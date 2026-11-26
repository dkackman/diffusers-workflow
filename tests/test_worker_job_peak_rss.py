"""Tests for `_job_scoped_peak_rss_mb`'s running-max behavior (#457).

`host_memory_job_peak_rss_mb` is read by `get_job_events` callers as a
job-scoped high-water mark; it must never decrease across a job's readings
even when the process-lifetime peak stops growing and current rss falls
(post-run cleanup, GC).
"""

import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.worker_protocol import MemoryInfo
from dw.worker import WorkflowWorker, _Job, _job_scoped_peak_rss_mb


def test_running_max_holds_when_rss_drops():
    """Reproduces #457's own repro: a light job run right after a heavy one
    inherits the heavy job's process-lifetime peak as its baseline, and
    never grows past it - so every reading falls to the current-rss branch,
    exactly as job 2's readings in the issue did (host_memory_peak_rss_mb
    flat at 62192.34 throughout)."""
    baseline_peak_mb = 62192.34
    running_max_mb = None

    # Process peak flat at the baseline: no growth, so no job peak (#1288)
    # and current rss never leaks in as one.
    readings = [
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 922.199},
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 949.852},
        # The peak then grows a little; the running max holds afterwards.
        {"host_memory_peak_rss_mb": 62292.34, "host_memory_rss_mb": 949.852},
        {"host_memory_peak_rss_mb": 62292.34, "host_memory_rss_mb": 100.0},
    ]

    reported = []
    for info in readings:
        running_max_mb = _job_scoped_peak_rss_mb(
            info, baseline_peak_mb, 900.0, running_max_mb
        )
        reported.append(running_max_mb)

    assert reported[:2] == [None, None]
    assert reported[2] == reported[3]
    assert abs(reported[3] - (62292.34 - 900.0)) < 1e-6


def test_running_max_none_baseline_returns_previous_max():
    assert (
        _job_scoped_peak_rss_mb({"host_memory_peak_rss_mb": 100.0}, None, None, 42.0)
        == 42.0
    )


def test_no_growth_records_null_not_current_rss():
    """#1288: a warm run that grows nothing must not report the worker's
    resident rss as its own peak."""
    result = _job_scoped_peak_rss_mb(
        {"host_memory_peak_rss_mb": 500.0, "host_memory_rss_mb": 123.0},
        500.0,
        100.0,
        None,
    )
    assert result is None


def test_growth_after_flat_readings_is_reported():
    flat = {"host_memory_peak_rss_mb": 500.0, "host_memory_rss_mb": 400.0}
    assert _job_scoped_peak_rss_mb(flat, 500.0, 400.0, None) is None
    grown = {"host_memory_peak_rss_mb": 800.0, "host_memory_rss_mb": 400.0}
    assert _job_scoped_peak_rss_mb(grown, 500.0, 400.0, None) == 400.0
    assert _job_scoped_peak_rss_mb(flat, 500.0, 400.0, 400.0) == 400.0


def _run_job_peak(monkeypatch, readings):
    """Drive the worker's own phase and final readings: `readings` is
    (peak, rss) pairs, the last one the post-run reading. Returns the job
    peak on the final memory_info."""
    pending = iter(readings)

    def _memory_info():
        peak, rss = next(pending)
        return {"host_memory_peak_rss_mb": peak, "host_memory_rss_mb": rss}

    replies = []
    worker = WorkflowWorker.__new__(WorkflowWorker)
    worker.settings = None
    worker.loaded_pipelines = {}
    worker.prior_step_keys = {}
    worker.run_count = 0
    worker.torch_flags_reported = True
    monkeypatch.setattr(worker, "_get_memory_info", _memory_info)
    monkeypatch.setattr(worker, "_reply", replies.append)
    monkeypatch.setattr(worker, "_failure_redactor", lambda command: lambda s: s)
    for name in (
        "_record_step_keys",
        "_report_torch_flags",
        "_evict_untouched_pipelines",
        "_cleanup_between_runs",
    ):
        monkeypatch.setattr(worker, name, lambda *args: None)
    watcher = SimpleNamespace(stop=lambda: None)
    monkeypatch.setattr(worker, "_watch_commands", lambda context: watcher)
    monkeypatch.setattr("dw.worker.configure_torch", lambda *args: None)

    def _run(arguments, pipelines, context=None, prior_step_keys=None):
        for _ in readings[:-1]:
            context.emit("phase", phase="load")

    job = _Job(workflow=SimpleNamespace(run=_run, manifest=[]))
    worker._run_job({"arguments": {}}, job)
    worker._report_success(job)
    infos = [r.info for r in replies if isinstance(r, MemoryInfo)]
    return infos[-1]["host_memory_job_peak_rss_mb"]


def test_earlier_jobs_mark_does_not_lower_this_jobs_peak(monkeypatch):
    """#1754: the issue's two jobs, one in a fresh worker and one under an
    earlier job's mark, started at the same size and reached the same peak."""
    fresh = _run_job_peak(monkeypatch, [(460.7, 460.7), (60197.2, 900.0)])
    used = _run_job_peak(monkeypatch, [(19527.0, 460.7), (60197.2, 900.0)])
    assert fresh == used == 60197.2 - 460.7


def test_warm_worker_residents_are_not_the_jobs_peak(monkeypatch):
    """A job in a worker holding 30 GB of models that lifts the mark by 1 GB
    reports that 1 GB, not the mark (#1288)."""
    peak = _run_job_peak(
        monkeypatch, [(30000.0, 30000.0), (31000.0, 30500.0), (31000.0, 30000.0)]
    )
    assert peak == 1000.0
