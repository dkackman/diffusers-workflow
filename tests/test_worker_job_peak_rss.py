"""Tests for `_job_scoped_peak_rss_mb`'s running-max behavior (#457).

`host_memory_job_peak_rss_mb` is read by `get_job_events` callers as a
job-scoped high-water mark; it must never decrease across a job's readings
even when the process-lifetime peak stops growing and current rss falls
(post-run cleanup, GC).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.worker import _job_scoped_peak_rss_mb


def test_running_max_holds_when_rss_drops():
    """Reproduces #457's own repro: a light job run right after a heavy one
    inherits the heavy job's process-lifetime peak as its baseline, and
    never grows past it - so every reading falls to the current-rss branch,
    exactly as job 2's readings in the issue did (host_memory_peak_rss_mb
    flat at 62192.34 throughout)."""
    baseline_peak_mb = 62192.34
    running_max_mb = None

    readings = [
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 922.199},
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 922.199},
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 936.961},
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 940.699},
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 949.852},
        # Final reading: post-run cleanup drops current rss well below the
        # 949.852 max already seen - the running max must hold, not drop.
        {"host_memory_peak_rss_mb": 62192.34, "host_memory_rss_mb": 932.563},
    ]

    reported = []
    for info in readings:
        running_max_mb = _job_scoped_peak_rss_mb(info, baseline_peak_mb, running_max_mb)
        reported.append(running_max_mb)

    assert reported == sorted(reported)
    assert reported[-1] == 949.852


def test_running_max_none_baseline_returns_previous_max():
    assert (
        _job_scoped_peak_rss_mb({"host_memory_peak_rss_mb": 100.0}, None, 42.0) == 42.0
    )


def test_running_max_first_reading_with_no_growth_uses_current_rss():
    result = _job_scoped_peak_rss_mb(
        {"host_memory_peak_rss_mb": 500.0, "host_memory_rss_mb": 123.0}, 500.0, None
    )
    assert result == 123.0
