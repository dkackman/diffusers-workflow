"""Host memory readings beside the device ones (#81).

`get_memory` is documented as VRAM *and* RAM, and on a box whose workflows
offload weights to host memory by design, the host half is where a leak is
visible at all.
"""

import builtins

import pytest

from dw import host_memory


def test_stats_answer_this_process():
    stats = host_memory.host_memory_stats()
    assert set(stats) == {"rss_mb", "peak_rss_mb", "total_mb", "available_mb"}
    # This test is itself a running python process, so its own footprint is
    # measurable on every platform CI runs on
    assert stats["rss_mb"] is None or stats["rss_mb"] > 1.0
    if stats["total_mb"] is not None:
        assert stats["total_mb"] > stats["available_mb"] - 1.0


def test_fields_carry_the_payload_names():
    fields = host_memory.host_memory_fields()
    assert all(name.startswith("host_memory_") for name in fields)
    # every name reported is one the payload declares
    assert set(fields) <= set(host_memory.FIELD_NAMES.values())


def test_a_reading_that_cannot_be_taken_is_absent_rather_than_null(monkeypatch):
    monkeypatch.setattr(host_memory, "_psutil_stats", lambda: {"rss_mb": None})
    monkeypatch.setattr(host_memory, "_proc_stats", lambda: {"rss_mb": None})
    monkeypatch.setattr(host_memory, "_peak_rss_mb", lambda: None)
    assert host_memory.host_memory_fields() == {}


def test_a_failing_method_does_not_raise(monkeypatch):
    def boom():
        raise RuntimeError("no /proc here")

    monkeypatch.setattr(host_memory, "_psutil_stats", boom)
    monkeypatch.setattr(host_memory, "_proc_stats", boom)
    stats = host_memory.host_memory_stats()
    assert stats["rss_mb"] is None
    assert stats["total_mb"] is None


def test_the_stdlib_fallback_covers_a_machine_without_psutil(monkeypatch):
    """psutil is opportunistic here, not a declared dependency."""
    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    stats = host_memory.host_memory_stats()
    monkeypatch.undo()
    # The peak comes from resource(2), which needs neither psutil nor /proc
    assert stats["peak_rss_mb"] is None or stats["peak_rss_mb"] > 1.0


def test_the_worker_reports_host_fields_beside_the_gpu_ones():
    from dw.worker import WorkflowWorker

    worker = WorkflowWorker.__new__(WorkflowWorker)
    worker.run_count = 0
    info = WorkflowWorker._get_memory_info(worker)

    assert "gpu_memory_allocated_mb" in info
    host = {k: v for k, v in info.items() if k.startswith("host_memory_")}
    if host:  # a platform that can answer answers in MB, as the gpu keys do
        assert all(isinstance(v, float) for v in host.values())
        assert host["host_memory_rss_mb"] > 1.0


@pytest.mark.parametrize("gpu_available", [True, False])
def test_the_repl_prints_host_memory_either_way(capsys, gpu_available):
    from dw.repl import DiffusersWorkflowREPL

    repl = DiffusersWorkflowREPL.__new__(DiffusersWorkflowREPL)
    DiffusersWorkflowREPL._print_memory_info(
        repl,
        {
            "gpu_available": gpu_available,
            "run_count": 1,
            "host_memory_rss_mb": 1234.5,
            "host_memory_total_mb": 64000.0,
            "host_memory_available_mb": 32000.0,
        },
    )
    out = capsys.readouterr().out
    assert "Worker RSS: 1234.5 MB" in out
    assert "32000.0 MB of 64000.0 MB" in out
