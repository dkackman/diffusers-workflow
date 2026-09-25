"""Host memory readings beside the device ones (#81).

`get_memory` is documented as VRAM *and* RAM, and on a box whose workflows
offload weights to host memory by design, the host half is where a leak is
visible at all.
"""

import builtins
import os
import sys

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
    refused = []

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            refused.append(name)
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    stats = host_memory.host_memory_stats()
    monkeypatch.undo()
    # the psutil path was taken and refused, and the call still answered
    assert refused
    assert set(stats) == {"rss_mb", "peak_rss_mb", "total_mb", "available_mb"}
    # The peak comes from resource(2), which needs neither psutil nor /proc
    if os.name == "posix":
        assert stats["peak_rss_mb"] > 1.0
    # /proc answers the rest on Linux
    if sys.platform.startswith("linux"):
        assert stats["rss_mb"] > 1.0
        assert stats["total_mb"] > 1.0


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


def test_the_peak_is_never_below_the_resident_figure(monkeypatch):
    """peak - rss is what the pair exists to answer; a small negative there
    reads as 'these fields are not comparable' - see issue #83."""
    monkeypatch.setattr(host_memory, "_peak_rss_mb", lambda: 764.1484375)
    monkeypatch.setattr(
        host_memory,
        "_psutil_stats",
        lambda: {"rss_mb": 764.79296875, "total_mb": 64000.0, "available_mb": 32000.0},
    )

    stats = host_memory.host_memory_stats()

    assert stats["peak_rss_mb"] == stats["rss_mb"] == 764.79296875


def test_a_genuine_peak_is_left_alone(monkeypatch):
    monkeypatch.setattr(host_memory, "_peak_rss_mb", lambda: 33044.98)
    monkeypatch.setattr(
        host_memory,
        "_psutil_stats",
        lambda: {"rss_mb": 2561.69, "total_mb": 64000.0, "available_mb": 32000.0},
    )

    assert host_memory.host_memory_stats()["peak_rss_mb"] == 33044.98


class TestReleasingHostCaches:
    """What a full cleanup hands back to the OS, and what it reports (#98).

    A worker that had released every model still sat on 14.5 GB of anonymous
    memory on the box this was measured on, which is the whole margin a
    template needing 96% of host RAM has.
    """

    def test_it_empties_the_pinned_cache_and_trims_the_heap(self, monkeypatch):
        called = []

        class FakeC:
            @staticmethod
            def _host_emptyCache():
                called.append("pinned")

        monkeypatch.setitem(
            __import__("sys").modules, "torch", type("torch", (), {"_C": FakeC})
        )
        monkeypatch.setattr(
            host_memory, "trim_host_memory", lambda: called.append("trim")
        )
        readings = iter([20000.0, 14000.0])
        monkeypatch.setattr(
            host_memory, "host_memory_stats", lambda: {"rss_mb": next(readings)}
        )

        released = host_memory.release_host_caches()

        assert called == ["pinned", "trim"]
        assert released == 6000.0

    def test_a_reading_it_cannot_take_is_not_a_number_it_invents(self, monkeypatch):
        monkeypatch.setattr(host_memory, "trim_host_memory", lambda: 0.0)
        monkeypatch.setattr(host_memory, "host_memory_stats", lambda: {"rss_mb": None})

        assert host_memory.release_host_caches() == 0.0

    def test_a_torch_without_the_hook_is_not_an_error(self, monkeypatch):
        monkeypatch.setitem(
            __import__("sys").modules, "torch", type("torch", (), {"_C": object})
        )
        monkeypatch.setattr(host_memory, "trim_host_memory", lambda: 0.0)
        monkeypatch.setattr(host_memory, "host_memory_stats", lambda: {"rss_mb": 100.0})

        assert host_memory.release_host_caches() == 0.0

    def test_the_pinned_figures_are_reported_in_mb(self, monkeypatch):
        class FakeCuda:
            @staticmethod
            def host_memory_stats():
                return {
                    "allocated_bytes.all.current": 512 * 1024 * 1024,
                    "reserved_bytes.all.current": 2048 * 1024 * 1024,
                }

        monkeypatch.setitem(
            __import__("sys").modules, "torch", type("torch", (), {"cuda": FakeCuda})
        )

        assert host_memory.pinned_host_memory_fields() == {
            "host_pinned_allocated_mb": 512.0,
            "host_pinned_reserved_mb": 2048.0,
        }

    def test_no_pinned_allocator_reports_nothing_rather_than_zero(self, monkeypatch):
        """A key that is absent says 'not measurable here'; a zero would say
        'measured, and there is none' - the same rule the host fields follow."""

        class FakeCuda:
            @staticmethod
            def host_memory_stats():
                raise RuntimeError("no CUDA")

        monkeypatch.setitem(
            __import__("sys").modules, "torch", type("torch", (), {"cuda": FakeCuda})
        )

        assert host_memory.pinned_host_memory_fields() == {}

    def test_trim_is_a_no_op_off_linux(self, monkeypatch):
        monkeypatch.setattr(__import__("sys"), "platform", "darwin")

        assert host_memory.trim_host_memory() == 0.0
