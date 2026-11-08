"""Host memory readings beside the device ones (#81).

`get_memory` is documented as VRAM *and* RAM, and on a box whose workflows
offload weights to host memory by design, the host half is where a leak is
visible at all.
"""

import builtins
import os
import sys

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
    assert set(fields) <= set(host_memory.FIELD_NAMES.values()) | set(
        host_memory.SPLIT_FIELD_NAMES.values()
    )


def test_a_reading_that_cannot_be_taken_is_absent_rather_than_null(monkeypatch):
    monkeypatch.setattr(host_memory, "_psutil_stats", lambda: {"rss_mb": None})
    monkeypatch.setattr(host_memory, "_proc_stats", lambda: {"rss_mb": None})
    monkeypatch.setattr(host_memory, "_peak_rss_mb", lambda: None)
    # the Linux-only anon/file split reads /proc directly (#709); without the
    # stub the test is green on macOS and red on CI
    monkeypatch.setattr(host_memory, "rss_split_mb", lambda: {})
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


def test_rss_split_reads_anon_and_file(tmp_path):
    status = tmp_path / "status"
    status.write_text("VmRSS:\t300 kB\nRssAnon:\t102400 kB\nRssFile:\t204800 kB\n")
    assert host_memory.rss_split_mb(str(status)) == {"anon_mb": 100.0, "file_mb": 200.0}


def test_rss_split_absent_without_the_split(tmp_path):
    status = tmp_path / "status"
    status.write_text("VmRSS:\t300 kB\n")
    assert host_memory.rss_split_mb(str(status)) == {}
    assert host_memory.rss_split_mb(str(tmp_path / "missing")) == {}


def test_read_oom_score_adj_reads_proc_or_none():
    from dw.host_memory import read_oom_score_adj

    assert read_oom_score_adj(0) is None
    assert read_oom_score_adj(None) is None
    if sys.platform.startswith("linux"):
        assert read_oom_score_adj(os.getpid()) == int(
            open("/proc/self/oom_score_adj").read()
        )
    else:
        assert read_oom_score_adj(os.getpid()) is None


# ------------------------------------------------- committed memory (#869)

# A /proc/<pid>/smaps excerpt shaped like lem's H3 worker: an anonymous
# mapping, a mapped checkpoint, a /dev/zero shared mapping and the NVIDIA
# driver's /dev/nvidiactl and /dev/nvidia-uvm mappings - the last two are
# pinned host pages that the kernel files under RssFile
SMAPS = """\
7d0000000000-7d0040000000 rw-p 00000000 00:00 0
Size:            1048576 kB
Rss:              102400 kB
Anonymous:        102400 kB
7d1000000000-7d1100000000 r--p 00000000 103:02 1234                       /models/h3/model.safetensors
Size:            4194304 kB
Rss:              409600 kB
Anonymous:             0 kB
7d2000000000-7d2000fc0000 rw-s 00000000 00:01 77                         /dev/zero (deleted)
Size:              16128 kB
Rss:               16128 kB
200400000-203400000 rw-s 00000000 00:07 881                              /dev/nvidiactl
Size:              49152 kB
Rss:               49152 kB
Anonymous:             0 kB
204800000-204a00000 rw-s 204800000 00:07 909                             /dev/nvidia-uvm
Size:               2048 kB
Rss:                2048 kB
200200000-200400000 rw-s 00000000 00:07 891                              /dev/nvidia0
Size:               2048 kB
Rss:                   0 kB
"""


def _status(tmp_path, text):
    path = tmp_path / "status"
    path.write_text(text)
    return str(path)


def _smaps(tmp_path, text=SMAPS):
    path = tmp_path / "smaps"
    path.write_text(text)
    return str(path)


def test_rss_split_reads_shmem(tmp_path):
    status = _status(
        tmp_path, "RssAnon:\t102400 kB\nRssFile:\t204800 kB\nRssShmem:\t16384 kB\n"
    )
    assert host_memory.rss_split_mb(status)["shmem_mb"] == 16.0


def test_device_mapped_pages_are_the_nvidia_mappings_rss(tmp_path):
    assert host_memory.device_mapped_mb(_smaps(tmp_path)) == (49152 + 2048) / 1024


def test_device_mapped_pages_unreadable_is_none(tmp_path):
    assert host_memory.device_mapped_mb(str(tmp_path / "missing")) is None


def test_committed_is_anon_plus_shmem_plus_driver_mapped(tmp_path):
    status = _status(
        tmp_path, "RssAnon:\t102400 kB\nRssFile:\t462848 kB\nRssShmem:\t16128 kB\n"
    )
    assert host_memory.committed_mb(status, _smaps(tmp_path)) == (
        (102400 + 16128 + 49152 + 2048) / 1024
    )


def test_committed_without_an_rssanon_line_is_none(tmp_path):
    status = _status(tmp_path, "VmRSS:\t300 kB\n")
    assert host_memory.committed_mb(status, _smaps(tmp_path)) is None
    assert host_memory.committed_rss_mb(None, status_path=status) is None


def test_a_failed_status_read_on_linux_is_none_not_full_rss(tmp_path, monkeypatch):
    # /proc is there, but the process's status is not readable (it exited
    # between the pid lookup and the read): its full RSS would under-charge
    monkeypatch.setattr(host_memory, "_has_proc", lambda: True)
    monkeypatch.setattr(host_memory, "process_rss_mb", lambda pid: 99999.0)
    monkeypatch.setattr(
        host_memory, "_proc_path", lambda pid, name: str(tmp_path / f"gone-{name}")
    )
    assert host_memory.committed_rss_mb(4242) is None


def test_without_proc_a_workers_committed_rss_is_its_whole_rss(monkeypatch):
    # macOS: no /proc and no split, so the peaks it is compared with are RSS
    # too, and the whole RSS is the reading on the same basis
    monkeypatch.setattr(host_memory, "_has_proc", lambda: False)
    monkeypatch.setattr(host_memory, "process_rss_mb", lambda pid: 1234.0)
    assert host_memory.committed_rss_mb(4242) == 1234.0


def test_the_pid_path_reads_that_processes_status_and_smaps(tmp_path, monkeypatch):
    _status(tmp_path, "RssAnon:\t102400 kB\nRssShmem:\t16128 kB\n")
    _smaps(tmp_path)
    asked = []

    def proc_path(pid, name):
        asked.append((pid, name))
        return str(tmp_path / name)

    monkeypatch.setattr(host_memory, "_has_proc", lambda: True)
    monkeypatch.setattr(host_memory, "_proc_path", proc_path)
    assert host_memory.committed_rss_mb(4242) == (
        (102400 + 16128 + 49152 + 2048) / 1024
    )
    assert asked == [(4242, "status"), (4242, "smaps")]


def test_no_pid_is_no_reading():
    assert host_memory.committed_rss_mb(None) is None
    assert host_memory.committed_rss_mb(0) is None


def test_fields_carry_the_committed_reading(monkeypatch):
    monkeypatch.setattr(
        host_memory,
        "rss_split_mb",
        lambda: {"anon_mb": 100.0, "file_mb": 400.0, "shmem_mb": 16.0},
    )
    monkeypatch.setattr(host_memory, "device_mapped_mb", lambda: 50.0)
    fields = host_memory.host_memory_fields()
    assert fields["host_memory_rss_shmem_mb"] == 16.0
    assert fields["host_memory_committed_mb"] == 166.0


def test_fields_reuse_a_recent_device_reading(monkeypatch):
    # smaps is read whole (seconds on a large worker); the memory report at
    # every phase boundary reuses a reading inside DEVICE_TERM_SECONDS
    reads = []

    def device():
        reads.append(1)
        return 50.0

    clock = [1000.0]
    monkeypatch.setattr(
        host_memory,
        "rss_split_mb",
        lambda: {"anon_mb": 100.0, "file_mb": 400.0, "shmem_mb": 16.0},
    )
    monkeypatch.setattr(host_memory, "device_mapped_mb", device)
    monkeypatch.setattr(host_memory.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(host_memory, "_device_term", [None, None, None])
    for _ in range(3):
        assert host_memory.host_memory_fields()["host_memory_committed_mb"] == 166.0
    assert len(reads) == 1
    clock[0] += host_memory.DEVICE_TERM_SECONDS + 1
    host_memory.host_memory_fields()
    assert len(reads) == 2
