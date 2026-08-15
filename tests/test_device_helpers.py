"""
Unit tests for the shared device/dtype helpers in dw/__init__.py
(preferred_task_dtype, empty_device_cache, device_memory_stats), and for
the worker-side translation of device_memory_stats() into the REPL's
memory-info dict shape.
"""

import queue

import pytest
import torch

import dw
from dw.worker import WorkflowWorker

# ---------------------------------------------------------------------------
# preferred_task_dtype
# ---------------------------------------------------------------------------


class TestPreferredTaskDtype:
    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
    def test_cuda_devices_get_float16(self, device):
        assert dw.preferred_task_dtype(device) == torch.float16

    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_non_cuda_devices_get_float32(self, device):
        assert dw.preferred_task_dtype(device) == torch.float32

    def test_defaults_to_the_configured_device(self, monkeypatch):
        monkeypatch.setenv("DW_DEVICE", "cuda:2")
        assert dw.preferred_task_dtype() == torch.float16

        monkeypatch.setenv("DW_DEVICE", "cpu")
        assert dw.preferred_task_dtype() == torch.float32


# ---------------------------------------------------------------------------
# empty_device_cache
# ---------------------------------------------------------------------------


class TestEmptyDeviceCache:
    def test_cuda_empties_cache_without_synchronizing_by_default(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        empty_cache = MagicMock()
        synchronize = MagicMock()
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

        dw.empty_device_cache()

        empty_cache.assert_called_once()
        synchronize.assert_not_called()

    def test_cuda_synchronizes_when_asked(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        empty_cache = MagicMock()
        synchronize = MagicMock()
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

        dw.empty_device_cache(synchronize=True)

        empty_cache.assert_called_once()
        synchronize.assert_called_once()

    def test_cuda_type_without_actual_availability_is_a_noop(self, monkeypatch):
        from unittest.mock import MagicMock

        # get_device_type() can say "cuda" (e.g. a stale DW_DEVICE) while the
        # backend itself isn't actually usable - the availability check must
        # still gate the call.
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        empty_cache = MagicMock()
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

        dw.empty_device_cache()

        empty_cache.assert_not_called()

    def test_mps_empties_cache_without_synchronizing_by_default(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        empty_cache = MagicMock()
        synchronize = MagicMock()
        monkeypatch.setattr(torch.mps, "empty_cache", empty_cache)
        monkeypatch.setattr(torch.mps, "synchronize", synchronize)

        dw.empty_device_cache()

        empty_cache.assert_called_once()
        synchronize.assert_not_called()

    def test_mps_synchronizes_when_asked(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        empty_cache = MagicMock()
        synchronize = MagicMock()
        monkeypatch.setattr(torch.mps, "empty_cache", empty_cache)
        monkeypatch.setattr(torch.mps, "synchronize", synchronize)

        dw.empty_device_cache(synchronize=True)

        empty_cache.assert_called_once()
        synchronize.assert_called_once()

    def test_cpu_is_a_noop(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cpu")
        cuda_empty = MagicMock()
        mps_empty = MagicMock()
        monkeypatch.setattr(torch.cuda, "empty_cache", cuda_empty)
        monkeypatch.setattr(torch.mps, "empty_cache", mps_empty)

        dw.empty_device_cache(synchronize=True)

        cuda_empty.assert_not_called()
        mps_empty.assert_not_called()


# ---------------------------------------------------------------------------
# device_memory_stats
# ---------------------------------------------------------------------------


class TestDeviceMemoryStats:
    def test_cuda_reports_full_stats(self, monkeypatch):
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100 * 1024 * 1024)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 200 * 1024 * 1024)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda: (300 * 1024 * 1024, 400 * 1024 * 1024)
        )

        stats = dw.device_memory_stats()

        assert stats == {
            "available": True,
            "device_name": "Fake GPU",
            "allocated_mb": 100.0,
            "reserved_mb": 200.0,
            "free_mb": 300.0,
            "total_mb": 400.0,
        }

    def test_cuda_mem_get_info_failure_leaves_free_and_total_none(self, monkeypatch):
        def raise_runtime_error():
            raise RuntimeError("no info")

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", raise_runtime_error)

        stats = dw.device_memory_stats()

        assert stats["available"] is True
        assert stats["free_mb"] is None
        assert stats["total_mb"] is None

    def test_mps_reports_zeroed_known_stats(self, monkeypatch):
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        stats = dw.device_memory_stats()

        assert stats == {
            "available": True,
            "device_name": "Apple Silicon (MPS)",
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
        }

    def test_cpu_reports_unavailable(self, monkeypatch):
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cpu")

        stats = dw.device_memory_stats()

        assert stats == {
            "available": False,
            "device_name": None,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": None,
            "total_mb": None,
        }


# ---------------------------------------------------------------------------
# WorkflowWorker._get_memory_info translation of device_memory_stats()
# ---------------------------------------------------------------------------


def _make_worker():
    return WorkflowWorker(queue.Queue(), queue.Queue())


class TestWorkerMemoryInfoTranslation:
    def test_total_mb_key_present_when_measurable(self, monkeypatch):
        import dw.worker as worker_module

        monkeypatch.setattr(
            worker_module,
            "device_memory_stats",
            lambda: {
                "available": True,
                "device_name": "Fake GPU",
                "allocated_mb": 1.0,
                "reserved_mb": 2.0,
                "free_mb": 3.0,
                "total_mb": 4.0,
            },
        )

        info = _make_worker()._get_memory_info()

        assert info["gpu_available"] is True
        assert info["gpu_memory_free_mb"] == 3.0
        assert info["gpu_memory_total_mb"] == 4.0

    def test_total_mb_key_absent_when_unmeasurable(self, monkeypatch):
        # Mirrors CUDA's mem_get_info failure case: free/total come back None
        # from device_memory_stats(), and the historical worker.py behavior
        # left gpu_memory_free_mb at its 0.0 default and omitted
        # gpu_memory_total_mb from the dict entirely rather than setting it
        # to None.
        import dw.worker as worker_module

        monkeypatch.setattr(
            worker_module,
            "device_memory_stats",
            lambda: {
                "available": True,
                "device_name": "Fake GPU",
                "allocated_mb": 1.0,
                "reserved_mb": 2.0,
                "free_mb": None,
                "total_mb": None,
            },
        )

        info = _make_worker()._get_memory_info()

        assert info["gpu_memory_free_mb"] == 0.0
        assert "gpu_memory_total_mb" not in info

    def test_get_gpu_memory_mb_uses_allocated_mb(self, monkeypatch):
        import dw.worker as worker_module

        monkeypatch.setattr(
            worker_module,
            "device_memory_stats",
            lambda: {
                "available": True,
                "device_name": "Fake GPU",
                "allocated_mb": 42.0,
                "reserved_mb": 0.0,
                "free_mb": None,
                "total_mb": None,
            },
        )

        assert _make_worker()._get_gpu_memory_mb() == 42.0
