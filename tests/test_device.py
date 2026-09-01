"""
Unit tests for device selection
Tests the configured device override and device type resolution
"""

import pytest
import dw


class TestGetDevice:
    """Test how the device dw runs on is chosen"""

    def test_environment_variable_wins_over_the_setting(self, monkeypatch):
        monkeypatch.setenv("DW_DEVICE", "cpu")
        monkeypatch.setattr(dw.settings, "device", "cuda:3")

        assert dw.get_device() == "cpu"

    def test_setting_is_used_without_an_environment_variable(self, monkeypatch):
        monkeypatch.delenv("DW_DEVICE", raising=False)
        monkeypatch.setattr(dw.settings, "device", "cuda:1")

        assert dw.get_device() == "cuda:1"

    def test_unconfigured_device_is_detected(self, monkeypatch):
        monkeypatch.delenv("DW_DEVICE", raising=False)
        monkeypatch.setattr(dw.settings, "device", None)

        assert dw.get_device() == dw.detect_device()

    def test_invalid_device_falls_back_to_detection(self, monkeypatch):
        # A typo should not take down a run that has already downloaded a model
        monkeypatch.delenv("DW_DEVICE", raising=False)
        monkeypatch.setattr(dw.settings, "device", "gpu")

        assert dw.get_device() == dw.detect_device()


class TestGetDeviceType:
    """Test resolution of a device identifier to its backend"""

    def test_index_is_stripped(self):
        assert dw.get_device_type("cuda:1") == "cuda"

    def test_plain_device_is_unchanged(self):
        assert dw.get_device_type("mps") == "mps"

    def test_defaults_to_the_device_dw_runs_on(self, monkeypatch):
        monkeypatch.setenv("DW_DEVICE", "cuda:2")

        assert dw.get_device_type() == "cuda"

    def test_autocast_follows_the_configured_device(self, monkeypatch):
        # A specific CUDA device still autocasts as CUDA
        monkeypatch.setenv("DW_DEVICE", "cuda:2")
        assert dw.get_autocast_device_type() == "cuda"

        monkeypatch.setenv("DW_DEVICE", "mps")
        assert dw.get_autocast_device_type() == "cpu"


class TestBackendAvailable:
    """Test whether a backend name is one this machine can actually run on"""

    def test_cpu_is_always_available(self):
        assert dw.backend_available("cpu") is True

    def test_cuda_follows_torch(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert dw.backend_available("cuda") is False

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert dw.backend_available("cuda") is True

    def test_mps_follows_torch(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert dw.backend_available("mps") is False

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert dw.backend_available("mps") is True

    def test_an_unrecognized_backend_is_left_alone(self):
        # dw does not know how to probe every accelerator torch may grow, and
        # guessing 'unavailable' would rewrite a device that works
        assert dw.backend_available("meta") is True


class TestResolveDevice:
    """Test translation of a workflow's device to one this machine has"""

    def test_a_device_on_an_available_backend_is_untouched(self, monkeypatch):
        monkeypatch.setattr(dw, "backend_available", lambda backend: backend == "cuda")

        assert dw.resolve_device("cuda") == "cuda"

    def test_an_index_survives_when_the_backend_matches(self, monkeypatch):
        # 'cuda:1' on a single-GPU CUDA box stays a genuine error, not a
        # portability problem to paper over
        monkeypatch.setattr(dw, "backend_available", lambda backend: backend == "cuda")

        assert dw.resolve_device("cuda:1") == "cuda:1"

    def test_an_unavailable_backend_becomes_the_machines_device(self, monkeypatch):
        monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
        monkeypatch.setattr(dw, "get_device", lambda: "mps")

        assert dw.resolve_device("cuda") == "mps"

    def test_translation_works_in_the_other_direction(self, monkeypatch):
        monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "mps")
        monkeypatch.setattr(dw, "get_device", lambda: "cuda")

        assert dw.resolve_device("mps") == "cuda"

    def test_cpu_is_never_rewritten(self, monkeypatch):
        # Pinning a step to the CPU is how a GPU-specific problem gets ruled
        # out, so it must never be helpfully upgraded to an accelerator
        monkeypatch.setattr(dw, "get_device", lambda: "cuda")

        assert dw.resolve_device("cpu") == "cpu"

    def test_none_stays_none(self):
        # Callers use None to mean 'no override', and inherit their default
        assert dw.resolve_device(None) is None

    def test_translation_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
        monkeypatch.setattr(dw, "get_device", lambda: "mps")

        with caplog.at_level("WARNING"):
            dw.resolve_device("cuda")

        assert "cuda" in caplog.text and "mps" in caplog.text

    def test_a_dropped_index_is_called_out(self, monkeypatch):
        # Splitting across accelerators has no equivalent on a one-device
        # backend, so the run is not what the workflow asked for
        monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
        monkeypatch.setattr(dw, "get_device", lambda: "mps")

        assert dw.resolve_device("cuda:1") == "mps"

    def test_an_invalid_device_is_left_for_torch_to_reject(self, monkeypatch):
        monkeypatch.setattr(dw, "get_device", lambda: "mps")

        assert dw.resolve_device("not-a-device") == "not-a-device"
