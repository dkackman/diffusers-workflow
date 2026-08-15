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
