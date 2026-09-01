"""Offload placement: what an 'offload' setting means on each backend.

Offloading trades copies for residency, and that trade only pays where the
accelerator has a memory pool of its own. The device-shaped downgrades live
in place_component, so they are checked here against a stand-in component
rather than a real pipeline.
"""

import logging

import pytest

from dw.pipeline_processors import pipeline as pipeline_module
from dw.pipeline_processors.pipeline import place_component


class FakeComponent:
    """Records the placement calls a real pipeline would act on."""

    def __init__(self):
        self.calls = []
        self._exclude_from_cpu_offload = []

    def enable_model_cpu_offload(self, device=None):
        self.calls.append(("model", device))

    def enable_sequential_cpu_offload(self, device=None):
        self.calls.append(("sequential", device))

    def to(self, device):
        self.calls.append(("to", device))
        return self


@pytest.fixture
def on_device(monkeypatch):
    """Run place_component as though the named backend were present."""

    def apply(device_type):
        monkeypatch.setattr(
            pipeline_module, "get_device_type", lambda device: device_type
        )

    return apply


def place(component, configuration, device):
    return place_component(component, "transformer", configuration, device)


class TestSequentialOnMps:
    def test_sequential_becomes_model_offload(self, on_device, caplog):
        on_device("mps")
        component = FakeComponent()
        with caplog.at_level(logging.WARNING, logger="dw"):
            place(component, {"offload": "sequential"}, "mps")
        assert component.calls == [("model", "mps")]
        assert "sequential" in caplog.text.lower()

    def test_the_warning_names_the_ignored_exclusions(self, on_device, caplog):
        # exclude_from_cpu_offload only means anything to the sequential path
        on_device("mps")
        component = FakeComponent()
        with caplog.at_level(logging.WARNING, logger="dw"):
            place(
                component,
                {"offload": "sequential", "exclude_from_cpu_offload": ["vae"]},
                "mps",
            )
        assert component.calls == [("model", "mps")]
        assert "exclude_from_cpu_offload" in caplog.text
        assert "vae" in caplog.text
        assert component._exclude_from_cpu_offload == []

    def test_model_offload_is_left_alone(self, on_device):
        on_device("mps")
        component = FakeComponent()
        place(component, {"offload": "model"}, "mps")
        assert component.calls == [("model", "mps")]


class TestOtherBackends:
    def test_sequential_stands_on_cuda(self, on_device):
        on_device("cuda")
        component = FakeComponent()
        place(
            component,
            {"offload": "sequential", "exclude_from_cpu_offload": ["vae"]},
            "cuda:1",
        )
        assert component.calls == [("sequential", "cuda:1")]
        assert component._exclude_from_cpu_offload == ["vae"]

    def test_cpu_drops_offload_entirely(self, on_device, caplog):
        on_device("cpu")
        component = FakeComponent()
        with caplog.at_level(logging.WARNING, logger="dw"):
            place(component, {"offload": "sequential"}, "cpu")
        assert component.calls == [("to", "cpu")]
        assert "not an accelerator" in caplog.text
