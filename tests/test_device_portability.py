"""
Tests that a device named in a workflow is translated to one the machine running
it actually has, so a workflow written on a CUDA box runs on a Mac and back again.
"""

import pytest
import torch

import dw
from dw.pipeline_processors.config_objects import get_group_offload_configuration
from dw.pipeline_processors.pipeline import Pipeline, configure_components
from dw.tasks.task import Task


@pytest.fixture
def cuda_is_missing(monkeypatch):
    """Stand this machine up as one with MPS and no CUDA"""
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")


class TestPipelineDevice:
    def test_a_steps_device_is_translated(self, cuda_is_missing):
        pipeline = Pipeline(
            {"configuration": {"device": "cuda"}, "arguments": {}}, 0, "mps"
        )

        assert pipeline.device == "mps"


class TestComponentDevice:
    def test_a_components_device_is_translated(self, cuda_is_missing):
        component = torch.nn.Linear(2, 2)
        pipeline = type("FakePipeline", (), {"vae": component})()

        configure_components(
            pipeline, {"components": {"vae": {"device": "cuda"}}}, "mps"
        )

        assert component.weight.device.type == "mps"


class TestGroupOffloadDevice:
    def test_the_onload_device_is_translated(self, cuda_is_missing):
        configuration = get_group_offload_configuration(
            {"group_offload": {"onload_device": "cuda"}}, "mps"
        )

        assert configuration["onload_device"] == torch.device("mps")


class TestTaskDevice:
    def test_a_tasks_device_override_is_translated(self, cuda_is_missing):
        task = Task({"command": "noop", "arguments": {}}, "mps")

        assert task.device_for({"device": "cuda"}) == "mps"


class TestPlacement:
    def test_a_components_device_is_translated_before_it_is_placed(
        self, cuda_is_missing
    ):
        from dw.pipeline_processors.pipeline import place_component

        class FakeComponent:
            def __init__(self):
                self.placed_on = None

            def to(self, device):
                self.placed_on = device
                return self

        component = FakeComponent()
        place_component(component, "transformer", {}, "cuda")

        assert component.placed_on == "mps"

    def test_a_translated_device_gets_its_backends_offload_downgrade(
        self, cuda_is_missing, caplog
    ):
        # The sequential-to-model downgrade keys off the backend, so a CUDA
        # workflow landing on MPS has to be seen as MPS by the time it is placed
        import logging

        from dw.pipeline_processors.pipeline import place_component

        class FakeComponent:
            def __init__(self):
                self.calls = []

            def enable_model_cpu_offload(self, device=None):
                self.calls.append(("model", device))

            def enable_sequential_cpu_offload(self, device=None):
                self.calls.append(("sequential", device))

            def to(self, device):
                self.calls.append(("to", device))
                return self

        component = FakeComponent()
        with caplog.at_level(logging.WARNING, logger="dw"):
            place_component(component, "transformer", {"offload": "sequential"}, "cuda")

        assert component.calls == [("model", "mps")]
