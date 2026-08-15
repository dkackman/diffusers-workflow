"""
Unit tests for modular pipeline support
Tests load_components handling in component loading
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from dw.arguments import realize_args
from dw.pipeline_processors.pipeline import load_component


def make_component_type(component):
    """Build a mock pipeline class whose from_pretrained returns the given component"""
    component_type = MagicMock()
    component_type.__name__ = "MockModularPipeline"
    component_type.from_pretrained.return_value = component
    return component_type


class TestLoadComponents:
    """Test load_components configuration for modular pipelines"""

    def test_load_components_called_with_arguments(self):
        component = MagicMock()
        configuration = {
            "component_type": make_component_type(component),
            "load_components": {"dtype": torch.bfloat16},
        }

        load_component("pipeline", configuration, {"model_name": "test-model"}, "cpu")

        component.load_components.assert_called_once_with(dtype=torch.bfloat16)

    def test_load_components_runs_before_device_placement(self):
        # Components have to exist before the pipeline can be moved to the device
        component = MagicMock()
        calls = []
        component.load_components.side_effect = lambda **kwargs: calls.append(
            "load_components"
        )
        component.to.side_effect = lambda device: calls.append("to") or component

        configuration = {
            "component_type": make_component_type(component),
            "load_components": {},
        }

        load_component("pipeline", configuration, {"model_name": "test-model"}, "cpu")

        assert calls == ["load_components", "to"]

    def test_not_called_when_not_configured(self):
        component = MagicMock()
        configuration = {"component_type": make_component_type(component)}

        load_component("pipeline", configuration, {"model_name": "test-model"}, "cpu")

        component.load_components.assert_not_called()

    def test_raises_when_pipeline_does_not_support_it(self):
        component = MagicMock(spec=["to"])
        configuration = {
            "component_type": make_component_type(component),
            "load_components": {"dtype": torch.bfloat16},
        }

        with pytest.raises(ValueError, match="modular pipelines"):
            load_component(
                "pipeline", configuration, {"model_name": "test-model"}, "cpu"
            )


class TestLoadingDevice:
    """Test the device component weights are materialized on while loading"""

    @pytest.fixture
    def default_device_is_not_cpu(self):
        # Stands in for a default device set outside of dw - loading an offloaded
        # component still has to land in system memory
        torch.set_default_device("meta")
        yield
        torch.set_default_device(None)

    def load_and_record_device(self, configuration):
        """Load a component that records the default device it was created under"""
        recorded = {}

        def from_pretrained(model_name, **kwargs):
            recorded["device"] = torch.empty(1).device.type
            return MagicMock()

        component_type = MagicMock()
        component_type.__name__ = "MockPipeline"
        component_type.from_pretrained.side_effect = from_pretrained
        configuration["component_type"] = component_type

        load_component("pipeline", configuration, {"model_name": "test-model"}, "cuda")

        return recorded["device"]

    def test_sequential_offload_loads_into_system_memory(
        self, default_device_is_not_cpu
    ):
        assert self.load_and_record_device({"offload": "sequential"}) == "cpu"

    def test_model_offload_loads_into_system_memory(self, default_device_is_not_cpu):
        assert self.load_and_record_device({"offload": "model"}) == "cpu"

    def test_group_offload_loads_into_system_memory(self, default_device_is_not_cpu):
        configuration = {"group_offload": {"offload_type": "leaf_level"}}
        assert self.load_and_record_device(configuration) == "cpu"

    def test_default_device_is_used_without_offloading(self, default_device_is_not_cpu):
        assert self.load_and_record_device({}) == "meta"


class TestComponentsManager:
    """Test components manager creation for modular pipelines"""

    def load(self, configuration, component=None):
        """Load a component with a patched ComponentsManager, returning both mocks"""
        component = component or MagicMock()
        component_type = make_component_type(component)
        configuration = {"component_type": component_type, **configuration}

        with patch("diffusers.ComponentsManager") as manager_type:
            load_component(
                "pipeline", configuration, {"model_name": "test-model"}, "cuda"
            )

        return component_type, manager_type.return_value

    def test_not_created_when_not_configured(self):
        component_type, manager = self.load({})

        _, kwargs = component_type.from_pretrained.call_args
        assert "components_manager" not in kwargs
        manager.enable_auto_cpu_offload.assert_not_called()

    def test_manager_passed_to_from_pretrained(self):
        component_type, manager = self.load({"components_manager": {}})

        _, kwargs = component_type.from_pretrained.call_args
        assert kwargs["components_manager"] is manager
        manager.enable_auto_cpu_offload.assert_not_called()

    def test_auto_cpu_offload_enabled_on_device(self):
        _, manager = self.load(
            {"components_manager": {"enable_auto_cpu_offload": True}}
        )

        manager.enable_auto_cpu_offload.assert_called_once_with(device="cuda")

    def test_memory_reserve_margin_forwarded(self):
        _, manager = self.load(
            {
                "components_manager": {
                    "enable_auto_cpu_offload": True,
                    "memory_reserve_margin": "6GB",
                }
            }
        )

        manager.enable_auto_cpu_offload.assert_called_once_with(
            device="cuda", memory_reserve_margin="6GB"
        )

    def test_auto_cpu_offload_leaves_device_placement_to_the_manager(self):
        component = MagicMock()
        self.load({"components_manager": {"enable_auto_cpu_offload": True}}, component)

        component.to.assert_not_called()

    def test_manager_without_offload_still_moves_to_device(self):
        component = MagicMock()
        self.load({"components_manager": {}}, component)

        component.to.assert_called_once_with("cuda")


class TestDtypeConversion:
    """Test that a bare 'dtype' key is converted to a torch dtype"""

    def test_dtype_key_converted(self):
        arguments = {"load_components": {"dtype": "torch.bfloat16"}}
        realize_args(arguments)
        assert arguments["load_components"]["dtype"] == torch.bfloat16

    def test_dtype_key_can_be_escaped(self):
        arguments = {"dtype": "{auto}"}
        realize_args(arguments)
        assert arguments["dtype"] == "auto"
