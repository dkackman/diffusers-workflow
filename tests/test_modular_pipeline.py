"""
Unit tests for modular pipeline support
Tests load_components handling in component loading
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from dw.arguments import realize_args
from dw.pipeline_processors.pipeline import (
    Pipeline,
    configure_components,
    get_component,
    load_component,
)


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


class TestConfiguredDevice:
    """Test the device a pipeline and its components are placed on"""

    def test_workflow_device_is_used_by_default(self):
        pipeline = Pipeline({"configuration": {}, "arguments": {}}, 42, "cuda")
        assert pipeline.device == "cuda"

    def test_configuration_device_overrides_the_workflow_device(self):
        definition = {"configuration": {"device": "cpu"}, "arguments": {}}
        assert Pipeline(definition, 42, "cuda").device == "cpu"

    def test_components_inherit_the_pipeline_device(self):
        definition = {
            "configuration": {"device": "cpu", "component_type": MagicMock()},
            "vae": {
                "configuration": {"component_type": MagicMock()},
                "from_pretrained_arguments": {"model_name": "test-vae"},
            },
            "arguments": {},
        }

        with patch("dw.pipeline_processors.pipeline.load_component") as load:
            Pipeline(definition, 42, "cuda").load({})

        # The component is loaded before the pipeline that holds it
        assert [call.args[0] for call in load.call_args_list] == ["vae", "pipeline"]
        assert [call.args[3] for call in load.call_args_list] == ["cpu", "cpu"]

    def test_a_component_can_pin_itself_to_another_device(self):
        definition = {
            "configuration": {"component_type": MagicMock()},
            "vae": {
                "configuration": {"component_type": MagicMock(), "device": "cpu"},
                "from_pretrained_arguments": {"model_name": "test-vae"},
            },
            "arguments": {},
        }

        with patch("dw.pipeline_processors.pipeline.load_component") as load:
            Pipeline(definition, 42, "cuda").load({})

        assert [call.args[3] for call in load.call_args_list] == ["cpu", "cuda"]


class TestTensorResults:
    """Test where a raw tensor result rests once a pipeline has produced it"""

    def run(self, output):
        pipeline = Pipeline({"configuration": {}, "arguments": {}}, 42, "cuda")
        pipeline.pipeline = MagicMock(return_value=output, spec=["__call__", "vocoder"])
        return pipeline.run({})

    def test_tensor_results_rest_in_system_memory(self):
        # Held for the whole workflow, they would otherwise occupy the accelerator the
        # next step needs
        result = self.run(torch.ones(2, 2))
        assert result.device.type == "cpu"

    def test_other_results_are_left_alone(self):
        images = ["an image"]
        assert self.run(images) is images


class TestOffloadDevice:
    """Test the accelerator components are offloaded onto"""

    def load(self, configuration, device):
        component = MagicMock()
        configuration = {
            "component_type": make_component_type(component),
            **configuration,
        }

        load_component("pipeline", configuration, {"model_name": "test-model"}, device)

        return component

    def test_model_offload_targets_the_configured_device(self):
        component = self.load({"offload": "model"}, "cuda:1")
        component.enable_model_cpu_offload.assert_called_once_with(device="cuda:1")

    def test_sequential_offload_targets_the_configured_device(self):
        component = self.load({"offload": "sequential"}, "mps")
        component.enable_sequential_cpu_offload.assert_called_once_with(device="mps")

    def test_offload_is_skipped_on_the_cpu(self):
        # There is no accelerator to stream the model onto
        component = self.load({"offload": "sequential"}, "cpu")

        component.enable_sequential_cpu_offload.assert_not_called()
        component.to.assert_called_once_with("cpu")

    def test_component_group_offload_is_not_moved_to_device(self):
        # configure_components() installs the per-component group-offload hooks after
        # load_component() returns - moving the whole pipeline to the device here would
        # load it in full first, defeating the offloading
        component = self.load(
            {
                "components": {
                    "transformer": {"group_offload": {"offload_type": "leaf_level"}}
                }
            },
            "cuda",
        )

        component.to.assert_not_called()

    def test_components_without_group_offload_still_move_to_device(self):
        # A 'components' block that only moves a component to a device, with no
        # group_offload anywhere, should not change the pipeline's own placement
        component = self.load(
            {"components": {"transformer": {"device": "cuda"}}}, "cuda"
        )

        component.to.assert_called_once_with("cuda")


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

    def test_component_group_offload_loads_into_system_memory(
        self, default_device_is_not_cpu
    ):
        # Only a per-component group_offload entry is configured - no top-level offload
        # or group_offload - but the pipeline itself still has to land in system memory
        configuration = {
            "components": {
                "transformer": {"group_offload": {"offload_type": "leaf_level"}}
            }
        }
        assert self.load_and_record_device(configuration) == "cpu"


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


class FakeQuantizationConfig:
    """Stands in for a quantization config, e.g. TorchAoConfig"""

    def __init__(self, **arguments):
        self.arguments = arguments


class FakeQuantType:
    """Stands in for a quantization type, e.g. Int8WeightOnlyConfig"""


class TestComponentQuantization:
    """Test quantization declared per component for a modular pipeline's load_components"""

    def transformer_configuration(self):
        return {
            "configuration": {"config_type": FakeQuantizationConfig},
            "arguments": {"quant_type": FakeQuantType},
        }

    def load(self, load_components):
        component = MagicMock()
        configuration = {
            "component_type": make_component_type(component),
            "load_components": load_components,
        }

        load_component("pipeline", configuration, {"model_name": "test-model"}, "cpu")

        return component.load_components.call_args.kwargs

    def test_configuration_is_built_for_the_named_component(self):
        arguments = self.load(
            {
                "dtype": torch.bfloat16,
                "quantization_config": {
                    "transformer": self.transformer_configuration()
                },
            }
        )

        assert arguments["dtype"] == torch.bfloat16
        config = arguments["quantization_config"]["transformer"]
        assert isinstance(config, FakeQuantizationConfig)
        # A quantization type is instantiated, not passed as the class
        assert isinstance(config.arguments["quant_type"], FakeQuantType)

    def test_components_that_are_not_named_stay_unquantized(self):
        arguments = self.load(
            {"quantization_config": {"transformer": self.transformer_configuration()}}
        )

        assert list(arguments["quantization_config"]) == ["transformer"]

    def test_load_components_without_quantization_is_unchanged(self):
        assert self.load({"dtype": torch.bfloat16}) == {"dtype": torch.bfloat16}

    def test_the_definition_is_left_as_written(self):
        # Built configurations must not leak back into the workflow definition, which is
        # loaded once and run repeatedly
        definition = self.transformer_configuration()
        self.load({"quantization_config": {"transformer": definition}})

        assert definition == self.transformer_configuration()

    def test_config_type_is_converted(self):
        from diffusers import TorchAoConfig

        arguments = {
            "load_components": {
                "quantization_config": {
                    "transformer": {
                        "configuration": {"config_type": "TorchAoConfig"},
                        "arguments": {},
                    }
                }
            }
        }
        realize_args(arguments)

        quantization_config = arguments["load_components"]["quantization_config"]
        assert quantization_config["transformer"]["configuration"]["config_type"] is (
            TorchAoConfig
        )


class TestConfigureComponents:
    """Test placement of the components a pipeline loaded for itself"""

    def configure(self, components, pipeline=None, device="cuda"):
        pipeline = pipeline if pipeline is not None else MagicMock()
        # Patched at its source - pipeline.py imports it at the call site
        with patch("diffusers.hooks.apply_group_offloading") as group_offload:
            configure_components(pipeline, {"components": components}, device)

        return pipeline, group_offload

    def test_group_offload_targets_the_pipeline_device(self):
        pipeline, group_offload = self.configure(
            {
                "transformer": {
                    "group_offload": {
                        "offload_type": "block_level",
                        "num_blocks_per_group": 1,
                        "use_stream": True,
                    }
                }
            }
        )

        group_offload.assert_called_once_with(
            pipeline.transformer,
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
        )

    def test_a_dotted_name_offloads_a_module_inside_a_component(self):
        pipeline, group_offload = self.configure(
            {"text_encoder.model": {"group_offload": {"offload_type": "leaf_level"}}}
        )

        assert group_offload.call_args.args[0] is pipeline.text_encoder.model

    def test_a_component_can_be_moved_to_a_device(self):
        pipeline, _ = self.configure({"audio_vae": {"device": "cuda"}})

        pipeline.audio_vae.to.assert_called_once_with("cuda")

    def test_components_are_left_alone_by_default(self):
        pipeline, group_offload = self.configure({})

        group_offload.assert_not_called()
        pipeline.to.assert_not_called()

    def test_an_unknown_component_raises(self):
        pipeline = MagicMock(spec=["vae"])

        with pytest.raises(ValueError, match="transformer"):
            self.configure({"transformer": {"device": "cuda"}}, pipeline)

    def test_get_component_follows_a_dotted_path(self):
        pipeline = MagicMock()
        assert get_component(pipeline, "text_encoder.model") is (
            pipeline.text_encoder.model
        )


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
