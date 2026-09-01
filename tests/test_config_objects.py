"""
Unit tests for building pipeline configuration objects from workflow JSON.

These builders sit between the declarative JSON and the diffusers constructors,
so a mistake here surfaces only after a multi-gigabyte model load - worth
catching at the dict level instead.
"""

import pytest
import torch

import dw

from dw.pipeline_processors.config_objects import (
    create_quantization_config,
    get_cache_configuration,
    get_group_offload_configuration,
    get_load_components_arguments,
    get_quantization_configuration,
)


class FakeConfig:
    """Stands in for a quantization config class resolved by realize_args."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def quantization_definition(**arguments):
    return {
        "configuration": {"config_type": FakeConfig},
        "arguments": arguments,
    }


class TestQuantizationConfiguration:
    def test_no_quantization_config_returns_none(self):
        assert get_quantization_configuration({}) is None
        assert get_quantization_configuration({"quantization_config": None}) is None

    def test_the_config_type_is_constructed_with_its_arguments(self):
        config = get_quantization_configuration(
            {"quantization_config": quantization_definition(load_in_4bit=True)}
        )

        assert isinstance(config, FakeConfig)
        assert config.kwargs == {"load_in_4bit": True}

    def test_class_valued_arguments_are_instantiated(self):
        # realize_args turns "torchao.quantization.Int8WeightOnlyConfig" into the
        # class; TorchAoConfig wants an instance, so the builder calls it
        config = create_quantization_config(
            quantization_definition(quant_type=FakeConfig, group_size=64)
        )

        assert isinstance(config.kwargs["quant_type"], FakeConfig)
        assert config.kwargs["group_size"] == 64

    def test_a_failing_constructor_propagates(self):
        class Exploding:
            def __init__(self, **kwargs):
                raise TypeError("unexpected keyword")

        with pytest.raises(TypeError):
            create_quantization_config(
                {
                    "configuration": {"config_type": Exploding},
                    "arguments": {"bogus": 1},
                }
            )


class TestLoadComponentsArguments:
    def test_a_pipeline_without_load_components_returns_none(self):
        assert get_load_components_arguments({"component_type": "FluxPipeline"}) is None

    def test_arguments_pass_through_when_nothing_is_quantized(self):
        arguments = get_load_components_arguments(
            {"load_components": {"names": ["transformer"]}}
        )

        assert arguments == {"names": ["transformer"]}

    def test_each_named_component_gets_its_own_config_object(self):
        arguments = get_load_components_arguments(
            {
                "load_components": {
                    "quantization_config": {
                        "transformer": quantization_definition(bits=4),
                        "text_encoder": quantization_definition(bits=8),
                    }
                }
            }
        )
        built = arguments["quantization_config"]

        assert built["transformer"].kwargs == {"bits": 4}
        assert built["text_encoder"].kwargs == {"bits": 8}

    def test_the_source_configuration_is_not_mutated(self):
        # Pipeline configurations are reused across cached loads and REPL reruns;
        # replacing the definition in place would leave a built object behind
        # where the next call expects a dict
        definition = quantization_definition(bits=4)
        configuration = {
            "load_components": {"quantization_config": {"transformer": definition}}
        }

        get_load_components_arguments(configuration)

        assert (
            configuration["load_components"]["quantization_config"]["transformer"]
            is definition
        )


class TestGroupOffloadConfiguration:
    def test_no_group_offload_returns_none(self):
        assert get_group_offload_configuration({}, "cuda") is None

    def test_device_strings_become_torch_devices(self, monkeypatch):
        # Stand CUDA up as present so the conversion is what is under test rather
        # than the translation a machine without CUDA would apply
        monkeypatch.setattr(dw, "backend_available", lambda backend: True)

        config = get_group_offload_configuration(
            {"group_offload": {"onload_device": "cuda:1", "offload_device": "cpu"}},
            "cuda",
        )

        assert config["onload_device"] == torch.device("cuda:1")
        assert config["offload_device"] == torch.device("cpu")

    def test_onload_defaults_to_the_pipeline_device(self, monkeypatch):
        # The default must follow DW_DEVICE / the device setting, not a hardcoded
        # accelerator - "cuda:1" here stands for a resolved non-default device
        monkeypatch.setattr(dw, "backend_available", lambda backend: True)

        config = get_group_offload_configuration(
            {"group_offload": {"num_blocks_per_group": 1}}, "cuda:1"
        )

        assert config["onload_device"] == torch.device("cuda:1")

    def test_offload_defaults_to_cpu(self):
        config = get_group_offload_configuration(
            {"group_offload": {"num_blocks_per_group": 1}}, "cuda"
        )

        assert config["offload_device"] == torch.device("cpu")

    def test_other_keys_are_carried_through(self):
        config = get_group_offload_configuration(
            {"group_offload": {"num_blocks_per_group": 2, "use_stream": True}}, "cuda"
        )

        assert config["num_blocks_per_group"] == 2
        assert config["use_stream"] is True

    def test_building_twice_from_one_configuration_is_stable(self):
        # The builder writes device objects back into the configuration it was
        # given; a cached pipeline reloaded a second time must still work
        configuration = {"group_offload": {"onload_device": "cpu"}}

        first = get_group_offload_configuration(configuration, "cuda")
        second = get_group_offload_configuration(configuration, "cuda")

        assert first["onload_device"] == second["onload_device"] == torch.device("cpu")


class TestCacheConfiguration:
    def test_no_cache_returns_none(self):
        assert get_cache_configuration({}) is None
        assert get_cache_configuration({"cache": None}) is None

    def test_first_block_defaults_its_threshold(self):
        config = get_cache_configuration({"cache": {"type": "first_block"}})

        assert type(config).__name__ == "FirstBlockCacheConfig"
        assert config.threshold == 0.05

    def test_first_block_honors_an_explicit_threshold(self):
        config = get_cache_configuration(
            {"cache": {"type": "first_block", "threshold": 0.2}}
        )

        assert config.threshold == 0.2

    @pytest.mark.parametrize(
        "cache_type, expected",
        [("faster", "FasterCacheConfig"), ("text_kv", "TextKVCacheConfig")],
    )
    def test_argument_free_cache_types(self, cache_type, expected):
        config = get_cache_configuration({"cache": {"type": cache_type}})

        assert type(config).__name__ == expected

    def test_taylorseer_forwards_its_tuning_arguments(self):
        config = get_cache_configuration(
            {"cache": {"type": "taylorseer", "cache_interval": 4, "max_order": 2}}
        )

        assert type(config).__name__ == "TaylorSeerCacheConfig"
        assert config.cache_interval == 4
        assert config.max_order == 2

    def test_taylorseer_omitted_arguments_keep_the_diffusers_defaults(self):
        explicit = get_cache_configuration(
            {"cache": {"type": "taylorseer", "max_order": 3}}
        )
        bare = get_cache_configuration({"cache": {"type": "taylorseer"}})

        assert explicit.max_order == 3
        assert bare.cache_interval == explicit.cache_interval

    def test_an_unknown_cache_type_raises(self):
        with pytest.raises(ValueError, match="Unknown cache type: nope"):
            get_cache_configuration({"cache": {"type": "nope"}})

    def test_a_cache_block_without_a_type_raises(self):
        with pytest.raises(KeyError):
            get_cache_configuration({"cache": {"threshold": 0.05}})

    def test_mag_forwards_its_tuning_arguments(self):
        config = get_cache_configuration(
            {
                "cache": {
                    "type": "mag",
                    "mag_ratios": [1.0, 0.9, 0.8],
                    "num_inference_steps": 3,
                    "threshold": 0.1,
                    "max_skip_steps": 2,
                    "retention_ratio": 0.3,
                }
            }
        )

        assert type(config).__name__ == "MagCacheConfig"
        assert config.threshold == 0.1
        assert config.max_skip_steps == 2
        assert config.retention_ratio == 0.3
        assert config.mag_ratios.tolist() == pytest.approx([1.0, 0.9, 0.8])

    def test_mag_resolves_a_named_ratio_preset(self):
        # JSON cannot carry a torch tensor, so a preset shipped by diffusers is
        # named instead - "flux" -> FLUX_MAG_RATIOS
        config = get_cache_configuration(
            {"cache": {"type": "mag", "mag_ratios": "flux", "num_inference_steps": 20}}
        )

        assert len(config.mag_ratios) == 20

    def test_mag_preset_names_are_case_insensitive(self):
        lower = get_cache_configuration(
            {"cache": {"type": "mag", "mag_ratios": "flux"}}
        )
        upper = get_cache_configuration(
            {"cache": {"type": "mag", "mag_ratios": "FLUX"}}
        )

        assert lower.mag_ratios.tolist() == upper.mag_ratios.tolist()

    def test_an_unknown_ratio_preset_lists_the_available_ones(self):
        with pytest.raises(ValueError, match="Unknown mag_ratios preset: wan"):
            get_cache_configuration({"cache": {"type": "mag", "mag_ratios": "wan"}})

    def test_mag_ratios_are_interpolated_to_the_step_count(self):
        # diffusers resizes checkpoint ratios to num_inference_steps, so a preset
        # does not have to match the workflow's step count
        config = get_cache_configuration(
            {
                "cache": {
                    "type": "mag",
                    "mag_ratios": [1.0, 0.9],
                    "num_inference_steps": 4,
                }
            }
        )

        assert len(config.mag_ratios) == 4

    def test_mag_calibration_mode_needs_no_ratios(self):
        # Calibration is how a user obtains ratios for a new model in the first
        # place, so it must build without them
        config = get_cache_configuration({"cache": {"type": "mag", "calibrate": True}})

        assert config.calibrate is True
        assert config.mag_ratios is None

    def test_mag_without_ratios_or_calibration_still_raises(self):
        # The diffusers guard must stay reachable - it tells the user how to get
        # ratios for their checkpoint
        with pytest.raises(ValueError, match="mag_ratios"):
            get_cache_configuration({"cache": {"type": "mag"}})


def cache_schema():
    """The cache block of the workflow schema."""
    import json
    import os

    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dw",
        "workflow_schema.json",
    )
    with open(schema_path) as f:
        schema = json.load(f)

    return schema["$defs"]["pipeline_configuration"]["properties"]["cache"]


class TestCacheTypesMatchTheSchema:
    """The schema and get_cache_configuration are maintained separately.

    A type or argument declared in one and missing from the other is a workflow
    that validates cleanly and then fails after a multi-gigabyte model load -
    which is exactly how the mag cache type shipped unusable.
    """

    # The minimum each type needs beyond "type" to construct. Anything not
    # listed builds from its diffusers defaults alone.
    MINIMUM_ARGUMENTS = {"mag": {"mag_ratios": [1.0, 0.9]}}

    def test_every_schema_cache_type_actually_builds(self):
        declared = cache_schema()["properties"]["type"]["enum"]

        for cache_type in declared:
            arguments = {
                "type": cache_type,
                **self.MINIMUM_ARGUMENTS.get(cache_type, {}),
            }

            config = get_cache_configuration({"cache": arguments})

            assert config is not None, f"schema type {cache_type} built nothing"

    def test_every_forwarded_mag_argument_is_declared_in_the_schema(self):
        from dw.pipeline_processors.config_objects import _MAG_CACHE_KEYS

        declared = set(cache_schema()["properties"])

        assert set(_MAG_CACHE_KEYS) | {"mag_ratios"} <= declared

    def test_every_forwarded_taylorseer_argument_is_declared_in_the_schema(self):
        from dw.pipeline_processors.config_objects import _TAYLORSEER_CACHE_KEYS

        declared = set(cache_schema()["properties"])

        assert set(_TAYLORSEER_CACHE_KEYS) <= declared

    def test_forwarded_arguments_are_real_diffusers_config_fields(self):
        """A key the schema and the builder agree on but diffusers does not have
        raises TypeError at construction - the other half of the same drift."""
        import dataclasses

        from diffusers import MagCacheConfig, TaylorSeerCacheConfig
        from dw.pipeline_processors.config_objects import (
            _MAG_CACHE_KEYS,
            _TAYLORSEER_CACHE_KEYS,
        )

        for keys, config_class in (
            (set(_MAG_CACHE_KEYS) | {"mag_ratios"}, MagCacheConfig),
            (set(_TAYLORSEER_CACHE_KEYS), TaylorSeerCacheConfig),
        ):
            fields = {f.name for f in dataclasses.fields(config_class)}
            assert keys <= fields, f"{config_class.__name__} lacks {keys - fields}"
