"""
Unit tests for component-name discovery and cache wiring in pipeline definitions
"""

from unittest.mock import MagicMock

from dw.pipeline_processors.pipeline import (
    declared_component_names,
    enable_cache_on_transformer,
    optional_component_names,
)


class TestDeclaredComponentNames:
    """A component outside the known list is loaded, not silently dropped"""

    def test_known_names_are_always_included(self):
        names = declared_component_names({})
        assert names == optional_component_names

    def test_component_shaped_keys_are_detected(self):
        definition = {
            "configuration": {"component_type": "SomePipeline"},
            "text_encoder_4": {
                "configuration": {},
                "from_pretrained_arguments": {"model_name": "some/repo"},
                "quantization_config": {"config_type": "{nf4}"},
            },
        }
        names = declared_component_names(definition)
        assert "text_encoder_4" in names

    def test_scheduler_is_not_a_component(self):
        # A scheduler carries from_config_args, not from_pretrained_arguments
        definition = {
            "scheduler": {
                "configuration": {"scheduler_type": "DDPMScheduler"},
                "from_config_args": {},
            }
        }
        names = declared_component_names(definition)
        assert "scheduler" not in names

    def test_reserved_keys_are_never_components(self):
        definition = {
            "from_pretrained_arguments": {"model_name": "some/repo"},
            "arguments": {"prompt": "a cat"},
            "loras": [],
        }
        names = declared_component_names(definition)
        assert names == optional_component_names

    def test_plain_values_are_not_components(self):
        definition = {"seed": 42, "vocoder": "not a dict shape"}
        names = declared_component_names(definition)
        assert "vocoder" not in names

    def test_vocoder_with_pretrained_arguments_is_detected(self):
        definition = {
            "vocoder": {
                "configuration": {},
                "from_pretrained_arguments": {"model_name": "some/vocoder"},
            }
        }
        assert "vocoder" in declared_component_names(definition)


class TestFasterCacheWiring:
    """FasterCache needs a current-timestep callback JSON cannot express"""

    def test_callback_is_wired_to_the_pipeline(self):
        from diffusers import FasterCacheConfig

        config = FasterCacheConfig()
        assert config.current_timestep_callback is None

        pipeline = MagicMock()
        pipeline._current_timestep = 17

        enable_cache_on_transformer(pipeline, config)

        pipeline.transformer.enable_cache.assert_called_once_with(config)
        assert config.current_timestep_callback() == 17

    def test_an_explicit_callback_is_left_alone(self):
        from diffusers import FasterCacheConfig

        callback = lambda: 3
        config = FasterCacheConfig(current_timestep_callback=callback)

        enable_cache_on_transformer(MagicMock(), config)

        assert config.current_timestep_callback is callback
