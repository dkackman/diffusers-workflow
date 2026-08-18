"""
Unit tests for component-name discovery and cache wiring in pipeline definitions
"""

from unittest.mock import MagicMock

import pytest

from dw.pipeline_processors.pipeline import (
    Pipeline,
    configure_components,
    declared_component_names,
    enable_cache_on_transformer,
    get_block_configs,
    load_component,
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


class ModularLike:
    """Stands in for a ModularPipeline - built from its own component index, and
    given already-loaded components afterwards rather than as constructor arguments"""

    def __init__(self):
        self.constructor_arguments = {}
        self.registered = {}
        self.load_order = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        pipeline = cls()
        pipeline.constructor_arguments = kwargs
        return pipeline

    def update_components(self, **kwargs):
        self.registered.update(kwargs)
        self.load_order.append("update")

    def load_components(self, **kwargs):
        self.load_order.append("load")


class StandardLike:
    """Stands in for a DiffusionPipeline - components are constructor arguments"""

    def __init__(self):
        self.constructor_arguments = {}

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        pipeline = cls()
        pipeline.constructor_arguments = kwargs
        return pipeline


class TestReusedComponents:
    """A component an earlier step loaded reaches the step that reuses it"""

    def test_a_modular_pipeline_is_given_them_after_it_is_built(self):
        text_encoder = MagicMock()
        pipeline = load_component(
            "pipeline",
            {
                "component_type": ModularLike,
                "preserve_device_placement": True,
                "load_components": {"dtype": "bfloat16"},
            },
            {"model_name": "some/repo", "workflow": "ref2va"},
            "cpu",
            {"text_encoder": text_encoder},
        )

        assert pipeline.registered == {"text_encoder": text_encoder}
        assert "text_encoder" not in pipeline.constructor_arguments
        # Registered first, so load_components skips what is already there rather
        # than pulling a second copy of the weights
        assert pipeline.load_order == ["update", "load"]

    def test_a_standard_pipeline_takes_them_as_constructor_arguments(self):
        vae = MagicMock()
        pipeline = load_component(
            "pipeline",
            {"component_type": StandardLike, "preserve_device_placement": True},
            {"model_name": "some/repo"},
            "cpu",
            {"vae": vae},
        )

        assert pipeline.constructor_arguments["vae"] is vae

    def test_nothing_reused_leaves_the_arguments_alone(self):
        pipeline = load_component(
            "pipeline",
            {"component_type": ModularLike, "preserve_device_placement": True},
            {"model_name": "some/repo"},
            "cpu",
        )

        assert pipeline.registered == {}


class TestComponentSharingNames:
    """The sharing lists are read from the configuration as well as the pipeline"""

    def pipeline_for(self, definition):
        return Pipeline({**definition, "arguments": {}}, 0, "cpu", MagicMock())

    def test_the_configuration_is_read(self):
        pipeline = self.pipeline_for(
            {"configuration": {"shared_components": ["text_encoder"]}}
        )

        assert pipeline.component_names("shared_components") == ["text_encoder"]

    def test_the_pipeline_level_is_read(self):
        pipeline = self.pipeline_for({"shared_components": ["vae"]})

        assert pipeline.component_names("shared_components") == ["vae"]

    def test_reusing_something_never_shared_is_an_error(self):
        pipeline = self.pipeline_for(
            {"configuration": {"reused_components": ["text_encoder"]}}
        )

        with pytest.raises(ValueError, match="no earlier step shared it"):
            pipeline.resolve_reused_components({})

    def test_a_shared_component_resolves(self):
        text_encoder = MagicMock()
        pipeline = self.pipeline_for(
            {"configuration": {"reused_components": ["text_encoder"]}}
        )

        resolved = pipeline.resolve_reused_components({"text_encoder": text_encoder})

        assert resolved == {"text_encoder": text_encoder}


class TestConfigureReusedComponents:
    """A reused component keeps the placement the step that shared it gave it"""

    def test_a_reused_component_is_not_placed_again(self):
        pipeline = MagicMock()
        configuration = {"components": {"text_encoder": {"device": "cuda"}}}

        configure_components(pipeline, configuration, "cpu", {"text_encoder": None})

        pipeline.text_encoder.to.assert_not_called()

    def test_a_path_into_a_reused_component_is_skipped_too(self):
        pipeline = MagicMock()
        configuration = {"components": {"text_encoder.model": {"device": "cuda"}}}

        configure_components(pipeline, configuration, "cpu", {"text_encoder": None})

        pipeline.text_encoder.model.to.assert_not_called()

    def test_a_component_this_step_loaded_is_still_placed(self):
        pipeline = MagicMock()
        configuration = {"components": {"vae": {"device": "cuda"}}}

        configure_components(pipeline, configuration, "cpu", {"text_encoder": None})

        pipeline.vae.to.assert_called_once_with("cuda")


class TestBlockConfigs:
    """A modular pipeline's blocks declare configs of their own"""

    def pipeline_with(self, *names):
        pipeline = ModularLike()
        pipeline._config_specs = {name: MagicMock() for name in names}
        return pipeline

    def test_declared_configs_are_collected(self):
        pipeline = self.pipeline_with("canvas_short_edge", "canvas_max_pixels")
        configuration = {"configs": {"canvas_short_edge": 1024}}

        assert get_block_configs(configuration, pipeline) == {"canvas_short_edge": 1024}

    def test_a_config_the_pipeline_does_not_declare_is_an_error(self):
        pipeline = self.pipeline_with("canvas_short_edge")
        configuration = {"configs": {"canvas_shrot_edge": 1024}}

        with pytest.raises(ValueError, match="declares no config named"):
            get_block_configs(configuration, pipeline)

    def test_a_pipeline_that_takes_no_configs_is_an_error(self):
        configuration = {"configs": {"canvas_short_edge": 1024}}

        with pytest.raises(ValueError, match="only supported on modular pipelines"):
            get_block_configs(configuration, StandardLike())

    def test_no_configs_is_not_an_error_on_any_pipeline(self):
        assert get_block_configs({}, StandardLike()) == {}

    def test_they_reach_the_pipeline_with_the_reused_components(self):
        text_encoder = MagicMock()
        pipeline = load_component(
            "pipeline",
            {
                "component_type": ModularLike,
                "preserve_device_placement": True,
                "configs": {"canvas_short_edge": 1024},
            },
            {"model_name": "some/repo"},
            "cpu",
            {"text_encoder": text_encoder},
        )

        assert pipeline.registered == {
            "canvas_short_edge": 1024,
            "text_encoder": text_encoder,
        }


class TestComponentTiling:
    """Tiled decoding for a component that is not the one named 'vae'"""

    class _Decoder:
        def __init__(self):
            self.tiling = None

        def enable_tiling(self, **arguments):
            self.tiling = arguments

        def to(self, device):
            return self

    class _Pipeline:
        def __init__(self, **components):
            for name, component in components.items():
                setattr(self, name, component)

    def test_true_uses_the_model_default_tile_size(self):
        from dw.pipeline_processors.pipeline import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {"components": {"diffusion_decoder": {"enable_tiling": True}}},
            "cpu",
        )

        assert decoder.tiling == {}

    def test_an_object_passes_the_tile_sizes_through(self):
        from dw.pipeline_processors.pipeline import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {
                "components": {
                    "diffusion_decoder": {
                        "enable_tiling": {
                            "tile_sample_min_height": 512,
                            "tile_sample_stride_height": 448,
                        }
                    }
                }
            },
            "cpu",
        )

        assert decoder.tiling == {
            "tile_sample_min_height": 512,
            "tile_sample_stride_height": 448,
        }

    def test_omitted_leaves_the_component_alone(self):
        from dw.pipeline_processors.pipeline import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {"components": {"diffusion_decoder": {"device": "cpu"}}},
            "cpu",
        )

        assert decoder.tiling is None

    def test_a_component_that_cannot_tile_says_so(self):
        from dw.pipeline_processors.pipeline import configure_components

        class Plain:
            def to(self, device):
                return self

        with pytest.raises(ValueError, match="does not support tiling"):
            configure_components(
                self._Pipeline(connectors=Plain()),
                {"components": {"connectors": {"enable_tiling": True}}},
                "cpu",
            )


class TestDefinitionIsNotMutatedByLoading:
    """A loaded component belongs to the load, not to the workflow definition.

    The definition outlives every step, so a component stored in it is one the run
    holds until it ends - release_pipeline frees nothing, and a workflow that loads
    a second large model after releasing the first holds both at once.
    """

    DEFINITION = {
        "configuration": {"component_type": "SomePipeline"},
        "from_pretrained_arguments": {"model_name": "some/model"},
        "transformer": {
            "configuration": {"component_type": "SomeTransformer"},
            "from_pretrained_arguments": {"model_name": "some/model"},
        },
    }

    def _populate(self, monkeypatch, definition):
        from dw.pipeline_processors import pipeline as pipeline_module

        loaded = object()
        monkeypatch.setattr(
            pipeline_module,
            "load_component",
            lambda *arguments, **keywords: loaded,
        )
        pipeline = pipeline_module.Pipeline(definition, 42, "cpu")
        return pipeline.populate_from_pretrained_arguments("cpu", {}), loaded

    def test_the_loaded_component_reaches_the_pipeline_arguments(self, monkeypatch):
        import copy

        definition = copy.deepcopy(self.DEFINITION)

        arguments, loaded = self._populate(monkeypatch, definition)

        assert arguments["transformer"] is loaded

    def test_the_definition_does_not_hold_the_component(self, monkeypatch):
        import copy

        definition = copy.deepcopy(self.DEFINITION)

        _, loaded = self._populate(monkeypatch, definition)

        assert "transformer" not in definition["from_pretrained_arguments"]
        assert definition["from_pretrained_arguments"] == {"model_name": "some/model"}

    def test_a_second_load_still_has_its_model_name(self, monkeypatch):
        # load_component consumes 'model_name' out of the arguments it is handed,
        # so a definition it consumed from would load an empty model next time
        import copy

        definition = copy.deepcopy(self.DEFINITION)

        self._populate(monkeypatch, definition)
        self._populate(monkeypatch, definition)

        assert definition["transformer"]["from_pretrained_arguments"] == {
            "model_name": "some/model"
        }

    def test_remote_text_encoder_does_not_edit_the_definition(self, monkeypatch):
        import copy

        definition = copy.deepcopy(self.DEFINITION)
        definition["remote_text_encoder"] = {"url": "https://example.invalid"}

        arguments, _ = self._populate(monkeypatch, definition)

        assert arguments["text_encoder"] is None
        assert "text_encoder" not in definition["from_pretrained_arguments"]
