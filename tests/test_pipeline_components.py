"""
Unit tests for component-name discovery and cache wiring in pipeline definitions
"""

import io
import torch
from unittest.mock import MagicMock

import pytest

from dw.pipeline_processors.components import (
    configure_components,
    enable_cache_on_transformer,
    get_block_configs,
    load_component,
)
from dw.pipeline_processors.pipeline import (
    Pipeline,
    declared_component_names,
    optional_component_names,
    warn_if_safety_checker_blanked,
)


class TestDeclaredComponentNames:
    """A component outside the known list is loaded, not silently dropped"""

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

        def callback():
            return 3

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

    def test_a_component_this_step_loaded_is_still_placed(self, all_backends_available):
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
        from dw.pipeline_processors.components import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {"components": {"diffusion_decoder": {"enable_tiling": True}}},
            "cpu",
        )

        assert decoder.tiling == {}

    def test_an_object_passes_the_tile_sizes_through(self):
        from dw.pipeline_processors.components import configure_components

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
        from dw.pipeline_processors.components import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {"components": {"diffusion_decoder": {"device": "cpu"}}},
            "cpu",
        )

        assert decoder.tiling is None

    def test_a_component_that_cannot_tile_says_so(self):
        from dw.pipeline_processors.components import configure_components

        class Plain:
            def to(self, device):
                return self

        with pytest.raises(ValueError, match="does not support tiling"):
            configure_components(
                self._Pipeline(connectors=Plain()),
                {"components": {"connectors": {"enable_tiling": True}}},
                "cpu",
            )


class TestComponentAttnProcessor:
    """The attention processor of a component that is not the unet or transformer"""

    class _Processor:
        pass

    class _Decoder:
        def __init__(self):
            self.processor = None

        def set_attn_processor(self, processor):
            self.processor = processor

        def to(self, device):
            return self

    class _Pipeline:
        def __init__(self, **components):
            for name, component in components.items():
                setattr(self, name, component)

    def test_the_named_type_is_constructed_and_set(self):
        from dw.pipeline_processors.components import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {
                "components": {
                    "diffusion_decoder": {"attn_processor_type": self._Processor}
                }
            },
            "cpu",
        )

        assert isinstance(decoder.processor, self._Processor)

    def test_omitted_leaves_the_component_alone(self):
        from dw.pipeline_processors.components import configure_components

        decoder = self._Decoder()
        configure_components(
            self._Pipeline(diffusion_decoder=decoder),
            {"components": {"diffusion_decoder": {"device": "cpu"}}},
            "cpu",
        )

        assert decoder.processor is None

    def test_a_component_that_takes_no_processor_says_so(self):
        from dw.pipeline_processors.components import configure_components

        class Plain:
            def to(self, device):
                return self

        with pytest.raises(ValueError, match="does not take an attention processor"):
            configure_components(
                self._Pipeline(connectors=Plain()),
                {
                    "components": {
                        "connectors": {"attn_processor_type": self._Processor}
                    }
                },
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


class TestSafetyCheckerWarning:
    """A blanked image looks like a repeated result - it has to say so."""

    def test_warns_when_the_safety_checker_blanked_an_image(self, caplog):
        output = MagicMock()
        output.nsfw_content_detected = [False, True]

        with caplog.at_level("WARNING", logger="dw"):
            warn_if_safety_checker_blanked(output)

        assert "safety checker" in caplog.text.lower()
        assert "1 of 2" in caplog.text

    def test_silent_when_nothing_was_flagged(self, caplog):
        output = MagicMock()
        output.nsfw_content_detected = [False, False]

        with caplog.at_level("WARNING", logger="dw"):
            warn_if_safety_checker_blanked(output)

        assert caplog.text == ""

    def test_silent_for_a_pipeline_without_a_safety_checker(self, caplog):
        output = MagicMock(spec=[])

        with caplog.at_level("WARNING", logger="dw"):
            warn_if_safety_checker_blanked(output)

        assert caplog.text == ""

    def test_it_reaches_the_run_as_a_warning_event(self):
        """The job's status is 'succeeded' and its file is solid black, so a
        consumer over the API or MCP - which sees the warnings list and
        nothing else - is the one party that cannot tell the difference
        (#133). The log alone never got there."""
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            output = MagicMock()
            output.nsfw_content_detected = [True]
            warn_if_safety_checker_blanked(output)
        finally:
            deactivate_context(token)

        warnings = [e for e in events if e["event"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["kind"] == "safety_checker_blanked"
        assert warnings[0]["blanked"] == 1
        assert "solid black" in warnings[0]["message"]


class TestAudiosSampleRate:
    """Audio-only pipelines put the waveform on `.audios` and the rate on a
    component config - AudioLDM2's vocoder, StableAudio's VAE. The workflow
    should not have to know which."""

    def _pipeline(self, **components):
        from types import SimpleNamespace

        return SimpleNamespace(**components)

    def _output(self):
        from types import SimpleNamespace

        import numpy

        return SimpleNamespace(audios=numpy.zeros((1, 2, 100), dtype="float32"))

    def test_a_vocoder_rate_is_recorded_for_audios(self):
        from types import SimpleNamespace

        from dw.pipeline_processors.pipeline import attach_audio_sample_rate

        pipeline = self._pipeline(
            vocoder=SimpleNamespace(config=SimpleNamespace(sampling_rate=16000))
        )
        output = self._output()

        attach_audio_sample_rate(pipeline, output)

        assert output.audio_sample_rate == 16000

    def test_a_vae_rate_is_recorded_for_audios(self):
        from types import SimpleNamespace

        from dw.pipeline_processors.pipeline import attach_audio_sample_rate

        pipeline = self._pipeline(
            vae=SimpleNamespace(config=SimpleNamespace(sampling_rate=44100))
        )
        output = self._output()

        attach_audio_sample_rate(pipeline, output)

        assert output.audio_sample_rate == 44100

    def test_no_rate_anywhere_records_nothing(self, caplog):
        from dw.pipeline_processors.pipeline import attach_audio_sample_rate

        output = self._output()

        with caplog.at_level("WARNING", logger="dw"):
            attach_audio_sample_rate(self._pipeline(), output)

        assert not hasattr(output, "audio_sample_rate")
        assert "no component reports its sample rate" in caplog.text

    def test_a_vae_rate_is_not_used_for_video_plus_audio(self):
        # A video+audio pipeline's `vae` is a video VAE - its config is not
        # where an audio rate lives, so it must not be consulted for `.audio`
        from types import SimpleNamespace

        from dw.pipeline_processors.pipeline import attach_audio_sample_rate

        pipeline = self._pipeline(
            vae=SimpleNamespace(config=SimpleNamespace(sampling_rate=44100))
        )
        output = SimpleNamespace(audio=[0.0, 0.0])

        attach_audio_sample_rate(pipeline, output)

        assert not hasattr(output, "audio_sample_rate")

    def test_an_audio_vae_rate_is_recorded_for_video_plus_audio(self):
        # Kandinsky 6's vocoder config carries no rate; its MMAudioVAE does,
        # as `sample_rate` (#663)
        from types import SimpleNamespace

        from dw.pipeline_processors.pipeline import attach_audio_sample_rate

        pipeline = self._pipeline(
            vocoder=SimpleNamespace(config=SimpleNamespace(num_mels=128)),
            audio_vae=SimpleNamespace(config=SimpleNamespace(sample_rate=44100)),
        )
        output = SimpleNamespace(audio=[0.0, 0.0])

        attach_audio_sample_rate(pipeline, output)

        assert output.audio_sample_rate == 44100

    def test_a_vocoder_rate_wins_over_the_audio_vae_rate(self):
        # LTX-2's audio VAE works at a lower rate than its vocoder outputs;
        # the vocoder's is the rate of the waveform
        from types import SimpleNamespace

        from dw.pipeline_processors.pipeline import attach_audio_sample_rate

        pipeline = self._pipeline(
            vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000)),
            audio_vae=SimpleNamespace(config=SimpleNamespace(sample_rate=16000)),
        )
        output = SimpleNamespace(audio=[0.0, 0.0])

        attach_audio_sample_rate(pipeline, output)

        assert output.audio_sample_rate == 24000


class TestRemoteTextEncoderResponse:
    """A retired endpoint answers with an HTML page. The error has to say
    that, not hand back torch's unpickling complaint about it."""

    def _response(self, status, content_type, body=b"<!DOCTYPE html>"):
        response = MagicMock()
        response.ok = status < 400
        response.status_code = status
        response.headers = {"Content-Type": content_type}
        response.content = body
        return response

    def test_an_html_page_is_refused_with_the_url_named(self, monkeypatch):
        from dw.pipeline_processors import remote

        monkeypatch.setattr(remote, "get_token", lambda: "tok")
        monkeypatch.setattr(
            remote,
            "safe_post",
            lambda *a, **k: self._response(206, "text/html; charset=utf-8"),
        )

        with pytest.raises(RuntimeError, match="https://example.invalid/predict"):
            remote.remote_text_encoder(
                ["a mug"], "https://example.invalid/predict", "cpu"
            )

    def test_an_error_status_is_reported_by_its_status(self, monkeypatch):
        """safe_post raises on an error status; the message still names it."""
        import requests

        from dw.pipeline_processors import remote

        def _raise(*a, **k):
            raise requests.HTTPError(response=self._response(503, "text/plain"))

        monkeypatch.setattr(remote, "get_token", lambda: "tok")
        monkeypatch.setattr(remote, "safe_post", _raise)

        with pytest.raises(RuntimeError, match="HTTP 503"):
            remote.remote_text_encoder(
                ["a mug"], "https://example.invalid/predict", "cpu"
            )

    def test_a_tensor_response_is_loaded(self, monkeypatch):
        from dw.pipeline_processors import remote

        buffer = io.BytesIO()
        torch.save(torch.zeros(2), buffer)
        monkeypatch.setattr(remote, "get_token", lambda: "tok")
        monkeypatch.setattr(
            remote,
            "safe_post",
            lambda *a, **k: self._response(
                200, "application/octet-stream", buffer.getvalue()
            ),
        )

        embeds = remote.remote_text_encoder(["a mug"], "https://example.invalid", "cpu")

        assert embeds.shape == (2,)


class TestFailedLoadIsTornDown:
    """#72: a load that raises partway left what it had built resident.

    Everything after the pipeline itself can fail - a quantized matmul pass, a
    LoRA, a group-offload placement - and each failure used to hand the
    workflow an exception with several GB still on the device, which the next
    attempt then loaded on top of.
    """

    def _definition(self):
        return {
            "configuration": {"component_type": "{MockPipeline}"},
            "from_pretrained_arguments": {"model_name": "some/repo"},
            "arguments": {"prompt": "a cat"},
        }

    def _pipeline(self):
        return Pipeline(self._definition(), 42, "cpu")

    def test_a_load_that_fails_after_the_pipeline_is_built_drops_it(self, monkeypatch):
        built = MagicMock()
        monkeypatch.setattr(
            "dw.pipeline_processors.pipeline.load_component",
            lambda *args, **kwargs: built,
        )
        monkeypatch.setattr(
            Pipeline,
            "configure_loaded_components",
            lambda self: (_ for _ in ()).throw(RuntimeError("placement failed")),
        )
        emptied = []
        monkeypatch.setattr(
            "dw.pipeline_processors.pipeline.empty_device_cache",
            lambda *args, **kwargs: emptied.append(True),
        )

        pipeline = self._pipeline()
        with pytest.raises(RuntimeError, match="placement failed"):
            pipeline.load({})

        assert pipeline.pipeline is None, "the half-loaded pipeline must be dropped"
        assert emptied, "the device cache is emptied on the way out"

    def test_a_failed_load_unpublishes_what_it_had_shared(self, monkeypatch):
        """A component published before the failure would otherwise be handed
        to a later step as if it belonged to a pipeline that exists."""
        definition = self._definition()
        definition["shared_components"] = ["vae"]
        monkeypatch.setattr(
            "dw.pipeline_processors.pipeline.load_component",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "dw.pipeline_processors.pipeline.load_loras",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lora failed")),
        )
        monkeypatch.setattr(
            "dw.pipeline_processors.pipeline.empty_device_cache",
            lambda *args, **kwargs: None,
        )

        shared = {}
        pipeline = Pipeline(definition, 42, "cpu")
        with pytest.raises(RuntimeError, match="lora failed"):
            pipeline.load(shared)

        assert shared == {}


class TestImageCrfMismatchDiagnosis:
    """#511: diffusers' `image_crf` re-compression error names a re-encoding
    knob that does not fix an argument holding a video where an image was
    required - the caller needs a still, not a different crf."""

    def _definition(self):
        return {
            "configuration": {"component_type": "{MockPipeline}"},
            "from_pretrained_arguments": {"model_name": "some/repo"},
            "arguments": {"image": "variable:image"},
        }

    def test_an_audiovideo_argument_is_named_in_the_rewritten_error(self, monkeypatch):
        from dw.media_types import AudioVideo

        def _raise(self, arguments):
            raise ValueError("re-compression requires you to set `image_crf`")

        monkeypatch.setattr(Pipeline, "_run_once", _raise)

        pipeline = Pipeline(self._definition(), 42, "cpu")
        pipeline.pipeline = MagicMock()

        video = AudioVideo(frames=[], audio=None, sample_rate=None)
        with pytest.raises(ValueError, match="expected an image but received a video"):
            pipeline.run({"image": video})

    def test_names_every_offending_argument(self):
        from dw.media_types import AudioVideo
        from dw.pipeline_processors.pipeline import _diagnose_image_crf_error

        video = AudioVideo(frames=[], audio=None, sample_rate=None)
        error = ValueError("re-compression requires you to set `image_crf`")

        diagnosed = _diagnose_image_crf_error(
            error, {"image": video, "conditioning_image": video, "prompt": "a cat"}
        )

        assert diagnosed is not None
        assert "conditioning_image" in str(diagnosed)
        assert "image" in str(diagnosed)
        assert "get_last_frame" in str(diagnosed)
        assert "video_frames" in str(diagnosed)

    def test_a_different_value_error_is_left_alone(self):
        from dw.pipeline_processors.pipeline import _diagnose_image_crf_error

        error = ValueError("some unrelated failure")
        assert _diagnose_image_crf_error(error, {"image": "not a video"}) is None

    def test_the_same_message_with_no_video_argument_is_left_alone(self):
        """The text alone isn't enough - it has to actually be this mismatch."""
        from dw.pipeline_processors.pipeline import _diagnose_image_crf_error

        error = ValueError("re-compression requires you to set `image_crf`")
        assert _diagnose_image_crf_error(error, {"image": "a/path.png"}) is None


class LoadablePipeline:
    """A stand-in for a diffusers pipeline class: constructed directly (no
    model_name, so no download), with the adapter and offload methods load()
    calls. Everything load() does to the definition around it is real"""

    def __init__(self, **constructor_arguments):
        self.constructor_arguments = constructor_arguments
        self.loras = []
        self.adapters = None
        self.ip_adapter = None
        self.group_offload = None

    def load_lora_weights(self, model_name, adapter_name=None, **kwargs):
        if "unexpected" in kwargs:
            raise TypeError("load_lora_weights() got an unexpected keyword")
        self.loras.append((model_name, adapter_name, kwargs))

    def set_adapters(self, names, weights):
        self.adapters = (list(names), list(weights))

    def load_ip_adapter(self, model_name, **kwargs):
        self.ip_adapter = (model_name, kwargs)

    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale

    def enable_group_offload(self, **kwargs):
        self.group_offload = kwargs


class TestLoadLeavesTheDefinitionAlone:
    """Loading edits what it is handed - group offload turns device names into
    torch.device objects and pops the stream flags off CUDA, load_loras and
    load_ip_adapter pop their keys, and the generator is set on the argument
    template. None of it may reach the workflow's own step definition"""

    @staticmethod
    def step_definition(image):
        return {
            "name": "generate",
            "pipeline": {
                "configuration": {
                    "component_type": LoadablePipeline,
                    "group_offload": {
                        "onload_device": "cpu",
                        "offload_device": "cpu",
                        "offload_type": "leaf_level",
                        "use_stream": True,
                        "record_stream": True,
                    },
                },
                "from_pretrained_arguments": {"torch_dtype": torch.bfloat16},
                "loras": [
                    {
                        "model_name": "some/lora",
                        "adapter_name": "style",
                        "scale": 0.8,
                        "weight_name": "style.safetensors",
                    },
                    {"model_name": "other/lora"},
                ],
                "ip_adapter": {
                    "model_name": "some/ip-adapter",
                    "scale": 0.5,
                    "subfolder": "models",
                },
                "arguments": {"prompt": "a cat", "image": image},
            },
        }

    def test_a_real_load_leaves_the_step_definition_as_it_was(self):
        import copy

        step = self.step_definition("a/path.png")
        before = copy.deepcopy(step)

        pipeline = Pipeline(step["pipeline"], 42, "cpu")
        pipeline.load({})

        # The load really did its edits - on its own copy
        loaded = pipeline.pipeline
        assert loaded.loras[0] == (
            "some/lora",
            "style",
            {"weight_name": "style.safetensors"},
        )
        assert loaded.adapters == (["style", "1"], [0.8, 1.0])
        assert loaded.ip_adapter == ("some/ip-adapter", {"subfolder": "models"})
        assert "use_stream" not in loaded.group_offload
        assert loaded.group_offload["onload_device"] == torch.device("cpu")

        assert step == before

    def test_the_generator_never_reaches_the_workflows_arguments(self):
        image = object()
        step = self.step_definition(image)
        arguments = step["pipeline"]["arguments"]

        pipeline = Pipeline(step["pipeline"], 42, "cpu")
        pipeline.load({})

        assert isinstance(pipeline.argument_template["generator"], torch.Generator)
        assert "generator" not in arguments
        # A shallow copy: realized media is shared, not duplicated
        assert pipeline.argument_template["image"] is image

    def test_a_failed_load_leaves_the_step_definition_as_it_was(self):
        import copy

        step = self.step_definition("a/path.png")
        # The second lora names nothing load_lora_weights accepts, after the
        # first has already been popped
        step["pipeline"]["loras"][1]["unexpected"] = True
        before = copy.deepcopy(step)

        pipeline = Pipeline(step["pipeline"], 42, "cpu")
        with pytest.raises(TypeError):
            pipeline.load({})

        assert step["pipeline"]["loras"] == before["pipeline"]["loras"]
        assert "generator" not in step["pipeline"]["arguments"]
