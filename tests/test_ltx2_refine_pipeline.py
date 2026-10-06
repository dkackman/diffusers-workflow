"""LTX2RefinePipeline (#638): `video` is VAE-encoded at width x height and
handed to LTX2Pipeline as `latents`.

The VAE is a real AutoencoderKLLTX2Video, shrunk to a few channels but with
the real 32x spatial / 8x temporal compression, so the encode under test is
the one a run does. The denoise itself is LTX2Pipeline's and is not re-run
here: the forwarding tests stop at the parent's __call__.
"""

import inspect
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from diffusers import AutoencoderKLLTX2Video, LTX2Pipeline

from dw.community_pipelines.pipeline_ltx2_refine import LTX2RefinePipeline
from dw.introspection import (
    describe_pipeline,
    list_pipelines,
    load_allowed_class,
    unknown_call_arguments,
)

NAME = "dw.community_pipelines.pipeline_ltx2_refine.LTX2RefinePipeline"
LATENT_CHANNELS = 8


@pytest.fixture(scope="module")
def pipe():
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(
        latent_channels=LATENT_CHANNELS,
        block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8),
        layers_per_block=(1, 1, 1, 1, 1),
        decoder_layers_per_block=(1, 1, 1, 1),
        decoder_inject_noise=(False, False, False, False),
        spatial_compression_ratio=32,
        temporal_compression_ratio=8,
    )
    components = {
        name: None
        for name in inspect.signature(LTX2Pipeline.__init__).parameters
        if name not in ("self", "vae")
    }
    return LTX2RefinePipeline(vae=vae, **components)


def clip(frames, width=100, height=60):
    return [Image.new("RGB", (width, height), (i * 10, 50, 90)) for i in range(frames)]


def expected_shape(width, height, num_frames):
    return (1, LATENT_CHANNELS, (num_frames - 1) // 8 + 1, height // 32, width // 32)


class TestEncode:
    def test_video_gives_latents_for_width_height_num_frames(self, pipe):
        latents = pipe.encode_video(clip(20), width=96, height=64, num_frames=17)
        assert tuple(latents.shape) == expected_shape(96, 64, 17)
        assert latents.dtype == torch.float32

    def test_the_source_size_does_not_matter_only_width_and_height(self, pipe):
        latents = pipe.encode_video(
            clip(9, 50, 200), width=128, height=32, num_frames=9
        )
        assert tuple(latents.shape) == expected_shape(128, 32, 9)

    def test_num_frames_defaults_to_the_clip_floored_to_the_grid(self, pipe):
        latents = pipe.encode_video(clip(20), width=96, height=64)
        assert tuple(latents.shape) == expected_shape(96, 64, 17)

    def test_the_latents_are_left_unnormalized_for_the_parent(self, pipe):
        """LTX2Pipeline.prepare_latents normalizes 5-D latents itself, so
        the encode must hand over the VAE's raw output, as the upsampler
        does - normalizing here would do it twice."""
        frames = clip(9)
        generator = torch.Generator().manual_seed(1)
        ours = pipe.encode_video(frames, width=96, height=64, generator=generator)
        pixels = pipe.video_processor.preprocess_video(frames, height=64, width=96)
        generator = torch.Generator().manual_seed(1)
        raw = pipe.vae.encode(pixels).latent_dist.sample(generator)
        torch.testing.assert_close(ours, raw.float())

    def test_the_parent_accepts_the_encoded_latents(self, pipe):
        """The real hand-off: LTX2Pipeline.prepare_latents normalizes, packs
        and renoises what encode_video returns."""
        latents = pipe.encode_video(clip(17), width=96, height=64)
        packed = pipe.prepare_latents(
            num_channels_latents=LATENT_CHANNELS,
            noise_scale=0.909375,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
            latents=latents,
        )
        _, channels, frames, height, width = latents.shape
        assert tuple(packed.shape) == (1, frames * height * width, channels)

    def test_an_off_grid_num_frames_is_refused(self, pipe):
        with pytest.raises(ValueError, match="8 \\* n \\+ 1"):
            pipe.encode_video(clip(20), width=96, height=64, num_frames=16)

    def test_num_frames_past_the_clip_is_refused(self, pipe):
        with pytest.raises(ValueError, match="only 9 frames"):
            pipe.encode_video(clip(9), width=96, height=64, num_frames=17)

    def test_a_clip_shorter_than_one_grid_step_is_refused(self, pipe):
        with pytest.raises(ValueError, match="at least 9"):
            pipe.encode_video(clip(5), width=96, height=64)


class TestCall:
    def test_video_and_latents_together_are_refused(self, pipe):
        with patch.object(LTX2Pipeline, "__call__") as parent:
            with pytest.raises(ValueError) as raised:
                pipe(
                    prompt="a cat",
                    video=clip(9),
                    latents=torch.zeros(1, LATENT_CHANNELS, 2, 2, 3),
                    width=96,
                    height=64,
                )
        assert "`video`" in str(raised.value) and "`latents`" in str(raised.value)
        parent.assert_not_called()

    def test_video_is_passed_to_the_parent_as_latents(self, pipe):
        with patch.object(LTX2Pipeline, "__call__", return_value="out") as parent:
            result = pipe(
                prompt="a cat",
                video=clip(20),
                width=96,
                height=64,
                num_frames=17,
                noise_scale=0.909375,
                sigmas=[0.909375, 0.725, 0.421875],
            )
        assert result == "out"
        call = parent.call_args.kwargs
        assert tuple(call["latents"].shape) == expected_shape(96, 64, 17)
        assert call["num_frames"] == 17
        assert call["noise_scale"] == 0.909375
        assert call["sigmas"] == [0.909375, 0.725, 0.421875]
        assert "video" not in call

    def test_num_frames_omitted_takes_the_clips_length(self, pipe):
        with patch.object(LTX2Pipeline, "__call__") as parent:
            pipe(prompt="a cat", video=clip(20), width=96, height=64)
        assert parent.call_args.kwargs["num_frames"] == 17

    def test_latents_are_repeated_per_video(self, pipe):
        with patch.object(LTX2Pipeline, "__call__") as parent:
            pipe(
                prompt=["a cat", "a dog"],
                video=clip(9),
                width=96,
                height=64,
                num_videos_per_prompt=2,
            )
        assert parent.call_args.kwargs["latents"].shape[0] == 4

    def test_without_video_the_call_is_the_parents(self, pipe):
        with patch.object(LTX2Pipeline, "__call__", return_value="out") as parent:
            assert pipe(prompt="a cat", width=96) == "out"
        assert parent.call_args.kwargs == {"prompt": "a cat", "width": 96}


class TestIntrospection:
    def test_it_is_listed_with_the_pipelines(self):
        assert NAME in list_pipelines()

    def test_its_signature_has_video_and_the_parents_arguments(self):
        description = describe_pipeline(NAME)
        parameters = {p["name"]: p for p in description["parameters"]}
        assert {"video", "latents", "noise_scale", "sigmas", "width"} <= set(parameters)
        assert "VAE-encoded" in parameters["video"]["description"]
        assert not description["accepts_kwargs"]

    def test_a_misspelt_argument_is_still_caught(self):
        assert unknown_call_arguments(NAME, ["video", "noise_scael"]) == ["noise_scael"]

    def test_only_the_shipped_modules_resolve(self):
        assert load_allowed_class(NAME) is LTX2RefinePipeline
        for blocked in (
            "dw.community_pipelines.os.path",
            "dw.community_pipelines.LTX2RefinePipeline",
            "dw.security.validate_path",
        ):
            with pytest.raises(ValueError):
                load_allowed_class(blocked)


def _with_doc(doc):
    def call(self):
        pass

    call.__doc__ = doc
    return call


def test_call_doc_survives_an_indented_parent_docstring():
    """A parent __call__ docstring that still carries its source indent (an
    older Python, or a wrapped call) must not fold into `video`'s entry."""
    from dw.community_pipelines import pipeline_ltx2_refine as module
    from dw.introspection import _parse_docstring_args

    class Parent:
        def __call__(self):
            pass

    # Args sits deeper than a stray shallow line, so cleandoc cannot strip it
    Parent.__call__.__doc__ = (
        "\n        Summary.\n\n        Args:\n"
        "            prompt (`str`):\n                The prompt.\n"
        "            height (`int`):\n                The height.\n"
        "    Stray.\n"
    )
    with patch.object(module, "LTX2Pipeline", Parent):
        documented, _ = _parse_docstring_args(
            inspect.getdoc(
                type("C", (), {"__call__": _with_doc(module._call_doc())}).__call__
            )
        )
    assert documented["prompt"]["description"] == "The prompt."
    assert documented["height"]["description"] == "The height."
    assert "The prompt" not in documented["video"]["description"]


def test_bare_community_name_hints_at_its_dotted_path():
    from dw.introspection import load_allowed_class

    with pytest.raises(
        ValueError, match="did you mean.*pipeline_ltx2_refine.LTX2RefinePipeline"
    ):
        load_allowed_class("LTX2RefinePipeline")
