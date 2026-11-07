"""MiniMax-H3 fixed guide noise (#1166, #1174): the cosine taper, the noised
pixels, the noised encode in the guide layout step, the pre-run checks and the
call-time refusal of `guide_noise`."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dw import validation
from dw.guides import guide_noise_errors, guide_noise_warnings
from dw.pipeline_processors.h3_guides import insert_guides
from dw.pipeline_processors.h3_hold import insert_audio_hold
from dw.pipeline_processors.h3_rules import (
    GUIDE_AUDIO_ROWS,
    GUIDE_NOISE_INPUT,
    guide_noise_problem,
    guide_noise_taper,
)
from dw.pipeline_processors.pipeline import Pipeline

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from dw.pipeline_processors import h3_hold_steps  # noqa: E402
from dw.pipeline_processors.h3_guide_steps import (  # noqa: E402
    DwH3GuideLayoutStep,
    noised_guide_pixels,
)

ENCODER = "diffusers.modular_pipelines.minimax_h3.encoders.encode_vae_condition"
TEXT = 10
SIDE = 64
FRAMES = 124
AUDIO_C = 4


class TestTaper:
    def test_cosine_values(self):
        values = guide_noise_taper(5, 0.4)
        expected = [0.4 * 0.5 * (1 + math.cos(math.pi * i / 4)) for i in range(5)]
        assert values == pytest.approx(expected)

    def test_endpoints_are_exact(self):
        values = guide_noise_taper(9, 0.4)
        assert values[0] == 0.4
        assert values[-1] == 0.0

    def test_monotonically_falling(self):
        values = guide_noise_taper(12, 0.7)
        assert all(a > b for a, b in zip(values, values[1:]))

    def test_one_frame_is_clean(self):
        assert guide_noise_taper(1, 0.4) == [0.0]

    def test_no_frames(self):
        assert guide_noise_taper(0, 0.4) == []


class TestNoisedPixels:
    def pixels(self, n=5):
        return torch.full((n, 8, 8, 3), 128, dtype=torch.uint8)

    def test_last_frame_is_unchanged_exactly(self):
        pixels = self.pixels()
        out = noised_guide_pixels(pixels, 0.4, 3)
        assert torch.equal(out[-1], pixels[-1].to(torch.float32))

    def test_first_frame_changes(self):
        pixels = self.pixels()
        out = noised_guide_pixels(pixels, 0.4, 3)
        assert not torch.equal(out[0], pixels[0].to(torch.float32))

    def test_noise_falls_along_the_guide(self):
        pixels = self.pixels(9)
        out = noised_guide_pixels(pixels, 0.2, 1)
        spread = (out - 128.0).abs().mean(dim=(1, 2, 3))
        assert spread[0] > spread[4] > spread[-1] == 0

    def test_clamped_to_pixel_range(self):
        for fill in (0, 255):
            pixels = torch.full((5, 8, 8, 3), fill, dtype=torch.uint8)
            out = noised_guide_pixels(pixels, 0.9, 0)
            assert out.min() >= 0.0 and out.max() <= 255.0
            assert out.dtype == torch.float32

    def test_same_seed_repeats(self):
        pixels = self.pixels()
        assert torch.equal(
            noised_guide_pixels(pixels, 0.4, 7), noised_guide_pixels(pixels, 0.4, 7)
        )

    def test_another_seed_differs(self):
        pixels = self.pixels()
        assert not torch.equal(
            noised_guide_pixels(pixels, 0.4, 7), noised_guide_pixels(pixels, 0.4, 8)
        )


# The layout step, with the VAE stubbed


def layout_components():
    return SimpleNamespace(
        _execution_device=torch.device("cpu"),
        canvas_multiple=32,
        config=SimpleNamespace(canvas_short_edge=768, canvas_max_pixels=768 * 1344),
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        fps=24,
        min_duration=5,
        max_duration=15,
        vae_spatial_compression_ratio=16,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=7,
        video_tag=9,
        pixel_mean=0,
        pixel_std=1,
        keyframe_encode_seed=100,
        vae=None,
        audio_sampling_rate=16000,
        audio_latent_channels=AUDIO_C,
    )


def layout_state(guides, guide_noise=None):
    state = PipelineState()
    state.set("text_token_tags", torch.ones(TEXT, dtype=torch.long))
    state.set("height", SIDE)
    state.set("width", SIDE)
    state.set("num_frames", FRAMES)
    state.set("keyframe_anchors", ())
    state.set("guides", guides)
    if guide_noise is not None:
        state.set(GUIDE_NOISE_INPUT, guide_noise)
    return state


def run_step(guides, guide_noise=None):
    return DwH3GuideLayoutStep()(
        layout_components(), layout_state(guides, guide_noise)
    )[1]


@pytest.fixture
def encodes(monkeypatch):
    """Records every (pixels, seed) the stubbed VAE encode is given. Its latent
    depends on the pixels, so a noisy encode differs from the clean one."""
    calls = []

    def encode(vae, pixels, mean, std, seed):
        calls.append((pixels.clone(), seed))
        frames = pixels.shape[2]
        latents = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
        shape = (1, 4, latents, SIDE // 16, SIDE // 16)
        value = pixels.to(torch.float32).mean(dim=(1, 3, 4)).mean() / 255.0
        ramp = torch.linspace(0, 1, latents).view(1, 1, -1, 1, 1)
        return torch.full(shape, 1.0) * value + ramp

    monkeypatch.setattr(ENCODER, encode)
    return calls


def video_guide(frame=17, n=22, fill=100):
    return {"video": np.full((n, 32, 32, 3), fill, np.uint8), "frame": frame}


class TestNoisedEncode:
    @pytest.mark.parametrize("noise", [None, 0, 0.0])
    def test_zero_is_todays_latent_with_one_encode_per_guide(self, encodes, noise):
        guides = [video_guide(0), video_guide(17)]
        state = run_step(guides, noise)
        assert len(encodes) == 2
        baseline = run_step(guides)
        for ours, theirs in zip(
            state.get("condition_latents"), baseline.get("condition_latents")
        ):
            assert torch.equal(ours, theirs)

    def test_zero_is_the_plain_encode(self, encodes):
        state = run_step([video_guide()], 0)
        ((pixels, seed),) = encodes
        assert seed == 100
        assert pixels.dtype == torch.uint8
        expected = encodes_value(pixels)
        assert torch.equal(state.get("condition_latents")[0], expected)

    def test_noise_encodes_once_per_guide_with_the_clean_seed(self, encodes):
        run_step([video_guide(0), video_guide(17)], 0.4)
        # The noisy pixels, encoded once each, under the clean encode seed
        assert len(encodes) == 2
        for noisy, seed in encodes:
            assert seed == 100
            assert noisy.dtype == torch.float32

    def test_result_is_the_noisy_encode(self, encodes):
        state = run_step([video_guide()], 0.4)
        ((noisy_pixels, _),) = encodes
        clean = encodes_value(torch.full_like(noisy_pixels, 100))
        noisy = encodes_value(noisy_pixels)
        assert not torch.equal(clean, noisy)
        assert torch.equal(state.get("condition_latents")[0], noisy)

    def test_the_noisy_frames_are_the_tapered_noise(self, encodes):
        run_step([video_guide()], 0.4)
        ((noisy_pixels, _),) = encodes
        clean_pixels = torch.full_like(noisy_pixels, 100)
        # (1, c, n, h, w): the last frame is the join, left clean
        assert torch.equal(noisy_pixels[:, :, -1], clean_pixels[:, :, -1])
        assert not torch.equal(noisy_pixels[:, :, 0], clean_pixels[:, :, 0])

    def test_another_guide_gets_another_noise_seed(self, encodes):
        run_step([video_guide(0, fill=100), video_guide(17, fill=100)], 0.4)
        first = encodes[0][0]
        second = encodes[1][0]
        assert not torch.equal(first, second)

    def test_a_run_repeats(self, encodes):
        run_step([video_guide()], 0.4)
        run_step([video_guide()], 0.4)
        assert torch.equal(encodes[0][0], encodes[1][0])


def encodes_value(pixels):
    frames = pixels.shape[2]
    latents = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
    shape = (1, 4, latents, SIDE // 16, SIDE // 16)
    value = pixels.to(torch.float32).mean(dim=(1, 3, 4)).mean() / 255.0
    ramp = torch.linspace(0, 1, latents).view(1, 1, -1, 1, 1)
    return torch.full(shape, 1.0) * value + ramp


LAYOUT_KEYS = [
    "num_frames",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
]


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return a == b


class TestRowsUnchanged:
    def test_layout_is_the_same_with_noise(self, encodes):
        guides = [video_guide(0), video_guide(17, n=39)]
        plain = run_step(guides)
        noised = run_step(guides, 0.5)
        for key in LAYOUT_KEYS:
            assert same(plain.get(key), noised.get(key)), key
        assert len(noised.get("condition_latents")) == 2
        for a, b in zip(
            plain.get("condition_latents"), noised.get("condition_latents")
        ):
            assert a.shape == b.shape

    def test_guide_audio_rows_are_the_same_with_noise(self, encodes, monkeypatch):
        spans = []

        def encode_span(encoder, components, waveform, num_latents):
            spans.append((tuple(waveform.shape), num_latents))
            return torch.full((2, num_latents, AUDIO_C), 5.0)

        monkeypatch.setattr(h3_hold_steps, "encode_audio_span", encode_span)
        guide = {
            "video": np.zeros((39, 32, 32, 3), np.uint8),
            "frame": 0,
            "audio": torch.ones(2, 16000 * 39 // 24),
            "sample_rate": 16000,
        }
        plain = run_step([guide])
        noised = run_step([dict(guide)], 0.6)
        assert spans[0] == spans[1]
        assert torch.equal(plain.get(GUIDE_AUDIO_ROWS), noised.get(GUIDE_AUDIO_ROWS))
        assert plain.get("num_condition_audio_rows") == noised.get(
            "num_condition_audio_rows"
        )
        assert torch.equal(plain.get("position_ids"), noised.get("position_ids"))


# Validation


def definition(workflow="t2va", chain=None, pipeline_class=None, **arguments):
    pipeline = {
        "configuration": {"component_type": "ModularPipeline"},
        "from_pretrained_arguments": {
            "model_name": "MiniMaxAI/MiniMax-H3",
            "workflow": workflow,
        },
        "arguments": arguments,
    }
    if pipeline_class:
        pipeline["configuration"] = {"component_type": pipeline_class}
        pipeline["from_pretrained_arguments"] = {"model_name": "some/model"}
    if chain is not None:
        pipeline["chain"] = chain
    return {"steps": [{"name": "video", "pipeline": pipeline}]}


GUIDE = [{"video": "clip.mp4", "frame": 0}]


class TestGuideNoiseErrors:
    @pytest.mark.parametrize("value", [1, 1.5, -0.1, "x", True, False, float("nan")])
    def test_refused_naming_the_range(self, value):
        (error,) = guide_noise_errors(definition(guide_noise=value, guides=GUIDE))
        assert error["path"] == "steps[0].pipeline.arguments.guide_noise"
        assert "[0, 1)" in error["message"]

    @pytest.mark.parametrize("value", [0, 0.0, 0.4, 0.999])
    @pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
    def test_in_range_is_clean(self, value, workflow):
        step = definition(workflow, guide_noise=value, guides=GUIDE)
        assert guide_noise_errors(step) == []

    def test_unset_and_null_are_clean(self):
        assert guide_noise_errors(definition()) == []
        assert guide_noise_errors(definition(guide_noise=None)) == []

    def test_unresolved_variable_is_not_refused(self):
        step = definition(guide_noise="variable:noise", guides=GUIDE)
        assert guide_noise_errors(step) == []
        assert guide_noise_problem("variable:noise") is None

    def test_off_h3_is_refused(self):
        step = definition(pipeline_class="WanPipeline", guide_noise=0.4)
        (error,) = guide_noise_errors(step)
        assert error["path"] == "steps[0].pipeline.arguments.guide_noise"

    def test_ref2va_is_refused(self):
        (error,) = guide_noise_errors(definition("ref2va", guide_noise=0.4))
        assert "ref2va" in error["message"]

    def test_references_are_refused(self):
        step = definition("fl2va", guide_noise=0.4, references=["a.png"])
        (error,) = guide_noise_errors(step)
        assert "references" in error["message"]

    def test_registered_as_a_check(self):
        assert "guide_noise" in [check.name for check in validation.ERROR_CHECKS]
        assert "guide_noise_warnings" in [
            check.name for check in validation.WARNING_CHECKS
        ]


class TestGuideNoiseWarnings:
    @pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
    def test_no_guides_and_no_chain_warns(self, workflow):
        (warning,) = guide_noise_warnings(definition(workflow, guide_noise=0.4))
        assert warning.startswith("steps[0].pipeline.arguments.guide_noise:")
        assert "does nothing" in warning

    def test_empty_guides_warn(self):
        step = definition(guide_noise=0.4, guides=[])
        assert len(guide_noise_warnings(step)) == 1

    def test_a_chain_without_guide_continuity_warns(self):
        step = definition(guide_noise=0.4, chain={"segments": 2})
        assert len(guide_noise_warnings(step)) == 1

    def test_guides_do_not_warn(self):
        assert guide_noise_warnings(definition(guide_noise=0.4, guides=GUIDE)) == []

    def test_guide_continuity_does_not_warn(self):
        chain = {"segments": 3, "continuity": "guide"}
        assert guide_noise_warnings(definition(guide_noise=0.4, chain=chain)) == []

    def test_unresolved_continuity_does_not_warn(self):
        chain = {"segments": 3, "continuity": "variable:continuity"}
        assert guide_noise_warnings(definition(guide_noise=0.4, chain=chain)) == []

    def test_zero_does_not_warn(self):
        assert guide_noise_warnings(definition(guide_noise=0)) == []

    def test_unset_does_not_warn(self):
        assert guide_noise_warnings(definition()) == []

    def test_unresolved_variable_value_is_not_an_error_and_may_warn(self):
        step = definition(guide_noise="variable:noise")
        assert guide_noise_errors(step) == []


# Pipeline._with_guide_noise


def fresh(workflow):
    pipeline = minimax.MiniMaxH3Blocks().get_workflow(workflow).init_pipeline()
    insert_audio_hold(pipeline)
    return pipeline


@pytest.fixture(scope="module")
def guided():
    pipeline = fresh("t2va")
    assert insert_guides(pipeline)
    return pipeline


def step(pipeline):
    return SimpleNamespace(name="video", pipeline=pipeline, base_dir=None)


class TestWithGuideNoise:
    def test_absent_passes_through(self, guided):
        arguments = {"prompt": "x"}
        assert Pipeline._with_guide_noise(step(guided), arguments) is arguments

    @pytest.mark.parametrize("value", [None, 0, 0.0])
    def test_zero_and_none_drop_the_key(self, guided, value):
        arguments = {"prompt": "x", GUIDE_NOISE_INPUT: value}
        result = Pipeline._with_guide_noise(step(guided), arguments)
        assert GUIDE_NOISE_INPUT not in result
        assert result["prompt"] == "x"
        assert GUIDE_NOISE_INPUT in arguments  # the caller's dict is left be

    def test_a_value_is_kept_as_a_float(self, guided):
        result = Pipeline._with_guide_noise(step(guided), {GUIDE_NOISE_INPUT: 0.3})
        assert result[GUIDE_NOISE_INPUT] == 0.3
        assert isinstance(result[GUIDE_NOISE_INPUT], float)

    def test_an_int_is_made_a_float(self, guided):
        # 0 drops; there is no other int in [0, 1)
        result = Pipeline._with_guide_noise(step(guided), {GUIDE_NOISE_INPUT: 0})
        assert GUIDE_NOISE_INPUT not in result

    @pytest.mark.parametrize("value", [1, 1.5, -0.1, "x", True, float("nan")])
    def test_out_of_range_raises(self, guided, value):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            Pipeline._with_guide_noise(step(guided), {GUIDE_NOISE_INPUT: value})

    def test_references_raise(self, guided):
        with pytest.raises(ValueError, match="references"):
            Pipeline._with_guide_noise(
                step(guided), {GUIDE_NOISE_INPUT: 0.3, "references": [object()]}
            )

    def test_a_pipeline_that_cannot_take_guides_raises(self):
        other = SimpleNamespace(name="other")
        with pytest.raises(ValueError, match="guide_noise"):
            Pipeline._with_guide_noise(step(other), {GUIDE_NOISE_INPUT: 0.3})
