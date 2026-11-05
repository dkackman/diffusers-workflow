"""The experimental latent carry (#1150 stage A, the latent-carry arm of #1044;
#1199 Task 17 Step 2): a guide chain hands segment k-1's own final latents -
the tail the pixel guide would have decoded and re-encoded - to segment k as
its guide's condition rows, video and audio, with no VAE encode.

The chain key and its rules (chain_rules), the tail it cuts (h3_guides
`latent_guide_tail`), the guide keys that carry it (h3_guides
`prepared_guide`), the layout that takes them in place of the encode
(h3_guide_steps), and the real H3 graph on tiny weights."""

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from dw.guides import anchor_errors  # noqa: E402
from dw.pipeline_processors import h3_guides, h3_hold_steps, h3_rules  # noqa: E402
from dw.pipeline_processors.chain import ChainConfig, run_chain  # noqa: E402
from dw.pipeline_processors.chain_rules import (  # noqa: E402
    CHAIN_LATENT_CARRY,
    chain_latent_carry_problem,
)
from dw.pipeline_processors.h3_rules import GUIDE_AUDIO_ROWS  # noqa: E402
from tests import test_chain, test_h3_guides  # noqa: E402
from tests.test_chain import FakePipeline  # noqa: E402
from tests.test_h3_guides import (  # noqa: E402
    AUDIO_C,
    SIDE,
    audio_layout_components,
    drive,
    layout_state,
)

# Through the modules, so pytest does not collect the classes a second time
GuideChainShape = test_chain.TestGuideContinuity
LayoutBlock = test_h3_guides.TestLayoutBlock

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
LATENT_SIDE = SIDE // 16
PACKED = (
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


def latent_frames(num_frames):
    return 5 * ((num_frames - 5) // 17) + 2


def audio_latents_for(num_frames):
    return int(round(num_frames * 40 / 24))


class LatentOutputs:
    """A FakePipeline output factory: TestGuideContinuity's frames and audio,
    plus `latents` (1, 4, T, 4, 4) and `audio_latents` (2, AUDIO_C, N) when
    the call asks for them - random, so a slice is told from any other. A
    clone of each is kept, since the chain takes them off the output."""

    def __init__(self):
        self.shape = GuideChainShape()
        self.kept = []

    def __call__(self, arguments, index):
        output = self.shape.output(arguments, index)
        names = arguments.get("output") or []
        if "latents" not in names:
            self.kept.append(None)
            return output
        num_frames = arguments.get("num_frames", 124)
        generator = torch.Generator().manual_seed(100 + index)
        output["latents"] = torch.randn(
            1,
            4,
            latent_frames(num_frames),
            LATENT_SIDE,
            LATENT_SIDE,
            generator=generator,
        )
        output["audio_latents"] = torch.randn(
            2, AUDIO_C, audio_latents_for(num_frames), generator=generator
        )
        self.kept.append(
            {key: output[key].clone() for key in ("latents", "audio_latents")}
        )
        return output


def run_latent_chain(**chain):
    outputs = LatentOutputs()
    pipeline = FakePipeline(outputs)
    result = run_chain(
        pipeline,
        outputs.shape.chain(**{CHAIN_LATENT_CARRY: True, **chain}),
        outputs.shape.arguments(output=["videos", "audio", "sampling_rate"]),
    )
    return pipeline, outputs, result


# 1. The flag gate, at every entry point


def test_the_chain_key_is_refused_with_the_flag_off(flag_off):
    problem = chain_latent_carry_problem(
        {"continuity": "guide", CHAIN_LATENT_CARRY: True}
    )
    assert FLAG in problem


def test_the_run_refuses_the_chain_key_with_the_flag_off(flag_off):
    shape = GuideChainShape()
    with pytest.raises(ValueError, match=FLAG):
        ChainConfig(shape.chain(**{CHAIN_LATENT_CARRY: True}), shape.arguments())


def test_validation_refuses_the_chain_key_with_the_flag_off(flag_off):
    errors = anchor_errors(h3_workflow({CHAIN_LATENT_CARRY: True}))
    assert [error["path"] for error in errors] == [
        f"steps[0].pipeline.chain.{CHAIN_LATENT_CARRY}"
    ]
    assert FLAG in errors[0]["message"]


@pytest.mark.parametrize("key", ["latents", "audio_latents"])
def test_the_guide_keys_are_refused_with_the_flag_off(flag_off, key):
    guide = {
        "video": np.zeros((22, 32, 32, 3), np.uint8),
        "frame": 0,
        key: torch.zeros(1, 4, 7, 2, 2) if key == "latents" else torch.zeros(2, 37, 4),
    }
    with pytest.raises(ValueError, match=FLAG):
        h3_guides.prepared_guide("guides", 0, guide, None)


def test_off_or_absent_is_never_refused(flag_off):
    assert chain_latent_carry_problem({"continuity": "guide"}) is None
    assert (
        chain_latent_carry_problem({"continuity": "guide", CHAIN_LATENT_CARRY: False})
        is None
    )
    assert (
        chain_latent_carry_problem({"continuity": "guide", CHAIN_LATENT_CARRY: None})
        is None
    )


# 2. What cannot combine, refused at validate time (and by the run's own rule)


def h3_workflow(chain=None, arguments=None):
    return {
        "steps": [
            {
                "name": "chain",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "fl2va",
                    },
                    "chain": {"segments": 3, "continuity": "guide", **(chain or {})},
                    "arguments": {"num_frames": 124, **(arguments or {})},
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "chain, arguments, match",
    [
        ({"continuity": "last_frame"}, None, "continuity 'guide'"),
        ({"continuity": "last_segment"}, None, "continuity 'guide'"),
        ({"look_hold": True}, None, "look_hold"),
        ({"look_hold": {"texture": True}}, None, "look_hold"),
        (
            {"match_audio": True, "segments": None},
            None,
            "match_audio",
        ),
        ({}, {"guide_noise": 0.4}, "guide_noise"),
        ({}, {"hold_audio": "asset:track.wav"}, "hold_audio"),
        ({CHAIN_LATENT_CARRY: "yes"}, None, "true or false"),
    ],
)
def test_an_incompatible_combination_is_refused_at_validate_time(
    flag_on, chain, arguments, match
):
    chain = {CHAIN_LATENT_CARRY: True, **chain}
    errors = anchor_errors(h3_workflow(chain, arguments))
    paths = [error["path"] for error in errors]
    assert f"steps[0].pipeline.chain.{CHAIN_LATENT_CARRY}" in paths
    assert any(match in error["message"] for error in errors)


def test_a_zero_guide_noise_is_not_a_conflict(flag_on):
    workflow = h3_workflow({CHAIN_LATENT_CARRY: True}, {"guide_noise": 0})
    assert anchor_errors(workflow) == []


def test_unresolved_values_are_left_to_the_run(flag_on):
    chain = {CHAIN_LATENT_CARRY: "variable:carry", "continuity": "variable:mode"}
    assert anchor_errors(h3_workflow(chain)) == []


def test_the_run_refuses_what_validation_refuses(flag_on):
    shape = GuideChainShape()
    with pytest.raises(ValueError, match="look_hold"):
        ChainConfig(
            shape.chain(**{CHAIN_LATENT_CARRY: True, "look_hold": True}),
            shape.arguments(),
        )


def test_the_run_refuses_a_step_holding_audio(flag_on):
    shape = GuideChainShape()
    pipeline = FakePipeline(LatentOutputs())
    with pytest.raises(ValueError, match="hold_audio"):
        run_chain(
            pipeline,
            shape.chain(**{CHAIN_LATENT_CARRY: True}),
            shape.arguments(hold_audio="asset:track.wav"),
        )
    assert pipeline.calls == []


def test_the_layout_refuses_guide_noise_on_a_latent_guide(monkeypatch):
    seen = []
    block = LayoutBlock().make_block(monkeypatch, seen)
    state = layout_state([latent_guide()])
    state.set(h3_rules.GUIDE_NOISE_INPUT, 0.4)
    with pytest.raises(ValueError, match="guide_noise"):
        drive(block(), state, audio_layout_components())
    assert seen == []


def test_a_latent_guide_off_the_canvas_is_refused(monkeypatch):
    block = LayoutBlock().make_block(monkeypatch, [])
    guide = latent_guide(latents=torch.zeros(1, 4, 7, 2, 2))
    with pytest.raises(ValueError, match="canvas"):
        drive(block(), layout_state([guide]), audio_layout_components())


# 3. The chain hands on the tail latents, bit-exact


@pytest.mark.parametrize("guide_frames", [22, 39])
def test_the_carried_latents_are_the_previous_tail_bit_exact(flag_on, guide_frames):
    pipeline, outputs, _ = run_latent_chain(guide_frames=guide_frames)
    count = latent_frames(guide_frames)
    audio = audio_latents_for(guide_frames)
    assert "guides" not in pipeline.calls[0]
    for index, call in enumerate(pipeline.calls[1:], start=1):
        (guide,) = call["guides"]
        previous = outputs.kept[index - 1]
        assert guide["frame"] == 0
        assert torch.equal(guide["latents"], previous["latents"][:, :, -count:])
        assert torch.equal(
            guide["audio_latents"],
            previous["audio_latents"][:, :, -audio:].permute(0, 2, 1),
        )
        # The pixels are still there for the keyframe, and no waveform rides
        assert len(guide["video"]) == guide_frames
        assert not guide.get("audio")


def test_a_backed_off_tail_is_the_latents_before_the_back_off(flag_on):
    pipeline, outputs, _ = run_latent_chain(guide_backoff=17)
    previous = outputs.kept[0]
    (guide,) = pipeline.calls[1]["guides"]
    # Frames 85..106 of 124: clip 5's five latents and clip 6's first two
    assert torch.equal(guide["latents"], previous["latents"][:, :, 25:32])
    end = audio_latents_for(124 - 17)
    assert torch.equal(
        guide["audio_latents"],
        previous["audio_latents"][:, :, end - 37 : end].permute(0, 2, 1),
    )


def test_the_carried_tails_own_only_their_tail_storage(flag_on):
    pipeline, outputs, _ = run_latent_chain()
    (guide,) = pipeline.calls[1]["guides"]
    for key in ("latents", "audio_latents"):
        tensor = guide[key]
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )
    assert guide["latents"].untyped_storage().nbytes() < (
        outputs.kept[0]["latents"].untyped_storage().nbytes()
    )


def test_the_audio_latent_rate_is_diffusers_own():
    constants = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.modular_pipeline",
        reason="diffusers' H3 pipeline is not importable here",
    )
    assert (
        h3_guides.H3_AUDIO_LATENTS_PER_SECOND
        == constants.MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
    )


def test_carry_audio_false_carries_video_latents_only(flag_on):
    pipeline, _, _ = run_latent_chain(carry_audio=False)
    (guide,) = pipeline.calls[1]["guides"]
    assert guide["latents"] is not None
    assert "audio_latents" not in guide


def test_every_segment_asks_for_its_latents_and_the_output_is_unchanged(flag_on):
    pipeline, _, result = run_latent_chain()
    for call in pipeline.calls:
        assert call["output"] == [
            "videos",
            "audio",
            "sampling_rate",
            "latents",
            "audio_latents",
        ]
    # The stitched take is the guide chain's: the tensors never reach it
    assert len(result.frames) == 124 + 2 * (124 - 22)


def test_a_segment_returning_no_latents_is_a_clear_error(flag_on):
    shape = GuideChainShape()
    pipeline = FakePipeline(shape.output)  # ignores the ask
    with pytest.raises(ValueError, match="latents"):
        run_chain(
            pipeline,
            shape.chain(**{CHAIN_LATENT_CARRY: True}),
            shape.arguments(output=["videos", "audio", "sampling_rate"]),
        )


def test_the_tail_refuses_a_start_off_a_vae_clip():
    with pytest.raises(ValueError, match="17"):
        h3_guides.latent_guide_tail(
            torch.zeros(1, 4, 37, 2, 2), None, num_frames=124, count=22, back=5
        )


def test_the_tail_refuses_latents_that_are_not_the_segments():
    with pytest.raises(ValueError, match="latent frames"):
        h3_guides.latent_guide_tail(
            torch.zeros(1, 4, 30, 2, 2), None, num_frames=124, count=22, back=0
        )


# 4. The layout takes the latents in place of the encode


def latent_guide(frames=22, latents=None, audio_latents=None, with_audio=True):
    generator = torch.Generator().manual_seed(7)
    guide = {
        "video": np.zeros((frames, 32, 32, 3), np.uint8),
        "frame": 0,
        "audio": None,
        "sample_rate": None,
        "latents": latents
        if latents is not None
        else torch.randn(
            1,
            4,
            latent_frames(frames),
            LATENT_SIDE,
            LATENT_SIDE,
            generator=generator,
        ),
    }
    if with_audio:
        guide["audio_latents"] = (
            audio_latents
            if audio_latents is not None
            else torch.randn(2, audio_latents_for(frames), AUDIO_C, generator=generator)
        )
    return guide


def pixel_guide(frames=22):
    return {
        "video": np.zeros((frames, 32, 32, 3), np.uint8),
        "frame": 0,
        "audio": torch.ones(2, 16000 * frames // 24),
        "sample_rate": 16000,
    }


def run_layout(monkeypatch, guides):
    """The guide layout over `guides` with the video and audio encodes
    replaced by spies: (state, video encodes seen, audio encodes seen)."""
    seen, encoded = [], []

    def encode_span(encoder, components, waveform, num_latents):
        encoded.append(num_latents)
        return torch.full((2, num_latents, AUDIO_C), 5.0)

    monkeypatch.setattr(h3_hold_steps, "encode_audio_span", encode_span)
    block = LayoutBlock().make_block(monkeypatch, seen)
    state = drive(block(), layout_state(guides), audio_layout_components())
    return state, seen, encoded


@pytest.mark.parametrize("frames", [22, 39])
def test_a_latent_guide_never_calls_either_encode(monkeypatch, frames):
    _, seen, encoded = run_layout(monkeypatch, [latent_guide(frames)])
    assert seen == [] and encoded == []


@pytest.mark.parametrize("frames", [22, 39])
def test_the_rows_are_the_carried_latents_bit_exact(monkeypatch, frames):
    guide = latent_guide(frames)
    state, _, _ = run_layout(monkeypatch, [guide])
    assert torch.equal(state.get("condition_latents")[-1], guide["latents"])
    assert torch.equal(
        state.get(GUIDE_AUDIO_ROWS), guide["audio_latents"].reshape(-1, AUDIO_C)
    )


@pytest.mark.parametrize("frames", [22, 39])
def test_positions_and_timing_match_the_pixel_guides(monkeypatch, frames):
    pixel, _, encoded = run_layout(monkeypatch, [pixel_guide(frames)])
    carried, _, _ = run_layout(monkeypatch, [latent_guide(frames)])
    assert encoded  # the pixel guide did encode its audio
    for key in PACKED:
        a, b = pixel.get(key), carried.get(key)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, key
    assert (
        pixel.get("condition_latents")[-1].shape
        == carried.get("condition_latents")[-1].shape
    )
    assert pixel.get(GUIDE_AUDIO_ROWS).shape == carried.get(GUIDE_AUDIO_ROWS).shape


def test_a_latent_guide_without_audio_latents_holds_no_audio(monkeypatch):
    state, _, encoded = run_layout(monkeypatch, [latent_guide(with_audio=False)])
    assert encoded == []
    assert not state.get("num_condition_audio_rows")
    assert state.get(GUIDE_AUDIO_ROWS) is None


def test_too_few_carried_audio_latents_is_refused(monkeypatch):
    guide = latent_guide(audio_latents=torch.zeros(2, 20, AUDIO_C))
    with pytest.raises(ValueError, match="audio latents"):
        run_layout(monkeypatch, [guide])


# 5. The guide keys, read


def test_prepared_guide_passes_the_latents_through(flag_on):
    latents = torch.randn(1, 4, 7, 2, 2)
    audio = torch.randn(2, 37, 4)
    guide = {
        "video": np.zeros((22, 32, 32, 3), np.uint8),
        "frame": 0,
        "latents": latents,
        "audio_latents": audio,
    }
    prepared = h3_guides.prepared_guide("guides", 0, guide, None)
    assert prepared["latents"] is latents
    assert prepared["audio_latents"] is audio
    assert prepared["audio"] is None


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"latents": torch.zeros(1, 4, 6, 2, 2)}, "7 latent frames"),
        ({"latents": "tensor"}, "tensor"),
        ({"latents": torch.zeros(1, 4, 7, 2, 2), "audio": True}, "audio"),
        ({"audio_latents": torch.zeros(2, 37)}, "audio_latents"),
    ],
)
def test_bad_latent_guide_keys_are_refused(flag_on, extra, match):
    guide = {"video": np.zeros((22, 32, 32, 3), np.uint8), "frame": 0, **extra}
    with pytest.raises(ValueError, match=match):
        h3_guides.prepared_guide("guides", 0, guide, None)


# 6. The existing guide path is unchanged


def test_a_guide_chain_without_the_key_asks_for_no_latents(flag_on):
    shape = GuideChainShape()
    pipeline = FakePipeline(shape.output)
    run_chain(
        pipeline,
        shape.chain(),
        shape.arguments(output=["videos", "audio", "sampling_rate"]),
    )
    for call in pipeline.calls:
        assert call["output"] == ["videos", "audio", "sampling_rate"]
    for call in pipeline.calls[1:]:
        (guide,) = call["guides"]
        assert set(guide) == {"video", "frame", "audio"}
        assert guide["audio"] is True


def test_a_pixel_guide_prepares_as_before(flag_on):
    guide = {"video": np.zeros((22, 32, 32, 3), np.uint8), "frame": 0}
    prepared = h3_guides.prepared_guide("guides", 0, guide, None)
    assert set(prepared) == {"video", "frame", "audio", "sample_rate"}


# 7. End to end: the real H3 graph on tiny weights


@pytest.fixture(scope="module")
def tiny_guided_h3():
    from dw.pipeline_processors.h3_guides import insert_guides
    from dw.pipeline_processors.h3_hold import insert_audio_hold
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    assert insert_guides(pipeline)
    insert_audio_hold(pipeline)
    return pipeline


def _render(pipeline, guide, output="latents"):
    return pipeline(
        prompt="a cat sat",
        height=32,
        width=32,
        num_frames=124,
        num_inference_steps=2,
        guides=[guide],
        generator=torch.Generator("cpu").manual_seed(1234),
        output=output,
    )


def test_end_to_end_a_latent_guide_is_the_pixel_guide_without_the_encode(
    tiny_guided_h3,
):
    from diffusers.modular_pipelines.minimax_h3 import encoders

    pipeline = tiny_guided_h3
    pixels = np.random.default_rng(5).integers(0, 255, (22, 32, 32, 3), "uint8")
    base = {"video": pixels, "frame": 0, "audio": None, "sample_rate": None}
    encoded = encoders.encode_vae_condition(
        pipeline.vae,
        torch.from_numpy(pixels).permute(3, 0, 1, 2)[None],
        pipeline.pixel_mean,
        pipeline.pixel_std,
        pipeline.keyframe_encode_seed,
    )
    from_pixels = _render(pipeline, base)
    from_latents = _render(pipeline, {**base, "latents": encoded})
    assert torch.equal(from_pixels, from_latents)


def test_end_to_end_a_segment_hands_on_latents_shaped_as_the_tail_takes(
    tiny_guided_h3,
):
    pixels = np.zeros((22, 32, 32, 3), "uint8")
    guide = {"video": pixels, "frame": 0, "audio": None, "sample_rate": None}
    output = _render(tiny_guided_h3, guide, ["latents", "audio_latents"])
    latents, audio = output["latents"], output["audio_latents"]
    assert latents.shape[2] == latent_frames(124)
    assert audio.shape[0] == 2 and audio.shape[2] == audio_latents_for(124)
    tail, rows = h3_guides.latent_guide_tail(latents, audio, 124, 22, 0)
    assert tail.shape[2] == 7 and rows.shape[:2] == (2, 37)
