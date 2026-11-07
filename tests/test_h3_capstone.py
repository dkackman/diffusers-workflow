"""The experimental late capstone (#1187, #1199 Task 14 Step 4): a one-frame
guide one latent past the render's end that joins the forward only below a
sigma gate. Its guide keys (h3_guides.prepared_guide), its place in the layout
(h3_guide_steps), the gate in the windowed denoiser (h3_windows) and the chain
key that feeds it segment 1's late frame (chain_rules, chain)."""

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from dw.pipeline_processors import h3_guides, h3_rules  # noqa: E402
from dw.pipeline_processors.chain_rules import chain_capstone_problem  # noqa: E402
from dw.pipeline_processors.h3_windows import (  # noqa: E402
    DwH3WindowedLoopDenoiser,
    gated_out_rows,
)
from tests.test_h3_guides import (  # noqa: E402
    LATENTS,
    SIDE,
    TEXT,
    TestLayoutBlock,
    drive,
    layout_components,
    layout_state,
)
from tests.test_h3_windows import (  # noqa: E402
    VIDEO_WIDTH,
    block_state,
    tiny_transformer,
)

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def capstone_guide(gate=0.9, frames=1):
    return {
        "video": np.zeros((frames, 32, 32, 3), np.uint8),
        "past_end": True,
        "enter_below_sigma": gate,
    }


# 1. The guide keys


def test_the_capstone_keys_are_refused_with_the_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(ValueError, match=FLAG):
        h3_guides.prepared_guide("guides", 0, capstone_guide(), None)


def test_a_capstone_needs_no_frame_and_keeps_its_gate(flag_on):
    prepared = h3_guides.prepared_guide("guides", 0, capstone_guide(0.7), None)
    assert prepared["past_end"] is True
    assert prepared["enter_below_sigma"] == 0.7
    assert prepared["frame"] is None
    assert prepared["video"].shape[0] == 1


@pytest.mark.parametrize(
    "guide, match",
    [
        (capstone_guide(frames=22), "one frame"),
        (capstone_guide(gate=0), "sigma in"),
        (capstone_guide(gate=1.5), "sigma in"),
        ({**capstone_guide(), "past_end": "yes"}, "true or false"),
    ],
)
def test_a_bad_capstone_is_refused(flag_on, guide, match):
    with pytest.raises(ValueError, match=match):
        h3_guides.prepared_guide("guides", 0, guide, None)


def test_a_plain_guide_is_unchanged(flag_on):
    guide = {"video": np.zeros((22, 32, 32, 3), np.uint8), "frame": 0}
    prepared = h3_guides.prepared_guide("guides", 0, guide, None)
    assert "past_end" not in prepared and "enter_below_sigma" not in prepared


# 2. The layout


def test_the_capstone_sits_one_latent_past_the_end_and_is_gated(monkeypatch):
    from diffusers.modular_pipelines.minimax_h3 import before_denoise

    block = TestLayoutBlock().make_block(monkeypatch, [])
    carry = {"video": np.zeros((22, 64, 64, 3), np.uint8), "frame": 0}
    capstone = {
        "video": np.zeros((1, 64, 64, 3), np.uint8),
        "frame": None,
        "past_end": True,
        "enter_below_sigma": 0.8,
    }
    state = drive(block(), layout_state([carry, capstone]), layout_components())
    rows_per_frame = (SIDE // 16 // 2) ** 2
    carry_rows = 7 * rows_per_frame
    assert state.get(h3_rules.GATED_GUIDE_ROWS) == [(carry_rows, rows_per_frame, 0.8)]
    assert state.get("num_condition_video_rows") == carry_rows + rows_per_frame
    # The capstone's rows carry the time one latent past the target's last
    times = before_denoise._temporal_position_grid(LATENTS + 1, float(TEXT))
    video = state.get("video_indices")
    capstone_times = state.get("position_ids")[
        video[carry_rows : carry_rows + rows_per_frame], 0
    ]
    target_last = state.get("position_ids")[video[-1], 0]
    assert torch.all(capstone_times == times[LATENTS])
    assert capstone_times[0] > target_last


def test_no_gate_leaves_no_gated_rows(monkeypatch):
    block = TestLayoutBlock().make_block(monkeypatch, [])
    carry = {"video": np.zeros((22, 64, 64, 3), np.uint8), "frame": 0}
    state = drive(block(), layout_state([carry]), layout_components())
    assert state.get(h3_rules.GATED_GUIDE_ROWS) is None


# 3. The gate in the denoiser


def test_rows_are_out_while_sigma_is_at_or_over_the_gate():
    gates = [(4, 2, 0.8)]
    # H3's timesteps are 1 - sigma
    assert gated_out_rows(gates, 0.0).tolist() == [4, 5]
    assert gated_out_rows(gates, 0.2).tolist() == [4, 5]
    assert gated_out_rows(gates, 0.25) is None
    assert gated_out_rows(None, 0.0) is None


def _with_condition_rows(count):
    """The windows' t2va test state, with its first `count` video rows taken
    as condition rows, and the same state without them."""
    full, scheduler = block_state()
    fields = full.denoiser_input_fields
    # The packed sequence less the first `count` video rows: what the gate runs
    drop = fields["video_indices"][:count]
    keep = torch.ones(fields["position_ids"].shape[0], dtype=torch.bool)
    keep[drop] = False
    sequence = keep.nonzero().squeeze(1)
    small = dict(full.__dict__)
    small_fields = {
        "position_ids": fields["position_ids"][sequence],
        "token_tags": fields["token_tags"][sequence],
        "video_indices": torch.searchsorted(sequence, fields["video_indices"][count:]),
        "audio_indices": torch.searchsorted(sequence, fields["audio_indices"]),
        "text_indices": torch.searchsorted(sequence, fields["text_indices"]),
    }
    small["denoiser_input_fields"] = small_fields
    small["latents"] = full.latents[count:]
    small["row_timestep_plan"] = [
        (unique, indices[sequence]) for unique, indices in full.row_timestep_plan
    ]
    full.dw_gated_guide_rows = [(0, count, 0.5)]
    return full, type(full)(**small), scheduler


def test_a_gated_step_is_the_forward_without_the_rows():
    from types import SimpleNamespace

    count = 2
    full, small, scheduler = _with_condition_rows(count)
    components = SimpleNamespace(transformer=tiny_transformer())
    t = scheduler.timesteps[0]  # sigma 1: the rows are out
    denoiser = DwH3WindowedLoopDenoiser()
    denoiser(components, full, 0, t)
    denoiser(components, small, 0, t)
    assert full.noise_pred.shape == (1, full.latents.shape[0], VIDEO_WIDTH)
    assert torch.all(full.noise_pred[0, :count] == 0)
    assert torch.allclose(full.noise_pred[0, count:], small.noise_pred[0], atol=1e-6)
    assert torch.allclose(full.audio_noise_pred, small.audio_noise_pred, atol=1e-6)


def test_below_the_gate_the_rows_are_in_the_forward():
    from types import SimpleNamespace

    full, _, _ = _with_condition_rows(2)
    components = SimpleNamespace(transformer=tiny_transformer())
    denoiser = DwH3WindowedLoopDenoiser()
    denoiser(components, full, 0, torch.tensor(0.9))  # sigma 0.1 < 0.5
    gated = full.noise_pred.clone()
    full.dw_gated_guide_rows = None
    denoiser(components, full, 0, torch.tensor(0.9))
    assert torch.equal(gated, full.noise_pred)


# 4. The chain key


def test_the_chain_key_is_refused_with_the_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    problem = chain_capstone_problem(
        {"experimental_capstone": {"enter_below_sigma": 0.5}}
    )
    assert FLAG in problem


@pytest.mark.parametrize(
    "value, match",
    [
        ({}, "enter_below_sigma"),
        ({"enter_below_sigma": 0}, "(0, 1]"),
        ({"enter_below_sigma": 2}, "(0, 1]"),
        ({"enter_below_sigma": 0.5, "from_end": 0}, "from_end"),
        ({"enter_below_sigma": 0.5, "sigma": 1}, "unknown"),
    ],
)
def test_a_bad_chain_key_is_refused(flag_on, value, match):
    assert match in chain_capstone_problem({"experimental_capstone": value})


def test_a_from_end_past_the_segment_is_refused(flag_on):
    chain = {"experimental_capstone": {"enter_below_sigma": 0.5, "from_end": 30}}
    problem = chain_capstone_problem(chain, num_frames=25)
    assert "from_end 30" in problem and "25 frames" in problem
    assert chain_capstone_problem(chain, num_frames=30) is None


def test_a_from_end_past_the_rendered_frames_is_a_clear_error():
    from dw.pipeline_processors.chain import _from_segment_1

    config = type(
        "C",
        (),
        {
            "anchor": None,
            "keystone": None,
            "capstone": {"enter_below_sigma": 0.5, "from_end": 30},
        },
    )()
    with pytest.raises(ValueError, match="from_end 30.*25 frames"):
        _from_segment_1(config, ["f"] * 25)


def test_a_good_chain_key_passes(flag_on):
    assert chain_capstone_problem({}) is None
    assert (
        chain_capstone_problem(
            {"experimental_capstone": {"enter_below_sigma": 0.9, "from_end": 5}}
        )
        is None
    )


def test_later_segments_get_segment_1s_frame_past_their_end(flag_on):
    from dw.pipeline_processors.chain import run_chain
    from tests.test_chain import FakePipeline, TestGuideContinuity

    shape = TestGuideContinuity()
    pipeline = FakePipeline(shape.output)
    run_chain(
        pipeline,
        shape.chain(experimental_capstone={"enter_below_sigma": 0.9}),
        shape.arguments(),
    )
    assert len(pipeline.calls[0].get("guides") or []) == 0
    for call in pipeline.calls[1:]:
        carry, capstone = call["guides"]
        assert carry["frame"] == 0
        assert capstone["past_end"] is True
        assert capstone["enter_below_sigma"] == 0.9
        # segment 1's frame num_frames - 5, every time
        assert [f.getpixel((0, 0)) for f in capstone["video"]] == [(0, 119, 7)]


def test_add_capstone_appends_a_gated_past_end_guide():
    from dw.pipeline_processors.chain import _add_capstone

    arguments = {"guides": [{"video": "carry", "frame": 0}]}
    _add_capstone(arguments, "late frame", {"enter_below_sigma": 0.9, "from_end": 5})
    assert arguments["guides"][0] == {"video": "carry", "frame": 0}
    assert arguments["guides"][1] == {
        "video": ["late frame"],
        "past_end": True,
        "enter_below_sigma": 0.9,
    }


# 5. End to end: the real H3 graph on tiny weights


@pytest.fixture(scope="module")
def tiny_guided_h3():
    from dw.pipeline_processors.h3_guides import insert_guides
    from dw.pipeline_processors.h3_hold import insert_audio_hold
    from dw.pipeline_processors.h3_windows import insert_refine_windows
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    assert insert_guides(pipeline)
    insert_audio_hold(pipeline)
    assert insert_refine_windows(pipeline)
    return pipeline


def _render(pipeline, capstone_gate=None):
    rng = np.random.default_rng(5)
    guides = [
        {
            "video": rng.integers(0, 255, (22, 32, 32, 3), "uint8"),
            "frame": 0,
            "audio": None,
            "sample_rate": None,
        }
    ]
    if capstone_gate is not None:
        guides.append(
            {
                "video": rng.integers(0, 255, (1, 32, 32, 3), "uint8"),
                "frame": None,
                "audio": None,
                "sample_rate": None,
                "past_end": True,
                "enter_below_sigma": capstone_gate,
            }
        )
    return pipeline(
        prompt="a cat sat",
        height=32,
        width=32,
        num_frames=124,
        num_inference_steps=3,
        guides=guides,
        generator=torch.Generator("cpu").manual_seed(1234),
        output="latents",
    )


def test_end_to_end_the_gate_decides_when_the_capstone_counts(tiny_guided_h3):
    # 3 steps at shift 12: forwards at sigma 1 and about 0.92. The draws are
    # the same for every capstone render (one more condition latent noised),
    # so only the gate differs between `never` and `late`; that a gated-out
    # forward is the forward without the rows is the unit test above
    never = _render(tiny_guided_h3, 1e-6)
    late = _render(tiny_guided_h3, 0.95)
    plain = _render(tiny_guided_h3)
    assert never.shape == late.shape == plain.shape
    assert torch.isfinite(late).all() and torch.isfinite(never).all()
    assert not torch.allclose(never, late, atol=1e-5)


def test_end_to_end_a_gated_one_frame_guide_inside_the_render(tiny_guided_h3):
    # The keystone slot's {"frame": 34, "enter_below_sigma"} guide
    # (chain_rules.place_keystone): a one-frame row on a chunk boundary past
    # the carried tail, gated like a capstone but not past the end
    rng = np.random.default_rng(5)

    def render(gate):
        guides = [
            {
                "video": rng.integers(0, 255, (22, 32, 32, 3), "uint8"),
                "frame": 0,
                "audio": None,
                "sample_rate": None,
            },
            {
                "video": np.full((1, 32, 32, 3), 200, "uint8"),
                "frame": 34,
                "audio": None,
                "sample_rate": None,
                "past_end": False,
                "enter_below_sigma": gate,
            },
        ]
        return tiny_guided_h3(
            prompt="a cat sat",
            height=32,
            width=32,
            num_frames=124,
            num_inference_steps=3,
            guides=guides,
            generator=torch.Generator("cpu").manual_seed(1234),
            output="latents",
        )

    rng_state = rng.bit_generator.state
    never = render(1e-6)
    rng.bit_generator.state = rng_state
    late = render(0.95)
    assert never.shape == late.shape
    assert torch.isfinite(late).all()
    assert not torch.allclose(never, late, atol=1e-5)


# 5. Validate (#1232): the capstone is refused before the run, not at the load


def _capstone_definition(capstone, num_frames=49):
    return {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "chain": {"experimental_capstone": capstone},
                    "arguments": {"num_frames": num_frames},
                },
            }
        ]
    }


def test_validate_refuses_the_capstone_with_the_flag_off(monkeypatch):
    from dw.guides import anchor_errors

    monkeypatch.delenv(FLAG, raising=False)
    errors = anchor_errors(_capstone_definition({"enter_below_sigma": 0.5}))
    assert any(FLAG in e["message"] for e in errors)
    assert any(e["path"].endswith("experimental_capstone") for e in errors)


def test_validate_checks_the_capstone_against_num_frames(flag_on):
    from dw.guides import anchor_errors

    capstone = {"enter_below_sigma": 0.5, "from_end": 60}
    errors = anchor_errors(_capstone_definition(capstone, num_frames=49))
    assert any("from_end 60" in e["message"] for e in errors)
    assert not [
        e
        for e in anchor_errors(_capstone_definition({"enter_below_sigma": 0.5}))
        if "capstone" in e["message"]
    ]
