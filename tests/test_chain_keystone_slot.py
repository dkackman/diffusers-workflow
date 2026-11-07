"""The keystone slot (#1187/#1188 follow-on, #1199 Task 17): where a guide
chain's `keystone: "segment_1"` frame sits in each later segment.

Today's keystone is the fl2va keyframe, at rotary time 0 - the same slot as the
guide's first latent, so the two condition rows contradict each other and the
burn resets only when the keystone happens to win, mid-segment. The slot moves
it: `"last"` makes it the fl2va `last_image` (the trained last-frame anchor),
`{"frame": 17j}` a one-frame guide past the carried tail, optionally gated by
`enter_below_sigma`; the keyframe goes back to the guide's first frame. A
stable chain key since Don's 2026-11-06 call (bench/records/structural.md,
*Keystone slot*); only the sigma gate, which the windowed denoiser runs, stays
behind DW_EXPERIMENTAL_REFINE_WINDOWS."""

import pytest

from dw.pipeline_processors import h3_rules
from dw.pipeline_processors.chain import ChainConfig, run_chain
from dw.pipeline_processors.chain_rules import (
    CHAIN_KEYSTONE_SLOT,
    chain_keystone_slot_problem,
)
from tests import test_chain
from tests.test_chain import FakePipeline, solid_frame

# Through the module, so pytest does not collect the class here a second time
GuideShape = test_chain.TestGuideContinuity

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
SEGMENT_1_FRAME_0 = (0, 0, 7)


def carried_frame_0(call_index):
    """The previous segment's frame 102, the first of the guide it carried."""
    return ((call_index - 1) * 50, 102, 7)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def chain(**overrides):
    return GuideShape().chain(keystone="segment_1", **overrides)


def render(slot, **arguments):
    shape = GuideShape()
    pipeline = FakePipeline(shape.output)
    run_chain(
        pipeline,
        chain(**{CHAIN_KEYSTONE_SLOT: slot}),
        shape.arguments(image=solid_frame((1, 2, 3)), **arguments),
    )
    return pipeline.calls


# 1. The rule


def test_the_key_is_stable():
    assert CHAIN_KEYSTONE_SLOT == "keystone_slot"


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


@pytest.mark.parametrize("slot", ["last", {"frame": 34}])
def test_last_and_a_frame_need_no_flag(flag_off, slot):
    assert chain_keystone_slot_problem(chain(**{CHAIN_KEYSTONE_SLOT: slot})) is None


def test_a_gated_frame_slot_still_needs_the_flag(flag_off):
    slot = {"frame": 34, "enter_below_sigma": 0.9}
    problem = chain_keystone_slot_problem(chain(**{CHAIN_KEYSTONE_SLOT: slot}))
    assert FLAG in problem and "enter_below_sigma" in problem


def test_the_spike_s_name_is_refused_naming_the_stable_one(flag_off):
    problem = chain_keystone_slot_problem(
        chain(**{"experimental_keystone_slot": "last"})
    )
    assert "experimental_keystone_slot" in problem
    assert "keystone_slot" in problem.replace("experimental_keystone_slot", "")
    with pytest.raises(ValueError, match="renamed"):
        ChainConfig(
            chain(**{"experimental_keystone_slot": "last"}),
            GuideShape().arguments(image=solid_frame((1, 2, 3))),
        )


def test_last_runs_with_the_flag_off(flag_off):
    calls = render("last")
    for call in calls[1:]:
        assert call["last_image"].getpixel((0, 0)) == SEGMENT_1_FRAME_0


def test_absent_null_and_first_are_off(flag_off):
    assert chain_keystone_slot_problem(chain()) is None
    assert chain_keystone_slot_problem(chain(**{CHAIN_KEYSTONE_SLOT: None})) is None
    assert chain_keystone_slot_problem(chain(**{CHAIN_KEYSTONE_SLOT: "first"})) is None


def test_the_slot_needs_the_keystone(flag_on):
    problem = chain_keystone_slot_problem(
        GuideShape().chain(**{CHAIN_KEYSTONE_SLOT: "last"})
    )
    assert "keystone 'segment_1'" in problem


@pytest.mark.parametrize(
    "slot, match",
    [
        ("middle", "first, last"),
        (17.0, "first, last"),
        ({"frame": 20}, "multiple of 17"),
        ({"frame": 17}, "past the guide"),
        ({"frame": 34, "enter_below_sigma": 0}, "(0, 1]"),
        ({"frame": 34, "enter_below_sigma": 1.5}, "(0, 1]"),
        ({"frame": 34, "sigma": 0.5}, "unknown"),
        ({}, "frame"),
    ],
)
def test_a_bad_slot_is_refused(flag_on, slot, match):
    assert match in chain_keystone_slot_problem(chain(**{CHAIN_KEYSTONE_SLOT: slot}))


def test_a_frame_slot_past_the_segment_is_refused(flag_on):
    slot = chain(**{CHAIN_KEYSTONE_SLOT: {"frame": 136}})
    assert "past the segment's 124 frames" in chain_keystone_slot_problem(slot, 124)
    assert chain_keystone_slot_problem(slot, 140) is None


def test_the_guide_span_follows_guide_frames(flag_on):
    slot = {CHAIN_KEYSTONE_SLOT: {"frame": 34}}
    assert chain_keystone_slot_problem(chain(**slot)) is None
    assert "past the guide" in chain_keystone_slot_problem(
        chain(guide_frames=39, **slot)
    )


def test_an_unresolved_reference_is_left_to_the_run(flag_on):
    assert (
        chain_keystone_slot_problem(chain(**{CHAIN_KEYSTONE_SLOT: "variable:slot"}))
        is None
    )


def test_the_run_refuses_a_bad_slot(flag_on):
    with pytest.raises(ValueError, match="multiple of 17"):
        ChainConfig(
            chain(**{CHAIN_KEYSTONE_SLOT: {"frame": 20}}),
            GuideShape().arguments(image=solid_frame((1, 2, 3))),
        )


def test_last_is_refused_when_the_step_has_its_own_last_image(flag_on):
    with pytest.raises(ValueError, match="last_image"):
        ChainConfig(
            chain(**{CHAIN_KEYSTONE_SLOT: "last"}),
            GuideShape().arguments(
                image=solid_frame((1, 2, 3)), last_image=solid_frame((4, 5, 6))
            ),
        )


def test_last_is_refused_without_a_keyframe(flag_on):
    with pytest.raises(ValueError, match="fl2va"):
        ChainConfig(
            chain(**{CHAIN_KEYSTONE_SLOT: "last"}),
            GuideShape().arguments(),
        )


def test_validation_wires_the_rule(flag_off):
    from dw.guides import anchor_errors

    definition = {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "chain": chain(**{CHAIN_KEYSTONE_SLOT: "middle"}),
                    "arguments": {},
                },
            }
        ]
    }
    errors = anchor_errors(definition)
    assert any(CHAIN_KEYSTONE_SLOT in e["path"] for e in errors)


# 2. The run


def test_first_is_todays_keystone(flag_on):
    shape = GuideShape()
    today = FakePipeline(shape.output)
    run_chain(today, chain(), shape.arguments(image=solid_frame((1, 2, 3))))
    first = render("first")
    for got, want in zip(first, today.calls):
        assert got["image"].getpixel((0, 0)) == want["image"].getpixel((0, 0))
        assert "last_image" not in got
        assert len(got.get("guides") or []) == len(want.get("guides") or [])


def test_last_makes_segment_1_frame_0_every_later_last_image(flag_on):
    calls = render("last")
    assert "last_image" not in calls[0]
    for index, call in enumerate(calls[1:], 1):
        assert call["last_image"].getpixel((0, 0)) == SEGMENT_1_FRAME_0
        # the keyframe is the guide's own first frame again: no collision
        assert call["image"].getpixel((0, 0)) == carried_frame_0(index)
        (guide,) = call["guides"]
        assert guide["frame"] == 0


def test_a_frame_slot_is_a_one_frame_guide_past_the_tail(flag_on):
    calls = render({"frame": 34})
    assert not calls[0].get("guides")
    for index, call in enumerate(calls[1:], 1):
        carry, keystone = call["guides"]
        assert carry["frame"] == 0
        assert keystone["frame"] == 34
        assert [f.getpixel((0, 0)) for f in keystone["video"]] == [SEGMENT_1_FRAME_0]
        assert "enter_below_sigma" not in keystone
        assert call["image"].getpixel((0, 0)) == carried_frame_0(index)
        assert "last_image" not in call


def test_a_gated_frame_slot_carries_its_gate(flag_on):
    calls = render({"frame": 34, "enter_below_sigma": 0.9})
    for call in calls[1:]:
        assert call["guides"][1]["enter_below_sigma"] == 0.9


def test_selflift_refuses_a_gated_slot(monkeypatch):
    from dw.guides import selflift_errors

    monkeypatch.setenv(FLAG, "1")
    definition = {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "chain": chain(
                        **{CHAIN_KEYSTONE_SLOT: {"frame": 34, "enter_below_sigma": 0.9}}
                    ),
                    "arguments": {"experimental_selflift": {"lift_step": 1}},
                },
            }
        ]
    }
    assert any(CHAIN_KEYSTONE_SLOT in e["message"] for e in selflift_errors(definition))


# 3. A match_audio plan's shortened last segment


def matched_chain(slot, total_frames, num_frames=175):
    """A guide chain matched to a track `total_frames` long at the shape's
    rate, segments of `num_frames`, the keystone at `slot`."""
    import torch

    from tests.test_chain import FakeAudioReference, FakeImageReference

    shape = GuideShape()
    definition = chain(
        match_audio=True, segment_argument="references", **{CHAIN_KEYSTONE_SLOT: slot}
    )
    definition.pop("segments")
    samples = round(total_frames / shape.FPS * shape.RATE)
    arguments = shape.arguments(
        num_frames=num_frames,
        image=solid_frame((1, 2, 3)),
        references=[
            FakeImageReference(solid_frame((9, 9, 9))),
            FakeAudioReference(torch.zeros(2, samples), shape.RATE),
        ],
    )
    return definition, arguments


def test_a_frame_slot_past_the_shortened_last_segment_is_refused_up_front(flag_on):
    # 175-frame segments over 205 frames: the last needs 52 and snaps up to
    # the 124 minimum, which frame 136 is past - refused before segment 1
    # renders, not when the last segment reaches the pipeline
    definition, arguments = matched_chain({"frame": 136}, 205)

    with pytest.raises(ValueError, match="frame 136 is past segment 2's 124"):
        ChainConfig(definition, arguments)


def test_a_frame_slot_inside_every_planned_segment_is_kept(flag_on):
    definition, arguments = matched_chain({"frame": 102}, 205)

    config = ChainConfig(definition, arguments)

    assert [s.num_frames for s in config.plan] == [175, 124]
    assert config.keystone_slot["frame"] == 102
