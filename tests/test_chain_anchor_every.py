"""`anchor_every` (Task 9, experimental): the keystone and the experimental
capstone apply only on later segments whose 0-based index i has
i % anchor_every == 0 - every third with 3 (indices 3, 6, 9), every one with 1
(the default). Refused without a keystone or a capstone to space, and with the
windowed refine's flag off."""

import pytest

from dw.guides import anchor_errors
from dw.pipeline_processors import h3_rules
from dw.pipeline_processors.chain import ChainConfig, run_chain
from dw.pipeline_processors.chain_rules import (
    CHAIN_ANCHOR_EVERY,
    chain_anchor_every_problem,
)
from tests import test_chain
from tests.test_chain import FakePipeline, solid_frame

GuideShape = test_chain.TestGuideContinuity
FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
KEYFRAME = (1, 2, 3)
# Segment 1's rendered frame 0 (the fake colours each frame by segment and
# frame); a carried guide's first frame is frame 102 of the segment before
SEGMENT_1_FRAME_0 = (0, 0, 7)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


def anchored_chain(**overrides):
    return GuideShape().chain(
        segments=10,
        keystone="segment_1",
        experimental_capstone={"enter_below_sigma": 0.7},
        **overrides,
    )


def render(**overrides):
    shape = GuideShape()
    pipeline = FakePipeline(shape.output)
    run_chain(
        pipeline,
        anchored_chain(**overrides),
        shape.arguments(image=solid_frame(KEYFRAME)),
    )
    return pipeline.calls


def anchored(call):
    """(keystone keyframe, capstone guide) a segment call took."""
    keystone = call["image"].getpixel((0, 0)) == SEGMENT_1_FRAME_0
    capstone = any(guide.get("past_end") for guide in call.get("guides") or [])
    return keystone, capstone


def test_anchor_every_3_anchors_segments_3_6_9(flag_on):
    calls = render(**{CHAIN_ANCHOR_EVERY: 3})
    assert len(calls) == 10
    taken = [index for index, call in enumerate(calls) if any(anchored(call))]
    assert taken == [3, 6, 9]
    for index in (3, 6, 9):
        assert anchored(calls[index]) == (True, True)
    for index, call in enumerate(calls):
        if index in (0, 3, 6, 9):
            continue
        assert anchored(call) == (False, False)
        # The plain guide chain: the keyframe is the carried guide's first frame
        assert len(call["guides"]) == 1
        assert call["image"].getpixel((0, 0))[1] == 102


def test_anchor_every_1_is_every_later_segment(flag_on):
    calls = render(**{CHAIN_ANCHOR_EVERY: 1})
    assert [anchored(call) for call in calls[1:]] == [(True, True)] * 9
    # ...as the default is
    assert [anchored(call) for call in render()[1:]] == [(True, True)] * 9


def test_a_digit_string_is_read_as_its_number(flag_on):
    calls = render(**{CHAIN_ANCHOR_EVERY: "3"})
    taken = [index for index, call in enumerate(calls) if any(anchored(call))]
    assert taken == [3, 6, 9]


def test_anchor_every_spaces_a_keystone_alone(flag_on):
    shape = GuideShape()
    pipeline = FakePipeline(shape.output)
    run_chain(
        pipeline,
        shape.chain(segments=5, keystone="segment_1", **{CHAIN_ANCHOR_EVERY: 2}),
        shape.arguments(image=solid_frame(KEYFRAME)),
    )
    keyed = [anchored(call)[0] for call in pipeline.calls]
    assert keyed == [False, False, True, False, True]


def test_anchor_every_without_an_anchor_is_refused(flag_on):
    chain = GuideShape().chain(**{CHAIN_ANCHOR_EVERY: 3})
    problem = chain_anchor_every_problem(chain)
    assert "keystone" in problem and "experimental_capstone" in problem
    with pytest.raises(ValueError, match=CHAIN_ANCHOR_EVERY):
        ChainConfig(chain, GuideShape().arguments())
    errors = anchor_errors(_workflow(chain))
    assert [e["path"] for e in errors] == [
        f"steps[0].pipeline.chain.{CHAIN_ANCHOR_EVERY}"
    ]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "0", "x"])
def test_anchor_every_must_be_a_whole_number_from_1(flag_on, value):
    problem = chain_anchor_every_problem(anchored_chain(**{CHAIN_ANCHOR_EVERY: value}))
    assert "whole number" in problem


def test_anchor_every_is_refused_with_the_flag_off(flag_off):
    problem = chain_anchor_every_problem(anchored_chain(**{CHAIN_ANCHOR_EVERY: 3}))
    assert FLAG in problem


def test_absent_null_and_unresolved_pass(flag_off):
    assert chain_anchor_every_problem(anchored_chain()) is None
    assert (
        chain_anchor_every_problem(anchored_chain(**{CHAIN_ANCHOR_EVERY: None})) is None
    )
    unresolved = {CHAIN_ANCHOR_EVERY: "variable:every"}
    assert chain_anchor_every_problem(GuideShape().chain(**unresolved)) is None


def _workflow(chain):
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
                    "chain": chain,
                    "arguments": {"num_frames": 124},
                },
            }
        ]
    }
