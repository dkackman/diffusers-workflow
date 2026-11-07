"""carry_pick (#1032): the sharpest of the last N frames as the next keyframe."""

import numpy
from PIL import Image

from dw.pipeline_processors.chain_pick import sharpest_index, sharpness


def flat(value=128):
    return numpy.full((16, 16, 3), value, dtype=numpy.uint8)


def checker():
    frame = numpy.zeros((16, 16, 3), dtype=numpy.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = 255
    return frame


def test_a_blurred_last_frame_loses_to_a_sharp_one_before_it():
    frames = [flat(), checker(), flat()]
    assert sharpest_index(frames, 3) == 1


def test_window_one_is_always_the_last_frame():
    assert sharpest_index([checker(), flat()], 1) == 0


def test_ties_go_to_the_latest_frame():
    assert sharpest_index([flat(), flat(), flat()], 3) == 0


def test_the_window_bounds_the_search():
    frames = [checker(), flat(), flat()]
    assert sharpest_index(frames, 2) == 0


def test_a_window_larger_than_the_clip_uses_the_whole_clip():
    assert sharpest_index([checker(), flat()], 8) == 1


def test_sharpness_accepts_pil_and_orders_detail_over_flat():
    assert sharpness(Image.fromarray(checker())) > sharpness(Image.fromarray(flat()))
    assert sharpness(flat()) == 0.0


def test_a_tiny_frame_has_no_sharpness_to_measure():
    assert sharpness(numpy.zeros((2, 2, 3), dtype=numpy.uint8)) == 0.0


def test_the_rule_names_the_allowed_values():
    from dw.pipeline_processors.h3_rules import chain_carry_pick_problem

    assert chain_carry_pick_problem({}) is None
    assert chain_carry_pick_problem({"carry_pick": {"sharpest": 4}}) is None
    for bad in ({"sharpest": 0}, {"sharpest": 9}, {"best": 3}, 3, {}):
        assert "1" in chain_carry_pick_problem({"carry_pick": bad})
    problem = chain_carry_pick_problem(
        {"carry_pick": {"sharpest": 3}, "continuity": "guide"}
    )
    assert "last_frame" in problem


def test_the_rule_refuses_match_audio_and_a_window_that_fills_the_segment():
    from dw.pipeline_processors.h3_rules import chain_carry_pick_problem

    pick = {"sharpest": 4}
    assert "match_audio" in chain_carry_pick_problem(
        {"carry_pick": pick, "match_audio": True}
    )
    problem = chain_carry_pick_problem({"carry_pick": pick, "trim_frames": 1}, 5)
    assert "5" in problem and "trim_frames 1" in problem
    assert chain_carry_pick_problem({"carry_pick": pick, "trim_frames": 1}, 6) is None
    assert chain_carry_pick_problem({"carry_pick": pick}) is None


def test_validation_refuses_match_audio():
    from dw.guides import anchor_errors

    definition = {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "chain": {"match_audio": True, "carry_pick": {"sharpest": 3}},
                    "arguments": {"num_frames": 49},
                },
            }
        ]
    }
    assert any("match_audio" in e["message"] for e in anchor_errors(definition))


def test_a_reference_for_match_audio_is_not_read_as_set():
    # a variable that resolves to false must not refuse the pick at validate (#1232)
    from dw.pipeline_processors.h3_rules import (
        chain_carry_pick_problem,
        guide_backoff_problems,
    )

    pick = {"carry_pick": {"sharpest": 3}, "match_audio": "variable:use_audio"}
    assert chain_carry_pick_problem(pick) is None
    assert chain_carry_pick_problem({**pick, "match_audio": True}) is not None
    backoff = {"continuity": "guide", "guide_backoff": 17}
    assert guide_backoff_problems({**backoff, "match_audio": "variable:a"}) == []
    assert guide_backoff_problems({**backoff, "match_audio": True})


def test_the_pick_window_is_checked_against_the_snapped_length():
    # 9 frames render as 22 (17n+5), so a window of 8 fits; the raw 9 - 1 = 8
    # would have refused it
    from dw.pipeline_processors.h3_rules import chain_carry_pick_problem

    snap = {"modulus": 17, "remainder": 5}
    chain = {"carry_pick": {"sharpest": 8}, "frame_snap": snap}
    assert chain_carry_pick_problem(chain, 9) is None
    assert chain_carry_pick_problem({"carry_pick": {"sharpest": 8}}, 9) is not None
