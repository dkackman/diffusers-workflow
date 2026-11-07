"""The per-segment check a chain runs before it keeps a segment: the seam jump
and look against the previous segment, and texture drift against segment 1
(dw/pipeline_processors/chain_check.py), and the worst-case minutes its
re-rolls can cost (dw/cost_scaling.py)."""

import numpy
import pytest
from PIL import Image

from dw.pipeline_processors.chain_check import (
    CUT_RATIO_MAX,
    JUMP_RATIO_MAX,
    LOOK_RATIO_MAX,
    TEXTURE_RATIO_MAX,
    SegmentCheck,
    check_segment,
    segment_check,
)
from dw.pipeline_processors.chain_rules import chain_check_problem


def noise(seed, size=32):
    rng = numpy.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=numpy.uint8)


def frames_of(array, count=8):
    return [Image.fromarray(array) for _ in range(count)]


def grey(level, size=32, count=8):
    return frames_of(numpy.full((size, size, 3), level, dtype=numpy.uint8), count)


def ramp(reverse=False, size=32):
    row = numpy.linspace(0, 255, size)
    plane = numpy.tile(row[::-1] if reverse else row, (size, 1))
    return numpy.stack([plane] * 3, axis=-1).astype(numpy.uint8)


def test_a_jump_over_the_threshold_fails_and_names_the_measure():
    # The picture flips across the seam, and its look does not move
    before = frames_of(ramp())
    after = frames_of(ramp(reverse=True))

    result = check_segment(before, after, before, SegmentCheck())

    assert not result.passed
    assert result.failed == ["jump_ratio"]
    assert result.measures["jump_ratio"] > JUMP_RATIO_MAX


def test_a_seamless_segment_passes():
    before = frames_of(noise(1))

    result = check_segment(before, frames_of(noise(1)), before, SegmentCheck())

    assert result.passed and result.failed == []
    assert result.measures["jump_ratio"] == 0


def test_a_static_shot_with_near_zero_motion_does_not_fail_on_jump():
    # Both shots static: their own typical delta is ~0, under the assess
    # floor, so the seam's change is divided by the floor and not by ~0
    result = check_segment(grey(100), grey(105), grey(100), SegmentCheck())

    assert "jump_ratio" not in result.failed
    assert result.measures["typical_delta"] == 2.0
    assert result.measures["jump_ratio"] == pytest.approx(2.5)


def test_hard_cut_seams_are_never_checked():
    before = frames_of(noise(1))
    after = frames_of(noise(2))

    result = check_segment(before, after, before, SegmentCheck(), hard_cut=True)

    assert result.passed and result.failed == []
    assert result.measures == {"skipped": "hard_cut"}


def test_a_look_step_over_the_threshold_fails_on_look():
    result = check_segment(grey(100), grey(160), grey(100), SegmentCheck())

    assert "look_ratio" in result.failed
    assert result.measures["look_ratio_measure"] == "lightness"


def test_texture_drift_from_segment_1_fails_both_ways():
    textured = frames_of(noise(1))
    smooth = grey(128)
    check = SegmentCheck(jump_ratio_max=1e9, look_ratio_max=1e9)

    blurred = check_segment(textured, smooth, textured, check)
    crusted = check_segment(smooth, textured, smooth + smooth, check)

    assert blurred.failed == ["texture_ratio"]
    assert blurred.measures["texture_ratio"] < 1 / TEXTURE_RATIO_MAX
    # A flat segment 1 has no fine band to measure a ratio against
    assert crusted.measures["texture_ratio"] is None
    assert "texture_ratio" not in crusted.failed


def test_texture_is_not_checked_when_its_threshold_is_none_or_frames_are_small():
    textured = frames_of(noise(1))
    off = SegmentCheck(1e9, 1e9, None)
    assert check_segment(textured, grey(128), textured, off).passed
    tiny = frames_of(noise(1, size=8))
    small = check_segment(tiny, tiny, tiny, SegmentCheck())
    assert small.measures["texture_ratio"] is None


def test_segment_check_reads_the_chain_value():
    assert segment_check(None) is None
    assert segment_check(False) is None
    assert segment_check(True) == SegmentCheck(
        JUMP_RATIO_MAX, LOOK_RATIO_MAX, TEXTURE_RATIO_MAX
    )
    custom = segment_check({"jump_ratio_max": 10, "texture_ratio_max": None})
    assert custom == SegmentCheck(10.0, LOOK_RATIO_MAX, None)


class TestRule:
    def test_absent_and_off_are_accepted(self):
        assert chain_check_problem({}) is None
        assert chain_check_problem({"check": False}) is None
        assert chain_check_problem({"check": True, "retries": 3}) is None
        assert chain_check_problem({"check": {"look_ratio_max": 8}}) is None
        assert chain_check_problem({"check": True, "retries": "variable:r"}) is None
        assert chain_check_problem({"check": "variable:c", "retries": 2}) is None

    @pytest.mark.parametrize(
        "chain, words",
        [
            ({"check": "yes"}, "check"),
            ({"check": {"gloss": 1}}, "unknown"),
            ({"check": {"jump_ratio_max": 0}}, "jump_ratio_max"),
            ({"check": {"jump_ratio_max": True}}, "jump_ratio_max"),
            ({"check": {"texture_ratio_max": 0.5}}, "texture_ratio_max"),
            ({"check": {"face": 1}}, "face"),
            ({"check": True, "retries": 4}, "0 and 3"),
            ({"check": True, "retries": -1}, "0 and 3"),
            ({"check": True, "retries": 1.5}, "0 and 3"),
            ({"check": True, "retries": True}, "0 and 3"),
            ({"retries": 1}, "needs check"),
            ({"check": False, "retries": 2}, "needs check"),
        ],
    )
    def test_bad_values_are_refused_naming_the_key(self, chain, words):
        assert words in chain_check_problem(chain)

    def test_chain_config_asks_the_rule(self):
        from dw.pipeline_processors.chain import ChainConfig

        with pytest.raises(ValueError, match="0 and 3"):
            ChainConfig({"segments": 2, "check": True, "retries": 5}, {})
        config = ChainConfig({"segments": 2}, {})
        assert config.check is None and config.retries == 0
        config = ChainConfig({"segments": 2, "check": True, "retries": 2}, {})
        assert config.check == SegmentCheck() and config.retries == 2

    def test_validation_wires_the_rule(self):
        from dw.guides import anchor_errors

        definition = {
            "steps": [
                {
                    "name": "s",
                    "pipeline": {
                        "chain": {"segments": 2, "check": True, "retries": 9},
                        "arguments": {},
                    },
                }
            ]
        }
        errors = anchor_errors(definition)
        assert any(
            "retries" in e["message"] and e["path"].endswith("chain.retries")
            for e in errors
        ), errors

    def test_the_schema_takes_both_keys(self):
        import json
        from pathlib import Path

        import jsonschema

        schema = json.loads(
            (Path(__file__).parent.parent / "dw/workflow_schema.json").read_text()
        )
        chain = {"$defs": schema["$defs"], **schema["$defs"]["chain"]}
        jsonschema.validate({"segments": 2, "check": True, "retries": 1}, chain)
        jsonschema.validate(
            {"segments": 2, "check": {"jump_ratio_max": 20, "face": None}}, chain
        )
        jsonschema.validate({"segments": 2, "retries": "variable:retries"}, chain)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"segments": 2, "retries": 4}, chain)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"segments": 2, "check": {"gloss": 1}}, chain)


def _priced(segments, check, retries):
    """A one-chain workflow measured at 3 segments in 30 minutes."""
    chain = {"segments": "variable:segments", "fps": 4}
    if check is not None:
        chain["check"] = check
    if retries is not None:
        chain["retries"] = "variable:retries"
    variables = {"segments": 3, "retries": 0}
    definition = {
        "cost": [{"device": "cuda", "name": "card", "minutes": 30}],
        "cost_drivers": ["segments"],
        "variables": variables,
        "steps": [{"pipeline": {"chain": chain, "arguments": {}}}],
    }
    expanded = {"variables": {**variables, "segments": segments, "retries": retries}}
    return definition, expanded


def _minutes(segments, check, retries):
    from dw.plan import estimate

    definition, expanded = _priced(segments, check, retries)
    return estimate(definition, expanded, {}, "cuda", None, None)


def test_the_estimate_quotes_the_worst_case_retries():
    per_segment = 10.0  # 30 minutes over the 3 segments it was measured at
    # Segment 1 is never checked, so never re-rolled: of 5 segments, 4 can
    # each run 1 + retries times
    for segments, retries in ((3, 1), (5, 2), (5, 3)):
        quoted = _minutes(segments, True, retries)
        worst = per_segment * (1 + (segments - 1) * (1 + retries))
        assert quoted["minutes"] == pytest.approx(worst), (segments, retries)
        assert "re-roll" in quoted["scaled"]


def test_the_estimate_is_unchanged_with_no_check_or_no_retries():
    assert _minutes(3, None, None)["minutes"] == pytest.approx(30)
    assert _minutes(3, True, 0)["minutes"] == pytest.approx(30)
    assert _minutes(5, True, 0)["minutes"] == pytest.approx(50)


def test_a_cut_inside_the_segment_fails_on_cut_ratio():
    # Seamless at the seam, then a cut to another picture two frames in
    before = frames_of(noise(1))
    after = frames_of(noise(1), 2) + frames_of(noise(7), 10)

    result = check_segment(before, after, before, SegmentCheck())

    assert result.failed == ["cut_ratio"]
    assert result.measures["jump_ratio"] == 0
    assert result.measures["cut_ratio"] > CUT_RATIO_MAX


def test_steady_motion_inside_the_segment_passes_cut_ratio():
    # The same step every frame: the largest is the typical one
    rng = numpy.random.default_rng(3)
    base = rng.integers(0, 200, (32, 32, 3)).astype(numpy.int16)
    after = [
        Image.fromarray(numpy.clip(base + 4 * n, 0, 255).astype(numpy.uint8))
        for n in range(12)
    ]

    result = check_segment(after[:1] * 4, after, after, SegmentCheck())

    assert "cut_ratio" not in result.failed
    assert result.measures["cut_ratio"] <= 1.5


def test_cut_ratio_null_skips_it():
    before = frames_of(noise(1))
    after = frames_of(noise(1), 2) + frames_of(noise(7), 10)

    check = segment_check({"cut_ratio_max": None})
    result = check_segment(before, after, before, check)

    assert result.passed
    assert chain_check_problem({"check": {"cut_ratio_max": 0}})
    assert chain_check_problem({"check": {"cut_ratio_max": None}}) is None
