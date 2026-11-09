"""The per-segment check a chain runs before it keeps a segment: the seam jump
and look against the previous segment, and texture drift against segment 1
(dw/pipeline_processors/chain_check.py), and the worst-case minutes its
re-rolls can cost (dw/cost_scaling.py)."""

import numpy
import pytest
from PIL import Image

from dw.pipeline_processors.chain_check import (
    AUDIO_DBFS_MIN,
    CUT_RATIO_MAX,
    JUMP_RATIO_MAX,
    LOOK_RATIO_MAX,
    TEXTURE_RATIO_MAX,
    WORDS_SIMILARITY_DEFAULT,
    CheckResult,
    SegmentCheck,
    audio_dbfs,
    best_attempt,
    check_audio,
    check_segment,
    segment_check,
    texture_base,
    with_words,
)
from dw.pipeline_processors.chain_config import (
    chain_checks_segment_1,
    chain_checks_words,
    segment_line,
)
from dw.pipeline_processors.chain_rules import (
    chain_check_problem,
    chain_lines_problem,
)


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

    result = check_segment(before, after, texture_base(before), SegmentCheck())

    assert not result.passed
    assert result.failed == ["jump_ratio"]
    assert result.measures["jump_ratio"] > JUMP_RATIO_MAX


def test_a_seamless_segment_passes():
    before = frames_of(noise(1))

    result = check_segment(
        before, frames_of(noise(1)), texture_base(before), SegmentCheck()
    )

    assert result.passed and result.failed == []
    assert result.measures["jump_ratio"] == 0


def test_a_static_shot_with_near_zero_motion_does_not_fail_on_jump():
    # Both shots static: their own typical delta is ~0, under the assess
    # floor, so the seam's change is divided by the floor and not by ~0
    result = check_segment(
        grey(100), grey(105), texture_base(grey(100)), SegmentCheck()
    )

    assert "jump_ratio" not in result.failed
    assert result.measures["typical_delta"] == 2.0
    assert result.measures["jump_ratio"] == pytest.approx(2.5)


def test_hard_cut_seams_are_never_checked():
    before = frames_of(noise(1))
    after = frames_of(noise(2))

    result = check_segment(
        before, after, texture_base(before), SegmentCheck(), hard_cut=True
    )

    assert result.passed and result.failed == []
    assert result.measures == {"skipped": "hard_cut"}


def test_a_look_step_over_the_threshold_fails_on_look():
    result = check_segment(
        grey(100), grey(160), texture_base(grey(100)), SegmentCheck()
    )

    assert "look_ratio" in result.failed
    assert result.measures["look_ratio_measure"] == "lightness"


def test_texture_drift_from_segment_1_fails_both_ways():
    textured = frames_of(noise(1))
    smooth = grey(128)
    check = SegmentCheck(jump_ratio_max=1e9, look_ratio_max=1e9)

    blurred = check_segment(textured, smooth, texture_base(textured), check)
    crusted = check_segment(smooth, textured, texture_base(smooth + smooth), check)

    assert blurred.failed == ["texture_ratio"]
    assert blurred.measures["texture_ratio"] < 1 / TEXTURE_RATIO_MAX
    # A flat segment 1 has no fine band to measure a ratio against
    assert crusted.measures["texture_ratio"] is None
    assert "texture_ratio" not in crusted.failed

    # And the sharper way: a faintly textured segment 1, a crusty later one
    rng = numpy.random.default_rng(2)
    faint = frames_of((118 + rng.random((32, 32, 3)) * 20).astype(numpy.uint8))
    sharpened = check_segment(faint, textured, texture_base(faint), check)
    assert sharpened.failed == ["texture_ratio"]
    assert sharpened.measures["texture_ratio"] > TEXTURE_RATIO_MAX


def test_segment_1_texture_is_the_base_measured_once():
    textured = frames_of(noise(1))
    smooth = grey(128)
    check = SegmentCheck(jump_ratio_max=1e9, look_ratio_max=1e9)
    base = texture_base(textured)
    assert isinstance(base, float) and base > 0
    by_base = check_segment(textured, smooth, base, check)
    assert by_base.failed == ["texture_ratio"]
    # No base to measure against reads as no ratio, not a failure
    unmeasured = check_segment(textured, smooth, None, check)
    assert unmeasured.measures["texture_ratio"] is None


def test_texture_is_not_checked_when_its_threshold_is_none_or_frames_are_small():
    textured = frames_of(noise(1))
    off = SegmentCheck(1e9, 1e9, None)
    assert check_segment(textured, grey(128), texture_base(textured), off).passed
    tiny = frames_of(noise(1, size=8))
    small = check_segment(tiny, tiny, texture_base(tiny), SegmentCheck())
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
        assert chain_check_problem({"check": {"audio_dbfs_min": -60}}) is None
        assert chain_check_problem({"check": {"audio_dbfs_min": None}}) is None
        assert chain_check_problem({"check": {"audio_dbfs_min": "variable:f"}}) is None

    @pytest.mark.parametrize(
        "chain, words",
        [
            ({"check": "yes"}, "check"),
            ({"check": {"gloss": 1}}, "unknown"),
            ({"check": {"jump_ratio_max": 0}}, "jump_ratio_max"),
            ({"check": {"jump_ratio_max": True}}, "jump_ratio_max"),
            ({"check": {"texture_ratio_max": 0.5}}, "texture_ratio_max"),
            ({"check": {"face": 1}}, "face"),
            ({"check": {"audio_dbfs_min": 0}}, "audio_dbfs_min"),
            ({"check": {"audio_dbfs_min": 6}}, "below 0"),
            ({"check": {"audio_dbfs_min": True}}, "audio_dbfs_min"),
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
    # A chain that generates its audio checks segment 1 for its level, so
    # every segment can run 1 + retries times
    for segments, retries in ((3, 1), (5, 2), (5, 3)):
        quoted = _minutes(segments, True, retries)
        worst = per_segment * segments * (1 + retries)
        assert quoted["minutes"] == pytest.approx(worst), (segments, retries)
        assert f"of each of {segments} checked" in quoted["scaled"]


@pytest.mark.parametrize(
    "check, chain_extra, step_extra",
    [
        ({"audio_dbfs_min": None}, {}, {}),
        (True, {}, {"hold_audio": "asset:track.wav"}),
    ],
)
def test_the_estimate_leaves_segment_1_out_when_no_audio_is_measured(
    check, chain_extra, step_extra
):
    # Floor off or the audio held: segment 1 has nothing to check, so of N
    # segments N - 1 can each run 1 + retries times
    from dw.plan import estimate

    for segments, retries in ((3, 1), (5, 2), (5, 3)):
        definition, expanded = _priced(segments, check, retries)
        step = definition["steps"][0]["pipeline"]
        step["chain"].update(chain_extra)
        step["arguments"].update(step_extra)
        quoted = estimate(definition, expanded, {}, "cuda", None, None)
        worst = 10.0 * (1 + (segments - 1) * (1 + retries))
        assert quoted["minutes"] == pytest.approx(worst), (segments, retries)
        assert f"of each of {segments - 1} checked" in quoted["scaled"]


def test_a_match_audio_chain_holds_its_audio():
    from dw.cost_scaling import _chain_checks_audio

    definition, _ = _priced(3, True, 1)
    assert _chain_checks_audio(definition, {})
    definition["steps"][0]["pipeline"]["chain"]["match_audio"] = True
    assert not _chain_checks_audio(definition, {})


def test_a_guide_chain_carrying_its_audio_checks_segment_1():
    # The guide holds only the span the head trim drops, so the kept audio
    # is generated and segment 1 is floored like any other chain's
    from dw.plan import estimate

    definition, expanded = _priced(3, True, 1)
    definition["steps"][0]["pipeline"]["chain"]["continuity"] = "guide"
    quoted = estimate(definition, expanded, {}, "cuda", None, None)
    assert quoted["minutes"] == pytest.approx(10.0 * 3 * 2)


def test_a_guide_chain_that_does_not_carry_its_audio_checks_segment_1():
    from dw.plan import estimate

    definition, expanded = _priced(3, True, 1)
    definition["steps"][0]["pipeline"]["chain"].update(
        {"continuity": "guide", "carry_audio": False}
    )
    quoted = estimate(definition, expanded, {}, "cuda", None, None)
    assert quoted["minutes"] == pytest.approx(10.0 * 3 * 2)


@pytest.mark.parametrize(
    "chain_extra, step_extra",
    [
        ({"check": True}, {}),
        ({"check": {"audio_dbfs_min": -50}}, {}),
        ({"check": {"audio_dbfs_min": None}}, {}),
        ({"check": {"audio_dbfs_min": "variable:floor"}}, {}),
        ({}, {}),
        ({"check": True}, {"hold_audio": "asset:track.wav"}),
        ({"check": True, "continuity": "guide"}, {}),
        ({"check": True, "continuity": "guide", "carry_audio": False}, {}),
    ],
)
def test_the_estimate_and_the_run_agree_on_segment_1(chain_extra, step_extra):
    # One owner (#1221): the estimate's checked-segment count and the run's
    # ChainConfig ask chain_config.chain_checks_segment_1, so they cannot drift
    from dw.cost_scaling import _chain_checks_audio
    from dw.pipeline_processors.chain import ChainConfig

    variables = {"floor": None}
    definition, _ = _priced(3, None, 1)
    step = definition["steps"][0]["pipeline"]
    step["chain"].update(chain_extra)
    step["arguments"].update(step_extra)
    estimated = _chain_checks_audio(definition, variables)

    run_chain = {"segments": 3, **chain_extra}
    if isinstance(run_chain.get("check"), dict):
        run_chain["check"] = {
            key: variables["floor"] if str(value).startswith("variable:") else value
            for key, value in run_chain["check"].items()
        }
    run_arguments = {"hold_audio": object()} if step_extra else {}
    assert ChainConfig(run_chain, run_arguments).checks_segment_1 == estimated


def test_the_estimate_is_unchanged_with_no_check_or_no_retries():
    assert _minutes(3, None, None)["minutes"] == pytest.approx(30)
    assert _minutes(3, True, 0)["minutes"] == pytest.approx(30)
    assert _minutes(5, True, 0)["minutes"] == pytest.approx(50)


def test_a_cut_inside_the_segment_fails_on_cut_ratio():
    # Seamless at the seam, then a cut to another picture two frames in
    before = frames_of(noise(1))
    after = frames_of(noise(1), 2) + frames_of(noise(7), 10)

    result = check_segment(before, after, texture_base(before), SegmentCheck())

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

    result = check_segment(after[:1] * 4, after, texture_base(after), SegmentCheck())

    assert "cut_ratio" not in result.failed
    assert result.measures["cut_ratio"] <= 1.5


def test_cut_ratio_null_skips_it():
    before = frames_of(noise(1))
    after = frames_of(noise(1), 2) + frames_of(noise(7), 10)

    check = segment_check({"cut_ratio_max": None})
    result = check_segment(before, after, texture_base(before), check)

    assert result.passed
    assert chain_check_problem({"check": {"cut_ratio_max": 0}})
    assert chain_check_problem({"check": {"cut_ratio_max": None}}) is None


def tone(dbfs, seconds=1.0, rate=100):
    """A stereo sine whose RMS level reads `dbfs`."""
    amplitude = 10 ** (dbfs / 20) * numpy.sqrt(2)
    t = numpy.arange(int(seconds * rate)) / rate
    wave = amplitude * numpy.sin(2 * numpy.pi * 7 * t)
    return numpy.stack([wave, wave])


class TestAudioFloor:
    """The audio-level floor (#1221): a segment whose generated audio reads
    under `audio_dbfs_min` fails and re-rolls, segment 1 included."""

    def test_the_measure_reads_silent_quiet_and_loud_audio(self):
        assert audio_dbfs(numpy.zeros((2, 100))) == -120.0
        assert audio_dbfs(tone(-88.5)) == pytest.approx(-88.5, abs=0.1)
        assert audio_dbfs(tone(-57)) == pytest.approx(-57, abs=0.1)
        assert audio_dbfs(tone(-12)) == pytest.approx(-12, abs=0.1)
        assert audio_dbfs(None) is None
        assert audio_dbfs(numpy.zeros((2, 0))) is None

    def test_the_default_floor_sits_between_a_dropout_and_the_quietest_take(self):
        assert SegmentCheck().audio_dbfs_min == AUDIO_DBFS_MIN == -70.0
        assert not check_audio(tone(-88.5), SegmentCheck()).passed
        assert check_audio(tone(-57), SegmentCheck()).passed
        silent = check_audio(numpy.zeros((2, 100)), SegmentCheck())
        assert silent.failed == ["audio_dbfs"]
        assert silent.measures == {"audio_dbfs": -120.0}

    def test_a_null_floor_or_no_audio_skips_it(self):
        off = segment_check({"audio_dbfs_min": None})
        assert off.audio_dbfs_min is None
        assert check_audio(numpy.zeros((2, 100)), off).passed
        none = check_audio(None, SegmentCheck())
        assert none.passed and none.measures == {"audio_dbfs": None}

    def test_segment_check_reads_the_floor(self):
        assert segment_check({"audio_dbfs_min": -50}).audio_dbfs_min == -50.0
        assert segment_check(True).audio_dbfs_min == AUDIO_DBFS_MIN

    def test_a_silent_seam_fails_on_audio_alongside_the_frame_measures(self):
        before = frames_of(noise(1))
        result = check_segment(
            before,
            frames_of(noise(1)),
            texture_base(before),
            SegmentCheck(),
            audio=numpy.zeros((2, 100)),
        )
        assert result.failed == ["audio_dbfs"]
        assert result.measures["audio_dbfs"] == -120.0
        loud = check_segment(
            before,
            frames_of(noise(1)),
            texture_base(before),
            SegmentCheck(),
            audio=tone(-20),
        )
        assert loud.passed
        assert loud.measures["audio_dbfs"] == pytest.approx(-20, abs=0.1)

    def test_held_audio_is_never_measured(self):
        from dw.pipeline_processors.chain import ChainConfig

        generated = ChainConfig({"segments": 2, "check": True}, {})
        assert generated.audio_floor == AUDIO_DBFS_MIN
        assert generated.checks_segment_1
        unchecked = ChainConfig({"segments": 2}, {})
        assert unchecked.audio_floor == AUDIO_DBFS_MIN
        assert not unchecked.checks_segment_1
        off = ChainConfig({"segments": 2, "check": {"audio_dbfs_min": None}}, {})
        assert off.audio_floor is None and not off.checks_segment_1
        step_held = ChainConfig(
            {"segments": 2, "check": True}, {"hold_audio": object()}
        )
        assert step_held.audio_held and step_held.audio_floor is None
        # A guide carrying its audio holds only the span its head trim drops:
        # the kept audio is generated, so it is floored and segment 1 checked
        guide = ChainConfig({"segments": 2, "check": True, "continuity": "guide"}, {})
        assert guide.guide_holds_audio
        assert not guide.audio_held and guide.audio_floor == AUDIO_DBFS_MIN
        assert guide.checks_segment_1


def _observed_quote(check, retries, **block):
    from dw.plan import estimate

    definition, expanded = _priced(3, check, retries)
    history = {"cold_minutes": 30.0, "cold_runs": 10, "device": "cuda", **block}
    return estimate(definition, expanded, {}, "cuda", None, None, observed=history)


def test_an_observed_estimate_quotes_the_worst_case_retries():
    # #1246: the box's own history is read as N renders, like the curated row
    quoted = _observed_quote(True, 1)
    assert quoted["minutes"] == pytest.approx(60.0)
    assert quoted["basis"] == "derived"
    assert "6 renders for 3" in quoted["scaled"]


def test_an_observed_worst_case_holds_the_fixed_share():
    # A re-roll renders a segment again; it does not load the model again.
    # The entry's fixed share (12 of 30 min) is held, as the curated path
    # holds it (#1224)
    from dw.plan import estimate

    definition, expanded = _priced(3, True, 1)
    definition["cost"][0]["fixed_minutes"] = 12
    history = {"cold_minutes": 30.0, "cold_runs": 10, "device": "cuda"}
    quoted = estimate(definition, expanded, {}, "cuda", None, None, observed=history)
    assert quoted["minutes"] == pytest.approx(12 + 18 * 2)


def test_an_inherited_estimate_quotes_the_worst_case_retries():
    quoted = _observed_quote(True, 1, inherited_from="templates/ltx2/chained-segments")
    assert quoted["minutes"] == pytest.approx(60.0)
    assert quoted["basis"] == "inherited"
    assert "6 renders for 3" in quoted["scaled"]
    held = _observed_quote({"audio_dbfs_min": None}, 2, inherited_from="t")
    assert held["minutes"] == pytest.approx(30.0 * (1 + 2 * 3) / 3)
    assert "7 renders for 3" in held["scaled"]


def test_an_observed_estimate_without_re_rolls_is_unchanged():
    for check, retries in ((None, None), (False, 0), (True, 0)):
        quoted = _observed_quote(check, retries, inherited_from="t")
        assert quoted["minutes"] == pytest.approx(30.0), (check, retries)
        assert "scaled" not in quoted


@pytest.mark.parametrize("floor", [0, 6, 0.5])
def test_a_literal_floor_at_or_above_0_is_refused_saying_below_0(floor, tmp_path):
    # #1246: the schema leaves the range to chain_check_problem, so a literal
    # number reads the same refusal as a reference that resolves to one
    import json
    from pathlib import Path

    from dw.validation import workflow_errors
    from dw.workflow import workflow_from_definition

    template = (
        Path(__file__).parent.parent / "workflows/templates/ltx2/chained-segments.json"
    )
    definition = json.loads(template.read_text())
    definition["steps"][0]["pipeline"]["chain"]["check"] = {"audio_dbfs_min": floor}
    errors = workflow_errors(workflow_from_definition(definition, str(tmp_path)))
    messages = " ".join(str(e) for e in errors)
    assert "audio_dbfs_min" in messages
    assert "below 0" in messages
    assert "greater than or equal to the maximum" not in messages


def _words(similarity=0.95, passed=True):
    return {
        "similarity": similarity,
        "heard": "hello there",
        "findings": [],
        "passed": passed,
        "seam_head_dropped": [],
    }


class TestWordsMeasure:
    def test_the_default_is_off(self):
        assert WORDS_SIMILARITY_DEFAULT is None
        assert SegmentCheck().words_similarity_min is None
        assert segment_check(True).words_similarity_min is None

    def test_segment_check_reads_the_minimum(self):
        assert segment_check({"words_similarity_min": 0.8}).words_similarity_min == 0.8
        assert segment_check({"words_similarity_min": 1}).words_similarity_min == 1.0
        assert (
            segment_check({"words_similarity_min": None}).words_similarity_min is None
        )

    def test_a_passing_answer_is_recorded_and_does_not_fail(self):
        result = CheckResult(True, {"jump_ratio": 1.0}, [], None)
        answer = _words()
        out = with_words(result, answer)
        assert out is result
        assert out.measures["words"] is answer
        assert out.passed and out.failed == []

    def test_a_failing_answer_fails_the_words_measure(self):
        result = CheckResult(True, {"jump_ratio": 1.0}, [], None)
        out = with_words(result, _words(0.5, passed=False))
        assert not out.passed
        assert out.failed == ["words"]

    def test_a_failing_answer_keeps_the_other_failures(self):
        result = CheckResult(False, {}, ["jump_ratio"], None)
        out = with_words(result, _words(0.5, passed=False))
        assert out.failed == ["jump_ratio", "words"]

    def test_none_records_null_and_never_fails(self):
        result = CheckResult(True, {}, [], None)
        out = with_words(result, None)
        assert "words" in out.measures and out.measures["words"] is None
        assert out.passed and out.failed == []

    def test_a_segment_one_result_can_carry_words(self):
        out = with_words(check_audio(None, SegmentCheck()), _words(0.4, False))
        assert out.failed == ["words"]


def _failed(*names, similarity=None, jump=None):
    measures = {}
    if jump is not None:
        measures["jump_ratio"] = jump
    if similarity is not None:
        measures["words"] = _words(similarity, passed=False)
    return CheckResult(not names, measures, list(names), None)


class TestBestAttemptWithWords:
    def test_words_only_failures_rank_by_higher_similarity(self):
        results = [
            _failed("words", similarity=0.3, jump=1.0),
            _failed("words", similarity=0.7, jump=9.0),
            _failed("words", similarity=0.5, jump=0.1),
        ]
        assert best_attempt(results) == 1

    def test_equal_similarity_keeps_the_earliest(self):
        results = [
            _failed("words", similarity=0.6),
            _failed("words", similarity=0.6),
        ]
        assert best_attempt(results) == 0

    def test_a_words_only_failure_ranks_after_another_one_measure_failure(self):
        results = [
            _failed("words", similarity=0.99, jump=0.5),
            _failed("jump_ratio", jump=50.0),
        ]
        assert best_attempt(results) == 1

    def test_a_pass_beats_a_words_only_failure(self):
        results = [_failed("words", similarity=0.99), _failed(jump=3.0)]
        assert best_attempt(results) == 1

    def test_a_words_only_failure_beats_a_two_measure_failure(self):
        results = [
            _failed("jump_ratio", "words", similarity=0.9, jump=1.0),
            _failed("words", similarity=0.1),
        ]
        assert best_attempt(results) == 1

    def test_a_null_similarity_ranks_last_among_words_only(self):
        results = [
            CheckResult(False, {"words": _words(None, False)}, ["words"], None),
            _failed("words", similarity=0.1),
        ]
        assert best_attempt(results) == 1

    def test_without_words_the_ranking_is_unchanged(self):
        results = [_failed("jump_ratio", jump=9.0), _failed("jump_ratio", jump=4.0)]
        assert best_attempt(results) == 1


class TestWordsRules:
    @pytest.mark.parametrize("value", [0.8, 1, 0.01, None, "variable:x"])
    def test_accepted(self, value):
        check = {"words_similarity_min": value}
        assert chain_check_problem({"check": check}) is None

    @pytest.mark.parametrize("value", [0, 0.0, -0.5, 1.5, "high", True, False, [0.5]])
    def test_refused_naming_the_key(self, value):
        problem = chain_check_problem({"check": {"words_similarity_min": value}})
        assert problem and "words_similarity_min" in problem

    def test_lines_accept_none_a_reference_and_the_right_length(self):
        assert chain_lines_problem(None, 3) is None
        assert chain_lines_problem("variable:lines", 3) is None
        assert chain_lines_problem(["a", "b", "c"], 3) is None
        assert chain_lines_problem(["a", "b"], None) is None
        assert chain_lines_problem(["", "b"], 2) is None

    @pytest.mark.parametrize(
        "lines, count, words",
        [
            ("just a string", 1, "list of strings"),
            ({"a": 1}, 1, "list of strings"),
            ([], 2, "list of strings"),
            (["a", 2], 2, "only strings"),
            ([None, "a"], 2, "only strings"),
            (["a", "b"], 3, "2 line(s) for 3"),
            (["a", "b", "c"], 2, "3 line(s) for 2"),
        ],
    )
    def test_lines_refused(self, lines, count, words):
        assert words in chain_lines_problem(lines, count)

    def test_segment_line_prefers_lines_and_never_repeats_them(self):
        lines = ["one", "", "  ", "four"]
        assert segment_line(lines, None, "p", 0) == "one"
        assert segment_line(lines, None, "p", 1) is None
        assert segment_line(lines, None, "p", 2) is None
        assert segment_line(lines, None, "p", 3) == "four"
        assert segment_line(lines, None, "<d>fallback</d>", 4) is None

    def test_lines_win_over_the_prompts(self):
        assert segment_line(["mine"], ["<d>theirs</d>"], "<d>step</d>", 0) == "mine"

    def test_segment_line_reads_the_prompts_repeating_the_last(self):
        prompts = ["<d>first</d>", "no dialogue here", "<d>third</d>"]
        assert segment_line(None, prompts, "<d>step</d>", 0) == "first"
        assert segment_line(None, prompts, "<d>step</d>", 1) is None
        assert segment_line(None, prompts, "<d>step</d>", 2) == "third"
        assert segment_line(None, prompts, "<d>step</d>", 7) == "third"

    def test_segment_line_falls_back_to_the_step_prompt(self):
        assert segment_line(None, None, "A man: <d>hi there</d>", 3) == "hi there"
        assert segment_line(None, [], "A man: <d>hi there</d>", 0) == "hi there"
        assert segment_line(None, None, "no dialogue", 0) is None
        assert segment_line(None, None, None, 0) is None
        assert segment_line([], None, "<d>x</d>", 0) == "x"

    @pytest.mark.parametrize(
        "chain, arguments, words, segment_1",
        [
            ({}, {}, False, False),
            ({"check": False}, {}, False, False),
            ({"check": True}, {}, False, True),
            ({"check": {"words_similarity_min": 0.8}}, {}, True, True),
            ({"check": {"words_similarity_min": None}}, {}, False, True),
            # words alone checks segment 1, with the audio floor off
            (
                {"check": {"words_similarity_min": 0.8, "audio_dbfs_min": None}},
                {},
                True,
                True,
            ),
            (
                {"check": {"words_similarity_min": None, "audio_dbfs_min": None}},
                {},
                False,
                False,
            ),
            ({"check": {"words_similarity_min": "variable:w"}}, {}, True, True),
            # a held audio track is not the segment's own speech
            (
                {"check": {"words_similarity_min": 0.8}, "match_audio": True},
                {},
                False,
                False,
            ),
            (
                {"check": {"words_similarity_min": 0.8}},
                {"hold_audio": "asset:t.wav"},
                False,
                False,
            ),
            # check true is the default ceilings: the words measure stays off
            ({"check": True, "match_audio": True}, {}, False, False),
            # no check, no words
            ({"words_similarity_min": 0.8}, {}, False, False),
        ],
    )
    def test_truth_table(self, chain, arguments, words, segment_1):
        assert chain_checks_words(chain, arguments) is words
        assert chain_checks_segment_1(chain, arguments) is segment_1

    def test_chain_config_reads_lines_and_asks_the_rules(self):
        from dw.pipeline_processors.chain import ChainConfig

        config = ChainConfig(
            {
                "segments": 2,
                "check": {"words_similarity_min": 0.8},
                "lines": ["a", "b"],
            },
            {"prompt": "<d>step</d>"},
        )
        assert config.checks_words and config.checks_segment_1
        assert config.check.words_similarity_min == 0.8
        assert config.segment_line(0) == "a" and config.segment_line(1) == "b"
        plain = ChainConfig({"segments": 2, "check": True}, {"prompt": "<d>s</d>"})
        assert not plain.checks_words
        assert plain.segment_line(1) == "s"

    def test_chain_config_refuses_lines_of_the_wrong_length(self):
        from dw.pipeline_processors.chain import ChainConfig

        with pytest.raises(ValueError, match="2 line"):
            ChainConfig({"segments": 3, "lines": ["a", "b"]}, {})

    def test_the_schema_takes_the_new_keys(self):
        import json
        from pathlib import Path

        import jsonschema

        schema = json.loads(
            (Path(__file__).parent.parent / "dw/workflow_schema.json").read_text()
        )
        chain = {"$defs": schema["$defs"], **schema["$defs"]["chain"]}
        for value in (0.8, None, "variable:w"):
            jsonschema.validate(
                {"segments": 2, "check": {"words_similarity_min": value}}, chain
            )
        jsonschema.validate({"segments": 2, "lines": ["a", "b"]}, chain)
        jsonschema.validate({"segments": 2, "lines": "variable:lines"}, chain)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"segments": 2, "lines": [1, 2]}, chain)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"segments": 2, "lines": 3}, chain)


class TestWordsEstimate:
    WORDS = {"words_similarity_min": 0.8}

    def test_a_words_checked_chain_quotes_more_than_without(self):
        unchecked = _minutes(3, None, None)
        null = _minutes(3, {"words_similarity_min": None}, 1)
        words = _minutes(3, self.WORDS, 1)
        assert unchecked["minutes"] == pytest.approx(30)
        assert null["minutes"] == pytest.approx(60)
        assert words["minutes"] > null["minutes"] > unchecked["minutes"]
        # 3 segments, segment 1 included, 2 attempts each = 6 passes
        assert words["minutes"] == pytest.approx(60 + 6 * 0.05)

    def test_the_scaled_text_names_the_words_check(self):
        words = _minutes(3, self.WORDS, 1)
        assert "words check" in words["scaled"]
        assert "6 Whisper pass" in words["scaled"]
        assert "words check" not in _minutes(3, True, 1)["scaled"]

    def test_the_term_is_additive_not_a_factor(self):
        # the ratio of 5 segments to 3 is not applied to the Whisper passes
        words = _minutes(5, self.WORDS, 2)
        assert words["minutes"] == pytest.approx(10.0 * 5 * 3 + 15 * 0.05, abs=0.06)

    def test_a_held_audio_chain_adds_nothing(self):
        from dw.plan import estimate

        definition, expanded = _priced(3, self.WORDS, 1)
        definition["steps"][0]["pipeline"]["chain"]["match_audio"] = True
        quoted = estimate(definition, expanded, {}, "cuda", None, None)
        assert "words check" not in quoted.get("scaled", "")

    def test_the_estimate_and_the_run_agree(self):
        from dw.cost_scaling import _chain_checks_words
        from dw.pipeline_processors.chain import ChainConfig

        for check, expected in (
            (self.WORDS, True),
            ({"words_similarity_min": None}, False),
            (True, False),
        ):
            definition, _ = _priced(3, check, 1)
            assert _chain_checks_words(definition, {}) is expected
            config = ChainConfig({"segments": 3, "check": check}, {})
            assert config.checks_words is expected


class TestWordsValidation:
    """The whole-workflow path: the chained-segments template with the words
    keys written into its chain block (#1262)."""

    @staticmethod
    def _workflow(tmp_path, **chain):
        import json
        from pathlib import Path

        from dw.workflow import workflow_from_definition

        template = (
            Path(__file__).parent.parent
            / "workflows/templates/ltx2/chained-segments.json"
        )
        definition = json.loads(template.read_text())
        block = definition["steps"][0]["pipeline"]["chain"]
        block.pop("duration_seconds", None)
        block["segments"] = 3
        block.update(chain)
        definition["variables"].update(words=0.8, check=True)
        return workflow_from_definition(definition, str(tmp_path))

    def _errors(self, tmp_path, **chain):
        from dw.validation import workflow_errors

        errors = workflow_errors(self._workflow(tmp_path, **chain))
        return [e for e in errors if "chain" in str(e.get("path"))], errors

    def _warnings(self, tmp_path, **chain):
        from dw import validation

        workflow = self._workflow(tmp_path, **chain)
        return validation.run_warning_check(workflow, "chain_lines_warnings", None)

    @pytest.mark.parametrize("value", [0, 1.5, "high", -1, True])
    def test_a_bad_minimum_is_refused(self, tmp_path, value):
        found, errors = self._errors(tmp_path, check={"words_similarity_min": value})
        messages = " ".join(str(e) for e in errors)
        assert "words_similarity_min" in messages, errors

    @pytest.mark.parametrize("value", [0.8, 1, None, "variable:words"])
    def test_a_good_minimum_is_accepted(self, tmp_path, value):
        _, errors = self._errors(tmp_path, check={"words_similarity_min": value})
        assert "words_similarity_min" not in " ".join(str(e) for e in errors)

    def test_lines_of_the_right_length_are_accepted(self, tmp_path):
        found, errors = self._errors(
            tmp_path,
            check={"words_similarity_min": 0.8},
            lines=["a", "b", "c"],
        )
        assert "lines" not in " ".join(str(e) for e in errors), errors

    def test_lines_of_the_wrong_length_are_refused_at_their_path(self, tmp_path):
        _, errors = self._errors(
            tmp_path, check={"words_similarity_min": 0.8}, lines=["a", "b"]
        )
        (error,) = [e for e in errors if str(e["path"]).endswith("chain.lines")]
        assert error["path"] == "steps[0].pipeline.chain.lines"
        assert "2 line(s) for 3" in error["message"]

    def test_a_non_string_entry_is_refused(self, tmp_path):
        _, errors = self._errors(
            tmp_path, check={"words_similarity_min": 0.8}, lines=["a", 2, "c"]
        )
        # the schema refuses the entry before the rule is asked
        assert any("chain.lines" in str(e["path"]) for e in errors), errors

    def test_lines_with_the_check_off_validate_but_warn(self, tmp_path):
        found, errors = self._errors(tmp_path, check=False, lines=["a", "b", "c"])
        assert "lines" not in " ".join(str(e) for e in errors), errors
        (warning,) = self._warnings(tmp_path, check=False, lines=["a", "b", "c"])
        assert "nothing reads 'lines'" in warning
        assert warning.startswith("steps[0].pipeline.chain.lines:")

    def test_lines_warn_with_a_null_minimum_or_no_check(self, tmp_path):
        lines = ["a", "b", "c"]
        assert self._warnings(tmp_path, check=True, lines=lines)
        assert self._warnings(
            tmp_path, check={"words_similarity_min": None}, lines=lines
        )
        assert self._warnings(tmp_path, lines=lines)

    def test_lines_do_not_warn_when_the_words_check_reads_them(self, tmp_path):
        lines = ["a", "b", "c"]
        assert (
            self._warnings(tmp_path, check={"words_similarity_min": 0.8}, lines=lines)
            == []
        )

    def test_unresolved_references_do_not_warn(self):
        from dw.guides import chain_lines_warnings

        lines = ["a", "b", "c"]
        for check in (
            "variable:check",
            {"words_similarity_min": "variable:words"},
        ):
            step = {
                "name": "s",
                "pipeline": {"chain": {"segments": 3, "check": check, "lines": lines}},
            }
            assert chain_lines_warnings({"steps": [step]}) == [], check

    def test_no_lines_no_warning(self, tmp_path):
        assert self._warnings(tmp_path, check=False) == []
        assert self._warnings(tmp_path, check=True) == []

    def test_the_warning_check_is_registered(self):
        from dw import validation

        assert "chain_lines_warnings" in [c.name for c in validation.WARNING_CHECKS]
