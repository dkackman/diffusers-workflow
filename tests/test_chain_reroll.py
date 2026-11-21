"""A chain's check re-rolling a segment through run_chain (#1241): the
re-roll under save_segments, carry_pick, and on the last segment, and that the
check keeps the previous segment's small measures rather than its frames."""

from PIL import Image

from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.chain_check import (
    CheckResult,
    check_segment,
    texture_base,
    measure_frames,
)
from tests.test_chain import FakePipeline, modular_output


def _grey(level):
    return Image.new("RGB", (32, 32), (level, level, level))


def _output(bad_calls, sample_rate=100):
    """Segments of flat grey 100; the calls named in `bad_calls` (0-based over
    every render, re-rolls included) come back flat 250 - a picture jump."""

    def factory(arguments, index):
        out = modular_output(arguments, index, sample_rate=sample_rate)
        level = 250 if index in bad_calls else 100
        out["videos"] = [[_grey(level) for _ in out["videos"][0]]]
        return out

    return factory


def _chain(**overrides):
    return {
        "segments": 3,
        "fps": 4,
        "trim_frames": 2,
        "check": True,
        "retries": 1,
    } | overrides


def _levels(frames):
    return [frame.getpixel((0, 0))[0] for frame in frames]


def test_a_failed_segment_is_re_rolled_and_the_good_attempt_kept():
    pipeline = FakePipeline(_output({1}))
    result = run_chain(pipeline, _chain(), {"num_frames": 8, "seed": 5})
    assert len(pipeline.calls) == 4  # 3 segments + one re-roll of segment 2
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]
    assert set(_levels(result.frames)) == {100}


def test_the_last_segment_is_re_rolled_too():
    pipeline = FakePipeline(_output({2}))
    result = run_chain(pipeline, _chain(), {"num_frames": 8, "seed": 5})
    assert len(pipeline.calls) == 4
    assert [s["attempts"] for s in result.shots] == [1, 1, 2]
    assert set(_levels(result.frames)) == {100}


def test_every_attempt_failing_keeps_one_and_warns():
    pipeline = FakePipeline(_output({1, 2}))
    result = run_chain(pipeline, _chain(), {"num_frames": 8, "seed": 5})
    assert result.shots[1]["attempts"] == 2
    assert result.shots[1]["check_failed"]


def test_a_re_roll_under_save_segments_keeps_the_good_attempt(tmp_path):
    pipeline = FakePipeline(
        _output({1}, sample_rate=8000), output_dir=str(tmp_path), file_prefix="wf-step"
    )
    result = run_chain(
        pipeline, _chain(save_segments=True), {"num_frames": 8, "seed": 5}
    )
    try:
        assert len(pipeline.calls) == 4
        assert [s["attempts"] for s in result.shots] == [1, 2, 1]
        # Lossy segment files: the bad attempt's 250 would sit far above 100
        means = [
            float(frame.float().mean()) for chunk in result.frames for frame in chunk
        ]
        assert means and max(means) < 150
    finally:
        result.frames.cleanup()


def test_a_re_roll_under_carry_pick_keeps_the_good_attempt():
    pipeline = FakePipeline(_output({1}))
    result = run_chain(
        pipeline,
        _chain(carry_pick={"sharpest": 2}),
        {"num_frames": 8, "seed": 5},
    )
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]


def test_the_check_reads_the_previous_segments_measures_not_its_frames():
    previous = [_grey(100) for _ in range(8)]
    frames = [_grey(250) for _ in range(8)]
    base = texture_base(frames)
    by_frames = check_segment(previous, frames, base, _chain_check())
    by_measures = check_segment(measure_frames(previous), frames, base, _chain_check())
    assert by_frames.failed == by_measures.failed
    assert by_frames.measures == by_measures.measures
    assert isinstance(by_measures, CheckResult) and by_measures.frame_data is not None


def test_the_run_holds_no_frames_of_the_previous_segment(monkeypatch):
    seen = []
    import dw.pipeline_processors.chain as chain_module

    real = chain_module.check_segment

    def spy(previous, *args, **kwargs):
        seen.append(previous)
        return real(previous, *args, **kwargs)

    monkeypatch.setattr(chain_module, "check_segment", spy)
    run_chain(FakePipeline(_output(set())), _chain(), {"num_frames": 8, "seed": 5})
    assert seen and all(isinstance(p, tuple) for p in seen)


def _chain_check():
    from dw.pipeline_processors.chain_check import SegmentCheck

    return SegmentCheck()


# The words check (#1262) through the run: `check_words` is replaced by a fake
# that hears each line as given, so no Whisper runs


def _heard(similarity, passed, heard="wrong words"):
    return {
        "similarity": similarity,
        "heard": heard,
        "findings": [] if passed else [{"rule": "extra_words", "detail": "x"}],
        "passed": passed,
        "seam_head_dropped": [],
    }


class FakeWords:
    """Stands in for chain.check_words: a line in `bad` fails, any other
    passes; every call is recorded as (line, previous_line, minimum)."""

    def __init__(self, bad=()):
        self.bad = set(bad)
        self.calls = []

    def __call__(self, audio, sample_rate, line, previous_line, minimum, **kwargs):
        self.calls.append((line, previous_line, minimum))
        if line in self.bad:
            return _heard(0.4, False)
        return _heard(0.97, True, heard=line)


def _words_chain(**overrides):
    return _chain(
        check={"words_similarity_min": 0.8},
        lines=["one one", "two two", "three three"],
        **overrides,
    )


def _run_words(monkeypatch, fake, chain, arguments=None):
    from unittest.mock import patch

    import dw.pipeline_processors.chain as chain_module

    monkeypatch.setattr(chain_module, "check_words", fake)
    pipeline = FakePipeline(_output(set()))
    with patch.object(chain_module, "emit_warning") as warn:
        result = run_chain(
            pipeline, chain, {"num_frames": 8, "seed": 5, **(arguments or {})}
        )
    return pipeline, result, warn


def _chain_check_warnings(warn):
    return [
        call
        for call in warn.call_args_list
        if call.kwargs.get("kind") == "chain_check_failed"
    ]


def test_a_mismatched_line_is_re_rolled_up_to_its_retries_and_warns(monkeypatch):
    fake = FakeWords(bad={"two two"})
    pipeline, result, warn = _run_words(monkeypatch, fake, _words_chain(retries=2))

    assert len(pipeline.calls) == 5  # 3 segments + 2 re-rolls of segment 2
    assert [s["attempts"] for s in result.shots] == [1, 3, 1]
    assert result.shots[1]["check_failed"] == ["words"]
    assert "check_failed" not in result.shots[0]
    (call,) = _chain_check_warnings(warn)
    kwargs = call.kwargs
    assert kwargs["segment"] == 2
    assert kwargs["measures"] == ["words"]
    assert kwargs["heard"] == "wrong words"
    assert kwargs["similarity"] == 0.4
    assert kwargs["findings"] == [{"rule": "extra_words", "detail": "x"}]
    assert kwargs["values"] == {"words": 0.4}
    assert kwargs["attempts"] == 3
    assert kwargs["every_attempt_failed"] is True
    assert "repair-segment" in kwargs["note"]
    assert "templates/minimax/repair-segment" in call.args[0]


def test_a_finding_at_a_passing_similarity_is_named_in_the_log_and_warning(
    monkeypatch,
):
    # #1298 bounce 2: words 1.0 failed on a finding the log did not name
    from unittest.mock import patch

    import dw.pipeline_processors.chain as chain_module

    def fake(audio, sample_rate, line, previous_line, minimum, **kwargs):
        if line != "two two":
            return _heard(0.97, True, heard=line)
        return {
            "similarity": 1.0,
            "heard": "two two two two",
            "findings": [{"rule": "unscripted_speech", "detail": "x"}],
            "passed": False,
            "seam_head_dropped": [],
        }

    monkeypatch.setattr(chain_module, "check_words", fake)
    pipeline = FakePipeline(_output(set()))
    with (
        patch.object(chain_module, "emit_warning") as warn,
        patch.object(chain_module, "emit_log") as log,
    ):
        run_chain(pipeline, _words_chain(retries=1), {"num_frames": 8, "seed": 5})

    (reroll,) = [c.args[0] for c in log.call_args_list if "re-rolling" in c.args[0]]
    assert "words 1.0 with unscripted_speech, heard 'two two two two'" in reroll
    (call,) = _chain_check_warnings(warn)
    assert "words 1.0 with unscripted_speech, heard 'two two two two'" in (call.args[0])
    assert call.kwargs["values"] == {"words": 1.0}
    assert call.kwargs["findings"] == [{"rule": "unscripted_speech", "detail": "x"}]


def test_failure_text_names_each_rule_once_and_leaves_other_measures_alone():
    from types import SimpleNamespace

    from dw.pipeline_processors.chain import _failures

    words = {
        "similarity": 0.5,
        "heard": "hi",
        "findings": [
            {"rule": "tag_spoken"},
            {"rule": "tag_spoken"},
            {"rule": "transcript_loop"},
        ],
    }
    result = SimpleNamespace(
        failed=["jump_ratio", "words"],
        measures={"jump_ratio": 2.5, "words": words},
    )
    text = _failures(result)
    assert text.startswith("jump_ratio ")
    assert text.endswith("words 0.5 with tag_spoken, transcript_loop, heard 'hi'")
    words["findings"] = []
    assert _failures(result).endswith("words 0.5, heard 'hi'")


def test_every_segment_including_segment_1_records_its_words(monkeypatch):
    fake = FakeWords()
    pipeline, result, warn = _run_words(monkeypatch, fake, _words_chain())

    assert len(pipeline.calls) == 3
    assert [s["words"]["similarity"] for s in result.shots] == [0.97, 0.97, 0.97]
    assert [call[0] for call in fake.calls] == ["one one", "two two", "three three"]
    # the line before it is handed on for the seam-head rule
    assert [call[1] for call in fake.calls] == [None, "one one", "two two"]
    assert {call[2] for call in fake.calls} == {0.8}
    assert _chain_check_warnings(warn) == []
    assert all("check_failed" not in s for s in result.shots)


def test_segment_1_is_re_rolled_on_its_words(monkeypatch):
    fake = FakeWords(bad={"one one"})
    pipeline, result, warn = _run_words(monkeypatch, fake, _words_chain(retries=1))

    assert [s["attempts"] for s in result.shots] == [2, 1, 1]
    assert result.shots[0]["check_failed"] == ["words"]
    (call,) = _chain_check_warnings(warn)
    assert call.kwargs["segment"] == 1


def test_a_reroll_that_passes_its_words_keeps_the_good_take(monkeypatch):
    class Once(FakeWords):
        def __call__(self, audio, sample_rate, line, previous_line, minimum, **kwargs):
            self.calls.append((line, previous_line, minimum))
            bad = line == "two two" and len(self.calls) == 2
            return _heard(0.4, False) if bad else _heard(0.97, True, heard=line)

    pipeline, result, warn = _run_words(monkeypatch, Once(), _words_chain())
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]
    assert "check_failed" not in result.shots[1]
    assert result.shots[1]["words"]["passed"]
    assert _chain_check_warnings(warn) == []


def test_not_every_attempt_failing_is_not_flagged_so(monkeypatch):
    # Words fail together with a picture jump on the first attempt only
    from unittest.mock import patch

    import dw.pipeline_processors.chain as chain_module

    class Mixed(FakeWords):
        def __call__(self, audio, sample_rate, line, previous_line, minimum, **kwargs):
            self.calls.append((line, previous_line, minimum))
            bad = line == "two two" and len(self.calls) == 2
            return _heard(0.4, False) if bad else _heard(0.97, True, heard=line)

    monkeypatch.setattr(chain_module, "check_words", Mixed())
    pipeline = FakePipeline(_output({1}))
    with patch.object(chain_module, "emit_warning") as warn:
        result = run_chain(
            pipeline, _words_chain(retries=1), {"num_frames": 8, "seed": 5}
        )
    # Attempt 1 failed jump and words, attempt 2 (jump fine, 250 level not
    # repeated) passed both: no warning at all
    assert result.shots[1]["attempts"] == 2
    assert _chain_check_warnings(warn) == []


def _opening_missing():
    return {
        "similarity": 0.9,
        "heard": "two",
        "findings": [
            {
                "rule": "line_opening_missing",
                "severity": "warn",
                "at": {"seconds": 0.0, "word": "two"},
                "value": 0,
                "threshold": 0,
                "missing": ["one"],
                "says": "x",
            }
        ],
        "passed": True,
        "seam_head_dropped": [],
        "first_word_start": 0.0,
    }


def test_opening_missing_rerolls_while_retries_remain(monkeypatch):
    class Opening(FakeWords):
        def __call__(self, audio, sample_rate, line, previous_line, minimum, **kwargs):
            self.calls.append((line, previous_line, minimum))
            if line == "two two" and len(self.calls) == 2:
                return _opening_missing()
            return _heard(0.97, True, heard=line)

    pipeline, result, warn = _run_words(monkeypatch, Opening(), _words_chain(retries=1))
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]
    assert "check_failed" not in result.shots[1]
    assert result.shots[1]["words"]["findings"] == []
    assert _chain_check_warnings(warn) == []


def test_opening_missing_on_every_attempt_is_reported(monkeypatch):
    class Opening(FakeWords):
        def __call__(self, audio, sample_rate, line, previous_line, minimum, **kwargs):
            self.calls.append((line, previous_line, minimum))
            if line == "two two":
                return _opening_missing()
            return _heard(0.97, True, heard=line)

    pipeline, result, warn = _run_words(monkeypatch, Opening(), _words_chain(retries=1))
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]
    assert result.shots[1]["check_failed"] == ["words_opening"]
    (call,) = _chain_check_warnings(warn)
    assert call.kwargs["every_attempt_failed"] is True
    assert call.kwargs["segment"] == 2
    assert "line_opening_missing" in call.args[0]
    assert call.kwargs["findings"][0]["rule"] == "line_opening_missing"
    # #1363: the similarity passed, so the note is not about the words or carry
    assert "failed its words" not in call.kwargs["note"]
    assert "repair-segment" not in call.kwargs["note"]
    assert "narrated take" in call.kwargs["note"]


def test_segment_one_words_failure_is_not_pointed_at_repair_segment(monkeypatch):
    fake = FakeWords(bad={"one one"})
    pipeline, result, warn = _run_words(monkeypatch, fake, _words_chain(retries=1))
    (call,) = _chain_check_warnings(warn)
    assert call.kwargs["segment"] == 1
    assert "repair-segment" not in call.kwargs["note"]
    assert "carry" not in call.kwargs["note"]
    assert "different seed" in call.kwargs["note"]


def test_a_segment_with_no_line_is_neither_checked_nor_re_rolled(monkeypatch):
    fake = FakeWords(bad={"nothing"})
    chain = _chain(check={"words_similarity_min": 0.8}, retries=2)
    pipeline, result, warn = _run_words(
        monkeypatch, fake, chain, {"prompt": "a quiet street, no dialogue"}
    )

    assert fake.calls == []
    assert len(pipeline.calls) == 3
    assert [s["attempts"] for s in result.shots] == [1, 1, 1]
    assert all(s["words"] is None for s in result.shots)
    assert all("check_failed" not in s for s in result.shots)
    assert _chain_check_warnings(warn) == []


def test_lines_come_from_the_prompt_dialogue_when_the_chain_has_none(monkeypatch):
    fake = FakeWords()
    chain = _chain(check={"words_similarity_min": 0.8}, retries=0)
    _run_words(monkeypatch, fake, chain, {"prompt": "A man <d>hello there</d>"})
    assert [call[0] for call in fake.calls] == ["hello there"] * 3


def test_a_blank_line_entry_is_not_checked(monkeypatch):
    fake = FakeWords(bad={"two two"})
    chain = _words_chain(retries=1)
    chain["lines"] = ["one one", "", "three three"]
    pipeline, result, _ = _run_words(monkeypatch, fake, chain)
    assert [c[0] for c in fake.calls] == ["one one", "three three"]
    assert result.shots[1]["words"] is None
    assert len(pipeline.calls) == 3


def test_a_null_minimum_never_calls_check_words_and_records_no_words(monkeypatch):
    for check in (True, {"words_similarity_min": None}, {"audio_dbfs_min": -60}):
        fake = FakeWords(bad={"two two"})
        chain = _chain(check=check, lines=["one one", "two two", "three three"])
        pipeline, result, warn = _run_words(monkeypatch, fake, chain)
        assert fake.calls == [], check
        assert all("words" not in shot for shot in result.shots), check
        assert len(pipeline.calls) == 3
        assert _chain_check_warnings(warn) == []


def test_held_audio_never_calls_check_words(monkeypatch):
    fake = FakeWords(bad={"two two"})
    _, result, _ = _run_words(
        monkeypatch, fake, _words_chain(), {"hold_audio": object()}
    )
    assert fake.calls == []
    assert all("words" not in shot for shot in result.shots)


def test_a_words_failure_beside_a_jump_names_both(monkeypatch):
    fake = FakeWords(bad={"two two"})
    from unittest.mock import patch

    import dw.pipeline_processors.chain as chain_module

    monkeypatch.setattr(chain_module, "check_words", fake)
    pipeline = FakePipeline(_output({1, 2}))
    with patch.object(chain_module, "emit_warning") as warn:
        result = run_chain(
            pipeline, _words_chain(retries=1), {"num_frames": 8, "seed": 5}
        )
    assert {"jump_ratio", "words"} <= set(result.shots[1]["check_failed"])
    (call,) = [c for c in _chain_check_warnings(warn) if c.kwargs["segment"] == 2]
    assert call.kwargs["values"]["words"] == 0.4
    assert call.kwargs["values"]["jump_ratio"] > 1


def test_lines_of_the_wrong_length_are_refused_by_the_config():
    import pytest

    from dw.pipeline_processors.chain import ChainConfig

    with pytest.raises(ValueError, match="2 line"):
        ChainConfig(_chain(lines=["a", "b"]), {})
    with pytest.raises(ValueError, match="only strings"):
        ChainConfig(_chain(lines=["a", 1, "c"]), {})
    # an empty list is a count mismatch, named with both numbers (#1262 bounce)
    with pytest.raises(ValueError, match=r"0 line\(s\) for 3"):
        ChainConfig(_chain(lines=[]), {})


# The kept attempt's measures on every checked segment's shot (#1474), through
# the run: passing or not, each against the ceiling in force


def _shots(bad_calls=(), **overrides):
    result = run_chain(
        FakePipeline(_output(set(bad_calls))),
        _chain(**overrides),
        {"num_frames": 8, "seed": 5},
    )
    return result.shots


def test_a_passing_segment_records_its_measures_with_their_ceilings():
    from dw.pipeline_processors.chain_check import (
        AUDIO_DBFS_MIN,
        CUT_RATIO_MAX,
        JUMP_RATIO_MAX,
        LOOK_RATIO_MAX,
    )

    shot = _shots()[1]
    assert "check_failed" not in shot
    measures = shot["measures"]
    assert {"jump_ratio", "look_ratio", "cut_ratio", "audio_dbfs"} <= set(measures)
    ceilings = {
        "jump_ratio": JUMP_RATIO_MAX,
        "look_ratio": LOOK_RATIO_MAX,
        "cut_ratio": CUT_RATIO_MAX,
        "audio_dbfs": AUDIO_DBFS_MIN,
    }
    for name, ceiling in ceilings.items():
        entry = measures[name]
        assert set(entry) == {"value", "ceiling", "bound", "warn_only", "failed"}
        assert isinstance(entry["value"], float)
        assert entry["ceiling"] == ceiling
        assert entry["failed"] is False
        assert entry["warn_only"] is False
    assert measures["jump_ratio"]["bound"] == "max"


def test_an_overridden_ceiling_shows_the_workflows_value():
    measures = _shots(check={"jump_ratio_max": 3.25})[1]["measures"]
    assert measures["jump_ratio"]["ceiling"] == 3.25
    assert measures["look_ratio"]["ceiling"] == 5.5


def test_a_failed_measure_agrees_with_check_failed():
    shot = _shots({1, 2})[1]
    assert shot["check_failed"]
    failed = {name for name, entry in shot["measures"].items() if entry["failed"]}
    assert failed == set(shot["check_failed"])


def test_check_off_records_no_measures_and_leaves_the_shot_as_it_was():
    shots = _shots(check=False, retries=0)
    assert shots and all("measures" not in shot for shot in shots)
    assert all("check" not in shot for shot in shots)


def test_segment_1_carries_only_what_it_measured():
    measures = _shots()[0]["measures"]
    # No seam before it: only its audio floor is held
    assert set(measures) == {"audio_dbfs"}


def test_audio_dbfs_stays_top_level_and_is_mirrored_as_a_floor():
    for shot in _shots():
        mirrored = shot["measures"]["audio_dbfs"]
        assert shot["audio_dbfs"] == mirrored["value"]
        assert mirrored["bound"] == "min"
        assert mirrored["ceiling"] == -70.0


def test_a_null_floor_records_a_null_ceiling():
    shots = _shots(check={"audio_dbfs_min": None})
    entry = shots[1]["measures"]["audio_dbfs"]
    assert entry["ceiling"] is None and entry["failed"] is False


def test_the_existing_shot_keys_are_unchanged():
    without = run_chain(
        FakePipeline(_output({1, 2})), _chain(), {"num_frames": 8, "seed": 5}
    ).shots
    keys = [
        "name",
        "start_frame",
        "num_frames",
        "start_sample",
        "num_samples",
        "attempts",
        "check",
        "check_failed",
        "audio_dbfs",
        "trim_frames",
        "crossfade_ms",
        "planned_segments",
    ]
    assert [k for k in without[1] if k != "measures"] == keys
    assert without[1]["check"] == {"face_drop": None}


def test_a_chain_manifest_entry_round_trips_through_the_api_model():
    from dw.server.api_models import ManifestEntry

    shots = _shots({1, 2}, check={"jump_ratio_max": 3.25})
    entry = {"step": "chain", "files": ["take.mp4"], "shots": shots}
    model = ManifestEntry.model_validate(entry)
    dumped = model.model_dump(exclude_unset=True)
    assert dumped == entry
    assert model.shots[1].measures["jump_ratio"].ceiling == 3.25
    # A shot from before #1474, with no measures, still validates
    old = {k: v for k, v in shots[1].items() if k != "measures"}
    assert ManifestEntry.model_validate(
        {"step": "chain", "files": [], "shots": [old]}
    ).model_dump(exclude_unset=True)["shots"] == [old]
