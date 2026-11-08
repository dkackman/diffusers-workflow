"""check_narration: the script against the speech, per take."""

import numpy
import pytest

from dw.tasks.narration_check import align, check_narration, normalize

TAKES = ["the cat sat.", "on the mat."]


def fake(transcript):
    return lambda waveform, rate, device, model_name: transcript


SILENCE = numpy.zeros((1, 160), dtype=numpy.float32)


def run(transcript, takes=TAKES, **kwargs):
    return check_narration(
        takes, SILENCE, sample_rate=16000, transcribe=fake(transcript), **kwargs
    )


def test_normalize_lowercases_and_drops_punctuation():
    assert normalize("The Cat, sat!") == ["the", "cat", "sat"]


def test_align_pairs_each_token():
    pairs = align(["a", "b"], ["a", "x", "b"])
    assert ("equal", 0, 0) in pairs
    assert ("inserted", None, 1) in pairs
    assert ("equal", 1, 2) in pairs


def test_a_word_perfect_narration_passes():
    result = run("The cat sat on the mat")
    assert all(take["ok"] for take in result["takes"])
    assert result["failed"] == []
    assert result["checked"] == 2
    assert result["worst"]["word_error_rate"] == 0.0


def test_a_dropped_word_is_attributed_to_its_take():
    result = run("the cat on the mat")
    first, second = result["takes"]
    assert first["dropped"] == ["sat"]
    assert first["word_error_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert first["ok"] is False
    assert second["ok"] is True
    assert result["failed"] == [0]
    assert result["worst"]["index"] == 0


def test_a_looped_take_shows_as_inserted_and_later_takes_still_align():
    result = run("the cat sat the cat sat on the mat")
    first, second = result["takes"]
    assert first["inserted"] == ["the", "cat", "sat"]
    assert second["ok"] is True
    assert second["dropped"] == [] and second["inserted"] == []


def test_the_lengths_need_not_match_but_a_string_is_split():
    text = "One two three. Four five six."
    result = check_narration(
        text, SILENCE, sample_rate=16000, transcribe=fake("one two three four five six")
    )
    assert result["checked"] == 1
    assert result["takes"][0]["ok"] is True


def test_each_failing_take_warns():
    from dw.events import RunContext, activate_context, deactivate_context

    events = []
    token = activate_context(RunContext(on_event=events.append))
    try:
        run("the cat on the mat")
    finally:
        deactivate_context(token)
    warnings = [e for e in events if e.get("kind") == "narration_take_mismatch"]
    assert len(warnings) == 1
    message = warnings[0]["message"]
    assert "take 0" in message and "0.33" in message and "sat" in message


def test_text_and_takes_are_alternatives_exactly_one():
    from dw.tasks.narration_check import check_narration as check

    with pytest.raises(ValueError, match="text.*takes"):
        check(None, SILENCE, sample_rate=16000, transcribe=fake(""))
    with pytest.raises(ValueError, match="text.*takes"):
        check(TAKES, SILENCE, sample_rate=16000, text="hello", transcribe=fake(""))
    ok = check(
        None,
        SILENCE,
        sample_rate=16000,
        text="hello there.",
        transcribe=fake("hello there"),
    )
    assert ok["failed"] == []


def test_the_threshold_is_a_unit_interval_domain():
    from dw.task_domains import TASK_ARGUMENT_DOMAINS, UNIT

    assert TASK_ARGUMENT_DOMAINS["check_narration"]["max_word_error_rate"] == UNIT


def test_a_misheard_last_word_is_blamed_on_its_own_take():
    result = run("the cat sad on the mat")
    first, second = result["takes"]
    assert first["dropped"] == ["sat"] and first["inserted"] == ["sad"]
    assert second["ok"] is True
    assert second["inserted"] == [] and second["dropped"] == []
    assert result["failed"] == [0]


def test_a_pure_insert_at_a_boundary_belongs_to_the_next_take():
    result = run("the cat sat um on the mat")
    first, second = result["takes"]
    assert first["inserted"] == []
    assert second["inserted"] == ["um"]
