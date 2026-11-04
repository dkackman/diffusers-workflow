"""split_takes: a long text into takes a speech model can say in one go."""

import re

import pytest

from dw.tasks.text_takes import split_takes

SENTENCES = [
    "Somebody finished the pistachio and nobody is saying anything.",
    "The fridge hums.",
    "Nobody looks up from their phone, and nobody admits to the crime.",
    "It was a quiet kind of guilt.",
    "Eventually someone laughs.",
]
TEXT = " ".join(SENTENCES * 4)


def words(text):
    return len(text.split())


def test_no_take_is_over_the_limit():
    takes = split_takes(TEXT, max_seconds=10, words_per_second=2.5)
    assert len(takes) > 1
    assert all(words(t) <= 25 for t in takes)


def test_takes_end_on_sentence_boundaries():
    for take in split_takes(TEXT, max_seconds=10, words_per_second=2.5):
        assert re.search(r"[.!?]$", take)


def test_takes_rejoin_to_the_text():
    takes = split_takes(TEXT, max_seconds=10, words_per_second=2.5)
    assert " ".join(takes).split() == TEXT.split()


def test_short_text_is_one_take():
    assert split_takes("Hello there.", 60, 2.5) == ["Hello there."]


def test_a_long_sentence_splits_at_its_last_comma_before_the_limit():
    sentence = "one two three, four five six, seven eight nine ten eleven twelve."
    takes = split_takes(sentence, max_seconds=4, words_per_second=2.5)  # 10 words
    assert takes[0] == "one two three, four five six,"
    assert " ".join(takes).split() == sentence.split()
    assert all(words(t) <= 10 for t in takes)


def test_a_long_sentence_without_commas_splits_at_a_word():
    sentence = " ".join(f"w{i}" for i in range(25)) + "."
    takes = split_takes(sentence, max_seconds=4, words_per_second=2.5)
    assert all(words(t) <= 10 for t in takes)
    assert " ".join(takes).split() == sentence.split()


def test_paragraph_breaks_are_whitespace():
    takes = split_takes("First one.\n\nSecond one.", 60, 2.5)
    assert " ".join(takes).split() == ["First", "one.", "Second", "one."]


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_bad_limits_are_refused(bad):
    with pytest.raises(ValueError):
        split_takes("Hi.", max_seconds=bad, words_per_second=2.5)
    with pytest.raises(ValueError):
        split_takes("Hi.", max_seconds=10, words_per_second=bad)


def test_empty_text_is_refused():
    with pytest.raises(ValueError):
        split_takes("   ", 60, 2.5)


def test_the_registered_task_answers_the_takes():
    import dw.tasks.task  # noqa: F401
    from dw.tasks.registry import _COMMAND_INFO

    assert _COMMAND_INFO["split_takes"]["returns"] == "json"
    from dw.tasks.text_takes import split_takes_task

    out = split_takes_task(TEXT, max_seconds=10)
    assert out["count"] == len(out["takes"]) > 1
