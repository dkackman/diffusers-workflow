"""#1297: dialogue_text, number-word normalising, and guard_words' folding."""

import numpy

from dw.script_lines import dialogue_text, fold_numbers, normalize_words
from dw.tasks.script_check import guard_words

RATE = 16000


def test_dialogue_text_joins_bodies_in_order():
    prompt = "Scene. <d>Hello there</d> prose <d>General Kenobi</d> end"
    assert dialogue_text(prompt) == "Hello there General Kenobi"


def test_dialogue_text_none_and_non_string():
    assert dialogue_text("just scene prose") == ""
    assert dialogue_text(None) == ""
    assert dialogue_text(42) == ""


def test_dialogue_text_keeps_markup():
    assert (
        dialogue_text("<d>[English] Hello <cutoff> there</d>")
        == "[English] Hello <cutoff> there"
    )
    assert dialogue_text("<d>(S1) Hi</d> <d>(S2) Yo</d>") == "(S1) Hi (S2) Yo"


def test_dialogue_text_case_insensitive_and_whitespace():
    assert dialogue_text("<D>  a\n\n  b \t c </D>") == "a b c"


def test_dialogue_text_excludes_outside_prose():
    assert dialogue_text("narration <d>spoken</d> more narration") == "spoken"


def test_dialogue_text_unclosed_runs_to_end():
    assert dialogue_text("intro <d>first</d> x <d>never closed here") == (
        "first never closed here"
    )


def test_number_words_equal_digits():
    assert normalize_words("Article One") == normalize_words("Article 1")
    assert normalize_words("twenty-one") == ["21"]
    assert normalize_words("twenty-one") == normalize_words("21")
    assert normalize_words("one thousand seven hundred and eighty-seven") == ["1787"]
    assert normalize_words("1,787") == ["1787"]
    assert normalize_words("twenty-one, twenty-two") == ["21", "22"]


def test_and_outside_a_number_is_untouched():
    assert normalize_words("bread and butter") == ["bread", "and", "butter"]


def test_numbers_false_leaves_words():
    assert normalize_words("twenty-one", numbers=False) == ["twenty", "one"]
    assert normalize_words("Article One", numbers=False) == ["article", "one"]


def test_fold_numbers_counts():
    assert fold_numbers(["article", "twenty", "one", "now"]) == [
        ("article", 1),
        ("21", 2),
        ("now", 1),
    ]
    assert fold_numbers(["bread", "and"]) == [("bread", 1), ("and", 1)]


def _voiced(seconds):
    n = int(seconds * RATE)
    return (0.1 * numpy.random.default_rng(1).standard_normal(n)).astype(numpy.float32)


def test_guard_words_folds_number_words_across_chunks():
    chunks = [
        {"start": 0.1, "end": 0.4, "text": " twenty"},
        {"start": 0.5, "end": 0.9, "text": " one"},
    ]
    heard, discarded = guard_words(chunks, _voiced(1.5), RATE)
    assert discarded == []
    assert len(heard) == 1
    assert heard[0]["word"] == "21"
    assert heard[0]["start"] == 0.1
    assert heard[0]["end"] == 0.9


def test_guard_words_lone_number_word_becomes_digits():
    heard, _ = guard_words(
        [{"start": 0.1, "end": 0.4, "text": " One"}], _voiced(1.0), RATE
    )
    assert [h["word"] for h in heard] == ["1"]


def test_guard_words_without_number_words_is_unfolded():
    """Stage A's fold is the identity on a take with no number word (the
    ep198 freezer/air take of C-F503): guard_words gives what it gave
    before the fold existed."""
    chunks = [
        {"start": 0.1, "end": 0.3, "text": " Who"},
        {"start": 0.3, "end": 0.5, "text": " left"},
        {"start": 0.5, "end": 0.8, "text": " the freezer"},
        {"start": 0.9, "end": 1.2, "text": " Uh,"},
    ]
    heard, _ = guard_words(chunks, _voiced(1.5), RATE)
    assert [(h["word"], h["text"], h["start"], h["end"]) for h in heard] == [
        ("who", "Who", 0.1, 0.3),
        ("left", "left", 0.3, 0.5),
        ("the", "the freezer", 0.5, 0.8),
        ("freezer", "the freezer", 0.5, 0.8),
        ("uh", "Uh,", 0.9, 1.2),
    ]


def test_digits_written_apart_are_not_one_number():
    """'2 1' (said 'two one') is not '21'; a year in halves still joins."""
    from dw.tasks.script_findings import line_similarity

    assert line_similarity(["article", "2", "1"], ["article", "21"]) < 1.0
    assert line_similarity(["article", "21"], ["article", "2", "1"]) < 1.0
    assert line_similarity(normalize_words("seventeen eighty-seven"), ["1787"]) == 1.0
    assert line_similarity(["candle", "crumbs"], ["candlecrumbs"]) == 1.0
