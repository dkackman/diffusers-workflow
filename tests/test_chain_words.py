"""#1262: check_words - whether a dialogue segment's audio speaks its line.

A fake `transcribe` returns synthetic word chunks over a synthetic voiced
waveform; no Whisper model is loaded.
"""

import numpy
import pytest

from dw.pipeline_processors import chain_check
from dw.pipeline_processors.chain_check import (
    SEAM_TAIL_WORDS,
    WORDS_FINDINGS,
    WORDS_SIMILARITY_MIN,
    check_words,
)

RATE = 16000


def voiced(seconds, amplitude=0.1):
    n = int(round(seconds * RATE))
    t = numpy.arange(n, dtype=numpy.float32) / RATE
    noise = numpy.random.default_rng(7).standard_normal(n) * 0.3
    return (amplitude * (numpy.sin(2 * numpy.pi * 220 * t) + noise)).astype(
        numpy.float32
    )


def words(text, start=0.0, step=0.5, length=0.4):
    return [
        {
            "start": start + i * step,
            "end": start + i * step + length,
            "text": f" {word}",
        }
        for i, word in enumerate(text.split())
    ]


def check(line, chunks, previous_line=None, minimum=0.8, seconds=None, audio=None):
    if audio is None:
        end = max((c["end"] for c in chunks), default=1.0)
        audio = voiced(seconds or end + 1.5)
    return check_words(
        audio, RATE, line, previous_line, minimum, transcribe=lambda w, r: chunks
    )


def test_constants():
    assert WORDS_SIMILARITY_MIN == 0.80
    assert SEAM_TAIL_WORDS == 3
    assert set(WORDS_FINDINGS) == {
        "line_clipped_at_end",
        "transcript_loop",
        "tag_spoken",
        "unscripted_speech",
    }


def test_matching_take_passes():
    answer = check("Hello there world", words("hello there world"))
    assert answer["similarity"] == 1.0
    assert answer["findings"] == []
    assert answer["passed"] is True
    assert answer["heard"] == "hello there world"
    assert answer["seam_head_dropped"] == []


def test_low_similarity_fails_unless_minimum_is_none():
    chunks = words("totally different words entirely")
    failed = check("hello there world again", chunks, minimum=0.8)
    assert failed["similarity"] < 0.8
    assert failed["passed"] is False
    held = check("hello there world again", chunks, minimum=None)
    assert held["similarity"] == failed["similarity"]
    assert held["findings"] == []
    assert held["passed"] is True


def test_line_clipped_at_end_fails():
    chunks = [
        {"start": 0.5, "end": 0.9, "text": " hello"},
        {"start": 1.6, "end": 1.95, "text": " world"},
    ]
    answer = check("hello world", chunks, seconds=2.0)
    assert [f["rule"] for f in answer["findings"]] == ["line_clipped_at_end"]
    assert answer["similarity"] == 1.0
    assert answer["passed"] is False


def test_transcript_loop_fails():
    chunks = words("how", 0.0) + words(
        " ".join(["the lid is gone"] * 8), 1.0, step=0.3, length=0.2
    )
    answer = check("the lid is gone", chunks, seconds=12.0)
    assert "transcript_loop" in [f["rule"] for f in answer["findings"]]
    assert answer["passed"] is False


def test_tag_spoken_fails():
    answer = check("<d>[English] Hello there</d>", words("english hello there"))
    # the spoken tag is also words matched to no line
    assert "tag_spoken" in [f["rule"] for f in answer["findings"]]
    assert answer["similarity"] == 1.0
    assert answer["passed"] is False


def test_unscripted_speech_line_spoken_twice_fails():
    chunks = words("alpha bravo charlie", 0.0) + words("alpha bravo charlie", 2.0)
    answer = check("alpha bravo charlie", chunks, minimum=None)
    assert "unscripted_speech" in [f["rule"] for f in answer["findings"]]
    assert answer["passed"] is False


def test_seam_head_word_of_previous_line_is_dropped():
    answer = check(
        "of the United States",
        words("people of the United States"),
        previous_line="we the people",
    )
    assert answer["seam_head_dropped"] == ["people"]
    assert answer["similarity"] == 1.0
    assert answer["heard"] == "of the United States"
    assert answer["passed"] is True


def test_seam_head_number_folded_across_chunks_is_dropped():
    # The previous line ends "twenty-one" (tail {"21"}); Whisper hears the
    # carry as two chunks, which fold to one word as check_script folds them
    answer = check(
        "of the United States",
        words("twenty one of the United States"),
        previous_line="Amendment twenty-one",
    )
    assert answer["seam_head_dropped"] == ["twenty", "one"]
    assert answer["heard"] == "of the United States"
    assert answer["similarity"] == 1.0
    assert answer["passed"] is True


def test_seam_head_chunk_sharing_the_lines_first_word_is_kept():
    chunks = [
        {"start": 0.0, "end": 0.4, "text": " people of"},
        *words("the United States", start=0.5),
    ]
    answer = check("of the United States", chunks, previous_line="we the people")
    assert answer["seam_head_dropped"] == []
    assert answer["heard"].startswith("people of")


def test_invented_head_word_is_kept():
    answer = check(
        "of the United States",
        words("banana of the United States"),
        previous_line="we the people",
    )
    assert answer["seam_head_dropped"] == []
    assert "banana" in answer["heard"]
    # similarity ignores the extra head word; unscripted_speech fails it
    assert [f["rule"] for f in answer["findings"]] == ["unscripted_speech"]
    assert answer["passed"] is False


def test_invented_word_inside_the_line_lowers_similarity():
    answer = check(
        "of the United States",
        words("banana the United States"),
        previous_line="we the people",
        minimum=0.99,
    )
    assert answer["similarity"] < 1.0
    assert answer["passed"] is False


def test_no_previous_line_drops_nothing():
    answer = check("of the United States", words("people of the United States"))
    assert answer["seam_head_dropped"] == []
    assert "people" in answer["heard"]
    assert [f["rule"] for f in answer["findings"]] == ["unscripted_speech"]
    assert answer["passed"] is False


def test_audio_shapes_1d_and_stereo():
    chunks = words("hello there world")
    mono = voiced(3.0)
    stereo = numpy.stack([mono, mono])
    for audio in (mono, stereo):
        answer = check("hello there world", chunks, audio=audio)
        assert answer["similarity"] == 1.0
        assert answer["passed"] is True


def test_default_transcriber_not_loaded_when_one_is_given(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("Whisper must not load")

    monkeypatch.setattr(chain_check, "_whisper_chunks", boom)
    assert check("hello", words("hello"))["passed"] is True


def test_default_transcriber_is_whisper_chunks(monkeypatch):
    seen = {}

    def fake(waveform, rate, device):
        seen["device"] = device
        seen["shape"] = waveform.shape
        return words("hello")

    monkeypatch.setattr(chain_check, "_whisper_chunks", fake)
    answer = check_words(voiced(2.0), RATE, "hello", None, 0.8, device="cuda:1")
    assert answer["passed"] is True
    assert seen["device"] == "cuda:1"
    assert seen["shape"][0] == 1


@pytest.mark.parametrize("previous", [None, ""])
def test_empty_previous_line_is_inert(previous):
    answer = check("hello there", words("hello there"), previous_line=previous)
    assert answer["seam_head_dropped"] == []
