"""retime_narration and narration_lines: pauses set by punctuation class, and
segment boundaries placed in word gaps on a chain's frame grid.

The track is synthetic - 220 Hz bursts as words over a low noise floor - and
the module's `_word_chunks` seam is replaced, so no Whisper model loads.
"""

import math

import numpy
import pytest

from dw.dsp import dbfs, rms
from dw.introspection import list_tasks
from dw.media_types import AudioTrack, JsonRecord
from dw.pipeline_processors.chain_config import plan_segments
from dw.task_domains import task_argument_errors
from dw.tasks import narration_retime
from dw.tasks.narration_retime import narration_lines, retime_narration
from dw.tasks.task import task_command_info

RATE = 16000
FLOOR_DBFS = -66.0
VOICE = 0.1


def build(words, seconds):
    """A track of `seconds` with a burst per (text, start, end) word, and the
    word chunks a word-timestamp transcription of it would answer."""
    n = int(round(seconds * RATE))
    rng = numpy.random.default_rng(11)
    wave = rng.normal(0.0, 10 ** (FLOOR_DBFS / 20.0), n)
    t = numpy.arange(n) / RATE
    for _text, start, end in words:
        a, b = int(round(start * RATE)), int(round(end * RATE))
        wave[a:b] += VOICE * numpy.sin(2 * numpy.pi * 220 * t[a:b])
    chunks = [{"start": s, "end": e, "text": f" {w}"} for w, s, e in words]
    return AudioTrack(wave[None, :].astype(numpy.float32), RATE), chunks


@pytest.fixture
def heard(monkeypatch):
    """Point the transcription seam at a fixed list of chunks."""

    def use(chunks):
        monkeypatch.setattr(
            narration_retime, "_word_chunks", lambda w, r, d, m: list(chunks)
        )

    return use


def gap_after(report, word):
    words = report["words"]
    i = next(k for k, w in enumerate(words) if w["word"] == word)
    return words[i + 1]["start"] - words[i]["end"]


PREAMBLE = [
    ("We", 0.2, 0.5),
    ("the", 0.6, 0.9),
    ("People", 1.0, 1.5),
    ("Union,", 1.6, 2.1),
    ("establish", 3.5, 4.1),
    ("America.", 4.2, 4.8),
    ("Article", 6.8, 7.3),
    ("One", 7.4, 7.8),
]


def test_comma_and_sentence_gaps_take_their_class_lengths(heard):
    track, chunks = build(PREAMBLE, 8.5)
    heard(chunks)

    result = retime_narration(track)

    report = result["report"]
    assert isinstance(report, JsonRecord)
    assert isinstance(result["audio"], AudioTrack)
    assert gap_after(report, "Union,") == pytest.approx(0.3, abs=0.02)
    assert gap_after(report, "America.") == pytest.approx(0.6, abs=0.02)
    # the 0.1 s gaps inside "We the People" carry no punctuation: untouched
    assert gap_after(report, "We") == pytest.approx(0.1, abs=0.005)
    expected = 8.5 - (1.4 - 0.3) - (2.0 - 0.6)
    assert report["duration_seconds"] == pytest.approx(expected, abs=0.01)
    audio = result["audio"]
    assert audio.audio.shape[1] / audio.sample_rate == pytest.approx(expected, abs=0.01)
    classes = {p["after"]: p["class"] for p in report["pauses"]}
    assert classes == {"Union,": "comma", "America.": "sentence"}
    union = next(p for p in report["pauses"] if p["after"] == "Union,")
    assert union["before_seconds"] == pytest.approx(1.4, abs=0.01)
    assert union["after_seconds"] == pytest.approx(0.3, abs=0.02)
    # the word "establish" really sits where the report says it does
    word = next(w for w in report["words"] if w["word"] == "establish")
    mid = int(round((word["start"] + word["end"]) / 2 * RATE))
    voiced = audio.audio[0, mid - 800 : mid + 800]
    assert dbfs(rms(voiced)) > -30.0
    assert report["mid_word_boundaries"] == []
    assert "boundaries" not in report


def test_gaps_shorter_than_min_gap_are_untouched(heard):
    words = [("Wait,", 0.2, 0.6), ("now.", 0.7, 1.1), ("Go", 1.2, 1.5)]
    track, chunks = build(words, 2.0)
    heard(chunks)

    report = retime_narration(track, min_gap_seconds=0.12)["report"]

    assert gap_after(report, "Wait,") == pytest.approx(0.1, abs=0.005)
    assert gap_after(report, "now.") == pytest.approx(0.1, abs=0.005)
    assert report["pauses"] == []
    assert report["duration_seconds"] == pytest.approx(2.0, abs=0.005)


def test_section_break_from_a_blank_line_in_text(heard):
    # Whisper heard no punctuation; the script's blank line makes it a section
    words = [
        ("We", 0.2, 0.5),
        ("the", 0.6, 0.9),
        ("People", 1.0, 1.5),
        ("Article", 3.5, 4.0),
        ("One", 4.1, 4.5),
    ]
    track, chunks = build(words, 5.0)
    heard(chunks)

    report = retime_narration(track, text="We the People.\n\nArticle One")["report"]

    assert gap_after(report, "People.") == pytest.approx(1.0, abs=0.02)
    assert [p["class"] for p in report["pauses"]] == ["section"]
    # the words come out in the script's spelling
    assert [w["word"] for w in report["words"]] == [
        "We",
        "the",
        "People.",
        "Article",
        "One",
    ]


def test_a_line_starting_amendment_opens_a_section(heard):
    words = [("ratified", 0.2, 0.8), ("Amendment", 1.6, 2.2), ("I", 2.3, 2.5)]
    track, chunks = build(words, 3.0)
    heard(chunks)

    report = retime_narration(track, text="ratified\nAmendment I")["report"]

    assert gap_after(report, "ratified") == pytest.approx(1.0, abs=0.02)


def test_inserted_gap_is_room_tone_not_zero(heard):
    words = [("America.", 0.3, 0.9), ("Article", 1.1, 1.6), ("One", 1.7, 2.1)]
    track, chunks = build(words, 2.6)
    heard(chunks)

    result = retime_narration(track)

    report, audio = result["report"], result["audio"]
    assert gap_after(report, "America.") == pytest.approx(0.6, abs=0.02)
    end = next(w["end"] for w in report["words"] if w["word"] == "America.")
    # the original 0.2 s gap is split at its middle, so the middle of the
    # retimed gap is the inserted bed
    a, b = int((end + 0.2) * RATE), int((end + 0.4) * RATE)
    inserted = dbfs(rms(audio.audio[0, a:b]), floor=-200.0)
    article = next(w for w in report["words"] if w["word"] == "Article")
    c, d = int(article["start"] * RATE), int(article["end"] * RATE)
    voiced = dbfs(rms(audio.audio[0, c:d]))
    assert -90.0 < inserted < voiced - 20.0
    assert numpy.count_nonzero(audio.audio[0, a:b]) == b - a


def even_words(count, start=0.2, step=0.5, length=0.35):
    return [
        (f"w{i}", start + i * step, start + i * step + length) for i in range(count)
    ]


def test_boundaries_land_in_gaps_on_the_chain_grid(heard):
    words = even_words(40)  # 0.15 s gaps between every word, ~20 s long
    # 8.0 s, the first kept-span end, lies inside w15 before any retiming
    assert any(s < 8.0 < e for _w, s, e in words)
    track, chunks = build(words, 20.5)
    heard(chunks)

    report = retime_narration(track, segment_frames=192, guide_frames=22, fps=24)[
        "report"
    ]

    boundaries = report["boundaries"]
    total = math.ceil(report["duration_seconds"] * 24 - 1e-9)
    plan = plan_segments(total, 192, 22)
    ends = [(s.audio_start_frame + s.num_frames) / 24 for s in plan[:-1]]
    assert len(boundaries) == len(ends) >= 2
    for boundary, end in zip(boundaries, ends):
        assert boundary["in_gap"] is True
        assert abs(boundary["seconds"] - end) <= 1 / 24
        assert not any(
            w["start"] < boundary["seconds"] < w["end"] for w in report["words"]
        )
    assert [b["segment"] for b in boundaries] == list(range(len(ends)))
    assert report["mid_word_boundaries"] == []


def test_boundary_inside_a_long_word_is_reported_not_forced(heard):
    words = [
        ("The", 0.2, 0.5),
        ("Congress", 6.8, 9.2),  # 8.0 s falls 1.2 s from either edge
        ("shall", 9.4, 9.8),
    ]
    track, chunks = build(words, 10.5)
    heard(chunks)

    report = retime_narration(track, segment_frames=192, guide_frames=22, fps=24)[
        "report"
    ]

    assert report["boundaries"][0]["in_gap"] is False
    assert report["boundaries"][0]["seconds"] == pytest.approx(8.0)
    assert report["mid_word_boundaries"] == [{"segment": 0, "word": "Congress"}]
    congress = next(w for w in report["words"] if w["word"] == "Congress")
    assert congress["start"] == pytest.approx(6.8, abs=0.01)


@pytest.mark.parametrize(
    "words",
    [
        # 4.0 s falls inside "d."; a grown gap lands it on the margin
        [
            ("a", 0.2, 0.5),
            ("b", 1.0, 1.5),
            ("c", 2.5, 3.7),
            ("d.", 3.9, 4.5),
            ("e", 4.7, 5.2),
        ],
        # the same with a short last word and a tighter run of words
        [
            ("a", 0.2, 0.5),
            ("b", 1.0, 1.5),
            ("c", 2.5, 3.6),
            ("d.", 3.9, 4.05),
            ("e", 4.1, 5.2),
        ],
    ],
)
def test_boundary_nudged_onto_the_margin_reports_in_gap(heard, words):
    track, chunks = build(words, 6.0)
    heard(chunks)

    report = retime_narration(track, segment_frames=96, guide_frames=10, fps=24)[
        "report"
    ]

    assert report["boundaries"][0]["in_gap"] is True
    assert report["mid_word_boundaries"] == []


def test_no_words_is_refused(heard):
    track, _ = build([], 2.0)
    heard([])

    with pytest.raises(ValueError, match="retime_narration"):
        retime_narration(track)


LINE_WORDS = [
    ("one", 0.2, 0.6),
    ("two", 1.0, 1.4),
    ("three", 1.8, 2.4),  # midpoint 2.1: segment 1, though it starts in 0
    ("four", 2.6, 3.0),
    ("five", 3.3, 3.8),  # midpoint 3.55: segment 2
    ("six", 4.0, 4.3),
]


def test_lines_split_words_by_kept_span_midpoint(heard):
    track, chunks = build(LINE_WORDS, 4.5)
    heard(chunks)

    answer = narration_lines(track, segment_frames=48, guide_frames=12, fps=24)

    plan = plan_segments(answer["total_frames"], 48, 12)
    assert answer["total_frames"] == 108
    assert answer["lines"] == ["one two", "three four", "five six"]
    assert len(answer["lines"]) == len(plan)
    assert [s["start_seconds"] for s in answer["segments"]] == [0.0, 2.0, 3.5]
    assert [s["end_seconds"] for s in answer["segments"]] == [2.0, 3.5, 4.5]
    assert answer["mid_word_boundaries"] == [
        {"segment": 0, "word": "three"},
        {"segment": 1, "word": "five"},
    ]
    assert "prompts" not in answer


def test_lines_take_the_script_spelling(heard):
    track, chunks = build(LINE_WORDS, 4.5)
    heard(chunks)

    answer = narration_lines(
        track,
        segment_frames=48,
        guide_frames=12,
        fps=24,
        text="One, two. Three four - five, six!",
    )

    assert answer["lines"] == ["One, two.", "Three four", "five, six!"]


def test_prompts_fill_the_template(heard):
    track, chunks = build(LINE_WORDS, 4.5)
    heard(chunks)

    answer = narration_lines(
        track,
        segment_frames=48,
        guide_frames=12,
        prompt_template="A {scene}. <d>{line}</d>",
        fill={"scene": "park"},
    )

    lines = answer["lines"]
    assert answer["prompts"] == [f"A park. <d>{line}</d>" for line in lines]


def test_fill_placeholder_missing_is_refused(heard):
    track, chunks = build(LINE_WORDS, 4.5)
    heard(chunks)

    with pytest.raises(ValueError, match="speaker"):
        narration_lines(
            track,
            segment_frames=48,
            guide_frames=12,
            prompt_template="{speaker} says <d>{line}</d>",
            fill={"scene": "park"},
        )


def test_an_attribute_placeholder_is_refused(heard):
    track, chunks = build(LINE_WORDS, 4.5)
    heard(chunks)

    with pytest.raises(ValueError, match="line.upper"):
        narration_lines(
            track,
            segment_frames=48,
            guide_frames=12,
            prompt_template="<d>{line.upper}</d>",
        )


def _lines_with(heard, template, **kwargs):
    track, chunks = build(LINE_WORDS, 4.5)
    heard(chunks)
    return narration_lines(
        track, segment_frames=48, guide_frames=12, prompt_template=template, **kwargs
    )


@pytest.mark.parametrize(
    "template",
    ["<d>{line:>999999999}</d>", "<d>{line!r}</d>", "{0}", "{line[0]}", "{ line }"],
)
def test_format_spec_placeholder_is_refused(heard, template):
    with pytest.raises(ValueError, match="narration_lines.*placeholder"):
        _lines_with(heard, template)


def test_escaped_braces_are_literal(heard):
    answer = _lines_with(heard, "{{style}} <d>{line}</d> }}")

    assert answer["prompts"] == [
        f"{{style}} <d>{line}</d> }}" for line in answer["lines"]
    ]


def test_unbalanced_brace_is_refused(heard):
    with pytest.raises(ValueError, match="narration_lines"):
        _lines_with(heard, "<d>{line</d>")


def test_filled_prompt_over_the_cap_is_refused(heard):
    with pytest.raises(ValueError, match=r"narration_lines.*segment 0"):
        _lines_with(heard, "{big} {big}", fill={"big": "x" * 15_000})


def test_template_longer_than_the_cap_is_refused(heard):
    with pytest.raises(ValueError, match=r"narration_lines.*prompt_template"):
        _lines_with(heard, "x" * 20_001)


def test_fill_value_longer_than_the_cap_is_refused(heard):
    with pytest.raises(ValueError, match=r"narration_lines.*'big'"):
        _lines_with(heard, "{big}", fill={"big": "x" * 20_001})


def test_many_placeholders_are_refused_before_building(heard, monkeypatch):
    built = []
    real = narration_retime._PLACEHOLDER

    class Spy:
        def finditer(self, text):
            return real.finditer(text)

        def sub(self, repl, text):
            built.append(text)
            return real.sub(repl, text)

    monkeypatch.setattr(narration_retime, "_PLACEHOLDER", Spy())
    with pytest.raises(ValueError, match=r"narration_lines.*segment 0"):
        _lines_with(heard, "{scene}" * 2_500, fill={"scene": "x" * 1_000})
    assert built == []


def test_narration_lines_with_no_words_is_refused(heard):
    track, _ = build([], 2.0)
    heard([])

    with pytest.raises(ValueError, match="narration_lines"):
        narration_lines(track, segment_frames=48)


def _definition(command, arguments):
    return {
        "steps": [
            {
                "name": "step",
                "task": {
                    "command": command,
                    "arguments": {"audio": "asset:narration.wav", **arguments},
                },
            }
        ]
    }


def test_registry_rules():
    commands = list_tasks()["commands"]
    assert "retime_narration" in commands
    assert "narration_lines" in commands
    assert task_command_info("narration_lines")["returns"] == "json"
    assert task_command_info("retime_narration")["returns"] == "artifact"
    for command, extra in (
        ("retime_narration", {}),
        ("narration_lines", {"segment_frames": 192}),
    ):
        errors = task_argument_errors(_definition(command, {"fps": 0, **extra}))
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.fps"]
    errors = task_argument_errors(
        _definition("retime_narration", {"comma_seconds": -0.1, "guide_frames": 2.5})
    )
    assert sorted(e["path"] for e in errors) == [
        "steps[0].task.arguments.comma_seconds",
        "steps[0].task.arguments.guide_frames",
    ]


def test_room_tone_falls_back_to_the_quietest_window(heard, monkeypatch):
    from dw.tasks import loop_bed

    monkeypatch.setattr(loop_bed, "find_loop_bed", lambda *a, **k: {"candidates": []})
    words = [("America.", 0.3, 0.9), ("Article", 1.1, 1.6), ("One", 1.7, 2.1)]
    track, chunks = build(words, 2.6)
    heard(chunks)

    result = retime_narration(track)

    report, audio = result["report"], result["audio"]
    end = next(w["end"] for w in report["words"] if w["word"] == "America.")
    a, b = int((end + 0.2) * RATE), int((end + 0.4) * RATE)
    inserted = dbfs(rms(audio.audio[0, a:b]), floor=-200.0)
    assert inserted == pytest.approx(FLOOR_DBFS, abs=3.0)


def test_boundary_at_a_word_junction_needs_no_move(heard):
    # 8.0 s, the first boundary, falls in the 0.03 s junction of "the" and
    # "People": the take holds its narration, so no word is cut there
    words = [
        ("We", 6.0, 7.0),
        ("the", 7.2, 7.98),
        ("People", 8.01, 9.0),
        ("shall", 9.05, 9.6),
    ]
    track, chunks = build(words, 10.0)
    heard(chunks)

    result = retime_narration(track, segment_frames=192, guide_frames=22, fps=24)

    report = result["report"]
    assert report["boundaries"][0]["seconds"] == pytest.approx(8.0)
    assert report["boundaries"][0]["in_gap"] is True
    assert report["mid_word_boundaries"] == []
    assert report["pauses"] == []
    numpy.testing.assert_array_equal(result["audio"].audio, track.audio)


def test_boundary_inside_a_word_moves_to_the_nearest_junction(heard):
    # 8.0 s is 0.1 s into "People"; the comma gap two words earlier is the
    # only one that may move, and lengthening it brings the junction of
    # "the" and "People" onto the boundary
    words = [
        ("Union,", 6.0, 7.0),
        ("and", 7.3, 7.6),
        ("the", 7.6, 7.9),
        ("People", 7.9, 8.6),
        ("of", 8.65, 9.0),
    ]
    track, chunks = build(words, 9.6)
    heard(chunks)

    report = retime_narration(track, segment_frames=192, guide_frames=22, fps=24)[
        "report"
    ]

    boundary = report["boundaries"][0]
    assert boundary["in_gap"] is True
    assert report["mid_word_boundaries"] == []
    assert not any(w["start"] < boundary["seconds"] < w["end"] for w in report["words"])
    shift = gap_after(report, "Union,") - 0.3
    assert 0.0 < shift <= 0.25 + 1e-6
    assert report["duration_seconds"] == pytest.approx(9.6 + shift, abs=0.01)


def test_mid_word_only_when_no_junction_is_reachable(heard):
    # 8.0 s is 2.5 s into a long word; the one eligible gap would have to
    # grow by 2.5 s, or shrink below nothing, to bring a junction there
    words = [
        ("a", 5.0, 5.3),
        ("Constitutionality", 5.5, 8.6),
        ("b", 8.6, 9.0),
    ]
    track, chunks = build(words, 9.5)
    heard(chunks)

    report = retime_narration(track, segment_frames=192, guide_frames=22, fps=24)[
        "report"
    ]

    assert report["boundaries"][0]["in_gap"] is False
    assert report["mid_word_boundaries"] == [
        {"segment": 0, "word": "Constitutionality"}
    ]
    assert report["duration_seconds"] == pytest.approx(9.5, abs=0.005)


def test_lines_use_script_spelling(heard):
    # Whisper hears "$20," for the script's "twenty dollars,": the line
    # carries the script's words, "dollars," included
    words = [("exceed", 0.2, 0.6), ("$20,", 0.7, 1.2), ("the", 1.8, 2.0)]
    track, chunks = build(words, 2.2)
    heard(chunks)

    answer = narration_lines(
        track,
        segment_frames=96,
        guide_frames=12,
        fps=24,
        text="exceed twenty dollars, the",
    )

    assert answer["lines"] == ["exceed twenty dollars, the"]
    report = retime_narration(track, text="exceed twenty dollars, the")["report"]
    assert [w["word"] for w in report["words"]] == [
        "exceed",
        "twenty dollars,",
        "the",
    ]
    # the comma the script puts after the word Whisper missed sets the pause
    assert gap_after(report, "twenty dollars,") == pytest.approx(0.3, abs=0.02)


def test_final_take_end_is_not_a_boundary(heard):
    # the last word ends at 6.0 s; the chain's final boundary (8.0 s) sits in
    # the silence after it, which is not a seam through a word
    words = [("a", 0.2, 0.6), ("b", 5.5, 6.0)]
    track, chunks = build(words, 9.0)
    heard(chunks)

    report = retime_narration(track, segment_frames=192, guide_frames=22, fps=24)[
        "report"
    ]

    assert report["mid_word_boundaries"] == []
    assert all(b["in_gap"] for b in report["boundaries"])


def test_lines_mid_word_tolerates_whisper_jitter(heard):
    # segment 0 ends at 8.0 s (192 frames at 24 fps)
    def mid_word(start, end):
        track, chunks = build([("a", 0.2, 0.6), ("word", start, end)], 12.0)
        heard(chunks)
        return narration_lines(track, segment_frames=192, guide_frames=22, fps=24)[
            "mid_word_boundaries"
        ]

    assert mid_word(7.95, 9.0) == []  # 0.05 s inside the word's start
    assert mid_word(7.8, 9.0) == [{"segment": 0, "word": "word"}]  # 0.2 s inside
    assert mid_word(6.5, 8.05) == []  # 0.05 s inside the word's end
