"""plan_cuts on analyze_phrases' result: timing with no text (#1095)."""

import numpy
import pytest

from dw.task_problems import cuts_errors
from dw.tasks.cuts import plan_cuts
from dw.tasks.phrases import analyze_phrases

FPS = 24
DURATION = 30.0
SPANS = [(4.0, 7.0), (7.5, 10.0), (18.0, 21.0), (21.5, 24.0)]
PHRASES = {
    "phrases": [{"start": a, "end": b} for a, b in SPANS],
    "duration_seconds": DURATION,
    "method": "centre",
    "warnings": [],
}
LYRICS = """[Verse]
Rise up the morning
Over the river

[Chorus]
Sing it out loud
Never let go
"""
LINES = ["Rise up the morning", "Over the river", "Sing it out loud", "Never let go"]


def beat_list(step=0.5, duration=DURATION):
    return [round(i * step, 4) for i in range(1, int(duration / step))]


def plan(**kwargs):
    kwargs.setdefault("transcript", PHRASES)
    return plan_cuts(**kwargs)


def vocal(result):
    return [s for s in result["shots"] if s["kind"] == "vocal"]


def assert_tiles(result):
    shots = result["shots"]
    assert shots[0]["start_frame"] == 0
    for before, after in zip(shots, shots[1:]):
        assert after["start_frame"] == before["start_frame"] + before["cut_frames"]
    assert sum(s["cut_frames"] for s in shots) == result["total_frames"]


class TestWithoutLyrics:
    def test_a_line_per_phrase_with_no_lyric(self):
        result = plan()
        shots = vocal(result)
        assert len(shots) == len(SPANS)
        assert all(s["lyric"] is None for s in shots)
        assert [s["start_frame"] for s in shots] == [round(a * FPS) for a, _ in SPANS]
        assert_tiles(result)

    def test_the_length_comes_from_the_phrase_result(self):
        result = plan()
        assert result["duration_s"] == DURATION
        assert not any(
            "can't tell" in w or "last line" in w for w in result["warnings"]
        )

    def test_no_transcript_warnings(self):
        warnings = plan()["warnings"]
        assert not any("suspect" in w or "heard" in w for w in warnings)

    def test_stanza_groups_phrases_by_silence(self):
        shots = vocal(plan(segment_by="stanza"))
        # The 8 s silence after 10.0 s is the only gap of min_gap_seconds (2.0)
        assert [s["start_frame"] for s in shots] == [
            round(4.0 * FPS),
            round(18.0 * FPS),
        ]
        assert all(s["lyric"] is None for s in shots)

    def test_beat_cuts_mark_the_sung_scenes(self):
        result = plan(segment_by="beat", beats=beat_list(2.0), min_scene_s=2.0)
        assert_tiles(result)
        assert vocal(result) and all(s["lyric"] is None for s in result["shots"])
        assert result["shots"][0]["kind"] == "instrumental"  # 0-2 s, nothing sung

    def test_no_phrases_is_one_instrumental_shot(self):
        result = plan(transcript=dict(PHRASES, phrases=[]))
        assert [s["kind"] for s in result["shots"]] == ["instrumental"]
        assert any("no sung phrases" in w for w in result["warnings"])


class TestMatchingLyrics:
    def test_lines_map_onto_phrases_in_order(self):
        result = plan(lyrics=LYRICS)
        shots = vocal(result)
        assert [s["lyric"] for s in shots] == LINES
        assert [s["start_frame"] for s in shots] == [round(a * FPS) for a, _ in SPANS]
        assert result["warnings"] == []

    def test_stanza_takes_the_lyrics_stanzas(self):
        shots = vocal(plan(lyrics=LYRICS, segment_by="stanza"))
        assert [s["lyric"] for s in shots] == [
            "\n".join(LINES[:2]),
            "\n".join(LINES[2:]),
        ]

    def test_beat_cuts_carry_the_lines(self):
        result = plan(
            lyrics=LYRICS, segment_by="beat", beats=beat_list(2.0), min_scene_s=2.0
        )
        assert_tiles(result)
        sung = [s["lyric"] for s in result["shots"] if s["lyric"]]
        assert "\n".join(sung).split("\n") == LINES

    def test_a_split_phrase_keeps_its_whole_lyric(self):
        long = dict(PHRASES, phrases=[{"start": 2.0, "end": 14.0}])
        result = plan(transcript=long, lyrics="One long held line", max_scene_s=5.0)
        sung = vocal(result)
        assert len(sung) > 1
        assert all(s["lyric"] == "One long held line" for s in sung)


class TestMismatchedLyrics:
    def mismatch(self, result):
        return [w for w in result["warnings"] if "spread evenly" in w]

    def test_more_phrases_than_lines_says_raise(self):
        result = plan(lyrics="\n".join(LINES[:3]))
        (warning,) = self.mismatch(result)
        assert "3 line(s)" in warning and "4 phrase(s)" in warning
        assert "raise analyze_phrases' 'min_gap_seconds'" in warning

    def test_fewer_phrases_than_lines_says_lower(self):
        result = plan(lyrics="\n".join(LINES + ["One more line", "And another"]))
        (warning,) = self.mismatch(result)
        assert "6 line(s)" in warning and "4 phrase(s)" in warning
        assert "lower analyze_phrases' 'min_gap_seconds'" in warning
        # Fused legato lines: the floor is the knob that moves the count
        assert "raise its 'floor_db'" in warning
        assert "by ear" in warning

    def test_more_phrases_than_lines_does_not_name_floor_db(self):
        (warning,) = self.mismatch(plan(lyrics="\n".join(LINES[:3])))
        assert "floor_db" not in warning

    def test_the_field_verse_shape_warns_with_both_knobs(self):
        # verse_s7.wav at the defaults: one fused phrase, four lines
        fused = dict(
            PHRASES, phrases=[{"start": 1.44, "end": 33.0}], duration_seconds=34.0
        )
        result = plan(transcript=fused, lyrics="\n".join(LINES))
        (warning,) = self.mismatch(result)
        assert "4 line(s)" in warning and "1 phrase(s)" in warning
        assert "1.44-33.00 s" in warning
        assert "'min_gap_seconds'" in warning and "'floor_db'" in warning
        assert [s["lyric"] for s in vocal(result) if s["lyric"]] == LINES

    def test_the_lines_spread_evenly_from_first_phrase_to_last(self):
        result = plan(lyrics="\n".join(LINES[:2]), include_instrumental_gaps=False)
        shots = vocal(result)
        assert [s["lyric"] for s in shots] == LINES[:2]
        # 4.0-24.0 s halved: the second line starts at 14.0 s, not at a phrase
        assert shots[1]["start_frame"] == round(14.0 * FPS)
        assert_tiles(result)


class TestUnchanged:
    def test_the_render_grid_holds(self):
        result = plan(
            lyrics=LYRICS, modulus=8, remainder=1, min_frames=25, max_frames=121
        )
        assert_tiles(result)
        for shot in result["shots"]:
            assert shot["num_frames"] % 8 == 1
            assert 25 <= shot["num_frames"] <= 121

    def test_cuts_snap_to_beats(self):
        beats = beat_list(0.7)
        result = plan(lyrics=LYRICS, beats=beats, snap_to_beats=True)
        beat_frames = {round(b * FPS) for b in beats}
        for shot in result["shots"][1:]:
            assert shot["start_frame"] in beat_frames

    def test_a_static_check_accepts_the_phrase_result(self):
        assert cuts_errors({"transcript": PHRASES}) == []

    def test_a_phrases_key_that_is_no_list_is_refused(self):
        errors = cuts_errors({"transcript": {"phrases": "4.0-7.0"}})
        assert errors and errors[0][0] == "transcript"
        with pytest.raises(ValueError, match="'phrases' key that is not a list"):
            plan(transcript={"phrases": "4.0-7.0"}, duration_s=DURATION)


def test_analyze_phrases_result_plans_with_lyrics():
    # The real chain: analyze_phrases' own result straight into plan_cuts
    rate, seconds = 22050, 8.0
    t = numpy.arange(int(seconds * rate)) / rate
    wave = numpy.zeros_like(t)
    bursts = [(1.0, 2.5), (3.5, 5.0), (6.0, 7.5)]
    for start, end in bursts:
        on = (t >= start) & (t < end)
        wave[on] = 0.3 * numpy.sin(2 * numpy.pi * 440 * t[on])
    phrases = analyze_phrases(wave.astype(numpy.float32), sample_rate=rate)
    assert len(phrases["phrases"]) == 3
    result = plan_cuts(
        phrases, lyrics="one\ntwo\nthree", min_scene_s=0.5, min_gap_seconds=0.5
    )
    shots = vocal(result)
    assert [s["lyric"] for s in shots] == ["one", "two", "three"]
    for shot, (start, _) in zip(shots, bursts):
        assert shot["start_frame"] == pytest.approx(start * FPS, abs=0.05 * FPS + 1)
    assert not any("spread evenly" in w for w in result["warnings"])


class TestDisproportion:
    """Matched counts on phrases far out of proportion (#1095 v3)."""

    def disproportion(self, result):
        return [w for w in result["warnings"] if "median" in w]

    def test_proportionate_phrases_give_no_warning(self):
        assert self.disproportion(plan(lyrics=LYRICS)) == []

    def test_the_field_verse_at_floor_db_minus_30_names_the_short_line(self):
        spans = [(2.66, 5.12), (7.26, 15.58), (16.02, 16.36), (18.40, 33.00)]
        odd = dict(
            PHRASES,
            phrases=[{"start": a, "end": b} for a, b in spans],
            duration_seconds=34.0,
        )
        result = plan(transcript=odd, lyrics=LYRICS)
        (warning,) = self.disproportion(result)
        assert "line 3" in warning and "Sing it out loud" in warning
        assert "0.34 s" in warning and "by ear" in warning
        for other in ("line 1", "line 2", "line 4"):
            assert other not in warning
        # The pairing still stands: advice only
        assert [s["lyric"] for s in vocal(result)] == LINES
        assert [s["start_frame"] for s in vocal(result)][1:] == [
            round(a * FPS) for a, _ in spans[1:]
        ]

    def test_an_over_long_phrase_is_named(self):
        spans = [(1.0, 3.0), (3.5, 5.5), (6.0, 8.0), (9.0, 27.0)]
        odd = dict(PHRASES, phrases=[{"start": a, "end": b} for a, b in spans])
        (warning,) = self.disproportion(plan(transcript=odd, lyrics=LYRICS))
        assert "line 4" in warning and "Never let go" in warning
        assert "line 1" not in warning
