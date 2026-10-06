"""plan_cuts: a cut list from a transcript, lyrics and beats (#600)."""

import json

import pytest

from dw.task_domains import cuts_errors, task_argument_errors
from dw.tasks.cuts import plan_cuts

FPS = 24
DURATION = 30.0

CHUNKS = [
    {"start": 4.0, "end": 7.0, "text": " Walking in the moonlight tonight"},
    {"start": 7.5, "end": 10.0, "text": " Dancing with the stars above"},
    {"start": 18.0, "end": 21.0, "text": " Singing the chorus loud"},
    {"start": 21.0, "end": 24.0, "text": " Never gonna stop now"},
]
TRANSCRIPT = {"text": " ".join(c["text"].strip() for c in CHUNKS), "chunks": CHUNKS}

LYRICS = """[Verse]
Walking in the moon light tonight
Dancing with the Stars above

[Chorus]
Singing the chorus LOUD
Never gonna stop now
"""
LINES = [
    "Walking in the moon light tonight",
    "Dancing with the Stars above",
    "Singing the chorus LOUD",
    "Never gonna stop now",
]

SHOT_KEYS = {
    "name",
    "start_frame",
    "num_frames",
    "cut_frames",
    "lead_frames",
    "lyric",
    "kind",
}


def beat_list(step=0.5, duration=DURATION):
    return [round(i * step, 4) for i in range(1, int(duration / step))]


def plan(**kwargs):
    kwargs.setdefault("transcript", TRANSCRIPT)
    kwargs.setdefault("duration_s", DURATION)
    return plan_cuts(**kwargs)


def sung(result):
    return "\n".join(s["lyric"] for s in result["shots"] if s["lyric"]).split("\n")


@pytest.fixture(autouse=True)
def captured_warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "dw.events.emit_warning", lambda message, **data: seen.append(message)
    )
    return seen


class TestLyrics:
    def test_the_lyrics_are_the_lines_verbatim_and_in_order(self):
        result = plan(lyrics=LYRICS)
        assert sung(result) == LINES
        # One shot per line, none carrying a tag or a blank
        lyrics = [s["lyric"] for s in result["shots"] if s["lyric"]]
        assert lyrics == LINES

    def test_a_list_of_lines_is_accepted(self):
        result = plan(lyrics=["[Verse]", *LINES[:2], "", *LINES[2:]])
        assert sung(result) == LINES

    def test_a_line_not_in_the_transcript_is_kept_and_warned_about(self):
        lyrics = LYRICS.replace(
            "Never gonna stop now", "Zzyzx quux flibber\nNever gonna stop now"
        )
        result = plan(lyrics=lyrics)
        assert sung(result) == [*LINES[:3], "Zzyzx quux flibber", LINES[3]]
        assert any("Zzyzx quux flibber" in w for w in result["warnings"])

    def test_an_unheard_line_between_spaced_lines_is_kept_and_warned_about(self):
        lyrics = LYRICS.replace(
            "Singing the chorus LOUD", "Zzyzx quux flibber\nSinging the chorus LOUD"
        )
        result = plan(lyrics=lyrics)
        assert sung(result) == [*LINES[:2], "Zzyzx quux flibber", *LINES[2:]]
        assert any("Zzyzx quux flibber" in w for w in result["warnings"])

    def test_without_lyrics_the_transcript_lines_are_used(self):
        result = plan()
        assert sung(result) == [c["text"].strip() for c in CHUNKS]

    def test_blank_lyrics_fall_back_to_the_transcript(self):
        assert sung(plan(lyrics="  \n")) == [c["text"].strip() for c in CHUNKS]


class TestNullEnds:
    def chunks(self, last_end=None):
        return [
            {"start": 1.0, "end": None, "text": "first line"},
            {"start": 3.0, "end": last_end, "text": "second line"},
        ]

    def test_a_null_end_takes_the_next_start_and_the_last_takes_the_song_end(self):
        from dw.tasks.cuts import _chunks

        parsed = _chunks({"chunks": self.chunks()}, 10.0)
        assert [(c["start"], c["end"]) for c in parsed] == [(1.0, 3.0), (3.0, 10.0)]

    def test_the_plan_runs_with_null_ends(self):
        result = plan_cuts({"chunks": self.chunks()}, duration_s=10.0)
        assert result["total_frames"] == 240
        assert sum(s["cut_frames"] for s in result["shots"]) == 240

    def test_the_song_end_can_come_from_the_beats(self):
        result = plan_cuts(
            {"chunks": self.chunks()}, beats={"beats": [1.0], "duration_seconds": 10.0}
        )
        assert result["duration_s"] == 10.0

    def test_no_duration_and_a_null_last_end_is_refused(self):
        with pytest.raises(ValueError, match="duration_s"):
            plan_cuts({"chunks": self.chunks()})


class TestBareStringTranscript:
    def test_a_string_is_refused_naming_return_timestamps(self):
        with pytest.raises(ValueError, match="return_timestamps"):
            plan(transcript="just some text")

    def test_the_static_check_reports_it_on_transcript(self):
        problems = cuts_errors({"transcript": "some text"})
        assert [name for name, _ in problems] == ["transcript"]
        assert "return_timestamps" in problems[0][1]

    def test_a_reference_is_not_complained_about(self):
        assert cuts_errors({"transcript": "previous_result:transcribe"}) == []


class TestResultShape:
    def test_a_dict_with_the_planned_keys(self):
        result = plan(lyrics=LYRICS)
        assert isinstance(result, dict)
        assert {
            "shots",
            "bpm",
            "fps",
            "warnings",
            "duration_s",
            "total_frames",
        } <= set(result)
        assert result["fps"] == FPS
        assert result["duration_s"] == DURATION
        assert result["total_frames"] == round(DURATION * FPS)
        json.dumps(result)

    def test_each_shot_has_exactly_the_shot_keys(self):
        for shot in plan(lyrics=LYRICS)["shots"]:
            assert set(shot) == SHOT_KEYS
            assert shot["num_frames"] == shot["cut_frames"]
            assert shot["lead_frames"] == 0
            assert shot["kind"] in ("vocal", "instrumental")

    def test_shots_are_named_in_order(self):
        names = [s["name"] for s in plan(lyrics=LYRICS)["shots"]]
        assert names == [f"shot_{i:02d}" for i in range(1, len(names) + 1)]


def assert_tiles(result):
    shots = result["shots"]
    assert shots[0]["start_frame"] == 0
    for before, after in zip(shots, shots[1:]):
        assert after["start_frame"] == before["start_frame"] + before["cut_frames"]
    assert all(s["cut_frames"] > 0 for s in shots)
    total = sum(s["cut_frames"] for s in shots)
    assert (
        total == result["total_frames"] == round(result["duration_s"] * result["fps"])
    )


class TestCoverage:
    @pytest.mark.parametrize("segment_by", ["line", "stanza", "beat"])
    @pytest.mark.parametrize("gaps", [True, False])
    @pytest.mark.parametrize("snap", [True, False])
    @pytest.mark.parametrize("lyrics", [None, LYRICS])
    def test_the_shots_tile_the_song(self, segment_by, gaps, snap, lyrics):
        result = plan(
            lyrics=lyrics,
            beats=beat_list(0.43),
            segment_by=segment_by,
            include_instrumental_gaps=gaps,
            snap_to_beats=snap,
        )
        assert_tiles(result)

    def test_a_fractional_fps_still_tiles(self):
        result = plan(lyrics=LYRICS, fps=23.976, beats=beat_list(0.47))
        assert_tiles(result)

    def test_no_sung_lines_is_one_instrumental_shot(self):
        result = plan_cuts({"chunks": []}, duration_s=10.0)
        assert [s["kind"] for s in result["shots"]] == ["instrumental"]
        assert_tiles(result)


class TestInstrumental:
    def test_intro_and_outro_become_instrumental_shots(self):
        result = plan(lyrics=LYRICS, vocal_tail_s=0.5)
        first, last = result["shots"][0], result["shots"][-1]
        assert first["kind"] == "instrumental" and first["lyric"] is None
        assert first["start_frame"] == 0
        assert first["cut_frames"] == round(4.0 * FPS)
        assert last["kind"] == "instrumental" and last["lyric"] is None
        assert last["start_frame"] == round(24.5 * FPS)

    def test_the_long_middle_gap_is_its_own_shot(self):
        result = plan(lyrics=LYRICS, vocal_tail_s=0.5)
        kinds = [s["kind"] for s in result["shots"]]
        assert kinds == [
            "instrumental",
            "vocal",
            "vocal",
            "instrumental",
            "vocal",
            "vocal",
            "instrumental",
        ]

    def test_without_the_option_gaps_are_held_by_the_neighbouring_shots(self):
        result = plan(lyrics=LYRICS, vocal_tail_s=0.5, include_instrumental_gaps=False)
        assert all(s["kind"] == "vocal" for s in result["shots"])
        assert len(result["shots"]) == 4
        assert result["shots"][0]["start_frame"] == 0
        assert result["shots"][0]["lyric"] == LINES[0]
        assert_tiles(result)

    def test_a_gap_under_min_gap_seconds_is_not_a_shot(self):
        result = plan(lyrics=LYRICS, min_gap_seconds=5.0, vocal_tail_s=0.0)
        # The 4 s intro is under 5 s; the 8 s middle gap and 6 s outro are over
        kinds = [s["kind"] for s in result["shots"]]
        assert kinds[0] == "vocal"
        assert kinds.count("instrumental") == 2


class TestSnapping:
    def test_every_internal_cut_lands_on_a_beat(self):
        beats = {"beats": beat_list(0.43), "bpm": 139.53, "duration_seconds": 30.0}
        result = plan(lyrics=LYRICS, beats=beats, snap_to_beats=True)
        frames = [round(b * FPS) for b in beats["beats"]]
        starts = [s["start_frame"] for s in result["shots"][1:]]
        assert starts
        for start in starts:
            assert min(abs(start - f) for f in frames) <= 1

    def test_snapping_without_beats_warns_and_keeps_the_cuts(self):
        result = plan(lyrics=LYRICS, snap_to_beats=True)
        assert any("snap_to_beats" in w for w in result["warnings"])
        assert_tiles(result)


class TestSceneLengths:
    def check_range(self, result, low, high):
        for shot in result["shots"]:
            seconds = shot["cut_frames"] / FPS
            named = any(shot["name"] in w for w in result["warnings"])
            inside = low - 0.5 / FPS <= seconds <= high + 0.5 / FPS
            assert inside or named, (shot["name"], seconds)

    @pytest.mark.parametrize("segment_by", ["line", "stanza", "beat"])
    def test_shots_stay_in_range_or_are_warned_about(self, segment_by):
        result = plan(
            lyrics=LYRICS,
            beats=beat_list(0.5),
            segment_by=segment_by,
            min_scene_s=2.0,
            max_scene_s=6.0,
        )
        self.check_range(result, 2.0, 6.0)
        assert_tiles(result)

    def test_a_short_line_merges_into_a_neighbour(self):
        chunks = [
            {"start": 0.0, "end": 0.4, "text": "oh"},
            {"start": 0.4, "end": 4.0, "text": "here we go again"},
        ]
        result = plan_cuts({"chunks": chunks}, duration_s=4.0, min_scene_s=1.0)
        assert len(result["shots"]) == 1
        assert result["shots"][0]["lyric"] == "oh\nhere we go again"

    def test_a_long_line_splits_on_a_beat(self):
        chunks = [{"start": 0.0, "end": 10.0, "text": "a very long held line"}]
        beats = [i * 0.7 for i in range(1, 14)]
        result = plan_cuts(
            {"chunks": chunks}, beats=beats, duration_s=10.0, max_scene_s=6.0
        )
        shots = result["shots"]
        assert len(shots) == 2
        cut = shots[1]["start_frame"]
        assert min(abs(cut - round(b * FPS)) for b in beats) <= 1
        # Both pieces keep the line
        assert [s["lyric"] for s in shots] == ["a very long held line"] * 2
        assert_tiles(result)

    def test_a_long_line_splits_evenly_without_beats(self):
        chunks = [{"start": 0.0, "end": 10.0, "text": "a very long held line"}]
        result = plan_cuts({"chunks": chunks}, duration_s=10.0, max_scene_s=6.0)
        assert [s["cut_frames"] for s in result["shots"]] == [120, 120]

    def test_a_long_line_is_split_into_enough_pieces(self):
        chunks = [{"start": 0.0, "end": 10.0, "text": "a very long held line"}]
        result = plan_cuts({"chunks": chunks}, duration_s=10.0, max_scene_s=3.0)
        assert [s["cut_frames"] for s in result["shots"]] == [60] * 4


class TestSegmentBy:
    def test_a_stanza_is_one_shot_of_several_lines(self):
        result = plan(lyrics=LYRICS, segment_by="stanza")
        lyrics = [s["lyric"] for s in result["shots"] if s["lyric"]]
        assert lyrics == ["\n".join(LINES[:2]), "\n".join(LINES[2:])]
        assert sung(result) == LINES

    def test_the_transcripts_own_gaps_make_stanzas(self):
        result = plan(segment_by="stanza")
        lyrics = [s["lyric"] for s in result["shots"] if s["lyric"]]
        assert len(lyrics) == 2
        assert all("\n" in lyric for lyric in lyrics)

    def test_beat_mode_cuts_on_beats_and_carries_lyrics(self):
        result = plan(lyrics=LYRICS, beats=beat_list(2.0), segment_by="beat")
        assert len(result["shots"]) == 15
        assert all(s["cut_frames"] == 2 * FPS for s in result["shots"])
        assert sung(result) == LINES

    def test_beat_mode_needs_beats(self):
        with pytest.raises(ValueError, match="beats"):
            plan(segment_by="beat")

    def test_a_bad_segment_by_is_refused(self):
        with pytest.raises(ValueError, match="segment_by"):
            plan(segment_by="bogus")

    def test_an_inverted_scene_range_is_refused(self):
        with pytest.raises(ValueError, match="min_scene_s"):
            plan(min_scene_s=5.0, max_scene_s=2.0)
        problems = cuts_errors({"min_scene_s": 5.0, "max_scene_s": 2.0})
        assert [name for name, _ in problems] == ["min_scene_s"]


class TestBeats:
    def test_a_bare_list_gives_a_bpm_from_the_median_spacing(self):
        result = plan(beats=beat_list(0.5))
        assert result["bpm"] == 120.0

    def test_the_bpm_ignores_an_outlying_gap(self):
        beats = [*beat_list(0.5)[:20], 11.3, 11.8]
        assert plan(beats=beats)["bpm"] == 120.0

    def test_an_analyze_beats_dict_brings_its_bpm_and_duration(self):
        result = plan_cuts(
            TRANSCRIPT,
            beats={"beats": beat_list(0.5), "bpm": 118.0, "duration_seconds": 30.0},
        )
        assert result["bpm"] == 118.0
        assert result["duration_s"] == 30.0

    def test_no_beats_means_no_bpm(self):
        assert plan()["bpm"] is None

    def test_a_bad_beats_value_is_refused(self):
        with pytest.raises(ValueError, match="beats"):
            plan(beats="fast")


class TestWarnings:
    def test_warnings_are_emitted(self, captured_warnings):
        lyrics = LYRICS + "\nZzyzx quux flibber\n"
        result = plan(lyrics=lyrics)
        assert result["warnings"]
        assert captured_warnings == result["warnings"]

    def test_no_duration_falls_back_to_the_last_line_with_a_warning(self):
        result = plan_cuts(TRANSCRIPT)
        assert result["duration_s"] == 24.0
        assert any("duration_s" in w for w in result["warnings"])


class TestRegistry:
    def test_the_registered_handler(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        result = _COMMAND_REGISTRY["plan_cuts"](
            None,
            {
                "transcript": TRANSCRIPT,
                "lyrics": LYRICS,
                "beats": beat_list(0.5),
                "duration_s": DURATION,
                "snap_to_beats": True,
            },
            {},
        )
        assert isinstance(result, dict)
        assert sung(result) == LINES
        assert_tiles(result)


def workflow_errors(arguments):
    return task_argument_errors(
        {
            "id": "cuts",
            "steps": [
                {
                    "name": "cuts",
                    "task": {"command": "plan_cuts", "arguments": arguments},
                    "result": {"content_type": "application/json"},
                }
            ],
        }
    )


class TestStaticValidation:
    def test_a_literal_string_transcript_is_reported_on_its_argument(self):
        errors = workflow_errors({"transcript": "la la la"})
        assert errors
        assert any(
            "arguments.transcript" in str(error) and "return_timestamps" in str(error)
            for error in errors
        ), errors

    def test_a_bad_segment_by_is_a_choice_error(self):
        errors = workflow_errors(
            {"transcript": "previous_result:transcribe", "segment_by": "bogus"}
        )
        assert any("segment_by" in str(error) for error in errors), errors

    def test_references_and_good_values_pass(self):
        assert (
            workflow_errors(
                {
                    "transcript": "previous_result:transcribe",
                    "beats": "previous_result:beats",
                    "segment_by": "beat",
                    "min_scene_s": 1,
                    "max_scene_s": 8,
                }
            )
            == []
        )
