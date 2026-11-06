"""A section shorter than its slice, kept at its own length (#1199 Task 22).

`slice_audio` zero-pads a slice that reaches past its source. Joined pieces
built from generated sections (`templates/minimax/music-long`) want the
opposite: a section that came back shorter than asked is kept as it is, so
the crossfade joins sound to sound rather than silence to the next intro.
`past_end: "stop"` asks for that, and the joined piece then comes out shorter
than its sections asked for - which `crossfade_audio` reports as a warning
naming both totals, since a caller reading the job sees the warnings list and
nothing else.
"""

import json
import os
import wave
from pathlib import Path

import numpy
import pytest

from dw.events import RunContext, activate_context, deactivate_context
from dw.tasks.audio_utils import crossfade_audio, fade_audio, slice_audio

RATE = 100
ROOT = Path(__file__).resolve().parent.parent


def tone(seconds, rate=RATE):
    return numpy.full((1, int(seconds * rate)), 0.5, dtype=numpy.float32)


def events_of(call):
    events = []
    token = activate_context(RunContext(on_event=events.append))
    try:
        result = call()
    finally:
        deactivate_context(token)
    return result, events


def kinds(events, kind):
    return [e for e in events if e.get("kind") == kind]


class TestSliceStop:
    def test_a_short_source_is_kept_at_its_own_length(self):
        sliced, events = events_of(
            lambda: slice_audio(
                tone(3.0),
                start_seconds=0,
                duration_seconds=8.5,
                sample_rate=RATE,
                past_end="stop",
            )
        )
        assert sliced.audio.shape == (1, 300)
        assert numpy.all(sliced.audio == 0.5)
        # Nothing was padded, so the padding warning has nothing to say
        assert kinds(events, "slice_past_end") == []
        assert sliced.asked_seconds == pytest.approx(8.5)

    def test_a_long_source_is_still_cut_at_the_length_asked(self):
        sliced = slice_audio(
            tone(10.0),
            start_seconds=0,
            duration_seconds=8.5,
            sample_rate=RATE,
            past_end="stop",
        )
        assert sliced.audio.shape == (1, 850)
        assert sliced.asked_seconds == pytest.approx(8.5)

    def test_the_default_still_pads(self):
        sliced, events = events_of(
            lambda: slice_audio(
                tone(3.0), start_seconds=0, duration_seconds=8.5, sample_rate=RATE
            )
        )
        assert sliced.audio.shape == (1, 850)
        assert len(kinds(events, "slice_past_end")) == 1
        assert getattr(sliced, "asked_seconds", None) is None

    def test_a_slice_starting_past_the_end_is_refused(self):
        with pytest.raises(ValueError, match="past the end"):
            slice_audio(
                tone(3.0),
                start_seconds=4.0,
                duration_seconds=2.0,
                sample_rate=RATE,
                past_end="stop",
            )

    def test_an_unknown_past_end_is_refused_at_run_time(self):
        with pytest.raises(ValueError, match="past_end"):
            slice_audio(
                tone(3.0),
                duration_seconds=2.0,
                sample_rate=RATE,
                past_end="loop",
            )

    def test_an_unknown_past_end_is_refused_at_validate_time(self):
        from dw.workflow import Workflow

        definition = {
            "id": "slice",
            "steps": [
                {
                    "name": "cut",
                    "task": {
                        "command": "slice_audio",
                        "arguments": {
                            "audio": "./a.wav",
                            "duration_seconds": 2,
                            "past_end": "loop",
                        },
                    },
                    "result": {"content_type": "audio/wav"},
                }
            ],
        }
        errors = Workflow(
            definition, str(ROOT), str(ROOT / "x.json")
        ).validation_errors()
        assert any(
            e["path"].endswith("past_end") and "one of" in e["message"] for e in errors
        ), errors

    def test_a_kept_short_slice_is_logged(self):
        _, events = events_of(
            lambda: slice_audio(
                tone(3.0),
                duration_seconds=8.5,
                sample_rate=RATE,
                past_end="stop",
            )
        )
        logs = [e for e in events if "kept" in str(e.get("message", ""))]
        assert logs, events


class TestFadeCarriesTheAskedLength:
    def test_fade_keeps_what_the_slice_was_asked_for(self):
        sliced = slice_audio(
            tone(3.0), duration_seconds=8.5, sample_rate=RATE, past_end="stop"
        )
        faded = fade_audio(sliced, fade_out_ms=400)
        assert faded.asked_seconds == pytest.approx(8.5)


class TestCrossfadeReportsAShortPiece:
    def sections(self, lengths, asked):
        return [
            fade_audio(
                slice_audio(
                    tone(length),
                    duration_seconds=wanted,
                    sample_rate=RATE,
                    past_end="stop",
                ),
                fade_out_ms=100,
            )
            for length, wanted in zip(lengths, asked)
        ]

    def test_the_shortfall_names_the_real_and_the_asked_total(self):
        tracks = self.sections([3.0, 10.0, 5.0], [8.5, 8.5, 8.5])
        joined, events = events_of(lambda: crossfade_audio(tracks, crossfade_ms=500))
        found = kinds(events, "joined_short")
        assert len(found) == 1
        warning = found[0]
        real = joined.audio.shape[1] / RATE
        # 3 + 8.5 + 5 less two 0.5 s overlaps
        assert real == pytest.approx(15.5)
        assert warning["seconds"] == pytest.approx(15.5)
        # 3 x 8.5 less the same two overlaps
        assert warning["asked_seconds"] == pytest.approx(24.5)
        assert "15.50 s" in warning["message"]
        assert "24.50 s" in warning["message"]
        # Each short track is named with its own two lengths
        assert warning["short_tracks"] == {
            "track 1": [3.0, 8.5],
            "track 3": [5.0, 8.5],
        }

    def test_sections_at_full_length_say_nothing(self):
        tracks = self.sections([10.0, 10.0], [8.5, 8.5])
        _, events = events_of(lambda: crossfade_audio(tracks, crossfade_ms=500))
        assert kinds(events, "joined_short") == []

    def test_tracks_that_carry_no_asked_length_say_nothing(self):
        _, events = events_of(
            lambda: crossfade_audio(
                [tone(3.0), tone(2.0)], crossfade_ms=100, sample_rate=RATE
            )
        )
        assert kinds(events, "joined_short") == []


def write_wav(path, seconds, rate=8000):
    samples = (numpy.full(int(seconds * rate), 0.4) * 32767).astype("<i2")
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_a_run_shaped_like_music_long_reaches_the_job_with_both_totals(
    tmp_path, monkeypatch
):
    """The template's trim -> fade -> crossfade chain, run through the engine,
    with the generated sections stood in for by assets: a short section is
    kept at its length and the job hears about the short piece."""
    from dw.workflow import Workflow

    template = json.loads(
        (ROOT / "workflows" / "templates" / "minimax" / "music-long.json").read_text()
    )
    by_name = {s["name"]: s for s in template["steps"]}
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    write_wav(asset_dir / "a.wav", 2.0)
    write_wav(asset_dir / "b.wav", 6.0)
    monkeypatch.setenv("DW_ASSET_DIR", str(asset_dir))

    trim = json.loads(json.dumps(by_name["trim"]))
    trim["task"]["arguments"]["audio"] = "item:audio"
    trim["task"]["arguments"]["sample_rate"] = 8000
    joined = json.loads(json.dumps(by_name["joined"]))
    joined["task"]["arguments"]["sample_rate"] = 8000
    definition = {
        "id": "music-long-shape",
        "variables": {
            "sections": [
                {"name": "one", "audio": "asset:a.wav", "seconds": 4},
                {"name": "two", "audio": "asset:b.wav", "seconds": 4},
            ],
            "crossfade_ms": 500,
            "section_fade_out_ms": 100,
        },
        "steps": [trim, by_name["fade"], joined],
    }
    workflow = Workflow(definition, str(tmp_path), str(tmp_path / "w.json"))
    assert workflow.validation_errors() == []
    _, events = events_of(lambda: workflow.run({}))
    assert kinds(events, "slice_past_end") == []
    found = kinds(events, "joined_short")
    assert len(found) == 1, events
    # 2 + 4 - 0.5 against 4 + 4 - 0.5
    assert found[0]["seconds"] == pytest.approx(5.5, abs=0.01)
    assert found[0]["asked_seconds"] == pytest.approx(7.5, abs=0.01)


def test_music_long_keeps_a_short_section_at_its_own_length():
    template = json.loads(
        (ROOT / "workflows" / "templates" / "minimax" / "music-long.json").read_text()
    )
    trim = next(s for s in template["steps"] if s["name"] == "trim")
    assert trim["task"]["arguments"]["past_end"] == "stop"
    description = template["description"]
    assert "lyric" in description and "length" in description


def test_slice_preflight_does_not_warn_a_slice_that_stops(tmp_path, monkeypatch):
    from dw.slice_preflight import slice_past_end_warnings

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    write_wav(asset_dir / "a.wav", 2.0)
    monkeypatch.setenv("DW_ASSET_DIR", str(asset_dir))
    step = {
        "name": "cut",
        "task": {
            "command": "slice_audio",
            "arguments": {"audio": "asset:a.wav", "duration_seconds": 5.0},
        },
        "result": {"content_type": "audio/wav"},
    }
    definition = {"id": "s", "steps": [step]}
    assert len(slice_past_end_warnings(definition, base_dir=str(tmp_path))) == 1
    step["task"]["arguments"]["past_end"] = "stop"
    assert slice_past_end_warnings(definition, base_dir=str(tmp_path)) == []
    assert os.path.exists(asset_dir / "a.wav")
