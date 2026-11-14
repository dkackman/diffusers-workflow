"""A slice that cuts a source while it is still playing (#1199 Task 22).

`templates/minimax/music-long` trims each generated section to its
`seconds`. A section that ran longer was cut wherever `seconds` fell -
mid-lyric, then full band into the next section's quiet intro - and no
warning fired, since the small-trim warning only covers a remainder under
5% of the slice and under 10 s. Two answers here: a `past_end: "stop"`
slice (a section, not an excerpt) that drops a tail still carrying real
signal warns `slice_dropped_loud_tail` whatever its length, and the opt-in
`end_snap_seconds` moves the cut back to the quietest point in that window
before the end - in a song, a phrase gap or a bar break.
"""

from pathlib import Path

import numpy
import pytest

from dw.events import RunContext, activate_context, deactivate_context
from dw.tasks.audio_slice import slice_audio
from dw.tasks.audio_utils import crossfade_audio, fade_audio

RATE = 1000
ROOT = Path(__file__).resolve().parent.parent


def tone(seconds, level=0.5, rate=RATE):
    count = int(round(seconds * rate))
    t = numpy.arange(count) / rate
    return (level * numpy.sin(2 * numpy.pi * 50 * t)).astype(numpy.float32)[None, :]


def song(*parts):
    """Concatenate (seconds, level) parts into one mono waveform."""
    return numpy.concatenate([tone(s, level) for s, level in parts], axis=1)


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


class TestLoudTailWarns:
    def test_a_long_loud_overrun_warns_with_seconds_and_level(self):
        sliced, events = events_of(
            lambda: slice_audio(
                tone(40.0),
                duration_seconds=20.0,
                sample_rate=RATE,
                past_end="stop",
            )
        )
        assert sliced.audio.shape[1] == 20 * RATE
        found = kinds(events, "slice_dropped_loud_tail")
        assert len(found) == 1, events
        warning = found[0]
        assert warning["dropped_seconds"] == pytest.approx(20.0)
        assert warning["dropped_rms_dbfs"] == pytest.approx(
            warning["kept_rms_dbfs"], abs=0.5
        )
        assert "20.00 s" in warning["message"]
        # The small-trim warning keeps to its own case
        assert kinds(events, "slice_trimmed_tail") == []

    def test_an_unsnapped_cut_says_it_fell_while_the_source_played(self):
        _, events = events_of(
            lambda: slice_audio(
                tone(40.0), duration_seconds=20.0, sample_rate=RATE, past_end="stop"
            )
        )
        message = kinds(events, "slice_dropped_loud_tail")[0]["message"]
        assert "mid-signal" not in message
        assert "cut at 20.00 s while it was still playing" in message
        assert "moved the cut" not in message

    def test_a_snapped_cut_says_the_tail_played_not_the_cut(self):
        # The snap moved the cut into the 15.0-15.3 s gap: the cut is quiet,
        # the dropped tail after it is not (#1199: "cut it mid-signal at
        # 82.17 s" when the cut sat in a -51 dBFS gap)
        source = song((15.0, 0.5), (0.3, 0.0), (24.7, 0.5))
        sliced, events = events_of(
            lambda: slice_audio(
                source,
                duration_seconds=20.0,
                sample_rate=RATE,
                past_end="stop",
                end_snap_seconds=8.0,
            )
        )
        warning = kinds(events, "slice_dropped_loud_tail")[0]
        message = warning["message"]
        kept = sliced.audio.shape[1] / RATE
        assert "mid-signal" not in message
        assert f"moved the cut {20.0 - kept:.2f} s back from 20.00 s" in message
        assert f"to a quiet point at {kept:.2f} s" in message
        assert "still playing" in message
        assert warning["snapped_seconds"] == pytest.approx(20.0 - kept, abs=1e-3)

    def test_a_quiet_decay_tail_does_not_warn(self):
        source = song((20.0, 0.5), (20.0, 0.005))
        _, events = events_of(
            lambda: slice_audio(
                source, duration_seconds=20.0, sample_rate=RATE, past_end="stop"
            )
        )
        assert kinds(events, "slice_dropped_loud_tail") == []

    def test_an_excerpt_with_the_default_past_end_does_not_warn(self):
        # A deliberate excerpt out of a long recording drops loud material
        # by design; only a slice declared a section warns
        _, events = events_of(
            lambda: slice_audio(tone(40.0), duration_seconds=20.0, sample_rate=RATE)
        )
        assert kinds(events, "slice_dropped_loud_tail") == []

    def test_the_small_trim_case_still_warns_its_own_kind_only(self):
        _, events = events_of(
            lambda: slice_audio(
                tone(20.5),
                duration_seconds=20.0,
                sample_rate=RATE,
                past_end="stop",
            )
        )
        assert len(kinds(events, "slice_trimmed_tail")) == 1
        assert kinds(events, "slice_dropped_loud_tail") == []


class TestEndSnap:
    def test_off_is_the_exact_cut(self):
        source = song((15.0, 0.5), (0.3, 0.0), (24.7, 0.5))
        sliced = slice_audio(source, duration_seconds=20.0, sample_rate=RATE)
        assert sliced.audio.shape[1] == 20 * RATE
        assert getattr(sliced, "snapped_seconds", None) is None

    def test_the_cut_lands_in_a_quiet_gap_inside_the_window(self):
        # A 0.3 s phrase gap at 15.0-15.3 s, 5 s before the 20 s end
        source = song((15.0, 0.5), (0.3, 0.0), (24.7, 0.5))
        sliced, events = events_of(
            lambda: slice_audio(
                source,
                duration_seconds=20.0,
                sample_rate=RATE,
                past_end="stop",
                end_snap_seconds=8.0,
            )
        )
        kept = sliced.audio.shape[1] / RATE
        assert 15.0 <= kept <= 15.3
        assert sliced.asked_seconds == pytest.approx(20.0)
        assert sliced.snapped_seconds == pytest.approx(20.0 - kept)
        # The snap is logged, and the loud overrun still warns
        assert any("quiet" in str(e.get("message", "")) for e in events), events
        assert len(kinds(events, "slice_dropped_loud_tail")) == 1

    def test_a_gap_outside_the_window_is_not_reached(self):
        source = song((10.0, 0.5), (0.3, 0.0), (29.7, 0.5))
        sliced = slice_audio(
            source, duration_seconds=20.0, sample_rate=RATE, end_snap_seconds=8.0
        )
        assert sliced.audio.shape[1] / RATE >= 12.0

    def test_an_even_level_keeps_the_exact_cut(self):
        sliced = slice_audio(
            tone(40.0),
            duration_seconds=20.0,
            sample_rate=RATE,
            end_snap_seconds=8.0,
        )
        assert sliced.audio.shape[1] == 20 * RATE

    def test_a_short_source_has_nothing_to_snap(self):
        sliced = slice_audio(
            tone(10.0),
            duration_seconds=20.0,
            sample_rate=RATE,
            past_end="stop",
            end_snap_seconds=8.0,
        )
        assert sliced.audio.shape[1] == 10 * RATE
        assert getattr(sliced, "snapped_seconds", None) is None

    def test_zero_is_off(self):
        source = song((15.0, 0.5), (0.3, 0.0), (24.7, 0.5))
        sliced = slice_audio(
            source, duration_seconds=20.0, sample_rate=RATE, end_snap_seconds=0
        )
        assert sliced.audio.shape[1] == 20 * RATE

    def test_a_window_longer_than_the_slice_stays_inside_it(self):
        source = song((2.0, 0.5), (0.3, 0.0), (10.0, 0.5))
        sliced = slice_audio(
            source, duration_seconds=4.0, sample_rate=RATE, end_snap_seconds=8.0
        )
        assert 2.0 <= sliced.audio.shape[1] / RATE <= 2.3

    def test_a_numeric_string_is_read_as_a_number(self):
        source = song((15.0, 0.5), (0.3, 0.0), (24.7, 0.5))
        sliced = slice_audio(
            source, duration_seconds=20.0, sample_rate=RATE, end_snap_seconds="8"
        )
        assert sliced.audio.shape[1] / RATE <= 15.3

    @pytest.mark.parametrize("bad", [-1.0, "abc", float("nan")])
    def test_a_bad_value_is_refused_at_run_time(self, bad):
        with pytest.raises(ValueError, match="end_snap_seconds"):
            slice_audio(
                tone(40.0),
                duration_seconds=20.0,
                sample_rate=RATE,
                end_snap_seconds=bad,
            )

    @pytest.mark.parametrize("bad", [-1.0, "abc"])
    def test_a_bad_literal_is_refused_at_validate_time(self, bad):
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
                            "end_snap_seconds": bad,
                        },
                    },
                    "result": {"content_type": "audio/wav"},
                }
            ],
        }
        errors = Workflow(
            definition, str(ROOT), str(ROOT / "x.json")
        ).validation_errors()
        assert any(e["path"].endswith("end_snap_seconds") for e in errors), errors


class TestJoinedShortTellsASnapFromAnEarlyEnd:
    def snapped(self):
        source = song((15.0, 0.5), (0.3, 0.0), (24.7, 0.5))
        return slice_audio(
            source,
            duration_seconds=20.0,
            sample_rate=RATE,
            past_end="stop",
            end_snap_seconds=8.0,
        )

    def early(self):
        return slice_audio(
            tone(10.0), duration_seconds=20.0, sample_rate=RATE, past_end="stop"
        )

    def test_a_snapped_section_alone_is_logged_not_warned_as_ended_early(self):
        tracks = [fade_audio(self.snapped(), fade_out_ms=100), tone(5.0)]
        _, events = events_of(
            lambda: crossfade_audio(tracks, crossfade_ms=500, sample_rate=RATE)
        )
        assert kinds(events, "joined_short") == []
        assert any(
            "quiet" in str(e.get("message", "")) and e.get("event") == "log"
            for e in events
        ), events

    def test_both_kinds_are_named_apart_and_the_total_holds(self):
        snapped = fade_audio(self.snapped(), fade_out_ms=100)
        early = fade_audio(self.early(), fade_out_ms=100)
        snapped_kept = snapped.audio.shape[1] / RATE
        joined, events = events_of(
            lambda: crossfade_audio([early, snapped], crossfade_ms=500)
        )
        found = kinds(events, "joined_short")
        assert len(found) == 1, events
        warning = found[0]
        real = joined.audio.shape[1] / RATE
        assert warning["seconds"] == pytest.approx(real, abs=0.01)
        # 20 + 20 less one overlap
        assert warning["asked_seconds"] == pytest.approx(39.5, abs=0.01)
        assert warning["short_tracks"] == {"track 1": [10.0, 20.0]}
        assert warning["snapped_tracks"] == {"track 2": [round(snapped_kept, 3), 20.0]}
        assert "1 of them ended early" in warning["message"]
        assert "quiet" in warning["message"]


def test_a_window_as_long_as_the_slice_never_cuts_into_its_first_half():
    # A quiet lead-in at the head and a loud end: a window as long as the
    # slice would find the lead-in and leave a sliver of the section
    source = song((0.5, 0.0), (5.5, 0.5), (4.0, 0.5))
    sliced = slice_audio(
        source, duration_seconds=6.0, sample_rate=RATE, end_snap_seconds=6.0
    )
    assert sliced.audio.shape[1] / RATE >= 3.0
