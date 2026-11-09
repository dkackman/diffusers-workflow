"""match_levels "lufs": a BS.1770 delivery target on the join."""

import numpy
import pytest
from PIL import Image

from dw import dsp
from dw.events import RunContext, activate_context, deactivate_context
from dw.media_types import AudioVideo
from dw.tasks.concat_videos import concat_videos
from dw.tasks.joins import match_levels

RATE = 16000


def noise(seconds, gain):
    rng = numpy.random.default_rng(7)
    wave = rng.standard_normal((2, int(seconds * RATE))).astype(numpy.float32)
    return (wave / numpy.abs(wave).max() * gain).astype(numpy.float32)


def lufs_of(wave):
    return dsp.integrated_lufs(wave.T, RATE)


def run_with_events(call):
    events = []
    token = activate_context(RunContext(on_event=events.append))
    try:
        result = call()
    finally:
        deactivate_context(token)
    return result, [e for e in events if e["event"] == "warning"]


def test_two_shots_at_different_gains_land_within_half_an_lu_of_the_target():
    matched = match_levels([noise(2, 0.5), noise(2, 0.02)], "lufs", sample_rate=RATE)

    assert [lufs_of(w) for w in matched] == [
        pytest.approx(-16.0, abs=0.5),
        pytest.approx(-16.0, abs=0.5),
    ]


def test_a_gain_that_would_pass_the_true_peak_ceiling_is_held_and_named():
    spiky = noise(2, 0.01)
    spiky[:, 8000] = 0.8

    matched, warnings = run_with_events(
        lambda: match_levels([spiky, noise(2, 0.3)], "lufs", sample_rate=RATE)
    )

    assert dsp.true_peak_dbfs(matched[0].T) == pytest.approx(-1.0, abs=0.3)
    assert lufs_of(matched[0]) < -16.5
    held = [w for w in warnings if w["kind"] == "match_levels_held"]
    assert [w["index"] for w in held] == [0]
    assert "video 1" in held[0]["message"]


def test_a_shot_too_short_for_lufs_is_matched_by_rms_and_named():
    short = noise(0.2, 0.05)

    matched, warnings = run_with_events(
        lambda: match_levels([short, noise(2, 0.3)], "lufs", sample_rate=RATE)
    )

    assert dsp.level_dbfs(matched[0], "rms") == pytest.approx(-20.0, abs=0.5)
    kinds = [w for w in warnings if w["kind"] == "match_levels_short_shot"]
    assert len(kinds) == 1
    assert "video 1" in kinds[0]["message"]


def test_concat_refuses_a_dbfs_target_with_lufs():
    video = AudioVideo([Image.new("RGB", (8, 8))] * 4, noise(2, 0.3), RATE)

    with pytest.raises(ValueError, match="match_levels_lufs"):
        concat_videos([video, video], match_levels="lufs", match_levels_dbfs=-20)


def test_a_broadcast_target_is_honoured():
    matched = match_levels(
        [noise(2, 0.5), noise(2, 0.02)],
        "lufs",
        sample_rate=RATE,
        target_lufs=-23.0,
    )

    assert [lufs_of(w) for w in matched] == [
        pytest.approx(-23.0, abs=0.5),
        pytest.approx(-23.0, abs=0.5),
    ]


def test_concat_matches_the_joined_inputs_to_the_lufs_target():
    def video(gain):
        return AudioVideo([Image.new("RGB", (8, 8))] * 4, noise(2, gain), RATE)

    result = concat_videos(
        [video(0.5), video(0.02)],
        match_levels="lufs",
        match_levels_lufs=-23.0,
        crossfade_ms=0,
    )

    assert lufs_of(numpy.asarray(result.audio)) == pytest.approx(-23.0, abs=0.7)


def test_a_silent_video_entry_passes_through_a_lufs_match():
    loud = noise(2, 0.5)

    matched = match_levels([None, loud], "lufs", sample_rate=RATE)

    assert matched[0] is None
    assert lufs_of(matched[1]) == pytest.approx(-16.0, abs=0.5)


def test_concat_joins_a_video_with_no_soundtrack_under_lufs():
    silent = AudioVideo([Image.new("RGB", (8, 8))] * 4, None, None, fps=30)
    voiced = AudioVideo([Image.new("RGB", (8, 8))] * 4, noise(2, 0.3), RATE, fps=30)

    result = concat_videos([silent, voiced], match_levels="lufs", fps=30)

    assert len(result.frames) == 8


def test_the_lufs_held_warning_names_lufs_and_dbtp_not_dbfs():
    spiky = noise(2, 0.01)
    spiky[:, 8000] = 0.8

    _, warnings = run_with_events(
        lambda: match_levels([spiky, noise(2, 0.3)], "lufs", sample_rate=RATE)
    )

    held = [w for w in warnings if w["kind"] == "match_levels_held"][0]
    assert "dBTP" in held["message"]
    assert "dBFS" not in held["message"]
    assert held["unit"] == "LUFS"


def test_the_rms_held_warning_keeps_its_dbfs_wording():
    spiky = noise(2, 0.01)
    spiky[:, 8000] = 0.95

    _, warnings = run_with_events(lambda: match_levels([spiky, noise(2, 0.3)], "rms"))

    held = [w for w in warnings if w["kind"] == "match_levels_held"][0]
    assert "dBFS peak" in held["message"]
    assert held["unit"] == "dBFS"


def test_a_lufs_match_of_only_silent_entries_needs_no_sample_rate():
    assert match_levels([None, None], "lufs", sample_rate=None) == [None, None]


def test_a_lufs_match_of_waveforms_still_needs_a_sample_rate():
    with pytest.raises(ValueError, match="needs a sample rate"):
        match_levels([noise(2, 0.3), None], "lufs", sample_rate=None)


def test_concat_joins_videos_with_no_soundtrack_at_all_under_lufs():
    silent = AudioVideo([Image.new("RGB", (8, 8))] * 4, None, None, fps=30)

    result = concat_videos([silent, silent], match_levels="lufs", fps=30)

    assert len(result.frames) == 8
