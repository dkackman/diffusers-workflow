"""Unit tests for join_into_song - dialogue shots, then shots sung to one track."""

import os
import sys

import numpy
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.loudness import integrated_lufs
from dw.media_types import AudioTrack, AudioVideo
from dw.shots import shot_references
from dw.tasks.audio_utils import frames_to_samples
from dw.tasks.join_into_song import join_into_song
from dw.tasks.task import _COMMAND_REGISTRY

FPS = 8
SR = 8000


def frames(count, size=(16, 16), color=(0, 0, 0)):
    return [Image.new("RGB", size, color) for _ in range(count)]


def clip(num_frames, fps, sample_rate, level=None, channels=2, size=(16, 16)):
    """An AudioVideo with constant-level audio exactly matching its frames, or
    silent when level is None. Carries no fps of its own - the caller passes
    fps to join_into_song directly, matching the plain-list idiom."""
    if level is None:
        audio = None
    else:
        samples = frames_to_samples(num_frames, fps, sample_rate)
        audio = numpy.full((channels, samples), float(level), dtype=numpy.float32)
    return AudioVideo(frames(num_frames, size=size), audio, sample_rate)


def song_track(num_samples, sample_rate, channels=2, ramp=False, value=0.0):
    if ramp:
        base = numpy.arange(1, num_samples + 1, dtype=numpy.float32)
        audio = numpy.tile(base, (channels, 1))
    else:
        audio = numpy.full((channels, num_samples), float(value), dtype=numpy.float32)
    return AudioTrack(audio, sample_rate)


def sine(num_samples, sample_rate, frequency, amplitude, channels=2):
    t = numpy.arange(num_samples) / sample_rate
    wave = amplitude * numpy.sin(2 * numpy.pi * frequency * t)
    return numpy.tile(wave, (channels, 1)).astype(numpy.float32)


# --- 1. Song landing sample -------------------------------------------------


def test_song_lands_exactly_at_song_entry_with_zero_cue():
    dialogue = [clip(8, FPS, SR, level=None)]
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(8000, SR, ramp=True)

    result = join_into_song(dialogue, song_shots, song, fps=FPS)

    d = frames_to_samples(8, FPS, SR)
    assert result.shots[1]["start_sample"] == d
    assert numpy.allclose(result.audio[:, d], song.audio[:, 0])


def test_song_lands_cue_seconds_before_dialogue_ends():
    dialogue = [clip(8, FPS, SR, level=None)]
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(8000, SR, ramp=True)

    result = join_into_song(dialogue, song_shots, song, fps=FPS, cue_seconds=0.5)

    d = frames_to_samples(8, FPS, SR)
    entry = d - round(0.5 * SR)
    assert numpy.allclose(result.audio[:, entry], song.audio[:, 0])


# --- 2. Ducking --------------------------------------------------------------


def test_duck_ramps_dialogue_under_the_song_entry():
    level = 1.0
    dialogue = [clip(16, FPS, SR, level=level)]
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(20000, SR, value=0.0)

    result = join_into_song(
        dialogue,
        song_shots,
        song,
        fps=FPS,
        cue_seconds=1.0,
        duck_delay_ms=100,
        duck_db=-6,
        duck_ramp_ms=250,
    )

    d = frames_to_samples(16, FPS, SR)
    entry = d - round(1.0 * SR)
    start = entry + round(100 * SR / 1000.0)
    ramp = round(250 * SR / 1000.0)
    floor = 10.0 ** (-6 / 20.0)
    mix = result.audio

    assert numpy.allclose(mix[:, start - 1], level)
    for k in (0, 500, ramp - 1):
        expected = level * (1.0 + (floor - 1.0) * k / ramp)
        assert numpy.allclose(mix[:, start + k], expected, atol=1e-4)
    assert numpy.allclose(mix[:, start + ramp + 50], level * floor, atol=1e-4)
    assert numpy.allclose(mix[:, d - 1], level * floor, atol=1e-4)


def test_duck_delay_past_dialogue_end_leaves_dialogue_unchanged():
    level = 1.0
    dialogue = [clip(16, FPS, SR, level=level)]
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(20000, SR, value=0.0)

    result = join_into_song(
        dialogue,
        song_shots,
        song,
        fps=FPS,
        cue_seconds=1.0,
        duck_delay_ms=20000,
        duck_db=-6,
    )

    d = frames_to_samples(16, FPS, SR)
    assert numpy.allclose(result.audio[:, :d], level)


def test_duck_db_zero_leaves_dialogue_unchanged():
    level = 1.0
    dialogue = [clip(16, FPS, SR, level=level)]
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(20000, SR, value=0.0)

    result = join_into_song(
        dialogue,
        song_shots,
        song,
        fps=FPS,
        cue_seconds=1.0,
        duck_delay_ms=100,
        duck_db=0,
    )

    d = frames_to_samples(16, FPS, SR)
    assert numpy.allclose(result.audio[:, :d], level)


# --- 3. Song shots' own audio is discarded ----------------------------------


def test_song_shot_audio_is_discarded():
    dialogue = [clip(8, FPS, SR, level=None)]
    song_shots = [clip(8, FPS, SR, level=5.0)]
    song = song_track(8000, SR, value=0.0)

    result = join_into_song(dialogue, song_shots, song, fps=FPS)

    d = frames_to_samples(8, FPS, SR)
    assert numpy.allclose(result.audio[:, d:], 0.0)


# --- 4. dialogue_target_lufs -------------------------------------------------


def test_dialogue_target_lufs_matches_each_shot_independently():
    quiet = AudioVideo(frames(8), sine(8000, SR, 440, 0.05), SR, fps=FPS)
    loud = AudioVideo(frames(8), sine(8000, SR, 440, 0.5), SR, fps=FPS)
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(24000, SR, value=0.0)
    target = -20.0

    result = join_into_song(
        [quiet, loud], song_shots, song, fps=FPS, dialogue_target_lufs=target
    )

    for shot in result.shots[:2]:
        start, num = shot["start_sample"], shot["num_samples"]
        segment = result.audio[:, start : start + num]
        measured = integrated_lufs(segment.T, SR)
        assert measured is not None
        assert abs(measured - target) < 0.5


def test_dialogue_target_lufs_leaves_a_short_shot_unmatched_with_a_warning(caplog):
    loud = AudioVideo(frames(8), sine(8000, SR, 440, 0.5), SR, fps=FPS)
    short = AudioVideo(frames(1), sine(1000, SR, 440, 0.5), SR, fps=FPS)
    song_shots = [clip(8, FPS, SR, level=None)]
    song = song_track(24000, SR, value=0.0)

    with caplog.at_level("WARNING"):
        result = join_into_song(
            [loud, short], song_shots, song, fps=FPS, dialogue_target_lufs=-20.0
        )

    assert "left at its own level" in caplog.text
    short_shot = result.shots[1]
    start, num = short_shot["start_sample"], short_shot["num_samples"]
    # Unmatched: still exactly the un-gained amplitude 0.5 sine.
    assert numpy.allclose(
        result.audio[:, start : start + num],
        sine(num, SR, 440, 0.5),
        atol=1e-4,
    )


# --- 5. Shot map --------------------------------------------------------------


def test_shot_map_partitions_frames_and_samples_with_names_and_hard_cuts():
    fps, sr = 4, 100
    dialogue = [frames(4), frames(6)]
    song_shots = [frames(5), frames(3)]
    song = song_track(1000, sr, value=0.0)

    result = join_into_song(dialogue, song_shots, song, fps=fps)

    shots = result.shots
    assert len(shots) == 4
    assert [shot["name"] for shot in shots] == [
        "video 1",
        "video 2",
        "video 3",
        "video 4",
    ]

    total_frames = 4 + 6 + 5 + 3
    frame_cursor = 0
    for shot in shots:
        assert shot["start_frame"] == frame_cursor
        frame_cursor += shot["num_frames"]
    assert frame_cursor == total_frames

    assert sum(shot["num_samples"] for shot in shots) == result.audio.shape[1]
    sample_cursor = 0
    for shot in shots:
        assert shot["start_sample"] == sample_cursor
        sample_cursor += shot["num_samples"]
    assert sample_cursor == result.audio.shape[1]

    assert "hard_cut" not in shots[0]
    for shot in shots[1:]:
        assert shot["hard_cut"] is True

    assert [shot["source_index"] for shot in shots] == [0, 1, 2, 3]

    assert result.fps == fps
    assert result.audio.shape[1] == frames_to_samples(total_frames, fps, sr)


# --- 6. Silent dialogue shot ---------------------------------------------------


def test_silent_dialogue_shot_leaves_silence_and_next_shot_starts_on_time():
    fps, sr = 4, 100
    silent = frames(4)
    loud = clip(4, fps, sr, level=2.0)
    song_shots = [frames(2)]
    song = song_track(1000, sr, value=0.0)

    result = join_into_song([silent, loud], song_shots, song, fps=fps)

    per_shot = frames_to_samples(4, fps, sr)
    mix = result.audio
    assert numpy.allclose(mix[:, :per_shot], 0.0)
    assert numpy.allclose(mix[:, per_shot : 2 * per_shot], 2.0)


# --- 7. Dialogue fitted to its frames -----------------------------------------


def test_dialogue_audio_longer_than_its_frames_is_trimmed_with_a_warning(caplog):
    frame_count = 8
    length = frames_to_samples(frame_count, FPS, SR)
    over = numpy.full((2, length + 2000), 3.0, dtype=numpy.float32)
    dialogue = [AudioVideo(frames(frame_count), over, SR, fps=FPS)]
    song_shots = [frames(8)]
    song = song_track(20000, SR, ramp=True)

    with caplog.at_level("WARNING"):
        result = join_into_song(dialogue, song_shots, song, fps=FPS)

    assert "trimmed" in caplog.text
    d = frames_to_samples(frame_count, FPS, SR)
    assert numpy.allclose(result.audio[:, d], song.audio[:, 0])
    assert numpy.allclose(result.audio[:, :d], 3.0)


def test_dialogue_audio_shorter_than_its_frames_is_padded_with_a_warning(caplog):
    frame_count = 8
    length = frames_to_samples(frame_count, FPS, SR)
    under = numpy.full((2, length - 2000), 3.0, dtype=numpy.float32)
    dialogue = [AudioVideo(frames(frame_count), under, SR, fps=FPS)]
    song_shots = [frames(8)]
    song = song_track(20000, SR, ramp=True)

    with caplog.at_level("WARNING"):
        result = join_into_song(dialogue, song_shots, song, fps=FPS)

    assert "padded with silence" in caplog.text
    d = frames_to_samples(frame_count, FPS, SR)
    assert numpy.allclose(result.audio[:, d], song.audio[:, 0])
    assert numpy.allclose(result.audio[:, : length - 2000], 3.0)
    assert numpy.allclose(result.audio[:, length - 2000 : d], 0.0)


# --- 8. Dialogue at a different sample rate -----------------------------------


def test_dialogue_at_a_different_sample_rate_is_resampled_to_the_songs_rate():
    frame_count = 8
    dialogue_sr = 4000
    dialogue_length = frames_to_samples(frame_count, FPS, dialogue_sr)
    audio = numpy.full((2, dialogue_length), 1.0, dtype=numpy.float32)
    dialogue = [AudioVideo(frames(frame_count), audio, dialogue_sr, fps=FPS)]
    song_shots = [frames(8)]
    song = song_track(20000, SR, value=0.0)

    result = join_into_song(dialogue, song_shots, song, fps=FPS)

    assert result.sample_rate == SR


# --- 9. Song too short ---------------------------------------------------------


def test_song_too_short_is_padded_with_silence_and_warns(caplog):
    dialogue = [clip(8, FPS, SR, level=None)]
    song_shots = [frames(8)]
    song = song_track(1000, SR, value=1.0)

    with caplog.at_level("WARNING"):
        result = join_into_song(dialogue, song_shots, song, fps=FPS)

    assert "padded with" in caplog.text
    d = frames_to_samples(8, FPS, SR)
    assert numpy.allclose(result.audio[:, d : d + 1000], 1.0)
    assert numpy.allclose(result.audio[:, d + 1000 :], 0.0)


# --- 10. Refusals ---------------------------------------------------------------


def test_refuses_empty_dialogue():
    with pytest.raises(ValueError, match="non-empty"):
        join_into_song([], [frames(4)], song_track(1000, SR), fps=4)


def test_refuses_empty_song_shots():
    with pytest.raises(ValueError, match="non-empty"):
        join_into_song([frames(4)], [], song_track(1000, SR), fps=4)


def test_refuses_non_list_dialogue():
    with pytest.raises(ValueError, match="non-empty"):
        join_into_song("not a list", [frames(4)], song_track(1000, SR), fps=4)


def test_refuses_cue_seconds_longer_than_dialogue():
    dialogue = [clip(8, FPS, SR, level=None)]
    song_shots = [frames(8)]
    song = song_track(20000, SR)

    with pytest.raises(ValueError, match="before the film"):
        join_into_song(dialogue, song_shots, song, fps=FPS, cue_seconds=2.0)


def test_refuses_mismatched_fps_among_the_inputs():
    dialogue = [AudioVideo(frames(4), None, SR, fps=8)]
    song_shots = [AudioVideo(frames(4), None, SR, fps=4)]

    with pytest.raises(ValueError, match="one frame rate"):
        join_into_song(dialogue, song_shots, song_track(1000, SR))


def test_refuses_a_shot_a_previous_step_wrote_at_another_rate(tmp_path):
    # The #513 bounce: pair_audio kept a 24 fps asset's rate, the step wrote
    # it with result.fps 12, and the join read the in-memory 24 - so the shot
    # was re-timed in silence. The saved step's rate is what the join sees
    from unittest.mock import patch

    from dw.result import Result

    shot = AudioVideo(frames(4), None, SR, fps=24)
    written = Result({"content_type": "video/mp4", "fps": 12})
    written.add_result(shot)
    with patch("dw.result.export_to_video"):
        written.save(str(tmp_path), "co12")
    [twelve] = written.get_artifacts()

    dialogue = [AudioVideo(frames(4), None, SR, fps=24)]
    song_shots = [twelve, AudioVideo(frames(4), None, SR, fps=24)]

    with pytest.raises(ValueError, match=r"24 fps.*12 fps|12 fps.*24 fps"):
        join_into_song(dialogue, song_shots, song_track(1000, SR))


def test_refuses_an_fps_the_videos_contradict():
    dialogue = [AudioVideo(frames(4), None, SR, fps=24)]
    song_shots = [AudioVideo(frames(4), None, SR, fps=24)]

    with pytest.raises(ValueError, match="given fps 12"):
        join_into_song(dialogue, song_shots, song_track(1000, SR), fps=12)


def test_an_fps_matching_the_carried_rate_is_accepted():
    dialogue = [AudioVideo(frames(4), None, SR, fps=8)]
    song_shots = [AudioVideo(frames(4), None, SR, fps=8)]

    result = join_into_song(dialogue, song_shots, song_track(8000, SR), fps=8)

    assert result.fps == 8


def test_refuses_when_no_fps_is_available_anywhere():
    with pytest.raises(ValueError, match="needs 'fps'"):
        join_into_song([frames(4)], [frames(4)], song_track(1000, SR))


def test_refuses_mismatched_frame_sizes():
    dialogue = [frames(4, size=(16, 16))]
    song_shots = [frames(4, size=(8, 8))]

    with pytest.raises(ValueError, match=r"\d+x\d+"):
        join_into_song(dialogue, song_shots, song_track(1000, SR), fps=4)


def test_refuses_negative_cue_seconds():
    with pytest.raises(ValueError, match="cue_seconds"):
        join_into_song([], [], None, cue_seconds=-1)


def test_refuses_negative_duck_delay_ms():
    with pytest.raises(ValueError, match="duck_delay_ms"):
        join_into_song([], [], None, duck_delay_ms=-1)


def test_refuses_negative_duck_ramp_ms():
    with pytest.raises(ValueError, match="duck_ramp_ms"):
        join_into_song([], [], None, duck_ramp_ms=-1)


def test_refuses_positive_duck_db():
    with pytest.raises(ValueError, match="duck_db"):
        join_into_song([], [], None, duck_db=1)


def test_refuses_fps_zero():
    with pytest.raises(ValueError, match="fps"):
        join_into_song([], [], None, fps=0)


# --- 11. Command registration and shot_references -------------------------------


def test_is_registered_as_a_task_command():
    assert "join_into_song" in _COMMAND_REGISTRY


def test_shot_references_reports_dialogue_then_song_shots():
    dialogue, song_shots = ["a", "b"], ["c"]
    assert shot_references({"dialogue": dialogue, "song_shots": song_shots}) == [
        "a",
        "b",
        "c",
    ]


def test_shot_references_reports_videos_for_concat_style_arguments():
    assert shot_references({"videos": ["x", "y"]}) == ["x", "y"]


def test_shot_references_reports_none_for_neither():
    assert shot_references({"other": 1}) is None


# --- 12. Song passed as a file path ---------------------------------------------


def test_song_as_a_file_path(tmp_path):
    import soundfile

    song_samples = 20000
    path = tmp_path / "song.wav"
    soundfile.write(path, numpy.zeros((song_samples, 2), dtype=numpy.float32), SR)
    dialogue = [clip(8, FPS, SR, level=None)]
    song_shots = [frames(8)]

    result = join_into_song(dialogue, song_shots, str(path), fps=FPS)

    assert result.sample_rate == SR
    d = frames_to_samples(8, FPS, SR)
    assert result.audio.shape[1] >= d
