#!/usr/bin/env python3
"""Tests for pairing a video with a soundtrack generated beside it.

A step that works on frames alone - a latent upsampler, an interpolator - drops
the audio a video-with-audio pipeline generated. pair_audio carries it across.
"""

import os
import sys

import numpy
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.media_types import AudioVideo
from dw.tasks.pair_audio import pair_audio


def _waveform(samples=100, channels=2):
    return numpy.zeros((channels, samples), dtype=numpy.float32)


@pytest.fixture
def warnings_emitted():
    """The messages of the warning events a call emits - what a consumer over
    the API or MCP actually reads, rather than the server's log (#82)."""
    from dw.events import RunContext, activate_context, deactivate_context

    messages = []

    def record(event):
        if event["event"] == "warning":
            messages.append(event["message"])

    token = activate_context(RunContext(on_event=record))
    try:
        yield messages
    finally:
        deactivate_context(token)


def test_pairs_frames_with_a_soundtrack_read_from_a_file(tmp_path):
    import soundfile

    path = tmp_path / "score.wav"
    soundfile.write(path, numpy.zeros((100, 2), dtype=numpy.float32), 16000)

    paired = pair_audio(["frame1"], str(path))

    assert paired.audio.shape == (2, 100)
    assert paired.sample_rate == 16000


def test_takes_the_soundtrack_of_an_mp4_file(tmp_path):
    """#548: templates/ltx2/upscale-clip names the caller's own clip as the
    'audio', so the deliverable carries the original track rather than one
    the IC-LoRA pass invented. The track is read out of the mp4 itself."""
    from diffusers.utils.export_utils import encode_video

    path = tmp_path / "clip.mp4"
    sample_rate = 16000
    encode_video(
        [Image.new("RGB", (16, 16)) for _ in range(8)],
        fps=4,
        output_path=str(path),
        audio=torch.full((2, 2 * sample_rate), 0.25),
        audio_sample_rate=sample_rate,
    )

    paired = pair_audio(
        [Image.new("RGB", (32, 32)) for _ in range(4)], str(path), fps=4, fit="video"
    )

    assert paired.sample_rate == sample_rate
    assert paired.audio.shape == (2, sample_rate)
    assert numpy.abs(paired.audio).max() > 0.1


def test_an_mp4_without_a_soundtrack_is_named_as_the_fault(tmp_path):
    """A silent source fails the pairing step with an error that says why."""
    from diffusers.utils.export_utils import encode_video

    path = tmp_path / "silent.mp4"
    encode_video(
        [Image.new("RGB", (16, 16)) for _ in range(4)], fps=4, output_path=str(path)
    )

    with pytest.raises(ValueError, match="carries no audio track"):
        pair_audio(["frame"], str(path), fps=4, fit="video")


def test_pairs_frames_with_a_bare_waveform():
    paired = pair_audio(["frame1", "frame2"], _waveform(), sample_rate=24000)

    assert isinstance(paired, AudioVideo)
    assert paired.frames == ["frame1", "frame2"]
    assert paired.sample_rate == 24000
    assert paired.audio.shape == (2, 100)


def test_takes_the_track_and_its_rate_from_another_result():
    """The common case: 'audio': 'previous_result:base_step'."""
    generated = AudioVideo(["low_res"], _waveform(), 16000)

    paired = pair_audio(["upscaled"], generated)

    assert paired.frames == ["upscaled"]
    assert paired.sample_rate == 16000


def test_explicit_rate_wins_over_the_carried_one():
    generated = AudioVideo(["low_res"], _waveform(), 16000)

    paired = pair_audio(["upscaled"], generated, sample_rate=24000)

    assert paired.sample_rate == 24000


def test_replaces_the_soundtrack_of_a_video_that_has_one():
    original = AudioVideo(["frame"], _waveform(samples=10), 24000)
    replacement = AudioVideo(["other"], _waveform(samples=50), 24000)

    paired = pair_audio(original, replacement)

    assert paired.frames == ["frame"]
    assert paired.audio.shape == (2, 50)


def test_normalizes_a_torch_waveform():
    paired = pair_audio(["frame"], torch.zeros(2, 100), sample_rate=24000)

    assert isinstance(paired.audio, numpy.ndarray)
    assert paired.audio.shape == (2, 100)


def test_frame_arrays_are_carried_through_untouched():
    """A long video must not be copied into PIL images just to be paired."""
    frames = numpy.zeros((8, 16, 16, 3), dtype=numpy.uint8)

    paired = pair_audio(frames, _waveform(), sample_rate=24000)

    assert paired.frames is frames


def test_raises_when_the_named_result_carries_no_audio():
    silent = AudioVideo(["frame"], None, None)

    with pytest.raises(ValueError, match="carries none"):
        pair_audio(["frame"], silent)


def test_raises_when_no_sample_rate_can_be_established():
    with pytest.raises(ValueError, match="sample_rate"):
        pair_audio(["frame"], _waveform())


def test_loads_a_string_video_path_the_same_way_concat_videos_does(tmp_path):
    """#553: 'audio' already loaded a string path via load_audio; 'video' did
    not, so a workflow naming its video by an asset:/output: path (resolved
    to a plain string by the time the task runs) failed rather than loading
    it the way concat_videos loads one of its own inputs."""
    from diffusers.utils.export_utils import encode_video

    path = tmp_path / "shot.mp4"
    encode_video(
        [Image.new("RGB", (16, 16)) for _ in range(4)],
        fps=4,
        output_path=str(path),
    )

    paired = pair_audio(str(path), _waveform(samples=40), sample_rate=24000)

    assert isinstance(paired, AudioVideo)
    assert len(paired.frames) == 4
    assert paired.audio.shape == (2, 40)


def test_loads_a_location_dict_video(tmp_path):
    """The same {"location": ...} idiom concat_videos and get_last_frame accept."""
    from diffusers.utils.export_utils import encode_video

    path = tmp_path / "shot.mp4"
    encode_video(
        [Image.new("RGB", (16, 16)) for _ in range(4)],
        fps=4,
        output_path=str(path),
    )

    paired = pair_audio(
        {"location": str(path)}, _waveform(samples=40), sample_rate=24000
    )

    assert isinstance(paired, AudioVideo)
    assert len(paired.frames) == 4


def test_registered_as_a_task_command():
    from dw.tasks.task import _COMMAND_REGISTRY

    assert "pair_audio" in _COMMAND_REGISTRY


def test_warns_when_a_real_measured_shot_position_is_regridded(warnings_emitted):
    """#563: assemble-and-score's 'film' step pairs the picture from
    concat_videos (real measured shots.start_sample) with an audio track that
    is a pointwise transform of that same audio (resample/fade/mix/normalize)
    rather than a track built shot by shot. remeasured_shots regrids every
    interior shot onto the frame grid regardless, silently discarding a real
    drift a join measured - shot 'b' here sits 31.67 ms off the grid, the
    kind of number an inner concat_videos seam actually reported."""
    fps = 24
    native_rate = 48000
    shots = [
        {"name": "a", "start_frame": 0, "num_frames": 40, "start_sample": 0},
        {
            "name": "b",
            "start_frame": 40,
            "num_frames": 40,
            # On-grid would be round(40 / 24 * 48000) == 80000; this is a
            # real measured position 1520 samples (31.67 ms) off it
            "start_sample": 78480,
        },
        {"name": "c", "start_frame": 80, "num_frames": 40, "start_sample": 160000},
    ]
    video = AudioVideo(
        [f"frame{i}" for i in range(120)],
        _waveform(samples=5 * native_rate),
        native_rate,
        fps=fps,
        shots=shots,
    )
    target_rate = 44100
    track = _waveform(samples=5 * target_rate)

    paired = pair_audio(video, track, sample_rate=target_rate)

    assert paired.shots[1]["name"] == "b"
    assert paired.shots[1]["start_sample"] == round(40 / fps * target_rate)
    regridded = [w for w in warnings_emitted if "off the frame grid" in w]
    assert len(regridded) == 1
    assert "b (-31.67 ms)" in regridded[0]


def test_a_shot_already_on_the_grid_is_not_warned_about(warnings_emitted):
    """The ordinary case - a join's shots already sit where the frame grid
    puts them - is not a false positive."""
    fps = 24
    rate = 44100
    shots = [
        {
            "name": "a",
            "start_frame": 0,
            "num_frames": 40,
            "start_sample": round(0 / fps * rate),
        },
        {
            "name": "b",
            "start_frame": 40,
            "num_frames": 40,
            "start_sample": round(40 / fps * rate),
        },
        {
            "name": "c",
            "start_frame": 80,
            "num_frames": 40,
            "start_sample": round(80 / fps * rate),
        },
    ]
    video = AudioVideo(
        [f"frame{i}" for i in range(120)],
        _waveform(samples=5 * rate),
        rate,
        fps=fps,
        shots=shots,
    )

    pair_audio(video, _waveform(samples=5 * rate), sample_rate=rate)

    assert warnings_emitted == []


def test_a_video_with_no_shots_is_not_warned_about(warnings_emitted):
    pair_audio(["frame1", "frame2"], _waveform(), sample_rate=24000)

    assert warnings_emitted == []


def test_get_task_says_what_the_frame_rate_does():
    """#104's fix is only useful where an author authoring a task step looks,
    and that is `get_task` - which reads the implementation's own docstring."""
    from dw.introspection import describe_task

    described = describe_task("pair_audio")
    video = next(p for p in described["parameters"] if p["name"] == "video")
    assert "result.fps" in video["description"]
    audio = next(p for p in described["parameters"] if p["name"] == "audio")
    assert "mono" in audio["description"]
