"""
Unit tests for the concat_videos task - the standalone counterpart of the
chained pipeline step's stitching.
"""

import numpy
import pytest
import torch
from PIL import Image

from dw.result import AudioVideo, get_artifact_list
from dw.tasks.concat_videos import concat_videos
from dw.tasks.task import Task


def frames(count, color=(0, 0, 0)):
    return [Image.new("RGB", (8, 8), color) for _ in range(count)]


def audio_video(num_frames, level, fps=4, sample_rate=100):
    samples = int(num_frames / fps * sample_rate)
    audio = numpy.full((2, samples), float(level), dtype=numpy.float32)
    return AudioVideo(frames(num_frames), audio, sample_rate)


class TestConcatVideos:
    def test_joins_plain_frame_lists(self):
        first, second = frames(4), frames(6)

        result = concat_videos([first, second])

        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 10
        assert result.audio is None

    def test_trims_the_head_of_every_later_video(self):
        result = concat_videos([frames(4), frames(4), frames(4)], trim_frames=1)

        assert len(result.frames) == 4 + 3 + 3

    def test_frames_are_carried_by_identity(self):
        first, second = frames(2), frames(2)

        result = concat_videos([first, second])

        assert result.frames[0] is first[0]
        assert result.frames[2] is second[0]

    def test_joins_audio_in_step_with_the_trimmed_video(self):
        videos = [audio_video(8, 1), audio_video(8, 2)]

        result = concat_videos(videos, trim_frames=2, fps=4)

        assert len(result.frames) == 14
        assert result.audio.shape == (2, int(14 / 4 * 100))
        assert result.sample_rate == 100

    def test_untrimmed_audio_concatenates_whole(self):
        videos = [audio_video(8, 1), audio_video(8, 2)]

        result = concat_videos(videos)

        assert result.audio.shape == (2, 400)

    def test_mixed_inputs_keep_the_audio_that_exists(self):
        videos = [frames(4), audio_video(8, 1)]

        result = concat_videos(videos)

        assert len(result.frames) == 12
        assert result.audio.shape == (2, 200)

    def test_mismatched_sample_rates_raise(self):
        videos = [audio_video(8, 1), audio_video(8, 2, sample_rate=200)]

        with pytest.raises(ValueError, match="different sample rates"):
            concat_videos(videos)

    def test_trimmed_audio_without_fps_raises(self):
        videos = [audio_video(8, 1), audio_video(8, 2)]

        with pytest.raises(ValueError, match="fps"):
            concat_videos(videos, trim_frames=2)

    def test_an_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            concat_videos([])

    def test_the_result_is_a_single_artifact(self):
        # add_result flattens lists - the joined video must never be one
        result = concat_videos([frames(3), frames(3)])

        assert len(get_artifact_list(result)) == 1


class TestTaskDispatch:
    def test_concat_videos_runs_through_task(self):
        task = Task({"command": "concat_videos", "arguments": {}}, "cpu")

        result = task.run({"videos": [frames(2), frames(2)]})

        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 4

    def test_slice_audio_runs_through_task(self):
        task = Task({"command": "slice_audio", "arguments": {}}, "cpu")

        result = task.run(
            {
                "audio": numpy.ones((2, 400), dtype=numpy.float32),
                "sample_rate": 100,
                "start_seconds": 1,
                "duration_seconds": 2,
            }
        )

        assert result.shape == (200, 2)

    def test_crossfade_audio_runs_through_task(self):
        task = Task({"command": "crossfade_audio", "arguments": {}}, "cpu")

        result = task.run(
            {
                "audios": [
                    numpy.ones((2, 200), dtype=numpy.float32),
                    numpy.ones((2, 200), dtype=numpy.float32),
                ],
                "sample_rate": 100,
                "crossfade_ms": 100,  # 10 samples of overlap
            }
        )

        assert result.shape == (390, 2)
