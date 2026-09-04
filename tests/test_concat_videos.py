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


class TestAudioBleed:
    def silent_head(self, num_frames, tail_level, fps=4, sample_rate=100):
        """A shot that opens on silence and ends on sound, the way H3 renders one."""
        samples = int(num_frames / fps * sample_rate)
        audio = numpy.zeros((2, samples), dtype=numpy.float32)
        audio[:, samples // 2 :] = tail_level
        return AudioVideo(frames(num_frames), audio, sample_rate)

    def test_the_tail_fills_the_next_shots_silent_head(self):
        first = self.silent_head(8, 0.5)
        second = self.silent_head(8, 0.5)

        result = concat_videos([first, second], audio_bleed_ms=500, fps=4)

        seam = first.audio.shape[1]
        assert result.audio[0, seam] == pytest.approx(0.5, abs=1e-3)
        assert result.audio[0, seam + 10] > 0.0  # the hole is filled

    def test_the_default_leaves_the_seam_alone(self):
        first = self.silent_head(8, 0.5)
        second = self.silent_head(8, 0.5)

        result = concat_videos([first, second], fps=4)

        seam = first.audio.shape[1]
        assert result.audio[0, seam + 10] == 0.0

    def test_it_does_not_change_the_length(self):
        first = self.silent_head(8, 0.5)
        second = self.silent_head(8, 0.5)

        bled = concat_videos([first, second], audio_bleed_ms=500, fps=4)
        plain = concat_videos([first, second], fps=4)

        assert bled.audio.shape == plain.audio.shape
        assert len(bled.frames) == len(plain.frames)

    def test_trimmed_seams_still_crossfade_instead(self):
        first = audio_video(8, 0.5)
        second = audio_video(8, 0.5)

        result = concat_videos(
            [first, second], trim_frames=2, crossfade_ms=100, audio_bleed_ms=500, fps=4
        )

        # trimmed material means a real crossfade; length loses the trim only
        expected = first.audio.shape[1] + second.audio.shape[1] - 50
        assert result.audio.shape[1] == expected

    def test_it_reaches_every_seam_of_a_longer_cut(self):
        shots = [self.silent_head(8, 0.5) for _ in range(3)]

        result = concat_videos(shots, audio_bleed_ms=500, fps=4)

        for index in (1, 2):
            seam = shots[0].audio.shape[1] * index
            assert result.audio[0, seam + 10] > 0.0

    def test_it_survives_the_task_layer(self):
        first = self.silent_head(8, 0.5)
        second = self.silent_head(8, 0.5)
        task = Task({"command": "concat_videos", "arguments": {}}, "cpu")

        result = task.run({"videos": [first, second], "audio_bleed_ms": 500, "fps": 4})

        seam = first.audio.shape[1]
        assert result.audio[0, seam + 10] > 0.0


class TestSeamFade:
    def test_the_default_seam_is_only_declicked(self):
        first, second = audio_video(8, 0.5), audio_video(8, 0.5)

        result = concat_videos([first, second], fps=4)

        # 3 ms at 100 Hz rounds to no ramp at all - the seam is untouched
        seam = first.audio.shape[1]
        assert result.audio[0, seam - 1] == pytest.approx(0.5)
        assert result.audio[0, seam] == pytest.approx(0.5)

    def test_a_longer_fade_eases_both_sides_of_the_seam(self):
        first, second = audio_video(8, 0.5), audio_video(8, 0.5)

        result = concat_videos([first, second], seam_fade_ms=200, fps=4)

        seam = first.audio.shape[1]
        assert result.audio[0, seam - 1] < 0.1  # faded out into the cut
        assert result.audio[0, seam] < 0.1  # and back in after it
        assert result.audio[0, seam - 25] == pytest.approx(0.5)  # before the ramp
        assert result.audio[0, seam + 25] == pytest.approx(0.5)  # after it

    def test_it_does_not_change_the_length(self):
        first, second = audio_video(8, 0.5), audio_video(8, 0.5)

        faded = concat_videos([first, second], seam_fade_ms=200, fps=4)
        plain = concat_videos([first, second], fps=4)

        assert faded.audio.shape == plain.audio.shape

    def test_a_bleed_takes_precedence_over_the_fade(self):
        first = AudioVideo(frames(8), numpy.full((2, 200), 0.5, numpy.float32), 100)
        second = AudioVideo(frames(8), numpy.zeros((2, 200), numpy.float32), 100)

        result = concat_videos(
            [first, second], audio_bleed_ms=500, seam_fade_ms=200, fps=4
        )

        # the bleed fills the seam, so nothing is faded away
        assert result.audio[0, 200] == pytest.approx(0.5, abs=1e-3)

    def test_the_fade_backs_up_a_bleed_with_no_material(self):
        first = AudioVideo(frames(8), numpy.full((2, 200), 0.5, numpy.float32), 100)
        second = AudioVideo(frames(8), numpy.full((2, 200), 0.5, numpy.float32), 100)

        result = concat_videos(
            [first, second], audio_bleed_ms=0, seam_fade_ms=200, fps=4
        )

        assert result.audio[0, 199] < 0.1


class TestResampleAudioTask:
    def test_resample_audio_runs_through_task(self):
        task = Task({"command": "resample_audio", "arguments": {}}, "cpu")

        result = task.run(
            {
                "audio": numpy.zeros((2, 44100), dtype=numpy.float32),
                "sample_rate": 44100,
                "target_sample_rate": 32000,
            }
        )

        assert result.shape == (32000, 2)
