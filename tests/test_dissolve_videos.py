"""
Unit tests for the dissolve_videos task - cross-dissolved seams and the
fades that open and close a piece.
"""

import numpy
import pytest
from PIL import Image

from dw.result import AudioVideo
from dw.tasks.dissolve_videos import dissolve_videos


def frames(count, level):
    return [Image.new("RGB", (4, 4), (level, level, level)) for _ in range(count)]


def levels(result):
    return [numpy.asarray(frame)[0, 0, 0] for frame in result.frames]


def audio_video(num_frames, level, tone, fps=4, sample_rate=100):
    samples = int(num_frames / fps * sample_rate)
    audio = numpy.full((2, samples), float(tone), dtype=numpy.float32)
    return AudioVideo(frames(num_frames, level), audio, sample_rate)


class TestDissolveVideos:
    def test_each_seam_shortens_the_join_by_one_overlap(self):
        result = dissolve_videos([frames(10, 0), frames(10, 0), frames(10, 0)], 3)

        assert len(result.frames) == 30 - 2 * 3

    def test_the_overlap_ramps_from_the_outgoing_to_the_incoming_picture(self):
        result = dissolve_videos([frames(6, 0), frames(6, 200)], dissolve_frames=4)

        seam = levels(result)[2:6]
        assert seam == [40, 80, 120, 160]
        assert levels(result)[:2] == [0, 0]
        assert levels(result)[6:] == [200, 200]

    def test_zero_overlap_is_a_cut(self):
        result = dissolve_videos([frames(3, 0), frames(3, 200)], dissolve_frames=0)

        assert levels(result) == [0, 0, 0, 200, 200, 200]

    def test_fades_open_and_close_on_the_fade_colour(self):
        result = dissolve_videos(
            [frames(8, 200)], 0, fade_in_frames=3, fade_out_frames=3
        )

        assert levels(result) == [50, 100, 150, 200, 200, 150, 100, 50]

    def test_a_fade_colour_other_than_black(self):
        result = dissolve_videos(
            [frames(3, 0)], 0, fade_in_frames=1, fade_color=(255, 255, 255)
        )

        assert levels(result)[0] == 128

    def test_a_video_too_short_for_its_seams_is_refused(self):
        with pytest.raises(ValueError, match="too few"):
            dissolve_videos([frames(10, 0), frames(5, 0), frames(10, 0)], 3)

    def test_negative_counts_are_refused(self):
        with pytest.raises(ValueError, match="negative"):
            dissolve_videos([frames(4, 0)], dissolve_frames=-1)

    def test_an_empty_list_is_refused(self):
        with pytest.raises(ValueError):
            dissolve_videos([])

    def test_audio_is_crossfaded_over_the_seam_span(self):
        first, second = audio_video(8, 0, 1.0), audio_video(8, 0, 1.0)

        result = dissolve_videos([first, second], dissolve_frames=4, fps=4)

        # 16 frames - 4 overlap = 12 frames at 4 fps = 3 s = 300 samples
        assert len(result.frames) == 12
        assert result.audio.shape == (2, 300)
        assert result.sample_rate == 100

    def test_audio_needs_fps_at_a_dissolve(self):
        with pytest.raises(ValueError, match="fps"):
            dissolve_videos([audio_video(8, 0, 1.0), audio_video(8, 0, 1.0)], 2)

    def test_a_silent_input_leaves_the_result_silent(self):
        result = dissolve_videos([audio_video(8, 0, 1.0), frames(8, 0)], 2, fps=4)

        assert result.audio is None

    def test_mismatched_sample_rates_are_refused(self):
        first = audio_video(8, 0, 1.0, sample_rate=100)
        second = audio_video(8, 0, 1.0, sample_rate=200)

        with pytest.raises(ValueError, match="sample rate"):
            dissolve_videos([first, second], 2, fps=4)
