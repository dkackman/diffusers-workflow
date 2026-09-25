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

    def test_mismatched_frame_sizes_are_refused(self):
        wide = [Image.new("RGB", (16, 8)) for _ in range(3)]

        with pytest.raises(ValueError, match="4x4.*16x8"):
            dissolve_videos([frames(3, 0), wide], 0)

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

    def test_mismatched_sample_rates_are_resampled_to_the_highest(self, caplog):
        """#287: parity with concat_videos (#108) - a rate mismatch between
        shots has no editorial meaning, so it is converted rather than
        failing the run mid-way with no remedy named."""
        first = audio_video(8, 0, 1.0, sample_rate=100)
        second = audio_video(8, 0, 1.0, sample_rate=200)

        result = dissolve_videos([first, second], 2, fps=4)

        assert result.sample_rate == 200

    def test_an_explicit_sample_rate_pins_the_target(self):
        first = audio_video(8, 0, 1.0, sample_rate=100)
        second = audio_video(8, 0, 1.0, sample_rate=200)

        result = dissolve_videos([first, second], 2, fps=4, sample_rate=100)

        assert result.sample_rate == 100

    def test_the_resample_warning_is_emitted_as_an_event(self):
        from dw.events import RunContext, activate_context, deactivate_context

        first = audio_video(8, 0, 1.0, sample_rate=100)
        second = audio_video(8, 0, 1.0, sample_rate=200)

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            dissolve_videos([first, second], 2, fps=4)
        finally:
            deactivate_context(token)

        warnings = [e for e in events if e.get("kind") == "sample_rate_mismatch"]
        assert len(warnings) == 1
        assert warnings[0]["command"] == "dissolve_videos"
        assert warnings[0]["sample_rate"] == 200
        assert warnings[0]["sample_rates"] == {"video 1": 100, "video 2": 200}

    def test_agreeing_rates_resampled_to_a_pinned_rate_draw_no_warning(self):
        """#453: every input at 32 kHz and the template pinning 44.1 kHz
        warned that the videos 'carry audio at different sample rates' -
        they didn't, and converting to the rate asked for is not a decision
        made on the caller's behalf."""
        from dw.events import RunContext, activate_context, deactivate_context

        first = audio_video(8, 0, 1.0, sample_rate=100)
        second = audio_video(8, 0, 1.0, sample_rate=100)

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            result = dissolve_videos([first, second], 2, fps=4, sample_rate=200)
        finally:
            deactivate_context(token)

        assert result.sample_rate == 200
        assert [e for e in events if e.get("kind") == "sample_rate_mismatch"] == []
        assert any(e["event"] == "log" and "200 Hz" in e["message"] for e in events)

    def test_matching_rates_are_left_alone(self, caplog):
        first = audio_video(8, 0, 1.0, sample_rate=100)
        second = audio_video(8, 0, 1.0, sample_rate=100)

        with caplog.at_level("WARNING"):
            result = dissolve_videos([first, second], 2, fps=4)

        assert result.sample_rate == 100
        assert "resampling" not in caplog.text


class TestLevelMatching:
    """The same pair concat_videos carries (#82) - a dissolve joins
    independently generated shots too, and a cross-dissolve between two
    loudnesses is a swell, not a match."""

    def test_matching_brings_the_shots_to_one_level(self):
        result = dissolve_videos(
            [audio_video(8, 0, 0.5), audio_video(8, 0, 0.05)],
            dissolve_frames=0,
            fps=4,
            match_levels="peak",
        )

        assert abs(result.audio[0][0]) == pytest.approx(abs(result.audio[0][-1]))

    def test_off_by_default_but_a_wide_spread_warns(self, caplog):
        result = dissolve_videos(
            [audio_video(8, 0, 0.5), audio_video(8, 0, 0.05)],
            dissolve_frames=0,
            fps=4,
        )

        assert abs(result.audio[0][-1]) == pytest.approx(0.05)
        assert "level jump" in caplog.text


class TestJoinedAudioFitsTheFrameGrid:
    """#435: the dissolved track was only ever as long as its crossfaded
    inputs measured, with nothing reconciling it against the frame count -
    so an input already short of its own frame grid (an ltx2/keyframes clip,
    in the reported repro) carried its shortfall into the join, and a
    further join built on that output compounded it. See #428 for the same
    remedy on pair_audio's single-track case."""

    def test_a_short_input_is_padded_to_the_frame_grid(self, caplog):
        from dw.tasks.audio_utils import frames_to_samples

        short = AudioVideo(
            frames(8, 0), numpy.full((2, 170), 0.5, dtype=numpy.float32), 100
        )

        result = dissolve_videos(
            [short, audio_video(8, 0, 0.5)], dissolve_frames=0, fps=4
        )

        expected = frames_to_samples(16, 4, 100)
        assert result.audio.shape[1] == expected
        assert "padded" in caplog.text

    def test_the_shot_map_lands_exactly_on_the_frame_grid_after_padding(self):
        from dw.tasks.audio_utils import frames_to_samples

        short = AudioVideo(
            frames(8, 0), numpy.full((2, 190), 0.5, dtype=numpy.float32), 100
        )

        result = dissolve_videos(
            [short, audio_video(8, 0, 0.5)], dissolve_frames=0, fps=4
        )

        expected = frames_to_samples(16, 4, 100)
        assert result.shots[-1]["start_sample"] + result.shots[-1]["num_samples"] == (
            expected
        )

    def test_padding_is_emitted_as_a_warning_event(self):
        from dw.events import RunContext, activate_context, deactivate_context

        short = AudioVideo(
            frames(8, 0), numpy.full((2, 170), 0.5, dtype=numpy.float32), 100
        )

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            dissolve_videos([short, audio_video(8, 0, 0.5)], dissolve_frames=0, fps=4)
        finally:
            deactivate_context(token)

        warnings = [
            e for e in events if e.get("kind") == "joined_audio_padded_to_frames"
        ]
        assert len(warnings) == 1
        assert warnings[0]["command"] == "dissolve_videos"
        assert warnings[0]["pad_samples"] == 30

    def test_a_track_already_on_the_grid_draws_no_warning(self, caplog):
        result = dissolve_videos(
            [audio_video(8, 0, 0.5), audio_video(8, 0, 0.5)],
            dissolve_frames=0,
            fps=4,
        )

        assert "padded" not in caplog.text
        assert "trimmed" not in caplog.text
        assert result.audio.shape[1] == 400


class TestFrameRateTravelsWithTheDissolve:
    """As with concat_videos - the rate the step was told is the rate the
    file is written at, rather than result.fps's default of 8 (#84)."""

    def test_the_tasks_fps_is_carried_to_the_result(self):
        result = dissolve_videos(
            [frames(8, 0), frames(8, 255)], dissolve_frames=2, fps=24
        )

        assert result.fps == 24
