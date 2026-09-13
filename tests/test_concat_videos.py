"""
Unit tests for the concat_videos task - the standalone counterpart of the
chained pipeline step's stitching.
"""

from unittest.mock import patch

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

    def test_mismatched_sample_rates_are_resampled_to_the_highest(self, caplog):
        """#108: a 24 kHz voice clip paired onto a 32 kHz generation used to
        fail the run mid-way, after the earlier steps had already written
        their files, with an error that named neither which shot to fix nor
        the `resample_audio` task that was the remedy. The difference has no
        editorial meaning - unlike a level jump - so it is converted."""
        videos = [audio_video(8, 1), audio_video(8, 2, sample_rate=200)]

        with caplog.at_level("WARNING"):
            result = concat_videos(videos)

        assert result.sample_rate == 200
        # both halves at the joined rate: two 2 s videos at 200 Hz. The
        # first was resampled up from 100, so its 200 samples became 400
        assert result.audio.shape[1] == pytest.approx(800, abs=4)
        assert "resampling them all to 200 Hz" in caplog.text

    def test_the_resample_warning_names_which_video(self, caplog):
        """ "24000 then 32000" does not say which entry of a six-shot list to
        look at."""
        videos = ["first.mp4", audio_video(8, 2, sample_rate=200)]

        with caplog.at_level("WARNING"):
            with patch(
                "dw.tasks.concat_videos.load_audio_video",
                return_value=audio_video(8, 1),
            ):
                concat_videos(videos)

        assert "first.mp4: 100 Hz" in caplog.text
        assert "video 2: 200 Hz" in caplog.text
        assert "resample_audio" in caplog.text

    def test_an_explicit_sample_rate_pins_the_target(self):
        videos = [audio_video(8, 1), audio_video(8, 2, sample_rate=200)]

        result = concat_videos(videos, sample_rate=100)

        assert result.sample_rate == 100

    def test_matching_rates_are_left_alone(self, caplog):
        videos = [audio_video(8, 1), audio_video(8, 2)]

        with caplog.at_level("WARNING"):
            result = concat_videos(videos)

        assert result.sample_rate == 100
        assert "resampling" not in caplog.text

    def test_trimmed_audio_without_fps_raises(self):
        videos = [audio_video(8, 1), audio_video(8, 2)]

        with pytest.raises(ValueError, match="fps"):
            concat_videos(videos, trim_frames=2)

    def test_an_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            concat_videos([])

    def test_mismatched_frame_sizes_raise(self):
        # Shots of different sizes were once "normalized" by a stabilize pass
        # that actually rescaled every frame; refusing them names the real
        # problem instead of quietly producing a film that jumps size mid-cut
        wide = [Image.new("RGB", (16, 8)) for _ in range(3)]

        with pytest.raises(ValueError, match="8x8.*16x8"):
            concat_videos([frames(3), wide])

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

        assert result.audio.shape == (2, 200)
        assert result.sample_rate == 100

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

        assert result.audio.shape == (2, 390)


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

        assert result.audio.shape == (2, 32000)
        assert result.sample_rate == 32000


class TestVideoFiles:
    """A shot an earlier run already wrote, named by path rather than passed in."""

    def write_shot(self, path, level, num_frames=4, fps=4, sample_rate=8000):
        from diffusers.utils.export_utils import encode_video

        encode_video(
            frames(num_frames, color=(level, 0, 0)),
            fps=fps,
            output_path=str(path),
            audio=torch.full(
                (2, int(num_frames / fps * sample_rate)),
                level / 255,
                dtype=torch.float32,
            ),
            audio_sample_rate=sample_rate,
        )
        return str(path)

    def test_a_path_is_loaded_with_its_audio(self, tmp_path):
        first = self.write_shot(tmp_path / "shot_1.mp4", 40)
        second = self.write_shot(tmp_path / "shot_2.mp4", 200)

        result = concat_videos([first, second], fps=4)

        assert len(result.frames) == 8
        assert result.sample_rate == 8000
        assert result.audio.shape == (2, 8 / 4 * 8000)

    def test_a_path_joins_a_video_passed_in_directly(self, tmp_path):
        path = self.write_shot(tmp_path / "shot.mp4", 40)

        result = concat_videos(
            [path, audio_video(4, 1, fps=4, sample_rate=8000)], fps=4
        )

        assert len(result.frames) == 8
        assert result.audio.shape[1] == 8 / 4 * 8000


class TestLevelMatching:
    """Independently generated shots land at whatever level the model chose,
    and cutting a -2.6 dBFS shot against a -12.6 dBFS one is audible as a
    drop no fade control can hide - see issue #82."""

    def test_off_by_default(self):
        loud, quiet = audio_video(4, 0.5), audio_video(4, 0.05)

        result = concat_videos([loud, quiet])

        assert result.audio[0][0] == pytest.approx(0.5)
        assert result.audio[0][-1] == pytest.approx(0.05)

    @pytest.mark.parametrize("measure", ["rms", "peak"])
    def test_matching_brings_the_shots_to_one_level(self, measure):
        loud, quiet = audio_video(4, 0.5), audio_video(4, 0.05)

        result = concat_videos([loud, quiet], match_levels=measure)

        # A steady tone's peak and rms are the same figure, so either
        # measure lands both shots on the same sample value
        head, tail = abs(result.audio[0][0]), abs(result.audio[0][-1])
        assert head == pytest.approx(tail, rel=1e-3)

    def test_the_target_level_is_the_callers_to_set(self):
        result = concat_videos(
            [audio_video(4, 0.5), audio_video(4, 0.05)],
            match_levels="peak",
            match_levels_dbfs=-6.0,
        )

        assert abs(result.audio[0][0]) == pytest.approx(10 ** (-6.0 / 20), rel=1e-3)

    def test_a_gain_that_would_clip_is_held_below_full_scale(self, caplog):
        # A track whose peak is far above its rms - matching the rms would
        # ask for a gain that puts the peak past 0 dBFS
        spiky = numpy.full((2, 100), 0.02, dtype=numpy.float32)
        spiky[:, 50] = 0.9
        video = AudioVideo(frames(4), spiky, 100)

        result = concat_videos([video, audio_video(4, 0.02)], match_levels="rms")

        assert float(numpy.abs(result.audio).max()) <= 1.0
        assert "held to" in caplog.text

    def test_a_silent_shot_is_left_alone(self):
        silent = AudioVideo(frames(4), numpy.zeros((2, 100), dtype=numpy.float32), 100)

        result = concat_videos([silent, audio_video(4, 0.5)], match_levels="rms")

        assert float(numpy.abs(result.audio[:, :100]).max()) == 0.0

    def test_a_shot_with_no_soundtrack_joins_as_before(self):
        result = concat_videos([frames(4), audio_video(4, 0.5)], match_levels="rms")

        assert result.audio is not None
        assert len(result.frames) == 8

    def test_an_unknown_measure_is_refused(self):
        with pytest.raises(ValueError, match="match_levels"):
            concat_videos([audio_video(4, 0.5)] * 2, match_levels="loudness")

    def test_a_target_above_full_scale_is_refused(self):
        with pytest.raises(ValueError, match="full scale"):
            concat_videos(
                [audio_video(4, 0.5)] * 2, match_levels="peak", match_levels_dbfs=3.0
            )

    def test_an_unmatched_join_warns_when_the_shots_are_levels_apart(self, caplog):
        concat_videos([audio_video(4, 0.5), audio_video(4, 0.05)])

        assert "level jump" in caplog.text
        assert "match_levels" in caplog.text

    def test_shots_already_at_one_level_draw_no_warning(self, caplog):
        concat_videos([audio_video(4, 0.5), audio_video(4, 0.45)])

        assert "level jump" not in caplog.text


class TestWarningsReachTheCaller:
    """A warning that only reaches the server's log does not exist from
    outside it - see issue #82, where match_levels verified but the spread
    warning was invisible over MCP."""

    def test_the_level_spread_warning_is_emitted_as_an_event(self):
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            concat_videos([audio_video(4, 0.5), audio_video(4, 0.05)])
        finally:
            deactivate_context(token)

        warnings = [e for e in events if e["event"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["kind"] == "level_spread"
        assert warnings[0]["command"] == "concat_videos"
        assert warnings[0]["spread_db"] == pytest.approx(20.0, abs=0.2)
        assert "match_levels" in warnings[0]["message"]

    def test_matched_shots_emit_nothing(self):
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            concat_videos(
                [audio_video(4, 0.5), audio_video(4, 0.05)], match_levels="rms"
            )
        finally:
            deactivate_context(token)

        assert [e for e in events if e["event"] == "warning"] == []


class TestFrameRateTravelsWithTheJoin:
    """result.fps defaults to 8, so a 24 fps cut that says nothing there
    used to be written three times slow against audio of the right length -
    see issue #84."""

    def test_the_tasks_fps_is_carried_to_the_result(self):
        result = concat_videos([frames(4), frames(4)], fps=24)

        assert result.fps == 24

    def test_an_input_videos_rate_is_carried_when_the_task_is_told_nothing(self):
        first = AudioVideo(frames(4), None, None, fps=30)

        result = concat_videos([first, frames(4)])

        assert result.fps == 30

    def test_the_tasks_own_fps_wins_over_its_inputs(self):
        first = AudioVideo(frames(4), None, None, fps=30)

        result = concat_videos([first, frames(4)], fps=24)

        assert result.fps == 24

    def test_nothing_is_carried_when_nothing_knows(self):
        assert concat_videos([frames(4), frames(4)]).fps is None
