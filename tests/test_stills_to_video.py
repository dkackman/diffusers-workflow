"""stills_to_video: storyboard stills held for a duration into a silent clip (#1144)."""

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.tasks.video_utils import MAX_STILLS_FPS, MAX_STILLS_SECONDS, stills_to_video


def still(width=512, height=512, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color)


class TestHold:
    def test_three_stills_hold_for_their_durations(self):
        video = stills_to_video([still(), still(), still()], [1, 2, 0.5], fps=24)
        assert isinstance(video, AudioVideo)
        assert video.frames.shape == (84, 512, 512, 3)
        assert video.frames.dtype == numpy.uint8
        assert video.fps == 24
        assert [shot["start_frame"] for shot in video.shots] == [0, 24, 72]
        assert [shot["num_frames"] for shot in video.shots] == [24, 48, 12]

    def test_each_still_fills_its_own_frames(self):
        video = stills_to_video(
            [still(color=(255, 0, 0)), still(color=(0, 0, 255))], [1, 1], fps=2
        )
        assert tuple(video.frames[0, 0, 0]) == (255, 0, 0)
        assert tuple(video.frames[3, 0, 0]) == (0, 0, 255)

    def test_one_duration_applies_to_all(self):
        video = stills_to_video([still(), still()], 2, fps=12)
        assert len(video.frames) == 48

    def test_a_sub_frame_duration_still_gets_one_frame(self):
        video = stills_to_video([still()], [0.01], fps=24)
        assert len(video.frames) == 1

    def test_arrays_are_accepted(self):
        array = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
        video = stills_to_video([array], 1, fps=4)
        assert video.frames.shape == (4, 64, 64, 3)

    def test_silent_clip_carries_no_audio(self):
        video = stills_to_video([still()], 1, fps=4)
        assert video.audio is None
        assert video.sample_rate is None


class TestRefusals:
    def test_mismatched_sizes_refused(self):
        with pytest.raises(ValueError, match=r"512x512.*512x768"):
            stills_to_video([still(512, 512), still(512, 768)], 1)

    def test_wrong_duration_count_refused(self):
        with pytest.raises(ValueError, match="durations"):
            stills_to_video([still(), still(), still()], [1, 2])

    @pytest.mark.parametrize("durations", [[1, 0], [-1], 0])
    def test_non_positive_duration_refused(self, durations):
        with pytest.raises(ValueError, match="durations"):
            stills_to_video(
                [still(), still()][
                    : len(durations) if isinstance(durations, list) else 1
                ],
                durations,
            )

    def test_fps_bounds(self):
        for bad in (0, 121):
            with pytest.raises(ValueError, match="fps"):
                stills_to_video([still()], 1, fps=bad)
        assert MAX_STILLS_FPS == 120
        assert len(stills_to_video([still()], 1, fps=120).frames) == 120

    def test_total_over_600_seconds_refused(self):
        assert MAX_STILLS_SECONDS == 600
        with pytest.raises(ValueError, match="600"):
            stills_to_video([still()], [601], fps=1)

    def test_no_images_refused(self):
        with pytest.raises(ValueError, match="images"):
            stills_to_video([], 1)


class TestWithConcat:
    def test_result_concats_with_concat_videos(self):
        from dw.tasks.concat_videos import concat_videos

        video = stills_to_video([still(64, 64)], 1, fps=4)
        joined = concat_videos([video, video], fps=4)
        assert len(joined.frames) == 2 * len(video.frames)

    def test_silent_clip_saves_as_a_video(self, tmp_path):
        from dw.result import Result

        video = stills_to_video([still(64, 64)], 1, fps=4)
        result = Result({"content_type": "video/mp4"})
        result.add_result([video])
        result.save(str(tmp_path), "silent")
        files = list(tmp_path.glob("silent*.mp4"))
        assert files and files[0].stat().st_size > 0


class TestRegistered:
    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "stills_to_video" in _COMMAND_REGISTRY


class TestStaticValidation:
    def _errors(self, **arguments):
        from dw.task_problems import stills_to_video_errors

        return stills_to_video_errors(arguments)

    def test_literal_bad_duration_found(self):
        assert [a for a, _ in self._errors(images=["a"], durations=[1, 0])] == [
            "durations"
        ]

    def test_fps_out_of_range_found(self):
        assert [a for a, _ in self._errors(durations=1, fps=121)] == ["fps"]

    def test_count_mismatch_found(self):
        errors = self._errors(images=["a", "b", "c"], durations=[1, 2])
        assert [a for a, _ in errors] == ["durations"]

    def test_total_found(self):
        assert self._errors(durations=[601])

    def test_references_are_left_alone(self):
        assert not self._errors(images="previous_result:x", durations="variable:d")

    def test_clean_arguments_pass(self):
        assert not self._errors(images=["a", "b"], durations=[1, 2], fps=24)


class TestHandlerLoadsLocations:
    def test_handler_reads_image_dicts(self, tmp_path):
        from dw.tasks.task import _COMMAND_REGISTRY

        path = tmp_path / "a.png"
        still(32, 32).save(path)
        handler = _COMMAND_REGISTRY["stills_to_video"]
        video = handler(
            None,
            {
                "images": [{"media_type": "image", "location": str(path)}],
                "durations": 1,
                "fps": 4,
            },
            {},
        )
        assert video.frames.shape == (4, 32, 32, 3)
