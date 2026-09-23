"""Tests for the grade task command (#349): CPU-only exposure, contrast,
saturation and white-balance adjustment for an image or a video.
"""

import numpy
import pytest
from PIL import Image

from dw.result import AudioVideo
from dw.task_domains import task_argument_errors
from dw.tasks.grade import grade_image
from dw.tasks.task import Task
from dw.workflow import Workflow


def _swatch(rgb=(128, 96, 160), size=16):
    return Image.new("RGB", (size, size), rgb)


def _mean_channels(image):
    array = numpy.asarray(image, dtype=numpy.float32)
    return array[..., 0].mean(), array[..., 1].mean(), array[..., 2].mean()


class TestIdentity:
    def test_no_arguments_returns_pixels_unchanged(self):
        source = _swatch()
        graded = grade_image(source)
        assert numpy.array_equal(numpy.asarray(source), numpy.asarray(graded))

    def test_identity_values_return_pixels_unchanged(self):
        source = _swatch()
        graded = grade_image(
            source,
            exposure=0.0,
            contrast=1.0,
            saturation=1.0,
            temperature=0.0,
            tint=0.0,
        )
        array = numpy.abs(
            numpy.asarray(source, dtype=numpy.int16)
            - numpy.asarray(graded, dtype=numpy.int16)
        )
        assert array.max() <= 1  # rounding only


class TestEachParameterMovesTheDirectionItDocuments:
    def test_positive_exposure_brightens(self):
        source = _swatch((100, 100, 100))
        graded = grade_image(source, exposure=1.0)
        assert _mean_channels(graded)[0] > _mean_channels(source)[0]

    def test_negative_exposure_darkens(self):
        source = _swatch((100, 100, 100))
        graded = grade_image(source, exposure=-1.0)
        assert _mean_channels(graded)[0] < _mean_channels(source)[0]

    def test_contrast_above_one_pushes_a_bright_pixel_brighter(self):
        source = _swatch((200, 200, 200))
        graded = grade_image(source, contrast=1.5)
        assert _mean_channels(graded)[0] > _mean_channels(source)[0]

    def test_contrast_below_one_pulls_a_bright_pixel_toward_grey(self):
        source = _swatch((200, 200, 200))
        graded = grade_image(source, contrast=0.5)
        assert _mean_channels(graded)[0] < _mean_channels(source)[0]

    def test_saturation_zero_greys_out_a_colour_swatch(self):
        source = _swatch((200, 50, 50))
        graded = grade_image(source, saturation=0.0)
        r, g, b = _mean_channels(graded)
        assert r == pytest.approx(g, abs=1.0)
        assert g == pytest.approx(b, abs=1.0)

    def test_positive_temperature_moves_red_up_and_blue_down(self):
        source = _swatch((128, 128, 128))
        graded = grade_image(source, temperature=1.0)
        r, _, b = _mean_channels(graded)
        sr, _, sb = _mean_channels(source)
        assert r > sr
        assert b < sb

    def test_negative_temperature_moves_blue_up_and_red_down(self):
        source = _swatch((128, 128, 128))
        graded = grade_image(source, temperature=-1.0)
        r, _, b = _mean_channels(graded)
        sr, _, sb = _mean_channels(source)
        assert r < sr
        assert b > sb

    def test_positive_tint_reduces_green(self):
        source = _swatch((128, 128, 128))
        graded = grade_image(source, tint=1.0)
        assert _mean_channels(graded)[1] < _mean_channels(source)[1]

    def test_negative_tint_increases_green(self):
        source = _swatch((128, 128, 128))
        graded = grade_image(source, tint=-1.0)
        assert _mean_channels(graded)[1] > _mean_channels(source)[1]


class TestAlphaPassesThrough:
    def test_an_alpha_channel_is_untouched(self):
        source = Image.new("RGBA", (4, 4), (200, 50, 50, 77))
        graded = grade_image(source, exposure=1.0)
        assert graded.mode == "RGBA"
        assert graded.getpixel((0, 0))[3] == 77


class TestVideoIsGradedPerFrame:
    def video(self):
        frames = [Image.new("RGB", (4, 4), (i * 40, 100, 100)) for i in range(3)]
        return AudioVideo(
            frames, numpy.zeros((2, 50), dtype=numpy.float32), 100, fps=24
        )

    def test_grade_dispatches_over_every_frame_and_keeps_audio_and_fps(self):
        task = Task({"command": "grade", "arguments": {}}, "cpu")
        result = task.run({"image": self.video(), "exposure": 1.0})

        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 3
        assert result.fps == 24
        assert result.sample_rate == 100
        assert result.audio.shape == (2, 50)
        # each frame was actually graded, not passed through untouched
        for source_frame, graded_frame in zip(self.video().frames, result.frames):
            assert not numpy.array_equal(
                numpy.asarray(source_frame), numpy.asarray(graded_frame)
            )


class TestDomains:
    def _errors(self, arguments):
        definition = {
            "id": "grade-domains",
            "steps": [
                {
                    "name": "grade",
                    "task": {"command": "grade", "arguments": arguments},
                    "result": {"content_type": "image/png"},
                }
            ],
        }
        return task_argument_errors(definition)

    def test_a_negative_contrast_is_refused_at_its_path(self):
        errors = self._errors({"image": "asset:a.png", "contrast": -0.5})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.contrast"]

    def test_a_negative_saturation_is_refused(self):
        errors = self._errors({"image": "asset:a.png", "saturation": -1.0})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.saturation"]

    def test_zero_saturation_is_a_legitimate_request(self):
        assert self._errors({"image": "asset:a.png", "saturation": 0.0}) == []

    def test_validate_workflow_refuses_it_before_a_run(self):
        workflow = Workflow(
            {
                "id": "grade-domains",
                "steps": [
                    {
                        "name": "grade",
                        "task": {
                            "command": "grade",
                            "arguments": {"image": "asset:a.png", "contrast": -1.0},
                        },
                        "result": {"content_type": "image/png"},
                    }
                ],
            },
            "outputs",
            None,
        )
        errors = workflow.validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.contrast"]

    def test_exposure_and_temperature_are_unconstrained(self):
        from dw.introspection import describe_task

        parameters = {p["name"]: p for p in describe_task("grade")["parameters"]}
        assert "domain" not in parameters["exposure"]
        assert "domain" not in parameters["temperature"]
        assert "domain" not in parameters["tint"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
