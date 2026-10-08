"""Tests for the grade task command (#349): CPU-only exposure, contrast,
saturation and white-balance adjustment for an image or a video.
"""

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.task_domains import task_argument_errors
from dw.tasks.grade import grade_image
from dw.tasks.task import Task
from dw.workflow import Workflow


def _swatch(rgb=(128, 96, 160), size=16):
    return Image.new("RGB", (size, size), rgb)


def _ramp(width=256, height=16):
    """A horizontal 0..255 ramp, so every luma level is present."""
    row = numpy.linspace(0, 255, width).round().astype(numpy.uint8)
    array = numpy.repeat(row[None, :, None], height, axis=0).repeat(3, axis=2)
    return Image.fromarray(array, mode="RGB")


def _noise(size=32, seed=7):
    rng = numpy.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=numpy.uint8))


def _checker(size=64, block=8, low=118, high=138):
    """A soft midtone checker around 128: local contrast for clarity to act on."""
    ys, xs = numpy.indices((size, size))
    plane = numpy.where(((ys // block) + (xs // block)) % 2 == 0, low, high)
    return Image.fromarray(numpy.stack([plane] * 3, axis=-1).astype(numpy.uint8))


def _gray(image):
    """The red channel of a grey-ish image as a float array."""
    return numpy.asarray(image, dtype=numpy.float32)[..., 0]


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
            highlights=0.0,
            shadows=0.0,
            whites=0.0,
            blacks=0.0,
            clarity=0.0,
            vignette=0.0,
            fade=0.0,
        )
        assert numpy.array_equal(numpy.asarray(source), numpy.asarray(graded))

    @pytest.mark.parametrize("make", [_ramp, _noise])
    def test_identity_is_exact_on_an_image_with_real_variation(self, make):
        source = make()
        explicit = grade_image(
            source,
            exposure=0.0,
            contrast=1.0,
            saturation=1.0,
            temperature=0.0,
            tint=0.0,
            highlights=0.0,
            shadows=0.0,
            whites=0.0,
            blacks=0.0,
            clarity=0.0,
            vignette=0.0,
            fade=0.0,
        )
        assert numpy.array_equal(numpy.asarray(source), numpy.asarray(explicit))
        assert numpy.array_equal(
            numpy.asarray(source), numpy.asarray(grade_image(source))
        )


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


class TestTonalControlsMoveTheDirectionTheyDocument:
    def _ramp_values(self, **kwargs):
        source = _gray(_ramp())[0]
        graded = _gray(grade_image(_ramp(), **kwargs))[0]
        return source, graded

    def test_negative_highlights_darken_bright_pixels_and_spare_deep_shadows(self):
        source, graded = self._ramp_values(highlights=-0.5)
        bright = source > 200
        assert (graded[bright] < source[bright]).all()
        deep = source < 40
        assert numpy.array_equal(graded[deep], source[deep])

    def test_positive_highlights_brighten_the_highlights(self):
        source, graded = self._ramp_values(highlights=0.5)
        bright = (source > 200) & (source < 250)
        assert (graded[bright] > source[bright]).all()
        assert numpy.array_equal(graded[source < 40], source[source < 40])

    def test_positive_shadows_lift_dark_pixels_and_spare_the_bright_ones(self):
        source, graded = self._ramp_values(shadows=0.5)
        dark = source < 100
        assert (graded[dark] > source[dark]).all()
        bright = source >= 128
        assert numpy.array_equal(graded[bright], source[bright])

    def test_negative_shadows_darken_the_shadows(self):
        source, graded = self._ramp_values(shadows=-0.5)
        dark = (source > 5) & (source < 100)
        assert (graded[dark] < source[dark]).all()
        bright = source >= 128
        assert numpy.array_equal(graded[bright], source[bright])

    @pytest.mark.parametrize("sign", [1, -1])
    def test_whites_move_near_white_more_than_near_black(self, sign):
        source, graded = self._ramp_values(whites=0.5 * sign)
        delta = (graded - source) * sign
        assert delta[240] > 0
        assert delta[240] > abs(delta[15])

    @pytest.mark.parametrize("sign", [1, -1])
    def test_blacks_move_near_black_more_than_near_white(self, sign):
        source, graded = self._ramp_values(blacks=0.5 * sign)
        delta = (graded - source) * sign
        assert delta[15] > 0
        assert delta[15] > abs(delta[240])

    def test_positive_clarity_raises_and_negative_clarity_lowers_local_contrast(self):
        source = _checker()
        base = _gray(source).std()
        assert _gray(grade_image(source, clarity=0.5)).std() > base
        assert _gray(grade_image(source, clarity=-0.5)).std() < base

    def test_positive_vignette_darkens_the_corner_and_spares_the_centre(self):
        source = _swatch((128, 128, 128), size=64)
        graded = grade_image(source, vignette=0.5)
        assert graded.getpixel((0, 0))[0] < 128
        assert graded.getpixel((32, 32)) == (128, 128, 128)

    def test_negative_vignette_lightens_the_corner(self):
        source = _swatch((128, 128, 128), size=64)
        graded = grade_image(source, vignette=-0.5)
        assert graded.getpixel((0, 0))[0] > 128
        assert graded.getpixel((32, 32)) == (128, 128, 128)

    def test_fade_lifts_black_and_keeps_white(self):
        black = grade_image(_swatch((0, 0, 0)), fade=0.5)
        white = grade_image(_swatch((255, 255, 255)), fade=0.5)
        assert black.getpixel((0, 0))[0] > 0
        assert white.getpixel((0, 0)) == (255, 255, 255)


class TestAlphaPassesThrough:
    def test_an_alpha_channel_is_untouched(self):
        source = Image.new("RGBA", (4, 4), (200, 50, 50, 77))
        graded = grade_image(source, exposure=1.0)
        assert graded.mode == "RGBA"
        assert graded.getpixel((0, 0))[3] == 77

    def test_alpha_is_exactly_preserved_by_the_vignette_and_fade(self):
        rgb = numpy.full((16, 16, 3), 140, dtype=numpy.uint8)
        alpha = (numpy.arange(256, dtype=numpy.uint8)).reshape(16, 16)
        source = Image.fromarray(numpy.dstack([rgb, alpha]), mode="RGBA")
        graded = grade_image(source, vignette=0.5, fade=0.3)
        assert graded.mode == "RGBA"
        assert numpy.array_equal(
            numpy.asarray(graded)[..., 3], numpy.asarray(source)[..., 3]
        )


class TestVideoIsGradedPerFrame:
    def video(self):
        frames = [Image.new("RGB", (4, 4), (i * 40, 100, 100)) for i in range(3)]
        return AudioVideo(
            frames, numpy.zeros((2, 50), dtype=numpy.float32), 100, fps=24
        )

    def test_grade_dispatches_over_every_frame_and_keeps_audio_and_fps(self):
        task = Task({"command": "grade", "arguments": {}}, "cpu")
        result = task.run({"media": self.video(), "exposure": 1.0})

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

    def test_new_controls_grade_every_frame_and_keep_the_audio_exactly(self):
        source = self.video()
        task = Task({"command": "grade", "arguments": {}}, "cpu")
        result = task.run({"media": self.video(), "highlights": -0.5, "vignette": 0.5})

        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 3
        assert result.fps == 24
        assert result.sample_rate == 100
        assert numpy.array_equal(result.audio, source.audio)
        assert all(frame.size == (4, 4) for frame in result.frames)

    def test_a_video_file_path_is_loaded_with_its_audio(self, tmp_path):
        # The regression this guards: `media` is deliberately not called
        # "image" or "video" - either name would make the engine's own
        # key-convention loading grab the string first (an image-only
        # loader that refuses a .mp4 extension, or a frame-only loader that
        # silently drops the audio), before the command ever saw it.
        import av

        path = tmp_path / "clip.mp4"
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264rgb", rate=24)
        stream.width, stream.height = 4, 4
        stream.pix_fmt = "rgb24"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(
                numpy.full((4, 4, 3), i * 40, dtype=numpy.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        task = Task({"command": "grade", "arguments": {}}, "cpu")
        result = task.run({"media": str(path), "exposure": 1.0})

        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 3
        assert result.fps == 24


class TestAppliedParametersAreLogged:
    def test_logs_the_non_identity_parameters_applied(self):
        # #392: with save:false a caller can only see what grade did through
        # job events, so the parameters actually applied must reach the log
        from dw.events import RunContext, activate_context, deactivate_context

        task = Task({"command": "grade", "arguments": {}}, "cpu")
        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            task.run({"media": _swatch(), "exposure": 1.0, "saturation": 0.5})
        finally:
            deactivate_context(token)

        logs = [e for e in events if e.get("event") == "log"]
        assert len(logs) == 1
        assert logs[0]["exposure"] == 1.0
        assert logs[0]["saturation"] == 0.5
        assert "contrast" not in logs[0]

    def test_logs_no_adjustment_when_every_parameter_is_identity(self):
        from dw.events import RunContext, activate_context, deactivate_context

        task = Task({"command": "grade", "arguments": {}}, "cpu")
        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            task.run({"media": _swatch()})
        finally:
            deactivate_context(token)

        logs = [e for e in events if e.get("event") == "log"]
        assert len(logs) == 1
        assert "no adjustment" in logs[0]["message"]

    def _log(self, values):
        from dw.events import RunContext, activate_context, deactivate_context

        task = Task({"command": "grade", "arguments": {}}, "cpu")
        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            task.run({"media": _swatch(), **values})
        finally:
            deactivate_context(token)
        logs = [e for e in events if e.get("event") == "log"]
        assert len(logs) == 1
        return logs[0]

    def test_logs_the_new_non_identity_controls_and_not_identity_ones(self):
        log = self._log({"highlights": -0.5, "fade": 0.2})
        assert log["highlights"] == -0.5
        assert log["fade"] == 0.2
        for name in ("exposure", "contrast", "shadows", "vignette", "clarity"):
            assert name not in log

    def test_every_signature_parameter_appears_in_the_log_when_non_identity(self):
        # The log's defaults are read off grade_image's signature, so a
        # parameter added there cannot silently drop out of it
        import inspect

        non_identity = {
            "exposure": 0.25,
            "contrast": 1.25,
            "saturation": 0.75,
            "temperature": 0.25,
            "tint": 0.25,
            "highlights": 0.25,
            "shadows": 0.25,
            "whites": 0.25,
            "blacks": 0.25,
            "clarity": 0.25,
            "vignette": 0.25,
            "fade": 0.25,
        }
        names = set(inspect.signature(grade_image).parameters) - {"media"}
        assert names == set(non_identity)
        log = self._log(non_identity)
        for name, value in non_identity.items():
            assert log[name] == value, name


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
        errors = self._errors({"media": "asset:a.png", "contrast": -0.5})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.contrast"]

    def test_a_negative_saturation_is_refused(self):
        errors = self._errors({"media": "asset:a.png", "saturation": -1.0})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.saturation"]

    def test_zero_saturation_is_a_legitimate_request(self):
        assert self._errors({"media": "asset:a.png", "saturation": 0.0}) == []

    def test_an_out_of_range_temperature_is_refused(self):
        errors = self._errors({"media": "asset:a.png", "temperature": 5.0})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.temperature"]

    def test_an_out_of_range_tint_is_refused(self):
        errors = self._errors({"media": "asset:a.png", "tint": -1.5})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.tint"]

    def test_validate_workflow_refuses_it_before_a_run(self):
        workflow = Workflow(
            {
                "id": "grade-domains",
                "steps": [
                    {
                        "name": "grade",
                        "task": {
                            "command": "grade",
                            "arguments": {"media": "asset:a.png", "contrast": -1.0},
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

    @pytest.mark.parametrize(
        "name,value,text",
        [
            ("highlights", 2.0, "between -1.0 and 1.0"),
            ("shadows", 2.0, "between -1.0 and 1.0"),
            ("whites", 2.0, "between -1.0 and 1.0"),
            ("blacks", 2.0, "between -1.0 and 1.0"),
            ("clarity", 2.0, "between -1.0 and 1.0"),
            ("vignette", 2.0, "between -1.0 and 1.0"),
            ("fade", 2.0, "between 0.0 and 1.0"),
            ("fade", -0.1, "between 0.0 and 1.0"),
        ],
    )
    def test_each_new_control_refuses_a_value_outside_its_range(
        self, name, value, text
    ):
        errors = self._errors({"media": "asset:a.png", name: value})
        assert [e["path"] for e in errors] == [f"steps[0].task.arguments.{name}"]
        assert name in errors[0]["message"]
        assert text in errors[0]["message"]

    def test_run_time_refuses_an_out_of_domain_value_from_a_reference(self):
        task = Task({"command": "grade", "arguments": {}}, "cpu")
        with pytest.raises(ValueError, match="shadows"):
            task.run({"media": _swatch(), "shadows": 2})

    def test_all_twelve_adjustments_are_described_with_domain_and_range(self):
        from dw.introspection import describe_task
        from dw.task_domains import CLOSED_UNIT, UNIT

        parameters = {p["name"]: p for p in describe_task("grade")["parameters"]}
        for name in (
            "exposure",
            "contrast",
            "saturation",
            "temperature",
            "tint",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "clarity",
            "vignette",
            "fade",
        ):
            assert name in parameters
        for name in (
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "clarity",
            "vignette",
        ):
            assert parameters[name]["domain"] == CLOSED_UNIT
            assert parameters[name]["range"] == "between -1.0 and 1.0"
        assert parameters["fade"]["domain"] == UNIT
        assert parameters["fade"]["range"] == "between 0.0 and 1.0"

    def test_exposure_is_any_finite_number_temperature_and_tint_are_closed_unit(self):
        from dw.introspection import describe_task
        from dw.task_domains import CLOSED_UNIT, FINITE

        parameters = {p["name"]: p for p in describe_task("grade")["parameters"]}
        assert parameters["exposure"]["domain"] == FINITE
        assert parameters["temperature"]["domain"] == CLOSED_UNIT
        assert parameters["tint"]["domain"] == CLOSED_UNIT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
