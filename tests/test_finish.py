"""Tests for the sharpen and film_grain task commands (#603): CPU-only
finishing passes for an image or a video.
"""

import numpy
import pytest
from PIL import Image, ImageFilter

from dw.media_types import AudioVideo
from dw.task_domains import task_argument_errors
from dw.tasks.finish import film_grain, sharpen_image
from dw.tasks.task import Task
from dw.workflow import Workflow


def _array(image):
    return numpy.asarray(image)


def _soft_step(width=48, height=32):
    """A black/white vertical step, blurred horizontally so the edge is soft."""
    row = numpy.where(numpy.arange(width) < width // 2, 0, 255).astype(numpy.uint8)
    array = numpy.repeat(row[None, :, None], height, axis=0).repeat(3, axis=2)
    return Image.fromarray(array, mode="RGB").filter(ImageFilter.GaussianBlur(3))


def _low_contrast(width=48, height=32):
    """A faint step, 120 beside 126: detail far below a large threshold."""
    row = numpy.where(numpy.arange(width) < width // 2, 120, 126).astype(numpy.uint8)
    array = numpy.repeat(row[None, :, None], height, axis=0).repeat(3, axis=2)
    return Image.fromarray(array, mode="RGB")


def _grey(value=128, size=(48, 32)):
    return Image.new("RGB", size, (value, value, value))


def _with_alpha(rgb_image):
    alpha = (numpy.arange(32 * 48) % 256).astype(numpy.uint8).reshape(32, 48)
    rgba = rgb_image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba


def _red(image):
    return numpy.asarray(image, dtype=numpy.float32)[..., 0]


def _video(frames=4, value=128):
    clip = [_grey(value) for _ in range(frames)]
    return AudioVideo(clip, numpy.full((2, 50), 0.25, dtype=numpy.float32), 100, fps=12)


def _run(command, media, workflow_seed=None, **arguments):
    """Run a command through Task; `workflow_seed` is Task's constructor seed,
    a `seed` keyword is the step's own argument."""
    return Task({"command": command, "arguments": {}}, "cpu", seed=workflow_seed).run(
        {"media": media, **arguments}
    )


class TestSharpen:
    def test_amount_zero_returns_the_pixels_exactly(self):
        source = _soft_step()
        assert numpy.array_equal(
            _array(sharpen_image(source, amount=0)), _array(source)
        )

    def test_amount_zero_leaves_rgba_alpha_and_pixels_alone(self):
        source = _with_alpha(_soft_step())
        result = sharpen_image(source, amount=0)
        assert result.mode == "RGBA"
        assert numpy.array_equal(_array(result), _array(source))

    def test_a_positive_amount_pushes_the_two_sides_of_an_edge_apart(self):
        source = _soft_step()
        base = _red(source)[0]
        mid = len(base) // 2
        # a few pixels either side of the centre, where the ramp is steep
        dark, bright = mid - 4, mid + 3
        weak = _red(sharpen_image(source, amount=0.5, radius=2))[0]
        strong = _red(sharpen_image(source, amount=2.0, radius=2))[0]
        assert weak[dark] < base[dark]
        assert weak[bright] > base[bright]
        assert strong[dark] <= weak[dark]
        assert strong[bright] >= weak[bright]
        assert (base[dark] - strong[dark]) + (strong[bright] - base[bright]) > (
            base[dark] - weak[dark]
        ) + (weak[bright] - base[bright])

    def test_a_threshold_above_the_detail_leaves_a_faint_image_unchanged(self):
        source = _low_contrast()
        held = sharpen_image(source, amount=2.0, radius=2, threshold=200)
        assert numpy.array_equal(_array(held), _array(source))

    def test_threshold_zero_does_change_the_faint_image(self):
        source = _low_contrast()
        changed = sharpen_image(source, amount=2.0, radius=2, threshold=0)
        assert not numpy.array_equal(_array(changed), _array(source))

    def test_the_mask_is_pillows_unsharp_mask(self):
        source = _soft_step()
        expected = source.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3)
        )
        result = sharpen_image(source, amount=0.8, radius=1.5, threshold=3)
        assert numpy.array_equal(_array(result), _array(expected))

    def test_a_fractional_threshold_is_rounded_to_whole_levels(self):
        source = _low_contrast()
        rounded = sharpen_image(source, amount=2.0, radius=2, threshold=6.4)
        whole = sharpen_image(source, amount=2.0, radius=2, threshold=6)
        assert numpy.array_equal(_array(rounded), _array(whole))

    def test_alpha_passes_through_untouched_at_a_positive_amount(self):
        source = _with_alpha(_soft_step())
        result = sharpen_image(source, amount=1.5)
        assert result.mode == "RGBA"
        assert numpy.array_equal(_array(result)[..., 3], _array(source)[..., 3])
        assert not numpy.array_equal(_array(result)[..., :3], _array(source)[..., :3])

    def test_a_video_is_sharpened_per_frame_keeping_audio_and_fps(self):
        source = AudioVideo(
            [_soft_step(), _soft_step()],
            numpy.zeros((2, 50), dtype=numpy.float32),
            100,
            fps=24,
        )
        result = _run("sharpen", source, amount=1.0)
        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 2
        assert result.fps == 24
        assert result.sample_rate == 100
        assert result.audio.shape == (2, 50)
        for before, after in zip(source.frames, result.frames):
            assert not numpy.array_equal(_array(before), _array(after))

    def test_the_command_sharpens_an_image(self):
        source = _soft_step()
        result = _run("sharpen", source, amount=1.0)
        assert numpy.array_equal(
            _array(result), _array(sharpen_image(source, amount=1.0))
        )


def _errors(command, arguments):
    definition = {
        "id": f"{command}-domains",
        "steps": [
            {
                "name": command,
                "task": {"command": command, "arguments": arguments},
                "result": {"content_type": "image/png"},
            }
        ],
    }
    return task_argument_errors(definition)


class TestSharpenDomains:
    @pytest.mark.parametrize(
        "name,value,text",
        [
            ("amount", -1, "zero or above"),
            ("radius", 0, "above zero"),
            ("threshold", 256, "between 0 and 255"),
            ("threshold", -1, "between 0 and 255"),
        ],
    )
    def test_refused_at_validate_time_naming_the_argument_and_range(
        self, name, value, text
    ):
        errors = _errors("sharpen", {"media": "asset:a.png", name: value})
        assert [e["path"] for e in errors] == [f"steps[0].task.arguments.{name}"]
        assert name in errors[0]["message"]
        assert text in errors[0]["message"]

    @pytest.mark.parametrize(
        "name,value", [("amount", -1), ("radius", 0), ("threshold", 256)]
    )
    def test_refused_at_run_time(self, name, value):
        with pytest.raises(ValueError, match=name):
            _run("sharpen", _soft_step(), **{name: value})

    def test_threshold_minus_one_is_refused_at_run_time(self):
        with pytest.raises(ValueError, match="threshold"):
            _run("sharpen", _soft_step(), threshold=-1)

    def test_validate_workflow_refuses_it_before_a_run(self):
        workflow = Workflow(
            {
                "id": "sharpen-domains",
                "steps": [
                    {
                        "name": "sharpen",
                        "task": {
                            "command": "sharpen",
                            "arguments": {"media": "asset:a.png", "radius": 0},
                        },
                        "result": {"content_type": "image/png"},
                    }
                ],
            },
            "outputs",
            None,
        )
        errors = workflow.validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.radius"]

    def test_the_edges_of_each_range_are_legitimate(self):
        arguments = {"media": "asset:a.png", "amount": 0, "threshold": 255}
        assert _errors("sharpen", arguments) == []
        assert _errors("sharpen", {"media": "asset:a.png", "threshold": 0}) == []


class TestFilmGrain:
    def test_amount_zero_is_identity(self):
        source = _soft_step()
        assert numpy.array_equal(
            _array(film_grain(source, amount=0, seed=1)), _array(source)
        )

    def test_the_same_seed_reproduces_the_output_exactly(self):
        source = _grey()
        first = _run("film_grain", source, workflow_seed=5, amount=0.3)
        second = _run("film_grain", source, workflow_seed=5, amount=0.3)
        assert numpy.array_equal(_array(first), _array(second))

    def test_a_different_seed_gives_different_output(self):
        source = _grey()
        first = _run("film_grain", source, workflow_seed=5, amount=0.3)
        other = _run("film_grain", source, workflow_seed=6, amount=0.3)
        assert not numpy.array_equal(_array(first), _array(other))
        assert not numpy.array_equal(_array(first), _array(source))

    def test_an_explicit_seed_argument_wins_over_the_workflow_seed(self):
        source = _grey()
        explicit = _run("film_grain", source, workflow_seed=1, amount=0.3, seed=9)
        from_workflow = _run("film_grain", source, workflow_seed=9, amount=0.3)
        other_workflow = _run("film_grain", source, workflow_seed=1, amount=0.3)
        assert numpy.array_equal(_array(explicit), _array(from_workflow))
        assert not numpy.array_equal(_array(explicit), _array(other_workflow))

    def test_a_video_keeps_count_fps_and_audio_with_different_grain_per_frame(self):
        source = _video()
        result = _run("film_grain", source, workflow_seed=3, amount=0.3)
        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 4
        assert result.fps == 12
        assert result.sample_rate == 100
        assert numpy.array_equal(result.audio, source.audio)
        arrays = [_array(frame) for frame in result.frames]
        for i in range(len(arrays) - 1):
            assert not numpy.array_equal(arrays[i], arrays[i + 1])

    def test_the_same_seed_reproduces_every_frame_of_a_video(self):
        first = _run("film_grain", _video(), workflow_seed=3, amount=0.3)
        second = _run("film_grain", _video(), workflow_seed=3, amount=0.3)
        for a, b in zip(first.frames, second.frames):
            assert numpy.array_equal(_array(a), _array(b))

    def test_chroma_zero_leaves_hue_unchanged_and_chroma_one_does_not(self):
        source = Image.new("RGB", (32, 48), (150, 100, 80))
        base = _array(source).astype(numpy.int32)

        def spread(result):
            out = _array(result).astype(numpy.int32)
            rg = (out[..., 0] - out[..., 1]) - (base[..., 0] - base[..., 1])
            gb = (out[..., 1] - out[..., 2]) - (base[..., 1] - base[..., 2])
            return numpy.abs(rg).max(), numpy.abs(gb).max()

        mono = film_grain(source, amount=0.2, chroma=0.0, seed=2)
        assert not numpy.array_equal(_array(mono), _array(source))
        assert max(spread(mono)) <= 1
        colour = film_grain(source, amount=0.2, chroma=1.0, seed=2)
        assert max(spread(colour)) > 1

    def test_a_larger_size_gives_coarser_grain(self):
        source = _grey(128, size=(64, 64))

        def roughness(size):
            grain = _red(film_grain(source, amount=0.5, size=size, seed=4)) - 128.0
            return numpy.abs(numpy.diff(grain, axis=1)).mean()

        assert roughness(4) < roughness(1)

    def test_grain_is_strongest_in_the_midtones(self):
        mid = _red(film_grain(_grey(128), amount=0.3, seed=8))
        dark = _red(film_grain(_grey(10), amount=0.3, seed=8))
        assert mid.std() > dark.std()

    def test_alpha_passes_through(self):
        source = _with_alpha(_grey())
        result = film_grain(source, amount=0.5, seed=1)
        assert result.mode == "RGBA"
        assert numpy.array_equal(_array(result)[..., 3], _array(source)[..., 3])
        assert not numpy.array_equal(_array(result)[..., :3], _array(source)[..., :3])

    def test_the_seed_is_logged(self):
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            _run("film_grain", _grey(), workflow_seed=11)
        finally:
            deactivate_context(token)
        logs = [e for e in events if e.get("event") == "log"]
        assert len(logs) == 1
        assert logs[0]["seed"] == 11


class TestFilmGrainDomains:
    @pytest.mark.parametrize(
        "name,value,text",
        [
            ("amount", 1.5, "between 0.0 and 1.0"),
            ("amount", -0.1, "between 0.0 and 1.0"),
            ("chroma", 2, "between 0.0 and 1.0"),
            ("size", 0.5, "1 or above"),
            ("seed", "abc", "a whole number, 0 or above"),
            ("seed", 1.5, "a whole number, 0 or above"),
            ("seed", -1, "a whole number, 0 or above"),
        ],
    )
    def test_refused_at_validate_time_naming_the_argument_and_range(
        self, name, value, text
    ):
        errors = _errors("film_grain", {"media": "asset:a.png", name: value})
        assert [e["path"] for e in errors] == [f"steps[0].task.arguments.{name}"]
        assert name in errors[0]["message"]
        assert text in errors[0]["message"]

    @pytest.mark.parametrize(
        "name,value",
        [
            ("amount", 1.5),
            ("amount", -0.1),
            ("chroma", 2),
            ("size", 0.5),
            ("seed", "abc"),
            ("seed", -1),
        ],
    )
    def test_refused_at_run_time(self, name, value):
        with pytest.raises(ValueError, match=name):
            _run("film_grain", _grey(), workflow_seed=1, **{name: value})

    def test_validate_workflow_refuses_a_non_integer_seed_before_a_run(self):
        # The #634 bounce: "abc" validated clean and failed only once queued
        workflow = Workflow(
            {
                "id": "film-grain-seed",
                "steps": [
                    {
                        "name": "film_grain",
                        "task": {
                            "command": "film_grain",
                            "arguments": {
                                "media": "asset:a.png",
                                "amount": 0.4,
                                "seed": "abc",
                            },
                        },
                        "result": {"content_type": "image/png"},
                    }
                ],
            },
            "outputs",
            None,
        )
        errors = workflow.validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.seed"]
        assert "a whole number, 0 or above" in errors[0]["message"]

    @pytest.mark.parametrize("seed", [0, 1108670404077236, "42", None])
    def test_a_whole_number_seed_is_legitimate(self, seed):
        assert _errors("film_grain", {"media": "asset:a.png", "seed": seed}) == []


class TestIntrospection:
    def test_describe_task_lists_a_seed_parameter(self):
        from dw.introspection import describe_task

        names = [p["name"] for p in describe_task("film_grain")["parameters"]]
        assert "seed" in names

    def test_both_commands_are_in_the_task_listing(self):
        from dw.introspection import list_tasks

        commands = list_tasks()["commands"]
        assert "sharpen" in commands
        assert "film_grain" in commands

    def test_the_new_domains_are_described_with_their_range_text(self):
        from dw.introspection import describe_task

        sharpen = {p["name"]: p for p in describe_task("sharpen")["parameters"]}
        assert sharpen["threshold"]["range"] == "between 0 and 255"
        grain = {p["name"]: p for p in describe_task("film_grain")["parameters"]}
        assert grain["size"]["range"] == "1 or above"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRgbaFileKeepsAlpha:
    """An RGBA file on disk - an asset:/output: path once resolved - loads with
    its alpha, so the finishing commands hand it back untouched (#775). The
    pipeline-input loader still flattens to RGB."""

    @staticmethod
    def _cutout(tmp_path):
        rgb = numpy.full((16, 16, 3), 200, dtype=numpy.uint8)
        alpha = numpy.zeros((16, 16), dtype=numpy.uint8)
        alpha[4:12, 4:12] = 255
        image = Image.fromarray(rgb).convert("RGBA")
        image.putalpha(Image.fromarray(alpha))
        path = tmp_path / "cutout.png"
        image.save(path)
        return str(path), alpha

    @pytest.mark.parametrize(
        "command, arguments",
        [
            ("grade", {"exposure": 0.5, "saturation": 0.5}),
            ("film_grain", {"amount": 0.3, "seed": 1}),
            ("apply_lut", {"palette": ["#102030", "#e0c090"]}),
            ("sharpen", {"amount": 1.0}),
        ],
    )
    def test_a_file_path_comes_back_rgba_with_its_alpha(
        self, tmp_path, command, arguments
    ):
        path, alpha = self._cutout(tmp_path)
        result = _run(command, path, **arguments)
        assert result.mode == "RGBA"
        assert numpy.array_equal(_array(result.getchannel("A")), alpha)

    def test_an_opaque_file_still_loads_rgb(self, tmp_path):
        from dw.tasks.task import _load_media

        path = tmp_path / "opaque.png"
        Image.new("RGB", (8, 8), "red").save(path)
        assert _load_media(str(path)).mode == "RGB"

    def test_a_palette_file_with_transparency_loads_rgba(self, tmp_path):
        from dw.tasks.task import _load_media

        path = tmp_path / "palette.png"
        image = Image.new("P", (8, 8), 0)
        image.info["transparency"] = 0
        image.save(path, transparency=0)
        assert _load_media(str(path)).mode == "RGBA"

    def test_a_pipeline_input_still_flattens_to_rgb(self, tmp_path):
        from dw.argument_media import fetch_image

        path, _ = self._cutout(tmp_path)
        assert fetch_image(path).mode == "RGB"
