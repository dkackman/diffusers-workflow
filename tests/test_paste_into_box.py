"""Unit tests for #1541 (#1066 stage A): fit_to_model's letterbox `mask` and
`align`, and paste_into_box (dw/tasks/paste_box.py) with its rules in
dw/task_problems.py."""

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.task_domains import task_argument_errors
from dw.tasks.fit import fit_to_model
from dw.tasks.paste_box import box_weights, paste_into_box

COUNT = 3
SOURCES = [(1344, 768), (960, 544)]
CANVASES = [(544, 960), (768, 1344), (768, 960), (768, 768)]


def smooth(width, height, count=COUNT):
    ys, xs = numpy.mgrid[0:height, 0:width].astype(numpy.float32)
    ys, xs = ys / height, xs / width
    frames = [
        numpy.stack(
            [
                0.5 + 0.4 * numpy.sin(3 * xs + t),
                0.5 + 0.4 * numpy.cos(3 * ys - t),
                0.5 + 0.4 * numpy.sin(xs + ys + t),
            ],
            axis=-1,
        )
        for t in range(count)
    ]
    return numpy.stack(frames).astype(numpy.float32)


def clip(width, height, count=COUNT, fps=24, audio=None, sample_rate=None):
    return AudioVideo(smooth(width, height, count), audio, sample_rate, fps=fps)


def fitted(source, canvas, **arguments):
    width, height = canvas
    return fit_to_model(
        clip(*source), width=width, height=height, num_frames=COUNT, **arguments
    )


def inside(box, width, height):
    """A (height, width) bool array, True inside the box."""
    where = numpy.zeros((height, width), dtype=bool)
    where[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]] = True
    return where


class TestLetterboxMask:
    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_mask_is_white_exactly_outside_the_box(self, source, canvas):
        result = fitted(source, canvas)
        mask = result["mask"]
        assert isinstance(mask, Image.Image)
        assert mask.mode == "L"
        assert mask.size == canvas
        pixels = numpy.asarray(mask)
        box = result["fit"]["content_box"]
        where = inside(box, *canvas)
        assert (pixels[where] == 0).all()
        assert (pixels[~where] == 255).all()

    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_mask_matches_the_black_bars(self, source, canvas):
        result = fitted(source, canvas)
        frames = numpy.asarray(result["video"])
        bars = numpy.asarray(result["mask"]) == 255
        if bars.any():
            assert (frames[:, bars] == 0).all()

    @pytest.mark.parametrize("mode", ["crop", "stretch"])
    def test_crop_and_stretch_return_no_mask(self, mode):
        result = fitted((1344, 768), (544, 960), mode=mode)
        assert "mask" not in result


class TestAlign:
    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_box_and_mask_sit_on_the_grid(self, source, canvas):
        result = fitted(source, canvas, align=32)
        box = result["fit"]["content_box"]
        assert all(box[key] % 32 == 0 for key in ("x", "y", "w", "h"))
        assert box["x"] + box["w"] <= canvas[0]
        assert box["y"] + box["h"] <= canvas[1]
        pixels = numpy.asarray(result["mask"])
        where = inside(box, *canvas)
        assert (pixels[where] == 0).all()
        assert (pixels[~where] == 255).all()

    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_the_crop_is_a_few_pixels(self, source, canvas):
        record = fitted(source, canvas, align=32)["fit"]
        box, kept = record["content_box"], record["source_box"]
        scale = box["w"] / kept["w"]
        # What the snap cut, in model pixels, is under one grid cell
        assert (source[0] - kept["w"]) * scale < 32
        assert (source[1] - kept["h"]) * scale < 32
        assert kept["x"] >= 0 and kept["y"] >= 0
        assert kept["x"] + kept["w"] <= source[0]
        assert kept["y"] + kept["h"] <= source[1]

    def test_1344x768_into_544x960(self):
        record = fitted((1344, 768), (544, 960), align=32)["fit"]
        assert record["content_box"] == {"x": 0, "y": 320, "w": 544, "h": 320}

    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_default_is_unchanged(self, source, canvas):
        plain, one = fitted(source, canvas), fitted(source, canvas, align=1)
        assert dict(plain["fit"]) == dict(one["fit"])
        assert numpy.array_equal(numpy.asarray(plain["video"]), one["video"])
        assert plain["fit"]["source_box"] == {
            "x": 0,
            "y": 0,
            "w": source[0],
            "h": source[1],
        }

    def test_align_is_refused_outside_letterbox(self):
        with pytest.raises(ValueError, match="align"):
            fitted((1344, 768), (544, 960), mode="crop", align=32)

    def test_align_must_divide_the_canvas(self):
        with pytest.raises(ValueError, match="align"):
            fitted((1344, 768), (544, 960), align=96)


SOURCE = (1344, 768)
CANVAS = (544, 960)


def generated(count=COUNT, fps=25, audio=True):
    rng = numpy.random.default_rng(7)
    frames = rng.random((count, CANVAS[1], CANVAS[0], 3), dtype=numpy.float32)
    track = numpy.full((2, 48000), 0.25, dtype=numpy.float32) if audio else None
    return AudioVideo(frames, track, 48000 if audio else None, fps=fps)


class TestPasteIntoBox:
    def setup_method(self):
        self.source = clip(*SOURCE)
        self.fit = fit_to_model(
            self.source, width=CANVAS[0], height=CANVAS[1], num_frames=COUNT, align=32
        )
        self.generated = generated()

    def test_source_is_exact_inside_the_box_beyond_the_feather(self):
        out = paste_into_box(self.generated, self.source, self.fit["fit"], feather=8)
        box = self.fit["fit"]["content_box"]
        fitted_frames = numpy.asarray(self.fit["video"])
        # x edges are flush with the frame, so only y is feathered
        rows = slice(box["y"] + 8, box["y"] + box["h"] - 8)
        cols = slice(box["x"], box["x"] + box["w"])
        assert numpy.array_equal(
            numpy.asarray(out.frames)[:, rows, cols], fitted_frames[:, rows, cols]
        )

    def test_generated_is_exact_outside_the_box(self):
        out = paste_into_box(self.generated, self.source, self.fit["fit"])
        where = inside(self.fit["fit"]["content_box"], *CANVAS)
        assert numpy.array_equal(
            numpy.asarray(out.frames)[:, ~where], self.generated.frames[:, ~where]
        )

    def test_the_ramp_is_monotonic(self):
        zeros = AudioVideo(
            numpy.zeros((COUNT, CANVAS[1], CANVAS[0], 3), numpy.float32), None, None
        )
        ones = AudioVideo(
            numpy.ones((COUNT, *SOURCE[::-1], 3), numpy.float32), None, None
        )
        out = numpy.asarray(paste_into_box(zeros, ones, self.fit["fit"]).frames)
        box = self.fit["fit"]["content_box"]
        column = out[0, box["y"] : box["y"] + box["h"], box["w"] // 2, 0]
        top = column[: box["h"] // 2]
        # A resize of a flat 1 is 1 to within float rounding
        assert (numpy.diff(top) >= -1e-6).all()
        assert (numpy.diff(column[box["h"] // 2 :]) <= 1e-6).all()
        assert 0 < top[0] < 1
        assert top[16] == pytest.approx(1)
        assert top[15] < 1 - 1e-3
        # The flush x edges are not feathered
        row = out[0, box["y"] + box["h"] // 2, :, 0]
        assert numpy.allclose(row, 1, atol=1e-6)

    def test_feather_zero_is_a_hard_edge(self):
        weight = box_weights({"x": 10, "y": 10, "w": 40, "h": 40}, 100, 100, 0)
        assert (weight == 1).all()

    def test_audio_fps_and_size_come_from_generated(self):
        source = clip(*SOURCE, fps=24, audio=numpy.zeros((2, 10)), sample_rate=8000)
        out = paste_into_box(self.generated, source, self.fit["fit"])
        assert out.fps == 25
        assert out.sample_rate == 48000
        assert numpy.array_equal(out.audio, self.generated.audio)
        assert numpy.asarray(out.frames).shape == (COUNT, CANVAS[1], CANVAS[0], 3)

    def test_a_bare_box_fills_and_centre_crops(self):
        box = {"x": 0, "y": 320, "w": 544, "h": 320}
        out = paste_into_box(self.generated, self.source, box, feather=0)
        from_record = paste_into_box(
            self.generated, self.source, self.fit["fit"], feather=0
        )
        assert numpy.array_equal(
            numpy.asarray(out.frames), numpy.asarray(from_record.frames)
        )
        listed = paste_into_box(self.generated, self.source, [0, 320, 544, 320], 0)
        assert numpy.array_equal(numpy.asarray(listed.frames), out.frames)

    def test_frame_count_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="one frame per frame"):
            paste_into_box(generated(count=2), self.source, self.fit["fit"])

    def test_a_box_outside_generated_is_refused(self):
        box = {"x": 0, "y": 700, "w": 544, "h": 320}
        with pytest.raises(ValueError, match="content_box"):
            paste_into_box(self.generated, self.source, box)

    def test_a_malformed_box_is_refused(self):
        for box in ({"x": 0, "y": 0, "w": 0, "h": 4}, [1, 2, 3], {"x": 1.5}):
            with pytest.raises(ValueError, match="content_box"):
                paste_into_box(self.generated, self.source, box)

    def test_a_record_of_another_source_is_refused(self):
        other = clip(960, 544)
        with pytest.raises(ValueError, match="fit record"):
            paste_into_box(self.generated, other, self.fit["fit"])

    @pytest.mark.parametrize("feather", [-1, 160, 2.5, "x"])
    def test_bad_feather_is_refused(self, feather):
        with pytest.raises(ValueError, match="feather"):
            paste_into_box(self.generated, self.source, self.fit["fit"], feather)

    def test_feather_just_below_half_the_short_side_runs(self):
        out = paste_into_box(self.generated, self.source, self.fit["fit"], 159)
        assert numpy.asarray(out.frames).shape[0] == COUNT


def _validate(arguments):
    workflow = {
        "steps": [
            {
                "name": "p",
                "task": {"command": "paste_into_box", "arguments": arguments},
            }
        ]
    }
    return [
        (error["path"].rsplit(".", 1)[-1], error["message"])
        for error in task_argument_errors(workflow)
    ]


GOOD = {
    "generated": "previous_result:gen",
    "source": "variable:clip",
    "content_box": "previous_result:fit.fit",
}


class TestValidate:
    def test_good_arguments_pass(self):
        assert _validate(GOOD) == []
        assert _validate({**GOOD, "feather": 0}) == []

    def test_a_literal_negative_feather_is_refused(self):
        errors = _validate({**GOOD, "feather": -1})
        assert errors and all("feather" in message for _, message in errors)

    def test_a_fractional_feather_is_refused(self):
        assert any(
            "feather" in message for _, message in _validate({**GOOD, "feather": 2.5})
        )

    def test_a_deferred_feather_is_not_checked(self):
        assert _validate({**GOOD, "feather": "variable:f"}) == []

    def test_feather_against_a_literal_box(self):
        box = {"x": 0, "y": 320, "w": 544, "h": 320}
        assert _validate({**GOOD, "content_box": box, "feather": 159}) == []
        errors = _validate({**GOOD, "content_box": box, "feather": 160})
        assert any("shorter side" in message for _, message in errors)

    def test_a_malformed_literal_box_is_refused(self):
        errors = _validate({**GOOD, "content_box": [1, 2, 3]})
        assert any(name == "content_box" for name, _ in errors)

    def test_fit_align_is_checked_at_validate(self):
        workflow = {
            "steps": [
                {
                    "name": "f",
                    "task": {
                        "command": "fit_to_model",
                        "arguments": {
                            "video": "variable:v",
                            "width": 544,
                            "height": 960,
                            "num_frames": 57,
                            "align": 96,
                        },
                    },
                }
            ]
        }
        assert any(
            "align" in error["message"] for error in task_argument_errors(workflow)
        )


def _bad_records(record):
    """A fit record broken one way at a time, each a case the shared rule
    (`task_domains.fit_record_problem`) refuses."""
    record = dict(record)
    float_box = {**record, "content_box": {**record["content_box"], "y": 320.0}}
    kept_outside = {**record, "source_box": {"x": 0, "y": 0, "w": 2000, "h": 768}}
    box_outside = {**record, "content_box": {"x": 0, "y": 900, "w": 544, "h": 320}}
    bad_size = {**record, "model_width": 0}
    bad_mode = {**record, "mode": "zoom"}
    return [float_box, kept_outside, box_outside, bad_size, bad_mode]


class TestOneRecordRule:
    """restore_to_source, paste_into_box and paste_into_box_errors read
    fit_to_model's record by one rule, so they refuse the same records."""

    def setup_method(self):
        self.source = clip(*SOURCE)
        self.fit = fit_to_model(
            self.source, width=CANVAS[0], height=CANVAS[1], num_frames=COUNT, align=32
        )
        self.generated = generated()

    def test_the_real_record_is_read_by_all_three(self):
        from dw.tasks.fit import restore_to_source

        record = dict(self.fit["fit"])
        assert _validate({**GOOD, "content_box": record}) == []
        paste_into_box(self.generated, self.source, record)
        restore_to_source(self.fit["video"], record)

    @pytest.mark.parametrize("case", range(5))
    def test_a_bad_record_is_refused_by_all_three(self, case):
        from dw.tasks.fit import restore_to_source

        bad = _bad_records(self.fit["fit"])[case]
        assert any(
            name == "content_box" for name, _ in _validate({**GOOD, "content_box": bad})
        )
        with pytest.raises(ValueError, match="content_box"):
            paste_into_box(self.generated, self.source, bad)
        with pytest.raises(ValueError, match="fit record|record fit_to_model"):
            restore_to_source(self.fit["video"], bad)

    def test_a_float_bare_box_is_refused_at_run_and_validate(self):
        box = {"x": 0, "y": 320.0, "w": 544, "h": 320}
        with pytest.raises(ValueError, match="content_box"):
            paste_into_box(self.generated, self.source, box)
        assert any(
            name == "content_box" for name, _ in _validate({**GOOD, "content_box": box})
        )

    def test_a_whole_float_feather_is_coerced_by_whole_number(self):
        out = paste_into_box(self.generated, self.source, self.fit["fit"], 8.0)
        exact = paste_into_box(self.generated, self.source, self.fit["fit"], 8)
        assert numpy.array_equal(numpy.asarray(out.frames), numpy.asarray(exact.frames))


def test_paste_into_box_is_registered_and_described():
    from dw.introspection import describe_task
    from dw.tasks.task import _COMMAND_REGISTRY

    assert "paste_into_box" in _COMMAND_REGISTRY
    found = {p["name"]: p for p in describe_task("paste_into_box")["parameters"]}
    assert set(found) >= {"generated", "source", "content_box", "feather"}
    assert found["feather"]["default"] == 16
    fit = describe_task("fit_to_model")
    assert "mask" in fit["summary"]
    parameters = {p["name"]: p for p in fit["parameters"]}
    assert parameters["align"]["default"] == 1
    assert "mask" in parameters["mode"]["description"]


class TestReframeTemplateGrid:
    """#1542: templates/minimax/reframe's own defaults put the picture's box on
    H3's mask cells, so no cell straddles the box edge."""

    @staticmethod
    def defaults():
        import json

        with open("workflows/templates/minimax/reframe.json") as f:
            return json.load(f)["variables"]

    @pytest.mark.parametrize("canvas", [None, (768, 768)])
    def test_the_box_lands_on_the_cell_grid(self, canvas):
        from dw.pipeline_processors.h3_rules import TILE_GRID

        variables = self.defaults()
        assert variables["align"] == TILE_GRID
        if canvas is None:
            canvas = (variables["width"], variables["height"])
            assert canvas == (544, 960)
        result = fitted((960, 544), canvas, mode="letterbox", align=variables["align"])
        box = result["fit"]["content_box"]
        if canvas == (544, 960):
            assert box == {"x": 0, "y": 320, "w": 544, "h": 320}
        assert all(box[key] % TILE_GRID == 0 for key in ("x", "y", "w", "h"))
        # every cell is all bar or all picture
        pixels = numpy.asarray(result["mask"])
        width, height = canvas
        cells = pixels.reshape(
            height // TILE_GRID, TILE_GRID, width // TILE_GRID, TILE_GRID
        )
        assert (cells.min(axis=(1, 3)) == cells.max(axis=(1, 3))).all()
        if canvas == (544, 960):
            column = pixels[:, 0]
            assert column[319] == 255 and column[320] == 0
            assert column[639] == 0 and column[640] == 255
