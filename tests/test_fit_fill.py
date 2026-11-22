"""Unit tests for #1567 (#1066 stage C): fit_to_model's letterbox `fill`, the
reframe template's frame-count refusal (C-F1154(7)) and its single feather
message (C-F1154(9))."""

import json
import os

import numpy
import pytest

from dw.task_domains import FIT_FILLS, task_argument_errors
from dw.tasks.fit import fit_to_model

from .test_paste_into_box import COUNT, clip, inside

SOURCES = [(1344, 768), (960, 544), (544, 960)]
CANVASES = [(544, 960), (768, 768), (960, 544), (768, 1344)]
TEMPLATE = os.path.join("workflows", "templates", "minimax", "reframe.json")


def fitted(source, canvas, **arguments):
    width, height = canvas
    return fit_to_model(
        clip(*source), width=width, height=height, num_frames=COUNT, **arguments
    )


def frames_of(result):
    return numpy.asarray(result["video"])


class TestFill:
    @pytest.mark.parametrize("fill", ["edge", "blur"])
    @pytest.mark.parametrize("align", [1, 32])
    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_only_the_bars_change(self, fill, align, source, canvas):
        black = fitted(source, canvas, align=align)
        painted = fitted(source, canvas, align=align, fill=fill)
        assert dict(painted["fit"]) == dict(black["fit"])
        assert numpy.array_equal(
            numpy.asarray(painted["mask"]), numpy.asarray(black["mask"])
        )
        where = inside(painted["fit"]["content_box"], *canvas)
        assert numpy.array_equal(
            frames_of(painted)[:, where], frames_of(black)[:, where]
        )
        assert frames_of(painted).shape == frames_of(black).shape

    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_black_is_the_default(self, source, canvas):
        plain = fitted(source, canvas, align=32)
        black = fitted(source, canvas, align=32, fill="black")
        assert numpy.array_equal(frames_of(plain), frames_of(black))
        bars = numpy.asarray(black["mask"]) == 255
        assert (frames_of(black)[:, bars] == 0).all()

    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_edge_starts_at_the_picture_s_edge(self, source, canvas):
        result = fitted(source, canvas, align=32, fill="edge")
        frames = frames_of(result)
        box = result["fit"]["content_box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        rows, columns = slice(y, y + h), slice(x, x + w)
        if y > 0:
            assert numpy.array_equal(frames[:, y - 1, columns], frames[:, y, columns])
        if y + h < canvas[1]:
            assert numpy.array_equal(
                frames[:, y + h, columns], frames[:, y + h - 1, columns]
            )
        if x > 0:
            assert numpy.array_equal(frames[:, rows, x - 1], frames[:, rows, x])
        if x + w < canvas[0]:
            assert numpy.array_equal(frames[:, rows, x + w], frames[:, rows, x + w - 1])

    def test_edge_blurs_with_distance(self):
        # A hard-striped picture: the far end of a bar is smoother than its
        # first row, which is the stripes themselves
        stripes = numpy.zeros((COUNT, 320, 544, 3), numpy.float32)
        stripes[:, :, ::8] = 1.0
        result = fit_to_model(stripes, 544, 960, COUNT, align=32, fill="edge")
        frames = frames_of(result)
        middle = slice(200, 344)
        first, far = frames[:, 319, middle], frames[:, 0, middle]
        assert first.std() > 4 * far.std()

    @pytest.mark.parametrize("source", SOURCES)
    @pytest.mark.parametrize("canvas", CANVASES)
    def test_blur_leaves_no_black(self, source, canvas):
        frames = frames_of(fitted(source, canvas, align=32, fill="blur"))
        assert frames.shape == (COUNT, canvas[1], canvas[0], 3)
        # The source's channels sit in 0.1-0.9, so no pixel is black
        assert (frames.max(axis=-1) > 0.05).all()

    @pytest.mark.parametrize("fill", ["edge", "blur"])
    @pytest.mark.parametrize("mode", ["crop", "stretch"])
    def test_crop_and_stretch_refuse_a_fill(self, fill, mode):
        with pytest.raises(ValueError, match="'fill'"):
            fitted((1344, 768), (544, 960), mode=mode, fill=fill)

    @pytest.mark.parametrize("mode", ["crop", "stretch"])
    def test_crop_and_stretch_take_black(self, mode):
        assert "mask" not in fitted((1344, 768), (544, 960), mode=mode, fill="black")

    def test_an_unknown_fill_is_refused(self):
        with pytest.raises(ValueError, match="'fill'"):
            fitted((1344, 768), (544, 960), fill="mirror")


def _validate_fit(**arguments):
    workflow = {
        "steps": [
            {
                "name": "fit",
                "task": {
                    "command": "fit_to_model",
                    "arguments": {
                        "video": "variable:clip",
                        "width": 544,
                        "height": 960,
                        "num_frames": 121,
                        **arguments,
                    },
                },
            }
        ]
    }
    return [
        (error["path"].rsplit(".", 1)[-1], error["message"])
        for error in task_argument_errors(workflow)
    ]


class TestFillValidate:
    @pytest.mark.parametrize("fill", FIT_FILLS)
    def test_letterbox_takes_every_fill(self, fill):
        assert _validate_fit(mode="letterbox", fill=fill) == []
        assert _validate_fit(fill=fill) == []

    @pytest.mark.parametrize("fill", ["edge", "blur"])
    @pytest.mark.parametrize("mode", ["crop", "stretch"])
    def test_crop_and_stretch_with_a_fill_are_refused(self, fill, mode):
        errors = _validate_fit(mode=mode, fill=fill)
        assert errors == [
            (
                "fill",
                f"fit_to_model takes 'fill' {fill!r} only with mode 'letterbox' "
                f"(it paints the letterbox's bars), got mode {mode!r}.",
            )
        ]

    @pytest.mark.parametrize("mode", ["crop", "stretch"])
    def test_crop_and_stretch_take_black(self, mode):
        assert _validate_fit(mode=mode, fill="black") == []

    def test_an_unknown_fill_is_refused_once(self):
        errors = _validate_fit(fill="mirror")
        assert [name for name, _ in errors] == ["fill"]
        assert "['black', 'edge', 'blur']" in errors[0][1]

    def test_a_deferred_fill_or_mode_is_not_checked(self):
        assert _validate_fit(mode="crop", fill="variable:fill") == []
        assert _validate_fit(mode="variable:mode", fill="edge") == []

    def test_a_fill_and_an_align_off_letterbox_are_both_named(self):
        names = [name for name, _ in _validate_fit(mode="crop", fill="edge", align=32)]
        assert sorted(names) == ["align", "fill"]


def _write_take(path, frames, size=(96, 54)):
    import av

    width, height = size
    container = av.open(path, "w")
    stream = container.add_stream("libx264", rate=24)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(
            numpy.full((height, width, 3), 40 + index % 100, numpy.uint8),
            format="rgb24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture(scope="module")
def take_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reframe")
    _write_take(str(directory / "take.mp4"), 124)
    return directory


def _template_errors(directory, template, args=None):
    from dw.validation import workflow_errors
    from dw.workflow import workflow_from_file

    path = directory / "workflow.json"
    path.write_text(json.dumps(template))
    workflow = workflow_from_file(str(path), str(directory))
    return [error["message"] for error in workflow_errors(workflow, args)]


def _reframe():
    with open(TEMPLATE) as f:
        template = json.load(f)
    template["variables"]["source_video"] = "take.mp4"
    template["variables"]["prompt"] = "a brick wall rises to a ceiling"
    return template


FRAME_COUNT = "they must match"


class TestReframeFrameCount:
    """C-F1154(7): reframe refuses a take whose length is not num_frames, as
    regional-edit does, reading the count from the fit step's video."""

    def test_the_defaults_are_clean(self, take_dir):
        messages = _template_errors(take_dir, _reframe())
        assert not [m for m in messages if FRAME_COUNT in m]

    @pytest.mark.parametrize("num_frames", [141, 125])
    def test_another_length_is_refused_naming_both_counts(self, take_dir, num_frames):
        messages = _template_errors(take_dir, _reframe(), {"num_frames": num_frames})
        refused = [m for m in messages if FRAME_COUNT in m]
        assert len(refused) == 1
        # 125 snaps up to 141, the count the step renders
        assert "124 frames" in refused[0] and "num_frames=141" in refused[0]
        assert "fit_to_model step 'fit'" in refused[0]

    def test_a_bare_fit_that_pads_is_unchanged(self, take_dir):
        template = {
            "id": "PadAShortClip",
            "steps": [
                {
                    "name": "fit",
                    "task": {
                        "command": "fit_to_model",
                        "arguments": {
                            "video": "take.mp4",
                            "width": 96,
                            "height": 64,
                            "num_frames": 141,
                        },
                    },
                    "result": {"content_type": "video/mp4"},
                }
            ],
        }
        assert _template_errors(take_dir, template) == []


class TestReframeFeather:
    """C-F1154(9): a bad feather is reported once."""

    @pytest.mark.parametrize("feather", [-1, -16])
    def test_a_negative_feather_is_one_message(self, take_dir, feather):
        messages = _template_errors(take_dir, _reframe(), {"feather": feather})
        assert len([m for m in messages if "feather" in m]) == 1

    def test_a_fractional_feather_is_one_message(self, take_dir):
        messages = _template_errors(take_dir, _reframe(), {"feather": 2.5})
        assert len([m for m in messages if "feather" in m]) == 1
