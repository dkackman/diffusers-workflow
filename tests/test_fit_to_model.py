"""Unit tests for fit_to_model / restore_to_source (dw/tasks/fit.py, #602) and
the fit rules in dw/task_domains.py. Frames are synthetic and smooth, so a
resize and its inverse stay close to the source."""

import json
import os

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo, JsonRecord
from dw.result import Result
from dw.task_domains import task_argument_errors
from dw.tasks.fit import fit_to_model, restore_to_source
from dw.tasks.video_utils import FrameList

SRC_W, SRC_H, SRC_FRAMES = 640, 480, 50
MODEL = dict(width=512, height=288, num_frames=57)


def smooth_frames(count=SRC_FRAMES, width=SRC_W, height=SRC_H):
    """A low-frequency float32 (count, h, w, 3) video in [0.1, 0.9]."""
    ys, xs = numpy.mgrid[0:height, 0:width].astype(numpy.float32)
    ys, xs = ys / height, xs / width
    frames = []
    for index in range(count):
        t = index / max(1, count - 1)
        frames.append(
            numpy.stack(
                [
                    0.5 + 0.4 * numpy.sin(2 * xs + t),
                    0.5 + 0.4 * numpy.cos(2 * ys - t),
                    0.5 + 0.4 * numpy.sin(xs + ys + 2 * t),
                ],
                axis=-1,
            )
        )
    return numpy.stack(frames).astype(numpy.float32)


def clip(count=SRC_FRAMES, width=SRC_W, height=SRC_H, fps=24):
    return AudioVideo(smooth_frames(count, width, height), None, None, fps=fps)


def fit(video=None, **overrides):
    arguments = dict(MODEL, mode="letterbox")
    arguments.update(overrides)
    return fit_to_model(clip() if video is None else video, **arguments)


def frames_of(video):
    return numpy.asarray(video.frames)


def upscaled(frames, factor):
    return numpy.repeat(numpy.repeat(frames, factor, axis=1), factor, axis=2)


def mean_error(a, b):
    return float(numpy.abs(a - b).mean())


@pytest.mark.parametrize("mode", ["letterbox", "stretch", "crop"])
class TestFitShape:
    def test_fitted_video_has_the_model_shape_and_fps(self, mode):
        result = fit(mode=mode)
        assert frames_of(result["video"]).shape == (57, 288, 512, 3)
        assert frames_of(result["video"]).dtype == numpy.float32
        assert result["video"].fps == 24
        assert result["video"].audio is None

    def test_record_fields(self, mode):
        record = fit(mode=mode)["fit"]
        assert isinstance(record, JsonRecord)
        assert record["mode"] == mode
        assert (record["source_width"], record["source_height"]) == (640, 480)
        assert record["source_frames"] == 50
        assert (record["model_width"], record["model_height"]) == (512, 288)
        assert record["model_frames"] == 57
        assert set(record) >= {"content_box", "source_box"}


class TestBoxes:
    def test_letterbox_centres_the_content_on_black_bars(self):
        result = fit(mode="letterbox")
        assert result["fit"]["content_box"] == {"x": 64, "y": 0, "w": 384, "h": 288}
        assert result["fit"]["source_box"] == {"x": 0, "y": 0, "w": 640, "h": 480}
        frames = frames_of(result["video"])
        assert (frames[:, :, :64] == 0).all()
        assert (frames[:, :, 448:] == 0).all()
        assert frames[:, :, 64:448].mean() > 0.2

    def test_stretch_fills_the_frame(self):
        record = fit(mode="stretch")["fit"]
        assert record["content_box"] == {"x": 0, "y": 0, "w": 512, "h": 288}
        assert record["source_box"] == {"x": 0, "y": 0, "w": 640, "h": 480}

    def test_crop_keeps_the_centre_of_the_source(self):
        record = fit(mode="crop")["fit"]
        assert record["source_box"] == {"x": 0, "y": 60, "w": 640, "h": 360}
        assert record["content_box"] == {"x": 0, "y": 0, "w": 512, "h": 288}


class TestRestore:
    @pytest.mark.parametrize("mode", ["letterbox", "stretch"])
    def test_one_x_comes_back_at_the_source_size(self, mode):
        result = fit(mode=mode)
        restored = restore_to_source(result["video"], result["fit"])
        frames = frames_of(restored)
        assert frames.shape == (50, 480, 640, 3)
        assert restored.fps == 24
        assert restored.audio is None
        assert mean_error(frames, smooth_frames()) < 0.02

    @pytest.mark.parametrize("mode", ["letterbox", "stretch"])
    def test_two_x_restores_to_twice_the_source(self, mode):
        result = fit(mode=mode)
        big = AudioVideo(upscaled(frames_of(result["video"]), 2), None, None, fps=24)
        restored = restore_to_source(big, result["fit"])
        frames = frames_of(restored)
        assert frames.shape == (50, 960, 1280, 3)
        assert restored.fps == 24
        assert mean_error(frames[:, ::2, ::2], smooth_frames()) < 0.03

    def test_crop_restores_the_kept_part_at_one_x(self):
        result = fit(mode="crop")
        restored = restore_to_source(result["video"], result["fit"])
        frames = frames_of(restored)
        assert frames.shape == (50, 360, 640, 3)
        assert mean_error(frames, smooth_frames()[:, 60:420]) < 0.02

    def test_crop_restores_to_twice_the_kept_part_at_two_x(self):
        result = fit(mode="crop")
        big = AudioVideo(upscaled(frames_of(result["video"]), 2), None, None, fps=24)
        frames = frames_of(restore_to_source(big, result["fit"]))
        assert frames.shape == (50, 720, 1280, 3)
        assert mean_error(frames[:, ::2, ::2], smooth_frames()[:, 60:420]) < 0.03


class TestFrameHandling:
    def test_a_short_source_holds_its_last_frame_and_restores_short(self):
        result = fit(clip(10), num_frames=17)
        frames = frames_of(result["video"])
        assert frames.shape[0] == 17
        for index in range(10, 17):
            assert numpy.array_equal(frames[index], frames[9])
        assert not numpy.array_equal(frames[8], frames[9])
        assert result["fit"]["source_frames"] == 10
        restored = restore_to_source(result["video"], result["fit"])
        assert frames_of(restored).shape[0] == 10

    def test_a_long_source_is_cut_to_its_first_frames(self):
        result = fit(clip(50), num_frames=17, mode="stretch")
        assert frames_of(result["video"]).shape[0] == 17
        restored = restore_to_source(result["video"], result["fit"])
        # Only 17 frames came back from the model, so 17 come out
        assert frames_of(restored).shape[0] == 17

    def test_an_output_shorter_than_the_source_restores_to_its_own_length(self):
        result = fit(mode="stretch")
        short = AudioVideo(frames_of(result["video"])[:20], None, None, fps=24)
        restored = restore_to_source(short, result["fit"])
        assert frames_of(restored).shape == (20, 480, 640, 3)


class TestInputs:
    def test_a_list_of_pil_images(self):
        images = [
            Image.fromarray((f * 255).astype(numpy.uint8)) for f in smooth_frames(5)
        ]
        result = fit(images, num_frames=9)
        assert frames_of(result["video"]).shape == (9, 288, 512, 3)
        assert result["fit"]["source_frames"] == 5

    def test_a_uint8_array(self):
        array = (smooth_frames(5) * 255).astype(numpy.uint8)
        result = fit(array, num_frames=9)
        out = frames_of(result["video"])
        assert out.shape == (9, 288, 512, 3)
        assert 0.0 <= out.min() and out.max() <= 1.0
        assert out[:, :, 64:448].mean() > 0.2

    def test_a_frame_list_keeps_its_fps(self):
        frames = FrameList(
            Image.fromarray((f * 255).astype(numpy.uint8)) for f in smooth_frames(5)
        )
        frames.fps = 30
        assert fit(frames, num_frames=9)["video"].fps == 30

    def test_string_numbers_are_coerced(self):
        result = fit_to_model(clip(5), width="512", height="288", num_frames="9")
        assert frames_of(result["video"]).shape == (9, 288, 512, 3)


class TestRefusals:
    @pytest.mark.parametrize("name", ["width", "height", "num_frames"])
    @pytest.mark.parametrize("bad", [0, -8, 2.5, True, "abc"])
    def test_a_bad_size_names_the_argument(self, name, bad):
        with pytest.raises(ValueError, match=name):
            fit(**{name: bad})

    @pytest.mark.parametrize("mode", ["fill", "", None, 3])
    def test_an_unknown_mode(self, mode):
        with pytest.raises(ValueError, match="mode"):
            fit(mode=mode)

    def test_an_empty_video(self):
        with pytest.raises(ValueError, match="video"):
            fit(AudioVideo([], None, None, fps=24))

    def good_record(self):
        return dict(fit(mode="letterbox")["fit"])

    def restore(self, record, video=None):
        if video is None:
            video = AudioVideo(
                numpy.zeros((3, 288, 512, 3), numpy.float32), None, None, fps=24
            )
        return restore_to_source(video, record)

    def test_the_good_record_restores(self):
        assert frames_of(self.restore(self.good_record())).shape[0] == 3

    @pytest.mark.parametrize("record", [None, [1, 2], 5])
    def test_a_record_that_is_not_a_dict(self, record):
        with pytest.raises(ValueError, match="fit"):
            self.restore(record)

    def test_a_missing_field(self):
        record = self.good_record()
        del record["source_width"]
        with pytest.raises(ValueError, match="source_width"):
            self.restore(record)

    @pytest.mark.parametrize("value", ["640", 6.4, True, 0, -1, None])
    def test_a_field_of_the_wrong_type(self, value):
        record = self.good_record()
        record["model_frames"] = value
        with pytest.raises(ValueError, match="model_frames"):
            self.restore(record)

    def test_a_bad_mode(self):
        record = self.good_record()
        record["mode"] = "fill"
        with pytest.raises(ValueError, match="mode"):
            self.restore(record)

    @pytest.mark.parametrize(
        "box",
        [
            {"x": 400, "y": 0, "w": 384, "h": 288},
            {"x": 0, "y": 10, "w": 384, "h": 288},
            {"x": -1, "y": 0, "w": 384, "h": 288},
            {"x": 0, "y": 0, "w": 0, "h": 288},
            {"x": 0, "y": 0, "w": 384},
        ],
    )
    def test_a_box_outside_the_frame(self, box):
        record = self.good_record()
        record["content_box"] = box
        with pytest.raises(ValueError, match="content_box"):
            self.restore(record)

    def test_a_crop_box_outside_the_source(self):
        record = dict(fit(mode="crop")["fit"])
        record["source_box"] = {"x": 0, "y": 200, "w": 640, "h": 360}
        with pytest.raises(ValueError, match="source_box"):
            self.restore(record)

    def test_a_non_uniform_scale_names_both_sizes(self):
        video = AudioVideo(
            numpy.zeros((3, 300, 512, 3), numpy.float32), None, None, fps=24
        )
        with pytest.raises(ValueError, match=r"512x300.*512x288"):
            self.restore(self.good_record(), video)


class TestSavedRecord:
    def test_a_saved_json_path_restores(self, tmp_path, monkeypatch):
        import dw.locations as locations

        result = fit(mode="stretch")
        path = tmp_path / "fit.json"
        path.write_text(json.dumps(result["fit"]))
        # A tmp path is outside the media roots unless workflows are trusted
        monkeypatch.setattr(locations, "workflows_are_trusted", lambda: True)
        restored = restore_to_source(result["video"], str(path))
        assert frames_of(restored).shape == (50, 480, 640, 3)

    def test_an_untrusted_path_outside_the_roots_is_refused(
        self, tmp_path, monkeypatch
    ):
        import dw.locations as locations

        result = fit(mode="stretch")
        path = tmp_path / "fit.json"
        path.write_text(json.dumps(result["fit"]))
        monkeypatch.setattr(locations, "workflows_are_trusted", lambda: False)
        with pytest.raises(Exception, match="Refusing"):
            restore_to_source(result["video"], str(path))


def _validate(arguments):
    workflow = {
        "steps": [
            {
                "name": "f",
                "task": {"command": "fit_to_model", "arguments": arguments},
            }
        ]
    }
    errors = task_argument_errors(workflow)
    return [(error["path"].rsplit(".", 1)[-1], error["message"]) for error in errors]


GOOD = {"video": "variable:v", "width": 512, "height": 288, "num_frames": 57}


class TestStaticValidation:
    def names(self, errors):
        return [name for name, _ in errors]

    def test_an_unknown_literal_mode(self):
        errors = _validate(dict(GOOD, mode="fill"))
        assert "mode" in self.names(errors), errors

    def test_width_zero(self):
        assert "width" in self.names(_validate(dict(GOOD, width=0)))

    def test_a_fractional_width(self):
        assert "width" in self.names(_validate(dict(GOOD, width=512.5)))

    @pytest.mark.parametrize("mode", ["variable:m", "previous_result:s.mode"])
    def test_a_reference_mode_is_silent(self, mode):
        assert not _validate(dict(GOOD, mode=mode))

    @pytest.mark.parametrize("mode", ["letterbox", "stretch", "crop"])
    def test_valid_literals_are_silent(self, mode):
        assert not _validate(dict(GOOD, mode=mode))

    def test_no_mode_is_silent(self):
        assert not _validate(GOOD)


class TestRealPath:
    def test_both_commands_through_the_task_dispatch_and_result_save(self, tmp_path):
        from dw.tasks.task import Task

        source = clip(12)
        fitted = Task({"command": "fit_to_model", "arguments": {}}, "cpu").run(
            dict(video=source, width=512, height=288, num_frames=17, mode="crop")
        )
        assert set(fitted) == {"video", "fit"}
        assert frames_of(fitted["video"]).shape == (17, 288, 512, 3)

        restored = Task({"command": "restore_to_source", "arguments": {}}, "cpu").run(
            dict(video=fitted["video"], fit=fitted["fit"])
        )
        assert frames_of(restored).shape == (12, 360, 640, 3)

        result = Result({"content_type": "video/mp4", "save": True})
        result.add_result(fitted)
        result.save(str(tmp_path), "wf-step.0")
        names = sorted(os.listdir(tmp_path), key=lambda n: n.endswith(".json"))
        assert [os.path.splitext(n)[1] for n in names] == [".mp4", ".json"]
        with open(tmp_path / names[1]) as handle:
            assert json.load(handle) == fitted["fit"]

    def test_a_crop_restore_is_saved_at_its_own_size(self, tmp_path):
        """640x360 is not a multiple of 16 high; the written file keeps it
        rather than growing to 640x368 (#631's bounce)."""
        from dw.media import video_shape

        fitted = fit(clip(12), num_frames=17, mode="crop")
        restored = restore_to_source(fitted["video"], fitted["fit"])
        result = Result({"content_type": "video/mp4", "save": True})
        result.add_result(restored)
        result.save(str(tmp_path), "wf-restore.0")
        (name,) = os.listdir(tmp_path)
        shape = video_shape(str(tmp_path / name))
        assert (shape["width"], shape["height"], shape["frame_count"]) == (
            640,
            360,
            12,
        )
