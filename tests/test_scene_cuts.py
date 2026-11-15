"""detect_cuts: a clip's cuts as shot records laid out for an H3 render (#1194)."""

import numpy as np
import pytest
from PIL import Image

import dw.tasks.scene_cuts as scene_cuts
from dw.media_types import AudioVideo
from dw.task_domains import frames_to_samples, task_argument_errors
from dw.tasks.scene_cuts import detect_cuts, render_window
from dw.variable_constraints import aligned

W, H = 64, 36


def solid(colour, count):
    return [Image.new("RGB", (W, H), colour) for _ in range(count)]


def textured(zoom=1.0):
    """A picture with structure but steady colours; `zoom` crops its centre
    and enlarges it back - a cut between two framings of one picture."""
    y, x = np.mgrid[0:H, 0:W]
    picture = np.stack(
        [(x * 4) % 256, (y * 7) % 256, ((x + y) * 3) % 256], axis=-1
    ).astype(np.uint8)
    image = Image.fromarray(picture)
    if zoom != 1.0:
        w, h = int(W / zoom), int(H / zoom)
        left, top = (W - w) // 2, (H - h) // 2
        image = image.crop((left, top, left + w, top + h)).resize((W, H))
    return image


def clip(pictures, fps=24, sample_rate=48000):
    samples = frames_to_samples(len(pictures), fps, sample_rate)
    audio = np.zeros((1, samples), dtype=np.float32)
    return AudioVideo(pictures, audio, sample_rate, fps=fps)


@pytest.fixture
def window():
    found = render_window()
    if found is None:
        pytest.skip("this diffusers does not declare MiniMax-H3's frame window")
    return found


def lengths(result):
    return [cut["num_frames"] for cut in result["cuts"]]


class TestCuts:
    def test_two_colour_shots_give_two_cuts(self, window):
        result = detect_cuts(clip(solid((200, 30, 30), 40) + solid((30, 30, 200), 60)))
        cuts = result["cuts"]
        assert [c["name"] for c in cuts] == ["cut-1", "cut-2"]
        assert [c["start_frame"] for c in cuts] == [0, 40]
        assert lengths(result) == [40, 60]
        assert result["num_frames"] == 100 and result["fps"] == 24.0
        assert cuts[0]["detail"] == {"reason": "clip start"}
        assert cuts[1]["detail"]["reason"] == "content"
        assert "histogram_correlation" in cuts[1]["detail"]

    def test_a_zoom_cut_that_keeps_its_colours_is_found(self, window):
        pictures = [textured()] * 30 + [textured(zoom=2.0)] * 30
        assert [c["start_frame"] for c in detect_cuts(clip(pictures))["cuts"]] == [
            0,
            30,
        ]

    def test_one_shot_gives_one_cut(self, window):
        result = detect_cuts(clip([textured()] * 50))
        assert lengths(result) == [50]
        assert result["cuts"][0]["start_frame"] == 0

    def test_the_samples_tile_the_track(self, window):
        source = clip(solid((200, 30, 30), 40) + solid((30, 30, 200), 60))
        cuts = detect_cuts(source)["cuts"]
        assert cuts[0]["start_sample"] == 0
        assert cuts[1]["start_sample"] == cuts[0]["num_samples"]
        assert sum(c["num_samples"] for c in cuts) == source.audio.shape[1]

    def test_a_silent_clip_has_no_sample_side(self, window):
        source = AudioVideo(solid((200, 30, 30), 30), None, None, fps=24)
        cut = detect_cuts(source)["cuts"][0]
        assert "start_sample" not in cut or cut["start_sample"] is None

    def test_a_file_is_read(self, window, tmp_path):
        from diffusers.utils.export_utils import encode_video

        path = tmp_path / "two.mp4"
        encode_video(
            solid((200, 30, 30), 24) + solid((30, 30, 200), 24),
            fps=24,
            output_path=str(path),
        )
        result = detect_cuts(str(path))
        starts = [c["start_frame"] for c in result["cuts"]]
        assert len(starts) == 2 and abs(starts[1] - 24) <= 1
        assert sum(lengths(result)) == result["num_frames"] == 48

    def test_a_path_argument_keeps_its_fps(self, window, tmp_path):
        # An asset:/output: video was decoded to bare frames before the task
        # saw it, so fps and the sample side came back null (#1373 bounce)
        import dw.arguments as arguments_module
        from diffusers.utils.export_utils import encode_video

        from dw.tasks.video_utils import VideoFileReference

        path = tmp_path / "two.mp4"
        encode_video(
            solid((200, 30, 30), 24) + solid((30, 30, 200), 24),
            fps=24,
            output_path=str(path),
        )
        task = {"command": "detect_cuts", "arguments": {"video": str(path)}}
        arguments_module.realize_args(task, base_dir=str(tmp_path))
        assert isinstance(task["arguments"]["video"], VideoFileReference)
        result = detect_cuts(**task["arguments"])
        assert result["fps"] == pytest.approx(24)
        assert len(result["cuts"]) == 2


class TestRenderFrames:
    @pytest.mark.parametrize("count", [1, 21, 22, 123, 124, 125, 141, 142, 344, 345])
    def test_render_frames_is_on_the_grid_and_covers_the_cut(self, window, count):
        frames = scene_cuts.render_frames(count, window)
        assert frames % window["modulus"] == window["remainder"]
        assert window["min_frames"] <= frames <= window["max_frames"]
        assert frames >= count
        assert frames == max(window["min_frames"], aligned(count, window))

    def test_known_values(self, window):
        assert scene_cuts.render_frames(10, window) == 124
        assert scene_cuts.render_frames(124, window) == 124
        assert scene_cuts.render_frames(125, window) == 141
        assert scene_cuts.render_frames(345, window) == 345

    def test_the_window_is_read_off_the_pipeline(self, window):
        assert (window["min_frames"], window["max_frames"]) == (124, 345)
        assert (window["modulus"], window["remainder"]) == (17, 5)

    def test_every_cut_carries_its_render_frames(self, window):
        result = detect_cuts(clip(solid((200, 30, 30), 40) + solid((30, 30, 200), 130)))
        assert [c["render_frames"] for c in result["cuts"]] == [124, 141]


class TestSplits:
    def test_a_long_shot_splits_into_near_equal_pieces(self, window):
        result = detect_cuts(clip([textured()] * 400), max_frames=150)
        pieces = lengths(result)
        assert sum(pieces) == 400 and max(pieces) <= 150
        assert max(pieces) - min(pieces) <= 1 and len(pieces) == 3
        for cut in result["cuts"]:
            assert cut["split"] is True
            assert cut["render_frames"] % window["modulus"] == window["remainder"]
        starts = [c["start_frame"] for c in result["cuts"]]
        assert starts == [0, pieces[0], pieces[0] + pieces[1]]

    def test_only_the_long_shot_is_split(self, window):
        pictures = solid((200, 30, 30), 40) + solid((30, 30, 200), 200)
        cuts = detect_cuts(clip(pictures), max_frames=124)["cuts"]
        assert [c["num_frames"] for c in cuts] == [40, 100, 100]
        assert "split" not in cuts[0]
        assert all(c["split"] for c in cuts[1:])
        assert [c["name"] for c in cuts] == ["cut-1", "cut-2", "cut-3"]

    def test_the_default_ceiling_is_345(self, window):
        cuts = detect_cuts(clip([textured()] * 346))["cuts"]
        assert [c["num_frames"] for c in cuts] == [173, 173]
        assert detect_cuts(clip([textured()] * 345))["cuts"][0]["num_frames"] == 345

    def test_a_split_warns_the_caller(self, window):
        from unittest.mock import patch

        with patch.object(scene_cuts, "emit_warning") as warn:
            detect_cuts(clip([textured()] * 400), max_frames=200)
        assert any("split" in call.args[0] for call in warn.call_args_list)

    @pytest.mark.parametrize("max_frames", [346, 1000])
    def test_max_frames_past_the_ceiling_is_refused_at_run_time(
        self, window, max_frames
    ):
        with pytest.raises(ValueError, match="max_frames"):
            detect_cuts(clip([textured()] * 10), max_frames=max_frames)

    @pytest.mark.parametrize("max_frames", [0, -1, 1.5, "x"])
    def test_a_bad_max_frames_is_refused(self, max_frames):
        with pytest.raises(ValueError, match="max_frames"):
            detect_cuts(clip([textured()] * 10), max_frames=max_frames)


class TestTooShort:
    def test_a_ten_frame_shot_is_flagged(self, window):
        pictures = solid((200, 30, 30), 40) + solid((30, 30, 200), 10)
        pictures += solid((30, 200, 30), 40)
        cuts = detect_cuts(clip(pictures))["cuts"]
        assert [c["num_frames"] for c in cuts] == [40, 10, 40]
        assert cuts[1]["too_short"] is True
        assert "too_short" not in cuts[0] and "too_short" not in cuts[2]
        assert cuts[1]["render_frames"] == window["min_frames"]

    def test_the_threshold_is_the_grids_first_step(self, window):
        assert scene_cuts.too_short_below(window) == 22
        at = detect_cuts(clip(solid((200, 30, 30), 22)))["cuts"][0]
        under = detect_cuts(clip(solid((200, 30, 30), 21)))["cuts"][0]
        assert "too_short" not in at and under["too_short"] is True

    def test_too_short_warns_the_caller(self, window):
        from unittest.mock import patch

        with patch.object(scene_cuts, "emit_warning") as warn:
            detect_cuts(clip(solid((200, 30, 30), 10)))
        assert any("cut-1" in call.args[0] for call in warn.call_args_list)


def workflow(arguments):
    return {
        "id": "x",
        "steps": [
            {"name": "s", "task": {"command": "detect_cuts", "arguments": arguments}}
        ],
    }


class TestValidation:
    def test_registered_as_a_json_task(self):
        from dw.tasks.registry import _COMMAND_INFO
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "detect_cuts" in _COMMAND_REGISTRY
        info = _COMMAND_INFO["detect_cuts"]
        assert info.get("returns") == "json"
        assert not info.get("required_weights")

    def test_the_task_describes_itself(self):
        from dw.introspection import describe_task

        described = describe_task("detect_cuts")
        assert "video" in str(described) and "max_frames" in str(described)

    @pytest.mark.parametrize("max_frames", [346, 0, 1.5, "x"])
    def test_a_bad_max_frames_is_an_error_at_its_argument(self, window, max_frames):
        errors = task_argument_errors(
            workflow({"video": "v.mp4", "max_frames": max_frames})
        )
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.max_frames"]

    @pytest.mark.parametrize("max_frames", [345, 124, 1, "variable:m"])
    def test_a_good_max_frames_passes(self, max_frames):
        errors = task_argument_errors(
            workflow({"video": "v.mp4", "max_frames": max_frames})
        )
        assert errors == []

    def test_a_still_image_is_refused_naming_the_argument(self):
        from dw.video_extensions import video_extension_errors

        errors = video_extension_errors(workflow({"video": "still.png"}))
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.video"]

    def test_a_json_result_saves_as_json(self):
        from dw.scalar_result_validation import scalar_result_errors

        definition = workflow({"video": "v.mp4"})
        definition["steps"][0]["result"] = {"content_type": "video/mp4"}
        assert scalar_result_errors(definition)
        definition["steps"][0]["result"] = {"content_type": "application/json"}
        assert scalar_result_errors(definition) == []
