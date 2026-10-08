"""
Unit tests for crop_face_track (dw/tasks/face_track.py) and its rules in
dw/task_domains.py. The YuNet detector is stubbed: a coloured square drawn
into synthetic frames is the "face", and the stub finds that colour's bounding
box in whatever array it is handed (the whole frame or an enlarged tile).
"""

import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

import dw.tasks.face_track as ft
from dw.media_types import AudioVideo, JsonRecord
from dw.result import Result
from dw.security import SecurityError
from dw.task_domains import task_argument_errors

W, H = 320, 240
FACE = (255, 0, 0)


def make_frame(face=None, background=(40, 160, 60)):
    """An RGB uint8 frame; face is (x, y, size) of a red square, or None."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = background
    if face is not None:
        x, y, s = face
        frame[y : y + s, x : x + s] = FACE
    return frame


def stub_detect(rgb):
    mask = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return [[float(x0), float(y0), float(x1 - x0), float(y1 - y0), 0.9]]


@pytest.fixture
def stub_detector(monkeypatch):
    monkeypatch.setattr(ft, "_detector", lambda *a, **k: stub_detect)


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(
        ft, "emit_warning", lambda message, **data: seen.append((message, data))
    )
    return seen


def clip_of(arrays, shots=None, fps=24):
    return AudioVideo(
        [Image.fromarray(a) for a in arrays], None, None, fps=fps, shots=shots
    )


def det(x=100, y=100, w=16, h=16, score=0.9):
    return [x, y, w, h, score]


class TestNms:
    def test_keeps_the_most_confident_of_overlapping_boxes(self):
        kept = ft.nms([[10, 10, 20, 20, 0.7], [11, 11, 20, 20, 0.95]])
        assert len(kept) == 1
        assert kept[0][4] == 0.95

    def test_keeps_non_overlapping_boxes(self):
        kept = ft.nms([[0, 0, 10, 10, 0.8], [100, 100, 10, 10, 0.9]])
        assert len(kept) == 2
        assert kept[0][4] == 0.9

    def test_no_detections_is_empty(self):
        assert ft.nms([]) == []

    def test_float_boxes_are_suppressed_by_their_overlap(self):
        # cv2.dnn.NMSBoxes reads float rows as Rect2d, so sub-pixel boxes
        # from the enlarged tiles still merge with the whole-frame copy
        kept = ft.nms([[10.25, 10.5, 20.0, 20.0, 0.8], [10.0, 10.0, 20.5, 20.0, 0.6]])
        assert kept == [[10.25, 10.5, 20.0, 20.0, 0.8]]


class TestDetectTiled:
    def test_one_face_found_once_near_its_true_box(self):
        frame = make_frame((10, 10, 20))
        found = ft.detect_tiled(stub_detect, frame)
        assert len(found) == 1
        x, y, w, h, _ = found[0]
        assert abs(x - 10) <= 2 and abs(y - 10) <= 2
        assert abs(w - 20) <= 3 and abs(h - 20) <= 3

    def test_a_tile_only_detection_maps_back_to_frame_coords(self):
        # the whole-frame search sees nothing; only an enlarged tile does
        calls = []

        def detect(rgb):
            calls.append(rgb.shape)
            if rgb.shape[:2] == (H, W):
                return []
            return [[20.0, 30.0, 40.0, 40.0, 0.9]]

        found = ft.detect_tiled(detect, make_frame())
        assert len(calls) == 5
        # every tile reports the same tile-local box, each mapping to a different
        # frame position; the top-left tile's maps to (10, 15, 20, 20)
        assert any(
            abs(f[0] - 10) < 1e-6 and abs(f[1] - 15) < 1e-6 and f[2] == 20
            for f in found
        )


class TestTracking:
    def test_ema_moves_partway_to_a_jumped_face(self):
        dets = [[det(100, 100)], [det(110, 100)]]
        frames, _ = ft.track_faces(dets, W, {}, 0.06, 0.12)
        assert frames[0]["box"][0] == 100
        expected = 100 + ft.EMA_ALPHA * 10
        assert frames[1]["box"][0] == pytest.approx(expected, abs=0.01)
        assert frames[1]["state"] == "tracked"

    def test_steady_face_keeps_its_box(self):
        frames, resets = ft.track_faces([[det()]] * 5, W, {}, 0.06, 0.12)
        assert all(f["box"] == [100, 100, 16, 16] for f in frames)
        assert resets == []

    def test_gap_is_held_with_decaying_strength(self):
        dets = [[det()]] + [[]] * 3
        frames, resets = ft.track_faces(dets, W, {}, 0.06, 0.12)
        assert frames[0]["strength"] == 1.0
        for i in (1, 2, 3):
            assert frames[i]["state"] == "held"
            assert frames[i]["box"] == frames[0]["box"]
            assert frames[i]["strength"] == pytest.approx(ft.HOLD_DECAY**i, abs=1e-3)
        assert resets == []

    def test_gap_of_max_gap_is_still_held_and_beyond_is_lost(self):
        dets = [[det()]] + [[]] * (ft.MAX_GAP + 2)
        frames, resets = ft.track_faces(dets, W, {}, 0.06, 0.12)
        assert frames[ft.MAX_GAP]["state"] == "held"
        lost_at = ft.MAX_GAP + 1
        assert frames[lost_at]["state"] == "none"
        assert frames[lost_at]["box"] is None and frames[lost_at]["strength"] == 0.0
        assert resets == [{"frame": lost_at, "reason": "lost", "detail": None}]

    def test_recovery_after_a_gap_restores_full_strength(self):
        dets = [[det()], [], [det()]]
        frames, _ = ft.track_faces(dets, W, {}, 0.06, 0.12)
        assert frames[2]["state"] == "tracked"
        assert frames[2]["strength"] == 1.0

    def test_a_reset_restarts_the_box_without_smoothing(self):
        dets = [[det(100, 100)], [det(100, 100)], [det(200, 50)]]
        frames, resets = ft.track_faces(dets, W, {2: ("shot", "b")}, 0.06, 0.12)
        assert frames[2]["box"][:2] == [200, 50]
        assert resets == [{"frame": 2, "reason": "shot", "detail": "b"}]


class TestShotStarts:
    def test_only_starts_after_frame_zero(self):
        shots = [
            {"name": "a", "start_frame": 0},
            {"name": "b", "start_frame": 5},
            {"name": "c"},
        ]
        assert ft.shot_starts(shots) == {5: "b"}
        assert ft.shot_starts(None) == {}


class TestGate:
    def test_ramp(self):
        assert ft.gate_strength(6, 100, 0.06, 0.12) == 1.0
        assert ft.gate_strength(3, 100, 0.06, 0.12) == 1.0
        assert ft.gate_strength(12, 100, 0.06, 0.12) == 0.0
        assert ft.gate_strength(50, 100, 0.06, 0.12) == 0.0
        assert ft.gate_strength(9, 100, 0.06, 0.12) == pytest.approx(0.5)

    def test_small_face_is_full_and_frame_filling_face_is_zero(self, stub_detector):
        small = ft.crop_face_track(clip_of([make_frame((100, 100, 12))] * 3))
        assert [f["strength"] for f in small["track"]["frames"]] == [1.0] * 3

        big = ft.crop_face_track(clip_of([make_frame((0, 0, 230))] * 3))
        assert [f["strength"] for f in big["track"]["frames"]] == [0.0] * 3
        assert big["track"]["face_found"] is True


class TestHistogramCuts:
    def test_cut_at_a_colour_change(self):
        a = make_frame((100, 100, 16), background=(40, 160, 60))
        b = make_frame((100, 100, 16), background=(200, 40, 200))
        cuts = ft.content_cuts([a, a, a, b, b])
        assert [index for index, _ in cuts] == [3]

    def test_no_spurious_cut_on_a_steady_clip(self):
        frame = make_frame((100, 100, 16))
        assert ft.content_cuts([frame] * 6) == []

    def test_end_to_end_cut_is_recorded_as_a_reset(self, stub_detector):
        a = make_frame((100, 100, 16), background=(40, 160, 60))
        b = make_frame((100, 100, 16), background=(200, 40, 200))
        result = ft.crop_face_track(clip_of([a, a, a, b, b]))
        resets = result["track"]["resets"]
        assert [(r["frame"], r["reason"]) for r in resets] == [(3, "cut")]


def framed_portrait(scale):
    """A textured portrait with a face square, shrunk by scale and letterboxed
    into a black W x H frame - two scales are two framings of one picture."""
    ph, pw = H, int(H * 0.75)
    y, x = np.mgrid[0:ph, 0:pw]
    portrait = np.stack(
        [
            90 + 60 * np.sin(x / 20),
            70 + 50 * np.cos(y / 16),
            60 + 40 * np.sin((x + y) / 30),
        ],
        axis=-1,
    ).astype(np.uint8)
    portrait[30:78, pw // 2 - 24 : pw // 2 + 24] = FACE
    small = np.asarray(
        Image.fromarray(portrait).resize(
            (int(pw * scale), int(ph * scale)), Image.Resampling.NEAREST
        )
    )
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    y0, x0 = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
    frame[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    return frame


class TestFramingCut:
    """A cut between two framings of one picture keeps its colours: the
    histogram barely moves, and the thumbnail difference spike finds it."""

    def test_zoom_cut_is_found_though_the_histogram_holds(self):
        far, near = framed_portrait(1 / 3.2), framed_portrait(1.0)
        cuts = ft.content_cuts([far] * 6 + [near] * 6)
        assert [index for index, _ in cuts] == [6]
        assert cuts[0][1]["histogram_correlation"] > ft.HISTOGRAM_CUT
        assert cuts[0][1]["frame_change"] >= ft.CUT_MIN_CHANGE

    def test_steady_pan_is_not_a_cut(self):
        near = framed_portrait(1.0)
        pan = [np.roll(near, 3 * i, axis=1) for i in range(20)]
        assert ft.content_cuts(pan) == []

    def test_zoom_cut_resets_the_track_with_no_held_frames(self, stub_detector):
        far, near = framed_portrait(1 / 3.2), framed_portrait(1.0)
        result = ft.crop_face_track(clip_of([far] * 6 + [near] * 6))
        track = result["track"]
        assert [(r["frame"], r["reason"]) for r in track["resets"]] == [(6, "cut")]
        states = [f["state"] for f in track["frames"]]
        assert states == ["tracked"] * 12
        far_width, near_width = (
            track["frames"][5]["box"][2],
            track["frames"][6]["box"][2],
        )
        assert near_width > 2.5 * far_width


class TestShotReset:
    def test_recorded_shot_start_resets_the_track(self, stub_detector):
        frames = [make_frame((100, 100, 16))] * 3 + [make_frame((200, 50, 16))] * 3
        shots = [
            {"name": "one", "start_frame": 0},
            {"name": "two", "start_frame": 3},
        ]
        result = ft.crop_face_track(clip_of(frames, shots=shots))
        track = result["track"]
        assert {"frame": 3, "reason": "shot", "detail": "two"} in track["resets"]
        # the box starts afresh at the new face rather than gliding to it
        assert track["frames"][3]["box"][0] == pytest.approx(200, abs=3)


class TestPadding:
    @pytest.mark.parametrize(
        "count,expected",
        [
            (1, (0, 0)),
            (8, (0, 1)),
            (9, (0, 0)),
            (10, (3, 4)),
            (17, (0, 0)),
            (30, (1, 2)),
        ],
    )
    def test_padding_to_grid_8n1(self, count, expected):
        before, after = ft.padding_to_grid(count, 8, 1)
        assert (before, after) == expected
        assert (count + before + after - 1) % 8 == 0

    @pytest.mark.parametrize("modulus,remainder", [(16, 0), (17, 5), (1, 0), (4, 3)])
    @pytest.mark.parametrize("count", [1, 3, 5, 6, 16, 17, 21, 22, 40])
    def test_padding_to_grid_is_the_next_grid_count(self, count, modulus, remainder):
        before, after = ft.padding_to_grid(count, modulus, remainder)
        total = count + before + after
        assert total >= count and total % modulus == remainder
        # the smallest such count, split evenly
        assert total - count < modulus
        assert before == (total - count) // 2

    def test_padding_to_grid_leaves_a_count_already_on_it(self):
        assert ft.padding_to_grid(21, 16, 5) == (0, 0)
        assert ft.padding_to_grid(32, 16, 0) == (0, 0)
        # below the remainder: pad up to it
        assert ft.padding_to_grid(2, 17, 5) == (1, 2)

    def test_pad_frames_mirrors_the_ends(self):
        crops = list(range(10, 15))  # 5 frames
        padded = ft.pad_frames(crops, 2, 3)
        assert len(padded) == 10
        assert padded[2:7] == crops
        assert padded[:2] == [12, 11]
        assert padded[7:] == [13, 12, 11]

    def test_pad_frames_single_frame(self):
        assert ft.pad_frames(["x"], 0, 0) == ["x"]
        assert ft.pad_frames(["x"], 3, 4) == ["x"] * 8

    def test_pad_frames_without_padding_is_identity(self):
        assert ft.pad_frames([1, 2, 3], 0, 0) == [1, 2, 3]


class TestCropFaceTrack:
    def test_end_to_end_shapes_and_record(self, stub_detector):
        n = 5
        result = ft.crop_face_track(
            clip_of([make_frame((100, 80, 16))] * n, fps=24), crop_size=64
        )
        crops, track = result["crops"], result["track"]
        assert isinstance(crops, AudioVideo)
        assert crops.audio is None
        assert crops.fps == 24
        assert len(crops.frames) == 9
        for frame in crops.frames:
            assert isinstance(frame, Image.Image) and frame.size == (64, 64)

        assert isinstance(track, JsonRecord)
        json.dumps(track)
        assert len(track["frames"]) == n
        for entry in track["frames"]:
            assert set(entry) >= {"box", "crop", "strength", "state"}
            assert entry["crop"][2] == entry["crop"][3]
        assert track["crop_frames"] == 9
        assert track["pad_before"] + track["pad_after"] == 4
        assert track["source"] == {"width": W, "height": H, "frames": n, "fps": 24}
        assert track["face_found"] is True
        assert "message" not in track

    def test_a_single_frame_clip_works(self, stub_detector):
        result = ft.crop_face_track(clip_of([make_frame((100, 80, 16))]), crop_size=32)
        assert len(result["crops"].frames) == 1
        assert len(result["track"]["frames"]) == 1

    def test_no_face_succeeds_with_zero_strength_and_a_warning(
        self, monkeypatch, warnings_seen
    ):
        monkeypatch.setattr(ft, "_detector", lambda *a, **k: lambda rgb: [])
        result = ft.crop_face_track(clip_of([make_frame()] * 4), crop_size=64)
        track = result["track"]
        assert track["face_found"] is False
        assert all(f["strength"] == 0.0 for f in track["frames"])
        assert "no face" in track["message"].lower()
        assert len(result["crops"].frames) == 9
        assert [d["kind"] for _, d in warnings_seen] == ["no_face_found"]

    def test_the_declared_grid_sets_the_crop_count(self, stub_detector):
        clip = clip_of([make_frame((100, 80, 16))] * 5)
        result = ft.crop_face_track(
            clip, crop_size=48, modulus=16, remainder=0, multiple=16
        )
        assert len(result["crops"].frames) == 16
        assert result["track"]["crop_frames"] == 16
        assert result["crops"].frames[0].size == (48, 48)

    def test_bad_arguments_are_refused_at_run_time(self, stub_detector):
        clip = clip_of([make_frame((100, 80, 16))])
        with pytest.raises(ValueError, match="crop_size"):
            ft.crop_face_track(clip, crop_size=500)
        with pytest.raises(ValueError, match="gate_zero"):
            ft.crop_face_track(clip, gate_full=0.2, gate_zero=0.1)
        with pytest.raises(ValueError, match="gate_zero"):
            ft.crop_face_track(clip, gate_full=0.1, gate_zero=0.1)
        with pytest.raises(ValueError, match="remainder"):
            ft.crop_face_track(clip, remainder=8)
        with pytest.raises(ValueError, match="multiple of 16"):
            ft.crop_face_track(clip, crop_size=520, multiple=16)


class TestDetectorSource:
    @pytest.fixture(autouse=True)
    def no_download(self, monkeypatch):
        import huggingface_hub

        def fail(*a, **k):
            pytest.fail("a refused detector source reached the download")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)

    def test_traversing_repo_is_refused(self):
        with pytest.raises(SecurityError):
            ft._detector("../evil", "x.onnx", 0.6, "cpu")

    def test_non_onnx_file_is_refused(self):
        with pytest.raises(SecurityError):
            ft._detector("opencv/face_detection_yunet", "weights.bin", 0.6, "cpu")


def workflow(arguments):
    return {
        "id": "x",
        "steps": [
            {
                "name": "s",
                "task": {"command": "crop_face_track", "arguments": arguments},
            }
        ],
    }


class TestStaticValidation:
    @pytest.mark.parametrize(
        "arguments,name",
        [
            ({"clip": "v.mp4", "crop_size": 500}, "crop_size"),
            ({"clip": "v.mp4", "gate_full": 0.2, "gate_zero": 0.1}, "gate_zero"),
            ({"clip": "v.mp4", "gate_full": 0.1, "gate_zero": 0.1}, "gate_zero"),
            ({"clip": "v.mp4", "crop_size": 48}, "crop_size"),
            ({"clip": "v.mp4", "modulus": 0}, "modulus"),
            ({"clip": "v.mp4", "remainder": -1}, "remainder"),
            ({"clip": "v.mp4", "remainder": 8}, "remainder"),
            ({"clip": "v.mp4", "modulus": 4, "remainder": 4}, "remainder"),
            ({"clip": "v.mp4", "multiple": 0}, "multiple"),
            ({"clip": "v.mp4", "multiple": 2.5}, "multiple"),
            ({"clip": "v.mp4", "padding": 4}, "padding"),
            ({"clip": "v.mp4", "padding": -1}, "padding"),
            ({"clip": "v.mp4", "detector_repo": "../x"}, "detector_repo"),
            ({"clip": "v.mp4", "detector_file": "w.bin"}, "detector_file"),
        ],
    )
    def test_bad_literal_is_an_error_at_its_argument(self, arguments, name):
        errors = task_argument_errors(workflow(arguments))
        path = f"steps[0].task.arguments.{name}"
        matching = [e for e in errors if e["path"] == path]
        assert matching, errors
        for error in errors:
            # the refusal may quote the offending value, but adds no traversal text
            assert ".." not in error["message"].replace("'../x'", "")

    def test_valid_workflow_has_no_errors(self):
        args = {
            "clip": "v.mp4",
            "crop_size": 512,
            "padding": 0.6,
            "gate_full": 0.06,
            "gate_zero": 0.12,
            "min_confidence": 0.6,
            "detector_repo": "opencv/face_detection_yunet",
            "detector_file": "face_detection_yunet_2023mar.onnx",
        }
        assert task_argument_errors(workflow(args)) == []

    def test_a_declared_grid_is_accepted(self):
        args = {"clip": "v.mp4", "crop_size": 48, "multiple": 16, "modulus": 16}
        assert task_argument_errors(workflow({**args, "remainder": 0})) == []
        assert task_argument_errors(workflow({**args, "remainder": 15})) == []

    @pytest.mark.parametrize(
        "arguments",
        [
            {"crop_size": 48, "multiple": "previous_result:x.m"},
            {"crop_size": 48, "multiple": "variable:m"},
            {"remainder": 12, "modulus": "variable:m"},
            {"remainder": "variable:r", "modulus": 4},
        ],
    )
    def test_a_deferred_grid_value_skips_its_rule(self, arguments):
        assert task_argument_errors(workflow({"clip": "v.mp4", **arguments})) == []

    def test_the_task_describes_its_grid_defaults(self):
        from dw.introspection import describe_task

        found = {p["name"]: p for p in describe_task("crop_face_track")["parameters"]}
        defaults = [found[n]["default"] for n in ("modulus", "remainder", "multiple")]
        assert defaults == [8, 1, 32]
        for name in ("modulus", "remainder", "multiple"):
            assert found[name]["description"]

    def test_variable_references_are_not_judged(self):
        args = {
            name: f"variable:{name}"
            for name in (
                "crop_size",
                "padding",
                "gate_full",
                "gate_zero",
                "modulus",
                "remainder",
                "multiple",
                "detector_repo",
                "detector_file",
            )
        }
        args["clip"] = "v.mp4"
        assert task_argument_errors(workflow(args)) == []


def test_command_is_registered():
    from dw.tasks.task import _COMMAND_REGISTRY

    assert "crop_face_track" in _COMMAND_REGISTRY


def test_json_record_is_saved_whole_under_a_video_content_type():
    record = JsonRecord({"a": 1, "frames": [{"box": None}], "nested": {"k": "v"}})
    with tempfile.TemporaryDirectory() as temp_dir:
        result = Result({"content_type": "video/mp4", "save": True})
        result.add_result(record)
        result.save(temp_dir, "wf-step.0")
        names = os.listdir(temp_dir)
        assert len(names) == 1 and names[0].endswith(".json")
        with open(os.path.join(temp_dir, names[0])) as file:
            assert json.load(file) == record


def test_the_task_result_saves_crops_as_video_and_track_as_one_json():
    crops = AudioVideo([Image.new("RGB", (64, 64))] * 9, None, None, fps=24)
    with tempfile.TemporaryDirectory() as temp_dir:
        result = Result({"content_type": "video/mp4", "save": True})
        result.add_result({"crops": crops, "track": JsonRecord({"a": {"b": 1}})})
        result.save(temp_dir, "wf-step.0")
        names = sorted(os.listdir(temp_dir))
        assert [os.path.splitext(n)[1] for n in names] == [".mp4", ".json"]
        with open(os.path.join(temp_dir, names[1])) as file:
            assert json.load(file) == {"a": {"b": 1}}


# ---- paste_face_track ----------------------------------------------------


def gradient_frame(face=None):
    """Smooth content (a gradient) with an optional red face square."""
    xs = np.linspace(20, 180, W, dtype=np.float32)[None, :]
    ys = np.linspace(20, 120, H, dtype=np.float32)[:, None]
    frame = np.stack([xs + 0 * ys, ys + 0 * xs, (xs + ys) / 2], axis=-1).astype(
        np.uint8
    )
    if face is not None:
        x, y, s = face
        frame[y : y + s, x : x + s] = FACE
    return frame


def tracked(frames, **clip_kwargs):
    clip = clip_of(frames, **clip_kwargs)
    result = ft.crop_face_track(clip)
    return clip, result["crops"], result["track"]


def diffs(out, source_arrays):
    return [
        np.abs(np.asarray(f).astype(int) - s.astype(int))
        for f, s in zip(out.frames, source_arrays)
    ]


SQUARE = [60, 60, 100, 100]


def crop_of(region):
    return Image.fromarray(region)


class TestFeatherMask:
    def test_centre_one_corners_zero(self):
        mask = ft.feather_mask(101, 0.3)
        assert mask[50, 50] == pytest.approx(1.0)
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            assert mask[corner] == 0

    def test_non_increasing_along_radius(self):
        mask = ft.feather_mask(101, 0.3)
        row = mask[50, 50:]
        assert np.all(np.diff(row) <= 1e-6)
        diagonal = np.array([mask[50 + i, 50 + i] for i in range(51)])
        assert np.all(np.diff(diagonal) <= 1e-6)

    def test_symmetric(self):
        mask = ft.feather_mask(64, 0.4)
        assert np.allclose(mask, mask[::-1, :])
        assert np.allclose(mask, mask[:, ::-1])
        assert np.allclose(mask, mask.T)

    def test_zero_feather_is_a_hard_disc(self):
        mask = ft.feather_mask(64, 0)
        assert set(np.unique(mask)) == {0.0, 1.0}
        assert mask[32, 32] == 1 and mask[0, 0] == 0

    def test_inner_part_is_exactly_one(self):
        side = 128
        mask = ft.feather_mask(side, 0.3)
        radius = side / 2
        coords = np.arange(side) + 0.5 - radius
        distance = np.hypot(coords[None, :], coords[:, None]) / radius
        inner = distance <= 0.7 - 1e-6
        assert inner.any()
        assert np.all(mask[inner] == 1.0)
        assert mask[(distance > 0.75) & (distance < 0.99)].max() < 1.0


class TestPasteCrop:
    def source_and_region(self):
        source = gradient_frame()
        x, y, side, _ = SQUARE
        return source, source[y : y + side, x : x + side].copy()

    def test_color_match_removes_a_constant_shift(self):
        source, region = self.source_and_region()
        shifted = region.copy()
        shifted[..., 0] += 40
        out = ft.paste_crop(source, crop_of(shifted), SQUARE, 1.0, 0.3, True)
        x, y, side, _ = SQUARE
        mask = ft.feather_mask(side, 0.3) > 0
        diff = np.abs(out[y : y + side, x : x + side].astype(int) - region.astype(int))
        assert diff[mask].max() <= 2

    def test_no_color_match_shows_the_shift(self):
        source, region = self.source_and_region()
        shifted = region.copy()
        shifted[..., 0] += 40
        out = ft.paste_crop(source, crop_of(shifted), SQUARE, 1.0, 0.3, False)
        centre = out[110, 110, 0].astype(int) - source[110, 110, 0].astype(int)
        assert centre == 40

    def test_altered_repair_only_around_the_face_without_a_seam(self):
        source = gradient_frame()
        white = np.full((100, 100, 3), 255, np.uint8)
        out = ft.paste_crop(source, crop_of(white), SQUARE, 1.0, 0.3, False)
        changed = np.any(out != source, axis=-1)
        x, y, side, _ = SQUARE
        # nothing outside the square moves
        outside = changed.copy()
        outside[y : y + side, x : x + side] = False
        assert not outside.any()
        # the corners of the square keep the source
        assert not changed[y, x] and not changed[y + side - 1, x + side - 1]
        # the centre is fully the repair
        assert tuple(out[110, 110]) == (255, 255, 255)
        # alpha ramps: no step between neighbours bigger than the ramp allows
        step = np.abs(np.diff(out[110, x : x + side, 0].astype(int)))
        assert step.max() <= 20

    def test_strength_scales_the_blend(self):
        source = gradient_frame()
        white = np.full((100, 100, 3), 255, np.uint8)
        out = ft.paste_crop(source, crop_of(white), SQUARE, 0.5, 0.3, False)
        expected = (int(source[110, 110, 0]) + 255) / 2
        assert abs(int(out[110, 110, 0]) - expected) <= 1


class TestPasteFaceTrack:
    def frames(self, count=5):
        return [gradient_frame((140 + i, 100, 16)) for i in range(count)]

    def test_identity_round_trip(self, stub_detector):
        arrays = self.frames()
        clip, crops, track = tracked(arrays)
        assert track["frames"][0]["strength"] > 0
        out = ft.paste_face_track(clip, crops, track)
        assert len(out.frames) == len(arrays)
        assert all(f.size == (W, H) for f in out.frames)
        d = diffs(out, arrays)
        assert max(x.mean() for x in d) < 0.1
        assert max(x.max() for x in d) <= 32

    def test_strength_zero_is_an_exact_passthrough(self, stub_detector):
        arrays = self.frames()
        clip, crops, track = tracked(arrays)
        for entry in track["frames"]:
            entry["strength"] = 0
        white = AudioVideo(
            [Image.new("RGB", (512, 512), (255, 255, 255)) for _ in crops.frames],
            None,
            None,
        )
        out = ft.paste_face_track(clip, white, track)
        for frame, source in zip(out.frames, arrays):
            assert np.array_equal(np.asarray(frame), source)

    def test_audio_fps_and_shots_are_carried(self, stub_detector):
        arrays = self.frames()
        shots = [
            {"name": "one", "start_frame": 0},
            {"name": "two", "start_frame": 3},
        ]
        clip, crops, track = tracked(arrays, shots=shots, fps=24)
        clip.audio = np.zeros((2, 48000), np.float32)
        clip.sample_rate = 48000
        out = ft.paste_face_track(clip, crops, track)
        assert np.array_equal(out.audio, clip.audio)
        assert out.sample_rate == 48000
        assert out.fps == 24
        assert out.shots == shots
        assert out.shots is not clip.shots

    def test_altered_repair_changes_the_face_only(self, stub_detector):
        arrays = self.frames()
        clip, crops, track = tracked(arrays)
        white = AudioVideo(
            [Image.new("RGB", f.size, (255, 255, 255)) for f in crops.frames],
            None,
            None,
        )
        out = ft.paste_face_track(clip, white, track, color_match=False)
        x, y, side, _ = (int(v) for v in track["frames"][0]["crop"])
        frame = np.asarray(out.frames[0])
        assert np.array_equal(frame[:10], arrays[0][:10])
        assert not np.array_equal(
            frame[y + side // 2, x + side // 2], arrays[0][y + side // 2, x + side // 2]
        )

    def test_track_read_from_saved_json(self, stub_detector, tmp_path):
        arrays = self.frames()
        clip, crops, track = tracked(arrays)
        path = tmp_path / "track.json"
        path.write_text(json.dumps(track))
        from_dict = ft.paste_face_track(clip, crops, track)
        from_file = ft.paste_face_track(clip, crops, str(path))
        for a, b in zip(from_dict.frames, from_file.frames):
            assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_track_path_with_traversal_is_refused(self, stub_detector):
        arrays = self.frames()
        clip, crops, track = tracked(arrays)
        with pytest.raises(SecurityError):
            ft.paste_face_track(clip, crops, "../track.json")

    def test_refusals(self, stub_detector):
        arrays = self.frames()
        clip, crops, track = tracked(arrays)

        with pytest.raises(ValueError, match="frames but this clip has"):
            ft.paste_face_track(clip_of(arrays[:3]), crops, track)

        small = clip_of([a[:200, :300] for a in arrays])
        with pytest.raises(ValueError, match="clip but this clip is"):
            ft.paste_face_track(small, crops, track)

        short = AudioVideo(crops.frames[:-1], None, None)
        with pytest.raises(ValueError, match="must keep every crop"):
            ft.paste_face_track(clip, short, track)

        with pytest.raises(ValueError, match="track record"):
            ft.paste_face_track(clip, crops, {})

        with pytest.raises(ValueError, match="feather"):
            ft.paste_face_track(clip, crops, track, feather=1.5)


class TestPasteValidation:
    @staticmethod
    def paste_workflow(feather):
        return {
            "id": "x",
            "steps": [
                {
                    "name": "s",
                    "task": {
                        "command": "paste_face_track",
                        "arguments": {
                            "clip": "v.mp4",
                            "repaired": "r.mp4",
                            "track": "t.json",
                            "feather": feather,
                        },
                    },
                }
            ],
        }

    @pytest.mark.parametrize("feather", [1.5, -0.1])
    def test_bad_feather_is_refused(self, feather):
        errors = task_argument_errors(self.paste_workflow(feather))
        assert any(e["path"] == "steps[0].task.arguments.feather" for e in errors)

    def test_good_feather_is_accepted(self):
        assert task_argument_errors(self.paste_workflow(0.3)) == []


def test_paste_command_is_registered():
    from dw.tasks.task import _COMMAND_REGISTRY

    assert "paste_face_track" in _COMMAND_REGISTRY
