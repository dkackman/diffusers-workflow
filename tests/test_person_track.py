"""track_people (#1375): one mask video per person, SAM2 video propagation per
cut. The transformers classes are faked at the module's own names, with
behaviour derived from the frames, so the cuts, numbering, exclusive masks,
assembly and contact sheet are the real code."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from dw.media_types import AudioVideo, JsonRecord
from dw.shots import shot_record
from dw.tasks import person_track
from dw.tasks.model_cache import clear_model_cache
from dw.tasks.person_track import _exclusive_masks, distinct_boxes, track_people

W, H = 160, 96
BACKGROUND = (200, 200, 200)
RED = (220, 30, 30)  # the larger figure, moving right
BLUE = (30, 30, 220)  # the smaller figure, moving left
GREEN = (30, 200, 30)  # a third, smallest figure, still


def rect_a(i):
    return (10 + 4 * i, 5, 50 + 4 * i, 45)  # 40 x 40


def rect_b(i):
    return (110 - 4 * i, 55, 140 - 4 * i, 85)  # 30 x 30


def rect_c(i):
    return (60, 50, 80, 70)  # 20 x 20


def make_frame(i, figures=("A", "B")):
    pixels = np.full((H, W, 3), BACKGROUND, np.uint8)
    table = {"A": (rect_a, RED), "B": (rect_b, BLUE), "C": (rect_c, GREEN)}
    for key in figures:
        rect, colour = table[key]
        x0, y0, x1, y1 = rect(i)
        pixels[y0:y1, x0:x1] = colour
    return Image.fromarray(pixels)


def make_video(count=10, figures=("A", "B"), shots=None):
    frames = [make_frame(i, figures) for i in range(count)]
    return AudioVideo(frames, None, None, fps=24, shots=shots)


def colour_box(image, colour):
    """(x0, y0, x1, y1) of the pixels of one colour, or None."""
    hit = (np.asarray(image.convert("RGB")) == colour).all(axis=-1)
    if not hit.any():
        return None
    ys, xs = np.nonzero(hit)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def colour_mask(image, colour):
    return (np.asarray(image.convert("RGB")) == colour).all(axis=-1)


COLOURS = (RED, BLUE, GREEN)


class FakeDino:
    """Stands in for AutoProcessor / AutoModelForZeroShotObjectDetection: the
    boxes are those of every figure colour in the image it was handed."""

    def __init__(self):
        self.calls = 0
        self.image = None

    # AutoProcessor / AutoModelForZeroShotObjectDetection
    def from_pretrained(self, *_args, **_kwargs):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        if images is not None:
            self.image = images
            self.calls += 1
        encoding = MagicMock()
        encoding.__getitem__ = lambda _self, key: torch.zeros(1, 1, dtype=torch.long)
        encoding.to.return_value = encoding
        return encoding

    def post_process_grounded_object_detection(self, *_args, **_kwargs):
        boxes, scores = [], []
        for colour in COLOURS:
            box = colour_box(self.image, colour)
            if box is not None:
                boxes.append(list(box))
                scores.append(0.9)
        return [
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(scores, dtype=torch.float32),
            }
        ]


class FakeSession:
    def __init__(self, frames):
        self.frames = frames
        self.colours = []
        self.recorded_obj_ids = None


class FakeSam:
    """Stands in for Sam2VideoProcessor / Sam2VideoModel. Each object follows
    the colour found at the centre of its box on the first frame; its logits
    on every frame are positive where that colour actually is."""

    def __init__(self, bleed=False):
        self.bleed = bleed  # the second object also claims the first's figure
        self.sessions = []

    def from_pretrained(self, *_args, **_kwargs):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def init_video_session(self, video=None, **_kwargs):
        session = FakeSession(list(video))
        self.sessions.append(session)
        return session

    def add_inputs_to_inference_session(
        self, inference_session, frame_idx, obj_ids, input_boxes
    ):
        inference_session.recorded_obj_ids = list(obj_ids)
        first = inference_session.frames[frame_idx]
        pixels = np.asarray(first.convert("RGB"))
        for x0, y0, x1, y1 in input_boxes[0]:
            centre = tuple(
                int(v) for v in pixels[int((y0 + y1) / 2), int((x0 + x1) / 2)]
            )
            inference_session.colours.append(centre)
        obj_ids.clear()  # the real processor consumes the list it is given

    def __call__(self, inference_session=None, frame_idx=0, **_kwargs):
        return MagicMock()

    def propagate_in_video_iterator(self, session):
        for index, frame in enumerate(session.frames):
            rows = []
            for slot, colour in enumerate(session.colours):
                logits = np.where(colour_mask(frame, colour), 5.0, -5.0)
                if self.bleed and slot == 1:
                    logits = np.where(
                        colour_mask(frame, session.colours[0]), 1.0, logits
                    )
                rows.append(logits)
            masks = torch.tensor(np.stack(rows), dtype=torch.float32).unsqueeze(1)
            yield MagicMock(
                frame_idx=index,
                object_ids=session.recorded_obj_ids,
                pred_masks=masks,
            )

    def post_process_masks(self, masks, original_sizes=None, binarize=True):
        assert binarize is False
        return [masks[0]]


@pytest.fixture
def models():
    """Patch the four transformers classes; hand back the two fakes."""
    clear_model_cache()
    dino, sam = FakeDino(), FakeSam()
    with (
        patch.object(person_track, "AutoProcessor", dino),
        patch.object(person_track, "AutoModelForZeroShotObjectDetection", dino),
        patch.object(person_track, "Sam2VideoProcessor", sam),
        patch.object(person_track, "Sam2VideoModel", sam),
        patch.object(person_track, "content_cuts", return_value=[]),
    ):
        yield dino, sam
    clear_model_cache()


@pytest.fixture
def warnings():
    with patch.object(person_track, "emit_warning") as warn:
        yield warn


def person_names(result):
    return sorted(k for k in result if k.startswith("person-"))


def inside(mask_image, box):
    mask = np.asarray(mask_image) > 0
    x0, y0, x1, y1 = box
    box_area = (x1 - x0) * (y1 - y0)
    return mask[y0:y1, x0:x1].sum() / box_area


def outside_count(mask_image, box):
    mask = np.asarray(mask_image) > 0
    x0, y0, x1, y1 = box
    return int(mask.sum() - mask[y0:y1, x0:x1].sum())


class TestTwoFigures:
    def test_each_figure_has_its_own_mask_video(self, models, warnings):
        result = track_people(make_video(), prompt="person")
        assert person_names(result) == ["person-1", "person-2"]
        for name in ("person-1", "person-2"):
            assert isinstance(result[name], AudioVideo)
            assert len(result[name].frames) == 10
            assert result[name].fps == 24
            assert result[name].audio is None
            assert result[name].frames[0].mode == "L"
            assert result[name].frames[0].size == (W, H)

    def test_masks_follow_their_figure_to_the_last_frame(self, models, warnings):
        result = track_people(make_video(), prompt="person")
        for index in (0, 9):
            one = result["person-1"].frames[index]
            two = result["person-2"].frames[index]
            assert inside(one, rect_a(index)) > 0.95
            assert outside_count(one, rect_a(index)) == 0
            assert inside(two, rect_b(index)) > 0.95
            assert outside_count(two, rect_b(index)) == 0
            assert inside(one, rect_b(index)) == 0
            assert inside(two, rect_a(index)) == 0

    def test_person_1_is_the_larger_figure(self, models, warnings):
        result = track_people(make_video(), prompt="person")
        area_one = (np.asarray(result["person-1"].frames[0]) > 0).sum()
        area_two = (np.asarray(result["person-2"].frames[0]) > 0).sum()
        assert area_one == 40 * 40 and area_two == 30 * 30
        people = result["people"]["cuts"][0]["people"]
        assert [p["name"] for p in people] == ["person-1", "person-2"]
        assert people[0]["area"] > people[1]["area"]

    def test_the_masks_never_overlap(self, models, warnings):
        result = track_people(make_video(), prompt="person")
        for one, two in zip(result["person-1"].frames, result["person-2"].frames):
            assert not ((np.asarray(one) > 0) & (np.asarray(two) > 0)).any()

    def test_the_object_ids_are_handed_to_the_processor_as_1_to_n(self, models):
        track_people(make_video(), prompt="person")
        assert models[1].sessions[0].recorded_obj_ids == [1, 2]

    def test_the_people_record_and_contact_sheet(self, models, warnings):
        result = track_people(make_video(), prompt="person", threshold=0.4)
        assert isinstance(result["people"], JsonRecord)
        record = result["people"]
        assert record["tracked"] is True
        assert record["people"] == 2
        assert record["num_frames"] == 10
        assert record["threshold"] == 0.4
        assert [c["name"] for c in record["cuts"]] == ["shot-1"]
        assert isinstance(result["contact_sheet"], Image.Image)
        assert result["contact_sheet"].mode == "RGB"
        assert list(result)[-2:] == ["contact_sheet", "people"]

    def test_one_cut_does_not_warn_about_numbering(self, models, warnings):
        track_people(make_video(), prompt="person")
        warnings.assert_not_called()

    def test_a_video_file_reference_is_read(self, models, warnings):
        from dw.tasks.video_utils import VideoFileReference

        with patch.object(
            person_track, "load_audio_video", return_value=make_video()
        ) as load:
            result = track_people(VideoFileReference("clip.mp4"), prompt="person")
        load.assert_called_once()
        assert len(result["person-1"].frames) == 10


class TestOverlap:
    def test_a_contested_pixel_goes_to_the_higher_logit(self, models, warnings):
        models[1].bleed = True
        result = track_people(make_video(), prompt="person")
        for one, two in zip(result["person-1"].frames, result["person-2"].frames):
            a, b = np.asarray(one) > 0, np.asarray(two) > 0
            assert not (a & b).any()
        # B's logits (+1) also covered A's figure; A's (+5) won it
        assert inside(result["person-1"].frames[0], rect_a(0)) > 0.95
        assert outside_count(result["person-2"].frames[0], rect_b(0)) == 0

    def test_exclusive_masks_gives_the_pixel_to_the_higher_one(self):
        logits = np.array(
            [
                [[3.0, 3.0, -1.0, -1.0]],
                [[1.0, 4.0, 2.0, -1.0]],
            ]
        )
        first, second = _exclusive_masks(logits, [0, 1])
        assert first.tolist() == [[True, False, False, False]]
        assert second.tolist() == [[False, True, True, False]]

    def test_a_pixel_no_object_scores_above_zero_belongs_to_nobody(self):
        logits = np.full((2, 2, 2), -0.5)
        for mask in _exclusive_masks(logits, [0, 1]):
            assert not mask.any()

    def test_exclusive_masks_follow_the_given_order(self):
        logits = np.array([[[2.0]], [[-2.0]]])
        # row 0 is person 1, row 1 is person 0
        by_person = _exclusive_masks(logits, [1, 0])
        assert by_person[1].tolist() == [[True]]
        assert by_person[0].tolist() == [[False]]


class TestMaxPeople:
    def test_the_largest_are_kept_and_the_cap_warns(self, models, warnings):
        video = make_video(figures=("A", "B", "C"))
        result = track_people(video, prompt="person", max_people=1)
        assert person_names(result) == ["person-1"]
        assert inside(result["person-1"].frames[0], rect_a(0)) > 0.95
        assert result["people"]["people"] == 1
        assert result["people"]["cuts"][0]["found"] == 3
        messages = [call.args[0] for call in warnings.call_args_list]
        assert any("max_people" in m and "largest" in m for m in messages)

    def test_three_people_are_numbered_by_size(self, models, warnings):
        video = make_video(figures=("C", "B", "A"))
        result = track_people(video, prompt="person", max_people=4)
        assert person_names(result) == ["person-1", "person-2", "person-3"]
        for name, rect in zip(
            ("person-1", "person-2", "person-3"), (rect_a, rect_b, rect_c)
        ):
            assert inside(result[name].frames[0], rect(0)) > 0.95

    def test_zero_tracks_nobody_and_loads_no_model(self, warnings):
        clear_model_cache()
        touched = []

        class Trip:
            def __getattr__(self, name):
                touched.append(name)
                raise AssertionError("a model class was touched")

        trip = Trip()
        with (
            patch.object(person_track, "AutoProcessor", trip),
            patch.object(person_track, "AutoModelForZeroShotObjectDetection", trip),
            patch.object(person_track, "Sam2VideoProcessor", trip),
            patch.object(person_track, "Sam2VideoModel", trip),
        ):
            result = track_people(make_video(), prompt="person", max_people=0)
        assert list(result) == ["people"]
        assert isinstance(result["people"], JsonRecord)
        assert result["people"]["tracked"] is False
        assert result["people"]["people"] == 0
        assert touched == []
        warnings.assert_called_once()

    @pytest.mark.parametrize("bad", [-1, 1.5, "x"])
    def test_a_bad_max_people_is_refused(self, models, bad):
        with pytest.raises(ValueError, match="max_people"):
            track_people(make_video(), max_people=bad)

    def test_a_missing_video_is_refused(self):
        with pytest.raises(ValueError, match="video"):
            track_people(None)

    def test_a_blank_prompt_is_refused(self):
        with pytest.raises(ValueError, match="prompt"):
            track_people(make_video(), prompt="  ")


class TestNoPerson:
    def test_a_clip_with_no_detection_is_an_error(self, models, warnings):
        with pytest.raises(ValueError, match="no person found"):
            track_people(make_video(figures=()), prompt="person")


class TestCuts:
    def shots(self):
        return [
            shot_record("shot-1", 0, 5),
            shot_record("shot-2", 5, 5),
        ]

    def video(self):
        frames = [make_frame(i, ("A", "B")) for i in range(5)]
        frames += [make_frame(i, ("A",)) for i in range(5, 10)]
        return AudioVideo(frames, None, None, fps=24, shots=self.shots())

    def test_a_recorded_shot_restarts_tracking(self, models, warnings):
        dino, sam = models
        result = track_people(self.video(), prompt="person")
        assert dino.calls == 2
        assert len(sam.sessions) == 2
        assert [len(s.frames) for s in sam.sessions] == [5, 5]
        cuts = result["people"]["cuts"]
        assert [c["start_frame"] for c in cuts] == [0, 5]
        assert [c["num_frames"] for c in cuts] == [5, 5]
        assert [len(c["people"]) for c in cuts] == [2, 1]
        assert len(result["person-1"].frames) == 10

    def test_a_cut_with_fewer_people_leaves_that_mask_black(self, models, warnings):
        result = track_people(self.video(), prompt="person")
        two = result["person-2"].frames
        assert len(two) == 10
        assert inside(two[0], rect_b(0)) > 0.95
        for frame in two[5:]:
            assert not (np.asarray(frame) > 0).any()
        one = result["person-1"].frames
        assert inside(one[5], rect_a(5)) > 0.95

    def test_several_cuts_warn_that_numbering_is_per_cut(self, models, warnings):
        track_people(self.video(), prompt="person")
        messages = [call.args[0] for call in warnings.call_args_list]
        assert any("2 cuts" in m for m in messages)

    def test_the_contact_sheet_has_a_tile_per_cut(self, models, warnings):
        result = track_people(self.video(), prompt="person")
        sheet = result["contact_sheet"]
        assert sheet.width == 2 * person_track.SHEET_TILE_WIDTH

    def test_a_detected_content_cut_starts_a_cut_too(self, models, warnings):
        with patch.object(person_track, "content_cuts", return_value=[(5, 0.9)]):
            result = track_people(make_video(), prompt="person")
        assert [c["start_frame"] for c in result["people"]["cuts"]] == [0, 5]

    def test_a_cut_with_nobody_is_black_and_warns(self, models, warnings):
        frames = [make_frame(i, ("A", "B")) for i in range(5)]
        frames += [make_frame(i, ()) for i in range(5, 10)]
        video = AudioVideo(frames, None, None, fps=24, shots=self.shots())
        result = track_people(video, prompt="person")
        for frame in result["person-1"].frames[5:]:
            assert not (np.asarray(frame) > 0).any()
        messages = [call.args[0] for call in warnings.call_args_list]
        assert any("nobody found" in m and "shot-2" in m for m in messages)

    def test_cut_starts_ignores_out_of_range_and_malformed_shots(self):
        starts = person_track.cut_starts(
            10,
            [{"start_frame": 4}, {"start_frame": 99}, {"start_frame": "x"}, {}],
            iter(()),
        )
        assert starts == [0, 4]


class TestBoxes:
    def test_a_near_duplicate_box_is_dropped(self):
        boxes = [[10, 10, 50, 50], [11, 11, 51, 51], [100, 10, 140, 50]]
        kept = distinct_boxes(boxes, [0.5, 0.9, 0.6])
        assert [score for _, score in kept] == [0.9, 0.6]
        assert kept[0][0] == [11, 11, 51, 51]

    def test_touching_boxes_are_both_kept(self):
        kept = distinct_boxes([[0, 0, 10, 10], [10, 0, 20, 10]], [0.5, 0.6])
        assert len(kept) == 2

    def test_no_boxes(self):
        assert distinct_boxes([], []) == []


class TestRegistration:
    def test_track_people_is_registered(self):
        from dw.tasks.registry import _COMMAND_INFO
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "track_people" in _COMMAND_REGISTRY
        info = _COMMAND_INFO["track_people"]
        assert info["result_content_type"] == "video/mp4"
        assert info["implementation"] == "dw.tasks.person_track.track_people"

    def test_it_consumes_the_device(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        task = MagicMock()
        task.device_for.return_value = "cuda:1"
        with patch.object(person_track, "track_people", return_value={}) as run:
            _COMMAND_REGISTRY["track_people"](task, {"video": "v.mp4"}, {})
        assert run.call_args.kwargs["device"] == "cuda:1"

    def test_video_is_a_media_argument(self):
        from dw.tasks.registry import _COMMAND_RULES

        assert "video" in _COMMAND_RULES["track_people"]["media_arguments"]
        assert _COMMAND_RULES["track_people"]["whole_numbers"] == ("max_people",)

    def test_the_video_is_realized_lazily(self):
        from dw.arguments import _LAZY_FRAME_COMMANDS

        assert "track_people" in _LAZY_FRAME_COMMANDS

    def test_the_task_describes_itself(self):
        from dw.introspection import describe_task

        described = str(describe_task("track_people"))
        assert "max_people" in described and "video" in described


class TestSaving:
    def result_dict(self):
        def mask_video(offset):
            frames = []
            for i in range(4):
                pixels = np.zeros((32, 48), np.uint8)
                pixels[4:20, offset + i : offset + i + 10] = 255
                frames.append(Image.fromarray(pixels, mode="L"))
            return AudioVideo(frames, None, None, fps=12)

        return {
            "person-1": mask_video(2),
            "person-2": mask_video(20),
            "contact_sheet": Image.new("RGB", (64, 40), (10, 20, 30)),
            "people": JsonRecord({"tracked": True, "people": 2}),
        }

    def test_the_real_writer_saves_every_entry(self, tmp_path):
        from dw.result import Result

        saved = Result({}).save_artifact(
            str(tmp_path), self.result_dict(), "step", "video/mp4", ".mp4"
        )
        names = sorted(
            p.name if hasattr(p, "name") else str(p).split("/")[-1] for p in saved
        )
        assert names == [
            "step-contact_sheet.png",
            "step-people.json",
            "step-person-1.mp4",
            "step-person-2.mp4",
        ]
        for name in names:
            assert (tmp_path / name).stat().st_size > 0
        assert json.loads((tmp_path / "step-people.json").read_text())["people"] == 2
        with Image.open(tmp_path / "step-contact_sheet.png") as sheet:
            assert sheet.format == "PNG" and sheet.size == (64, 40)

    def test_a_pil_image_under_a_video_type_is_written_as_png(self, tmp_path):
        from dw.result import Result

        saved = Result({}).save_artifact(
            str(tmp_path),
            {"contact_sheet": Image.new("RGB", (8, 8))},
            "still",
            "video/mp4",
            ".mp4",
        )
        assert [str(p).rsplit("/", 1)[-1] for p in saved] == ["still-contact_sheet.png"]
        assert (tmp_path / "still-contact_sheet.png").exists()
        assert not (tmp_path / "still-contact_sheet.mp4").exists()

    def test_the_rule_applies_only_to_a_video_content_type(self):
        from dw.result import _is_still_beside_video

        image = Image.new("RGB", (4, 4))
        assert _is_still_beside_video(image, "video/mp4")
        assert not _is_still_beside_video(image, "image/png")
        assert not _is_still_beside_video("text", "video/mp4")


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cached(repo):
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo, "config.json")
    return isinstance(found, str)


@pytest.mark.integration
class TestRealModels:
    def test_two_rectangles_get_separate_masks(self):
        for repo in ("IDEA-Research/grounding-dino-base", "facebook/sam2-hiera-large"):
            if not _cached(repo):
                pytest.skip(f"{repo} is not in the Hugging Face cache")
        clear_model_cache()
        width, height, count = 320, 192, 12
        frames = []
        for i in range(count):
            pixels = np.full((height, width, 3), 200, np.uint8)
            pixels[40:160, 30 + 4 * i : 90 + 4 * i] = RED
            pixels[60:150, 220 - 4 * i : 260 - 4 * i] = BLUE
            frames.append(Image.fromarray(pixels))
        result = track_people(
            AudioVideo(frames, None, None, fps=24),
            prompt="rectangle",
            threshold=0.2,
            device=_device(),
        )
        assert person_names(result) == ["person-1", "person-2"]
        assert len(result["person-1"].frames) == count

        def centre_x(image):
            xs = np.nonzero(np.asarray(image) > 0)[1]
            assert xs.size > 0
            return xs.mean()

        for index in (0, count - 1):
            red_x = 60 + 4 * index
            blue_x = 240 - 4 * index
            assert abs(centre_x(result["person-1"].frames[index]) - red_x) < 20
            assert abs(centre_x(result["person-2"].frames[index]) - blue_x) < 20
            both = (np.asarray(result["person-1"].frames[index]) > 0) & (
                np.asarray(result["person-2"].frames[index]) > 0
            )
            assert not both.any()
