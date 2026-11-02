"""Tests for analyze_face_drift (dw/tasks/face_drift.py, #1146).

YuNet and SFace are replaced by fakes, so no weights are downloaded: a face
is a flat coloured square on a black frame, the detector finds each distinct
colour's bounding box, and the recognizer embeds a crop as its mean colour -
two squares of the same colour are the same person, different colours are
orthogonal (cosine 0).
"""

import numpy as np
import pytest
from PIL import Image

import dw.tasks.face_drift as fd
from dw.media_types import AudioVideo
from dw.shots import shot_record
from dw.tasks.registry import required_weights_hook
from dw.tasks.task import task_command_info

PERSON_A = (255, 0, 0)
PERSON_B = (0, 255, 0)
SIZE = 64


def frame(*faces):
    """A black frame; each face is (colour, x, y, side)."""
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for colour, x, y, side in faces:
        pixels[y : y + side, x : x + side] = colour
    return Image.fromarray(pixels)


def a_face(colour=PERSON_A, side=24):
    return frame((colour, 8, 8, side))


class FakeRecognizer:
    def alignCrop(self, bgr, row):
        x, y, w, h = (int(v) for v in row[:4])
        return bgr[y : y + h, x : x + w]

    def feature(self, crop):
        return crop.reshape(-1, 3).mean(axis=0).astype(np.float32)

    def match(self, first, second, flag):
        assert flag == fd.COSINE
        return float(
            first @ second / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-9)
        )


class Probe:
    """Counts the frames the fake detector was shown."""

    def __init__(self):
        self.calls = 0

    def detect(self, bgr):
        self.calls += 1
        rows = []
        for colour in {tuple(p) for p in bgr.reshape(-1, 3)} - {(0, 0, 0)}:
            ys, xs = np.nonzero((bgr == colour).all(axis=2))
            row = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
            rows.append(row + [0.0] * 10 + [0.9])
        return rows


@pytest.fixture
def probe(monkeypatch):
    seen = Probe()
    monkeypatch.setattr(
        fd, "_models", lambda device=None: (seen.detect, FakeRecognizer())
    )
    return seen


def clip(frames, shots=None):
    return AudioVideo(frames, None, None, fps=24, shots=shots)


def shots_of(*counts):
    start = 0
    records = []
    for index, count in enumerate(counts):
        records.append(shot_record(f"shot{index}", start, count))
        start += count
    return records


def test_same_face_no_drift_finding(probe):
    answer = fd.analyze_face_drift(
        clip([a_face()] * 6, shots_of(3, 3)), a_face(), samples_per_shot=3
    )
    assert answer["findings"] == []
    assert [m["faces_found"] for m in answer["measurements"]] == [3, 3]
    assert all(m["similarity_median"] > 0.99 for m in answer["measurements"])
    assert answer["shots_source"] == "artifact"
    assert answer["rules_applied"] == ["face_drift", "no_face"]


def test_other_face_drift_on_every_shot(probe):
    answer = fd.analyze_face_drift(
        clip([a_face(PERSON_B)] * 6, shots_of(3, 3)), a_face(), samples_per_shot=3
    )
    drifted = [f for f in answer["findings"] if f["rule"] == "face_drift"]
    assert [f["at"] for f in drifted] == [{"shot": "shot0"}, {"shot": "shot1"}]
    assert drifted[0]["threshold"] == 0.363
    assert drifted[0]["value"] < 0.363


def test_empty_shot_reports_no_face_only_there(probe):
    frames = [frame()] * 3 + [a_face()] * 3
    answer = fd.analyze_face_drift(
        clip(frames, shots_of(3, 3)), a_face(), samples_per_shot=3
    )
    assert [(f["rule"], f["at"]) for f in answer["findings"]] == [
        ("no_face", {"shot": "shot0"})
    ]
    first, second = answer["measurements"]
    assert first["faces_found"] == 0 and first["similarity_median"] is None
    assert first["similarity_min"] is None
    assert second["faces_found"] == 3


def test_reference_without_face_refused(probe):
    with pytest.raises(ValueError, match="reference"):
        fd.analyze_face_drift(clip([a_face()]), frame())


def test_reference_with_two_faces_uses_largest_and_notes_it(probe):
    # the larger face is person A; a second, smaller person B stands beside it
    reference = frame((PERSON_A, 4, 4, 30), (PERSON_B, 40, 4, 12))
    answer = fd.analyze_face_drift(clip([a_face()] * 2), reference)
    assert answer["findings"] == []
    assert any("reference" in note and "largest" in note for note in answer["notes"])
    swapped = fd.analyze_face_drift(clip([a_face(PERSON_B)] * 2), reference)
    assert [f["rule"] for f in swapped["findings"]] == ["face_drift"]


def test_whole_file_is_one_shot_without_boundaries(probe):
    answer = fd.analyze_face_drift(clip([a_face()] * 4), a_face())
    assert answer["shots_source"] == "none"
    assert len(answer["measurements"]) == 1


@pytest.mark.parametrize("threshold", [-1.01, 1.01])
def test_threshold_bounds(probe, threshold):
    with pytest.raises(ValueError, match="threshold"):
        fd.analyze_face_drift(clip([a_face()]), a_face(), threshold=threshold)


@pytest.mark.parametrize("samples", [0, -2, 1.5])
def test_samples_per_shot_must_be_a_positive_whole_number(probe, samples):
    with pytest.raises(ValueError, match="samples_per_shot"):
        fd.analyze_face_drift(clip([a_face()]), a_face(), samples_per_shot=samples)


def test_samples_per_shot_caps_reads(probe):
    answer = fd.analyze_face_drift(clip([a_face()] * 100), a_face())
    # one detection for the reference, eight for the shot
    assert probe.calls == 1 + 8
    assert answer["measurements"][0]["frames_sampled"] == 8


def test_a_shot_shorter_than_the_cap_reads_each_frame_once(probe):
    answer = fd.analyze_face_drift(clip([a_face()] * 3), a_face())
    assert answer["measurements"][0]["frames_sampled"] == 3


def test_required_weights_lists_sface_and_yunet():
    entries = fd.required_weights({})
    assert {
        "repo": fd.SFACE_REPO,
        "filename": fd.SFACE_FILE,
        "revision": fd.SFACE_REVISION,
    } in entries
    assert any(entry["repo"] == "opencv/face_detection_yunet" for entry in entries)


def test_command_registered():
    import dw.tasks  # noqa: F401  registers the handlers

    info = task_command_info("analyze_face_drift")
    assert info["returns"] == "json"
    assert required_weights_hook("analyze_face_drift") is fd.required_weights


class TestStaticCheck:
    """The settings refused at validate, in the run's words (one rule)."""

    @staticmethod
    def _errors(**arguments):
        from dw.task_problems import face_drift_errors

        return face_drift_errors(arguments)

    @pytest.mark.parametrize(
        "arguments, argument",
        [
            ({"threshold": 1.5}, "threshold"),
            ({"threshold": -1.01}, "threshold"),
            ({"samples_per_shot": 0}, "samples_per_shot"),
            ({"samples_per_shot": 65}, "samples_per_shot"),
            ({"samples_per_shot": 1.5}, "samples_per_shot"),
        ],
    )
    def test_out_of_range_found(self, arguments, argument):
        assert [a for a, _ in self._errors(**arguments)] == [argument]

    def test_clean_and_referenced_values_pass(self):
        assert not self._errors(threshold=0.363, samples_per_shot=64)
        assert not self._errors(threshold=-1, samples_per_shot=1)
        assert not self._errors(
            threshold="variable:t", samples_per_shot="previous_result:x"
        )

    def test_the_run_refuses_in_the_same_words(self, probe):
        (_, message), *_ = self._errors(threshold=1.5)
        with pytest.raises(ValueError) as refused:
            fd.analyze_face_drift(clip([a_face()]), a_face(), threshold=1.5)
        assert str(refused.value) == message


TEMPLATE = "workflows/templates/face-drift-check.json"


class TestTemplateValidation:
    @staticmethod
    def _errors(tmp_path, arguments):
        import os

        from dw.validation import workflow_errors
        from dw.workflow import workflow_from_file
        from tests.test_examples import REPO_ROOT

        workflow = workflow_from_file(os.path.join(REPO_ROOT, TEMPLATE), str(tmp_path))
        return workflow_errors(workflow, arguments)

    @pytest.mark.parametrize(
        "arguments, argument",
        [
            ({"threshold": 1.5}, "threshold"),
            ({"samples_per_shot": 0}, "samples_per_shot"),
            ({"samples_per_shot": 65}, "samples_per_shot"),
        ],
    )
    def test_validate_refuses_out_of_range_settings(
        self, tmp_path, arguments, argument
    ):
        errors = self._errors(tmp_path, arguments)
        assert errors
        assert all(error["path"].endswith(f".{argument}") for error in errors)

    def test_defaults_are_clean(self, tmp_path):
        assert self._errors(tmp_path, {}) == []

    def test_template_declares_and_wires_samples_per_shot(self):
        import json
        import os

        from tests.test_examples import REPO_ROOT

        with open(os.path.join(REPO_ROOT, TEMPLATE), encoding="utf-8") as handle:
            definition = json.load(handle)
        assert definition["variables"]["samples_per_shot"] == fd.DEFAULT_SAMPLES
        arguments = definition["steps"][0]["task"]["arguments"]
        assert arguments["samples_per_shot"] == "variable:samples_per_shot"


def test_sface_cache_key_carries_the_revision(monkeypatch):
    """A revision bump must not be answered by the recognizer cached before it."""
    import dw.tasks.model_cache as model_cache

    keys = []
    monkeypatch.setattr(fd, "face_detector_model", lambda *args: object())
    monkeypatch.setattr(
        model_cache, "cached_model", lambda key, load: keys.append(key) or object()
    )
    fd._models()
    assert keys and fd.SFACE_REVISION in keys[0]
