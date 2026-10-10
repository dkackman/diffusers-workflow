"""restore_faces builds its facexlib helper once per (settings, device), not per frame."""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from dw.tasks import model_cache


@pytest.fixture
def fake_facexlib(monkeypatch):
    """A facexlib whose helper records constructions and detects no faces."""
    constructed = []

    class FakeHelper:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.cleaned = 0

        def clean_all(self):
            self.cleaned += 1

        def read_image(self, image):
            self.image = image

        def get_face_landmarks_5(self, **kwargs):
            return 0

    module = types.ModuleType("facexlib.utils.face_restoration_helper")
    module.FaceRestoreHelper = FakeHelper
    package = types.ModuleType("facexlib")
    utils = types.ModuleType("facexlib.utils")
    monkeypatch.setitem(sys.modules, "facexlib", package)
    monkeypatch.setitem(sys.modules, "facexlib.utils", utils)
    monkeypatch.setitem(sys.modules, "facexlib.utils.face_restoration_helper", module)
    model_cache.clear_model_cache()
    yield constructed
    model_cache.clear_model_cache()


def test_two_frames_share_one_helper(fake_facexlib, monkeypatch):
    from dw.tasks import restore_faces as module

    descriptor = MagicMock(supports_half=False)
    monkeypatch.setattr(module, "_load_face_model", lambda path, device: descriptor)
    monkeypatch.setattr(
        "dw.tasks.upscale.resolve_model_path",
        lambda model_name, filename: "weights.pth",
    )
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    module.restore_faces(frame, "m", device="cpu")
    module.restore_faces(frame, "m", device="cpu")

    assert len(fake_facexlib) == 1
    helper_kwargs = fake_facexlib[0]
    assert helper_kwargs["det_model"] == "retinaface_resnet50"


def test_helper_state_is_cleared_before_each_frame(fake_facexlib, monkeypatch):
    from dw.tasks import restore_faces as module

    descriptor = MagicMock(supports_half=False)
    monkeypatch.setattr(module, "_load_face_model", lambda path, device: descriptor)
    monkeypatch.setattr(
        "dw.tasks.upscale.resolve_model_path",
        lambda model_name, filename: "weights.pth",
    )
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    module.restore_faces(frame, "m", device="cpu")
    module.restore_faces(frame, "m", device="cpu")

    helper = model_cache._cache[("restore_faces_helper", 1, 512, True, "cpu")]
    assert helper.cleaned == 2


def test_different_settings_build_different_helpers(fake_facexlib, monkeypatch):
    from dw.tasks import restore_faces as module

    descriptor = MagicMock(supports_half=False)
    monkeypatch.setattr(module, "_load_face_model", lambda path, device: descriptor)
    monkeypatch.setattr(
        "dw.tasks.upscale.resolve_model_path",
        lambda model_name, filename: "weights.pth",
    )
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    module.restore_faces(frame, "m", device="cpu", face_size=512)
    module.restore_faces(frame, "m", device="cpu", face_size=256)
    assert len(fake_facexlib) == 2
