"""restore_faces refuses a single upsample_img with a video input (#615)."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.tasks.task import _handle_restore_faces


def _task():
    task = MagicMock()
    task.device_for.return_value = "cpu"
    return task


def test_video_with_upsample_img_is_refused():
    video = AudioVideo([Image.new("RGB", (8, 8)) for _ in range(2)], None, None, fps=8)
    args = {
        "image": video,
        "model_name": "m",
        "upsample_img": Image.new("RGB", (16, 16)),
    }
    with pytest.raises(ValueError, match="upsample_img"):
        _handle_restore_faces(_task(), args, {})


def test_image_with_upsample_img_still_runs():
    background = Image.new("RGB", (16, 16))
    args = {
        "image": Image.new("RGB", (8, 8)),
        "model_name": "m",
        "upsample_img": background,
    }
    with patch("dw.tasks.restore_faces.restore_faces", return_value="out") as run:
        assert _handle_restore_faces(_task(), args, {}) == "out"
    assert run.call_args.kwargs["upsample_img"] is background
