import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


def _make_test_frames(count=4, width=64, height=64):
    """Create a list of test frames with different colors."""
    frames = []
    for i in range(count):
        shade = int(255 * i / max(count - 1, 1))
        frames.append(Image.new("RGB", (width, height), color=(shade, shade, shade)))
    return frames


class TestInterpolateFrames:
    """Test interpolate_frames function."""

    @patch("dw.tasks.interpolate_frames._load_rife_model")
    def test_2x_doubles_frame_count(self, mock_load):
        """2x multiplier should produce 2N-1 frames from N input frames."""
        from dw.tasks.interpolate_frames import interpolate_frames

        mock_model = MagicMock()

        def fake_inference(img1, img2):
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            mid = ((arr1 + arr2) / 2).astype(np.uint8)
            return Image.fromarray(mid)

        mock_model.side_effect = fake_inference
        mock_load.return_value = mock_model

        frames = _make_test_frames(4)
        result = interpolate_frames(frames, multiplier=2)

        # 4 frames with 2x: (4-1)*2 + 1 = 7
        assert len(result) == 7
        assert all(isinstance(f, Image.Image) for f in result)

    @patch("dw.tasks.interpolate_frames._load_rife_model")
    def test_4x_quadruples_frame_count(self, mock_load):
        """4x multiplier should run two passes of 2x."""
        from dw.tasks.interpolate_frames import interpolate_frames

        mock_model = MagicMock()

        def fake_inference(img1, img2):
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            mid = ((arr1 + arr2) / 2).astype(np.uint8)
            return Image.fromarray(mid)

        mock_model.side_effect = fake_inference
        mock_load.return_value = mock_model

        frames = _make_test_frames(4)
        result = interpolate_frames(frames, multiplier=4)

        # Two passes of 2x: 4 -> 7 -> 13
        assert len(result) == 13
        assert all(isinstance(f, Image.Image) for f in result)

    def test_invalid_multiplier_raises(self):
        """multiplier must be 2, 4, or 8."""
        from dw.tasks.interpolate_frames import interpolate_frames

        with pytest.raises(ValueError, match="multiplier"):
            interpolate_frames(_make_test_frames(2), multiplier=3)

    def test_single_frame_raises(self):
        """Need at least 2 frames to interpolate."""
        from dw.tasks.interpolate_frames import interpolate_frames

        with pytest.raises(ValueError, match="at least 2"):
            interpolate_frames(_make_test_frames(1), multiplier=2)


class TestInterpolateFramesRegistration:
    def test_interpolate_frames_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "interpolate_frames" in _COMMAND_REGISTRY
