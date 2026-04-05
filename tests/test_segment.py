import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from PIL import Image


def _make_test_image(width=640, height=480):
    """Create a simple test image."""
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_batch_encoding(data):
    """Create a MagicMock that behaves like a BatchEncoding (dict-like + .to())."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: data[key]
    mock.__contains__ = lambda self, key: key in data
    mock.to.return_value = mock
    return mock


class TestSegmentImage:
    """Test segment_image function with mocked models."""

    @patch("dw.tasks.segment.Sam2Model")
    @patch("dw.tasks.segment.Sam2Processor")
    @patch("dw.tasks.segment.AutoModelForZeroShotObjectDetection")
    @patch("dw.tasks.segment.AutoProcessor")
    def test_returns_pil_image_mode_l(
        self, mock_auto_proc, mock_auto_model, mock_sam_proc, mock_sam_model
    ):
        """segment_image should return a grayscale PIL Image."""
        from dw.tasks.segment import segment_image

        mock_processor_instance = MagicMock()
        mock_auto_proc.from_pretrained.return_value = mock_processor_instance
        dino_inputs = _make_batch_encoding({"input_ids": torch.zeros(1, 10)})
        mock_processor_instance.return_value = dino_inputs
        mock_processor_instance.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 300.0, 300.0]]),
                "scores": torch.tensor([0.9]),
                "labels": ["dog"],
            }
        ]

        mock_model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = MagicMock()

        mock_sam_proc_instance = MagicMock()
        mock_sam_proc.from_pretrained.return_value = mock_sam_proc_instance
        sam_inputs = _make_batch_encoding(
            {
                "pixel_values": torch.zeros(1, 3, 256, 256),
                "input_boxes": torch.tensor([[[100.0, 100.0, 300.0, 300.0]]]),
                "original_sizes": torch.tensor([[480, 640]]),
                "reshaped_input_sizes": torch.tensor([[256, 256]]),
            }
        )
        mock_sam_proc_instance.return_value = sam_inputs

        mock_sam_model_instance = MagicMock()
        mock_sam_model.from_pretrained.return_value = mock_sam_model_instance
        mock_sam_model_instance.to.return_value = mock_sam_model_instance
        mask_tensor = torch.zeros(1, 1, 3, 480, 640)
        mask_tensor[0, 0, 0, 100:300, 100:300] = 1.0
        mock_sam_model_instance.return_value = MagicMock(pred_masks=mask_tensor)

        post_mask = torch.zeros(1, 1, 480, 640)
        post_mask[0, 0, 100:300, 100:300] = 1.0
        mock_sam_proc_instance.post_process_masks.return_value = [post_mask]

        image = _make_test_image()
        result = segment_image(image, "dog")

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.size == (640, 480)

    @patch("dw.tasks.segment.Sam2Model")
    @patch("dw.tasks.segment.Sam2Processor")
    @patch("dw.tasks.segment.AutoModelForZeroShotObjectDetection")
    @patch("dw.tasks.segment.AutoProcessor")
    def test_no_detections_returns_black_mask(
        self, mock_auto_proc, mock_auto_model, mock_sam_proc, mock_sam_model
    ):
        """When nothing is detected, return an all-black mask."""
        from dw.tasks.segment import segment_image

        mock_processor_instance = MagicMock()
        mock_auto_proc.from_pretrained.return_value = mock_processor_instance
        dino_inputs = _make_batch_encoding({"input_ids": torch.zeros(1, 10)})
        mock_processor_instance.return_value = dino_inputs
        mock_processor_instance.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": [],
            }
        ]

        mock_model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = MagicMock()

        image = _make_test_image()
        result = segment_image(image, "nonexistent_object")

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        arr = np.array(result)
        assert arr.max() == 0

    @patch("dw.tasks.segment.Sam2Model")
    @patch("dw.tasks.segment.Sam2Processor")
    @patch("dw.tasks.segment.AutoModelForZeroShotObjectDetection")
    @patch("dw.tasks.segment.AutoProcessor")
    def test_invert_flag(
        self, mock_auto_proc, mock_auto_model, mock_sam_proc, mock_sam_model
    ):
        """When invert=True, mask should be inverted."""
        from dw.tasks.segment import segment_image

        mock_processor_instance = MagicMock()
        mock_auto_proc.from_pretrained.return_value = mock_processor_instance
        dino_inputs = _make_batch_encoding({"input_ids": torch.zeros(1, 10)})
        mock_processor_instance.return_value = dino_inputs
        mock_processor_instance.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": [],
            }
        ]

        mock_model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = MagicMock()

        image = _make_test_image()
        result = segment_image(image, "nonexistent_object", invert=True)

        assert isinstance(result, Image.Image)
        arr = np.array(result)
        assert arr.min() == 255


class TestSegmentTaskRegistration:
    def test_segment_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "segment" in _COMMAND_REGISTRY
