"""
Unit tests for the depth estimator hint
Tests the dtype a depth hint is produced in
"""

import torch
from unittest.mock import MagicMock, patch
from PIL import Image

from dw.tasks.depth_estimator import make_hint_tensor, make_hint_image


def estimator_returning_depth():
    """Build a mock transformers pipeline that returns a depth map"""
    estimator = MagicMock(return_value={"depth": Image.new("L", (8, 8), 128)})
    return MagicMock(return_value=estimator)


class TestHintDtype:
    """Test the dtype of the hint tensor a controlnet pipeline consumes"""

    def make(self, device, dtype_choice=None, **kwargs):
        with patch("dw.tasks.depth_estimator.pipeline", estimator_returning_depth()):
            if dtype_choice is None:
                return make_hint_tensor(Image.new("RGB", (8, 8)), device, **kwargs)

            # Exercise a backend's dtype choice without needing that hardware -
            # preferred_task_dtype() is what make_hint_tensor defers to.
            with patch(
                "dw.tasks.depth_estimator.preferred_task_dtype",
                return_value=dtype_choice,
            ):
                return make_hint_tensor(Image.new("RGB", (8, 8)), device, **kwargs)

    def test_cuda_uses_float16(self):
        assert self.make("cpu", dtype_choice=torch.float16).dtype == torch.float16

    def test_mps_uses_float32(self):
        # float16 produces NaN values on Apple Silicon
        assert self.make("cpu", dtype_choice=torch.float32).dtype == torch.float32

    def test_cpu_uses_float32(self):
        assert self.make("cpu").dtype == torch.float32

    def test_dtype_can_be_requested(self):
        # The hint has to match the dtype of the pipeline consuming it
        hint = self.make("cpu", dtype=torch.bfloat16)
        assert hint.dtype == torch.bfloat16

    def test_shape_is_a_batch_of_three_channels(self):
        assert self.make("cpu").shape == (1, 3, 8, 8)

    def test_hint_image_survives_a_half_precision_hint(self):
        with patch("dw.tasks.depth_estimator.pipeline", estimator_returning_depth()):
            image = make_hint_image(
                Image.new("RGB", (8, 8)), "cpu", dtype=torch.float16
            )

        assert isinstance(image, Image.Image)
