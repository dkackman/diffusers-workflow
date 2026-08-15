import numpy as np
import pytest
import torch
from PIL import Image

from dw.tasks.tensor_image import pil_to_float_tensor, float_tensor_to_pil


class TestPilToFloatTensor:
    def test_shape_and_range(self):
        """Output is (1, 3, H, W), float, values in [0, 1]."""
        image = Image.new("RGB", (16, 9), color=(10, 128, 250))
        tensor = pil_to_float_tensor(image, "cpu")

        assert tensor.shape == (1, 3, 9, 16)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_non_rgb_input_is_converted(self):
        """Grayscale ('L') and RGBA inputs are coerced to 3-channel RGB."""
        gray = Image.new("L", (8, 8), color=200)
        tensor = pil_to_float_tensor(gray, "cpu")
        assert tensor.shape == (1, 3, 8, 8)

        rgba = Image.new("RGBA", (8, 8), color=(1, 2, 3, 128))
        tensor = pil_to_float_tensor(rgba, "cpu")
        assert tensor.shape == (1, 3, 8, 8)

    def test_device_placement_cpu(self):
        image = Image.new("RGB", (4, 4), color=(1, 2, 3))
        tensor = pil_to_float_tensor(image, "cpu")
        assert tensor.device.type == "cpu"

    def test_device_placement_explicit_device_object(self):
        image = Image.new("RGB", (4, 4), color=(1, 2, 3))
        device = torch.device("cpu")
        tensor = pil_to_float_tensor(image, device)
        assert tensor.device == device

    def test_dtype_cast(self):
        """Optional dtype argument casts after device placement."""
        image = Image.new("RGB", (4, 4), color=(1, 2, 3))
        tensor = pil_to_float_tensor(image, "cpu", dtype=torch.float64)
        assert tensor.dtype == torch.float64

    def test_dtype_defaults_to_float32(self):
        image = Image.new("RGB", (4, 4), color=(1, 2, 3))
        tensor = pil_to_float_tensor(image, "cpu")
        assert tensor.dtype == torch.float32


class TestFloatTensorToPil:
    def test_accepts_batched_and_unbatched(self):
        """Both (1, 3, H, W) and (3, H, W) tensors are accepted."""
        batched = torch.rand(1, 3, 5, 7)
        unbatched = batched.squeeze(0)

        img_a = float_tensor_to_pil(batched)
        img_b = float_tensor_to_pil(unbatched)

        assert img_a.size == (7, 5) == img_b.size
        assert np.array_equal(np.array(img_a), np.array(img_b))

    def test_output_mode_and_size(self):
        tensor = torch.rand(1, 3, 12, 20)
        image = float_tensor_to_pil(tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (20, 12)

    def test_clamps_out_of_range_values(self):
        """Values outside [0, 1] don't wrap or error — they clamp."""
        tensor = torch.tensor([[[[-1.0]], [[0.5]], [[2.0]]]])
        image = float_tensor_to_pil(tensor)
        r, g, b = image.getpixel((0, 0))
        assert r == 0
        assert g == 128  # round(0.5 * 255) = round(127.5) = 128 (round-half-to-even)
        assert b == 255


class TestRoundTripFidelity:
    def test_exact_8bit_values_round_trip_losslessly(self):
        """PIL -> tensor -> PIL must reproduce exact 8-bit pixel values.

        This is the whole point of the fix: the old hand-rolled conversions
        truncated (`.byte()` on a plain float multiply), so a value like
        128/255 could land at 127.999... in float32 and get chopped down to
        127. Rounding avoids that drift.
        """
        rng = np.random.default_rng(1234)
        arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        original = Image.fromarray(arr, mode="RGB")

        tensor = pil_to_float_tensor(original, "cpu")
        result = float_tensor_to_pil(tensor)

        assert result.size == original.size
        assert result.mode == original.mode
        assert np.array_equal(np.array(result), np.array(original))

    def test_all_256_channel_values_round_trip(self):
        """Sweep every possible uint8 channel value through the round trip."""
        values = np.arange(256, dtype=np.uint8)
        arr = np.stack([values, values, values], axis=-1).reshape(16, 16, 3)
        original = Image.fromarray(arr, mode="RGB")

        tensor = pil_to_float_tensor(original, "cpu")
        result = float_tensor_to_pil(tensor)

        assert np.array_equal(np.array(result), np.array(original))


class TestRoundingNotTruncation:
    def test_near_half_maps_by_rounding_not_floor(self):
        """A value that truncation would floor down must round up instead.

        127.6/255 -> raw float*255 == 127.6, which floor/truncate would chop
        to 127 but round() correctly sends to 128.
        """
        value = 127.6 / 255.0
        tensor = torch.full((1, 3, 1, 1), value)
        image = float_tensor_to_pil(tensor)
        pixel = image.getpixel((0, 0))

        assert pixel == (128, 128, 128)
        # Sanity: prove truncation really would have given a different (wrong) answer.
        truncated = int(value * 255)
        assert truncated == 127
        assert pixel[0] != truncated

    def test_half_value_rounds_up_not_down(self):
        """0.5/255-scaled exact half (127.5) rounds to nearest even (128)."""
        tensor = torch.full((1, 3, 1, 1), 127.5 / 255.0)
        image = float_tensor_to_pil(tensor)
        assert image.getpixel((0, 0)) == (128, 128, 128)


class TestModuleAdoption:
    """Verify upscale.py and interpolate_frames.py actually route through the
    shared helpers (rather than a copy), and restore_faces.py deliberately
    keeps its own local logic because its data is BGR numpy, not PIL/RGB."""

    def test_upscale_uses_shared_functions(self):
        import dw.tasks.upscale as upscale
        import dw.tasks.tensor_image as tensor_image

        assert upscale.pil_to_float_tensor is tensor_image.pil_to_float_tensor
        assert upscale.float_tensor_to_pil is tensor_image.float_tensor_to_pil

    def test_interpolate_frames_uses_shared_pil_to_tensor(self):
        import dw.tasks.interpolate_frames as interpolate_frames
        import dw.tasks.tensor_image as tensor_image

        # Imported under its historical private name so existing patch.object
        # tests (test_interpolate_frames.py) keep working unchanged.
        assert interpolate_frames._pil_to_tensor is tensor_image.pil_to_float_tensor
        assert (
            interpolate_frames.float_tensor_to_pil is tensor_image.float_tensor_to_pil
        )
