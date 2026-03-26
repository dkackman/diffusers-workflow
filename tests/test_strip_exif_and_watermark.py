"""Tests for strip_exif and add_watermark image processing commands."""

import unittest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from dw.tasks.image_utils import strip_exif, add_watermark, process_image


class TestStripExif(unittest.TestCase):
    """Tests for the strip_exif function."""

    def test_returns_image_same_size(self):
        img = Image.new("RGB", (200, 100), color="red")
        result = strip_exif(img)
        self.assertEqual(result.size, (200, 100))

    def test_preserves_pixel_data(self):
        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        result = strip_exif(img)
        # Verify pixel content is preserved
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.getpixel((1, 1)), (255, 0, 0))

    def test_removes_png_metadata(self):
        img = Image.new("RGB", (10, 10))
        info = PngInfo()
        info.add_text("Comment", "secret location data")
        # Simulate an image with metadata by setting .info
        img.info["Comment"] = "secret location data"

        result = strip_exif(img)
        # The clean image should have no info dict entries carried over
        self.assertNotIn("Comment", result.info)

    def test_preserves_mode(self):
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        result = strip_exif(img)
        self.assertEqual(result.mode, "RGBA")

    def test_dispatch_via_process_image(self):
        img = Image.new("RGB", (50, 50))
        result = process_image(img, "strip_exif", "cpu", {})
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (50, 50))


class TestAddWatermark(unittest.TestCase):
    """Tests for the add_watermark function."""

    def test_returns_rgb_image(self):
        img = Image.new("RGB", (200, 100), color="blue")
        result = add_watermark(img)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (200, 100))

    def test_modifies_pixels(self):
        img = Image.new("RGB", (200, 100), color=(0, 0, 0))
        result = add_watermark(img, text="WATERMARK", opacity=255)
        # The watermarked image should differ from the all-black original
        self.assertNotEqual(img.tobytes(), result.tobytes())

    def test_default_text(self):
        # Should not raise with defaults
        img = Image.new("RGB", (400, 200))
        result = add_watermark(img)
        self.assertIsInstance(result, Image.Image)

    def test_custom_text(self):
        img = Image.new("RGB", (400, 200))
        result = add_watermark(img, text="DO NOT DISTRIBUTE")
        self.assertIsInstance(result, Image.Image)

    def test_all_positions(self):
        img = Image.new("RGB", (400, 200))
        for pos in ["bottom-right", "bottom-left", "top-right", "top-left", "center"]:
            result = add_watermark(img, position=pos)
            self.assertEqual(result.size, (400, 200))

    def test_invalid_position_falls_back(self):
        img = Image.new("RGB", (400, 200))
        # Unknown position should fall back to bottom-right
        result = add_watermark(img, position="nonsense")
        self.assertIsInstance(result, Image.Image)

    def test_custom_color(self):
        img = Image.new("RGB", (400, 200))
        result = add_watermark(img, color=(255, 0, 0))
        self.assertIsInstance(result, Image.Image)

    def test_custom_font_size(self):
        img = Image.new("RGB", (400, 200))
        result = add_watermark(img, font_size=24)
        self.assertIsInstance(result, Image.Image)

    def test_rgba_input_converted(self):
        img = Image.new("RGBA", (200, 100))
        result = add_watermark(img)
        self.assertEqual(result.mode, "RGB")

    def test_dispatch_via_process_image(self):
        img = Image.new("RGB", (200, 100))
        result = process_image(img, "add_watermark", "cpu", {"text": "TEST"})
        self.assertIsInstance(result, Image.Image)


if __name__ == "__main__":
    unittest.main()
