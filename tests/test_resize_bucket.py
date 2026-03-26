"""Tests for resize_bucket image processing command."""

import unittest
from PIL import Image

from dw.tasks.image_utils import resize_bucket, _DEFAULT_RATIOS


class TestResizeBucket(unittest.TestCase):
    """Tests for the resize_bucket function."""

    def test_square_image_picks_1_1(self):
        img = Image.new("RGB", (500, 500))
        result = resize_bucket(img, resolution=1024)
        self.assertEqual(result.width, result.height)

    def test_landscape_picks_landscape_ratio(self):
        # 1600x900 is 16:9
        img = Image.new("RGB", (1600, 900))
        result = resize_bucket(img, resolution=1024)
        self.assertGreater(result.width, result.height)

    def test_portrait_picks_portrait_ratio(self):
        # 900x1600 is 9:16
        img = Image.new("RGB", (900, 1600))
        result = resize_bucket(img, resolution=1024)
        self.assertGreater(result.height, result.width)

    def test_dimensions_aligned_to_64(self):
        img = Image.new("RGB", (1600, 900))
        result = resize_bucket(img, resolution=1024)
        self.assertEqual(result.width % 64, 0)
        self.assertEqual(result.height % 64, 0)

    def test_custom_alignment(self):
        img = Image.new("RGB", (800, 600))
        result = resize_bucket(img, resolution=512, alignment=32)
        self.assertEqual(result.width % 32, 0)
        self.assertEqual(result.height % 32, 0)

    def test_custom_ratios(self):
        # 1800x900 is 2:1, should pick [2, 1] over [1, 1]
        img = Image.new("RGB", (1800, 900))
        result = resize_bucket(img, resolution=1024, ratios=[[1, 1], [2, 1]])
        self.assertGreater(result.width, result.height)

    def test_4_3_image(self):
        # 800x600 is exactly 4:3
        img = Image.new("RGB", (800, 600))
        result = resize_bucket(img, resolution=1024)
        ratio = result.width / result.height
        # Should pick 4:3 (1.333) — verify it's close
        self.assertAlmostEqual(ratio, 4 / 3, delta=0.1)

    def test_3_2_image(self):
        # 1200x800 is 3:2
        img = Image.new("RGB", (1200, 800))
        result = resize_bucket(img, resolution=1024)
        ratio = result.width / result.height
        self.assertAlmostEqual(ratio, 3 / 2, delta=0.1)

    def test_resolution_controls_short_side(self):
        img = Image.new("RGB", (500, 500))
        result = resize_bucket(img, resolution=512)
        # 1:1 ratio, so both sides should be ~512
        self.assertEqual(result.width, 512)
        self.assertEqual(result.height, 512)

    def test_converts_to_rgb(self):
        img = Image.new("RGBA", (500, 500))
        result = resize_bucket(img, resolution=512)
        self.assertEqual(result.mode, "RGB")

    def test_default_ratios_has_expected_entries(self):
        # Sanity check that we have the standard ratios
        ratio_values = {(r[0], r[1]) for r in _DEFAULT_RATIOS}
        self.assertIn((1, 1), ratio_values)
        self.assertIn((16, 9), ratio_values)
        self.assertIn((9, 16), ratio_values)
        self.assertIn((4, 3), ratio_values)
        self.assertIn((3, 4), ratio_values)


class TestResizeBucketRegistration(unittest.TestCase):
    """Test that resize_bucket is accessible via process_image."""

    def test_process_image_dispatches(self):
        from dw.tasks.image_utils import process_image

        img = Image.new("RGB", (800, 600))
        result = process_image(img, "resize_bucket", "cpu", {"resolution": 512})
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.width % 64, 0)
        self.assertEqual(result.height % 64, 0)


if __name__ == "__main__":
    unittest.main()
