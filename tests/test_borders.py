"""
Unit tests for outpainting border and mask generation.

add_border_and_mask produces the (image, mask) pair an inpainting pipeline
needs to extend an image outward: the original pasted onto a larger black
canvas, plus a mask that is white everywhere the model should paint.
"""

import pytest
from PIL import Image

from dw.tasks.borders import add_border_and_mask, add_border_and_mask_with_size

BLACK = (0, 0, 0)
RED = (255, 0, 0)
PAINT = 255  # mask white - the model fills here
KEEP = 0  # mask black - the original pixels survive


@pytest.fixture
def image():
    return Image.new("RGB", (100, 100), RED)


class TestAddBorderAndMask:
    def test_returns_a_bordered_image_and_a_matching_mask(self, image):
        result = add_border_and_mask(image, zoom_left=0.5)

        assert set(result) == {"bordered_image", "mask"}
        assert result["bordered_image"].size == result["mask"].size
        assert result["bordered_image"].mode == "RGB"
        assert result["mask"].mode == "L"

    def test_padding_extends_only_the_requested_side(self, image):
        result = add_border_and_mask(image, zoom_left=0.5)

        # 100 + 50 left pad = 150, snapped up to the nearest multiple of 32
        assert result["bordered_image"].size == (160, 96)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"zoom_left": 0.5}, (160, 96)),
            ({"zoom_right": 0.5}, (160, 96)),
            ({"zoom_up": 0.25, "zoom_down": 0.25}, (96, 160)),
            ({"zoom_all": 2.0}, (192, 192)),
        ],
    )
    def test_dimensions_are_always_multiples_of_32(self, image, kwargs, expected):
        size = add_border_and_mask(image, **kwargs)["bordered_image"].size

        assert size == expected
        assert size[0] % 32 == 0 and size[1] % 32 == 0

    def test_no_zoom_snaps_the_canvas_to_the_nearest_multiple_of_32(self, image):
        # 100 rounds *down* to 96, so a 100x100 input with no zoom loses its
        # outermost rows and columns. Pinned because it is easy to reintroduce
        # by switching round() for ceil() and quietly changing every output.
        result = add_border_and_mask(image)

        assert result["bordered_image"].size == (96, 96)

    def test_the_original_is_pasted_past_the_left_padding(self, image):
        result = add_border_and_mask(image, zoom_left=0.5)
        bordered = result["bordered_image"]

        assert bordered.getpixel((10, 50)) == BLACK  # inside the new border
        assert bordered.getpixel((60, 50)) == RED  # original content

    def test_the_mask_marks_the_border_and_spares_the_original(self, image):
        mask = add_border_and_mask(image, zoom_left=0.5)["mask"]

        assert mask.getpixel((10, 50)) == PAINT
        assert mask.getpixel((100, 50)) == KEEP

    def test_overlap_pulls_the_mask_in_over_the_original(self, image):
        # Overlap lets the model repaint a seam of real pixels so the join
        # between original and generated content blends
        mask = add_border_and_mask(image, zoom_left=0.5, overlap=0.1)["mask"]

        # left_pad 50, overlap 10 -> the kept region starts at x=60, not x=50
        assert mask.getpixel((55, 50)) == PAINT
        assert mask.getpixel((60, 50)) == KEEP

    def test_without_overlap_the_mask_meets_the_original_exactly(self, image):
        mask = add_border_and_mask(image, zoom_left=0.5)["mask"]

        assert mask.getpixel((49, 50)) == PAINT
        assert mask.getpixel((50, 50)) == KEEP


class TestAddBorderAndMaskWithSize:
    def test_output_matches_the_requested_size(self):
        result = add_border_and_mask_with_size(
            Image.new("RGB", (200, 100), RED), 512, 512
        )

        assert result["bordered_image"].size == (512, 512)
        assert result["mask"].size == (512, 512)

    def test_requested_size_is_snapped_to_a_multiple_of_32(self):
        result = add_border_and_mask_with_size(
            Image.new("RGB", (100, 200), RED), 500, 500
        )

        assert result["bordered_image"].size == (512, 512)

    @pytest.mark.parametrize("source_size", [(200, 100), (100, 200), (100, 100)])
    def test_the_aspect_ratio_survives_the_fit(self, source_size):
        # The source is letterboxed, never stretched - so a row through the
        # middle of a wide source still shows original pixels edge to edge
        result = add_border_and_mask_with_size(
            Image.new("RGB", source_size, RED), 256, 256
        )
        bordered = result["bordered_image"]

        assert bordered.size == (256, 256)
        assert bordered.getpixel((128, 128)) == RED

    def test_a_wide_source_gets_top_and_bottom_bars(self):
        result = add_border_and_mask_with_size(
            Image.new("RGB", (400, 100), RED), 256, 256
        )

        assert result["bordered_image"].getpixel((128, 4)) == BLACK
        assert result["mask"].getpixel((128, 4)) == PAINT
        assert result["mask"].getpixel((128, 128)) == KEEP

    def test_a_tall_source_gets_left_and_right_bars(self):
        result = add_border_and_mask_with_size(
            Image.new("RGB", (100, 400), RED), 256, 256
        )

        assert result["bordered_image"].getpixel((4, 128)) == BLACK
        assert result["mask"].getpixel((4, 128)) == PAINT
        assert result["mask"].getpixel((128, 128)) == KEEP

    def test_a_matching_aspect_ratio_needs_no_bars(self):
        result = add_border_and_mask_with_size(
            Image.new("RGB", (100, 100), RED), 256, 256
        )

        assert result["bordered_image"].getpixel((4, 4)) == RED


class TestBorderDispatch:
    """Both functions are reachable as image processors from a workflow"""

    @pytest.mark.parametrize(
        "processor, kwargs",
        [
            ("add_border_and_mask", {"zoom_left": 0.5}),
            ("add_border_and_mask_with_size", {"width": 256, "height": 256}),
        ],
    )
    def test_dispatch_via_process_image(self, image, processor, kwargs):
        from dw.tasks.image_utils import process_image

        result = process_image(image, processor, "cpu", dict(kwargs))

        assert set(result) == {"bordered_image", "mask"}
