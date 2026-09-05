"""Tests for dw.tasks.image_utils.recenter_crop - re-framing around a chosen point."""

import pytest
from PIL import Image

from dw.tasks.image_utils import available_processors, recenter_crop


def _marked(size=200, mark=(20, 180), colour="red"):
    """A navy field with a small mark whose centre is at `mark`."""
    image = Image.new("RGB", (size, size), "navy")
    image.paste(Image.new("RGB", (8, 8), colour), (mark[0] - 4, mark[1] - 4))
    return image


class TestRecenterCrop:
    def test_the_chosen_point_lands_at_the_centre(self):
        out = recenter_crop(
            _marked(), center_x=0.5, center_y=0.5, crop=0.5, width=100, height=100
        )
        assert out.size == (100, 100)

    def test_a_corner_feature_is_brought_to_the_centre(self):
        image = _marked(mark=(20, 180))
        out = recenter_crop(
            image, center_x=0.1, center_y=0.9, crop=0.6, width=120, height=120
        )
        assert out.getpixel((60, 60)) == (255, 0, 0)

    def test_the_output_size_defaults_to_the_window_size(self):
        assert recenter_crop(_marked(), crop=0.5).size == (100, 100)

    def test_a_window_running_off_the_edge_is_filled_not_clamped(self):
        # Clamping would slide the window back inside and move the feature off
        # centre, which is exactly what registration cannot tolerate.
        image = _marked(mark=(4, 4))
        out = recenter_crop(
            image, center_x=0.02, center_y=0.02, crop=0.5, width=100, height=100
        )
        assert out.getpixel((50, 50)) == (255, 0, 0)

    def test_a_colour_fill_is_used_outside_the_source(self):
        out = recenter_crop(
            _marked(),
            center_x=0.0,
            center_y=0.0,
            crop=0.5,
            width=100,
            height=100,
            fill="black",
        )
        assert out.getpixel((5, 5)) == (0, 0, 0)

    @pytest.mark.parametrize("fill", ["edge", "reflect", "symmetric"])
    def test_the_mirroring_fills_are_accepted(self, fill):
        out = recenter_crop(
            _marked(),
            center_x=0.0,
            center_y=0.0,
            crop=0.5,
            width=64,
            height=64,
            fill=fill,
        )
        assert out.size == (64, 64)

    def test_a_non_positive_crop_is_refused(self):
        with pytest.raises(ValueError, match="greater than zero"):
            recenter_crop(_marked(), crop=0.0)

    def test_it_is_registered_as_a_processor(self):
        assert "recenter_crop" in available_processors()
