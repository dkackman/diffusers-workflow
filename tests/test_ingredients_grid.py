"""ingredients_grid: individual images laid out as one reference sheet (#607)."""

import pytest
from PIL import Image

from dw.tasks.image_utils import (
    _grid_compositions as _compositions,
    _grid_row_cells as _row_cells,
    ingredients_grid,
)


def solid(width, height, color):
    return Image.new("RGB", (width, height), color)


RED, GREEN, BLUE, BLACK = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)


class TestCanvas:
    def test_is_exactly_the_requested_size(self):
        sheet = ingredients_grid([solid(100, 200, RED)], width=768, height=448)
        assert sheet.size == (768, 448)
        assert sheet.mode == "RGB"

    def test_every_image_shows_up_on_the_sheet(self):
        images = [solid(300, 200, c) for c in (RED, GREEN, BLUE)]
        sheet = ingredients_grid(images, background="black", gap=4)
        colors = {color for _, color in sheet.getcolors(maxcolors=1 << 20)}
        assert {RED, GREEN, BLUE} <= colors

    def test_background_fills_the_unused_canvas(self):
        sheet = ingredients_grid([solid(100, 100, RED)], background="#102030")
        assert sheet.getpixel((0, 0)) == (16, 32, 48)

    def test_an_image_stays_in_reading_order(self):
        sheet = ingredients_grid(
            [solid(200, 200, RED), solid(200, 200, BLUE)],
            layout="rows",
            background="black",
        )
        reds = [x for x in range(sheet.width) if sheet.getpixel((x, 224)) == RED]
        blues = [x for x in range(sheet.width) if sheet.getpixel((x, 224)) == BLUE]
        assert max(reds) < min(blues)

    def test_transparency_is_flattened_onto_white(self):
        clear = Image.new("RGBA", (100, 100), (255, 0, 0, 0))
        sheet = ingredients_grid([clear], background="black", gap=0)
        assert sheet.getpixel((384, 224)) == (255, 255, 255)


class TestLayout:
    def test_auto_picks_rows_for_few_and_panels_for_many(self):
        few = [solid(300, 200, RED)] * 3
        many = [solid(300, 200, RED)] * 6
        assert (
            ingredients_grid(few).tobytes()
            == ingredients_grid(few, layout="rows").tobytes()
        )
        assert (
            ingredients_grid(many).tobytes()
            == ingredients_grid(many, layout="panels").tobytes()
        )

    def test_the_partition_search_considers_every_split(self):
        assert sum(1 for _ in _compositions(4)) == 8
        assert [3] in list(_compositions(3))
        assert [1, 1, 1] in list(_compositions(3))

    def test_wide_images_stack_in_rows_and_tall_ones_sit_side_by_side(self):
        wide = _row_cells([4.0, 4.0, 4.0], 768, 448, 0)
        assert len({y for _, y, _, _ in wide}) == 3
        tall = _row_cells([0.4, 0.4, 0.4], 768, 448, 0)
        assert len({y for _, y, _, _ in tall}) == 1

    def test_rows_stay_inside_the_canvas(self):
        cells = _row_cells([1.0] * 7, 752, 432, 8)
        for x, y, w, h in cells:
            assert x >= 0 and y >= 0 and x + w <= 752 and y + h <= 432

    def test_panels_are_equal_cells(self):
        sheet = ingredients_grid(
            [solid(100, 100, c) for c in (RED, GREEN, BLUE, RED)],
            layout="panels",
            width=400,
            height=400,
            gap=0,
            background="black",
        )
        assert sheet.getpixel((100, 100)) == RED
        assert sheet.getpixel((300, 100)) == GREEN
        assert sheet.getpixel((100, 300)) == BLUE


class TestFit:
    def test_contain_pads_and_cover_crops(self):
        wide = solid(400, 100, RED)
        kwargs = dict(layout="panels", width=200, height=200, gap=0, background="black")
        contained = ingredients_grid([wide], fit="contain", **kwargs)
        covered = ingredients_grid([wide], fit="cover", **kwargs)
        # the single panel is the whole canvas: contain letterboxes, cover fills
        assert contained.getpixel((100, 5)) == BLACK
        assert covered.getpixel((100, 5)) == RED


class TestRefusals:
    def test_no_images(self):
        with pytest.raises(ValueError, match="no images"):
            ingredients_grid([])

    def test_more_than_max_images_is_refused_not_dropped(self):
        with pytest.raises(ValueError, match="max_images"):
            ingredients_grid([solid(10, 10, RED)] * 3, max_images=2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"layout": "spiral"},
            {"fit": "stretch"},
            {"width": 0},
            {"gap": -1},
            {"background": "notacolour"},
        ],
    )
    def test_bad_arguments(self, kwargs):
        with pytest.raises(ValueError, match="ingredients_grid"):
            ingredients_grid([solid(10, 10, RED)], **kwargs)


class TestRegistered:
    def test_the_command_is_registered_with_its_arguments(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "ingredients_grid" in _COMMAND_REGISTRY


class TestStaticValidation:
    """validate refuses literal layout/fit/background and over-max image lists (#646)."""

    def _errors(self, **arguments):
        from dw.task_domains import task_argument_errors

        step = {
            "name": "s",
            "task": {"command": "ingredients_grid", "arguments": arguments},
        }
        return task_argument_errors({"steps": [step]})

    def test_bad_layout_and_fit(self):
        paths = {
            e["path"]
            for e in self._errors(images=["asset:a"], layout="grid", fit="stretch")
        }
        assert paths == {
            "steps[0].task.arguments.layout",
            "steps[0].task.arguments.fit",
        }

    def test_bad_background(self):
        assert self._errors(images=["asset:a"], background="notacolour")

    def test_too_many_images(self):
        errors = self._errors(images=[f"asset:{i}" for i in range(13)])
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.images"]
        assert not self._errors(images=[f"asset:{i}" for i in range(13)], max_images=13)

    def test_references_left_to_run_time(self):
        assert not self._errors(
            images=["gather:x"],
            layout="variable:l",
            fit="previous_result:f",
            background="#fff",
        )
