"""The Z-Image catalog entry's VRAM estimate (#945): a batch-size voxel
variable projects like frames do, and the numbers match what lem measured."""

import json
import os

from dw.vram_estimate import required_gb, vram_estimate_errors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _entry():
    with open(os.path.join(REPO_ROOT, "workflows", "models", "z-image.json")) as f:
        return json.load(f)


def _at(entry, **overrides):
    """The entry as the run executes it: variables substituted into the step."""
    entry["variables"].update(overrides)
    arguments = entry["steps"][0]["pipeline"]["arguments"]
    for key, value in arguments.items():
        if isinstance(value, str) and value.startswith("variable:"):
            arguments[key] = entry["variables"][value.split(":", 1)[1]]
    return entry


def test_the_projection_matches_the_lem_measurements():
    estimate = _entry()["vram_estimate"]
    for n, w, h, measured in [
        (1, 512, 512, 12.16),
        (1, 1024, 1024, 12.88),
        (4, 960, 544, 13.83),
        (4, 1344, 768, 15.80),
    ]:
        projected = required_gb(
            estimate, {"num_images_per_prompt": n, "width": w, "height": h}
        )
        assert abs(projected - measured) < 0.4, (n, w, h, projected)


def test_batch_size_moves_the_projection_and_is_not_a_frame_count():
    estimate = _entry()["vram_estimate"]
    one = required_gb(
        estimate, {"num_images_per_prompt": 1, "width": 1344, "height": 768}
    )
    eight = required_gb(
        estimate, {"num_images_per_prompt": 8, "width": 1344, "height": 768}
    )
    assert eight > one + 5


def test_the_default_and_the_field_report_shape_fit_a_24gb_card():
    for overrides in ({}, {"width": 1344, "height": 768}):
        entry = _at(_entry(), **overrides)
        assert vram_estimate_errors(entry, device_type="cuda", capacity_gb=23.6) == []


def test_a_batch_past_the_card_warns():
    entry = _at(_entry(), width=1344, height=768, num_images_per_prompt=16)
    errors = vram_estimate_errors(entry, device_type="cuda", capacity_gb=23.6)
    assert errors and "RTX 3090" in errors[0]["message"]
