"""The H3 frame-anchored templates quote a price at the family canvas (#1110)."""

import json
from pathlib import Path

import pytest

from dw.cost_scaling import scale_curated

ROOT = Path(__file__).resolve().parent.parent / "workflows/templates/minimax"


@pytest.mark.parametrize("name", ["last-frame-only", "first-and-last-frame"])
def test_960x544_is_priced_near_the_measured_run(name):
    definition = json.loads((ROOT / f"{name}.json").read_text())
    minutes = definition["cost"][0]["minutes"]
    defaults = definition["variables"]
    scaled, axes = scale_curated(
        minutes,
        {k: defaults[k] for k in ("width", "height")},
        {"width": 960, "height": 544},
    )
    assert axes and scaled == pytest.approx(3.5, abs=0.1)


def test_a_scales_with_field_that_cannot_be_summed_leaves_no_figure():
    # music-long's sections: one side's entries omit audio_duration, so the
    # sum cannot be taken - quoting the curated figure unscaled would hide
    # sections asking 3x the default
    from dw.cost_scaling import _list_frames_scaling

    default = [{"audio_duration": 60}, {"audio_duration": 60}]
    moved = [{"audio_duration": 180}, {"prompt": "no duration"}]
    assert (
        _list_frames_scaling("sections", default, moved, {}, "audio_duration") is None
    )
    # Unmoved, an unsummable field is still the curated figure
    assert _list_frames_scaling("sections", moved, moved, {}, "audio_duration") == (
        1.0,
        None,
    )
