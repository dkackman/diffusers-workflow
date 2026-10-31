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
