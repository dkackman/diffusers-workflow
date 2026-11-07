"""A `cost` entry's `fixed_minutes` stays out of the scaling (#1224)."""

import pytest

from dw.cost_scaling import _driver_scaling, _scaled, fixed_minutes
from dw.plan import _own_price


def _definition(**cost):
    return {
        "variables": {"num_inference_steps": 5, "num_frames": 124},
        "cost_drivers": ["num_inference_steps", "num_frames"],
        "cost": [{"device": "cuda", "vram_gb": 24, "minutes": 13.5, **cost}],
    }


def _quote(definition, **moved):
    variables = {**definition["variables"], **moved}
    return _own_price(definition, {"variables": variables}, {}, "cuda", {})


def test_the_fixed_share_is_not_multiplied():
    plain = _quote(_definition(), num_inference_steps=12, num_frames=345)
    fixed = _quote(
        _definition(fixed_minutes=9.0), num_inference_steps=12, num_frames=345
    )
    factor = 12 / 5 * 345 / 124
    assert plain["minutes"] == pytest.approx(13.5 * factor)
    assert fixed["minutes"] == pytest.approx(9.0 + 4.5 * factor)
    assert "9 min of it fixed" in fixed["scaled"]
    assert fixed["basis"] == "derived"


def test_a_run_at_the_defaults_is_the_curated_figure():
    assert _quote(_definition(fixed_minutes=9.0))["minutes"] == 13.5


def test_fixed_minutes_is_read_from_the_device_entry():
    cost = [
        {"device": "mps", "vram_gb": 0, "minutes": 5, "fixed_minutes": 1},
        {"device": "cuda", "vram_gb": 24, "minutes": 13.5, "fixed_minutes": 9},
    ]
    assert fixed_minutes(cost, "cuda") == 9.0
    assert fixed_minutes(cost, "mps") == 1.0
    assert fixed_minutes(cost, "cpu") == 1.0  # the first entry, as _price does
    assert fixed_minutes(None, "cuda") == 0.0
    assert (
        fixed_minutes([{"device": "cuda", "minutes": 1, "fixed_minutes": -2}], "cuda")
        == 0.0
    )


def test_a_fixed_share_above_the_figure_is_capped():
    scaling = _driver_scaling(
        _definition(), {"variables": {"num_inference_steps": 10, "num_frames": 124}}, {}
    )
    price = {"minutes": 4.0, "basis": "catalog", "measured_on": None}
    assert _scaled(price, scaling, 99.0)["minutes"] == pytest.approx(4.0)


def test_the_schema_accepts_it_and_refuses_a_negative():
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "dw/workflow_schema.json").read_text()
    )
    item = schema["properties"]["cost"]["items"]
    ok = {"device": "cuda", "vram_gb": 24, "minutes": 13.5, "fixed_minutes": 9}
    jsonschema.validate(ok, item)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**ok, "fixed_minutes": -1}, item)


def test_the_h3_chain_template_quotes_the_measured_runs_within_1_3x():
    """The 7 x 345 f, 12 step drill took 71.0 min (ab7f54f73b2f, with the check
    on); the same chain at 3 segments took 28.3 min (8a669d31258e)."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    definition = json.loads(
        (root / "workflows/templates/minimax/chained-segments.json").read_text()
    )
    base = {"num_inference_steps": 12, "num_frames": 345, "lora_model_name": None}
    three = _quote(definition, segments=3, **base)["minutes"]
    seven = _quote(definition, segments=7, **base)["minutes"]
    assert 28.3 / 1.3 < three < 28.3 * 1.3
    assert 71.0 / 1.3 < seven < 71.0 * 1.3
