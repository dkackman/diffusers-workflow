"""A ref2va guide's rows are priced in `plan.estimate` (#1426): a step that
lays in a `guides` clip costs the workflow's declared `guide_factor` times one
without, on the catalog figure and on a figure inherited from a template."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from dw.cost_scaling import declared_guide_factor, guide_scaling
from dw.plan import _own_price

from .test_inherited_cost import PIPELINE, inherited
from .test_observed_cost import MINUTE, run

ROOT = Path(__file__).resolve().parent.parent
REF2VA = "workflows/templates/minimax/reference-to-video.json"
GUIDE = [{"video": "asset:take.mp4", "strength": 1.0}]


def _toggled(**cost):
    """A one-step workflow whose guide is a variable, unguided by default."""
    return {
        "variables": {"num_frames": 124, "guides": None},
        "cost_drivers": ["num_frames"],
        "cost": [{"device": "cuda", "vram_gb": 24, "minutes": 4.0, **cost}],
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    **PIPELINE,
                    "arguments": {
                        "num_frames": "variable:num_frames",
                        "guides": "variable:guides",
                    },
                },
            }
        ],
    }


def _quote(definition, **moved):
    variables = {**definition["variables"], **moved}
    return _own_price(definition, {"variables": variables}, {}, "cuda", {})


class TestCatalogFigure:
    def test_a_guide_doubles_the_scaling_part(self):
        price = _quote(_toggled(guide_factor=2.0), guides=GUIDE)
        assert price["minutes"] == pytest.approx(8.0)
        assert "guides on 1 of 1 pipeline steps vs 0 of 1" in price["scaled"]
        assert "x2 a guided step" in price["scaled"]

    def test_the_fixed_share_is_not_doubled(self):
        price = _quote(_toggled(guide_factor=2.0, fixed_minutes=1.0), guides=GUIDE)
        assert price["minutes"] == pytest.approx(1.0 + 3.0 * 2)

    def test_it_multiplies_with_the_declared_drivers(self):
        price = _quote(_toggled(guide_factor=2.0), guides=GUIDE, num_frames=248)
        assert price["minutes"] == pytest.approx(4.0 * 2 * 248 / 124)

    def test_no_guide_is_the_curated_figure(self):
        assert _quote(_toggled(guide_factor=2.0))["minutes"] == 4.0

    def test_a_null_video_is_not_a_guide(self):
        price = _quote(_toggled(guide_factor=2.0), guides=[{"video": None}])
        assert price["minutes"] == 4.0

    def test_without_a_declared_factor_a_guide_is_not_priced(self):
        assert _quote(_toggled(), guides=GUIDE)["minutes"] == 4.0


def test_declared_guide_factor_reads_only_a_number_above_one():
    def of(value):
        return declared_guide_factor({"cost": [{"minutes": 1, "guide_factor": value}]})

    assert of(2) == 2.0
    assert of(1) is None
    assert of(True) is None
    assert of("2") is None
    assert declared_guide_factor({}) is None


def test_the_ratio_is_against_the_guides_the_figure_was_measured_with():
    guided = _toggled(guide_factor=2.0)
    guided["variables"]["guides"] = GUIDE
    # measured guided, run guided: nothing moves
    assert guide_scaling(guided, {}, guided, {}, guided) == (1.0, None)
    # measured guided, run without: half
    ratio, reason = guide_scaling(guided, {}, guided, {"guides": None}, guided)
    assert ratio == pytest.approx(0.5)
    assert "0 of 1 pipeline steps vs 1 of 1" in reason


def _chained(continuity, segments=3):
    definition = _toggled(guide_factor=2.0)
    definition["steps"][0]["pipeline"]["chain"] = {
        "segments": segments,
        "continuity": continuity,
    }
    return definition


def test_a_guide_chain_is_a_guided_step_as_the_vram_owner_counts_it():
    """The cost rule asks `guides.guide_lengths` whether a step is guided, so a
    `continuity: "guide"` chain - which appends a guide to every segment after
    the first - is priced as guided, as the VRAM projection charges it."""
    from dw.guides import guide_lengths

    chain = _chained("guide")
    assert guide_lengths(chain["steps"][0]["pipeline"])
    unchained = _toggled(guide_factor=2.0)
    ratio, reason = guide_scaling(unchained, {}, chain, {}, chain)
    assert ratio == pytest.approx(2.0)
    assert "1 of 1 pipeline steps vs 0 of 1" in reason
    # a chain carried some other way, or one segment long, lays in no guide
    for other in (_chained("last_frame"), _chained("guide", segments=1)):
        assert not guide_lengths(other["steps"][0]["pipeline"])
        assert guide_scaling(unchained, {}, other, {}, other) == (1.0, None)


def _ref2va():
    return json.loads((ROOT / REF2VA).read_text())


def test_the_ref2va_template_declares_a_valid_factor():
    definition = _ref2va()
    assert declared_guide_factor(definition) == 2.0
    schema = json.loads((ROOT / "dw/workflow_schema.json").read_text())
    item = schema["properties"]["cost"]["items"]
    for entry in definition["cost"]:
        jsonschema.validate(entry, item)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**definition["cost"][0], "guide_factor": 0.5}, item)


class TestInheritedFigure:
    """The #1426 repro: an inline copy of the ref2va template with a literal
    guide inherits the template's unguided runs (3.8 min control, 7.4 min
    guided on lem), and was quoted the unguided figure."""

    def _block(self, inline):
        template = _ref2va()
        return inherited(
            inline,
            [(REF2VA.removeprefix("workflows/").removesuffix(".json"), template)],
            {
                REF2VA.removeprefix("workflows/").removesuffix(".json"): [
                    run(3.8 * MINUTE, {})
                ]
            },
        )

    def _inline(self, guided):
        inline = copy.deepcopy(_ref2va())
        inline["id"] = "inline"
        inline.pop("cost", None)
        if guided:
            inline["steps"][0]["pipeline"]["arguments"]["guides"] = GUIDE
        return inline

    def test_a_guided_copy_quotes_about_twice_the_unguided_one(self):
        plain = self._block(self._inline(False))
        guided = self._block(self._inline(True))
        assert plain is not None and guided is not None
        assert plain["cold_minutes"] == pytest.approx(3.8)
        # 7.4 min measured guided (9334eb09b659)
        assert guided["cold_minutes"] == pytest.approx(7.6)
        assert "guides on 1 of 1 pipeline steps vs 0 of 1" in guided["scaled"]
        assert "scaled" not in plain
