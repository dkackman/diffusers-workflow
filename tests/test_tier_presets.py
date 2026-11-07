"""The long-form templates' `tier` dial (draft, standard, best) through
`presets`, measured in bench/records/tiers.md: `standard` is today's
defaults, every tier validates, and an explicit argument still wins."""

import json
from pathlib import Path

import pytest

from dw.presets import preset_errors
from dw.workflow import Workflow

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "workflows" / "templates"

TIERED = [
    "minimax/chained-segments",
    "minimax/chain-video-continuity",
    "minimax/chain-matched-to-audio",
    "minimax/chain-matched-and-aligned",
    "minimax/dialogue-short",
    "minimax/music-video",
]
TIERS = ("draft", "standard", "best")
RECORD = ROOT / "bench" / "records" / "tiers.md"


def _definition(name):
    return json.loads((TEMPLATES / f"{name}.json").read_text(encoding="utf-8"))


def _workflow(name):
    path = TEMPLATES / f"{name}.json"
    return Workflow(_definition(name), None, str(path))


def _variables(name, arguments):
    return _workflow(name).expanded_definition(arguments)["variables"]


@pytest.mark.parametrize("name", TIERED)
def test_the_tier_dial_is_draft_standard_best_defaulting_to_standard(name):
    definition = _definition(name)
    assert definition["variables"]["tier"] == "standard"
    assert sorted(definition["presets"]["tier"]) == sorted(TIERS)
    assert preset_errors(definition) == []


@pytest.mark.parametrize("name", TIERED)
def test_standard_is_todays_defaults(name):
    definition = _definition(name)
    standard = definition["presets"]["tier"]["standard"]
    assert standard, name
    for variable, value in standard.items():
        assert definition["variables"][variable] == value, variable
    assert _variables(name, {}) == _variables(name, {"tier": "standard"})


@pytest.mark.parametrize("name", TIERED)
def test_every_tier_states_the_same_settings(name):
    bundles = _definition(name)["presets"]["tier"]
    keys = {tier: sorted(bundles[tier]) for tier in TIERS}
    assert keys["draft"] == keys["standard"] == keys["best"]


@pytest.mark.parametrize("name", TIERED)
@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_validates(name, tier, monkeypatch):
    # on a normal server: no tier needs the windowed refine's scratch flag
    monkeypatch.delenv("DW_EXPERIMENTAL_REFINE_WINDOWS", raising=False)
    assert _workflow(name).validation_errors({"tier": tier}) == []


@pytest.mark.parametrize("name", TIERED)
def test_draft_is_cheaper_and_best_dearer_than_standard(name):
    def work(tier):
        v = _variables(name, {"tier": tier})
        return v["width"] * v["height"] * v["num_inference_steps"]

    assert work("draft") < work("standard") <= work("best")
    assert _variables(name, {"tier": "best"}) != _variables(name, {})


@pytest.mark.parametrize("name", TIERED)
def test_a_caller_value_beats_the_tier(name):
    draft = _variables(name, {"tier": "draft"})
    mixed = _variables(name, {"tier": "draft", "width": 960, "height": 544})
    assert (mixed["width"], mixed["height"]) == (960, 544)
    for variable in _definition(name)["presets"]["tier"]["draft"]:
        if variable not in ("width", "height"):
            assert mixed[variable] == draft[variable], variable


@pytest.mark.parametrize("name", TIERED)
def test_tier_is_a_cost_driver_and_described(name):
    definition = _definition(name)
    assert "tier" in definition["cost_drivers"]
    for phrase in ("`tier`", "draft", "best", "bench/records/tiers.md"):
        assert phrase in definition["description"], phrase


def test_tier_and_shot_move_different_variables():
    presets = _definition("minimax/chained-segments")["presets"]
    tiered = {k for bundle in presets["tier"].values() for k in bundle}
    shot = {k for bundle in presets["shot"].values() for k in bundle}
    assert tiered and shot and not tiered & shot


def test_the_record_names_every_tiered_template_and_the_skipped_ones():
    text = RECORD.read_text(encoding="utf-8")
    for name in TIERED + ["ltx2/chained-segments", "ltx2/extend-clip"]:
        assert name in text, name
