"""chained-segments: `shot` selects the drift counters (static, moving,
dialogue) through the template's `presets`, measured in bench/records/counters.md."""

import json
from pathlib import Path


from dw.workflow import Workflow

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "workflows/templates/minimax/chained-segments.json"
BEFORE = ROOT / "fixtures/chained_segments_before_shot.json"


def _workflow():
    definition = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    return Workflow(definition, None, str(TEMPLATE))


def _chain(arguments=None):
    expanded = _workflow().expanded_definition(arguments or {})
    return expanded["steps"][0]["pipeline"]["chain"]


def test_template_validates_for_every_shot():
    for shot in ("static", "moving", "dialogue"):
        assert _workflow().validation_errors({"shot": shot}) == []
    assert _workflow().validation_errors({}) == []


def test_shot_static_applies_the_anchor_at_a_quarter_on_guide():
    chain = _chain({"shot": "static"})
    assert chain["continuity"] == "guide"
    assert chain["anchor"] == "segment_1"
    assert chain["anchor_strength"] == 0.25


def test_shot_moving_is_guide_with_no_counters():
    chain = _chain({"shot": "moving"})
    assert chain["continuity"] == "guide"
    assert chain.get("anchor") in (None, "none")
    assert chain.get("anchor_strength") is None


def test_shot_dialogue_is_guide_with_no_counters():
    chain = _chain({"shot": "dialogue"})
    assert chain["continuity"] == "guide"
    assert chain.get("anchor") in (None, "none")
    assert chain.get("anchor_strength") is None


def test_explicit_anchor_strength_beats_the_static_bundle():
    chain = _chain({"shot": "static", "anchor_strength": 0.5})
    assert chain["anchor"] == "segment_1"
    assert chain["anchor_strength"] == 0.5


def test_explicit_continuity_beats_the_bundle():
    assert (
        _chain({"shot": "static", "continuity": "last_frame"})["continuity"]
        == "last_frame"
    )


def test_default_is_guide_with_no_counters():
    chain = _chain()
    assert chain["continuity"] == "guide"
    assert chain["guide_frames"] == 22
    assert chain["anchor"] == "none"
    assert chain.get("anchor_strength") is None
    for shot in ("static", "moving", "dialogue"):
        assert _chain({"shot": shot})["continuity"] == "guide"
    assert json.loads(TEMPLATE.read_text())["variables"]["continuity"] == "guide"


def test_continuity_last_frame_is_still_the_old_chain_block():
    """The template as it was before `shot`, expanded: with continuity
    last_frame its chain block is today's plus the anchor's explicit 'none'."""
    old = Workflow(json.loads(BEFORE.read_text()), None, str(BEFORE))
    before = old.expanded_definition({})["steps"][0]["pipeline"]["chain"]
    chain = _chain({"continuity": "last_frame"})
    assert chain.pop("anchor") == "none"
    assert chain == before


def test_description_states_the_rules():
    description = json.loads(TEMPLATE.read_text())["description"]
    for phrase in (
        "static",
        "moving",
        "dialogue",
        "30 s",
        "look_hold",
        "carry_pick",
        "camera move",
        "Guide is the default",
    ):
        assert phrase in description, phrase
