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


def test_cost_row_is_the_guide_figure():
    # The seam A/B at this template's shape (3 x 124) measured guide at 13.5 min
    # against last_frame's 9.6 (docs/proposals/complete/h3-guides-complete.md).
    definition = json.loads(TEMPLATE.read_text())
    assert [row["minutes"] for row in definition["cost"]] == [13.5]
    assert "h3-guides-complete" in definition["description"]


def test_cost_drivers_bucket_continuity_shot_and_anchor():
    drivers = json.loads(TEMPLATE.read_text())["cost_drivers"]
    for name in ("continuity", "shot", "anchor"):
        assert name in drivers, name


def test_summary_says_guide_not_last_frame():
    summary = json.loads(TEMPLATE.read_text())["summary"]
    assert "guide" in summary.lower()
    assert "last frame" not in summary


def test_readme_does_not_call_the_example_last_frame():
    readme = (TEMPLATE.parent / "README.md").read_text()
    row = next(x for x in readme.splitlines() if "[chained-segments.json]" in x)
    assert "`last_frame`" not in row


def test_turning_the_anchor_off_on_a_static_shot_is_refused_as_documented():
    errors = _workflow().validation_errors({"shot": "static", "anchor": "none"})
    messages = " ".join(e["message"] for e in errors)
    assert "anchor_strength needs anchor 'segment_1'" in messages
    # the advice the description gives: null the strength, or use shot moving
    description = json.loads(TEMPLATE.read_text())["description"]
    assert "anchor_strength: null" in description
    assert "shot: moving" in description
    for arguments in (
        {"shot": "static", "anchor": "none", "anchor_strength": None},
        {"shot": "moving"},
    ):
        errors = _workflow().validation_errors(arguments)
        assert not any("anchor_strength needs" in e["message"] for e in errors)
