"""templates/minimax/narrated-take: a guide chain matched to and holding a
retimed narration, whose per-segment prompts and lines come from a
`narration_lines` step. Validates as a probe (`stop_after`) with its
required variables given, refuses them by name when they are missing, and
plans its length from the narration rather than a segment count."""

import copy
import json
from pathlib import Path

import pytest

from dw.plan import build_plan
from dw.server.observed_cost import observed_for
from dw.validation import workflow_errors
from dw.workflow import workflow_from_definition
from tests.test_observed_cost import MINUTE, run

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "workflows/templates/minimax/narrated-take.json"
)
REQUIRED = ("image", "narration", "script", "scene", "speaker")
ARGUMENTS = {
    "image": "asset:speaker.png",
    "narration": "asset:narration-retimed.wav",
    "script": "We the People of the United States, in Order to form a more perfect Union.",
    "scene": "a medium close-up of a woman in a charcoal overcoat on a gravel path.",
    "speaker": "a woman in her sixties with a low, warm alto voice",
    "stop_after": 12,
}


def load():
    return json.loads(TEMPLATE.read_text())


def steps_by_name(definition):
    return {step["name"]: step for step in definition["steps"]}


@pytest.fixture
def plan(tmp_path, monkeypatch):
    """build_plan with the hub cache empty and the hub unreachable."""
    import dw.plan_downloads

    def offline(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )
    monkeypatch.setattr(dw.plan_downloads, "model_info", offline)

    def make(arguments, **overrides):
        candidate = workflow_from_definition(
            load(), str(tmp_path / "outputs"), str(tmp_path), str(tmp_path)
        )
        kwargs = dict(device="cuda", lookup_sizes=False) | overrides
        return build_plan(candidate, copy.deepcopy(arguments), **kwargs)

    return make


def _errors(tmp_path, arguments):
    workflow = workflow_from_definition(load(), str(tmp_path))
    return workflow_errors(workflow, arguments)


def test_validates_with_a_probe(tmp_path, plan):
    assert _errors(tmp_path, ARGUMENTS) == []

    # The chain's length is the narration's: no count or duration is asked
    # for, the track is matched and held, and the segment prompts are the
    # narration_lines step's
    steps = steps_by_name(load())
    chain = steps["take"]["pipeline"]["chain"]
    assert chain["match_audio"] is True
    assert "segments" not in chain and "duration_seconds" not in chain
    assert steps["take"]["pipeline"]["arguments"]["hold_audio"] == "variable:narration"
    assert chain["prompts"] == "previous_result:lines.prompts"
    # No chain 'lines': only the words check reads it, and a held chain has
    # none, so naming it drew the nothing-reads-'lines' warning on every validate
    assert "lines" not in chain
    lines = steps["lines"]["task"]
    assert lines["command"] == "narration_lines"
    assert lines["arguments"]["audio"] == "variable:narration"

    # ...and the plan says so: a probe of 12 segments, priced per segment,
    # whose full length comes from the track. The history is one full take
    # of 10 segments in 40 minutes, so 4 minutes a segment
    take = [
        {
            "step": "take",
            "shots": [
                {"name": f"segment {i}", "planned_segments": 10} for i in range(10)
            ],
        }
    ]
    rows = [run(40 * MINUTE, manifest=take)]

    def history(arguments):
        return observed_for(
            load(), rows, device="cuda", device_name="RTX 3090", arguments=arguments
        )

    assert history({})["cold_minutes"] == 4.0
    assert history({})["per_segment"] is True

    answer = plan(ARGUMENTS, observed=history)
    assert answer["steps"] == 2
    scaled = answer["estimate"]["scaled"]
    assert "from the observed 4 min" in scaled
    assert "probe: 12 segment(s) at the per-segment figure" in scaled
    assert "comes from the track" in scaled
    # segment 1 unchecked (the track is held), 11 checked with 1 re-roll:
    # 1 + 11 x 2 = 23 renders of the 4-minute segment, not of the 40-minute take
    assert "23 renders" in scaled
    assert answer["estimate"]["minutes"] == pytest.approx(92.0)

    # the full take, whose count only the track knows, is one segment's figure
    full = plan({**ARGUMENTS, "stop_after": None}, observed=history)
    assert full["estimate"]["minutes"] == pytest.approx(4.0)
    assert "per segment; the track sets the count" in full["estimate"]["scaled"]


@pytest.mark.parametrize("missing", REQUIRED)
def test_required_variables_are_refused_when_missing(tmp_path, missing):
    arguments = {k: v for k, v in ARGUMENTS.items() if k != missing}
    messages = [str(e.get("message")) for e in _errors(tmp_path, arguments)]
    assert any(f"'{missing}' is required" in m for m in messages), messages


def test_the_template_checks_the_document_without_its_required_values(tmp_path):
    """Checked as a file (no arguments), the required nulls are not judged -
    the held narration is not reported missing a second time."""
    workflow = workflow_from_definition(load(), str(tmp_path))
    assert workflow_errors(workflow) == []


def test_the_check_holds_no_words_or_lip_sync_measure():
    check = load()["variables"]["check"]
    assert check["words_similarity_min"] is None
    assert check["face_drop_max"] is None
    assert "lip_sync_min" not in check


def test_the_default_prompt_template_has_no_cue_or_breath():
    template = load()["variables"]["prompt_template"]
    for placeholder in ("{scene}", "{speaker}", "{line}", "{soundscape}"):
        assert placeholder in template
    lowered = template.lower()
    assert "breath" not in lowered
    assert "first second" not in lowered
    assert "silent walking" not in lowered
    assert "non_diegetic_music: N/A" in template


def test_the_template_declares_no_cost_yet():
    assert "cost" not in load()


def test_a_held_track_that_is_not_required_is_still_missing(tmp_path):
    """The required-null exemption is the constraint's, not the null's: drop
    the `required` rule and the matched chain's missing track is reported."""
    definition = load()
    del definition["variable_constraints"]["narration"]
    workflow = workflow_from_definition(definition, str(tmp_path))
    messages = [str(e.get("message")) for e in workflow_errors(workflow)]
    assert any("match_audio chain needs a 'hold_audio'" in m for m in messages)
