"""Every multi-step template says which of its outputs is the deliverable.

A run writes everything into one directory, so a finished episode sits
beside the scratch that went into it. A step's `result.subfolder` places
its files - by convention `final` for the deliverable and `intermediate`
for the rest (docs/WORKFLOW_GUIDE.md, "Saying which output is the
deliverable"). Agents compose by copying a template, so the convention
propagates only if every shipped template follows it; this test is what
keeps it from drifting.

The rule: a *saving step* is one
whose `result` sets `content_type` and does not set `save: false`; a
template is in scope when it has two or more saving steps; every saving
step of an in-scope template carries `final` or `intermediate`, and at
least one is `final`. The packaged builtins in dw/workflows/ stay unmarked:
a builtin is a step list a parent composes, and a role is the parent's to
assign - the parent's own `result` block, not the child's, is where it
says so.
"""

import json
import os

import pytest

from tests.test_examples import BUILTIN_DIR, REPO_ROOT, get_example_files

CONVENTION = {"final", "intermediate"}


def load(relative_path):
    with open(os.path.join(REPO_ROOT, relative_path), encoding="utf-8") as file:
        return json.load(file)


def saving_steps(definition):
    """The steps whose result is written to disk."""
    for step in definition.get("steps", []):
        result = step.get("result") or {}
        if result.get("content_type") and result.get("save", True) is not False:
            yield step


TEMPLATES = [f for f in get_example_files() if f.startswith("workflows/templates/")]
IN_SCOPE = [f for f in TEMPLATES if len(list(saving_steps(load(f)))) >= 2]


def test_the_scope_is_what_the_design_counted():
    """Eighteen templates had two or more saving steps when the convention
    landed. A template added later with several saving steps joins the
    parametrized test below on its own; this pins that none has quietly
    left it (a step that stopped saving would drop a template from scope
    without failing anything else)."""
    assert len(IN_SCOPE) >= 18, IN_SCOPE


@pytest.mark.parametrize("template", IN_SCOPE)
def test_every_saving_step_of_a_multi_step_template_names_its_role(template):
    steps = list(saving_steps(load(template)))
    roles = {step["name"]: step["result"].get("subfolder") for step in steps}

    unmarked = sorted(name for name, role in roles.items() if role not in CONVENTION)
    assert not unmarked, (
        f"{template}: saving steps without a final/intermediate subfolder: {unmarked}"
    )
    assert "final" in roles.values(), f"{template}: no step is marked final"


@pytest.mark.parametrize("template", IN_SCOPE)
def test_the_subfolder_is_the_last_key_of_the_result(template):
    """One added line per step, and every template reads alike."""
    for step in saving_steps(load(template)):
        assert list(step["result"])[-1] == "subfolder", (
            f"{template}: step {step['name']!r} does not end its result with subfolder"
        )


@pytest.mark.parametrize(
    "builtin",
    sorted(name for name in os.listdir(BUILTIN_DIR) if name.endswith(".json")),
)
def test_the_packaged_builtins_stay_unmarked(builtin):
    with open(os.path.join(BUILTIN_DIR, builtin), encoding="utf-8") as file:
        definition = json.load(file)
    marked = [
        step["name"]
        for step in definition.get("steps", [])
        if "subfolder" in (step.get("result") or {})
    ]
    assert not marked, (
        f"dw/workflows/{builtin} marks {marked}; a role is the parent's to assign"
    )
