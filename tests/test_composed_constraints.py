"""#1756: a composing step's arguments are held to the child's variable rules."""

import json

import pytest

from dw.workflow import workflow_from_file

TEMPLATES = "workflows/templates"


def refusals(path, arguments=None, workflow_dir="workflows"):
    """validation_errors without the VRAM finding, which is the test host's."""
    workflow = workflow_from_file(str(path), "/tmp", str(workflow_dir))
    return {
        error["path"]: error["message"]
        for error in workflow.validation_errors(arguments=arguments)
        if "which this workflow does not accept" in error["message"]
    }


@pytest.mark.parametrize(
    "arguments, path, rule",
    [
        ({"num_frames": 96}, "arguments.num_frames", "must be 8 * n + 1"),
        ({"num_frames": 1}, "arguments.num_frames", "must be at least 9"),
        ({"video_width": 500}, "arguments.video_width", "must be 32 * n + 0"),
        ({"video_height": 500}, "arguments.video_height", "must be 32 * n + 0"),
    ],
)
def test_best_of_n_to_video_refuses_what_the_ltx_child_refuses(arguments, path, rule):
    found = refusals(f"{TEMPLATES}/best-of-n-to-video.json", arguments)

    assert list(found) == [path]
    assert rule in found[path]
    assert "./ltx2/image-to-video.json" in found[path]


def test_values_the_child_accepts_are_not_refused():
    arguments = {"num_frames": 97, "video_width": 512, "video_height": 512}

    assert refusals(f"{TEMPLATES}/best-of-n-to-video.json", arguments) == {}


def _write(directory, name, definition):
    path = directory / name
    path.write_text(json.dumps(definition))
    return path


def _pair(tmp_path, parent_variables, passed, constraints=None, child_rule=None):
    step = {"name": "s", "task": {"command": "gather_images", "arguments": {}}}
    _write(
        tmp_path,
        "child.json",
        {
            "id": "child",
            "variables": {"frames": 9},
            "variable_constraints": {
                "frames": child_rule or {"modulus": 8, "remainder": 1}
            },
            "steps": [step],
        },
    )
    parent = {
        "id": "parent",
        "variables": parent_variables,
        "steps": [
            {
                "name": "video",
                "workflow": {"path": "./child.json", "arguments": {"frames": passed}},
            }
        ],
    }
    if constraints:
        parent["variable_constraints"] = constraints
    return _write(tmp_path, "parent.json", parent)


def test_a_stored_default_is_reported_at_the_variable(tmp_path):
    path = _pair(tmp_path, {"length": 10}, "variable:length")

    assert list(refusals(path, workflow_dir=tmp_path)) == ["variables.length"]


def test_a_literal_is_reported_at_the_step(tmp_path):
    path = _pair(tmp_path, {}, 10)

    assert list(refusals(path, workflow_dir=tmp_path)) == [
        "steps[0].workflow.arguments.frames"
    ]


def test_a_value_the_parents_own_rule_refuses_is_reported_once(tmp_path):
    path = _pair(
        tmp_path,
        {"length": 9},
        "variable:length",
        constraints={"length": {"modulus": 8, "remainder": 1}},
    )
    workflow = workflow_from_file(str(path), "/tmp", str(tmp_path))

    errors = workflow.validation_errors(arguments={"length": 10})

    assert [error["path"] for error in errors] == ["arguments.length"]


def test_a_variable_with_no_value_yet_is_not_judged(tmp_path):
    path = _pair(
        tmp_path,
        {"length": None},
        "variable:length",
        child_rule={"required": True},
    )

    assert (
        workflow_from_file(str(path), "/tmp", str(tmp_path)).validation_errors() == []
    )
