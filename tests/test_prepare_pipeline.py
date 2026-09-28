"""Validation, the run and the record prepare a definition through one
pipeline - fold, then expand - so what validation checks is what the run
executes and what the realized workflow records."""

import copy

import pytest

from dw.workflow import ConstantError, Workflow, workflow_from_definition

# One definition exercising every stage the three paths used to disagree on:
# a snap-up constraint, a for_each whose entry references another variable,
# and a "constraint:" frame_snap. (A frame_snap in a task's arguments is a
# validation error by design - loop_frames takes no such argument - but the
# prepare stages walk the whole definition, so it is the smallest place to
# put one. Verified against f07eaf2d: the run resolves it, validation does
# not, and that is the divergence the first test catches.)
DEFINITION = {
    "id": "prep",
    "seed": 7,
    "variables": {
        "num_frames": 108,
        "tail": 5,
        "clips": [{"name": "a", "len": "variable:tail"}, {"name": "b", "len": 9}],
    },
    "variable_constraints": {
        "num_frames": {
            "modulus": 17,
            "remainder": 5,
            "min_frames": 124,
            "max_frames": 345,
            "snap": "up",
        }
    },
    "steps": [
        {
            "name": "clip",
            "for_each": "variable:clips",
            "task": {
                "command": "loop_frames",
                "arguments": {"video": "item:name", "num_frames": "item:len"},
            },
            "result": {"content_type": "video/mp4"},
        },
        {
            "name": "long",
            "task": {
                "command": "loop_frames",
                "arguments": {
                    "video": "gather:clip",
                    "num_frames": "variable:num_frames",
                    "frame_snap": "constraint:num_frames",
                },
            },
            "result": {"content_type": "video/mp4"},
        },
    ],
}


def test_validation_expands_exactly_what_the_run_prepares(tmp_path):
    wf = workflow_from_definition(copy.deepcopy(DEFINITION), str(tmp_path))
    arguments = {"num_frames": 108}
    validated = wf.expanded_definition(arguments)
    prepared, seed, _ = wf._prepare_definition(
        copy.deepcopy(wf.workflow_definition), arguments, str(tmp_path)
    )
    assert validated["steps"] == prepared["steps"]
    assert seed == 7


def test_the_record_carries_the_run_s_folded_values(tmp_path):
    wf = workflow_from_definition(copy.deepcopy(DEFINITION), str(tmp_path))
    _, _, recorded = wf._prepare_definition(
        copy.deepcopy(wf.workflow_definition), {"num_frames": 108}, str(tmp_path)
    )
    from dw.realize import realize_workflow

    realized, _ = realize_workflow(wf.workflow_definition, recorded, seed=7)
    assert realized["variables"]["num_frames"] == 124  # snapped, as run
    assert realized["variables"]["clips"][0]["len"] == 5  # entry resolved
    assert realized["steps"][0]["for_each"] == "variable:clips"  # unexpanded


def test_a_failing_constant_names_its_variable_at_run_time(tmp_path):
    definition = copy.deepcopy(DEFINITION)
    definition["variables"]["sigmas"] = "constant:diffusers.no_such_module.X"
    wf = workflow_from_definition(definition, str(tmp_path))
    with pytest.raises(ConstantError) as raised:
        wf._prepare_definition(copy.deepcopy(definition), {}, str(tmp_path))
    assert raised.value.path == "variables.sigmas"


def test_expansion_is_computed_once_per_workflow_and_arguments(tmp_path):
    wf = workflow_from_definition(copy.deepcopy(DEFINITION), str(tmp_path))
    calls = []
    original = Workflow._fold

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Workflow, "_fold", counting)
        first = wf.expanded_definition({"num_frames": 108})
        first["steps"].clear()
        second = wf.expanded_definition({"num_frames": 108})
        wf.validation_errors({"num_frames": 108})
    assert len(calls) == 1
    assert second["steps"]  # the cache was not mutated through `first`


def test_an_undeclared_frame_snap_name_is_reported_at_its_path_not_raised(
    tmp_path,
):
    definition = copy.deepcopy(DEFINITION)
    definition["steps"][1]["task"]["arguments"]["frame_snap"] = "constraint:nope"
    wf = workflow_from_definition(definition, str(tmp_path))
    # Expansion resolves "constraint:" names now, and raises a bare
    # ValueError on one it cannot - validation must answer it at its path
    errors = wf.validation_errors({"num_frames": 108})
    assert {
        "path": "steps[1].task.arguments.frame_snap",
        "message": "'constraint:nope' names no entry of this workflow's "
        "'variable_constraints'. Declared: num_frames",
    } in errors


def test_an_undeclared_frame_snap_name_reached_through_a_variable_is_a_finding(
    tmp_path,
):
    """The literal check runs on the definition as written, where this one
    is still "variable:snap"; substitution makes it "constraint:nope" and
    expansion must report it at the frame_snap's path, not raise."""
    definition = copy.deepcopy(DEFINITION)
    definition["variables"]["snap"] = "constraint:nope"
    definition["steps"][1]["task"]["arguments"]["frame_snap"] = "variable:snap"
    wf = workflow_from_definition(definition, str(tmp_path))

    errors = wf.validation_errors({"num_frames": 108})

    assert errors == [
        {
            "path": "steps[1].task.arguments.frame_snap",
            "message": "'constraint:nope' names no entry of this workflow's "
            "'variable_constraints'. Declared: num_frames",
        }
    ]
