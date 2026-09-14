"""A task step whose arguments its command's signature refuses.

`validate_workflow` used to answer `valid: true` for a task step that left a
*required* argument unset, and then the job failed on Python's own
"resample_audio() missing 1 required positional argument: 'audio'" - the one
class of mistake a free pre-flight most obviously exists for (#141). An
unknown argument sat beside it as a warning, so a step with every argument it
was given rejected and every argument it needs missing still validated. Both
are the same guaranteed TypeError at the same call, so both are errors, and
the command itself refuses the same thing at run time for a value that
arrived from a variable or an earlier step.
"""

import json
import pathlib

import pytest

from dw.introspection import (
    missing_task_arguments,
    task_signature_errors,
    unknown_task_arguments,
    workflow_argument_warnings,
)
from dw.tasks.task import Task


def task_step(command, arguments, name="a"):
    return {
        "name": name,
        "task": {"command": command, "arguments": arguments},
        "result": {"content_type": "audio/wav"},
    }


def errors_for(command, arguments):
    return task_signature_errors({"id": "sig", "steps": [task_step(command, arguments)]})


class TestARequiredArgumentLeftUnset:
    def test_the_reported_repro_is_refused(self):
        """#141's first repro: resample_audio without its audio."""
        errors = errors_for("resample_audio", {"target_sample_rate": 16000})
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.audio"
        assert "resample_audio" in errors[0]["message"]
        assert "'audio'" in errors[0]["message"]

    def test_a_supplied_argument_passes(self):
        assert errors_for("resample_audio", {"audio": "x", "target_sample_rate": 16000}) == []

    def test_a_reference_counts_as_supplied(self):
        """The key's presence is what the check is - the value may still be a
        reference that only resolves during the run."""
        assert (
            errors_for(
                "resample_audio",
                {"audio": "previous_result:earlier", "target_sample_rate": 16000},
            )
            == []
        )

    def test_device_is_never_required(self):
        assert "device" not in missing_task_arguments("resample_audio", ["audio"])

    def test_kwargs_does_not_excuse_a_required_argument(self):
        """An image processor takes any keys and still needs its image."""
        assert missing_task_arguments("canny", []) == ["image"]
        assert missing_task_arguments("canny", ["image"]) == []

    def test_an_unknown_command_reports_nothing(self):
        assert missing_task_arguments("not_a_command", []) == []
        assert errors_for("not_a_command", {}) == []


class TestAnArgumentTheCommandDoesNotTake:
    def test_the_near_miss_from_the_report_is_refused(self):
        """#141's second repro: crossfade_audio's parameter is `audios`, a
        list, so `audio`/`other` are rejected and `audios` is missing - every
        argument wrong, and it used to validate."""
        errors = errors_for(
            "crossfade_audio", {"audio": "x", "other": "y", "crossfade_ms": 0}
        )
        paths = [e["path"] for e in errors]
        assert paths == [
            "steps[0].task.arguments.audios",
            "steps[0].task.arguments.audio",
            "steps[0].task.arguments.other",
        ]

    def test_it_is_no_longer_also_a_warning(self):
        """Reported once, by the pass whose verdict it changes."""
        definition = {
            "id": "sig",
            "steps": [task_step("crossfade_audio", {"audios": ["x"], "nope": 1})],
        }
        assert task_signature_errors(definition)
        assert not [w for w in workflow_argument_warnings(definition) if "nope" in w]

    def test_a_free_form_command_accepts_anything(self):
        assert unknown_task_arguments("gather_inputs", ["whatever"]) == []
        assert errors_for("gather_inputs", {"whatever": 1}) == []


class TestTheRunTimeBackstop:
    """The static pass sees literals. A required argument that arrived from a
    variable or an earlier step and resolved to nothing reaches the command,
    which refuses it in the validator's wording rather than Python's."""

    def test_the_command_refuses_and_says_which_argument(self):
        task = Task({"command": "resample_audio", "arguments": {}}, "cpu")
        with pytest.raises(ValueError) as caught:
            task.run({"target_sample_rate": 16000})
        message = str(caught.value)
        assert "resample_audio" in message
        assert "'audio'" in message
        assert "positional argument" not in message

    def test_an_inputs_list_template_is_left_alone(self):
        """`inputs` is consumed whole - there are no names to miss."""
        task = Task({"command": "gather_inputs", "inputs": [1, 2]}, "cpu")
        task._check_required_arguments([1, 2])


class TestTheCatalogItself:
    """Every workflow shipped in the repo passes the new check - a template
    that did not would be a run nothing could start."""

    @pytest.mark.parametrize(
        "path",
        sorted(
            str(p)
            for p in list(pathlib.Path("workflows").rglob("*.json"))
            + list(pathlib.Path("dw/workflows").glob("*.json"))
        ),
    )
    def test_workflow_has_no_task_signature_error(self, path):
        definition = json.loads(pathlib.Path(path).read_text())
        if not isinstance(definition, dict) or "steps" not in definition:
            pytest.skip("not a workflow")
        assert task_signature_errors(definition) == []
