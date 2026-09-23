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
    return task_signature_errors(
        {"id": "sig", "steps": [task_step(command, arguments)]}
    )


class TestARequiredArgumentLeftUnset:
    def test_the_reported_repro_is_refused(self):
        """#141's first repro: resample_audio without its audio."""
        errors = errors_for("resample_audio", {"target_sample_rate": 16000})
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.audio"
        assert "resample_audio" in errors[0]["message"]
        assert "'audio'" in errors[0]["message"]

    def test_a_supplied_argument_passes(self):
        assert (
            errors_for("resample_audio", {"audio": "x", "target_sample_rate": 16000})
            == []
        )

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

    def test_an_unknown_command_is_reported_at_the_command(self):
        """The per-argument helpers have nothing to say about a command that
        does not exist; the pass reports the command itself (#285)."""
        assert missing_task_arguments("not_a_command", []) == []
        problems = errors_for("not_a_command", {})
        assert [p["path"] for p in problems] == ["steps[0].task.command"]
        assert "not a registered task command" in problems[0]["message"]


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


class TestARequiredArgumentFedByANullVariable:
    """A step that names the argument by `variable:name`, where `name`'s
    value is null, is not a step that "does not supply" it (#364) - the
    error carries a `variable` key so a caller with no arguments of its own
    can tell the two apart."""

    def test_the_repro_carries_the_variable_key_and_a_clearer_message(self):
        written = {
            "id": "sig",
            "variables": {"audio": None},
            "steps": [
                task_step(
                    "resample_audio",
                    {"audio": "variable:audio", "target_sample_rate": 16000},
                )
            ],
        }
        # replace_variables drops a variable: reference resolved to null
        # from its containing dict (#209) - this is what the expanded
        # definition looks like once that has happened
        expanded = {
            "id": "sig",
            "steps": [task_step("resample_audio", {"target_sample_rate": 16000})],
        }
        errors = task_signature_errors(expanded, written_definition=written)
        assert len(errors) == 1
        assert errors[0]["variable"] == "audio"
        assert errors[0]["path"] == "steps[0].task.arguments.audio"
        assert "variable 'audio'" in errors[0]["message"]
        assert "does not supply" not in errors[0]["message"]

    def test_no_written_definition_keeps_the_original_wording(self):
        """Without the author-written form to compare against - the #141
        call sites already in the codebase before #364 - nothing changes."""
        errors = errors_for("resample_audio", {"target_sample_rate": 16000})
        assert "variable" not in errors[0]
        assert "does not supply" in errors[0]["message"]

    def test_a_genuinely_missing_argument_is_unaffected(self):
        """No `variable:` reference at all in the written step - still the
        plain #141 message, even with a written_definition available."""
        written = {
            "id": "sig",
            "steps": [task_step("resample_audio", {"target_sample_rate": 16000})],
        }
        errors = task_signature_errors(written, written_definition=written)
        assert "variable" not in errors[0]
        assert "does not supply" in errors[0]["message"]

    def test_a_variable_not_declared_is_unaffected(self):
        """`variable:audio` written but nothing declares `audio` - not the
        shape #364 covers, so the original wording stands."""
        written = {
            "id": "sig",
            "steps": [
                task_step(
                    "resample_audio",
                    {"audio": "variable:audio", "target_sample_rate": 16000},
                )
            ],
        }
        expanded = {
            "id": "sig",
            "steps": [task_step("resample_audio", {"target_sample_rate": 16000})],
        }
        errors = task_signature_errors(expanded, written_definition=written)
        assert "variable" not in errors[0]
        assert "does not supply" in errors[0]["message"]


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
