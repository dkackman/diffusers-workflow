"""dw/validation.py - one finding type, one context per request, one
exception policy for every check (B10)."""

import ast
import copy
import logging
import os
import pathlib
import wave

from dw import validation
from dw.validation import (
    Check,
    Finding,
    ValidationContext,
    run_checks,
    to_errors,
    to_warnings,
)


def context(**overrides):
    values = {
        "workflow": object(),
        "arguments": None,
        "expanded": {"steps": []},
        "source_indices": [],
        "base_dir": None,
    }
    values.update(overrides)
    return ValidationContext(**values)


def write_wav(path, frames=800, sample_rate=8000):
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def boom(_context):
    raise RuntimeError("disk on fire")


def one_error(_context):
    return [{"path": "steps[0].name", "message": "bad name"}]


def another_error(_context):
    return [{"path": "steps[1].name", "message": "worse name"}]


class TestExceptionPolicy:
    def test_a_raising_check_is_one_internal_error_and_the_rest_still_run(self, caplog):
        checks = [
            Check("first", one_error),
            Check("broken", boom),
            Check("last", another_error),
        ]
        with caplog.at_level(logging.ERROR):
            findings = run_checks(context(), checks, "error")

        assert [f.kind for f in findings] == ["first", "internal", "last"]
        internal = findings[1]
        assert internal.severity == "error"
        assert internal.path is None
        assert internal.message == (
            "check 'broken' failed (RuntimeError: disk on fire) - "
            "the server log has the traceback"
        )
        # the traceback goes to the log, at ERROR
        records = [r for r in caplog.records if "broken" in r.getMessage()]
        assert records and records[0].levelno == logging.ERROR
        assert records[0].exc_info is not None

    def test_a_raising_warning_source_is_one_warning_and_never_an_error(self):
        checks = [Check("noisy", boom), Check("fine", lambda c: ["a.b: careful"])]

        findings = run_checks(context(), checks, "warning")

        assert [f.severity for f in findings] == ["warning", "warning"]
        assert to_warnings(findings) == [
            "internal: warning check 'noisy' failed (RuntimeError: disk on fire)",
            "a.b: careful",
        ]
        assert to_errors([f for f in findings if f.severity == "error"]) == []


class TestLegacyShapesRoundTrip:
    def test_error_dicts_keep_their_extra_keys(self):
        legacy = [
            {"path": "steps[0].task.arguments.x", "message": "unset", "variable": "x"},
            {"path": "steps[1]", "message": "plain"},
            {"path": None, "message": "no path at all"},
        ]

        findings = run_checks(context(), [Check("legacy", lambda c: legacy)], "error")

        assert findings[0].extra == {"variable": "x"}
        assert to_errors(findings) == legacy
        # key order is part of the JSON a caller reads
        assert [list(e) for e in to_errors(findings)] == [list(e) for e in legacy]

    def test_warning_strings_round_trip(self):
        legacy = [
            "steps[0].pipeline.arguments.lora: turbo lora switched off",
            "a message with no path",
            "variables.shots: one: two",
        ]

        findings = run_checks(context(), [Check("legacy", lambda c: legacy)], "warning")

        assert findings[0].path == "steps[0].pipeline.arguments.lora"
        assert findings[0].message == "turbo lora switched off"
        assert findings[1].path is None
        assert to_warnings(findings) == legacy

    def test_a_finding_without_a_path_serializes_as_its_bare_message(self):
        warning = Finding(severity="warning", kind="k", path=None, message="m")
        error = Finding(severity="error", kind="k", path=None, message="m")
        assert to_warnings([warning, error]) == ["m"]
        assert to_errors([warning, error]) == [{"path": None, "message": "m"}]


class TestProbeMemo:
    def test_probe_is_memoized_within_one_context(self, tmp_path):
        path = tmp_path / "tone.wav"
        write_wav(path)
        ctx = context()

        first = ctx.probe(str(path))

        assert first["kind"] == "audio"
        assert ctx.probe(str(path)) is first

    def test_the_memo_is_keyed_by_real_path(self, tmp_path):
        path = tmp_path / "tone.wav"
        write_wav(path)
        alias = tmp_path / "alias.wav"
        os.symlink(path, alias)
        ctx = context()

        assert ctx.probe(str(alias)) is ctx.probe(str(path))

    def test_two_contexts_do_not_share_entries(self, tmp_path):
        path = tmp_path / "tone.wav"
        write_wav(path)
        one, two = context(), context()

        assert one.probe(str(path)) is not two.probe(str(path))
        # a replaced file is seen by the next validation, not served stale
        write_wav(path, frames=1600)
        assert (
            context().probe(str(path))["duration_seconds"]
            > (one.probe(str(path))["duration_seconds"])
        )


def test_validation_does_not_import_the_workflow_module():
    tree = ast.parse(pathlib.Path(validation.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
            modules.extend(f"{node.module or ''}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
    assert not [m for m in modules if m.split(".")[-1] == "workflow"], modules


ERROR_ORDER = [
    "previous_result_references",
    "subfolders",
    "fps",
    "reference_names",
    "video_extensions",
    "content_types",
    "scalar_results",
    "locations",
    "reference_limits",
    "vram_estimate",
    "null_media",
    "adapters",
    "task_argument_domains",
    "voices",
    "dissolve_frames",
    "video_sizes",
    "select",
    "task_signatures",
    "component_types",
    "component_names",
    "constraints",
    "kernel_availability",
    "sub_workflows",
]


def test_the_error_registry_runs_in_its_pinned_order():
    """error order is part of the /api/validate response; a reorder changes
    what the editor shows first"""
    assert [check.name for check in validation.ERROR_CHECKS] == ERROR_ORDER


def _for_each_definition():
    return {
        "id": "memo",
        "variables": {"shots": [{"name": "a", "text": "A"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": ["item:text"]},
                },
                "result": {"content_type": "text/plain"},
            }
        ],
    }


def test_validation_leaves_the_definition_and_its_expansion_memo_alone(tmp_path):
    """The expansion is memoized per arguments with no invalidation, which
    holds only because nothing mutates the definition after construction -
    validation included."""
    from dw.workflow import workflow_from_definition

    workflow = workflow_from_definition(
        _for_each_definition(), str(tmp_path), str(tmp_path), None
    )
    arguments = {"shots": [{"name": "x", "text": "X"}]}
    written = copy.deepcopy(workflow.workflow_definition)
    expanded = workflow.expanded_definition(arguments)

    assert workflow.validation_errors(arguments=arguments) == []
    assert workflow.validation_errors() == []

    assert workflow.workflow_definition == written
    assert workflow.expanded_definition(arguments) == expanded


def test_validation_errors_answers_a_raising_check_as_an_internal_error(
    tmp_path, monkeypatch
):
    registry = list(validation.ERROR_CHECKS)
    index = [check.name for check in registry].index("select")
    registry[index] = Check("select", boom)
    monkeypatch.setattr(validation, "ERROR_CHECKS", registry)
    from dw.workflow import workflow_from_definition

    workflow = workflow_from_definition(
        _for_each_definition(), str(tmp_path), str(tmp_path), None
    )

    assert workflow.validation_errors() == [
        {
            "path": None,
            "message": "check 'select' failed (RuntimeError: disk on fire) - "
            "the server log has the traceback",
        }
    ]
