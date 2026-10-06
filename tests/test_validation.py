"""dw/validation.py - one finding type, one context per request, one
exception policy for every check (B10)."""

import ast
import copy
import json
import logging
import os
import pathlib
import wave

import pytest

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
        # the exception's type only: its text may carry internals, and the
        # log has it
        assert internal.message == (
            "check 'broken' failed (RuntimeError) - the server log has the detail"
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
            "internal: warning check 'noisy' failed (RuntimeError) - "
            "the server log has the detail",
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
    "hold_audio",
    "refine_strength",
    "content_types",
    "scalar_results",
    "locations",
    "reference_limits",
    "vram_estimate",
    "null_media",
    "adapters",
    "task_argument_domains",
    "voices",
    "script_lines",
    "dissolve_frames",
    "window_count",
    "video_sizes",
    "select",
    "chain_prompts",
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
            "message": "check 'select' failed (RuntimeError) - "
            "the server log has the detail",
        }
    ]


def test_a_childs_pathless_finding_is_placed_at_the_sub_workflow_step(
    tmp_path, monkeypatch
):
    """A child's internal finding has no path of its own; the parent names
    where it came from, not "... -> None"."""
    registry = list(validation.ERROR_CHECKS)
    index = [check.name for check in registry].index("select")
    registry[index] = Check("select", boom)
    monkeypatch.setattr(validation, "ERROR_CHECKS", registry)
    from dw.workflow import Workflow

    (tmp_path / "child.json").write_text(json.dumps(_for_each_definition()))
    parent = {
        "id": "parent",
        "steps": [{"name": "delegate", "workflow": {"path": "child.json"}}],
    }
    workflow = Workflow(parent, str(tmp_path), str(tmp_path / "parent.json"))

    paths = [error["path"] for error in workflow.validation_errors()]

    assert "steps[0].workflow.path" in paths
    assert not any(isinstance(p, str) and p.endswith("None") for p in paths)


WARNING_ORDER = [
    "workflow_argument_warnings",
    "constraint_warnings",
    "entry_field_warnings",
    "unseeded_cache_warnings",
    "adapter_warnings",
    "null_variable_argument_warnings",
    "sub_workflow_warnings",
    "slice_past_end_warnings",
    "shot_span_warnings",
    "inherited_vram_warnings",
]


def test_the_warning_registry_runs_in_its_pinned_order():
    """warning order is part of the /api/validate response - admit()'s
    helper order before the registry existed"""
    assert [check.name for check in validation.WARNING_CHECKS] == WARNING_ORDER


def _unexpandable_definition():
    """Passes the schema; its for_each names a variable holding no list,
    which the expansion refuses."""
    definition = _for_each_definition()
    definition["variables"]["shots"] = "not a list"
    return definition


def test_building_a_context_does_not_expand(tmp_path):
    """The gates in validation_errors answer an expansion failure as a
    finding; a context built ahead of them (admit builds one per request)
    must not raise that failure first."""
    from dw.workflow import workflow_from_definition

    workflow = workflow_from_definition(
        _unexpandable_definition(), str(tmp_path), str(tmp_path), None
    )

    context = validation.workflow_context(workflow)
    errors = workflow.validation_errors(context=context)

    assert errors == workflow.validation_errors()
    assert errors and errors[0]["path"] is not None
    assert "check '" not in errors[0]["message"]


def test_a_context_with_other_arguments_is_a_caller_bug(tmp_path):
    import pytest

    from dw.workflow import workflow_from_definition

    workflow = workflow_from_definition(
        _for_each_definition(), str(tmp_path), str(tmp_path), None
    )
    arguments = {"shots": [{"name": "x", "text": "X"}]}
    context = validation.workflow_context(workflow, arguments)

    # the context alone, or the context with the same arguments, is fine
    assert workflow.validation_errors(context=context) == []
    assert workflow.validation_errors(arguments, context=context) == []
    with pytest.raises(ValueError):
        workflow.validation_errors({"shots": []}, context=context)
    with pytest.raises(ValueError):
        workflow.validation_errors(composing=["/a.json"], context=context)


def test_a_warning_method_on_an_unexpandable_definition_says_it_failed(tmp_path):
    """Each Workflow warning method used to answer [] when its expansion
    raised; it runs its registry check now, and a check that raises is one
    internal warning (B10)."""
    from dw.workflow import workflow_from_definition

    workflow = workflow_from_definition(
        _unexpandable_definition(), str(tmp_path), str(tmp_path), None
    )
    calls = {
        "adapter_warnings": lambda: validation.run_warning_check(
            workflow, "adapter_warnings", None
        ),
        "null_variable_argument_warnings": (
            lambda: validation.run_warning_check(
                workflow, "null_variable_argument_warnings", None
            )
        ),
        "sub_workflow_warnings": lambda: validation.run_warning_check(
            workflow, "sub_workflow_warnings", None
        ),
        "slice_past_end_warnings": lambda: validation.run_warning_check(
            workflow, "slice_past_end_warnings", None
        ),
        "shot_span_warnings": lambda: validation.run_warning_check(
            workflow, "shot_span_warnings", None
        ),
        "inherited_vram_warnings": (
            lambda: validation.run_warning_check(
                workflow,
                "inherited_vram_warnings",
                None,
                ceiling_index={"identity": {}},
            )
        ),
    }
    for name, call in calls.items():
        assert call() == [
            f"internal: warning check '{name}' failed (ForEachError) - "
            "the server log has the detail"
        ], name


def test_an_unknown_warning_check_name_is_a_key_error():
    with pytest.raises(KeyError):
        validation.warning_check("no_such_check")


def test_a_pipeline_configuration_naming_teacache_is_refused():
    from dw.workflow import Workflow

    definition = {
        "id": "teacache-removed",
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {
                        "component_type": "{Fake}",
                        "teacache": {"rel_l1_thresh": 0.4},
                    },
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {"prompt": "d"},
                },
            }
        ],
    }

    errors = Workflow(definition, ".", "teacache-removed.json").validation_errors()

    assert any("teacache" in str(error["message"]) for error in errors), errors
