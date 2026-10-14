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
    "cache",
    "reference_names",
    "video_extensions",
    "hold_audio",
    "refine_strength",
    "guides",
    "content_types",
    "scalar_results",
    "locations",
    "reference_limits",
    "vram_estimate",
    "null_media",
    "adapters",
    "task_argument_domains",
    "dissolve_frames",
    "script_shot_names",
    "window_count",
    "video_sizes",
    "select",
    "chain_prompts",
    "guide_chain",
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
    "acceleration_warnings",
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
        "acceleration_warnings": lambda: validation.run_warning_check(
            workflow, "acceleration_warnings", None
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


class TestAccelerationWarnings:
    """dw/acceleration_warnings.py through the real path: a Workflow, its
    expansion, the registry check. Every code is a warning, never a refusal,
    and a value unknown before the run leaves its warning silent."""

    CODES = (
        "cache_short_schedule",
        "taylorseer_state_unestimated",
        "compile_fullgraph_with_cache",
        "attention_backend_unavailable",
    )

    @staticmethod
    def definition(
        cache=None,
        steps=20,
        components=None,
        attention_backend=None,
        variables=None,
        extra=None,
    ):
        configuration = {"component_type": "FluxPipeline"}
        if cache is not None:
            configuration["cache"] = cache
        if components is not None:
            configuration["components"] = components
        if attention_backend is not None:
            configuration["attention_backend"] = attention_backend
        arguments = {"prompt": "a cat"}
        if steps is not None:
            arguments["num_inference_steps"] = steps
        definition = {
            "id": "accel",
            "steps": [
                {
                    "name": "gen",
                    "pipeline": {
                        "configuration": configuration,
                        "from_pretrained_arguments": {
                            "model_name": "black-forest-labs/FLUX.1-dev"
                        },
                        "arguments": arguments,
                    },
                }
            ],
        }
        if variables is not None:
            definition["variables"] = variables
        definition.update(extra or {})
        return definition

    @staticmethod
    def workflow(definition):
        from dw.workflow import Workflow

        return Workflow(definition, None, "")

    def run(self, monkeypatch, definition, arguments=None, device="cuda", **fields):
        monkeypatch.setattr(validation, "get_device_type", lambda: device)
        return validation.run_warning_check(
            self.workflow(definition), "acceleration_warnings", arguments, **fields
        )

    def direct(self, definition, **fields):
        """A hand-built context, for the fields run_warning_check cannot
        leave None (capacity_gb) or the device it cannot name."""
        workflow = self.workflow(definition)
        source_indices = []
        expanded = workflow.expanded_definition(None, source_indices)
        from dw.acceleration_warnings import acceleration_warnings

        values = {
            "workflow": workflow,
            "expanded": expanded,
            "source_indices": source_indices,
            "device_type": "cuda",
        }
        values.update(fields)
        return acceleration_warnings(context(**values))

    @staticmethod
    def coded(warnings, code):
        return [w for w in warnings if f": {code}:" in w]

    # cache_short_schedule

    def test_a_threshold_cache_fires_below_twenty_steps_and_not_at_twenty(
        self, monkeypatch
    ):
        cache = {"type": "first_block"}
        short = self.run(monkeypatch, self.definition(cache, steps=19))
        enough = self.run(monkeypatch, self.definition(cache, steps=20))

        found = self.coded(short, "cache_short_schedule")
        assert len(found) == 1
        assert "19 steps" in found[0]
        assert "first_block" in found[0]
        assert self.coded(enough, "cache_short_schedule") == []

    @pytest.mark.parametrize("cache_type", ["first_block", "mag"])
    def test_both_threshold_caches_fire_at_twelve_steps(self, monkeypatch, cache_type):
        warnings = self.run(
            monkeypatch, self.definition({"type": cache_type}, steps=12)
        )

        assert len(self.coded(warnings, "cache_short_schedule")) == 1

    @pytest.mark.parametrize("cache_type", ["taylorseer", "faster", "text_kv"])
    def test_the_other_caches_are_not_held_to_the_schedule_rule(
        self, monkeypatch, cache_type
    ):
        warnings = self.run(monkeypatch, self.definition({"type": cache_type}, steps=9))

        assert self.coded(warnings, "cache_short_schedule") == []

    def test_steps_come_from_a_variable_default(self, monkeypatch):
        definition = self.definition(
            {"type": "first_block"},
            steps="variable:steps",
            variables={"steps": 12},
        )

        warnings = self.run(monkeypatch, definition)

        found = self.coded(warnings, "cache_short_schedule")
        assert len(found) == 1 and "12 steps" in found[0]

    def test_steps_come_from_an_argument_over_the_default(self, monkeypatch):
        definition = self.definition(
            {"type": "first_block"},
            steps="variable:steps",
            variables={"steps": 30},
        )

        short = self.run(monkeypatch, definition, {"steps": 9})
        default = self.run(monkeypatch, definition)

        assert len(self.coded(short, "cache_short_schedule")) == 1
        assert self.coded(default, "cache_short_schedule") == []

    def test_steps_that_resolve_to_nothing_are_silent(self, monkeypatch):
        definition = self.definition(
            {"type": "first_block"},
            steps="variable:steps",
            variables={"steps": None},
        )

        assert (
            self.coded(self.run(monkeypatch, definition), "cache_short_schedule") == []
        )

    def test_no_num_inference_steps_is_silent(self, monkeypatch):
        definition = self.definition({"type": "first_block"}, steps=None)

        assert (
            self.coded(self.run(monkeypatch, definition), "cache_short_schedule") == []
        )

    # taylorseer_state_unestimated

    def test_taylorseer_always_says_its_state_is_unestimated(self, monkeypatch):
        default = self.run(monkeypatch, self.definition({"type": "taylorseer"}))
        second = self.run(
            monkeypatch, self.definition({"type": "taylorseer", "max_order": 2})
        )

        first = self.coded(default, "taylorseer_state_unestimated")
        higher = self.coded(second, "taylorseer_state_unestimated")
        assert len(first) == 1 and "this one is 1" in first[0]
        assert len(higher) == 1 and "this one is 2" in higher[0]

    def test_taylorseer_names_the_declared_estimate_against_the_capacity(
        self, monkeypatch
    ):
        definition = self.definition(
            {"type": "taylorseer"},
            extra={
                "vram_estimate": {
                    "base_gb": 10,
                    "gb_per_voxel": 0,
                    "voxel_variables": [],
                }
            },
        )

        warnings = self.run(monkeypatch, definition, capacity_gb=24)

        found = self.coded(warnings, "taylorseer_state_unestimated")
        assert len(found) == 1
        assert "10.0 of 24.0 GB" in found[0]

    def test_taylorseer_without_capacity_carries_no_gb_clause(self):
        definition = self.definition(
            {"type": "taylorseer"},
            extra={
                "vram_estimate": {
                    "base_gb": 10,
                    "gb_per_voxel": 0,
                    "voxel_variables": [],
                }
            },
        )

        warnings = self.direct(definition, capacity_gb=None)

        found = self.coded(warnings, "taylorseer_state_unestimated")
        assert len(found) == 1 and "GB" not in found[0]

    def test_taylorseer_without_a_declared_estimate_carries_no_gb_clause(
        self, monkeypatch
    ):
        warnings = self.run(
            monkeypatch, self.definition({"type": "taylorseer"}), capacity_gb=24
        )

        found = self.coded(warnings, "taylorseer_state_unestimated")
        assert len(found) == 1 and "GB" not in found[0]

    # compile_fullgraph_with_cache

    def test_fullgraph_with_a_cache_fires_at_the_component(self, monkeypatch):
        definition = self.definition(
            {"type": "faster"},
            components={"transformer": {"compile": {"fullgraph": True}}},
        )

        warnings = self.run(monkeypatch, definition)

        found = self.coded(warnings, "compile_fullgraph_with_cache")
        assert len(found) == 1
        assert (
            found[0].split(": ")[0].endswith("components.transformer.compile.fullgraph")
        )

    def test_partial_graph_with_a_cache_is_silent(self, monkeypatch):
        definition = self.definition(
            {"type": "faster"},
            components={"transformer": {"compile": {"fullgraph": False}}},
        )

        warnings = self.run(monkeypatch, definition)

        assert self.coded(warnings, "compile_fullgraph_with_cache") == []

    def test_fullgraph_without_a_cache_is_silent(self, monkeypatch):
        definition = self.definition(
            components={"transformer": {"compile": {"fullgraph": True}}}
        )

        warnings = self.run(monkeypatch, definition)

        assert self.coded(warnings, "compile_fullgraph_with_cache") == []

    # attention_backend_unavailable

    @staticmethod
    def with_backend(level, backend):
        if level == "pipeline":
            return {"attention_backend": backend}
        return {"components": {"transformer": {"attention_backend": backend}}}

    @pytest.mark.parametrize("level", ["pipeline", "component"])
    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_a_cuda_only_backend_fires_off_cuda(self, monkeypatch, level, device):
        definition = self.definition(**self.with_backend(level, "sage_hub"))

        warnings = self.run(monkeypatch, definition, device=device)

        found = self.coded(warnings, "attention_backend_unavailable")
        assert len(found) == 1
        assert "'sage_hub'" in found[0] and f"'{device}'" in found[0]
        if level == "component":
            assert "components.transformer.attention_backend" in found[0]

    @pytest.mark.parametrize("level", ["pipeline", "component"])
    @pytest.mark.parametrize("device", ["cuda", "cuda:1"])
    def test_a_cuda_only_backend_is_silent_on_cuda(self, monkeypatch, level, device):
        definition = self.definition(**self.with_backend(level, "sage_hub"))

        warnings = self.run(monkeypatch, definition, device=device)

        assert self.coded(warnings, "attention_backend_unavailable") == []

    @pytest.mark.parametrize("level", ["pipeline", "component"])
    def test_an_unknown_device_is_silent(self, level):
        definition = self.definition(**self.with_backend(level, "sage_hub"))

        warnings = self.direct(definition, device_type=None)

        assert self.coded(warnings, "attention_backend_unavailable") == []

    @pytest.mark.parametrize("backend", ["native", "_native_cudnn", "flex"])
    def test_a_native_family_backend_is_silent_on_mps(self, monkeypatch, backend):
        definition = self.definition(attention_backend=backend)

        warnings = self.run(monkeypatch, definition, device="mps")

        assert self.coded(warnings, "attention_backend_unavailable") == []

    def test_an_unresolvable_backend_is_silent_on_mps(self, monkeypatch):
        definition = self.definition(
            attention_backend="variable:backend", variables={"backend": None}
        )

        warnings = self.run(monkeypatch, definition, device="mps")

        assert self.coded(warnings, "attention_backend_unavailable") == []

    # shape of the answer

    def test_a_for_each_step_says_each_warning_once(self, monkeypatch):
        """Members share their written path, and each one's message names
        its member - the warning is said once there, not once a member"""
        definition = self.definition(
            {"type": "taylorseer"},
            variables={"shots": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        )
        definition["steps"][0]["for_each"] = "variable:shots"
        definition["steps"][0]["name"] = "gen"

        warnings = self.run(monkeypatch, definition)

        assert len(self.coded(warnings, "taylorseer_state_unestimated")) == 1

    def test_warnings_never_refuse(self, monkeypatch):
        definition = self.definition(
            {"type": "first_block"},
            steps=9,
            components={
                "transformer": {
                    "compile": {"fullgraph": True},
                    "attention_backend": "sage_hub",
                }
            },
        )
        monkeypatch.setattr(validation, "get_device_type", lambda: "mps")

        warnings = self.run(monkeypatch, definition, device="mps")

        assert len(self.coded(warnings, "cache_short_schedule")) == 1
        assert len(self.coded(warnings, "compile_fullgraph_with_cache")) == 1
        assert len(self.coded(warnings, "attention_backend_unavailable")) == 1
        assert self.workflow(definition).validation_errors() == []

    # the catalog

    def test_no_catalog_workflow_trips_a_device_independent_acceleration_warning(
        self, monkeypatch
    ):
        """The catalog defaults are held to the same rules a hand-written
        workflow meets. attention_backend_unavailable is left out on purpose:
        a CUDA-only backend is a legitimate catalog choice."""
        import glob

        from dw.workflow import Workflow

        root = pathlib.Path(__file__).resolve().parent.parent
        paths = sorted(
            glob.glob(str(root / "workflows" / "**" / "*.json"), recursive=True)
            + glob.glob(
                str(root / "dw" / "workflows" / "**" / "*.json"), recursive=True
            )
        )
        monkeypatch.setattr(validation, "get_device_type", lambda: "cuda")
        checked = 0
        offenders = []
        for path in paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    definition = json.load(handle)
            except (OSError, ValueError):
                continue
            if not isinstance(definition, dict) or "steps" not in definition:
                continue
            workflow = Workflow(definition, None, path)
            warnings = validation.run_warning_check(
                workflow, "acceleration_warnings", None
            )
            checked += 1
            for code in self.CODES[:3]:
                offenders += [
                    f"{os.path.relpath(path, root)}: {w}"
                    for w in self.coded(warnings, code)
                ]

        assert checked
        assert offenders == []


class TestCacheVariable:
    """A `cache: "variable:cache"` is checked as the object it substitutes to
    (#865): the raw schema only sees the reference string, so this is what
    stops a bad cache object from passing validate and failing at load."""

    @staticmethod
    def definition(default=None):
        return {
            "id": "cache-variable",
            "variables": {"cache": default},
            "steps": [
                {
                    "name": "gen",
                    "pipeline": {
                        "configuration": {
                            "component_type": "FluxPipeline",
                            "cache": "variable:cache",
                        },
                        "from_pretrained_arguments": {
                            "model_name": "black-forest-labs/FLUX.1-dev"
                        },
                        "arguments": {"prompt": "a cat", "num_inference_steps": 30},
                    },
                }
            ],
        }

    @staticmethod
    def errors(definition, arguments=None):
        from dw.workflow import Workflow

        return Workflow(definition, None, "").validation_errors(arguments)

    @staticmethod
    def expanded_configuration(definition, arguments=None):
        from dw.workflow import Workflow

        expanded = Workflow(definition, None, "").expanded_definition(arguments)
        return expanded["steps"][0]["pipeline"]["configuration"]

    def test_a_valid_object_validates_and_reaches_the_step(self):
        cache = {"type": "first_block", "threshold": 0.05}

        assert self.errors(self.definition(), {"cache": cache}) == []
        assert (
            self.expanded_configuration(self.definition(), {"cache": cache})["cache"]
            == cache
        )

    def test_null_drops_the_key(self):
        assert self.errors(self.definition()) == []
        assert "cache" not in self.expanded_configuration(self.definition())

    def test_an_unknown_type_is_refused_at_the_field(self):
        errors = self.errors(self.definition(), {"cache": {"type": "bogus"}})

        assert [e["path"] for e in errors] == [
            "steps[0].pipeline.configuration.cache.type"
        ]
        assert "bogus" in errors[0]["message"]

    def test_a_bare_string_is_refused(self):
        errors = self.errors(self.definition(), {"cache": "first_block"})

        assert [e["path"] for e in errors] == ["steps[0].pipeline.configuration.cache"]
        assert "object" in errors[0]["message"]

    def test_a_missing_type_is_refused(self):
        errors = self.errors(self.definition(), {"cache": {"threshold": 0.05}})

        assert [e["path"] for e in errors] == ["steps[0].pipeline.configuration.cache"]
        assert "'type'" in errors[0]["message"]

    def test_an_invalid_default_is_refused_with_no_arguments(self):
        errors = self.errors(self.definition(default={"type": "bogus"}))

        assert [e["path"] for e in errors] == [
            "steps[0].pipeline.configuration.cache.type"
        ]

    def test_a_literal_string_other_than_a_variable_fails_the_schema(self):
        definition = self.definition()
        definition["steps"][0]["pipeline"]["configuration"]["cache"] = "first_block"

        errors = self.errors(definition)

        assert [e["path"] for e in errors] == ["steps[0].pipeline.configuration.cache"]

    def test_a_literal_object_still_validates(self):
        definition = self.definition()
        definition["steps"][0]["pipeline"]["configuration"]["cache"] = {
            "type": "first_block",
            "threshold": 0.05,
        }

        assert self.errors(definition) == []


class TestCacheVariableOnAForEachTemplate:
    """The same check through a real for_each template: every member's
    substituted cache is checked, and one bad value is said once."""

    PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "workflows",
        "templates",
        "minimax",
        "dialogue-short.json",
    )

    def errors(self, arguments):

        from dw.workflow import Workflow

        with open(self.PATH, encoding="utf-8") as handle:
            definition = json.load(handle)
        errors = Workflow(definition, None, self.PATH).validation_errors(arguments)
        return [e for e in errors if ".cache" in e["path"]]

    def test_a_valid_object_is_clean(self):
        assert self.errors({"cache": {"type": "first_block", "threshold": 0.05}}) == []

    def test_an_unknown_type_is_refused_once(self):
        errors = self.errors({"cache": {"type": "bogus"}})

        assert len(errors) == 1, errors
        assert errors[0]["path"].endswith("pipeline.configuration.cache.type")
        assert "member 'shot@" in errors[0]["message"]

    def test_a_bare_string_is_refused(self):
        errors = self.errors({"cache": "first_block"})

        assert len(errors) == 1, errors
        assert errors[0]["path"].endswith("pipeline.configuration.cache")
