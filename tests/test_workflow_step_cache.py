"""Integration tests: Workflow.run consults the step cache to skip
re-executing a step whose resolved definition, seed, and upstream results
are unchanged since the last run in this process.
"""

import copy
import os
from unittest.mock import MagicMock, patch

import pytest

from dw.events import RunContext
from dw.step_cache import step_cache
from dw.step import Step
from dw.pipeline_processors.pipeline import Pipeline
from dw import workflow as workflow_module
from dw.workflow import Workflow


def _mock_pipeline_load(self, shared_components):
    self.pipeline = MagicMock()


class FakeResult:
    """Minimal stand-in for dw.result.Result - just enough surface for the
    workflow loop: .save() records saved_files, .result_list is returned
    from a run(), and a cache hit reads .saved_files without calling save()
    again."""

    def __init__(self):
        self.result_list = []
        self.saved_files = []

    def save(self, output_dir, base_name):
        # Files are really written: a cache hit only counts when every file
        # the cached entry names still exists on disk
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{base_name}.png")
        with open(path, "wb") as handle:
            handle.write(b"")
        self.saved_files = [path]
        return self.saved_files


def _workflow_def():
    return {
        "id": "test_step_cache",
        # Fixed so step_seed is identical across runs - Workflow.run draws a
        # fresh random seed per run when none is set, which would make the
        # cache key differ every time and defeat these tests' whole point.
        "seed": 42,
        "variables": {"prompt": "a cat"},
        "steps": [
            {
                "name": "generate",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-generate"},
                    "arguments": {"prompt": "variable:prompt"},
                },
            }
        ],
    }


def build_test_workflow_and_call_count_spy():
    """Returns (workflow, call_count) where call_count() reports how many
    times the step's body (Step.run) actually executed."""
    workflow = Workflow(_workflow_def(), "/tmp/test_output", "test.json")

    calls = {"n": 0}

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        calls["n"] += 1
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
    ]
    for p in patchers:
        p.start()
    workflow._test_patcher = patchers

    def call_count():
        return calls["n"]

    return workflow, call_count


def test_second_run_with_unchanged_step_reuses_cached_result():
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()

    try:
        workflow.run({})
        workflow.run({})

        assert call_count() == 1
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_second_run_with_changed_variable_recomputes_that_step():
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()

    try:
        workflow.run({"prompt": "a cat"})
        workflow.run({"prompt": "a dog"})

        assert call_count() == 2
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_uncopyable_step_argument_degrades_to_no_caching_rather_than_crashing():
    """A realized argument that copy.deepcopy chokes on makes the step
    uncacheable - the run must continue normally, not abort."""

    class NotCopyable:
        def __deepcopy__(self, memo):
            raise TypeError("this object cannot be copied")

    original_realize_args = workflow_module.realize_args

    def realize_and_poison(target, base_dir):
        original_realize_args(target, base_dir)
        if isinstance(target, list):  # the steps list, not the variables dict
            target[0]["pipeline"]["arguments"]["image"] = NotCopyable()

    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()

    try:
        with patch.object(workflow_module, "realize_args", realize_and_poison):
            workflow.run({})
            workflow.run({})

        # Neither run raised, and neither was served from cache
        assert call_count() == 2
    finally:
        for p in workflow._test_patcher:
            p.stop()


def _pipeline_reference_workflow_def():
    return {
        "id": "test_step_cache_pipeline_reference",
        "seed": 42,
        "variables": {"prompt_b": "x"},
        "steps": [
            {
                "name": "A",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-a"},
                    # Constant - never changes across runs, so A alone would
                    # otherwise be eligible for a cache hit.
                    "arguments": {"prompt": "a fixed prompt"},
                },
            },
            {
                "name": "B",
                "pipeline_reference": {
                    "reference_name": "A",
                    "arguments": {"prompt": "variable:prompt_b"},
                },
            },
        ],
    }


def build_pipeline_reference_workflow_and_call_count_spy():
    workflow = Workflow(
        _pipeline_reference_workflow_def(), "/tmp/test_output", "test.json"
    )

    calls = {"n": 0}

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        calls["n"] += 1
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
    ]
    for p in patchers:
        p.start()
    workflow._test_patcher = patchers

    def call_count():
        return calls["n"]

    return workflow, call_count


def test_pipeline_reference_still_resolves_when_referenced_step_is_cache_eligible():
    """A step reached by a later step's pipeline_reference may be served
    from cache - create_step_action runs on a hit too, so
    _pipeline_keys_by_step still records the referenced step's pipeline and
    the referencing step's lookup resolves.

    Here A's inputs never change (so A hits on run 2) while B's inputs
    change every run (so B always re-executes and always needs to resolve
    its pipeline_reference to 'A' this run)."""
    step_cache.clear()
    workflow, call_count = build_pipeline_reference_workflow_and_call_count_spy()

    try:
        workflow.run({"prompt_b": "first"})
        workflow.run({"prompt_b": "second"})

        # A executed once (run 2 was a cache hit), B executed both runs.
        assert call_count() == 3
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_cache_hit_still_touches_the_steps_pipeline():
    """A hit must run create_step_action's bookkeeping - it is the only
    caller of touch_pipeline, and the worker evicts every pipeline a run
    did not touch."""
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()
    pipelines = {}

    try:
        cold = RunContext()
        workflow.run({}, previous_pipelines=pipelines, context=cold)
        warm = RunContext()
        workflow.run({}, previous_pipelines=pipelines, context=warm)

        assert call_count() == 1  # the second run was a cache hit
        assert cold.touched_pipelines
        assert warm.touched_pipelines == cold.touched_pipelines
    finally:
        for p in workflow._test_patcher:
            p.stop()


def _shared_components_workflow_def():
    return {
        "id": "test_step_cache_shared_components",
        "seed": 42,
        "variables": {"prompt_b": "x"},
        "steps": [
            {
                "name": "A",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-a"},
                    "shared_components": ["text_encoder"],
                    # Constant across runs, so A is cache-eligible from run 2
                    "arguments": {"prompt": "a fixed prompt"},
                },
            },
            {
                "name": "B",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-b"},
                    "reused_components": ["text_encoder"],
                    "arguments": {"prompt": "variable:prompt_b"},
                },
            },
        ],
    }


def _mock_pipeline_load_with_sharing(self, shared_components):
    """Stand-in for Pipeline.load that keeps the sharing contract: a fresh
    load resolves what it reuses and publishes what it shares."""
    self.resolve_reused_components(shared_components)
    self.pipeline = MagicMock()
    self.publish_shared_components(shared_components)


def test_cache_hit_republishes_shared_components_for_a_later_cold_step():
    """A hit on the sharing step must still republish into this run's
    shared_components dict, or a later step that has to load fresh raises
    'Cannot reuse component ... Shared so far: nothing'."""
    step_cache.clear()
    workflow = Workflow(
        _shared_components_workflow_def(), "/tmp/test_output", "test.json"
    )
    pipelines = {}

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load_with_sharing),
    ]
    for p in patchers:
        p.start()
    try:
        workflow.run({"prompt_b": "first"}, previous_pipelines=pipelines)
        workflow.run({"prompt_b": "first"}, previous_pipelines=pipelines)

        # B's pipeline is gone (released, or evicted by the worker), so run
        # three must load it fresh while A is served from cache
        b_key = workflow._pipeline_keys_by_step["B"]
        pipelines.pop(b_key, None)

        seen = []
        original_resolve = Pipeline.resolve_reused_components

        def recording_resolve(self, shared_components):
            seen.append(sorted(shared_components))
            return original_resolve(self, shared_components)

        with patch.object(Pipeline, "resolve_reused_components", recording_resolve):
            workflow.run({"prompt_b": "second"}, previous_pipelines=pipelines)

        # Only B loaded on run three, and A's cache hit had republished the
        # component B reuses before that load resolved it
        assert seen == [["text_encoder"]]
    finally:
        for p in patchers:
            p.stop()


def test_release_pipeline_on_a_cache_hit_step_releases_its_pipeline():
    """release_pipeline is not a no-op on a hit - create_step_action ran,
    so the step's key is recorded and the pop finds it."""
    step_cache.clear()
    definition = _workflow_def()
    definition["steps"][0]["release_pipeline"] = True
    workflow = Workflow(definition, "/tmp/test_output", "test.json")

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
    ]
    for p in patchers:
        p.start()
    pipelines = {}
    try:
        workflow.run({}, previous_pipelines=pipelines)
        assert pipelines == {}
        # Put it back so the hit run has something to release
        pipelines[workflow._pipeline_keys_by_step["generate"]] = MagicMock()

        workflow.run({}, previous_pipelines=pipelines)

        assert pipelines == {}
    finally:
        for p in patchers:
            p.stop()


def test_workflow_without_a_seed_skips_the_step_cache_entirely():
    """A workflow that names no seed draws a fresh one every run, so no
    step can ever hit - it must not pay the deepcopy or pin a Result."""
    step_cache.clear()
    definition = _workflow_def()
    del definition["seed"]
    workflow = Workflow(definition, "/tmp/test_output", "test.json")

    copied = []
    real_deepcopy = copy.deepcopy

    def recording_deepcopy(value, *args, **kwargs):
        copied.append(value)
        return real_deepcopy(value, *args, **kwargs)

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
        patch.object(step_cache, "put"),
        patch.object(copy, "deepcopy", recording_deepcopy),
    ]
    started = [p.start() for p in patchers]
    put_mock = started[2]
    try:
        workflow.run({})
        workflow.run({})

        put_mock.assert_not_called()
        assert not [
            value
            for value in copied
            if isinstance(value, dict) and value.get("name") == "generate"
        ]
    finally:
        for p in patchers:
            p.stop()


def test_cache_hit_marks_its_manifest_entry_and_event_reused():
    """A hit republishes an earlier run's files - both the manifest entry
    and the step_end event say so, so nothing downstream credits this run
    with writing them."""
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()

    try:
        workflow.run({})
        assert not any("reused" in entry for entry in workflow.manifest)

        events = []
        workflow.run({}, context=RunContext(on_event=events.append))

        assert call_count() == 1
        assert workflow.manifest == [
            {
                "step": "generate",
                "files": workflow.manifest[0]["files"],
                "reused": True,
            }
        ]
        step_end = [e for e in events if e["event"] == "step_end"]
        assert len(step_end) == 1
        assert step_end[0]["reused"] is True
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_sub_workflow_of_a_seedless_parent_does_not_cache(tmp_path):
    """A parent that names no seed injects its randomly drawn one into the
    child, so the child would look seeded - and cacheable - while nothing it
    does can ever hit."""
    import json

    child = {
        "id": "child",
        "seed": 7,
        "variables": {"prompt": "a cat"},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-child"},
                    "arguments": {"prompt": "variable:prompt"},
                },
            }
        ],
    }
    (tmp_path / "child.json").write_text(json.dumps(child))
    parent = {
        "id": "parent",
        # No seed - drawn fresh every run
        "steps": [
            {
                "name": "delegate",
                "workflow": {"path": "child.json", "arguments": {"prompt": "a cat"}},
            }
        ],
    }

    step_cache.clear()
    workflow = Workflow(parent, str(tmp_path), str(tmp_path / "parent.json"))

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        # Stand in for Step.run, but still delegate to a sub-workflow the way
        # the real one does - the child's own step loop is what is under test
        if isinstance(step_action, Workflow):
            step_action.run(step_action.argument_template, previous_pipelines)
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
        patch.object(step_cache, "put"),
    ]
    started = [p.start() for p in patchers]
    put_mock = started[2]
    try:
        workflow.run({})
        put_mock.assert_not_called()
    finally:
        for p in patchers:
            p.stop()


def _two_step_def(second_reads_first, workflow_id="test_step_cache_two"):
    """A then B, where B either reads A's result or is independent of it."""
    b_arguments = {"prompt": "b fixed"}
    if second_reads_first:
        b_arguments["image"] = "previous_result:A"
    return {
        "id": workflow_id,
        "seed": 42,
        "variables": {"a_prompt": "one"},
        "steps": [
            {
                "name": "A",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-a"},
                    "arguments": {"prompt": "variable:a_prompt"},
                },
            },
            {
                "name": "B",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-b"},
                    "arguments": b_arguments,
                },
            },
        ],
    }


def _run_with_per_step_counts(workflow, arguments, fail_on=None):
    """Run `workflow`, counting Step.run per step name; optionally abort the
    run by raising before the named step's body executes (a cancel landing
    between two steps' puts)."""
    counts = {}

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        if fail_on is not None and self.name == fail_on:
            raise RuntimeError("cancelled")
        counts[self.name] = counts.get(self.name, 0) + 1
        result = FakeResult()
        result.result_list = [f"{self.name} artifact"]
        return result

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
    ]
    for p in patchers:
        p.start()
    try:
        try:
            workflow.run(arguments)
        except RuntimeError as ex:
            if fail_on is None or "cancelled" not in str(ex):
                raise
    finally:
        for p in patchers:
            p.stop()
    return counts


def test_renaming_the_workflow_id_does_not_reuse_the_old_ids_entry():
    """Saved files carry the workflow id, so an entry keyed by the bare step
    name would republish the previous id's paths and write none of its own."""
    step_cache.clear()
    first = Workflow(_workflow_def(), "/tmp/test_output", "test.json")
    assert _run_with_per_step_counts(first, {}) == {"generate": 1}

    renamed_def = _workflow_def()
    renamed_def["id"] = "test_step_cache_renamed"
    renamed = Workflow(renamed_def, "/tmp/test_output", "test.json")

    assert _run_with_per_step_counts(renamed, {}) == {"generate": 1}


def test_step_whose_upstream_was_recomputed_by_a_cancelled_run_misses():
    """A -> B, fixed seed. Change A, run, cancel after A's put but before
    B's: the next unchanged run must not serve B computed from the old A."""
    step_cache.clear()
    workflow = Workflow(_two_step_def(True), "/tmp/test_output", "test.json")

    assert _run_with_per_step_counts(workflow, {"a_prompt": "one"}) == {"A": 1, "B": 1}
    # A changes and is re-put; the run dies before B's put
    assert _run_with_per_step_counts(workflow, {"a_prompt": "two"}, fail_on="B") == {
        "A": 1
    }

    # Same inputs as the cancelled run: A hits, but B's cached result was
    # computed from the *old* A and must not be served
    assert _run_with_per_step_counts(workflow, {"a_prompt": "two"}) == {"B": 1}


def test_unreferenced_non_final_step_is_cached_without_its_result_list():
    """B does not read A and A is not the workflow's return value, so A's
    entry keeps its saved_files but drops the realized media."""
    step_cache.clear()
    workflow = Workflow(_two_step_def(False), "/tmp/test_output", "test.json")

    _run_with_per_step_counts(workflow, {})

    a_entry = step_cache._entries[("test_step_cache_two", "A")]
    b_entry = step_cache._entries[("test_step_cache_two", "B")]
    assert a_entry["result"].result_list == []
    assert a_entry["result"].saved_files
    # B is the last step - its result is the workflow's return value
    assert b_entry["result"].result_list == ["B artifact"]


def test_adding_a_downstream_reference_misses_on_a_result_that_was_not_retained():
    """A ran unreferenced (so its entry holds no result); a later run whose
    B reads A needs the real thing and must re-run A."""
    step_cache.clear()
    without = Workflow(_two_step_def(False), "/tmp/test_output", "test.json")
    assert _run_with_per_step_counts(without, {}) == {"A": 1, "B": 1}

    with_reference = Workflow(_two_step_def(True), "/tmp/test_output", "test.json")

    counts = _run_with_per_step_counts(with_reference, {})

    assert counts.get("A") == 1
