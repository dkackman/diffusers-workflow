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


@pytest.fixture(autouse=True)
def _clear_step_cache_after_test():
    """step_cache is a process-global singleton - leaving it populated
    after a test would let it leak cached results into unrelated tests
    that run later in the same process."""
    yield
    step_cache.clear()


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

        workflow.run({"prompt_b": "second"}, previous_pipelines=pipelines)
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
