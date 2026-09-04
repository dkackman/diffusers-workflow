"""Integration tests: Workflow.run consults the step cache to skip
re-executing a step whose resolved definition, seed, and upstream results
are unchanged since the last run in this process.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

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
    """A step reached by a later step's pipeline_reference must never be
    served from cache, even when its own inputs are unchanged - otherwise
    create_step_action never runs for it on the later pass, so
    _pipeline_keys_by_step never records it and the referencing step's
    pipeline_reference lookup raises ValueError.

    Here A's inputs never change (a naive cache would serve A from cache on
    run 2) while B's inputs change every run (so B always re-executes and
    always needs to resolve its pipeline_reference to 'A' this run)."""
    step_cache.clear()
    workflow, call_count = build_pipeline_reference_workflow_and_call_count_spy()

    try:
        workflow.run({"prompt_b": "first"})
        workflow.run({"prompt_b": "second"})

        # Both A and B must have actually executed on both runs - A because
        # it is a pipeline_reference target, B because its own args changed.
        assert call_count() == 4
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_import_order_workflow_then_step_cache_does_not_cycle():
    """dw.workflow must not import dw.step_cache at module scope - that
    would cycle against step_cache's top-level `from .workflow import
    referenced_result_names` when dw.workflow is imported first."""
    import importlib
    import dw.workflow as wf_module
    import dw.step_cache as sc_module

    importlib.reload(wf_module)
    importlib.reload(sc_module)
