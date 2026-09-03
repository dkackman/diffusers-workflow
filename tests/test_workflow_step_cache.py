"""Integration tests: Workflow.run consults the step cache to skip
re-executing a step whose resolved definition, seed, and upstream results
are unchanged since the last run in this process.
"""

from unittest.mock import MagicMock, patch

from dw.step_cache import step_cache
from dw.step import Step
from dw.pipeline_processors.pipeline import Pipeline
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
        self.saved_files = [f"{base_name}.png"]
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


def test_import_order_workflow_then_step_cache_does_not_cycle():
    """dw.workflow must not import dw.step_cache at module scope - that
    would cycle against step_cache's top-level `from .workflow import
    referenced_result_names` when dw.workflow is imported first."""
    import importlib
    import dw.workflow as wf_module
    import dw.step_cache as sc_module

    importlib.reload(wf_module)
    importlib.reload(sc_module)
