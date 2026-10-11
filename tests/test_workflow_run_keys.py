"""pipeline_keys() answers the worker's question on a workflow switch: which
resident pipelines does the next workflow load anyway."""

from dw import workflow_run as workflow_run_module
from dw.step_cache import step_cache
from tests.test_workflow_step_cache import build_test_workflow_and_call_count_spy


def test_pipeline_keys_match_what_a_run_keys_on(tmp_path):
    step_cache.clear()
    workflow, _ = build_test_workflow_and_call_count_spy(str(tmp_path))
    try:
        pipelines = {}
        workflow.run({}, pipelines)
        assert pipelines
        assert workflow_run_module.pipeline_keys(workflow, {}) == set(pipelines)
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_an_unseeded_workflow_still_answers(tmp_path):
    step_cache.clear()
    workflow, _ = build_test_workflow_and_call_count_spy(str(tmp_path))
    del workflow.workflow_definition["seed"]
    try:
        assert workflow_run_module.pipeline_keys(workflow, {})
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_the_probe_writes_nothing(tmp_path):
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy(str(tmp_path))
    try:
        workflow_run_module.pipeline_keys(workflow, {})
        assert list(tmp_path.iterdir()) == []
        assert call_count() == 0
    finally:
        for p in workflow._test_patcher:
            p.stop()
