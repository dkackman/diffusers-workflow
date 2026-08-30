"""Phase 0 plumbing: progress events, cancellation, manifests, cache identity."""

import json
import logging
import pytest
from unittest.mock import patch

from PIL import Image

from dw.events import RunContext, WorkflowCancelled, get_context, current_context
from dw.log_setup import setup_logging
from dw.result import Result
from dw.workflow import Workflow, pipeline_cache_key
from dw.pipeline_processors.pipeline import Pipeline


class FakeOutput:
    def __init__(self):
        self.images = [Image.new("RGB", (2, 2))]


class FakePipeline:
    """Stands in for a diffusers pipeline that supports step callbacks."""

    def __init__(self, steps=3):
        self.steps = steps

    def __call__(
        self,
        prompt=None,
        num_inference_steps=None,
        generator=None,
        callback_on_step_end=None,
    ):
        self._num_timesteps = self.steps
        for i in range(self.steps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, 0, {})
        return FakeOutput()


class FakePipelineNoCallback:
    """A pipeline whose signature does not name callback_on_step_end."""

    def __call__(self, prompt=None, num_inference_steps=None, generator=None):
        return FakeOutput()


def _workflow_def(steps=1):
    return {
        "id": "events_test",
        "steps": [
            {
                "name": f"gen{i}",
                "pipeline": {
                    "configuration": {
                        "component_type": "{FakePipeline}",
                        "no_generator": True,
                    },
                    "from_pretrained_arguments": {"model_name": f"model-{i}"},
                    "arguments": {"prompt": "p", "num_inference_steps": 3},
                },
            }
            for i in range(steps)
        ],
    }


def _run(workflow_def, context, fake=None):
    def mock_load(self, shared_components):
        self.pipeline = fake if fake is not None else FakePipeline()

    workflow = Workflow(workflow_def, "/tmp/test_output", "test.json")
    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            return workflow.run({}, previous_pipelines={}, context=context)


def test_progress_event_sequence():
    events = []
    context = RunContext(on_event=events.append)
    _run(_workflow_def(), context)

    names = [event["event"] for event in events]
    assert names[0] == "workflow_start"
    assert names[-1] == "workflow_end"
    assert "step_start" in names and "step_end" in names
    assert "iteration_start" in names
    assert names.count("pipeline_step") == 3

    start = events[0]
    assert start["workflow"] == "events_test"
    assert start["total_steps"] == 1 and start["steps"] == ["gen0"]
    assert isinstance(start["seed"], int)
    denoise = [event for event in events if event["event"] == "pipeline_step"]
    assert [event["step"] for event in denoise] == [1, 2, 3]
    assert all(event["total_steps"] == 3 for event in denoise)


def test_pre_cancelled_run_raises_before_any_step():
    events = []
    context = RunContext(on_event=events.append)
    context.cancel()
    with pytest.raises(WorkflowCancelled):
        _run(_workflow_def(), context)
    assert "step_start" not in [event["event"] for event in events]


def test_cancel_mid_denoise_stops_the_pipeline_call():
    events = []
    context = RunContext(on_event=events.append)

    original = events.append

    def cancelling_sink(event):
        original(event)
        if event["event"] == "pipeline_step":
            context.cancel()

    context._on_event = cancelling_sink
    with pytest.raises(WorkflowCancelled):
        _run(_workflow_def(), context)
    denoise = [event for event in events if event["event"] == "pipeline_step"]
    assert len(denoise) == 1, "the callback after the cancel must raise"


def test_no_callback_injection_without_signature_support():
    events = []
    context = RunContext(on_event=events.append)
    _run(_workflow_def(), context, fake=FakePipelineNoCallback())
    names = [event["event"] for event in events]
    assert "pipeline_step" not in names
    assert names[-1] == "workflow_end"


def test_manifest_and_save_paths(tmp_path):
    workflow_def = _workflow_def()
    workflow_def["steps"][0]["result"] = {"content_type": "image/png"}

    def mock_load(self, shared_components):
        self.pipeline = FakePipeline()

    workflow = Workflow(workflow_def, str(tmp_path), "test.json")
    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            workflow.run({}, previous_pipelines={})

    assert len(workflow.manifest) == 1
    entry = workflow.manifest[0]
    assert entry["step"] == "gen0"
    assert len(entry["files"]) == 1
    saved = entry["files"][0]
    assert saved.endswith(".png") and (tmp_path / saved.split("/")[-1]).exists()


def test_result_save_returns_json_paths(tmp_path):
    result = Result({"content_type": "application/json"})
    result.add_result({"a": 1})
    saved = result.save(str(tmp_path), "base")
    assert len(saved) == 1
    assert json.load(open(saved[0])) == {"a": 1}
    assert result.saved_files == saved


def test_result_save_disabled_returns_empty(tmp_path):
    result = Result({"content_type": "image/png", "save": False})
    result.add_result(FakeOutput())
    assert result.save(str(tmp_path), "base") == []


def test_pipeline_cache_key_identity():
    definition = _workflow_def()["steps"][0]["pipeline"]
    key = pipeline_cache_key(definition)

    # Per-call variation does not change identity
    changed_arguments = json.loads(json.dumps(definition))
    changed_arguments["arguments"]["prompt"] = "something else"
    changed_arguments["seed"] = 99
    assert pipeline_cache_key(changed_arguments) == key

    # A different model is a different pipeline
    changed_model = json.loads(json.dumps(definition))
    changed_model["from_pretrained_arguments"]["model_name"] = "other"
    assert pipeline_cache_key(changed_model) != key


def test_no_ambient_context_outside_a_run():
    assert current_context() is None
    # and the fallback context is inert
    context = get_context()
    context.emit("anything", value=1)
    context.check_cancelled()


def test_setup_logging_is_idempotent(tmp_path):
    log_path = str(tmp_path / "dw.log")
    setup_logging(log_path, "INFO")
    setup_logging(log_path, "INFO")
    logger = logging.getLogger("dw")
    assert len(logger.handlers) == 1

    setup_logging(log_path, "INFO", log_to_console=True)
    assert len(logger.handlers) == 2
    # leave global state clean for other tests
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_pipeline_reference_resolves_by_step_name_and_errors_when_missing():
    workflow = Workflow({"id": "ref", "steps": []}, "/tmp/test_output", "t.json")
    cache = {}

    def mock_load(self, shared_components):
        self.pipeline = FakePipeline()

    step_def = _workflow_def()["steps"][0]  # named gen0
    with patch.object(Pipeline, "load", mock_load):
        action = workflow.create_step_action(step_def, {}, cache, 1, "cpu")
        reference = workflow.create_step_action(
            {
                "name": "again",
                "pipeline_reference": {"reference_name": "gen0", "arguments": {}},
            },
            {},
            cache,
            1,
            "cpu",
        )
    assert reference.pipeline is action.pipeline

    with pytest.raises(ValueError, match="does not name"):
        workflow.create_step_action(
            {
                "name": "bad",
                "pipeline_reference": {"reference_name": "nope", "arguments": {}},
            },
            {},
            cache,
            1,
            "cpu",
        )


def test_sub_workflow_events_flow_into_parent_context(tmp_path):
    """A child workflow inherits the parent run's ambient context: its
    progress reaches the parent's sink and its pipelines are recorded in the
    shared touched set the worker's eviction relies on."""
    child = {
        "id": "child",
        "variables": {"prompt": "default"},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {
                        "component_type": "{FakePipeline}",
                        "no_generator": True,
                    },
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {
                        "prompt": "variable:prompt",
                        "num_inference_steps": 2,
                    },
                },
                "result": {"content_type": "image/png"},
            }
        ],
    }
    (tmp_path / "child.json").write_text(json.dumps(child))
    parent = {
        "id": "parent",
        "steps": [
            {
                "name": "delegate",
                "workflow": {"path": "child.json", "arguments": {"prompt": "hello"}},
            }
        ],
    }

    events = []
    context = RunContext(on_event=events.append)

    def mock_load(self, shared_components):
        self.pipeline = FakePipeline(steps=2)

    workflow = Workflow(parent, str(tmp_path), str(tmp_path / "parent.json"))
    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            workflow.run({}, previous_pipelines={}, context=context)

    workflows_seen = {e["workflow"] for e in events if "workflow" in e}
    assert workflows_seen == {"parent", "child"}
    # the child's saves roll up into the parent's manifest, so job history
    # and the gallery see every file the run produced
    child_files = [
        entry
        for entry in workflow.manifest
        if entry["step"] == "gen" and entry["files"]
    ]
    assert child_files, "child workflow saves must appear in the parent manifest"
    assert any(e["event"] == "pipeline_step" for e in events)
    assert context.touched_pipelines, "child pipelines must land in the shared set"
    assert current_context() is None, "context must deactivate after the run"
