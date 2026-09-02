"""Phase events: what a run is doing when it is not counting denoise steps.

A step's wall clock is mostly load, decode and encode, none of which the
step counter can see. These cover the emits that fill that silence.
"""

from types import SimpleNamespace
from unittest.mock import patch
import pytest
from PIL import Image

from dw.events import RunContext, WorkflowCancelled, emit_phase
from dw.events import activate_context, deactivate_context
from dw.result import Result
from dw.workflow import Workflow
from dw.pipeline_processors.pipeline import Pipeline
from dw.tasks import task as task_module
from dw.tasks.task import Task


class FakeOutput:
    def __init__(self):
        self.images = [Image.new("RGB", (2, 2))]


class FakePipeline:
    """A diffusers pipeline that honors the step callback."""

    def __call__(
        self,
        prompt=None,
        num_inference_steps=None,
        generator=None,
        callback_on_step_end=None,
    ):
        self._num_timesteps = 3
        for i in range(3):
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, 0, {})
        return FakeOutput()


class FakePipelineNoCallback:
    """One whose signature names no callback - it reports nothing itself."""

    def __call__(self, prompt=None, num_inference_steps=None, generator=None):
        return FakeOutput()


def _pipeline_workflow(save=False):
    step = {
        "name": "gen",
        "pipeline": {
            "configuration": {
                "component_type": "{FakePipeline}",
                "no_generator": True,
            },
            "from_pretrained_arguments": {"model_name": "acme/model"},
            "arguments": {"prompt": "p", "num_inference_steps": 3},
        },
    }
    if save:
        step["result"] = {"content_type": "image/png"}
    return {"id": "phase_test", "steps": [step]}


def _run(
    workflow_def, context, output_dir="/tmp/test_output", pipelines=None, fake=None
):
    def mock_load(self, shared_components):
        self.pipeline = fake if fake is not None else FakePipeline()

    workflow = Workflow(workflow_def, output_dir, "test.json")
    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            workflow.run({}, previous_pipelines=pipelines, context=context)


def _phases(events):
    return [
        (event["phase"], event["detail"])
        for event in events
        if event["event"] == "phase"
    ]


def test_a_run_reports_every_phase_in_order(tmp_path):
    events = []
    _run(
        _pipeline_workflow(save=True), RunContext(on_event=events.append), str(tmp_path)
    )

    assert [phase for phase, _ in _phases(events)] == [
        "loading",
        "generating",
        "decoding",
        "saving",
    ]


def test_loading_names_the_model_it_is_waiting_on():
    events = []
    _run(_pipeline_workflow(), RunContext(on_event=events.append))

    loading = [detail for phase, detail in _phases(events) if phase == "loading"]
    assert loading == ["acme/model"]


def test_a_cache_hit_is_reported_as_cached_not_loading():
    pipelines = {}
    _run(_pipeline_workflow(), RunContext(), pipelines=pipelines)

    events = []
    _run(
        _pipeline_workflow(),
        RunContext(on_event=events.append),
        pipelines=pipelines,
    )

    phases = _phases(events)
    assert ("cached", "acme/model") in phases
    assert "loading" not in [phase for phase, _ in phases]


def test_decoding_is_reported_after_the_last_denoise_step():
    events = []
    _run(_pipeline_workflow(), RunContext(on_event=events.append))

    names = [
        event.get("phase") if event["event"] == "phase" else event["event"]
        for event in events
        if event["event"] in ("phase", "pipeline_step")
    ]
    # Three steps, then the decode - not before, and not once per step
    assert names == [
        "loading",
        "generating",
        "pipeline_step",
        "pipeline_step",
        "pipeline_step",
        "decoding",
    ]


def test_a_pipeline_without_a_step_callback_still_reports_generating():
    events = []
    _run(
        _pipeline_workflow(),
        RunContext(on_event=events.append),
        fake=FakePipelineNoCallback(),
    )

    phases = [phase for phase, _ in _phases(events)]
    assert phases == ["loading", "generating"], "a dark step is the thing to avoid"


def test_saving_reports_the_content_type_it_is_writing(tmp_path):
    events = []
    context = RunContext(on_event=events.append)
    from dw.events import activate_context, deactivate_context

    token = activate_context(context)
    try:
        result = Result({"content_type": "application/json"})
        result.add_result({"a": 1})
        result.save(str(tmp_path), "base")
    finally:
        deactivate_context(token)

    assert _phases(events) == [("saving", "application/json")]


def test_saving_reports_nothing_when_saving_is_disabled(tmp_path):
    events = []
    context = RunContext(on_event=events.append)
    from dw.events import activate_context, deactivate_context

    token = activate_context(context)
    try:
        result = Result({"content_type": "image/png", "save": False})
        result.add_result(FakeOutput())
        result.save(str(tmp_path), "base")
    finally:
        deactivate_context(token)

    assert _phases(events) == []


def test_a_task_step_names_its_command():
    events = []
    context = RunContext(on_event=events.append)
    from dw.events import activate_context, deactivate_context

    task_module._COMMAND_REGISTRY["phase_probe"] = lambda task, args, pipelines: "done"
    token = activate_context(context)
    try:
        task = Task({"command": "phase_probe", "arguments": {}}, "cpu")
        assert task.run({}) == "done"
    finally:
        deactivate_context(token)
        del task_module._COMMAND_REGISTRY["phase_probe"]

    assert _phases(events) == [("task", "phase_probe")]


class ChainedFakePipeline:
    """Records the segment label in force at each segment's call."""

    def __init__(self):
        self.output_dir = None
        self.file_prefix = None
        self.segment_label = None
        self.labels = []

    def _run_once(self, arguments):
        self.labels.append(self.segment_label)
        return SimpleNamespace(frames=[[Image.new("RGB", (8, 8)) for _ in range(4)]])


def test_a_chain_labels_each_segment_for_the_restarting_counter():
    from dw.pipeline_processors.chain import run_chain

    pipeline = ChainedFakePipeline()
    with patch("dw.pipeline_processors.chain.empty_device_cache"):
        run_chain(pipeline, {"segments": 3}, {"prompt": "p"})

    assert pipeline.labels == ["segment 1/3", "segment 2/3", "segment 3/3"]
    # Cleared at the end - the label belongs to the run that set it
    assert pipeline.segment_label is None


# ---------------------------------------------------------- cancel_pending
#
# A model load or a task step has no checkpoint of its own to catch the
# cancel flag - cancel() has to tell the client the request landed but will
# only take effect once that phase finishes on its own.


def test_cancel_during_loading_emits_cancel_pending():
    events = []
    context = RunContext(on_event=events.append)
    token = activate_context(context)
    try:
        emit_phase("loading", detail="acme/model")
        context.cancel()
    finally:
        deactivate_context(token)

    pending = [e for e in events if e["event"] == "cancel_pending"]
    assert len(pending) == 1
    assert pending[0]["phase"] == "loading"


def test_cancel_during_task_emits_cancel_pending():
    events = []
    context = RunContext(on_event=events.append)
    token = activate_context(context)
    try:
        emit_phase("task", detail="some_task")
        context.cancel()
    finally:
        deactivate_context(token)

    pending = [e for e in events if e["event"] == "cancel_pending"]
    assert len(pending) == 1
    assert pending[0]["phase"] == "task"


def test_cancel_during_an_interruptible_phase_emits_nothing_extra():
    """generating/decoding/saving/cached all have (or are near) a checkpoint
    of their own - cancel_pending would be noise there, since the run stops
    at the very next callback."""
    events = []
    context = RunContext(on_event=events.append)
    for phase in ("generating", "decoding", "saving", "cached"):
        token = activate_context(context)
        try:
            emit_phase(phase)
            context.cancel()
        finally:
            deactivate_context(token)
        # cancel() is idempotent to call again, but the Event only fires once
        context._cancel.clear()

    assert not [e for e in events if e["event"] == "cancel_pending"]


def test_cancel_pending_is_eventually_honored_at_the_next_checkpoint():
    """The pending state is not a dead end: once the non-interruptible phase
    ends and execution reaches a checkpoint, the run still stops."""
    events = []
    context = RunContext(on_event=events.append)
    token = activate_context(context)
    try:
        emit_phase("loading", detail="acme/model")
        context.cancel()
        assert any(e["event"] == "cancel_pending" for e in events)
        # The load "finishes" and the next checkpoint (e.g. step.py's
        # per-step check) is reached
        with pytest.raises(WorkflowCancelled):
            context.check_cancelled()
    finally:
        deactivate_context(token)
