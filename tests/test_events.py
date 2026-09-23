"""Phase 0 plumbing: progress events, cancellation, manifests, cache identity."""

import json
import logging
import os
import pathlib
import threading
import time
import pytest
from unittest.mock import patch

from PIL import Image

from dw import events as events_module
from dw.events import RunContext, WorkflowCancelled, get_context, current_context
from dw.log_setup import setup_logging
from dw.result import Result
from dw.runs import is_run_id
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
    assert names[0] == "run_start"
    # the run's ordinal, so a job can name its run the way the gallery will
    # (the output root here is shared, so only its shape is fixed)
    assert isinstance(events[0]["version"], int) and events[0]["version"] >= 1
    assert names[1] == "workflow_start"
    assert names[-1] == "workflow_end"
    assert "step_start" in names and "step_end" in names
    assert "iteration_start" in names
    assert names.count("pipeline_step") == 3

    start = events[1]
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
    # The manifest names the file by the absolute path it was written to,
    # which the default layout puts in this run's own directory under the
    # workflow's identity - '<output>/test/<run id>/'
    assert saved.endswith(".png") and os.path.exists(saved)
    run_dir = os.path.dirname(saved)
    assert os.path.dirname(run_dir) == str(tmp_path / "test")
    assert is_run_id(os.path.basename(run_dir))
    # and the run records itself beside what it made
    manifest = json.loads((pathlib.Path(run_dir) / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["workflow"]["identity"] == "test"
    assert manifest["steps"][0]["files"] == [os.path.basename(saved)]


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


def _fast_watchdog(threshold=0.05, interval=0.02):
    """Patches the watchdog's timing constants down to something a test can
    wait out in real time, without touching the production defaults."""
    return patch.multiple(
        events_module,
        PHASE_STALL_THRESHOLD_SECONDS=threshold,
        PHASE_STALL_CHECK_INTERVAL_SECONDS=interval,
    )


# Production checks six times per threshold (30s / 5s). A test that asserts
# something does *not* happen within a window keeps that ratio, so the margin
# between the window and the threshold is several check intervals wide rather
# than a scheduler hiccup wide.
_RATIO_THRESHOLD = 0.6
_RATIO_INTERVAL = 0.1


def _stalls(events):
    return [e for e in events if e.get("kind") == "phase_stall"]


def _wait_for_stall(events, deadline_seconds=5.0):
    """Poll until the watchdog has reported at least one stall."""
    deadline = time.monotonic() + deadline_seconds
    while not _stalls(events):
        assert time.monotonic() < deadline, "watchdog never reported a stall"
        time.sleep(0.005)


def test_watchdog_fires_after_threshold_with_no_events():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("generating")
            time.sleep(0.2)
        finally:
            context.exit_run()

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert stalls, "watchdog must report a stall once the phase runs past the threshold"
    assert stalls[0]["phase"] == "generating"
    assert stalls[0]["seconds_since_phase_start"] > 0


def test_watchdog_does_not_fire_before_threshold():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog(_RATIO_THRESHOLD, _RATIO_INTERVAL):
        context.enter_run()
        try:
            context.note_phase("generating")
            # Two check intervals, so the watchdog has really looked (a
            # watchdog that ignored the threshold would fire here), and well
            # short of the threshold
            time.sleep(2.5 * _RATIO_INTERVAL)
        finally:
            context.exit_run()

    assert not _stalls(events)


def test_watchdog_repeats_while_the_stall_continues():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("generating")
            time.sleep(0.3)
        finally:
            context.exit_run()

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert len(stalls) >= 2, "a continuing stall must be reported more than once"
    # seconds_since_phase_start increases across repeats - the message embeds
    # it, so a message-string-deduplicating consumer (job.warnings) sees more
    # than one distinct text rather than silently dropping every repeat
    assert (
        stalls[-1]["seconds_since_phase_start"] > stalls[0]["seconds_since_phase_start"]
    )
    assert len({s["message"] for s in stalls}) > 1


def test_watchdog_stops_once_a_new_event_arrives():
    # The stall report bumps the silence clock itself, so it repeats one
    # threshold after the last report. A progress event part-way through that
    # wait must restart the clock: the window below runs past when the repeat
    # would have come without the reset, and ends well before one threshold
    # after the progress event.
    threshold = _RATIO_THRESHOLD
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog(threshold, _RATIO_INTERVAL):
        context.enter_run()
        try:
            context.note_phase("generating")
            _wait_for_stall(events)
            stalls_before_progress = len(_stalls(events))
            time.sleep(0.6 * threshold)
            context.emit("pipeline_step", step=1)
            time.sleep(0.65 * threshold)
            stalls_after_progress = _stalls(events)[stalls_before_progress:]
        finally:
            context.exit_run()

    assert stalls_after_progress == [], (
        "a fresh event must reset the silence clock, not just a fresh phase"
    )


def test_watchdog_is_shared_and_not_double_started_across_sub_workflow_runs():
    context = RunContext(on_event=lambda e: None)
    context.enter_run()
    thread_after_outer = context._watchdog_thread
    assert thread_after_outer is not None and thread_after_outer.is_alive()

    context.enter_run()  # nested sub-workflow call, same shared context
    assert context._watchdog_thread is thread_after_outer, (
        "a nested run must not start a second watchdog thread"
    )

    context.exit_run()  # nested call returns - watchdog must stay up
    assert context._watchdog_thread is thread_after_outer
    assert context._watchdog_thread.is_alive()

    context.exit_run()  # outermost call returns - watchdog must stop
    assert context._watchdog_thread is None
    assert not thread_after_outer.is_alive()


def test_watchdog_thread_is_cleanly_stopped_at_run_end():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        context.note_phase("generating")
        context.exit_run()

    live_watchdog_threads = [
        t for t in threading.enumerate() if t.name == "dw-phase-stall"
    ]
    assert not live_watchdog_threads, "no watchdog thread must survive exit_run()"


def test_watchdog_event_carries_the_required_fields():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("saving")
            time.sleep(0.2)
        finally:
            context.exit_run()

    stall = next(e for e in events if e.get("kind") == "phase_stall")
    assert stall["event"] == "warning"
    assert stall["phase"] == "saving"
    assert isinstance(stall["seconds_since_phase_start"], (int, float))
    assert isinstance(stall["seconds_since_last_progress"], (int, float))
    assert "message" in stall


def test_watchdog_reports_last_progress_kind_and_does_not_reset_on_repeat():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("generating")
            context.emit("pipeline_step", step=1)
            time.sleep(0.3)
        finally:
            context.exit_run()

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert len(stalls) >= 2
    assert "pipeline_step" in stalls[0]["message"]
    # seconds_since_last_progress climbs across repeats rather than
    # resetting each time the watchdog itself emits (#357)
    assert (
        stalls[-1]["seconds_since_last_progress"]
        > stalls[0]["seconds_since_last_progress"]
    )


def test_each_run_records_its_own_version(tmp_path):
    """Consecutive runs of one workflow number themselves 1, 2, 3 - the
    ordinal the gallery shows as 'v2' and an agent quotes."""

    def mock_load(self, shared_components):
        self.pipeline = FakePipeline()

    versions = []
    for _ in range(3):
        workflow_def = _workflow_def()
        workflow_def["steps"][0]["result"] = {"content_type": "image/png"}
        workflow = Workflow(workflow_def, str(tmp_path), "test.json")
        with patch.object(Pipeline, "load", mock_load):
            with patch("dw.workflow.empty_device_cache"):
                workflow.run({}, previous_pipelines={})
        # The run's own directory, not the one its files came from: a
        # cached step reports the earlier run's files while still being a
        # run of its own with its own number
        run_dir = pathlib.Path(workflow._run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        versions.append(manifest["version"])

    assert versions == [1, 2, 3]


def test_the_version_is_on_disk_before_the_first_step_runs(tmp_path):
    """A run killed mid-step - which is how a stuck server gets restarted -
    never reaches the closing manifest, so the number has to land when the
    run opens. Also what lets a second process opening a run of the same
    workflow see this one's number rather than taking it too."""
    seen = {}

    def mock_load(self, shared_components):
        manifest_path = pathlib.Path(workflow._run_dir) / "manifest.json"
        seen.update(json.loads(manifest_path.read_text()))
        self.pipeline = FakePipeline()

    workflow_def = _workflow_def()
    workflow_def["steps"][0]["result"] = {"content_type": "image/png"}
    workflow = Workflow(workflow_def, str(tmp_path), "test.json")
    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            workflow.run({}, previous_pipelines={})

    assert seen["version"] == 1
    assert seen["status"] == "running"
    assert seen["finished_at"] is None
    closing = json.loads(
        (pathlib.Path(workflow._run_dir) / "manifest.json").read_text()
    )
    assert closing["status"] == "completed"
    assert closing["version"] == 1
    assert closing["finished_at"] is not None
