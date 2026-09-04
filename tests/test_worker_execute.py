"""Worker behavior added in Phase 0: cancel mid-run, inline workflows,
identity-based cache eviction, progress forwarding."""

import queue
import time
from unittest.mock import patch

from dw.events import WorkflowCancelled


def _make_worker():
    with patch("dw.worker.setup_logging"):
        from dw.worker import WorkflowWorker

        return WorkflowWorker(queue.Queue(), queue.Queue())


class StubWorkflow:
    name = "stub"

    def __init__(self, behavior=None):
        self.behavior = behavior
        self.manifest = [{"step": "s", "files": ["/out/a.png"]}]

    def validate(self):
        pass

    def run(self, arguments, previous_pipelines=None, context=None):
        if self.behavior == "wait_for_cancel":
            for _ in range(200):
                if context.cancelled:
                    raise WorkflowCancelled()
                time.sleep(0.05)
            raise AssertionError("cancel never arrived")
        if self.behavior == "emit":
            context.emit("step_start", step="s", index=0, total_steps=1)
        if context is not None:
            context.touch_pipeline("kept-key")
        return []


def _drain(result_queue):
    messages = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            return messages


def _execute(worker, workflow, command=None):
    command = command or {
        "workflow_path": "x.json",
        "arguments": {},
        "output_dir": "/tmp",
    }
    with patch("dw.worker.workflow_from_file", return_value=workflow):
        worker._handle_execute(command)
    return _drain(worker.result_queue)


def test_cancel_command_stops_a_running_workflow():
    worker = _make_worker()
    worker.command_queue.put({"type": "cancel"})
    messages = _execute(worker, StubWorkflow("wait_for_cancel"))
    types = [message["type"] for message in messages]
    assert "cancelled" in types
    assert "success" not in types


def test_success_carries_the_manifest_and_forwards_progress():
    worker = _make_worker()
    messages = _execute(worker, StubWorkflow("emit"))
    by_type = {message["type"]: message for message in messages}
    assert by_type["success"]["manifest"] == [{"step": "s", "files": ["/out/a.png"]}]
    progress = [m for m in messages if m["type"] == "progress"]
    assert progress and progress[0]["event"] == "step_start"


def test_workflow_switch_evicts_cache_and_untouched_keys_dropped():
    worker = _make_worker()
    _execute(worker, StubWorkflow())
    # a pipeline something loaded, plus one the run will not touch
    worker.loaded_pipelines["stale-key"] = object()

    # same identity: stale key evicted after the run, models otherwise kept
    _execute(worker, StubWorkflow())
    assert "stale-key" not in worker.loaded_pipelines

    # different identity: everything evicted before the run
    worker.loaded_pipelines["kept-key"] = object()
    with patch.object(worker, "_cleanup_all") as cleanup:
        _execute(
            worker,
            StubWorkflow(),
            command={
                "workflow_path": "other.json",
                "arguments": {},
                "output_dir": "/tmp",
            },
        )
    cleanup.assert_called_once()


def test_inline_workflow_definition_executes():
    worker = _make_worker()
    with patch(
        "dw.worker.workflow_from_definition",
        lambda data, out, base_dir=None, workflow_dir=None: StubWorkflow(),
    ):
        worker._handle_execute(
            {
                "workflow": {"id": "inline_test", "steps": []},
                "arguments": {},
                "output_dir": "/tmp/test_output",
            }
        )
    types = [message["type"] for message in _drain(worker.result_queue)]
    assert "success" in types
    assert worker.workflow_identity == ("inline", "inline_test")


def test_cancel_keeps_cached_models():
    """Cancelling a run must not cost the model cache - that is the whole
    point of cooperative cancellation."""
    worker = _make_worker()
    worker.loaded_pipelines["warm-model"] = object()
    worker.command_queue.put({"type": "cancel"})
    messages = _execute(worker, StubWorkflow("wait_for_cancel"))
    assert "cancelled" in [m["type"] for m in messages]
    assert "warm-model" in worker.loaded_pipelines


def test_shutdown_during_run_cancels_then_flags_shutdown():
    """A shutdown command arriving mid-run stops the workflow and leaves
    the worker set to exit its loop."""
    worker = _make_worker()
    worker.command_queue.put({"type": "shutdown"})
    messages = _execute(worker, StubWorkflow("wait_for_cancel"))
    assert "cancelled" in [m["type"] for m in messages]
    assert worker.pending_shutdown is True


class StubResult:
    saved_files = []


def test_full_cleanup_clears_step_cache():
    """A full cleanup ('memory clear') must drop cached step results, or a
    'clear' leaves results pinned in RAM."""
    from dw.step_cache import step_cache

    step_cache.put({"name": "cleanup_probe"}, 42, StubResult())

    worker = _make_worker()
    worker._cleanup_all()

    assert step_cache.get({"name": "cleanup_probe"}, 42, hits_this_run=set()) is None
