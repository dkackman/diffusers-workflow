"""Worker behavior added in Phase 0: cancel mid-run, inline workflows,
identity-based cache eviction, progress forwarding."""

import json
import os
import queue
import time
from unittest.mock import patch

import pytest

from dw import workflow_run
from dw.events import WorkflowCancelled
from dw.pipeline_ownership import PipelineOwnership


def _make_worker():
    with patch("dw.worker.setup_logging"):
        from dw.worker import WorkflowWorker

        return WorkflowWorker(queue.Queue(), queue.Queue())


class StubWorkflow:
    name = "stub"

    def __init__(self, behavior=None):
        self.behavior = behavior
        self.manifest = [{"step": "s", "files": ["/out/a.png"]}]

    def run(
        self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
    ):
        self.seen_prior_step_keys = dict(prior_step_keys or {})
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


def snapshot_command(file_spec="/w/x.json", source="path", **fields):
    """An execute command carrying an admitted snapshot, as _run_job sends."""
    return {
        "definition": {"id": "stub", "steps": []},
        "file_spec": file_spec,
        "source": source,
        "arguments": {},
        "output_dir": "/tmp",
        **fields,
    }


def _execute(worker, workflow, command=None):
    command = command or snapshot_command()
    with patch("dw.worker.workflow_from_snapshot", return_value=workflow):
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
            command=snapshot_command("/w/other.json"),
        )
    cleanup.assert_called_once()


def test_inline_workflow_definition_executes(tmp_path):
    worker = _make_worker()
    with patch("dw.worker.workflow_from_snapshot", return_value=StubWorkflow()):
        worker._handle_execute(
            snapshot_command(
                str(tmp_path / "__inline__.json"),
                source="inline",
                definition={"id": "inline_test", "steps": []},
                output_dir=str(tmp_path),
            )
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
    result_list = []


def test_full_cleanup_clears_step_cache():
    """A full cleanup ('memory clear') must drop cached step results, or a
    'clear' leaves results pinned in RAM."""
    from dw.step_cache import step_cache

    step_cache.put(
        "probe_workflow", {"name": "cleanup_probe"}, 42, StubResult(), "/out", True
    )

    worker = _make_worker()
    worker._cleanup_all()

    assert (
        step_cache.get(
            "probe_workflow", {"name": "cleanup_probe"}, 42, set(), "/out", True
        )
        is None
    )


def test_failure_carries_the_manifest_of_the_steps_that_ran():
    """T015: the steps before the failure wrote real files, and a report
    that omits them reads as 'this run produced nothing'."""

    class FailingWorkflow(StubWorkflow):
        def run(
            self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
        ):
            raise KeyError("Previous result 'first_renamed' not found")

    worker = _make_worker()
    messages = _execute(worker, FailingWorkflow())
    error = next(m for m in messages if m["type"] == "error")
    assert error["manifest"] == [{"step": "s", "files": ["/out/a.png"]}]


def test_cancellation_carries_the_manifest_too():
    worker = _make_worker()
    worker.command_queue.put({"type": "cancel"})
    messages = _execute(worker, StubWorkflow("wait_for_cancel"))
    cancelled = next(m for m in messages if m["type"] == "cancelled")
    assert cancelled["manifest"] == [{"step": "s", "files": ["/out/a.png"]}]


def test_a_failure_before_the_workflow_loads_reports_no_manifest():
    worker = _make_worker()
    with patch("dw.worker.workflow_from_snapshot", side_effect=OSError("no such file")):
        worker._handle_execute(snapshot_command())
    error = next(m for m in _drain(worker.result_queue) if m["type"] == "error")
    assert error["manifest"] == []


def test_a_failed_run_reclaims_memory_and_drops_untouched_pipelines():
    """#72: a run that fails leaks everything it had loaded.

    Success and cancellation both reclaim; failure did neither, so a
    half-loaded pipeline and any variant the failed attempt superseded stayed
    resident - and the next attempt loaded its own on top of them, which is
    how three retries starved a decode by 1.88 GiB.
    """

    class FailingWorkflow(StubWorkflow):
        def run(
            self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
        ):
            context.touch_pipeline("kept-key")
            raise RuntimeError("CUDA out of memory")

    worker = _make_worker()
    worker.loaded_pipelines["kept-key"] = object()
    worker.loaded_pipelines["dead-variant"] = object()

    with patch.object(worker, "_cleanup_between_runs") as cleanup:
        messages = _execute(worker, FailingWorkflow())

    assert "error" in [message["type"] for message in messages]
    cleanup.assert_called_once()
    # the variant this run never touched is dead weight - the one it did touch
    # is still the warm model a retry would reuse
    assert "dead-variant" not in worker.loaded_pipelines
    assert "kept-key" in worker.loaded_pipelines


def test_a_failed_run_drops_the_traceback_before_reclaiming():
    """The traceback is what pins a half-finished load in memory.

    Every frame between the handler and the failure is reachable through
    `__traceback__`, and those frames hold whatever the load had built when it
    raised - so a collection that runs while the exception is still live frees
    none of it. The report is a formatted string by then, so the traceback
    goes first and the reclaim afterwards has something to reclaim.
    """
    import gc
    import weakref

    class Weight:
        pass

    held = []
    alive_at_cleanup = []

    class FailingWorkflow(StubWorkflow):
        def run(
            self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
        ):
            half_loaded = Weight()
            held.append(weakref.ref(half_loaded))
            raise RuntimeError("load failed partway")

    worker = _make_worker()

    def record():
        gc.collect()
        alive_at_cleanup.append(held[0]() is not None)

    # the log record itself carries exc_info, and pytest's capture keeps every
    # record for the length of the test - which would pin the traceback here
    # no matter what the worker does with it
    with patch("dw.worker.logger"):
        with patch.object(worker, "_cleanup_between_runs", side_effect=record):
            _execute(worker, FailingWorkflow())

    assert alive_at_cleanup == [False]


def probe_answering(hits, seen=None):
    """A stand-in for `workflow_run.cache_hits`, recording what it was asked."""

    def cache_hits(workflow, arguments):
        if seen is not None:
            seen["workflow"] = workflow
            seen["arguments"] = arguments
        return list(hits)

    return cache_hits


def test_probe_cache_answers_with_the_workflows_hits():
    worker = _make_worker()
    workflow = StubWorkflow()
    seen = {}
    command = snapshot_command(
        type="probe_cache", request_id="p-1", arguments={"prompt": "p"}
    )
    with (
        patch("dw.worker.workflow_from_snapshot", return_value=workflow),
        patch.object(workflow_run, "cache_hits", probe_answering(["gen"], seen)),
    ):
        worker._handle_probe_cache(command)
    assert _drain(worker.result_queue) == [
        {"type": "probe_cache", "request_id": "p-1", "cached": ["gen"]}
    ]
    assert seen["workflow"] is workflow
    assert seen["arguments"] == {"prompt": "p"}


def test_probe_cache_reports_a_failure_as_unknown_not_as_a_crash():
    worker = _make_worker()
    with patch("dw.worker.workflow_from_snapshot", side_effect=ValueError("bad file")):
        worker._handle_probe_cache(snapshot_command(type="probe_cache"))
    [answer] = _drain(worker.result_queue)
    assert answer["type"] == "probe_cache"
    assert answer["request_id"] is None  # echoed even when the command had none
    assert answer["cached"] is None
    assert "bad file" in answer["error"]


def test_probe_cache_activates_the_jobs_asset_dir(tmp_path):
    worker = _make_worker()
    seen = {}

    def asset_aware_cache_hits(workflow, arguments):
        from dw.assets import get_asset_dir

        seen["asset_dir"] = get_asset_dir()
        return []

    with (
        patch("dw.worker.workflow_from_snapshot", return_value=StubWorkflow()),
        patch.object(workflow_run, "cache_hits", asset_aware_cache_hits),
    ):
        worker._handle_probe_cache(
            snapshot_command(type="probe_cache", asset_dir=str(tmp_path))
        )
    assert seen["asset_dir"] == str(tmp_path)


def test_prior_step_keys_carry_from_one_command_to_the_next():
    """A same-workflow rerun must know what its steps loaded last time.

    The release path that frees a superseded pipeline before its replacement
    loads is keyed on the step's previous cache key, and a command builds a
    fresh Workflow every time - so without the worker carrying the map, a
    rerun of the same workflow with different arguments reloads on top of the
    resident stack and is OOM-killed (#150).
    """
    worker = _make_worker()

    first = StubWorkflow()
    _execute(worker, first)
    assert first.seen_prior_step_keys == {}
    first_keys = {"gen": "key-a"}
    worker.prior_step_keys.update(first_keys)

    second = StubWorkflow()
    _execute(worker, second)
    assert second.seen_prior_step_keys == first_keys


def test_the_worker_records_what_each_run_loaded_even_when_it_fails():
    """A run that died partway still leaves its loads to be released."""
    worker = _make_worker()

    class Dying(StubWorkflow):
        def run(self, *args, **kwargs):
            self.pipeline_ownership = PipelineOwnership()
            self.pipeline_ownership.record("gen", "key-a")
            raise RuntimeError("boom")

    _execute(worker, Dying())
    assert worker.prior_step_keys == {"gen": "key-a"}


def test_a_workflow_switch_forgets_the_prior_keys():
    """Cleanup dropped the pipelines those keys addressed."""
    worker = _make_worker()
    _execute(worker, StubWorkflow())
    worker.prior_step_keys.update({"gen": "key-a"})
    _execute(
        worker,
        StubWorkflow(),
        command=snapshot_command("/w/other.json"),
    )
    assert worker.prior_step_keys == {}


def test_execute_runs_the_callers_arguments_not_the_default(tmp_path):
    """#415: a document-default 'text/html' content_type that the caller's
    own argument overrides to 'text/plain' must actually run, not just queue.

    Admission (dw.server.admission.admit, where #415's first bounce was
    fixed) checks the caller's arguments before the job is queued; the
    worker once re-validated with none and failed the job against the
    unsubstituted default. This drives a real Workflow (not StubWorkflow)
    through the actual worker path, the one StubWorkflow-based tests above
    cannot catch."""
    worker = _make_worker()
    definition = {
        "id": "se-415",
        "variables": {"ct": "text/html"},
        "steps": [
            {
                "name": "t",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": ["<b>x</b>"]},
                },
                "result": {"content_type": "variable:ct"},
            }
        ],
    }
    worker._handle_execute(
        snapshot_command(
            str(tmp_path / "__inline__.json"),
            source="inline",
            definition=definition,
            arguments={"ct": "text/plain"},
            output_dir=str(tmp_path),
        )
    )
    messages = _drain(worker.result_queue)
    types = [m["type"] for m in messages]
    assert "success" in types, messages
    success = next(m for m in messages if m["type"] == "success")
    files = success["manifest"][0]["files"]
    assert len(files) == 1
    assert (tmp_path / files[0]).read_text() == "<b>x</b>"


class AssetRecordingWorkflow(StubWorkflow):
    def run(self, arguments, *args, **kwargs):
        from dw.assets import get_asset_dir

        self.asset_dir_at_run = get_asset_dir()
        return super().run(arguments, *args, **kwargs)


def test_the_run_sees_the_jobs_asset_directory(tmp_path):
    """B8: steps like dissolve_videos and location policy resolve asset:
    references, so the run must see the job's own workspace asset
    directory - not whatever the default discovery would find - or a job
    against a non-default workspace reads the wrong library."""
    worker = _make_worker()
    workflow = AssetRecordingWorkflow()
    _execute(
        worker,
        workflow,
        snapshot_command(output_dir=str(tmp_path), asset_dir=str(tmp_path / "assets")),
    )
    assert workflow.asset_dir_at_run == str(tmp_path / "assets")


def test_between_run_cleanup_releases_host_caches_without_clearing_pipelines():
    """#368: a job's own cleanup left ~10GB resident that only clear_memory
    reclaimed - the pinned-host staging buffers of group_offload and the
    glibc arenas a released pipeline's weights were read into. Neither is
    touched by gc.collect()/empty_device_cache() alone, so the light,
    every-job cleanup must also call release_host_caches() - and must keep
    loaded_pipelines/shared_components warm while doing it, since those
    exist for exactly this (inter-run) cleanup to leave alone.
    """
    worker = _make_worker()
    worker.loaded_pipelines["warm-key"] = object()
    worker.shared_components["warm-component"] = object()

    with patch("dw.worker.release_host_caches", return_value=512.0) as released:
        worker._cleanup_between_runs()

    released.assert_called_once()
    # the whole point: still-warm state for the next run survives this call
    assert "warm-key" in worker.loaded_pipelines
    assert "warm-component" in worker.shared_components


def _text_workflow(text):
    return {
        "id": "edited",
        "steps": [
            {
                "name": "t",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": [text]},
                },
                "result": {"content_type": "text/plain"},
            }
        ],
    }


@pytest.mark.parametrize("change", ["edited", "deleted"])
def test_an_edited_file_runs_as_admitted(tmp_path, change):
    """Review Focus 2c/2: the job carries the definition admission checked,
    and the worker runs that - not whatever the file says by the time the
    job reaches the front of the queue. Edited, the file is still valid and
    its run would write different text, so a read of it shows in the
    output; deleted, any read of it fails the run."""
    from dw.workflow import workflow_from_file

    root = tmp_path / "workflows"
    root.mkdir()
    path = root / "edited.json"
    path.write_text(json.dumps(_text_workflow("admitted")))
    output_dir = tmp_path / "outputs"
    admitted = workflow_from_file(str(path), str(output_dir), str(root))
    if change == "edited":
        path.write_text(json.dumps(_text_workflow("edited after admission")))
    else:
        os.remove(path)

    worker = _make_worker()
    worker._handle_execute(
        {
            "definition": admitted.workflow_definition,
            "file_spec": admitted.file_spec,
            "source": "path",
            "workflow_dir": str(root),
            "arguments": {},
            "output_dir": str(output_dir),
        }
    )

    messages = _drain(worker.result_queue)
    success = next((m for m in messages if m["type"] == "success"), None)
    assert success is not None, messages
    [written] = success["manifest"][0]["files"]
    assert (output_dir / written).read_text() == "admitted"
    assert worker.workflow_identity == ("path", admitted.file_spec)


@pytest.mark.parametrize("command_type", ["memory_status", "clear_memory"])
def test_a_request_whose_handler_fails_is_answered_with_its_request_id(command_type):
    """The command loop's error reply echoes the failed request's id, so
    WorkerManager.request returns it at once - without the echo it would be
    discarded as someone else's reply and the caller would wait out its
    timeout (clear_memory's is 30s) instead of failing now."""
    worker = _make_worker()

    def broken_reading():
        raise RuntimeError("no reading")

    worker._get_memory_info = broken_reading
    worker.command_queue.put({"type": command_type, "request_id": "r-7"})
    worker.command_queue.put({"type": "shutdown"})
    worker.run()

    [answer] = _drain(worker.result_queue)
    assert answer["type"] == "error"
    assert answer["request_id"] == "r-7"
    assert "no reading" in answer["message"]


def test_a_failure_names_an_asset_by_reference_not_by_server_path(tmp_path):
    """GHSA-fwg5-jfjg-fxpf: a task failing on an 'asset:' input reported the
    path it had resolved to - the server's home directory and workspace
    layout - to every API and MCP caller."""
    assets = tmp_path / "ws" / "assets"
    outputs = tmp_path / "ws" / "outputs"
    source = assets / "refine" / "src.mp4"

    class FailingWorkflow(StubWorkflow):
        def run(
            self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
        ):
            raise ValueError(
                f"{source} carries no audio track; wrote {outputs / 'id' / 'a.png'}"
            )

    worker = _make_worker()
    messages = _execute(
        worker,
        FailingWorkflow(),
        command=snapshot_command(asset_dir=str(assets), output_dir=str(outputs)),
    )
    error = next(m for m in messages if m["type"] == "error")
    assert "asset:refine/src.mp4 carries no audio track" in error["message"]
    assert "output:id/a.png" in error["message"]
    assert str(tmp_path) not in error["message"]
    assert str(tmp_path) not in error["traceback"]


def test_a_warning_names_an_asset_by_reference_not_by_server_path(tmp_path):
    """GHSA-cr8g-q9j9-j68g (and -m7jg-h8hm-m45q, -j924-mr8f-x2x4): a task
    warning naming the 'asset:' input it resampled printed the path it had
    resolved to - in the message and as a key of `sample_rates`. Free-text
    events are rewritten as the failure is; an event's `files` reach the
    server absolute, which relativises them itself."""
    assets = tmp_path / "ws" / "assets"
    outputs = tmp_path / "ws" / "outputs"
    source = str(assets / "cast" / "cut.mp4")
    written = str(outputs / "id" / "a.wav")

    class WarningWorkflow(StubWorkflow):
        def run(
            self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
        ):
            context.emit(
                "warning",
                message=f"mix_audio: tracks differ ({source}: 48000 Hz)",
                sample_rates={source: 48000, "track 2": 16000},
            )
            context.emit("log", message=f"wrote {written}")
            context.emit("step_end", step="mix", files=[written])

    worker = _make_worker()
    messages = _execute(
        worker,
        WarningWorkflow(),
        command=snapshot_command(asset_dir=str(assets), output_dir=str(outputs)),
    )
    events = [m for m in messages if m["type"] == "progress"]
    warning = next(e for e in events if e["event"] == "warning")
    assert (
        warning["message"] == "mix_audio: tracks differ (asset:cast/cut.mp4: 48000 Hz)"
    )
    assert warning["sample_rates"] == {"asset:cast/cut.mp4": 48000, "track 2": 16000}
    log = next(e for e in events if e["event"] == "log")
    assert log["message"] == "wrote output:id/a.wav"
    step_end = next(e for e in events if e["event"] == "step_end")
    assert step_end["files"] == [written]
