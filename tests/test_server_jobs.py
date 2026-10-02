"""The job store's record of which run a job was: the run_start event, the
two columns, and reading the realized workflow back off disk.

The manager tests proper live in tests/test_server.py; this file is the run
record, which is a store concern rather than a routing one.
"""

import json
import logging
import os
import threading
import time

import pytest

from dw.runs import REALIZED_FILE_NAME, new_run_id
from dw.server.jobs import JobManager
from dw.server.job_history import JobHistory
from dw.server.job_record import TERMINAL_STATES

from .test_server import (  # noqa: F401 - `server` is a fixture
    DyingWorkerManager,
    ScriptedWorkerManager,
    admitted_for,
    server,
    success_script,
    valid_workflow,
    wait_for_status,
)

RUN_ID = new_run_id({"workflow": "spec"})
RUN_DIR = f"server_test/{RUN_ID}"


def tracked_script(command):
    """A worker that reports its run before doing anything else - what
    Workflow.run emits once the run directory is chosen."""
    yield {
        "type": "progress",
        "event": "run_start",
        "run_id": RUN_ID,
        "identity": "server_test",
        "run_dir": RUN_DIR,
        "version": 4,
    }
    yield {"type": "success", "message": "ok", "run_count": 1, "manifest": []}


@pytest.fixture
def manager(tmp_path):
    made = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(tracked_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    yield made
    made.shutdown()


def finished_job(manager):
    job = manager.submit(
        admitted=admitted_for(manager, valid_workflow()), workflow=valid_workflow()
    )
    deadline = time.time() + 5
    while job.status not in TERMINAL_STATES and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "succeeded", job.error
    return job


def test_run_start_populates_the_job(manager):
    job = finished_job(manager)
    assert job.run_id == RUN_ID
    assert job.run_dir == RUN_DIR
    assert job.summary()["run_id"] == RUN_ID
    assert job.detail()["run_dir"] == RUN_DIR
    # the ordinal the gallery shows for this run's files
    assert job.run_version == 4
    assert job.summary()["run_version"] == 4


def test_both_persist_and_read_back(manager):
    job = finished_job(manager)
    historical = manager.history.get(job.id)
    assert historical["run_id"] == RUN_ID
    assert historical["run_dir"] == RUN_DIR
    assert historical["run_version"] == 4
    # and in the polled list, not only the detail
    (summary,) = manager.history.recent_summaries()
    assert summary["run_version"] == 4


def test_realized_reads_the_file_the_run_wrote(manager, tmp_path):
    job = finished_job(manager)
    run_dir = tmp_path / "outputs" / "server_test" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / REALIZED_FILE_NAME).write_text(json.dumps({"id": "realized"}))

    assert manager.realized(job.id) == {"id": "realized"}


def test_realized_is_none_without_the_file(manager):
    job = finished_job(manager)
    assert manager.realized(job.id) is None


def test_realized_is_none_for_a_pre_tracking_row(manager):
    """A job recorded before run tracking has no run_dir, and the manager
    does not guess one from file paths."""
    job = finished_job(manager)
    with manager.history._connect() as connection:
        connection.execute(
            "UPDATE jobs SET run_id = NULL, run_dir = NULL WHERE id = ?", (job.id,)
        )
    manager.jobs.pop(job.id)

    assert manager.realized(job.id) is None


def test_realized_refuses_a_run_dir_that_escapes_the_output_root(manager, tmp_path):
    job = finished_job(manager)
    job.run_dir = "../../etc"
    assert manager.realized(job.id) is None


def test_a_database_without_the_columns_is_migrated(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.sqlite")
    # A store written before run tracking: the ALTER is the whole migration
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
            " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
            " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
        )
        connection.execute(
            "INSERT INTO jobs (id, status) VALUES ('old-1', 'succeeded')"
        )

    history = JobHistory(path)
    row = history.get("old-1")
    assert row["run_id"] is None and row["run_dir"] is None


def _for_each_workflow(**overrides):
    """Copied from tests/test_workflow.py's helper of the same name (Task 5) -
    not imported across test modules, per that task's convention."""
    definition = {
        "id": "fe",
        "variables": {
            "shots": [{"name": "a", "text": "A"}, {"name": "b", "text": "B"}]
        },
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": ["item:text"]},
                },
                "result": {"content_type": "text/plain"},
            },
            {
                "name": "edit",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": "gather:shot"},
                },
                "result": {"content_type": "text/plain"},
            },
        ],
    }
    definition.update(overrides)
    return definition


def for_each_script(command):
    """Runs the real Workflow.run() in-process - the actual for_each
    expansion and compose_text execution, not a canned response - standing
    in for the spawned worker process the way this file's other scripts do
    (ScriptedWorkerManager replaces the process, not the workflow code)."""
    from dw.workflow import workflow_from_snapshot

    # Mirrors dw/worker.py's _handle_execute: the admitted snapshot, run
    # with the caller's own arguments, which run() substitutes and expands
    workflow = workflow_from_snapshot(
        command["definition"],
        command["output_dir"],
        command["file_spec"],
        command.get("workflow_dir"),
    )
    workflow.run(command["arguments"], {})
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": workflow.manifest,
    }


def test_a_for_each_job_expands_and_runs_through_the_server_job_path(tmp_path):
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(for_each_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        job = manager.submit(
            admitted=admitted_for(manager, _for_each_workflow()),
            workflow=_for_each_workflow(),
            arguments={
                "shots": [
                    {"name": "one", "text": "1"},
                    {"name": "two", "text": "2"},
                    {"name": "three", "text": "3"},
                ]
            },
        )
        deadline = time.time() + 5
        while job.status not in TERMINAL_STATES and time.time() < deadline:
            time.sleep(0.01)
        assert job.status == "succeeded", job.error
        assert [entry["step"] for entry in job.manifest] == [
            "shot@one",
            "shot@two",
            "shot@three",
            "edit",
        ]

        edit_files = job.manifest[-1]["files"]
        assert len(edit_files) == 1
        edit_path = tmp_path / "outputs" / edit_files[0]
        # compose_text's default separator is a blank line
        assert edit_path.read_text() == "1\n\n2\n\n3"
    finally:
        manager.shutdown()


def failing_script(command):
    """A run that writes a file, then dies on the next step - the shape of a
    dialogue-short whose last step names a renamed one (T015)."""
    output_dir = command["output_dir"]
    yield {
        "type": "progress",
        "event": "step_end",
        "step": "first",
        "files": [f"{output_dir}/QaDanglingRef/run/first.txt"],
    }
    yield {
        "type": "error",
        "message": "Workflow execution error: Previous result 'x' not found",
        "traceback": "...",
        "manifest": [
            {"step": "first", "files": [f"{output_dir}/QaDanglingRef/run/first.txt"]}
        ],
    }


def workflow_end_script(command):
    """A run that emits a workflow_end progress event carrying its own
    nested manifest, the way Workflow.run's real emit does (#284)."""
    output_dir = command["output_dir"]
    files = [f"{output_dir}/QaWorkflowEnd/run/film.mp4"]
    yield {
        "type": "progress",
        "event": "step_end",
        "step": "film",
        "files": files,
    }
    yield {
        "type": "progress",
        "event": "workflow_end",
        "workflow": "QaWorkflowEnd",
        "manifest": [{"step": "film", "files": files}],
    }
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": [{"step": "film", "files": files}],
    }


def test_workflow_end_event_manifest_matches_get_job_manifest(tmp_path):
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(workflow_end_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        job = manager.submit(
            admitted=admitted_for(manager, valid_workflow()), workflow=valid_workflow()
        )
        deadline = time.time() + 5
        while job.status not in TERMINAL_STATES and time.time() < deadline:
            time.sleep(0.01)
        assert job.status == "succeeded", job.error

        workflow_end_events = [
            e for e in job.events if e.get("event") == "workflow_end"
        ]
        assert len(workflow_end_events) == 1
        assert workflow_end_events[0]["manifest"] == job.manifest
        assert workflow_end_events[0]["manifest"] == [
            {"step": "film", "files": ["QaWorkflowEnd/run/film.mp4"]}
        ]
    finally:
        manager.shutdown()


def test_a_failed_job_reports_the_steps_that_completed(tmp_path):
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(failing_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        job = manager.submit(
            admitted=admitted_for(manager, valid_workflow()), workflow=valid_workflow()
        )
        deadline = time.time() + 5
        while job.status not in TERMINAL_STATES and time.time() < deadline:
            time.sleep(0.01)
        assert job.status == "failed"
        # named the way clients address outputs, exactly as a success is
        assert job.manifest == [
            {"step": "first", "files": ["QaDanglingRef/run/first.txt"]}
        ]
        assert manager.history.get(job.id)["manifest"] == job.manifest
    finally:
        manager.shutdown()


def _wait_terminal(job):
    deadline = time.time() + 5
    while job.status not in TERMINAL_STATES and time.time() < deadline:
        time.sleep(0.01)
    return job


def _unknown_type_warnings(caplog):
    return [r for r in caplog.records if "Unknown worker message" in r.getMessage()]


def test_a_late_probe_reply_cannot_poison_the_next_request(tmp_path, caplog):
    """Review Focus 3. A probe that gave up still answers eventually, onto
    the queue the next request reads. The next memory_status must discard
    it and answer live, and the job after that must find no stray reply."""
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        manager.worker_manager.ensure_worker()
        manager.worker_manager._results.put(
            {"type": "probe_cache", "request_id": "gave-up", "cached": ["stale"]}
        )
        with caplog.at_level(logging.DEBUG, logger="dw"):
            status = manager.memory_status()
            assert status["live"] is True
            assert status["reason"] is None
            assert status["info"] == {"gpu_available": True}

            job = _wait_terminal(
                manager.submit(
                    admitted=admitted_for(manager, valid_workflow()),
                    workflow=valid_workflow(),
                )
            )
        assert job.status == "succeeded", job.error
        assert _unknown_type_warnings(caplog) == []
    finally:
        manager.shutdown()


def stray_reply_script(command):
    """A run whose reply stream carries a request reply nobody waited for -
    a memory_status answer whose reader timed out before it landed."""
    yield {"type": "memory_status", "request_id": "gave-up", "info": {}}
    yield {"type": "success", "message": "ok", "run_count": 1, "manifest": []}


def test_a_stray_request_reply_during_a_run_is_discarded_quietly(tmp_path, caplog):
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(stray_reply_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        with caplog.at_level(logging.DEBUG, logger="dw"):
            job = _wait_terminal(
                manager.submit(
                    admitted=admitted_for(manager, valid_workflow()),
                    workflow=valid_workflow(),
                )
            )
        assert job.status == "succeeded", job.error
        assert _unknown_type_warnings(caplog) == []
        assert any(
            r.levelno == logging.DEBUG and "memory_status" in r.getMessage()
            for r in caplog.records
        )
    finally:
        manager.shutdown()


def unknown_reply_script(command):
    yield {"type": "no_such_reply"}
    yield {"type": "success", "message": "ok", "run_count": 1, "manifest": []}


def test_an_unknown_reply_type_is_still_a_warning(tmp_path, caplog):
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(unknown_reply_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        with caplog.at_level(logging.DEBUG, logger="dw"):
            job = _wait_terminal(
                manager.submit(
                    admitted=admitted_for(manager, valid_workflow()),
                    workflow=valid_workflow(),
                )
            )
        assert job.status == "succeeded", job.error
        [warning] = _unknown_type_warnings(caplog)
        assert warning.levelno == logging.WARNING
        assert "no_such_reply" in warning.getMessage()
    finally:
        manager.shutdown()


class TestAWorkerThatDiesMidRequest:
    """Review Focus 4: the reply dispatcher changes how a reply is read, not
    what a dead worker means - the crash is still marked, and each request
    answers the way it did."""

    @pytest.fixture
    def dying(self, tmp_path):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=DyingWorkerManager(),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=str(tmp_path),
        )
        manager.worker_manager.worker_active = True
        manager.last_memory = {"gpu_available": True, "used": 42}
        yield manager
        manager.shutdown()

    def test_memory_status_marks_the_crash_and_answers_unreachable(self, dying):
        status = dying.memory_status()
        assert status["live"] is False
        assert status["reason"] == "worker_unreachable"
        assert status["info"] == {"gpu_available": True, "used": 42}
        assert dying.worker_manager.crashed is True

    def test_a_probe_is_unknown(self, dying):
        probe = {
            "definition": {"id": "x", "steps": []},
            "file_spec": "/w/x.json",
            "source": "path",
            "arguments": {},
            "output_dir": "/tmp",
        }
        assert dying.probe_cache(probe) is None

    def test_clear_memory_raises_for_the_route_to_answer_503(self, dying):
        with pytest.raises(RuntimeError, match="died"):
            dying.clear_memory()


def submit_path_job(manager, tmp_path):
    path = tmp_path / "Basic.json"
    path.write_text(json.dumps(valid_workflow("basic"), indent=2))
    job = manager.submit(
        admitted=admitted_for(manager, workflow_path=str(path)),
        workflow_path=str(path),
    )
    deadline = time.time() + 5
    while job.status not in TERMINAL_STATES and time.time() < deadline:
        time.sleep(0.01)
    return job, path


def test_definition_of_a_live_path_job_equals_its_file(manager, tmp_path):
    """rerun and get_job_workflow consume definition(), so answering from the
    snapshot must give exactly what json.load of the file gives."""
    job, path = submit_path_job(manager, tmp_path)
    assert manager.definition(job.id) == json.loads(path.read_text())


def test_definition_of_a_live_path_job_survives_its_file_going(manager, tmp_path):
    job, path = submit_path_job(manager, tmp_path)
    expected = json.loads(path.read_text())
    path.unlink()
    assert manager.definition(job.id) == expected


def test_definition_of_a_live_job_is_a_copy(manager, tmp_path):
    job, _ = submit_path_job(manager, tmp_path)
    manager.definition(job.id)["id"] = "mutated"
    assert manager.definition(job.id)["id"] == "basic"


def test_definition_of_a_restored_path_job_rereads_its_file(manager, tmp_path):
    """Only a restored job has no snapshot, so only it depends on the file."""
    job, path = submit_path_job(manager, tmp_path)
    # The in-memory status turns terminal before the history row is
    # written; a restore reads only the row
    deadline = time.time() + 5
    while manager.history.get(job.id) is None and time.time() < deadline:
        time.sleep(0.01)
    restored = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(tracked_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        assert restored.definition(job.id) == json.loads(path.read_text())
        path.unlink()
        assert restored.definition(job.id) is None
    finally:
        restored.shutdown()


class _HistoricalRow:
    """A finished job straight into history, with the run fields a test sets."""

    workflow_name = "w"
    catalog_name = None
    status = "succeeded"
    created_at = 1.0
    started_at = 1.0
    finished_at = 2.0
    manifest = []
    warnings = []
    error = None
    events = []
    run_id = None
    acknowledged = "none"

    def __init__(self, job_id, run_dir, output_dir=None):
        self.id = job_id
        self.run_dir = run_dir
        self.spec = {"workspace": "default"}
        if output_dir:
            self.spec["output_dir"] = output_dir


class TestDeleteAJobsRun:
    """DELETE /api/jobs/{id}/run: the job record names its run directory and
    the root it ran against, so the server deletes the run without the
    caller deriving either."""

    def test_deleting_a_jobs_run_removes_its_run_directory(self, server, tmp_path):
        run = tmp_path / "outputs" / RUN_DIR
        run.mkdir(parents=True)
        (run / "manifest.json").write_text("{}")
        with server(tracked_script) as client:
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], TERMINAL_STATES)
            answer = client.delete(f"/api/jobs/{job['id']}/run")
            assert answer.status_code == 200, answer.text
            body = answer.json()
            assert body["job_id"] == job["id"]
            assert body["run_dir"] == RUN_DIR
            assert body["deleted"] is True
            assert not run.exists()

    def test_a_job_without_a_run_dir_is_404(self, server):
        with server(success_script) as client:
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], TERMINAL_STATES)
            answer = client.delete(f"/api/jobs/{job['id']}/run")
            assert answer.status_code == 404
            assert "no run directory" in answer.json()["detail"]

    def test_an_unknown_job_is_404(self, server):
        with server(success_script) as client:
            assert client.delete("/api/jobs/nope/run").status_code == 404

    def test_a_running_job_is_409(self, server, tmp_path):
        release = threading.Event()

        def held_script(command):
            yield {
                "type": "progress",
                "event": "run_start",
                "run_id": RUN_ID,
                "identity": "server_test",
                "run_dir": RUN_DIR,
                "version": 1,
            }
            release.wait(5)
            yield {"type": "success", "message": "ok", "run_count": 1, "manifest": []}

        (tmp_path / "outputs" / RUN_DIR).mkdir(parents=True)
        with server(held_script) as client:
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], {"running"})
            try:
                answer = client.delete(f"/api/jobs/{job['id']}/run")
                assert answer.status_code == 409
                assert (tmp_path / "outputs" / RUN_DIR).is_dir()
            finally:
                release.set()

    def test_a_run_dir_escaping_the_root_is_refused(self, server, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        JobHistory(str(tmp_path / "jobs.sqlite")).record(
            _HistoricalRow("escape", "../outside")
        )
        with server(success_script) as client:
            answer = client.delete("/api/jobs/escape/run")
            assert answer.status_code in (400, 404)
            assert outside.is_dir()


def test_run_location_is_the_jobs_own_root(manager, tmp_path):
    """A job in a named workspace wrote under that workspace's outputs."""
    other = tmp_path / "ws" / "outputs"
    other.mkdir(parents=True)
    manager.history.record(_HistoricalRow("named", RUN_DIR, output_dir=str(other)))
    root, run_dir = manager.run_location("named")
    assert root == os.path.realpath(str(other))
    assert run_dir == RUN_DIR
