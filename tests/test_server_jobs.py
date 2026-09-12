"""The job store's record of which run a job was: the run_start event, the
two columns, and reading the realized workflow back off disk.

The manager tests proper live in tests/test_server.py; this file is the run
record, which is a store concern rather than a routing one.
"""

import json
import time

import pytest

from dw.runs import REALIZED_FILE_NAME, new_run_id
from dw.server.jobs import TERMINAL_STATES, JobHistory, JobManager

from .test_server import ScriptedWorkerManager, valid_workflow

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
    job = manager.submit(workflow=valid_workflow(), base_dir=None)
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


def test_both_persist_and_read_back(manager):
    job = finished_job(manager)
    historical = manager.history.get(job.id)
    assert historical["run_id"] == RUN_ID
    assert historical["run_dir"] == RUN_DIR


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
    from dw.workflow import workflow_from_definition

    workflow = workflow_from_definition(
        command["workflow"],
        command["output_dir"],
        command["base_dir"],
        command.get("workflow_dir"),
    )
    # Mirrors dw/worker.py's _handle_execute: validated against the
    # defaults first, then run() substitutes and expands the real arguments.
    workflow.validate()
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
            workflow=_for_each_workflow(),
            base_dir=None,
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


def test_a_failed_job_reports_the_steps_that_completed(tmp_path):
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(failing_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    try:
        job = manager.submit(workflow=valid_workflow(), base_dir=None)
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
