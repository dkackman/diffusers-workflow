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
