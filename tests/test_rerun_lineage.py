"""A rerun names the job it reran (`rerun_of`), live and from history."""

import sqlite3

from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.job_history import JobHistory
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, success_script, valid_workflow
from .test_server import wait_for_status


def make_client(tmp_path):
    (tmp_path / "workflows").mkdir(exist_ok=True)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
    )
    return TestClient(app, base_url="http://localhost")


def run_and_rerun(client):
    original = client.post(
        "/api/jobs", json={"workflow": valid_workflow("lineage"), "arguments": {}}
    ).json()["id"]
    wait_for_status(client, original, ["succeeded"])
    rerun = client.post(f"/api/jobs/{original}/rerun").json()["id"]
    wait_for_status(client, rerun, ["succeeded"])
    return original, rerun


def test_a_rerun_names_the_job_it_reran(tmp_path):
    with make_client(tmp_path) as client:
        original, rerun = run_and_rerun(client)
        assert client.get(f"/api/jobs/{rerun}").json()["rerun_of"] == original
        assert "rerun_of" not in client.get(f"/api/jobs/{original}").json()
        listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        assert listed[rerun]["rerun_of"] == original


def test_the_lineage_survives_a_restart(tmp_path):
    with make_client(tmp_path) as client:
        original, rerun = run_and_rerun(client)
    with make_client(tmp_path) as client:
        assert client.get(f"/api/jobs/{rerun}").json()["rerun_of"] == original
        listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        assert listed[rerun]["rerun_of"] == original


def test_an_older_history_database_gains_the_column(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
            " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
            " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
        )
    JobHistory(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert "rerun_of" in columns
