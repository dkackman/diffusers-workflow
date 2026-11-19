# ruff: noqa: F811
"""last_job_use (#1445): GET /api/models serves it, a delete forgets it, and a
job stamps the repos its admission found when its run starts."""

import time

import pytest

from dw import model_use
from dw.library import library_path
from dw.server.admission import admit
from dw.server.jobs import JobManager
from dw.workspace import Workspace

from .test_server import (  # noqa: F401 - `server` is a fixture
    ScriptedWorkerManager,
    admitted_for,
    hanging_script,
    server,
    success_script,
    valid_workflow,
    wait_for_status,
)
from .test_server_jobs import tracked_script


def scan(*repos):
    return lambda *args, **kwargs: {
        "cache_dir": "/hub",
        "size_on_disk": 2,
        "warnings": [],
        "disk_free": None,
        "disk_total": None,
        "repos": [
            {
                "repo_id": repo,
                "repo_type": "model",
                "size_on_disk": 1,
                "nb_files": 1,
                "last_accessed": 111.0 + index,
                "last_modified": 1.0,
                "revisions": [],
            }
            for index, repo in enumerate(repos)
        ],
    }


class TestModelsRoute:
    def test_each_row_carries_last_job_use(self, server, monkeypatch):
        monkeypatch.setattr(
            "dw.server.routes.system.scan_models", scan("org/used", "org/unused")
        )
        model_use.record(["org/used"], 1234.5)
        with server(success_script) as client:
            rows = {
                row["repo_id"]: row for row in client.get("/api/models").json()["repos"]
            }
        assert rows["org/used"]["last_job_use"] == 1234.5
        assert "last_job_use" in rows["org/unused"]
        assert rows["org/unused"]["last_job_use"] is None
        assert rows["org/used"]["last_accessed"] == 111.0
        assert rows["org/unused"]["last_accessed"] == 112.0

    def test_delete_forgets_the_record(self, server, monkeypatch):
        monkeypatch.setattr("dw.server.routes.system.delete_model", lambda repo: 42)
        model_use.record(["org/a", "org/b"], 5.0)
        with server(success_script) as client:
            response = client.delete("/api/models", params={"repo": "org/a"})
        assert response.status_code == 200
        assert model_use.read() == {"org/b": 5.0}

    def test_a_failed_delete_keeps_the_record(self, server, monkeypatch):
        def refuse(repo):
            raise ValueError("not cached")

        monkeypatch.setattr("dw.server.routes.system.delete_model", refuse)
        model_use.record(["org/a"], 5.0)
        with server(success_script) as client:
            response = client.delete("/api/models", params={"repo": "org/a"})
        assert response.status_code == 404
        assert model_use.read() == {"org/a": 5.0}


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


def wait_done(job, timeout=5):
    deadline = time.time() + timeout
    while job.status not in ("succeeded", "failed", "cancelled"):
        assert time.time() < deadline, job.status
        time.sleep(0.01)


class TestJobStamps:
    def test_submit_sets_repos_named(self, tmp_path):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(hanging_script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=str(tmp_path),
        )
        try:
            job = manager.submit(
                admitted=admitted_for(manager, valid_workflow()),
                workflow=valid_workflow(),
                repos_named=["org/a", "org/b"],
            )
            assert job.repos_named == ["org/a", "org/b"]
            manager.cancel(job.id)
        finally:
            manager.shutdown()

    def test_on_run_start_stamps_the_repos(self, manager):
        job = manager.submit(
            admitted=admitted_for(manager, valid_workflow()),
            workflow=valid_workflow(),
            repos_named=["org/a", "org/b"],
        )
        wait_done(job)
        stamps = model_use.read()
        assert set(stamps) == {"org/a", "org/b"}
        for stamp in stamps.values():
            assert abs(stamp - time.time()) < 30

    def test_a_job_naming_nothing_stamps_nothing(self, manager):
        wait_done(
            manager.submit(
                admitted=admitted_for(manager, valid_workflow()),
                workflow=valid_workflow(),
            )
        )
        assert model_use.read() == {}

    def test_a_job_cancelled_while_queued_stamps_nothing(self, tmp_path):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(hanging_script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=str(tmp_path),
        )
        try:
            first = manager.submit(
                admitted=admitted_for(manager, valid_workflow()),
                workflow=valid_workflow(),
            )
            deadline = time.time() + 5
            while first.status != "running" and time.time() < deadline:
                time.sleep(0.01)
            second = manager.submit(
                admitted=admitted_for(manager, valid_workflow()),
                workflow=valid_workflow(),
                repos_named=["org/queued"],
            )
            assert second.status == "queued"
            assert manager.cancel(second.id) == "cancelled"
            manager.cancel(first.id)
            assert "org/queued" not in model_use.read()
        finally:
            manager.shutdown()

    def test_a_rerun_forwards_repos_named(self, manager):
        job = manager.submit(
            admitted=admitted_for(manager, valid_workflow()),
            workflow=valid_workflow(),
        )
        wait_done(job)
        rerun = manager.rerun(
            job.id,
            admitted=admitted_for(manager, valid_workflow()),
            repos_named=["org/again"],
        )
        assert rerun.repos_named == ["org/again"]
        wait_done(rerun)
        assert "org/again" in model_use.read()


def quantized_workflow():
    return {
        "id": "quantized",
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {
                        "component_type": "{Fake}",
                        "no_generator": True,
                        "components": {},
                    },
                    "from_pretrained_arguments": {"model_name": "org/base-repo"},
                    "transformer": {
                        "configuration": {"component_type": "{FakeModel}"},
                        "quantization_config": {
                            "configuration": {"config_type": "sdnq.SDNQConfig"},
                            "arguments": {"weights_dtype": "int8"},
                        },
                        "from_pretrained_arguments": {
                            "model_name": "org/source-repo",
                            "subfolder": "transformer",
                        },
                    },
                    "arguments": {"prompt": "p"},
                },
            }
        ],
    }


def test_a_quantized_component_names_its_source_repo_and_stamps_it(manager, tmp_path):
    """A quantized-cache hit loads from dw-quantized, but the job still names
    the source repo, so the stamp (and last_job_use) covers it."""
    root = Workspace(tmp_path / "studio", "flag").ensure()
    admission = admit(
        workflow_path=None,
        workflow=quantized_workflow(),
        arguments={},
        base_dir=None,
        workspace=root,
        ceiling_index={},
        output_dir=root.outputs,
        workflow_dir=root.workflows,
        asset_library=library_path("assets", root),
        prompt_library=library_path("prompts", root),
    )
    assert admission.ok, admission.errors
    assert "org/source-repo" in admission.repos_named
    job = manager.submit(
        admitted=admission.workflow,
        workflow=quantized_workflow(),
        repos_named=admission.repos_named,
    )
    wait_done(job)
    assert "org/source-repo" in model_use.read()
