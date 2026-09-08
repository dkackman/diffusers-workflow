"""Exporting one finished job: a directory that stands on its own, and the
same tree as a zip."""

import io
import json
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from dw.runs import REALIZED_FILE_NAME, new_run_id
from dw.server.app import create_app
from dw.server.jobs import JobManager, TERMINAL_STATES
from dw.workspace import Workspace

from .test_server import (
    ScriptedWorkerManager,
    hanging_script,
    valid_workflow,
    wait_for_status,
)

RUN_ID = new_run_id({"workflow": "export"})
RUN_DIR = f"server_test/{RUN_ID}"


def exporting_script(command):
    """A run that reports its directory and writes one file."""
    output_dir = command["output_dir"]
    run_dir = os.path.join(output_dir, "server_test", RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "still.png"), "wb") as file:
        file.write(b"an image")
    with open(os.path.join(run_dir, REALIZED_FILE_NAME), "w") as file:
        json.dump(
            {
                "id": "server_test",
                "seed": 7,
                "steps": [
                    {
                        "name": "gen",
                        "pipeline": {
                            "configuration": {"component_type": "{Fake}"},
                            "from_pretrained_arguments": {"model_name": "m"},
                            "arguments": {"image": "asset:iris.png"},
                        },
                    }
                ],
            },
            file,
        )
    with open(os.path.join(run_dir, "manifest.json"), "w") as file:
        json.dump(
            {
                "run_id": RUN_ID,
                "status": "completed",
                "seed": 7,
                "steps": [{"step": "gen", "files": ["still.png"]}],
            },
            file,
        )
    yield {
        "type": "progress",
        "event": "run_start",
        "run_id": RUN_ID,
        "identity": "server_test",
        "run_dir": RUN_DIR,
    }
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": [{"step": "gen", "files": [os.path.join(run_dir, "still.png")]}],
    }


PRIOR_RUN_ID = new_run_id({"workflow": "prior"})
PRIOR_REFERENCE = f"output:prior/{PRIOR_RUN_ID}/prior.png"


def untracked_script(command):
    """A run from before run tracking: no run_start event, so the job never
    learns a run directory and its own row is the only manifest there is."""
    output_dir = command["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "legacy.png")
    with open(path, "wb") as file:
        file.write(b"an older image")
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": [{"step": "gen", "files": [path]}],
    }


def chained_script(command):
    """A run whose realized workflow names an earlier run's file."""
    output_dir = command["output_dir"]
    prior_dir = os.path.join(output_dir, "prior", PRIOR_RUN_ID)
    os.makedirs(prior_dir, exist_ok=True)
    with open(os.path.join(prior_dir, "prior.png"), "wb") as file:
        file.write(b"the first stage")

    run_dir = os.path.join(output_dir, "server_test", RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "still.png"), "wb") as file:
        file.write(b"an image")
    with open(os.path.join(run_dir, REALIZED_FILE_NAME), "w") as file:
        json.dump(
            {
                "id": "server_test",
                "seed": 7,
                "steps": [
                    {
                        "name": "gen",
                        "pipeline": {
                            "configuration": {"component_type": "{Fake}"},
                            "from_pretrained_arguments": {"model_name": "m"},
                            "arguments": {"image": PRIOR_REFERENCE},
                        },
                    }
                ],
            },
            file,
        )
    with open(os.path.join(run_dir, "manifest.json"), "w") as file:
        json.dump(
            {
                "run_id": RUN_ID,
                "status": "completed",
                "seed": 7,
                # The same file twice: a chain's step names the one before
                # it, and the export holds one copy of it
                "steps": [
                    {"step": "gen", "files": ["still.png"]},
                    {"step": "post", "files": ["still.png"]},
                ],
            },
            file,
        )
    yield {
        "type": "progress",
        "event": "run_start",
        "run_id": RUN_ID,
        "identity": "server_test",
        "run_dir": RUN_DIR,
    }
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": [{"step": "gen", "files": [os.path.join(run_dir, "still.png")]}],
    }


@pytest.fixture
def workspace_root(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.assets, "iris.png"), "wb") as file:
        file.write(b"an iris")
    return root


@pytest.fixture
def server(workspace_root, tmp_path):
    def make(script=exporting_script):
        manager = JobManager(
            workspace_root.outputs,
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=workspace_root.workflows,
        )
        # The test's handle on the manager the app is talking to, so a test
        # can evict a finished job and make the server answer from history
        make.manager = manager
        app = create_app(
            workflow_dir=workspace_root.workflows,
            output_dir=workspace_root.outputs,
            job_manager=manager,
            prompt_dir=workspace_root.prompts,
            asset_dir=workspace_root.assets,
            workspace=workspace_root.root,
        )
        return TestClient(app, base_url="http://localhost")

    return make


def finished(client):
    submitted = client.post(
        "/api/jobs", json={"workflow": valid_workflow(), "arguments": {}}
    ).json()
    wait_for_status(client, submitted["id"], TERMINAL_STATES)
    return submitted["id"]


class TestExportDirectory:
    def test_it_gathers_the_whole_run(self, server, workspace_root):
        with server() as client:
            job_id = finished(client)
            response = client.post(f"/api/jobs/{job_id}/export")

        assert response.status_code == 201
        body = response.json()
        directory = body["directory"]
        assert directory == os.path.join(workspace_root.root, "exports", job_id)
        for name in ("README.md", "workflow.json", "manifest.json", "job.json"):
            assert os.path.isfile(os.path.join(directory, name))
        assert os.path.isfile(os.path.join(directory, "assets", "iris.png"))
        assert os.path.isfile(os.path.join(directory, "outputs", "still.png"))
        assert body["total_bytes"] > 0
        assert body["missing"] == []
        assert body["zip_url"] == f"/exports/{job_id}.zip"

    def test_the_workflow_is_the_realized_one(self, server):
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        recorded = json.loads(open(os.path.join(body["directory"], "job.json")).read())
        assert recorded["realized"] is True
        assert "traceback" not in recorded and "event_count" not in recorded
        assert body["workflow"]["seed"] == 7

    def test_a_historical_job_records_the_same_keys(self, server):
        with server() as client:
            live_id = finished(client)
            live = client.post(f"/api/jobs/{live_id}/export").json()["job"]

            job_id = finished(client)
            # Evicted from memory: the server has to answer from the
            # history store, whose dict is a different shape
            server.manager.jobs.clear()
            historical = client.post(f"/api/jobs/{job_id}/export").json()["job"]

        assert sorted(historical) == sorted(live)
        assert "spec" not in historical and "historical" not in historical
        assert historical["id"] == job_id
        assert historical["run_dir"] == RUN_DIR

    def test_a_job_with_no_run_directory_gets_a_synthesized_manifest(self, server):
        with server(untracked_script) as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        directory = body["directory"]
        assert body["manifest"]["synthesized"] is True
        assert body["manifest"]["steps"] == [{"step": "gen", "files": ["legacy.png"]}]
        # The job predates run tracking, so the workflow is the definition
        # as submitted rather than a realized one - and the file the row
        # names still lands, resolved against the output root itself
        assert (
            json.loads(open(os.path.join(directory, "job.json")).read())["realized"]
            is False
        )
        assert os.path.isfile(os.path.join(directory, "outputs", "legacy.png"))
        assert body["missing"] == []

    def test_an_output_reference_is_copied_under_the_run_it_names(self, server):
        with server(chained_script) as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        copied = os.path.join("inputs", "prior", PRIOR_RUN_ID, "prior.png")
        assert os.path.isfile(os.path.join(body["directory"], copied))
        paths = [entry["path"] for entry in body["files"]]
        assert copied.replace(os.sep, "/") in paths
        # Listed by two steps, copied and counted once
        assert paths.count("outputs/still.png") == 1
        assert body["missing"] == []

    def test_the_readme_names_the_job_and_says_how_to_run_it(self, server):
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        readme = open(os.path.join(body["directory"], "README.md")).read()
        assert job_id in readme
        assert "python -m dw.run workflow.json" in readme
        assert "Git LFS" in readme

    def test_an_unresolvable_asset_is_reported_missing(self, server, workspace_root):
        os.unlink(os.path.join(workspace_root.assets, "iris.png"))
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        assert body["missing"] == ["asset:iris.png"]

    def test_an_unknown_job_is_404(self, server):
        with server() as client:
            assert client.post("/api/jobs/nope/export").status_code == 404

    def test_a_live_job_is_409(self, server):
        with server(hanging_script) as client:
            submitted = client.post(
                "/api/jobs", json={"workflow": valid_workflow(), "arguments": {}}
            ).json()
            wait_for_status(client, submitted["id"], ("running",))
            response = client.post(f"/api/jobs/{submitted['id']}/export")
            client.post(f"/api/jobs/{submitted['id']}/cancel")

        assert response.status_code == 409

    def test_a_second_export_without_overwrite_is_409(self, server):
        with server() as client:
            job_id = finished(client)
            assert client.post(f"/api/jobs/{job_id}/export").status_code == 201
            again = client.post(f"/api/jobs/{job_id}/export")
            assert again.status_code == 409
            forced = client.post(f"/api/jobs/{job_id}/export?overwrite=true")
            assert forced.status_code == 201


class TestExportZip:
    def test_it_lists_the_same_entries_as_the_directory(self, server):
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()
            response = client.get(f"/exports/{job_id}.zip")

        assert response.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        assert sorted(archive.namelist()) == sorted(
            f"{job_id}/{entry['path']}" for entry in body["files"]
        )

    def test_no_export_is_404(self, server):
        with server() as client:
            job_id = finished(client)
            assert client.get(f"/exports/{job_id}.zip").status_code == 404


class TestReservedName:
    def test_exports_cannot_name_a_workspace(self, server):
        with server() as client:
            response = client.post("/api/workspaces", json={"name": "exports"})
        assert response.status_code == 400
        assert "cannot name a workspace" in response.json()["detail"]

    def test_an_exports_folder_is_not_listed_as_a_workspace(
        self, server, workspace_root
    ):
        with server() as client:
            job_id = finished(client)
            client.post(f"/api/jobs/{job_id}/export")
            names = [
                w["name"] for w in client.get("/api/workspaces").json()["workspaces"]
            ]
        assert names == ["default"]
