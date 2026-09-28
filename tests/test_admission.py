"""One admission service for validate, submit and rerun.

A request used to be loaded and checked in several places - the validate
route, `_candidate_for`, `JobManager.submit` - each building its own
Workflow, expanding the definition again, activating the workspace's asset
library around some of its checks and not others, and computing a different
subset of the warnings. `dw.server.admission.admit` loads it once, checks it
once, with the library active for all of it.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw import assets
from dw.server.admission import admit
from dw.server.app import create_app
from dw.server.jobs import TERMINAL_STATES, JobManager
from dw.workflow import Workflow
from dw.workspace import Workspace

from .test_server import (
    ScriptedWorkerManager,
    success_script,
    valid_workflow,
    wait_for_status,
)


def for_each_workflow():
    """A task-only workflow with a for_each: expansion is where the work
    goes, and a default entry carries a key no step reads, so a warning
    derived from the expansion is there to report."""
    return {
        "id": "fe",
        "variables": {
            "shots": [
                {"name": "a", "text": "A", "mood": "calm"},
                {"name": "b", "text": "B", "mood": "calm"},
            ]
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


def image_workflow():
    workflow = valid_workflow("imaged")
    workflow["variables"]["image"] = "asset:present.png"
    return workflow


@pytest.fixture
def root(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.assets, "present.png"), "wb") as file:
        file.write(b"png")
    return root


@pytest.fixture
def server(root, tmp_path):
    def make(script=success_script):
        manager = JobManager(
            root.outputs,
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=root.workflows,
        )
        app = create_app(
            workflow_dir=root.workflows,
            output_dir=root.outputs,
            job_manager=manager,
            prompt_dir=root.prompts,
            asset_dir=root.assets,
            workspace=root.root,
        )
        return TestClient(app, base_url="http://localhost")

    return make


@pytest.fixture
def folds(monkeypatch):
    """Every call of Workflow._fold, the stage that expansion starts with."""
    calls = []
    original = Workflow._fold

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Workflow, "_fold", counting)
    return calls


def test_a_submit_expands_once(server, folds):
    with server() as client:
        response = client.post(
            "/api/jobs",
            json={
                "workflow": for_each_workflow(),
                "arguments": {"shots": [{"name": "x", "text": "X"}]},
            },
        )
        assert response.status_code == 201, response.text
        assert len(folds) == 1


def test_a_validate_expands_once(server, folds):
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false",
            json={
                "workflow": for_each_workflow(),
                "arguments": {"shots": [{"name": "x", "text": "X"}]},
            },
        ).json()
        assert answer["valid"], answer
        assert answer["plan"] is not None
        assert len(folds) == 1


def test_a_validate_without_arguments_expands_once(server, folds):
    """Omitted arguments check the document (validation_errors is asked
    with None) while the warnings and the plan are asked with {} - the same
    fold, which the expansion memo has to key as one."""
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false", json={"workflow": for_each_workflow()}
        ).json()
        assert answer["valid"], answer
        assert answer["plan"] is not None
        assert len(folds) == 1


def test_every_admission_check_sees_the_workspace_asset_library(
    server, root, monkeypatch
):
    seen = []
    original = Workflow.shot_span_warnings

    def spy(self, *args, **kwargs):
        seen.append(assets._active_asset_dir.get())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Workflow, "shot_span_warnings", spy)
    shots_assets = os.path.join(root.root, "shots", "assets")
    with server() as client:
        client.post("/api/workspaces", json={"name": "shots"})
        body = {"workflow": valid_workflow(), "workspace": "shots"}

        assert client.post("/api/validate?sizes=false", json=body).json()["valid"]
        assert seen == [shots_assets]

        assert client.post("/api/jobs", json=body).status_code == 201
        assert seen == [shots_assets, shots_assets]


def test_bad_arguments_are_refused_before_the_queue_and_validate_still_expands(
    server, folds
):
    body = {"workflow": for_each_workflow(), "arguments": {"prmopt": "a cat"}}
    with server() as client:
        manager = client.app.state.job_manager

        response = client.post("/api/jobs", json=body)
        assert response.status_code == 400
        assert "arguments.prmopt" in response.json()["detail"]
        assert manager.worker_manager.commands == []

        folds.clear()
        answer = client.post("/api/validate?sizes=false", json=body).json()
        assert answer["valid"] is False
        assert answer["errors"][0]["path"] == "arguments.prmopt"
        # the expansion ran over the defaults, whose entries carry a key no
        # step reads - so its warning is there beside the error
        assert any("mood" in warning for warning in answer["warnings"])
        assert len(folds) == 1


def test_a_rerun_rechecks_its_references(server, root):
    with open(os.path.join(root.assets, "gone.png"), "wb") as file:
        file.write(b"png")
    with server() as client:
        submitted = client.post(
            "/api/jobs",
            json={
                "workflow": image_workflow(),
                "arguments": {"image": "asset:gone.png"},
            },
        )
        assert submitted.status_code == 201, submitted.text
        job_id = submitted.json()["id"]
        wait_for_status(client, job_id, TERMINAL_STATES)
        os.remove(os.path.join(root.assets, "gone.png"))
        executed = [
            command
            for command in client.app.state.job_manager.worker_manager.commands
            if command["type"] == "execute"
        ]

        rerun = client.post(f"/api/jobs/{job_id}/rerun")

        assert rerun.status_code == 400
        assert "arguments.image" in rerun.json()["detail"]
        assert "gone.png" in rerun.json()["detail"]
        assert [
            command
            for command in client.app.state.job_manager.worker_manager.commands
            if command["type"] == "execute"
        ] == executed


def test_a_submitted_job_records_the_warnings_validate_reports(server):
    with server() as client:
        validated = client.post(
            "/api/validate?sizes=false", json={"workflow": valid_workflow()}
        ).json()
        unseeded = [w for w in validated["warnings"] if "sets no 'seed'" in w]
        assert unseeded

        submitted = client.post("/api/jobs", json={"workflow": valid_workflow()})

        assert submitted.status_code == 201
        assert unseeded[0] in submitted.json()["warnings"]


def test_admit_answers_a_schema_failure_with_no_warnings(root):
    broken = {"id": "broken", "steps": [{"name": "s"}]}

    admission = admit(
        workflow_path=None,
        workflow=broken,
        arguments={},
        base_dir=None,
        workspace=root,
        ceiling_index={},
        output_dir=root.outputs,
        workflow_dir=root.workflows,
        asset_roots=[root.assets],
        prompt_roots=[root.prompts],
    )

    assert not admission.ok
    assert admission.schema_errors
    assert admission.warnings == []
    assert json.dumps(admission.errors)


def test_admission_checks_content_type_against_the_callers_arguments(root):
    """A document-default 'text/html' content_type that the caller's own
    argument overrides to 'text/plain' must be admitted - JobManager.submit
    used to validate the unsubstituted document (loaded.validate(), no
    arguments), refusing a run that validate_workflow had already accepted
    for the same call (#415, the run_workflow mirror of #414). Moved from
    tests/test_server_jobs.py when admission left submit."""
    definition = {
        "id": "se-415",
        "variables": {"ct": "text/html"},
        "steps": [
            {
                "name": "t",
                "task": {"command": "compose_text", "arguments": {"parts": ["x"]}},
                "result": {"content_type": "variable:ct"},
            }
        ],
    }

    def admitted(arguments):
        return admit(
            workflow_path=None,
            workflow=definition,
            arguments=arguments,
            base_dir=None,
            workspace=root,
            ceiling_index={},
            output_dir=root.outputs,
            workflow_dir=root.workflows,
            asset_roots=[root.assets],
            prompt_roots=[root.prompts],
        )

    assert admitted({"ct": "text/plain"}).ok
    assert not admitted({}).ok
