"""#1758: validating a stored workflow with `arguments` omitted refuses a
`required` variable, as `arguments: {}` does and as a run would."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from tests.test_server import ScriptedWorkerManager, success_script

TEMPLATES = Path(__file__).resolve().parent.parent / "workflows" / "templates"

REQUIRED_TEMPLATES = [
    ("voice-clone", "reference_audio"),
    ("narration-long", "reference_audio"),
    ("narration-from-takes", "reference_audio"),
    ("narration-retime", None),
    ("minimax/narrated-take", None),
    ("minimax/h3-seam-pass", None),
]


@pytest.fixture
def client(tmp_path):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(workflows),
    )
    app = create_app(
        workflow_dir=str(workflows),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        examples_dirs=[str(TEMPLATES)],
    )
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def required_errors(answer):
    return sorted(
        (e["path"], e["message"])
        for e in answer["errors"]
        if "is required and has no default" in e["message"]
    )


@pytest.mark.parametrize("name,variable", REQUIRED_TEMPLATES)
def test_omitted_arguments_refuse_like_empty_ones(client, name, variable):
    body = {"workflow_path": name}
    omitted = client.post("/api/validate?sizes=false", json=body).json()
    empty = client.post(
        "/api/validate?sizes=false", json={**body, "arguments": {}}
    ).json()

    assert omitted["valid"] is False, omitted
    assert not omitted.get("plan")
    assert required_errors(omitted)
    assert required_errors(omitted) == required_errors(empty)
    if variable:
        assert required_errors(omitted)[0][0] == f"variables.{variable}"


def test_a_run_with_no_arguments_queues_nothing(client):
    response = client.post("/api/jobs", json={"workflow_path": "voice-clone"})

    assert response.status_code == 400, response.text
    assert "'reference_audio' is required" in response.text
    assert client.get("/api/jobs").json()["jobs"] == []


def test_an_inline_definition_with_no_arguments_is_a_document_check(client):
    import json

    definition = json.loads((TEMPLATES / "voice-clone.json").read_text())
    answer = client.post(
        "/api/validate?sizes=false", json={"workflow": definition}
    ).json()

    assert required_errors(answer) == []
