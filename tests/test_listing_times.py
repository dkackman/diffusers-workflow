"""`times=true` on the three library listings (#1599): each entry says when
it last changed (`mtime`) and, where the filesystem records one, when it was
first written (`created`). Without the opt-in a listing is what it was, so
the MCP tools, which never ask, answer exactly as before."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import Workspace, file_times
from dw_mcp import catalog, prompts, workspaces

from .test_server import ScriptedWorkerManager, success_script, valid_workflow

ROUTES = ("/api/workflows", "/api/prompts", "/api/workspaces")


@pytest.fixture
def client(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    manager = JobManager(
        root.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
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
    with TestClient(app, base_url="http://localhost") as client:
        saved = client.put("/api/workflows/shot", json={"workflow": valid_workflow()})
        assert saved.status_code == 200
        saved = client.put("/api/prompts/fox", json={"prompt": {"text": "a fox"}})
        assert saved.status_code == 200
        assert client.post("/api/workspaces", json={"name": "takes"}).status_code == 201
        yield client


def _entries(route, body):
    if route == "/api/workspaces":
        return body["workspaces"]
    return list(body["details"].values())


def _no_times(value):
    text = json.dumps(value)
    return '"mtime"' not in text and '"created"' not in text


class _Mcp:
    """What dw_mcp's client does with a GET, answered by the test server."""

    workspace = "default"
    mounted = False

    def __init__(self, http):
        self.http = http
        self.asked = []

    def get_json(self, path, params=None, workspace=None, headers=None):
        self.asked.append(params or {})
        response = self.http.get(path, params=params)
        response.raise_for_status()
        return response.json()


@pytest.mark.parametrize("route", ROUTES)
def test_times_are_opt_in(client, route):
    plain = client.get(route)
    assert plain.status_code == 200
    assert _entries(route, plain.json())
    assert _no_times(plain.json())
    # false is the default spelled out: the same bytes
    assert client.get(route, params={"times": "false"}).content == plain.content


def test_times_are_opt_in_for_the_mcp_listings(client):
    mcp = _Mcp(client)
    answers = [
        catalog.list_workflows(mcp),
        catalog.list_workflows(mcp, shape="image"),
        prompts.list_prompts(mcp),
        workspaces.list_workspaces(mcp),
        workspaces.list_workspaces(mcp, detail=True),
    ]
    assert all(_no_times(answer) for answer in answers)
    assert all("times" not in params for params in mcp.asked)


@pytest.mark.parametrize("route", ROUTES)
def test_each_entry_says_when_it_changed_when_asked(client, route):
    entries = _entries(route, client.get(route, params={"times": "true"}).json())
    assert entries
    for entry in entries:
        assert isinstance(entry["mtime"], (int, float)) and entry["mtime"] > 0
        # a birth time where the filesystem has one, else no key: never mtime
        # dressed up as one
        if "created" in entry:
            assert isinstance(entry["created"], (int, float))


def test_a_workflow_s_times_are_its_file_s(client, tmp_path):
    path = os.path.join(tmp_path, "studio", "workflows", "shot.json")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    detail = client.get("/api/workflows", params={"times": "true"}).json()["details"]
    assert detail["shot"]["mtime"] == 1_700_000_000
    assert ("created" in detail["shot"]) == ("created" in file_times(path))


def test_a_prompt_s_time_is_its_file_s(client, tmp_path):
    path = os.path.join(tmp_path, "studio", "prompts", "fox.json")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    detail = client.get("/api/prompts", params={"times": "true"}).json()["details"]
    assert detail["fox"]["mtime"] == 1_700_000_000


def test_a_workspace_changes_when_a_run_lands_in_it(client, tmp_path):
    takes = os.path.join(tmp_path, "studio", "takes")
    for folder, _dirs, files in os.walk(takes):
        for name in [*_dirs, *files, "."]:
            os.utime(os.path.join(folder, name), (1_600_000_000, 1_600_000_000))

    def changed():
        listing = client.get("/api/workspaces", params={"times": "true"}).json()
        return next(w for w in listing["workspaces"] if w["name"] == "takes")["mtime"]

    assert changed() == 1_600_000_000
    # a new run folder inside outputs/<workflow>/ moves the workspace's time,
    # though the workspace folder itself is untouched
    run = os.path.join(takes, "outputs", "shot")
    os.makedirs(run)
    os.utime(os.path.join(takes, "outputs"), (1_600_000_000, 1_600_000_000))
    os.utime(run, (1_700_000_000, 1_700_000_000))
    assert changed() == 1_700_000_000


def test_file_times_has_no_created_without_a_birth_time(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    path.write_text("{}")
    real = os.stat

    class NoBirth:
        def __init__(self, result):
            self.st_mtime = result.st_mtime

    monkeypatch.setattr(os, "stat", lambda p, *a, **k: NoBirth(real(p, *a, **k)))
    assert file_times(str(path)) == {"mtime": real(path).st_mtime}
