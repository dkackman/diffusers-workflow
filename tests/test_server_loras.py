"""The LoRA catalog over HTTP: the server's own library at <root>/loras,
shared by every workspace, with an examples tree's loras/ behind it."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.server.routes import loras as lora_routes
from dw.workspace import Workspace, create_workspace

from .test_server import ScriptedWorkerManager, success_script

QWEN = "Qwen/Qwen-Image-2.1"


def entry(**overrides):
    body = {
        "model_name": "prithivMLmods/Qwen-Image-2.1-Voxel-Style",
        "base_models": [QWEN],
        "description": "Blocky voxel look",
        "use_when": "voxel or blocky 3D requests",
        "status": "trial",
    }
    body.update(overrides)
    return body


def write(root, name, body):
    path = os.path.join(root, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(body, file)


@pytest.fixture
def server(tmp_path, monkeypatch):
    for variable in ("DW_PROMPT_PATH", "DW_ASSET_PATH", "DW_WORKFLOW_PATH", "DW_PROMPT_DIR", "DW_ASSET_DIR"):
        monkeypatch.delenv(variable, raising=False)
    checkout = tmp_path / "repo"
    (checkout / "workflows" / "models").mkdir(parents=True)
    write(str(checkout / "workflows"), "models/qwen", {
        "steps": [{"name": "s", "pipeline": {"from_pretrained_arguments": {"model_name": QWEN}}}]
    })
    write(str(checkout / "workflows"), "models/h3-ref", {
        "steps": [{"name": "s", "pipeline": {"from_pretrained_arguments": {"model_name": "MiniMaxAI/MiniMax-H3", "workflow": "ref2va"}}}]
    })
    write(str(checkout / "loras"), "qwen-image/voxel", entry())
    write(str(checkout / "loras"), "minimax-h3/realism", entry(
        model_name="fal/R", base_models=["MiniMaxAI/MiniMax-H3"], workflow="t2va",
        status="proven", evidence=[{"note": "best"}]))
    write(str(checkout / "loras"), "broken", {"model_name": "not valid"})
    workspace = Workspace(tmp_path / "studio", "flag").ensure()
    manager = JobManager(
        workspace.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=workspace.workflows,
    )
    app = create_app(
        workflow_dir=workspace.workflows,
        output_dir=workspace.outputs,
        job_manager=manager,
        prompt_dir=workspace.prompts,
        asset_dir=workspace.assets,
        examples_dirs=[str(checkout / "workflows")],
        workspace=workspace.root,
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.workspace = workspace
        client.checkout = checkout
        yield client


class TestListing:
    def test_lists_valid_entries_and_skips_an_invalid_file(self, server):
        body = server.get("/api/loras").json()
        assert [row["name"] for row in body["loras"]] == ["minimax-h3/realism", "qwen-image/voxel"]
        assert body["loras"][0]["origin"] == "examples"
        assert body["loras"][0]["writable"] is False

    def test_a_repo_id_filters_exactly(self, server):
        body = server.get("/api/loras", params={"model": QWEN}).json()
        assert [row["name"] for row in body["loras"]] == ["qwen-image/voxel"]
        assert body["resolved"] == [{"repo": QWEN, "workflow": None}]
        assert server.get("/api/loras", params={"model": "Qwen/Qwen-Image-2.1-2509"}).json()["loras"] == []

    def test_a_workflow_name_resolves_to_its_bases(self, server):
        body = server.get("/api/loras", params={"model": "models/qwen"}).json()
        assert [row["name"] for row in body["loras"]] == ["qwen-image/voxel"]

    def test_a_reference_workflow_does_not_list_a_t2va_entry(self, server):
        body = server.get("/api/loras", params={"model": "models/h3-ref"}).json()
        assert body["loras"] == []
        assert body["resolved"] == [{"repo": "MiniMaxAI/MiniMax-H3", "workflow": "ref2va"}]

    def test_the_workflow_parameter_narrows_a_repo(self, server):
        params = {"model": "MiniMaxAI/MiniMax-H3", "workflow": "ref2va"}
        assert server.get("/api/loras", params=params).json()["loras"] == []

    def test_status_and_tag_filter(self, server):
        assert [r["name"] for r in server.get("/api/loras", params={"status": "proven"}).json()["loras"]] == ["minimax-h3/realism"]

    def test_neither_a_workflow_nor_a_repo_is_a_400(self, server):
        response = server.get("/api/loras", params={"model": "no-such-thing"})
        assert response.status_code == 400
        assert "no-such-thing" in response.json()["detail"]


class TestWrites:
    def test_save_lands_in_the_roots_library_and_shadows_the_shipped_one(self, server):
        response = server.put("/api/loras/qwen-image/voxel", json={"entry": entry(scale={"default": 0.8})})
        assert response.status_code == 200
        saved = os.path.join(server.workspace.root, "loras", "qwen-image", "voxel.json")
        assert os.path.isfile(saved)
        got = server.get("/api/loras/qwen-image/voxel")
        assert got.json()["scale"] == {"default": 0.8}
        assert got.headers["X-Lora-Origin"] == "workspace"

    def test_an_invalid_entry_is_a_400(self, server):
        response = server.put("/api/loras/x", json={"entry": entry(status="proven")})
        assert response.status_code == 400

    def test_a_traversing_name_is_a_400(self, server):
        # the client may normalise the '..' away, which lands on no route at all
        assert server.put("/api/loras/../x", json={"entry": entry()}).status_code in (400, 404, 405)

    def test_the_name_recommend_is_refused(self, server):
        response = server.put("/api/loras/recommend", json={"entry": entry()})
        assert response.status_code == 400

    def test_a_named_workspace_sees_the_same_library(self, server):
        server.put("/api/loras/mine", json={"entry": entry()})
        create_workspace(server.workspace, "other")
        names = [r["name"] for r in server.get("/api/loras", params={"workspace": "other"}).json()["loras"]]
        assert "mine" in names

    def test_deleting_a_shipped_entry_is_a_403(self, server):
        assert server.delete("/api/loras/qwen-image/voxel").status_code == 403

    def test_deleting_an_own_entry(self, server):
        server.put("/api/loras/mine", json={"entry": entry()})
        assert server.delete("/api/loras/mine").json() == {"name": "mine", "deleted": True}
        assert server.get("/api/loras/mine").status_code == 404

    def test_the_schema_has_its_own_route(self, server):
        assert server.get("/api/lora-schema").json()["title"] == "LoRA catalog entry"

    def test_a_workflow_that_is_not_an_object_is_a_400(self, server):
        write(str(server.checkout / "workflows"), "models/listy", [1, 2])
        response = server.get("/api/loras", params={"model": "models/listy"})
        assert response.status_code == 400
        assert "models/listy" in response.json()["detail"]


@pytest.fixture
def hub(monkeypatch):
    calls = {}

    def fake(bases, query, terms, limit, rejected, api=None, timeout=None):
        calls.update(bases=bases, query=query, terms=terms, limit=limit, rejected=rejected)
        return calls.get("results", []), calls.get("error")

    monkeypatch.setattr(lora_routes, "search_hub", fake)
    return calls


class TestRecommend:
    def test_catalog_first_then_hub(self, server, hub):
        hub["results"] = [{"source": "hub", "status": "candidate", "model_name": "x/y"}]
        body = server.get("/api/loras/recommend", params={"model": "models/qwen", "query": "voxel style"}).json()
        assert [row["name"] for row in body["catalog"]] == ["qwen-image/voxel"]
        assert body["catalog"][0]["source"] == "catalog"
        assert body["hub"] == hub["results"]
        assert hub["bases"] == ["Qwen/Qwen-Image-2.1"]
        assert hub["terms"] == ["voxel"]
        assert "trial" in body["note"]

    def test_rejected_entries_are_not_offered_but_suppress_the_hub(self, server, hub):
        server.put("/api/loras/qwen-image/bad", json={"entry": entry(model_name="bad/one", status="rejected", evidence=[{"note": "noise"}])})
        body = server.get("/api/loras/recommend", params={"model": QWEN, "query": ""}).json()
        assert "qwen-image/bad" not in [row["name"] for row in body["catalog"]]
        assert hub["rejected"] == {"bad/one": "noise"}

    def test_a_hub_failure_still_returns_the_catalog(self, server, hub):
        hub["error"] = "Hub search timed out after 20 s"
        body = server.get("/api/loras/recommend", params={"model": QWEN, "query": "voxel"}).json()
        assert body["hub_error"] == "Hub search timed out after 20 s"
        assert body["catalog"]

    def test_recommend_is_not_read_as_an_entry_name(self, server, hub):
        response = server.get("/api/loras/recommend", params={"model": QWEN})
        assert response.status_code == 200
        assert "catalog" in response.json()

    def test_limit_is_bounded(self, server, hub):
        assert server.get("/api/loras/recommend", params={"model": QWEN, "limit": 0}).status_code == 422
        assert server.get("/api/loras/recommend", params={"model": QWEN, "limit": 26}).status_code == 422

    def test_a_model_with_no_repo_bases_skips_the_hub(self, server, hub):
        write(str(server.checkout / "workflows"), "models/local", {"steps": [{"name": "s", "pipeline": {"from_pretrained_arguments": {"model_name": "./weights"}}}]})
        body = server.get("/api/loras/recommend", params={"model": "models/local"}).json()
        assert body["resolved"] == [] and body["hub"] == [] and "bases" not in hub

    def test_an_overlong_query_is_a_422(self, server, hub):
        query = "x" * 201
        assert server.get("/api/loras/recommend", params={"model": QWEN, "query": query}).status_code == 422

