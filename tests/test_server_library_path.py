"""The server's consumers of `LibraryPath`: the prompt and asset search paths
the API builds from `app.state`, set against the ones the worker builds from
the environment `dw.serve` pins."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.assets import resolve_asset_reference
from dw.library import (
    ASSETS_KIND,
    COMMON_ORIGIN,
    EXAMPLES_ORIGIN,
    PROMPTS_KIND,
    WORKFLOWS_KIND,
    WORKSPACE_ORIGIN,
    library_path_from_env,
    pin_library_path,
)
from dw.server.app import create_app
from dw.server.deps import prompt_library, sources_for, workspace_for
from dw.server.jobs import JobManager
from dw.server.outputs import asset_file, asset_library
from dw.workspace import Workspace, named_workspace

from .test_server import ScriptedWorkerManager, success_script, valid_workflow


def write_prompt(directory, name, text):
    path = os.path.join(directory, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump({"text": text}, file)
    return path


class Server:
    def __init__(self, client, workspace, checkout, prompt_dir, app):
        self.client = client
        self.workspace = workspace
        self.checkout = checkout
        self.prompt_dir = prompt_dir
        self.app = app

    @property
    def state(self):
        return self.app.state


@pytest.fixture
def make_server(tmp_path, monkeypatch):
    """A server over a workspace root with an examples tree beside it and the
    root's `common/assets`. `prompt_dir` is the server's own prompt library;
    left alone it is the root's `prompts/`."""
    for variable in (
        "DW_PROMPT_PATH",
        "DW_ASSET_PATH",
        "DW_WORKFLOW_PATH",
        "DW_PROMPT_DIR",
        "DW_ASSET_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)
    opened = []

    def make(prompt_dir=None):
        workspace = Workspace(tmp_path / "studio", "flag").ensure()
        os.makedirs(workspace.common_assets, exist_ok=True)
        checkout = tmp_path / "repo"
        for sub in ("workflows", "prompts", "assets"):
            (checkout / sub).mkdir(parents=True, exist_ok=True)
        prompts = prompt_dir or workspace.prompts
        os.makedirs(prompts, exist_ok=True)
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
            prompt_dir=prompts,
            asset_dir=workspace.assets,
            examples_dirs=[str(checkout / "workflows")],
            workspace=workspace.root,
        )
        client = TestClient(app, base_url="http://localhost")
        client.__enter__()
        opened.append(client)
        return Server(client, workspace, checkout, prompts, app)

    yield make
    for client in opened:
        client.__exit__(None, None, None)


def prompt_workflow(name):
    workflow = valid_workflow()
    workflow["variables"]["prompt"] = f"prompt:{name}"
    return workflow


class TestNamedWorkspacePrompts:
    def test_a_named_workspace_plans_against_the_servers_prompt_dir(
        self, make_server, tmp_path
    ):
        # --prompt-dir names a library that is not <root>/prompts; prompts are
        # shared by reference, so a named workspace's plan must read it too
        server = make_server(prompt_dir=str(tmp_path / "elsewhere" / "prompts"))
        write_prompt(server.prompt_dir, "only-here", "a lighthouse")
        server.client.post("/api/workspaces", json={"name": "shots"})

        workflow = prompt_workflow("only-here")
        default = server.client.post("/api/validate", json={"workflow": workflow})
        named = server.client.post(
            "/api/validate", json={"workflow": workflow, "workspace": "shots"}
        )

        assert default.json()["valid"] is True
        assert named.json()["valid"] is True
        # The fingerprint is taken over the workflow with each stored prompt
        # inlined, so a named workspace that read another library would
        # fingerprint the bare reference and quote a different plan
        assert (
            named.json()["plan"]["fingerprint"] == default.json()["plan"]["fingerprint"]
        )

    def test_the_named_workspace_hands_the_prompt_dir_back(self, tmp_path):
        root = Workspace(tmp_path / "studio", "flag")
        elsewhere = str(tmp_path / "elsewhere")
        named = named_workspace(root, "shots", prompts_root=elsewhere)
        assert named.prompts == elsewhere
        assert named_workspace(root, "shots").prompts == root.prompts


class TestPromptRootsDropAMissingExampleDir:
    def test_the_listing_names_only_directories_that_exist(self, make_server, tmp_path):
        server = make_server()
        # An examples tree whose prompts/ is gone: the engine already drops it
        gone = tmp_path / "repo" / "prompts"
        gone.rmdir()
        body = server.client.get("/api/prompts").json()
        assert body["prompt_dirs"] == [server.workspace.prompts]


def described(path):
    return [(root.root, root.origin, root.writable) for root in path.roots()]


class TestTheWorkerBuildsTheAPIsPath:
    """Review Focus 1. `dw.serve` pins the read-only tails in the environment
    for the worker and hands the API the same arguments; the two must name the
    same roots, in the same order, with the same origins."""

    @pytest.fixture
    def pinned(self, make_server, monkeypatch):
        server = make_server()
        monkeypatch.setenv("DW_WORKSPACE", server.workspace.root)
        monkeypatch.setenv("DW_WORKSPACE_SOURCE", "flag")
        examples = [str(server.checkout / "workflows")]
        for variable in ("DW_ASSET_PATH", "DW_PROMPT_PATH", "DW_WORKFLOW_PATH"):
            # pin_library_path writes os.environ directly; registering the
            # variable first makes monkeypatch restore it afterwards
            monkeypatch.setenv(variable, "")
        for kind in (ASSETS_KIND, PROMPTS_KIND, WORKFLOWS_KIND):
            pin_library_path(kind, server.workspace, examples)
        server.client.post("/api/workspaces", json={"name": "shots"})
        return server

    @pytest.mark.parametrize("workspace", [None, "shots"])
    def test_assets(self, pinned, workspace):
        ws = workspace_for(pinned.state, workspace)
        api = asset_library(pinned.state, ws)
        worker = library_path_from_env(ASSETS_KIND, ws.assets).existing()
        assert [root.origin for root in api.roots()] == [
            WORKSPACE_ORIGIN,
            COMMON_ORIGIN,
            EXAMPLES_ORIGIN,
        ]
        assert described(worker) == described(api)

    @pytest.mark.parametrize("workspace", [None, "shots"])
    def test_prompts(self, pinned, workspace):
        # Prompts are shared: the named workspace reads the same library
        api = prompt_library(pinned.state)
        worker = library_path_from_env(PROMPTS_KIND, pinned.prompt_dir)
        assert [root.origin for root in api.roots()] == [
            WORKSPACE_ORIGIN,
            EXAMPLES_ORIGIN,
        ]
        assert described(worker) == described(api)
        ws = workspace_for(pinned.state, workspace)
        assert ws.prompts == pinned.prompt_dir

    @pytest.mark.parametrize("workspace", [None, "shots"])
    def test_workflows(self, pinned, workspace):
        ws = workspace_for(pinned.state, workspace)
        api = sources_for(pinned.state, ws)
        worker = library_path_from_env(WORKFLOWS_KIND, ws.workflows)
        assert [root.origin for root in api.roots()] == [
            WORKSPACE_ORIGIN,
            EXAMPLES_ORIGIN,
        ]
        assert described(worker) == described(api)


class TestTheWorkspaceShadowsAnExample:
    """Review Focus 2, assets. Every way of asking for a name answers the
    workspace's copy, and the example's is reported as hidden."""

    @pytest.fixture
    def server(self, make_server):
        server = make_server()
        for directory, content in (
            (server.workspace.assets, b"workspace-copy"),
            (str(server.checkout / "assets"), b"example-copy"),
        ):
            with open(os.path.join(directory, "shared.png"), "wb") as file:
                file.write(content)
        with open(str(server.checkout / "assets" / "only-example.png"), "wb") as file:
            file.write(b"example-only")
        return server

    def test_asset_resolution_takes_the_workspace_copy(self, server):
        ws = server.state.default_workspace
        library = asset_library(server.state, ws)
        expected = os.path.realpath(os.path.join(server.workspace.assets, "shared.png"))
        assert resolve_asset_reference("asset:shared.png", library=library) == expected
        assert asset_file(server.state, "asset:shared.png", ws) == expected

    def test_the_inputs_route_serves_the_workspace_copy(self, server):
        response = server.client.get("/inputs/shared.png")
        assert response.content == b"workspace-copy"
        assert server.client.get("/inputs/only-example.png").content == b"example-only"

    def test_the_listing_offers_the_workspace_copy_and_hides_the_example(self, server):
        body = server.client.get("/api/assets").json()
        offered = [a for a in body["assets"] if a["name"] == "shared.png"]
        assert [a["origin"] for a in offered] == [WORKSPACE_ORIGIN]
        hidden = [a for a in body["shadowed"] if a["name"] == "shared.png"]
        assert [(a["origin"], a["shadowed_by"]) for a in hidden] == [
            (EXAMPLES_ORIGIN, WORKSPACE_ORIGIN)
        ]
        # A name only the example has is offered, not hidden
        assert [
            a["origin"] for a in body["assets"] if a["name"] == "only-example.png"
        ] == [EXAMPLES_ORIGIN]
