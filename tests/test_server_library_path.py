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
    merge_patch,
    pin_library_path,
)
from dw.server.app import create_app
from dw.server.deps import server_prompt_library, sources_for, workspace_for
from dw.server.jobs import JobManager
from dw.server.outputs import asset_file, workspace_asset_library
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

    def make(prompt_dir=None, layout="separate"):
        # "checkout": the workspace root is the checkout itself, so the
        # examples directory is the workspace's own workflows/
        checkout = tmp_path / "repo"
        workspace = Workspace(
            checkout if layout == "checkout" else tmp_path / "studio", "flag"
        ).ensure()
        os.makedirs(workspace.common_assets, exist_ok=True)
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
        assert [library["root"] for library in body["libraries"]] == [
            server.workspace.prompts
        ]


def described(path):
    return [(root.root, root.origin, root.writable) for root in path.roots()]


class TestTheWorkerBuildsTheAPIsPath:
    """Review Focus 1. `dw.serve` pins the read-only tails in the environment
    for the worker and hands the API the same arguments; the two must name the
    same roots, in the same order, with the same origins."""

    @pytest.fixture(
        params=[
            ("default", "separate"),
            ("elsewhere", "separate"),
            ("default", "checkout"),
        ]
    )
    def pinned(self, request, make_server, monkeypatch, tmp_path):
        # --prompt-dir left alone, and pointed outside the root; and the
        # checkout layout, where --examples-dir is the workspace's own
        # workflows/
        prompts, layout = request.param
        prompt_dir = (
            str(tmp_path / "elsewhere" / "prompts") if prompts == "elsewhere" else None
        )
        server = make_server(prompt_dir=prompt_dir, layout=layout)
        server.layout = layout
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
        api = workspace_asset_library(pinned.state, ws)
        worker = library_path_from_env(ASSETS_KIND, ws.assets).existing()
        expected = [WORKSPACE_ORIGIN, COMMON_ORIGIN]
        if pinned.layout == "separate" or workspace == "shots":
            expected.append(EXAMPLES_ORIGIN)
        assert [root.origin for root in api.roots()] == expected
        assert described(worker) == described(api)

    @pytest.mark.parametrize("workspace", [None, "shots"])
    def test_prompts(self, pinned, workspace):
        # Prompts are shared: the named workspace reads the same library
        api = server_prompt_library(pinned.state)
        worker = library_path_from_env(PROMPTS_KIND, pinned.prompt_dir)
        expected = [WORKSPACE_ORIGIN]
        if pinned.layout == "separate":
            expected.append(EXAMPLES_ORIGIN)
        assert [root.origin for root in api.roots()] == expected
        assert described(worker) == described(api)
        ws = workspace_for(pinned.state, workspace)
        assert ws.prompts == pinned.prompt_dir

    @pytest.mark.parametrize("workspace", [None, "shots"])
    def test_workflows(self, pinned, workspace):
        ws = workspace_for(pinned.state, workspace)
        api = sources_for(pinned.state, ws)
        worker = library_path_from_env(WORKFLOWS_KIND, ws.workflows)
        expected = [WORKSPACE_ORIGIN]
        if pinned.layout == "separate" or workspace == "shots":
            expected.append(EXAMPLES_ORIGIN)
        assert [root.origin for root in api.roots()] == expected
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
        library = workspace_asset_library(server.state, ws)
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


class TestWorkspaceRegistryNamesTheServersPrompts:
    def test_a_named_workspace_reports_the_servers_prompt_dir(
        self, make_server, tmp_path
    ):
        elsewhere = str(tmp_path / "elsewhere" / "prompts")
        server = make_server(prompt_dir=elsewhere)
        created = server.client.post("/api/workspaces", json={"name": "shots"})
        assert created.json()["prompts"] == elsewhere
        listed = server.client.get("/api/workspaces").json()["workspaces"]
        assert {w["name"]: w["prompts"] for w in listed} == {
            "default": elsewhere,
            "shots": elsewhere,
        }

    def test_creating_one_leaves_no_prompts_folder_in_the_root(
        self, make_server, tmp_path
    ):
        server = make_server(prompt_dir=str(tmp_path / "elsewhere" / "prompts"))
        os.rmdir(server.workspace.prompts)  # the fixture made the root's own
        server.client.post("/api/workspaces", json={"name": "shots"})
        assert not os.path.exists(server.workspace.prompts)
        assert not os.path.exists(
            os.path.join(server.workspace.root, "shots", "prompts")
        )


class TestAPromptLinkOutOfItsRoot:
    """The API treats a prompt symlink that leaves its root as a miss and
    looks on; the engine resolver refuses it. Both are confinement."""

    @pytest.fixture
    def linked(self, make_server, tmp_path):
        server = make_server()
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps({"text": "from outside"}))
        os.symlink(outside, os.path.join(server.workspace.prompts, "escape.json"))
        write_prompt(str(server.checkout / "prompts"), "escape", "the example's")
        return server

    def test_the_api_skips_the_link_and_answers_the_next_root(self, linked):
        body = linked.client.get("/api/prompts/escape").json()
        assert body["text"] == "the example's"

    def test_the_engine_resolver_refuses_it(self, linked):
        from dw.prompts import resolve_prompt_reference
        from dw.security import SecurityError

        library = server_prompt_library(linked.state)
        with pytest.raises(SecurityError):
            resolve_prompt_reference("prompt:escape", library=library)


REMOVED_LISTING_FIELDS = (
    "workflow_dir",
    "prompt_dir",
    "asset_dir",
    "sources",
    "prompt_dirs",
    "asset_dirs",
    "origins",
)


class TestOneListingEnvelope:
    """The three listings share one envelope: `libraries` (the search path in
    order), `origin` and `writable` on every entry, `shadowed` for all three."""

    @pytest.fixture
    def server(self, make_server):
        server = make_server()
        definition = valid_workflow("dup")
        for directory in (
            server.workspace.workflows,
            str(server.checkout / "workflows"),
        ):
            with open(os.path.join(directory, "dup.json"), "w") as file:
                json.dump(definition, file)
        with open(
            os.path.join(str(server.checkout / "workflows"), "example-only.json"), "w"
        ) as file:
            json.dump(valid_workflow("example-only"), file)
        write_prompt(server.workspace.prompts, "dup", "the workspace's")
        write_prompt(str(server.checkout / "prompts"), "dup", "the example's")
        write_prompt(str(server.checkout / "prompts"), "example-only", "only here")
        for directory, content in (
            (server.workspace.assets, b"workspace-copy"),
            (str(server.checkout / "assets"), b"example-copy"),
        ):
            with open(os.path.join(directory, "dup.png"), "wb") as file:
                file.write(content)
        return server

    @pytest.mark.parametrize("route", ["workflows", "prompts", "assets"])
    def test_libraries_are_the_search_path_in_order(self, server, route):
        body = server.client.get(f"/api/{route}").json()
        for removed in REMOVED_LISTING_FIELDS:
            assert removed not in body
        libraries = body["libraries"]
        assert all(
            sorted(entry) == ["origin", "root", "writable"] for entry in libraries
        )
        origins = [entry["origin"] for entry in libraries]
        if route == "assets":
            assert origins == [WORKSPACE_ORIGIN, COMMON_ORIGIN, EXAMPLES_ORIGIN]
        else:
            assert origins == [WORKSPACE_ORIGIN, EXAMPLES_ORIGIN]
        # The writable root a save targets is the workspace entry
        own = [
            e for e in libraries if e["writable"] and e["origin"] == WORKSPACE_ORIGIN
        ]
        assert len(own) == 1
        assert os.path.isabs(own[0]["root"])

    def test_the_workspace_is_echoed_by_workflows_and_assets_only(self, server):
        assert server.client.get("/api/workflows").json()["workspace"] == "default"
        assert server.client.get("/api/assets").json()["workspace"] == "default"
        assert "workspace" not in server.client.get("/api/prompts").json()

    def test_every_entry_carries_origin_and_writable(self, server):
        workflows = server.client.get("/api/workflows").json()["details"]
        prompts = server.client.get("/api/prompts").json()["details"]
        assets = server.client.get("/api/assets").json()["assets"]
        pairs = lambda items: {  # noqa: E731
            name: (d["origin"], d["writable"]) for name, d in items.items()
        }
        assert pairs(workflows)["dup"] == (WORKSPACE_ORIGIN, True)
        assert pairs(workflows)["example-only"] == (EXAMPLES_ORIGIN, False)
        assert pairs(prompts)["dup"] == (WORKSPACE_ORIGIN, True)
        assert pairs(prompts)["example-only"] == (EXAMPLES_ORIGIN, False)
        by_name = {a["name"]: (a["origin"], a["writable"]) for a in assets}
        assert by_name["dup.png"] == (WORKSPACE_ORIGIN, True)

    @pytest.mark.parametrize(
        "route,name", [("workflows", "dup"), ("prompts", "dup"), ("assets", "dup.png")]
    )
    def test_the_hidden_copy_is_listed_as_shadowed(self, server, route, name):
        hidden = [
            entry
            for entry in server.client.get(f"/api/{route}").json()["shadowed"]
            if entry["name"] == name
        ]
        assert [(e["origin"], e["shadowed_by"]) for e in hidden] == [
            (EXAMPLES_ORIGIN, WORKSPACE_ORIGIN)
        ]

    @pytest.mark.parametrize(
        "route,name,kind",
        [
            ("workflows", "example-only", "workflows"),
            ("prompts", "example-only", "prompts"),
        ],
    )
    def test_deleting_a_read_only_entry_gives_one_message(
        self, server, route, name, kind
    ):
        response = server.client.delete(f"/api/{route}/{name}")
        assert response.status_code == 403
        examples = str(
            server.checkout / ("workflows" if route == "workflows" else "prompts")
        )
        assert response.json()["detail"] == (
            f"'{name}' is in the read-only examples library ({examples}); "
            f"only the workspace's own {kind} can be deleted"
        )

    def test_deleting_a_read_only_asset_gives_the_same_message(self, server):
        with open(str(server.checkout / "assets" / "example-only.png"), "wb") as file:
            file.write(b"x")
        response = server.client.delete("/api/assets/example-only.png")
        assert response.status_code == 403
        assert response.json()["detail"] == (
            "'example-only.png' is in the read-only examples library "
            f"({server.checkout / 'assets'}); only the workspace's own assets "
            "can be deleted"
        )


class TestAnUploadWithNoAssetLibrary:
    def test_it_is_a_409_rather_than_a_write_into_outputs(self, tmp_path):
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
        with TestClient(app, base_url="http://localhost") as client:
            response = client.post(
                "/api/uploads", params={"filename": "a.png"}, content=b"bytes"
            )
        assert response.status_code == 409
        assert not (tmp_path / "outputs" / "uploads").exists()


class TestLibraryRouteStatuses:
    """B10: a refusal is 400, a failure after the request was understood is 500."""

    def test_an_invalid_workflow_save_is_a_400(self, make_server):
        server = make_server()
        response = server.client.put(
            "/api/workflows/bad", json={"workflow": {"id": "bad", "steps": "no"}}
        )
        assert response.status_code == 400

    def test_a_crash_while_validating_a_save_is_a_500(self, make_server, monkeypatch):
        from dw.workflow import Workflow

        server = make_server()

        def crash(self, *args, **kwargs):
            raise RuntimeError("internals: /secret/path")

        monkeypatch.setattr(Workflow, "validation_errors", crash)
        response = server.client.put(
            "/api/workflows/ok", json={"workflow": valid_workflow("ok")}
        )
        assert response.status_code == 500
        assert "/secret/path" not in response.text

    def test_a_failed_enhance_submit_is_a_500(self, make_server, monkeypatch):
        server = make_server()
        manager = server.state.job_manager

        def crash(*args, **kwargs):
            raise RuntimeError("internals: /secret/path")

        monkeypatch.setattr(manager, "submit", crash)
        response = server.client.post("/api/enhance", json={"idea": "a cat"})
        assert response.status_code == 500
        assert "/secret/path" not in response.text

    def test_an_unknown_enhance_preset_is_a_400(self, make_server):
        server = make_server()
        response = server.client.post(
            "/api/enhance", json={"idea": "a cat", "preset": "nope"}
        )
        assert response.status_code == 400


class TestNoAssetLibraryHasOneWording:
    def test_upload_keep_and_delete_answer_the_same_409(self, tmp_path):
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
        with TestClient(app, base_url="http://localhost") as client:
            upload = client.post(
                "/api/uploads", params={"filename": "a.png"}, content=b"bytes"
            )
            keep = client.post("/api/assets/keep", json={"name": "x.png"})
            delete = client.delete("/api/assets/x.png")
        assert {r.status_code for r in (upload, keep, delete)} == {409}
        assert len({r.json()["detail"] for r in (upload, keep, delete)}) == 1


class TestPromptSaveWithNoPromptLibrary:
    def test_the_save_answers_409_rather_than_500(self, tmp_path):
        app = create_app(
            workflow_dir=str(tmp_path / "workflows"),
            output_dir=str(tmp_path / "outputs"),
            prompt_dir=None,
            job_manager=JobManager(
                str(tmp_path / "outputs"),
                worker_manager=ScriptedWorkerManager(success_script),
                history_path=str(tmp_path / "jobs.sqlite"),
            ),
        )
        with TestClient(app, base_url="http://localhost") as client:
            response = client.put("/api/prompts/x", json={"prompt": {"text": "hello"}})
        assert response.status_code == 409
        assert response.json()["detail"] == "This server has no prompt library"


def test_merge_patch_is_rfc_7396():
    target = {"a": 1, "b": {"c": 2, "d": 3}, "keep": [1, 2]}
    patch = {"a": None, "b": {"c": 9, "e": 4}, "keep": [3]}
    assert merge_patch(target, patch) == {"b": {"c": 9, "d": 3, "e": 4}, "keep": [3]}
    assert target["a"] == 1, "the target is not mutated"


class TestPatchWorkflow:
    """PATCH /api/workflows/{name}: a merge patch onto the stored version,
    read, merged and written under one lock, so a save made between the
    read and the write is not silently lost."""

    def test_patch_merges_onto_the_stored_definition(self, make_server):
        server = make_server()
        server.client.put("/api/workflows/ok", json={"workflow": valid_workflow("ok")})
        answer = server.client.patch(
            "/api/workflows/ok", json={"description": "Patched in place."}
        )
        assert answer.status_code == 200, answer.text
        stored = server.client.get("/api/workflows/ok").json()
        assert stored["description"] == "Patched in place."
        assert stored["steps"] == valid_workflow("ok")["steps"]

    def test_patch_null_deletes_a_key(self, make_server):
        server = make_server()
        server.client.put(
            "/api/workflows/ok",
            json={"workflow": {**valid_workflow("ok"), "description": "Gone soon."}},
        )
        server.client.patch("/api/workflows/ok", json={"description": None})
        assert "description" not in server.client.get("/api/workflows/ok").json()

    def test_patch_of_an_example_writes_a_writable_copy(self, make_server):
        server = make_server()
        example = os.path.join(str(server.checkout / "workflows"), "ex.json")
        with open(example, "w") as file:
            json.dump(valid_workflow("ex"), file)
        answer = server.client.patch(
            "/api/workflows/ex", json={"description": "My copy."}
        )
        assert answer.status_code == 200, answer.text
        with open(os.path.join(server.workspace.workflows, "ex.json")) as file:
            assert json.load(file)["description"] == "My copy."
        with open(example) as file:
            assert "description" not in json.load(file)

    def test_an_invalid_patch_result_is_a_400_and_writes_nothing(self, make_server):
        server = make_server()
        server.client.put("/api/workflows/ok", json={"workflow": valid_workflow("ok")})
        answer = server.client.patch("/api/workflows/ok", json={"steps": "no"})
        assert answer.status_code == 400
        assert server.client.get("/api/workflows/ok").json() == valid_workflow("ok")

    def test_patch_accepts_the_merge_patch_media_type(self, make_server):
        server = make_server()
        server.client.put("/api/workflows/ok", json={"workflow": valid_workflow("ok")})
        answer = server.client.patch(
            "/api/workflows/ok",
            content=json.dumps({"description": "Typed."}),
            headers={"content-type": "application/merge-patch+json"},
        )
        assert answer.status_code == 200, answer.text
        assert server.client.get("/api/workflows/ok").json()["description"] == "Typed."
