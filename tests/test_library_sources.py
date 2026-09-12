"""The prompt and asset search paths: an --examples-dir tree brings the
prompts and media its workflows reference along with it, read-only, behind
the workspace's own libraries."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.assets import asset_search_path, resolve_asset_reference
from dw.prompts import fetch_prompt, prompt_search_path, resolve_prompt_reference
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workflow_sources import EXAMPLES_ORIGIN, WORKSPACE_ORIGIN
from dw.workspace import (
    ASSETS_SUBDIR,
    PROMPTS_SUBDIR,
    Workspace,
    example_libraries,
    library_fallbacks,
    set_library_fallbacks,
)

from .test_server import ScriptedWorkerManager, success_script, valid_workflow


def write_prompt(directory, name, text):
    path = os.path.join(directory, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump({"text": text}, file)
    return path


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """A workspace and a checkout beside it: the checkout's workflows/ is
    what --examples-dir names, and its prompts/ and assets/ sit next to it."""
    workspace = Workspace(tmp_path / "studio", "flag").ensure()

    checkout = tmp_path / "repo"
    (checkout / "workflows").mkdir(parents=True)
    (checkout / "prompts" / "flux").mkdir(parents=True)
    (checkout / "assets").mkdir(parents=True)
    write_prompt(str(checkout / "prompts"), "flux/daffodil", "a biomechanical daffodil")
    write_prompt(str(checkout / "prompts"), "shared", "the example's text")
    (checkout / "assets" / "iris.png").write_bytes(b"example-png")
    (checkout / "assets" / "shared.png").write_bytes(b"example-shared")

    # No environment left over from another test, and none leaking out of this
    for variable in ("DW_PROMPT_PATH", "DW_ASSET_PATH"):
        monkeypatch.delenv(variable, raising=False)
    return workspace, checkout


class TestDerivation:
    def test_the_libraries_beside_a_workflows_tree_are_found(self, trees):
        _workspace, checkout = trees
        found = example_libraries([str(checkout / "workflows")])
        assert found[PROMPTS_SUBDIR] == [str(checkout / "prompts")]
        assert found[ASSETS_SUBDIR] == [str(checkout / "assets")]

    def test_a_directory_holding_them_itself_is_found(self, trees):
        # --examples-dir pointed at a whole workspace rather than its
        # workflows/ subfolder
        _workspace, checkout = trees
        found = example_libraries([str(checkout)])
        assert found[PROMPTS_SUBDIR] == [str(checkout / "prompts")]

    def test_a_tree_with_no_libraries_brings_none(self, tmp_path):
        bare = tmp_path / "elsewhere" / "workflows"
        bare.mkdir(parents=True)
        found = example_libraries([str(bare)])
        assert found == {PROMPTS_SUBDIR: [], ASSETS_SUBDIR: []}

    def test_fallbacks_round_trip_through_the_environment(self, trees):
        # This is how the worker subprocess learns them: spawn inherits the
        # environment, it does not inherit the argument parser
        _workspace, checkout = trees
        set_library_fallbacks(PROMPTS_SUBDIR, [str(checkout / "prompts")])
        assert library_fallbacks(PROMPTS_SUBDIR) == [str(checkout / "prompts")]

    def test_the_primary_library_is_not_repeated_as_a_fallback(self, trees):
        _workspace, checkout = trees
        set_library_fallbacks(PROMPTS_SUBDIR, [str(checkout / "prompts")])
        assert library_fallbacks(PROMPTS_SUBDIR, str(checkout / "prompts")) == []

    def test_a_missing_root_is_dropped(self, tmp_path):
        set_library_fallbacks(PROMPTS_SUBDIR, [str(tmp_path / "gone")])
        assert library_fallbacks(PROMPTS_SUBDIR) == []


class TestResolution:
    @pytest.fixture(autouse=True)
    def libraries(self, trees, monkeypatch):
        workspace, checkout = trees
        monkeypatch.setenv("DW_PROMPT_DIR", workspace.prompts)
        monkeypatch.setenv("DW_ASSET_DIR", workspace.assets)
        found = example_libraries([str(checkout / "workflows")])
        monkeypatch.setenv("DW_PROMPT_PATH", os.pathsep.join(found[PROMPTS_SUBDIR]))
        monkeypatch.setenv("DW_ASSET_PATH", os.pathsep.join(found[ASSETS_SUBDIR]))
        return workspace, checkout

    def test_the_workspace_comes_first_on_the_path(self, libraries):
        workspace, checkout = libraries
        assert prompt_search_path() == [workspace.prompts, str(checkout / "prompts")]
        assert asset_search_path() == [workspace.assets, str(checkout / "assets")]

    def test_an_example_prompt_resolves(self, libraries):
        _workspace, checkout = libraries
        assert fetch_prompt("prompt:flux/daffodil") == "a biomechanical daffodil"
        assert resolve_prompt_reference("prompt:flux/daffodil") == os.path.realpath(
            checkout / "prompts" / "flux" / "daffodil.json"
        )

    def test_an_example_asset_resolves(self, libraries):
        _workspace, checkout = libraries
        assert resolve_asset_reference("asset:iris.png") == os.path.realpath(
            checkout / "assets" / "iris.png"
        )

    def test_the_workspace_shadows_the_example(self, libraries):
        workspace, _checkout = libraries
        write_prompt(workspace.prompts, "shared", "mine")
        with open(os.path.join(workspace.assets, "shared.png"), "wb") as file:
            file.write(b"mine")
        assert fetch_prompt("prompt:shared") == "mine"
        assert resolve_asset_reference("asset:shared.png").startswith(
            os.path.realpath(workspace.assets)
        )

    def test_a_missing_name_names_every_root_it_looked_in(self, libraries):
        workspace, checkout = libraries
        with pytest.raises(ValueError) as error:
            fetch_prompt("prompt:nowhere")
        assert workspace.prompts in str(error.value)
        assert str(checkout / "prompts") in str(error.value)


class TestServer:
    @pytest.fixture
    def client(self, trees, tmp_path):
        workspace, checkout = trees
        write_prompt(workspace.prompts, "mine", "my own text")
        with open(os.path.join(workspace.assets, "mine.png"), "wb") as file:
            file.write(b"mine-png")
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
            yield client, workspace, checkout

    def test_the_prompt_listing_spans_the_path(self, client):
        api, workspace, checkout = client
        body = api.get("/api/prompts").json()
        assert set(body["prompts"]) == {"mine", "flux/daffodil", "shared"}
        assert body["origins"]["mine"] == WORKSPACE_ORIGIN
        assert body["origins"]["flux/daffodil"] == EXAMPLES_ORIGIN
        assert body["prompt_dir"] == workspace.prompts
        assert body["prompt_dirs"] == [workspace.prompts, str(checkout / "prompts")]
        assert body["details"]["flux/daffodil"]["text"] == "a biomechanical daffodil"

    def test_an_example_prompt_reads(self, client):
        api, _workspace, _checkout = client
        assert api.get("/api/prompts/flux/daffodil").json()["text"] == (
            "a biomechanical daffodil"
        )
        assert api.get("/api/prompts/flux/daffodil/download").status_code == 200

    def test_a_missing_prompt_is_still_a_404(self, client):
        api, _workspace, _checkout = client
        assert api.get("/api/prompts/nowhere").status_code == 404

    def test_an_example_prompt_cannot_be_deleted(self, client):
        api, _workspace, checkout = client
        response = api.delete("/api/prompts/flux/daffodil")
        assert response.status_code == 403
        assert os.path.isfile(checkout / "prompts" / "flux" / "daffodil.json")

    def test_saving_an_example_prompt_writes_a_copy(self, client):
        api, workspace, checkout = client
        saved = api.put(
            "/api/prompts/flux/daffodil",
            json={"prompt": {"text": "changed"}},
        )
        assert saved.status_code == 200
        assert saved.json()["path"].startswith(os.path.abspath(workspace.prompts))
        with open(checkout / "prompts" / "flux" / "daffodil.json") as file:
            assert json.load(file)["text"] == "a biomechanical daffodil"
        assert api.get("/api/prompts/flux/daffodil").json()["text"] == "changed"

    def test_the_asset_listing_spans_the_path(self, client):
        api, workspace, checkout = client
        body = api.get("/api/assets").json()
        origins = {entry["name"]: entry["origin"] for entry in body["assets"]}
        assert origins == {
            "mine.png": WORKSPACE_ORIGIN,
            "iris.png": EXAMPLES_ORIGIN,
            "shared.png": EXAMPLES_ORIGIN,
        }
        assert body["asset_dir"] == workspace.assets
        assert body["asset_dirs"] == [
            os.path.abspath(workspace.assets),
            str(checkout / "assets"),
        ]

    def test_an_example_asset_is_served(self, client):
        api, _workspace, _checkout = client
        assert api.get("/inputs/iris.png").content == b"example-png"
        assert api.get("/inputs/mine.png").content == b"mine-png"
        assert api.get("/inputs/nothing.png").status_code == 404

    def test_validated_arguments_reach_the_whole_asset_path(self, client, tmp_path):
        """An argument may name an example asset or the workspace's own, and
        a miss has to name the workspace's library rather than whichever root
        happened to be searched last - a message that leaves out the library
        the caller works in reads as though it was never looked in."""
        api, workspace, checkout = client
        definition = valid_workflow("refs")
        definition["variables"]["image"] = "asset:mine.png"

        for name in ("asset:mine.png", "asset:iris.png"):
            answer = api.post(
                "/api/validate",
                json={"workflow": definition, "arguments": {"image": name}},
            ).json()
            assert answer["valid"] is True, answer

        missing = api.post(
            "/api/validate",
            json={"workflow": definition, "arguments": {"image": "asset:nope.png"}},
        ).json()

        assert missing["valid"] is False
        assert missing["errors"][0]["path"] == "arguments.image"
        assert os.path.abspath(workspace.assets) in missing["errors"][0]["message"]
