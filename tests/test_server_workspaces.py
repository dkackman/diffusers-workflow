"""Several workspaces on one server: named subdirectories of the workspace
root, each with its own workflows, assets and outputs, all sharing the one
prompt library."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import Workspace

from .test_server import ScriptedWorkerManager, success_script, valid_workflow
from .test_server import wait_for_status


@pytest.fixture
def workspace_root(tmp_path):
    """A workspace root: its own folders are the default workspace, and the
    prompt library at the root is shared by everything under it."""
    root = Workspace(tmp_path / "studio", "flag").ensure()
    return root


@pytest.fixture
def server(workspace_root, tmp_path):
    def make(script=success_script):
        manager = JobManager(
            workspace_root.outputs,
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=workspace_root.workflows,
        )
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


class TestRegistry:
    def test_a_fresh_root_has_only_the_default(self, server):
        with server() as client:
            body = client.get("/api/workspaces").json()
        assert [w["name"] for w in body["workspaces"]] == ["default"]
        assert body["workspaces"][0]["default"] is True

    def test_creating_one_makes_its_folders(self, server, workspace_root):
        with server() as client:
            created = client.post("/api/workspaces", json={"name": "shots"})
            assert created.status_code == 201
            body = created.json()

        assert body["name"] == "shots"
        for folder in ("workflows", "assets", "outputs"):
            assert os.path.isdir(os.path.join(workspace_root.root, "shots", folder))
        # and it shares the root's one prompt library rather than making its own
        assert body["prompts"] == workspace_root.prompts
        assert not os.path.exists(os.path.join(workspace_root.root, "shots", "prompts"))

    def test_a_reserved_name_is_refused(self, server):
        with server() as client:
            response = client.post("/api/workspaces", json={"name": "outputs"})
        assert response.status_code == 400
        assert "cannot name a workspace" in response.json()["detail"]

    @pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", ""])
    def test_a_name_that_is_not_a_name_is_refused(self, server, name):
        with server() as client:
            response = client.post("/api/workspaces", json={"name": name})
        assert response.status_code in (400, 422)

    def test_creating_the_same_name_twice_conflicts(self, server):
        with server() as client:
            assert (
                client.post("/api/workspaces", json={"name": "shots"}).status_code
                == 201
            )
            again = client.post("/api/workspaces", json={"name": "shots"})
        assert again.status_code == 409


class TestDeletion:
    def test_deletion_reports_what_it_would_remove_and_waits(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            outputs = os.path.join(workspace_root.root, "shots", "outputs")
            with open(os.path.join(outputs, "still.png"), "wb") as handle:
                handle.write(b"png-bytes")

            refused = client.delete("/api/workspaces/shots")
            assert refused.status_code == 409
            detail = refused.json()["detail"]
            assert detail["contents"]["outputs"]["files"] == 1
            assert os.path.isdir(outputs)

            deleted = client.delete("/api/workspaces/shots?acknowledged=true")
            assert deleted.status_code == 200
            assert not os.path.exists(os.path.join(workspace_root.root, "shots"))

    def test_the_default_workspace_cannot_be_deleted(self, server, workspace_root):
        with server() as client:
            response = client.delete("/api/workspaces/default?acknowledged=true")
        assert response.status_code == 400
        assert os.path.isdir(workspace_root.prompts)

    def test_a_directory_holding_more_than_a_workspace_is_refused(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            stray = os.path.join(workspace_root.root, "shots", "notes.txt")
            with open(stray, "w") as handle:
                handle.write("mine")

            refused = client.delete("/api/workspaces/shots?acknowledged=true")
            assert refused.status_code == 409
            assert refused.json()["detail"]["entries"] == ["notes.txt"]
            assert os.path.isfile(stray)

    def test_a_source_tree_beside_the_root_is_not_a_workspace(
        self, server, workspace_root
    ):
        # A checkout as the root has dw/workflows/ - one folder must not make
        # the package a workspace
        os.makedirs(os.path.join(workspace_root.root, "dw", "workflows"))
        with server() as client:
            names = [
                w["name"] for w in client.get("/api/workspaces").json()["workspaces"]
            ]
            assert names == ["default"]
            assert client.get("/api/workflows?workspace=dw").status_code == 404

    def test_an_unknown_workspace_is_a_404(self, server):
        with server() as client:
            assert (
                client.delete("/api/workspaces/nope?acknowledged=true").status_code
                == 404
            )


class TestScopedRoutes:
    def test_workflows_are_per_workspace(self, server):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            client.put(
                "/api/workflows/Mine?workspace=shots",
                json={"workflow": valid_workflow("mine")},
            )

            assert client.get("/api/workflows").json()["workflows"] == []
            scoped = client.get("/api/workflows?workspace=shots").json()
            assert scoped["workflows"] == ["Mine"]
            assert scoped["workspace"] == "shots"
            assert client.get("/api/workflows/Mine?workspace=shots").status_code == 200
            assert client.get("/api/workflows/Mine").status_code == 404

    def test_assets_and_uploads_are_per_workspace(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            uploaded = client.post(
                "/api/uploads",
                params={"filename": "iris.png", "workspace": "shots"},
                content=b"png",
            )
            assert uploaded.status_code == 201
            assert uploaded.json()["path"].startswith("asset:uploads/")

            assert client.get("/api/assets").json()["assets"] == []
            scoped = client.get("/api/assets?workspace=shots").json()
            assert len(scoped["assets"]) == 1
            assert scoped["asset_dir"] == os.path.join(
                workspace_root.root, "shots", "assets"
            )

    def test_an_unknown_workspace_is_refused_rather_than_created(
        self, server, workspace_root
    ):
        with server() as client:
            assert client.get("/api/workflows?workspace=ghost").status_code == 404
        assert not os.path.exists(os.path.join(workspace_root.root, "ghost"))


class TestConfinement:
    """A named workspace's own workflows/ is the confinement root for that
    workspace's requests, not the server's default workflow_dir - the bug
    this class guards against would reject a base_dir or a relative
    sub-workflow path that lives entirely inside the named workspace."""

    def test_validate_accepts_a_base_dir_inside_the_named_workspace(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            base_dir = os.path.join(workspace_root.root, "shots", "workflows")

            response = client.post(
                "/api/validate",
                params={"workspace": "shots"},
                json={"workflow": valid_workflow("mine"), "base_dir": base_dir},
            )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_saving_a_workflow_with_a_relative_sub_workflow_path(
        self, server, workspace_root
    ):
        sub = {
            "id": "sub",
            "variables": {},
            "steps": [
                {
                    "name": "gen",
                    "pipeline": {
                        "configuration": {
                            "component_type": "{Fake}",
                            "no_generator": True,
                        },
                        "from_pretrained_arguments": {"model_name": "m"},
                        "arguments": {},
                    },
                }
            ],
        }
        main = {
            "id": "main",
            "variables": {},
            "steps": [
                {
                    "name": "sub",
                    "workflow": {"path": "parts/Sub.json", "arguments": {}},
                }
            ],
        }
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            client.put(
                "/api/workflows/parts/Sub?workspace=shots", json={"workflow": sub}
            )
            response = client.put(
                "/api/workflows/Main?workspace=shots", json={"workflow": main}
            )
        assert response.status_code == 200

    def test_an_unknown_workspace_query_param_is_a_404(self, server):
        with server() as client:
            assert client.get("/api/workflows?workspace=nope").status_code == 404

    def test_a_traversal_attempt_as_a_workspace_name_is_a_400(self, server):
        with server() as client:
            response = client.get("/api/workflows", params={"workspace": "../x"})
        assert response.status_code == 400


class TestServingFiles:
    def test_outputs_are_served_from_the_workspace_that_made_them(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            shots = os.path.join(workspace_root.root, "shots", "outputs")
            with open(os.path.join(shots, "still.png"), "wb") as handle:
                handle.write(b"shots-bytes")
            with open(
                os.path.join(workspace_root.outputs, "still.png"), "wb"
            ) as handle:
                handle.write(b"default-bytes")

            assert client.get("/outputs/still.png").content == b"default-bytes"
            scoped = client.get("/outputs/still.png?workspace=shots")
            assert scoped.content == b"shots-bytes"

    def test_a_file_only_in_a_named_workspace_is_not_served_by_default(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            shots = os.path.join(workspace_root.root, "shots", "outputs")
            with open(os.path.join(shots, "only.png"), "wb") as handle:
                handle.write(b"x")

            assert client.get("/outputs/only.png").status_code == 404
            assert client.get("/outputs/only.png?workspace=shots").status_code == 200

    def test_serving_still_answers_range_requests(self, server, workspace_root):
        """Video scrubbing depends on it, and the static mount this route
        replaced supported it."""
        with open(os.path.join(workspace_root.outputs, "clip.mp4"), "wb") as handle:
            handle.write(b"0123456789")
        with server() as client:
            response = client.get("/outputs/clip.mp4", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/10"

    def test_a_file_cannot_be_read_from_outside_the_workspace(
        self, server, workspace_root
    ):
        with open(os.path.join(workspace_root.root, "secret.png"), "wb") as handle:
            handle.write(b"not yours")
        with server() as client:
            assert client.get("/outputs/../secret.png").status_code in (404, 400)
            assert client.get("/outputs/%2e%2e%2fsecret.png").status_code in (404, 400)

    def test_assets_are_served_per_workspace(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            library = os.path.join(workspace_root.root, "shots", "assets")
            with open(os.path.join(library, "iris.png"), "wb") as handle:
                handle.write(b"iris")

            assert client.get("/inputs/iris.png").status_code == 404
            served = client.get("/inputs/iris.png?workspace=shots")
            assert served.status_code == 200
            assert served.content == b"iris"

    def test_the_gallery_is_scoped_and_its_urls_carry_the_workspace(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            shots = os.path.join(workspace_root.root, "shots", "outputs")
            with open(os.path.join(shots, "still.png"), "wb") as handle:
                handle.write(b"x")

            assert client.get("/api/gallery").json()["files"] == []
            body = client.get("/api/gallery?workspace=shots").json()
            assert body["workspace"] == "shots"
            assert len(body["files"]) == 1
            url = body["files"][0]["url"]
            assert "workspace=shots" in url
            # and that URL is one the server actually serves
            assert client.get(url).content == b"x"

    def test_an_output_answers_304_when_unmodified(self, server, workspace_root):
        """The bare FileResponse this route used to return never answers
        304 (no If-None-Match handling) - going through StaticFiles is what
        restores it, and is what keeps the gallery from re-streaming every
        file on every load."""
        with open(os.path.join(workspace_root.outputs, "still.png"), "wb") as handle:
            handle.write(b"bytes")
        with server() as client:
            first = client.get("/outputs/still.png")
            assert first.status_code == 200
            etag = first.headers["etag"]

            second = client.get("/outputs/still.png", headers={"If-None-Match": etag})
        assert second.status_code == 304

    def test_an_uploaded_asset_is_reachable_at_its_scoped_url(
        self, server, workspace_root
    ):
        """upload_media must scope the returned URL the same way the
        gallery does, or the editor's preview of a just-uploaded file in a
        named workspace 404s."""
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            uploaded = client.post(
                "/api/uploads",
                params={"filename": "iris.png", "workspace": "shots"},
                content=b"iris-bytes",
            )
            assert uploaded.status_code == 201
            url = uploaded.json()["url"]
            assert "workspace=shots" in url

            fetched = client.get(url)
        assert fetched.status_code == 200
        assert fetched.content == b"iris-bytes"

    def test_asset_listing_entries_carry_a_fetchable_url(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            library = os.path.join(workspace_root.root, "shots", "assets")
            with open(os.path.join(library, "iris.png"), "wb") as handle:
                handle.write(b"iris")

            body = client.get("/api/assets?workspace=shots").json()
            assert len(body["assets"]) == 1
            url = body["assets"][0]["url"]
            assert "workspace=shots" in url

            fetched = client.get(url)
        assert fetched.status_code == 200
        assert fetched.content == b"iris"

    def test_get_workflow_reports_its_origin_and_writability(
        self, server, workspace_root
    ):
        with server() as client:
            client.put(
                "/api/workflows/Basic", json={"workflow": valid_workflow("mine")}
            )
            response = client.get("/api/workflows/Basic")
        assert response.status_code == 200
        assert response.headers["x-workflow-origin"] == "workspace"
        assert response.headers["x-workflow-writable"] == "true"


class TestKeepingOutputs:
    """Generated files are named by the run that made them; keeping one puts
    it in the asset library under a name later workflows can rely on."""

    def written(self, root, name, content=b"png-bytes"):
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def test_an_output_is_kept_under_a_stable_name(self, server, workspace_root):
        self.written(workspace_root.outputs, "Gyre/20260905-101500-aaaaaaaa/still.png")
        with server() as client:
            response = client.post(
                "/api/assets/keep",
                json={
                    "name": "Gyre/20260905-101500-aaaaaaaa/still.png",
                    "asset_name": "gyre/hero.png",
                },
            )
        assert response.status_code == 201
        body = response.json()
        assert body["reference"] == "asset:gyre/hero.png"
        kept = os.path.join(workspace_root.assets, "gyre", "hero.png")
        assert open(kept, "rb").read() == b"png-bytes"
        # the run's own copy is untouched - keeping is not moving
        assert os.path.exists(
            os.path.join(
                workspace_root.outputs, "Gyre/20260905-101500-aaaaaaaa/still.png"
            )
        )

    def test_it_links_rather_than_copying_when_it_can(self, server, workspace_root):
        """One frame of a multi-gigabyte render should not cost another copy
        of it; both names refer to the same content."""
        source = self.written(workspace_root.outputs, "Gyre/run/clip.mp4")
        with server() as client:
            body = client.post(
                "/api/assets/keep", json={"name": "Gyre/run/clip.mp4"}
            ).json()
        assert body["reference"] == "asset:clip.mp4"
        if body["linked"]:
            assert os.stat(source).st_ino == os.stat(body["path"]).st_ino

    def test_the_name_defaults_to_the_files_own(self, server, workspace_root):
        self.written(workspace_root.outputs, "Gyre/run/still.png")
        with server() as client:
            body = client.post(
                "/api/assets/keep", json={"name": "Gyre/run/still.png"}
            ).json()
        assert body["reference"] == "asset:still.png"

    def test_an_existing_asset_is_not_replaced_silently(self, server, workspace_root):
        self.written(workspace_root.outputs, "Gyre/run/still.png", b"new")
        self.written(workspace_root.assets, "hero.png", b"old")
        with server() as client:
            refused = client.post(
                "/api/assets/keep",
                json={"name": "Gyre/run/still.png", "asset_name": "hero.png"},
            )
            assert refused.status_code == 409
            assert (
                open(os.path.join(workspace_root.assets, "hero.png"), "rb").read()
                == b"old"
            )

            replaced = client.post(
                "/api/assets/keep",
                json={
                    "name": "Gyre/run/still.png",
                    "asset_name": "hero.png",
                    "overwrite": True,
                },
            )
            assert replaced.status_code == 201
        assert (
            open(os.path.join(workspace_root.assets, "hero.png"), "rb").read() == b"new"
        )

    @pytest.mark.parametrize("asset_name", ["../escape.png", "/etc/passwd.png"])
    def test_a_destination_cannot_leave_the_library(
        self, server, workspace_root, asset_name
    ):
        self.written(workspace_root.outputs, "Gyre/run/still.png")
        with server() as client:
            response = client.post(
                "/api/assets/keep",
                json={"name": "Gyre/run/still.png", "asset_name": asset_name},
            )
        assert response.status_code == 400

    def test_keeping_stays_inside_the_workspace(self, server, workspace_root):
        """The source is read from the named workspace's outputs and the copy
        lands in its assets - neither reaches the default workspace."""
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            shots = os.path.join(workspace_root.root, "shots")
            self.written(os.path.join(shots, "outputs"), "Gyre/run/still.png")

            assert (
                client.post(
                    "/api/assets/keep", json={"name": "Gyre/run/still.png"}
                ).status_code
                == 404
            )
            kept = client.post(
                "/api/assets/keep?workspace=shots",
                json={"name": "Gyre/run/still.png"},
            )
        assert kept.status_code == 201
        assert os.path.exists(os.path.join(shots, "assets", "still.png"))
        assert not os.path.exists(os.path.join(workspace_root.assets, "still.png"))

    def test_a_kept_asset_is_listed_and_referable(self, server, workspace_root):
        self.written(workspace_root.outputs, "Gyre/run/still.png")
        with server() as client:
            client.post(
                "/api/assets/keep",
                json={"name": "Gyre/run/still.png", "asset_name": "gyre/hero.png"},
            )
            listed = client.get("/api/assets").json()["assets"]
        assert [entry["reference"] for entry in listed] == ["asset:gyre/hero.png"]


class TestRunning:
    def test_a_job_runs_in_the_workspace_it_named(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            client.put(
                "/api/workflows/Mine?workspace=shots",
                json={"workflow": valid_workflow("mine")},
            )
            response = client.post(
                "/api/jobs", json={"workflow_path": "Mine", "workspace": "shots"}
            )
            assert response.status_code == 201
            detail = wait_for_status(
                client, response.json()["id"], {"succeeded", "failed"}
            )

        assert detail["status"] == "succeeded"

    def test_enhance_runs_in_the_selected_workspace(self, server):
        """The enhance job used to be submitted unscoped, so its text landed
        in the default workspace's outputs while the editor read it back
        from the selected one and got a 404."""
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            response = client.post(
                "/api/enhance?workspace=shots",
                json={"idea": "a cat in the rain", "preset": "h3"},
            )
            assert response.status_code == 201
            detail = wait_for_status(
                client, response.json()["id"], {"succeeded", "failed"}
            )
        assert detail["workspace"] == "shots"

    def test_an_inline_job_is_confined_to_its_own_workspace(
        self, server, workspace_root
    ):
        """An inline (body-supplied) workflow in a named workspace used to
        record the manager's process-wide workflow_dir as its confinement
        while base_dir defaulted to the job's own workflow_dir - the worker
        then re-validated base_dir against a workflow_dir it did not match,
        and a 201 always failed. base_dir and workflow_dir must agree."""
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            response = client.post(
                "/api/jobs",
                json={"workflow": valid_workflow("inline"), "workspace": "shots"},
            )
            assert response.status_code == 201
            job_id = response.json()["id"]
            detail = wait_for_status(client, job_id, {"succeeded", "failed"})
            assert detail["status"] == "succeeded"

            manager = client.app.state.job_manager
            command = manager.worker_manager.commands[0]
            shots_workflows = os.path.join(workspace_root.root, "shots", "workflows")
            assert command["workflow_dir"] == shots_workflows
            assert (
                os.path.commonpath([command["base_dir"], shots_workflows])
                == shots_workflows
            )

    def test_rerun_stays_in_the_workspace_it_ran_in(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            client.put(
                "/api/workflows/Mine?workspace=shots",
                json={"workflow": valid_workflow("mine")},
            )
            original = client.post(
                "/api/jobs", json={"workflow_path": "Mine", "workspace": "shots"}
            ).json()
            wait_for_status(client, original["id"], {"succeeded", "failed"})

            rerun = client.post(f"/api/jobs/{original['id']}/rerun")
            assert rerun.status_code == 201
            rerun_id = rerun.json()["id"]
            wait_for_status(client, rerun_id, {"succeeded", "failed"})

            summary = client.get(f"/api/jobs/{rerun_id}").json()
            assert summary["status"] == "succeeded"

            manager = client.app.state.job_manager
            rerun_command = manager.worker_manager.commands[-1]
            assert rerun_command["output_dir"] == os.path.join(
                workspace_root.root, "shots", "outputs"
            )

            listed = {job["id"]: job for job in client.get("/api/jobs").json()["jobs"]}
            assert listed[rerun_id]["workspace"] == "shots"

    def test_jobs_list_filters_by_workspace(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            client.put(
                "/api/workflows/Mine?workspace=shots",
                json={"workflow": valid_workflow("mine")},
            )
            shots_job = client.post(
                "/api/jobs", json={"workflow_path": "Mine", "workspace": "shots"}
            ).json()
            wait_for_status(client, shots_job["id"], {"succeeded", "failed"})

            default_job = client.post(
                "/api/jobs", json={"workflow": valid_workflow("default")}
            ).json()
            wait_for_status(client, default_job["id"], {"succeeded", "failed"})

            scoped = client.get("/api/jobs?workspace=shots").json()["jobs"]
            assert [job["id"] for job in scoped] == [shots_job["id"]]

            everything = client.get("/api/jobs").json()["jobs"]
            ids = {job["id"] for job in everything}
            assert {shots_job["id"], default_job["id"]} <= ids

    def test_gallery_metadata_is_scoped_to_its_workspace(self, server, workspace_root):
        from PIL import Image

        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            shots_outputs = os.path.join(workspace_root.root, "shots", "outputs")

            Image.new("RGB", (4, 4)).save(
                os.path.join(workspace_root.outputs, "same-name.png")
            )
            Image.new("RGB", (4, 4)).save(os.path.join(shots_outputs, "same-name.png"))

            default_job = client.post(
                "/api/jobs", json={"workflow": valid_workflow("default")}
            ).json()
            wait_for_status(client, default_job["id"], {"succeeded", "failed"})

            client.put(
                "/api/workflows/Mine?workspace=shots",
                json={"workflow": valid_workflow("mine")},
            )
            shots_job = client.post(
                "/api/jobs", json={"workflow_path": "Mine", "workspace": "shots"}
            ).json()
            wait_for_status(client, shots_job["id"], {"succeeded", "failed"})

            # success_script's manifest names /out/a.png, not the fixture
            # image above - both files just need to exist so the metadata
            # route can serve them; only job_for_file's scoping is at stake.
            # Reuse the real manifest name so job linkage actually happens.
            Image.new("RGB", (4, 4)).save(os.path.join(workspace_root.outputs, "a.png"))
            Image.new("RGB", (4, 4)).save(os.path.join(shots_outputs, "a.png"))

            default_meta = client.get("/api/gallery/a.png/metadata").json()
            shots_meta = client.get(
                "/api/gallery/a.png/metadata?workspace=shots"
            ).json()
            assert default_meta["job"]["id"] == default_job["id"]
            assert shots_meta["job"]["id"] == shots_job["id"]


def test_a_server_without_a_workspace_root_has_one_workspace(tmp_path):
    """Individual directory overrides and no workspace: the server has the
    one workspace it was configured with, and says so rather than pretending
    it can make more."""
    (tmp_path / "workflows").mkdir()
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "outputs"),
        job_manager=JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        ),
    )
    with TestClient(app, base_url="http://localhost") as client:
        body = client.get("/api/workspaces").json()
        assert body["workspace_root"] is None
        assert [w["name"] for w in body["workspaces"]] == ["default"]
        assert client.post("/api/workspaces", json={"name": "shots"}).status_code == 409


def test_history_migrates_rows_that_predate_workspaces(tmp_path):
    """A jobs.sqlite written before workspaces existed gains the column, and
    the jobs already in it belong to the default workspace - history that
    cannot say where a job ran stops making sense once there are two."""
    import sqlite3

    from dw.server.jobs import JobHistory

    path = str(tmp_path / "jobs.sqlite")
    # the schema as it was: no workspace column
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
            " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
            " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
        )
        connection.execute(
            "INSERT INTO jobs (id, workflow, status) VALUES ('old', 'w', 'succeeded')"
        )

    JobHistory(path)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        stored = connection.execute(
            "SELECT workspace FROM jobs WHERE id='old'"
        ).fetchone()
    assert "workspace" in columns
    assert stored[0] == "default"


def test_the_listing_reports_each_workspace_s_disk_usage(server, workspace_root):
    """Every workspace in the listing carries roughly how much disk it holds.

    The shared prompt library counts once - against the default workspace,
    whose folder it is - rather than once per workspace, and a named
    workspace counts only what is under its own root.
    """
    from dw.workspace import forget_workspace_usage

    with server() as client:
        assert client.post("/api/workspaces", json={"name": "shots"}).status_code == 201
        with open(os.path.join(workspace_root.prompts, "scenic.json"), "w") as f:
            f.write("x" * 500)
        shots_assets = os.path.join(workspace_root.root, "shots", "assets")
        with open(os.path.join(shots_assets, "big.bin"), "wb") as f:
            f.write(b"0" * 4096)

        # the listing caches its walk for a minute, and the writes above
        # landed after the create already primed it
        forget_workspace_usage()
        spaces = {
            w["name"]: w["usage"]
            for w in client.get("/api/workspaces").json()["workspaces"]
        }

    assert spaces["shots"] == {"files": 1, "bytes": 4096}
    assert spaces["default"]["files"] == 1
    assert spaces["default"]["bytes"] == 500


class TestSharedAssets:
    """Assets are per workspace, which is right for the work that made them
    and wrong for a recurring cast: episode four, started in a fresh
    workspace, could not see the character portraits episode one uploaded
    (2026-09-11). The shared library is the prompt library's treatment
    applied to assets - one copy, reachable from every workspace."""

    def test_a_shared_upload_is_visible_from_every_workspace(
        self, server, workspace_root
    ):
        with server() as client:
            client.post("/api/workspaces", json={"name": "episode-four"})
            uploaded = client.post(
                "/api/uploads",
                params={
                    "filename": "priya.png",
                    "asset_name": "cast/priya",
                    "shared": "true",
                },
                content=b"png",
            )
            assert uploaded.status_code == 201
            body = uploaded.json()
            assert body["path"] == "asset:uploads/cast/priya.png"
            assert body["shared"] is True

            for workspace in ("", "?workspace=episode-four"):
                listed = client.get(f"/api/assets{workspace}").json()
                names = {asset["name"]: asset for asset in listed["assets"]}
                assert "uploads/cast/priya.png" in names, workspace
                assert names["uploads/cast/priya.png"]["origin"] == "common"

    def test_it_is_stored_once_at_the_root(self, server, workspace_root):
        with server() as client:
            client.post(
                "/api/uploads",
                params={"filename": "hal.png", "shared": "true"},
                content=b"png",
            )

        shared = os.path.join(workspace_root.root, "common", "assets", "uploads")
        assert len(os.listdir(shared)) == 1
        assert not os.path.exists(os.path.join(workspace_root.assets, "uploads"))

    def test_a_workspace_asset_shadows_a_shared_one_of_the_same_name(
        self, server, workspace_root
    ):
        """The same order 'asset:' resolves in - the workspace's own first."""
        with server() as client:
            client.post(
                "/api/uploads",
                params={
                    "filename": "hal.png",
                    "asset_name": "cast/hal",
                    "shared": "true",
                },
                content=b"shared-bytes",
            )
            client.post(
                "/api/uploads",
                params={"filename": "hal.png", "asset_name": "cast/hal"},
                content=b"workspace-bytes",
            )

            listed = client.get("/api/assets").json()["assets"]
            entries = [a for a in listed if a["name"] == "uploads/cast/hal.png"]
            assert len(entries) == 1
            assert entries[0]["origin"] == "workspace"

            served = client.get("/inputs/uploads/cast/hal.png")
            assert served.content == b"workspace-bytes"

    def test_a_shared_asset_previews_like_any_other(self, server, workspace_root):
        with server() as client:
            client.post("/api/workspaces", json={"name": "episode-four"})
            client.post(
                "/api/uploads",
                params={
                    "filename": "priya.png",
                    "asset_name": "cast/priya",
                    "shared": "true",
                },
                content=b"png-bytes",
            )
            listed = client.get("/api/assets?workspace=episode-four").json()
            url = listed["assets"][0]["url"]

            assert client.get(url).content == b"png-bytes"

    def test_an_output_can_be_kept_as_a_shared_asset(self, server, workspace_root):
        generated = os.path.join(workspace_root.outputs, "still-gen.0-0.0.png")
        with open(generated, "wb") as file:
            file.write(b"generated")

        with server() as client:
            client.post("/api/workspaces", json={"name": "episode-four"})
            kept = client.post(
                "/api/assets/keep",
                json={
                    "name": "still-gen.0-0.0.png",
                    "asset_name": "cast/priya.png",
                    "shared": True,
                },
            )
            assert kept.status_code == 201
            assert kept.json()["shared"] is True
            assert kept.json()["reference"] == "asset:cast/priya.png"

            listed = client.get("/api/assets?workspace=episode-four").json()
            assert [a["origin"] for a in listed["assets"]] == ["common"]

        assert os.path.isfile(
            os.path.join(workspace_root.root, "common", "assets", "cast", "priya.png")
        )

    def test_the_shared_library_is_not_a_workspace(self, server):
        """'common' holds one library, not workflows and outputs - naming it
        as a workspace has to be refused rather than making a folder."""
        with server() as client:
            refused = client.post("/api/workspaces", json={"name": "common"})
            assert refused.status_code == 400
            assert [
                w["name"] for w in client.get("/api/workspaces").json()["workspaces"]
            ] == ["default"]
