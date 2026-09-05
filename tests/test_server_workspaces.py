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
