"""A symlink planted inside a workspace must not carry anything out of it.

Every root the server works in - outputs, assets, the shared asset library,
workflows, prompts, exports - is a directory a local user (or a mounted
volume, or an unpacked archive) can put a symlink into. `validate_path`
resolves symlinks before its containment check, so a path that is *checked*
is safe; the question this file asks is which code paths reach the disk
without being checked. For each root: read, write, delete and enumeration
through a link pointing at a sibling directory the server was never told
about, and the same for `download_output`'s destination over a mounted MCP
endpoint (gap 7 of the live suite's "Not covered here" list).

Everything is under `tmp_path`: "outside" is a sibling of the workspace, and
the "secret" is a marker string, so a failing boundary leaks a marker and
overwrites a scratch file.
"""

import io
import json
import os
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dw.security import TRUST_WORKFLOWS_ENV_VAR
from dw.server.app import create_app
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, success_script, valid_workflow

SECRET = "outside-the-roots-probe"

pytestmark = pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt",
    reason="symlinks need POSIX semantics",
)


def _png(path, color="red"):
    Image.new("RGB", (4, 4), color).save(path)
    return path


@pytest.fixture
def tree(tmp_path):
    """<tmp>/ws is the workspace root (default workspace), <tmp>/outside the
    directory nothing may reach. Each outside file carries SECRET."""
    root = tmp_path / "ws"
    paths = {
        "root": root,
        "workflows": root / "workflows",
        "outputs": root / "outputs",
        "assets": root / "assets",
        "prompts": root / "prompts",
        "common": root / "common" / "assets",
        "outside": tmp_path / "outside",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    outside = paths["outside"]
    _png(outside / "secret.png")
    (outside / "secret.txt").write_text(SECRET)
    (outside / "secret.json").write_text(
        json.dumps(
            {
                "id": "secret",
                "description": SECRET,
                "variables": {SECRET.replace("-", "_"): 1},
                "steps": [],
            }
        )
    )
    (outside / "prompt.json").write_text(json.dumps({"text": SECRET}))
    (outside / "victim.txt").write_text("untouched")
    (outside / "dir").mkdir()
    _png(outside / "dir" / "inner.png")
    (outside / "dir" / "inner.txt").write_text(SECRET)
    (paths["workflows"] / "Basic.json").write_text(json.dumps(valid_workflow("b")))
    return paths


def link(at, target):
    at.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, at)
    return at


@pytest.fixture
def client(tree, tmp_path, monkeypatch):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
    manager = JobManager(
        str(tree["outputs"]),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tree["workflows"]),
        output_dir=str(tree["outputs"]),
        job_manager=manager,
        prompt_dir=str(tree["prompts"]),
        asset_dir=str(tree["assets"]),
        workspace=str(tree["root"]),
    )
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client


def _leaks(response):
    return response.status_code < 300 and SECRET.encode() in response.content


def _outside_untouched(tree):
    outside = tree["outside"]
    assert (outside / "victim.txt").read_text() == "untouched"
    assert (outside / "secret.txt").read_text() == SECRET
    assert (outside / "dir" / "inner.txt").read_text() == SECRET
    assert (outside / "secret.png").exists()
    assert sorted(p.name for p in outside.iterdir()) == [
        "dir",
        "prompt.json",
        "secret.json",
        "secret.png",
        "secret.txt",
        "victim.txt",
    ], "something was written into the outside directory"


# ------------------------------------------------------------------ outputs


class TestOutputs:
    @pytest.fixture(autouse=True)
    def plant(self, tree):
        link(tree["outputs"] / "leak.png", tree["outside"] / "secret.png")
        link(tree["outputs"] / "leak.txt", tree["outside"] / "secret.txt")
        link(tree["outputs"] / "linked_run", tree["outside"] / "dir")

    @pytest.mark.parametrize(
        "path",
        [
            "/outputs/leak.txt",
            "/outputs/leak.png",
            "/outputs/linked_run/inner.txt",
            "/api/gallery/leak.txt/download",
            "/api/gallery/leak.png/download",
            "/api/gallery/linked_run/inner.txt/download",
            "/api/gallery/leak.png/metadata",
            "/api/gallery/leak.png/thumbnail",
            "/api/gallery/linked_run/inner.png/thumbnail",
        ],
    )
    def test_reads_do_not_follow_the_link(self, client, path):
        response = client.get(path)
        assert not _leaks(response), path
        assert response.status_code >= 400, (path, response.status_code)

    def test_the_archive_route_does_not_follow_the_link(self, client):
        response = client.post("/api/gallery/archive", json={"names": ["leak.txt"]})
        assert response.status_code >= 400
        assert not _leaks(response)

    @pytest.mark.xfail(
        strict=True,
        reason="_iter_gallery_files walks outputs with os.walk and os.stat, "
        "so GET /api/gallery lists a symlink pointing outside, with the "
        "target's size and mtime",
    )
    def test_the_gallery_listing_does_not_enumerate_the_link(self, client):
        names = [entry["name"] for entry in client.get("/api/gallery").json()["files"]]
        assert "leak.png" not in names

    def test_delete_removes_nothing_outside(self, client, tree):
        for name in ("leak.txt", "linked_run/inner.txt", "linked_run"):
            client.delete(f"/api/gallery/{name}")
        _outside_untouched(tree)

    def test_keep_output_does_not_copy_the_target_into_assets(self, client, tree):
        response = client.post(
            "/api/assets/keep", json={"name": "leak.png", "asset_name": "kept.png"}
        )
        assert response.status_code >= 400
        assert not (tree["assets"] / "kept.png").exists()

    def test_an_output_reference_does_not_follow_a_linked_run_directory(self, tree):
        from dw.runs import resolve_output_reference
        from dw.security import SecurityError

        with pytest.raises((SecurityError, ValueError)):
            resolve_output_reference(
                "output:linked_run/inner.png", root=str(tree["outputs"])
            )


# ------------------------------------------------------------------- assets


class TestAssets:
    @pytest.fixture(autouse=True)
    def plant(self, tree):
        link(tree["assets"] / "leak.png", tree["outside"] / "secret.png")
        link(tree["assets"] / "cast", tree["outside"] / "dir")
        link(tree["common"] / "shared-leak.png", tree["outside"] / "secret.png")
        link(tree["common"] / "shared-dir", tree["outside"] / "dir")

    @pytest.mark.parametrize(
        "path",
        [
            "/inputs/leak.png",
            "/inputs/cast/inner.txt",
            "/inputs/cast/inner.png",
            "/inputs/shared-leak.png",
            "/inputs/shared-dir/inner.txt",
        ],
    )
    def test_reads_do_not_follow_the_link(self, client, path):
        response = client.get(path)
        assert response.status_code >= 400, (path, response.status_code)

    def test_the_asset_archive_does_not_follow_the_link(self, client):
        for names in (["leak.png"], ["cast/inner.txt"], ["shared-dir/inner.txt"]):
            response = client.post("/api/assets/archive", json={"names": names})
            assert response.status_code >= 400, names
            assert not _leaks(response)

    @pytest.mark.parametrize(
        "reference",
        [
            "asset:leak.png",
            "asset:cast/inner.png",
            "asset:shared-leak.png",
            "asset:shared-dir/inner.png",
        ],
    )
    def test_an_asset_reference_does_not_follow_the_link(
        self, tree, monkeypatch, reference
    ):
        from dw.assets import resolve_asset_reference
        from dw.security import SecurityError

        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        monkeypatch.setenv("DW_ASSET_DIR", str(tree["assets"]))
        monkeypatch.setenv("DW_ASSET_PATH", str(tree["common"]))
        with pytest.raises((SecurityError, ValueError)):
            resolve_asset_reference(reference)

    @pytest.mark.xfail(
        strict=True,
        reason="the asset listing walks the library with os.walk/os.stat and "
        "lists a symlink that resolves outside it",
    )
    def test_the_asset_listing_does_not_enumerate_the_link(self, client):
        listing = client.get("/api/assets").json()
        assert "leak.png" not in json.dumps(listing)

    def test_an_upload_does_not_write_through_a_linked_name(self, client, tree):
        """Uploads land in <library>/uploads/, so that is where a link that
        could redirect one would be planted."""
        link(tree["assets"] / "uploads" / "victim.png", tree["outside"] / "victim.txt")
        response = client.post(
            "/api/uploads?filename=victim.png&asset_name=victim.png",
            content=b"\x89PNG overwritten",
        )
        assert response.status_code >= 400
        _outside_untouched(tree)

    def test_an_upload_does_not_write_into_a_linked_folder(self, client, tree):
        link(tree["assets"] / "uploads" / "cast", tree["outside"] / "dir")
        response = client.post(
            "/api/uploads?filename=new.png&asset_name=cast/new.png",
            content=b"\x89PNG new",
        )
        assert response.status_code >= 400
        assert not (tree["outside"] / "dir" / "new.png").exists()
        _outside_untouched(tree)

    def test_a_shared_upload_does_not_write_through_a_link_in_common(
        self, client, tree
    ):
        link(tree["common"] / "uploads" / "victim.png", tree["outside"] / "victim.txt")
        response = client.post(
            "/api/uploads?filename=victim.png&asset_name=victim.png&shared=true",
            content=b"\x89PNG overwritten",
        )
        assert response.status_code >= 400
        _outside_untouched(tree)

    def test_keep_output_does_not_write_through_a_linked_destination(
        self, client, tree
    ):
        _png(tree["outputs"] / "real.png", "blue")
        link(tree["assets"] / "dest.png", tree["outside"] / "victim.txt")
        for overwrite in (False, True):
            client.post(
                "/api/assets/keep",
                json={
                    "name": "real.png",
                    "asset_name": "dest.png",
                    "overwrite": overwrite,
                },
            )
            client.post(
                "/api/assets/keep",
                json={
                    "name": "real.png",
                    "asset_name": "cast/new.png",
                    "overwrite": overwrite,
                },
            )
        _outside_untouched(tree)

    def test_delete_removes_nothing_outside(self, client, tree):
        for name in ("leak.png", "cast/inner.png", "cast", "shared-leak.png"):
            client.delete(f"/api/assets/{name}")
        _outside_untouched(tree)


# ------------------------------------------------------ gather_images globs


class TestGlobs:
    """tests/test_locations.py drops a *file* match that escapes; a linked
    *directory* in the pattern's fixed part is refused before expansion."""

    @pytest.fixture(autouse=True)
    def untrusted(self, monkeypatch, tree):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        monkeypatch.setenv("DW_ASSET_DIR", str(tree["assets"]))
        link(tree["assets"] / "cast", tree["outside"] / "dir")
        link(tree["assets"] / "leak.png", tree["outside"] / "secret.png")
        _png(tree["assets"] / "own.png")

    def test_a_linked_directory_in_the_pattern_is_refused(self, tree):
        from dw.security import PathTraversalError
        from dw.tasks.gather import gather_images

        with pytest.raises(PathTraversalError):
            gather_images(glob=str(tree["assets"] / "cast" / "*.png"))

    def test_a_wildcard_does_not_descend_through_a_linked_directory(self, tree):
        """The only match is under the linked directory; dropped, it leaves
        nothing, which gather_images reports as no images."""
        from dw.tasks.gather import gather_images

        with pytest.raises(ValueError, match="No images found"):
            gather_images(glob=str(tree["assets"] / "*" / "*.png"))

    def test_a_linked_file_match_is_dropped(self, tree):
        from dw.tasks.gather import gather_images

        images = gather_images(glob=str(tree["assets"] / "*.png"))
        assert len(images) == 1

    def test_gather_videos_drops_a_linked_match(self, tree):
        from dw.tasks.gather import gather_videos

        (tree["outside"] / "clip.mp4").write_bytes(b"not really a video")
        link(tree["assets"] / "clip.mp4", tree["outside"] / "clip.mp4")
        # a decode error here would mean the linked file was opened
        with pytest.raises(ValueError, match="No videos found"):
            gather_videos(glob=str(tree["assets"] / "*.mp4"))


# ------------------------------------------------------- workflows, prompts


class TestWorkflows:
    @pytest.fixture(autouse=True)
    def plant(self, tree):
        link(tree["workflows"] / "leak.json", tree["outside"] / "secret.json")
        link(tree["workflows"] / "linked", tree["outside"])

    @pytest.mark.parametrize(
        "path",
        [
            "/api/workflows/leak",
            "/api/workflows/leak/download",
            "/api/workflows/leak/variables",
            "/api/workflows/linked/secret",
        ],
    )
    def test_reads_do_not_follow_the_link(self, client, path):
        response = client.get(path)
        assert not _leaks(response), path

    @pytest.mark.xfail(
        strict=True,
        reason="workflow_details opens every *.json os.walk finds without a "
        "containment check, so GET /api/workflows reads a linked file's "
        "description and variable names",
    )
    def test_the_listing_does_not_read_through_the_link(self, client):
        response = client.get("/api/workflows")
        assert SECRET not in response.text
        assert SECRET.replace("-", "_") not in response.text

    def test_a_save_does_not_write_through_the_link(self, client, tree):
        for name in ("leak", "linked/victim"):
            client.put(f"/api/workflows/{name}", json=valid_workflow("overwrite"))
        assert json.loads((tree["outside"] / "secret.json").read_text())["id"] == (
            "secret"
        )
        _outside_untouched(tree)

    def test_a_delete_removes_nothing_outside(self, client, tree):
        for name in ("leak", "linked/secret"):
            client.delete(f"/api/workflows/{name}")
        _outside_untouched(tree)


class TestPrompts:
    @pytest.fixture(autouse=True)
    def plant(self, tree):
        link(tree["prompts"] / "leak.json", tree["outside"] / "prompt.json")
        link(tree["prompts"] / "linked", tree["outside"])

    @pytest.mark.parametrize(
        "path", ["/api/prompts/leak", "/api/prompts/leak/download"]
    )
    def test_reads_do_not_follow_the_link(self, client, path):
        assert not _leaks(client.get(path)), path

    @pytest.mark.xfail(
        strict=True,
        reason="list_prompts hands every *.json workflow_names finds to "
        "prompt_details, which opens it without a containment check - the "
        "listing carries a linked file's text",
    )
    def test_the_listing_does_not_read_through_the_link(self, client):
        assert SECRET not in client.get("/api/prompts").text

    def test_a_prompt_reference_does_not_follow_the_link(self, tree, monkeypatch):
        from dw.prompts import fetch_prompt
        from dw.security import SecurityError

        for reference in ("prompt:leak", "prompt:linked/prompt"):
            with pytest.raises((SecurityError, ValueError)):
                fetch_prompt(reference, prompt_dir=str(tree["prompts"]))

    def test_a_save_does_not_write_through_the_link(self, client, tree):
        for name in ("leak", "linked/victim"):
            client.put(f"/api/prompts/{name}", json={"prompt": {"text": "overwrite"}})
        assert json.loads((tree["outside"] / "prompt.json").read_text()) == {
            "text": SECRET
        }
        _outside_untouched(tree)

    def test_a_delete_removes_nothing_outside(self, client, tree):
        for name in ("leak", "linked/prompt"):
            client.delete(f"/api/prompts/{name}")
        assert (tree["outside"] / "prompt.json").exists()
        _outside_untouched(tree)


# -------------------------------------------------------- exports, workspaces


class TestExportsAndWorkspaces:
    @pytest.mark.xfail(
        strict=True,
        reason="GET /exports/<job>.zip (no token) zips the export directory "
        "with os.walk + ZipFile.write, which follows a planted file symlink "
        "and archives the target's bytes",
    )
    def test_the_export_zip_does_not_follow_a_planted_link(self, client, tree):
        export = tree["root"] / "exports" / "job-1"
        export.mkdir(parents=True)
        (export / "README.md").write_text("an export")
        link(export / "leak.txt", tree["outside"] / "secret.txt")

        response = client.get("/exports/job-1.zip")
        assert response.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        for name in archive.namelist():
            assert SECRET.encode() not in archive.read(name), name

    def test_the_export_zip_does_not_follow_a_linked_export_directory(
        self, client, tree
    ):
        link(tree["root"] / "exports" / "job-2", tree["outside"] / "dir")
        response = client.get("/exports/job-2.zip")
        assert response.status_code == 404
        assert SECRET.encode() not in response.content

    def test_deleting_a_workspace_does_not_follow_a_link_inside_it(self, client, tree):
        assert (
            client.post("/api/workspaces", json={"name": "doomed"}).status_code == 201
        )
        doomed = tree["root"] / "doomed"
        link(doomed / "outputs" / "out", tree["outside"])
        link(doomed / "assets" / "cast", tree["outside"] / "dir")
        client.delete("/api/workspaces/doomed?acknowledged=true")
        _outside_untouched(tree)

    def test_the_workspace_size_count_does_not_follow_a_link(self, client, tree):
        assert client.post("/api/workspaces", json={"name": "sized"}).status_code == 201
        link(tree["root"] / "sized" / "outputs" / "big", tree["outside"])
        response = client.delete("/api/workspaces/sized")
        contents = response.json()["detail"]["contents"]
        assert all(entry.get("files", 0) == 0 for entry in contents.values()), contents


# ------------------------------------------------------ download_output (7)


def _mounted(workspace_root, body=b"downloaded-bytes"):
    """A DwClient shaped like the one dw.serve builds for its own /mcp."""
    from dw_mcp.client import DwClient

    def handler(request):
        if request.url.path == "/api/server":
            return httpx.Response(
                200, json={"directories": {"workspace": str(workspace_root)}}
            )
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    client = DwClient(transport=httpx.MockTransport(handler))
    client.mounted = True
    return client


class TestDownloadOutputDestination:
    """Over a mounted endpoint the file lands on the server, so the
    destination is confined to the workspace (#113). A legal relative
    destination writes inside it and nowhere else."""

    @pytest.fixture
    def workspace(self, tree):
        return tree["root"]

    def _download(self, workspace, destination, overwrite=False):
        from dw_mcp.media import download_output

        return download_output(
            _mounted(workspace),
            "run/probe.png",
            destination=destination,
            overwrite=overwrite,
        )

    @pytest.mark.parametrize(
        "destination",
        ["kept/probe.png", "probe.png", "kept/", "deep/er/still/probe.png"],
    )
    def test_a_relative_destination_lands_inside(self, workspace, tree, destination):
        before = {p for p in tree["outside"].rglob("*")}
        result = self._download(workspace, destination)
        saved = os.path.realpath(result["saved_to"])
        assert saved.startswith(os.path.realpath(workspace) + os.sep)
        assert open(saved, "rb").read() == b"downloaded-bytes"
        assert {p for p in tree["outside"].rglob("*")} == before

    @pytest.mark.parametrize(
        "destination",
        [
            "../outside/escaped.png",
            "kept/../../outside/escaped.png",
            "OUTSIDE_ABSOLUTE",
            "~/escaped.png",
        ],
    )
    def test_dot_dot_absolute_and_home_are_refused(
        self, workspace, tree, monkeypatch, destination
    ):
        from dw_mcp.client import DwApiError

        monkeypatch.setenv("HOME", str(tree["outside"]))
        if destination == "OUTSIDE_ABSOLUTE":
            destination = str(tree["outside"] / "escaped.png")
        with pytest.raises(DwApiError):
            self._download(workspace, destination)
        assert not (tree["outside"] / "escaped.png").exists()
        _outside_untouched(tree)

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_a_linked_parent_is_refused(self, workspace, tree, overwrite):
        from dw_mcp.client import DwApiError

        link(workspace / "kept", tree["outside"])
        with pytest.raises(DwApiError):
            self._download(workspace, "kept/escaped.png", overwrite=overwrite)
        with pytest.raises(DwApiError):
            self._download(workspace, "kept/victim.txt", overwrite=overwrite)
        _outside_untouched(tree)

    def test_a_linked_grandparent_is_refused_even_when_the_parent_is_missing(
        self, workspace, tree
    ):
        """The parent does not exist yet, so realpath of the destination
        alone would stop at the missing segment - the check has to resolve
        the nearest existing ancestor, which is the link."""
        from dw_mcp.client import DwApiError

        link(workspace / "kept", tree["outside"])
        with pytest.raises(DwApiError):
            self._download(workspace, "kept/new/deeper/escaped.png")
        assert not (tree["outside"] / "new").exists()
        _outside_untouched(tree)

    def test_overwrite_cannot_clobber_a_linked_file(self, workspace, tree):
        from dw_mcp.client import DwApiError

        link(workspace / "victim.png", tree["outside"] / "victim.txt")
        with pytest.raises(DwApiError):
            self._download(workspace, "victim.png", overwrite=True)
        _outside_untouched(tree)

    def test_a_dangling_link_is_replaced_not_followed(self, workspace, tree):
        """A link to a file that does not exist yet: writing through it
        would create the file outside. The write must replace the link (or
        refuse), never create its target."""
        target = tree["outside"] / "created-by-download.png"
        link(workspace / "dangling.png", target)
        try:
            self._download(workspace, "dangling.png", overwrite=True)
        except Exception:
            pass
        assert not target.exists()
        _outside_untouched(tree)

    def test_a_workspace_root_that_is_itself_a_link_still_confines(
        self, tree, tmp_path
    ):
        """The operator's workspace may be a symlink (a data volume); the
        root is resolved, and a destination is confined to what it resolves
        to - not to the link's own parent."""
        from dw_mcp.client import DwApiError

        linked_root = link(tmp_path / "ws-link", tree["root"])
        result = self._download(linked_root, "kept/probe.png")
        assert os.path.realpath(result["saved_to"]).startswith(
            os.path.realpath(tree["root"]) + os.sep
        )
        with pytest.raises(DwApiError):
            self._download(linked_root, str(tmp_path / "outside" / "escaped.png"))
        _outside_untouched(tree)
