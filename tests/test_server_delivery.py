"""How the server delivers bytes: gzip where it helps, caching for the
built UI, and nothing compressed where compression would break the
transport (event streams, ranges, media, zips, /mcp). Transport only:
every gzipped body decodes to the body the client gets without it."""

import json

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.trust import TRUST_WORKFLOWS_ENV_VAR

from .test_server import (
    ScriptedWorkerManager,
    success_script,
    valid_workflow,
    wait_for_status,
)
from .test_server_mcp import INITIALIZE, MCP_HEADERS

GZIP = {"Accept-Encoding": "gzip"}
IDENTITY = {"Accept-Encoding": "identity"}
ASSET = "assets/app-abc123.js"


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "ws"
    for sub in ("workflows", "outputs", "assets", "prompts"):
        (root / sub).mkdir(parents=True)
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>dw</title>")
    (ui / ASSET).write_text("console.log('dw');\n" * 280)  # ~5 KB
    (tmp_path / "secret.txt").write_text("beside the ui dir")
    (root / "outputs" / "note.txt").write_text("0123456789" * 400)
    (root / "outputs" / "run.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 3000)
    export = root / "exports" / "job1"
    export.mkdir(parents=True)
    (export / "a.json").write_text(json.dumps({"k": "v" * 3000}))
    return {"root": root, "ui": ui, "tmp": tmp_path}


def build(tree, mcp=False):
    root = tree["root"]
    manager = JobManager(
        str(root / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tree["tmp"] / "jobs.sqlite"),
    )
    return create_app(
        workflow_dir=str(root / "workflows"),
        output_dir=str(root / "outputs"),
        job_manager=manager,
        prompt_dir=str(root / "prompts"),
        asset_dir=str(root / "assets"),
        workspace=str(root),
        ui_dir=str(tree["ui"]),
        mcp=mcp,
    )


@pytest.fixture
def client(tree, monkeypatch):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
    with TestClient(build(tree), base_url="http://localhost") as test_client:
        yield test_client


def test_json_is_gzipped_when_asked(client, tree):
    for index in range(40):
        (tree["root"] / "workflows" / f"w{index}.json").write_text(
            json.dumps(valid_workflow(f"w{index}"))
        )
    response = client.get("/api/workflows", headers=GZIP)
    assert response.headers["content-encoding"] == "gzip"
    assert "accept-encoding" in response.headers["vary"].lower()
    assert (
        "content-encoding" not in client.get("/api/workflows", headers=IDENTITY).headers
    )


@pytest.mark.parametrize("path", ["/api/jobs", "/api/server", "/api/workflows"])
def test_a_gzipped_body_is_todays_body(client, tree, path):
    for index in range(40):
        (tree["root"] / "workflows" / f"w{index}.json").write_text(
            json.dumps(valid_workflow(f"w{index}"))
        )
    plain = client.get(path, headers=IDENTITY)
    packed = client.get(path, headers=GZIP)
    assert plain.status_code == packed.status_code == 200
    assert plain.content == packed.content


def test_assets_are_immutable(client):
    response = client.get(f"/{ASSET}")
    assert response.status_code == 200
    cache = response.headers["cache-control"]
    assert cache == "public, max-age=31536000, immutable"
    assert client.get(f"/{ASSET}", headers=GZIP).headers["content-encoding"] == "gzip"


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_index_html_revalidates(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert "immutable" not in response.headers["cache-control"]


def test_event_stream_is_not_compressed(client):
    job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
    wait_for_status(client, job["id"], {"succeeded", "failed"})
    with client.stream("GET", f"/api/jobs/{job['id']}/events", headers=GZIP) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "content-encoding" not in r.headers
        assert "data: " in "".join(r.iter_lines())


def test_a_range_request_is_not_compressed(client):
    response = client.get("/outputs/note.txt", headers={**GZIP, "Range": "bytes=0-3"})
    assert response.status_code == 206
    assert response.content == b"0123"
    assert "content-encoding" not in response.headers
    # a ranged text file well past the minimum size is still left alone
    big = client.get("/outputs/note.txt", headers={**GZIP, "Range": "bytes=0-2999"})
    assert big.status_code == 206 and len(big.content) == 3000
    assert "content-encoding" not in big.headers


def test_media_and_zip_are_not_compressed(client):
    png = client.get("/outputs/run.png", headers=GZIP)
    assert png.status_code == 200 and "content-encoding" not in png.headers
    zipped = client.get("/exports/job1.zip", headers=GZIP)
    assert zipped.status_code == 200
    assert zipped.content[:2] == b"PK"
    assert "content-encoding" not in zipped.headers


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_mcp_is_never_compressed(tree, tmp_path, path):
    pytest.importorskip("mcp")
    with TestClient(build(tree, mcp=True), base_url="http://localhost") as c:
        response = c.post(
            path,
            json=INITIALIZE,
            headers={**MCP_HEADERS, **GZIP},
            follow_redirects=True,
        )
        assert response.status_code == 200, response.text
        assert "content-encoding" not in response.headers
        # while the API beside it is still compressed
        assert c.get("/api/server", headers=GZIP).headers["content-encoding"] == "gzip"


@pytest.mark.parametrize(
    "path", ["/assets/../../secret.txt", "/assets/%2e%2e/%2e%2e/secret.txt"]
)
def test_the_ui_mount_serves_nothing_outside_its_directory(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert "beside the ui dir" not in response.text


def test_api_answers_keep_their_cache_headers(client):
    response = client.get("/api/jobs")
    assert "immutable" not in response.headers.get("cache-control", "")


def test_a_small_body_is_not_compressed(client):
    response = client.get("/api/jobs", headers=GZIP)
    assert response.status_code == 200
    assert len(response.content) < 1000
    assert "content-encoding" not in response.headers
    assert response.headers["content-length"] == str(len(response.content))


def test_a_large_body_is_compressed_and_still_secured(client, tree):
    for index in range(40):
        (tree["root"] / "workflows" / f"w{index}.json").write_text(
            json.dumps(valid_workflow(f"w{index}"))
        )
    response = client.get("/api/workflows", headers=GZIP)
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_an_uncompressed_download_keeps_its_length(client):
    response = client.get("/outputs/run.png", headers=GZIP)
    assert "content-encoding" not in response.headers
    assert response.headers["content-length"] == str(len(response.content))
