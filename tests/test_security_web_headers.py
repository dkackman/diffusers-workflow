"""What the browser is told about each response, and what the UI's origin
can be made to serve.

The web UI keeps the API token in localStorage, and the server serves the
UI, generated outputs (`/outputs`, no token) and input media (`/inputs`, no
token) from one origin. So the question for the HTTP layer is not only who
may call the API but what a browser will *execute* when it loads something
from that origin - and whether a page elsewhere can frame or read it.

ui/e2e/security.spec.ts drives the same boundary through a real browser;
these tests pin the headers and content types that decide it, fast and
without one.
"""

import json

import pytest
from fastapi.testclient import TestClient

from dw.security import TRUST_WORKFLOWS_ENV_VAR
from dw.server.app import create_app
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, success_script

# Content types a browser renders as a document, where script runs
ACTIVE_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/xml",
    "application/xml",
    "image/svg+xml",
)
SCRIPT = "<script>document.title=localStorage.getItem('dw-api-token')</script>"


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "ws"
    for sub in ("workflows", "outputs", "assets", "prompts"):
        (root / sub).mkdir(parents=True)
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>dw</title>")
    (root / "outputs" / "run.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "outputs" / "note.txt").write_text(SCRIPT)
    return {"root": root, "ui": ui, "tmp": tmp_path}


@pytest.fixture
def client(tree, monkeypatch):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
    root = tree["root"]
    manager = JobManager(
        str(root / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tree["tmp"] / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(root / "workflows"),
        output_dir=str(root / "outputs"),
        job_manager=manager,
        prompt_dir=str(root / "prompts"),
        asset_dir=str(root / "assets"),
        workspace=str(root),
        ui_dir=str(tree["ui"]),
    )
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client


def _page(content_type):
    return {
        "id": "page",
        "steps": [
            {
                "name": "t",
                "task": {"command": "compose_text", "arguments": {"parts": [SCRIPT]}},
                "result": {"content_type": content_type},
            }
        ],
    }


def _document_is_inert(response):
    """A response a browser will not execute as a same-origin document:
    not an active type, or forced to download, or sandboxed by CSP."""
    content_type = response.headers.get("content-type", "").split(";")[0].strip()
    disposition = response.headers.get("content-disposition", "")
    csp = response.headers.get("content-security-policy", "")
    return (
        content_type not in ACTIVE_TYPES
        or disposition.startswith("attachment")
        or "sandbox" in csp
    )


# ------------------------------------------------ active content in outputs


class TestActiveOutputs:
    """A workflow - which an MCP agent may author - could write an .html or
    .xml output, and /outputs needs no token and shares the UI's origin.
    Validation and the writer refuse those two result types (dw/
    content_types.py); a planted file still reaches /outputs, so it is
    served sandboxed."""

    @pytest.mark.parametrize(
        "content_type", ["text/html", "text/xml", "Text/HTML; charset=utf-8"]
    )
    def test_validation_refuses_it_at_its_path(self, monkeypatch, content_type):
        from dw.workflow import Workflow

        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        errors = Workflow(_page(content_type), "", "").validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].result.content_type"]
        assert content_type in errors[0]["message"]

    @pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
    def test_an_inert_text_type_still_validates(self, monkeypatch, content_type):
        from dw.workflow import Workflow

        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        assert Workflow(_page(content_type), "", "").validation_errors() == []

    @pytest.mark.parametrize("content_type", ["text/html", "text/xml"])
    def test_the_writer_refuses_it_without_validation(
        self, tmp_path, monkeypatch, content_type
    ):
        """A run that skipped validation still writes nothing active."""
        from dw.security import InvalidInputError
        from dw.workflow import Workflow

        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        workflow = Workflow(_page(content_type), str(tmp_path / "out"), "")
        with pytest.raises(InvalidInputError, match=content_type):
            workflow.run({})
        assert not [
            p for p in (tmp_path / "out").rglob("*") if p.suffix in (".html", ".xml")
        ]

    def test_the_api_refuses_to_queue_it(self, client):
        response = client.post("/api/jobs", json={"workflow": _page("text/html")})
        assert response.status_code == 400, response.text
        assert "text/html" in response.text

    @pytest.mark.parametrize(
        "name", ["page.html", "page.xhtml", "page.xml", "page.svg"]
    )
    def test_outputs_does_not_serve_it_as_a_live_document(self, client, tree, name):
        (tree["root"] / "outputs" / name).write_text(SCRIPT)
        response = client.get(f"/outputs/{name}")
        assert response.status_code == 200
        assert _document_is_inert(response), response.headers

    def test_the_sandbox_keeps_range_and_etag(self, client, tree):
        (tree["root"] / "outputs" / "page.html").write_text(SCRIPT)
        ranged = client.get("/outputs/page.html", headers={"Range": "bytes=0-3"})
        assert ranged.status_code == 206
        assert ranged.headers["content-security-policy"] == "sandbox"
        etag = client.get("/outputs/page.html").headers["etag"]
        cached = client.get("/outputs/page.html", headers={"If-None-Match": etag})
        assert cached.status_code == 304

    def test_an_inert_output_is_not_sandboxed(self, client):
        for path in ("/outputs/run.png", "/outputs/note.txt"):
            assert "content-security-policy" not in client.get(path).headers

    def test_a_text_output_is_served_as_plain_text(self, client):
        response = client.get("/outputs/note.txt")
        assert response.headers["content-type"].startswith("text/plain")

    @pytest.mark.parametrize(
        "name", ["page.html", "page.svg", "page.xml", "page.xhtml"]
    )
    def test_an_upload_cannot_plant_one(self, client, name):
        response = client.post(f"/api/uploads?filename={name}", content=SCRIPT.encode())
        assert response.status_code == 400

    def test_keep_output_cannot_carry_one_into_assets(self, client, tree):
        (tree["root"] / "outputs" / "page.html").write_text(SCRIPT)
        response = client.post(
            "/api/assets/keep", json={"name": "page.html", "asset_name": "cast.html"}
        )
        if response.status_code < 400:
            assert _document_is_inert(client.get("/inputs/cast.html"))

    def test_keep_output_cannot_rename_one_to_svg(self, client, tree):
        (tree["root"] / "outputs" / "page.html").write_text(SCRIPT)
        response = client.post(
            "/api/assets/keep", json={"name": "page.html", "asset_name": "cast.svg"}
        )
        assert response.status_code == 400
        assert not (tree["root"] / "assets" / "cast.svg").exists()


# ------------------------------------------------------------ the headers

UI_AND_MEDIA = ["/", "/index.html", "/outputs/run.png", "/outputs/note.txt"]


class TestBrowserHeaders:
    @pytest.mark.xfail(
        strict=True,
        reason="deferred by #407: no Content-Security-Policy on the UI, so "
        "nothing limits what a script injected into the page may load or "
        "where it may send the token",
    )
    def test_the_ui_carries_a_content_security_policy(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "script-src" in response.headers.get(
            "content-security-policy", ""
        ) or "default-src" in response.headers.get("content-security-policy", "")

    def test_the_ui_cannot_be_framed(self, client):
        response = client.get("/")
        frame_options = response.headers.get("x-frame-options", "").upper()
        csp = response.headers.get("content-security-policy", "")
        assert frame_options in ("DENY", "SAMEORIGIN") or "frame-ancestors" in csp

    @pytest.mark.parametrize("path", UI_AND_MEDIA)
    def test_nosniff(self, client, path):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_the_api_answers_json_as_json(self, client):
        """The one thing that keeps an API body from being rendered as a
        page today: its declared type."""
        response = client.get("/api/health")
        assert response.headers["content-type"].startswith("application/json")


# ------------------------------------------------------------------ CORS


class TestCrossOriginReads:
    EVIL = "https://evil.example"

    @pytest.mark.parametrize(
        "path", ["/api/health", "/api/workflows", "/outputs/run.png", "/"]
    )
    def test_no_response_grants_a_foreign_origin_read_access(self, client, path):
        response = client.get(path, headers={"Origin": self.EVIL})
        assert "access-control-allow-origin" not in response.headers
        assert "access-control-allow-credentials" not in response.headers

    def test_a_preflight_from_a_foreign_origin_is_refused(self, client):
        response = client.options(
            "/api/jobs",
            headers={
                "Origin": self.EVIL,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert response.status_code >= 400
        assert "access-control-allow-origin" not in response.headers

    def test_a_same_origin_page_is_not_refused(self, client):
        response = client.get("/api/health", headers={"Origin": "http://localhost"})
        assert response.status_code == 200


# ------------------------------------------------- download file names


class TestDownloadNames:
    """A file name on disk ends up in a Content-Disposition header. It must
    not be able to add a header or break out of the quoted value."""

    @pytest.mark.parametrize(
        "name",
        [
            'quote"; filename="evil.html.png',
            "semi;colon.png",
            "unicode-‮txt.png",
            "crlf\r\nX-Injected: 1.png",
            "lf\nSet-Cookie: dw=1.png",
        ],
    )
    def test_a_hostile_name_cannot_inject_a_header(self, client, tree, name):
        try:
            (tree["root"] / "outputs" / name).write_bytes(b"\x89PNG")
        except OSError:
            pytest.skip("this filesystem refuses the name")
        from urllib.parse import quote

        response = client.get(f"/api/gallery/{quote(name)}/download")
        assert "x-injected" not in response.headers
        assert "set-cookie" not in response.headers
        disposition = response.headers.get("content-disposition", "")
        assert "\r" not in disposition and "\n" not in disposition
        # the name, however spelled, is one parameter, not a second filename=
        assert disposition.count("filename=") <= 1

    def test_the_gallery_listing_reports_a_hostile_name_as_data(self, client, tree):
        """Escaping is the UI's job (ui/e2e/security.spec.ts); the API's is
        to report the name exactly, inside JSON, so nothing is pre-rendered."""
        name = "x\"'><img src=x onerror=alert(1)>.png"
        (tree["root"] / "outputs" / name).write_bytes(b"\x89PNG")
        response = client.get("/api/gallery")
        assert response.headers["content-type"].startswith("application/json")
        names = [entry["name"] for entry in response.json()["files"]]
        assert name in names
        json.dumps(names)  # round-trips as plain data
