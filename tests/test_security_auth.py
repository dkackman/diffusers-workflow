"""The HTTP server's own boundary: token, Origin/Host, the ungated routes.

tests/test_server.py covers the happy shapes - a missing or wrong token is a
401, a foreign Origin a 403, a foreign Host a 400, the query-token allowance
is per route - and tests/test_serve_main.py the `--mcp` refusal on 0.0.0.0.
This file is the adversarial side of the same boundary: path spellings that
might slip past a prefix check, credentials in the wrong place, Origin and
Host values built to fool a hostname comparison, and the routes that are
ungated *on purpose* (`/outputs`, `/inputs`, `/exports`), pinned to serving
their own roots and nothing else.

Every probe target is under `tmp_path`; "outside" means a sibling directory
the server was never told about.
"""

import json

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, success_script, valid_workflow

TOKEN = "s3cr3t-token"
SECRET = "outside-the-roots-probe"


@pytest.fixture
def layout(tmp_path):
    """A workspace-shaped tree plus a sibling directory holding a secret."""
    paths = {
        "workflows": tmp_path / "ws" / "workflows",
        "outputs": tmp_path / "ws" / "outputs",
        "assets": tmp_path / "ws" / "assets",
        "prompts": tmp_path / "ws" / "prompts",
        "outside": tmp_path / "outside",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    (paths["workflows"] / "Basic.json").write_text(json.dumps(valid_workflow("b")))
    (paths["outputs"] / "run.png").write_bytes(b"generated")
    (paths["assets"] / "iris.png").write_bytes(b"asset")
    (paths["outside"] / "secret.txt").write_text(SECRET)
    (paths["outside"] / "secret.png").write_text(SECRET)
    paths["root"] = tmp_path / "ws"
    return paths


@pytest.fixture
def make_client(layout, tmp_path):
    def make(token=None, host="127.0.0.1", base_url="http://localhost"):
        manager = JobManager(
            str(layout["outputs"]),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        app = create_app(
            workflow_dir=str(layout["workflows"]),
            output_dir=str(layout["outputs"]),
            job_manager=manager,
            prompt_dir=str(layout["prompts"]),
            asset_dir=str(layout["assets"]),
            workspace=str(layout["root"]),
            token=token,
            host=host,
        )
        return TestClient(app, base_url=base_url)

    return make


def _jobs(client, headers=None):
    listing = client.get("/api/jobs", headers=headers or {}).json()
    return listing.get("jobs", listing) if isinstance(listing, dict) else listing


def _leaks(response):
    return response.status_code == 200 and SECRET in response.text


# ---------------------------------------------------------------- the token


class TestTheTokenGate:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/health",
            "/api/health/",
            "/api/workflows",
            "/api/jobs",
            "/api/server",
            "/api/gallery",
            "/api/prompts",
            "/api/assets",
            "/%61pi/health",
            "/api/%68ealth",
            "/api/./health",
            "/mcp",
            "/mcp/",
        ],
    )
    def test_every_api_spelling_needs_the_token(self, make_client, path):
        with make_client(token=TOKEN) as client:
            response = client.get(path)
        assert response.status_code == 401, (path, response.status_code)

    @pytest.mark.parametrize("path", ["//api/health", "/API/health", "/api"])
    def test_a_spelling_that_misses_the_prefix_reaches_no_api_route(
        self, make_client, path
    ):
        """If the gate's prefix check misses a spelling, the router must miss
        it too - otherwise an ungated spelling of a gated route exists."""
        with make_client(token=TOKEN) as client:
            response = client.get(path)
        assert response.status_code == 404, (path, response.status_code)

    @pytest.mark.parametrize(
        "authorization",
        [
            "",
            "Bearer",
            "Bearer ",
            f"Basic {TOKEN}",
            f"Token {TOKEN}",
            f"{TOKEN}",
            f"Bearer {TOKEN}x",
            f"Bearer x{TOKEN}",
            f"Bearer {TOKEN[:-1]}",
            f"Bearer {TOKEN.upper()}",
            f"Bearer {TOKEN}\x00",
        ],
    )
    def test_anything_but_the_exact_bearer_token_is_a_401(
        self, make_client, authorization
    ):
        with make_client(token=TOKEN) as client:
            try:
                response = client.get(
                    "/api/health", headers={"Authorization": authorization}
                )
            except Exception:
                # httpx refuses to send some header values at all - nothing
                # reached the server, which is the outcome the test wants
                return
        assert response.status_code == 401

    def test_the_scheme_is_case_insensitive_but_the_token_is_not(self, make_client):
        with make_client(token=TOKEN) as client:
            assert (
                client.get(
                    "/api/health", headers={"Authorization": f"bearer {TOKEN}"}
                ).status_code
                == 200
            )

    @pytest.mark.parametrize(
        "method, path",
        [
            ("GET", "/api/health"),
            ("GET", "/api/workflows"),
            ("GET", "/api/jobs"),
            ("POST", "/api/validate"),
            ("DELETE", "/api/prompts/x"),
        ],
    )
    def test_the_query_token_is_refused_where_it_is_not_allowed(
        self, make_client, method, path
    ):
        """?token= exists for <img>/EventSource GETs on marked routes only; a
        route that is not marked must not accept it."""
        with make_client(token=TOKEN) as client:
            response = client.request(method, f"{path}?token={TOKEN}", json={})
        assert response.status_code == 401

    def test_the_query_token_is_get_only_even_on_a_marked_route(
        self, make_client, layout
    ):
        with make_client(token=TOKEN) as client:
            assert (
                client.get(f"/api/gallery/run.png/download?token={TOKEN}").status_code
                == 200
            )
            assert (
                client.delete(f"/api/gallery/run.png?token={TOKEN}").status_code == 401
            )
        assert (layout["outputs"] / "run.png").exists()

    def test_a_valid_token_does_not_excuse_a_foreign_origin(self, make_client):
        """The Origin check is not an auth check the token can satisfy: a
        page that somehow holds the token still may not drive the API."""
        with make_client(token=TOKEN) as client:
            response = client.post(
                "/api/jobs",
                json={"workflow": valid_workflow()},
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Origin": "https://evil.example",
                },
            )
        assert response.status_code == 403

    def test_a_refused_request_queues_nothing(self, make_client):
        with make_client(token=TOKEN) as client:
            client.post("/api/jobs", json={"workflow": valid_workflow()})
            assert _jobs(client, {"Authorization": f"Bearer {TOKEN}"}) == []


# ---------------------------------------------------------- Origin and Host


class TestOriginAndHostSpoofing:
    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost@evil.example",
            "http://localhost:8765@evil.example",
            "http://localhost.evil.example",
            "http://127.0.0.1.evil.example",
            "http://evil.example#localhost",
            "http://evil.example/localhost",
            "http://evil.example?localhost",
            "http://evil-localhost",
            "null",
            "file://",
        ],
    )
    def test_a_lookalike_origin_is_refused(self, make_client, origin):
        with make_client() as client:
            response = client.post(
                "/api/jobs",
                json={"workflow": valid_workflow()},
                headers={"Origin": origin},
            )
        assert response.status_code == 403, origin

    @pytest.mark.parametrize(
        "origin", ["http://[::1].evil.example", "http://[localhost]", "http://["]
    )
    def test_an_unparseable_origin_is_not_processed(self, make_client, origin):
        """urlparse raises on these; whatever the middleware answers, the
        request must not reach a route."""
        app = make_client().app
        with TestClient(
            app, base_url="http://localhost", raise_server_exceptions=False
        ) as client:
            response = client.post(
                "/api/jobs",
                json={"workflow": valid_workflow()},
                headers={"Origin": origin},
            )
            assert response.status_code >= 400
            assert _jobs(client) == []

    @pytest.mark.xfail(
        strict=True,
        reason="reject_foreign_origins calls urlparse outside a try, so a "
        "bracketed non-IPv6 Origin is a 500 from an unhandled ValueError "
        "rather than the 403 every other refused Origin gets",
    )
    def test_an_unparseable_origin_is_a_403_not_a_500(self, make_client):
        app = make_client().app
        with TestClient(
            app, base_url="http://localhost", raise_server_exceptions=False
        ) as client:
            response = client.post(
                "/api/jobs",
                json={"workflow": valid_workflow()},
                headers={"Origin": "http://[::1].evil.example"},
            )
        assert response.status_code == 403

    @pytest.mark.parametrize(
        "host",
        [
            "localhost.evil.example",
            "127.0.0.1.evil.example",
            "evil.example",
            "evil.example:8765",
            "127.0.0.2",
            "0.0.0.0",
        ],
    )
    def test_a_lookalike_host_is_refused_on_a_loopback_bind(self, make_client, host):
        with make_client() as client:
            response = client.get("/api/health", headers={"Host": host})
        assert response.status_code == 400, host

    @pytest.mark.parametrize("path", ["/outputs/run.png", "/inputs/iris.png"])
    def test_the_host_check_covers_the_ungated_routes_too(self, make_client, path):
        """DNS rebinding reads through whatever route answers: the static
        routes need the Host check as much as /api does."""
        with make_client() as client:
            assert client.get(path).status_code == 200
            assert (
                client.get(path, headers={"Host": "rebind.evil.example"}).status_code
                == 400
            )


# ------------------------------------------------ the ungated static routes


class TestUngatedRoutesServeOnlyTheirRoots:
    """/outputs, /inputs and /exports take no token by design - an <img> or
    a download link cannot attach one. What they serve is therefore exactly
    what anyone who can reach the port can read, so pin it: the workspace's
    outputs, its asset search path, and finished exports, never a byte from
    anywhere else."""

    def test_what_they_serve_without_a_token(self, make_client):
        with make_client(token=TOKEN) as client:
            assert client.get("/outputs/run.png").content == b"generated"
            assert client.get("/inputs/iris.png").content == b"asset"
            assert client.get("/api/gallery").status_code == 401

    @pytest.mark.parametrize(
        "path",
        [
            "/outputs/../outside/secret.txt",
            "/outputs/..%2foutside/secret.txt",
            "/outputs/%2e%2e/outside/secret.txt",
            "/outputs/%2e%2e%2foutside%2fsecret.txt",
            "/outputs/..%5coutside%5csecret.txt",
            "/outputs/....//outside/secret.txt",
            "/outputs//etc/hostname",
            "/outputs/%2fetc%2fhostname",
            "/outputs/~/secret.txt",
            "/inputs/../outside/secret.png",
            "/inputs/..%2foutside/secret.png",
            "/inputs/%2e%2e%2f%2e%2e%2foutside%2fsecret.png",
            "/inputs/..%5coutside%5csecret.png",
            "/exports/..%2f..%2foutside.zip",
            "/exports/%2e%2e.zip",
        ],
    )
    def test_traversal_spellings_serve_nothing_outside(self, make_client, path):
        with make_client(token=TOKEN) as client:
            response = client.get(path)
        assert not _leaks(response), path
        assert response.status_code in (400, 404, 405), (path, response.status_code)

    def test_an_absolute_path_in_the_name_is_not_joined_as_absolute(
        self, make_client, layout
    ):
        """os.path.join(root, '/abs') is '/abs' - a name that starts with a
        separator must still resolve under the root."""
        secret = layout["outside"] / "secret.txt"
        with make_client(token=TOKEN) as client:
            for route in ("/outputs", "/inputs"):
                response = client.get(f"{route}/{secret}")
                assert not _leaks(response), route
                response = client.get(f"{route}/%2F{str(secret).lstrip('/')}")
                assert not _leaks(response), route

    @pytest.mark.parametrize("workspace", ["..", "../outside", "%2e%2e", "/tmp", "."])
    def test_a_hostile_workspace_parameter_is_refused(self, make_client, workspace):
        with make_client(token=TOKEN) as client:
            response = client.get(f"/outputs/secret.txt?workspace={workspace}")
        assert not _leaks(response)
        assert response.status_code in (400, 404, 422)

    def test_the_gallery_download_route_is_contained_as_well(self, make_client):
        with make_client(token=TOKEN) as client:
            for name in ("..%2Foutside%2Fsecret.txt", "%2e%2e/outside/secret.txt"):
                response = client.get(f"/api/gallery/{name}/download?token={TOKEN}")
                assert not _leaks(response), name


# ------------------------------------------------------------- dw.serve CLI


@pytest.fixture
def serve(monkeypatch, tmp_path):
    """dw.serve.main with the app factory, uvicorn and startup replaced."""
    import uvicorn

    import dw
    import dw.serve as serve_module
    from dw.server import app as app_module

    calls = {}

    def fake_create_app(**kwargs):
        calls["create_app"] = kwargs
        return object()

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(ran=True))
    monkeypatch.setattr(dw, "startup", lambda *a, **k: calls.update(started=True))
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    monkeypatch.setenv("DW_PROMPT_DIR", str(tmp_path / "prompts"))
    monkeypatch.setenv("DW_ASSET_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("DW_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("DW_WORKSPACE_SOURCE", "flag")
    (tmp_path / "workflows").mkdir()

    def run(*argv):
        calls.clear()
        monkeypatch.setattr(
            "sys.argv",
            ["dw-serve", "--workflow-dir", str(tmp_path / "workflows"), *argv],
        )
        serve_module.main()
        return calls

    return run


class TestServeRefusesAnOpenMcpEndpoint:
    @pytest.mark.parametrize(
        "host",
        [
            "0.0.0.0",
            "::",
            "10.0.0.2",
            "192.168.1.5",
            "gpu-box.local",
            # loopback in fact, but not a name the check knows - the
            # conservative direction, pinned so it stays conservative
            "127.0.0.2",
            "LOCALHOST",
        ],
    )
    def test_no_token_means_no_start(self, serve, host):
        with pytest.raises(SystemExit) as exit_info:
            serve("--mcp", "--host", host)
        assert exit_info.value.code == 2

    @pytest.mark.parametrize("token_args", [["--token", ""], []])
    def test_an_empty_token_is_no_token(self, serve, monkeypatch, token_args):
        monkeypatch.setenv("DW_API_TOKEN", "")
        with pytest.raises(SystemExit):
            serve("--mcp", "--host", "0.0.0.0", *token_args)

    def test_nothing_starts_before_the_refusal(self, serve):
        calls = {}
        try:
            calls = serve("--mcp", "--host", "0.0.0.0")
        except SystemExit:
            pass
        assert "started" not in calls and "ran" not in calls

    def test_the_environment_token_satisfies_it_and_reaches_the_app(
        self, serve, monkeypatch
    ):
        monkeypatch.setenv("DW_API_TOKEN", "from-env")
        calls = serve("--mcp", "--host", "0.0.0.0")
        assert calls["create_app"]["token"] == "from-env"
        assert calls["create_app"]["mcp"] is True

    def test_the_flag_token_wins_over_the_environment(self, serve, monkeypatch):
        monkeypatch.setenv("DW_API_TOKEN", "from-env")
        calls = serve("--mcp", "--host", "0.0.0.0", "--token", "from-flag")
        assert calls["create_app"]["token"] == "from-flag"
