# Remote GPU Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `dw.serve` as a long-lived service on the GPU machine and use it from other LAN machines - a browser for the UI and Claude Code over Streamable-HTTP MCP - with nothing installed on the client.

**Architecture:** Fix the `Origin` middleware so a non-loopback bind actually accepts its own browser; add host identity to `/api/health`; make the stdio `dw-mcp` refuse an unauthenticated remote and probe the server at startup; mount the existing MCP tool surface inside `dw.serve` at `/mcp` (behind the same bearer token) with `dw.serve --mcp`; ship a systemd unit and a remote-setup guide. No TLS in the app - beyond the LAN a reverse proxy or Tailscale is documented.

**Tech Stack:** FastAPI/Starlette + uvicorn (server), `mcp` 2.1.1 SDK (`MCPServer.streamable_http_app`), httpx (`dw_mcp` client), `httpx2` (the SDK's own client, used in the MCP-over-ASGI test), pytest + `fastapi.testclient.TestClient`.

**Spec:** `docs/superpowers/specs/2026-09-04-remote-gpu-server-design.md`

## Global Constraints

- `dw_mcp/` must never import any `dw.*` module (`tests/test_mcp_server.py::TestStartupWeight` guards it). `dw/` may import `dw_mcp/`.
- `mcp` is an optional extra. Import it lazily, only where `--mcp` is used; a missing package must produce a clear `SystemExit`, not an `ImportError` traceback.
- Never use `eval()`, `exec()`, `shell=True`. Paths through `dw/security.py`.
- The token stays a single shared static secret. No sessions, no users.
- Run `black dw/ dw_mcp/ tests/` before every commit. Run the touched test file, then the full suite (`pytest -q -x -p no:cacheprovider`) before the last commit of each task.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work on branch `remote-gpu-server` off `master`.

---

## File map

| File | Responsibility |
| --- | --- |
| `dw/server/app.py` | Origin rule (Task 1), health identity (Task 2), token gate covers `/mcp` and `create_app(mcp=, port=)` + lifespan drives the MCP session manager (Task 4) |
| `dw/server/mcp_mount.py` (new) | Builds the MCP ASGI app over a loopback `DwClient` and mounts it; the only place `dw/` imports `mcp` |
| `dw/serve.py` | `--mcp` flag, hard refusal for non-loopback + no token + `--mcp` (Task 5) |
| `dw_mcp/client.py` | `LOOPBACK_HOSTS`, `is_loopback_url()` (Task 3) |
| `dw_mcp/__main__.py` | Refuse-without-token, health probe, `--no-probe`, `main(argv, transport=None)` (Task 3) |
| `dw_mcp/server.py` | `download_output` docstring says which machine it writes to (Task 4) |
| `contrib/systemd/` (new) | `dw-serve.service`, `dw-serve.env.example`, `README.md` (Task 6) |
| `docs/REMOTE.md` (new) | LAN and beyond-LAN setup guide (Task 6) |
| `docs/MCP.md`, `docs/SERVER.md`, `docs/SECURITY.md`, `CLAUDE.md`, `docs/proposals/remote-gpu-server.md` | Updated (Task 6) |
| `tests/test_server.py` | Origin + health tests (Tasks 1, 2) |
| `tests/test_mcp_main.py` (new) | `dw-mcp` startup behavior (Task 3) |
| `tests/test_server_mcp.py` (new) | `/mcp` mount, token gate, real MCP client over ASGI (Task 4) |
| `tests/test_serve_main.py` (new) | `dw.serve` flag validation (Task 5) |

---

### Task 0: Branch

- [ ] **Step 1: Create the branch**

```bash
git checkout -b remote-gpu-server master
```

---

### Task 1: Origin rule accepts the request's own host

**Files:**
- Modify: `dw/server/app.py` (the `reject_foreign_origins` middleware, currently ~lines 413-425, and the `allowed_hosts` computation just below it ~lines 438-441)
- Test: `tests/test_server.py`

**Interfaces:**
- Produces: nothing new outside the middleware. Behavior: an `Origin` is accepted when its hostname is loopback, equals the non-wildcard `--host`, or equals the request's `Host` hostname.

Background: `create_app(host=...)` already computes `wildcard_bind` and `allowed_hosts` for the Host check, but that computation sits *after* the Origin middleware definition. Python closures resolve at call time, so the Origin middleware may reference `allowed_hosts` as long as it is defined somewhere in `create_app` before the app serves a request - but for readability, move the two lines up above the Origin middleware.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py` right after `test_foreign_origin_requests_are_rejected`:

```python
def _app_bound_to(tmp_path, host):
    """A create_app bound to `host`, with a scripted worker - the shape the
    Host/Origin tests share."""
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    return create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        prompt_dir=str(tmp_path / "prompts"),
        host=host,
    )


def test_origin_matching_the_request_host_is_accepted_on_a_wildcard_bind(tmp_path):
    """A browser on another machine reaches `--host 0.0.0.0` by LAN IP or
    hostname and sends that as its Origin. Same-origin must pass, or the UI
    cannot make a single POST off-loopback."""
    app = _app_bound_to(tmp_path, "0.0.0.0")
    for base in ("http://192.168.1.50:8765", "http://gpu-box.local:8765"):
        with TestClient(app, base_url=base) as client:
            response = client.post(
                "/api/jobs",
                json={"workflow": valid_workflow()},
                headers={"Origin": base},
            )
            assert response.status_code == 201, base


def test_origin_that_differs_from_the_request_host_is_still_rejected(tmp_path):
    """DNS rebinding: the attacker's page has its own Origin while Host is
    whatever their DNS name resolved to. The two differ, so it is refused -
    the same-origin allowance does not weaken the check."""
    app = _app_bound_to(tmp_path, "0.0.0.0")
    with TestClient(app, base_url="http://192.168.1.50:8765") as client:
        response = client.post(
            "/api/jobs",
            json={"workflow": valid_workflow()},
            headers={"Origin": "http://evil.example"},
        )
        assert response.status_code == 403


def test_origin_comparison_ignores_scheme_and_port(tmp_path):
    """A TLS-terminating proxy forwards Host as-is while the browser's Origin
    is https and may carry a different port; only the hostname matters."""
    app = _app_bound_to(tmp_path, "0.0.0.0")
    with TestClient(app, base_url="http://gpu-box.local:8765") as client:
        response = client.get(
            "/api/health", headers={"Origin": "https://gpu-box.local"}
        )
        assert response.status_code == 200


def test_origin_naming_the_configured_bind_host_is_accepted(tmp_path):
    """`--host my-server.local` accepts that Origin regardless of Host, the
    same allowance the Host check already makes for the bind address."""
    app = _app_bound_to(tmp_path, "my-server.local")
    with TestClient(app, base_url="http://my-server.local") as client:
        response = client.get(
            "/api/health", headers={"Origin": "http://my-server.local:9999"}
        )
        assert response.status_code == 200


def test_loopback_origin_is_accepted_for_any_host(tmp_path):
    """An SSH tunnel's browser sends a loopback Origin; that keeps working."""
    app = _app_bound_to(tmp_path, "0.0.0.0")
    with TestClient(app, base_url="http://192.168.1.50:8765") as client:
        response = client.get(
            "/api/health", headers={"Origin": "http://127.0.0.1:8765"}
        )
        assert response.status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server.py -k "origin" -v -p no:cacheprovider`
Expected: the first, third and fourth new tests FAIL with 403; the rebinding and loopback tests PASS (they assert existing behavior and must keep passing).

- [ ] **Step 3: Implement the rule**

In `dw/server/app.py`, inside `create_app`, move these two lines from below the Origin middleware to just above the `@app.middleware("http")` that defines `reject_foreign_origins` (keep the comment block that explains the Host check with the Host middleware):

```python
    wildcard_bind = host in WILDCARD_HOSTS
    allowed_hosts = set(LOOPBACK_HOSTS)
    if host and not wildcard_bind:
        allowed_hosts.add(host.lower())
```

Then replace the body of `reject_foreign_origins`:

```python
    @app.middleware("http")
    async def reject_foreign_origins(request, call_next):
        """Refuse browser cross-origin requests - a drive-by web page must
        not be able to queue jobs on this server. Requests without an
        Origin header (curl, scripts, same-origin GETs) pass.

        An Origin is accepted when its hostname is a loopback name, the
        configured bind host, or the hostname the request itself was
        addressed to (same-origin). The last clause is what lets a browser
        on another machine use a `--host 0.0.0.0` server by its LAN IP or
        hostname - and it stays safe against DNS rebinding, where the
        attacker's page carries its own Origin while Host is whatever
        resolved: the two differ, so the request is refused. Scheme and
        port are ignored, matching the Host check: a TLS-terminating proxy
        forwards Host unchanged while the browser's Origin is https."""
        origin = request.headers.get("origin")
        if origin:
            origin_host = (urlparse(origin).hostname or "").lower()
            request_host = (request.url.hostname or "").lower()
            if origin_host not in allowed_hosts and origin_host != request_host:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin requests are not allowed"},
                )
        return await call_next(request)
```

Note the change `if host and not wildcard_bind:` - previously `allowed_hosts` gained `"0.0.0.0"` on a wildcard bind, which was harmless for the Host check (skipped on wildcard) but would now let an `Origin: http://0.0.0.0` through. Nothing sends that; excluding it is just tidier.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_server.py -k "origin or host_header or bind_host or wildcard" -v -p no:cacheprovider`
Expected: all PASS, including the pre-existing `test_foreign_origin_requests_are_rejected`, `test_foreign_host_header_requests_are_rejected`, `test_configured_bind_host_is_allowed`, `test_wildcard_bind_accepts_requests_addressed_to_any_host`.

- [ ] **Step 5: Format and commit**

```bash
black dw/server/app.py tests/test_server.py
git add dw/server/app.py tests/test_server.py
git commit -m "fix(server): accept same-origin browser requests on a non-loopback bind

The Origin check only allowed loopback origins, so a browser reaching a
--host 0.0.0.0 server by LAN IP got 403 on every POST. An Origin whose
hostname matches the request's own Host is now accepted; a rebinding
attack still differs from Host and is still refused.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Health payload identifies the machine

**Files:**
- Modify: `dw/server/app.py` (the `health()` route, ~line 1295)
- Test: `tests/test_server.py::test_health_and_memory`

**Interfaces:**
- Produces: `GET /api/health` JSON gains `"hostname": str`, `"device": str` (one of `cuda`/`mps`/`cpu`/`xpu`...), `"mcp": bool` (always `False` until Task 4 wires it). Later tasks read `app.state.mcp_mounted` (a bool, default `False`) for the `mcp` field.

- [ ] **Step 1: Write the failing test**

Replace the top of `test_health_and_memory` in `tests/test_server.py`:

```python
def test_health_and_memory(server):
    import socket

    with server(success_script) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        # a remote client uses these to confirm which machine answered
        assert health["hostname"] == socket.gethostname()
        assert health["device"] in {"cuda", "mps", "cpu", "xpu"}
        assert health["mcp"] is False

        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])
        memory = client.get("/api/memory").json()
        assert memory["live"] is True
        assert memory["info"]["gpu_available"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_server.py::test_health_and_memory -v -p no:cacheprovider`
Expected: FAIL with `KeyError: 'hostname'`.

- [ ] **Step 3: Implement**

In `create_app`, right after `app.state.prompt_dir = prompt_dir`, add:

```python
    app.state.mcp_mounted = False
```

Replace the `health()` route:

```python
    @app.get("/api/health")
    def health():
        import socket

        from .. import __version__, get_device, get_device_type

        worker = manager.worker_manager
        return {
            "status": "ok",
            "version": __version__,
            "worker_alive": bool(
                worker.worker_active
                and worker.worker_process is not None
                and worker.worker_process.is_alive()
            ),
            "current_job": manager._current_job_id,
            "queued": sum(1 for j in manager.list() if j["status"] == "queued"),
            # which machine answered - the thing a remote client cannot
            # otherwise tell apart from a stale tunnel pointed at nothing
            "hostname": socket.gethostname(),
            "device": get_device_type(get_device()),
            "mcp": bool(app.state.mcp_mounted),
        }
```

`get_device` and `get_device_type` are both defined in `dw/__init__.py` (lines ~106 and ~134); `get_device_type(device)` strips the index from `cuda:1`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_server.py::test_health_and_memory -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black dw/server/app.py tests/test_server.py
git add dw/server/app.py tests/test_server.py
git commit -m "feat(server): report hostname, device and mcp in /api/health

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `dw-mcp` refuses an unauthenticated remote and probes the server

**Files:**
- Modify: `dw_mcp/client.py` (add `LOOPBACK_HOSTS`, `is_loopback_url`)
- Modify: `dw_mcp/__main__.py`
- Create: `tests/test_mcp_main.py`

**Interfaces:**
- Produces in `dw_mcp/client.py`:
  - `LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})`
  - `def is_loopback_url(url: str) -> bool`
- Produces in `dw_mcp/__main__.py`: `def main(argv=None, transport=None) -> int` - `transport` is an `httpx.BaseTransport` injected by tests; `--no-probe` flag. Exit code 2 on every refusal.
- Consumes: `/api/health` fields `hostname`, `version`, `device` from Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_main.py`:

```python
"""`dw-mcp` startup: what it refuses, what it checks, what it prints."""

import json

import httpx
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw_mcp import __main__ as cli  # noqa: E402
from dw_mcp.client import is_loopback_url  # noqa: E402


@pytest.fixture(autouse=True)
def no_ambient_config(monkeypatch):
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    monkeypatch.delenv("DW_MCP_URL", raising=False)


@pytest.fixture
def no_stdio(monkeypatch):
    """Stop main() from actually serving stdio once startup checks pass."""
    ran = {}

    class FakeServer:
        def run(self, transport):
            ran["transport"] = transport

    monkeypatch.setattr(cli, "build_server", lambda client: FakeServer())
    return ran


def healthy(request):
    return httpx.Response(
        200,
        json={"status": "ok", "hostname": "gpu-box", "version": "1.2.3", "device": "cuda"},
    )


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:8765", True),
        ("http://localhost:8765", True),
        ("http://[::1]:8765", True),
        ("http://192.168.1.50:8765", False),
        ("http://gpu-box.local:8765", False),
    ],
)
def test_is_loopback_url(url, expected):
    assert is_loopback_url(url) is expected


def test_a_remote_url_without_a_token_is_refused_before_any_request(capsys):
    requests = []

    def handler(request):
        requests.append(request)
        return healthy(request)

    code = cli.main(
        ["--url", "http://192.168.1.50:8765"], transport=httpx.MockTransport(handler)
    )

    assert code == 2
    assert requests == []
    err = capsys.readouterr().err
    assert "--token" in err and "DW_API_TOKEN" in err
    assert "192.168.1.50" in err


def test_a_loopback_url_without_a_token_is_allowed(no_stdio):
    code = cli.main([], transport=httpx.MockTransport(healthy))
    assert code == 0
    assert no_stdio["transport"] == "stdio"


def test_the_probe_reports_a_server_that_wants_a_token(capsys):
    def handler(request):
        return httpx.Response(401, json={"detail": "Missing or invalid bearer token"})

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "wrong"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 2
    assert "token" in capsys.readouterr().err.lower()


def test_the_probe_reports_an_unreachable_server(capsys):
    def handler(request):
        raise httpx.ConnectError("refused")

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "t"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 2
    assert "http://192.168.1.50:8765" in capsys.readouterr().err


def test_a_successful_probe_prints_the_server_identity(no_stdio, capsys):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        return healthy(request)

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "s3cr3t"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 0
    assert seen == [("GET", "/api/health", "Bearer s3cr3t")]
    err = capsys.readouterr().err
    assert "gpu-box" in err and "1.2.3" in err and "cuda" in err


def test_no_probe_sends_nothing(no_stdio):
    seen = []

    def handler(request):
        seen.append(request)
        return healthy(request)

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "t", "--no-probe"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 0
    assert seen == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_main.py -v -p no:cacheprovider`
Expected: FAIL at import (`cannot import name 'is_loopback_url'`).

- [ ] **Step 3: Add the loopback helper to `dw_mcp/client.py`**

After `DEFAULT_BASE_URL = "http://127.0.0.1:8765"`:

```python
# Twin of dw.server.app.LOOPBACK_HOSTS. Duplicated rather than imported:
# importing anything under dw/ runs dw/__init__.py and pulls in torch, which
# this pure HTTP client must not do (tests/test_mcp_server.py guards that).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_loopback_url(url):
    """True when `url` names this machine's loopback interface - the case
    where an unauthenticated dw.serve is only reachable by this user."""
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower() in LOOPBACK_HOSTS
```

- [ ] **Step 4: Rewrite `dw_mcp/__main__.py`**

```python
"""`python -m dw_mcp` / `dw-mcp`: serve the tool surface over stdio."""

import argparse
import sys

import httpx

from dw_mcp.client import (
    DwApiError,
    DwClient,
    is_loopback_url,
    resolve_base_url,
    resolve_token,
)
from dw_mcp.server import build_server


def _refuse(message):
    print(f"dw-mcp: {message}", file=sys.stderr)
    return 2


def main(argv=None, transport=None):
    parser = argparse.ArgumentParser(
        prog="dw-mcp",
        description="MCP server for diffusers-workflow. Requires a running "
        "dw.serve - start one with `dw-serve` first.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL of the running dw.serve "
        "(default: $DW_MCP_URL, else http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token the dw.serve was started with, if any "
        "(default: $DW_API_TOKEN). Required when --url is not loopback.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait on any one API request (default: 30)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        default=False,
        help="Skip the startup GET /api/health that confirms the server is "
        "reachable and the token is accepted",
    )
    args = parser.parse_args(argv)

    base_url = resolve_base_url(args.url)
    token = resolve_token(args.token)

    # A remote dw.serve with no token would let anyone on that network run
    # workflows as this user; refusing here is the client-side half of the
    # server's own non-loopback-without-token warning.
    if not is_loopback_url(base_url) and not token:
        return _refuse(
            f"{base_url} is not a loopback address and no token is set. "
            "A dw.serve reachable from other machines must be started with "
            "--token / DW_API_TOKEN, and the same token passed here with "
            "--token or DW_API_TOKEN."
        )

    client = DwClient(
        base_url=base_url, timeout=args.timeout, token=token, transport=transport
    )
    try:
        if not args.no_probe:
            code = _probe(client, base_url)
            if code:
                return code
        build_server(client).run(transport="stdio")
    finally:
        client.close()
    return 0


def _probe(client, base_url):
    """One GET /api/health so a wrong URL or token fails here, once, with a
    message - not on every tool call as an unexplained 401."""
    try:
        health = client.get_json("/api/health")
    except DwApiError as e:
        return _refuse(f"could not reach dw.serve at {base_url}: {e}")
    print(
        "dw-mcp: connected to {host} (dw {version}, {device}) at {url}".format(
            host=health.get("hostname", "?"),
            version=health.get("version", "?"),
            device=health.get("device", "?"),
            url=base_url,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Check how `DwClient._raise_for_status` phrases a 401 (in `dw_mcp/client.py`, search `401`). The test only asserts the word "token" appears; if the existing 401 message does not contain it, catch that case in `_probe` explicitly:

```python
    except DwApiError as e:
        text = str(e)
        if "401" in text or "token" in text.lower():
            return _refuse(
                f"dw.serve at {base_url} requires a bearer token and rejected "
                f"the one given (or none was given): {text}"
            )
        return _refuse(f"could not reach dw.serve at {base_url}: {text}")
```

Use the second form regardless - it is the better message.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_main.py tests/test_mcp_client.py tests/test_mcp_server.py -v -p no:cacheprovider`
Expected: all PASS. If `test_the_probe_reports_an_unreachable_server` fails because `httpx.ConnectError` escapes as a raw exception rather than `DwApiError`, look at how `DwClient._request` translates transport errors (`tests/test_mcp_client.py::test_a_refused_connection_says_how_to_start_the_server` shows the existing behavior) and catch `httpx.HTTPError` alongside `DwApiError` in `_probe`.

- [ ] **Step 6: Format and commit**

```bash
black dw_mcp/ tests/test_mcp_main.py
git add dw_mcp/client.py dw_mcp/__main__.py tests/test_mcp_main.py
git commit -m "feat(mcp): refuse an unauthenticated remote server and probe at startup

dw-mcp exits 2 when --url is not loopback and no token is set, and makes
one GET /api/health before serving so a wrong URL or token is reported
once with a message rather than as a 401 on every tool call. --no-probe
skips the check.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Mount MCP over Streamable HTTP inside `dw.serve`

**Files:**
- Create: `dw/server/mcp_mount.py`
- Modify: `dw/server/app.py` (`create_app` signature, lifespan, `require_bearer_token` path test, `app.state.mcp_mounted`)
- Modify: `dw_mcp/server.py` (`download_output` docstring)
- Create: `tests/test_server_mcp.py`

**Interfaces:**
- Produces in `dw/server/mcp_mount.py`:
  - `def build_mcp_app(*, port: int, token: str | None) -> tuple[Starlette, MCPServer, DwClient]` - the ASGI app to mount, the MCP server (whose `session_manager.run()` the parent lifespan must enter), and the loopback client to close at shutdown. Raises `SystemExit` with an install hint if `mcp` is missing.
- Produces in `create_app`: keyword params `mcp: bool = False`, `port: int = 8765`. When `mcp=True` the app has `/mcp` mounted and `app.state.mcp_mounted = True`.
- Consumes: `dw_mcp.client.DwClient`, `dw_mcp.server.build_server`, `app.state.mcp_mounted` from Task 2.

How the SDK piece fits: `MCPServer.streamable_http_app(streamable_http_path="/")` returns a Starlette app with one route at `/` and a lifespan that runs `server.session_manager.run()`. Starlette does not run a mounted sub-app's lifespan, so `create_app`'s own lifespan must enter `session_manager.run()` itself. Verified against `mcp` 2.1.1 (`mcp/server/lowlevel/server.py`, the `Starlette(... lifespan=lambda app: session_manager.run())` at the end of `streamable_http_app`).

`json_response=True` + `stateless_http=True`: every MCP request gets a plain JSON reply rather than an SSE stream. Simplest mode, survives server restarts, and is what makes the in-process ASGI test possible (`httpx2.ASGITransport` buffers the response body).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_mcp.py`:

```python
"""MCP over Streamable HTTP, mounted inside dw.serve at /mcp."""

import json

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw.server.app import create_app  # noqa: E402
from dw.server.jobs import JobManager  # noqa: E402

from tests.test_mcp_server import EXPECTED_TOOLS  # noqa: E402
from tests.test_server import ScriptedWorkerManager, success_script  # noqa: E402

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def make_app(tmp_path, token=None, mcp=True):
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    return create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        prompt_dir=str(tmp_path / "prompts"),
        host="0.0.0.0",
        token=token,
        mcp=mcp,
        port=8765,
    )


def test_mcp_is_not_mounted_by_default(tmp_path):
    app = make_app(tmp_path, mcp=False)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health").json()["mcp"] is False
        assert client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 404


def test_mcp_mount_answers_initialize(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health").json()["mcp"] is True
        response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["result"]["serverInfo"]["name"] == "diffusers-workflow"


def test_mcp_mount_is_gated_by_the_bearer_token(tmp_path):
    app = make_app(tmp_path, token="s3cr3t")
    with TestClient(app, base_url="http://localhost") as client:
        assert client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 401
        wrong = {**MCP_HEADERS, "Authorization": "Bearer nope"}
        assert client.post("/mcp", json=INITIALIZE, headers=wrong).status_code == 401
        # never as a query parameter - that allowance is for <img>/<a> only
        assert (
            client.post("/mcp?token=s3cr3t", json=INITIALIZE, headers=MCP_HEADERS).status_code
            == 401
        )
        right = {**MCP_HEADERS, "Authorization": "Bearer s3cr3t"}
        assert client.post("/mcp", json=INITIALIZE, headers=right).status_code == 200


@pytest.mark.asyncio
async def test_a_real_mcp_client_lists_every_tool_over_http(tmp_path):
    """The SDK's own client, speaking Streamable HTTP into the mounted app
    in-process, sees the same tool surface the stdio server exposes."""
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = make_app(tmp_path, token="s3cr3t")
    # TestClient drives the lifespan (the MCP session manager); the SDK
    # client rides an ASGI transport straight into the same app object.
    with TestClient(app, base_url="http://localhost"):
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer s3cr3t"},
        )
        async with http:
            async with streamable_http_client(
                "http://localhost/mcp", http_client=http
            ) as streams:
                async with ClientSession(streams.read_stream, streams.write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
    assert {t.name for t in tools.tools} == EXPECTED_TOOLS


def test_download_output_says_it_writes_on_the_server_side():
    """Over /mcp the tool runs on the GPU box; the description has to say so
    or an agent on another machine will look for the file locally."""
    from dw_mcp.client import DwClient
    from dw_mcp.server import build_server

    import asyncio

    server = build_server(DwClient())
    tools = asyncio.run(server.list_tools())
    description = next(t.description for t in tools if t.name == "download_output")
    assert "machine running the MCP server" in description
```

If the `streams` object returned by `streamable_http_client` is a tuple rather than an object with `.read_stream`/`.write_stream`, unpack it as `read, write, _ = streams` - check `mcp.client.streamable_http.TransportStreams` in the installed SDK (`python -c "import mcp.client.streamable_http as m; help(m.TransportStreams)"`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server_mcp.py -v -p no:cacheprovider`
Expected: FAIL - `create_app() got an unexpected keyword argument 'mcp'`.

- [ ] **Step 3: Create `dw/server/mcp_mount.py`**

```python
"""Mount the MCP tool surface inside the HTTP server.

`dw_mcp` is a pure HTTP client of dw.serve; mounting it here does not
change that - the tools reach the REST API over loopback with the same
token the API requires, a few milliseconds per call. This is the only
module under dw/ that imports the mcp SDK, and it does so lazily: the
package is an optional extra.
"""


def build_mcp_app(*, port, token):
    """The ASGI app to mount at /mcp, the MCPServer behind it, and the
    loopback client the tools use.

    The returned Starlette app carries its own lifespan (the SDK's session
    manager), which Starlette does not run for a mounted sub-app - the
    parent's lifespan must enter `server.session_manager.run()` itself.
    """
    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:
        raise SystemExit(
            "--mcp needs the mcp extra: pip install 'diffusers-workflow[mcp]'"
        )

    from dw_mcp.client import DwClient
    from dw_mcp.server import build_server

    client = DwClient(base_url=f"http://127.0.0.1:{port}", token=token)
    server = build_server(client)
    asgi = server.streamable_http_app(
        # the SDK app routes at "/"; the parent mounts it under /mcp
        streamable_http_path="/",
        # one JSON reply per request, no sessions to strand on a restart
        json_response=True,
        stateless_http=True,
        # dw.server.app's own Origin/Host middleware runs first and owns this
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    return asgi, server, client
```

- [ ] **Step 4: Wire it into `create_app`**

In `dw/server/app.py`:

(a) Signature - add two keyword parameters at the end:

```python
def create_app(
    workflow_dir="./workflows",
    output_dir="./outputs",
    log_level="INFO",
    job_manager=None,
    ui_dir=None,
    download_manager=None,
    diffusers_updater=None,
    prompt_dir="./prompts",
    host="127.0.0.1",
    token=None,
    mcp=False,
    port=8765,
):
```

and extend the docstring: `mcp=True mounts the MCP tool surface at /mcp (Streamable HTTP), gated by the same token; port is the port this server listens on, which the mounted tools call back to over loopback.`

(b) Build the MCP pieces before the lifespan is defined (just after the `manager.workflow_dir` check):

```python
    mcp_asgi = mcp_server = mcp_client = None
    if mcp:
        from .mcp_mount import build_mcp_app

        mcp_asgi, mcp_server, mcp_client = build_mcp_app(port=port, token=token)
```

(c) Replace the lifespan:

```python
    @asynccontextmanager
    async def lifespan(app):
        if mcp_server is None:
            yield
        else:
            # the SDK's session manager is the mounted app's own lifespan,
            # which Starlette does not run for a sub-app
            async with mcp_server.session_manager.run():
                yield
            mcp_client.close()
        manager.shutdown()
```

(d) Set state right after `app.state.mcp_mounted = False` (from Task 2):

```python
    app.state.mcp_mounted = mcp_asgi is not None
```

(delete the `= False` line from Task 2 and keep this one).

(e) In `require_bearer_token`, change the path test so `/mcp` is gated exactly like `/api/`:

```python
        path = request.url.path
        if not (path.startswith("/api/") or path == "/mcp" or path.startswith("/mcp/")):
            return await call_next(request)
```

The query-parameter allowance below it is matched per route via `_matched_route` and `query_token_ok`; the mounted app has no such attribute, so `?token=` never applies to `/mcp` - the test asserts it.

(f) Mount it. Just before `app.mount("/outputs", ...)` (so `/api` routes keep precedence and the UI catch-all stays last):

```python
    # ------------------------------------------------------------------ mcp

    if mcp_asgi is not None:
        app.mount("/mcp", mcp_asgi, name="mcp")
```

- [ ] **Step 5: Update the `download_output` description in `dw_mcp/server.py`**

Replace the docstring of `download_output`:

```python
        """Save one output file to disk on the machine running the MCP
        server - for the stdio `dw-mcp` that is your own machine; for a
        `dw.serve --mcp` endpoint it is the GPU box, and this tool is not
        the way to get a file to where you are (use get_output_image /
        get_output_text for inline content, or the /outputs URL). Unlike
        those two, this works for any file type, streams the body straight
        to disk rather than buffering it, and returns no content to the
        conversation - only where it was saved. `destination` may be a
        full path, a directory, or omitted to save into the current
        working directory under the output's own name; a '..' path segment
        in it is refused. An existing file at the resolved path is left
        alone unless `overwrite=True`."""
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_server_mcp.py tests/test_mcp_server.py tests/test_server.py -v -p no:cacheprovider`
Expected: all PASS. Likely wrinkles and their fixes:
- `initialize` returns 406: the `Accept` header must list both `application/json` and `text/event-stream` (it does in `MCP_HEADERS`).
- 404 on `/mcp` but 200 on `/mcp/`: Starlette's `Mount` redirects or requires the trailing slash for a sub-app whose route is `/`. If so, mount at `/mcp` and pass `streamable_http_path="/"` still, but add `redirect_slashes=False` is not a Mount option - instead mount the SDK app at `""`? No: the cleanest fix is to keep the mount and have the token gate + tests use whichever of `/mcp` / `/mcp/` the SDK answers, then normalize by adding a tiny `@app.post("/mcp")` forwarding route only if needed. Try the plain mount first; the SDK's `Route("/")` inside a `Mount("/mcp")` matches `/mcp` in Starlette ≥ 0.30 (a Mount strips its prefix and the sub-app sees path `""`, which `Route("/")` matches).
- `test_a_real_mcp_client_lists_every_tool_over_http` hangs: `httpx2.ASGITransport` needs the app's lifespan already started - the surrounding `with TestClient(app)` does that. If it still hangs, `json_response=True` did not take effect; confirm `build_mcp_app` passes it.

- [ ] **Step 7: Format and commit**

```bash
black dw/ dw_mcp/ tests/test_server_mcp.py
git add dw/server/mcp_mount.py dw/server/app.py dw_mcp/server.py tests/test_server_mcp.py
git commit -m "feat(server): mount MCP over Streamable HTTP at /mcp

create_app(mcp=True) mounts the dw_mcp tool surface inside the HTTP
server, behind the same bearer token as /api. The tools call back to
the REST API over loopback, so dw_mcp stays a pure HTTP client. An
agent on another machine needs no local install: one URL and a token.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `dw.serve --mcp`

**Files:**
- Modify: `dw/serve.py`
- Create: `tests/test_serve_main.py`

**Interfaces:**
- Consumes: `create_app(..., mcp=, port=)` from Task 4.
- Produces: `--mcp` flag; hard `SystemExit(2)` when `--mcp` and the bind is non-loopback and no token; the startup banner names the MCP URL.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_serve_main.py`:

```python
"""dw.serve's argument handling, without starting uvicorn."""

import pytest


@pytest.fixture
def serve(monkeypatch, tmp_path):
    """dw.serve.main with uvicorn and the app factory replaced, so a call
    returns what it would have served instead of serving it."""
    import uvicorn

    import dw.serve as serve_module
    from dw.server import app as app_module

    calls = {}

    def fake_create_app(**kwargs):
        calls["create_app"] = kwargs
        return object()

    def fake_run(app, **kwargs):
        calls["uvicorn"] = kwargs

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    (tmp_path / "workflows").mkdir()

    def run(*argv):
        monkeypatch.setattr(
            "sys.argv",
            ["dw-serve", "--workflow-dir", str(tmp_path / "workflows"), *argv],
        )
        serve_module.main()
        return calls

    return run


def test_mcp_is_off_by_default(serve):
    calls = serve()
    assert calls["create_app"]["mcp"] is False
    assert calls["create_app"]["port"] == 8765


def test_mcp_flag_is_passed_through(serve):
    calls = serve("--mcp", "--port", "9000", "--token", "t")
    assert calls["create_app"]["mcp"] is True
    assert calls["create_app"]["port"] == 9000


def test_mcp_on_a_non_loopback_bind_requires_a_token(serve, capsys):
    with pytest.raises(SystemExit) as exit_info:
        serve("--mcp", "--host", "0.0.0.0")
    assert exit_info.value.code == 2
    assert "--token" in capsys.readouterr().err


def test_mcp_on_loopback_needs_no_token(serve):
    calls = serve("--mcp")
    assert calls["create_app"]["mcp"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_serve_main.py -v -p no:cacheprovider`
Expected: FAIL - `unrecognized arguments: --mcp`; `test_mcp_is_off_by_default` fails on `KeyError: 'mcp'`.

- [ ] **Step 3: Implement in `dw/serve.py`**

Add the flag after `--trust-workflows`:

```python
    parser.add_argument(
        "--mcp",
        action="store_true",
        default=False,
        help="Also serve the MCP tool surface at /mcp over Streamable HTTP, "
        "behind the same token as /api, so an agent on another machine "
        "needs no local install (`claude mcp add --transport http dw "
        "http://<host>:<port>/mcp --header 'Authorization: Bearer <token>'`). "
        "Refused on a non-loopback --host without a token.",
    )
```

Immediately after `token = args.token or os.environ.get("DW_API_TOKEN") or None` (before any `dw` import, so the refusal costs nothing):

```python
    from .server.app import LOOPBACK_HOSTS  # noqa: E402 - light import

    if args.mcp and args.host not in LOOPBACK_HOSTS and not token:
        print(
            f"dw-serve: --mcp on {args.host} needs a token. An MCP endpoint "
            "can author and run workflows, and unlike the web UI there is "
            "no page to type a token into - pass --token or set "
            "DW_API_TOKEN, or bind to 127.0.0.1.",
            file=sys.stderr,
        )
        raise SystemExit(2)
```

Note: `dw.server.app` is not light - it imports `dw/__init__` (torch). That is already the cost of `dw.serve`; the point is only that the refusal comes before `startup()` and before the worker is spawned. Add `import sys` at the top of the file. Move the existing `from .server.app import LOOPBACK_HOSTS` further down (it is imported again there) or drop the later one - one import is enough.

Pass the new arguments to `create_app` and extend the banner:

```python
    app = create_app(
        workflow_dir=os.path.abspath(args.workflow_dir),
        output_dir=args.output_dir,
        log_level=args.log_level,
        prompt_dir=prompt_dir,
        host=args.host,
        token=token,
        mcp=args.mcp,
        port=args.port,
    )
    ui = " - UI at /" if default_ui_dir() else ""
    mcp = " - MCP at /mcp" if args.mcp else ""
    print(
        f"diffusers-workflow server on http://{args.host}:{args.port}  (docs at /docs{ui}{mcp})"
    )
```

Note the test's `fake_create_app(**kwargs)` requires `create_app` be called with keyword arguments only - it already is. The test patches `dw.server.app.create_app`; `dw/serve.py` does `from .server.app import create_app` *inside* `main()`, after the patch, so the fake is what it gets.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_serve_main.py -v -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Format, run the whole suite, commit**

```bash
black dw/serve.py tests/test_serve_main.py
pytest -q -x -p no:cacheprovider
git add dw/serve.py tests/test_serve_main.py
git commit -m "feat(serve): --mcp serves the tool surface at /mcp

Refused on a non-loopback bind without a token: an MCP endpoint can
write and run workflows and has no UI-style token prompt in front of it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Docs, systemd unit, proposal status

**Files:**
- Create: `contrib/systemd/dw-serve.service`, `contrib/systemd/dw-serve.env.example`, `contrib/systemd/README.md`
- Create: `docs/REMOTE.md`
- Modify: `docs/MCP.md` (Client configuration + Security sections), `docs/SERVER.md` (Security model, health line, options), `docs/SECURITY.md` (MCP Server section), `CLAUDE.md`, `docs/proposals/remote-gpu-server.md`, `README.md` (one link)

No code; the "test" is a docs consistency check plus a real smoke run.

- [ ] **Step 1: Write `contrib/systemd/dw-serve.service`**

```ini
# diffusers-workflow server as a system service. See README.md alongside.
#   sudo cp dw-serve.service /etc/systemd/system/
#   sudo cp dw-serve.env.example /etc/dw-serve.env && sudo $EDITOR /etc/dw-serve.env
#   sudo systemctl daemon-reload && sudo systemctl enable --now dw-serve
[Unit]
Description=diffusers-workflow server (dw-serve)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# The user whose home holds the Hugging Face cache and the checkout
User=dw
Group=dw
# The git checkout that install.sh set up
WorkingDirectory=/home/dw/diffusers-workflow
EnvironmentFile=/etc/dw-serve.env
# --host 0.0.0.0 makes it reachable from the LAN; DW_API_TOKEN in the env
# file is what gates it. Drop --mcp if no agent will connect.
ExecStart=/home/dw/diffusers-workflow/venv/bin/dw-serve --host 0.0.0.0 --port 8765 --mcp --workflow-dir workflows --output-dir outputs
Restart=on-failure
RestartSec=5
# Model loads are slow to interrupt; give a job time to unwind
TimeoutStopSec=60
# journalctl -u dw-serve -f
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write `contrib/systemd/dw-serve.env.example`**

```bash
# Copied to /etc/dw-serve.env by the README's steps. KEY=value, no quotes, no export.

# Required when --host is not 127.0.0.1. Generate one:  openssl rand -hex 32
DW_API_TOKEN=replace-me

# Where models are cached (default ~/.cache/huggingface of the service user)
#HF_HOME=/data/hf

# Gated models (FLUX.1-dev etc.) need a Hugging Face token
#HF_TOKEN=hf_...

# Pin the accelerator: cuda, cuda:1, cpu. Default: autodetect
#DW_DEVICE=cuda

# Stored prompt library (default: ./prompts under the working directory)
#DW_PROMPT_DIR=/home/dw/diffusers-workflow/prompts
```

- [ ] **Step 3: Write `contrib/systemd/README.md`**

```markdown
# Running dw-serve as a systemd service

For a Linux GPU box that should serve `dw` to the rest of your network
without anyone keeping a terminal open. The unit file assumes a user `dw`
with the repository checked out and installed (`bash ./install.sh`) at
`/home/dw/diffusers-workflow`; edit `User`, `Group`, `WorkingDirectory` and
`ExecStart` if yours differ.

    sudo cp dw-serve.service /etc/systemd/system/
    sudo cp dw-serve.env.example /etc/dw-serve.env
    sudo chmod 600 /etc/dw-serve.env           # it holds the API token
    sudo $EDITOR /etc/dw-serve.env             # set DW_API_TOKEN at least
    sudo systemctl daemon-reload
    sudo systemctl enable --now dw-serve
    systemctl status dw-serve
    journalctl -u dw-serve -f                  # logs

Then, from another machine on the LAN, `curl http://<box>:8765/api/health
-H "Authorization: Bearer <token>"` should answer with the box's hostname.
The full client-side setup is in [docs/REMOTE.md](../../docs/REMOTE.md).

Updating: `git pull && bash ./install.sh` in the checkout, then
`sudo systemctl restart dw-serve`. (The web UI's Models page can update
`diffusers` alone without a restart.)

macOS as the server: there is no unit here; a `launchd` plist with the same
`ExecStart` and environment does the job, or run it under `tmux`.
```

- [ ] **Step 4: Write `docs/REMOTE.md`**

```markdown
# Using dw from another machine

`dw.serve` runs on the machine with the GPU; a browser and Claude Code on
your laptop use it over the network. Nothing is installed on the laptop.

## On the GPU box

1. Install as usual: `git clone …`, `bash ./install.sh`.
2. Make a token: `openssl rand -hex 32`. This one string is the only thing
   between "on your network" and "can run workflows on your GPU" - keep it
   long and random.
3. Run it bound to the network, with the token, with MCP:

       DW_API_TOKEN=<token> dw-serve --host 0.0.0.0 --mcp

   To keep it running across logouts and reboots, install the systemd unit
   in [contrib/systemd](../contrib/systemd/README.md) instead.
4. Open port 8765 in the box's firewall for your LAN only (for example
   `sudo ufw allow from 192.168.1.0/24 to any port 8765`).

Never pass `--trust-workflows` on a server other machines can reach: it
lets any workflow - including one an agent authored - execute arbitrary
Python. See [SECURITY.md](SECURITY.md#trust-model).

Check it from the laptop:

    curl http://<box>:8765/api/health -H "Authorization: Bearer <token>"
    {"status":"ok","version":"…","hostname":"gpu-box","device":"cuda","mcp":true,…}

`hostname` and `device` are there so you can tell which machine answered.

## Browser

Open `http://<box>:8765`. Click the key icon next to the theme toggle,
paste the token once; it is kept in the browser's `localStorage` and sent
with every request.

## Claude Code, no local install

    claude mcp add --transport http dw http://<box>:8765/mcp \
      --header "Authorization: Bearer <token>"

Start a new Claude Code session; `/mcp` should list `dw` as connected.
Pick the scope with `-s user` to have it in every project.

Two things differ from the local stdio setup:

- `download_output` writes on the GPU box (where the MCP server runs), not
  on your laptop. Use `get_output_image` / `get_output_text` to see a
  result, or open `http://<box>:8765/outputs/<name>` in the browser.
- The connection is a plain HTTP call per tool invocation; there is no
  subprocess to restart.

## Claude Code with a local install (stdio)

If the laptop also has `dw` installed, the stdio server works against a
remote box too:

    claude mcp add dw -- /path/to/venv/bin/dw-mcp \
      --url http://<box>:8765 --token <token>

`dw-mcp` refuses to start against a non-loopback URL without a token, and
makes one `GET /api/health` at startup so a wrong URL or token fails once,
with a message, instead of on every tool call.

## Beyond your LAN

Everything above is plaintext HTTP: the token and every prompt and result
are readable by anything on the network path. That is acceptable on a
network you control and nowhere else. **Do not port-forward 8765 on your
router.** Two ways to reach the box from outside:

**Tailscale (or another WireGuard overlay).** Install it on the box and the
laptop; use the box's Tailscale IP or MagicDNS name in every URL above.
Traffic is encrypted end to end, no certificates to manage, and `dw-serve`
can stay bound to the Tailscale interface (`--host 100.x.y.z`) rather than
`0.0.0.0`. This is the recommended option.

**A TLS-terminating reverse proxy.** Bind `dw-serve` back to loopback and
put Caddy in front of it with a real hostname:

    # /etc/caddy/Caddyfile
    dw.example.com {
        reverse_proxy 127.0.0.1:8765
    }

Caddy obtains and renews a certificate automatically. Use
`https://dw.example.com` (no port) in every URL above. `dw-serve`'s Origin
and Host checks work unchanged behind the proxy because Caddy forwards the
`Host` header as-is. nginx works the same way with `proxy_pass` and
`proxy_set_header Host $host;` plus your own certificate.

The token is still the only authentication in either setup; a proxy or a
VPN protects the transport, not the door.

## What is not here

- TLS inside `dw-serve` itself: a reverse proxy does it better.
- More than one token, or users: one shared secret, deliberately
  ([SERVER.md](SERVER.md#authentication)).
- Docker: `install.sh` on the box is the supported install.
```

- [ ] **Step 5: Update `docs/MCP.md`**

In **Client configuration**, add a new subsection before "### Claude Code":

```markdown
### Remote server, no local install

If `dw.serve` runs on another machine with `--mcp` (see
[REMOTE.md](REMOTE.md)), Claude Code connects to it directly:

    claude mcp add --transport http dw http://<box>:8765/mcp \
      --header "Authorization: Bearer <token>"

Nothing from this repository is installed on the client. The stdio setup
below is for a machine that has its own `dw` install, and also works
against a remote `--url` with `--token`.
```

Replace the **Security** section's paragraph beginning "Because it adds no authentication" with:

```markdown
`dw-mcp` may be pointed at a `dw.serve` on another machine only when that
server was started with a token, and the same token is passed here
(`--token` / `DW_API_TOKEN`); it refuses to start otherwise. The token is
the only authentication, and the connection is plaintext HTTP - use it on
a network you control, or through Tailscale or a TLS proxy beyond that.
[REMOTE.md](REMOTE.md) has the full setup. The same applies to
`dw.serve --mcp`, which serves this tool surface itself at `/mcp` behind
the same token; in that setup `download_output` writes on the server
machine, not the client's.
```

In **Tool reference › Media**, on the `download_output` line, append: "Writes on the machine running the MCP server - over `dw.serve --mcp` that is the GPU box."

- [ ] **Step 6: Update `docs/SERVER.md`**

In **Security model**, replace the bullet beginning "Requests carrying a non-local `Origin` header are rejected (403)" with:

```markdown
- Requests carrying an `Origin` header are rejected (403) unless its
  hostname is a loopback name, the configured `--host`, or the hostname
  the request itself was addressed to (`Host`). The last clause lets a
  browser on another machine use a `--host 0.0.0.0` server by LAN IP or
  hostname; it still blocks cross-site pages and DNS rebinding, where the
  attacker's page carries its own `Origin` while `Host` is whatever
  resolved. Scheme and port are ignored, so a TLS-terminating proxy that
  forwards `Host` unchanged needs no configuration.
```

Add a bullet after the `--trust-workflows` one:

```markdown
- `--mcp` mounts the MCP tool surface at `/mcp` (Streamable HTTP) behind
  the same token as `/api`, for an agent on another machine with no local
  install. It is refused on a non-loopback bind without a token. See
  [REMOTE.md](REMOTE.md).
```

Update the `GET /api/health` line in the API list: "`GET /api/memory`, `GET /api/health` — worker VRAM/RAM stats and liveness; health also reports `hostname`, `device` and whether `mcp` is mounted, so a remote client can tell which machine answered". Add a "Running on another machine" one-liner at the end of Authentication pointing to `REMOTE.md`.

- [ ] **Step 7: Update `docs/SECURITY.md`, `CLAUDE.md`, `README.md`, the proposal**

`docs/SECURITY.md` › **MCP Server**: replace "Localhost only — see [MCP Server](MCP.md#security)." with "A remote `dw.serve` is allowed only with a token — see [MCP Server](MCP.md#security) and [REMOTE.md](REMOTE.md)." In the `download_output` paragraph, after "the client's own filesystem permissions allow", add "(the machine running the MCP server - the GPU box when served by `dw.serve --mcp`)".

`CLAUDE.md`: in **Server & Web UI**, after the sentence ending "See docs/SERVER.md.", add:

```markdown
`dw.serve --mcp` additionally mounts the MCP tool surface at `/mcp`
(`dw/server/mcp_mount.py`, Streamable HTTP, same bearer token) so an
agent on another machine needs no local install; `contrib/systemd/` has a
unit file and docs/REMOTE.md the LAN/NAT setup. The `Origin` check accepts
the request's own `Host` hostname, which is what makes a non-loopback bind
usable from a browser.
```

`README.md`: find the docs list (search for `SERVER.md`) and add a line: `- [docs/REMOTE.md](docs/REMOTE.md) — using the server from another machine`.

`docs/proposals/remote-gpu-server.md`: change the Status line to:

```markdown
Status: **accepted and implemented** - see the design in
[docs/superpowers/specs/2026-09-04-remote-gpu-server-design.md](../superpowers/specs/2026-09-04-remote-gpu-server-design.md)
and the user guide in [REMOTE.md](../REMOTE.md). The notes below are the
original scoping and are kept for the reasoning; one claim in them turned
out wrong: direct-LAN browser access did **not** already work, because the
`Origin` check accepted loopback origins only. Decisions taken: MCP both as
stdio and mounted in `dw.serve --mcp`; plain HTTP + token on the LAN with a
reverse proxy or Tailscale beyond it; a shipped systemd unit; install still
via clone + install.sh.
```

- [ ] **Step 8: Consistency check**

Run: `grep -rn "must not be pointed\|Localhost only" docs/ CLAUDE.md` - Expected: no matches. Run: `grep -rn "REMOTE.md" docs/ README.md CLAUDE.md contrib/ | wc -l` - Expected: ≥ 6. Run `pytest tests/test_mcp_server.py tests/test_server_mcp.py -q -p no:cacheprovider` (doc edits to `dw_mcp/server.py` were in Task 4; this just confirms nothing drifted).

- [ ] **Step 9: Smoke test on this machine**

```bash
source ./activate
DW_API_TOKEN=abc dw-serve --host 0.0.0.0 --mcp --workflow-dir workflows &
sleep 8
curl -s http://$(hostname):8765/api/health -H "Authorization: Bearer abc"   # expect mcp:true, hostname
curl -s -o /dev/null -w "%{http_code}\n" http://$(hostname):8765/mcp        # expect 401
curl -s -X POST http://$(hostname):8765/mcp -H "Authorization: Bearer abc" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
kill %1
```

Expected: health JSON with `"mcp": true`; `401`; an `initialize` result naming `diffusers-workflow`. Record the actual output in the commit message body if anything differs from the plan.

- [ ] **Step 10: Commit**

```bash
git add contrib/ docs/REMOTE.md docs/MCP.md docs/SERVER.md docs/SECURITY.md CLAUDE.md README.md docs/proposals/remote-gpu-server.md
git commit -m "docs: remote GPU server guide, systemd unit, MCP security rewrite

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Verification on the GPU box (after merge)

1. `git pull && bash ./install.sh` on the box; install the unit per `contrib/systemd/README.md`.
2. From the laptop: `curl http://<box>:8765/api/health -H "Authorization: Bearer <token>"` shows the box's hostname and `"device": "cuda"`.
3. Browser: open `http://<box>:8765`, paste the token, run `workflows/sd15.json`. The POST must return 201, not 403 - that is Task 1 working.
4. `claude mcp add --transport http dw http://<box>:8765/mcp --header "Authorization: Bearer <token>"`, new session, `/mcp` shows connected, `get_health` returns the box's hostname, `list_workflows` returns the catalog, `run_workflow` on `sd15.json` with `acknowledged_cost=true` queues a job and `wait_for_job` sees it succeed.
