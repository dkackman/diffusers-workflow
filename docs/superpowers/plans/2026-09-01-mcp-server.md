# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a stdio MCP server (`dw/mcp/`) that wraps the existing `dw.serve` REST API so an MCP client can author, validate, save, run, and diagnose workflows without shell access.

**Architecture:** `dw/mcp/` is an HTTP client of a *running* `dw.serve` — it owns no job state, no GPU worker, and starts no subprocess. `client.py` speaks HTTP and translates errors; `catalog.py` / `authoring.py` / `diagnose.py` / `media.py` hold plain handler functions taking `(client, **kwargs)`; only `server.py` imports the MCP SDK, registering each handler as a tool with annotations. One new REST endpoint (`GET /api/jobs/{id}/event-log`) is added because the existing event stream is SSE-only.

**Tech Stack:** Python 3.10+, the `mcp` Python SDK (`FastMCP`, stdio transport), `httpx` (with `httpx.MockTransport` in tests), FastAPI (existing server), Pillow, pytest + pytest-asyncio.

**Spec:** [../specs/2026-09-01-mcp-server-design.md](../specs/2026-09-01-mcp-server-design.md)

## Global Constraints

- Handler modules (`client.py`, `catalog.py`, `authoring.py`, `diagnose.py`, `media.py`) **must not import the `mcp` SDK**. Only `server.py` and `__main__.py` may.
- Handlers are plain functions with the signature `handler(client, **kwargs)` returning JSON-serializable values. This is what makes them testable without an MCP session.
- No test may start a real server, touch a GPU, or open a network socket. `httpx.MockTransport` backs every client test.
- Every new tool carries MCP annotations. `readOnlyHint: True` for catalog + `validate_workflow`; `destructiveHint: True` for `save_workflow`, `delete_workflow`; `openWorldHint: False` on every tool.
- `run_workflow` **never blocks** waiting for a job, and refuses unless `acknowledged_cost is True`.
- `base_dir` is never exposed through any MCP tool.
- Path/security validation stays server-side in `dw/security.py`. The MCP layer adds no second validation layer — only clear error text.
- Format every touched Python file with `black dw/ tests/` before committing.
- Coverage bar: `pytest --cov=dw.mcp --cov-report=term-missing` at ≥90% line coverage on `dw/mcp/`.
- Follow-ups F1–F7 in [../scope/mcp.md](../scope/mcp.md) are **out of scope**. Do not implement them.

## Subagent Assignment

| Task | Suggested model | Why |
| --- | --- | --- |
| 1. Event-log endpoint | **Sonnet** | Mechanical: one route mirroring siblings, TDD against an existing test harness |
| 2. `DwClient` + packaging | **Sonnet** | Mechanical, but error-translation table must be followed exactly |
| 3. Catalog handlers | **Sonnet** | Repetitive one-line-per-route wrappers |
| 4. Media / image tool | **Sonnet** | Self-contained; the downscale loop is the only real logic |
| 5. Authoring handlers | **Sonnet** | Small; the "exactly one source" rule is the only subtlety |
| 6. Diagnose handlers + confirm-gating | **Opus** | Gating semantics are the design's load-bearing decision |
| 7. Server assembly + annotations + entry point | **Opus** | SDK surface, annotation correctness, async tool dispatch |
| 8. Documentation | **Sonnet** | Prose against a finished, verifiable surface |
| 9. Verification + review sweep | **Opus** | Cross-task consistency, coverage, spec coverage |

Dependency order: **1** and **2** are independent and may run in parallel. **3**, **4**, **5** each depend only on 2. **6** depends on 1 and 2. **7** depends on 3, 4, 5, 6. **8** depends on 7. **9** depends on everything.

---

### Task 1: Non-streaming job event-log endpoint

The MCP server needs job events in a request/response shape. Today they exist only behind SSE (`GET /api/jobs/{id}/events`), and `Job.detail()` returns `event_count`, not the events. This adds a sibling route.

**Files:**
- Modify: `dw/server/app.py` (insert after the `job_events` SSE route, which ends around line 386)
- Modify: `docs/SERVER.md`
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `GET /api/jobs/{job_id}/event-log?after=-1&limit=200` returning
  `{"id": str, "status": str, "events": list[dict], "last_seq": int, "truncated": bool, "note": str | None}`.
  Task 6 consumes this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`. The file already defines `valid_workflow()` and `ScriptedWorkerManager`; read the existing tests near the SSE tests and reuse whatever app/client fixture they use, matching that style exactly.

```python
def test_event_log_returns_events_for_a_live_job(client_and_manager):
    client, manager = client_and_manager
    job_id = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()["id"]
    wait_for_terminal(client, job_id)

    body = client.get(f"/api/jobs/{job_id}/event-log").json()

    assert body["id"] == job_id
    assert body["events"], "a completed job should have recorded events"
    assert [event["seq"] for event in body["events"]] == list(
        range(len(body["events"]))
    )
    assert body["last_seq"] == body["events"][-1]["seq"]
    assert body["truncated"] is False
    assert body["note"] is None


def test_event_log_pages_with_after_and_limit(client_and_manager):
    client, manager = client_and_manager
    job_id = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()["id"]
    wait_for_terminal(client, job_id)
    total = client.get(f"/api/jobs/{job_id}/event-log").json()["events"]
    assert len(total) >= 3, "test needs a job with at least three events"

    first = client.get(f"/api/jobs/{job_id}/event-log?limit=2").json()
    assert len(first["events"]) == 2
    assert first["truncated"] is True
    assert first["last_seq"] == 1

    rest = client.get(
        f"/api/jobs/{job_id}/event-log?after={first['last_seq']}"
    ).json()
    assert rest["events"][0]["seq"] == 2
    assert rest["truncated"] is False


def test_event_log_clamps_a_negative_after(client_and_manager):
    client, manager = client_and_manager
    job_id = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()["id"]
    wait_for_terminal(client, job_id)

    body = client.get(f"/api/jobs/{job_id}/event-log?after=-99").json()

    assert body["events"][0]["seq"] == 0


def test_event_log_explains_a_historical_job_has_no_trail(client_and_manager):
    """A job recovered from sqlite is a plain dict with no event list. Say so
    rather than returning an empty list that reads as 'nothing happened'."""
    client, manager = client_and_manager
    manager.get = lambda job_id: {"id": job_id, "status": "complete"}

    body = client.get("/api/jobs/historical/event-log").json()

    assert body["events"] == []
    assert body["truncated"] is False
    assert "not retained" in body["note"]


def test_event_log_404s_for_an_unknown_job(client_and_manager):
    client, _manager = client_and_manager

    response = client.get("/api/jobs/nope/event-log")

    assert response.status_code == 404
```

If `tests/test_server.py` has no `client_and_manager` fixture or `wait_for_terminal` helper, use whatever the neighbouring SSE tests use for the same purposes and rename accordingly — do not invent a second harness.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server.py -k event_log -v`
Expected: FAIL — every case 404s, because the route does not exist.

- [ ] **Step 3: Implement the route**

Insert into `dw/server/app.py` immediately after the `job_events` SSE handler:

```python
    @app.get("/api/jobs/{job_id}/event-log")
    def job_event_log(job_id: str, after: int = -1, limit: int = 200):
        """Job events as one JSON page rather than a stream, for clients that
        poll instead of holding a connection open (the MCP server). `after` is
        exclusive, matching the SSE route's parameter of the same name."""
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        if isinstance(job, dict):
            # A job restored from sqlite: history keeps the summary, not the
            # event log, so there is nothing to page through
            return {
                "id": job_id,
                "status": job.get("status"),
                "events": [],
                "last_seq": after,
                "truncated": False,
                "note": "This job's event log was not retained - events live "
                "only while the server process that ran the job is alive.",
            }
        limit = max(1, min(limit, 1000))
        pending = job.events_after(after)
        page = pending[:limit]
        return {
            "id": job_id,
            "status": job.status,
            "events": page,
            "last_seq": page[-1]["seq"] if page else max(after, -1),
            "truncated": len(pending) > len(page),
            "note": None,
        }
```

`job.events_after` already clamps an `after` below `-1` (`jobs.py:197`), so no extra clamping is needed here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_server.py -k event_log -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Run the full server suite for regressions**

Run: `pytest tests/test_server.py -q`
Expected: PASS, no failures.

- [ ] **Step 6: Document the endpoint**

Add to `docs/SERVER.md`, in the same table/section style the file already uses for `/api/jobs/{id}/events`:

```markdown
`GET /api/jobs/{id}/event-log?after=-1&limit=200` — the same events as the
SSE stream, as one JSON page: `{id, status, events, last_seq, truncated,
note}`. `after` is exclusive; page by passing back the previous `last_seq`.
For a job restored from history the list is empty and `note` explains that
events are not retained across a server restart.
```

- [ ] **Step 7: Format and commit**

```bash
black dw/ tests/
git add dw/server/app.py tests/test_server.py docs/SERVER.md
git commit -m "feat(server): add non-streaming job event-log endpoint"
```

---

### Task 2: `DwClient` HTTP layer and packaging

**Files:**
- Create: `dw/mcp/__init__.py`
- Create: `dw/mcp/client.py`
- Modify: `pyproject.toml`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces — every later task depends on these exact names:
  - `class DwApiError(Exception)` — the single error type handlers let propagate.
  - `class DwClient` with `__init__(self, base_url=None, timeout=30.0, transport=None)`; attribute `base_url: str`; methods
    `get_json(path, params=None) -> Any`,
    `post_json(path, payload=None) -> Any`,
    `put_json(path, payload) -> Any`,
    `delete_json(path) -> Any`,
    `get_bytes(path) -> tuple[bytes, str]` (body, content-type),
    `close() -> None`.
  - `def resolve_base_url(explicit=None) -> str` — `explicit`, else `$DW_MCP_URL`, else `http://127.0.0.1:8765`; trailing slashes stripped.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_client.py`:

```python
"""The HTTP layer under the MCP tools: URL resolution and the one place
every API failure is turned into a message a non-developer can act on."""

import httpx
import pytest

from dw.mcp.client import DwApiError, DwClient, resolve_base_url


def client_with(handler, **kwargs):
    return DwClient(transport=httpx.MockTransport(handler), **kwargs)


def test_resolve_base_url_prefers_the_explicit_value(monkeypatch):
    monkeypatch.setenv("DW_MCP_URL", "http://127.0.0.1:9999")
    assert resolve_base_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234"


def test_resolve_base_url_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("DW_MCP_URL", "http://127.0.0.1:9999/")
    assert resolve_base_url(None) == "http://127.0.0.1:9999"


def test_resolve_base_url_defaults_to_the_serve_port(monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    assert resolve_base_url(None) == "http://127.0.0.1:8765"


def test_get_json_returns_the_decoded_body():
    def handler(request):
        assert request.url.path == "/api/health"
        return httpx.Response(200, json={"status": "ok"})

    assert client_with(handler).get_json("/api/health") == {"status": "ok"}


def test_get_json_passes_query_parameters():
    def handler(request):
        assert request.url.params["limit"] == "5"
        return httpx.Response(200, json={"files": []})

    client_with(handler).get_json("/api/gallery", params={"limit": 5})


def test_post_put_and_delete_send_the_right_method_and_body():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.read()))
        return httpx.Response(200, json={"ok": True})

    client = client_with(handler)
    client.post_json("/api/validate", {"workflow": {"id": "w"}})
    client.put_json("/api/workflows/w", {"workflow": {"id": "w"}})
    client.delete_json("/api/workflows/w")

    assert [entry[0] for entry in seen] == ["POST", "PUT", "DELETE"]
    assert b'"id":"w"' in seen[0][2].replace(b" ", b"")


def test_get_bytes_returns_the_body_and_content_type():
    def handler(request):
        return httpx.Response(
            200, content=b"\x89PNG", headers={"content-type": "image/png"}
        )

    body, content_type = client_with(handler).get_bytes("/outputs/a.png")

    assert body == b"\x89PNG"
    assert content_type == "image/png"


def test_a_refused_connection_says_how_to_start_the_server():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = client_with(handler, base_url="http://127.0.0.1:8765")
    with pytest.raises(DwApiError) as caught:
        client.get_json("/api/health")

    message = str(caught.value)
    assert "http://127.0.0.1:8765" in message
    assert "dw-serve" in message


def test_a_timeout_names_the_request():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/models")

    assert "/api/models" in str(caught.value)
    assert "timed out" in str(caught.value).lower()


def test_a_400_surfaces_the_servers_detail_verbatim():
    def handler(request):
        return httpx.Response(400, json={"detail": "steps must be a list"})

    with pytest.raises(DwApiError) as caught:
        client_with(handler).post_json("/api/validate", {})

    assert str(caught.value) == "steps must be a list"


def test_a_404_names_what_was_missing():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown job"})

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/jobs/nope")

    assert "Unknown job" in str(caught.value)


def test_a_500_is_labelled_a_server_side_failure():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/memory")

    message = str(caught.value)
    assert "500" in message
    assert "boom" in message


def test_a_non_json_error_body_does_not_mask_the_status():
    def handler(request):
        return httpx.Response(404, text="<html>not found</html>")

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/workflows/x")

    assert "404" in str(caught.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.mcp'`.

- [ ] **Step 3: Implement the client**

Create `dw/mcp/__init__.py`:

```python
"""MCP server for diffusers-workflow.

A stdio MCP server that is an HTTP client of a running `dw.serve`. It owns
no job state and no GPU worker - every tool is a call against the REST API
that the web UI already uses.
"""

from dw import __version__

__all__ = ["__version__"]
```

Create `dw/mcp/client.py`:

```python
"""HTTP access to a running dw.serve, and the single place an API failure
becomes a message a non-developer can act on."""

import os

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8765"


class DwApiError(Exception):
    """A request to dw.serve failed. The message is meant to be read by the
    person driving the MCP client, not by a developer with a stack trace."""


def resolve_base_url(explicit=None):
    """Where dw.serve is: the explicit value, else DW_MCP_URL, else the
    default port."""
    url = explicit or os.environ.get("DW_MCP_URL") or DEFAULT_BASE_URL
    return url.rstrip("/")


class DwClient:
    """One method per kind of REST call. Knows nothing about MCP - the tool
    handlers are plain functions over this."""

    def __init__(self, base_url=None, timeout=30.0, transport=None):
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    def close(self):
        self._http.close()

    # ------------------------------------------------------------- requests

    def get_json(self, path, params=None):
        return self._json(self._request("GET", path, params=params), path)

    def post_json(self, path, payload=None):
        return self._json(self._request("POST", path, json=payload or {}), path)

    def put_json(self, path, payload):
        return self._json(self._request("PUT", path, json=payload), path)

    def delete_json(self, path):
        return self._json(self._request("DELETE", path), path)

    def get_bytes(self, path):
        """Raw body plus content type - for the output media served from the
        /outputs static mount rather than an /api route."""
        response = self._request("GET", path)
        self._raise_for_status(response, path)
        return response.content, response.headers.get("content-type", "")

    # ------------------------------------------------------------ internals

    def _request(self, method, path, **kwargs):
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.ConnectError:
            raise DwApiError(
                f"Cannot reach diffusers-workflow at {self.base_url}. "
                "Start the server with `dw-serve` (or `python -m dw.serve`) "
                "and try again."
            )
        except httpx.TimeoutException:
            raise DwApiError(
                f"Request to {path} timed out after {self.timeout}s. The "
                "server may be busy loading a model."
            )
        except httpx.HTTPError as e:
            raise DwApiError(f"Request to {path} failed: {e}")

    def _json(self, response, path):
        self._raise_for_status(response, path)
        try:
            return response.json()
        except ValueError:
            raise DwApiError(f"{path} returned a non-JSON body: {response.text[:200]}")

    def _raise_for_status(self, response, path):
        if response.status_code < 400:
            return
        detail = None
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = body.get("detail")
        except ValueError:
            detail = None
        if detail:
            # The API writes these for humans already - 400s carry validation
            # messages, 404s and 409s carry the reason
            raise DwApiError(str(detail))
        raise DwApiError(
            f"{path} failed with HTTP {response.status_code}: "
            f"{response.text[:200] or 'no body'}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_client.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Add the packaging entries**

In `pyproject.toml`, under `[project.optional-dependencies]`, add the extra after `server` and add `mcp` to `dev`:

```toml
mcp = ["mcp", "httpx>=0.28.1"]
```

Pin the `mcp` floor to the version that actually installs (`pip install mcp`, then `python -c "import importlib.metadata as m; print(m.version('mcp'))"`) and write it as `mcp>=<that version>`, matching how every other floor in this file is set (see the comment above `dependencies` and `scripts/refresh_dep_floors.py`). Add `"mcp>=<that version>"` to the `dev` list too, so the MCP tests always run in CI.

Then add the console script alongside the existing ones in `[project.scripts]`:

```toml
dw-mcp = "dw.mcp.__main__:main"
```

- [ ] **Step 6: Verify the package metadata still builds**

Run: `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print('ok')"`
Expected: `ok`

- [ ] **Step 7: Format and commit**

```bash
black dw/ tests/
git add dw/mcp/__init__.py dw/mcp/client.py tests/test_mcp_client.py pyproject.toml
git commit -m "feat(mcp): add DwClient HTTP layer and mcp packaging extra"
```

---

### Task 3: Catalog (read-only) handlers

**Files:**
- Create: `dw/mcp/catalog.py`
- Test: `tests/test_mcp_catalog.py`

**Interfaces:**
- Consumes: `DwClient` from Task 2 (`get_json`).
- Produces — each takes `client` first and returns the API body unchanged unless noted:
  `list_workflows(client)`, `get_workflow(client, name)`, `get_schema(client)`,
  `list_pipelines(client)`, `get_pipeline_signature(client, name)`,
  `list_classes(client, kind)`, `get_class(client, name, target="init")`,
  `list_tasks(client)`, `get_task(client, command)`, `list_models(client)`,
  `get_memory(client)`, `get_health(client)`, `list_jobs(client)`,
  `list_gallery(client, limit=50)`, `get_gallery_metadata(client, name)`.

Note `GET /api/classes` requires `kind` — the server has no default (`app.py:417`), so `kind` is a required parameter here too.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_catalog.py`:

```python
"""The read-only tools: each is a thin pass-through, so the tests pin the
route, the parameters, and that nothing is reshaped on the way back."""

import httpx
import pytest

from dw.mcp import catalog
from dw.mcp.client import DwApiError, DwClient


def recording_client(body=None, status=200):
    """A client whose transport records the request it was given."""
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(status, json=body if body is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


@pytest.mark.parametrize(
    "call, path",
    [
        (lambda c: catalog.list_workflows(c), "/api/workflows"),
        (lambda c: catalog.get_workflow(c, "folder/w"), "/api/workflows/folder/w"),
        (lambda c: catalog.get_schema(c), "/api/schema"),
        (lambda c: catalog.list_pipelines(c), "/api/pipelines"),
        (lambda c: catalog.get_pipeline_signature(c, "FluxPipeline"),
         "/api/pipelines/FluxPipeline"),
        (lambda c: catalog.list_classes(c, "schedulers"), "/api/classes"),
        (lambda c: catalog.get_class(c, "diffusers.AutoencoderKL"),
         "/api/classes/diffusers.AutoencoderKL"),
        (lambda c: catalog.list_tasks(c), "/api/tasks"),
        (lambda c: catalog.get_task(c, "upscale"), "/api/tasks/upscale"),
        (lambda c: catalog.list_models(c), "/api/models"),
        (lambda c: catalog.get_memory(c), "/api/memory"),
        (lambda c: catalog.get_health(c), "/api/health"),
        (lambda c: catalog.list_jobs(c), "/api/jobs"),
        (lambda c: catalog.list_gallery(c), "/api/gallery"),
        (lambda c: catalog.get_gallery_metadata(c, "a.png"),
         "/api/gallery/a.png/metadata"),
    ],
)
def test_each_catalog_tool_calls_its_route(call, path):
    client, seen = recording_client()

    call(client)

    assert seen["method"] == "GET"
    assert seen["path"] == path


def test_list_classes_sends_the_required_kind():
    client, seen = recording_client()

    catalog.list_classes(client, "quantization")

    assert seen["params"]["kind"] == "quantization"


def test_get_class_sends_the_target():
    client, seen = recording_client()

    catalog.get_class(client, "diffusers.FluxPipeline", target="call")

    assert seen["params"]["target"] == "call"


def test_list_gallery_sends_its_limit():
    client, seen = recording_client()

    catalog.list_gallery(client, limit=7)

    assert seen["params"]["limit"] == "7"


def test_a_pass_through_tool_returns_the_body_unchanged():
    client, _seen = recording_client({"workflows": ["a"], "details": {}})

    assert catalog.list_workflows(client) == {"workflows": ["a"], "details": {}}


def test_a_missing_workflow_propagates_the_api_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "No such workflow: ghost"})

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError, match="ghost"):
        catalog.get_workflow(client, "ghost")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_catalog.py -v`
Expected: FAIL — `ImportError: cannot import name 'catalog'`.

- [ ] **Step 3: Implement the handlers**

Create `dw/mcp/catalog.py`:

```python
"""Read-only tools: everything an agent can look at without spending GPU
time. Each is a pass-through - the API's shapes are already the ones the
web UI consumes, and reshaping them here would only add a second thing to
keep in sync."""


def list_workflows(client):
    """Workflow names in the server's workflow directory, with details."""
    return client.get_json("/api/workflows")


def get_workflow(client, name):
    """One workflow's full JSON definition."""
    return client.get_json(f"/api/workflows/{name}")


def get_schema(client):
    """The workflow JSON schema every definition is validated against."""
    return client.get_json("/api/schema")


def list_pipelines(client):
    """Every diffusers pipeline class this install exports."""
    return client.get_json("/api/pipelines")


def get_pipeline_signature(client, name):
    """A pipeline's real __call__ arguments - check before proposing a fix."""
    return client.get_json(f"/api/pipelines/{name}")


def list_classes(client, kind):
    """Class names of one kind: pipelines, models, schedulers, quantization."""
    return client.get_json("/api/classes", params={"kind": kind})


def get_class(client, name, target="init"):
    """A class's argument schema. target: init, call, or load."""
    return client.get_json(f"/api/classes/{name}", params={"target": target})


def list_tasks(client):
    """Every task command a workflow's task step can name."""
    return client.get_json("/api/tasks")


def get_task(client, command):
    """A task command's argument schema."""
    return client.get_json(f"/api/tasks/{command}")


def list_models(client):
    """What the Hugging Face hub cache holds, largest repo first."""
    return client.get_json("/api/models")


def get_memory(client):
    """Worker VRAM/RAM stats - the first thing to check on an OOM."""
    return client.get_json("/api/memory")


def get_health(client):
    """Server liveness."""
    return client.get_json("/api/health")


def list_jobs(client):
    """The live queue plus recent history, oldest first."""
    return client.get_json("/api/jobs")


def list_gallery(client, limit=50):
    """Generated media in the output directory, newest first."""
    return client.get_json("/api/gallery", params={"limit": limit})


def get_gallery_metadata(client, name):
    """Metadata embedded in a saved file: the full workflow that made it,
    plus the job that produced it when history remembers one."""
    return client.get_json(f"/api/gallery/{name}/metadata")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_catalog.py -v`
Expected: PASS, 20 tests.

- [ ] **Step 5: Format and commit**

```bash
black dw/ tests/
git add dw/mcp/catalog.py tests/test_mcp_catalog.py
git commit -m "feat(mcp): add read-only catalog tool handlers"
```

---

### Task 4: Output image handler

The agent cannot answer "why does this look wrong" without seeing the image. Output media is served from the `/outputs` **static mount** (`app.py:840`), not an `/api` route.

**Files:**
- Create: `dw/mcp/media.py`
- Test: `tests/test_mcp_media.py`

**Interfaces:**
- Consumes: `DwClient.get_bytes` and `DwApiError` from Task 2.
- Produces: `get_output_image(client, name, max_dimension=768) -> dict` with keys
  `{"name": str, "data": str (base64), "mime_type": "image/png" | "image/jpeg", "original_size": [w, h], "returned_size": [w, h], "bytes": int}`.
  Task 7 wraps this into an MCP image content block — this module must **not** import the MCP SDK.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_media.py`:

```python
"""Returning a generated image to the agent: downscaled enough to be worth
a context window, honest about what it refuses."""

import base64
import io

import httpx
import pytest
from PIL import Image

from dw.mcp.client import DwApiError, DwClient
from dw.mcp.media import MAX_RETURNED_BYTES, get_output_image


def png_bytes(width, height, color=(120, 30, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 10, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


def serving(content, content_type):
    def handler(request):
        return httpx.Response(
            200, content=content, headers={"content-type": content_type}
        )

    return DwClient(transport=httpx.MockTransport(handler))


def decoded(result):
    return Image.open(io.BytesIO(base64.b64decode(result["data"])))


def test_a_large_image_is_downscaled_to_max_dimension():
    client = serving(png_bytes(2048, 1024), "image/png")

    result = get_output_image(client, "big.png", max_dimension=512)

    assert decoded(result).size == (512, 256)
    assert result["original_size"] == [2048, 1024]
    assert result["returned_size"] == [512, 256]


def test_the_taller_side_governs_the_downscale():
    client = serving(png_bytes(600, 1200), "image/png")

    result = get_output_image(client, "tall.png", max_dimension=600)

    assert decoded(result).size == (300, 600)


def test_a_small_image_is_returned_at_its_own_size():
    client = serving(png_bytes(64, 48), "image/png")

    result = get_output_image(client, "small.png", max_dimension=768)

    assert decoded(result).size == (64, 48)
    assert result["returned_size"] == [64, 48]


def test_a_jpeg_source_comes_back_as_jpeg():
    client = serving(jpeg_bytes(300, 300), "image/jpeg")

    result = get_output_image(client, "photo.jpg")

    assert result["mime_type"] == "image/jpeg"


def test_a_png_source_comes_back_as_png():
    client = serving(png_bytes(300, 300), "image/png")

    assert get_output_image(client, "a.png")["mime_type"] == "image/png"


def test_the_result_stays_under_the_byte_ceiling():
    """A hard cap matters more than fidelity - a payload over the ceiling
    would crowd out the conversation it is meant to inform."""
    import random

    noise = Image.new("RGB", (4000, 4000))
    noise.putdata(
        [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(4000 * 4000)
        ]
    )
    buffer = io.BytesIO()
    noise.save(buffer, format="PNG")
    client = serving(buffer.getvalue(), "image/png")

    result = get_output_image(client, "noise.png", max_dimension=4000)

    assert result["bytes"] <= MAX_RETURNED_BYTES
    assert len(base64.b64decode(result["data"])) == result["bytes"]


def test_a_video_output_is_refused_by_name():
    client = serving(b"\x00\x00\x00\x18ftypmp42", "video/mp4")

    with pytest.raises(DwApiError) as caught:
        get_output_image(client, "clip.mp4")

    assert "video/mp4" in str(caught.value)


def test_an_undecodable_body_is_refused_clearly():
    client = serving(b"not an image at all", "image/png")

    with pytest.raises(DwApiError, match="could not be decoded"):
        get_output_image(client, "broken.png")


def test_a_missing_file_propagates_the_api_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown file"})

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError, match="Unknown file"):
        get_output_image(client, "ghost.png")


def test_the_name_is_url_quoted_in_the_request():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(
            200, content=png_bytes(10, 10), headers={"content-type": "image/png"}
        )

    get_output_image(DwClient(transport=httpx.MockTransport(handler)), "a b#1.png")

    assert "%23" in seen["path"] or "#" not in seen["path"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_media.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.mcp.media'`.

- [ ] **Step 3: Implement the handler**

Create `dw/mcp/media.py`:

```python
"""Hand a generated image back to the agent.

Output media is served from the /outputs static mount rather than an /api
route, so this is the one tool that reaches outside /api. Everything is
downscaled and capped before it is returned: a full-resolution render would
cost more context than the answer it is meant to support.
"""

import base64
import io
from urllib.parse import quote

from PIL import Image

from dw.mcp.client import DwApiError

# Roughly 4MB of encoded image. Past this the payload crowds out the
# conversation it is supposed to inform
MAX_RETURNED_BYTES = 4 * 1024 * 1024
MIN_DIMENSION = 64


def get_output_image(client, name, max_dimension=768):
    """One image from the output directory, downscaled, as base64 plus the
    sizes it went in and came out at."""
    body, content_type = client.get_bytes(f"/outputs/{quote(name)}")
    if content_type and not content_type.startswith("image/"):
        raise DwApiError(
            f"{name} is {content_type}, not an image - this tool returns "
            "images only. Use get_gallery_metadata to inspect other media."
        )
    try:
        image = Image.open(io.BytesIO(body))
        image.load()
    except Exception:
        raise DwApiError(f"{name} could not be decoded as an image.")

    original_size = [image.width, image.height]
    fmt = "JPEG" if (image.format or "").upper() == "JPEG" else "PNG"
    if image.mode not in ("RGB", "L") and fmt == "JPEG":
        image = image.convert("RGB")

    limit = max(MIN_DIMENSION, int(max_dimension))
    encoded, sized = _encode_within_budget(image, limit, fmt)
    return {
        "name": name,
        "data": base64.b64encode(encoded).decode("ascii"),
        "mime_type": "image/jpeg" if fmt == "JPEG" else "image/png",
        "original_size": original_size,
        "returned_size": [sized.width, sized.height],
        "bytes": len(encoded),
    }


def _encode_within_budget(image, limit, fmt):
    """Shrink until the encoded bytes fit the ceiling. Two loops rather than
    one calculation because compressed size does not follow from pixel count
    - noise and flat colour differ by an order of magnitude."""
    while True:
        sized = _fit(image, limit)
        buffer = io.BytesIO()
        sized.save(buffer, format=fmt)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_RETURNED_BYTES or limit <= MIN_DIMENSION:
            return encoded, sized
        limit = max(MIN_DIMENSION, limit // 2)


def _fit(image, limit):
    """A copy no larger than `limit` on its longest side, aspect preserved.
    An image already inside the limit is returned as-is - upscaling would
    invent detail the model would then reason about."""
    longest = max(image.width, image.height)
    if longest <= limit:
        return image
    scale = limit / longest
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_media.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Format and commit**

```bash
black dw/ tests/
git add dw/mcp/media.py tests/test_mcp_media.py
git commit -m "feat(mcp): return downscaled output images to the agent"
```

---

### Task 5: Authoring handlers

**Files:**
- Create: `dw/mcp/authoring.py`
- Test: `tests/test_mcp_authoring.py`

**Interfaces:**
- Consumes: `DwClient`, `DwApiError` (Task 2); `catalog.get_workflow` (Task 3).
- Produces:
  `validate_workflow(client, workflow=None, name=None) -> dict` (the API's `{"valid", "error", "warnings"}`),
  `save_workflow(client, name, workflow) -> dict`,
  `delete_workflow(client, name) -> dict`.

`POST /api/validate` rejects a request without an inline workflow (`app.py:442`), so validating by `name` must fetch the definition first and post it inline.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_authoring.py`:

```python
"""Authoring tools: validate before saving, and never guess which workflow
the caller meant."""

import httpx
import pytest

from dw.mcp import authoring
from dw.mcp.client import DwApiError, DwClient


WORKFLOW = {"id": "w", "steps": []}


def scripted(routes):
    """routes: {(method, path): (status, json_body)}"""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append(key)
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_validate_posts_an_inline_workflow():
    client, seen = scripted(
        {("POST", "/api/validate"): (200, {"valid": True, "error": None, "warnings": []})}
    )

    result = authoring.validate_workflow(client, workflow=WORKFLOW)

    assert result["valid"] is True
    assert seen == [("POST", "/api/validate")]


def test_validate_by_name_fetches_then_posts_inline():
    """/api/validate refuses a request that carries only a path, so a
    name has to be resolved to a definition first."""
    client, seen = scripted(
        {
            ("GET", "/api/workflows/mine"): (200, WORKFLOW),
            ("POST", "/api/validate"): (
                200,
                {"valid": True, "error": None, "warnings": []},
            ),
        }
    )

    authoring.validate_workflow(client, name="mine")

    assert seen == [("GET", "/api/workflows/mine"), ("POST", "/api/validate")]


def test_validate_refuses_both_sources_at_once():
    client, _seen = scripted({})

    with pytest.raises(DwApiError, match="exactly one"):
        authoring.validate_workflow(client, workflow=WORKFLOW, name="mine")


def test_validate_refuses_neither_source():
    client, _seen = scripted({})

    with pytest.raises(DwApiError, match="exactly one"):
        authoring.validate_workflow(client)


def test_validate_returns_an_invalid_verdict_rather_than_raising():
    """An invalid workflow is the answer the agent asked for, not a failure."""
    client, _seen = scripted(
        {
            ("POST", "/api/validate"): (
                200,
                {"valid": False, "error": "steps must not be empty", "warnings": []},
            )
        }
    )

    result = authoring.validate_workflow(client, workflow=WORKFLOW)

    assert result["valid"] is False
    assert "steps" in result["error"]


def test_save_puts_the_definition_under_its_name():
    body_seen = {}

    def handler(request):
        body_seen["method"] = request.method
        body_seen["path"] = request.url.path
        body_seen["body"] = request.read()
        return httpx.Response(200, json={"name": "mine", "path": "/w/mine.json",
                                         "warnings": []})

    client = DwClient(transport=httpx.MockTransport(handler))

    result = authoring.save_workflow(client, "mine", WORKFLOW)

    assert body_seen["method"] == "PUT"
    assert body_seen["path"] == "/api/workflows/mine"
    assert b'"workflow"' in body_seen["body"]
    assert result["name"] == "mine"


def test_save_surfaces_a_rejected_definition():
    client, _seen = scripted(
        {("PUT", "/api/workflows/mine"): (400, {"detail": "steps must be a list"})}
    )

    with pytest.raises(DwApiError, match="steps must be a list"):
        authoring.save_workflow(client, "mine", WORKFLOW)


def test_save_surfaces_a_path_the_server_refuses():
    client, _seen = scripted(
        {
            ("PUT", "/api/workflows/../escape"): (
                400,
                {"detail": "Path traversal is not allowed"},
            )
        }
    )

    with pytest.raises(DwApiError, match="traversal"):
        authoring.save_workflow(client, "../escape", WORKFLOW)


def test_delete_calls_delete():
    client, seen = scripted(
        {("DELETE", "/api/workflows/mine"): (200, {"name": "mine", "deleted": True})}
    )

    assert authoring.delete_workflow(client, "mine")["deleted"] is True
    assert seen == [("DELETE", "/api/workflows/mine")]


def test_delete_surfaces_a_missing_workflow():
    client, _seen = scripted(
        {("DELETE", "/api/workflows/ghost"): (404, {"detail": "No such workflow"})}
    )

    with pytest.raises(DwApiError, match="No such workflow"):
        authoring.delete_workflow(client, "ghost")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_authoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'authoring'`.

- [ ] **Step 3: Implement the handlers**

Create `dw/mcp/authoring.py`:

```python
"""Writing workflows. Validation is free and comes first; saving overwrites,
so it is annotated destructive at the tool layer.

Path confinement is the server's job (dw/security.py already refuses
traversal and anything outside the workflow directory). Nothing here
re-implements it - a second, subtly different check is how the two drift.
"""

from dw.mcp.catalog import get_workflow
from dw.mcp.client import DwApiError


def validate_workflow(client, workflow=None, name=None):
    """Schema- and signature-check a workflow without queuing anything.
    Give either an inline definition or the name of a stored one."""
    if (workflow is None) == (name is None):
        raise DwApiError(
            "Provide exactly one of `workflow` (an inline definition) or "
            "`name` (a stored workflow)."
        )
    if workflow is None:
        # /api/validate only accepts an inline definition, so resolve first
        workflow = get_workflow(client, name)
    return client.post_json("/api/validate", {"workflow": workflow})


def save_workflow(client, name, workflow):
    """Write a workflow into the server's workflow directory, overwriting any
    file already under that name. The server validates before writing."""
    return client.put_json(f"/api/workflows/{name}", {"workflow": workflow})


def delete_workflow(client, name):
    """Remove a workflow from the server's workflow directory."""
    return client.delete_json(f"/api/workflows/{name}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_authoring.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Format and commit**

```bash
black dw/ tests/
git add dw/mcp/authoring.py tests/test_mcp_authoring.py
git commit -m "feat(mcp): add workflow validate/save/delete handlers"
```

---

### Task 6: Diagnose handlers and the run gate

**Model: Opus.** The gating contract is the design's load-bearing decision — a run costs real GPU time on a single-job-at-a-time engine.

**Files:**
- Create: `dw/mcp/diagnose.py`
- Test: `tests/test_mcp_diagnose.py`

**Interfaces:**
- Consumes: `DwClient`, `DwApiError` (Task 2); `GET /api/jobs/{id}/event-log` (Task 1).
- Produces:
  `run_workflow(client, workflow_path=None, inline_workflow=None, arguments=None, acknowledged_cost=False) -> dict`
  returning `{"job_id", "status", "queue_position", "next"}`;
  `get_job(client, job_id)`, `get_job_events(client, job_id, after=-1, limit=200)`,
  `cancel_job(client, job_id)`, `rerun_job(client, job_id)`,
  `move_job(client, job_id, direction)`.
  Also `COST_REFUSAL: str` — the message Task 7's tool description quotes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_diagnose.py`:

```python
"""Running and diagnosing jobs. The gate on run_workflow is the point: a
run costs GPU time on an engine that runs one job at a time."""

import httpx
import pytest

from dw.mcp import diagnose
from dw.mcp.client import DwApiError, DwClient


WORKFLOW = {"id": "w", "steps": []}
SUBMITTED = {"id": "job-1", "status": "queued", "queue_position": 2}


def scripted(routes):
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append({"key": key, "body": request.read(),
                     "params": dict(request.url.params)})
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def submitting():
    return scripted({("POST", "/api/jobs"): (201, SUBMITTED)})


def test_run_refuses_without_an_acknowledged_cost():
    """The client-agnostic floor of the confirm gate. Claude Code does not
    implement MCP elicitation today, so this is what actually fires."""
    client, seen = submitting()

    with pytest.raises(DwApiError) as caught:
        diagnose.run_workflow(client, workflow_path="w.json")

    assert "acknowledged_cost" in str(caught.value)
    assert seen == [], "nothing may be queued before the cost is acknowledged"


def test_run_submits_once_the_cost_is_acknowledged():
    client, seen = submitting()

    result = diagnose.run_workflow(
        client, workflow_path="w.json", acknowledged_cost=True
    )

    assert result["job_id"] == "job-1"
    assert result["status"] == "queued"
    assert result["queue_position"] == 2
    assert len(seen) == 1


def test_run_returns_immediately_rather_than_waiting_for_the_job():
    """A generation takes minutes; no MCP client will hold a call open. The
    contract is submit-then-poll, so exactly one request goes out."""
    client, seen = submitting()

    result = diagnose.run_workflow(
        client, workflow_path="w.json", acknowledged_cost=True
    )

    assert [entry["key"] for entry in seen] == [("POST", "/api/jobs")]
    assert "get_job_events" in result["next"]


def test_run_sends_an_inline_workflow_when_given_one():
    client, seen = submitting()

    diagnose.run_workflow(
        client, inline_workflow=WORKFLOW, acknowledged_cost=True
    )

    assert b'"workflow"' in seen[0]["body"]


def test_run_never_sends_base_dir():
    """base_dir decides where an inline workflow's relative paths resolve -
    a path-authority parameter the tool surface deliberately withholds."""
    client, seen = submitting()

    diagnose.run_workflow(
        client, inline_workflow=WORKFLOW, acknowledged_cost=True
    )

    assert b"base_dir" not in seen[0]["body"]


def test_run_refuses_both_workflow_sources():
    client, seen = submitting()

    with pytest.raises(DwApiError, match="exactly one"):
        diagnose.run_workflow(
            client,
            workflow_path="w.json",
            inline_workflow=WORKFLOW,
            acknowledged_cost=True,
        )
    assert seen == []


def test_run_refuses_neither_workflow_source():
    client, seen = submitting()

    with pytest.raises(DwApiError, match="exactly one"):
        diagnose.run_workflow(client, acknowledged_cost=True)
    assert seen == []


def test_run_passes_variable_overrides():
    client, seen = submitting()

    diagnose.run_workflow(
        client,
        workflow_path="w.json",
        arguments={"prompt": "a cat"},
        acknowledged_cost=True,
    )

    assert b"a cat" in seen[0]["body"]


def test_run_surfaces_a_rejected_workflow():
    client, _seen = scripted(
        {("POST", "/api/jobs"): (400, {"detail": "steps must not be empty"})}
    )

    with pytest.raises(DwApiError, match="steps must not be empty"):
        diagnose.run_workflow(
            client, inline_workflow=WORKFLOW, acknowledged_cost=True
        )


def test_get_job_returns_the_detail_payload():
    client, _seen = scripted(
        {("GET", "/api/jobs/job-1"): (200, {"id": "job-1", "status": "failed",
                                            "error": "CUDA out of memory"})}
    )

    assert diagnose.get_job(client, "job-1")["error"] == "CUDA out of memory"


def test_get_job_events_pages_from_the_event_log():
    client, seen = scripted(
        {
            ("GET", "/api/jobs/job-1/event-log"): (
                200,
                {"id": "job-1", "status": "running",
                 "events": [{"seq": 3, "event": "phase"}],
                 "last_seq": 3, "truncated": True, "note": None},
            )
        }
    )

    result = diagnose.get_job_events(client, "job-1", after=2, limit=50)

    assert seen[0]["params"] == {"after": "2", "limit": "50"}
    assert result["last_seq"] == 3
    assert result["truncated"] is True


def test_get_job_events_defaults_to_the_whole_log():
    client, seen = scripted(
        {("GET", "/api/jobs/job-1/event-log"): (200, {"events": [], "last_seq": -1})}
    )

    diagnose.get_job_events(client, "job-1")

    assert seen[0]["params"]["after"] == "-1"


def test_cancel_rerun_and_move_call_their_routes():
    client, seen = scripted(
        {
            ("POST", "/api/jobs/job-1/cancel"): (200, {"id": "job-1",
                                                       "status": "cancelled"}),
            ("POST", "/api/jobs/job-1/rerun"): (201, {"id": "job-2",
                                                      "status": "queued"}),
            ("POST", "/api/jobs/job-1/move"): (200, {"id": "job-1", "queue": []}),
        }
    )

    diagnose.cancel_job(client, "job-1")
    diagnose.rerun_job(client, "job-1")
    diagnose.move_job(client, "job-1", "front")

    assert [entry["key"][1] for entry in seen] == [
        "/api/jobs/job-1/cancel",
        "/api/jobs/job-1/rerun",
        "/api/jobs/job-1/move",
    ]
    assert b"front" in seen[2]["body"]


def test_move_surfaces_a_job_that_has_left_the_queue():
    client, _seen = scripted(
        {
            ("POST", "/api/jobs/job-1/move"): (
                409,
                {"detail": "Job is not queued - only queued jobs move"},
            )
        }
    )

    with pytest.raises(DwApiError, match="only queued jobs move"):
        diagnose.move_job(client, "job-1", "up")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_diagnose.py -v`
Expected: FAIL — `ImportError: cannot import name 'diagnose'`.

- [ ] **Step 3: Implement the handlers**

Create `dw/mcp/diagnose.py`:

```python
"""Queue a run and work out what happened.

Two rules shape this module. A run costs real GPU time on an engine that
runs one job at a time, so `run_workflow` refuses until the caller has
acknowledged that. And a generation takes minutes, longer than any MCP
client will hold a tool call open, so submitting returns immediately and
progress is polled from the event log.
"""

from dw.mcp.client import DwApiError

COST_REFUSAL = (
    "Running a workflow occupies the GPU for minutes and the engine runs one "
    "job at a time. Tell the user what is about to run, get their go-ahead, "
    "then call again with acknowledged_cost=true. `validate_workflow` is free "
    "and checks the definition first."
)


def run_workflow(
    client,
    workflow_path=None,
    inline_workflow=None,
    arguments=None,
    acknowledged_cost=False,
):
    """Queue a workflow. Returns as soon as it is queued - it does not wait
    for the job to finish. Poll `get_job_events` for progress."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    if (workflow_path is None) == (inline_workflow is None):
        raise DwApiError(
            "Provide exactly one of `workflow_path` (a workflow on the "
            "server) or `inline_workflow` (a definition to run as-is)."
        )
    payload = {"arguments": arguments or {}}
    if workflow_path is not None:
        payload["workflow_path"] = workflow_path
    else:
        payload["workflow"] = inline_workflow
    # base_dir is deliberately absent: it decides where an inline workflow's
    # relative paths resolve, and the MCP surface does not hand that out
    job = client.post_json("/api/jobs", payload)
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "queue_position": job.get("queue_position"),
        "next": "Poll get_job_events(job_id) for progress, then get_job(job_id) "
        "for the manifest or the error.",
    }


def get_job(client, job_id):
    """A job's status, arguments, warnings, manifest, error and traceback."""
    return client.get_json(f"/api/jobs/{job_id}")


def get_job_events(client, job_id, after=-1, limit=200):
    """One page of a job's progress events. `after` is exclusive - pass back
    the previous call's `last_seq` to continue."""
    return client.get_json(
        f"/api/jobs/{job_id}/event-log", params={"after": after, "limit": limit}
    )


def cancel_job(client, job_id):
    """Ask a queued or running job to stop."""
    return client.post_json(f"/api/jobs/{job_id}/cancel")


def rerun_job(client, job_id):
    """Queue a fresh job from a previous job's stored spec."""
    return client.post_json(f"/api/jobs/{job_id}/rerun")


def move_job(client, job_id, direction):
    """Reorder a queued job: up, down, front, or back."""
    return client.post_json(f"/api/jobs/{job_id}/move", {"direction": direction})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_diagnose.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Format and commit**

```bash
black dw/ tests/
git add dw/mcp/diagnose.py tests/test_mcp_diagnose.py
git commit -m "feat(mcp): add run/diagnose handlers with a cost gate on runs"
```

---

### Task 7: MCP server assembly and entry point

**Model: Opus.** This is the only code touching the SDK; annotation correctness and the elicitation fallback are the risk.

**Files:**
- Create: `dw/mcp/server.py`
- Create: `dw/mcp/__main__.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: every handler module (Tasks 3–6) and `DwClient` (Task 2).
- Produces: `build_server(client) -> FastMCP` and `main(argv=None) -> int`.

**Read first:** the installed SDK's `FastMCP` API — `pip show mcp`, then `python -c "from mcp.server.fastmcp import FastMCP; help(FastMCP.tool)"`. If `FastMCP.tool` does not accept an `annotations=` argument in the installed version, pass annotations however that version supports (check `mcp.types.ToolAnnotations`) and note the deviation in the commit message. Do not skip annotations.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_server.py`:

```python
"""The assembled MCP server: what a client actually sees when it connects."""

import json

import httpx
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw.mcp.client import DwClient  # noqa: E402
from dw.mcp.server import build_server  # noqa: E402


EXPECTED_TOOLS = {
    "list_workflows",
    "get_workflow",
    "get_schema",
    "list_pipelines",
    "get_pipeline_signature",
    "list_classes",
    "get_class",
    "list_tasks",
    "get_task",
    "list_models",
    "get_memory",
    "get_health",
    "list_jobs",
    "list_gallery",
    "get_gallery_metadata",
    "get_output_image",
    "validate_workflow",
    "save_workflow",
    "delete_workflow",
    "run_workflow",
    "get_job",
    "get_job_events",
    "cancel_job",
    "rerun_job",
    "move_job",
}

READ_ONLY_TOOLS = EXPECTED_TOOLS - {
    "save_workflow",
    "delete_workflow",
    "run_workflow",
    "cancel_job",
    "rerun_job",
    "move_job",
}

DESTRUCTIVE_TOOLS = {"save_workflow", "delete_workflow"}


def server_over(handler):
    client = DwClient(transport=httpx.MockTransport(handler))
    return build_server(client)


def ok(body):
    def handler(request):
        return httpx.Response(200, json=body)

    return handler


async def tools_of(server):
    return {tool.name: tool for tool in await server.list_tools()}


@pytest.mark.asyncio
async def test_every_designed_tool_is_registered():
    tools = await tools_of(server_over(ok({})))

    assert set(tools) == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_every_tool_has_a_description():
    tools = await tools_of(server_over(ok({})))

    missing = [name for name, tool in tools.items() if not (tool.description or "").strip()]
    assert missing == []


@pytest.mark.asyncio
async def test_read_only_tools_are_annotated_read_only():
    tools = await tools_of(server_over(ok({})))

    for name in READ_ONLY_TOOLS:
        assert tools[name].annotations is not None, name
        assert tools[name].annotations.readOnlyHint is True, name


@pytest.mark.asyncio
async def test_writing_tools_are_not_annotated_read_only():
    tools = await tools_of(server_over(ok({})))

    for name in EXPECTED_TOOLS - READ_ONLY_TOOLS:
        assert tools[name].annotations.readOnlyHint is not True, name


@pytest.mark.asyncio
async def test_overwriting_tools_are_annotated_destructive():
    tools = await tools_of(server_over(ok({})))

    for name in DESTRUCTIVE_TOOLS:
        assert tools[name].annotations.destructiveHint is True, name


@pytest.mark.asyncio
async def test_no_tool_claims_an_open_world():
    """Every tool talks to one known local server and nothing else."""
    tools = await tools_of(server_over(ok({})))

    for name, tool in tools.items():
        assert tool.annotations.openWorldHint is False, name


@pytest.mark.asyncio
async def test_no_tool_exposes_base_dir():
    tools = await tools_of(server_over(ok({})))

    for name, tool in tools.items():
        assert "base_dir" not in json.dumps(tool.inputSchema), name


@pytest.mark.asyncio
async def test_run_workflow_takes_an_acknowledged_cost_flag():
    tools = await tools_of(server_over(ok({})))

    assert "acknowledged_cost" in tools["run_workflow"].inputSchema["properties"]


@pytest.mark.asyncio
async def test_a_read_only_tool_round_trips_to_the_api():
    server = server_over(ok({"workflows": ["a"], "details": {}}))

    result = await server.call_tool("list_workflows", {})

    assert "workflows" in json.dumps(_text_of(result))


@pytest.mark.asyncio
async def test_run_workflow_refuses_without_acknowledgement():
    server = server_over(ok({"id": "job-1", "status": "queued"}))

    with pytest.raises(Exception) as caught:
        await server.call_tool("run_workflow", {"workflow_path": "w.json"})

    assert "acknowledged_cost" in str(caught.value)


@pytest.mark.asyncio
async def test_an_unreachable_server_reports_how_to_start_it():
    def refusing(request):
        raise httpx.ConnectError("refused", request=request)

    server = server_over(refusing)

    with pytest.raises(Exception) as caught:
        await server.call_tool("get_health", {})

    assert "dw-serve" in str(caught.value)


def _text_of(result):
    """FastMCP returns either a content list or a (content, structured) pair
    depending on version - normalise so the assertion reads the same."""
    payload = result[0] if isinstance(result, tuple) else result
    return [getattr(item, "text", "") for item in payload]
```

If `pytest-asyncio` needs an explicit mode, check `pyproject.toml`/`pytest.ini` for an existing `asyncio_mode` setting and follow it; if none exists, keep the `@pytest.mark.asyncio` markers and add `asyncio_mode = "auto"` only if the markers alone do not work.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.mcp.server'` (or a skip if the `mcp` extra is not installed; install it first with `pip install mcp`).

- [ ] **Step 3: Implement the server**

Create `dw/mcp/server.py`:

```python
"""Assemble the MCP tool surface over a DwClient.

The only module that imports the MCP SDK. Every tool body is a one-line
call into a handler, so the handlers stay testable without a session and
this file stays a description of the surface rather than logic.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, ToolAnnotations

from dw.mcp import authoring, catalog, diagnose, media

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITES = ToolAnnotations(readOnlyHint=False, openWorldHint=False)
OVERWRITES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)
DELETES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)


def build_server(client):
    """An MCP server whose tools all run against `client`."""
    server = FastMCP(
        "diffusers-workflow",
        instructions=(
            "Author, run and diagnose diffusers-workflow jobs against a "
            "running dw.serve. Validate a workflow before running it - "
            "validation is free, a run occupies the GPU for minutes."
        ),
    )

    def tool(fn, annotations, name=None):
        server.add_tool(fn, name=name or fn.__name__, annotations=annotations)

    # ------------------------------------------------------------- catalog

    def list_workflows() -> dict:
        """List the workflows stored on the server, with their descriptions,
        output kinds and variables."""
        return catalog.list_workflows(client)

    def get_workflow(name: str) -> dict:
        """Get one stored workflow's full JSON definition."""
        return catalog.get_workflow(client, name)

    def get_schema() -> dict:
        """Get the JSON schema every workflow definition must satisfy."""
        return catalog.get_schema(client)

    def list_pipelines() -> dict:
        """List every diffusers pipeline class this installation provides."""
        return catalog.list_pipelines(client)

    def get_pipeline_signature(name: str) -> dict:
        """Get a pipeline's real call arguments. Check this before proposing
        pipeline arguments - a plausible-looking argument that the pipeline
        does not accept is the most common workflow bug."""
        return catalog.get_pipeline_signature(client, name)

    def list_classes(kind: str) -> dict:
        """List class names of one kind: pipelines, models, schedulers, or
        quantization."""
        return catalog.list_classes(client, kind)

    def get_class(name: str, target: str = "init") -> dict:
        """Get a class's argument schema. target: init, call, or load."""
        return catalog.get_class(client, name, target=target)

    def list_tasks() -> dict:
        """List every task command a workflow's task step can name."""
        return catalog.list_tasks(client)

    def get_task(command: str) -> dict:
        """Get a task command's argument schema."""
        return catalog.get_task(client, command)

    def list_models() -> dict:
        """List what the Hugging Face model cache holds, largest first."""
        return catalog.list_models(client)

    def get_memory() -> dict:
        """Get the worker's VRAM and RAM statistics. Check this first when a
        job fails with an out-of-memory error."""
        return catalog.get_memory(client)

    def get_health() -> dict:
        """Check that the server is alive."""
        return catalog.get_health(client)

    def list_jobs() -> dict:
        """List queued, running and recent jobs."""
        return catalog.list_jobs(client)

    def list_gallery(limit: int = 50) -> dict:
        """List generated output files, newest first."""
        return catalog.list_gallery(client, limit=limit)

    def get_gallery_metadata(name: str) -> dict:
        """Get the metadata embedded in a generated file: the exact workflow
        and arguments that produced it. Use this to reproduce a bad result."""
        return catalog.get_gallery_metadata(client, name)

    for fn in (
        list_workflows,
        get_workflow,
        get_schema,
        list_pipelines,
        get_pipeline_signature,
        list_classes,
        get_class,
        list_tasks,
        get_task,
        list_models,
        get_memory,
        get_health,
        list_jobs,
        list_gallery,
        get_gallery_metadata,
    ):
        tool(fn, READ_ONLY)

    # --------------------------------------------------------------- media

    def get_output_image(name: str, max_dimension: int = 768) -> ImageContent:
        """Look at a generated image. Use this to judge output quality - it
        is the only way to see what a workflow actually produced. The image
        is downscaled to `max_dimension` on its longest side."""
        result = media.get_output_image(client, name, max_dimension=max_dimension)
        return ImageContent(
            type="image", data=result["data"], mimeType=result["mime_type"]
        )

    tool(get_output_image, READ_ONLY)

    # ----------------------------------------------------------- authoring

    def validate_workflow(workflow: dict = None, name: str = None) -> dict:
        """Check a workflow against the schema and against real pipeline
        signatures. Free and instant - always run this before run_workflow.
        Give exactly one of `workflow` or `name`."""
        return authoring.validate_workflow(client, workflow=workflow, name=name)

    def save_workflow(name: str, workflow: dict) -> dict:
        """Save a workflow to the server, overwriting any existing workflow
        of that name. Validate it first."""
        return authoring.save_workflow(client, name, workflow)

    def delete_workflow(name: str) -> dict:
        """Permanently delete a stored workflow."""
        return authoring.delete_workflow(client, name)

    tool(validate_workflow, READ_ONLY)
    tool(save_workflow, OVERWRITES)
    tool(delete_workflow, DELETES)

    # ------------------------------------------------------------ diagnose

    def run_workflow(
        workflow_path: str = None,
        inline_workflow: dict = None,
        arguments: dict = None,
        acknowledged_cost: bool = False,
    ) -> dict:
        """Queue a workflow for generation. THIS COSTS GPU TIME: a run
        occupies the machine for minutes and the engine runs one job at a
        time. Tell the user what will run and get their go-ahead, then pass
        acknowledged_cost=true. Returns as soon as the job is queued; poll
        get_job_events for progress. Give exactly one of `workflow_path` or
        `inline_workflow`."""
        return diagnose.run_workflow(
            client,
            workflow_path=workflow_path,
            inline_workflow=inline_workflow,
            arguments=arguments,
            acknowledged_cost=acknowledged_cost,
        )

    def get_job(job_id: str) -> dict:
        """Get a job's status, warnings, output manifest, error and
        traceback."""
        return diagnose.get_job(client, job_id)

    def get_job_events(job_id: str, after: int = -1, limit: int = 200) -> dict:
        """Get a page of a job's progress events - phase transitions, memory
        readings and log lines. `after` is exclusive: pass back the previous
        call's `last_seq` to continue."""
        return diagnose.get_job_events(client, job_id, after=after, limit=limit)

    def cancel_job(job_id: str) -> dict:
        """Ask a queued or running job to stop."""
        return diagnose.cancel_job(client, job_id)

    def rerun_job(job_id: str) -> dict:
        """Queue a fresh job from a previous job's stored specification."""
        return diagnose.rerun_job(client, job_id)

    def move_job(job_id: str, direction: str) -> dict:
        """Reorder a queued job: up, down, front, or back."""
        return diagnose.move_job(client, job_id, direction)

    tool(get_job, READ_ONLY)
    tool(get_job_events, READ_ONLY)
    for fn in (run_workflow, cancel_job, rerun_job, move_job):
        tool(fn, WRITES)

    return server
```

If `FastMCP.add_tool` in the installed SDK does not take `name=`/`annotations=`, use the `@server.tool(...)` decorator form instead — keep the handler bodies and docstrings exactly as written.

Create `dw/mcp/__main__.py`:

```python
"""`python -m dw.mcp` / `dw-mcp`: serve the tool surface over stdio."""

import argparse
import sys

from dw.mcp.client import DwClient, resolve_base_url
from dw.mcp.server import build_server


def main(argv=None):
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
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait on any one API request (default: 30)",
    )
    args = parser.parse_args(argv)

    client = DwClient(base_url=resolve_base_url(args.url), timeout=args.timeout)
    try:
        build_server(client).run(transport="stdio")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mcp_server.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Verify the entry point starts and stops**

Run: `python -m dw.mcp --help`
Expected: the argparse help, exit code 0.

- [ ] **Step 6: Verify against a real server end to end**

In one terminal: `python -m dw.serve`. In another:

```bash
python - <<'EOF'
import asyncio, json
from dw.mcp.client import DwClient
from dw.mcp.server import build_server

async def main():
    server = build_server(DwClient())
    names = sorted(tool.name for tool in await server.list_tools())
    print(len(names), "tools:", names)
    print(await server.call_tool("get_health", {}))
    print(await server.call_tool("list_workflows", {}))

asyncio.run(main())
EOF
```

Expected: 25 tools, a health payload, and the real workflow list. Then stop `dw.serve` and re-run: expect the "Start the server with `dw-serve`" message, not a traceback.

- [ ] **Step 7: Format and commit**

```bash
black dw/ tests/
git add dw/mcp/server.py dw/mcp/__main__.py tests/test_mcp_server.py
git commit -m "feat(mcp): assemble the MCP tool surface and stdio entry point"
```

---

### Task 8: Documentation

**Files:**
- Create: `docs/MCP.md`
- Modify: `README.md`, `CLAUDE.md`, `docs/SECURITY.md`, `docs/SECURITY_QUICKREF.md`, `docs/TESTING.md`

**Interfaces:**
- Consumes: the finished tool surface from Task 7. Every tool name and argument in the docs must match `dw/mcp/server.py` exactly.
- Produces: documentation only.

- [ ] **Step 1: Read the neighbouring docs for house style**

Read `docs/SERVER.md` and `docs/REPL_COMMANDS.md` in full before writing. Match their heading depth, table style, and voice. Note that `tests/test_docs_links.py` checks cross-document links, so every link must resolve.

- [ ] **Step 2: Write `docs/MCP.md`**

Cover, in this order:

1. **What it is** — a fourth way to drive the engine, alongside `dw.run`, `dw.repl`, and `dw.serve`. A stdio MCP server that is an HTTP client of a running `dw.serve`; it owns no GPU worker and starts no server of its own.
2. **Install and run** — `pip install -e ".[server,mcp]"`, then `dw-serve` in one terminal and `dw-mcp` as the client's configured command. Flags: `--url`, `--timeout`; environment variable `DW_MCP_URL`.
3. **Client configuration** — a `.mcp.json` block for Claude Code and a `claude_desktop_config.json` block for Claude Desktop, both invoking `dw-mcp`. Copy the exact JSON shape from the MCP documentation; do not invent field names.
4. **Tool reference** — one table per group (catalog, media, authoring, diagnose) with tool name, arguments, and one line of purpose. Generate the rows from `dw/mcp/server.py` so they cannot drift; all 25 tools must appear.
5. **The run gate** — `run_workflow` requires `acknowledged_cost=true`, returns as soon as the job is queued, and never waits. The polling loop: `validate_workflow` → `run_workflow` → `get_job_events` until a terminal status → `get_job` for the manifest, `get_output_image` to look at the result.
6. **Security** — inherits the REST API's posture exactly: localhost binding, no authentication, path confinement in `dw/security.py`, `Origin` checks. State plainly that the MCP server adds no authentication and must not be exposed beyond localhost. Link to `docs/SECURITY.md`.
7. **Known limits** — a job's event log does not survive a server restart (`get_job_events` returns a `note` saying so); `get_output_image` returns images only, not video; model download, dependency management and the prompt library are not exposed yet.
8. **Troubleshooting** — "Cannot reach diffusers-workflow at …" means `dw.serve` is not running; a tool timing out usually means a model is loading; a `run_workflow` refusal is the cost gate, not an error.

- [ ] **Step 3: Update the surrounding docs**

- `README.md` — in the section listing how to drive the engine, add MCP as the fourth mode with a one-line description and a link to `docs/MCP.md`.
- `CLAUDE.md` — add an "MCP Server" subsection under Architecture, after "Server & Web UI":

```markdown
### MCP Server

`dw/mcp/` is a stdio MCP server (`dw-mcp`, `python -m dw.mcp`) that wraps the
`dw.serve` REST API in a structured tool surface: workflow catalog and
introspection, validate/save/delete, queue a run, poll its events, and view a
generated image. It is an HTTP client of a *running* `dw.serve` — it owns no
job state and no GPU worker. Only `dw/mcp/server.py` imports the MCP SDK; the
handlers in `catalog.py`, `authoring.py`, `diagnose.py` and `media.py` are
plain `(client, **kwargs)` functions, which is what makes them testable
without an MCP session. `run_workflow` requires `acknowledged_cost=True` and
returns as soon as the job is queued — a generation outlasts any client's
tool-call timeout. See docs/MCP.md.
```

- `docs/SECURITY.md` and `docs/SECURITY_QUICKREF.md` — a short paragraph each: the MCP server introduces no new file access and no authentication; every path still goes through the server's own validation; localhost only, same as the REST API.
- `docs/TESTING.md` — a note that the MCP tests fake the REST API with `httpx.MockTransport` and never start a server or touch a GPU, and that `tests/test_mcp_server.py` skips when the `mcp` extra is absent.

- [ ] **Step 4: Verify every documented tool exists**

Run:

```bash
python - <<'EOF'
import asyncio, pathlib, re
from dw.mcp.client import DwClient
from dw.mcp.server import build_server

async def main():
    tools = {tool.name for tool in await build_server(DwClient()).list_tools()}
    documented = set(re.findall(r"`([a-z_]+)\(", pathlib.Path("docs/MCP.md").read_text()))
    documented &= tools | {name for name in documented if name in tools}
    print("undocumented:", sorted(tools - documented))

asyncio.run(main())
EOF
```

Expected: `undocumented: []`.

- [ ] **Step 5: Verify documentation links**

Run: `pytest tests/test_docs_links.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/MCP.md README.md CLAUDE.md docs/SECURITY.md docs/SECURITY_QUICKREF.md docs/TESTING.md
git commit -m "docs: document the MCP server surface, gate and limits"
```

---

### Task 9: Verification sweep

**Model: Opus.** Cross-task consistency is what a per-task reviewer cannot see.

**Files:**
- Modify: whatever the sweep finds. No new files expected.

**Interfaces:**
- Consumes: everything.
- Produces: a green suite at the coverage bar, and a written report.

- [ ] **Step 1: Run the whole suite**

Run: `pytest -q`
Expected: PASS, no failures, no new warnings attributable to `dw/mcp/`.

- [ ] **Step 2: Check coverage against the bar**

Run: `pytest tests/test_mcp_client.py tests/test_mcp_catalog.py tests/test_mcp_media.py tests/test_mcp_authoring.py tests/test_mcp_diagnose.py tests/test_mcp_server.py --cov=dw.mcp --cov-report=term-missing`
Expected: ≥90% line coverage on `dw/mcp/`. Add tests for any uncovered branch that represents real behaviour; do not add tests solely to move the number.

- [ ] **Step 3: Verify the suite still runs without the `mcp` extra**

Run: `pip uninstall -y mcp && pytest tests/ -k mcp -q && pip install mcp`
Expected: `tests/test_mcp_server.py` skips; every other MCP test passes. If a handler test fails, a handler module has imported the SDK — fix the import, since that violates a global constraint.

- [ ] **Step 4: Check the global constraints hold**

Run:

```bash
grep -rn "^import mcp\|^from mcp" dw/mcp/ | grep -v "server.py\|__main__.py"
grep -rn "base_dir" dw/mcp/
```

Expected: the first prints nothing; the second prints only the explanatory comment in `diagnose.py`.

- [ ] **Step 5: Check the spec is fully covered**

Read `docs/superpowers/specs/2026-09-01-mcp-server-design.md` alongside the implementation. Confirm each section has landed: topology, module layout, the event-log endpoint, all 25 tools, the three-layer gate, the error table, packaging, tests, docs. Confirm nothing from the "Out of scope" section (follow-ups F1–F7, `base_dir`, UI changes) was implemented.

- [ ] **Step 6: Format and report**

Run: `black --check dw/ tests/`
Expected: no files would be reformatted.

Report: tools registered, tests added, coverage figure, and any spec deviation with its reason.

- [ ] **Step 7: Commit any fixes**

```bash
git add -A
git commit -m "chore(mcp): verification sweep fixes"
```

---

## Notes for the executor

- **Elicitation is not implemented in this plan.** The design names it as gate layer 2, but Claude Code does not advertise the capability, so there is nothing to test against and an untested code path is worse than an absent one. Layers 1 (annotations) and 3 (`acknowledged_cost`) are implemented and tested. If the SDK version you install supports `Context.elicit` and you can exercise it in a test, adding it to `run_workflow` is in scope; adding it untested is not.
- **The `mcp` SDK version floor is unpinned in this plan** because the SDK is not installed in the development venv. Task 2 Step 5 pins it from the version that actually installs, matching `scripts/refresh_dep_floors.py`.
- **If `FastMCP`'s API differs** from what Task 7 assumes, adapt the registration mechanics but keep the surface identical: same 25 tool names, same arguments, same annotations, same docstrings. The tests in Task 7 assert the surface, not the mechanics.
