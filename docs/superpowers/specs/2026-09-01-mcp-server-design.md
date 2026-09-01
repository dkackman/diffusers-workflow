# MCP Server — Design

Date: 2026-09-01
Scope doc: [../scope/mcp.md](../scope/mcp.md)

## Summary

Add a fourth way to drive the workflow engine: a stdio **MCP server**
(`dw/mcp/`) that wraps the existing `dw.serve` REST API in a bounded,
structured tool surface. An MCP client — Claude Code first — can list and
read workflows, introspect real pipeline signatures, validate and save a
workflow, queue a run, poll its progress, and look at the image it
produced, without shell access or a repo checkout.

One server-side endpoint is added; everything else maps onto routes that
already exist.

## Architecture

### Topology

The MCP server is an HTTP client of a **running** `dw.serve`. It holds no
job state, owns no GPU worker, and starts no subprocess. Base URL comes
from `--url`, else `DW_MCP_URL`, else `http://127.0.0.1:8765`.

Embedding the FastAPI app in-process was rejected: `create_app()`
constructs its own `JobManager` and GPU worker, so an in-process copy
alongside a real `dw.serve` would put two workers on one GPU, which the
single-job-at-a-time design cannot tolerate.

When the API is unreachable, every tool fails with one specific message
naming the URL tried and telling the user to start `dw-serve` — not a raw
`httpx.ConnectError`.

### Module layout

| File | Responsibility | Depends on |
| --- | --- | --- |
| `dw/mcp/__init__.py` | package marker, version re-export | — |
| `dw/mcp/client.py` | `DwClient`: `httpx.Client` wrapper. One method per REST call. Owns base URL, timeouts, and HTTP-status→`ToolError` translation. Knows nothing about MCP. | `httpx` |
| `dw/mcp/catalog.py` | read-only tool handlers | `client` |
| `dw/mcp/authoring.py` | `validate_workflow`, `save_workflow`, `delete_workflow` | `client` |
| `dw/mcp/diagnose.py` | `run_workflow`, `get_job`, `get_job_events`, `cancel_job`, `rerun_job`, `move_job` | `client` |
| `dw/mcp/media.py` | fetch `/outputs/{name}`, downscale, return an MCP image content block | `client`, `PIL` |
| `dw/mcp/server.py` | builds the `FastMCP` server, registers every tool with its annotations and description | all of the above |
| `dw/mcp/__main__.py` | argument parsing, stdio transport, `python -m dw.mcp` | `server` |

Each handler module exports plain functions taking `(client, **args)` and
returning JSON-serializable values, so they are testable without an MCP
session. `server.py` is the only file that imports the MCP SDK.

### Data flow

```
MCP client (Claude Code)
  └─ stdio ─> dw/mcp/server.py  (tool dispatch, annotations)
                └─> handler fn  (catalog / authoring / diagnose / media)
                      └─> DwClient  (httpx)
                            └─> http://127.0.0.1:8765  (dw.serve)
                                  └─> JobManager -> GPU worker
```

## Server-side change

`GET /api/jobs/{job_id}/event-log?after=-1&limit=200`

```json
{ "id": "...", "status": "running", "events": [ ... ],
  "last_seq": 41, "truncated": false, "note": null }
```

- Live job: `job.events_after(after)`, capped at `limit`; `truncated` is
  true when more remain, so the caller knows to page.
- Historical job (a dict from `JobHistory`): `events: []`, `truncated`
  false, and `note` explaining the trail was not retained across the
  process — the honest answer rather than an empty one. Follow-up F1 in
  the scope doc removes this case.
- Unknown job: 404, matching the sibling routes.

A separate path rather than a `?stream=false` flag on the SSE route, so
each route keeps one response type in the OpenAPI schema.

The web UI is not changed by this work; the endpoint is additive.

## Tool surface

Annotations are set on every tool. `readOnlyHint: true` on everything in
Catalog and on `validate_workflow`. `openWorldHint: false` throughout —
the server talks only to a known local API.

### Catalog (read-only)

`list_workflows`, `get_workflow(name)`, `get_schema`, `list_pipelines`,
`get_pipeline_signature(name)`, `list_classes(kind=None)`,
`get_class(name)`, `list_tasks`, `get_task(command)`, `list_models`,
`get_memory`, `get_health`, `list_jobs`, `list_gallery(limit=50)`,
`get_gallery_metadata(name)`. (`get_job` is read-only too, but lives in
`diagnose.py` with the rest of the job loop.)

### Media

`get_output_image(name, max_dimension=768)` — GETs `/outputs/{name}`,
downscales with PIL preserving aspect ratio, re-encodes as PNG (JPEG when
the source is JPEG), and returns an MCP `ImageContent` block. Refuses
non-image media types with a message naming the actual type, so video
outputs fail clearly rather than returning garbage. A hard byte ceiling
(~4 MB after re-encode) guards the client's context.

### Authoring

- `validate_workflow(workflow=None, name=None)` — exactly one of the two;
  `name` fetches then validates. Free, no GPU.
- `save_workflow(name, workflow)` — `PUT`. `destructiveHint: true`
  (overwrites), `idempotentHint: true`.
- `delete_workflow(name)` — `DELETE`. `destructiveHint: true`.

Path safety is the server's job and stays there: `dw/security.py` already
confines these to the workflow directory. The MCP layer adds no second
validation, only clear error text when the server refuses.

### Diagnose

- `run_workflow(workflow_path=None, inline_workflow=None, arguments=None,
  acknowledged_cost=False)` — exactly one workflow source. Returns
  `{job_id, status, queue_position}` **immediately**; it never waits for
  the job. Refuses with a cost explanation unless `acknowledged_cost` is
  true. `base_dir` is not exposed.
- `get_job_events(job_id, after=-1, limit=200)` — the new endpoint.
- `cancel_job(job_id)`, `rerun_job(job_id)`, `move_job(job_id, direction)`.

### Confirm-gating

Three layers, in order of preference:

1. **Annotations** — the client can surface that this tool is not
   read-only before calling.
2. **Elicitation** — when the client advertises the capability (MCP
   2025-06-18), `run_workflow` elicits an explicit confirmation.
3. **`acknowledged_cost`** — a required-to-be-true argument, the floor
   that works on every client. Claude Code does not implement elicitation
   today, so in practice this is the gate that fires.

The tool description states the cost plainly and instructs the agent to
run `validate_workflow` first.

## Error handling

`DwClient` translates once, centrally:

| Condition | Result |
| --- | --- |
| `httpx.ConnectError` | "Cannot reach diffusers-workflow at `{url}`. Start the server with `dw-serve`." |
| `httpx.TimeoutException` | names the operation and the timeout |
| 400 | the server's `detail` verbatim — it is already a user-facing validation message |
| 404 | "No such {kind}: {name}" |
| 409 | the server's `detail` (e.g. a job that has left the queue) |
| 5xx | status plus body, labeled a server-side failure |

Handlers do not catch these; they propagate as MCP tool errors with the
message intact.

## Packaging

- `pyproject.toml`: new optional extra `mcp = ["mcp", "httpx>=0.28.1"]`, the `mcp` floor pinned from the
  installed version per `scripts/refresh_dep_floors.py` (the SDK is not
  currently installed in this venv);
  `mcp` also added to `dev` so the tests always run; console script
  `dw-mcp = "dw.mcp.__main__:main"`.
- `docs/MCP.md` carries the client configuration snippet (`.mcp.json` for
  Claude Code, `claude_desktop_config.json` for Desktop).

## Testing

No GPU, no live server, no network. `httpx.MockTransport` backs `DwClient`
with a scripted fake of the REST API; MCP round-trips use the SDK's
in-memory client/server session pair so `list_tools` output and
annotations are asserted as a client actually sees them.

| File | Covers |
| --- | --- |
| `tests/test_mcp_client.py` | base-URL resolution, every error translation above, timeout config |
| `tests/test_mcp_catalog.py` | each read-only handler: success shape, 404 path |
| `tests/test_mcp_authoring.py` | validate (both sources, and the "exactly one" rule), save, delete, server-refusal messages |
| `tests/test_mcp_diagnose.py` | run gating (refusal without `acknowledged_cost`, non-blocking return), event paging, cancel/rerun/move |
| `tests/test_mcp_media.py` | downscale math, aspect ratio, format choice, byte ceiling, non-image refusal |
| `tests/test_mcp_server.py` | tool registration: every tool listed, annotations correct, input schemas well-formed, one end-to-end call per group |
| `tests/test_server.py` (extended) | the new event-log endpoint: live paging, `truncated`, historical `note`, 404 |

Bar: every tool has a success case, an error case, and a schema case.
`pytest --cov=dw.mcp` at 90% line coverage or better. The suite must stay
runnable without the `mcp` extra installed — `pytest.importorskip("mcp")`
in `test_mcp_server.py` only; the handler tests import no SDK.

## Documentation

| Document | Change |
| --- | --- |
| `docs/MCP.md` | **new** — what it is, install, client config, full tool reference, the confirm-gating contract, security posture, troubleshooting |
| `docs/SERVER.md` | document `GET /api/jobs/{id}/event-log` |
| `README.md` | add MCP as the fourth way to drive the engine |
| `CLAUDE.md` | an "MCP Server" subsection under Architecture |
| `docs/SECURITY.md`, `docs/SECURITY_QUICKREF.md` | MCP inherits the REST API's posture and adds no authentication; localhost only |
| `docs/TESTING.md` | how the MCP tests fake the API |

`tests/test_docs_links.py` already guards cross-document links, so new
links are checked automatically.

## Out of scope

Everything in the scope doc's "Post-v1 follow-ups" (F1–F7): event
persistence, run-time estimation, server-log exposure, model and prompt
tools, MCP resources/prompts, and the remote auth story. Also out: any
change to the web UI, and `base_dir` on inline runs.
