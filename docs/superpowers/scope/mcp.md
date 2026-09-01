# MCP Server for diffusers-workflow — Scoping Doc

## Background

`diffusers-workflow` (dkackman/diffusers-workflow) is a declarative JSON
workflow engine for Hugging Face Diffusers, with a local HTTP server
(`dw.serve`, default port 8765) that exposes a full REST API: workflow
CRUD, a job queue with SSE progress streaming, model management,
introspection of diffusers pipeline signatures, and a gallery of past
generations with embedded metadata.

Today, workflow authoring and debugging, for the project developer,
can happen via Claude Code in VS Code,
with direct repo/filesystem access: "create a workflow that does X," or
"workflow Y produces bad output, find out why" — Claude Code runs the
workflow, reads logs, checks GPU memory, and proposes fixes to code or
workflow config. It has even gone so far as to identify bad RAM causing
intermittent generation failures.

That flow works well for the developer (me), but it assumes a dev
environment: a repo checkout, a terminal, VS Code, comfort reading stack
traces. The next audience is **non-developer end users** of the packaged
product, who won't have any of that. They'll have the app (web UI /
`dw.serve`, or an embedded product surface), not a coding agent with shell
access.

## Goal

Turn this from a dev-only workflow authoring/debugging tool into a **non-dev-friendly MCP server** that exposes the same "author a workflow / run it / diagnose why it
failed or produced bad output" capability, but in a structured, bounded way that
doesn't require shell access or a dev environment. The MCP server will wrap the existing `dw.serve` HTTP API, exposing a constrained tool surface for workflow authoring and debugging, suitable for non-developer users.

The modes the underlying workflow can be authored and run today are::

- a text editor and CLI `python -m dw.run workflow.json`
- with command line REPL `python -m dw.repl`
- via the HTTP API `dw.serve` and existing web UI

So this adds a fourth mode, which would be cool if we turned this into an AI first product: an MCP server that can be driven by a Claude Code agent, or eventually a local LLM-backed agent, to author and run workflows without deep
knowledge of diffusers, my workflow engine, or any of that complexity. The MCP server would expose a structured tool surface for workflow authoring and debugging, suitable for non-developer users.

- via the MCP server (new) — structured tool surface

Expose the same "author a workflow / run it / diagnose why it failed or
produced bad output" capability as an **MCP server**, wrapping the existing
`dw.serve` HTTP API. This lets any MCP-compatible client — Claude Code
first, later Claude Desktop/claude.ai, and eventually a local-LLM-backed
client — drive the engine through a bounded, structured tool surface
instead of raw shell/API access.

## Why MCP (not just keep using Claude Code)

- Claude Code assumes a dev environment; end users won't have one.
- MCP tools are a defined, constrained surface (structured in/out, no
  arbitrary code execution) — appropriate once non-developers are the
  audience, and safer than raw shell access.
- The underlying HTTP API already exists and is already defensive
  (path validation, traversal blocking, `Origin` header checks, JSON size
  limits) — MCP adds a protocol layer in front of it, not new attack
  surface.
- Protocol-agnostic: same server can later be pointed at Claude Code, a
  future embedded chat UI in the product, or a local-LLM-backed client.

## Rollout sequence (decided)

1. **Start with MCP server + Claude Code** as the client, for continued dev
   use and to validate the tool surface.
2. **Move to embedded-in-app** later, once the product is packaged for
   non-developer users. This later phase will need its own auth story,
   since the server currently binds to `127.0.0.1` with no authentication
   (see Security, below) — fine for local/trusted-LAN use, not fine for a
   hosted/remote product.
3. **Local LLM support** stays a goal throughout — keep the option open
   for users who won't have/want a paid Claude subscription. MCP is
   protocol-agnostic, so this is a separate, later concern from the MCP
   server itself, not a blocker to starting.

## Proposed MCP tool set

Mapped to existing `dw.serve` routes — no new server-side endpoints needed
for the initial build.

### Read-only / low-cost (build first)

| Tool | Route(s) | Notes |
| --- | --- | --- |
| `list_workflows` | `GET /api/workflows` | descriptions, output kinds, variable counts |
| `get_workflow(name)` | `GET /api/workflows/{name}` | full JSON |
| `validate_workflow(json)` | `POST /api/validate` | schema + signature-level argument checks; no GPU cost — good pre-flight tool |
| `get_schema` | `GET /api/schema` | workflow JSON schema |
| `get_pipeline_signature(name)` | `GET /api/pipelines/{name}` | lets the agent check real pipeline arguments before proposing a config fix |
| `list_classes(kind)` / `get_class(name)` | `GET /api/classes?kind=...`, `GET /api/classes/{name}` | broader introspection beyond pipelines |
| `list_tasks` / `get_task(command)` | `GET /api/tasks`, `GET /api/tasks/{command}` | task-step argument schemas |
| `list_models` | `GET /api/models` | hub cache inventory: sizes, revisions, last-used |
| `get_memory` | `GET /api/memory` | worker VRAM/RAM stats — answers the GPU-diagnosis need directly, no new endpoint required |
| `get_health` | `GET /api/health` | liveness |
| `get_gallery_item(name)` | `GET /api/gallery/{name}/metadata` | pull back a prior run's exact workflow + seed, e.g. to reproduce a failure |

### Diagnostic loop (build second)

| Tool | Route(s) | Notes |
| --- | --- | --- |
| `run_workflow(workflow_path \| inline_workflow, arguments)` | `POST /api/jobs` | **Costs GPU time; one job runs at a time.** See confirm-gating below. |
| `get_job(job_id)` | `GET /api/jobs/{id}` | full detail: spec, events, manifest, error — primary diagnosis payload |
| `stream_job_events(job_id, after?)` | `GET /api/jobs/{id}/events` | SSE; `job_status`/`phase`/`memory`/`log` events cover exactly the "instrument the process" need (phase transitions: loading → generating → decoding → saving) |
| `cancel_job(job_id)` | `POST /api/jobs/{id}/cancel` | cooperative cancel |
| `rerun_job(job_id)` | `POST /api/jobs/{id}/rerun` | re-queue after a fix |
| `move_job(job_id, direction)` | `POST /api/jobs/{id}/move` | reorder queued jobs |

### Deferred / optional

- `POST /api/models/download`, `DELETE /api/models`, `/api/system/diffusers*`
  — model and dependency management; useful eventually, not needed for the
  first author/run/diagnose loop.
- Prompt library endpoints (`/api/prompts*`, `/api/enhance`) — relevant if
  the agent should also help build/enhance prompts, not core to diagnosis.

## Key design decisions

1. **`run_workflow` is confirm-gated.** A run costs real GPU time (and
   possibly money) and the engine is single-GPU/single-job-at-a-time. MCP
   has no built-in elicitation/confirmation primitive, so the confirm step
   should live in the client/agent behavior (state what's about to run and
   get explicit go-ahead) rather than the tool silently firing.
   `validate_workflow` is free and can run without confirmation first.

2. **Security model carries over, doesn't change.** The server already
   validates every path (`dw/security.py`), blocks traversal, rejects
   non-local `Origin` headers, and confines file access to the configured
   workflow/output/prompt directories. MCP wrapping this doesn't add new
   risk for the local/Claude-Code phase. It does **not** solve remote
   auth — the server binds to `127.0.0.1` by default with no
   authentication; `--host 0.0.0.0` exists but is explicitly documented as
   trusted-LAN-only. The embedded-in-app phase needs a real auth story
   before any hosted/remote exposure.

3. **Build order:** read-only tools first (workflow authoring assistant
   usable almost immediately), then the run/diagnose loop once the
   confirm-gating pattern for `run_workflow` is settled.

## Open questions for implementation

- Which MCP transport/SDK — Python `mcp` SDK (project is already Python)
  talking to `http://127.0.0.1:8765/api/...` via `httpx`, run as a stdio
  server for Claude Code.
- Exact confirmation UX for `run_workflow` — client-side prompt text,
  and whether to surface estimated cost/time if the API can supply it.
- Whether GPU/worker diagnostics need anything beyond `/api/memory`, or
  whether that's sufficient for the diagnostic loop as scoped.
