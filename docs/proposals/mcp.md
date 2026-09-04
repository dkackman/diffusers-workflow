# MCP Server for diffusers-workflow — Scoping Doc

> Status: scoped and verified against the code on 2026-09-01. The design
> that came out of this document is
> [2026-09-01-mcp-server-design.md](../specs/2026-09-01-mcp-server-design.md).

## Background

`diffusers-workflow` (dkackman/diffusers-workflow) is a declarative JSON
workflow engine for Hugging Face Diffusers, with a local HTTP server
(`dw.serve`, default port 8765) that exposes a full REST API: workflow
CRUD, a job queue with SSE progress streaming, model management,
introspection of diffusers pipeline signatures, and a gallery of past
generations with embedded metadata.

Today, workflow authoring and debugging, for the project developer, can
happen via Claude Code in VS Code, with direct repo/filesystem access:
"create a workflow that does X," or "workflow Y produces bad output, find
out why" — Claude Code runs the workflow, reads logs, checks GPU memory,
and proposes fixes to code or workflow config. It has even gone so far as
to identify bad RAM causing intermittent generation failures.

That flow works well for the developer (me), but it assumes a dev
environment: a repo checkout, a terminal, VS Code, comfort reading stack
traces. The next audience is **non-developer end users** of the packaged
product, who won't have any of that. They'll have the app (web UI /
`dw.serve`, or an embedded product surface), not a coding agent with shell
access.

## Goal

The modes a workflow can be authored and run in today:

- a text editor plus the CLI, `python -m dw.run workflow.json`
- the command-line REPL, `python -m dw.repl`
- the HTTP API `dw.serve` and its web UI

Add a fourth: an **MCP server** wrapping the existing `dw.serve` HTTP API,
exposing the same "author a workflow / run it / diagnose why it failed or
produced bad output" capability through a bounded, structured tool surface
— no shell, no repo checkout, no stack traces. Any MCP-compatible client
can drive it: Claude Code first, later an embedded chat UI in the product,
and eventually a local-LLM-backed client for users without a paid Claude
subscription.

## Why MCP (not just keep using Claude Code)

- Claude Code assumes a dev environment; end users won't have one.
- MCP tools are a defined, constrained surface (structured in/out, no
  arbitrary code execution) — appropriate once non-developers are the
  audience, and safer than raw shell access.
- The underlying HTTP API already exists and is already defensive (path
  validation, traversal blocking, `Origin` header checks, JSON size
  limits) — MCP adds a protocol layer in front of it, not new attack
  surface.
- Protocol-agnostic: the same server can later be pointed at Claude Code,
  a future embedded chat UI, or a local-LLM-backed client.

## Rollout sequence (decided)

1. **MCP server + Claude Code** as the client, for continued dev use and
   to validate the tool surface.
2. **Embedded-in-app** later, once the product is packaged for
   non-developer users. This phase needs its own auth story, since the
   server binds to `127.0.0.1` with no authentication (see Security,
   below) — fine for local/trusted-LAN use, not fine for a hosted product.
3. **Local LLM support** stays a goal throughout. MCP is
   protocol-agnostic, so this is a separate, later concern, not a blocker.

## Topology

`dw_mcp/` is a **stdio MCP server that is an HTTP client of a running
`dw.serve`**. (It shipped as `dw/mcp/` and moved to a top-level package in
F8, below.) It owns no job state, no GPU, and no worker process; it
holds no state the REST API doesn't already hold. If `dw.serve` isn't
reachable it fails with an instruction to start it, rather than starting
one itself.

Rejected: embedding the FastAPI app in-process. `create_app()` builds its
own `JobManager` and GPU worker, so an in-process copy running alongside a
real `dw.serve` would mean two workers contending for one GPU — which the
single-job-at-a-time design cannot tolerate.

## Proposed MCP tool set

Mapped to existing `dw.serve` routes. One new endpoint is required (see
"Corrections", below); everything else exists.

### Read-only / low-cost (build first)

| Tool | Route(s) | Notes |
| --- | --- | --- |
| `list_workflows` | `GET /api/workflows` | descriptions, output kinds, variable counts |
| `get_workflow(name)` | `GET /api/workflows/{name}` | full JSON |
| `validate_workflow(json)` | `POST /api/validate` | schema + signature-level argument checks; no GPU cost — the pre-flight tool |
| `get_schema` | `GET /api/schema` | workflow JSON schema |
| `list_pipelines` | `GET /api/pipelines` | what pipelines the installed diffusers offers |
| `get_pipeline_signature(name)` | `GET /api/pipelines/{name}` | real pipeline arguments, before proposing a config fix |
| `list_classes(kind)` / `get_class(name)` | `GET /api/classes?kind=...`, `GET /api/classes/{name}` | broader introspection beyond pipelines |
| `list_tasks` / `get_task(command)` | `GET /api/tasks`, `GET /api/tasks/{command}` | task-step argument schemas |
| `list_models` | `GET /api/models` | hub cache inventory: sizes, revisions, last-used |
| `get_memory` | `GET /api/memory` | worker VRAM/RAM stats — answers the GPU-diagnosis need directly |
| `get_health` | `GET /api/health` | liveness |
| `list_jobs` | `GET /api/jobs` | live queue plus recent history |
| `list_gallery(limit)` | `GET /api/gallery` | recent outputs, newest first |
| `get_gallery_metadata(name)` | `GET /api/gallery/{name}/metadata` | a prior run's exact workflow + seed, e.g. to reproduce a failure |
| `get_output_image(name, max_dimension)` | `GET /outputs/{name}` (static mount) | returns a downscaled MCP image block — this is how the agent actually *sees* bad output |

### Authoring (build second)

| Tool | Route(s) | Notes |
| --- | --- | --- |
| `save_workflow(name, json)` | `PUT /api/workflows/{name}` | overwrites; `destructiveHint: true`, `idempotentHint: true` |
| `delete_workflow(name)` | `DELETE /api/workflows/{name}` | `destructiveHint: true` |

Without these the server is a read-only advisor: it can propose a workflow
but never complete the validate-then-save loop.

### Diagnostic loop (build third)

| Tool | Route(s) | Notes |
| --- | --- | --- |
| `run_workflow(workflow_path \| inline_workflow, arguments, acknowledged_cost)` | `POST /api/jobs` | **Costs GPU time; one job runs at a time.** Submits and returns immediately. See confirm-gating below. |
| `get_job(job_id)` | `GET /api/jobs/{id}` | status, arguments, warnings, manifest, error, traceback, `event_count` |
| `get_job_events(job_id, after, limit)` | `GET /api/jobs/{id}/event-log` **(new)** | live or persisted `job_status` / `phase` / `memory` / `log` events — phase transitions (loading → generating → decoding → saving) |
| `cancel_job(job_id)` | `POST /api/jobs/{id}/cancel` | cooperative cancel |
| `rerun_job(job_id, acknowledged_cost)` | `POST /api/jobs/{id}/rerun` | re-queue after a fix — cost-gated like a run |
| `move_job(job_id, direction)` | `POST /api/jobs/{id}/move` | reorder queued jobs |

### Deferred / optional

- `POST /api/models/download`, `DELETE /api/models`, `/api/system/diffusers*`
  — model and dependency management; useful eventually, not needed for the
  first author/run/diagnose loop.
- Prompt library endpoints (`/api/prompts*`, `/api/enhance`,
  `/api/enhancers`, `/api/prompt-schema`) — relevant if the agent should
  also help build/enhance prompts, not core to diagnosis.
- `base_dir` on `POST /api/jobs` is deliberately **not** exposed. It is a
  path-authority parameter that decides where an inline workflow's
  relative paths resolve; v1 lets them resolve against the server's own
  workflow directory only.

## Corrections found while verifying against the code

Checked against `dw/server/app.py` and `dw/server/jobs.py` on 2026-09-01.

1. **`get_job` returns no events and no spec.** `Job.detail()`
   (`jobs.py:222`) returns `event_count`, not `events`, and `arguments`,
   not the whole spec. The event trail exists only behind the SSE stream,
   which a request/response MCP tool cannot consume sensibly. Hence the
   one new endpoint, `GET /api/jobs/{job_id}/event-log`.
2. **Outputs are not on an `/api` route.** Generated media is served from
   a `StaticFiles` mount at `/outputs` (`app.py:840`). Any tool that shows
   the agent an image must read from there.
3. **MCP does have a confirmation primitive.** The 2025-06-18 spec
   revision added **elicitation**, and tool **annotations**
   (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`)
   exist for exactly this purpose. The earlier claim that confirmation had
   to live entirely in client prose was wrong.
4. **Events did not survive a restart.** They live in the in-memory `Job`
   object (`jobs.py:182`); `JobHistory` persisted eleven columns and none
   of them was the event log, so a job recovered from history had no
   trail. Originally deferred as F1, this has been pulled into the build —
   history now persists a bounded tail of each job's events.

## Key design decisions

1. **`run_workflow` is confirm-gated, in three layers.** A run costs real
   GPU time and the engine is single-GPU/single-job-at-a-time. The tool
   carries annotations so a client can surface the cost; it uses MCP
   elicitation when the client advertises that capability; and it requires
   an explicit `acknowledged_cost` argument as the client-agnostic floor,
   since Claude Code does not implement elicitation today.
   `validate_workflow` is free and runs first, without confirmation.

2. **`run_workflow` never blocks.** It submits and returns
   `{job_id, status, queue_position}` immediately. A generation takes
   minutes; no MCP client will hold a tool call open that long. Progress
   comes from polling `get_job_events`.

3. **Security model carries over, doesn't change.** The server already
   validates every path (`dw/security.py`), blocks traversal, rejects
   non-local `Origin` headers, and confines file access to the configured
   workflow/output/prompt directories. MCP wrapping this doesn't add new
   risk for the local/Claude-Code phase. It does **not** solve remote
   auth — the server binds to `127.0.0.1` by default with no
   authentication; `--host 0.0.0.0` exists but is explicitly documented as
   trusted-LAN-only. The embedded-in-app phase needs a real auth story
   before any hosted/remote exposure.

4. **Tools only, no resources or prompts, in v1.** `get_schema` and
   `get_workflow` are natural MCP *resources*, and "diagnose this job" is
   a natural MCP *prompt* — but client support for tools is universal and
   support for the other two is not. Revisit once the tool surface has
   been validated.

5. **Build order:** read-only tools first (a workflow authoring assistant
   usable almost immediately), then authoring writes, then the
   run/diagnose loop once confirm-gating is settled.

## Post-v1 follow-ups

Deliberately out of the first build. Each is its own change. (F1, persisting
a job's event tail, was pulled into the build instead — without it the
diagnostic loop only explains jobs from the current server process, which is
the wrong trade for the non-developer audience this is aimed at.)

- **F2 — Cost/time estimation for the confirm prompt.** Nothing in the API
  can say "this will take about four minutes." Job history holds durations
  per workflow id; a `GET /api/workflows/{name}/stats` could turn that
  into an estimate worth putting in front of a human before a run.
- **F3 — Expose the server log.** `log_filename` is already a setting; a
  `get_server_log(tail)` tool would help diagnosis, but it widens the read
  surface beyond the confined directories and needs its own redaction
  thinking first.
- **F4 — Model and dependency management tools.** *(Done 2026-09-01.)* The
  deferred `/api/models/*` and `/api/system/diffusers*` routes — the first
  wall a non-developer hits is "that model isn't downloaded," and until this
  the agent could see the gap and not close it. Six tools in `dw_mcp/models.py`:
  `download_model`, `list_downloads`, `cancel_download`, `delete_model`,
  `get_diffusers_state`, `update_diffusers`.

  This widened the `acknowledged_cost` gate past its original GPU-only
  charter. `download_model` (tens of gigabytes), `delete_model`
  (unrecoverable without re-downloading) and `update_diffusers` (an untagged
  development build, no undo) each carry it, with a distinct refusal message
  apiece: one shared message would be wrong for each of them in a different
  way, and a gate the user learns to wave through is not a gate. The two
  cancels stay ungated — they end a cost rather than starting one, and
  gating them would make the safe direction the harder one.
- **F5 — Prompt library tools.** `/api/prompts*` and `/api/enhance`, for
  an agent that helps compose prompts rather than only workflows.
- **F6 — MCP resources and prompts.** Re-expose the schema, workflows, and
  gallery as resources, and ship a "diagnose this job" prompt template,
  once target clients support them.
- **F7 — Remote/auth story.** Required before the embedded-in-app phase.
  Blocks any non-localhost exposure of either the REST API or the MCP
  server.
- **F8 — The MCP server imported the whole engine.** *(Found during the v1
  build; not part of the original scoping. Done 2026-09-01.)* The package
  shipped as `dw/mcp/`, and importing any `dw.*` submodule runs
  `dw/__init__.py`, which imports torch and diffusers — so a process that is
  a pure HTTP client, owning no models, paid for the model framework at
  startup. Measured: `import dw` 1.05s (1.40s cumulative) against 0.35s for
  the client's own dependencies (`httpx`, `PIL`, `mcp`). Paid once per
  client session, since a stdio server is spawned once and lives for the
  session — a real but modest cost, which is why this is recorded with its
  numbers rather than asserted as severe. Fixed by moving the package to a
  top-level `dw_mcp/`, taking startup to 0.36s with torch never imported.
  A regression guard in `tests/test_mcp_server.py` asserts the boundary
  (`torch`, `diffusers` and `dw` absent from `sys.modules` after importing
  `dw_mcp.server`), because the failure mode is one convenience import away
  and is otherwise invisible.
