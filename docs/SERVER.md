# Server & Web UI

`dw.serve` runs the workflow engine as a local HTTP server with a full web
UI: browse and run workflows, build them in a form-based editor with
introspection-driven autocomplete, watch jobs stream live progress, review
past generations in a gallery, and manage the models on disk.

```bash
python -m dw.serve                       # http://127.0.0.1:8765
python -m dw.serve --port 8000 --workflow-dir ./workflows --output-dir ./outputs --prompt-dir ./prompts
python -m dw.serve --host 0.0.0.0 --token "some-long-random-string"   # reachable off this machine
python -m dw.serve --trust-workflows      # only if nothing untrusted can reach POST /api/jobs - see Security model
```

Installed as a package, the same server is `dw-serve`. Interactive API docs
(OpenAPI) are at `/docs`.

The server keeps the REPL's persistent GPU worker underneath: models stay
loaded between runs, so re-running a workflow with a new prompt skips the
load entirely.

## The pages

- **Workflows** — every JSON file under `--workflow-dir`, as cards with
  descriptions, output kinds, and variable counts. Folders one level deep
  become sections. Click through to a run form generated from the workflow's
  variables, with the raw JSON alongside.
- **Prompts** — the prompt library under `--prompt-dir` (default: discovered
  the way a CLI run discovers it, then pinned for every job, so the page and
  `prompt:` resolution always agree): stored prompts as
  cards with descriptions, intended-model badges, and tags, foldered the
  same way workflows are. Each opens in an editor with form, split, and
  schema-aware JSON views, and an **Enhance with AI** panel that expands an
  idea into a full prompt with a local language model (a preset per target
  model family; the model runs as an ordinary queued job). A workflow
  argument written as `prompt:name` loads the stored text at run time,
  and deleting a prompt warns which workflows reference it.
- **Jobs** — the queue and full run history (persisted in
  `~/.diffusers_helper/jobs.sqlite`). A running job streams step-by-step
  progress, per-step denoising ticks, what each step is doing when it is not
  denoising (loading a model, decoding, saving), and its result files as
  they land.
  Jobs can be cancelled mid-denoise and re-run with one click.
- **Editor** — build or modify workflows without writing JSON by hand.
  Forms are generated from the live pipeline signatures (see
  [introspection](#introspection)), references
  (`variable:` / `previous_result:`) autocomplete from the workflow itself,
  and three views — form, split, and raw JSON — edit the same definition.
  The split view puts the form beside the JSON with both sides editable;
  changes apply when a side loses focus. Validate, save, and run from the
  same screen. A Monaco editor with the workflow JSON schema backs the
  JSON views. A fourth view, **flow**, renders the workflow's data-flow
  graph read-only: one box per step, arrows for each `previous_result`
  reference labeled with the argument it feeds, entry-point steps marked
  apart from steps that depend on earlier ones, and fan-in points -
  steps combining more than one upstream producer - flagged with the
  cartesian-product multiplier where it's known statically (e.g. a
  literal `num_images_per_prompt` on both producers). It's a diagram of
  the JSON, not a second way to edit it; clicking a step jumps to it in
  the form view.
- **Gallery** — everything in the output directory. Images generated with
  `embed_metadata` carry their full workflow definition and seed; **open as
  workflow** loads that definition into the editor with the seed pinned, so
  any image can be reproduced or riffed on.
- **Models** — the Hugging Face hub cache: every cached repo with sizes,
  revisions, and last-used dates, plus free disk space. Download a repo by
  id with live progress (cancellable; partial files resume on retry), and
  delete to free disk (refused while a job is running; the next workflow
  that needs the model downloads it again). The page also shows the
  installed diffusers version (with its commit for a git install) and can
  upgrade it to GitHub HEAD - new model pipelines usually land there
  before a PyPI release. The idle worker restarts on success so the next
  job imports the new version; the upgrade is refused while a job runs.
- **Schema** — the workflow JSON schema the running server validates
  against, as a browsable tree: the document root plus every definition,
  with types, required markers, defaults, enums, and descriptions.
  `$ref` labels jump to their definition; a filter narrows the list.

## Jobs API

| Route | What it does |
| --- | --- |
| `POST /api/jobs` | Queue a run: `{"workflow_path": ...}` or an inline `{"workflow": {...}, "base_dir": ...}`, plus `arguments` for variable overrides. `workflow_path` accepts a stored workflow name as listed by `/api/workflows` (with or without `.json`, nested names included), or a relative/absolute path that still resolves under `--workflow-dir` - confined the same way the `/api/workflows` CRUD routes are; a path that names a real file outside that directory is rejected with 400, not opened. Answers with argument warnings from signature checking. |
| `GET /api/jobs` | Queue + history summaries |
| `GET /api/jobs/{id}` | Full detail: spec, events, manifest, error |
| `GET /api/jobs/{id}/events` | Server-sent events stream; `?after=N` / `Last-Event-ID` replay missed events, so reconnects are lossless |
| `GET /api/jobs/{id}/event-log?after=-1&limit=200` | The same events as the SSE stream, as one JSON page: `{id, status, events, last_seq, truncated, note}`. `after` is exclusive; page by passing back the previous `last_seq`. A job restored from history serves the bounded event tail persisted with it; a job that finished before events were retained returns an empty list and a `note` saying so. |
| `POST /api/jobs/{id}/cancel` | Cooperative cancel (takes effect at the next step boundary or denoise step) |
| `POST /api/jobs/{id}/rerun` | Re-queue a finished job's spec |
| `POST /api/jobs/{id}/move` | Reorder a queued job: `{"direction": "up"\|"down"\|"front"\|"back"}`. Job listings carry each waiting job's `queue_position`. |

One job runs at a time (it is one GPU); submissions queue in order, and
the waiting portion of the queue can be reordered.

### Progress events

Every event in the stream carries a `seq` and an `event` name:

| event | when | payload |
| --- | --- | --- |
| `job_status` | queued/running/terminal transitions | `status` |
| `log` | worker output lines | `message` |
| `memory` | device memory after a run | `info` |
| `workflow_start` | the run begins | `workflow`, `total_steps`, `steps`, `seed` |
| `step_start` / `step_end` | each step | `step`, `index`, `total_steps`; `files` at the end |
| `iteration_start` | each argument combination in a step | `step`, `iteration`, `total_iterations` |
| `pipeline_step` | each denoise step | `step`, `total_steps` |
| `phase` | the step changes what it is doing | `phase`, `detail` |
| `workflow_end` | the run finishes | `manifest` |

A step spends most of its wall clock outside the denoise loop, and
`pipeline_step` cannot see any of it. `phase` is what fills that silence:
`loading` (with the model or component in `detail`), `cached` (the same
pipeline as a previous run - milliseconds, not minutes), `generating`
(the denoise loop, or a chain's `segment N/M` - which is why the counter
restarts), `decoding` (latents, after the last denoise step), `saving`
(writing files, including video encode) and `task` (a task step, named in
`detail`). Emits are a handful per step, not per denoise tick.

## Introspection API

The editor's forms come from these; they are just as usable from scripts:

- `GET /api/pipelines`, `GET /api/pipelines/{name}` — diffusers pipeline
  classes and their call signatures
- `GET /api/classes?kind=...`, `GET /api/classes/{name}?target=call|init|load` —
  any allowed class (diffusers + registered extension modules), described
  for calling, constructing, or `from_pretrained` loading
- `GET /api/tasks` — the task commands and processors
- `GET /api/tasks/{command}` — a task's argument schema, read from its
  registered implementation's real signature
- `GET /api/schema` — the workflow JSON schema
- `POST /api/validate` — schema validation plus signature-level argument
  warnings for pipeline and task steps (catches the typo before the model
  loads). Accepts `workflow_path` (same resolution and confinement as
  `/api/jobs`, above) as an alternative to inline `workflow` - exactly one
  of the two, or a 400

## Files and models

- `GET /api/workflows` — the stored workflow names, plus a `details` entry
  per workflow: `description`, `kinds` (the output content types' top-level
  halves), `steps`, `variables` (a count) and `variable_names`, and
  `prompt_refs` naming the stored prompts it leans on. Enough to choose a
  workflow and know what to pass it without reading each one; the variable
  defaults are deliberately left out, being an order of magnitude more
  payload on a listing the UI reloads. Cached by file mtime
- `GET/PUT/DELETE /api/workflows/{name}` — read, save, delete workflow files
  (confined to `--workflow-dir`)
- `GET /api/prompts`, `GET/PUT/DELETE /api/prompts/{name}` — the prompt
  library (confined to `--prompt-dir`, names held to what a `prompt:`
  reference can load); saves are validated against the prompt schema,
  served at `GET /api/prompt-schema`
- `GET /api/enhancers`, `POST /api/enhance` — prompt-enhancement presets,
  and `{"idea": ..., "preset": ..., "model_name": ..., "device": ...}` to
  queue an enhancement as an ordinary job whose saved text file is the
  result
- `GET /api/gallery`, `GET /api/gallery/{name}/metadata`,
  `DELETE /api/gallery/{name}` — outputs and their embedded metadata
- `GET /api/models`, `DELETE /api/models?repo={repo_id}` — hub cache
  inventory and deletion
- `POST /api/models/download` (`{"repo_id": ...}`), `GET /api/models/downloads`,
  `POST /api/models/downloads/{id}/cancel` — background snapshot downloads
  with byte-level progress
- `GET /api/system/diffusers`, `POST /api/system/diffusers/update` —
  installed diffusers version/commit, and a background diffusers install/
  update (refused while a job is running or queued). The POST body is
  optional JSON, `{"commit": ..., "revert": ...}`: with neither, it
  `pip install --upgrade`s from GitHub HEAD; `commit` (7-40 hex characters,
  validated before it reaches the command line) pins the git install to
  that commit instead of HEAD; `revert: true` pins back to the known-good
  published release instead of installing from git - the diffusers floor
  version read from `pyproject.toml` (`pip install diffusers==<floor>`).
  `commit` and `revert` are mutually exclusive. The status response
  includes `before` (the version/commit that was installed when the update
  started) alongside the live `version`/`commit`, so a revert has a
  concrete before/after to compare
- `GET /api/memory`, `GET /api/health` — worker VRAM/RAM stats and liveness

## Security model

The server is built to serve **your own GPU to your own browser**, not the
network:

- Binds to `127.0.0.1` by default. `--host 0.0.0.0` (or any other
  non-loopback address) is possible; without a token configured (see
  Authentication, below) the server logs a startup warning, since anything
  that can reach that address can queue jobs and browse/delete files.
- Requests carrying a non-local `Origin` header are rejected (403), which
  blocks cross-site requests from web pages you happen to have open.
- Requests carrying a `Host` header that names neither a loopback address
  nor the configured `--host` are rejected (400). A wildcard bind
  (`--host 0.0.0.0` or `::`) skips this check - clients reach such a
  server by the machine's LAN IP or hostname, never by the bind address,
  so there is no allowlist to build from it. This is defense-in-depth,
  not the DNS-rebinding fix by itself - the `Origin` check above already
  covers browser requests, since a browser's `Origin` reflects the real
  requesting origin regardless of what DNS name resolved to this address.
  The `Host` check closes the remaining gap: a non-browser client (curl, a
  script, the MCP client) that never sends `Origin` at all.
- Every path from HTTP input goes through `dw/security.py` validation;
  workflow files (both the `/api/workflows` CRUD routes and a
  `workflow_path` given to `/api/jobs` or `/api/validate`) are confined to
  the workflow directory, prompt files to the prompt directory, outputs to
  the output directory, and traversal (`../`) is blocked throughout.
- Inline workflow definitions are schema-validated before queueing, and
  their `base_dir` is validated like any other path input.
- A workflow JSON file can execute arbitrary Python (`pre_load_modules`,
  dotted `*_type`/`config_type` values - see [Trust
  model](SECURITY.md#trust-model)). `dw-serve` refuses that surface by
  default for every job it runs, inline or from a file, MCP-submitted or
  not; `--trust-workflows` lifts the refusal for the whole server and
  should only be passed when nothing untrusted can reach `POST /api/jobs`.

### Authentication

There is no authentication by default - the checks above assume a trusted
local machine or LAN. An optional static bearer token closes that gap:

```bash
python -m dw.serve --token "some-long-random-string"
# or
export DW_API_TOKEN="some-long-random-string"
python -m dw.serve
```

When a token is configured, every `/api/*` request must carry
`Authorization: Bearer <token>` or gets a 401. The UI's own static files and
`/outputs` (generated media) stay reachable without it - the page has to
load far enough for a user to enter the token, and an `<img>`/`<script>`
tag cannot attach a header anyway. Two API routes additionally accept the
token as a `?token=...` query parameter, because the browser loads them
without being able to set headers: the SSE stream,
`GET /api/jobs/{id}/events` (`EventSource`), and the gallery grid's
`GET /api/gallery/{name}/thumbnail` (an `<img>` tag). That is a deliberate,
narrower trade-off (a token that can leak into logs or browser history for
those URLs) rather than a general alternative to the header - every other
route accepts the header only.

The web UI has a one-time token field (next to the theme toggle) that
stores the token in `localStorage` and attaches it to every API call,
including the two query-parameter routes above. The MCP server reads the
same `DW_API_TOKEN` variable (or `dw-mcp --token`), so one export
configures both ends - see [MCP.md](MCP.md). It is a convenience, not a
credential vault - anyone with access to the browser profile can read it
back out of `localStorage`.

A token configured this way is a single shared static secret, not a login
system: there is one token, checked with a constant-time comparison, and no
notion of separate users or sessions. It raises the bar for exposing the
server on a LAN or beyond; it is not a substitute for a real network
boundary (a firewall, a VPN, or simply binding to `127.0.0.1`) for anything
more exposed than that.
