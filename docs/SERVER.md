# Server & Web UI

`dw.serve` runs the workflow engine as a local HTTP server with a full web
UI: browse and run workflows, build them in a form-based editor with
introspection-driven autocomplete, watch jobs stream live progress, review
past generations in a gallery, and manage the models on disk.

```bash
python -m dw.serve                       # http://127.0.0.1:8765
python -m dw.serve --port 8000 --workflow-dir ./examples --output-dir ./outputs
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
- **Jobs** — the queue and full run history (persisted in
  `~/.diffusers_helper/jobs.sqlite`). A running job streams step-by-step
  progress, per-step denoising ticks, and its result files as they land.
  Jobs can be cancelled mid-denoise and re-run with one click.
- **Editor** — build or modify workflows without writing JSON by hand.
  Forms are generated from the live pipeline signatures (see
  [introspection](#introspection)), references
  (`variable:` / `previous_result:`) autocomplete from the workflow itself,
  and a split view shows the JSON updating as you edit. Validate, save, and
  run from the same screen. A Monaco editor with the workflow JSON schema
  backs the JSON views.
- **Gallery** — everything in the output directory. Images generated with
  `embed_metadata` carry their full workflow definition and seed; **open as
  workflow** loads that definition into the editor with the seed pinned, so
  any image can be reproduced or riffed on.
- **Models** — the Hugging Face hub cache: every cached repo with sizes,
  revisions, and last-used dates, plus free disk space. Deleting a repo
  frees it from disk (refused while a job is running; the next workflow
  that needs the model downloads it again).

## Jobs API

| Route | What it does |
| --- | --- |
| `POST /api/jobs` | Queue a run: `{"workflow_path": ...}` or an inline `{"workflow": {...}, "base_dir": ...}`, plus `arguments` for variable overrides. Answers with argument warnings from signature checking. |
| `GET /api/jobs` | Queue + history summaries |
| `GET /api/jobs/{id}` | Full detail: spec, events, manifest, error |
| `GET /api/jobs/{id}/events` | Server-sent events stream; `?after=N` / `Last-Event-ID` replay missed events, so reconnects are lossless |
| `POST /api/jobs/{id}/cancel` | Cooperative cancel (takes effect at the next step boundary or denoise step) |
| `POST /api/jobs/{id}/rerun` | Re-queue a finished job's spec |

One job runs at a time (it is one GPU); submissions queue in order.

## Introspection API

The editor's forms come from these; they are just as usable from scripts:

- `GET /api/pipelines`, `GET /api/pipelines/{name}` — diffusers pipeline
  classes and their call signatures
- `GET /api/classes?kind=...`, `GET /api/classes/{name}?target=call|init|load` —
  any allowed class (diffusers + registered extension modules), described
  for calling, constructing, or `from_pretrained` loading
- `GET /api/tasks` — the task commands and processors
- `GET /api/schema` — the workflow JSON schema
- `POST /api/validate` — schema validation plus signature-level argument
  warnings (catches the typo before the model loads)

## Files and models

- `GET/PUT/DELETE /api/workflows/{name}` — read, save, delete workflow files
  (confined to `--workflow-dir`)
- `GET /api/gallery`, `GET /api/gallery/{name}/metadata`,
  `DELETE /api/gallery/{name}` — outputs and their embedded metadata
- `GET /api/models`, `DELETE /api/models?repo={repo_id}` — hub cache
  inventory and deletion
- `GET /api/memory`, `GET /api/health` — worker VRAM/RAM stats and liveness

## Security model

The server is built to serve **your own GPU to your own browser**, not the
network:

- Binds to `127.0.0.1` by default. `--host 0.0.0.0` is possible but there is
  no authentication — treat it as trusted-LAN only.
- Requests carrying a non-local `Origin` header are rejected (403), which
  blocks cross-site requests from web pages you happen to have open.
- Every path from HTTP input goes through `dw/security.py` validation;
  workflow files are confined to the workflow directory, outputs to the
  output directory, and traversal (`../`) is blocked.
- Inline workflow definitions are schema-validated before queueing, and
  their `base_dir` is validated like any other path input.
