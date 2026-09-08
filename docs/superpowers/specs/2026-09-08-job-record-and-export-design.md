# Job record and export: design

Date: 2026-09-08. Proposal: [docs/proposals/job-record-and-export.md](../../proposals/job-record-and-export.md).
Companion: [docs/proposals/resume.md](../../proposals/resume.md), which this
design serves but does not implement.

## Goal

Every run leaves a realized copy of the workflow that produced it beside its
manifest; a server job knows which run is its own; an MCP client can read
that workflow and can export one job as a git-ready directory and a zip.

## Non-goals

- Resuming a run from a previous run's files (resume.md stage 3).
- Rewriting references inside an exported workflow.
- Rebuilding job history from run directories.
- Widening the schema's `workflow` step to hold an inline definition.

## Global constraints

- Model knowledge stays out of engine code; nothing here names a model.
- Every path read or written goes through `dw/security.py` validators:
  `validate_output_path`, `validate_path`, the asset and output resolvers.
- The realized file must validate against `dw/workflow_schema.json`
  unchanged: annotations that the schema would reject live in the manifest.
- Writing the realized file and the manifest is best effort: a run that
  produced its files has succeeded whether or not the record landed
  (`write_manifest`'s rule).
- Nothing bundled requires `--trust-workflows`.
- `exports` is a reserved workspace name, beside `workflows`, `prompts`,
  `assets`, `outputs`.

## 1. Realization

### Module

New `dw/realize.py`:

```python
def realize_workflow(definition, arguments, seed, base_dir=None,
                     prompt_dir=None, output_root=None):
    """A copy of `definition` with every mutable input pinned.
    Returns (realized, annotations)."""
```

- `definition` is the workflow as loaded (before `Workflow.run`'s deep copy
  mutates it); the function never mutates its input.
- `arguments` is the run's argument dict; `seed` is the seed the run
  resolved (`resolved_seed` in `Workflow.run`), never `None` here because
  `run` draws one when the workflow names none.
- `annotations` is a dict the manifest carries:
  `{"prompts": [...names...], "sub_workflows": {path: sha256_hex}}`.

### Rules

| Reference | Realized as |
|---|---|
| `variables` | The defaults after `set_variables(arguments, variables)`, which is exactly what the run computed. `variable:` references elsewhere in the file are left alone. |
| `seed` | The integer `seed`, written at the top level even when the definition had none. |
| `prompt:name` (any string value, anywhere in the tree) | `fetch_prompt(reference, prompt_dir, base_dir)`'s text. `name` is appended to `annotations["prompts"]` (deduplicated, in first-seen order). |
| `output:<identity>/latest/<file>` | `output:<identity>/<run id>/<file>` where run id comes from `resolve_output_reference(reference, output_root)` relative to the root. An explicit run id is kept as written. |
| `asset:`, `constant:`, `previous_result:`, `builtin:` | Kept. |
| `workflow.path` on a step, not `builtin:` | Kept; `annotations["sub_workflows"][path]` is the SHA-256 of the file, resolved as `Workflow` resolves it (relative to `base_dir`, confined to `workflow_dir`). Unreadable file: the digest is `null`, no error. |
| A prompt or output reference that fails to resolve | Left as written; the run will raise on it later with the engine's own error. Realization never fails a run. |

The tree walk is one recursive function over dicts, lists and strings, the
shape `referenced_result_names` in `dw/step_cache.py` uses. A stored
prompt's text may not itself begin with a reference prefix (an existing
engine rule), so inlining cannot introduce a second resolution.

### When and where it is written

In `Workflow.run`, immediately after the run directory is chosen
(`self._run_dir` set, `run_id` known) and before the step loop, when
`self._run_dir` is set and not inherited:

```python
realized, annotations = realize_workflow(
    self.workflow_definition, arguments, resolved_seed,
    base_dir=self.base_dir, output_root=self.output_dir)
write_realized_workflow(self._run_dir, realized)
```

New in `dw/runs.py`: `REALIZED_FILE_NAME = "workflow.json"` and
`write_realized_workflow(run_dir, realized)`, the same best-effort shape as
`write_manifest`, returning the path or `None`.

The manifest's `workflow` block gains `"realized": "workflow.json"` (or
`null` when the write failed) and the two annotation keys:

```json
"workflow": {
  "id": "...", "file": "...", "identity": "...",
  "realized": "workflow.json",
  "prompts": ["scenic/dusk"],
  "sub_workflows": {"steps/upscale.json": "ab12..."}
}
```

`realized_seed` is already the manifest's `seed`. A sub-workflow inherits
the run directory and writes neither file, as today.

Flat output layout writes no run directory, so it writes no realized file;
the CLI does not warn, matching how the manifest behaves there.

## 2. The job knows its run

### Event

`Workflow.run` emits, right after the realized file is written (also when
the write failed):

```python
run_context.emit("run_start", run_id=run_id,
                 identity=workflow_identity(self.file_spec, workflow_id),
                 run_dir=<run dir relative to self.output_dir>)
```

The worker already forwards every emitted event as a `progress` message,
so the server sees it with no worker change.

### Job and store

`Job` gains `run_id = None` and `run_dir = None`; `JobManager`'s progress
branch sets them when `event["event"] == "run_start"`. `summary()` includes
`run_id`; `detail()` includes `run_id` and `run_dir`.

`jobs.sqlite` gains `run_id TEXT` and `run_dir TEXT`, migrated like
`workflow_name` (ALTER when absent, NULL for older rows). Persisted at job
finish with the rest of the row; read back into the history dict.

### Reading the realized workflow

```python
def realized(self, job_id):
    """The realized workflow a job ran, or None when the job predates
    run tracking or its run directory no longer holds the file."""
```

Reads `<output_dir>/<run_dir>/workflow.json` where `output_dir` is the
job's own (`spec["output_dir"]`, or the manager's default for history
rows without one), after `validate_output_path` confines the join.

`GET /api/jobs/{job_id}/workflow` returns
`{"id", "definition", "realized": bool}`: the realized workflow when
`realized()` finds one, else the submitted definition as today. The UI's
graph keeps working unchanged; a later UI change may label it.

### MCP

New tool in `dw_mcp/diagnose.py`, registered in `dw_mcp/server.py` beside
`get_job`:

```python
def get_job_workflow(client, job_id):
    """The workflow a job ran. `realized: true` means every mutable input
    is pinned (arguments, seed, prompts, output:latest); false means the
    job predates run tracking and this is the definition as submitted.
    Pass it to save_workflow to rerun it by name, or edit it and pass it
    to run_workflow as inline_workflow."""
```

Returns `{"job_id", "realized", "workflow", "next"}` where `next` is one
sentence naming `save_workflow` and `run_workflow`.

## 3. Export

### Server module

New `dw/server/exports.py`:

```python
EXPORTS_SUBDIR = "exports"

@dataclass
class ExportSummary:
    job_id: str
    directory: str          # absolute, on the server
    files: list[dict]       # [{"path": "outputs/x.mp4", "bytes": n}, ...]
    total_bytes: int
    missing: list[str]      # references the export could not find

def export_job(manager, job_id, workspace_root, asset_roots,
               overwrite=False) -> ExportSummary
```

Behaviour:

1. `manager.get(job_id)` must exist and be terminal; a live or unknown job
   is a `ValueError` the route turns into 409 or 404.
2. Target is `<workspace_root>/exports/<job_id>/`, validated with
   `validate_output_path` against the workspace root. Existing target and
   `overwrite=False` raises `FileExistsError` (409); `overwrite=True`
   removes it first.
3. `workflow.json`: `manager.realized(job_id)`, else the submitted
   definition, and `job.json` records which (`"realized": bool`).
4. `manifest.json`: copied from the run directory when `run_dir` is known;
   else synthesized from the job row's `manifest` list, marked
   `"synthesized": true`.
5. `job.json`: the job's `detail()` minus `traceback` and `event_count`.
6. `assets/<name>`: for every `asset:` string in `workflow.json`, the file
   `resolve_asset_reference` finds on `asset_roots` (the workspace's asset
   search path, in the order `_asset_roots` in `app.py` builds it), copied
   under its reference name (folders preserved). Unresolvable: added to
   `missing`.
7. `inputs/<identity>/<run id>/<file>`: for every `output:` string, the
   file `resolve_output_reference` finds under the job's output root.
   Unresolvable: `missing`.
8. `outputs/<file>`: every file the manifest lists, copied with the
   manifest's relative names.
9. `README.md`: generated text (a module constant with format fields):
   what the job was (workflow id, catalog name, status, started and finished
   times, device, engine version), the seed, the arguments as a JSON block,
   the stored prompts inlined by name, the sub-workflow digests, where each
   `inputs/` file came from, how to run it again
   (`python -m dw.run workflow.json` from a checkout, or `run_workflow`
   with the file as `inline_workflow`), the `missing` list, and one
   paragraph saying media under `assets/`, `inputs/` and `outputs/` belongs
   in Git LFS if the tree is committed.

Copies are copies, never hard links; the file list and `total_bytes` are
computed from what landed.

### Routes

- `POST /api/jobs/{job_id}/export?workspace=&overwrite=` → 201 with the
  `ExportSummary` as JSON plus `"zip_url"` built by `_served_url` for
  `/exports/<job_id>.zip`. 404 unknown job, 409 live job or existing
  export without `overwrite`.
- `GET /exports/{job_id}.zip?workspace=` → `StreamingResponse` of a zip
  built on the fly from the export directory with `zipfile` at
  `ZIP_DEFLATED`, entries named `<job_id>/<relative path>`. 404 when the
  directory does not exist. Same auth treatment as `/outputs`: gated only
  where `/outputs` is.
- `GET /exports/{job_id}/{path}` is not added; the zip and the directory
  are the two forms.

`RESERVED_WORKSPACE_NAMES` gains `exports`; `_foreign_entries` skips it
when listing workspaces.

### MCP

New `dw_mcp/exports.py`:

```python
def export_job(client, job_id, overwrite=False):
    """Gather one finished job into a directory on the machine running
    dw.serve: workflow.json (realized), manifest.json, job.json, README,
    assets/, inputs/, outputs/. Returns the directory, the zip URL, the
    file list with sizes and the total, and the three JSON files inline.
    The directory is on the server machine, not this one - use the zip
    URL to fetch it elsewhere."""
```

Registered in `dw_mcp/server.py` in the jobs group. The response includes
`"where": "<directory> on the machine running the MCP server"` so an
agent does not report a local path, the lesson `download_output` taught.

## 4. Tests

- `tests/test_realize.py`: one test per row of the realization table,
  using a temporary prompt library and output root; a test that the input
  definition is not mutated; a test that an unresolvable prompt is left as
  written; a test that the realized file validates against the schema
  (`dw.validate`).
- `tests/test_runs.py` (existing): `write_realized_workflow` best-effort
  return and the manifest's `realized`, `prompts`, `sub_workflows` keys
  after a task-only workflow runs; flat layout writes none.
- `tests/test_server_jobs.py` (new; the manager tests today live in `tests/test_server.py`): `run_start` populates
  `run_id`/`run_dir`; both persist and read back; `realized()` returns the
  file, `None` for a pre-tracking row.
- `tests/test_server_exports.py` (new, on the `tests/test_server.py` client fixture): the workflow route's
  `realized` flag; export 201 tree contents; 409 on a live job; 409 without
  overwrite; the zip lists the same entries as the directory; `exports` is
  refused as a workspace name.
- `tests/test_mcp_diagnose.py`: `get_job_workflow` shape and `next`.
- `tests/test_mcp_exports.py`: `export_job` shape, `where` sentence, the
  refusal path when the server answers 409.
- `tests/test_docs_links.py` keeps passing for every path the docs name.

## 5. Docs

- `docs/WORKFLOW_GUIDE.md`, run directories: `workflow.json` beside the
  manifest, what "realized" means, the reproduction sentence. The
  "Authoring a workflow from an agent" section: after a long inline run,
  `get_job_workflow` then `save_workflow` so the next run is by name; the
  same sentence in `CLAUDE.md`'s type-system list where the conventions
  are mirrored.
- `CLAUDE.md` run-directories gotcha: the realized file and the manifest
  keys; `exports` reserved.
- `docs/MCP.md`: the two tools, with the server-machine caveat.
- `docs/SERVER.md`: the export routes and the `exports/` directory.
- `docs/WORKSPACES.md`: `exports/` beside the four folders.
- `plugins/dw/skills/*/SKILL.md`, "Run and judge": one bullet each, "After
  an inline run worth keeping, `get_job_workflow` and `save_workflow` it;
  `export_job` bundles the run for git."
- `docs/proposals/job-record-and-export.md` status line moves to
  "implemented" with the PR; `resume.md` gains a sentence that its stage 2
  is satisfied by the realized file.

## Sequence for the plan

1. Realization module and tests.
2. Writing it from `Workflow.run`, manifest keys, runs tests.
3. `run_start`, job fields, store migration, `realized()`, route flag.
4. `get_job_workflow` tool.
5. Export module, routes, reserved name, tests.
6. `export_job` tool.
7. Docs and skills.
