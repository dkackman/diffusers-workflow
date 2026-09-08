# Proposal: the realized workflow as a job's record, and exporting a job

Status: implemented 2026-09-08 (design:
`docs/superpowers/specs/2026-09-08-job-record-and-export-design.md`, plan:
`docs/superpowers/plans/2026-09-08-job-record-and-export.md`). Supplies the
"manifest carries step identity" stage of [resume.md](resume.md), which stays
the design for resuming a run; this proposal is the record that resume reads,
plus the two ways to get it off the server.

## Problem

On 2026-09-08 an MCP session assembled a six-shot MiniMax H3 short as an
inline workflow, ran it for most of an hour, and lost it at the last step to
a bad argument. The retry regenerated every shot. Three things about that
run are worth fixing, and none of them is the bad argument:

- **The workflow that produced the cut existed nowhere the user could read.**
  The job store keeps an inline definition in its spec column and the job
  page can draw it, but no MCP tool returns it, and the run directory holds
  a manifest that says only `"file": ".../__inline__.json"`.
- **A template job records less than that.** Its spec holds a path and the
  arguments. Edit the template, or a stored prompt it names, or make a newer
  run that `output:latest` now picks, and the record no longer describes what
  ran. The manifest pins the seed and the arguments, but not the definition.
- **Nothing packages a run.** Manifest, workflow, the assets it referenced
  and the files it made are four places on one machine, and `download_output`
  moves one file at a time onto that same machine. A run that is worth
  keeping, sharing or committing has to be gathered by hand.

MCP is turning out to be the primary way work is submitted, and an agent
writes inline workflows freely: an inline job is now the common case, not
the exception the job store treated it as.

## What exists today

| Piece | Where | What it records |
|---|---|---|
| Submitted definition | `jobs.sqlite` `spec` (`workflow` for inline, `workflow_path` for a file) | What was sent, not what ran |
| Read-only view of it | `GET /api/jobs/{id}/workflow`, `JobManager.definition` | UI only; no MCP tool |
| Rerun | `POST /api/jobs/{id}/rerun`, `rerun_job` | Resubmits the spec verbatim, arguments included |
| Run directory | `<output_dir>/<identity>/<run id>/` (`dw/runs.py`) | Every file a step wrote |
| Manifest | `manifest.json` beside them | Status, times, engine version, device, seed actually used, arguments, per-step files |
| Run id | Directory name only | Never reported to the job or the job store |
| Reuse of a file | `output:<identity>/<run id or latest>/<file>` | Resolves out of that tree |
| Keeping a file | `keep_output` | Copies one output into the assets library |

The record of a job is therefore split between a database row that knows the
definition and a directory that knows the outcome, and neither is complete.

## Proposed shape

### 1. Every run writes its realized workflow beside its manifest

`Workflow.run` writes `workflow.json` into the run directory at the start of
the run, before the first step, so a crash or a cancel still leaves it. The
manifest points at it (`"workflow": {..., "realized": "workflow.json"}`).
CLI runs and server jobs get it alike, since the manifest is written there
too. A sub-workflow inherits its parent's run directory and writes none of
its own, as with the manifest.

*Realized* means every mutable input is pinned, so the file reproduces the
run whatever changes later:

| Reference | In the realized file |
|---|---|
| `variables` and the job's `arguments` | Arguments folded into the variable defaults; `variable:` references stay, so the file remains runnable with overrides |
| `seed` | The seed the run actually used (drawn if the workflow named none) |
| `prompt:name` | The stored prompt's text, inlined |
| `output:<identity>/latest/<file>` | Rewritten to the concrete run id it resolved to |
| `output:` with an explicit run id | Kept |
| `asset:name` | Kept: a name in the library, which the export bundles |
| `constant:` | Kept: a value in the Python module the manifest's `dw_version` pins |
| `builtin:` sub-workflow | Kept: packaged with the engine, pinned by `dw_version` |
| sub-workflow by local path | Kept; the manifest records the file's SHA-256 |
| `previous_result:` | Kept: it names a step in the same file |

The realized file is a valid workflow. `validate_workflow` accepts it and
`run_workflow(inline_workflow=...)` reproduces the run, which is the
simplest possible resume-by-hand and needs no new engine path.

### 2. The job knows its run

The worker reports the run id and the run directory (relative to the output
root) in a `run_start` event; `Job` records them and `jobs.sqlite` gains a
`run_id` column. `JobManager.realized(job_id)` reads the run's
`workflow.json`. A job from before this feature has no run id: the manager
falls back to the submitted definition and says so, rather than deriving a
run directory from file paths.

### 3. MCP can read it

`get_job_workflow(job_id)` returns the realized workflow, with `realized:
true`, or the submitted definition with `realized: false` and a sentence
saying why. Small enough to return inline: a workflow is text.

### 4. Exporting a job

`POST /api/jobs/{id}/export` gathers one job into
`<workspace>/exports/<job id>/`:

```
exports/<job id>/
    README.md          what this is, how it was made, how to run it again
    workflow.json      the realized workflow (section 1)
    manifest.json      the run's manifest
    job.json           the job row: status, times, arguments, warnings, error
    assets/            every asset:name the workflow references, by name
    inputs/            every file an output: reference named from another run
    outputs/           every file the manifest lists, in the run's layout
```

The tree is git-ready as it stands: text at the top, media in folders, no
absolute paths (the README notes that large media belongs in Git LFS).
`GET /exports/<job id>.zip` streams the same tree as one archive for a
browser or a laptop, built on request from the directory rather than kept as
a second copy. `exports` joins the reserved workspace names.

MCP `export_job(job_id)` runs the export and returns the directory, the zip
URL, and the file list with sizes; the JSON files it also returns inline.
Like `download_output`, the directory is on the machine running the server,
and the tool says so.

### 5. Recoverability

This proposal writes the record; [resume.md](resume.md) is the design for
acting on it. Its stage 2, a per-step identity in the manifest, is served by
the realized file: each step's definition is there to compare, and the
manifest's per-step files say what it made. Its stage 3, rehydrating the
step cache from a previous run, is what turns the lighthouse retry into a
one-step rerun, and is planned as the next piece after this one lands. Until
then the realized file is what a user hands back to `run_workflow` to
reproduce a run, and what an agent edits to change one step of it.

## Staging

1. **The realized file.** Realization rules, written at run start, manifest
   pointer, tests for every row of the table above. CLI and server alike.
2. **Run id on the job, and the MCP read.** The `run_start` event, the
   column, `JobManager.realized`, `get_job_workflow`.
3. **Export.** The route, the directory layout, the README, the zip, the MCP
   tool, the reserved name.
4. **Resume**, as resume.md stage 3, its own proposal and plan.

Stages 1 and 2 are the useful unit; 3 stands on them.

## Open questions

- **Sub-workflows by local path.** Inlining them would make the realized file
  self-contained, but the schema's `workflow` step takes a path, not a
  definition. Decided in the design: keep the path, record the file's
  digest in the manifest, and inline only if the digest turns out to be
  what people trip over.
- **Realizing `prompt:` loses the name.** Decided in the design: the
  manifest lists the stored prompts the run inlined, so the realized file
  stays schema-clean and the name is not lost.
- **Export size.** A six-shot H3 run is hundreds of megabytes of video. The
  export copies rather than hard-links, since `exports/` is what a user moves
  or deletes; the response reports the total so an agent can quote it.
- **`inputs/` and the realized file.** The export copies what an `output:`
  reference named from another run, but does not rewrite the reference,
  since the realized file is the immutable record of the run. The README
  says where each input came from.
- **History rebuild.** A run directory with `workflow.json` and
  `manifest.json` is enough to reconstruct a job row without the database.
  Not proposed here; noted because this is what makes it possible.
