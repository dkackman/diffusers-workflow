# Proposal: parameter sweeps, batch rerun, and side-by-side comparison

Status: **not committed** - scoping notes written against audit finding U6.
Nothing here should be implemented without a fresh look at whether it is
still worth the cost.

## U6, as filed

> No parameter sweep, no side-by-side run comparison, no batch rerun.

`JobManager.rerun(job_id)` (`dw/server/jobs.py:347`) re-submits exactly the
spec a previous job carried - `workflow_path`/`workflow`/`base_dir` plus that
job's `arguments` dict, unchanged - through `manager.submit(...)`. One job
id in, one new job id out. `POST /api/jobs/{job_id}/rerun`
(`dw/server/app.py:440`) is a thin wrapper over that. There is no notion of
"rerun this with prompt A, B, and C" or "run this same workflow at
`num_inference_steps` 20/30/40 and show me all three." A user who wants that
today opens `JobPage`, hits rerun, waits, edits `arguments` by hand, reruns
again - once per value, serially, with no grouping tying the results
together afterward.

## 1. Sweep spec: reuse `previous_result`'s cartesian machinery, or something new?

### What the existing machinery actually does

`dw/previous_results.py:get_iterations()` is the engine's cartesian-product
expansion. It walks one step's argument template, finds every
`previous_result:` reference (`find_previous_result_refs`), and for each
combination of values those references could take, produces one full
argument dict via `itertools.product`. It is guarded by `MAX_ITERATIONS =
10000`. This is what turns "4 images x 3 masks" into 12 pipeline calls -
**within a single running job, at a single step**, fed by artifacts an
earlier step already produced. It has no notion of a job boundary, a
workflow *variable*, or anything the manager's `submit()`/`rerun()` sees:
by the time `get_iterations` runs, the job is already `RUNNING` and the
worker process is deep inside `step.py`.

A sweep, by contrast, needs to vary a **workflow variable** (`variable:` -
`dw/arguments.py:389` - the substitution class fed by
`JobRequest.arguments`, a flat `Dict[str, Any]`) *before* a job is even
submitted, and it needs the result to be N separate jobs the queue can run,
track, cancel, and display individually - not N pipeline calls fused inside
one job's manifest and one event stream.

### Recommendation: a new construct at the API/submission layer, not a reuse of `previous_result`

Reusing `get_iterations` directly is the wrong fit and loses on three counts:

- **Wrong axis.** It expands references to *prior step output* inside one
  running job. A sweep expands *submission-time variable values* into
  *multiple jobs*. Coercing a sweep into a `previous_result:`-shaped
  reference would require either inventing a fake "step" whose artifacts are
  the sweep values (semantically backwards - nothing produced them) or
  running `get_iterations` in the manager instead of the worker, which
  is a rewrite of what the function is for, not a reuse of it.
- **Wrong granularity for the rest of this doc's asks.** Batch grouping,
  per-run cancellation, and a comparison UI all need N distinct job rows
  with their own ids, statuses, and manifests (section 2). A cartesian
  expansion *inside* one job gives one job id, one manifest, one event
  stream carrying all N outputs interleaved - exactly the shape the
  Gallery/JobPage side-by-side view (section 3) would then have to un-merge.
- **Wrong safety envelope.** `MAX_ITERATIONS = 10000` exists to bound
  in-process pipeline calls sharing one GPU worker's memory within a single
  job's lifetime. A sweep is bounded by "how many jobs am I willing to queue
  and wait for" - a much smaller number in practice (worth a much lower
  default cap, see below) and a different kind of limit (queue depth /
  wall-clock, not memory).

Concretely, add a `sweep` field to the job submission request rather than
extending `arguments`:

```jsonc
POST /api/jobs
{
  "workflow_path": "workflows/ZImage.json",
  "arguments": { "prompt": "a cat", "num_images_per_prompt": 4 },
  "sweep": { "variable": "steps", "values": [20, 30, 40] }
}
```

`sweep.variable` names one key already present (or addable) in `arguments`;
`sweep.values` is the list to substitute it with. The manager expands this
into N `submit()` calls, one per value, each with `arguments` overridden at
that key - reusing `submit()`'s existing validation (schema, path security,
argument warnings) unchanged, per job. This is deliberately the *simplest*
construct that satisfies "vary argument X over these values": a single
variable name and a flat value list, not a workflow-JSON-level DSL. See
section 6 for why multi-argument sweeps are phase 2, not phase 1.

**Alternative considered and rejected**: a `sweep` block inside the workflow
JSON itself (schema-level, like `steps`/`variables`). This was rejected
because a sweep is a property of *one submission*, not of the workflow
definition - the same `ZImage.json` should be sweepable over `steps` today
and over `prompt` tomorrow without editing the file each time, matching how
`arguments` already overrides workflow variables per-submission rather than
being baked into the JSON. It would also entangle `dw/workflow_schema.json`
and schema validation with a concern that belongs to the job queue, not the
engine - workflows run standalone via `dw.run` with no queue underneath
them at all.

## 2. Batch grouping in `JobManager`

`Job.__init__` (`dw/server/jobs.py:200`) has no batch concept; `Job.spec`
carries only what one run needs. Minimal additions:

- **`Job`**: a `batch_id: Optional[str]` field, `None` for an ordinary job.
  All jobs from one sweep submission (or one batch rerun, section 4) share
  a `uuid4().hex[:12]` batch id, generated once by the new
  `submit_sweep()`/`rerun_batch()` entry point, the same id-shape `Job.id`
  already uses.
- **`Job.summary()`/`detail()`**: include `batch_id` (default `None`) so
  `JobsPage`/`JobPage` can group without a second lookup.
- **`JobHistory` (sqlite)**: add a `batch_id TEXT` column via the same
  additive-`ALTER TABLE` pattern already used for `events`
  (`jobs.py:74-76`: check `PRAGMA table_info`, add the column if absent, old
  rows keep `NULL`). No migration script, no version table - the codebase's
  existing convention for schema growth in this file. `recent_summaries()`
  and `_to_detail()` add `batch_id` to their `SELECT`s and dict.
- **`JobManager`**: a `submit_batch(base_kwargs, variable, values)` method
  that loops `values`, calling the existing `submit()` once per value with
  `arguments = {**base_kwargs['arguments'], variable: value}`, stamping the
  shared `batch_id` onto each returned `Job` before queuing. Validation
  failure on one value should not silently drop the rest of the batch or
  submit a partial batch invisibly - fail the whole request before any job
  is queued (validate every expanded arg dict first, exactly as `submit()`
  already validates before queuing today), so a batch is all-or-nothing at
  submission time even though the N jobs run independently afterward.
- **No change to `_run_loop`/`_run_job`/`_consume_results`.** Batch jobs are
  ordinary jobs in the pending queue - `batch_id` is metadata for grouping
  and display, not a new execution mode. This keeps the one-worker-FIFO
  invariant untouched, which matters: nothing about sweeps should imply
  concurrent GPU use.

`GET /api/jobs` already returns everything `list()` produces; adding
`batch_id` to summaries is enough for the UI to group client-side (`Map` by
`batch_id`, matching the existing `statusFilter`-driven `$derived` filter in
`JobsPage.svelte:16`) without a new endpoint. A `GET /api/jobs?batch_id=...`
filter is a nice-to-have, not required for phase 1, since the full list is
already small enough to filter client-side (`recent_summaries(limit=200)`).

## 3. UI surface

Per U5's finding, `ArgumentsEditor.svelte` is the right home for the sweep
*control* - sweeps vary an argument, and this is the component that already
owns per-argument rendering (`ui/src/lib/editor/ArgumentsEditor.svelte:88-159`,
one `.row` per key with its widget chosen by `widgetFor`). Concretely:

- **`ArgumentsEditor.svelte`**: next to each row's existing input, a small
  toggle ("sweep this") that swaps the single-value widget for a
  comma/newline-separated value-list input (reusing the `textarea`/`json`
  widget machinery already present for other multi-value cases, e.g. the
  `widget === 'json'` branch at line 119). Only one row may be in sweep mode
  at a time for phase 1 (single-argument sweep, see section 6) - the
  component enforces this by clearing any other row's sweep state when one
  is turned on, not by hiding the toggle elsewhere, so the constraint is
  visible rather than silent. The submitting caller (wherever `args` is
  currently POSTed to `/api/jobs`, e.g. the workflow run page) reads this
  state and, when a row is in sweep mode, sends `sweep: {variable, values}`
  instead of folding the value into `arguments` directly.

- **`JobsPage.svelte`**: this list is already a flat, filterable table
  (`nameFilter`/`statusFilter`, `$derived` over `jobs`, `ui/src/lib/pages/JobsPage.svelte:10-17`).
  Grouping needs a small, additive change: when consecutive/matching
  `batch_id`s are present, render a collapsed batch header row (workflow
  name, value list summary, aggregate status - "3 running, 1 succeeded")
  that expands to the same per-job `<a href="#/jobs/{id}">` rows already
  rendered today (`JobsPage.svelte:91-145`). Jobs with no `batch_id` render
  exactly as they do now - this is additive, not a rewrite of the list.

- **`JobPage.svelte`**: currently one job's detail plus a `Rerun` button
  (`JobPage.svelte:163`, `api.rerunJob(jobId).then((j) => go('jobs', j.id))`).
  When `job.batch_id` is set, add a small "N other runs in this batch" link
  that navigates to a new comparison view (below) rather than trying to
  cram side-by-side comparison into the single-job page, which is laid out
  for one job's manifest/events/warnings and shouldn't be overloaded.

- **Side-by-side comparison - new, not a GalleryPage retrofit.**
  `GalleryPage.svelte` is a *file* browser (loaded outputs across all jobs,
  filtered by folder/name, `ui/src/lib/pages/GalleryPage.svelte:20-93`) -
  its unit is a file, not a job, and it already resolves `sourceJob` per
  file for the "open the job that produced this file" link
  (`GalleryPage.svelte:79`, `284-287`). Comparison's unit is a *batch*, and
  what a comparison view needs (fixed set of N jobs, N argument-value
  headers, N result grids side by side) doesn't fit Gallery's filtered,
  open-ended file list without twisting its query model. The concrete,
  low-risk answer is a new route/page, `BatchPage.svelte`
  (`#/batches/{batch_id}`), fed by `GET /api/jobs?batch_id=...` (section 2):
  one column per job, each column's header showing the swept value and
  status chip (reusing `JobPage`'s `<span class="chip {job.status}">`
  pattern), each column's body showing that job's manifest thumbnails
  (reusing whatever `JobPage`/`GalleryPage` already use to render an output
  file - the manifest-to-thumbnail logic should be factored out once shared
  between three call sites rather than copied a third time). Diffing
  (pixel-level or metadata-level) is explicitly out of phase 1 (section 6).

## 4. Batch rerun

Extending single-job rerun to a batch is small if `rerun()` and the new
`submit_batch()` compose: a `rerun_batch(batch_id)` method collects every
job (live + historical, same `jobs.get`/`history.get` fallback `rerun()`
already does at `jobs.py:349-366`) sharing that `batch_id`, reads each one's
stored `spec`/`arguments` pair, and resubmits each through `submit()` under
a *new* shared `batch_id` - it does not reuse the old one, so history keeps
the original batch and the rerun batch as distinct groups rather than
silently merging reruns into an old batch's row count. Route:
`POST /api/jobs/batches/{batch_id}/rerun` (`201`, mirroring
`POST /api/jobs/{job_id}/rerun`), returning the list of newly queued job
summaries plus the new `batch_id`. `JobsPage`'s batch header (section 3)
gets a "rerun batch" action next to (not replacing) each row's existing
single-job rerun.

One thing this does *not* need to solve: rerunning a batch with *different*
sweep values than the original. Phase 1 batch rerun replays the same N
values that already ran (useful after a code/model update, or to retry a
partially-failed batch) - "rerun with new values" is just submitting a new
sweep from `ArgumentsEditor`, not a rerun-batch concern.

## 5. Interaction with cancellation (S6)

`JobManager.cancel(job_id)` (`jobs.py:408`) already operates per-job: pop
from `_pending` and finish immediately if `QUEUED`, forward to
`worker_manager.cancel()` if it's the one `RUNNING` job. Nothing about batch
grouping changes that mechanism - a batch's jobs are ordinary queue entries,
so per-job cancel already works unmodified once `batch_id` exists on `Job`.

What's worth adding, cheaply, is a **"cancel batch" convenience** that is
sugar over the existing per-job call, not new cancellation semantics: a
`cancel_batch(batch_id)` on `JobManager` that calls the existing `cancel()`
for every job (live or historical no-op) sharing that `batch_id` - queued
ones drop instantly, the one currently running gets the same
`worker_manager.cancel()` signal a single cancel would, the rest stay queued
until their turn (matching FIFO - there is still exactly one worker). This
is a loop over the existing method, not a change to `cancel()`'s contract,
matching how `aggressive-cancellation.md` frames the cooperative-flag model
as something to build *on top of*, not around. UI: one "cancel batch" button
on the `JobsPage` batch header and on `BatchPage`, next to (not replacing)
each job's own per-row cancel - a user who only dislikes one variant's
progress shouldn't have to kill the whole batch to stop it.

## 6. Effort/risk and phasing

**Phase 1 (minimal, recommended scope):**
- Single-argument sweep only (`sweep: {variable, values}`), values as a
  flat list of scalars matching the variable's existing type - no nested
  sweeps, no cross-argument products.
- `JobManager.submit_batch()` + sqlite `batch_id` column + `batch_id` on
  `Job.summary()`/`detail()`. `POST /api/jobs` accepts an optional `sweep`
  field (`JobRequest` gains `sweep: Optional[SweepRequest]`); the manager
  branches to `submit_batch` when present, otherwise unchanged single-job
  `submit()`.
- A **low** cap on sweep size distinct from `MAX_ITERATIONS` - values in the
  range of 2-20 make sense for a queue a person is going to sit and watch
  fill up job-by-job; something like `MAX_SWEEP_VALUES = 25` guards against
  a fat-fingered value list serializing hundreds of multi-minute jobs onto
  the one worker. Reject at submission (`ValueError`, same 400 path
  `submit()` already uses), before anything is queued.
- `JobsPage` grouping (collapsed batch rows), `ArgumentsEditor` sweep toggle
  on one row at a time, a minimal `BatchPage` grid view (thumbnails + status
  per value, no diffing).
- Batch cancel (section 5) - genuinely cheap once `batch_id` exists, worth
  bundling into phase 1 rather than deferring.
- Rough size: comparable to or smaller than `aggressive-cancellation.md`'s
  own estimate for its "option 2" - a few hundred lines split across
  `jobs.py`/`app.py` (backend, well-contained, extends patterns already in
  the file) and a similar amount of new Svelte (`BatchPage.svelte` new,
  `ArgumentsEditor`/`JobsPage`/`JobPage` each get a focused, additive
  change). No changes to `step.py`, `previous_results.py`, `worker.py`, or
  the schema - this stays entirely in the server/UI layer, which keeps risk
  low: a bug here cannot corrupt a running job's execution, only its
  submission/grouping/display.

**Phase 2 (later, only if phase 1 sees use):**
- Multi-argument sweeps (`sweep: {variables: [...], values: [[...], ...]}`
  or a Cartesian `{variable: values}` map) - genuinely more complex: needs
  its own combination-explosion guard analogous to `MAX_ITERATIONS`, a
  richer `ArgumentsEditor` UI (more than one row in sweep mode at once, with
  a combination-count preview so a user sees "12 jobs" before submitting),
  and a comparison grid that's 2-D instead of a single row of columns.
- Batch rerun with edited values (rerun this batch's *shape* against a new
  value list, not just a replay).
- Comparison-view diffing: metadata diff (which arguments actually differ
  between two jobs in a batch - trivial, just a dict diff over stored
  `spec`/`arguments`) as a cheap first step; visual/pixel diffing of output
  images as a materially bigger, separate piece of work if wanted at all.
- `GET /api/jobs?batch_id=...` as a real server-side filter, if the client-
  side `Map`-group in `JobsPage` ever becomes a real cost (it won't at
  `recent_summaries(limit=200)` scale).

Phase 1 deliberately stops short of anything that touches `dw/step.py`,
`dw/pipeline.py`, or the worker protocol - a sweep is purely "submit N jobs
instead of 1, remember they're related." That's what keeps this a
server/UI-layer feature rather than an engine change, and why it's scoped
smaller than U6's framing might suggest at first read.
