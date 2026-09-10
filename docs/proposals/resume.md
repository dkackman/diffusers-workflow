# Proposal: resuming a run — making the step cache survive the process

Status: draft / scoping only — no implementation.

## Problem

A multi-step run that dies part way through — the OOM killer taking the
worker, a cancelled job, a crashed box, a mistake in step 6 of 8 — starts again
at step 1 the next time it is asked for. It regenerates work that is already
sitting on disk, and on a video workflow that is measured in GPU-hours rather
than seconds. Reported from an MCP session as roughly an hour of GPU time lost
to a single crash.

The engine already has the two halves this needs and has never connected them:

- **The step cache** (`dw/step_cache.py`) knows precisely when a step may be
  skipped, and its criteria are already the strict ones. But it is a
  process-wide singleton, so it dies with the worker that crashed.
- **Run directories** (`dw/runs.py`) already write every step's files under
  `<output_dir>/<identity>/<run id>/` with a `manifest.json` beside them, and
  `output:` already resolves a path *out of* that tree from another run. But
  nothing reads a manifest back as an input to execution.

So the durable record of "step 4 produced these files" exists on disk and is
thrown away, while the component that would act on it lives only in RAM.

## Current behavior

| Piece | Where | Lifetime |
|---|---|---|
| Step skip decision | `StepCache.get`, keyed `(workflow_id, step_name)` | Process |
| Cached entry payload | `step_data`, `step_seed`, `result`, `output_dir`, `generation`, `upstream_generations`, `retained` | Process |
| Files a step wrote | `manifest["steps"][i]["files"]`, relative to the run dir | Disk, forever |
| Seed the run used | `manifest["seed"]` | Disk, forever (as of the seed fix) |
| Run identity | `new_run_id({"workflow": workflow_def, "arguments": arguments})` | Disk, in the directory name |

`Workflow.run` consults the cache for every step, server jobs included, and a
hit reports the earlier run's files with `reused: true` and writes nothing new.
That behavior — the thing a resume wants — is fully built. It just cannot be
reached across a process boundary.

Note also that the cache is disabled outright for a workflow that sets no seed
(`cache_enabled_this_run`), because a fresh random seed each run means no entry
could ever match. Any resume inherits that constraint: **a run is resumable only
if it is reproducible.** The companion seed changes (variable `seed`, and
recording the drawn seed in the manifest) are a prerequisite, not a nicety.

## Proposed shape

**Rehydrate the existing step cache from manifests, rather than build a second
resume path.**

The alternative — an explicit `--resume-from <run-id>` that replays a run and
skips steps by name — means a second set of rules about when skipping is safe,
sitting beside the six criteria `step_cache.py` already documents and enforces.
Two mechanisms answering the same question is how they drift. Rehydration adds
one loader and reuses every existing invariant.

### Mechanism

1. On `Workflow.run` start, when resume is enabled, read the manifests under
   this workflow's identity in the output root, newest first.
2. For each manifest whose recorded seed and step definitions still match this
   run, synthesize `StepCache` entries for its steps and `put` them before the
   step loop begins.
3. Everything after that is unchanged. `StepCache.get` decides. Its criterion 5
   (every named file must still exist) already handles a partially deleted run
   directory; criterion 4 (output root must match) already handles a resume
   pointed somewhere else.

For entry validity across processes, the manifest must carry enough to
reconstruct the comparison. Today it holds the run's `arguments` and seed but
not each step's *resolved* definition. Two options:

- **(a)** Store a hash of each step's resolved `step_data` in the manifest entry
  and compare hashes rather than values. Cheap, small, and matches how
  `pipeline_cache_key` and `new_run_id` already establish identity.
- **(b)** Store the resolved `step_data` itself. Larger and leaks realized
  values (a resolved argument can hold a PIL image), so (a) is preferred.

With (a), `StepCache` needs to accept an entry whose `step_data` is known only
by digest. That is a real change to its shape and the main design decision here.

### The hard boundary: results vs. files

A cache entry may hold a live `Result` (`retained`) or only `saved_files`. Only
the second survives a process. So:

- A step whose downstream consumers need only its **files** — which is most
  image and video work, and everything the `output:` reference already supports
  — can be resumed.
- A step whose result is consumed **in memory** by a later step (latents, an
  embedding, a mask handed to an inpaint step via `previous_result:`) cannot be
  rehydrated from disk, because those bytes were never written.

The proposal is to let such a step **miss** rather than to invent a
serialization format for arbitrary pipeline outputs. `StepCache.get` already
takes `needs_result` and already returns None when an entry lacks one, so this
falls out of the existing code with no new branch. Consequence worth stating
plainly: a workflow whose steps chain in memory resumes at the first such step
and re-runs from there. A workflow whose steps chain through saved files —
the multi-stage pattern the GYRE and marmot workflows use — resumes at the true
failure point.

A later stage could widen this by reloading saved media into a `Result` (the
loaders already exist for `asset:` and `output:`), turning "files only" entries
into serviceable ones for the common image/video case. Deliberately out of
scope for stage one.

### Surface

Resume should be **off by default and explicit**, at least initially. A run
that silently reuses a previous run's files is the kind of behavior that is
delightful until the one time it is wrong, and the step cache's in-process
version is already scoped to a session where the user can see what happened.

- CLI: `--resume` (reuse whatever matches from previous runs of this workflow),
  and `--resume-from <run-id>` to name one explicitly.
- Server/MCP: a `resume` flag on job submission, defaulting off. Note that a
  job carries its own `output_dir`, so resume is naturally confined to its
  workspace with no extra work.
- Reporting: reused steps already surface as `reused: true` in the manifest and
  in `step_end` events, so the gallery and job detail need no changes to show
  what was skipped. Worth confirming the UI actually distinguishes it.

## Staging

1. **Prerequisite (done): seed.** `seed` accepts a `variable:` reference, and
   the manifest records the seed actually used. Without both, a seedless
   workflow has nothing to resume against.
2. **Manifest carries step identity.** *Satisfied by the realized workflow*
   (shipped 2026-09-08; see CLAUDE.md's run-directory notes, and
   `docs/superpowers/specs/2026-09-08-job-record-and-export-design.md` for the
   design): every run now writes
   `workflow.json` beside its manifest with every mutable input pinned, so each
   step's definition as it actually ran is on disk to compare against, and the
   manifest's per-step files say what it made. A per-step digest may still be
   worth adding for a cheaper comparison, but the information is no longer
   missing.
3. **Rehydration.** A loader that turns manifests into `StepCache` entries, plus
   whatever `StepCache` needs to compare by digest. Behind `--resume`.
4. **Reload saved media into results.** Widen resumability past files-only
   steps, if stage 3 shows it is the common blocker.

Stages 1–3 are the useful unit; 4 is an optimization.

## Open questions

- **Scope of a match.** Should resume consider only the most recent matching
  run, or any run of that identity? Any-run is strictly more useful and makes
  the "fix one still" workflow work across days, but it also means a step can
  be served from a run the user has forgotten about. Leaning: most recent
  matching run in stage 3, widened later if it proves too narrow.
- **Interaction with pruning.** If output pruning ever lands, a resumable run
  is one whose files must not be pruned. Criterion 5 makes a pruned run fail
  safe (it misses and re-runs), so this is a quality-of-life question, not a
  correctness one.
- **Sub-workflows.** A sub-workflow inherits the parent's run directory and
  writes no manifest of its own, so its steps are invisible to a manifest-based
  resume. Its steps *do* appear in the parent's rolled-up manifest, but keyed by
  the child's `workflow_id`. Needs a decision before stage 3.
- **Flat layout.** `--output-layout flat` writes no manifests at all, so resume
  is simply unavailable there. Should it say so, or stay silent?
- **Confinement.** Reading manifests means reading paths written by an earlier
  run and republishing them. They are already confined to the output root and
  already validated on the `output:` path, but the resume loader is new code
  reading path-shaped data off disk and should go through the same validators.
