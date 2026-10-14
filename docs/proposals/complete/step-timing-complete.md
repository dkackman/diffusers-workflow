# Per-step timing and step-cache skip counts (#867)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-14). The idea came from the H3 quality matrix on lem (workspace
`h3-quality`), which had to rebuild phase minutes and s/pass from the event
log with `tools/phases.py`, and could only infer from s/pass whether a
`first_block` cache had skipped anything. Plan v1 was designed on #867. Don's
review made v2, which moved the carrier to the manifest entry and added a
real-pipeline cache-context test. Don approved v2. It was built as stages
#885 (1) and #886 (2). The plan comment on #867 stays the full design; this
is the record of what shipped.

## The problem

Two questions had no answer from dw: *where did a generation step's time
go*, and *did the step cache skip anything*. A script could rebuild the
first from events, but only while the 200-event history tail still held
load and encode. Nothing could rebuild the second, because diffusers' cache
hooks keep only the latest decision.

## Verdict: build smaller

Cut from the proposal, with what would bring each back:
- **`compile_s` as its own field.** Compile is lazy and runs inside the
  first denoise forward, so dw can't separate it. Shipped instead:
  `first_pass_s`, `pass_s` (the median of the rest) and `compiled`. On a
  cold run, compile ≈ `first_pass_s − pass_s`. *Trigger:* a diffusers or
  torch hook that reports compile time.
- **`skipped_passes_equiv`.** It is `skipped` in other units.
- **Skip counts for `taylorseer`, `faster`, `text_kv`.** Their hooks keep no
  per-forward decision, so they report type and config with null counts
  (Don, Q3). *Trigger:* a matrix run that needs them. For taylorseer, the
  route is to wrap `_measure_should_compute`.

Non-goals stayed out: pricing from timings (#866), per-block stats, UI
display, and timing on `wait_for_job` (Don, Q1).

## Plan history

- **v1:** a top-level `JobDetail.timing` list, kept in a new history column.
- **v2 (approved):** Don's review moved `timing` onto each step's
  `ManifestEntry`. That entry already has `extra="allow"`, is stored in
  history and is written to the run's `manifest.json`. The change dropped the
  history column, the migration and the `Job` fold, and timing survives
  export. Stage 2's first test became a tiny real pipeline with CFG, to catch
  a counter that reads outside the nested `cache_context("cond"/"uncond")`.
  Q1–Q4: no `wait_for_job` timing, one summed entry per chain step,
  TaylorSeer type-only, timing first.

## What was built

### Stage 1 (#885): `timing` on manifest entries

Commits 6342ab36 (engine), 7dd0bf89 (tests and docs), 831fc660 (map row,
`workflow_run.py` back under its size ceiling) and 23bb31d5 (the `passes`
ruling).
- `StepTimer` (`dw/pipeline_processors/timing.py`) runs on the worker's
  monotonic clock. It records `load_s`, `load_cached`, `encode_s`,
  `denoise_s`, `passes`, `first_pass_s`, `pass_s`, `compiled`, `decode_s`,
  `save_s` and `segments`, rounded to 0.1. A mark that never fired gives
  `null`.
- The denoise start comes from a self-removing denoiser pre-forward hook on
  callback pipelines, and from the denoise bar opening on modular pipelines.
- The timing dict reaches the manifest entry and `step_end` through
  `record_step`. A reused step has no `timing`. A chain step gets one summed
  entry.
- `ManifestEntry.timing` and `StepTiming` are in the API contract, with
  OpenAPI and the UI types regenerated. `docs/SERVER.md` and `docs/MCP.md`
  correct "generating → first step is encode". `docs/WORKSPACES.md` has a
  *Step timing* field table. `docs/ACCELERATION.md` explains how to read
  `first_pass_s − pass_s`.

**Ruling made in the build:** `passes` counts the denoise passes that
actually ran (one per scheduler timestep), not `num_inference_steps`. SD
1.5's PNDM runs N+1 timesteps in PLMS mode (20 → 21). Reporting 20 would
have broken `denoise_s ≈ first_pass_s + (passes − 1) × pass_s`, which #866
needs. Only descriptions changed. The suite's `passes: 20` literals in
C-F430 and C-F434 were sent for amendment (dkackman/harnest#110, #111).

### Stage 2 (#886): `timing.cache` and the `cache_stats` event

Commits 95f9cffa, ca8ad005 (folded into `timing.py`), ea5943dd (map row) and
0ead3b7f (the event's wire fix).
- A forward hook on the cache transformer, registered in
  `stateful_cache_context`, reads the leader hook's `should_compute` once
  per forward, inside the pipeline's cond/uncond context. It counts
  `forwards`, `computed` and `skipped` for `first_block` and `mag`, and
  reports `hit_rate = skipped / forwards` to 3 places. `mag` also reports its
  config keys. A chain sums its segments' counts.
- One `cache_stats` event per pipeline call, emitted before dw resets the
  cache state. It carries `step` and, on a chain segment, `segment`.
- If the installed diffusers moves the private hook state, the counts are
  null, one `cache_stats_unavailable` warning fires, and the run continues.
  Pin tests on the installed diffusers check the hook names and state
  attributes.
- `docs/ACCELERATION.md` *Reading the skip counts*: a 0-skip run is not
  bit-identical to an uncached one (the matrix's P2 finding).

**Deviations:** there is no separate `cache_stats.py`, and
`StepTiming.cache` is typed as an open `dict` rather than its own model,
because the architecture ratchet refused a new module and a crossing of
`api_models.py`'s line ceiling. The shape is the same. The MCP surface
budget went from 14,990 to 15,005 (+15 description tokens, dated).

## Verification and bounces

| Stage | Bounces | What bounced |
|---|---|---|
| #885 | 2 | Architecture review: there was no map row for the new owner (fixed 831fc660). Verify on mini-ai: C-F430 `passes` was N+1 (ruled correct, docs 23bb31d5). |
| #886 | 2 | Architecture review: there was no map row (fixed ea5943dd). Verify on mini-ai: no `cache_stats` event ever reached the job, because the event's own `type` key collided on the worker wire (fixed 0ead3b7f, which sends it as `event_type`). |

- #885 was verified on mps (mini-ai) and cross-checked on CUDA (lem). The
  modular-pipeline case (C-F433) was proven on lem against a MiniMax Music3
  job, since neither box has a modular image template.
- #886 was verified on mps against an inline FLUX.1-dev pipeline. SD 1.5 has
  no transformer, so `first_block` is "not applied" there, and Z-Image
  refuses `enable_cache()`. The H3-on-CUDA case (C-F445) is now an ordinary
  case for lem. It had not run when #886 closed.
- Cost is not recorded: the stage comments name no `usage:` figures.

## Found along the way (filed separately)

- #895: `validate_workflow` accepted `faster` on guidance-distilled
  FLUX.1-dev, and the run then failed inside diffusers' FasterCache hook.
  Fixed on develop (0dcf45ad).
- `validate_workflow` reported `downloads_required: []` for FLUX.1-dev
  before a ~33 GB download (raised by the #886 tester).
- MiniMax Music3 accepted `num_inference_steps` without a warning and ran
  40 passes anyway (noted on #885, not filed).

## Follow-up

#866's estimator, and `observed_cost.py`'s cold/warm split, can now read
`timing.load_s`, `load_cached`, `passes` and `pass_s`. That work belongs to
#866.
