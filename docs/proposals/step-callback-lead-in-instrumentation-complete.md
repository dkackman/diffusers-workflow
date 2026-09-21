# Design: naming the pre-denoise lead-in on step-callback pipelines

Status: **proposal**. Written for #171 (filed by the regression agent, model
`opus` via provider `anthropic`, quoting a `triage: work` comment from
dkackman), investigated and drafted by the implementer agent, model `sonnet`
via provider `anthropic`, 2026-09-16.

## The ask, as filed

`templates/ltx2/text-to-video` ran 213.2s against a curated 1.8min (108s),
with 98.6s of it a silent "generating" lead-in before the first
`pipeline_step` event - unlike #95's block narration, this template emits no
block logs at all during that lead-in. The filer named two candidate causes:
the curated cost being stale, or the lead-in itself having grown as a
regression. A `triage: work` comment said this needs a measured run on `lem`
to tell the two apart.

## What the measured run on `lem` shows

I pulled the `events` JSON for three `templates/ltx2/text-to-video` jobs
directly from `jobs.sqlite` and read the phase timeline off `phase` /
`pipeline_step` / `step_end` entries (`at` is seconds since job start):

**`bd2f45b50862`, 2026-09-13, 108.1s total (closest to curated 1.8min):**
loading 6.8→69.8 (63.0s: transformer 21.0s, text_encoder 17.4s, pipeline
24.6s) → generating phase starts 69.8, first denoise step at 77.7 (**7.9s
lead-in**) → 8 denoise steps 77.7→101.2 (23.5s) → decode 4.4s → save 2.5s.

**`58430c14f254`, 2026-09-16, 213.2s total (today's #171 report):**
loading 1.8→84.0 (82.2s - already ~19s slower than the baseline run above)
→ generating phase starts 84.0, first denoise step at 182.6 (**98.6s
lead-in**, matching the issue exactly) → 8 denoise steps 182.6→206.0 (23.4s,
same pace as the baseline run) → decode 4.7s → save 1.9s.

**`83444a2ff3f1`, 2026-09-13, 249.0s total (the C-F005 job #171 cites for
"no block logs during the lead-in"):** loading 2.3→69.8 (67.5s, normal) →
generating phase starts 69.8, first denoise step at 79.6 (**9.8s lead-in**,
*not* anomalous) → 8 denoise steps 23.4s (normal) → decode 12.1s (a little
high but not the outlier) → save phase starts 115.1, but `step_end` does not
land until 248.2 - **a 133.1s stall inside "saving"**, with none of the
`writing .../wrote ... in N.Ns` log pair the other two runs both have at this
point.

Two conclusions:

1. **The curated `cost.minutes: 1.8` is not stale.** `bd2f45b50862` lands at
   108.1s against a curated 108s almost exactly. Candidate cause (1) from the
   issue is ruled out by this measurement.
2. **Two different intermittent stalls are being conflated as one symptom.**
   Today's regression run stalls in the pre-denoise lead-in (encode_prompt /
   connectors, per the architecture read below). The C-F005 job #171 cites as
   corroboration has a *normal* lead-in and instead stalls in the save phase
   (file write/mux). Denoise-step pacing (23.4-23.5s for 8 steps) and loading
   time (63-82s) are the same order of magnitude across all three runs;
   nothing in `git log -- dw/pipeline_processors/pipeline.py` between
   2026-09-13 and 2026-09-16 (`cd71fc5`, `20e59d3`) touches group offload,
   LTX2, or the save/mux path, so there is no code change to pin either stall
   to. Both look like host-side variance (GPU/CPU contention or I/O)
   surfacing in two different uninstrumented places, not a single regressing
   code path.

## Why this needs a decision and not an edit

`LTX2Pipeline` is a classic `DiffusionPipeline` subclass whose `__call__`
signature genuinely names `callback_on_step_end`, so dw takes the
step-callback instrumentation route (`_takes_step_callback` /
`_with_step_callback` in `dw/pipeline_processors/pipeline.py`), not the
block-narration route #95 added for true `ModularPipeline` instances. That
route only wraps the denoise loop itself - everything `__call__` does before
entering it (`check_inputs`, `self.encode_prompt(...)` against the
group-offloaded Gemma text encoder, `self.connectors(...)` against the
group-offloaded connectors component, latent prep) runs with no event of any
kind. That is architectural, not a bug in this template: any step-callback
pipeline with expensive pre-loop work has the same blind spot, and #95's
narration approach (patching tqdm / walking `SequentialPipelineBlocks`) does
not apply here since there is no blocks tree to walk.

Adding visibility here is a new instrumentation concept for a whole class of
pipelines (patching or wrapping specific pre-loop calls - `encode_prompt`,
`connectors`, or a generic "time between phase-start and first callback"
watchdog), not a one-line fix, so it falls under guardrail 7.

## Options

### A. A generic "still working" watchdog log, no pipeline-specific hooks

While waiting for the first `callback_on_step_end` invocation after a
`generating` phase event, emit a `log` event every N seconds (e.g. "still
generating, Ns since phase start, no denoise step yet"). Cheap, applies to
every step-callback pipeline uniformly, needs no per-pipeline knowledge of
`encode_prompt`/`connectors`. Downside: it says *that* something is slow, not
*what* - a consumer still can't tell text encoding from connector loading
from a genuine denoise-loop hang.

### B. Named sub-phase events for LTX2's specific lead-in calls

Wrap `encode_prompt` and `connectors` (or monkeypatch them the way
`_with_step_callback` wraps the pipeline's `__call__`) to emit `phase`/`log`
events naming each, mirroring #95's spirit but per-pipeline rather than
generic. Gives the precise breakdown this investigation needed to do by hand
against `jobs.sqlite`. Downside: bespoke per pipeline family - LTX2 today,
whichever pipeline is next tomorrow - and a maintenance burden if
`LTX2Pipeline.__call__`'s internals change upstream (this is exactly the kind
of coupling the ModularPipeline block-narration route was designed to avoid
by walking a declared tree instead of hardcoding call names).

### C. Do nothing beyond noting the finding

The curated cost is confirmed accurate; the two stalls found are each
one-off measurements, not a reproducible regression tied to a code change.
Recommend closing #171 as informational/wontfix and letting a future
instance of either stall (now that its shape is known) motivate whichever of
A/B is worth the maintenance cost.

## Recommendation

**A**, generic watchdog logging, as the first increment: it is the cheap half
that would have made both stalls visible in real time without per-pipeline
coupling, and it generalizes past LTX2 to any future step-callback pipeline
with expensive pre-loop work. B can follow later, scoped to LTX2 specifically,
if the watchdog shows this lead-in stalling often enough to be worth naming
precisely rather than just flagging.

## Open questions for Don

1. Is a periodic "still working, no denoise step yet" log acceptable, or does
   it need to be structured (a distinct event type a client can key off, like
   `phase`/`pipeline_step`) rather than a plain `log` message?
2. The C-F005 job's 133s save-phase stall (no `writing`/`wrote` log pair
   appearing at all) looks like a separate, possibly more concerning gap -
   worth its own issue, or fold into this one's watchdog scope since it's the
   same "phase started, nothing else logged for a long time" shape?
3. Given the curated cost is confirmed accurate, should #171 stay open only
   for the instrumentation question, or would you rather close it now (cost
   figure vindicated) and let the watchdog work track separately if approved?
