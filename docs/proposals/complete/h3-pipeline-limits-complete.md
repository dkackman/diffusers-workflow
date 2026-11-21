# Validate refuses H3 literals the pipeline would refuse (#1361)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-15). Plan v1 approved by Don as "build smaller", option (b), on
2026-11-14; one stage, #1364 (A), verified.

## The report

A tester session on lem (develop @ 388de1e9, verifying #1359) ran an inline
`ModularPipeline` step on `MiniMaxAI/MiniMax-H3`, `workflow: "ref2va"`,
960x544, with a literal `num_frames: 25`. `validate_workflow` returned
`valid: true`; `run_workflow` then failed job `445dc323ca8c` in diffusers
("must be between 120 and 360, got 25 (rounded up to 39)") after the model
load. It was filed as a regression of #584.

## Decisions

- **Triage ruling (2026-11-14):** not a regression. #584 was a docs fix, and
  validate checked only the rules a workflow declares; inline JSON declares
  none. Sent to the lead for a plan weighing four options: (a) catalog- or
  plugin-held constraints applied by model id, (b) limits read off the
  diffusers pipeline class, (c) a warning when a catalog-model step declares
  no rule, (d) a guide line only.
- **Plan v1: build smaller, (b) only.**
  - (a) cut: a new rules home outside the workflow, a model-id lookup, a copy
    of numbers diffusers already publishes, and template-test rework. It
    comes back for a family whose limits diffusers does not publish as
    readable class attributes.
  - (c) cut: more machinery (an inheritance-index sibling, a new
    `workflow_context` field) for a weaker result, since the run still starts
    and fails. It comes back for a constraint that is a house rule from the
    templates rather than the pipeline's own.
  - (d) folded in as one line in the guide. The skill already stated the rule
    and it did not stop this workflow.
- **Q1 [scope]:** (b). **Q2 [record]:** `CLAUDE.md`'s "model knowledge never
  in engine code" line amended: an engine check may name the diffusers class
  that enforces a limit, never a number. **Q3 [none]:** width/height
  multiples of 32 included (#789's class).
- **Non-goals held:** no catalog-held rules, no "you declared no rule"
  warning, no refusal of an off-grid count the pipeline accepts and rounds
  (130 runs as 141), no change to `variable_constraints`, no family but H3.

## What was built

**Stage A, #1364** (server, lem), `d1c1ed10`, merged to `develop` as
`a95fdbd8`:

- **`dw/pipeline_limits.py`** (sibling of `dw/reference_limits.py`):
  `pipeline_limit_errors` reads `fps`, `min_duration`, `max_duration`,
  `vae_frames_per_chunk` and `canvas_multiple` off
  `MiniMaxH3ModularPipeline` from a bare instance with no components loaded.
  The engine holds only the class pointer; the +5 of the 17n+5 grid comes
  from `h3_rules.RENDER_GRID`. For steps `h3_rules.is_h3_step` accepts
  (t2va, fl2va, ref2va), on the substituted, expanded definition:
  - a literal int `num_frames` whose grid rounding falls outside 5-15 s is
    refused at `steps[i].pipeline.arguments.num_frames`, naming the seconds,
    fps, class, rounded count and nearest accepted count (124..345 today);
  - a literal `width`/`height` off a multiple of 32 is refused at its own
    path, naming the values below and above.
  - Skipped: references and non-ints, absent arguments, non-H3 steps, and a
    `variable:`/`item:` argument whose variable declares a constraint (that
    one is reported once, at `arguments.<name>`, as before). A missing
    class, module or property checks nothing and logs at debug. A `for_each`
    member's error is at the written step and names the member.
- **Wiring:** `Check("pipeline_limits", ...)` after `reference_limits` in
  `dw/validation.py`; `ERROR_ORDER` updated. No MCP, REST, schema or syntax
  change: only new entries in `validate_workflow`'s existing `errors`.
- **Docs:** the `CLAUDE.md` amendment (Q2); `docs/ARCHITECTURE.md` rows for
  `pipeline_limits` and the previously missing `reference_limits`; the
  `dw/variable_constraints.py` docstring pointer; `docs/WORKFLOW_GUIDE.md`
  ("The loop" step 1 and a line in "Authoring a workflow from an agent");
  the `validate_workflow` description ("a bound the workflow declares, or
  the pipeline publishes, is an error here").
- **Tests:** `tests/test_pipeline_limits.py` (limits match the template
  derivation, boundaries, each workflow kind, for_each, skips, one error per
  value inline and through `templates/minimax/reference-to-video`, missing
  class), plus the #1361 repro through `Workflow.validation_errors`.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #1364 | 0 | Verified first time on lem @ a95fdbd8: M-F319..M-F326 pass. |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
The plan estimated ~$3-5. The architecture review passed.

## Verification notes

- The full #1361 workflow (quantization, LoRA, 2 steps) at `num_frames: 25`
  gives exactly one error, at `steps[0].pipeline.arguments.num_frames`.
- 346 and 360 are refused (round up to 362, 15.08 s); the 120..360 in
  diffusers' own message is the raw-count range, and the accepted on-grid
  counts are 124..345.
- `run_workflow` of the refused case is refused before queueing; no job.

## Deviations and follow-ups

- **`bench/workflows/seam-restore.json` now fails validate:** its ref2va
  restore passes use `num_frames: 22`, the refusal lem hit at run time in
  #1351. Fixing it belongs to #1325's re-plan. Its case in
  `tests/test_bench_workflows.py::test_every_bench_workflow_validates` is a
  strict xfail citing #1325/#1351; remove the mark when the workflow is
  fixed.
- WORKFLOW_GUIDE's validate list had no separate reference-limits item, so
  the new line sits at the end of "The loop" step 1.

## Deferred

- **Other families.** One joins by adding its class pointer once diffusers
  publishes the same properties.
- **(a) and (c)**, with the triggers above.
- **Risk kept open:** a diffusers rename makes the check go quiet (the run
  still refuses, as before #1364); the unit test reading the class fails at
  the bump.
