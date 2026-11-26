# Composition pass-through: a parent's run reaches its child's knobs (#1642)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v2 on #1642 is the full design. This file records what
shipped, what was cut, and why.

**Result:** every catalog template that composes another workflow now passes
the child's run-affecting values (size, frames, steps, image count, model)
through parent variables. Each of those variables defaults to the child's own
value. A seedless parent's drawn seed now reaches a child that declares its own
seed, so `compose-workflows` reproduces from its recorded seed. A catalog test
(`tests/test_composition_passthrough.py`) and a paragraph in
`docs/WORKFLOW_GUIDE.md` (*Composing a stored workflow*) keep new templates
from reintroducing hard-coded child values. There is no validate warning.

## The problem

The engine's rule is that a workflow's defaults live in `variables` and a run
overrides any of them with `name=value`. Several composing templates passed
child values as literals, or not at all, so a parent run couldn't reach them:
`describe-and-regenerate` (model path, steps, size), `best-of-n-to-video` (the
LTX child's frames and size), and the flux3 upsampler builtins, which had their
model, revision and token budget as undeclared literals. The plan's sweep also
found two things the issue missed. `compose-workflows` lost the parent's seed,
because `models/z-image.json` declares `seed: 42`, and it made 4 stills by
default.

## Verdict: build smaller

Don approved plan v1 with the delegate's recommendations on 2026-11-25. He
approved plan v2 the same way on 2026-11-26.
- **Q1:** a catalog test plus a guide paragraph, not a `validate_workflow`
  warning. A warning would fire on every UI Enhance run and on deliberate
  literals.
- **Q2:** no `device` pass-through to the flux3 callers. A literal `"cuda"` is
  correct on every worker: a `--devices` worker sees one card, and on mps
  `resolve_device` translates the value. Passing `cuda:1` down would break a
  pinned worker.
- **Q3:** `compose-workflows` defaults `num_images_per_prompt` to 1, so one
  still feeds one video. This is the one default that changed.
- **Q4:** names as written.
- **Q5 (v2):** a narrow engine change so that a null parent seed hands down the
  parent's drawn seed.

Corrections the plan made against the issue:
- The flux3 builtins declared no variables for `model_name`, `revision` or
  `max_new_tokens`, so they had to declare them before any parent could pass
  them.
- `device: "cuda"` is not a bug.
- `minimax/enhance-prompt-with-image.json` already followed the convention.

**Why v2.** Building stage A showed that v1's seed assumption was false.
`replace_variables` (`dw/variables.py`) drops any key whose `variable:`
reference resolves to null. So under a null parent seed, `"seed":
"variable:seed"` handed the child nothing, and z-image's 42 still won. Plan v2
added Q5. Its default was an engine exception, which Don approved.

## What shipped

### Stage A (#1725): thread the child knobs, and the null-seed hand-down

Branch `feat/1642-A-thread-child-knobs`. v1 work ended at `6e2a8f2d` and the
engine change is `9faede98`. It merged to `develop` at `309c6d02`. The
architecture fix (`feat/1642-A-arch-row`, `b1a541bb`) merged at `ec93c6b4`.
`deploy: server`.

- **Builtins.** `flux3_klein_upsample.json` and `flux3_klein_upsample_edit.json`
  declare `model_name`, `revision` and `max_new_tokens`, defaulting to the
  pinned values. Both revision sites and `from_hub.repo` read the variables, so
  the repo and its pin can't drift apart. The flux3 parents don't expose them
  (cut 2 below).
- **`templates/describe-and-regenerate.json`.**
  - Its path is `variable:image_workflow`, with default `../models/flux-dev.json`.
  - It passes `num_images_per_prompt` (1), `num_inference_steps` (25), `width`
    (768) and `height` (768).
- **`templates/best-of-n-to-video.json`.** It passes `video_width` (768),
  `video_height` (768) and `num_frames` (401) to `ltx2/image-to-video`. The
  inline SD 1.5 stills step stays.
- **`templates/compose-workflows.json`.** It passes `"seed": "variable:seed"` to
  both children, and a new `num_images_per_prompt` (default 1) to the image
  child.
- **`templates/sub-workflow.json`.** Its description says `lora` is fixed on
  purpose.
- **Engine (Q5).** `hold_null_seed_arguments` in `dw/workflow.py` is called
  from `Workflow._expand` before variable substitution.
  - When a `workflow` step authors `seed` as a `variable:` reference to a null
    variable, the child is handed null instead of the key being dropped. The
    #1641 fallback in `dw/workflow_run.py: prepare_definition` then gives the
    child the parent's run seed.
  - A literal seed still wins. A reference to a set variable passes its value.
    A step that passes no seed keeps the child's default.
  - Validation expansion behaves the same way.
- **Docs.** `docs/ARCHITECTURE.md` has a new **Composed child seed** row under
  Engine core, which the architecture bounce asked for. The
  `docs/WORKFLOW_GUIDE.md` seed paragraph says a null `variable:seed` hands down
  the drawn seed.
- **Tests.**
  - `tests/test_composition_knobs.py` is new. It runs the real expansion and
    `prepare_definition`.
  - `tests/test_flux3_catalog.py` now reads the declared defaults.
  - `COMPACT_BUDGET` rose to 16_055, measured at 16_030, with a dated comment.

Deviations the build recorded:
- The plan put the Q5 change in `_sub_workflow_action`. By the time that runs,
  `replace_variables` has already dropped the key, so the change lives in
  `Workflow._expand` instead. The behaviour is the one planned.
- `describe-and-regenerate`'s validate fingerprint changed, because the child's
  arguments are now passed explicitly. Its downloads, estimate and warnings did
  not change.

### Stage B (#1726): the guard and the rule

Branch `feat/1642-B-passthrough-guard`, commit `4710ec10`, merged to `develop`
at `63387056`. `deploy: server`, because `get_guide` serves the guide.

- **`tests/test_composition_passthrough.py`** walks every `workflow` step in
  `workflows/` and `dw/workflows/` and resolves each child.
  - A `variable:` path resolves through the parent's default. A `preference:`
    path resolves to every target in the key's ladder, and each literal is
    checked against each target.
  - Every literal passed for a variable the child declares needs a `DELIBERATE`
    entry with its reason. A stale entry fails the test.
  - These count as references: `variable:`, `previous_result:`, `prompt:`,
    `asset:`, `output:`, `gather:` and `item:` strings, and non-empty lists or
    dicts whose leaves are all references. A `constant:` counts as a literal.
  - The file has 35 tests.
- **`docs/WORKFLOW_GUIDE.md`, *Composing a stored workflow*** has one new
  paragraph, which states three rules:
  - pass a child's run-affecting values through parent variables whose defaults
    equal the child's;
  - pass `"seed": "variable:seed"` when the child declares a non-null seed (a
    seedless run hands down its drawn seed);
  - a literal is for a value the composition fixes on purpose.

Deviations:
- `DELIBERATE` has more entries than the planned one (`sub-workflow`'s `lora`).
  #1722 had moved seven portrait and board steps onto
  `variable:portrait_workflow` → `preference:text_to_image`. Those steps are in
  `minimax/dialogue-short`, `generated-subject-reference`, `music-video` and
  `storyboard`, and they pin `width`/`height` 768 and `num_images_per_prompt: 1`
  on purpose. A ladder has no single child default to pass through.
  `preferred/text-to-image`'s `num_images_per_prompt: 1` is also listed.
- The `preference:` handling above was added, because the plan was written
  before `preference:` existed.

## Cut, and what brings each back

- **Passing `device` through the three flux3 callers (Q2).** This comes back if
  a job's upsampler runs on the wrong backend.
- **Exposing the upsampler's model, revision and tokens on the flux3 parents.**
  The builtins declare them, so a hand-written parent can pass them, but the
  catalog parents keep BFL's pinned upsampler. This comes back with a field
  report that needs another upsampler.
- **A validate warning for literal child arguments (Q1).** This comes back if
  user-authored compositions keep hard-coding child values where the catalog
  test can't see them.
- **A general run-time override of any child's variable** (for example
  `video.num_frames=97`). It was never proposed as a stage. It would be its own
  `idea` if threading knobs keeps falling short.
- **`preference:text_to_image` as `describe-and-regenerate`'s default.** This
  is #1626's to do. The `variable:` path with the flux-dev default is a complete
  fix without it.
- **The Enhancer's literal `model_name`** (`dw/server/enhancers.py`). This is
  #1643's.

## Found along the way (filed separately)

- **#1755:** `best-of-n-to-video`'s judge step failed the run when the VLM reply
  carried no score ("Malformed limbs."). This blocked C-F1434 until it was
  fixed. It is closed and verified.
- **#1756:** `validate_workflow` accepted `num_frames`/`video_width` values that
  the LTX child refuses, because constraints weren't propagated through
  threaded knobs. It is closed and verified.
- **Not filed:** on a 17.8 GB mps machine, `sub_workflow_errors` checks the
  ltx2 child at its own defaults (768×768×401), not at the arguments handed to
  it, so it refuses `best-of-n-to-video` even when the parent passes smaller
  values. `develop` behaves the same way. lem's 24 GB card passes.

## Stages, bounces and cost

| Stage | Bounces | Notes |
|---|---|---|
| #1725 thread child knobs + null-seed hand-down | 2 | The first build stopped on the false v1 seed assumption, which led to the v2 re-plan and Q5. Architecture review bounced once for the missing map row, fixed at `ec93c6b4`. The first verify (19:03) passed acceptance 1–7 but left C-F1434 blocked by #1755. It passed on re-run at `7487de03`. |
| #1726 catalog pass-through test + guide paragraph | 0 | Verified on the first pass at `63387056`. |

The stage comments name no `usage:` figures, so cost isn't recorded here. The
plan estimated about $7 for v1 and about $8–9 for v2.

Verification on lem in workspace `regression-complete`:
- C-F1429 to C-F1435, SE-F086, C-F1466, C-F1467 and C-F1436 pass.
- `compose-workflows` with seed 7, 7 and 8 gave identical stills for the two
  7s (SSIM 1.0), embedded child seeds of 7, 7 and 8, and no 42 anywhere.
- A seedless run drew 5474498048674222. A `rerun_job` at that seed reproduced
  its still (SSIM 1.0).
- `best-of-n-to-video` with `num_frames=97 video_width=512 video_height=512`
  made a 512×512 clip of 97 frames at 24 fps.
