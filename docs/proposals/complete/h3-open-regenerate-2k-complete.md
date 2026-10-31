# H3 Open-Regenerate-2K refiner LoRA and `refine_sigmas` (#1065)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 approved by Don with "approve defaults" (Q1 Kijai's
refiner-only rank-384 extraction stacked with `turbo-4step-keyframe`, Q2
Don's read of the crop as the pass line with PSNR/SSIM as a drift guard,
Q3 `refine_sigmas` and `num_inference_steps` both required and agreeing).
Two stages, #1081 and #1082, both verified 2026-10-31.

Prior art: HomoLapis/justae-flow (GPL-3.0, ideas only): its "Open-H3 2K"
preset pairs the regenerate LoRA with the H3 latent upscaler and runs 3
steps on a 768p draft. No code was taken.

## The question

Does a LoRA trained for the regenerate step beat dw's low-sigma refine
(#471 latent upscaler, #1043/#1048 `refine_tiles`) on the #1049 arms, and
does it load and fit on lem's 24 GB cards?

## What the design sweep found

- **Source and licence.** NVlabs, `Efficient-Large-Model/Open-H3-Regenerate-2K`
  (code in `NVlabs/Sana`, branch `sol-engine`), under the MiniMax H3
  Community License. The issue's `..._r128_comfyui_bf16` file is a
  community re-extraction with no stated provenance or licence: not used.
  `Kijai/MiniMax-H3-experimental` holds refiner-only extractions at rank 16
  and 384.
- **NVlabs ships a merge plus a non-LoRA part.** One FP32 adapter (~12.45
  GB, rank 576 on 312 linears) merging the refiner (r384), LightX2V FL2VA
  Turbo and LMS Sharp r64, plus `sharp_curve_branches.safetensors`: 50
  AdaLN branches on an 8-D timestep curve, which `load_lora_weights` can't
  carry (the same wall that made `loras/minimax-h3/lms.json` `rejected`).
- **The recipe.** 768p draft → 3D-conv latent upscaler → start at
  `(1-0.909375)·up + 0.909375·noise` → 3 Euler steps on
  `[0.909375, 0.857143, 0.666667, 0]` at CFG 1 → 2560x1440.
- **Corrections to the proposal.** Tiled refine at σ 0.91 keeps only 9% of
  the plate, so tiles would each re-render the whole prompt: full-frame
  1920x1088 was tried first. #1049's crop PSNR/SSIM measures drift, which a
  regenerate causes by design, so it could not be the pass line. The
  recipe's sigma points could not be written in dw (the refine schedule
  was always shift-spaced), hence stage B.

## Verdict

**Build smaller**, as a probe with a gate: A (catalog, docs, probe jobs),
then Don's gate, then B (`refine_sigmas` and the faithful arm).

## What was built

### Stage A, #1081: catalog entry, doc corrections, probes

- `loras/minimax-h3/open-h3-regenerate-2k.json`, `status: trial`: Kijai's
  `loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors`
  (5.9 GB) at revision `f5c3b1d5…`, `workflow: [t2va, fl2va]`,
  `stacks_with: [minimax-h3/turbo-4step-keyframe]`, no `lora_alpha`, the
  sharp-branch note, licence `minimax-h3-community`.
- The "API-only and not open-sourced" lines in
  `audits/2026-09-07-minimax-h3-audit.md` and
  `docs/superpowers/plans/2026-09-07-ltx-h3-catalog-repair.md` now point at
  Open-H3-Regenerate-2K. `WORKFLOW_GUIDE.md`'s "25-30 min, not measured"
  tiled-refine figure is now the measured ~12 min (RTX 3090, 124 frames).
- Probes on lem, workspace `qa-h3-regen-2k`, on #1049 arm (d)'s realized
  workflow (#500 crowd-gate draft, seed 42, 124 frames):
  - **P1, load:** clean on the default unpruned `transformer`; no
    unmatched-key or skipped-module warning. The only warning was the
    `adapter_unrecognized` filename heuristic, which the catalog entry
    clears.
  - **P2, fit:** full-frame 1920x1088 refine, strength 0.909375, 4 steps,
    shift 2: succeeded (`33feb79dab33`), **21.5 GB** reserved, 13.4 min
    (refine only; base/up cached).
  - **P3, tiles:** not run, since P2 fit.

### Stage B, #1082: `refine_sigmas`

- An optional `refine_sigmas` list on an H3 refine step.
  `DwH3RefineScheduleStep` runs it as given through `explicit_sigmas()` in
  `dw/pipeline_processors/h3_hold_steps.py`, unshifted, for both the video
  and audio schedulers, and rebuilds `row_timestep_plan` from it. Without
  it the shift grid is unchanged.
- Rules in one owner, `h3_rules.refine_problems`, called by the static
  check (`hold_audio.refine_strength_errors`) and the run-time check
  (`Pipeline._check_refine`): at least 2 points, strictly decreasing,
  first point equal to `refine_strength` within 1e-6, last point 0, length
  equal to `num_inference_steps`. Refused without `refine_strength`;
  allowed with `refine_tiles`.
- Tests: `TestSigmaListStatic`, `TestSigmaListRuntime`,
  `TestSigmaListOneOwner`, `TestSigmaListSchedule` in
  `tests/test_h3_refine.py`.
- Docs: `WORKFLOW_GUIDE.md` › *Pipeline Configuration* › *Explicit refine
  sigmas*, with the recipe as its example; the `ARCHITECTURE.md` H3 row.

## The measurements

Crop `[1200,700,600,388]` at 2.0 s and 4.5 s, against #1049 arm (a)
(`PreGateFitF-a_1080.2-0.0-video.mp4`):

| arm | job | peak reserved | wall | PSNR / SSIM vs (a) |
|---|---|---|---|---|
| #1049 (d) tiled refine, no LoRA | `0be863816672` | - | 14.8 min | 26.40 dB / 0.773 |
| #1049 (e) spandrel 2x | - | - | 7.05 min | 34.4 / 0.94 |
| #1049 (f) LTX upscale | - | - | ≈9.3 min | 18.7 / 0.50 |
| P2, regen LoRA, shift-2 grid | `33feb79dab33` | 21.5 GB | 13.4 min | 18.58 / 0.458 |
| faithful, regen LoRA, recipe sigmas | `51e1c17b4022` | 23.5 GB | 10.2 min | 18.63 / 0.451 |

The regenerate is far sharper (pores, stubble, coat weave, crisp flames
and cobbles; no seams or duplicated subjects), but at strength 0.909 it
redraws identity: it scores at (f)'s level, not (d)'s. The recipe's exact
sigma points are visually near-identical to the shift-2 grid and agree
within noise (+0.05 dB, -0.007 SSIM): at this strength they buy nothing
measurable.

## Deviations from the plan

1. **The gate added a step that was not recorded.** Don's ruling on the
   stage A record (2026-10-31) was "proceed, with a cheap lower-strength
   arm before stage B": run one or two arms at `refine_strength` about 0.6
   and 0.7, and plan stage B only if one kept (d)'s fidelity (about 26 dB /
   0.77) with visibly sharper detail; otherwise close the idea. **No
   lower-strength arm appears on #1065, #1081 or #1082.** Stage B was built
   and the faithful arm run without that record. The question Don asked,
   whether the detail survives once identity drift is gone, is still open
   (see Deferred).
2. **The refine step needs explicit `width`/`height` at 1920x1088.**
   Without them the layout defaults to 1344x768 and the transformer fails
   (`fc673788417c`, `index_copy_` size mismatch). Validate didn't catch it.
   Filed with the other side finding as #1087 and #1088.
3. **The guide subsection moved.** It was written as a `####` under the
   *refine_strength* subsection, which `get_guide` can't resolve; it is now
   a `###` under *Pipeline Configuration* (the stage B bounce).

## Bounces per stage

- **#1081 (A):** none. The tester counted one clause of M-F231 (d)
  (warnings beside a refusal) as the case's own error and filed a suite
  amendment on dkackman/harnest; validate drops warnings on any invalid
  result, for every model.
- **#1082 (B):** one. M-F233 step 1, `get_guide("workflows",
  section="Explicit refine sigmas")`, didn't resolve because of the
  heading level. Fixed docs-only (`02ebe65d`, merged as `1cc889fa`) and
  verified.

Cost: the stage comments name no `usage:` figures, so none is recorded.
The plan's estimates were $2-3 plus ~45 min GPU for A and $3-5 for B.

## Deferred and left open

- **The lower-strength arms (0.6, 0.7).** Don's gate question. They need
  no new code: the P2 workflow with a lower `refine_strength`, reusing the
  cached base/up, scored on the same crop. Their result decides whether
  the entry goes toward `proven` (and a template) or `rejected`.
- **The catalog entry stays `trial`.** Its `use_when` still says the exact
  sigmas "need refine_sigmas, #1065 stage B"; that input now exists, so
  the line should name it as available (a catalog change, left to the
  next build that touches the entry).
- **Sharp curve branches.** Need a new AdaLN module fed by the step
  timestep. Reopen if a lower-strength arm is close but soft and NVlabs'
  ablation shows the branches carry the sharpness.
- **2560x1440 output.** Needs the upscaler and tile caps raised and new
  VRAM pricing; the faithful arm already reserved 23.5 GB at 1920x1088, so
  it won't fit full-frame on 24 GB.
- **A template and a `minimax-h3` skill row.** Their own issue, after a
  passing gate.
- **The NVlabs merged FP32 file and the community r128 file.** Not used
  (Q1 default; unknown provenance).
