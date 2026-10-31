# H3 regional edit: source_video and a mask (#1062)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 approved by Don with its defaults (Q1 per-step blend,
Q2 build smaller, Q3 no paste-back, Q4 names as proposed). Plan v2, written
after Don's ruling on #1077's M-F218 bounce, approved 2026-10-31 with its
defaults (Q5 keep the feature, Q6 document the shifted-σ meaning rather
than re-scale). Two stages, #1077 and #1078, both verified 2026-10-31.

Prior art: einhorn13/mmh3_media (MIT, ComfyUI): `av_edit.py`'s per-stream
masks, `inpaint_prepare.py`, and `h3_tile_refine.py`. The concept was
re-implemented on the diffusers modular pipeline. No code was taken.

## The question

Does the diffusers MiniMax-H3 modular pipeline let the noise or latents be
masked, per stream, during the denoise? The audio half existed already
(`hold_audio`); the video half was the unknown.

## The answer: feasible as dw blocks, no upstream needed

The modular pipeline has no mask input, but it has what a mask needs:
- **per-row noise levels are native** (`row_timestep_plan`): conditioning
  and held audio rows already ride clean beside noisy rows;
- **one block writes rows** (`MiniMaxH3LoopSchedulerStep`), so a dw loop
  sub-block after `update` can put the kept rows back after every step,
  which is ComfyUI's per-stream `noise_mask`;
- **the audio-locked tile refine had already shipped** (#1043 / #1048,
  `refine_tiles`).

A `denoise_mask` input on `MiniMaxH3LoopSchedulerStep` would be a clean
upstream PR later, and the dw block would shrink to a pass-through. Not
pursued here.

## Verdict

**Build smaller.** A (engine) and B (template and skill). Audio intervals,
masked tiled refine and masked chained steps were cut, each with a
trigger (below).

## What was built

### Stage A, #1077: engine regional edit

Arguments on an H3 `t2va`, `fl2va` or `ref2va` step:
- `source_video`: the take to edit. It must match the step's `width`,
  `height` and `num_frames`. It stands in for a refine's `latents`.
- `mask_video` **or** `mask_image`: white regenerates, binarized at luma
  128 through Pillow's `L` conversion (the same rule as
  `remove_object.binarize_mask`), max-pooled onto the patch grid (16 px ×
  patch size; one latent frame, about 3.4 pixel frames, × patch time
  size), then dilated by `mask_grow` cells (a whole number, 0 or more,
  default 1). With no mask the whole source is re-denoised.
- `refine_strength`: required, in (0, 1] with a `source_video` (1.0
  regenerates the region from noise). Without a source the refine's range
  stays (0, 1), with its message unchanged.
- `hold_audio`: normally `source_video` itself; the source mp4's own track
  is held.

Code:
- `dw/pipeline_processors/h3_mask.py`: `insert_mask_blocks`, and the
  mask-to-rows arithmetic (`binarized`, `regenerate_cells`, `keep_rows`).
- `dw/pipeline_processors/h3_mask_steps.py`:
  - the source-latents block before `set_timesteps`, which VAE-encodes the
    source on the canvas;
  - the mask-blend block in the `denoise` loop after `update`. It writes
    each kept row back as the source noised to the next sigma, with the
    refine's own noise, so the kept rows end as the source latent;
  - `edit_summary`, the regional-edit `log` job event (regenerated/kept
    cells, `mask_grow`, `refine_strength`), added after Don's ruling.
- `dw/pipeline_processors/h3_rules.py`: `mask_problems`, `media_problems`
  and `edits_region`, shared by validate and `_with_mask` in
  `pipeline.py`.
- `dw/hold_audio.py`: `mask_errors`, Check `regional_edit`. It probes the
  source and mask through `ValidationContext.probe`, so every refusal
  fires at `validate_workflow` and nothing is queued.
- Docs: `WORKFLOW_GUIDE.md` › *H3: regional edit with source_video and a
  mask* and *Choosing refine_strength for a regional edit* (the measured
  bands and the shifted-σ note). Also the `ARCHITECTURE.md` H3 row.
- Tests: `tests/test_h3_mask.py` (arithmetic, insertion, the blend ending
  at the source, rules, static media, strength 1.0, the job event) and
  additions to `tests/test_h3_hold_audio.py`.

The output is the decode as-is (Q3: no pixel paste-back), so kept pixels
are near-identical to the source, not bit-identical. Outside-mask SSIM
measured about 0.98 at every strength.

### Plan v2: the measured strength bands

v1 promised that `refine_strength: 0.8` changes the masked region "per the
prompt". On the M-F218 take (`ep6-cold-open.mp4` + `h3-mask/region.png`,
`mask_grow` 1, 5 turbo steps, `video_shift` 6, seed 1062; prompt asking for
a bright red raincoat) it didn't. Don ruled for a sweep:

| strength | inside SSIM | outside SSIM | inside the mask |
|---|---|---|---|
| 0.80 / 0.85 | 0.86 / 0.83 | 0.978 | re-detail only: the source wins |
| 0.90 | 0.78 | 0.977 | a held object is replaced; same person, pose, clothing |
| 0.95 | 0.65 | 0.978 | garments reworked too; face, hair and pose mostly hold |
| 1.00 | 0.44 | 0.975 | the prompt lands, but a fresh generation (identity lost) |

The numbers sit high because `refine_strength` is σ₀ on the shifted grid.
At shift 6 the unshifted start is u = s / (6 − 5s): 0.8 → 0.40,
0.9 → 0.60, 0.95 → 0.76. Q6 kept the semantics (re-scaling would move
`refine_tiles` too). A big recolour of a large region needs 1.0, which is
a documented limit, not a promise.

### Stage B, #1078: template and skill branch

- `workflows/templates/minimax/regional-edit.json`: shape `shot`, traits
  `needs-input-media` and `has-audio`. Defaults are 960×544×124,
  `refine_strength` 0.9, 5 turbo steps, shift 6, and
  `hold_audio: variable:source_video`. Cost **2.1 min on an RTX 3090**,
  measured: a cold run took 120-123 s (encode 45 s, denoise 45-47 s,
  decode 19 s).
- `minimax-h3` skill: the "fix one region of a good take" branch in
  `SKILL.md`, with `references/regional-edit.md`. It covers making the
  mask (segment on a frame, or upload), choosing the strength from the
  bands, the recolour limit, and the patch-cell grain.

## Deviations from the plan

- **Every refusal is at validate.** The first build checked media only at
  the call, so an all-black mask queued a job (the first tester bounce).
  `media_problems` now runs at validate on the files that resolve.
- **Strength 1.0 is accepted with a `source_video`**, following "1.0
  regenerates from noise". It stays refused without one.
- **The template runs 5 steps, not the sketch's 8**, matching the turbo
  LoRA and the measured bands.
- **The acceptance strength moved from 0.8 to 0.9** (plan v2). The red
  raincoat is no longer required inside the mask.

## Bounces per stage

- **#1077: one architecture bounce, four tester bounces, then a park.**
  - architecture: the mask arithmetic was hand-written instead of on
    Pillow, torch and scipy, and `mask_image` took an RGB round trip that
    could binarize differently from `mask_video`;
  - tester: refusals fired at the run rather than at validate (M-F216), and
    M-F214/M-F215 (a)(d) were short. Fixed by validate-time media checks,
    holding the source mp4's own track, and strength 1.0 with a source;
  - tester: M-F218, the region changed but not per the prompt at 0.8, and
    no job event named the edit. The driver parked the stage after four
    hand-offs. Don ruled: add the event, and re-plan (v2) on a measured
    sweep;
  - tester: M-F216 (k), a string `"0.8"` validated clean. Not a code
    defect: the carrier's variable default was a number, so the argument
    was converted before the rule saw it. Tests were added that pin where
    a string reaches the rule;
  - tester: M-F217, the latent-refine range message had changed wording
    for the no-source case. Restored verbatim, and the exact sentence is
    pinned.

  Twice an architecture review sent the stage back with no findings and no
  record, which looked like a failed reviewer session. Nothing was changed
  for either. It verified at `a34b4d15`.
- **#1078: one.** M-F249: the listed cost of 3.0 min was 32% over the
  measured 2.05 min. It was recalibrated to 2.1, and the stage verified at
  `5e04cab0`.

No `usage:` figures were recorded on the stages, so cost is left out. The
plan's estimate was $6-9.

## Deferred and left open

- **Audio intervals and "audio follows video mask".** Cut until a field
  report wants a line re-sung or re-spoken inside a kept take. The
  mechanism is the same blend on audio rows, with `refine_problems`'
  "every audio row held" rule relaxed.
- **Mask + `refine_tiles`.** Refused (`TILE_CONFLICTS`). It comes back when
  a 1080p regional fix is asked for, by cropping the mask per tile the way
  `latents` already is.
- **Mask on a chained step.** Refused, until a chained regional-edit
  request comes in.
- **The big-recolour case**, where identity holds at strength 1.0. Revisit
  with an H3 refiner LoRA (#1065) or an upstream mask input.
- **Strength generality.** The bands come from one take, one seed and 5
  turbo steps at shift 6. The guide states those conditions, and the u
  formula says which way another shift moves them.
- **Upstream `denoise_mask`** on `MiniMaxH3LoopSchedulerStep`: an
  optional diffusers PR. It would reduce the blend block to a
  pass-through.
- **Q1's alternative** (pin kept rows clean at t = 1.0) was not tried.
- **Stale v1 cases.** M-F218, M-F222 and the v1 lines of M-F220/M-F221
  expect v1's 0.8 / 0.7 / "0.5 / 1.0". The tester asked the curator to
  retire or amend them in dkackman/harnest#158. M-F244 to M-F249 are
  authoritative.
