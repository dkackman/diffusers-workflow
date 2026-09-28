# `ltx2/refine-clip`: LTX-2.5 two-stage refine of an existing mp4 (#543)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-09-28). Plan v2 approved by Don (Q1-Q3 answered as the defaults) on
2026-09-27; one stage, #549, verified 2026-09-28.

## The report

`templates/ltx2/two-stage`'s `refine` step (renoise at 0.909375, the three
`STAGE_2_DISTILLED_SIGMA_VALUES`) could only start from latents its own
`base` pass made. #543 asked for the same refine on a clip dw did not make,
and assumed that meant a new engine task to VAE-encode an mp4 into LTX-2.5
latents, plus answers on audio latents, the encode symbol, size/frame
constraints and VRAM. It is the companion to #542 (`upscale-clip`, the
IC-LoRA route).

## The finding that shrank it

**The 2x case needs no encode task.** `LTX2LatentUpsamplePipeline.__call__`
takes `video=` as an alternative to `latents`: it VAE-encodes the frames,
resizes them to `height`/`width` and returns un-normalized 5D latents at 2x,
the same form `two-stage`'s `upscale` hands to `refine`. `LTX2Pipeline`
normalizes 5D input itself, so there is no double-normalization risk. The
whole chain is existing pieces in one template.

Other corrections the plan made to the issue:
- **`audio_latents=None` is accepted** (refine samples fresh audio noise),
  but three sigmas of noise is not a deliverable, so the template drops the
  generated track and pairs the source's own back with `pair_audio`.
- **No encode symbol to pin.** What is pinned is the renoise scale
  (`STAGE_2_DISTILLED_SIGMA_VALUES[0]`) and the sigmas by `constant:`.
- **The upsampler encodes with `sample_mode="sample"`**, not argmax. Noted
  as a risk; deterministic under the seeded generator, and the renoise at
  0.91 swamps it.

## Decisions (plan v2)

- **Q1: keep both routes.** #542 re-renders with the source as a reference
  (invents detail, center-crops, needs a gated LoRA); this one renoises the
  source's own latents (stays closer to its structure, stretches, no LoRA).
- **Q2: refuse a silent source early**, before any pipeline loads. Don may
  revisit (e.g. keeping refine's generated track) in a new plan version.
- **Q3: the 1x refine and the audio encode stay cut** (see Deferred).

## What was built

**Stage A, #549** (server catalog + plugin skill; no engine code):
`workflows/templates/ltx2/refine-clip.json` (id `LTX2RefineClip`, shape
`shot`, traits `needs-input-media`, `has-audio`). Steps:

1. `source_audio`: `normalize_audio` of `source_video` to -3 dBFS, not
   saved. It is also the silent-source refusal: `load_audio` raises on a
   video with no audio stream, so the job fails at step 0 in about 1 s.
2. `source_frames`: `loop_frames(video: variable:source_video, num_frames:
   variable:num_frames)`, not saved. Added after the bounce (below).
3. `upscale`: `LTX2LatentUpsamplePipeline` on `previous_result:source_frames`,
   latents out, not saved; its VAE is an explicit component with tiling,
   published as `shared_components: ["vae"]`.
4. `refine`: `LTX2Pipeline`, `latents: previous_result:upscale.frames`, no
   `audio_latents`, `noise_scale` 0.909375, the stage-two sigmas, guidance 1,
   STG/modality off, `reused_components: ["vae"]`; saved to `intermediate/`.
5. `with_source_audio`: `pair_audio(fit: "video")` of the refined picture
   and `previous_result:source_audio`; saved to `final/`.

- Variables: `source_video` (placeholder `asset:clip.mp4`, which does not
  exist, like `upscale-clip`'s), `prompt`, `negative_prompt`, `width` 512,
  `height` 288, `num_frames` 121, `frame_rate` 24.0, the SDNQ dtypes, `seed`.
- `variable_constraints`: `num_frames` on `8n+1`, `width`/`height` multiples
  of 32, no `snap` (copied from `two-stage`).
- `cost_drivers`: `num_frames`, `width`, `height`. `cost`: **2.9 min**,
  RTX 3090 24 GB, measured cold on lem at the defaults (job `dad16ac7253d`).
  No `vram_estimate`, like the siblings; the measured run fit 24 GB.
- Tests: `TestLtxRefineClip` in `tests/test_catalog_structure.py` (the
  renoise scale, the latents and trim wiring, the -3 dBFS pair, and a real
  `loop_frames` run cutting 130 frames to the first 97). `COMPACT_BUDGET`
  raised to 9_150.
- Docs: the `ltx-2.5` skill (two-stage entry, the Repairing section's
  contrast with `upscale-clip`, the audio rule, the cost line), the
  `workflows/templates/ltx2/README.md` row, and a `docs/RECIPES_24GB.md`
  pointer.

Deviations from the plan, as built:
- **`refine`'s `width`/`height` are the source size, not 2x.** `LTX2Pipeline`
  reads its shape off the latents, as `two-stage`'s refine does; the output
  is 2x the stated size.
- **The silent-source check is the first step**, not a separate probe. The
  tester accepted a first-step refusal (M-F066 arm b).
- **The upsampler reads `previous_result:source_frames`**, not
  `variable:source_video`. The plan's premise that "the upsampler truncates"
  to `num_frames` was wrong (next section).
- **A source shorter than `num_frames` is lapped from its first frame**
  (`loop_frames`' behavior), not refused; the description says not to do
  it. Probing the source at validate stayed cut.
- **An off-ratio source is stretched**, as planned, and documented.

## The plan's gap: the upsampler does not truncate

`LTX2LatentUpsamplePipeline.__call__` overwrites `num_frames` with
`len(video)` when handed `video` (`pipeline_ltx2_latent_upsample.py:354`);
its only trim is a floor to `8k+1`. So a 121-frame source asked for 97 came
out at 121 frames of picture while refine's (discarded) audio followed
`num_frames`, and `pair_audio` paired the full track. The `source_frames`
trim step fixes it; its float [0,1] frames match the old PIL path through
`VideoProcessor.preprocess_video` exactly at the same size and within 1/255
after a resize. `cost` was not re-measured for a CPU frame copy.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #549 | 1 | M-F065 short arm: `num_frames: 97` gave a 121-frame picture (the upsampler ignores `num_frames` for `video=`). Fixed with the `source_frames` trim. |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
The plan estimated $4-6.

Open against the suite, not the code: the tester filed an amendment on
dkackman/harnest to bring M-F062's text in line with the as-built wiring
(`source_frames`, source-size refine, the first-step audio check). The
silent fixture M-F066 uses is `asset:refine/src-512x288-silent.mp4`, made
on-box and kept.

## Deferred

- **A 1x refine** (re-detail at the source's own size). The one case that
  really needs a VAE-encode task: a latents `returns` kind or a
  `save: false` artifact task, plus a second LTX VAE through `cached_model`.
  Back on a field report of a clip to clean up without doubling it.
- **Encoding the source soundtrack into `audio_latents`.** No diffusers LTX2
  pipeline calls `audio_vae.encode`, and nothing provides the log-mel front
  end to pin. Back when diffusers ships one, or on a report that the paired
  original track is wrong for the refined picture.
- **Keeping refine's generated track for a silent source** (Q2's revisit).
- **Probing the source at validate** (its length or aspect ratio), cut the
  same way as #542.
- **Quality on foreign footage** is evidenced by the verify runs only: the
  stage-two sigmas were tuned for the same model's base-pass latents.
