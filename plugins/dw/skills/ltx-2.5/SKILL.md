---
name: ltx-2.5
description: Use when a dw MCP server is connected and the user wants LTX-2.5 video - a short clip with its own soundtrack, a clip from one picture or between two, a sharper full-size render, a 2x upscale of its own or the user's clip, a subject held across a clip from a reference sheet, a blurry or compressed clip restored, or a clip extended or chained longer. Picks the template, states schedule and size rules, quotes cost, and carries the trained caption spec the prompt must follow.
---

# LTX-2.5 on a dw server

LTX-2.5 generates video and a soundtrack together, 24 fps, on a distilled
schedule that is not a knob (except refine-in-place's `strength`, 0 preserve
.. 4 reinterpret). Every template here fits a 24 GB card.

## Before anything

1. `get_server_info`: the device and workspace. On `mps` they run
   slower than CUDA `cost` says (only text-to-video has run
   there); on `cpu` say so, stop.
2. `list_workflows(shape="shot")`: every template in the family carries
   that shape. Take names, `summary`, `traits` and `cost` from the
   listing over names quoted below.
3. `get_workflow` on the one chosen, for its variables and defaults.
4. Before anything near the card's ceiling - a full-size refine, 481 frames, a
   2x upscale - `get_memory` on an idle server; a leftover `live: true` figure
   is an earlier run's, and `clear_memory` (idle only) drops it and the step
   cache.

## Which shape is the request

- **A clip from text**: `templates/ltx2/text-to-video` (960x544, 121 frames, 5s).
- **From a picture**: `templates/ltx2/image-to-video` (first frame; 481 frames
  is 20s in one pass, so extend or chain only past that);
  `templates/ltx2/keyframes` (first and last frames pinned);
  `templates/ltx2/enhance-prompt` (a one-line idea plus a picture; the model's
  enhancer writes the caption).
- **Sharper at full size**: `templates/ltx2/two-stage` - eight sigmas at
  768x448, a 2x latent upsample, then renoise and three stage-two sigmas at
  1536x896 carrying audio latents through. The upsample alone is soft;
  the refine adds the detail. On the user's clip: `refine-clip`.
- `templates/ltx2/diffusion-decode` compares decoders. Do not offer it: without
  a `shi-labs/natten` build its fallback OOMs on 24GB at any size.
- **A generative 2x render**: `templates/ltx2/generative-upscale` draws its own
  low-res pass, then an IC-LoRA re-renders it twice the size,
  inventing detail. `base_width`/`base_height` are the first render's size,
  `width`/`height` the doubled target. The user's clip: `templates/ltx2/upscale-clip`.
- **Keeping a subject across a clip**: `templates/ltx2/reference-sheet`. The
  family's only identity route; the user must author the reference sheet
  - one composite image, a clean panel per character, prop and location, no
  text, bigger panels for what matters. What isn't on the sheet won't
  appear. `reference_frames` must stay at or above 121.
- **Repairing the user's footage**: `templates/ltx2/restore-deblur` for
  spatial defocus, `templates/ltx2/restore-decompression` for low-bitrate
  artefacts. Each inverts one defect and no other - neither upscales or
  removes motion blur or grain - so name the defect and let the user
  correct you. To upscale or sharpen: `templates/ltx2/upscale-clip`
  (IC-LoRA re-render), `width`/`height` 2x the source's, or `refine-clip`
  (its own latents, no LoRA), `width`/`height` the source's and output 2x,
  other ratios stretched. Both: `num_frames` at most its length, soundtrack
  kept, silent source refused. Same size and length, no 2x:
  `templates/ltx2/refine-in-place` (`strength` 0-4, default 2). Footage
  longer than `num_frames`: `templates/ltx2/restore-long` restores it window
  by window. Its `windows` list needs `ceil(source_frames / (num_frames -
  overlap))` entries, `index` 0 up - read `source_frames` from
  `get_gallery_metadata`, keep `num_frames` 121 and `overlap` 16 (105 new
  frames per window), and let `validate_workflow` name any entry to add or
  drop. At most 32 windows.
- **Longer**: `templates/ltx2/extend-clip` continues an opening conditioned
  on all of it, not one frame; `clip` extends an existing clip (`width`/
  `height` matched, shorter than `num_frames`; clip_frames unused) instead
  of generating; `templates/ltx2/chained-segments` re-runs per segment on
  the previous last frame and stitches. Both are dw's own recipes (Lightricks
  added a chunked one 2026-09-29); one 481-frame pass reaches 20 seconds first.

If none fits, compose from `list_tasks` before authoring a new workflow;
read the `workflows` guide's authoring section first. Other
LoRAs: `list_loras` first.

## Hard rules

- `num_frames` is `8k + 1` (121, 241, 481); `width` and `height` are multiples of 32,
  generated at 24 fps. RoPE time is `frame / fps` and the model is
  trained around 24, 25, 30 and 60, so for a higher-fps request generate at 24
  (or condition at 60 at most - `MAX_CONDITIONING_FPS`, never 120) and let
  playback set the rate. 48 and 96 fps are the DFR pipelines', unused here.
- The distilled transformer runs its eight trained sigmas (`DISTILLED_SIGMA_VALUES`)
  with `guidance_scale` 1.0, STG and modality guidance off. No
  `num_inference_steps`. Those knobs are the dev transformer's; no 24 GB
  template ships them.
- Stage two of the two-stage flow: renoise at 0.909375 (the first
  `STAGE_2_DISTILLED_SIGMA_VALUES` entry), three sigmas at full size.
- A hand-authored ladder: three sigmas, no trailing 0 (the scheduler appends
  it), `noise_scale = sigmas[0]`.
- An image condition is re-compressed at CRF 18 to match training and needs a
  PIL image; a multi-frame video condition is not.
- Audio is generated in the first pass and nothing refines it, so carry the
  audio latents (two-stage) or pair the track back (`pair_audio`, as the
  `-clip` upscales do) on a frames-only step.

## Prompts

A caption, not a tag list: one paragraph of roughly 150 to 220 words in the
present progressive, opening on the action, stating for every shot a shot
type, a camera motion (say static when none) and a viewpoint, soundscape
interleaved with the action rather than appended, in plain
observable words. For an image-conditioned clip the image gives the look and
the caption the motion: open by matching the image faithfully, never
contradict it, one continuous take, no cuts. The `ltx2/` stored prompts
(`list_prompts`) follow it. The duration predictor times the clip as written:
put beats in the prompt ("she pauses") or set a duration. On-screen text and
chaotic physics are unreliable.

**Several shots**: one chronological paragraph, no shot list or sluglines. At
each cut name the transition in prose ("a hard cut to"), re-establish scale,
angle and lighting, reuse the same identifiers for recurring subjects, and
say what the audio does. 2 to 4 shots; stay single-take for image-to-video,
lip-synced dialogue or an unbroken camera move. Skip the enhancer.

Before writing a caption, read `references/caption-spec.md`: the training
caption spec, verbatim from the pipeline, which the caption must follow.

## Run and judge

1. `validate_workflow` first - free, and catches bad arguments.
2. Quote `plan.estimate` from the validate answer (wall clock, loading
   included) and name any `downloads_required` - an IC-LoRA template pulls a
   gated weight the box may not have. Only `text-to-video`, `two-stage`,
   `refine-clip` and `refine-in-place` (~1.5 min, RTX 3090) carry a `cost`;
   for the rest give the shape - a 121-frame clip at 960x544 is under two
   minutes cold on a 24 GB card, a minute loading; extend and chain
   multiply by their passes.
   Get the go-ahead, then `run_workflow` with `acknowledged_cost` set to the
   plan's `{fingerprint, minutes, downloads}`.
3. `wait_for_job`, `timeout_seconds` = estimate plus margin;
   `timeout_applied_seconds` is what you got (`timeout_capped`: the cap cut
   it). Call again while `still_running`.
4. Writing costs on a long chain: `"result": {"save": false}` on every step
   not worth keeping, as `two-stage` does for `base` and `upscale`; a miss
   is silent. A saving step carries a `subfolder` - the one shown to the user
   `final`, the rest `intermediate` - so `list_gallery(subfolder="final")`
   lists only deliverables. Keep both in anything you compose.
5. Judge it yourself: `get_output_frames(count=12)` for a clip's shape,
   `seams=true` for a chained clip's joins, `at` near the end for a scene cut
   where the prompt contradicted the image or softness where the refine pass
   was skipped, and `get_output_audio` for a near-silent soundtrack. Then
   `get_job` for the manifest and its warnings, `get_gallery_metadata` for
   duration, size and audio presence, and give the user the gallery `url`
   (`list_gallery`, or the manifest's file name).
6. Save a keeper (`get_job_workflow`, `save_workflow`) to rerun it by name.

## Sources

Lightricks/LTX-2.5-Diffusers model card, the `ltx-pipelines` docs and CHANGELOG
(github.com/Lightricks/LTX-2), docs.ltx.io prompting guide, the diffusers LTX-2
pipelines and `utils.py`. Read 2026-10-02; audit
`docs/proposals/audits/2026-10-02-ltx-2.5-audit.md`.
