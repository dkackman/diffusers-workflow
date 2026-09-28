---
name: ltx-2.5
description: Use when a dw MCP server is connected and the user wants LTX-2.5 video - a short clip with its own soundtrack, a clip from one picture or between two, a sharper full-size render, a 2x upscale of its own or the user's clip, a subject held across a clip from a reference sheet, a blurry or compressed clip restored, or a clip extended or chained longer. Picks the template, states schedule and size rules, quotes cost, and carries the trained caption spec the prompt must follow.
---

# LTX-2.5 on a dw server

LTX-2.5 generates video and a soundtrack together, 24 fps, on a distilled
schedule that is not a knob. Every template here fits a 24 GB card. This
skill picks the template and arguments; the prompt follows the caption
spec below.

## Before anything

1. `get_server_info`: the device and workspace. On `mps` they run
   slower than CUDA `cost` says (only text-to-video has run
   there); on `cpu` say so and stop.
2. `list_workflows(shape="shot")`: every template in the family carries
   that shape. Take current names, `summary`, `traits` and `cost` from
   the listing, trusting it over names quoted below.
3. `get_workflow` on the one chosen, for its variables and defaults.
4. Before anything near the card's ceiling - a full-size refine, 481 frames, a
   2x upscale - `get_memory` on an idle server and read a `live: true`
   reading's `gpu_memory_allocated_mb`; only those are the worker's.
   `info: null` means nothing is resident, and `live: false` is cached from
   another moment. A non-trivial idle figure is an earlier run's leftover,
   subtracted from what this one has. With the server idle, `clear_memory`
   clears it (refused while queued or running) and drops the step cache, so
   the next run, even a seeded rerun, is cold and regenerates. Re-read
   `get_memory` to confirm; don't retry a failed attempt unless cleared.

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
  the refine pass supplies the detail.
- `templates/ltx2/diffusion-decode` compares decoders. Do not offer it: without
  a `shi-labs/natten` build its fallback OOMs on 24GB at any size.
- **A generative 2x render**: `templates/ltx2/generative-upscale` draws its own
  low-res pass, then an IC-LoRA re-renders it twice the size,
  inventing detail. `base_width`/`base_height` are the first render's size,
  `width`/`height` the doubled target. For the user's own clip,
  `templates/ltx2/upscale-clip`: `width`/`height` 2x the source's,
  `num_frames` at most its length; its soundtrack is kept.
- **Keeping a subject across a clip**: `templates/ltx2/reference-sheet`. The
  family's only identity route; the user must author the reference sheet
  - one composite image, a clean panel per character, prop and location, no
  text, bigger panels for what matters. What isn't on the sheet won't
  appear. `reference_frames` must stay at or above 121.
- **Repairing the user's footage**: `templates/ltx2/restore-deblur` for
  spatial defocus, `templates/ltx2/restore-decompression` for low-bitrate
  artefacts. Each
  inverts one defect and no other - neither is an upscale, neither
  removes motion blur or grain - so name the defect and let
  the user correct you.
- **Longer**: `templates/ltx2/extend-clip` continues an opening conditioned
  on all of it, not one frame; `clip` extends an existing clip (`width`/
  `height` matched, shorter than `num_frames`; clip_frames unused) instead
  of generating; `templates/ltx2/chained-segments` re-runs per segment on
  the previous last frame and stitches. Neither is a Lightricks recipe; a
  single 481-frame pass reaches 20 seconds before either is needed.

If none fits, compose from `list_tasks` before authoring a new workflow;
read the `workflows` guide's authoring section first.

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
- An image condition is re-compressed at CRF 18 to match training and needs a
  PIL image; a multi-frame video condition is not.
- Audio is generated in the first pass and nothing refines it, so carry the
  audio latents (two-stage) or pair the track back (`pair_audio`) on a
  frames-only step.

## Prompts

A caption, not a tag list: one paragraph of roughly 150 to 220 words in the
present progressive, opening on the action, stating for every shot a shot
type, a camera motion (say static when none) and a viewpoint, soundscape
interleaved with the action rather than appended, in plain
observable words. For an image-conditioned clip describe only what changes
from the image; restating it invites a scene cut. The `ltx2/`
stored prompts (`list_prompts`) follow it. The spec, from
`diffusers.pipelines.ltx2.utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` (the
image-to-video variant, `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`, adds the
describe-only-changes rule):

## The trained caption spec

```
You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.

```

## Run and judge

1. `validate_workflow` first - free, and catches bad arguments.
2. Quote `plan.estimate` from the validate answer (wall clock, loading
   included) and name any `downloads_required` - an IC-LoRA template pulls a
   gated weight the box may not have. Only `text-to-video` and `two-stage`
   carry a `cost`; for the rest give the shape - a 121-frame clip at 960x544
   is under two minutes cold on a 24 GB card, a minute loading; extend
   and chain multiply by their passes.
   Get the go-ahead, then `run_workflow` with `acknowledged_cost` set to the
   plan's `{fingerprint, minutes, downloads}`.
3. `wait_for_job` with `timeout_seconds` = the estimate plus a margin
   (`timeout_capped` says the server's cap cut it; call again while
   `still_running`), then `get_job` for the manifest.
4. Writing still costs on a long chain, so only worthwhile steps
   should: `"result": {"save": false}` on the rest, as `two-stage` does for
   `base` and `upscale`. Missing it is silent. What does write carries a
   `subfolder` - the step the user is shown `final`, every other saving step
   `intermediate` - so `list_gallery(subfolder="final")` lists only
   deliverables. Keep both in anything you compose.
5. Judge it yourself: `get_output_frames(count=12)` for a clip's shape,
   `seams=true` for a chained clip's joins, `at` near the end for a scene cut
   where the prompt contradicted the image or softness where the refine pass
   was skipped, and `get_output_audio` for a near-silent soundtrack. Then
   `get_job` for the manifest and its warnings, `get_gallery_metadata` for
   duration, size and audio presence, and hand the user the
   gallery `url` (`list_gallery`, or the manifest's file name).
6. After a run worth keeping, `get_job_workflow` and `save_workflow` it, so
   the next run is by name not pasted JSON; `export_job` bundles it on the
   server. `auth_required: false` - fetch `open_url` into `exports/` under
   the working dir (never a temp dir; unpacks into a job-id folder).
   `true` - hand `open_url` to the person instead, keep using
   `get_output_image`/`_audio`/`_frames`

## Sources

Lightricks/LTX-2.5-Diffusers model card, the `ltx-pipelines` docs and CHANGELOG
(github.com/Lightricks/LTX-2), the diffusers LTX-2 pipelines and `utils.py`.
Read 2026-09-07; audit `docs/proposals/audits/2026-09-07-ltx-2.5-audit.md`.
