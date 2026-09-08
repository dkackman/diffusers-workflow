---
name: ltx-2.5
description: Use when a dw MCP server is connected and the user wants LTX-2.5 video - a short clip with its own soundtrack, a clip that starts from a picture or runs between two, a sharper full-size render, a 2x upscale, or a clip extended or chained longer. Picks the template for the shape, states the schedule and size rules, quotes cost, and carries the trained caption spec the prompt must follow.
---

# LTX-2.5 on a dw server

LTX-2.5 generates video and a soundtrack together, 24 fps, on a distilled
schedule that is not a knob. Every template here fits a 24 GB card. This
skill chooses the template and the arguments; the prompt is written to
Lightricks' own caption spec, quoted below from diffusers.

## Before anything

1. `get_server_info`: the device and the workspace. These templates quantize
   with SDNQ on CUDA; on an `mps` or `cpu` server say so and stop, because
   none of them fit there.
2. `list_workflows(shape="shot")`: all eight of the family's templates carry
   that shape. Take their current names, `summary`, `traits` and `cost` from
   the listing and trust it over the names quoted below.
3. `get_workflow` on the one chosen, for its variables and their defaults.

## Which shape is the request

- **A clip from text**: `templates/ltx2/text-to-video` (960x544, 121 frames, 5 s).
- **From a picture**: `templates/ltx2/image-to-video` (first frame; 481 frames
  is 20 s in one pass, so reach for extend or chain only past that);
  `templates/ltx2/keyframes` (first and last frames pinned);
  `templates/ltx2/enhance-prompt` (a one-line idea plus a picture; the model's
  own enhancer writes the caption).
- **Sharper at full size**: `templates/ltx2/two-stage`, the three-move distilled
  flow - eight sigmas at 768x448, a 2x latent upsample, then renoise and three
  stage-two sigmas at 1536x896 with the audio latents carried through. The
  upsample alone is soft; the refine pass is where the detail comes from.
- **A generative 2x render**: `templates/ltx2/generative-upscale` draws its own
  low-resolution pass and has the IC-LoRA re-render it at twice the size,
  inventing detail rather than interpolating. `base_width` and `base_height`
  are that first render's size, `width` and `height` the doubled target; both
  passes run the same eight distilled sigmas at guidance 1.0. Nothing in this
  family takes a user-supplied video, so a user's own footage is not a fit for
  any of these templates.
- **Longer**: `templates/ltx2/extend-clip` generates an opening and then
  continues it conditioned on the whole opening, not on a single frame - a
  supplied clip is not an input here either; `templates/ltx2/chained-segments`
  re-runs per segment on the previous last frame and stitches. Neither is a Lightricks recipe; both are dw's, and
  a single 481-frame pass reaches 20 seconds before either is needed.

If none fits, compose from `list_tasks` before authoring a new workflow, and
read the `workflows` guide's authoring section first.

## Hard rules

- `num_frames` is `8k + 1` (121, 241, 481); `width` and `height` are in multiples of 32.
  The templates generate at 24 fps. RoPE time is `frame / fps` and the model is
  trained around 24, 25, 30 and 60, so for a higher-fps request generate at 24
  (or condition at 60 at most - `MAX_CONDITIONING_FPS`, never 120) and let
  playback carry the rate. The temporal-upscaling path that renders 48 and 96
  fps belongs to the DFR pipelines, which no template here uses yet.
- The distilled transformer runs its eight trained sigmas (`DISTILLED_SIGMA_VALUES`)
  with `guidance_scale` 1.0 and STG and modality guidance off. No
  `num_inference_steps`. Those knobs mean something only against the dev
  transformer, which no 24 GB template ships.
- Stage two of the two-stage flow: renoise at 0.909375 (the first
  `STAGE_2_DISTILLED_SIGMA_VALUES` entry) and run its three sigmas at full size.
- An image condition is re-compressed at CRF 18 to match training and needs a
  PIL image; a multi-frame video condition is not re-compressed.
- Audio is generated in the first pass; nothing refines it afterwards, so
  carry the audio latents (two-stage) or pair the soundtrack back
  (`pair_audio`) on any step that works on frames alone.

## Prompts

A caption, not a tag list: one paragraph of roughly 150 to 220 words in the
present progressive, opening on the action, stating for every shot a shot
type, a camera motion (say static when there is none) and a viewpoint, with
the soundscape interleaved with the action rather than appended, in plain
observable words. For an image-conditioned clip describe only what changes
from the image; restating it invites a scene cut. The stored prompts under
`prompts/ltx2/` are written to this spec. For a one-line idea,
`templates/ltx2/enhance-prompt` runs the model's own enhancer with the same
spec. The spec itself, from `diffusers.pipelines.ltx2.utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`
(the image-to-video variant, `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`, adds the
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

1. `validate_workflow` first - free, and it catches arguments the pipeline
   does not accept.
2. Quote the listing's `cost`. Only `templates/ltx2/text-to-video` and
   `templates/ltx2/two-stage` declare one; for the other six say so and give
   the shape of the spend instead - a 121-frame clip at 960x544 is under a
   minute warm on a 24 GB card, the two-stage flow about eight, and extend
   and chain multiply by their passes. Either way get the user's go-ahead
   before `run_workflow` with `acknowledged_cost=true`.
3. `wait_for_job`, then `get_job` for the manifest. Writing a 121-frame
   1536x896 clip takes minutes after the last step ends; the job is not stuck.
4. You cannot watch a video: no tool returns a frame from one, and this family
   has no image steps for `get_output_image` to read. Hand the user the gallery
   `url` (`list_gallery`, or the manifest's file name) and ask them to look, and
   check what you can yourself - `get_job` for the manifest and its warnings,
   `get_gallery_metadata` for duration, size and whether an audio stream is
   present. Ask the user to look for the family's failure modes: a scene cut
   where the prompt contradicted the image; softness where the refine pass was
   skipped; a near-silent soundtrack where the caption gave the sound nothing
   to do.
5. After an inline run worth keeping, `get_job_workflow` and `save_workflow` it,
   so the next run is by name rather than by pasting JSON; `export_job` bundles
   the run — workflow, manifest, job row and media — for git. The bundle is on
   the server: fetch its zip URL and unpack it into `exports/` under the
   session's working directory, never a temp directory, and do not make a
   folder named after the job id first, since the archive already unpacks
   into one.

## Sources

Lightricks/LTX-2.5-Diffusers model card, the `ltx-pipelines` docs and
CHANGELOG (github.com/Lightricks/LTX-2), the diffusers LTX-2 pipelines and
`utils.py`. Read 2026-09-07; the audit is
`docs/proposals/audits/2026-09-07-ltx-2.5-audit.md`. Since 2026-08 Lightricks
route production quality through their DFR pipeline, which diffusers ships
and no template here uses yet.
