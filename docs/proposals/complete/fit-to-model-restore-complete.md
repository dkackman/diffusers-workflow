# `fit_to_model` / `restore_to_source`: exact size and frame-count round trip for v2v (#602)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-06). Plan v1 approved by Don 2026-10-05 and folded into v2; that
approval was withdrawn when stage B's build found v2 couldn't be wired. v3
approved 2026-10-06 (Q2′) and recorded as v4. Two stages, #631 and #632,
both verified 2026-10-06. Builds on `ltx2-upscale-clip-complete.md` (#542)
and `ltx2-refine-clip-complete.md` (#543), whose crop and stretch rules it
replaces.

The concept came from vrgamegirl19/comfyui-vrgamedevgirl, whose
source-available license is incompatible with Apache-2.0. Everything here
was re-implemented from the concept. No code, prompt text or presets were
taken.

## The report

The v2v templates made the caller conform the footage, and the two 2x
templates did it in opposite ways: `upscale-clip` centre-cropped an
off-ratio source (diffusers' `preprocess_video(resize_mode="crop")`),
`refine-clip` stretched it, and a short source was lapped from frame 0
(`loop_frames`) in one and left without a reference in the other. The idea:
a fit task that resizes into the model's size and frame count and records
what it did, and a restore task that undoes it.

The guided-enhancement-anchors extension was split out to #613.

## Verdict

**Build smaller, in two stages.** A model-bound round trip with a record,
not a general resize task (#512, declined). Evidence of demand was thin
(every run of either template was in `regression-model-specific`); it rested
on Don's priority:2 approval and on both templates' descriptions warning
about exactly this.

## What the plan found against the issue

- **The task picks no bucket or grid.** `width`, `height` and `num_frames`
  stay template variables, because `variable_constraints`, `cost_drivers`
  and the VRAM estimate only see variables. A run-time size would lose the
  32n / 8n+1 refusals and the cost quote. "Pad to the grid" means padding a
  short source up to the caller's `num_frames`.
- **Restore infers its scale** as output width / model width, and refuses
  a non-uniform one. The proposal assumed restore always went back to 1x.
- **A short source holds its last frame**, not lapped to frame 0.
- **Crop can't restore the cut edges**: it returns the kept `source_box` at
  source density (640×480 into 512×288 restores to 640×360).
- **Cut:** the `orig*(1-s) + out*s` blend (#606's sigma ladder is the
  model-native version) and "keep the tail frames the model dropped"
  (a visible seam, and nothing below the source length is dropped).

### Found in stage B's build (plan v3)

- **v2's `ref_width`/`ref_height` couldn't be wired.** `LTX2InContextPipeline`
  takes `width`/`height` at the *output* size and derives the reference as
  `width // reference_downscale_factor` (`pipeline_ltx2_ic_lora.py:1403`).
  Workflow JSON can't multiply, so nothing could pass it `2 × ref_width`.
- **An `AudioVideo` can't feed a pipeline argument or a condition's
  `frames`**: nothing in the engine unwraps it, and
  `previous_result:fit.video.frames` doesn't resolve (properties are read
  one level deep). v2's fallback, saving the fit and referencing the file,
  didn't work either: `output:` names only earlier runs.

## Decisions (Don)

- **Q1 (2026-10-05):** `letterbox` is both templates' default, flipping to
  `stretch` if stage B's runs showed an edge halo. They didn't (C-F246:
  no bars, edges match the source's field of view), so letterbox stands.
- **Q2 (2026-10-05), superseded:** upscale-clip moves to
  `ref_width`/`ref_height` as a `breaking-change`.
- **Q2′ (2026-10-06):** upscale-clip keeps `width`/`height` as the output
  size; `fit_to_model` gains `downscale`, and upscale-clip passes 2. No
  engine change, no `breaking-change`. The two templates' size variables
  mean different things (output for upscale-clip, working size for
  refine-clip) and the descriptions say which. `fit.video` becomes an
  fps-carrying float32 ndarray.
- **Q3 (2026-10-05):** adopting the pair in `restore-deblur` /
  `restore-decompression` is a separate follow-up.

## What was built

**Stage A, #631: the two tasks.**
- `fit_to_model(video, width, height, num_frames, mode="letterbox")` and
  `restore_to_source(video, fit)` in `dw/tasks/fit.py`, registered in
  `dw/tasks/task.py`; `FIT_MODES` and the argument checks in
  `dw/task_domains.py`.
- Fit returns `{video, fit}`, read as `previous_result:<step>.video` and
  `.fit`. The record (`mode`, source and model width/height/frames,
  `content_box`, and `source_box` in crop mode) is a `JsonRecord`, saved as
  one `.json`.
- Restore crops the letterbox, resizes to source × scale, trims to
  `min(source_frames, frames)`, and passes fps through.
- `docs/TASKS.md`: a row each and a round-trip section.
- **A save bug it surfaced:** `Result` wrote frame-only video through
  `export_to_video`, whose imageio `macro_block_size` of 16 grew any side not
  a multiple of 16 (360 → 368). Both calls in `dw/result.py` now pass
  `VIDEO_MACRO_BLOCK = 2`, so a saved video keeps its size. That changes
  every frame-only video save, not just this pair's.

**Stage B, #632: the templates, and the v3 task changes.**
- `fit_to_model` gains `downscale` (default 1), checked statically and at
  run time by one function, `fit_downscale_problem`, which names only the
  side or sides it doesn't divide.
- The fitted `video` is `dw.media_types.FittedVideo`, a float32 ndarray
  carrying `.fps`.
- `upscale-clip` and `refine-clip` both run fit → pipeline → restore →
  `pair_audio`, with a new `fit` variable (default `letterbox`).
  upscale-clip's fit has `downscale: 2`; refine-clip's replaces
  `loop_frames`. `intermediate/` keeps the model-frame output.
- Descriptions, summaries, `workflows/templates/ltx2/README.md`,
  `plugins/dw/skills/ltx-2.5/SKILL.md` (plugin version bumped) and
  `docs/RECIPES_24GB.md` describe the round trip and which size each
  template's variables name.

**Measured (C-F246, C-F247):** a 640×480, 50-frame, 24 fps source comes back
from either template at 1280×960, 50 frames, 24 fps, with the source's
2.083 s of audio, no bars and no lost edge content. A source already at the
working size (C-F321) gives the same output size as before.

## Bounces per stage

- **#631: one.** C-F241's crop arm restored to 640×368, not 640×360: the
  restore was right and the save grew the side (the macroblock fix above).
  Verified on the second hand-off.
- **#632: one stopped build, then one tester bounce.** The first build
  stopped before any code on v2's two unworkable parts, which led to plan
  v3. After the v3 build, C-F320's refusal named `512x288` for both sides
  instead of the one `downscale` doesn't divide. Verified on the next
  hand-off. Every architecture review passed.

No `usage:` figures were recorded on the stages, so cost is left out. The
plan's estimate was $4–6 for A and $5–8 for B.

## Deferred and left open

- **Q3:** the pair in `restore-deblur` and `restore-decompression`, which
  still crop silently. To be filed to Don as its own idea now that stage B
  showed letterbox holds under a generative model.
- **`loop_frames` stays:** `templates/ltx2/reference-sheet.json` still
  uses it, so the plan's removal follow-up doesn't apply.
- **#613 (anchors)** builds on the fitted frame. #601 (overlap windowing)
  is complementary: fit once, window, join, restore once.
- **The blend and tail-splice cuts** come back only on a field report (or,
  for the blend, if #606 is declined).
- **Empty-video refusal** is covered by unit tests only: no MCP tool makes a
  zero-frame video.
- **Retiring plan-v2 cases** C-F245, C-F248 and C-F250 is requested in
  dkackman/harnest#60 and #65; until applied they stay `pending: #632`.
