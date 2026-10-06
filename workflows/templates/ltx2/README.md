# LTX-2.5 workflows

Text- and image-to-video with a generated soundtrack, using
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) fitted onto a single
24GB consumer GPU. The memory configuration the examples share is explained in
[docs/RECIPES_24GB.md](../../../docs/RECIPES_24GB.md).

The prompt format is Lightricks' own: the model was trained on one-paragraph
captions of roughly 150-220 words that carry a shot type, a camera motion and a
viewpoint in prose, with the soundscape interleaved with the action. That spec ships
inside diffusers as `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and
`LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` in `diffusers.pipelines.ltx2.utils`, which is what
the [enhancer](enhance-prompt.json) runs; the stored prompts under `prompts/ltx2/` are
written to it. Settings come from the
[model card](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) and Lightricks'
`ltx-pipelines` notes. Audited against those sources on 2026-09-07
([the audit](../../../docs/proposals/audits/2026-09-07-ltx-2.5-audit.md)).

An agent driving this family from Claude Code has the `ltx-2.5` skill of the
[dw plugin](../../../plugins/dw/README.md), which chooses among these templates
and carries the caption spec.

Read them in this order and each introduces one new idea on top of the last.

## The basics

| Example | What it introduces |
| ------- | ------------------ |
| [text-to-video.json](text-to-video.json) | The baseline: text to video plus soundtrack on a fixed eight-step distilled schedule, quantized per component |
| [image-to-video.json](image-to-video.json) | A supplied still becomes the first frame, with VAE tiling for the longer clip |
| [keyframes.json](keyframes.json) | Pinning the first and last frames as conditions, each with the latent index it lands on |

## Writing the prompt with a model

| Example | What it introduces |
| ------- | ------------------ |
| [enhance-prompt.json](enhance-prompt.json) | LTX-2.5's own enhancer rewrites a one-line idea into a trained-format prompt, conditioned on the reference frame |

## Quality and scale

| Example | What it introduces |
| ------- | ------------------ |
| [two-stage.json](two-stage.json) | The distilled two-stage flow in its three moves: render at half size, double the latents, renoise and refine at full size. Lightricks' newer DFR pipeline is the follow-up |
| [generative-upscale.json](generative-upscale.json) | A generative 2x upscale: an in-context LoRA re-renders a clip at twice the size, inventing detail |
| [upscale-clip.json](upscale-clip.json) | The same 2x re-render of a clip the caller brings (`source_video`), with its soundtrack paired back on by `pair_audio`. `width`/`height` are the render size: the source is fitted to half of it by `fit_to_model` (`downscale: 2`) and restored after, so the output is exactly 2x the source at its own length - another aspect ratio is letterboxed and the bars cut back off (`fit`: `letterbox`, `stretch` or `crop`), a shorter source is held on its last frame and trimmed back. Generative, not a faithful resize; restore a compressed source first (#548, #602) |
| [refine-clip.json](refine-clip.json) | `two-stage`'s refine pass on a clip the caller brings (`source_video`): the latent upsampler encodes the source and doubles its latents, then the three stage-two sigmas refine them - the source's own latents, not a reference re-render, and no LoRA. `width`/`height` and `num_frames` are the working size and length the source is fitted to; the result is restored to exactly 2x the source at its own length, another aspect ratio letterboxed and cut back (`fit`), a shorter source held and trimmed. The source's soundtrack is paired back on; a silent source is refused before any pipeline loads (#543, #602) |
| [refine-in-place.json](refine-in-place.json) | `refine-clip` at the source's own size and length: fit to the model, refine, restore to the source's size, so nothing doubles. `strength` 0-4 (default 2) picks one of five three-sigma ladders, each with `noise_scale` equal to its first sigma: 0 stays close to the source, 4 reinterprets it (identity and framing hold; skin texture, tone and expression are re-rendered). Run time is flat across strengths, about 1.5 minutes on an RTX 3090. The soundtrack is paired back on; a silent source is refused (#639) |
| [face-repair.json](face-repair.json) | Repairs a wide shot's small, smeared face and leaves the rest of the frame alone: `crop_face_track` cuts a `crop_size` square (a multiple of 32, default 512) around the tracked face on every frame, `LTX2RefinePipeline` re-renders the crops at that size, and `paste_face_track` feathers them back in, colour-matched. How much each frame is repaired comes from the face's size - full while it is at most `gate_full` (0.03) of the frame's width, none from `gate_zero` (0.06) - so a near face, or a clip with no face, comes back as the source. `strength` 0-4 (default 2) picks one of five lem-tuned three-sigma ladders: 0 stays close to the smear, 4 is sharp but hair and clothing can drift. About 2 minutes on an RTX 3090; the soundtrack, frame rate and shots are the source's own (#599, #624) |
| [diffusion-decode.json](diffusion-decode.json) | LTX-2.5's other video decoder: a small diffusion model in place of the convolutional VAE, held against `text-to-video` frame for frame. An experiment, not a recommendation - it is silent (audio comes back as latents), and it needs a `shi-labs/natten` build for the installed torch: the FlexAttention fallback needs ~25.5GiB for the smallest canvas its kernel accepts, so on 24GB it decodes nothing at all (#153) |

## Keeping a subject

| Example | What it introduces |
| ------- | ------------------ |
| [reference-sheet.json](reference-sheet.json) | The family's identity route: a reference sheet - one composite image with a clean panel per character, prop and location - held across the clip by the Ingredients IC-LoRA. The sheet is a still, looped into a static video by a `loop_frames` step because the LoRA reads it through a 121-frame bucket; the prompt is in the trained `Reference sheet: … / Generated video: …` form. A 0.9 preview weight, trained at 768x448x121 @ 24fps (#151) |

## Restoring footage you did not generate

Both of these (and `upscale-clip` above) read a clip the workflow did not make, with the source attached
as an in-context reference for the whole denoise - so identity, framing and
background geometry are the source's and only the defect changes. A different
trade from `two-stage`, which invents a sharper version of a scene it
generated. Both are 0.9 preview weights trained at 960x544x121 @ 24fps;
generating far above that bucket weakens the effect.

| Example | What it introduces |
| ------- | ------------------ |
| [restore-deblur.json](restore-deblur.json) | Spatial defocus only - not motion blur, not noise, not low resolution. Lower `lora_scale` toward 0.8 if it over-sharpens into haloing (#152) |
| [restore-decompression.json](restore-decompression.json) | Macroblocking, chroma bleed, ringing and banding from a low bit-rate source. Not a deblur and not an upscale (#152) |
| [restore-long.json](restore-long.json) | `restore-deblur` over a source longer than one bucket: `window_video` slices it, a `for_each` restores each window, `join_windows` cross-fades them back to the source's length. `windows` is `ceil(source_frames / (num_frames - overlap))` entries, checked by `validate_workflow` (#601) |

## Going long

| Example | What it introduces |
| ------- | ------------------ |
| [extend-clip.json](extend-clip.json) | Continuing a clip by conditioning on it in full; `clip` extends an existing clip instead of generating one |
| [chained-segments.json](chained-segments.json) | A chain re-runs the pipeline per segment on the previous last frame and stitches frames and audio back together |
