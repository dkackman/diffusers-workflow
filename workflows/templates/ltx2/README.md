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

## Going long

| Example | What it introduces |
| ------- | ------------------ |
| [extend-clip.json](extend-clip.json) | Continuing a clip by conditioning on it in full, both steps sharing one loaded model |
| [chained-segments.json](chained-segments.json) | A chain re-runs the pipeline per segment on the previous last frame and stitches frames and audio back together |
