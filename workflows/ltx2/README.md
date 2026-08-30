# LTX-2.5 workflows

Text- and image-to-video with a generated soundtrack, using
[LTX-2.5](https://huggingface.co/Lightricks/LTX-Video) fitted onto a single
24GB consumer GPU. The memory configuration the examples share is explained in
[docs/RECIPES_24GB.md](../../docs/RECIPES_24GB.md).

Read them in this order and each introduces one new idea on top of the last.

## The basics

| Example | What it introduces |
| ------- | ------------------ |
| [LTX2.json](LTX2.json) | The baseline: text to video plus soundtrack on a fixed eight-step distilled schedule, quantized per component |
| [LTX2I2V.json](LTX2I2V.json) | A supplied still becomes the first frame, with VAE tiling for the longer clip |
| [LTX2Keyframes.json](LTX2Keyframes.json) | Pinning the first and last frames as conditions, each with the latent index it lands on |

## Writing the prompt with a model

| Example | What it introduces |
| ------- | ------------------ |
| [LTX2I2VEnhancePrompt.json](LTX2I2VEnhancePrompt.json) | LTX-2.5's own enhancer rewrites a one-line idea into a trained-format prompt, conditioned on the reference frame |

## Quality and scale

| Example | What it introduces |
| ------- | ------------------ |
| [LTX2TwoStage.json](LTX2TwoStage.json) | The recommended quality flow: render small, double in latent space, pair the original soundtrack back on |
| [LTX2ICLora.json](LTX2ICLora.json) | A generative 2x upscale: an in-context LoRA re-renders a clip at twice the size, inventing detail |

## Going long

| Example | What it introduces |
| ------- | ------------------ |
| [LTX2Extend.json](LTX2Extend.json) | Continuing a clip by conditioning on it in full, both steps sharing one loaded model |
| [LTX2I2VChained.json](LTX2I2VChained.json) | A chain re-runs the pipeline per segment on the previous last frame and stitches frames and audio back together |
