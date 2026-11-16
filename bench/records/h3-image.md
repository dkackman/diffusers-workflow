# MiniMax-H3-Image probe (#1317, stage A #1345)

A bench probe only. Nothing here touches the catalog, the engine, MCP, REST or the
workflow syntax. It answers one question with numbers and side-by-sides: should dw ship
MiniMax-H3-Image (`MiniMaxH3ImageModularPipeline`), and is it a better identity still
than Klein for feeding MiniMax-H3 image-to-video?

## Workflows

All four are in `bench/workflows/` and pass validation.

| File | What it runs |
|---|---|
| `h3-image-t2i.json` | One H3-Image text-to-image still at 2048x2048. |
| `h3-image-edit3.json` | One H3-Image edit at 2048x2048 from three references, named `<Picture 1>`..`<Picture 3>` in the prompt. |
| `klein-edit3.json` | The Klein control: `templates/flux3/multi-reference` as shipped, with the same refs, the same intent worded as "image 1..3", and the same seed. |
| `h3-image-to-h3.json` | The H3-Image edit and the Klein edit (each with `release_pipeline: true`), then `templates/minimax/image-to-video` (960x544, int4) run on each still with one prompt. |

The references are `asset:cast/estela.jpg` and `asset:cast/aurelio.jpg`, both common, and
`asset:parhelion/kf-open.png` from the default workspace. The seed is 1317.

## H3-Image recipe

- **Transformer:** SDNQ at `weights_dtype` (int8 by default), with H3's quantizer
  arguments. `modules_to_not_convert` is read off the H3-Image transformer. It is
  block-level group offloaded.
- **Text encoder:** loaded from `MiniMaxAI/MiniMax-H3` at `0919fb68`, through the
  per-component `load_components` override, so it shares H3's cache. It is quantized with
  SDNQ, `lm_head` is removed, and its language model is truncated to 51 layers, because
  the pipeline reads `hidden_states[50]`. It runs under leaf-level offload.
- **VAE:** bf16, loaded on demand. It is H3-Image's own `AutoencoderKLMiniMaxH3Image`, so
  H3's int8 VAE recipe isn't carried over.
- **Sampler:** fixed at 6 steps and guidance-distilled, so the workflows pass no
  `num_inference_steps`, `guidance_scale` or `negative_prompt`.
- **Canvas:** width and height are multiples of 32. `min_generation_area` defaults to
  4194304; at 1024 the pipeline generates at the requested size.

## Download gate (first action, passed)

`validate_workflow` on `h3-image-t2i.json` (lem, develop @ 8e1b27f6) lists exactly one
download in `downloads_required`:

- the repo is `MiniMaxAI/MiniMax-H3-Image`;
- the size is 63.93 GiB (about 68.6 GB);
- the components are processor, scheduler, tokenizer, transformer_diffusers and
  vae_diffusers.

No text encoder is fetched from H3-Image, and the run does not download the repo's full
~204 GB. The override is honoured.

## Method

- **Timing:** time each arm cold (first load) and then warm (a rerun on the same worker).
  Record the peak reserved VRAM from the job and the bytes downloaded from
  `list_downloads` or the job events.
- **1024 run:** run `h3-image-t2i.json` with `width=1024 height=1024
  min_generation_area=1024`.
- **Identity:** score every edit still and every clip with
  `bench/workflows/t26-face-check.json` (SFace) against each portrait.
- **Comparison:** post the side-by-sides on #1317, using `compare_outputs` plus a visual
  look.
- **Failures:** an OOM or a load failure is a result. Record it with the error, not a
  blank.

## Results

| Arm | Cold min | Warm min | Peak VRAM GiB | Bytes downloaded | SFace estela / aurelio | Notes |
|---|---|---|---|---|---|---|
| H3-Image t2i 2048 | | | | | n/a | |
| H3-Image t2i 1024 (min_generation_area=1024) | | | | | n/a | |
| H3-Image edit3 2048 | | | | | | |
| Klein edit3 | | | | | | |
| H3-Image still → H3 i2v 544p | | | | | | |
| Klein still → H3 i2v 544p | | | | | | |
