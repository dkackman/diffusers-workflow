# Fast on 24GB

Recommended configurations per model family for a 24GB consumer GPU (RTX 3090/4090 class). Every knob here is documented individually in [ACCELERATION.md](ACCELERATION.md) and [QUANTIZATION.md](QUANTIZATION.md); this page is about the combinations that work.

The general recipe, in order of impact:

1. **Fit the transformer first.** If it fits in bf16 with room for activations, don't quantize. If it doesn't, prefer float8/int8 quantization (TorchAO, GGUF Q8) over offloading - quantization costs quality once, offloading costs speed every step.
2. **Compile the transformer** (`"compile": {"repeated_blocks": true}`). 1.3-1.5x, stacks with everything below. The REPL worker keeps compiled pipelines loaded, so the compile cost is paid once per session. Add `fullgraph: true` only when no cache is configured - cache hooks need a graph break.
3. **Cache** (`"cache": {"type": "first_block"}`). Another 1.5-2x at mild quality cost; raise `threshold` to taste.
4. **Offload only what doesn't fit.** Text encoders and VAE tolerate `offload: "model"` cheaply - they run once per generation, not once per step. In a modular pipeline the same components take `"residency": "on_demand"`, which frees their VRAM for the denoise loop at the cost of one pair of transfers per call.
5. **Pin the attention backend** on compiled components (`"attention_backend": "flash_hub"` or `"sage_hub"` - fetched from the Hub, no local build).

## Flux dev (12B)

The bf16 transformer is ~24GB - it does not fit alongside the T5 encoder. Two good options:

| Approach | Config | Notes |
| -------- | ------ | ----- |
| **int8 TorchAO** (RTX 30-series) | transformer `quant_type: "torchao.quantization.Int8WeightOnlyConfig"` + `compile` + `cache: first_block` + `offload: "model"` | Needs compile to be fast. |
| **float8 TorchAO** (RTX 40-series+) | same, with `Float8DynamicActivationFloat8WeightConfig` | Fastest, but fp8 needs compute capability 8.9+ (Ada). |
| **GGUF Q8** | `from_single_file` Q8_0 transformer + `offload: "model"` | Simplest, best quality retention, slower than TorchAO+compile. |

**Examples:** [FluxDevFast.json](../examples/flux/FluxDevFast.json) (the float8 recipe), [FluxGGUF.json](../examples/flux/FluxGGUF.json), [FluxDevFirstBlockCache.json](../examples/flux/FluxDevFirstBlockCache.json)

## Qwen-Image (20B)

Too large for bf16 on 24GB. Quantize the transformer (TorchAO int8/float8 or GGUF Q4/Q5) and model-offload the rest; add `compile` + `first_block` cache as for Flux. The Qwen2.5-VL text encoder is large - quantize it too, or group-offload it (`"text_encoder.model"` with `leaf_level`).

**Example:** [QwenImageEdit.json](../examples/archive/QwenImageEdit.json)

## Wan 2.2

- **TI2V-5B**: fits in bf16 on 24GB. No quantization needed - just `compile` + `cache` + VAE tiling for longer clips.
- **T2V/I2V-A14B**: two 14B transformers (`transformer` + `transformer_2`). Quantize both (GGUF Q4/Q5 or TorchAO int8) and use `offload: "model"`; enable `vae.enable_tiling`.

**Examples:** [Wan22TI2V5B.json](../examples/archive/Wan22TI2V5B.json), [Wan22T2V14B.json](../examples/archive/Wan22T2V14B.json)

## HunyuanVideo (13B)

Same shape as Flux: quantize the transformer (GGUF Q6/Q8, or TorchAO int8/float8 per the Flux table) + `offload: "model"` + `vae.enable_tiling`. The Llama text encoder benefits from group offload. Use `first_block` cache - video steps are expensive, caching pays off more than on images.

**Examples:** [HunyuanVideoGguf.json](../examples/archive/HunyuanVideoGguf.json), [hunyuan15.json](../examples/archive/hunyuan15.json)

## MiniMax-H3

A modular pipeline, so everything is per component in `components` rather than a
pipeline-level `offload`. The working 24GB configuration at 960x544:

| Component | Config |
| --------- | ------ |
| `transformer` / `transformer_ref` | SDNQ int4 (`quantization_device: "cuda"`, `return_device: "cpu"`), `group_offload` `block_level` with `num_blocks_per_group: 1-2` and `use_stream: true` |
| `text_encoder.model` | SDNQ int4, `group_offload` `leaf_level` |
| `vae`, `audio_vae` | SDNQ int8, `device: "cuda"`, `residency: "on_demand"` |
| pipeline | `cache: first_block` (`threshold: 0.1`), `vae.enable_tiling` |

The VAEs are the piece worth calling out. They hold roughly 3GiB, are used only to
encode references and decode the result, and group offloading them is worse than useless
because tiled decode restreams the model once per tile. Every H3 workflow adds
`"residency": "on_demand"` to both. The reference workflows, which carry the most
conditioning, go from 23.2GiB peak reserved with 40 allocator retries to 18.9GiB with
none; the frame-conditioned ones from 22.7GiB with 22 retries to 18.0GiB with none. Both
for about 1% in wall time.
Spend the headroom on length: carrying a frame between chained segments adds a reference
and ~1.9GiB, which is what made the chained variants OOM on their second segment before.

**Examples:** [MiniMaxH3Ref2VA.json](../examples/MiniMaxH3Ref2VA.json), [MiniMaxH3Ref2VAChained.json](../examples/MiniMaxH3Ref2VAChained.json), [MiniMaxH3I2V.json](../examples/MiniMaxH3I2V.json)

## LTX-2.5 (22B, video + audio)

A standard pipeline, but placed per component rather than with a pipeline-level
`offload` - the transformer is the only thing that wants to be resident, and the text
encoder is nearly as large as it is. The working 24GB configuration at 960x544:

| Component | Config |
| --------- | ------ |
| `transformer` | SDNQ `uint4` (`quantization_device: "cuda"`, `return_device: "cuda"`, `use_quantized_matmul: true`), resident |
| `text_encoder` (Gemma 4, 23GB) | SDNQ `int8`, `return_device: "cpu"`, `group_offload` `leaf_level` with `use_stream: true` |
| `connectors` (12GB) | `group_offload` `leaf_level` with `use_stream: true` |
| `vae`, `audio_vae`, `vocoder`, `duration_head` | `device: "cuda"` - small, and used once per generation |
| pipeline | `vae.enable_tiling` for anything above the base resolution |

Two things about the checkpoint are worth knowing before tuning anything:

- **`transformer` is the distilled model.** It runs a fixed 8-step schedule at
  `guidance_scale: 1.0`, with STG and modality guidance off, and the `sigmas` every
  example passes are its trained schedule - not a knob. They are referenced from
  diffusers (`constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES`) rather
  than copied, so the schedule stays whatever the library says it is. `num_inference_steps`,
  `guidance_scale`, `stg_scale` and the rest only mean anything against
  `subfolder: "transformer_full"`, the dev model, which is not a 24GB configuration:
  it is the same ~38GB in bf16, and the guidance those knobs turn on costs three
  transformer passes per step against CFG-doubled batches. Nothing here ships it.
- **The checkpoint ships a diffusion decoder that `LTX2Pipeline` ignores**, and on
  24GB you are not missing much. It is listed in `model_index.json` but is not a
  constructor argument, so diffusers logs "not expected ... will be ignored" and
  decodes with the convolutional VAE. Reaching it means `LTX2VideoDiffusionDecodePipeline`
  on a step run with `output_type: "{latent}"`, and two things get in the way. Its
  first three stages run on the full volume by design - only stage 4 and the diffusion
  blocks tile - and the attention mask they build is quadratic in the output grid:
  70GiB at 1536x896x121, which no tile size reduces. Base resolution fits comfortably.
  And a step that returns latents returns *audio* latents too, which nothing outside a
  pipeline call can vocode, so that path is silent. Nothing here ships it.

Spend headroom on the two-stage flow rather than on base resolution: render at 768x448,
upsample the latents 2x, and the result is sharper than a single pass at 1536x896 and
fits where that would not. The base step shares its `vae` into the upsampler and sets
`release_pipeline`, which frees the 11GB transformer before the 2x decode runs.

**Examples:** [LTX2.json](../examples/LTX2.json) (t2v),
[LTX2TwoStage.json](../examples/LTX2TwoStage.json) (base -> latent upsample -> mux),
[LTX2Keyframes.json](../examples/LTX2Keyframes.json) (first and
last frame), [LTX2Extend.json](../examples/LTX2Extend.json) (continue a clip),
[LTX2ICLora.json](../examples/LTX2ICLora.json) (generative 2x upscale via IC-LoRA),
[LTX2I2VEnhancePrompt.json](../examples/LTX2I2VEnhancePrompt.json) (native prompt
enhancer and duration head)

## SDXL (2.6B UNet)

Fits several times over in 24GB. Skip quantization and offloading entirely; `compile` the UNet if you generate many images per session. Use `num_images_per_prompt` batching with `vae.enable_slicing`.

**Example:** [sdxl.json](../examples/archive/sdxl.json)

## Multi-model workflows

When a workflow chains two large models (generate → upscale, generate → interpolate), release the first pipeline instead of offloading everything:

```json
{ "name": "generate", "release_pipeline": true, "pipeline": { ... } }
```

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#releasing-a-pipeline-mid-workflow).
