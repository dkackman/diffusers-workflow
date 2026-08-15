# Fast on 24GB

Recommended configurations per model family for a 24GB consumer GPU (RTX 3090/4090 class). Every knob here is documented individually in [ACCELERATION.md](ACCELERATION.md) and [QUANTIZATION.md](QUANTIZATION.md); this page is about the combinations that work.

The general recipe, in order of impact:

1. **Fit the transformer first.** If it fits in bf16 with room for activations, don't quantize. If it doesn't, prefer float8/int8 quantization (TorchAO, GGUF Q8) over offloading - quantization costs quality once, offloading costs speed every step.
2. **Compile the transformer** (`"compile": {"repeated_blocks": true}`). 1.3-1.5x, stacks with everything below. The REPL worker keeps compiled pipelines loaded, so the compile cost is paid once per session. Add `fullgraph: true` only when no cache is configured - cache hooks need a graph break.
3. **Cache** (`"cache": {"type": "first_block"}`). Another 1.5-2x at mild quality cost; raise `threshold` to taste.
4. **Offload only what doesn't fit.** Text encoders and VAE tolerate `offload: "model"` cheaply - they run once per generation, not once per step.
5. **Pin the attention backend** on compiled components (`"attention_backend": "flash_hub"` or `"sage_hub"` - fetched from the Hub, no local build).

## Flux dev (12B)

The bf16 transformer is ~24GB - it does not fit alongside the T5 encoder. Two good options:

| Approach | Config | Notes |
| -------- | ------ | ----- |
| **int8 TorchAO** (RTX 30-series) | transformer `quant_type: "torchao.quantization.Int8WeightOnlyConfig"` + `compile` + `cache: first_block` + `offload: "model"` | Needs compile to be fast. |
| **float8 TorchAO** (RTX 40-series+) | same, with `Float8DynamicActivationFloat8WeightConfig` | Fastest, but fp8 needs compute capability 8.9+ (Ada). |
| **GGUF Q8** | `from_single_file` Q8_0 transformer + `offload: "model"` | Simplest, best quality retention, slower than TorchAO+compile. |

**Examples:** [FluxDevFast.json](../examples/FluxDevFast.json) (the float8 recipe), [FluxGGUF.json](../examples/FluxGGUF.json), [FluxDevFirstBlockCache.json](../examples/FluxDevFirstBlockCache.json)

## Qwen-Image (20B)

Too large for bf16 on 24GB. Quantize the transformer (TorchAO int8/float8 or GGUF Q4/Q5) and model-offload the rest; add `compile` + `first_block` cache as for Flux. The Qwen2.5-VL text encoder is large - quantize it too, or group-offload it (`"text_encoder.model"` with `leaf_level`).

**Example:** [QwenImageEdit.json](../examples/QwenImageEdit.json)

## Wan 2.2

- **TI2V-5B**: fits in bf16 on 24GB. No quantization needed - just `compile` + `cache` + VAE tiling for longer clips.
- **T2V/I2V-A14B**: two 14B transformers (`transformer` + `transformer_2`). Quantize both (GGUF Q4/Q5 or TorchAO int8) and use `offload: "model"`; enable `vae.enable_tiling`.

**Examples:** [Wan22TI2V5B.json](../examples/Wan22TI2V5B.json), [Wan22T2V14B.json](../examples/Wan22T2V14B.json)

## HunyuanVideo (13B)

Same shape as Flux: quantize the transformer (GGUF Q6/Q8, or TorchAO int8/float8 per the Flux table) + `offload: "model"` + `vae.enable_tiling`. The Llama text encoder benefits from group offload. Use `first_block` cache - video steps are expensive, caching pays off more than on images.

**Examples:** [HunyuanVideoGguf.json](../examples/HunyuanVideoGguf.json), [hunyuan15.json](../examples/hunyuan15.json)

## SDXL (2.6B UNet)

Fits several times over in 24GB. Skip quantization and offloading entirely; `compile` the UNet if you generate many images per session. Use `num_images_per_prompt` batching with `vae.enable_slicing`.

**Example:** [sdxl.json](../examples/sdxl.json)

## Multi-model workflows

When a workflow chains two large models (generate → upscale, generate → interpolate), release the first pipeline instead of offloading everything:

```json
{ "name": "generate", "release_pipeline": true, "pipeline": { ... } }
```

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#releasing-a-pipeline-mid-workflow).
