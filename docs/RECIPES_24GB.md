# Fast on 24GB

Recommended configurations per model family for a 24GB consumer GPU (RTX 3090/4090 class). Every knob here is documented individually in [ACCELERATION.md](ACCELERATION.md) and [QUANTIZATION.md](QUANTIZATION.md); this page is about the combinations that work.

The general recipe, in order of impact:

1. **Fit the transformer first.** If it fits in bf16 with room for activations, don't quantize. If it doesn't, prefer float8/int8 quantization (TorchAO, GGUF Q8) over offloading - quantization costs quality once, offloading costs speed every step.
2. **Compile the transformer** (`"compile": {"repeated_blocks": true}`). 1.3-1.5x, stacks with everything below. The server's persistent worker keeps compiled pipelines loaded, so the compile cost is paid once per session. Add `fullgraph: true` only when no cache is configured - cache hooks need a graph break.
3. **Cache** (`"cache": {"type": "first_block"}`). Another 1.5-2x at mild quality cost; raise `threshold` to taste.
4. **Offload only what doesn't fit.** Text encoders and VAE tolerate `offload: "model"` cheaply - they run once per generation, not once per step. In a modular pipeline the same components take `"residency": "on_demand"`, which frees their VRAM for the denoise loop at the cost of one pair of transfers per call.
5. **Pin the attention backend** on compiled components (`"attention_backend": "flash_hub"` or `"sage_hub"` - fetched from the Hub, no local build).

## Flux dev (12B)

The bf16 transformer is ~22 GB, so on 24 GB it cannot sit beside its own activations
with any margin, and it cannot sit beside the T5 encoder or the VAE at all. Three
placements, measured on an RTX 3090 (2026-10-10, cold load included):

| Placement | 1024x1024, 25 steps, 1 image, dev + LoRA | 768x768, 25 steps, 4 images, Krea | 1232x1632, 50 steps, Fill |
| --------- | ----------------------------------------- | --------------------------------- | ------------------------- |
| `offload: "sequential"` | 273 s (10.8 s/step) | not measured - same per-step cost | not measured |
| `offload: "model"` | 61 s (1.7 s/step) at **23.3 GB reserved, 491 MB free** | **OOM** at 23.45 GiB | not attempted |
| **transformer `group_offload`, block-level, streamed, pinned host copies** | 78 s (1.5 s/step), 17.7 GB peak reserved, 6 GB free | 131 s (3.2 s/step) | 152 s denoise (3.0 s/step), 1 s decode |

Model offload is the fastest placement that loads, and the one #580 (9c708a12) retired:
at 1024x1024 it runs with a few hundred MB to spare and fails on anything larger -
a batch, a second encoder resident for prompt weighting, a cache's state. Sequential
offload is safe everywhere and 6x slower per step. The streamed group offload is both:
the transformer's blocks are copied in one at a time from pinned host memory under a
CUDA stream, so each step costs the same as the resident model's, and the card holds
one block plus the activations. It is what every unquantized FLUX template now uses:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "components": {
        "transformer": {
            "group_offload": {
                "offload_type": "block_level",
                "num_blocks_per_group": 1,
                "use_stream": true,
                "record_stream": true
            }
        },
        "text_encoder": { "device": "cuda" },
        "text_encoder_2": { "device": "cuda" },
        "vae": { "device": "cuda" }
    }
}
```

Two things the shape depends on. The non-offloaded components are placed explicitly:
with no pipeline-level `offload`, nothing else moves them, and a T5 or VAE left on the
CPU costs a minute per prompt encode and per decode (measured: 70 s before the first
step, 185 s decoding). And the host copies are pinned - diffusers' default,
`low_cpu_mem_usage: false` - which holds ~22 GB of unswappable host memory per loaded
FLUX pipeline and takes ~35 s once per load; with `low_cpu_mem_usage: true` the copies
come from pageable memory at 8 s/step, barely ahead of sequential. A host with less
than 32 GB free should take `offload: "sequential"` instead.

Compile and caching stack on top as before: `compile: {repeated_blocks: true}` measured
52 s vs 55 s per image on the old model-offload placement; `cache: first_block` is
[step-caching.json](../workflows/templates/step-caching.json). TorchAO int8 is over 60 s
per *denoising step* on Ampere - do not use it there; float8 needs compute capability
8.9+ (Ada) and is unmeasured; GGUF Q8 ([flux-gguf.json](../workflows/models/flux-gguf.json))
loads with the least VRAM and is unmeasured for speed.

**Examples:** [lora.json](../workflows/templates/lora.json), [flux-dev-compile.json](../workflows/models/flux-dev-compile.json), [flux-gguf.json](../workflows/models/flux-gguf.json), [step-caching.json](../workflows/templates/step-caching.json)

## Z-Image Turbo (6B)

Fits a 24 GB card in bf16 with model offload, and that is the configuration to use.
Measured on an RTX 3090 (960x544, 9 steps, 4 images per prompt, cold load included,
2026-10-10):

| Offload | Load | Denoise | Decode | Total | Peak VRAM reserved |
| ------- | ---- | ------- | ------ | ----- | ------------------ |
| `sequential` | 18 s | 84 s | 2 s | 107 s | 1.1 GB |
| **`model`** | 4 s | 25 s | 9 s | 41 s | 13.0 GB |
| none (resident) | 11 s | 17 s | 2 s | 33 s | 23.7 GB, 95 MB free at decode |

Sequential offload streams the transformer's layers every step and costs 3.4x on the
denoise for a model that does not need the room. Resident is faster still but decodes
four images with nothing to spare, so [z-image.json](../workflows/models/z-image.json)
and the Z-Image portrait steps of the MiniMax templates use `"offload": "model"`. The
decode's 9 s under model offload is the VAE swapping in; `vae.enable_slicing` would
trade it for per-image decodes.

## Qwen-Image (20B)

Too large for bf16 on 24GB. Quantize the transformer (TorchAO int8/float8 or GGUF Q4/Q5) and model-offload the rest; add `compile` + `first_block` cache as for Flux. The Qwen2.5-VL text encoder is large - quantize it too, or group-offload it (`"text_encoder.model"` with `leaf_level`).

The catalog no longer ships a Qwen-Image workflow - author one from [text-to-image.json](../workflows/templates/text-to-image.json) with the configuration above.

## Wan 2.2

- **TI2V-5B**: fits in bf16 on 24GB. No quantization needed - just `compile` + `cache` + VAE tiling for longer clips.
- **T2V/I2V-A14B**: two 14B transformers (`transformer` + `transformer_2`). Quantize both (GGUF Q4/Q5 or TorchAO int8) and use `offload: "model"`; enable `vae.enable_tiling`.

The catalog no longer ships Wan workflows - author one from [minimax/image-to-video.json](../workflows/templates/minimax/image-to-video.json) with the configuration above.

## HunyuanVideo (13B)

Same shape as Flux: quantize the transformer (GGUF Q6/Q8, or TorchAO int8/float8 per the Flux table) + `offload: "model"` + `vae.enable_tiling`. The Llama text encoder benefits from group offload. Use `first_block` cache - video steps are expensive, caching pays off more than on images.

The catalog no longer ships a HunyuanVideo workflow - author one from [ltx2/text-to-video.json](../workflows/templates/ltx2/text-to-video.json) with the configuration above.

## MiniMax-H3

A modular pipeline, so everything is per component in `components` rather than a
pipeline-level `offload`. The working 24GB configuration at 960x544:

| Component | Config |
| --------- | ------ |
| `transformer` / `transformer_ref` | SDNQ int4 (`quantization_device: "cuda"`, `return_device: "cpu"`), `group_offload` `block_level` with `num_blocks_per_group: 1-2` and `use_stream: true` |
| `text_encoder` | `remove_modules: ["lm_head"]` - the encoder path never calls it |
| `text_encoder.model` | SDNQ int4, `truncate_layers: {"language_model.layers": 51}`, `group_offload` `leaf_level` |
| `vae`, `audio_vae` | SDNQ int8, `device: "cuda"`, `residency: "on_demand"` |
| pipeline | `cache: first_block` (`threshold: 0.1`) on the 20-step ref2va workflows only - measured on the 9-step turbo schedule it never skips (consecutive distilled steps differ too much for the threshold), so the turbo examples omit it rather than hold cache state for nothing |

The VAEs are the piece worth calling out. They hold roughly 3GiB, are used only to
encode references and decode the result, and group offloading them is worse than useless
because tiled decode restreams the model once per tile. Every H3 workflow adds
`"residency": "on_demand"` to both. The reference workflows, which carry the most
conditioning, go from 23.2GiB peak reserved with 40 allocator retries to 18.9GiB with
none; the frame-conditioned ones from 22.7GiB with 22 retries to 18.0GiB with none. Both
for about 1% in wall time.
Spend the headroom on length: carrying a frame between chained segments adds a reference
and ~1.9GiB, which is what made the chained variants OOM on their second segment before.

The text encoder pruning exists because H3 conditions on `hidden_states[50]` of its
64-layer Qwen3-VL: layers 51-63 and the LM head run (and stream) for nothing on every
encode. Keeping 51 layers is bit-identical - index 50 of the tuple is recorded before
layer 50 runs; keeping only 50 would hand back the final-norm output, a different
tensor - and it returns a few GiB of system RAM on a host that needs every one of them
(a full t2va run peaks around 63GiB RSS on a 64GiB box). Note H3's VAE constructs with
tiling already enabled, so a pipeline-level `vae.enable_tiling` adds nothing here.

Two levers measured and *rejected* on a 3090 (A/B, t2va 960x544x124f, 2026-08):
`low_cpu_mem_usage: false` on the transformer's group offload (pin host copies once
instead of re-pinning per onload) left step time unchanged at ~15.4s - at int4 the
step is compute-bound and the transfers already hide under it - while the ~27GiB of
unswappable pinned memory pushed the host into an OOM kill. `use_stream` on the text
encoder's leaf offload was backed out with it for the same host-memory reason. On a
faster GPU (or a host with more RAM) both are worth re-testing; the step budget there
may actually expose the transfer time.

Length costs VRAM but the configuration holds to the model's full range: a single
345-frame take (14.4s, the `17n+5` maximum) peaks at 23.6GiB reserved at 960x544 -
inside 24GB with nothing to spare - and denoises in ~13 minutes on a 3090 with the
9-step turbo schedule (~85-100s a step once warm, against ~15s at 124 frames).

Host RAM is the tighter budget than VRAM on a 64GiB box. Loading H3 peaks around
59GiB RSS and a running ref2va shot sits at 45-53GiB, so a workflow that ran another
model first (Z-Image drawing a subject, Music3 writing a song) must free it with
`release_pipeline` before H3 loads - with it, the multi-model digital-short
workflows below fit; without it, the load is an OOM kill, not a slowdown.

Ref2VA reaches 1344x768 too - the canvas isn't tied to the FL2VA 768p checkpoint, only
the shift/alpha pairing is (above). The trade-off is frame count, not a config change:
VAE decode memory scales with `width * height * num_frames`, so the same 24GB budget
that holds 345 frames at 960x544 only holds 209 at 1344x768 with 2 references, and each
reference costs about 1 GB more (175 at 3) - 175 frames with 2 references measured at
~31 minutes on a 3090. Give up length, or a reference, before reaching for a smaller canvas;
the model's `17n+5` ceiling (345) is reachable at 960x544, not at 1344x768.

**Examples:** [reference-to-video.json](../workflows/templates/minimax/reference-to-video.json), [chain-matched-to-audio.json](../workflows/templates/minimax/chain-matched-to-audio.json), [image-to-video.json](../workflows/templates/minimax/image-to-video.json), [dialogue-short.json](../workflows/templates/minimax/dialogue-short.json) (five ref2va shots + two Z-Image portraits in ~35 minutes end to end)

## MiniMax-Music3

Music3 runs at about 22GiB in bfloat16 under the templates' `components_manager`
auto CPU offload, which keeps only the running component resident; no quantization
is needed on a 24GB card. The language model is the part worth offloading harder:
a leaf-level `group_offload` of `language_model` brings it to about 8GiB (the model
card's low-VRAM recipe). Two things the examples carry: `release_pipeline` on the
music step in any workflow that loads H3 afterwards, since host RAM is the binding
constraint (see Multi-model workflows below), and the run's time follows the length
the model actually sings, not `audio_duration`, which is a ceiling of at most 9000
frames at 25 frames per second (360 seconds). Output is 44.1 kHz stereo.

**Examples:** [music.json](../workflows/templates/minimax/music.json), [music-video.json](../workflows/templates/minimax/music-video.json)

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
  neighborhood attention has two processors, and the one you get by default is the
  portable FlexAttention fallback: it densifies a `seq_len x seq_len` block mask and then
  runs uncompiled `flex_attention`, which falls to the eager reference path. Both
  allocations are quadratic in the output grid, and neither is reduced by tiling (stages
  1-3 always run on the full volume) or by shrinking the clip (the stage-4 grid is near
  output resolution either way). Measured on an otherwise empty 3090 (#153): 10.05GiB
  inside stages 1-3 at 960x544x121, 69.77GiB for one stage-4 attention at 512x288x25
  (17.44GiB just to densify that stage's mask, whichever it reaches first), and ~25.5GiB
  at 224x224x25, which is the smallest canvas its 7x7 kernel accepts at all. Nothing
  fits - not base resolution, not the smallest clip the decoder will take. The path that does is NATTEN's `na3d` kernel, named per component as
  `"attn_processor_type": "diffusers.models.autoencoders.ltx2_diffusion_decoder.LTX2VideoVaeNeighborhoodNattenProcessor"`,
  which builds no mask at all - but it is fetched from the Hub by the `kernels` package
  and needs a `shi-labs/natten` build matching the installed torch, which as of
  torch 2.14 does not exist.
  And a step that returns latents returns *audio* latents too, which nothing outside a
  pipeline call can vocode - the two-stage template below feeds them back into one,
  which is the only way they become sound. Nothing here ships the diffusion decoder.

Spend headroom on the two-stage flow rather than on base resolution: render at 768x448
on the eight distilled sigmas, double the video latents with the latent upsampler, then
renoise them and run the three stage-two sigmas at 1536x896 through the same pipeline.
That refine pass is what puts the detail back - the upsampler alone gives a soft 2x -
and it is the flow the model card, Lightricks' pipeline notes and the diffusers docs all
describe. The base pass keeps its pipeline loaded so the refine pass is served from the
cache; the refine pass releases it. Since 2026-08 Lightricks route production quality
through their DFR pipeline instead, which diffusers ships and nothing here uses yet.
Measured on an RTX 3090 the refined clip is sharper than the 2x upsample alone at the
same seed: fur, branches and snow texture resolve where the upsample-only frame is a
soft blur. About eight warm minutes, three and a half of them writing the full-size clip.
The same refine on a clip dw did not make is
[refine-clip.json](../workflows/templates/ltx2/refine-clip.json): the upsampler encodes
the source itself, fitted to the working size first and restored to exactly twice the
source's size and length after, and the source's soundtrack is paired back on.

**Examples:** [text-to-video.json](../workflows/templates/ltx2/text-to-video.json) (t2v),
[two-stage.json](../workflows/templates/ltx2/two-stage.json) (base -> latent upsample -> refine),
[keyframes.json](../workflows/templates/ltx2/keyframes.json) (first and
last frame), [extend-clip.json](../workflows/templates/ltx2/extend-clip.json) (continue a clip),
[generative-upscale.json](../workflows/templates/ltx2/generative-upscale.json) (generative 2x upscale via IC-LoRA),
[enhance-prompt.json](../workflows/templates/ltx2/enhance-prompt.json) (native prompt
enhancer and duration head)

## SDXL (2.6B UNet)

Fits several times over in 24GB. Skip quantization and offloading entirely; `compile` the UNet if you generate many images per session. Use `num_images_per_prompt` batching with `vae.enable_slicing`.

**Example:** [base-and-refiner.json](../workflows/templates/base-and-refiner.json)

## Multi-model workflows

When a workflow chains two large models (generate → upscale, generate → interpolate), release the first pipeline instead of offloading everything:

```json
{ "name": "generate", "release_pipeline": true, "pipeline": { ... } }
```

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#releasing-a-pipeline-mid-workflow).
