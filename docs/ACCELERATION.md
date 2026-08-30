# Inference Acceleration

Speed up generation by caching intermediate computations and skipping redundant transformer steps. Two systems are available: diffusers built-in caching and TeaCache. Beyond caching, `torch.compile`, attention backend selection, layerwise casting, and device-level settings (TF32, cuDNN) also affect throughput - see below. Memory offloading trades speed for VRAM and is covered in depth in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#memory-offloading).

For ready-made configurations that combine these levers per model family, see [RECIPES_24GB.md](RECIPES_24GB.md).

## Diffusers Built-in Cache

Applied at pipeline load time via the `cache` configuration. Hooks auto-reset between runs.

### Models diffusers has not registered

`first_block`, `mag` and `layer_skip` look a model's transformer block class up in
diffusers' own registry and raise when it is absent, which is how a model that supports
`enable_cache()` ends up with no usable cache. [cache_blocks.json](../dw/cache_blocks.json)
fills those gaps in, registering the missing block metadata on demand; entries become
redundant, not wrong, once diffusers registers the same class upstream. MiniMax-H3 and
LTX-2 are listed there today.

LTX-2's block needs one thing more than the metadata diffusers defines. It returns two
streams - video and audio - and diffusers reads the second one back out of a forward
argument named literally `encoder_hidden_states`, the only two-stream shape it registers
upstream (text beside image). LTX-2's blocks take an `encoder_hidden_states` of their
own, the text conditioning, so the fixed name reads the wrong tensor and feeds the text
embeddings back as the audio stream on every skipped block. The entry names the argument
its second stream actually comes from (`encoder_hidden_states_argument_name`), which is
what makes caching correct there rather than merely quiet.

### FirstBlockCache

Simplest and broadest support. Compares first-block residuals to decide whether to skip remaining blocks.

```json
"configuration": {
    "component_type": "FluxPipeline",
    "cache": {
        "type": "first_block",
        "threshold": 0.05
    }
}
```

Higher threshold = more speedup, more quality loss. Start with `0.05` and increase to taste.

**Example:** [FluxDevFirstBlockCache.json](../workflows/flux/FluxDevFirstBlockCache.json)

### MagCache

Magnitude-based caching with error accumulation. Requires `num_inference_steps` to match the pipeline arguments, and `mag_ratios` — the per-step magnitude ratios, which are checkpoint-dependent:

```json
"cache": {
    "type": "mag",
    "mag_ratios": "flux",
    "threshold": 0.06,
    "num_inference_steps": 28,
    "max_skip_steps": 3,
    "retention_ratio": 0.2
}
```

| Property | Default | Description |
| -------- | ------- | ----------- |
| `mag_ratios` | required | Preset name or explicit per-step ratio array — see below |
| `threshold` | 0.06 | Accumulated error threshold for skipping |
| `num_inference_steps` | required | Must match pipeline arguments |
| `max_skip_steps` | 3 | Max consecutive steps to skip |
| `retention_ratio` | 0.2 | Fraction of initial steps where skipping is disabled |
| `calibrate` | false | Measure ratios for a new model instead of skipping — see below |

#### Supplying `mag_ratios`

MagCache needs to know how each denoising step's output magnitude typically behaves for *your* checkpoint, so unlike the other cache types it cannot run on defaults alone. Give it either:

- **A preset name** — `"mag_ratios": "flux"` resolves to the ratios diffusers ships for Flux. Any preset a later diffusers release adds is usable by name without a change here.
- **An explicit array** — `"mag_ratios": [1.0, 0.98, 0.96, ...]`. The array is interpolated automatically when its length differs from `num_inference_steps`, so ratios measured at one step count can be reused at another.

For a model with no preset, run once with `"calibrate": true`. Calibration skips nothing and logs the measured ratios at the end of the run; paste that array into `mag_ratios` and drop the `calibrate` flag for subsequent runs.

```json
"cache": { "type": "mag", "calibrate": true, "num_inference_steps": 28 }
```

### TaylorSeerCache

Taylor series approximation of cached outputs:

```json
"cache": {
    "type": "taylorseer",
    "cache_interval": 5,
    "max_order": 1
}
```

| Property | Default | Description |
| -------- | ------- | ----------- |
| `cache_interval` | 5 | Full computation every N steps |
| `max_order` | 1 | Taylor series order (higher = better approximation, more memory) |

### FasterCache

Experimental, video-oriented. Uses FFT frequency decomposition:

```json
"cache": {
    "type": "faster"
}
```

Best for video models like CogVideoX. No additional parameters needed for basic use.

### TextKVCache

Caches the transformer's key/value projections of the (unchanging) text
embeddings across denoising steps, recomputing only what the latents need:

```json
"cache": {
    "type": "text_kv"
}
```

No parameters.

## TeaCache

Training-free acceleration that monkey-patches the transformer's forward function. Uses polynomial-rescaled L1 distance to determine when to skip computation.

```json
"configuration": {
    "component_type": "FluxPipeline",
    "teacache": {
        "rel_l1_thresh": 0.6
    }
}
```

TeaCache requires `num_inference_steps` in the pipeline arguments — it needs to know the total step count.

### Configuration

| Property | Description |
| -------- | ----------- |
| `rel_l1_thresh` | Cache threshold. Model-specific defaults apply if omitted. |
| `coefficients` | Array of 5 polynomial coefficients. Override model defaults. |
| `variant` | Explicit model variant for multi-variant architectures. |

### Supported Models

Model coefficients and defaults are stored in [teacache_models.json](../dw/teacache_models.json). Currently implemented with a custom forward function:

- **Flux** (FluxTransformer2DModel) — thresholds: 0.25 (~1.5x), 0.4 (~1.8x), 0.6 (~2.0x), 0.8 (~2.25x)

Registry includes coefficients for Mochi, LTX-Video, CogVideoX, HunyuanVideo, Wan2.1, and Lumina2 (forward functions pending). For any model other than Flux, use the [diffusers built-in caches](#diffusers-built-in-cache) instead - `first_block` or `mag` cover the models the registry lists.

### Variants

Some models have multiple variants with different coefficients:

```json
"teacache": {
    "rel_l1_thresh": 0.2,
    "variant": "cogvideox_2b"
}
```

**Example:** [FluxDevTeaCache.json](../workflows/flux/FluxDevTeaCache.json)

## Cache vs TeaCache

| | Diffusers Cache | TeaCache |
| --- | --- | --- |
| Setup | Built into diffusers | Custom forward functions |
| Model support | Any transformer with CacheMixin | Requires per-model implementation |
| Maintenance | Maintained by HuggingFace | Maintained in this project |
| Configuration | Set once at load time | Applied per-execution via context manager |
| Approach | Various algorithms (block, magnitude, Taylor) | Polynomial-rescaled L1 distance |

They are **mutually exclusive** — use one or the other, not both.

For most cases, start with `first_block` cache. Use TeaCache when you need fine-tuned control over Flux acceleration thresholds.

## Attention Backends

Select the attention implementation diffusers uses for the duration of each pipeline call, via a context manager wrapped around `pipeline(...)`:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "attention_backend": "flash_hub"
}
```

Common values: `"flash"`, `"flash_hub"`, `"sage"`, `"sage_hub"`, `"native"`, `"flex"`. The full set is diffusers' `AttentionBackendName` enum - availability depends on what's installed (`flash-attn`, `sageattention`, etc.) and the platform. `_hub`-suffixed backends are fetched from the Hugging Face Hub kernel registry on first use rather than needing a local install.

A component can also pin its backend persistently instead, via `set_attention_backend`:

```json
"configuration": {
    "components": {
        "transformer": { "attention_backend": "flash_hub" }
    }
}
```

Prefer the pinned form for a compiled component - the per-call context manager switches implementations under the compiled graph and forces a recompile on every run.

**Example:** [Flux2Dev.json](../workflows/flux/Flux2Dev.json), [hunyuan15.json](../workflows/archive/hunyuan15.json), [Wan22TI2V5B.json](../workflows/archive/Wan22TI2V5B.json)

## Attention Slicing

```json
"configuration": {
    "component_type": "FluxPipeline",
    "enable_attention_slicing": true
}
```

Processes attention in slices to reduce memory at some cost to speed. Enabled automatically on MPS (unified memory benefits from slicing) unless `disable_attention_slicing` is set. Modular pipelines have no `enable_attention_slicing()` method - the setting is silently skipped rather than failing when the pipeline doesn't support it.

## torch.compile

Compile a component once it is fully configured - the graph captures final dtypes, adapters, quantization, and offload hooks. Configured per component under `components`:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "components": {
        "transformer": {
            "compile": {
                "repeated_blocks": true,
                "fullgraph": true
            }
        }
    }
}
```

| Property | Description |
| -------- | ----------- |
| `repeated_blocks` | Compile only the model's repeated block classes (diffusers regional compilation). Near the same speedup as full compilation with a fraction of the cold-start cost. Recommended. |
| `mode` | torch.compile mode: `"default"`, `"reduce-overhead"`, `"max-autotune"`. |
| `fullgraph` | Require a single graph with no breaks - fails fast instead of silently losing speedup. |
| `dynamic` | Compile with dynamic shapes. Set `true` when resolutions or frame counts vary between runs to avoid recompiles. |

Typical gains are 1.3-1.5x on diffusion transformers, and compilation stacks with the caches above. Notes:

- **First run pays the compile cost.** The [REPL](REPL_COMMANDS.md)'s persistent worker keeps compiled pipelines loaded between runs, so the cost is paid once per session rather than once per generation.
- **Pin the attention backend** on a compiled component (`"attention_backend"` in the same `components` entry) rather than using the pipeline-level per-call context manager, which forces recompiles.
- **Composes with offloading**: apply `group_offload` and `compile` on the same component and the offload hooks are installed first, as required. Skipped with a warning on MPS.
- **Don't combine `fullgraph` with a `cache`**: the cache hooks decide skip-or-compute per step, a data-dependent branch diffusers wraps in `torch.compiler.disable` - it needs the graph break that `fullgraph: true` forbids. Compile with the default (partial) graph mode when a cache is active.
- **TorchAO quantization needs compile to be fast** - see [QUANTIZATION.md](QUANTIZATION.md#torchao).

**Example:** [FluxDevFast.json](../workflows/flux/FluxDevFast.json), [FluxTorchAO.json](../workflows/flux/FluxTorchAO.json)

## Layerwise Casting

Store a component's weights in a narrow dtype and upcast only for compute, per component:

```json
"transformer": {
    "configuration": { "component_type": "FluxTransformer2DModel" },
    "enable_layerwise_casting": {
        "storage_dtype": "torch.float8_e4m3fn",
        "compute_dtype": "torch.bfloat16"
    },
    "from_pretrained_arguments": { ... }
}
```

Both `storage_dtype` and `compute_dtype` are required. Applied via the component's own `enable_layerwise_casting()` right after it loads, so it composes with quantization and group offloading on the same component.

## Memory Offloading

`offload` (`"model"` or `"sequential"`) and `group_offload` trade speed for VRAM by streaming weights between system memory and the accelerator instead of keeping everything resident. `"model"` moves whole submodules and costs the least speed; `"sequential"` moves individual layers and is the slowest but uses the least memory; block/leaf-level `group_offload` sits between the two and is what a modular pipeline's self-loaded components use, since they aren't reachable in time for `offload`. Full configuration syntax is in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#memory-offloading). Omit both for the fastest run, when VRAM allows it.

`"residency": "on_demand"` on a component is the cheap case of the same trade: the model rests in system memory and is moved to the device whole around each of its own calls. That is a bad deal for anything called once per step, and a good one for a VAE called twice a run - it frees the VAE's VRAM for the denoise loop at the cost of two transfers, where group offloading the same VAE would restream it once per decode tile. See [On-demand components](WORKFLOW_GUIDE.md#on-demand-components).

**Example:** [FluxDev.json](../workflows/flux/FluxDev.json) (`"offload": "model"`), [ZImage.json](../workflows/ZImage.json) (`"offload": "sequential"`), [MiniMaxH3.json](../workflows/minimax/MiniMaxH3.json) (`group_offload` per component), [MiniMaxH3Ref2VA.json](../workflows/minimax/MiniMaxH3Ref2VA.json) (`group_offload` for the transformer, `on_demand` for the VAEs)

## TF32 and cuDNN

Device-level settings, read once at startup from `~/.diffusers_helper/settings.json`:

| Setting | Default | Effect |
| ------- | ------- | ------ |
| `enable_tf32` | `true` | Sets `torch.set_float32_matmul_precision("high")`, and on CUDA also `torch.backends.cuda.matmul.allow_tf32 = True`. ~2x faster matmuls on Ampere+ GPUs (RTX 30/40 series, A100, H100) with minor precision loss. No effect outside CUDA. |
| `cudnn_benchmark` | `true` | CUDA only. Autotunes cuDNN algorithm selection - fastest for a workflow with fixed input sizes, can add overhead when sizes vary run to run. |
| `cudnn_deterministic` | `false` | CUDA only. Set `true` to trade speed for reproducible output given the same seed. |

```json
{ "enable_tf32": true, "cudnn_benchmark": true, "cudnn_deterministic": false }
```

## Environment Defaults

Set automatically at import unless already present in the environment (export your own value to override):

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Lets the CUDA allocator grow segments instead of fragmenting fixed-size ones. Multi-step workflows churn differently-shaped allocations (generate, upscale, interpolate); fragmentation is what OOMs a card that nominally has room. |
| `HF_ENABLE_PARALLEL_LOADING` | `true` | Loads sharded checkpoints in parallel - faster cold starts. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.0` | MPS only - use all available unified memory. |

For faster model downloads, optionally `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`. Not enabled automatically - it bypasses the Python HTTP stack and breaks some proxy setups.

## MPS Notes

Apple Silicon has narrower acceleration support than CUDA:

- No flash-attn, no Triton, no bitsandbytes - `attention_backend` is effectively CUDA-only; use `"native"`-family backends or leave it unset on MPS. `compile` is skipped with a warning (inductor support on MPS is immature).
- No `torch.autocast` support - autocast-related warnings from other libraries are suppressed automatically rather than surfaced.
- `enable_attention_slicing` is on by default (set `disable_attention_slicing` to turn it off).
- `float16` produces NaN values on Apple Silicon - use `float32` or `bfloat16` for `torch_dtype` instead; dw only warns, it doesn't override the dtype for you.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` defaults to `0.0` (use all unified memory) unless already set in the environment.
- Offloading has less benefit than on CUDA, since unified memory is already shared between CPU and GPU.
- `export PYTORCH_ENABLE_MPS_FALLBACK=1` to fallback to cpu for operations that MPS doesn't support
