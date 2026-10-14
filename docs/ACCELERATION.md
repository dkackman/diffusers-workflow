# Inference Acceleration

Speed up generation by caching intermediate computations and skipping redundant transformer steps. Diffusers' built-in caching (`first_block`, `mag` and the others below) provides it. Beyond caching, `torch.compile`, attention backend selection, layerwise casting, and device-level settings (TF32, cuDNN) also affect throughput - see below. Memory offloading trades speed for VRAM and is covered in depth in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#memory-offloading).

For ready-made configurations that combine these levers per model family, see [RECIPES_24GB.md](RECIPES_24GB.md).

## Diffusers Built-in Cache

Applied at pipeline load time via the `cache` configuration. Hooks auto-reset between runs.

`validate_workflow` warns (it never refuses) about two cache settings that cost a render to find out:

- **`cache_short_schedule`: a threshold cache on a short schedule.** `first_block` and `mag` decide each step's skip on a residual threshold, and the consecutive steps of a short schedule differ too much for it: under 20 `num_inference_steps` such a cache will likely skip nothing and still perturbs the result. On MiniMax-H3 a `first_block` cache at `0.05` skipped nothing at 12 steps and took 22% off at 30 ([FirstBlockCache](#firstblockcache)). Drop the cache or run 20 steps or more. A turbo schedule (9 steps or fewer) never benefits. `taylorseer` and `faster` skip on a fixed interval and `text_kv` skips no step, so the rule does not apply to them.
- **`taylorseer_state_unestimated`: TaylorSeer's state is not in the VRAM estimate.** Said whenever `taylorseer` is set - see [TaylorSeerCache](#taylorseercache).

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

Higher threshold = more speedup, more quality loss. Start with `0.05` and increase to taste. On MiniMax-H3 (RTX 3090, 1344x768, 4 references) `0.05` took 22% off a 30-step base-model job and nothing off a 12-step or turbo one, which it never skips; `torch.compile` there cost 289 s on the first pass for ~1% per pass, `text_kv` did nothing on Ref2VA, and `taylorseer` ran out of memory. Every MiniMax-H3 template takes a `cache` argument, `null` (off) by default and wired to the H3 step only: pass `{"type": "first_block", "threshold": 0.05}` rather than editing the JSON, and only at ~20 or more base steps, never with the turbo LoRAs.

**Example:** [step-caching.json](../workflows/templates/step-caching.json)

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

For a model with no preset, run once with `"calibrate": true`. Calibration skips nothing. At the end of the run the measured ratios appear as a job warning and a `log` event (`get_job`'s `warnings`, `get_job_events`, `export_job`): `MagCache calibration (N steps): [ ... ] — paste into mag_ratios`. While calibrating, the run's own `num_inference_steps` is the step count, so the cache's can be left at its default (a mismatch is reported as a warning). The cache attaches to a pipeline's `transformer`; a UNet pipeline such as SD 1.5 has none, and gets a warning that the cache was not applied. Paste that array into `mag_ratios` and drop the `calibrate` flag for subsequent runs. A pipeline that uses classifier-free guidance (an SD-style template, not a guidance-distilled one such as Flux) keeps a separate state for its `cond` and `uncond` passes, so it emits **two** lists, each labeled `[cond]` or `[uncond]`; use the `cond` one, since `mag_ratios` takes one array.

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
| `max_order` | 1 | Taylor series order (higher = better approximation, and more state kept on the GPU) |

TaylorSeer keeps per-block state on the GPU that no `vram_estimate` charges, so a job that fits without the cache can run out of memory with it. On MiniMax-H3 at 1344x768, 30 steps and `max_order` 1, a run that `validate_workflow` had passed ran out of memory 2 min in on a 24 GB card. Validate says so as `taylorseer_state_unestimated` whenever `taylorseer` is set, with the job's estimate and the card's capacity when both are known; the state itself is on top of that figure, and is not sized.

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

## Attention Backends

Select the attention implementation diffusers uses for the duration of each pipeline call, via a context manager wrapped around `pipeline(...)`:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "attention_backend": "flash_hub"
}
```

Common values: `"flash"`, `"flash_hub"`, `"sage"`, `"sage_hub"`, `"native"`, `"flex"`. The full set is diffusers' `AttentionBackendName` enum - availability depends on what's installed (`flash-attn`, `sageattention`, etc.) and the platform. `_hub`-suffixed backends are fetched from the Hugging Face Hub kernel registry on first use, which needs the `kernels` package installed (`pip install kernels`) - it is not a dw dependency, and no bundled workflow sets a backend, so each runs on a plain install. diffusers also refuses a Hub kernel whose publisher it does not list as trusted - `sage_hub` among them, since SageAttention 2 comes from `SageAttention/sage-attention` - until `DIFFUSERS_TRUST_REMOTE_KERNELS=true`, which allows remote code from any such publisher. On an RTX 3090 `sage_hub` measured about 30% faster per MiniMax-H3 step (13.8 s against 19.2 s at 960x544x124, 54.7 s against 81.2 s at 345 frames); the local `sage` backend needs `sageattention>=2.1.1`, which PyPI does not carry.

Every MiniMax-H3 template in [templates/minimax/](../workflows/templates/minimax/)
exposes this as a variable: `attention_backend`, pinned on the H3 step's transformer
(`components.transformer`, or `components.transformer_ref` for a ref2va workflow) and
`null` by default. That covers the shots, chains and episodes - `dialogue-short`,
`shots-batch`, `music-video` and `storyboard` among them - and only their H3 steps: a
Z-Image still or Music3 song beside an H3 shot keeps the process default. Pass
`attention_backend=sage_hub` to a box with the `kernels` package and
`DIFFUSERS_TRUST_REMOTE_KERNELS=true` in its `.env`; on MPS leave it null, since a
CUDA-only kernel fails there at load (validation warns, below). LTX-2.5 cannot use it: its transformer
passes an attention mask, and sage attention refuses one ("`attn_mask` is not supported
for sage attention", measured 2026-10-10), so the LTX templates expose no backend.

A component can also pin its backend persistently instead, via `set_attention_backend`:

```json
"configuration": {
    "components": {
        "transformer": { "attention_backend": "flash_hub" }
    }
}
```

Prefer the pinned form for a compiled component - the per-call context manager switches implementations under the compiled graph and forces a recompile on every run.

The flash, flash-3, sage and xformers backends need CUDA, `_hub` variants included. On a server whose device is not CUDA (`mps`, `cpu`), `validate_workflow` warns `attention_backend_unavailable` for one set at pipeline or component level, naming the backend and the device: the job would fail at load. It never refuses, and a `native`-family backend never warns. A backend diffusers adds later is not on the list until dw's is updated, and draws no warning.

Either form stays on what it names. diffusers' `set_attention_backend` also makes its backend the process-wide default, which the persistent worker would keep for every later model; dw restores the default after pinning, so an H3 `sage_hub` pin cannot reach a Music3 or LTX job that follows it. The server-wide default is `DIFFUSERS_ATTN_BACKEND` in `.env` - leave it `native`, since it reaches every model and sage cannot serve all of them.

## Attention Slicing

```json
"configuration": {
    "component_type": "FluxPipeline",
    "enable_attention_slicing": true
}
```

Processes attention in slices to reduce memory at some cost to speed. Automatic on MPS unless `disable_attention_slicing` is set, opt-in elsewhere. Only UNet/ControlNet models take it. On MPS, whether it helps depends on the model's attention head dim: MPS SDPA is slow at 40, 48 and 160 (slicing made SD 1.5 ~20% faster end to end) and fast at 64-128 (slicing made SDXL-shaped attention 2.4x slower - set `disable_attention_slicing` for SDXL-family models). Modular pipelines have no `enable_attention_slicing()` method - the setting is silently skipped rather than failing when the pipeline doesn't support it.

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

- **First run pays the compile cost.** The [server](SERVER.md)'s persistent worker keeps compiled pipelines loaded between runs, so the cost is paid once per session rather than once per generation. A new session still retraces, but the kernels it built come back from the on-disk cache, which dw keeps across reboots (`TORCHINDUCTOR_CACHE_DIR`, see [Environment Defaults](#environment-defaults)). SDNQ's `triton.jit`/autotune kernels are cached separately by Triton itself (`TRITON_CACHE_DIR`, else `$TRITON_HOME/.triton/cache`, else `~/.triton/cache`); dw sets no `TRITON_*` variable and leaves that cache where it is, shared with any other Triton user on the machine. Both caches' sizes show in `get_server_info`'s `caches` and `GET /api/caches`, and `DELETE /api/caches/{name}` empties one ([SERVER.md](SERVER.md)); the next first step of each shape pays the compile or re-autotune again (100-300 s per shape on MiniMax-H3 for Triton).
- **Measure the cold cost from `timing`.** A pipeline step's manifest entry carries `timing` ([WORKSPACES.md](WORKSPACES.md#step-timing)): on a cold run the compile cost is about `first_pass_s - pass_s`, and `compiled` says whether compile was on. `pass_s` is wall time at step boundaries, not kernel time - on CUDA kernels run asynchronously, so a pass boundary is when the host saw it.
- **Pin the attention backend** on a compiled component (`"attention_backend"` in the same `components` entry) rather than using the pipeline-level per-call context manager, which forces recompiles.
- **Quantize-on-load time is a separate cost.** An SDNQ on-the-fly component re-reads and re-quantizes its bf16 weights on every cold load; `DW_QUANTIZED_CACHE=1` saves the quantized result and reads it back instead - see [QUANTIZATION.md](QUANTIZATION.md#cached-on-the-fly-dw_quantized_cache).
- **Composes with offloading**: apply `group_offload` and `compile` on the same component and the offload hooks are installed first, as required. Skipped with a warning on MPS. The compile cost is paid again every time the worker releases the pipeline ("Workflow changed - releasing cached models"), i.e. on any workflow change, so it pays off only across many runs of one workflow at one shape.
- **Don't combine `fullgraph` with a `cache`**: the cache hooks decide skip-or-compute per step, a data-dependent branch diffusers wraps in `torch.compiler.disable` - it needs the graph break that `fullgraph: true` forbids. Compile with the default (partial) graph mode when a cache is active. `validate_workflow` warns `compile_fullgraph_with_cache` on a component that sets `fullgraph: true` in a pipeline with a `cache`.
- **TorchAO quantization needs compile to be fast** - see [QUANTIZATION.md](QUANTIZATION.md#torchao).

**Example:** [flux-dev-compile.json](../workflows/models/flux-dev-compile.json), [flux-torchao.json](../workflows/models/flux-torchao.json)

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

## Memory Format and Attention Processors

Three older per-component knobs, set in the pipeline `configuration` beside
`vae` / `unet` / `transformer` and applied right after the components load:

| Key | Where | Effect |
| --- | ----- | ------ |
| `channels_last` | `vae`, `unet` | `to(memory_format=torch.channels_last)`. Faster convolutions on CUDA for a convolutional UNet or VAE; nothing to gain on a transformer |
| `enable_forward_chunking` | `unet` | Runs the UNet's feed-forward layers in chunks - less peak memory, slightly slower |
| `attn_processor_type` | `unet`, `transformer` | Names an attention processor class to install with `set_attn_processor` (the name is resolved and constructed, so it goes through the `_type` conversion: `"AttnProcessor2_0"`). For a per-call backend instead, see [Attention Backends](#attention-backends) |

```json
"configuration": {
    "component_type": "StableDiffusionPipeline",
    "unet": { "channels_last": true, "enable_forward_chunking": true },
    "vae": { "enable_slicing": true, "channels_last": true }
}
```

## Memory Offloading

`offload` (`"model"` or `"sequential"`) and `group_offload` trade speed for VRAM by streaming weights between system memory and the accelerator instead of keeping everything resident. `"model"` moves whole submodules and costs the least speed; `"sequential"` moves individual layers and is the slowest but uses the least memory; block/leaf-level `group_offload` sits between the two and is what a modular pipeline's self-loaded components use, since they aren't reachable in time for `offload`. Full configuration syntax is in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#memory-offloading). Omit both for the fastest run, when VRAM allows it. Measured on an RTX 3090 (RECIPES_24GB.md): sequential
offload costs about 6x per step on FLUX dev and 3.4x on Z-Image against model offload;
a block-level, streamed `group_offload` of the transformer from pinned host memory
matches the resident per-step cost while holding one block on the card, at ~22 GB of
pinned host RAM per FLUX pipeline - the placement the unquantized FLUX templates use.

`"residency": "on_demand"` on a component is the cheap case of the same trade: the model rests in system memory and is moved to the device whole around each of its own calls. That is a bad deal for anything called once per step, and a good one for a VAE called twice a run - it frees the VAE's VRAM for the denoise loop at the cost of two transfers, where group offloading the same VAE would restream it once per decode tile. See [On-demand components](WORKFLOW_GUIDE.md#on-demand-components).

**Example:** [flux-dev.json](../workflows/models/flux-dev.json) (`"offload": "model"`), [z-image.json](../workflows/models/z-image.json) (`"offload": "sequential"`), [video-with-audio.json](../workflows/templates/minimax/video-with-audio.json) (`group_offload` per component), [reference-to-video.json](../workflows/templates/minimax/reference-to-video.json) (`group_offload` for the transformer, `on_demand` for the VAEs)

## Reading Memory While Offloading

A workflow that offloads keeps its weights in host memory by design, so the
card can sit near-empty through a generation and the VRAM figures alone say
nothing about what a run holds or fails to release. `get_memory` (MCP) and
`GET /api/memory` report both: `gpu_*` is the card, `host_memory_rss_mb` is
what the worker process holds and `host_memory_peak_rss_mb` the most it has
ever held, beside the machine's `host_memory_total_mb` /
`host_memory_available_mb`.

`host_pinned_reserved_mb` / `host_pinned_allocated_mb`, where the platform
reports them, are torch's pinned-host cache - the staging buffers group
offloading moves weights through. They are part of `host_memory_rss_mb` and
invisible in every `gpu_*` figure, so a worker that has released every model
and still holds gigabytes is usually holding these; they are returned when
the worker switches to a different workflow (#98). A host field is absent,
rather than null, on a platform that cannot measure it.

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

Set automatically at import unless already present in the environment (export your own value to override, or put it in a `.env` file in the directory dw starts from - [.env.example](../.env.example) lists these and the other knobs worth knowing: the Hub, diffusers, transformers and SDNQ):

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Lets the CUDA allocator grow segments instead of fragmenting fixed-size ones. Multi-step workflows churn differently-shaped allocations (generate, upscale, interpolate); fragmentation is what OOMs a card that nominally has room. |
| `HF_ENABLE_PARALLEL_LOADING` | `true` | Loads sharded checkpoints in parallel - faster cold starts. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.0` | MPS only - use all available unified memory. |
| `TORCHINDUCTOR_CACHE_DIR` | `$XDG_CACHE_HOME/dw/torchinductor` (`~/.cache/dw/torchinductor`) | Keeps `torch.compile`'s on-disk cache across reboots - Inductor's own default is under `/tmp`. A new worker still retraces, but a cache hit skips codegen and kernel builds, the bulk of a cold compile. Entries are per shape, so a new size or frame count compiles once more. Listed as `inductor` by `GET /api/caches` and cleared with `DELETE /api/caches/inductor`. Triton's cache is not set by dw and stays at its own default (`~/.triton/cache`) - see the compile notes above. |

For faster model downloads, optionally `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`. Not enabled automatically - it bypasses the Python HTTP stack and breaks some proxy setups.

## MPS Notes

Apple Silicon has narrower acceleration support than CUDA:

- No flash-attn, no Triton, no bitsandbytes - `attention_backend` is effectively CUDA-only; use `"native"`-family backends or leave it unset on MPS. Validate warns `attention_backend_unavailable` for a CUDA-only one ([Attention Backends](#attention-backends)). `compile` is skipped with a warning (inductor support on MPS is immature). A workflow that loads a bitsandbytes checkpoint (`flux2-dev`, `multi-image-reference`) or a TorchAO int4 config (`flux-torchao`) has no Mac path yet.
- Workflows written for CUDA are adapted rather than refused: a `cuda` device becomes `mps`, SDNQ's `quantization_device` follows it and its quantized matmul is switched off (it runs through `torch._int_mm`, ~500x slower on MPS), group-offload CUDA streams are dropped, and `"offload": "sequential"` becomes `"model"`. Each change is logged as a warning.
- Attention slicing is automatic; set `disable_attention_slicing` for SDXL-family models (head dim 64), where it is 2.4x slower in attention. It helps SD 1.5.
- `float16` can produce NaN values on Apple Silicon - use `bfloat16`; dw only warns, it doesn't override the dtype for you.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` defaults to `0.0` (use all unified memory) unless already set in the environment.
- Ops the MPS backend lacks: with `PYTORCH_ENABLE_MPS_FALLBACK=1` (set by the `fp4-fp8-for-torch-mps` package install.sh adds) they run on the CPU, and dw logs torch's warning when one does. Without it they raise.
- Memory figures are real: allocated, driver-reserved, and Metal's recommended working set as the total. `vram_estimate` checks a Mac against that total, since the catalog's `cost` entries are CUDA cards; the per-voxel figures were calibrated on CUDA.
- Cost estimates quoted on a Mac are the CUDA figures (`basis: "other_device"`) until the Mac's own runs build observed history - expect them to be optimistic.
- Parallel checkpoint loading (`HF_ENABLE_PARALLEL_LOADING`) defaults to off on macOS - concurrent loader threads copying onto MPS segfaulted a quantized load. Set it explicitly to override.
- Offloading has less benefit than on CUDA, since unified memory is already shared between CPU and GPU.
