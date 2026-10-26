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
redundant, not wrong, once diffusers registers the same class upstream. MiniMax-H3,
LTX-2 and Qwen-Image-2.1 are listed there today.

Qwen-Image-2.1's pipeline keeps a key/value cache of its own (`use_kv_cache`, on by
default) that drops the text tokens from the sequence after the first step. `first_block`
and `mag` compare one step's residual against the previous one, so the sequence length
changing under them fails the second step; pass `"use_kv_cache": false` in the step's
`arguments` to cache it.

A transformer that does not support `enable_cache()` at all - Z-Image's, for one - takes
no cache of any type. The step still runs, uncached, and the job carries a warning that the
cache was not applied.

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
| `num_inference_steps` | required | The transformer forwards the call runs: the pipeline's `num_inference_steps`, except where the scheduler runs fewer (MiniMax-H3 runs one fewer) |
| `max_skip_steps` | 3 | Max consecutive steps to skip |
| `retention_ratio` | 0.2 | Fraction of initial steps where skipping is disabled |
| `calibrate` | false | Measure ratios for a new model instead of skipping — see below |

#### Supplying `mag_ratios`

MagCache needs to know how each denoising step's output magnitude typically behaves for *your* checkpoint, so unlike the other cache types it cannot run on defaults alone. Give it either:

- **A preset name** — `"mag_ratios": "flux"` resolves to the ratios diffusers ships for Flux. Any preset a later diffusers release adds is usable by name without a change here.
- **An explicit array** — `"mag_ratios": [1.0, 0.98, 0.96, ...]`. The array is interpolated automatically when its length differs from `num_inference_steps`, so ratios measured at one step count can be reused at another.

For a model with no preset, run once with `"calibrate": true`. Calibration skips nothing. At the end of the run the measured ratios appear as a job warning and a `log` event (`get_job`'s `warnings`, `get_job_events`, `export_job`): `MagCache calibration (N steps): [ ... ] — paste into mag_ratios`. While calibrating, the run's own `num_inference_steps` is the step count, so the cache's can be left at its default (a mismatch is reported as a warning). The cache attaches to a pipeline's `transformer`, and only one that supports `enable_cache()`: a UNet pipeline such as SD 1.5 has no transformer, and Z-Image's transformer has no cache support, so both get a warning that the cache was not applied and no calibration. Paste that array into `mag_ratios` and drop the `calibrate` flag for subsequent runs. `N` is the forwards the run actually made, which is not always the call's `num_inference_steps`: MiniMax-H3's scheduler turns N sigmas into N - 1 evaluations, so a 30-step H3 call reports 29 ratios. Set the cache's `num_inference_steps` to that `N` when you paste them, so each ratio lines up with its forward. A pipeline that runs classifier-free guidance as two transformer passes in their own `cond` and `uncond` cache contexts (Qwen-Image-2.1 with `true_cfg_scale` above 1 and a `negative_prompt`, or the other Qwen-Image pipelines) keeps a separate state for each, so it emits **two** lists, each labeled `[cond]` or `[uncond]`; use the `cond` one, since `mag_ratios` takes one array. A guidance-distilled pipeline such as Flux, or one that batches both passes into one forward emits one unlabeled list.

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

| Property | Default | Description |
| -------- | ------- | ----------- |
| `is_guidance_distilled` | false | Set `true` for a guidance-distilled model (FLUX.1-dev, FLUX.1-Krea-dev), where `guidance_scale` is an embedding and the transformer sees a batch of 1. FasterCache otherwise assumes a cond/uncond batch of 2 and fails at the first step with `tuple index out of range` (#895). Validate cannot tell a distilled model from a CFG one, so it does not check this |

### TextKVCache

Caches the transformer's key/value projections of the (unchanging) text
embeddings across denoising steps, recomputing only what the latents need:

```json
"cache": {
    "type": "text_kv"
}
```

No parameters.

### Reading the skip counts

A step that set a `cache` reports `timing.cache` on its manifest entry (fields in [WORKSPACES.md](WORKSPACES.md#step-timing)). `hit_rate` is the fraction of denoiser forwards whose remaining blocks were skipped. It counts forwards, not steps: classifier-free guidance runs two per step, so a 30-step run is 60 forwards. Only `first_block` and `mag` are counted; `taylorseer`, `faster` and `text_kv` report null counts. A cache with zero skips can still change the output, so a 0-skip run is not guaranteed bit-identical to an uncached one. With `first_block` the head block always runs, and enabling any cache wraps the blocks in hooks that recompute the residual (the tail's output minus the head's, in the transformer's dtype), which changes the order and precision of operations; the MiniMax-H3 matrix run P2 saw a changed output with nothing skipped. Judge quality against an uncached run, not against `skipped == 0`. A threshold sweep (#875) reads `hit_rate` against quality.

## Attention Backends

Select the attention implementation diffusers uses for the duration of each pipeline call, via a context manager wrapped around `pipeline(...)`:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "attention_backend": "flash_hub"
}
```

Common values: `"flash"`, `"flash_hub"`, `"sage"`, `"sage_hub"`, `"native"`, `"flex"`. The full set is diffusers' `AttentionBackendName` enum - availability depends on what's installed (`flash-attn`, `sageattention`, etc.) and the platform. `_hub`-suffixed backends are fetched from the Hugging Face Hub kernel registry on first use, which needs the `kernels` package installed (`pip install kernels`) - it is not a dw dependency. The only backend a bundled workflow sets is `sparse`, the default on every MiniMax-H3 template, and it needs no `kernels` install (comfy-kitchen is a core dependency), so each bundled workflow runs on a plain install. diffusers also refuses a Hub kernel whose publisher it does not list as trusted - `sage_hub` among them, since SageAttention 2 comes from `SageAttention/sage-attention` - until `DIFFUSERS_TRUST_REMOTE_KERNELS=true`, which allows remote code from any such publisher. On an RTX 3090 `sage_hub` measured about 30% faster per MiniMax-H3 step (13.8 s against 19.2 s at 960x544x124, 54.7 s against 81.2 s at 345 frames); the local `sage` backend needs `sageattention>=2.1.1`, which PyPI does not carry.

Every MiniMax-H3 template in [templates/minimax/](../workflows/templates/minimax/)
exposes this as a variable: `attention_backend`, pinned on the H3 step's transformer
(`components.transformer`, or `components.transformer_ref` for a ref2va workflow) and
`"sparse"` by default (with `sparse_tau`, below). That covers the shots, chains and episodes - `dialogue-short`,
`shots-batch`, `music-video` and `storyboard` among them - and only their H3 steps: a
Z-Image still or Music3 song beside an H3 shot keeps the process default. Pass
`attention_backend=sage_hub` (or `native`) to override the default; `sage_hub` needs a box with the `kernels` package and
`DIFFUSERS_TRUST_REMOTE_KERNELS=true` in its `.env`; on MPS the default falls back to dense, but a CUDA-only kernel such as `sage_hub` fails there at load (validation warns, below). LTX-2.5 cannot use it: its transformer
passes an attention mask, and sage attention refuses one ("`attn_mask` is not supported
for sage attention", measured 2026-10-10), so the LTX templates expose no backend.

### `sparse` (MiniMax-H3)

`attention_backend: "sparse"` runs H3's attention through comfy-kitchen's block-sparse kernel: each query tile attends only to the key tiles that matter, so the cost of the long video-token sequence falls. It applies to the H3 transformer only, at pipeline or component level, and is the default on every H3 template (#931). Set at both levels, the component's own `attention_backend` wins, as it does for every backend: a pipeline-level `sparse` skips a component pinned to `sage_hub` (or anything else) and warns `attention_backend 'sparse' at pipeline level skips '<component>', which pins '<backend>'`. When it does install, the run logs `Using attention backend: sparse (tau <tau>) on <components>`, or `(topk <k>)` with top-k routing. `sparse_tau` (number, minimum 0, default 0.5; anything else, such as a `variable:` that resolves to a string or a bool, fails the step rather than running dense) sets how much of the attention mass a tile may drop; 0 keeps the most. Observed at Ref2VA 1344x768, seed 42: tau 1.0 ran 42.1 s/forward against 44.8 s at 0.5, and by Don's eye 1.0 gave smoother skin while 0.5 matched dense (the table below). That is one shape and one seed, not a rule that higher is always faster or smoother.

`sparse_topk` (a number strictly between 0 and 1; the H3 templates set 0.15, and a workflow that sets none routes by `sparse_tau`) switches the kernel to top-k routing: each query tile keeps that fraction of key tiles and drops the rest outright, with `sparse_tau` ignored. It is the selection lightx2v's Turbo-SLA LoRAs were trained against (85% sparsity, so `0.15`). Measured on `video-with-audio-768p`, seed 42 (#932): top-k 0.15 ran 21.1 s/forward against 23.6 s at tau 0.5, about 11% faster, and Don saw no difference on that clip, with either the Turbo-SLA LoRA or the default v1.2 turbo (so dw does not adopt the SLA LoRA). That clip has no faces, so it was checked on the H3 quality ramp's 4-reference sitcom shot at seeds 42, 7 and 1234 against tau 0.5. Don judged top-k equal or better: on the turbo L7 rung tau showed blue ghosting at seed 7 and slight smearing at all three, and on the 30-step base-model L5 rung the skin held. Top-k 0.15 is therefore the H3 templates' default; pass `sparse_topk=null` to route by `sparse_tau` instead.

| rung (minutes per job, seeds 42 / 7 / 1234) | tau 0.5 | top-k 0.15 |
|---|---|---|
| L7: ref2v 4-step turbo, int8 | 6.5 / 6.5 / 6.5 | 6.1 / 6.2 / 5.8 |
| L5: base model, int8, 30 steps | 26.8 / 26.6 / 26.4 | 23.9 / 25.5 / 23.9 |

**What the default buys depends on what it replaces.** Against `native` attention, which is what an H3 template ran before #931 unless `attention_backend` was set, sparse is 2.2-4.4x faster per step. Against `sage_hub`, the gain is much smaller: 6-16% depending on the shape (the second table). So sparse's main win is for a run that set no backend, and it needs no `kernels` package or `DIFFUSERS_TRUST_REMOTE_KERNELS`. A box already on `sage_hub` gains a little, at equal quality: on the H3 skill's quality-ramp rungs, Don judged faces, skin and draft-to-final staging equal under both backends across three seeds (#931 stage 3).

Measured on an RTX 3090, seed 42, s per transformer forward (one per denoise step), lower is faster:

| shape | native | `sage_hub` | sparse | sparse vs native | sparse vs `sage_hub` |
|---|---|---|---|---|---|
| T2VA 960x544, 4-step turbo, tau 1.0 | 24.3 | - | 10.8 | 2.25x | - |
| T2VA 1344x768, 4-step turbo, tau 1.0 | 96.1 | - | 21.9 | 4.38x | - |
| Ref2VA 1344x768, 8-step ref2v turbo, tau 1.0 | 108.4 | - | 42.1 | 2.58x | - |
| Ref2VA 1344x768, 8-step ref2v turbo, tau 0.5 (the default) | 108.4 | 51.0 | 44.1-44.8 | 2.42x | 1.16x |

Wall time, same order: 150 -> 95 s, 457 -> 158 s, 965 -> 430 s, and for the last row 965 s native, 489 s `sage_hub`, 442-450 s sparse.

Against `sage_hub` on the H3 skill's quality-ramp rungs (ep 6 cold open, 4 references, 1344x768, 124 frames; minutes per job, seeds 42 / 7 / 1234, all on the same diffusers):

| rung | `sage_hub` | sparse | sparse vs `sage_hub` |
|---|---|---|---|
| L1: ref2v 4-step turbo, int4, 5 steps | 5.5 / 5.5 / 5.5 | 5.2 / 5.2 / 5.1 | 1.06-1.08x |
| L7: ref2v 4-step turbo, int8, 5 steps | 7.1 / 6.9 / 7.1 | 6.5 / 6.5 / 6.5 | 1.06-1.09x |
| B3: base model, int8, 12 steps | 13.0 / 12.9 / 12.9 | 12.3 / 12.0 / 12.1 | 1.06-1.07x |
| L5: base model, int8, 30 steps | 29.7 / 29.7 / 29.7 | 26.8 / 26.6 / 26.4 | 1.11-1.12x |

The `first_block` cache on `sage_hub` (23.1 min at L5's settings) is still faster than sparse alone there; sparse with that cache is not measured. The native and tau 1.0 figures come from the #931 prototype. The `sage_hub` figure and the 44.1 s come from the engine on lem after the deploy, with a newer diffusers (jobs 94c04976d21d and 593131ca713f), so the sage comparison is between those two jobs only. The T2VA rows have no `sage_hub` run. Peak VRAM did not change. Judged by eye, tau 0.5 matched dense; tau 1.0 was smoother in skin.

It needs CUDA with compute capability 8.0 or higher. Anywhere else (MPS, CPU, an older card, an unsupported dtype or head size) the run goes dense instead of failing, and says so twice: `validate_workflow` warns `sparse_attention_fallback` when the server's device is not CUDA, and the run reports `<label>: sparse attention ran dense - <reason>` once per pipeline call. A job of one call warns once; a chained, `shots-batch` or `dialogue-short` job calls the pipeline once per segment or shot, so it warns once for each, under that segment's or shot's label. A comfy-kitchen kernel that fails on its first invocation in a call also runs that call dense, with `sparse kernel failed: <error>` as the reason; one that fails after it has run in the call fails the run, since some blocks have already run sparse.

Unlike sage, sparse **moves composition** relative to dense: the same subject, story and cuts, but different framing and set dressing. A draft and its final must therefore use the same backend; mixing `sparse` and `sage_hub` between tiers means the draft no longer predicts the final.

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
