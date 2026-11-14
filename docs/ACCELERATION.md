# Inference Acceleration

The levers that make a generation faster: diffusers' step caches, attention backends, `torch.compile`, layerwise casting, device settings (TF32, cuDNN) and offloading. Offloading trades speed for VRAM; its syntax is in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#memory-offloading). Tested combinations per model family are in [RECIPES_24GB.md](RECIPES_24GB.md); quantization is in [QUANTIZATION.md](QUANTIZATION.md).

## Diffusers Built-in Cache

A `cache` block in the pipeline `configuration` attaches one of diffusers' cache hooks to the pipeline's `transformer` at load time. The hooks reset between runs. `cache` may also be a whole-object `variable:` reference; one that resolves to `null` means no cache ([WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#cache-acceleration)). Every MiniMax-H3 template takes a `cache` argument, `null` by default and wired to the H3 step only.

`validate_workflow` warns, and never refuses, about cache settings that cannot pay off (`dw/acceleration_warnings.py`):

- **`cache_short_schedule`**: a `first_block` or `mag` cache under 20 `num_inference_steps`. Both decide each step's skip on a residual threshold, and the consecutive steps of a short schedule differ too much for it, so the cache likely skips nothing and still perturbs the result. On MiniMax-H3 a `first_block` cache at `0.05` skipped nothing at 12 steps or on a turbo schedule, and took 22% off at 30. Drop the cache or run 20 steps or more. `taylorseer` and `faster` skip on a fixed interval and `text_kv` skips no step, so the rule does not apply to them.
- **`cache_unsupported`**: the pipeline class has no `transformer` (a UNet pipeline such as SD 1.5), or its transformer class has no `enable_cache()` (Z-Image's). The step runs uncached and the job warns that the cache was not applied.
- **`taylorseer_state_unestimated`**: said whenever `taylorseer` is set - see [TaylorSeerCache](#taylorseercache).
- **`compile_fullgraph_with_cache`**: see [torch.compile](#torchcompile).

### Models diffusers has not registered

`first_block` and `mag` look the transformer's block class up in diffusers' block registry and raise when it is absent, so a model that supports `enable_cache()` can still have no usable cache. [cache_blocks.json](../dw/cache_blocks.json) registers the missing block metadata on demand (`dw/cache_blocks.py`); an entry becomes redundant, not wrong, once diffusers registers the class itself. It lists MiniMax-H3, LTX-2 and Qwen-Image-2.1.

- **LTX-2**: its blocks return video and audio streams, and diffusers reads the second stream back from a forward argument named `encoder_hidden_states`. LTX-2's own `encoder_hidden_states` is the text conditioning, so a skipped block would feed the text embeddings back as audio. The entry's `encoder_hidden_states_argument_name` names the argument the second stream actually comes from.
- **Qwen-Image-2.1**: the pipeline's own key/value cache (`use_kv_cache`, on by default) drops the text tokens from the sequence after the first step, and the changed sequence length fails `first_block` and `mag` on the second step. Pass `"use_kv_cache": false` in the step's `arguments` to cache it.

### FirstBlockCache

Compares the first block's residual between steps to decide whether to skip the remaining blocks. The broadest support.

```json
"configuration": {
    "component_type": "FluxPipeline",
    "cache": {
        "type": "first_block",
        "threshold": 0.05
    }
}
```

`threshold` defaults to `0.05`. Higher skips more and costs more quality. On MiniMax-H3 (RTX 3090, 1344x768, 4 references) `0.05` took 22% off a 30-step base-model job and nothing off a 12-step or turbo one. On H3 `torch.compile` cost 289 s on the first pass for about 1% per pass, `text_kv` did nothing on Ref2VA, and `taylorseer` ran out of memory. On an H3 template pass `cache={"type": "first_block", "threshold": 0.05}`, at 20 or more base steps and never with a turbo LoRA. `turbo_lead_in` ([WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#laying-out-the-composition-without-the-turbo-lora-turbo_lead_in)) is refused with a `first_block` cache: its residual threshold reads the adapter switch as a change, and cached steps skip it.

**Example:** [step-caching.json](../workflows/templates/step-caching.json)

### MagCache

Magnitude-based skipping with error accumulation:

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
| `mag_ratios` | required unless `calibrate` | Preset name or per-step ratio array - see below |
| `threshold` | 0.06 | Accumulated error threshold for skipping |
| `num_inference_steps` | 28 (diffusers) | The transformer forwards the call runs: the pipeline's `num_inference_steps`, except where the scheduler runs fewer (MiniMax-H3 runs one fewer). Set it to match the call |
| `max_skip_steps` | 3 | Max consecutive steps to skip |
| `retention_ratio` | 0.2 | Fraction of initial steps where skipping is disabled |
| `calibrate` | false | Measure ratios for a new model instead of skipping - see below |

#### Supplying `mag_ratios`

The ratios describe how each step's output magnitude behaves for one checkpoint, so MagCache cannot run on defaults alone. Give either:

- **A preset name**: `"mag_ratios": "flux"` resolves to diffusers' `FLUX_MAG_RATIOS`. A preset a later diffusers adds (`<NAME>_MAG_RATIOS`) works by name with no change here; an unknown name fails with the list of presets.
- **An array**: `"mag_ratios": [1.0, 0.98, 0.96, ...]`. An array whose length differs from `num_inference_steps` is interpolated.

For a model with no preset, run once with `"calibrate": true`:

```json
"cache": { "type": "mag", "calibrate": true, "num_inference_steps": 28 }
```

Calibration skips nothing. At the end of the run the ratios appear as a job warning and a `log` event (`get_job`'s `warnings`, `get_job_events`, `export_job`): `MagCache calibration (N steps): [ ... ] — paste into mag_ratios`. While calibrating, the run's own `num_inference_steps` is used, and a mismatch with the cache's is reported as a warning. Paste the array into `mag_ratios`, drop `calibrate`, and set the cache's `num_inference_steps` to the reported `N`, which counts the forwards actually made: MiniMax-H3's scheduler turns N sigmas into N - 1 evaluations, so a 30-step H3 call reports 29 ratios.

A pipeline whose config fixes a `denoising_step_list` (a distilled sampler that takes no `num_inference_steps`) runs `len - 1` passes, and that count is what the short-schedule warning, progress and cost scaling read. MagCache calibration and TST are keyed on `num_inference_steps`, so on such a pipeline both are skipped with a warning that they are not supported on a fixed step-list sampler (`dw/step_passes.py`).

A pipeline that runs classifier-free guidance as two passes in separate `cond` and `uncond` cache contexts (Qwen-Image-2.1 with `true_cfg_scale` above 1 and a `negative_prompt`, or the other Qwen-Image pipelines) emits two lists, labeled `[cond]` and `[uncond]`; use the `cond` one. A guidance-distilled pipeline such as Flux, or one that batches both passes into one forward, emits one unlabeled list. A pipeline the cache cannot attach to (see `cache_unsupported` above) reports nothing.

### TaylorSeerCache

Approximates skipped outputs with a Taylor series:

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
| `max_order` | 1 | Taylor series order (higher = better approximation, more state on the GPU) |

TaylorSeer keeps per-block state on the GPU that no `vram_estimate` charges, so a job that fits without it can run out of memory with it. On MiniMax-H3 at 1344x768, 30 steps and `max_order` 1, a job `validate_workflow` had passed ran out of memory 2 min in on a 24 GB card. Validate warns `taylorseer_state_unestimated`, with the job's estimate and the card's capacity when both are known; the state comes on top of that figure and is not sized.

### FasterCache

Experimental and video-oriented (CogVideoX, for example); uses FFT frequency decomposition. dw wires diffusers' `current_timestep_callback` to the pipeline, so no other parameter is needed for basic use:

```json
"cache": {
    "type": "faster"
}
```

| Property | Default | Description |
| -------- | ------- | ----------- |
| `is_guidance_distilled` | inferred | `true` for a guidance-distilled model (FLUX.1-dev, FLUX.1-Krea-dev), where `guidance_scale` is an embedding and the transformer sees a batch of 1. Unset, dw sets it `true` when the transformer's config has `guidance_embeds: true`, else diffusers' `false`. Wrong, FasterCache assumes a cond/uncond batch of 2 and fails at the first step with `tuple index out of range` (#895). Validate does not check it |

### TextKVCache

Caches the transformer's key/value projections of the unchanging text embeddings across steps. No parameters.

```json
"cache": {
    "type": "text_kv"
}
```

### Reading the skip counts

A step with a `cache` reports `timing.cache` on its manifest entry (fields in [WORKSPACES.md](WORKSPACES.md#step-timing)). `hit_rate` is the fraction of denoiser forwards whose remaining blocks were skipped. It counts forwards, not steps: classifier-free guidance runs two per step, so a 30-step run is 60 forwards. Only `first_block` and `mag` are counted; `taylorseer`, `faster` and `text_kv` report null counts.

Zero skips is not an uncached run. With `first_block` the head block always runs, and any cache wraps the blocks in hooks that recompute the residual in the transformer's dtype, which changes the order and precision of operations; the MiniMax-H3 matrix run P2 changed output with nothing skipped. Judge quality against an uncached run (a `null` cache), not against `skipped == 0`. A threshold sweep (#875) reads `hit_rate` against quality.

## Attention Backends

`attention_backend` in the pipeline `configuration` selects diffusers' attention implementation for the duration of each pipeline call, via a context manager around `pipeline(...)`:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "attention_backend": "flash_hub"
}
```

A component can pin its backend instead, persistently, via `set_attention_backend`:

```json
"configuration": {
    "components": {
        "transformer": { "attention_backend": "flash_hub" }
    }
}
```

Pin it on a compiled component: the per-call context manager switches implementations under the compiled graph and forces a recompile on every run. Set at both levels, the component's own wins.

Values are diffusers' `AttentionBackendName` enum, plus dw's [`sparse`](#sparse-minimax-h3). Common ones: `native`, `flex`, `flash`, `flash_hub`, `sage`, `sage_hub`. Availability depends on the installed packages and the platform:

- `_hub` backends are fetched from the Hugging Face Hub kernel registry on first use through the `kernels` package, which dw installs.
- diffusers refuses a Hub kernel from a publisher it does not list as trusted - `sage_hub` among them (SageAttention 2 from `SageAttention/sage-attention`) - until `DIFFUSERS_TRUST_REMOTE_KERNELS=true` is set in the server's environment (`.env`). That allows remote code from any such publisher.
- The local `sage` backend needs `sageattention>=2.1.1`, which PyPI does not carry.
- The flash, flash-3, sage and xformers families need CUDA, `_hub` variants included. On a server whose device is not CUDA, `validate_workflow` warns `attention_backend_unavailable`, naming the backend and the device: the job would fail at load. A `native`-family backend never warns, and a backend diffusers adds later draws no warning until dw's list (`CUDA_ONLY_ATTENTION_BACKENDS`) is updated.
- The backends the server offers (`/api/health`'s `attention_backends`, the web UI's menu) follow diffusers' own requirement check: a backend whose package is missing here, or older than the installed diffusers needs, is left out (#1315). diffusers 0.43 raised several of those minimums, so an upgrade can shorten the list until the package is updated. dw states none of the minimums itself; they move with diffusers.

On an RTX 3090 `sage_hub` measured about 30% faster per MiniMax-H3 step than `native` (13.8 s against 19.2 s at 960x544x124, 54.7 s against 81.2 s at 345 frames). LTX-2.5 cannot use sage: its transformer passes an attention mask, which sage refuses ("`attn_mask` is not supported for sage attention", measured 2026-10-10), so the LTX templates expose no backend.

Either form stays on what it names. diffusers' `set_attention_backend` also makes its backend the process-wide default, which the persistent worker would keep for every later model; dw restores the default after pinning, so an H3 `sage_hub` pin cannot reach a Music3 or LTX job that follows. The server-wide default is `DIFFUSERS_ATTN_BACKEND` in `.env`; leave it `native`, since it reaches every model and sage cannot serve all of them.

### `sparse` (MiniMax-H3)

`attention_backend: "sparse"` runs H3's attention through comfy-kitchen's block-sparse kernel (`dw/sparse_attention.py`, #931): each query tile attends only to the key tiles that matter, so the cost of the long video-token sequence falls. Text, references and audio stay dense. comfy-kitchen is a core dependency, so it needs no `kernels` package or `DIFFUSERS_TRUST_REMOTE_KERNELS`. The only backend a bundled workflow sets is `sparse`.

Every H3 template in [templates/minimax/](../workflows/templates/minimax/) takes it as variables, pinned on the H3 step's transformer (`components.transformer`, or `components.transformer_ref` for a ref2va workflow): `attention_backend` (default `"sparse"`), `sparse_tau` (0.5) and `sparse_topk` (0.15). That covers the shots, chains and episodes (`dialogue-short`, `shots-batch`, `music-video`, `storyboard` among them), and only their H3 steps: a Z-Image still or Music3 song beside an H3 shot keeps the process default. Pass `attention_backend=sage_hub` (or `native`) to override.

- **Where it applies**: the H3 transformer, at pipeline or component level. On any other component, or a pipeline with no H3 transformer, it runs dense with a warning. A pipeline-level `sparse` skips a component pinned to another backend and warns `attention_backend 'sparse' at pipeline level skips '<component>', which pins '<backend>'`. When it installs, the run logs `Using attention backend: sparse (tau <tau>) on <components>`, or `(topk <k>)`.
- **`sparse_tau`**: number >= 0, default 0.5. How much of the attention mass a tile may drop: higher is sparser and faster, 0 keeps the most. Anything else, such as a `variable:` that resolves to a string or a bool, fails the step rather than running dense.
- **`sparse_topk`**: a number strictly between 0 and 1, unset by default outside the templates. Switches to top-k routing: each query tile keeps that fraction of key tiles and drops the rest, and `sparse_tau` is ignored. It is the selection lightx2v's Turbo-SLA LoRAs were trained against (85% sparsity, so `0.15`). Pass `sparse_topk=null` to an H3 template to route by `sparse_tau`.
- Both are runtime settings: changing one reuses the loaded pipeline.

**Top-k 0.15 against tau 0.5** (#932). On `video-with-audio-768p`, seed 42, top-k ran 21.1 s/forward against 23.6 s at tau 0.5, about 11% faster, with no visible difference under the Turbo-SLA LoRA or the default v1.2 turbo (so dw does not adopt the SLA LoRA). On the H3 quality ramp's 4-reference sitcom shot, seeds 42, 7 and 1234, Don judged top-k equal or better: on the turbo L7 rung tau showed blue ghosting at seed 7 and slight smearing at all three, and on the 30-step base-model L5 rung the skin held.

| rung (minutes per job, seeds 42 / 7 / 1234) | tau 0.5 | top-k 0.15 |
|---|---|---|
| L7: ref2v 4-step turbo, int8 | 6.5 / 6.5 / 6.5 | 6.1 / 6.2 / 5.8 |
| L5: base model, int8, 30 steps | 26.8 / 26.6 / 26.4 | 23.9 / 25.5 / 23.9 |

**Sparse against `native` and `sage_hub`.** Against `native`, which an H3 template ran before #931 unless `attention_backend` was set, sparse is 2.2-4.4x faster per step. Against `sage_hub` the gain is 6-16% depending on the shape, at equal quality: on the H3 skill's quality-ramp rungs Don judged faces, skin and draft-to-final staging equal under both across three seeds (#931 stage 3). RTX 3090, seed 42, s per transformer forward (one per denoise step):

| shape | native | `sage_hub` | sparse | sparse vs native | sparse vs `sage_hub` |
|---|---|---|---|---|---|
| T2VA 960x544, 4-step turbo, tau 1.0 | 24.3 | - | 10.8 | 2.25x | - |
| T2VA 1344x768, 4-step turbo, tau 1.0 | 96.1 | - | 21.9 | 4.38x | - |
| Ref2VA 1344x768, 8-step ref2v turbo, tau 1.0 | 108.4 | - | 42.1 | 2.58x | - |
| Ref2VA 1344x768, 8-step ref2v turbo, tau 0.5 (the default) | 108.4 | 51.0 | 44.1-44.8 | 2.42x | 1.16x |

Wall time, same order: 150 -> 95 s, 457 -> 158 s, 965 -> 430 s, and for the last row 965 s native, 489 s `sage_hub`, 442-450 s sparse. Peak VRAM did not change. By eye, tau 0.5 matched dense and tau 1.0 gave smoother skin; that is one shape and one seed, not a rule that higher tau is always faster or smoother. The native and tau 1.0 figures come from the #931 prototype; the `sage_hub` figure and the 44.1 s come from the engine on lem after the deploy, with a newer diffusers (jobs 94c04976d21d and 593131ca713f), so the sage comparison is between those two jobs only.

On the quality-ramp rungs (ep 6 cold open, 4 references, 1344x768, 124 frames; minutes per job, seeds 42 / 7 / 1234, same diffusers):

| rung | `sage_hub` | sparse | sparse vs `sage_hub` |
|---|---|---|---|
| L1: ref2v 4-step turbo, int4, 5 steps | 5.5 / 5.5 / 5.5 | 5.2 / 5.2 / 5.1 | 1.06-1.08x |
| L7: ref2v 4-step turbo, int8, 5 steps | 7.1 / 6.9 / 7.1 | 6.5 / 6.5 / 6.5 | 1.06-1.09x |
| B3: base model, int8, 12 steps | 13.0 / 12.9 / 12.9 | 12.3 / 12.0 / 12.1 | 1.06-1.07x |
| L5: base model, int8, 30 steps | 29.7 / 29.7 / 29.7 | 26.8 / 26.6 / 26.4 | 1.11-1.12x |

The `first_block` cache on `sage_hub` (23.1 min at L5's settings) is still faster than sparse alone there; sparse with that cache is not measured.

**Where it cannot run.** It needs CUDA with compute capability 8.0 or higher. Anywhere else (MPS, CPU, an older card, an unsupported dtype or head size) the run goes dense instead of failing, and says so twice: `validate_workflow` warns `sparse_attention_fallback` when the server's device is not CUDA, and the run reports `<label>: sparse attention ran dense - <reason>` once per pipeline call, so a chained, `shots-batch` or `dialogue-short` job warns once per segment or shot, under its label. Whether a call runs sparse is decided once, before its first forward: a comfy-kitchen kernel that fails on its first invocation runs that call dense (`sparse kernel failed: <error>`), and one that fails after it has run fails the run.

**Sparse moves composition** relative to dense, unlike sage: the same subject, story and cuts, but different framing and set dressing. A draft and its final must use the same backend; mixing `sparse` and `sage_hub` between tiers means the draft no longer predicts the final.

## Attention Slicing

```json
"configuration": {
    "component_type": "FluxPipeline",
    "enable_attention_slicing": true
}
```

Processes attention in slices to reduce memory at some cost to speed. Automatic on MPS unless `disable_attention_slicing` is set, opt-in elsewhere. Only UNet/ControlNet models take it; a pipeline with no `enable_attention_slicing()` (a modular pipeline) skips it. On MPS the effect depends on the attention head dim: MPS SDPA is slow at 40, 48 and 160 (slicing made SD 1.5 ~20% faster end to end) and fast at 64-128 (slicing made SDXL-shaped attention 2.4x slower), so set `disable_attention_slicing` for SDXL-family models.

## torch.compile

Configured per component under `components`. The component is compiled once fully configured, so the graph captures final dtypes, adapters, quantization and offload hooks:

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
| `repeated_blocks` | Compile only the model's repeated block classes (diffusers regional compilation): near the speedup of full compilation at a fraction of the cold-start cost. Recommended. A component without `compile_repeated_blocks()` fails |
| `mode` | torch.compile mode: `"default"`, `"reduce-overhead"`, `"max-autotune"` |
| `fullgraph` | Require a single graph with no breaks: fails fast instead of silently losing speedup |
| `dynamic` | Compile with dynamic shapes. Set `true` when resolutions or frame counts vary between runs, to avoid recompiles |

Typical gains are 1.3-1.5x on diffusion transformers, and compile stacks with the caches above.

- **The first run pays the compile cost.** The [server](SERVER.md)'s persistent worker keeps compiled pipelines loaded between runs. It releases them on any workflow change ("Workflow changed - releasing cached models"), and the compile is paid again, so compile pays off across many runs of one workflow at one shape. A new worker retraces, but Inductor's kernels come back from its on-disk cache (`TORCHINDUCTOR_CACHE_DIR`, see [Environment Defaults](#environment-defaults)). SDNQ's Triton kernels are cached by Triton itself (`TRITON_CACHE_DIR`, else `$TRITON_HOME/.triton/cache`, else `~/.triton/cache`); dw sets no `TRITON_*` variable. Both caches' sizes show in `get_server_info`'s `caches` and `GET /api/caches`, and `DELETE /api/caches/{name}` empties one ([SERVER.md](SERVER.md)); the next first step of each shape pays the compile or re-autotune again (100-300 s per shape on MiniMax-H3 for Triton).
- **Measure the cold cost from `timing`.** A pipeline step's manifest entry carries `timing` ([WORKSPACES.md](WORKSPACES.md#step-timing)): on a cold run the compile cost is about `first_pass_s - pass_s`, and `compiled` says whether compile was on. `pass_s` is wall time at step boundaries, not kernel time.
- **Pin the attention backend** on a compiled component ([Attention Backends](#attention-backends)).
- **Don't combine `fullgraph` with a `cache`.** The cache's per-step skip decision is a data-dependent branch diffusers wraps in `torch.compiler.disable`, which needs the graph break `fullgraph: true` forbids. `validate_workflow` warns `compile_fullgraph_with_cache` on such a component.
- **Composes with offloading**: `group_offload` and `compile` on the same component install the offload hooks first, as required.
- **`turbo_lead_in` changes a LoRA scale mid-call.** It sets the `turbo` adapter to 0, then back, after N evaluations. Under `torch.compile` peft's scale is a Python float the compiled forward guards on, so the switch can cost a recompile.
  <!-- turbo_lead_in compile figure: measured on lem, #1352 -->
- **Skipped on MPS** with a warning.
- **Quantize-on-load is a separate cost.** An SDNQ on-the-fly component re-quantizes on every cold load unless `DW_QUANTIZED_CACHE=1` ([QUANTIZATION.md](QUANTIZATION.md#cached-on-the-fly-dw_quantized_cache)).
- **TorchAO quantization needs compile to be fast** ([QUANTIZATION.md](QUANTIZATION.md#torchao)).

**Example:** [flux-dev-compile.json](../workflows/models/flux-dev-compile.json), [flux-torchao.json](../workflows/models/flux-torchao.json)

## Per-Component Dtype

A pipeline's `from_pretrained_arguments.torch_dtype` can be diffusers' per-component dict instead of one dtype. This keeps sensitive components at higher precision while the rest of the pipeline stays narrow:

```json
"from_pretrained_arguments": {
    "model_name": "Lightricks/LTX-2",
    "torch_dtype": {
        "default": "torch.bfloat16",
        "audio_vae": "torch.float32",
        "vocoder": "torch.float32"
    }
}
```

**Components a pipeline chains together must share a dtype** unless the pipeline casts between them. For LTX-2, load `audio_vae` and `vocoder` in the same dtype. The decode casts the audio latents to `audio_vae.dtype`, then hands the mel to the vocoder with no cast. A vocoder-only dict validates, then fails at decode with a dtype mismatch (on MPS: `Input type (c10::BFloat16) and bias type (float) should be the same`). The engine doesn't check this, because which components chain is model knowledge.

- **Keys** are the pipeline's component names, passed through to diffusers as written. The engine knows no model's components.
- **`default`** covers every component the dict doesn't name. Without it, those components load as `torch.float32`, and `validate_workflow` warns.
- **Values** must each name a dtype. A `{}`-escaped value or an unknown name is refused, and the error names the component.
- **Readers:**
  - The MPS float16 warning names each float16 component.
  - The quantized cache keys each component on its own dtype.
  - The warm-pipeline cache keys each dict separately from the single-dtype form.
- **Scope:** a component declared in its own `components` entry still takes its own single `torch_dtype`.

## Layerwise Casting

Store a component's weights in a narrow dtype and upcast only for compute:

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

Both dtypes are required. It is applied through the component's own `enable_layerwise_casting()` right after it loads, so it composes with quantization and group offloading on the same component. The pipeline `configuration` takes the same block.

## Memory Format and Attention Processors

Three per-component keys, set in the pipeline `configuration` beside `vae` / `unet` / `transformer` and applied right after the components load:

| Key | Where | Effect |
| --- | ----- | ------ |
| `channels_last` | `vae`, `unet` | `to(memory_format=torch.channels_last)`. Faster convolutions on CUDA for a convolutional UNet or VAE; nothing to gain on a transformer |
| `enable_forward_chunking` | `unet` | Runs the UNet's feed-forward layers in chunks: less peak memory, slightly slower |
| `attn_processor_type` | `unet`, `transformer` | An attention processor class to construct and install with `set_attn_processor`, named through the `_type` conversion (`"AttnProcessor2_0"`). For a per-call backend instead, see [Attention Backends](#attention-backends) |

```json
"configuration": {
    "component_type": "StableDiffusionPipeline",
    "unet": { "channels_last": true, "enable_forward_chunking": true },
    "vae": { "enable_slicing": true, "channels_last": true }
}
```

## Memory Offloading

`offload` (`"model"` or `"sequential"`) and `group_offload` stream weights between system memory and the accelerator instead of keeping them resident; syntax is in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#memory-offloading). Omit both for the fastest run when VRAM allows.

- `"model"` moves whole submodules and costs the least speed.
- `"sequential"` moves individual layers: the slowest, and the least memory.
- `group_offload` (block or leaf level) sits between them, and is what a modular pipeline's self-loaded components use, since they are not reachable in time for `offload`.
- `"residency": "on_demand"` on a component rests the model in system memory and moves it to the device whole around each of its own calls. A bad deal for anything called once per step; a good one for a VAE called twice a run, where group offloading would restream it once per decode tile ([On-demand components](WORKFLOW_GUIDE.md#on-demand-components)).

Measured on an RTX 3090 ([RECIPES_24GB.md](RECIPES_24GB.md)): sequential offload costs about 6x per step on FLUX dev and 3.4x on Z-Image against model offload. A block-level, streamed `group_offload` of the transformer from pinned host memory matches the resident per-step cost while holding one block on the card, at ~22 GB of pinned host RAM per FLUX pipeline; the unquantized FLUX templates use it.

**Example:** [flux-dev.json](../workflows/models/flux-dev.json) (`"offload": "model"`), [krea2.json](../workflows/models/krea2.json) (`"offload": "sequential"`), [video-with-audio.json](../workflows/templates/minimax/video-with-audio.json) (`group_offload` per component), [reference-to-video.json](../workflows/templates/minimax/reference-to-video.json) (`group_offload` for the transformer, `on_demand` for the VAEs)

## Reading Memory While Offloading

A workflow that offloads keeps its weights in host memory, so the card can sit near-empty through a generation and the VRAM figures alone say nothing about what a run holds or fails to release. `get_memory` (MCP) and `GET /api/memory` report both: `gpu_*` is the card; `host_memory_rss_mb` is what the worker process holds and `host_memory_peak_rss_mb` the most it has held, beside the machine's `host_memory_total_mb` and `host_memory_available_mb`.

`host_pinned_reserved_mb` / `host_pinned_allocated_mb`, where the platform reports them, are torch's pinned-host cache: the staging buffers group offloading moves weights through. They are part of `host_memory_rss_mb` and invisible in every `gpu_*` figure, so a worker that has released every model and still holds gigabytes is usually holding these; they are returned when the worker switches to a different workflow (#98). A host field is absent, not null, on a platform that cannot measure it.

## TF32 and cuDNN

Device-level settings from `~/.diffusers_helper/settings.json` (the directory moves with `DIFFUSERS_HELPER_ROOT`). They are process-wide torch flags, so every process that runs a pipeline applies them (`configure_torch` in `dw/__init__.py`): the CLI at startup, and each `dw.serve` worker when it starts and again at the start of every job. A settings change reaches served jobs after a server restart. The first job a fresh worker runs narrates what it ran with, read back from torch after the pipelines loaded, as a `log` event: `torch: cudnn benchmark=false deterministic=true, tf32=false`.

The defaults give the same output for the same seed, whatever else the host is doing (#1229; `bench/records/determinism.md`):

| Setting | Default | Effect |
| ------- | ------- | ------ |
| `enable_tf32` | `false` | `true` sets `torch.set_float32_matmul_precision("high")`, and on CUDA also `torch.backends.cuda.matmul.allow_tf32 = True`: faster fp32 matmuls on Ampere and newer (RTX 30/40, A100, H100), numerically different from full fp32. Most pipelines run in bf16, so this mostly touches fp32 text encoders and VAEs |
| `cudnn_benchmark` | `false` | CUDA only. `true` autotunes cuDNN's algorithm choice by timing it. Timing moves with host load, so on a two-card server, or a card next to other GPU work, the same seed can give different output |
| `cudnn_deterministic` | `true` | CUDA only. Restricts cuDNN to its deterministic algorithms. With `cudnn_benchmark` also on, the autotuner picks among those only |

```json
{ "enable_tf32": false, "cudnn_benchmark": false, "cudnn_deterministic": true }
```

The catalog `cost` rows were measured with benchmark and TF32 off. Turning either on is a speed opt-in that gives up same-seed reproducibility.

## Environment Defaults

Set at import unless already in the environment. Export your own value to override, or put it in a `.env` file in the directory dw starts from; [.env.example](../.env.example) lists these and the other Hub, diffusers, transformers and SDNQ knobs.

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Lets the CUDA allocator grow segments instead of fragmenting fixed-size ones. Multi-step workflows churn differently-shaped allocations; fragmentation is what OOMs a card that nominally has room |
| `HF_ENABLE_PARALLEL_LOADING` | `true`, `false` on macOS | Loads sharded checkpoints in parallel. Off on macOS: concurrent loader threads copying onto MPS segfaulted a quantized load |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.0` | MPS only: use all available unified memory |
| `TORCHINDUCTOR_CACHE_DIR` | `$XDG_CACHE_HOME/dw/torchinductor` (`~/.cache/dw/torchinductor`) | Keeps `torch.compile`'s on-disk cache across reboots (Inductor's own default is under `/tmp`). A cache hit skips codegen and kernel builds, the bulk of a cold compile; entries are per shape. Listed as `inductor` by `GET /api/caches`, cleared with `DELETE /api/caches/inductor` |

For faster downloads, optionally `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`. It is not enabled automatically: it bypasses the Python HTTP stack and breaks some proxy setups.

## MPS Notes

Apple Silicon has narrower acceleration support than CUDA:

- No flash-attn, Triton or bitsandbytes. Use a `native`-family `attention_backend` or leave it unset; validate warns `attention_backend_unavailable` for a CUDA-only one, and `sparse` runs dense. `compile` is skipped with a warning. A workflow that loads a bitsandbytes checkpoint (`flux2-dev`, `multi-image-reference`) or a TorchAO int4 config (`flux-torchao`) has no Mac path.
- Workflows written for CUDA are adapted rather than refused, each change logged as a warning: a `cuda` device becomes `mps`, SDNQ's `quantization_device` and `return_device` follow it and its `use_quantized_matmul` is switched off (it runs through `torch._int_mm`, ~500x slower on MPS), group-offload CUDA streams are dropped, and `"offload": "sequential"` becomes `"model"`.
- Attention slicing is automatic; see [Attention Slicing](#attention-slicing) for when to disable it.
- `float16` can produce NaN values on Apple Silicon; use `bfloat16`. dw warns and does not override the dtype.
- Ops the MPS backend lacks run on the CPU with `PYTORCH_ENABLE_MPS_FALLBACK=1` (set by the `fp4-fp8-for-torch-mps` package install.sh adds), and dw logs torch's warning when one does. Without it they raise.
- Memory figures are real: allocated, driver-reserved, and Metal's recommended working set as the total. `vram_estimate` checks a Mac against that total; the per-voxel figures were calibrated on CUDA.
- Cost estimates on a Mac are the CUDA figures (`basis: "other_device"`) until the Mac's own runs build observed history; expect them to be optimistic.
- A 24 GB Mac (about 17.8 GB recommended working set; an approximate figure, not re-measured - the exact limit is what the device reports and varies by macOS version and machine) is refused at the `minimax/chained-segments` defaults under either continuity (about 18.3 GB projected with a guide, 18.0 GB without). `num_frames` cannot be lowered (the template accepts 124 and up), so lower the canvas: 768x448 passes under guide (`Workflow.validation_errors`); 864x480 passes only under `last_frame`.
- Offloading gains less than on CUDA, since unified memory is already shared between CPU and GPU.
