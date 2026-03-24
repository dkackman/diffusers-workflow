# Acceleration & Extra Features Analysis

Analysis of [dgenerate/extras](https://github.com/Teriks/dgenerate/tree/d83b839033cc22c5101fb0f987bd4eb2de3d5d12/dgenerate/extras) modules and upstream alternatives, evaluated for inclusion in diffusers-workflow.

Date: 2026-03-23

## Already Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| **TeaCache** | Done | Custom implementation in `dw/teacache.py` with JSON registry for 14 model variants across 7 architectures. Flux forward implemented; others registry-only. |
| **SDNQ** | Done | Uses `sdnq` PyPI package. `pre_load_modules` registration + optional `sdnq_optimize` (CUDA/XPU only). |
| **controlnet_aux** | Done | Installed from PyPI. dgenerate vendors it only for specific bugfixes. |
| **FP8 on MPS** | Done | `fp4-fp8-for-torch-mps` added to macOS install. Auto-activates via `torch.backends`. |

## TODO (Ranked by Recommendation)

### 1. Diffusers Built-in Caching (replace/complement TeaCache)

- [ ] Evaluate diffusers 0.38+ native caching: `FasterCacheConfig`, `FirstBlockCacheConfig`, `MagCacheConfig`, `TaylorSeerCacheConfig`
- [ ] These are maintained by HuggingFace, work with any supported model, and need no custom forward functions
- [ ] Could add as a `cache` config option alongside or replacing `teacache`
- [ ] Biggest maintenance win: eliminates need to write per-model forward functions for TeaCache

**Effort:** Low — diffusers API, no vendoring needed
**Benefit:** High — multi-model caching with zero maintenance, officially supported

### 2. sd_embed (Long Prompt Support)

- [ ] Breaks the 77-token CLIP limit for SD 1.5/SDXL/SD3/Flux
- [ ] Not on PyPI — must vendor ~3 files from https://github.com/xhinker/sd_embed
- [ ] Broadly useful for any user writing detailed prompts

**Effort:** Low-medium — 3 files, clean API
**Benefit:** Medium-high — token limit is a common pain point

### 3. Compel (Advanced Prompt Weighting)

- [ ] `pip install compel` — prompt weighting syntax like `(word)+` for emphasis
- [ ] PyPI package works as-is; dgenerate vendors for minor Stable Cascade/clip_skip mods
- [ ] Would need integration into argument processing

**Effort:** Low — pip package, no vendoring
**Benefit:** Medium — power-user feature for prompt control

### 4. HiDiffusion (Resolution Boost)

- [ ] `pip install hidiffusion` — training-free resolution/speed boost
- [ ] UNet-only (SD 1.5, SDXL) — **not applicable to Flux/SD3/transformer models**
- [ ] Ecosystem is moving toward transformers, limiting future relevance

**Effort:** Low — pip package
**Benefit:** Low-medium — only helps legacy UNet models

### 5. ASDFF/ADetailer (Auto Face Detail Fix)

- [ ] YOLO-based face/body detection + inpainting refinement pass
- [ ] dgenerate version adds SD3/Flux support not in upstream `adetailer`
- [ ] Must vendor ~4 files + requires `ultralytics` dependency
- [ ] Tightly coupled to dgenerate internals — needs significant rework

**Effort:** High — rework needed, heavy dependency
**Benefit:** Medium — useful for portrait generation workflows

## Not Recommended

| Feature | Why Skip |
|---------|----------|
| **RAS** (Reinforced Attention Scheduling) | SD3-only, requires Triton — CUDA-only, dead on MPS |
| **SADA** (Self-Adaptive Diffusion Acceleration) | UNet-only, not pip-installable, overlaps with TeaCache/diffusers caching |
| **Kolors** (Kwai Kolors pipelines) | Niche model. Basic support already in diffusers. Extras are inpaint/ControlNet variants |
| **sd_latent_interposer** | Cross-model latent conversion. Extremely niche workflow |
| **DistillT5** | Distilled text encoder. Requires specific model weights, niche optimization |
| **UltraEdit** | InstructPix2Pix for SD3. Not in diffusers, niche editing workflow |
| **argostranslate** | Offline translation. Vendored in dgenerate for Py 3.13 sentencepiece issues, likely fixed on 3.14 |

## Sources

- dgenerate extras: https://github.com/Teriks/dgenerate/tree/d83b839033cc22c5101fb0f987bd4eb2de3d5d12/dgenerate/extras
- TeaCache: https://github.com/ali-vilab/TeaCache
- Diffusers caching docs: https://huggingface.co/docs/diffusers/optimization/cache
- sd_embed: https://github.com/xhinker/sd_embed
- HiDiffusion: https://github.com/megvii-research/HiDiffusion
- ASDFF: https://github.com/Bing-su/asdff
