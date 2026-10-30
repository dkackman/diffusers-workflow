# Attributions

diffusers-workflow is Apache-2.0. This page credits the work it builds on that
a dependency list does not show: code that was copied in, and other projects
whose ideas were reimplemented here. Python dependencies are in
[requirements.txt](../requirements.txt) and [pyproject.toml](../pyproject.toml).
Model and LoRA weights keep their own licences, and each catalog entry
([workflows/models/](../workflows/models/), [loras/](../loras/)) names the
repository it downloads from.

## Code included in this repository

Each file keeps its upstream licence notice in its header.

| File | Source | Licence |
|---|---|---|
| [dw/tasks/rife_model.py](../dw/tasks/rife_model.py) | The RIFE interpolation network from [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | MIT, Copyright (c) Megvii Inc. |
| [dw/tasks/h3_latent_upscaler_model.py](../dw/tasks/h3_latent_upscaler_model.py) | The MiniMax H3 3D-convolution latent upscaler from [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | MIT, Copyright (c) LBH-123-AI |
| [dw/community_pipelines/pipeline_flux_rf_inversion.py](../dw/community_pipelines/pipeline_flux_rf_inversion.py) | The diffusers community RF-Inversion pipeline, after [RF Inversion](https://rf-inversion.github.io/) by Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai and Wen-Sheng Chu | Apache-2.0, Copyright 2024 Black Forest Labs and The HuggingFace Team |
| [dw/prompt_weighting.py](../dw/prompt_weighting.py) | Based on [sd_embed](https://github.com/xhinker/sd_embed) by Andrew Zhu | Apache-2.0 |

## Ideas reimplemented here

These projects showed an approach that dw rebuilt in its own code for
diffusers. No code, prompt text or data was copied from them; where a licence
does not allow reuse, only the concept was taken. The issue numbers link to the
design discussion, which names the source files consulted.

### [vrgamegirl19/comfyui-vrgamedevgirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl)

A ComfyUI node pack and set of workflows for MiniMax H3 and LTX-2 film making,
assessed in October 2026. It is source-available under a licence that does not
allow reuse here, so only concepts were taken. It inspired:

- holding the reference audio during H3 denoising, and a refine pass after the
  latent upscale ([#598](https://github.com/dkackman/diffusers-workflow/issues/598))
- tracked, distance-gated face repair with LTX on face crops ([#599](https://github.com/dkackman/diffusers-workflow/issues/599))
- the music-video timing stack: beat analysis, a lyric- and beat-aware cut
  planner, pad-then-trim shots ([#600](https://github.com/dkackman/diffusers-workflow/issues/600))
- overlapping windows for long video sources, joined with a cosine blend
  (`window_video`, `join_windows`; [#601](https://github.com/dkackman/diffusers-workflow/issues/601))
- the exact size and frame-count round trip for video-to-video templates
  (`fit_to_model`, `restore_to_source`; [#602](https://github.com/dkackman/diffusers-workflow/issues/602))
- the finishing tasks: LUT apply and make, film grain, sharpen, tonal controls
  in `grade` ([#603](https://github.com/dkackman/diffusers-workflow/issues/603))
- stem separation as its own task (`separate_stems`; [#604](https://github.com/dkackman/diffusers-workflow/issues/604))
- strength-selectable same-size LTX refine and per-segment strength decay in
  chains ([#606](https://github.com/dkackman/diffusers-workflow/issues/606))
- laying out a reference sheet from individual images (`ingredients_grid`; [#607](https://github.com/dkackman/diffusers-workflow/issues/607))
- the H3 dialogue-budget warnings ([#608](https://github.com/dkackman/diffusers-workflow/issues/608))
- checking a take's transcript against the expected lines (`check_script`; [#609](https://github.com/dkackman/diffusers-workflow/issues/609))
- the script-to-video shot-list contract and voice bible ([#610](https://github.com/dkackman/diffusers-workflow/issues/610))
- H3 chunk continuation in latent space ([#611](https://github.com/dkackman/diffusers-workflow/issues/611))
- trying the Alissonerdx H3 LoRAs, which its workflows use ([#612](https://github.com/dkackman/diffusers-workflow/issues/612))

### [Kablex/ComfyUI-Ref2VA-VSA](https://github.com/Kablex/ComfyUI-Ref2VA-VSA)

Apache-2.0. Showed that MiniMax H3's Ref2VA attention could run block-sparse,
building on the [FastVideo](https://github.com/hao-ai-lab/FastVideo) team's
VSA-H3 sparse-attention work. dw's `sparse` attention backend
([dw/sparse_attention.py](../dw/sparse_attention.py),
[#931](https://github.com/dkackman/diffusers-workflow/issues/931)) runs on
[comfy-kitchen](https://pypi.org/project/comfy-kitchen/)'s Sol-Attn kernel, a
dependency.

### [mo-akkakk/ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles](https://github.com/mo-akkakk/ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles)

MIT. The tiled low-sigma refine after an H3 upscale (`refine_tiles`,
[#1043](https://github.com/dkackman/diffusers-workflow/issues/1043)), and the
approach of the test that checks a one-segment H3 chain is bit-identical to a
native run on a tiny random-weight model on CPU
([#1046](https://github.com/dkackman/diffusers-workflow/issues/1046)).

### [harlixay7/Kinoloom](https://github.com/harlixay7/Kinoloom)

MIT. Its seamless-chain research gave `analyze_seams` its saturation,
lightness, contrast and detail measures
([#1031](https://github.com/dkackman/diffusers-workflow/issues/1031)), and the
`script-to-video` skill its guidance on re-rolling a chained segment whose seam
fails ([#1034](https://github.com/dkackman/diffusers-workflow/issues/1034)).
