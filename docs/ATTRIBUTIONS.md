# Attributions

diffusers-workflow is Apache-2.0. This page credits the work it builds on that
a dependency list does not show: code that was copied in, and other projects
whose ideas were reimplemented here. Python dependencies are in
[pyproject.toml](../pyproject.toml).
Model and LoRA weights keep their own licences, and each catalog entry
([workflows/models/](../workflows/models/), [loras/](../loras/)) names the
repository it downloads from.

## Code included in this repository

Each copied or ported file keeps its upstream licence notice in its header.

| File | Source | Licence |
|---|---|---|
| [dw/tasks/rife_model.py](../dw/tasks/rife_model.py) | The RIFE interpolation network from [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | MIT, Copyright (c) Megvii Inc. |
| [dw/tasks/h3_latent_upscaler_model.py](../dw/tasks/h3_latent_upscaler_model.py) | The MiniMax H3 3D-convolution latent upscaler from [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | MIT, Copyright (c) LBH-123-AI |
| [dw/vendored/audio8_tts_model.py](../dw/vendored/audio8_tts_model.py) | The Audio8-TTS (ArkTTS) dual-AR speech model, codec and processor from [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b/tree/1e857568bfd4f02931aca3988ee773a44d2d26d3), ported (upstream NOTICE kept in the file header) | Apache-2.0 |
| [dw/vendored/mmaudio_model.py](../dw/vendored/mmaudio_model.py), [dw/vendored/mmaudio_clip.py](../dw/vendored/mmaudio_clip.py) | The MMAudio flow-matching network, its sequence lengths and classifier-free guidance, and the CLIP conditioning from [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio/tree/974010a026c731054592d8f777218bd9d85a6c24), ported (the sampler is dw's own Euler loop, and the DFN5B towers are transformers' CLIP models filled by a key map) | MIT, Copyright (c) 2024 Sony Research Inc. |
| [dw/vendored/mmaudio_sync_model.py](../dw/vendored/mmaudio_sync_model.py) | MMAudio's Synchformer visual encoder (a MotionFormer feature extractor) and its 16-frame segmenting, from [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio/tree/974010a026c731054592d8f777218bd9d85a6c24)'s `mmaudio/ext/synchformer/`, ported without timm, einops or omegaconf | MIT, Copyright (c) 2024 Vladimir Iashin |
| [dw/tasks/mmaudio_decoder.py](../dw/tasks/mmaudio_decoder.py) | Loads diffusers' `MMAudioVAE` and `MMAudioVocoder`; the vocoder is [BigVGAN](https://github.com/NVIDIA/BigVGAN) (no BigVGAN code is copied here: the classes are diffusers') | BigVGAN: MIT, Copyright (c) 2024 NVIDIA CORPORATION |
| [dw/community_pipelines/pipeline_flux_rf_inversion.py](../dw/community_pipelines/pipeline_flux_rf_inversion.py) | The diffusers community RF-Inversion pipeline, after [RF Inversion](https://rf-inversion.github.io/) by Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai and Wen-Sheng Chu | Apache-2.0, Copyright 2024 Black Forest Labs and The HuggingFace Team |
| [dw/prompt_weighting.py](../dw/prompt_weighting.py) | Based on [sd_embed](https://github.com/xhinker/sd_embed) by Andrew Zhu | Apache-2.0 |

## Model weights with their own terms

Most weights are named only by their catalog entry. These are listed here as
well, because their terms are narrower than this repository's licence and
apply to anything generated with them.

| Weights | Used by | Licence |
|---|---|---|
| [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) (`weights/mmaudio_large_44k_v2.pth` and its `ext_weights/`) | `generate_sound_effects`: [dw/vendored/mmaudio_model.py](../dw/vendored/mmaudio_model.py) | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only |
| [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) `ext_weights/synchformer_state_dict.pth` | `generate_sound_effects`, MMAudio's sync features, [dw/vendored/mmaudio_sync_model.py](../dw/vendored/mmaudio_sync_model.py) | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only |
| [kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers/tree/4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81) `audio_vae/` and `vocoder/` | `generate_sound_effects`, MMAudio's decoder, [dw/tasks/mmaudio_decoder.py](../dw/tasks/mmaudio_decoder.py) | The MMAudio VAE tensors: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial use only, as MMAudio's weights (K6's model card states MIT; dw follows the stricter MMAudio terms). The BigVGAN vocoder: MIT. |
| [apple/DFN5B-CLIP-ViT-H-14-384](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384) | `generate_sound_effects`, MMAudio's CLIP conditioning, [dw/vendored/mmaudio_clip.py](../dw/vendored/mmaudio_clip.py) | [Apple Machine Learning Research Model License](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384/blob/main/LICENSE): research purposes only. Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement. dw maps the weights into transformers' layout in memory at load time and never writes or ships converted tensors. |

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

### [r34vtraining/Longshot_Studio](https://github.com/r34vtraining/Longshot_Studio)

A ComfyUI studio for making long MiniMax H3 films shot by shot, assessed in
October 2026. It has no licence, so only concepts are taken. Its ideas are
filed as dialogue/effects stem separation
([#1091](https://github.com/dkackman/diffusers-workflow/issues/1091)), an H3
face refine with the audio held
([#1092](https://github.com/dkackman/diffusers-workflow/issues/1092)), cutaway
and lead-in joins ([#1093](https://github.com/dkackman/diffusers-workflow/issues/1093)),
rendering the odd shots of a long take first and bridging the gaps
([#1094](https://github.com/dkackman/diffusers-workflow/issues/1094)), and
finding phrase starts from the centre of the stereo mix
([#1095](https://github.com/dkackman/diffusers-workflow/issues/1095)).

### [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)

A ComfyUI node pack for refining faces in MiniMax H3 takes, under the MIT
licence. Two concepts are reimplemented in `crop_face_track`
([dw/tasks/face_track.py](../dw/tasks/face_track.py)): choosing which subject to
follow when several faces are visible (`select`), and staying lost after a lost
face track rather than jumping to another face. No code was adapted.

### [knoic/ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream)

A ComfyUI node pack for MiniMax H3 prefix streaming, under the MIT licence. It
inspired the texture-band measurement on `compare_outputs`
(`texture_bands`, [dw/texture_bands.py](../dw/texture_bands.py)): RMS energy by
wavelength band, to see whether a change in a chain adds or removes texture.
It also inspired `guide_noise`
([dw/pipeline_processors/h3_guide_steps.py](../dw/pipeline_processors/h3_guide_steps.py);
[#1166](https://github.com/dkackman/diffusers-workflow/issues/1166)), from its prefix
noise at `775be90`: a cosine-tapered noise on each guide, folded in as the difference of
a paired VAE encode, so the guide's hold loosens toward its start.

### [1113800942/comfy-h3md-director](https://github.com/1113800942/comfy-h3md-director)

H3MD Director, a ComfyUI node pack for directing long MiniMax H3 videos, under
AGPL-3.0-or-later, so only the concept was taken. It inspired the latent
statistics anchor: rescaling each later segment's video latent to the first
segment's per-channel mean and standard deviation before decode
(`latent_anchor`, `chain.anchor: "segment_1"`,
[dw/pipeline_processors/h3_anchor.py](../dw/pipeline_processors/h3_anchor.py);
[#1166](https://github.com/dkackman/diffusers-workflow/issues/1166)). The
diagnosis it answers, colour and contrast drifting from segment to segment of
an H3 chain, was described by Lumifel and by roycho87 (NerdNotes). It also
inspired repairing one segment of a finished chain in place, held to both of its
neighbours, and its rules: no tail hold on the last segment, a new seed for the
re-render, and the chain's own text encoder, length and sizes kept
([templates/minimax/repair-segment](../workflows/templates/minimax/repair-segment.json),
[dw/tasks/chain_repair.py](../dw/tasks/chain_repair.py);
[#1169](https://github.com/dkackman/diffusers-workflow/issues/1169)).

### ComfyUI-Continuity

Concept only, no code taken. It inspired `turbo_lead_in`
([dw/pipeline_processors/h3_lead_in.py](../dw/pipeline_processors/h3_lead_in.py);
[#1326](https://github.com/dkackman/diffusers-workflow/issues/1326)): running a
call's first denoising steps without the turbo (step-distilled) LoRA, so the base
model lays out the composition.

### [matlowai/h3-long-video-research](https://github.com/matlowai/h3-long-video-research)

A research repository on long MiniMax H3 videos, under CC BY 4.0. Its findings
informed the narrated take (`templates/minimax/narrated-take`), which dw built
and measured itself; no text or data was copied. The measurements dw reports are
its own, in
[bench/records/narrated.md](../bench/records/narrated.md).

### [roycho87/NerdNotes](https://github.com/roycho87/NerdNotes)

Tests 16-18 of the NerdNotes collection, under CC BY-SA, informed the narrated
take (`templates/minimax/narrated-take`), which dw rebuilt and measured in its
own code. No text or data was copied; because the licence is share-alike, only
the concepts were taken.

### [roycho87/geiru_context_suite](https://github.com/roycho87/geiru_context_suite)

A set of ComfyUI nodes for long MiniMax H3 videos, under GPL-3.0, so only the
concept was taken, together with seitanism's context hold. It inspired
`guide_backoff`
([dw/pipeline_processors/chain.py](../dw/pipeline_processors/chain.py);
[#1189](https://github.com/dkackman/diffusers-workflow/issues/1189)): handing the
next segment the frames one block before the end and dropping the previous
segment's last block at assembly, so the join lands inside the held copy. The
same suite's idea of a first-segment keyframe, and NerdNotes' (CC BY-SA, so
concepts only), inspired `keystone`
([#1187](https://github.com/dkackman/diffusers-workflow/issues/1187)): segment 1's
frame 0 as every later segment's keyframe, pulling the chain back toward its
clean start. Its Anchoring node, a source's head held in a shot's last frames
with its audio, gave the repair's tail hold and the rule that the tail hold
outnumbers the head (39 frames against 22 in
[templates/minimax/repair-segment](../workflows/templates/minimax/repair-segment.json);
[#1169](https://github.com/dkackman/diffusers-workflow/issues/1169)).

harlixay7/Kinoloom (MIT; concept only, no code taken) inspired `carry_pick`
([dw/pipeline_processors/chain_pick.py](../dw/pipeline_processors/chain_pick.py);
[#1032](https://github.com/dkackman/diffusers-workflow/issues/1032)): picking the
sharpest of a segment's last frames as the next keyframe rather than always the
last, and ending the segment there.

Smite79/MiniMax-H3-LongVideos (a custom no-redistribution licence; concepts only,
no code taken) inspired `look_hold`
([dw/pipeline_processors/chain_look.py](../dw/pipeline_processors/chain_look.py);
[#1191](https://github.com/dkackman/diffusers-workflow/issues/1191)): a quantile
grade of a segment's opening to the carried frame, an ease-out of contrast and
saturation creep, an ease-out of faint texture, and a keyframe pre-compensated for
the measured boost.

Smite79/MiniMax-H3-LongVideos (concepts only, under the licence above) and WanGP
(concept only, from [#984](https://github.com/dkackman/diffusers-workflow/issues/984))
inspired the H3 audio pass
([dw/pipeline_processors/h3_mask.py](../dw/pipeline_processors/h3_mask.py);
[#1192](https://github.com/dkackman/diffusers-workflow/issues/1192)): a finished
picture's video latents locked clean while only the audio is generated over it, and
the listen pass on a smaller canvas.

leonardofrito/Directors-Cut (MIT; concept and schema shape, no code taken)
inspired `export_timeline`
([dw/tasks/timeline_export.py](../dw/tasks/timeline_export.py);
[#1197](https://github.com/dkackman/diffusers-workflow/issues/1197)): a joined
piece written as an OpenTimelineIO `.otioz` bundle with one clip per shot, so an
editor opens it as a timeline.
