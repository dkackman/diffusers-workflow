# Catalog inventory, 2026-09-06

Verdicts for every workflow under `workflows/`. Produced by Task 2 of the
[restructure plan](2026-09-06-workflow-catalog-restructure.md); the file lists
in Tasks 3-5 are this table, filtered by verdict.

Status: AWAITING APPROVAL

Counts: COLLAPSE INTO 6, DELETE 42, MODEL 9, TEMPLATE 64 (total 121)

| Current path | Verdict | Destination | `id` | Why |
| --- | --- | --- | --- | --- |
| workflows/InterpolateFrames.json | TEMPLATE | templates/interpolate-frames.json | InterpolateFrames | Only RIFE interpolation demo |
| workflows/Krea2.json | MODEL | models/krea2.json | Krea2 | One-step t2i with Krea 2 Turbo |
| workflows/Krea2Edit.json | TEMPLATE | templates/image-edit.json | Krea2Edit | Identity-preserving edit via LoRA; needs a description written |
| workflows/ZImage.json | MODEL | models/z-image.json | ZImage | One-step t2i with Z-Image Turbo |
| workflows/ZImageSDNQ.json | MODEL | models/z-image-sdnq.json | ZImageSDNQ | Pre-quantized SDNQ uint4 - a VRAM fact |
| workflows/archive/Allegro.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/CogVideoX-5B-I2V.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/CogVideoX-5b.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/CogVideoX1.5-5B-I2V.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/CogVideoX1.5-5B.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/ErnieImageTurbo.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/HunyuanVideo.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/HunyuanVideoGguf.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Ideogram4.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Kolors.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Lumina.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Mochi.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/NucleusAI.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/QwenImageEdit.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Wan22I2V14B.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Wan22T2V14B.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/Wan22TI2V5B.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/WanI2V14.json | DELETE | — | — | Outpaint-then-animate; both halves have templates |
| workflows/archive/WanT2V1.3.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/WanT2V14.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/bnb_quant.json | DELETE | — | — | Shares the id sd35 with archive/sd35.json |
| workflows/archive/hunyuan15.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/img2vid.json | DELETE | — | — | Four checkpoints through one image-to-video shape |
| workflows/archive/ip-adapter.json | DELETE | — | — | templates/ip-adapter.json covers the mechanism |
| workflows/archive/kandinsky.json | DELETE | — | — | Kandinsky-specific prior + ControlNet chain |
| workflows/archive/kandinsky3.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/kandinsky_i2i.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/ltx.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/owl.json | DELETE | — | — | Workflow composition survives in templates/compose-workflows.json |
| workflows/archive/qr_code.json | TEMPLATE | templates/qr-code.json | qr_code | Unique task command, still linked from TASKS.md |
| workflows/archive/sana.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/sd15.json | DELETE | — | — | Duplicate of workflows/sd15.json, same id |
| workflows/archive/sd35.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/sd35ip.json | DELETE | — | — | Model-gallery entry; no feature the tree lacks |
| workflows/archive/sdxl.json | TEMPLATE | templates/base-and-refiner.json | sdxl | The two-stage base-plus-refiner pattern |
| workflows/archive/sdxl_refiner.json | DELETE | — | — | The refiner pattern survives in templates/base-and-refiner.json |
| workflows/archive/txt2img2vid.json | TEMPLATE | templates/compose-workflows.json | txt2img2vid | Composes any t2i with any i2v, both named as variables |
| workflows/attention_processors_example.json | TEMPLATE | templates/attention-processor.json | attention_processors_example | Only custom attention-processor demo |
| workflows/controlnet.json | TEMPLATE | templates/controlnet-component.json | sd15-controlnet | ControlNetModel loaded as a separate component, unlike FluxControlPipeline |
| workflows/flux/Flux2Dev.json | MODEL | models/flux2-dev.json | Flux2Dev | One-step t2i with FLUX.2 dev, 4-bit |
| workflows/flux/Flux2DevImageCombine.json | TEMPLATE | templates/multi-image-reference.json | Flux2DevImageCombine | Combining several reference images into one generation |
| workflows/flux/FluxCanny.json | TEMPLATE | templates/controlnet.json | FluxCanny | Preprocessor step feeding a control pipeline |
| workflows/flux/FluxDepth.json | COLLAPSE INTO | templates/controlnet.json | FluxDepth | Same shape as FluxCanny; only the preprocessor differs |
| workflows/flux/FluxDev.json | MODEL | models/flux-dev.json | FluxDev | One-step t2i with FLUX.1 dev |
| workflows/flux/FluxDevFast.json | MODEL | models/flux-dev-fast.json | FluxDevFast | 24GB tuning: int8 TorchAO resident + regional offload |
| workflows/flux/FluxDevFirstBlockCache.json | TEMPLATE | templates/step-caching.json | FluxDevFirstBlockCache | Skipping redundant transformer work across denoise steps |
| workflows/flux/FluxDevKrea.json | MODEL | models/flux-krea.json | FluxDevKrea | One-step t2i with the FLUX.1 Krea dev checkpoint |
| workflows/flux/FluxDevTeaCache.json | COLLAPSE INTO | templates/step-caching.json | FluxDevTeaCache | Same cache block, different type |
| workflows/flux/FluxFill.json | TEMPLATE | templates/inpaint.json | FluxFill | Only inpainting demo |
| workflows/flux/FluxGGUF.json | MODEL | models/flux-gguf.json | FluxGGUF | GGUF-quantized transformer checkpoint |
| workflows/flux/FluxIP.json | TEMPLATE | templates/ip-adapter.json | FluxIP | Only IP-Adapter demo outside archive |
| workflows/flux/FluxImg2Img.json | TEMPLATE | templates/image-to-image.json | FluxImg2Img | Only img2img demo |
| workflows/flux/FluxInContext.json | TEMPLATE | templates/lora-styles.json | FluxInContext | Ten adapters, one image each - per-step LoRA switching |
| workflows/flux/FluxLogo.json | TEMPLATE | templates/sub-workflow.json | FluxLogo | Delegates to another workflow, then post-processes |
| workflows/flux/FluxLora.json | TEMPLATE | templates/lora.json | FluxLora | The LoRA reference |
| workflows/flux/FluxOutpaint.json | TEMPLATE | templates/outpaint.json | FluxOutpaint | Border mask then fill - only outpaint demo |
| workflows/flux/FluxRFInversion.json | TEMPLATE | templates/community-pipeline.json | FluxRfInversion | Only community pipeline + pipeline_reference demo |
| workflows/flux/FluxRedux.json | TEMPLATE | templates/image-variation.json | FluxRedux | Prior redux encode then regenerate |
| workflows/flux/FluxSchnellWeighted.json | TEMPLATE | templates/prompt-weighting.json | FluxSchnellWeighted | Only A1111-style weighting demo |
| workflows/flux/FluxTorchAO.json | MODEL | models/flux-torchao.json | FluxTorchAO | int8 TorchAO quantization |
| workflows/flux/img2txt2img.json | TEMPLATE | templates/describe-and-regenerate.json | img2txt2img | VLM then LLM then generate, across sub-workflows |
| workflows/gyre/GyreAssemble.json | TEMPLATE | templates/assemble-and-score.json | GyreAssemble | Edit-only pass: stabilize, cut, mix a score under shot audio; needs generalizing off GYRE |
| workflows/gyre/GyreDissolve.json | TEMPLATE | templates/dissolve-between-shots.json | GyreDissolve | Only dissolve_videos user; needs generalizing off GYRE |
| workflows/gyre/GyreFilm.json | DELETE | — | — | Project-specific 24-step pass; SitcomShort teaches multi-shot generation |
| workflows/gyre/GyreFrames.json | TEMPLATE | templates/recenter-crop.json | GyreFrames | recenter_crop registration; needs generalizing off GYRE |
| workflows/gyre/GyreReshoot.json | DELETE | — | — | One-off repair of GYRE shot 4 |
| workflows/gyre/GyreStills.json | DELETE | — | — | Project-specific keyframe pass |
| workflows/gyre/GyreStillsFix.json | DELETE | — | — | One-off repair of two GYRE stills |
| workflows/gyre/GyreStillsFix2.json | DELETE | — | — | Third attempt at GYRE's second shot |
| workflows/lora.json | COLLAPSE INTO | templates/lora.json | sd35_lora | Same LoRA mechanism as FluxLora, different base |
| workflows/ltx2/LTX2.json | TEMPLATE | templates/ltx2/text-to-video.json | LTX2 | LTX-2.5 baseline: video with a soundtrack on one 24GB card |
| workflows/ltx2/LTX2Extend.json | TEMPLATE | templates/ltx2/extend-clip.json | LTX2Extend | Conditioning on a full clip to carry motion forward |
| workflows/ltx2/LTX2I2V.json | TEMPLATE | templates/ltx2/image-to-video.json | LTX2I2V | Still becomes the first frame |
| workflows/ltx2/LTX2I2VChained.json | TEMPLATE | templates/ltx2/chained-segments.json | LTX2I2VChained | Long clip from short ones, frames and audio stitched |
| workflows/ltx2/LTX2I2VEnhancePrompt.json | TEMPLATE | templates/ltx2/enhance-prompt.json | LTX2I2VEnhancePrompt | LTX-2.5's own enhancer plus the duration head |
| workflows/ltx2/LTX2ICLora.json | TEMPLATE | templates/ltx2/generative-upscale.json | LTX2ICLora | In-context LoRA re-renders at 2x, inventing detail |
| workflows/ltx2/LTX2Keyframes.json | TEMPLATE | templates/ltx2/keyframes.json | LTX2Keyframes | Pinning first and last frames as conditions |
| workflows/ltx2/LTX2TwoStage.json | TEMPLATE | templates/ltx2/two-stage.json | LTX2TwoStage | Render small then upscale in latent space |
| workflows/minimax/MiniMaxH3.json | TEMPLATE | templates/minimax/video-with-audio.json | MiniMaxH3 | H3 baseline every other H3 example builds on |
| workflows/minimax/MiniMaxH3EnhancePrompt.json | TEMPLATE | templates/minimax/enhance-prompt.json | MiniMaxH3EnhancePrompt | Sub-workflow output feeding the next step's argument |
| workflows/minimax/MiniMaxH3FL2VA.json | TEMPLATE | templates/minimax/first-and-last-frame.json | MiniMaxH3FL2VA | Both ends pinned, motion interpolated |
| workflows/minimax/MiniMaxH3GeneratedVoice.json | TEMPLATE | templates/minimax/voice-timbre-reference.json | MiniMaxH3GeneratedVoice | Generated speech as an <Audio 1> timbre reference |
| workflows/minimax/MiniMaxH3I2V.json | TEMPLATE | templates/minimax/image-to-video.json | MiniMaxH3I2V | First frame pinned |
| workflows/minimax/MiniMaxH3I2VChained.json | TEMPLATE | templates/minimax/chained-segments.json | MiniMaxH3I2VChained | Fixed segment count with last_frame continuity |
| workflows/minimax/MiniMaxH3I2VEnhancePrompt.json | TEMPLATE | templates/minimax/enhance-prompt-with-image.json | MiniMaxH3I2VEnhancePrompt | Enhancer sees the same still the pipeline pins |
| workflows/minimax/MiniMaxH3L2V.json | TEMPLATE | templates/minimax/last-frame-only.json | MiniMaxH3L2V | Final frame pinned, everything before it invented |
| workflows/minimax/MiniMaxH3MusicVideo.json | TEMPLATE | templates/minimax/music-video.json | MiniMaxH3MusicVideo | Cuts against a soundtrack that never touches a chain |
| workflows/minimax/MiniMaxH3Ref2VA.json | TEMPLATE | templates/minimax/reference-to-video.json | MiniMaxH3Ref2VA | Identity conditioning via typed reference objects |
| workflows/minimax/MiniMaxH3Ref2VAChained.json | TEMPLATE | templates/minimax/chain-matched-to-audio.json | MiniMaxH3Ref2VAChained | match_audio sizes the chain from a supplied track |
| workflows/minimax/MiniMaxH3Ref2VAChainedAligned.json | TEMPLATE | templates/minimax/chain-matched-and-aligned.json | MiniMaxH3Ref2VAChainedAligned | match_audio and last_segment used together |
| workflows/minimax/MiniMaxH3Ref2VAChainedVideo.json | TEMPLATE | templates/minimax/chain-video-continuity.json | MiniMaxH3Ref2VAChainedVideo | last_segment hands over a video tail, not one frame |
| workflows/minimax/MiniMaxH3Ref2VAGeneratedSubject.json | TEMPLATE | templates/minimax/generated-subject-reference.json | MiniMaxH3Ref2VAGeneratedSubject | from_previous_result builds a reference from a generated still |
| workflows/minimax/MiniMaxH3Ref2VAVideo.json | TEMPLATE | templates/minimax/composable-references.json | MiniMaxH3Ref2VAVideo | Image supplies the subject, video supplies the shot |
| workflows/minimax/MiniMaxH3SitcomShort.json | TEMPLATE | templates/minimax/dialogue-short.json | MiniMaxH3SitcomShort | Multi-shot cut sequence with per-shot references |
| workflows/minimax/MiniMaxH3Storyboard.json | TEMPLATE | templates/minimax/storyboard.json | MiniMaxH3Storyboard | Several stills in one generation, each with a stated role |
| workflows/minimax/MiniMaxMusic.json | TEMPLATE | templates/minimax/music.json | MiniMaxMusic | Audio-only output from a pipeline |
| workflows/sd15.json | TEMPLATE | templates/text-to-image.json | test_job | Ungated baseline - the no-login quickstart CLAUDE.md points at |
| workflows/tasks/CaptionToImage.json | DELETE | — | — | img2txt2img is the richer caption-then-generate demo |
| workflows/tasks/DiffusionUpscale.json | DELETE | — | — | Generate-then-upscale is previous_result chaining, shown everywhere |
| workflows/tasks/DiffusionUpscaleImage.json | TEMPLATE | templates/upscale-diffusion.json | DiffusionUpscaleImage | Upscales any existing image, no generation step |
| workflows/tasks/ExpandAndGenerate.json | COLLAPSE INTO | templates/expand-prompt.json | ExpandAndGenerate | Expand-then-generate is the same task plus a generation step |
| workflows/tasks/ExpandPrompt.json | TEMPLATE | templates/expand-prompt.json | ExpandPrompt | LLM prompt expansion, text out |
| workflows/tasks/FaceRestore.json | TEMPLATE | templates/restore-faces.json | FaceRestore | Only face restoration demo |
| workflows/tasks/GenerateSpeech.json | TEMPLATE | templates/generate-speech.json | GenerateSpeech | Only text-to-speech demo |
| workflows/tasks/ImageToText.json | TEMPLATE | templates/image-to-text.json | ImageToText | Captioning with the small default |
| workflows/tasks/ImageToTextVLM.json | COLLAPSE INTO | templates/image-to-text.json | ImageToTextVLM | Same task, a VLM and a specific question |
| workflows/tasks/MarigoldDepth.json | TEMPLATE | templates/depth-marigold.json | MarigoldDepth | Depth maps with Marigold |
| workflows/tasks/MarigoldNormals.json | TEMPLATE | templates/surface-normals.json | MarigoldNormals | Surface normals - a different output, different pipeline class |
| workflows/tasks/MetadataEmbed.json | TEMPLATE | templates/embed-metadata.json | MetadataEmbed | Only metadata-embedding demo |
| workflows/tasks/Segment.json | TEMPLATE | templates/segment.json | Segment | Text-prompted segmentation |
| workflows/tasks/SegmentAndInpaint.json | TEMPLATE | templates/segment-and-inpaint.json | SegmentAndInpaint | Segment then inpaint the selected region |
| workflows/tasks/SpandrelUpscale.json | DELETE | — | — | Generate-then-upscale; UpscaleImage covers the task |
| workflows/tasks/TrimFadeAudio.json | TEMPLATE | templates/audio-trim-fade.json | TrimFadeAudio | Only audio finishing-pass demo |
| workflows/tasks/UpscaleImage.json | TEMPLATE | templates/upscale-spandrel.json | UpscaleImage | Upscales any existing image, no generation step |
| workflows/tasks/face-swap.json | COLLAPSE INTO | templates/ip-adapter.json | face_swap | IP-Adapter with a face model - same mechanism |
| workflows/tasks/image_processors.json | TEMPLATE | templates/image-processors.json | image_processors | Every preprocessor side by side - the preprocessor catalog |

## Judgement calls worth a second look

- **`templates/text-to-image.json` inherits the id `test_job`** from
  `workflows/sd15.json`. The plan forbids changing an id, so it carries a poor
  name into a prominent file. Overriding that constraint for this one file is
  reasonable if you would rather it read `text-to-image`.
- **The one-step text-to-image models become `models/` entries** - `flux-dev`,
  `krea2`, `z-image`, `flux2-dev`, `flux-krea`. The collapse rule stops at a
  pipeline class boundary and these are four different classes, so they cannot
  merge into the template; as per-checkpoint setups they are model configs.
- **`gyre/` keeps three templates that still name GYRE's own shots.** They need
  generalizing as part of Task 3, not just moving.
- **`Krea2Edit.json` has no description at all** and needs one written before it
  can pass the template test.
- **`archive/sdxl.json` graduates but `archive/sdxl_refiner.json` does not** - the
  two-stage file demonstrates the pattern, the standalone refiner is a subset.
