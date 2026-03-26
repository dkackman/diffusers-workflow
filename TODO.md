# diffusers-workflow TODO

Utilities and enhancements from the stable diffusion ecosystem.

## Post-Processing & Enhancement

- [ ] **CCSR / StableSR** — Diffusion-based upscalers with better detail than ESRGAN-family, especially for faces and textures. New task type alongside existing Spandrel upscaler.
- [ ] **Real-ESRGAN Video** — Frame-consistent video upscaling with temporal smoothing. Current upscaler is image-only.
- [x] **RIFE / FILM frame interpolation** — Generate intermediate frames for smoother video output. Video post-processing task.

## Image Preprocessing & Conditioning

- [x] **Marigold depth / DSINE normals** — Newer, more accurate depth/normal estimators than MiDaS/DPT. Better ControlNet conditioning maps.
- [x] **GroundingDINO + SAM2** — Text-prompted object detection to segmentation. "Segment the dog" as a task, producing masks for inpainting workflows.
- [x] **Florence-2** — Microsoft vision-language model for captioning, detection, segmentation. Powers an `auto_caption` task for img2img or IP-Adapter workflows.
- [ ] **PuLID / InstantID** — Identity-preserving face conditioning (better than IP-Adapter for faces). Works with Flux and SDXL.

## Video-Specific

- [ ] **FramePack** — Context-aware video generation with efficient memory usage for long video generation.
- [ ] **PySceneDetect keyframe extraction** — Smarter than `get_frame` for selecting keyframes from input video based on scene detection.
- [x] **RIFE/IFRNet optical flow interpolation** — Optical flow-based frame interpolation as a post-processing step.

## Workflow Utilities

- [x] **Prompt expansion via local LLM** — Task that takes a short prompt and expands it using a small language model (Llama 3.2 1B, Qwen2.5, etc.).
- [ ] **Image comparison / SSIM / LPIPS scoring** — Task that scores similarity between images for iterative refinement workflows.
- [ ] **Color palette extraction / transfer** — Extract dominant colors from a reference image or apply color grading from one image to another.
- [ ] **Tiled generation / outpainting helper** — Automate the border+mask+generation loop for progressive outpainting.

## Model Management

- [ ] **Automatic VRAM estimation** — Given a workflow JSON, estimate peak VRAM before running. Helps pick the right quantization/offload settings.
- [ ] **Model predownload/warmup** — Dry-run mode that downloads all models without executing. Useful for deployment.

## Quick Wins

- [x] **Image metadata embedding** — Store generation params in PNG info chunks for reproducibility.
- [ ] **EXIF stripping** on input images — Privacy-safe preprocessing.
- [ ] **Image hashing** (perceptual hash) — Dedup detection across workflow runs.
- [ ] **Watermark embedding/detection** — Responsible AI compliance.
- [ ] **Aspect ratio bucketing** — Auto-resize inputs to model-native aspect ratios.

## Architectural Enhancements

- [ ] **Conditional branching** — e.g., "if image has faces, run face restore; otherwise skip." Enables more sophisticated pipelines.
- [ ] **Parallel step execution** — Steps with no data dependencies run concurrently. Matters for multi-GPU setups.
