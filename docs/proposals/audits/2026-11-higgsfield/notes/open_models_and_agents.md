# Open-weights models, LoRAs and agents for Higgsfield-style cinematic controls (as of 2026-11-01)

Scope note: about 17 searches and fetches. Primary sources (Hugging Face model listings and cards, GitHub, arXiv, docs.ltx.io) are preferred. Where only aggregators or vendor blogs were found, that is said. Anything I know only from training data, and could not re-verify this session, is in **Inferences** or **Gaps** and marked "(unverified)". The dw engine's supported families are MiniMax H3, LTX-2.5 (22B, `Lightricks/LTX-2.5-Diffusers`), Kandinsky 6 and FLUX 3 Klein. The repo already catalogues these LTX-2.5 IC-LoRAs: Pixel-Spatial-Upscaler, Deblur, Ingredients, Decompression, Restore, Refine-Details, Colorization and Alpha-Gen (grep of `dw/`, `workflows/` and `plugins/`).

## Q1. Camera motion control in video (dolly, orbit, crane, crash zoom), and which engine families it fits

### Takeaway
Lightricks publishes the only first-party, named-preset camera LoRAs. These are seven moves: dolly in, dolly out, dolly left, dolly right, jib up, jib down and static. **They target LTX-2 19B, not LTX-2.5 22B**, and I found no LTX-2.5 camera-preset LoRA. On LTX-2.5, camera moves come from the prompt: the trained caption spec requires a stated camera move. Orbit and crash-zoom presets exist only as Wan 2.1 community effect LoRAs. Trajectory-conditioned methods (ReCamMaster, Uni3C, and older ones like CameraCtrl and MotionCtrl) are Wan- or SVD-based and would need a new base model plus custom loaders.

### Cited Findings
- The Lightricks Hugging Face listing (sorted by last modified, all re-uploaded 2026-10-22) has these camera LoRAs: `Lightricks/LTX-2-19b-LoRA-Camera-Control-{Dolly-In, Dolly-Out, Dolly-Left, Dolly-Right, Jib-Up, Jib-Down, Static}`. LTX-2 19B control IC-LoRAs: `LTX-2-19b-IC-LoRA-{Union-Control, Pose-Control, Depth-Control, Canny-Control, Detailer}`. LTX-2.3 22B IC-LoRAs: Union-Control, Motion-Track-Control, Relight, In-Outpainting, HDR, DubIt, Clean-Plate, Day-To-Night, Water-Simulation, Instant-Shave, Cross-Eyed and others, plus a `LTX-2.3-22b-LoRA-Foley-V2A`. LTX-2.5 22B: IC-LoRAs Layout-To-Render, Day-To-Night, Clean-Plate, Water-Simulation, SDR-To-HDR and the restoration set, plus plain LoRAs **Slow-Motion-Control** and **Cinemagraph**. **There is no LTX-2.5 or LTX-2.3 camera-preset LoRA in the listing.** — [HF API listing, author=Lightricks](https://huggingface.co/api/models?author=Lightricks&search=LoRA&limit=100&sort=lastModified)
- The Dolly-In card says the LoRA "biases outputs toward a forward push-in". It loads through LTX-2's two-stage HQ pipeline at strength 1.0, alongside the base checkpoint, a distilled LoRA and a Gemma 3 text encoder. — [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In); [Dolly-Out](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out)
- The Dolly-Left card advises naming in the prompt what should come into view during the move. — [Dolly-Left mirror](https://huggingface.co/Galatea777/LTX-2-19b-LoRA-Camera-Control-Dolly-Left)
- Lightricks' VFX product page lists "seven pre-trained camera LoRAs (dolly in/out/left/right, jib up/down, static)" under LTX-2.3, plus Union Control and Motion Track Control IC-LoRAs. This conflicts with the HF repo names, which say LTX-2-19b, so it is unclear which base the camera LoRAs are validated on. — [ltx.io VFX capabilities](https://ltx.io/model/capabilities/vfx)
- LTX-2.5 was released on 2026-08-11. Camera moves on 2.5 are mainly prompt-driven: the HF examples include a dolly-out prompt. Official ComfyUI examples cover union control and IC-LoRA workflows, including motion tracking. One aggregator says some LTX-2.3 LoRAs "partly work" with 2.5, with variable results. — [gradually.ai](https://www.gradually.ai/en/ai-models/ltx-2-5/); [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5); [comfyui-wiki LTX-2.5](https://comfyui-wiki.com/es/models/ltx-video/ltx-2-5); [aireiter (aggregator)](https://aireiter.com/es/blog/ltx-2-5-api-pricing-guide)
- RunComfy's hosted LTX 2.5 Pro image-to-video API exposes an optional "camera-motion preset". That is a hosted-API feature, not a published open LoRA. — [RunComfy](https://www.runcomfy.com/zh-CN/models/lightricks/ltx-2.5/image-to-video/pro/api)
- The dw repo's LTX-2.5 caption spec already demands "Camera motion (always stated; if none, explicitly say the camera remains static)" and a viewpoint (front-facing, over-the-shoulder, top-down, low-angle and so on), written as flowing prose. The Kandinsky 6 skill demands "exactly one camera sentence ('The camera is static.' when none is asked for)". — `plugins/dw/skills/ltx-2.5/references/caption-spec.md` lines 25-27; `plugins/dw/skills/kandinsky-6/SKILL.md` lines 77-78 (local repo)
- Remade-AI publishes Wan 2.1 14B I2V 480p effect LoRAs for camera moves, each with a trigger phrase:
  - **360 Orbit**: trigger "0rb4it 360 degree orbit". Strength 1.0, guidance 6.0, flow shift 5.0. Trained on about 50 s of video from 10 clips, for 30 epochs.
  - **Crash Zoom In**: trigger "cr34sh crash zoom in effect".
  - **Crash Zoom Out** and **Rotate**: listed, but no details found.

  The org title says T2V, but the individual cards say I2V. — [Remade-AI org](https://huggingface.co/Remade-AI); [360-Orbit mirror](https://cnb.cool/ai-models/Remade-AI/360-Orbit); [Crash-zoom-in mirror](https://cnb.cool/ai-models/Remade-AI/Crash-zoom-in); [I2V collection](https://huggingface.co/collections/Remade-AI/wan21-14b-480p-i2v-loras)
- Uni3C (ewrfcas/Uni3C) is a "unified 3D-enhanced framework for precise control of both camera and human motion". It has a pose-extraction step and an inference step. It is used in ComfyUI to transfer camera motion from a reference video. — [cnb mirror](https://cnb.cool/wan2222222/uni3cv2); [RunComfy Uni3C workflow](https://www.runcomfy.com/comfyui-workflows/uni3c-comfyui-workflow-video-referenced-camera-motion-transfer)

### Inferences
- **Named presets:** two sets of LoRAs are trained on camera vocabulary and can be exposed directly as presets. The Lightricks set has dolly in, dolly out, dolly left, dolly right, jib/crane up, jib/crane down and static. The Remade set has orbit 360, crash zoom in, crash zoom out and rotate, each with a trigger phrase. Neither targets LTX-2.5. For dw, the cheapest path is **prompt presets**: a named move maps to a canonical caption sentence in the LTX-2.5 and Kandinsky 6 skill caption specs, which already require one camera statement. LoRA-backed presets would come later, if Lightricks ships 2.5 camera LoRAs or a 2.3 camera LoRA is shown to load on 2.5.
- The LTX-2.3 and LTX-2.5 IC-LoRA family uses the same mechanism that dw already loads for the LTX-2.5 restoration set. So Motion-Track-Control and Union-Control, if ported to 2.5, would be the lowest-friction route to trajectory and reference-video camera control inside the engine.
- Uni3C, ReCamMaster and the Remade LoRAs all need Wan 2.1 or 2.2 as a new base family. Wan has native diffusers pipelines (WanPipeline, WanImageToVideoPipeline, WanVACEPipeline, and WanAnimatePipeline per the PR cited in Q4), but the camera-conditioning paths of ReCamMaster and Uni3C are not in diffusers.
- CameraCtrl (AnimateDiff and SVD), MotionCtrl (VideoCrafter and SVD) and AC3D are older U-Net or CogVideoX-era methods. As of late 2026 they are superseded by Wan-based methods and by first-party LoRAs (unverified, from prior knowledge).

### Gaps
- No camera-control LoRA for LTX-2.5 was found, and no statement that the LTX-2-19b camera LoRAs load on 2.5. docs.ltx.io's camera-control page returned 404.
- I could not verify licences for the LTX-2 camera LoRAs this session. They are presumably under the LTX-2 community licence, which in earlier versions allowed commercial use below a revenue threshold (unverified). Remade-AI licences are also unverified.
- No camera-control LoRA or adapter was found for MiniMax H3, Kandinsky 6 or FLUX 3 Klein. H3 camera control appears to be prompt-only. The dw staging notes say "a moving camera weren't tested".
- CameraCtrl, MotionCtrl, TrajectoryAttention and AC3D licences and status were not re-verified.

## Q2. Re-shooting a video from a new angle (novel-view video)

### Takeaway
The leading open options are ReCamMaster (KlingTeam, ported to Wan 2.1) and TrajectoryCrafter (Tencent ARC, ICCV 2025). Both are re-render-from-source-video models with preset or explicit trajectories. Neither has a diffusers pipeline, and both need a new base (Wan 2.1, or CogVideoX-based for TrajectoryCrafter, unverified) and a custom loader.

### Cited Findings
- ReCamMaster (ICCV'25 Oral, Best Paper Finalist) is "camera-controlled generative rendering from a single video". The paper's model is an internal Kuaishou T2V model that "can't be open-sourced". The released checkpoint is a port to Wan2.1, at `KlingTeam/ReCamMaster-Wan2.1`, and its results may not match the paper's. — [GitHub KlingAIResearch/ReCamMaster](https://github.com/KlingAIResearch/ReCamMaster); [README](https://raw.githubusercontent.com/KlingAIResearch/ReCamMaster/fcf98bc86e876bb534518cd99e8a65b282f0f16e/README.md); [HF KlingTeam/ReCamMaster-Wan2.1](https://huggingface.co/KlingTeam/ReCamMaster-Wan2.1)
- A ComfyUI workflow host describes a 1.3B ReCamMaster on the Wan 2.1 architecture. This comes from a marketing page. — [floyo.ai](https://www.floyo.ai/workflows/wan-2-1-recammaster-v2v-camera-contr-f9y6ufftq2i6)
- TrajectoryCrafter (Mark Yu, Wenbo Hu, Jinbo Xing, Ying Shan; Tencent ARC and CUHK; ICCV 2025) generates "high-fidelity novel views from casually captured monocular video" with precise pose control. The repo recommends GPUs with at least 28 GB of VRAM. — [arXiv 2503.05638](https://arxiv.org/html/2503.05638v1); [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_TrajectoryCrafter_Redirecting_Camera_Trajectory_for_Monocular_Videos_via_Diffusion_Models_ICCV_2025_paper.pdf)
- ViewCrafter, the same group's earlier static-scene novel-view model, is Apache-2.0 on HF. — [Drexubery/ViewCrafter_25](https://huggingface.co/Drexubery/ViewCrafter_25/raw/35af104d9781e9ed5cdaddfcfbbdca5835e1c36f/README.md)

### Inferences
- In ReCamMaster, the camera is chosen from a fixed set of preset trajectories (pan, arc, translate, zoom and so on, per the README, unverified in detail). That maps naturally onto named "re-shoot" presets.
- Neither model fits the dw engine's current families. Each would be a new pipeline class wrapping a Wan 2.1 transformer (ReCamMaster), or a point-cloud render plus a video-diffusion stage (TrajectoryCrafter), and that is a custom loader.
- The LTX-2.3 Motion-Track-Control IC-LoRA (Q1) is the nearest LTX-family substitute, though it is not novel-view re-rendering.

### Gaps
- Licences for ReCamMaster (code and weights) and TrajectoryCrafter were not confirmed. The official TrajectoryCrafter repo page was not returned.
- No 2026 successor to either was found in this session, and no newer novel-view video model was checked (for example Wan 2.2-based or LTX-based ones).

## Q3. Multi-angle stills from one image ("Shots: 9 angles")

### Takeaway
The open standard is a Qwen-Image-Edit multi-angle LoRA: fal's 96-position LoRA for the 2511 base, or dx8152's for the 2509 base. Both are prompt-driven ("Rotate the camera 45 degrees to the left"), so they map directly to named angle presets. Qwen-Image-Edit is a native diffusers pipeline (QwenImageEditPlusPipeline). dw already ran Qwen in the 2026-09-20 field report, so this is the lowest-cost add. No equivalent LoRA for FLUX 3 Klein was found.

### Cited Findings
- `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` is a "multi-angle camera control LoRA" with 96 camera positions, trained on more than 3000 Gaussian-splatting renders. It ships a safetensors file and a ComfyUI workflow, with a suggested strength of 0.9. — [HF fal](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA)
- `dx8152/Qwen-Edit-2509-Multiple-angles` (weight file `镜头转换.safetensors`) is wrapped by HF Spaces. Those Spaces build the prompt from rotation, zoom, tilt and wide-angle controls, with positive values rotating left and negative rotating right. — [linoyts Space app.py](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles/blob/bbbeff133c3a7714b98a8ab99a61895294dfb4cf/app.py); [rideshare Space](https://huggingface.co/spaces/rideshare/Qwen-Image-Edit-Angles/blob/main/app.py)
- Replicate's wrapper uses ±180° rotation and a `multiple_angles_strength` of 0.8-1.3. — [Replicate qwen-edit-multiangle](https://replicate.com/qwen/qwen-edit-multiangle/readme)
- A community Rapid-AIO merge variant is tested with "Rotate the camera 45 degrees to the left." — [Manojb/Qwen-Image-Edit-Rapid-AIO-MultipleAngle](https://huggingface.co/Manojb/Qwen-Image-Edit-Rapid-AIO-MultipleAngle)

### Inferences
- A "9 angles" preset can be a fan-out over one Qwen-Image-Edit plus multi-angle LoRA step, using dw's list-driven steps, with nine fixed prompt strings. Examples: front, ±45° rotate, ±90°, high angle, low angle, close-up and wide. The Space's rotate, zoom and tilt prompt grammar is the vocabulary to pin.
- FLUX 3 Klein multi-reference editing (in dw already) can approximate angle changes from prompting, but it has no trained camera-pose vocabulary that I found.
- MV-Adapter (SDXL and SD2.1, Apache-2.0, diffusers-based custom pipeline; unverified this session) produces orthographic turnaround views. It is better for character sheets than for cinematic angles.

### Gaps
- Licences of the fal and dx8152 LoRAs were not checked. The Qwen-Image-Edit base is Apache-2.0 (unverified this session).
- No multi-angle LoRA for FLUX 3 Klein or Kandinsky was found.

## Q4. Character or actor swap in video, and identity consistency

### Takeaway
Wan2.2-Animate-14B is the strongest open character-replacement model. Its animation and replacement modes have a diffusers `WanAnimatePipeline`, but the pose, mask and face preprocessing is outside diffusers. Identity-reference options (VACE, Phantom, MAGREF, Stand-In) are all Wan 2.1-based. Within dw's current families, identity holding comes from LTX-2.5's Ingredients IC-LoRA (already catalogued) and H3's Ref2VA reference mode.

### Cited Findings
- Diffusers PR #12442 adds `WanAnimatePipeline`, loading `Wan-AI/Wan2.2-Animate-14B-Diffusers`. It has Animation Mode (animates a character image with motion from a reference video) and Replacement Mode (replaces a character in a background video). Replacement needs pose video and other extracted inputs. The PR notes "integration of these preprocessing steps into Diffusers is planned for a future release". — [diffusers PR #12442 diff](https://github.com/huggingface/diffusers/pull/12442.diff); [issue #12441](https://github.com/huggingface/diffusers/issues/12441)
- The official card's replacement mode uses `generate.py --replace_flag` plus a relighting LoRA, and the card lists diffusers integration. — [HF Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/wan-ai/wan2.2-animate-14b)
- The ComfyUI version uses a 14B FP8 checkpoint, a SAM2 character mask and DWPose extraction. — [comfy.org workflow](https://comfy.org/hub/workflows/wan2-2-animate-character-animation-and-replacement-c2c4b45b7df5/)
- Wan 2.2 is Apache-2.0, released July 2025. TI2V-5B runs 720p at 24 fps on 8 GB of VRAM. — [MindStudio (aggregator)](https://www.mindstudio.ai/blog/what-is-wan-2-2-video-open-source)
- **Stand-In** is a Wan 2.1 identity LoRA that locks identity from a single image. **Phantom** (ByteDance) is built on Wan 2.1 in 14B and 1.3B sizes and handles cross-modal subject consistency. **MAGREF** generates from multiple reference images, with a Wan2.1 I2V 14B base. **VACE** is Wan's all-in-one editing module (inject subjects, animate, transfer motion). — [RunComfy Stand-In](https://www.runcomfy.com/comfyui-workflows/wan2-1-stand-in-in-comfyui-character-consistent-video-workflow); [comfyui-wiki Phantom](https://comfyui-wiki.com/en/models/wan/phantom); [ThinkDiffusion MAGREF](https://learn.thinkdiffusion.com/magref-generate-ai-videos-with-multiple-people-and-objects-from-images/); [Wan2GP VACE doc](https://huggingface.co/spaces/attong39/Wan2GP/blob/main/docs/VACE.md)
- The repo's own LTX-2.5 catalogue already includes `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, its subject-reference adapter. — local grep

### Inferences
- An actor swap needs a Wan family plus a preprocessing toolchain (DWPose, SAM2), and that toolchain would be new utility tasks in dw. The diffusers pipeline covers only the diffusion stage.
- Wan VACE has a native diffusers `WanVACEPipeline` (unverified this session). Phantom, MAGREF and Stand-In need custom loaders.

### Gaps
- Merge status of PR #12442 at Nov 2026 was not confirmed, though it very likely merged in late 2025 (unverified).
- Licences of Phantom, MAGREF and Stand-In were not found in this session's results. Wan2.2-Animate VRAM was not measured.

## Q5. Lip sync, talking heads and dialogue

### Takeaway
For dubbing an existing video, LatentSync 1.6 (ByteDance) is the default with a clear commercial path: Apache-2.0 code and OpenRAIL++ weights, about 18 GB of VRAM. MuseTalk (MIT) is the lightweight option. For audio-driven generation of long talking or multi-person video, InfiniteTalk and MultiTalk (MeiGen, Wan-based) lead, and LongCat-Video-Avatar 1.5 (MIT) is a newer option. dw already has native audio-video families (H3 dialogue, LTX-2.5, Kandinsky 6) that generate speech and lips jointly. That is the closest match to Higgsfield's "speak" and should come first. LTX-2.3's DubIt IC-LoRA is the in-family lip-redub option.

### Cited Findings
- From the 2026-09-16 licence comparison:
  - **LatentSync**: Apache-2.0 code, commercial use permitted. 8 GB VRAM for v1.5 and 18 GB for v1.6. v1.6 is trained at 512×512. 25 fps video, 16 kHz audio.
  - **MuseTalk**: MIT, commercial use permitted. 4 GB minimum (fp16), 256×256 face region.
  - **Wav2Lip**: commercial use "strictly prohibited".
  - **SadTalker**: Apache-2.0 with third-party carve-outs.
  - **LongCat-Video-Avatar 1.5**: MIT. Whisper-Large audio encoder, 8-step DMD2 distillation, 480p or 720p.

  — [Creative AI News, 2026-09-16](https://www.creativeainews.com/articles/open-source-lip-sync-licenses-commercial-use-2026/)
- LatentSync weights are labelled OpenRAIL++ in HF metadata and the code is Apache. A Civitai page shows CreativeML Open RAIL-M text, which conflicts and is likely a mislabel. — [HF discussion](https://huggingface.co/chunyu-li/LatentSync/discussions/2); [GitHub bytedance/LatentSync](https://github.com/bytedance/LatentSync); [Civitai (conflicting)](https://civitai.com/models/license/2409072)
- InfiniteTalk (MeiGen-AI) is a "sparse-frame video dubbing" framework for unlimited length. It runs on a single GPU, on multiple GPUs, or in low-VRAM setups. It claims better lip sync than MultiTalk and fewer hand and body distortions. These are the project's own claims. — [sourcepulse](https://www.sourcepulse.org/projects/12893512); [knightli (2026-06)](https://knightli.com/es/2026/06/15/infinitetalk-audio-driven-video-generation/)
- Lightricks lists `LTX-2.3-22b-IC-LoRA-DubIt` and `LTX-2.3-22b-LoRA-Foley-V2A`. — [HF listing](https://huggingface.co/api/models?author=Lightricks&search=LoRA&limit=100&sort=lastModified)

### Inferences
- InfiniteTalk and MultiTalk are built on Wan 2.1 I2V 14B with an audio cross-attention adapter (unverified this session). They need the Wan family plus a custom loader, and InfiniteTalk's licence should be checked (Apache-2.0 per prior knowledge, unverified).
- Hallo3 (CogVideoX-based) and Sonic (Tencent) are, to my knowledge, research or non-commercial licences (unverified). Flag them as non-commercial until checked.
- LatentSync is a standalone U-Net latent-diffusion pipeline, not diffusers-native. It could be a dw utility task, because it is post-processing on an existing clip.

### Gaps
- No licence was found this session for InfiniteTalk, MultiTalk, Hallo3, Sonic, FantasyTalking, OmniAvatar or Wan2.2-S2V.
- The DubIt IC-LoRA card was not read, so its function (lip re-dub to new audio?) and whether it is validated on LTX-2.5 are unconfirmed.

## Q6. VFX effect presets (explosion, disintegrate and so on)

### Takeaway
Effect LoRAs are a community ecosystem centred on Wan 2.1 and 2.2 I2V: Remade-AI's set (orbit, crash zoom, rotate and others) and a few Apache-2.0 one-offs. Older Lightricks LTX-Video effect LoRAs (Cakeify, Squish) target the deprecated LTX-Video 0.9 base. On LTX-2.5, Lightricks' own "effect" adapters are utility-grade (Slow-Motion-Control, Cinemagraph, Water-Simulation, Day-To-Night, Clean-Plate), not Higgsfield-style spectacle presets.

### Cited Findings
- `Ashmotv/exploded_effect_wan` is an Apache-2.0 Wan2.2-I2V-A14B LoRA with trigger "expl0ded effect". It is an exploded-diagram disassembly effect, not a pyrotechnic explosion. — [HF README](https://huggingface.co/Ashmotv/exploded_effect_wan/blob/main/README.md)
- Remade-AI publishes a collection of Wan 2.1 14B 480p I2V effect LoRAs with trigger phrases. — [Remade-AI collection](https://huggingface.co/collections/Remade-AI/wan21-14b-480p-i2v-loras)
- Lightricks: `LTX-Video-Cakeify-LoRA` and `LTX-Video-Squish-LoRA` (2025-03, LTX-Video 0.9 base). LTX-2.5 22B adapters include `LoRA-Slow-Motion-Control`, `LoRA-Cinemagraph`, `IC-LoRA-Water-Simulation`, `IC-LoRA-Day-To-Night` and `IC-LoRA-Clean-Plate` (2026-10-22). — [HF listing](https://huggingface.co/api/models?author=Lightricks&search=LoRA&limit=100&sort=lastModified)
- A dev.to post claims Higgsfield launched Wan 2.2. That is contradicted by the primary attribution to Alibaba, so treat it as unreliable. — [dev.to](https://dev.to/ytosko/higgsfield-ai-launches-wan-22-its-most-advanced-open-source-model-for-video-amp-image-3373)

### Inferences
- The LTX-2.5 Slow-Motion-Control, Cinemagraph, Water-Simulation and Day-To-Night LoRAs are immediately usable in dw: same base and same loader path as the catalogued IC-LoRAs. They are good first "effect presets" for dw.
- Higgsfield-style spectacle effects (explode, disintegrate, melt, inflate, levitate) exist in open form mainly as Wan 2.1 I2V community LoRAs (Remade, Civitai). Adopting them requires adding Wan 2.1/2.2 I2V 14B as a family. diffusers' Wan pipelines load LoRAs natively via `load_lora_weights`, so no custom loader would be needed once Wan is in.

### Gaps
- No open "disintegrate" or pyrotechnic-explosion LoRA was located with a confirmed licence. Civitai was not searched.
- No HunyuanVideo effect-LoRA survey was done. Licences on Remade LoRAs were not checked.

## Q7. Relighting and lens or cinematography look

### Takeaway
The most directly usable video relight is Lightricks' LTX-2.3 IC-LoRA Relight (gated). The docs say it is not yet validated on LTX-2.5. Training-free alternatives (Light-A-Video, LightCtrl) build on IC-Light, the image relighter fine-tuned from Stable Diffusion, and have no diffusers pipeline. LTX-2.5 already ships SDR-To-HDR and Day-To-Night IC-LoRAs as look controls. I found nothing on lens or depth-of-field control LoRAs.

### Cited Findings
- `Lightricks/LTX-2.3-22b-IC-LoRA-Relight` relights exterior video to a chosen sun direction and hardness. It is driven by a "light-direction ball" composited into the reference video's corner and runs in a single pass. Access requires accepting the model's conditions. The docs say it has not been validated on LTX-2.5. — [HF card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Relight); [docs.ltx.io relight](https://docs.ltx.io/open-source-model/feature-guides/editing-effects/relight)
- Light-A-Video is training-free video relighting. It reports a motion error of 1.833 against 5.969 for per-frame IC-Light, which flickers. IC-Light is fine-tuned from Stable Diffusion. — [arXiv 2502.08590](https://arxiv.org/pdf/2502.08590)
- LightCtrl (2026) adds user-defined light trajectories, training-free, on an IC-Light base. — [arXiv 2603.27083](https://arxiv.org/pdf/2603.27083)
- LTX-2.5 has the IC-LoRAs `SDR-To-HDR` (2026-09-30) and `Day-To-Night`. — [HF listing](https://huggingface.co/api/models?author=Lightricks&search=LoRA&limit=100&sort=lastModified)
- Community `FuzzPuppy/LTX-2.3-Black-Magic-LoRA` appears in relight-adjacent results. Its purpose was not verified. — [HF](https://huggingface.co/FuzzPuppy/LTX-2.3-Black-Magic-LoRA)

### Inferences
- For stills, IC-Light (SD1.5-based, with an FC and FBC model) can be loaded in diffusers with custom UNet channel patching. The IC-Light v2 Flux variant was not released openly (unverified). Relighting a FLUX 3 Klein still is better done by prompt editing with Klein itself.
- Lens and look presets (anamorphic, shallow DOF, 35mm film, golden hour) are best handled as caption-sentence presets in each family's prompt spec, plus the HDR and Day-To-Night IC-LoRAs on LTX-2.5.

### Gaps
- No open lens, focal-length or DOF-control LoRA was found (the search did not target it directly). IC-Light licence: Apache-2.0 per prior knowledge (unverified).

## Q8. Shot-planning and storyboard agents, and prompt upsamplers for video

### Takeaway
ViMax (HKUDS, MIT, about 5.5k stars, paper arXiv 2606.07649) is the most complete open agentic film pipeline. Its decomposition is screenwriting, then shot planning and storyboard, then character design (reference images), then keyframes, then video generation per shot, then VLM quality control with best-of-k, then assembly. It calls external image and video APIs rather than a local engine. FilmAgent (Xu et al. 2025) is a multi-agent research system in virtual 3D spaces. dw's existing `script-to-video` and `series-episodes` skills already implement a comparable script, shot list, cast-once, per-shot template, validate and cost, run, cut-and-score loop.

### Cited Findings
- ViMax "orchestrates scriptwriting, storyboarding, character creation, and final video generation, all end-to-end". The paper describes "specialized agents for screenwriting, shot planning, character styling, video generation, and VLM-based quality control with best-of-k selection". — [HF papers 2606.07649](https://huggingface.co/papers/2606.07649.md); [alphaxiv](https://www.alphaxiv.org/abs/2606.07649.md)
- ViMax code is at github.com/HKUDS/ViMax and is "fully open-sourced under the MIT license" (secondary source). It reports more than 5,500 stars and 935 forks (news site, may be stale). It recently added Seedance 2.0 Fast as a video backend. Cost is about $1-5 per video in external API fees (unverified blog estimate). — [themenonlab blog](https://themenonlab.blog/blog/vimax-agentic-video-generation-open-source); [mirror repo](https://github.com/2456868764/ViMax)
- FilmAgent is cited by ViMax as "a multi-agent framework for end-to-end film automation in virtual 3d spaces (Xu et al., 2025)". — [HF papers 2606.07649](https://huggingface.co/papers/2606.07649.md)
- The LTX-2.5 caption spec in dw is itself Lightricks' prompt-upsampler system prompt ("Write a single, highly detailed audio-visual caption… in the EXACT style of the training captions"). Kandinsky 6 has a built-in `expand_prompts` text-encoder upsampler. — local `plugins/dw/skills/ltx-2.5/references/caption-spec.md` line 13; `plugins/dw/skills/kandinsky-6/SKILL.md` line 87-88

### Inferences
- The common decomposition across ViMax, MovieAgent and AnimDirector (the latter two unverified this session) is: script, then scene breakdown, then shot list (shot size, camera move, duration, dialogue), then a character and location bible with reference images, then a first-frame keyframe per shot (T2I or edit), then I2V per shot, then VLM judge and retry, then edit and score. dw's MCP tools (`assess_output`, `compare_outputs`) can serve as the VLM judge stage, and list-driven steps cover the per-shot fan-out.
- ViMax's value for dw is as a design reference (prompt templates, best-of-k QC) rather than a dependency, because it targets hosted APIs.

### Gaps
- MovieAgent, StoryAgent, AnimDirector and DreamFactory were not verified this session (repos, licences, decomposition details).
- Open prompt upsamplers dedicated to video, beyond vendor caption specs, were not surveyed.

## Cross-cutting: fit to dw's supported families (summary of the above)
- **Works now, same base:**
  - LTX-2.5 22B LoRAs and IC-LoRAs: Slow-Motion-Control, Cinemagraph, Water-Simulation, Day-To-Night, Clean-Plate, Layout-To-Render, SDR-To-HDR and Ingredients.
  - Prompt-level camera, lens and look presets for LTX-2.5, Kandinsky 6 and H3.
  - Multi-reference editing on FLUX 3 Klein.
- **Probably works, needs validation:** LTX-2.3 IC-LoRAs (Relight, Motion-Track-Control, Union-Control, DubIt) on LTX-2.5. The docs say Relight is not validated on 2.5, and an aggregator says 2.3 LoRAs "partly work".
- **Needs a new base:**
  - Qwen-Image-Edit 2509 or 2511, with the multi-angle LoRAs. It is diffusers-native and the easiest add.
  - Wan 2.1 or 2.2 14B: orbit and crash-zoom effect LoRAs, VACE, Animate, Phantom, MAGREF, Stand-In, ReCamMaster, Uni3C, InfiniteTalk and MultiTalk. The base pipelines are diffusers-native. Most of the conditioning adapters need custom loaders and preprocessing tasks.
  - LTX-2 19B, for the seven named camera LoRAs.
- **Non-commercial flags:** Wav2Lip is confirmed non-commercial. Hallo3, Sonic and the ReCamMaster and TrajectoryCrafter weights are unconfirmed and should be treated as unknown until checked. Lightricks LoRAs are gated and under their own licence terms (not re-verified).
