# MiniMax-H3-Image probe (#1317, stage A #1345)

A bench probe only. Nothing here touches the catalog, the engine, MCP, REST or the
workflow syntax. It answers one question with numbers and side-by-sides: should dw ship
MiniMax-H3-Image (`MiniMaxH3ImageModularPipeline`), and is it a better identity still
than Klein for feeding MiniMax-H3 image-to-video?

## Workflows

All four, plus `t26-face-check.json`, are in `bench/workflows/` and are saved in lem's
`bench` workspace under these MCP names: `bench/h3-image-t2i`, `bench/h3-image-edit3`,
`bench/klein-edit3`, `bench/h3-image-to-h3` and `bench/t26-face-check`. All of them
validate clean. Two changes were needed to get there:

- **The chain's `summary`** was 125 characters, over the schema's 120-character limit,
  so `save_workflow` refused it. It is shorter now.
- **The prompts:** the edit, Klein and chain prompts said "a room", but
  `parhelion/kf-open.png` is a frozen lake at dusk. They now ask to keep "the frozen-lake
  setting, its dusk light and its colour grade".

| File | What it runs |
|---|---|
| `h3-image-t2i.json` | One H3-Image text-to-image still at 2048x2048. |
| `h3-image-edit3.json` | One H3-Image edit at 2048x2048 from three references, named `<Picture 1>`..`<Picture 3>` in the prompt. |
| `klein-edit3.json` | The Klein control: `templates/flux3/multi-reference` as shipped, with the same refs, the same intent worded as "image 1..3", and the same seed. |
| `h3-image-to-h3.json` | The H3-Image edit and the Klein edit (each with `release_pipeline: true`), then `templates/minimax/image-to-video` (960x544, int4) run on each still with one prompt. |

The references are `asset:cast/estela.jpg` and `asset:cast/aurelio.jpg`, both common, and
`asset:parhelion/kf-open.png`, which was copied from `default` into `bench` for these
runs. The seed is 1317.

## H3-Image recipe

- **Transformer:** SDNQ at `weights_dtype` (int8 by default), with H3's quantizer
  arguments. `modules_to_not_convert` is read off the H3-Image transformer. It is
  block-level group offloaded.
- **Text encoder:** loaded from `MiniMaxAI/MiniMax-H3` at `0919fb68`, through the
  per-component `load_components` override, so it shares H3's cache. It is quantized with
  SDNQ, `lm_head` is removed, and its language model is truncated to 51 layers, because
  the pipeline reads `hidden_states[50]`. It runs under leaf-level offload.
- **VAE:** bf16, loaded on demand. It is H3-Image's own `AutoencoderKLMiniMaxH3Image`, so
  H3's int8 VAE recipe isn't carried over.
- **Sampler:** fixed at 6 steps and guidance-distilled, so the workflows pass no
  `num_inference_steps`, `guidance_scale` or `negative_prompt`.
- **Canvas:** width and height are multiples of 32. `min_generation_area` defaults to
  4194304; at 1024 the pipeline generates at the requested size.

## Download gate (first action, passed)

`validate_workflow` on `h3-image-t2i.json` (lem, develop @ 8e1b27f6) lists exactly one
download in `downloads_required`:

- the repo is `MiniMaxAI/MiniMax-H3-Image`;
- the size is 63.93 GiB (about 68.6 GB);
- the components are processor, scheduler, tokenizer, transformer_diffusers and
  vae_diffusers.

No text encoder is fetched from H3-Image, and the run does not download the repo's full
~204 GB. The override is honoured.

## Method

- **Timing:** time each arm cold (first load) and then warm (a rerun on the same worker).
  Record the peak reserved VRAM from the job and the bytes downloaded from
  `list_downloads` or the job events.
- **1024 run:** run `h3-image-t2i.json` with `width=1024 height=1024
  min_generation_area=1024`.
- **Identity:** score every edit still and every clip with
  `bench/workflows/t26-face-check.json` (SFace) against each portrait.
- **Comparison:** post the side-by-sides on #1317, using `compare_outputs` plus a visual
  look.
- **Failures:** an OOM or a load failure is a result. Record it with the error, not a
  blank.

## Results

All runs were on lem (2× RTX 3090 24 GB, 128 GB host RAM), in workspace `bench`, on
develop @ 0365ece6, on 2026-11-16 and 17 (UTC).

| Arm | Cold min | Warm min | Peak VRAM GiB | Bytes downloaded | SFace estela / aurelio | Notes |
|---|---|---|---|---|---|---|
| H3-Image t2i 2048 | 38.2 | 2.0 | ~3.7 reserved (snapshot) | 63.93 GiB planned; transformer cache miss 34.0 GB written | n/a | Completed. Coherent at 2048, and clean at 1:1. |
| H3-Image t2i 1024 (min_generation_area=1024) | — | 2.5 | not sampled | 0 | n/a | Completed at a true 1024x1024. No faster than 2048. |
| H3-Image edit3 2048 | 3.8 (quantized cache, new card) | — | ~5.5 reserved at decode (chain step) | 0 | not scored (stills) | Completed. Both faces and costumes kept; scene light and grade kept. |
| Klein edit3 | — | 1.6 | ~22.0 reserved at decode (chain step) | 0 | not scored (stills) | Completed. Both recognisable, flatter grade, accordion added. |
| H3-Image still → H3 i2v 544p (chain) | 8.1 for the whole chain, both arms (`ad655f0bec2d`) | — | ~9.6 reserved at decode | 0 | **0.423 / 0.316** (16/16 faces) | Completed in the chain after #1408. Standalone (first run): 1.6 min, 0.459 / 0.324. |
| Klein still → H3 i2v 544p (chain) | (same chain job) | — | ~10.0 reserved at decode | 0 | **0.243 / 0.292** (6/16 faces) | Completed in the chain. Face found in only 6 of 16 frames. Standalone: 1.5 min, 0.211 / 0.282. |

### Jobs

| Arm | Job | Run | Card |
|---|---|---|---|
| t2i 2048 cold | `e91d74b829bc` | `20261116-234230-06e036eb` | cuda:0 |
| t2i 2048 warm (seed 1318) | `d2b52f18d0ee` | `20261117-002117-7dad89ea` | cuda:0 |
| t2i 1024 | `a4720895a4fd` | `20261117-002325-de966aa4` | cuda:0 |
| H3-Image edit3 | `f2e738122c65` | `20261117-002619-62c9958a` | cuda:1 |
| Klein edit3 | `d068d6f51d44` | `20261117-003010-21d97902` | cuda:0 |
| Chain (OOM, before #1408) | `02101087ea65` | `20261117-003210-d314f740` | cuda:0 |
| Chain (rerun after #1408, completed) | `ad655f0bec2d` | `20261117-024514-d314f740` | cuda:0 |
| H3 i2v on the chain's H3-Image still | `e281147c8f7f` | `20261117-003902-915bf993` | cuda:0 |
| H3 i2v on the chain's Klein still | `ba62ab6bf574` | `20261117-004051-82ca6f96` | cuda:0 |
| SFace runs, standalone clips | `135e51308622`, `f5f0c84e64cc`, `b201a4e6b0da`, `a1de06d73717` | | cuda:0 |
| SFace runs, chain clips (H3-Image estela, aurelio; Klein estela, aurelio) | `67704a1303de`, `8e9095e3ea2d`, `941ee01ddb09`, `cfc2c0a85a91` | | cuda:0 |

### Timing

| Run | Load | Text encode | Denoise | Decode |
|---|---|---|---|---|
| t2i 2048 cold | 2110.8 s (includes the transformer download and quantization) | 33.4 s | 136.6 s (6 passes; first pass 60.9 s, then 15.2 s each) | 3.9 s |
| t2i 2048 warm | cached | 22.3 s | 91.4 s (15.2 s per pass) | 3.4 s |
| t2i 1024 | — | 22.0 s | 124.5 s (first pass 48.7 s, then 15.1 s each) | — |
| H3-Image edit3 | 56.9 s (quantized-cache hit, cuda:1) | 23.6 s | 132.1 s (first pass 89.1 s, then 8.6 s each) | 3.5 s |
| H3 i2v, H3-Image still | — | 16.6 s | 53.1 s (4 passes, 12.0 s each) | 20.0 s |
| H3 i2v, Klein still | — | 16.8 s | 48.4 s (4 passes, 12.0 s each) | 19.9 s |
| Chain rerun `ad655f0bec2d` | H3-Image still 57.2 s (cache hit); Klein ~17 s; H3 ~6 s | H3-Image 41.9 s | H3-Image 92.9 s (6 passes, 15.2 s each); H3 i2v generate to decode 72.5 s and 64.6 s | 3.9 s; H3 i2v ~20 s each |

A 1024 pass takes the same 15.1 s as a 2048 pass, so the block-offloaded transformer is
bound by weight transfer, not by compute. Generating at 1024 doesn't speed it up.

### Memory

The binding resource is host RAM, not VRAM.

- **Cold t2i:** the job's host peak RSS was 116,760 MB of 128,712 MB, during the
  quantize. Peak committed host memory was 66.5 GB.
- **Warm runs, edit3 and the chain's H3-Image step:** 63 to 64 GB of committed host
  memory. The transformer and the text encoder live in host RAM under offload.
- **VRAM:** `get_memory` has no peak-VRAM field, so the VRAM figures are the job's
  memory-event snapshots. Under block offload, H3-Image reserved ~3.7 GiB at the end of
  the denoise and 5.5 GiB at the start of the decode. It never comes near 24 GB.

### Bytes downloaded

The first run is the only one that downloaded anything. No event carries a byte count,
and `list_downloads` doesn't track a download made during a run, so the figure is built
from these:

- **Before the run:** the validate plan listed `MiniMaxAI/MiniMax-H3-Image` at 63.93 GiB.
- **After the run:** `downloads_required` is `[]`.
- **Quantized cache:**
  - text_encoder: hit, 35.6 GB, from H3's cache, not fetched;
  - transformer: miss, `incomplete_snapshot`, 2064 s for the download and the quantize;
    34.0 GB written in 44.8 s.

Every later run fetched 0 bytes.

### The chain OOM

`bench/h3-image-to-h3` (job `02101087ea65`) produced both stills:

- `h3_image_still`: load 51.9 s, encode 37.5 s, denoise 93.3 s;
- `klein_still`: completed.

It then failed at `h3_image_clip` → `keyframe_to_video_audio`
(templates/minimax/image-to-video, 960x544, int4, turbo LoRA, sparse attention). The
failure came in the generating phase, after the H3 text and VAE encodes and before the
first denoise step:

> CUDA out of memory. Tried to allocate 280.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 153.69 MiB is free ... this process has 23.14 GiB memory in use. Of the allocated memory 22.43 GiB is allocated by PyTorch

The cause is in the job's memory events. Klein's weights were still on the card when H3
loaded:

- after `klein_still`, `gpu_memory_allocated_mb` read 19,216 MB at the start of the H3
  load;
- it still read 19,216 MB at the start of H3's generating phase.

`release_pipeline: true` on a `workflow:` step (the composed `templates/flux3/multi-reference`)
did not free the composed pipeline's GPU weights. On the H3-Image step it did: VRAM went
back to 8 MB after that step.

So the OOM is an engine defect in releasing composed steps. It is not H3-Image's cost, and
not a host-RAM handoff failure. Host memory was fine throughout: H3-Image peaked at 63.8 GB
committed, and after its release RSS fell back to 1.9 GB.

The H3-Image → H3 handoff that the probe measures held on host RAM. H3 i2v on its own
then ran on each chain still; those standalone clips were scored first.

### The chain rerun (after #1408)

#1408 (`release_pipeline` on a `workflow:` step pops the composed child's pipelines,
`9399df4e`) was verified and deployed (lem on develop @ 8384837e). The same
`bench/h3-image-to-h3`, unchanged, was rerun as job `ad655f0bec2d` and completed in 483 s
(8.1 min), both arms through to their clips.

In its memory events, Klein's release now holds:

- `klein_still` reads 19,216 MB allocated while generating (t=255.2) and 19,241 MB at its
  decode (t=294.4);
- after its save, at t=295.8, allocated is back to 8 MB;
- H3 video loads at t=296.1 with 8 MB allocated, and the H3 i2v steps run without an OOM
  (~9.6 and ~10.0 GiB reserved at their decodes).

The H3-Image step released as before (8 MB at t=207.5). The committed host peak for the
job was 63.2 GB.

The chain's clips are scored with the same `t26-face-check` route. The SFace pairs are
0.423 / 0.316 (H3-Image, 16/16 faces) against 0.243 / 0.292 (Klein, 6/16). They differ a
little from the standalone pair (0.459 / 0.324 and 0.211 / 0.282) because the chain
re-rendered both stills (no step-cache reuse, `cached_steps: 0`). The direction is the
same: the H3-Image arm scores higher on both portraits and keeps a detectable face in
every sampled frame.

### Look

These judgements are by eye. `compare_outputs` gives a grid only: there are no PSNR or
SSIM metrics, because the sizes differ (2048² against 896x1152).

- **t2i 2048:** the lighthouse keeper on the gallery, with a storm over the sea. The
  image is coherent across the whole canvas, and a 1:1 crop is clean.
- **t2i 1024:** a close three-quarter framing of the keeper in a flat cap, with strong
  face detail. It is a true 1024² image; the composition differs from the 2048 run.
- **H3-Image edit3:**
  - Estela (the purple gown with butterflies, a silver bun and a tiara) and Aurelio (a
    sombrero, a striped poncho and a moustache) sit at a round table on the frozen lake
    at dusk, with the low sun and the treeline.
  - It is faithful to both portraits and keeps the scene's light and grade.
  - They face each other in near-profile.
- **Klein edit3:**
  - Both characters are recognisable.
  - Aurelio holds the accordion from his portrait, which nobody asked for.
  - The grade is flatter and greyer than the scene's, and the table is square.
- **Clips:** both are coherent and hold the static two-shot.
  - From the H3-Image still: the faces stay stable across the 5.2 s.
  - From the Klein still: the faces are turned and small, so SFace finds a face in only 6
    of 16 frames, and its scores are low and noisy (minimums 0.02).
  - The audio in both is quiet: -47 dBFS mean, ambience level. The line "You came back."
    is not clearly carried.

The SFace caveat: `analyze_face_drift` compares each sampled frame's face with one
reference. In a two-person frame, which face it matches isn't controlled. Read the pairs
as a relative A/B between the arms, not as absolute identity scores.

### Deviations

- **The warm t2i** used seed 1318, because an identical rerun would have been served
  from the step cache.
- **The chain's clips** first came from standalone `templates/minimax/image-to-video`
  runs on the chain's own stills, because the chain OOMed (#1408). After the fix, the
  chain itself was rerun (`ad655f0bec2d`) and its clips scored; both sets are recorded.
- **The download gate** was measured on develop @ 8e1b27f6; the runs used 0365ece6.
- **The edit stills** weren't scored with SFace. `t26-face-check` takes a video, and the
  bounce asked for SFace on the two video arms.

GPU time for the whole probe was about 57 minutes, 35 of which were the one-time
download and quantize, plus about 8 minutes for the chain rerun.
