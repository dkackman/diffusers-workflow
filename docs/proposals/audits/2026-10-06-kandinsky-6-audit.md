# Kandinsky 6.0 - verification of issue #663 against primary sources

Research date: 2026-10-06. Everything here is about Kandinsky **6.0**, released 2026-10-06. Kandinsky 5.0 (arXiv 2511.14993: 2B Lite / 19B Pro, 10 s video, no audio) is a different family, and none of its facts are used here. The one place 5.0 leaks into 6.0 is the vendor's own beautifier code, which reuses the K5 t2va instruction text. It is flagged where it comes up.

Abbreviations: `P` = `venv/lib/python3.14/site-packages/diffusers/` (diffusers main @ c6df88a5. Its `__version__` still reads **0.41.0.dev0**, the same string as the pre-K6 venv, so the version string cannot tell the two apart. Check that the class is present instead.)

## Sources

| URL | Type | Date | Contribution |
|---|---|---|---|
| `P/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py`, `pipeline_kandinsky6_sr.py`, `pipeline_output.py` | primary (code) | merged 2026-10-06 (bc5e3bd9) | What the pipelines accept, their defaults, constants, beautifier instructions |
| `P/models/transformers/transformer_kandinsky6.py`, `transformer_kandinsky6_sr.py`, `P/schedulers/scheduling_piflow.py`, `P/models/autoencoders/autoencoder_mmaudio.py`, `autoencoder_kandinsky6_sr.py`, `P/models/latent_upscaler/latent_upscaler_kandinsky6_sr.py` | primary (code) | same | Attention backends, RoPE limits, fp32 modules, Piflow behaviour, audio rate |
| https://github.com/huggingface/diffusers/pull/14949 (`gh pr view`, review comments) | primary | opened 2026-10-05, merged 2026-10-06T12:04:58Z, commit bc5e3bd9 | Vendor author (leffff) corrects the docs: distilled = **10 steps**, CFG 1.0. Pretrain = 50/5.0 |
| https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/kandinsky6.md (rendered at https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky6) | primary | 2026-10-06 | Model table with steps/CFG per checkpoint, SR usage, memory notes, input rules |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers (raw README, model_index, scheduler/transformer/audio_vae configs) | primary | created 2026-09-09, modified 2026-10-06 | 10 steps / CFG 1.0, PiFlow. T2AV + TI2AV examples, SR guide, component table |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers | primary | 2026-09-09 / 2026-10-06 | 50 steps / CFG 5.0, FlowMatchEuler shift 5.0 |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers | primary | 2026-09-29 / 2026-10-06 | "same architecture and sampler settings as the main Lite model", listed for "generation" |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers | primary | 2026-09-09 / 2026-10-06 | 10 steps / CFG 1.0, PiFlow, not gated |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers | primary | 2026-09-09 / 2026-10-06 | **Gated (`auto`)**: the raw README and configs return 401. The HTML card (WebFetch) shows 50 steps / CFG 5.0 and FlowMatchEuler |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers | primary | 2026-09-29 / 2026-10-06 | 50 / 5.0, "same sampler settings as the main Pro model" |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers | primary | 2026-09-21 / 2026-10-06 | Flow-matching SR, 4 Euler steps/tile, 1.41B DiT + 1.74B KVAE + 3.65B latent-upscaler bank |
| https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers | primary | 2026-09-15 / 2026-10-06 | π-Flow SR, 2 evals/tile, PiflowScheduler `nfe=2, shift=3.5` |
| https://huggingface.co/api/collections/kandinskylab/kandinsky-60-diffusers | primary | updated 2026-10-05 | Lists the 6 generation repos only. **The VSR repos are not in the collection** |
| https://github.com/kandinskylab/kandinsky-6 (README, `kandinsky/configs/checkpoints/*.yaml`, `configs/devices/*.yaml`, `cli.py`, `core/components/beautifier/qwen25.py`, `pipeline/config.py`, `comfyui/kandinsky6/beautifier_prompts/*.txt`) | primary | created 2026-10-05, pushed 2026-10-06 | Reference settings per checkpoint, GPU presets (24 GB and 16 GB run **Pro-distill**), timing table, beautifier texts, default negative prompt |
| https://github.com/kandinskylab/kandinsky-6-sr (README) | primary | created 2026-10-05, pushed 2026-10-06 | SR config table, H100 timings, padding/crop behaviour, `target_resolution` delivery tiers |
| https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s (`app.py`, `beautifier_os_plain/config.json`) | primary (vendor demo) | created 2026-09-27 | Production settings for the demo: 480x864, 121 f, 24 fps, 10 steps, CFG 1.0. I2V sizing by aspect ratio. Qwen3.5-9B beautifier. Memory notes |
| https://arxiv.org/abs/2610.05608 (abstract only) | primary (report) | submitted 2026-10-04 | 3B Lite / 29B Pro, dual-stream CrossDiT, 5 s, 44 kHz, lip-sync, Full-HD via SR, SFT + RL + distillation. Full PDF not read |
| Local measurement, `venv/bin/python`, torch 2.14.1, **Apple M4 Pro / 24 GB** (this dev box, *not* the 64 GB onboarding Mac) | primary (measured) | 2026-10-06 | MPS SDPA at K6 token counts: fused, no memory blow-up. Timings below |
| Web search "Kandinsky 6.0 video audio model release VRAM" | secondary | 2026-10-06 | **No independent secondary coverage found** (released today). Results were K5/K4 pages only. The vLLM-Omni docs page returned HTTP 429 and was not read |

No secondary source corroborates or contradicts anything. Every verdict below rests on primary sources alone.

## Claim-by-claim verdict (issue #663)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Licence MIT | CONFIRMED | Card front-matter `license: mit`. GitHub repos MIT. (The diffusers port is Apache-2.0.) |
| 2 | Joint text/image-to-video+audio | CONFIRMED | Card "T2AV and TI2AV". `Kandinsky6TI2VAPipeline` `image=` at ti2va.py:649 |
| 3 | 5 s clips | CONFIRMED | Cards, arXiv abstract. 121 frames at 24 fps (vendor configs `sample_frames: 121`, `latent_frames: 31`) |
| 4 | 44.1 kHz audio incl. lip-sync | CONFIRMED | `audio_vae/config.json` `sample_rate: 44100`. Fallback `44_100` at ti2va.py:254-256. Cards say "44 kHz ... including lip-sync" |
| 5 | Separate tiled SR pipeline | CONFIRMED | `Kandinsky6SRPipeline` (sr.py:143). Overlapping tiles with Hann blend |
| 6 | Merged to main 2026-10-06, #14949, `bc5e3bd9` | CONFIRMED | `mergedAt 2026-10-06T12:04:58Z`, merge oid bc5e3bd9c554… |
| 7 | Classes `Kandinsky6TI2VAPipeline`, `Kandinsky6SRPipeline`, `Kandinsky6Transformer3DModel` (+ SR transformer) | CONFIRMED (incomplete) | Also new: `Kandinsky6SRTransformer3DModel`, `Kandinsky6SRVAE`, `Kandinsky6SRLatentUpscalerBank`, `PiflowScheduler`, `MMAudioVAE`, `MMAudioVocoder`, `Kandinsky6TI2VAPipelineOutput`, `Kandinsky6SRPipelineOutput` (`P/__init__.py:310-330, 470, 713-716`) |
| 8 | Not in a release. Latest v0.41.0 | CONFIRMED | v0.41.0 published 2026-10-06T06:43Z, before the merge. `pipelines/kandinsky6` is 404 at tag v0.41.0 |
| 9 | "Repo venv's 0.41.0.dev0 has no `Kandinsky6*` classes" | STALE | The venv is now main @ c6df88a5 and has them, still labelled 0.41.0.dev0. Gate on class presence, not the version string |
| 10 | Components: Qwen2.5-VL + CLIP, `AutoencoderKLHunyuanVideo`, `MMAudioVAE` + vocoder, audio optional via `sample_audio` | CONFIRMED | `model_index.json`. ti2va.py:220-231, `_optional_components = ["audio_vae","vocoder"]` (:213). Vocoder class is `MMAudioVocoder` (BigVGAN-style). Note `sample_audio` **defaults True** (:667) |
| 11 | All checkpoints `*-5s-Diffusers` | CONFIRMED | 6 generation repos + `Kandinsky-6.0-VSR-5s-Diffusers` + `Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` |
| 12 | Lite-distill: 3B, 10 steps / CFG 1.0, PiflowScheduler, must use 1.0 | CONFIRMED | Card ("Use 10 steps and guidance 1.0"), docs table, vendor `lite-distill.yaml`, scheduler_config `PiflowScheduler shift 5.0 n_grid 10`. ti2va.py:700 "Must be `1.0` with a PiflowScheduler" |
| 13 | Lite: 3B, default 50 | CONFIRMED (incomplete) | 50 steps **and CFG 5.0**, FlowMatchEuler shift 5.0 (card, `lite.yaml`) |
| 14 | Lite-pretrain: "base, not for end use" | CONTRADICTED | Card: "the pretrained Lite checkpoint. It uses the same architecture and sampler settings as the main Lite model", Use column "generation", full T2AV/TI2AV examples at 50 / 5.0. Nothing in any source says "not for end use". (The vendor still treats `pro-distill` as the default and pretrain as the pre-SFT/RL base. That is an inference from the arXiv abstract's SFT + RL + distill pipeline, not a stated rule.) |
| 15 | Pro-distill: 29-30B, **~16 steps** / CFG 1.0 | CONTRADICTED (steps) | **10 steps**: card ("samples in 10 steps with the PiFlow scheduler"), diffusers docs table, vendor `pro-distill.yaml` `num_steps: 10`, the Space `K6_NUM_STEPS=10`, and the vendor's PR review comment ("num_inference_steps=10"). The "16" is a stale leftover in the diffusers source: `EXAMPLE_DOC_STRING` ti2va.py:53, `__call__` docstring ti2va.py:694 ("Use `16` with the distilled checkpoints"), and sr.py:47. Size confirmed: transformer 60.34 GB bf16 on disk, ~30B (wider PiFlow heads) |
| 16 | Pro: 29B, 50 / ~5.0, FlowMatchEuler | CONFIRMED | Gated card (HTML) and `pro.yaml`: 50 / **5.0** exactly, FlowMatchEuler shift 5.0 |
| 17 | Pro-pretrain: 29-30B, base | PARTLY | 50 / 5.0 per card. "Base" is accurate as lineage but not a use restriction (see #14). Transformer is 69.73 GB on disk, more than 29B x 2 bytes (likely some fp32 tensors; unverified) |
| 18 | Native 480x864, 121 frames | CONFIRMED | Every card example and vendor config. **But the diffusers `__call__` defaults are `height=512, width=768`** (ti2va.py:651-652). Templates must set 480/864 explicitly |
| 19 | SR to 1920x1080 via x2 / x2.25 / x4 | PARTLY | Scales confirmed: `if resolution_scale not in (2, 2.25, 4)` (sr.py:196). "1920x1080" is marketing for Full-HD. In diffusers, 480x864 at x2.25 → pre-upscale 1.125 snapped to 16 px (`round(540/16)*16=544`, `round(972/16)*16=976`, sr.py:286-293) → x2 = **1088x1952**, not 1080x1920. The vendor CLI crops to exactly source x scale (1080x1944) and has a separate `target_resolution: fullhd` downscale. Diffusers has neither |
| 20 | 4 steps per tile, 2 distilled | CONFIRMED | `num_inference_steps: int = 4` (sr.py:234). Docstring "Use `2` with the distilled checkpoints" (sr.py:254). The distilled card says 2 "must be set explicitly" |
| 21 | 24 GB: 3B transformer ~6 GB bf16 | CONFIRMED | Lite-distill transformer 6.38 GB, Lite 7.48 GB on disk |
| 22 | 24 GB: Qwen2.5-VL is the largest piece, offloaded | CONFIRMED | `text_encoder/` 16.6 GB (bf16 7B). Offload seq puts it first: `text_encoder->text_encoder_2->transformer->vae->audio_vae->vocoder` (ti2va.py:212). Lite-distill repo total 26.7 GB |
| 23 | Pro "only quantized" on 24 GB, 29B bf16 ~58 GB | CONTRADICTED (feasibility) | Size is right (60.3 GB Pro-distill). But the vendor's own `configs/devices/rtx-4090.yaml` (24 GB) runs **pro-distill unquantized** with `offload.strategy: block` and SageAttention. `rtx-5060-ti.yaml` (16 GB) does the same plus `quantized_qwen: true` (NF4 Qwen, "saves ~10-12 GiB", `pipeline/config.py:149`). The README timing table has Pro on an RTX 4090 at 936 s (SD, non-distilled). Block offload needs ~60 GB+ of host RAM. The diffusers equivalent is `enable_group_offload(offload_type="block_level", use_stream=True)`, which the vendor Space says pins the whole 56 GiB DiT up front (~45 s per request) |
| 24 | Pro with CFG = 50 steps x 2 passes | CONFIRMED | `do_classifier_free_guidance` is `guidance_scale > 1.0` (ti2va.py:629-630). The uncond pass is at :901-917 |
| 25 | SR is tiled, memory should be bounded | PARTLY | Tile compute is bounded. But the latent path KVAE-encodes the **whole video** first (sr.py:322-323), and the output accumulator is a full-canvas fp32 tensor on the device (sr.py:346): 3x121x1088x1952x4 B ≈ 3.1 GB at x2.25. Uncompiled flex attention "does not fit in memory at video resolutions" (transformer_kandinsky6_sr.py:161-166) |
| 26 | "No VRAM numbers are published" | PARTLY | No peak-VRAM table exists. The vendor does publish GPU presets (16/24/32 GB consumer cards run Pro-distill with block offload), per-GPU timings, and Space notes ("9B + Qwen + VAEs (~37 GB) fit the 46 GB slot", "56 GiB DiT") |
| 27 | `<S>` / `<AUDCAP>` prompt tags | CONFIRMED | Full forms are `<S>…<E>` (direct speech) and `<AUDCAP>…<ENDAUDCAP>` (audio caption), in `_T2VA_EXPANSION_INSTRUCTION` (ti2va.py:82-90) and `_I2VA_EXPANSION_INSTRUCTION` (:91-101) |
| 28 | Built-in prompt-beautifier system prompt in the pipeline | CONFIRMED (with correction) | The beautifier instructions are user-turn texts, not a system prompt: `_T2VA_EXPANSION_INSTRUCTION` and `_I2VA_EXPANSION_INSTRUCTION`, used by `Kandinsky6TI2VAPipeline.expand_prompts` (:413-485) when `expand_prompts=True` (default **False**, :668). The actual system prompt in the file is the encoder template `_PROMPT_TEMPLATE` (:66-78), which is always applied, then cropped by `_QWEN_CROP_START = 129`. The vendor has **three different** beautifier texts (see Missing knowledge) |
| 29 | Lite-distill is the right first template | SUPPORTED | Smallest download (26.7 GB), 10 steps, one pass. Card examples. But the vendor's own default and demo are Pro-distill |
| 30 | HunyuanImage-3.0 "Not filed" section | NOT CHECKED | Out of scope |

## Pipeline facts (diffusers main @ c6df88a5)

**`Kandinsky6TI2VAPipeline`** (`P/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py:178`)
- `__call__` (:646-674): `prompt=None, image=None, negative_prompt=None, height=512, width=768, num_frames=121, frame_rate=24.0, num_inference_steps=50, timesteps=None, sigmas=None, guidance_scale=5.0, num_videos_per_prompt=1, generator=None, latents=None, audio_latents=None, prompt_embeds=None, pooled_prompt_embeds=None, negative_prompt_embeds=None, negative_pooled_prompt_embeds=None, sample_audio=True, expand_prompts=False, max_sequence_length=1024, output_type="pil", return_dict=True, callback_on_step_end=None, callback_on_step_end_tensor_inputs=["latents"]`.
- Resolution: `height` and `width` must be divisible by `vae_scale_factor_spatial * max(patch_size[1:])` = 8x2 = **16**. This is a hard `ValueError` (:503-507).
- Frames: `num_frames >= 1` (:508). If `num_frames % 4 != 1` it warns and sets `num_frames = num_frames // 4 * 4 + 1` (:756-760). That rounds **down** (122 → 121), even though the message says "nearest". Rule: **4k+1**.
- Hard code limits (RoPE tables, index error beyond): video RoPE `max_pos=(128,128,128)` (transformer_kandinsky6.py:122). So latent frames ≤ 128 (≤ 509 frames; image conditioning adds one tail latent frame), and height, width ≤ 128x16 = 2048 px. Audio and text 1D RoPE `max_pos=2048` (:97). The model is trained only at 5 s / 121 frames. The vendor ships only `*-5s` checkpoints and its SR repo calls the 121-frame budget a "model contract".
- fps: `frame_rate` (default 24.0) only sets the audio length: `audio_length = ceil(((latent_frames-1)*4+1)/frame_rate*sample_rate/hop)` (:836-841), hop = `audio_vae.latent_hop_length` (fallback 1024, :257-259). The video is always encoded at 24 fps in every vendor example.
- Audio output: `output.audio` has shape `(batch, num_samples)`, mono, float32 in [-1, 1], at `pipe.audio_sample_rate` (= `audio_vae.config.sample_rate`, 44100). It is a **torch tensor** unless `output_type="np"` (:950-955). It is `None` when `sample_audio=False`. With `output_type="latent"` it holds audio latents `(B, C, L)`. Vendor muxing: `encode_video(frames, fps=24, audio=output.audio[0][None], audio_sample_rate=pipe.audio_sample_rate)`. The Space duplicates the mono track to stereo.
- Image conditioning: PIL images are **resized and center-cropped** to `height x width` (`resize_mode="crop"`, :567-572). Tensors must already be that size. Requires `transformer.config.visual_cond` and `visual_token_type_num_embeddings >= 2` (:543-549). All six released configs have `visual_cond: true`, `visual_token_type_num_embeddings: 2`. The reference is appended as a masked tail latent frame (:850-867).
- CFG: on when `guidance_scale > 1.0`. Default negative `_DEFAULT_NEGATIVE_PROMPT` (:215-218) = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards". This is identical to the vendor CLI `--negative` default (`kandinsky/cli.py:39-42`).
- Text: Qwen tokens padded to `max_length = max_sequence_length + 129`. `_CLIP_MAX_LENGTH = 77` (:81). `_QWEN_CROP_START = 129` (:80).
- Constants to pin: `_PROMPT_TEMPLATE` (:66-78), `_QWEN_CROP_START` (:80), `_CLIP_MAX_LENGTH` (:81), `_T2VA_EXPANSION_INSTRUCTION` (:82-90), `_I2VA_EXPANSION_INSTRUCTION` (:91-101), `Kandinsky6TI2VAPipeline._DEFAULT_NEGATIVE_PROMPT` (:215), `model_cpu_offload_seq` (:212).
- `expand_prompts` (staticmethod, :413) generates with `max_new_tokens=max_sequence_length` (1024) and greedy defaults. With a generator it seeds the **global** RNG via `torch.manual_seed(generator.initial_seed())` (:481-482). It expects a PIL image for i2v.
- dtype: compute dtype = `transformer.dtype` (:772). `_keep_in_fp32_modules = ["time_embeddings", "modulation"]` (transformer_kandinsky6.py:648) upcasts those under `torch_dtype=bfloat16`. The vendor Space overrides this to `["time_embeddings"]` because it adds "+9 GB in RAM" on Pro.
- Attention: `Kandinsky6AttnProcessor` uses `dispatch_attention_fn` with the configurable backend (default native SDPA, transformer_kandinsky6.py:160-207). `MMAudioVAE` forces `AttentionBackendName.NATIVE` because its head_dim is too big for flash (autoencoder_mmaudio.py:143-150).

**Schedulers per checkpoint** (repo `scheduler/scheduler_config.json`, 2026-10-06)

| Checkpoint | Scheduler | Key config | Steps / CFG (card + vendor yaml) |
|---|---|---|---|
| Lite-distill | `PiflowScheduler` | shift 5.0, n_grid 10, final_step_size_scale 0.5, num_policy_substeps 128 | 10 / 1.0 |
| Pro-distill | `PiflowScheduler` | same | 10 / 1.0 |
| Lite, Lite-pretrain, Pro, Pro-pretrain | `FlowMatchEulerDiscreteScheduler` | shift 5.0, no dynamic shifting | 50 / 5.0 |
| VSR | `FlowMatchEulerDiscreteScheduler` | shift 5.0 | 4 per tile |
| VSR-distilled2steps | `PiflowScheduler` | nfe 2, shift 3.5, n_grid 10 | 2 per tile |

`PiflowScheduler` rejects custom `sigmas`, `timesteps` and `mu` (scheduling_piflow.py:241-242) and rolls out in float32 (:344-347). It cannot be swapped for another scheduler, because the distilled transformer outputs `n_grid x` channels (`out_visual_dim: 160`, `out_audio_dim: 400`).

**`Kandinsky6SRPipeline`** (`pipeline_kandinsky6_sr.py:143`)
- `__call__(video, resolution_scale=2.25, num_inference_steps=4, timesteps=None, sigmas=None, lq_noise_scale=0.7, min_overlap=0.2, tiles_batch_size=1, generator=None, output_type="pil", return_dict=True)` (:230-243). There is no audio, no prompt and no seed parameter beyond `generator`.
- `video` takes any `VideoProcessor.preprocess_video` input: a list of PIL images, an ndarray or a tensor. Sizes are rounded down to multiples of 16 (KVAE `spatial_compression_ratio = 2**(len(blocks)-1) = 16`, autoencoder_kandinsky6_sr.py:582). The frame count **must be 1 + 4k** or it raises a `ValueError` (sr.py:207-211).
- Scales are only 2, 2.25 (1.125 bilinear pre-upscale + 2x) or 4 (:196, :286). `lq_noise_scale` must be in (0, 1]. `min_overlap` must be in [0, 1).
- Tiles: the trained tile sizes default to `((512,512),(512,768),(768,512))` (:186-190). The pipeline picks the one closest to the input aspect (:303). The input-side tile is the base tile divided by the scale, and the input must be at least that big (:311-315). For example, x2 needs at least 256x384 for landscape.
- Output size is exactly `input x tiling_scale` after any pre-upscale: 480x864 at x2.25 gives **1088x1952**, at x2 gives 960x1728, at x4 gives 1920x3456.
- Offload seq: `latent_upscaler->transformer->vae` (:167).
- Attention: `Kandinsky6SRAttnProcessor` **always** uses `AttentionBackendName.FLEX` with a NABLA `BlockMask` (transformer_kandinsky6_sr.py:170-211). The constructor raises `ImportError` without flex attention (torch ≥ 2.5) (:476-480). It warns when uncompiled (:161-166). The token grid must be divisible by `FRACTAL_BLOCK_SIZE = 8` (:44). Vendor recipe: `torch._inductor.config.max_autotune = True` ("required"), `set_attention_backend("flex")`, `enable_model_cpu_offload()`, `compile_repeated_blocks(fullgraph=True)`.
- The VSR repos' `transformer/config.json` and `latent_upscaler/config.json` are in the vendor's legacy key format (`attention_params`, `sr_params`, a `models` list). Diffusers ignores the unknown keys and falls back to the class defaults (`tile_sizes`, `nabla_threshold=0.8`, `scales=(2,4)`), which happen to match the vendor values. If a later re-key changes the defaults, this breaks silently.

**Checkpoint drift:** every repo has a `rename-module-keys` branch. The vendor Space pins a revision because "upstream re-keys the weights whenever the module names change (last time 2026-10-03)" (Space `app.py:63-65`). Catalog entries should pin a `revision` per repo, checked against the installed diffusers commit.

## Apple Silicon / MPS

- **No float64, no CUDA-only kernels** in the TI2VA path. `grep` finds no `float64`/`double`/`cuda` in the K6 transformer, pipeline, Piflow or MMAudio code. RoPE, modulation, gates and Piflow run in float32 by design. `torch.stft(return_complex=True)` exists only in the MMAudio mel *encoder* (autoencoder_mmaudio.py:344), which generation does not call. **Unverified:** whether the BigVGAN snakebeta vocoder and HunyuanVideo VAE decode run cleanly on MPS in bf16. Run a first smoke test with `sample_audio=True`.
- **Attention on MPS (measured on this M4 Pro 24 GB, torch 2.14.1, not the 64 GB target):** native SDPA at the K6 video token count (31 latent frames x 30 x 54 = 50,220 tokens) is fused and allocates no N² matrix.
  - Lite shape (28 heads x 64): **3.3 s per self-attention call**.
  - Pro shape (32 x 128): **7.4 s**.
  - Masked cross-attention to 1024 text tokens: 0.3 s.
  - Rough lower bound for self-attention alone: Lite-distill 32 blocks x 10 steps ≈ **18 min**. Lite at 50 steps with CFG ≈ 3 h. Pro-distill 60 x 10 ≈ 74 min.
  - These are estimates. Measure end to end on the 64 GB Mac.
- **Memory on a 64 GB Mac:** Lite-distill bf16 is 26.7 GB of weights and fits resident, so skip CPU offload (it buys nothing on unified memory). Pro-distill (60 GB transformer + 16.6 GB Qwen) **does not fit** without quantizing the transformer (the dw MPS path is SDNQ; bitsandbytes NF4, which the vendor uses for Qwen, has no Mac path per docs/ACCELERATION.md). Treat Pro as out of scope for the Mac onboarding.
- **SR on MPS: high risk, likely blocked.** NABLA needs flex attention. Eager flex fallback materializes full attention, which the source says does not fit. dw skips `compile` on MPS (ACCELERATION.md:191, :289). A trivial `torch.compile(flex_attention)` call did run on MPS here, but compiled flex with a NABLA `BlockMask` and `max_autotune` on MPS is untested. Mark SR as CUDA-only until a measured run says otherwise.
- The vendor stack itself is NVIDIA-only (README: "An NVIDIA GPU"). Its kernels are SageAttention/FA3 and the BigVGAN CUDA kernel. The diffusers port avoids all of these.
- Vendor examples use `torch.Generator("cuda")` (VSR card), which must become `mps`/`cpu` on a Mac. dw's device adaptation should cover this.

## Missing knowledge (not in the issue)

1. **Per-checkpoint audio VAE scaling differs** (vendor yaml and HF `audio_vae/config.json`, 2026-10-06). Lite and Pro use `scaling_factor: 0.5302`. The distill and pretrain checkpoints use `0.417`. Each repo ships its own audio_vae, so never share the component across checkpoints.
2. **Beautifier is on by default in the vendor stack, off in diffusers.** Every vendor device preset sets `beautifier: qwen25`. The demo uses Qwen3.5-9B with a long system prompt. Diffusers defaults to `expand_prompts=False`. Short prompts will under-perform unless the template enables it or the user writes long prompts. (2026-10-06)
3. **Three beautifier texts exist; they disagree.** (a) Diffusers `_T2VA/_I2VA_EXPANSION_INSTRUCTION`: short, "English prompts". (b) The vendor CLI `qwen25.py` `_t2va_instruction`: K5's few-shot instruction, labelled "K5 t2va instruction" in source. (c) The vendor ComfyUI/Space `beautifier_prompts/t2av_system.txt` and `i2av_system.txt` (Qwen3.5-9B, temperature 0.3, repetition_penalty 1.1, max_tokens 1500). (c) is the most detailed prompting guide the vendor publishes:
   - Keep the request's language (do not translate).
   - One paragraph of 180-300 words, in this order: frame and shot size, exactly one camera sentence ("The camera is static." when none is requested), appearance, the 5 s action in order, then light and style last.
   - Drop technical tags (masterpiece, 4k, --ar).
   - Never use "atmosphere/mood/feeling".
   - Give every moving object a direction.
   - Speech goes inside `<S>…<E>`, with the speaker and a speech verb right before the tag. One pair per line, never translated.
   - A silent visible person gets "lips closed".
   - Keep sound words out of the body. End with exactly one `<AUDCAP>…<ENDAUDCAP>` sentence: the loudest source marked "loudly", music written as "music" + instrument, no "echo", no speech text, no voices if nobody speaks.
   - I2V: the image is ground truth and the first frame. Quote visible lettering exactly. Subjects not in the image "enter the frame". Flat images have no background depth.
4. **Card example prompt shape** (Lite-distill card, 2026-10-06): style lead ("cinematic shot: …"), action, camera move, look, then "Audio: …" list, then "No dialogue, text, or logos." The diffusers expansion uses `<AUDCAP>` instead. Both appear in vendor material.
5. **I2V sizing**: the vendor CLI (`max_area` resize, divisibility 16, `pipeline/config.py:135-139`) and the Space (`size_for_image`: area of 480x864 at the image's aspect ratio, multiples of 16) keep the reference aspect, so portrait in gives portrait out. Diffusers center-crops to the given `height x width`. The template should compute height/width from the image, or the user loses framing. (2026-10-06)
6. **SR applies to video only**: re-mux the generation's audio (cards). The vendor `kandy-sr` pads to the 16 px stride, crops back to exactly source x scale, and offers `target_resolution: hd|fullhd|2k`. H100 timing for 768x512, 121 f, distilled: x2 = 41 s (9 tiles), x4 = 118 s (25 tiles) (kandinsky-6-sr README, 2026-10-06).
7. **Vendor timings** (non-distilled, 5 s clip, after warmup, kandinsky-6 README, 2026-10-06): Lite SD is 437 s on RTX 4090 and 239 s on H100. Pro SD is 936 s on RTX 4090 and 356 s on H100.
8. **Memory figures** (Space `app.py`, 2026-10-06): the Pro DiT is "56 GiB". "9B [beautifier] + Qwen + VAEs (~37 GB) fit the 46 GB slot". NF4 Qwen saves ~10-12 GiB (`pipeline/config.py:149`). Repo totals on disk: Lite-distill 26.7 GB, Lite and Lite-pretrain 27.8 GB, Pro-distill 80.6 GB, Pro-pretrain 90.0 GB, each VSR 17.5 GB.
9. **Pro is gated** (`gated: auto`, accept terms + HF login). The other seven repos are not gated. (2026-10-06)
10. Diffusers docs memory notes: `enable_sequential_cpu_offload()` and `pipe.vae.enable_tiling()` for high-res or long video. `compile_repeated_blocks(fullgraph=True)` for speed.
11. Vendor CLI frame rule when `time_length` is given: `num_frames = tl*24//4 + 1` latent frames (`pipeline.py:442`). Only 5 s is shipped.

## What the templates should teach (reading order)

1. **Lite-distill text-to-video+audio**: `kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`, bf16, `height=480, width=864, num_frames=121, num_inference_steps=10, guidance_scale=1.0`, `sample_audio=True`, mux at fps 24 with `audio_sample_rate=pipe.audio_sample_rate`. Use the card's DEFAULT_PROMPT (stone samurai) shape as the example. Teaches: always set 480/864 (the defaults are 512/768), keep CFG at 1.0, and use 4k+1 frames.
2. **Lite-distill image-to-video+audio**: the same settings plus `image=`, with the prompt describing what happens *next* (the card's dragon example, `assets/i2va_input.jpg`). Height/width come from the image aspect at the 480x864 area, in multiples of 16 (the vendor Space rule). Teaches the first-frame conditioning and the aspect rule.
3. **Prompt expansion**: the same as 1 with `expand_prompts=True` and a short prompt. Teaches the `<S>…<E>` / `<AUDCAP>…<ENDAUDCAP>` tags and the cost (an extra Qwen generation pass, no extra weights). Point at `_T2VA_EXPANSION_INSTRUCTION` and the vendor `t2av_system.txt` rules; do not copy them.
4. **Speech / lip-sync**: Lite-distill with an explicit `<S>line<E>` after "X says:", ending with an `<AUDCAP>` sentence. This comes from the vendor beautifier rules; there is no standalone card example.
5. **Lite (quality)**: `Kandinsky-6.0-Lite-5s-Diffusers`, 50 steps, CFG 5.0, the default negative prompt. Teaches the CFG cost (2 passes).
6. **SR (CUDA-only until proven)**: `Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers`, `resolution_scale=2.25, num_inference_steps=2` (VSR-5s: 4). Requires `max_autotune` + flex + compile, and audio re-mux. Optionally downscale to 1920x1080 afterwards, because diffusers outputs 1088x1952.
7. **Pro-distill** (CUDA, 10 / 1.0, block/group offload) and **Pro** (50 / 5.0, gated) last, as cost/quality upgrades. Not for the Mac.
8. Pretrain checkpoints: catalog entries only, with no template, since no vendor use case distinguishes them.

## Assessment

A skill can trust these parts of #663 as written: the family shape (T2AV/TI2AV, MIT, 3B/29B, 5 s at 121 frames and 24 fps, 44.1 kHz mono audio, 480x864 native), the component list, the merge facts and "not in a release", Lite-distill/Lite/Pro settings (with CFG 5.0 added for Lite), the SR scale set and per-tile step counts, the tag names, and the Lite memory arithmetic. Lite-distill as the first template is sound.

These parts need correcting:
- **Pro-distill is 10 steps, not ~16.** The 16 comes from stale docstrings in the diffusers source (ti2va.py:53, :694, sr.py:47). Every vendor source says 10. A test should pin 10 to the card, not to the docstring.
- **Pretrain checkpoints are not "not for end use"**: the cards present them as generation checkpoints at 50 / 5.0.
- **"Pro only quantized on 24 GB" is wrong per the vendor**: their RTX 4090 and 16 GB presets run Pro-distill unquantized with block offload, given enough host RAM.
- **"SR to 1920x1080" is approximate**: the diffusers x2.25 output from 480x864 is 1088x1952. Getting exact 1080p needs a resize step.
- The "beautifier system prompt" is a user-turn instruction pair (`_T2VA/_I2VA_EXPANSION_INSTRUCTION`). It is off by default, unlike in the vendor stack, and it is the shortest of three divergent vendor texts. The ComfyUI `*_system.txt` files are the real prompting guide.
- The "venv lacks Kandinsky6" line is stale, and the version string cannot detect the difference.
- Add these items, missing from the issue: diffusers defaults of 512x768, CFG 5.0 and `sample_audio=True`. Center-crop i2v versus vendor aspect-preserving sizing. Per-checkpoint audio VAE scaling. Pro gating. Weight re-keying, so pin revisions. SR's hard dependence on flex + compile, which makes SR CUDA-only for the Mac onboarding, and Pro infeasible there without transformer quantization.
