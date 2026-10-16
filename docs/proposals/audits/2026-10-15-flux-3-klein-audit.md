# FLUX 3 Image Klein: vendor knowledge for onboarding

Researched 2026-10-15, the vendor's launch day. Every source below is less than 72 hours old, and the vendor repos were still taking README commits at 22:58Z. Re-check the volatile items (marked **[volatile]**) before you pin anything.

Naming: the vendor calls the model "FLUX 3 Image Klein", not "FLUX.3 [klein]". Diffusers calls the family "FLUX.3", with classes named `Flux3*`.

## Sources

| URL | Type | Date | Contributes |
|---|---|---|---|
| https://huggingface.co/black-forest-labs/flux-3-image-klein (+ README via `/resolve/main/README.md`, readable without auth) | primary | repo created 2026-10-13, last modified 2026-10-15T22:57Z | Variant table (16 steps, no guidance), "do not use without upsampler", Diffusers `ModularPipeline` snippet, VRAM table, box format, license |
| https://huggingface.co/api/models/black-forest-labs/flux-3-image-klein | primary | 2026-10-15 | `gated: auto`, `library_name: diffusers`, file list (no `model_index.json`; has `modular_model_index.json`), 5,548,681,984 BF16 params |
| https://huggingface.co/black-forest-labs/flux-3-image-klein-base | primary | created 2026-10-13 | Undistilled model: 50 steps, CFG 3.5, for LoRA and fine-tuning; FP8 VRAM 29 GB, NVFP4 26 GB |
| https://huggingface.co/black-forest-labs/flux-3-image-klein-pup | primary | created 2026-10-14, sha 3f3edb52 | Prompt upsampler: Qwen3.5-9B fine-tune, Apache-2.0, not gated, 18.8 GB; `system_prompts.json` (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`); decoding settings |
| https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy | primary | created 2026-10-14 | int8/NVFP4/BF16 single files; int8 runs on 8 GB GPUs; VAE file is `flux2-vae.safetensors`; sampler defaults |
| https://github.com/black-forest-labs/flux-3-image-klein (commit bbfe317d) | primary | created 2026-10-15 | Reference code: `src/flux3/pipeline.py` constants (MAX_REFERENCES=10, sides 64-4096, alpha 6.93), CLI defaults, example JSONs holding the upsampled captions, base-vs-distilled timing |
| https://bfl.ai/blog/flux-3-image-klein | primary | 2026-10-15 | Announcement: 5.6B, 10 refs, 4K, Elo against FLUX.2 [klein] 9B, int8 on 8 GB RTX, LoRA via Diffusers / AI Toolkit / NeMo |
| https://docs.bfl.ml/flux_3/flux3_image_klein.md | primary | undated (launch) | Spec table (lists distilled "Guidance (CFG) 1"), LoRA training config, LoRA-on-distilled scale 0.75 |
| https://docs.bfl.ml/flux_3/flux3_image_bounding_boxes.md | primary | undated | Box format: `[y0,x0,y1,x1]` on a 0-1000 grid; edit rows `from`/`src_bbox`/`tgt_bbox`; kinds new/move/anchor |
| https://docs.bfl.ml/guides/prompting_summary.md and `prompting_unified_basics`, `_building`, `_style`, `_technical`, `prompting_layout`, `prompting_editing_*` | primary | undated | Prompting guide. **Written for the FLUX 3 Image API model**, which upsamples on the server side; covers quoting text, refs by position, no negative prompts, aspect ratios |
| https://docs.bfl.ml/api-reference/utility/generate-an-image-with-flux-3-klein.md | primary | undated | API: `steps` 1-16 (default 16), `seed`, `aspect_ratio` (default `auto`), up to 10 `images`; "pixel restoration" is API-only |
| https://github.com/huggingface/diffusers/pull/15046 (merge commit 7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941) | primary (library) | merged 2026-10-15T16:30Z | Adds `src/diffusers/modular_pipelines/flux3/`, `Flux3Transformer2DModel`, `BFLQuantizationConfig`, `Flux3LoraLoaderMixin`, DreamBooth LoRA scripts |
| https://github.com/huggingface/diffusers/blob/7ecb1e1b4f/docs/source/en/api/pipelines/flux3.md (rendered at https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux3) | primary (library) | 2026-10-15 | Diffusers docs: defaults, guider, refs `<ref_image_N>`, memory, single-file and quantized loading |
| https://github.com/huggingface/diffusers/blob/7ecb1e1b4f/docs/source/en/quantization/bfl.md | primary (library) | 2026-10-15 | fp8r needs CC >= 8.9 to run fused; NVFP4 needs Blackwell + flashinfer; on other devices weights are dequantized per forward pass |
| https://github.com/huggingface/diffusers/pull/15058 | primary (library), **open** | opened 2026-10-15T22:32Z | `Flux3PromptUpsampleStep`, plus the constants `SYSTEM_PROMPTS` and `PROMPT_UPSAMPLER_REPO` in `diffusers/modular_pipelines/flux3/prompt_upsampler.py` (not merged) |
| https://huggingface.co/api/models?filter=base_model:adapter:black-forest-labs/flux-3-image-klein-base | secondary | 2026-10-15 | First community LoRAs, e.g. `Muinez/flux3-klein-distil-4step` and `diffusers-modular/flux3-tarot-lora` |
| Web search "FLUX 3 Image Klein" | secondary | 2026-10-15 | **No independent coverage yet.** Every result is about FLUX.2 [klein] (Jan 2026). |

Fetch failures:
- `raw/main/model_index.json`: 401. The file does not exist; the repo ships `modular_model_index.json` instead.
- `modular_model_index.json`, `transformer/config.json` and the other config files: 401. They are gated, and there is no local HF token.
- `github.com/black-forest-labs/flux-3`, linked from the diffusers docs (flux3.md line 175): does not resolve. The real repo is `flux-3-image-klein`.

## Model facts

- **Parameters.** The transformer is 5.6B by the vendor's count, 5,548,681,984 by the Hub safetensors count, and 5.5B in the diffusers docs (flux3.md:203). The Qwen3-VL-4B-Instruct text encoder (~4B) is not counted in that figure.
- **Variants** (HF card table):

  | Variant | Repo | Weights | Steps | Guidance | Built for |
  |---|---|---|---|---|---|
  | Klein (distilled) | `flux-3-image-klein` | step-distilled flow map, bf16 | 16 | none | generation and editing; the model behind the BFL API |
  | Klein Base | `flux-3-image-klein-base` | undistilled | 50 | CFG 3.5 | fine-tuning, LoRA, research |

  Both repos carry `variants/fp8r` and `variants/nvfp4`. The vendor names no size other than 5.6B.
- **Siblings on the Hub (org listing):**
  - `flux-3-image-klein-pup`: the prompt upsampler.
  - `flux-3-image-klein-comfy`: ComfyUI repack.
  - `flux-3-action-{base,droid,so101}`: a 7B robot-action model released 2026-09-22. It is irrelevant here.
  - "FLUX 3 Image", the full model, is API-only. No weights.
- **License.** Weights are under the FLUX Kommunity License v1.0: non-commercial use, plus commercial *output* use for Qualifying Users under US$5M gross annual revenue. The text encoder and the upsampler are Apache-2.0. The reference code is Apache-2.0.
- **Gating.** Both weight repos are gated with `gated: "auto"`: you click-accept and access is granted automatically. The pup repo is not gated. `hf auth login` plus accepting the terms on both the Klein and Klein Base repos is required.
- **Text encoder.** Qwen3-VL-4B-Instruct, class `transformers.Qwen3VLForConditionalGeneration` with an `AutoProcessor` tokenizer. Embeddings stack hidden layers (4, 8, …, 32).
- **VAE.** `AutoencoderKLFlux2`, the same FLUX.2 autoencoder: the comfy file is `vae/flux2-vae.safetensors`, and diffusers flux3.md:17 agrees. 8x downsample with 2x2 packing gives a 16 px token. Latent channels 32 x 4 = 128.
- **Tasks.** One model covers text-to-image, single-reference edit, multi-reference composition (up to 10 refs), bounding-box layout and box edits, and text rendering.
- **Resolution.** "Up to 4K" (GitHub README, BFL docs). Reference code: `MIN_SIDE, MAX_SIDE = 64, 4096` (`src/flux3/pipeline.py:26`). Default output is 1024x1024.
- **VRAM, as the vendor states it:**
  - B200, 1024², upsampler off: distilled 31 GB bf16 / 26 GB FP8 / 25 GB NVFP4; base 31 / 29 / 26 GB.
  - The upsampler adds ~28 GB.
  - BFL docs say it "fits on consumer GPUs, down to an RTX 3090".
  - Blog and comfy card: int8 runs on 8 GB RTX (3070) in ComfyUI.
  - Diffusers: ~20 GB of bf16 weights in total, with the VAE in fp32 (flux3.md:203).
- **Speed.** H200 at 1 MP: about 3 s distilled vs 12 s base (GitHub README). RTX 4070 int8 distilled, 1024², 16 steps: 8.9 s (blog chart).
- **Quantized checkpoints from the vendor:**
  - fp8r (rowwise FP8) and NVFP4 (ModelOpt layout), under `variants/` in both repos.
  - Comfy int8 "convrot" files for the DiT, the text encoder and the upsampler.
  - NVFP4 needs Blackwell. fp8r runs fused only on CC >= 8.9 (Ada and newer); the RTX 3090 (8.6) falls back to per-forward dequantization (bfl.md).

## Pipeline facts (diffusers @ 7ecb1e1b4f, PR #15046)

**Local install.** `venv/.../diffusers` is 0.41.0.dev0 at commit 1e004eb6. It has `pipelines/flux`, `pipelines/flux2` (`Flux2Pipeline`, `Flux2KleinPipeline`, `…KleinInpaint`, `…KleinKV`) and `modular_pipelines/flux2`. It has **no flux3**. Reinstall diffusers from main at or after 7ecb1e1b. Local transformers 5.19.0 already has `Qwen3VLForConditionalGeneration` and `Qwen3_5ForConditionalGeneration`.

**Klein does NOT load through `Flux2Pipeline` or `Flux2KleinPipeline`.** It needs a new transformer (`Flux3Transformer2DModel`) and a new text encoder. It ships **only as a modular pipeline**: there is no `pipelines/flux3` and no `DiffusionPipeline` class. (The HF sidebar snippet's `DiffusionPipeline.from_pretrained` is the Hub's auto-generated snippet; the card itself uses `ModularPipeline`.)

**Classes and modules:**
- `src/diffusers/modular_pipelines/flux3/modular_pipeline.py:24`: `Flux3ModularPipeline(ModularPipeline, Flux3LoraLoaderMixin)`. Its default blocks are `Flux3AutoBlocks`.
- `modular_pipeline.py:52`: `Flux3DistilledModularPipeline`. Its default blocks are `Flux3DistilledAutoBlocks` (`modular_blocks_flux3_distilled.py:256`).
- `modular_pipelines/modular_pipeline.py` maps the `"flux3"` and `"flux3-distilled"` model names. Per diffusers docs flux3.md:161, the repo's `modular_model_index.json` resolves to the distilled classes.
- Load with `ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein")`, then `pipe.load_components(dtype=torch.bfloat16)`.

**trust_remote_code.** Inferred: not needed. Every class named is in the diffusers library, and the file list has no `.py` files. The `modular_model_index.json` itself could not be read (gated). Verify after you accept the gate.

**Components** (`modular_blocks_flux3_distilled.py:265-268`):

| Component | Class |
|---|---|
| `text_encoder` | `Qwen3VLForConditionalGeneration` |
| `tokenizer` | `AutoProcessor` |
| `image_processor` | `Flux3ImageProcessor` (from_config) |
| `vae` | `AutoencoderKLFlux2` |
| `transformer` | `Flux3Transformer2DModel` (`models/transformers/transformer_flux3.py:341`, `flow_map` config at :406) |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` (from_config, shift 6.93) |
| `guider` | `ClassifierFreeGuidance`, base only |

**Call inputs and defaults** (`Flux3DistilledAutoBlocks` docstring, `modular_blocks_flux3_distilled.py:270-304`):
- `prompt`: a **str only**. Batched prompts raise (`encoders.py:229`). Use `num_images_per_prompt` (default 1) for several images.
- `max_sequence_length`: 8192 (`encoders.py:196`). The text window is right-padded to a multiple of 80 tokens (`encoders.py:166`). The padding "is part of the model recipe".
- `text_encoder_out_layers`: `(4, 8, 12, 16, 20, 24, 28, 32)`.
- `image`: a PIL image or a list of them. Non-PIL input raises (`encoders.py:84`).
- `height` / `width`:
  - Text-to-image default is 1024 x 1024 (`before_denoise.py:186`).
  - With `image`, the default is 1 MP at the **first reference's** aspect ratio, rounded to 16 (`inputs.py:246, 302-309`).
  - Both must be divisible by 16 or the call raises (`before_denoise.py:232`).
- `num_inference_steps`: 16 distilled (`modular_blocks_flux3_distilled.py:48,116`), 50 base (`before_denoise.py:95`). The default sigmas are `np.linspace(1.0, 1/n, n)` with scheduler `shift=6.93` (`before_denoise.py:95,151`).
- Also accepted: `timesteps` and `sigmas` (overrides), `latents`, `generator`, `output_type` (`pil`).
- `output="images"` selects the return value.

**Guidance:**
- Distilled: no guider, no `negative_prompt`. The transformer is a flow map that gets the current and the target timestep at each step (flux3.md:161; `transformer_flux3.py:546`).
- Base: real CFG through the `guider` component, `ClassifierFreeGuidance(guidance_scale=3.5)` (`denoise.py:48`). **There is no `guidance_scale` call argument.** You change it with `pipe.guider = pipe.guider.new(guidance_scale=…)` (flux3.md:151). The base model takes `negative_prompt`.
- Diffusers says the distilled model "is trained for 4 to 16 steps" (flux3.md:171). The API caps `steps` at 1-16.

**References:**
- Each reference keeps its own resolution. It is downscaled only above 4096x4096 and center-cropped to the 16 px grid (`encoders.py:88`).
- In prompts, refer to references as `<ref_image_0>`, `<ref_image_1>`, … (flux3.md:108).
- **Diffusers enforces no maximum count.** The reference code enforces `MAX_REFERENCES = 10` (`src/flux3/pipeline.py:27`), and the vendor states 10 everywhere.

**Reference handling differs between diffusers and the vendor code:**
- **Vendor, single reference:** pads the image to a canvas of at most 4 MP (`MATCH_MAX_PIXELS`), then crops the output back. The output size defaults to the reference's own size.
- **Vendor, multi-reference:** each reference is resized to at most 1 MP.
- **Diffusers:** keeps references at up to 16 MP and defaults the output to 1 MP.

**Prompt upsampler / enhancer helper:**
- **Not in merged diffusers.** Open PR #15058 adds `Flux3PromptUpsampleStep` and the module constants `SYSTEM_PROMPTS` (a dict of `t2i`/`i2i`/`t2i_bbox`/`i2i_bbox`) and `PROMPT_UPSAMPLER_REPO = "black-forest-labs/flux-3-image-klein-pup"`, in `diffusers/modular_pipelines/flux3/prompt_upsampler.py`. The PR copies `system_prompts.json` at pup commit 8b069751. The pup head is now 3f3edb52 **[volatile]**.
- Until that merges, the vendor text lives only in the pup repo's `system_prompts.json` and in the reference repo's `src/flux3/pup_prompts.json`. Pin to the diffusers constant once it lands, rather than copying the text.

**Quantized loading:**
- `BFLQuantizationConfig(compute_dtype=torch.bfloat16)` works **only through `Flux3Transformer2DModel.from_single_file(…/variants/{fp8r,nvfp4}/model.safetensors, config=<repo>, subfolder="transformer")`**, followed by `pipe.update_components(transformer=…)`. Pipeline-level loading of these checkpoints is unsupported (bfl.md).
- Load the other components by name first. A bare `load_components()` would also load the bf16 transformer.

**Other library facts:**
- **LoRA:** `Flux3LoraLoaderMixin` is on the pipeline. DreamBooth LoRA scripts are at `examples/dreambooth/train_dreambooth_lora_flux3_klein{,_img2img}.py`.
- **`_repeated_blocks`:** `["Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"]`, which is usable for regional compile.
- **Memory:** flux3.md:208 suggests `pipe.enable_auto_cpu_offload(device="cuda")`. The base-model docs use VAE fp32 (`dtype={"vae": torch.float32, "default": torch.bfloat16}`); the distilled card uses bf16 throughout.

## Vendor recommended settings

| Variant | Steps | Guidance | Resolution | Source |
|---|---|---|---|---|
| Klein (distilled) | 16 (trained for 4-16; API 1-16) | none. BFL docs list "CFG 1" (equivalent to none), and the reference code raises on anything but None/1.0 | 1024² default; up to 4K; sides 64-4096, multiple of 16 | HF card table; `src/flux3/generate.py`; docs.bfl.ml klein page; flux3.md:171 |
| Klein Base | 50 | CFG 3.5 | same | HF card; CLI `--model` help; `pipeline.py:20` |
| LoRA on distilled | 16 | CFG 1 | — | LoRA scale 0.75 (0.5-1.0), docs.bfl.ml klein page |
| LoRA training | rank/alpha 32/32, AdamW (0.9, 0.95), lr 1e-4 with 200 warmup steps then linear decay, 4000 steps, ~1 MP native aspect ratio, train on **Base** | — | — | docs.bfl.ml klein page |
| Upsampler decoding | greedy, `max_new_tokens=3584`, `enable_thinking=False` | — | — | pup card (note below) |

The upsampler decoding note: the pup card also says "top_p 0.95, top_k 20, presence penalty 1.5 for t2i / 0 for edits". That reads as contradictory alongside greedy decoding; the card's own snippet uses `do_sample=False`.

Vendor example sizes are 1024x1024, 832x1248 (portrait) and 1248x832 (landscape) (`examples/README.md`). The vendor does not state a scheduler shift; it is alpha 6.93 in `pipeline.py:63` and in diffusers.

## Prompt format

- **Mandatory upsampling.** "Do not use the model without a prompt upsampler" (HF card; GitHub README `[!IMPORTANT]`). The model is trained on the captions the upsampler writes, and short prompts give generic output.
  - **Text-to-image:** the upsampler (`t2i` system prompt) expands the request into one English scene caption, about 200 words in the vendor examples (e.g. `examples/distilled/man_dog.json`). The caption covers medium/style, framing, subject with colors, materials and positions, background and lighting.
  - **Edits** (`i2i`): the user message is the reference images followed by a three-line text: "Reference images follow in order (the first image, the second image, ...).\nUser request: …\nOutput the caption." The output is a complete caption of the reference with the edit folded in. Pass the same images to both the upsampler and the pipeline.
  - **Any other LLM** may stand in, given these system prompts.
- **Natural language, not tags.** Describe attributes (palette, stroke, texture) rather than naming an artwork (HF card *Prompting*).
- **Text rendering.** Quote the exact words, then say where they go and how they look (docs.bfl.ml `prompting_unified_basics`; the upsampler keeps quoted text verbatim, per the pup card).
- **No negative prompts.** Rephrase each negation as a positive description (docs.bfl.ml `prompting_unified_technical`). The distilled pipeline also rejects `negative_prompt`.
- **Layout / JSON.**
  - Write the caption with `<id>` tags, then append a JSON array of `{"id","bbox":[top,left,bottom,right] on 0-1000,"desc"}`.
  - For box edits, add `from: "ref_image_N"`, `src_bbox` and `tgt_bbox`. A `null` target removes the element; equal boxes anchor it.
  - Boxes are a hint, not a hard mask.
  - Sources: HF card; https://docs.bfl.ml/flux_3/flux3_image_bounding_boxes; flux3.md:122-147.
  - The reference CLI's `boxes.json` uses 0-1 floats and converts them (`bbox.to_canvas`).
- **Naming references.** Diffusers and the vendor box format use `<ref_image_0>` (zero-based). The BFL API guide uses "image 1", "image 2" (one-based). The upsampler input uses "the first image, the second image". Use the convention of whichever layer the prompt is going to.
- **Scope of the API guide.** Its prompting guide (https://docs.bfl.ml/guides/prompting_summary.md) is written for the FLUX 3 Image *API*, where a short prompt is expanded on the server side ("A short prompt works"). For local Klein, that advice holds only once an upsampler is in front of the model.

## Missing knowledge / caveats

- **Diffusers support is hours old.** It merged 2026-10-15 16:30Z and is not in any release. The upsampler block (PR #15058) is unmerged. The diffusers docs link a nonexistent `github.com/black-forest-labs/flux-3`.
- **Gated configs unread.** I could not read `modular_model_index.json`, `transformer/config.json` or `variants/*/hf_quant_config.json` (gated, no token). The trust_remote_code status and the exact block mapping are inferred, not observed.
- **No guider support in the repo.** A grep of `docs/WORKFLOW_GUIDE.md` and `dw/pipeline_processors/` found no handling for modular `guider` components, so a Base CFG other than 3.5 has no knob today. The distilled template is unaffected.
- **The vendor's quantized variants need repo support** for `from_single_file` + `update_components` on a modular pipeline. That is unverified for this engine. The fallbacks are SDNQ/TorchAO through `load_components.quantization_config`, which is how the H3 templates quantize.
- **24 GB fit.** ~11.1 GB transformer + ~8.9 GB text encoder + VAE ≈ 20 GB bf16 is tight on 24 GB at 1 MP. The upsampler (18.8 GB bf16; 9.8 GB int8 in the comfy repo) cannot be resident alongside. It must run as a separate prior step, or as a remote LLM. `enable_auto_cpu_offload` (`components_manager`) is CUDA-only per the repo docs.
- **MPS** is completely unstated by the vendor. The fp8r and NVFP4 kernels are CUDA-only; elsewhere they dequantize per forward pass. bf16 on MPS needs ~20 GB of unified memory plus activations.
- **Known failure modes** (HF card *Limitations*):
  - Rendered text can be inaccurate.
  - Boxes are not masks; small boxed elements may not appear, and elements can extend past their box.
  - Edits can change content outside the requested region. API "pixel restoration" fixes this, but it is API-only.
  - A one-line edit instruction sent raw "reframes the shot".
  - The upsampler resizes references to ≤1 MP, so it can misdescribe small details. It never refuses, and its output is English only.
- **Seeds.** Seeds reproduce only when noise is drawn as the vendor draws it (flux3.md:175-185).
- **Text encoder version quirk.** Qwen3-VL padding position ids differ between transformers 4.57 and 5.x. Diffusers passes them explicitly (`encoders.py:269-273`), so do not bypass the pipeline's text encoder step.
- **LoRA.** Train on Base; the LoRA runs on distilled without conversion at scale ~0.75. Trainers: Diffusers DreamBooth scripts, Ostris AI Toolkit (`arch: flux3`), NVIDIA NeMo AutoModel. A LoRA adds ~5% to generation time (blog). Community LoRAs exist from day one, including a "4-step distil" LoRA by Muinez, not vendor-verified.
- **FLUX.2 vs FLUX.3.** FLUX.2 [klein] (4B Apache / 9B non-commercial, released 2026-01-15) is the previous release.
  - It has its own diffusers classes (`Flux2KleinPipeline`, already local). The blog claims Klein beats FLUX.2 [klein] 9B by +91 Elo t2i and +70 edit with 38% fewer parameters (internal evaluation).
  - Shared with FLUX.2: the VAE, 10 references, and "up to 4MP"-style editing.
  - Different from FLUX.2: the transformer, the text encoder (Qwen3-VL-4B), mandatory upsampling, bbox prompts, and the license (Kommunity rather than Apache/non-commercial).
  - Do not carry FLUX.2 [klein] settings or guides over; the BFL docs' FLUX.2 pages (training, HEX/JSON prompting use cases) are about the earlier release.
- **No independent secondary coverage** exists yet. Every claim above rests on vendor or diffusers sources alone.

## What carries over from `workflows/models/flux2-dev.json`

- **Carries over:** the result block (`image/jpeg`, `embed_metadata`), `seed: variable:seed`, `cost_drivers`, `configures: templates/text-to-image`, and the `hf auth login` gating pattern.
- **Differs:**
  - `component_type` becomes `ModularPipeline` (as in the H3 templates), not `Flux2Pipeline`.
  - There is no `offload`; use `components_manager` / on-demand residency instead.
  - Per-component dtype and quantization go through `load_components`.
  - There is no `guidance_scale` argument (distilled has none; base uses the guider).
  - `output: "images"`.
  - A prompt-upsampler step must precede the generation step.
  - Steps are 16, not 40.
  - The text encoder is a transformers Qwen3-VL model and is not pre-quantized by the vendor except in comfy int8.
- **Repo conventions** (RECIPES_24GB, ACCELERATION, QUANTIZATION):
  - Unquantized FLUX uses `offload: "model"`, which modular pipelines cannot use.
  - `mag_ratios: "flux"` is a FLUX.1 preset and is not valid for Flux3; calibrate instead.
  - `is_guidance_distilled` (FasterCache) is FLUX.1 guidance embedding, which does not apply here. The distilled Klein is a flow map with no CFG batch.

## What the templates should teach (reading order)

1. **text-to-image (distilled)**
   - Setup: `black-forest-labs/flux-3-image-klein`, 16 steps, 1024², bf16, no guidance.
   - The prompt is already a full caption, or comes from an upsampler step.
   - Vendor example: "A man walking with his dog, afternoon sun", seed 7 (`examples/README.md`); the expanded caption is in `examples/distilled/man_dog.json`.
2. **upsample-then-generate**
   - Two steps: a `text_generation` step with `flux-3-image-klein-pup` (`Qwen3_5ForConditionalGeneration`, system prompt `t2i`, greedy, `max_new_tokens` 3584, `enable_thinking=False`), whose output is passed by `previous_result:` as `prompt`.
   - Unload between the steps on 24 GB.
   - Pin the system prompt to diffusers `SYSTEM_PROMPTS` once PR #15058 merges.
   - Source: HF card *Prompt upsampling*.
3. **image edit (single reference)**
   - `image` = one asset; the `i2i` upsampler caption gets the same image; 16 steps.
   - Output defaults to 1 MP at the reference's aspect ratio.
   - Vendor examples: "put a red bandana around the dog's neck" (seed 22), "fill the mug with coffee and put a croissant next to it" (seed 21).
4. **multi-reference**
   - 2-10 images, referred to as `<ref_image_N>`; the first image sets the aspect ratio.
   - Vendor example: "print the dog from the second image onto the mug" (seed 23). Diffusers example: fox + bench at 1360x768.
5. **layout with bounding boxes**
   - Caption with `<id>` tags plus a JSON box table on 0-1000.
   - Vendor examples: the silhouette example (HF card) and the balloons boxes (`examples/inputs/balloons_boxes.json`). Docs: docs.bfl.ml bounding-box page.
6. **typography**
   - Quoted text, 832x1248 portrait.
   - Vendor example: the gig poster, seed 1 (`examples/typography/`).
7. **base + CFG** (optional, for LoRA users)
   - `flux-3-image-klein-base`, 50 steps, guider 3.5, `negative_prompt` allowed.
   - The repo needs guider support to vary the CFG.
8. **LoRA on distilled**
   - A Base-trained LoRA at scale 0.75, 16 steps (docs.bfl.ml klein page).
9. **low-VRAM 24 GB CUDA**
   - Option A, the vendor fp8r `from_single_file` + `BFLQuantizationConfig`: fused on Ada+ only, dequant fallback on a 3090. Needs engine support.
   - Option B, SDNQ/TorchAO int8 through `load_components.quantization_config` on the transformer and text encoder (H3 pattern).
   - With either option, `components_manager` auto-offload and the upsampler as a separate step. Measure; the vendor gives no 24 GB recipe.
10. **Apple MPS**
    - bf16 with no vendor quantization (the kernels are CUDA-only), 1024² or lower.
    - Upsampler run separately or replaced by another LLM.
    - The vendor says nothing about MPS; it must be measured.

## Assessment

**Trust as-is.** These facts are consistent across the HF card, the reference code, the BFL docs and the diffusers source:
- Distilled: 16 steps, no CFG. Base: 50 steps, CFG 3.5. Shift 6.93.
- 1024² default; sides a multiple of 16, at most 4096.
- Up to 10 references (vendor-enforced; diffusers does not enforce it).
- Qwen3-VL-4B text encoder and the FLUX.2 VAE.
- `max_sequence_length` 8192; prompt must be a single string.
- No negative prompt on distilled.
- The bbox format.
- The mandatory upsampler and its decoding settings.
- The LoRA recipe (train on Base, scale 0.75 on distilled).
- The class names: `Flux3DistilledModularPipeline` / `Flux3ModularPipeline`, loaded via `ModularPipeline.from_pretrained`.

**Uncertain:**
- The upsampler system prompts are still moving (pup repo updated hours ago; the diffusers PR pins an older commit). Do not pin text yet.
- Diffusers and the vendor code prepare references differently (1 MP vs native-size output default; 16 MP vs 4 MP/1 MP reference cap). Expect diffusers output to differ from the vendor's for edits.
- The distilled guidance label differs: "none" (card) vs "CFG 1" (BFL docs). They are equivalent.
- 24 GB and MPS fit are unmeasured. The vendor's VRAM figures are B200 peaks.
- trust_remote_code: inferred not needed; the gated index is unread.

**Blockers:**
1. **The local diffusers has no flux3.** Diffusers must be updated to main ≥ 7ecb1e1b (merged today; unreleased).
2. **Gated weights.** Accept the Kommunity license on the Klein and Base repos, and set an HF token on the server.
3. **The upsampler.** A faithful template needs either a 9B upsampler step (18.8 GB bf16) or a different LLM given BFL's system prompts. The library constant for those prompts is not merged yet.
4. **Engine gaps.** The vendor fp8r/NVFP4 variants and a non-default Base CFG require engine support for modular `from_single_file`/`update_components` and the `guider`. Neither is present today.
