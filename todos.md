# todos

## chaining

- Keyframe pre-planning: since fl2va takes first and last keyframes, generate a storyboard of keyframes first, then fill each segment between consecutive pairs. Segments become independent — no cumulative drift, and parallelizable. This is arguably a better long-video strategy than sequential chaining
- Anti-drift correction: histogram/color matching each segment's frames back to segment 0 — cheap, addresses the best-known failure mode of autoregressive video chaining
- Per-segment prompt scheduling: "prompt": ["intro shot...", "then the camera...", ...] — one prompt per segment, for narrative arcs (falls out of the chain loop almost for free)
- Chained prompt-embed reuse — LTX2I2VChained only. chain.py:255 re-runs the full prompt through the 14 GB Gemma once per segment; at 3 segments that's two redundant encodes per run. Engine change, not config.

## performance

- Save a pre-quantized checkpoint. The 45 s SDNQ pass re-quantizes identical weights on every cold start. Save once locally, point model_name at it, and cold starts drop to plain weight loading. Also speeds the REPL's first load. This is the one real remaining structural win for non-REPL use.
- save compiled checkpoint like above
- torch.compile with repeated_blocks. Attacks the 25 s denoise across 48 repeated blocks. It only became viable when the transformer went resident — compile and group-offload hooks fight each other, and that's gone now. But first-run compilation costs more than it saves, so it only pays off paired with #1, where the graph survives between runs.

## ltx 2.5 round-out

Priority ranked. Assessment: we shipped t2v + i2v + chaining out of ~11 native
capabilities. Most of the gap was two generic engine limits, not per-model work.

1. [x] **Cache blocks for LTX-2.5.** `LTX2VideoTransformerBlock` registered in
   [cache_blocks.json](dw/cache_blocks.json). Its second returned stream is audio, not
   `encoder_hidden_states`, so the registry grew an `encoder_hidden_states_argument_name`
   remap - without it a skipped block feeds the text embeddings back as the audio
   stream (pinned by a test that reproduces exactly that). Of little use on the
   distilled model's 8-step schedule; it is there for anything longer.
2. [x] **Frames and audio across a step boundary.** `Result.get_artifact_properties`
   reads object artifacts by attribute, so `previous_result:step.frames` works on the
   AudioVideo a task or a chain produces. Two tasks carry the pieces: `video_frames`
   (frames as one 0-255 array, the shape conditions want) and `pair_audio` (puts a
   soundtrack back beside frames a frames-only step returned). Ships LTX2TwoStage.json.
3. [x] **Argument objects built from named fields.** `from_arguments` in
   [arguments.py](dw/arguments.py) constructs a type from the arguments it names, for
   types with no `from_file()` and no media `kind` - LTX-2's conditions and IC-LoRA
   references. Defers to `build_objects` when one of those arguments names a step.
   Ships LTX2Keyframes.json, LTX2Extend.json, LTX2ICLora.json.
4. [-] **Diffusion decoder.** Built, run, dropped. `LTX2VideoDiffusionDecodePipeline`
   works, but two things make it a bad deal on 24GB: its first three stages run on the
   full volume by design (only stage 4 and the diffusion blocks tile) and the attention
   mask they build is quadratic in the output grid - 70GiB at 1536x896x121, unaffected
   by tile size - and a step run with `output_type: "{latent}"` also returns audio
   latents that nothing outside a pipeline call can vocode, so the flow is silent.
   Base resolution would fit, but that is a silent clip at the same size as LTX2.json.
   Documented in RECIPES_24GB. The per-component `enable_tiling` this exposed is kept:
   the engine could only tile a component literally named 'vae' before.
5. [x] **Prompt enhancement.** LTX2I2VEnhancePrompt.json declares `google/gemma-4-E2B-it`
   as the `prompt_enhancer` plus its `processor`, image-conditioned off the reference
   frame. Quantized `uint4` because the pipeline moves the enhancer onto the accelerator
   and never moves it back.
6. [-] **Non-distilled example.** Dropped: `transformer_full` is ~38GB in bf16 and its
   guidance knobs cost three transformer passes per step, so it is not a 24GB
   configuration. The knobs are documented in RECIPES_24GB; no example ships them.
7. [x] **Auto duration.** LTX2I2VEnhancePrompt.json omits `num_frames` and gives the
   duration head `min_seconds` / `max_seconds` instead.
8. [x] **Docs.** LTX-2.5 section in RECIPES_24GB, `from_arguments` and the
   frames/audio hand-off in WORKFLOW_GUIDE, both tasks in TASKS, the dual-stream cache
   block in ACCELERATION.

### left over

- **Stage-2 refine.** The full LTX flow re-denoises the upsampled latents at
  `STAGE_2_DISTILLED_SIGMA_VALUES` with `noise_scale: 0.909375` (the standard pipeline
  defaults it to 0.0, so it must be passed). Blocked on the batch dimension: a step run
  with `output_type: "{latent}"` pairs its latents with the audio latents per batch item,
  which drops the leading axis every pipeline's `latents` argument expects. The
  `previous_result:step.frames` route keeps it, but then the audio latents have no
  decoder of their own - `audio_vae` + `vocoder` are only reachable inside a pipeline
  call. Needs either a standalone audio decode step or batch-preserving latent artifacts.
- **HDR** (`LTX2HDRPipeline`). Still deferred - needs pre-computed connector embeddings
  from a safetensors file we have no path to produce.
- [x] **Released pipelines did not give their VRAM back.** Found and fixed.
  `populate_from_pretrained_arguments` loaded each declared sub-component *into the
  step's own definition dict* (`from_pretrained_arguments[name] = component`), and the
  definition belongs to the workflow, which outlives every step - so `release_pipeline`
  freed the wrapper while the weights stayed reachable. LTX2ICLora.json OOMed with
  22.6GiB allocated: two 22B transformers alive at once. It only showed up on workflows
  that declare sub-components, which is why an sd15 pipeline (model_name only) released
  cleanly and hid it. Both mutation sites now copy; a second load also keeps its
  'model_name', which load_component used to consume out of the definition.
  Confirmed end to end: LTX2TwoStage.json now drops from 13.3GB to 2.8GB at its
  release boundary, where it used to hold 12.2GB. LTX2ICLora.json keeps sharing its
  transformer between the two steps - that is now a preference (it skips a second
  38GB load) rather than the workaround it was.

- **Gated repo.** `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` is behind a
  license click-through, accepted on this machine as of 2026-08-18.
  `google/gemma-4-E2B-it` needs nothing.

## introspection

- Task argument discovery needs a design decision. Task handlers read their
  arguments via dict lookups inside the function body, so their argument sets
  are not signature-visible the way pipeline/config classes are. Options to
  weigh: hand-written metadata on the @register_command registry (explicit,
  another thing to keep in sync) vs a docstring convention the introspection
  layer parses (cheaper to author, easier to drift). Whichever wins, the
  editor's task-step forms consume it through the same describe/classes API.
