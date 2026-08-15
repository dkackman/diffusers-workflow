# TODO — Code Review Findings (2026-08-15)

Findings from a full-project correctness/quality review, adversarially verified against the
code and the installed dependencies (diffusers 0.40.0.dev0, accelerate 1.14, torch 2.13).
Line numbers are as of commit `12c6c8d`. Priority-ranked: P0 = silent wrong results or
security, P1 = crashes/broken features, P2 = performance, P3 = design/extensibility debt,
P4 = cleanup.

Checked-off items should also be removed from this file once verified fixed (with a test
where feasible).

## Progress (2026-08-15)

**DONE — P0 (all):** #1 `99fa7ba`, #2 `6ff9621`, #3+#4 `6896b41`, #5 `5e617f2`,
#6 `72a2f76`, #7 `446e454`.
**DONE — P1:** #9+#10 `2631c54`, #11+#13 `2f173b3`, #12+#14 `6896b41`, #15 `9b8274e`,
#16 `5e617f2`. #8 in progress.
**DONE — P1 (all):** #8 `d8b423c`.
**DONE — P2 (all):** #17 `3148d12`, #18 `0224885`, #19 `a26074e`, #20 (Phase T,
`0a185ba`), #21 `6e1d00c`.
**DONE — P3/P4 (all) + S7/S9:** #22-#33 and the staged-diff follow-ups, committed
2026-08-15 (see git log). Every finding from the 2026-08-15 review is now addressed;
this file remains as the record and for the open question below.

**Open question (from #21 work):** vendored `dw/tasks/rife_model.py` block0 is declared
for 23 input channels but its forward concatenates 39 (3+3+16+16+1) — flagged by an
agent, unverified; if real, some IFNet path would crash at runtime. Verify against
upstream RIFE v4.6 before touching.

---

## P0 — Security & silently wrong results

### 1. `gather` task bypasses path validation entirely
**File:** `dw/tasks/gather.py:36` (images), `:91` (videos)
`gather_images`/`gather_videos` pass the workflow-supplied `glob` straight to `glob.glob()`
and `load_image`/`load_video` with no `validate_path()` and no
`ALLOWED_IMAGE_EXTENSIONS`/`ALLOWED_VIDEO_EXTENSIONS` check — `ALLOWED_IMAGE_EXTENSIONS` is
imported at line 8 and never used. No upstream layer validates the `glob` key either:
`task.py` handlers splat `**arguments`, and `realize_args` only inspects keys named
`image`/`video`/`*_image`/`*_video`. Relative globs resolve against the process cwd, so
`"glob": "../../../home/*/Pictures/**/*.jpg"` reads files outside the workflow dir, violating
the security layer's documented guarantee.
**Fix:** validate each glob match with `validate_path()` + extension allowlist before loading
(mirror `arguments.py` `fetch_image`/`fetch_video`, or just call them per matched path).

### 2. TeaCache silently corrupts output under Flux true CFG
**File:** `dw/teacache.py:185`
The step counter `cnt` assumes one transformer forward per denoising step, but diffusers
`FluxPipeline` with `negative_prompt` + `true_cfg_scale > 1` calls the transformer **twice**
per step (cond ~line 899, uncond line 916 in `pipeline_flux.py`). `cnt` wraps at
`num_inference_steps` mid-run, the forced-compute first/last-step logic fires at wrong
timesteps, and `previous_modulated_input`/`previous_residual` are shared across cond/uncond
passes — corrupted images, no error.
**Fix:** detect the double-forward case (e.g. track calls per timestep value, or key cache
state on the actual timestep tensor) or refuse to enable TeaCache when true-CFG args are
present. TeaCache applies only to `FluxTransformer2DModel` (`_FORWARD_FACTORIES`).

### 3. `"no_generator": false` disables the generator (inverted semantics)
**Files:** `dw/pipeline_processors/pipeline.py:197`, `dw/workflow.py:232`
Both sites use presence-only checks (`"no_generator" not in configuration`) although
`workflow_schema.json` declares `no_generator` as a boolean. `"no_generator": false`
(explicitly requesting a generator) therefore skips generator creation — the pipeline runs
unseeded and the workflow/step seed is silently ignored.
**Fix:** `if not configuration.get("no_generator", False)` at both sites.

### 4. Step-level `seed` is silently ignored
**File:** `dw/step.py:19` (stored, never read); see also `dw/workflow.py:149-155`,
`dw/pipeline_processors/pipeline.py:201`
The schema documents a per-step `seed` ("Default seed for the entire step").
`Step.__init__` stores it in `self.default_seed`, but nothing reads it — `create_step_action`
receives the *workflow-level* seed and the generator seeds from
`pipeline_definition.get("seed", default_seed)` (pipeline-level). Workflow seed 111 + step
seed 222 → generator seeded 111.
**Fix:** thread `step.default_seed` into the seed-resolution chain
(pipeline seed > step seed > workflow seed) on both the fresh-load and cached paths.

### 5. CLI boolean/list variable overrides coerce wrongly
**File:** `dw/variables.py:123`
`get_value` special-cases only the strings `'true'`/`'false'` for bool defaults, then falls
through to `desired_type(v)`: `upscale=0` / `upscale=no` → `bool("0")` → `True` — the user's
"off" silently turns the feature **on**. A list-typed default mangles a string via
`list("cat")` → `['c','a','t']`.
**Fix:** for bool defaults, accept a proper truthy/falsy string set (`0/1/yes/no/on/off`) and
raise on anything else; for list defaults, split on commas or raise.

### 6. REPL: `config set output_dir` silently ignored on cache hit
**File:** `dw/worker.py:139` (`workflow_changed` check), `:163-166` (only place output_dir is
applied)
The worker's changed-check compares only file hash + path. `output_dir` arrives with every
`execute` command but is only applied in the reload branch, so after
`config set output_dir /new/dir` + re-run, results save to the **old** directory while the
REPL reports the new one.
**Fix:** apply `output_dir` to the cached workflow too (`self.current_workflow.output_dir =
output_dir` on cache hit).

### 7. Unknown component keys validate, then are silently dropped
**File:** `dw/pipeline_processors/pipeline.py:20` (`optional_component_names`),
`dw/workflow_schema.json` (`$defs.pipeline` — no `additionalProperties: false`)
Component handling iterates a hardcoded whitelist; the schema allows arbitrary extra keys. A
component outside the list (e.g. a future `text_encoder_4`, `vocoder`) passes validation and
its `quantization_config`/dtype are silently dropped → model loads full-precision and OOMs
with no explanation. The Python list and the schema must currently be maintained in lockstep.
**Fix (either):** add `additionalProperties: false` (or schema-level warning) so unknown keys
fail loudly; better, detect component-shaped keys dynamically (dict containing
`configuration`/`from_pretrained_arguments`/`quantization_config`) instead of a whitelist.

---

## P1 — Crashes / broken features

### 8. TeaCache breaks CPU offload (device-mismatch crash)
**File:** `dw/teacache.py:366`; interaction with `dw/pipeline_processors/pipeline.py:310-313`
(teacache wrap) and `:666` (offload hooks installed at load)
`transformer.forward = teacache_forward.__get__(...)` overwrites the accelerate `CpuOffload`
hook wrapper (accelerate installs it as an instance-attribute `forward`,
`hooks.py:180-200`). The hook's `pre_forward` (which onloads weights) never runs → with
`"offload": "model"` + `"teacache"`, the CPU-resident transformer gets CUDA latents →
RuntimeError. The `finally` restore then re-enables the hook, masking the cause.
**Fix:** wrap the *inner* function accelerate stored (`transformer._hf_hook` /
`._old_forward`) instead of the outermost `forward`, or install TeaCache before offload
hooks, or refuse the combination with a clear error.

### 9. Prompt weighting hardcodes `cuda:0` and breaks offload/multi-GPU
**File:** `dw/prompt_weighting.py:178-189` (`_get_device`), `:217-219` (`.to()` on
offloaded encoders); callers pass no device (`prompt_weighting.py:322`,
`dw/pipeline_processors/pipeline.py:254-255` — even though `self.device` is in scope there)
Two verified failure modes: (a) `DW_DEVICE=cuda:1` + `"offload": "model"` → encoders/tokens
go to `cuda:0` while hooks execute on `cuda:1` → mismatch/OOM on the excluded GPU;
(b) `"offload": "sequential"` → `.to()` on an `AlignDevicesHook` module raises
`NotImplementedError: Cannot copy out of meta tensor` (empirically confirmed).
**Fix:** delete `_get_device`; pass `Pipeline.self.device` down from the call site; never
`.to()` a hook-managed module (check `hasattr(module, "_hf_hook")`) — let the hooks move
weights, or gate prompt weighting against sequential offload with a clear error.

### 10. FasterCache option is guaranteed-broken
**File:** `dw/pipeline_processors/config_objects.py:108`; applied at
`dw/pipeline_processors/pipeline.py:771`
`FasterCacheConfig()` is constructed bare; `current_timestep_callback` (a callable, default
`None`) cannot be expressed in JSON. diffusers 0.40 no longer raises at apply time, but
`FasterCacheDenoiserHook.new_forward` calls `self.current_timestep_callback()`
unconditionally → first inference step dies with `TypeError: 'NoneType' object is not
callable`.
**Fix:** wire the callback at apply time, e.g.
`FasterCacheConfig(current_timestep_callback=lambda: pipe._current_timestep, ...)` — needs
the pipeline reference, so construct it in `enable_cache_on_transformer` where both are
available. Add a test that runs one denoise step with each cache type.

### 11. Audio pipeline outputs crash on save (audio never written)
**File:** `dw/result.py:456`
`get_artifact_list` handles standalone audio via `audio.T.float().cpu().numpy()`, but
`AudioPipelineOutput.audios` is a **numpy ndarray** under the default `output_type="np"`
(AudioLDM2 line 1092, StableAudio line 729 call `.numpy()` before returning) → runtime-
confirmed `AttributeError: 'numpy.ndarray' object has no attribute 'float'` after generation
completes. (The LTX-2 frames+audio muxing branch is unaffected — `as_audio_track` handles
numpy.)
**Fix:** branch on type: torch tensor → `.T.float().cpu().numpy()`; ndarray → transpose with
`.T`/`np.transpose` only. Torch shape note: `audios` is (batch, channels, samples), items are
(channels, samples), so `.T` → (samples, channels) is correct as-is for tensors.

### 12. Cached-pipeline path builds the generator on the wrong device
**File:** `dw/workflow.py:240` (cached path) vs `dw/pipeline_processors/pipeline.py:199-201`
(fresh path, honors `configuration["device"]`)
On REPL re-run the generator is rebuilt with `torch.Generator(device)` using the
workflow-default `get_device()`, ignoring the pipeline's `device` override. Pipeline pinned
to `"cpu"` on a CUDA box: run 1 OK, run 2 → diffusers `randn_tensor` raises
`ValueError: Cannot generate a cpu tensor from a generator of type cuda`.
**Fix:** the cached branch must resolve the device the same way `Pipeline.__init__` does
(`configuration.get("device", device)`) — the cached `Pipeline` wrapper already knows it via
`self.device`; use that. Related to #4 (same seed-resolution chain).

### 13. String step results break `previous_result:step.property` references
**File:** `dw/result.py:138-141` (`get_artifact_properties`)
A string result (e.g. `image_to_text` caption) passes the `isinstance(result, Iterable)`
check; `property_name in result` becomes a substring test. Substring present →
`result[property_name]` raises `TypeError: string indices must be integers`; absent →
returns `[]` silently and the referencing step is skipped with only "no iterations to
execute". Both modes runtime-confirmed.
**Fix:** check `isinstance(result, Mapping)` for the property path; give strings/lists a
clear error ("result of step X has no property Y").

### 14. `torch.seed()` evaluated eagerly, reseeding global RNG every run
**File:** `dw/workflow.py:113`
`workflow_def.get("seed", torch.seed())` — Python evaluates the default eagerly, so
`torch.seed()` runs (and nondeterministically reseeds global CPU/CUDA/MPS RNG,
`torch/random.py:89-104`) on **every** `Workflow.run`, even with an explicit workflow seed.
Clobbers any `manual_seed` done in the persistent REPL worker; affects all paths drawing
from global RNG (`no_generator` pipelines, tasks).
**Fix:** `seed = workflow_def.get("seed"); if seed is None: seed = torch.seed()` — or better,
draw a random seed without reseeding: `torch.randint(0, 2**63 - 1, ()).item()`.

### 15. Multi-dot `previous_result` references crash with unpack error
**File:** `dw/previous_results.py:99`
`result_name, property_name = previous_result_name.split(".")` — step names are unrestricted
strings per the schema, so `previous_result:v1.0.mask` raises
`ValueError: too many values to unpack` with no mention of the offending reference, and a
step literally named `v1.0` mis-resolves into result `v1` + property `0` → KeyError.
**Fix:** `split(".", 1)` + membership check against known step names with a descriptive
error; or restrict step names in the schema (`^[^.]+$`) and validate references.

### 16. Unknown CLI variable → bare KeyError
**File:** `dw/variables.py:86`
`set_variables` dereferences `variables[validated_name]` inside a try that only catches
`SecurityError`. A typo (`promt="a cat"`) surfaces as
`Error running workflow 'wf.json': 'promt'` with no guidance. Sibling inconsistency: if the
workflow declares **no** `variables` block, extra args are silently ignored instead.
**Fix:** membership check with an error listing valid variable names; make the
no-variables-block case behave the same way.

---

## P2 — Performance

### 17. Task models reloaded from disk on every iteration (largest perf win)
**Files:** `dw/tasks/segment.py:49-52,79-80` (GroundingDINO+SAM2, ~3GB per image),
`dw/tasks/diffusion_upscale.py:66` (full SD upscale pipeline per image),
`dw/tasks/zoe_depth.py:9` (`torch.hub.help` — fresh download — per call),
also `upscale.py:52`, `restore_faces.py:57`, `image_to_text.py:41`, `text_generation.py:40`,
and every `controlnet_aux` branch in `image_utils.py`
`step.py:63-69` runs the task handler once per cartesian-product iteration;
`Task.run` dispatches with no cache; `previous_pipelines` (workflow.py:256) caches only
`Pipeline` objects. N inputs = N full model loads: segmenting 20 images loads ~3GB twenty
times; load time dominates inference by minutes and allocator churn can OOM.
**Fix:** module-level cache keyed on `(model_name, device)` per task module (mirror
`previous_pipelines`), plus an unload hook wired into `memory clear` / `_cleanup_between_runs`.

### 18. `get_iterations` deep-copies loaded media per iteration
**File:** `dw/previous_results.py:55`
`copy.deepcopy(argument_template)` inside the product loop, after `realize_args`
(workflow.py:139) has already loaded PIL images / full video frame lists into the template —
each iteration duplicates all pixel data in RAM, and all iterations are materialized up
front. Verified: downstream mutations are only top-level dict ops (`pop`ing image/device,
prompt→prompt_embeds replacement), so a **shallow** per-iteration dict copy is safe.
Previous-result values inserted at line 61 are already shared references.
**Fix:** replace deepcopy with `dict(argument_template)`; optionally make get_iterations a
generator. Add a comment documenting the shallow-copy contract (no in-place mutation of
nested values).

### 19. All step results retained in RAM for the whole workflow
**File:** `dw/workflow.py:158`
`results[step.name] = result` — no eviction; every intermediate image/frame list lives until
`Workflow.run` returns even when no later step references it (artifacts are already on
disk). Long video chains (generate → interpolate → upscale → encode) accumulate GB of dead
frames and starve the host RAM CPU-offloading depends on.
**Fix:** before the loop, scan remaining steps' argument templates for
`previous_result:<name>` refs; after each step, `del` results no longer referenced, then
`gc.collect()` (already called between steps).

### 20. Heavy imports at module load — paid even by `dw.validate` — **DONE 2026-08-15**
Fixed as part of the test-reliability pass (see "Phase T" below): lazy imports in
`image_utils.py` (cv2/controlnet_aux/transformers/zoe/background_remover/depth_estimator),
`task.py` (7 model-backed handlers), `config_objects.py` (diffusers cache configs → peft),
and `pipeline.py` (`diffusers.hooks` → peft/bitsandbytes, `dw.prompt_weighting` →
transformers). `import dw.workflow`: 5.6s → 2.6s (floor is torch + diffusers core).
Two tests re-targeted their patches to source modules (`test_modular_pipeline.py`,
`test_task.py`). Residual idea for #26: dispatch-table refactor can consolidate the
per-branch imports.

### 21. RIFE rebuilds constant tensors per frame pair
**File:** `dw/tasks/interpolate_frames.py:122-135` (grid/divisor/timestep), `:109-112`
(double PIL→tensor conversion)
`tenFlow_div`, the full-resolution backwarp grid, and the timestep tensor are rebuilt inside
the closure for every frame pair (all frames share one resolution — 8x on 121 frames ≈ 840
rebuilds), and each interior frame is converted PIL→device tensor twice (as `img2` of pair i,
again as `img1` of pair i+1).
**Fix:** hoist the constant tensors out of the closure (compute once per resolution); carry
the padded tensor of frame i+1 forward as the next pair's first input.

---

## P3 — Design / extensibility (diffusers surface-area tracking)

### 22. Output discovery is a hardcoded `hasattr` chain
**File:** `dw/result.py:430-459` (`get_artifact_list`)
Fixed chain: `images` → `image_embeds` → `image_embeddings` → `frames` (with inline LTX
audio-pairing branch) → `audios` → `return [result]` fallback. Any new diffusers output
field (`videos`, `depth`, standalone `audio`) falls through and the whole output object is
treated as one artifact → confusing content-type mismatch error on save.
**Fix:** registry mapping output-field name → extractor (audio-pairing becomes one
registered extractor); optionally allow the workflow's `result` block to name the output
attribute explicitly. Log a clear warning when falling through.

### 23. Resource loading keyed on argument-name suffixes
**File:** `dw/arguments.py:28-34` (`realize_args`)
Only keys matching `image`/`*_image`/`video`/`*_video` get loaded as media; a path under
`mask`, `depth_map`, etc. silently stays a raw string and the pipeline crashes deep inside
diffusers. (`mask_image` — the common inpaint spelling — IS covered.) The dict form
`{"location": ...}` already exists as an explicit marker.
**Fix:** promote the explicit dict form (or a `resource:` prefix mirroring
`variable:`/`previous_result:`) as the general mechanism; keep suffix matching as legacy
sugar in one table.

### 24. Unregistered task commands silently fall back to image/video processing
**File:** `dw/tasks/task.py:269-286` (fallback at 274)
Commands missing from `_COMMAND_REGISTRY` route to `process_image`/`process_video` if an
`image`/`video` argument is present, before the "Unknown task command" raise. A typo'd
command dies with the misleading "Unknown image processor type" error, and the registry's
"Registered commands" listing permanently omits all image/video commands.
**Fix:** register image/video processor names in `_COMMAND_REGISTRY` (they're enumerable);
make unknown commands always raise the task-level error listing valid commands.

### 25. Prompt-weighting support keyed on exact class-name strings
**File:** `dw/prompt_weighting.py:280+` (`_PIPELINE_FUNCTIONS` — four Flux variants by name)
`FluxKontextPipeline`, `FluxFillPipeline`, `FluxControlPipeline`, or any subclass has the
identical CLIP+T5 stack yet raises "not supported"; a diffusers rename silently breaks
workflows.
**Fix:** dispatch on `isinstance`/base class, or detect encoder topology from
`pipe.components` (tokenizer_2 + T5 text_encoder_2 present → flux path).

### 26. `process_image`: 34-branch if-chain, two Canny implementations
**File:** `dw/tasks/image_utils.py:35-170`; dual Canny at `:50-51` (`canny_cv`, raw cv2,
native resolution) vs `:100-102` (`canny`, controlnet_aux `CannyDetector`, resizes to 512)
12 branches share the exact shape `XDetector.from_pretrained("lllyasviel/Annotators")
.to(device)(image, **kwargs)`; the `processor` param is shadowed by detector instances at
lines 97-109. The two Canny paths produce different output for near-identical spellings.
**Fix:** module-level dict `name → (detector_class, from_pretrained args, call kwargs)` +
one generic loader; custom processors get function entries. Either delegate `canny_cv` to
`CannyDetector` or document the difference. Do together with #17 (caching) and #20 (lazy
imports).

### 27. cuda-fp16 dtype rule copy-pasted in four task modules
**Files:** `dw/tasks/image_to_text.py:38`, `text_generation.py:39`,
`diffusion_upscale.py:65`, `depth_estimator.py:32` (only this one has the explanatory
comment: fp16 NaNs on MPS, unsupported ops on CPU)
`torch.float16 if get_device_type(device) == "cuda" else torch.float32` — verbatim ×4.
**Fix:** one helper in `dw/__init__.py` next to `get_device_type()` (e.g.
`preferred_task_dtype(device)`), carrying the comment; call it from all four sites.

### 28. PIL↔tensor round-trip hand-rolled in three tasks
**Files:** `dw/tasks/upscale.py:75-77,90-92`, `restore_faces.py:99-111`,
`interpolate_frames.py:109-112,148-151`
All three use truncating `(x.clamp(0,1) * 255).byte()`; diffusers'
`VaeImageProcessor.pil_to_numpy/numpy_to_pt/pt_to_numpy/numpy_to_pil` are the tested
equivalents (and round properly instead of truncating).
**Fix:** use the diffusers helpers (or one local shared pair of functions if avoiding the
class); fixes the subtle truncation-vs-round difference everywhere at once.

### 29. cuda/mps availability ladder repeated four times in worker
**File:** `dw/worker.py:238` (`_cleanup_between_runs`), `:283` (`_cleanup_all`), `:362`
(`_get_gpu_memory_mb`), `:387` (`_get_memory_info`); echoed in `prompt_weighting.py:266-269`
Same `cuda available / mps available` branching ×4, already drifted (synchronize and
stats-reset in some copies only); MPS branch of `_get_memory_info` (424-435) mostly
re-assigns default zeros. xpu (first-class in `apply_sdnq_optimizations`) gets no cache
cleanup and reports zero memory.
**Fix:** shared helpers in `dw/__init__.py` built on `get_device_type()`: `empty_cache()`,
`synchronize()`, `memory_stats()`; use everywhere.

---

## P4 — Cleanup

### 30. Duplicated log-and-reraise exception ladders (~150 lines, double/triple logging)
**Files:** `dw/step.py:72-99` nested inside `:104-122`; `dw/workflow.py:169-198`;
`dw/pipeline_processors/pipeline.py:273-291`, `:683-703`
Every clause in every ladder does exactly `logger.error(..., exc_info=True); raise`. One
exception in Step.run logs twice (three times counting workflow.py). Clause wording has
already drifted with no behavioral difference.
**Fix:** single `except Exception as e: logger.error(f"{type(e).__name__} …",
exc_info=True); raise` per site; drop the inner Step.run ladder, folding the iteration
number into the outer message.

### 31. Dead gradient-checkpointing branches in TeaCache forward
**File:** `dw/teacache.py:194-218` and `:247-271` (verbatim duplicates)
Only entry point is `Pipeline.run`, decorated `@torch.inference_mode()`
(pipeline.py:205), so `torch.is_grad_enabled()` is always False — both branches are dead
(~50 lines in a 200-line closure); block-call signature changes must currently be made in
four places.
**Fix:** delete both checkpointing branches (or hoist one shared helper if keeping them for
future training use — but nothing in dw/ or tests/ can reach them).

### 32. REPL `run ask`: unreachable guard, duplicated import, deep nesting
**File:** `dw/repl_commands.py:451-454` (dead guard), `:468` + `:505` (duplicate
`import readline`)
If `arg[4:].strip()` is empty then `arg.strip() == "ask"` already returned at line 444, so
the inner `if not arg_name` cannot fire. The 60-line ask block sits 4-5 deep in a 160-line
`_workflow_run` mixing prompting, readline history, validation, and result streaming.
**Fix:** extract `_prompt_for_argument(arg_name)` with the readline toggling isolated;
delete the dead guard; flatten `_workflow_run` to guard clauses.

### 33. `save_artifact` revalidates already-validated paths on every artifact
**File:** `dw/result.py:236-243`
Only callers are inside result.py (`save()` at 211, recursion at 251/286), all passing
already-validated values; N images → N+1 identical realpath/pattern validations; the
`SecurityError` handler only logs and re-raises.
**Fix:** validate once in `save()`; have `save_artifact` trust its (internal, derived)
arguments; drop the redundant handler.

---

## S — Staged-diff findings (MiniMax-H3 modular pipeline work, reviewed 2026-08-15)

Review of the staged diff (modular `components` block, `from_file` argument mechanism,
modular result handling, MiniMax-H3 examples/docs). Verified against diffusers 0.40.0.dev0
modular-pipeline source and `git show HEAD:` for pre-change behavior.

**STATUS 2026-08-15: S1–S6 and S8 are FIXED in the working tree (Phase 0 executed):**
- S1/S2: `validate_media_location` + `fetch_image`/`fetch_video` now resolve relative
  paths against the workflow file's dir (`realize_args(arg, base_dir)`, threaded from
  `Workflow.run`); `previous_result:`/unresolved `variable:` refs in `from_file` raise
  clear errors at load (full deferred construction is a follow-up, see below);
  `MiniMaxH3Ref2VA.json` voice default switched to a stable HF URL (from_file downloads
  URLs itself). Extension check now reuses `security.validate_file_extension`.
- S3: `modular_artifacts` appends leftover dict keys as one extra artifact (saved via the
  existing dict recursion); `first_value` → `first_item` tracks consumed keys.
- S4: new `has_component_group_offload()` feeds both `loading_device()` and the
  `.to(device)` gate, so `components.*.group_offload` no longer needs a manual
  `do_not_send_to_device`.
- S5: object construction now keys on a single `*_type` key (NON_TYPE_KEYS excluded);
  dicts with `from_file` but no `_type` key pass through untouched.
- S6: `{}` escape is honored for NON_TYPE_KEYS (`offload_type`), so both spellings work.
- S8: LTX2I2V `num_frames` 484 → 481 (8k+1).
- Docs: WORKFLOW_GUIDE from_file section updated (relative paths, reference semantics).

**Follow-up (not yet done):** S7, S9 items, and deferred `from_file` construction from
`previous_result:` references (needs nested-ref substitution in previous_results.py plus
per-iteration realization — design alongside todo #23).

### S1. `validate_media_location` rejects deferred references (breaks cross-step flow)
**File:** `dw/arguments.py:142`
`fetch_image`/`fetch_video` both early-return on `previous_result:`/`variable:` prefixes
(resolved later, at step-run time); `validate_media_location` does not, so
`"from_file": "previous_result:tts_step"` dies at load with a misleading
"Path does not exist" SecurityError. Also bites `from_file` objects in the `variables`
block (realize_args runs before replace_variables there — `MiniMaxH3Ref2VA.json` works only
because its reference sits in `steps`).
**Fix:** same guard as the siblings, ideally factored into one shared
"resolve media location" helper (deferred-skip + URL/path validation + extension set) used
by all three callers. Test: workflow with `from_file: previous_result:x` loads.

### S2. `from_file` paths resolve against cwd; `MiniMaxH3Ref2VA.json` unrunnable as shipped
**File:** `dw/arguments.py:150`; `examples/MiniMaxH3Ref2VA.json` (`"voice": "voice.wav"`)
`validate_path(location, allow_create=False)` is called without `base_dir` → cwd-relative,
contradicting CLAUDE.md's "file paths in workflows are relative to the workflow file". No
`voice.wav` exists in the repo, so the example fails at load from any cwd; placing the file
next to the JSON does not help. Note: `fetch_image`/`fetch_video` have the same pre-existing
cwd behavior — the real fix threads the workflow dir into all media loading.
**Fix:** pass the workflow file's dir as `base_dir` through realize_args → media loaders
(workflow.py already knows it, see the sub-workflow handling at workflow.py:298-300); ship a
`voice.wav` (or switch the example to a URL/HF asset). Test: example loads from repo root.

### S3. `modular_artifacts` silently drops non-video dict keys
**File:** `dw/result.py:491` (dict branch triggers at 466 on key-name match)
Returns only videos (+paired audio); before the change a dict fell to `return [result]` and
`save_artifact` recursed over `items()` writing one file per key. `"output": ["videos",
"images"]` now silently loses `images` from disk and from `get_artifacts()` (still reachable
by-name via `get_artifact_properties`). No warning. Staged test only covers dicts without a
video key.
**Fix:** after extracting video+audio, route the remaining keys through the existing dict
recursion (or at minimum `logger.warning` the dropped keys). Test: dict with
videos+images saves both.

### S4. `components.*.group_offload` unenforced pairing with `do_not_send_to_device`
**File:** `dw/pipeline_processors/pipeline.py:747` (`.to(device)` gate), `:635-644`
(`loading_device`), `:201` (configure_components runs after)
`ModularPipeline.to()` moves all loaded components; the `.to(device)` fires before
`configure_components` installs group-offload hooks unless the user *also* sets
`do_not_send_to_device: true`. The pairing is documented in WORKFLOW_GUIDE and honored by
the three examples, but enforced nowhere — omitting it moves e.g. MiniMax-H3's ~62GB
transformer to CUDA and OOMs, the exact failure the config prevents, with no diagnostic.
**Fix:** derive it — if any `components` entry has `group_offload`, skip the `.to()` (and
have `loading_device()` treat it like top-level `group_offload`); or schema-enforce the
pairing with a clear validation error.

### S5. `realize_object`/`from_file` trigger is too broad, detection too clever
**File:** `dw/arguments.py:80` (trigger), `:103-112` (type detection)
(a) Any dict anywhere in steps carrying a literal `from_file` key without exactly one
class-valued sibling now hard-aborts at load ("needs exactly one '_type' argument") where it
previously passed through inert — runtime-verified. (b) The constructed type is picked by
`isinstance(v, type)` over ALL keys rather than `*_type`-named keys, so a second
class-valued kwarg trips the "exactly one" error (narrow: torch dtypes aren't `type`
instances).
**Fix:** select the type by key convention (`k.endswith("_type")`, matching realize_args and
the error message); if a dict has `from_file` but no `_type` key, leave it untouched
(recurse as before) rather than raising.

### S6. `offload_type` brace-escape regression
**File:** `dw/arguments.py:18` (`NON_TYPE_KEYS`)
Adding `offload_type` fixes the pre-existing crash for the bare spelling
(`load_type_from_name("leaf_level")` → AttributeError), but the documented — and previously
*mandatory* — escape `"{leaf_level}"` now bypasses the only brace-strip site and reaches
diffusers verbatim → `ValueError: '{leaf_level}' is not a valid GroupOffloadingType`.
In-repo nothing used it; external user workflows that followed the docs break.
**Fix:** strip `{}` for NON_TYPE_KEYS values too (unescape regardless of which branch
handles the key). Test both spellings.

### S7. `get_component` raises on None-valued modular components (PLAUSIBLE)
**File:** `dw/pipeline_processors/pipeline.py:492`
diffusers ModularPipeline registers unloaded components as None-valued attributes
(`load_components` itself tests `getattr(self, name, None) is None`); `get_component`
raises ValueError on None and `configure_components` applies the map unconditionally.
Shipped examples are safe (verified: each selected workflow loads every named component),
but a components map reused across workflow selections, or a component diffusers
warned-and-continued past at load, aborts with a misleading "has no component".
**Fix:** distinguish missing attribute (typo → raise, fail-fast is right) from
attribute-present-but-None (unloaded → warn and skip, naming the workflow selection).

### S8. `examples/LTX2I2V.json` num_frames 484 is not 8k+1 (PLAUSIBLE, pre-existing scaled)
LTX-2: latent frames = (n-1)//8+1, decodes (l-1)*8+1 frames; audio duration = n/frame_rate.
484 → 481 video frames vs 484/24s audio → ~0.13s trailing audio in the muxed mp4. 242 had
the same defect at 1 frame (~0.04s); the change tripled it. No validation anywhere.
**Fix:** use 481 (or 485); consider a dw-side warning for non-8k+1 values on LTX-2
pipelines (ties into todo #7's component/config validation theme).

### S9. Smaller items from this diff
- **`dw/arguments.py:148`** — URL branch of `validate_media_location` skips the extension
  allowlist (local-only enforcement). Matches pre-existing fetch_image/fetch_video design;
  resolve deliberately repo-wide (check URL path extensions, or document URLs as trusted).
- **`dw/pipeline_processors/pipeline.py:436`** — `configure_components`
  (`configuration.components.<name>`) is a second per-component config namespace parallel to
  `configure_loaded_components` (`configuration.<name>`), disjoint keys, no cross-checking;
  MiniMaxH3.json configures `vae` in both. Unify or cross-validate (relates to todo #7).
- **`dw/arguments.py:151-153`** — extension check re-implements
  `security.validate_file_extension`; use the helper (raises InvalidInputError, a
  SecurityError subclass — staged test still passes).
- **`dw/result.py:51-53`** — `MODULAR_*_KEYS` alias tuples support spellings nothing
  produces; deepens the output-sniffing chain todo #22 wants replaced with a registry —
  fold this dict branch into that refactor when doing #22.
- **`dw/pipeline_processors/pipeline.py:471`** — route `configure_loaded_components` and
  `apply_sdnq_optimizations` lookups through the new `get_component` for dotted-name
  support and one consistent miss behavior.

### Verified non-issues in this diff
- `pair_audio_with_frames` signature change: single caller path, LTX-2 behavior unchanged.
- `config_objects.py` quantization move: line-for-line identical.
- `get_load_components_arguments` shallow copy and `get_group_offload_configuration`
  in-place mutation: safe (Workflow.run deepcopies per run; cached pipelines skip load()).
- Channel-major audio mux corruption claim: refuted — all shipped dict-path sources emit
  batch-major audio (MiniMax-H3 decoder permutes to (1,2,N); LTX-2 vocoder documents
  (batch, channels, samples)), and a mismatched shape fails loudly in _write_audio, not
  silently. Latent shape-assumption hazard only.
- twimg example URLs: in-convention for this repo's examples.

---

## Phase T — test reliability pass (DONE 2026-08-15)

Executed between Phase 0 and Phase 1 after the suite was found to hang indefinitely at
exit. Root cause: `test_worker.py` used 5s queue timeouts racing the worker child's
~6s import cost; on `_queue.Empty` failure the non-daemon child was never stopped and
multiprocessing's atexit join hung pytest forever (2-hour-old hung pytest processes and
a dozen orphaned workers were found and killed).

- Worker spawn/import cost: see item #20 (5.6s → 2.6s).
- `tests/test_worker.py`: lifecycle fixture guarantees shutdown → terminate → kill in
  teardown; daemon=True; 60s readiness timeout (covers child imports), 10s thereafter.
- `dw/worker.py`: command loop polls with a 5s timeout and exits cleanly when its parent
  pid changes/dies, so orphaned workers self-terminate instead of lingering.
- Full suite: 439 passed, 2 skipped, ~33s, exits cleanly — verified twice; worker tests
  verified twice more in isolation. No stray processes after runs.

## Remediation plan

Ordered so each phase leaves the tree working; one focused commit per cluster, each with a
regression test; `pytest -v` + `black dw/ tests/` before every commit.

**Phase 0 — land the staged work cleanly (S1–S6 + S8 example tweak).** These are
regressions in uncommitted work; fix them in/with the staged commit rather than on top.
Order: S1+S2 together (both live in `validate_media_location` — add the deferred guard and
`base_dir` threading in one pass, factoring the shared media-location helper), then S5+S6
(both in the realize_object/from_file trigger), then S3, S4, S8. S7/S9 items can trail as a
follow-up commit.

**Phase 1 — P0 correctness (items 1–7).** Suggested clusters:
(a) seed/generator semantics: #3 + #4 + #12 + #14 share one resolution chain — fix
together, add a reproducibility test matrix (workflow/step/pipeline seed × fresh/cached ×
no_generator true/false/absent);
(b) #1 gather validation (reuse fetch_image/fetch_video per match — dovetails with the
Phase-0 media-location helper);
(c) #2 TeaCache×true-CFG: minimum safe fix is refusing the combination loudly;
(d) #5 bool/list coercion + #16 unknown-variable error (same file);
(e) #6 REPL output_dir; (f) #7 component whitelist (schema `additionalProperties` or
dynamic detection — coordinate with S9's namespace unification).

**Phase 2 — P1 crashes (items 8–15).** #8 teacache×offload (wrap the hook's inner forward)
and #9 prompt-weighting devices (delete `_get_device`, pass `Pipeline.self.device`) are the
highest-value; #10 FasterCache needs the pipeline-reference plumbing; #11 audio save is a
small type branch; #13/#15 are error-handling hardening.

**Phase 3 — P2 performance (items 17–21).** #17 task-model cache first (largest win;
mirrors `previous_pipelines`, needs an unload hook wired to `memory clear`), then #18
shallow-copy, #19 result eviction, #20 lazy imports, #21 RIFE hoisting.

**Phase 4 — P3/P4 (items 22–33).** Group by file to avoid churn: `image_utils.py` trio
(#26 dispatch table + #20 lazy imports + per-detector caching from #17); `result.py`
registry (#22 + S9 alias-tuple fold-in); worker/device helpers (#29 + #27); exception
ladders (#30); the rest opportunistically when touching those files.

---

## Verified non-issues (do not re-report)

- **Group offload + `.to(device)` fallthrough** (`pipeline.py:677-679`): benign — diffusers
  0.40 `ModelMixin.to`/`DiffusionPipeline.to` detect group offload and no-op with a warning;
  `DiffusionPipeline.enable_group_offload` exists for the pipeline-level case. Worst case is
  a spurious warning log line.
- **Destructive `pop()` of `model_name`/lora keys during load** (`pipeline.py:605` etc.):
  safe — `Workflow.run` deepcopies the whole definition (workflow.py:95) before any step
  runs; REPL re-runs and per-iteration sub-workflows always operate on fresh copies.
- **`remove_background` rejecting extra kwargs** (`image_utils.py:48`): loud rejection of
  invalid input; the task has no options and all legitimate usage passes none.
