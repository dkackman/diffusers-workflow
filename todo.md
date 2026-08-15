# TODO — Code Review Findings (2026-08-15)

Findings from a full-project correctness/quality review, adversarially verified against the
code and the installed dependencies (diffusers 0.40.0.dev0, accelerate 1.14, torch 2.13).
Line numbers are as of commit `12c6c8d`. Priority-ranked: P0 = silent wrong results or
security, P1 = crashes/broken features, P2 = performance, P3 = design/extensibility debt,
P4 = cleanup.

Checked-off items should also be removed from this file once verified fixed (with a test
where feasible).

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

### 20. Heavy imports at module load — paid even by `dw.validate`
**File:** `dw/tasks/image_utils.py:2-27`; chain: `dw/validate.py:3` → `workflow.py:13` →
`task.py:4` → `image_utils`
`cv2`, 17 `controlnet_aux` detector classes, and transformers model classes import at module
top level, so schema-only validation and every REPL worker spawn pay seconds of import and
tens of MB. The file already demonstrates the fix (lazy `PIL.ImageDraw` in `add_watermark`).
**Fix:** move detector/transformers/cv2 imports inside the branches that use them (pairs
naturally with #26's dispatch-table refactor).

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
