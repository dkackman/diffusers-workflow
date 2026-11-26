# Engine defaults and device rewrites reach the caller (#1643)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v1 on #1643 is the full design. This file records what
shipped, what was cut, and why.

**Result:** every device correction the engine makes on MPS or CPU is now a
`device_adjusted` job warning, not a log line. A video written at the 8 fps
fallback carries a `video_fps_defaulted` warning. Each run's `manifest.json`
records a `torch` block with the numerics that applied. The h3 prompt enhancer
lets its builtin choose the model, and a `text_generation` job says which model
it ran. The H3 default step count is read off the diffusers block instead of
being copied from it.

## The problem

The engine's rule (`docs/ARCHITECTURE.md`, *Run-time warnings*) is that a
run-affecting value is a variable the caller can override, and that when the
engine changes a value the caller is told. An audit found exceptions:
- defaults in engine code that no variable reaches (task dtype, hidden task
  defaults, the H3 step count, per-model processor defaults, the 8 fps
  fallback, the enhancer's model);
- MPS/CPU device rewrites that were only logged, so they never reached an API
  or MCP caller;
- server-wide torch numerics (`enable_tf32`, `cudnn_benchmark`,
  `cudnn_deterministic`) that were not recorded with the run.

## Verdict: build smaller

Don approved plan v1 with the delegate's recommendations on 2026-11-25. The
delegate passed it to him because it adds reply surface: a new warning kind
(`device_adjusted`) and a new manifest key (`torch`).
- **Q1 (scope):** cut the task `dtype` argument and validate-time MPS warnings.
- **Q2 (scope):** keep the per-model processor defaults (H3 windows, keystone
  attention, H3 rules, `sparse_tau`) as the processor's documented contract.
- **Q3 (loosen):** G and C are verified by the builder's unit tests, plus a check
  that a CUDA run carries no `device_adjusted` warning.

Corrections the plan made against the issue:
- The device-rewrite sites are under `dw/pipeline_processors/`, not `dw/`.
- The attention-slicing note was `debug`, not a warning.
- The H3 `DEFAULT_STEPS` is reached only by hand-written workflows.
- The upscale defaults were already template variables and already documented.
- The manifest already recorded `device` and `dw_version`.
- Dropping `default_model` from the enhancer presets would break the REST model,
  the UI picker, MCP readers and about 8 tests. The fix belongs in the call.

## What shipped

### Stage 1 (#1727): device corrections and the fps fallback become job warnings

Branch `feat/1643-A-device-warnings` (569e140c, fa4698ad, 494f0e94), merged to
`develop` at `f837e469`. `deploy: server`.

- **G.** Each log-only correction goes through `emit_warning(kind="device_adjusted",
  setting=…, requested=…, applied=…)`. The message says what was asked, what
  ran, and why. The sites:
  - RoPE `double_precision` float64 → float32 on MPS
    (`placement.apply_mps_rope_precision`);
  - automatic attention slicing on MPS, moved from a debug log in `pipeline.py`
    into `placement.apply_attention_slicing`. It warns only when the device
    turned slicing on;
  - `enable_auto_cpu_offload` skipped on MPS (`placement.create_components_manager`);
  - `offload` dropped on CPU (`placement.place_component`);
  - `sequential` → `model` offload on MPS, naming any dropped
    `exclude_from_cpu_offload` entries;
  - `torch.compile` skipped on MPS (`components.apply_compile`);
  - float16 on MPS (`components._warn_float16_on_mps`), a warning only, with
    no rewrite claim;
  - SDNQ quantized-matmul options off on MPS
    (`config_objects.portable_quantization_arguments`).
- **E.** `Result.warn_if_fps_defaulted` warns at the mp4, audio+video and gif
  write sites when the file is written at 8 fps. It is quiet when a rate was
  declared or carried, and for an unsaved child.
- **Docs.** `docs/ACCELERATION.md` (MPS notes, float16 and compile lines) and
  the `docs/ARCHITECTURE.md` *Run-time warnings* row's pin.
- **Tests.** `tests/test_device_adjusted_warnings.py` (each site on a real
  `Job`, a quiet cuda case for each, de-duplication per step) and
  `tests/test_result.py::TestVideoFrameRate`.

Deviations the build recorded:
- The two #1744 `portable_loaded_components` warnings now also carry
  `kind="device_adjusted"`.
- The gif fallback warns too.
- `resolve_device`'s `cuda` → `mps` translation and the group-offload stream
  drop stay log-only. The translation would warn on almost every catalog run on
  a Mac. `docs/ACCELERATION.md` says so.

### Stage 2 (#1728): recorded numerics and defaults read from their source

Branch `feat/1643-B-recorded-numerics` (`c2e4fd2d`), merged to `develop` at
`3272d304`. The bounce fix, `feat/1643-B-model-that-ran` (`562bdd6d`), merged at
`94a442fd`. `deploy: server`.

- **H.** `write_run_manifest` (`dw/workflow_run.py`) adds
  `"torch": {version, diffusers_version, enable_tf32, cudnn_benchmark,
  cudnn_deterministic}` from `dw.torch_numerics()`, read live in the worker
  that ran the job. It is additive. A manifest from before the deploy has no
  `torch` key. `docs/WORKSPACES.md` (Runs) describes it.
- **F.** `dw/server/enhancers.py`: the h3 preset passes `model_name` only when
  the caller names one, so `h3_context_ir.json`'s variable decides. t2i keeps
  `default_model`. `default_model` stays in `list_enhancers`, the REST model
  and the UI. A test pins the h3 `default_model` to the builtin's variable
  default.
- **Bounce fix (which model ran).** The tester couldn't see over MCP which
  model an h3 enhance job ran, once `model_name` was left to the builtin.
  `text_generation` now emits a `log` event on every run,
  `text_generation: <model> on <device>`, with `model_name` and `device`
  fields. `docs/TASKS.md` (Text Generation) states it.
- **C.** `h3_rules.default_num_inference_steps()` (cached, diffusers imported
  lazily) reads `MiniMaxH3SetTimestepsStep`'s declared default, with a
  fallback of 50. `h3_tst.DEFAULT_STEPS` and `h3_lead_in.DEFAULT_STEPS` stay
  readable through a module `__getattr__`.
- **B docs.** No edit: `docs/TASKS.md` already stated `max_new_tokens` 500.
- **Tests.** `tests/test_recorded_numerics.py` covers the manifest block, the
  enhancer call and pin, the logged model, and the H3 step default with each
  fallback path.

## Cut, and what brings each back

- **A task `dtype` argument (Q1).** It would be new surface on 4 tasks, and a
  `**kwargs` task ignores a misspelling silently (`introspection.py:694`).
  Reopen on a field report of fp16 overflow/NaN in a CUDA task, or of memory
  pressure on a Mac.
- **Validate-time MPS warnings in `acceleration_warnings` (Q1).** Reopen if a Mac
  session is surprised by a rewrite that the job warning reported only after
  the GPU time was spent.
- **Moving the per-model processor defaults to template variables (Q2).** Moving
  `sparse_tau` would change step-cache keys (`step_cache.py:179`).
- **Task-registry defaults (B).** The registry has no notion of defaults;
  building one is its own feature.
- **Reconsidering automatic attention slicing.** That is #1625's. This feature
  only makes slicing visible.
- **Reading builtins through `get_workflow`.** It would add MCP surface; the
  model-name event answers the case that needed it.

## Found along the way (filed separately)

- **#1751:** a `gather_images` list saved as mp4 and `loop_frames` saved as gif
  crashed at save. Closed and verified.
- **dkackman/harnest#312:** C-F1437's "bare" steps no longer reach the fallback,
  because `video_frames` now carries the source rate. The amendment moves them
  to `loop_frames`.
- **Settings not on `get_server_info`.** The three numeric flags aren't reported
  by `get_server_info` or `get_health`, so C-F1439 checks them only as
  booleans that agree across cards. Not filed: the plan doesn't promise it.

## Stages, bounces and cost

| Stage | Bounces | Notes |
|---|---|---|
| #1727 device warnings + fps fallback | 0 | The first verify pass had no verdict: the mini-ai GPU was held by another job. Verified on mps (mini-ai) at `11bff16d`. The CUDA no-leak half (C-F1438) is left to lem's regression run. |
| #1728 manifest numerics, enhancer model, H3 steps | 1 | C-F1441 step 2 bounced: no MCP reply named the model that ran. Fixed by the `text_generation` model event at `94a442fd`, re-verified on lem. |

The stage comments name no `usage:` figures, so cost isn't recorded here. The
plan estimated about $8 for the two stages.

Verification: C-F1437 and C-F1438 on mps (mini-ai.lan), C-F1439 to C-F1442 on
lem, all in workspace `regression-complete`.
