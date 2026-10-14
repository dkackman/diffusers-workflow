# Validate warnings for acceleration settings that can't pay off (#868)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-14). Don approved plan v2 on 2026-10-14. Stage A, #887, shipped the
same day at `develop` @ `c96e9531` and was verified on mini-ai (mps).

## The idea

Don's H3 quality matrix on lem (2x RTX 3090 24 GB, workspace `h3-quality`)
had three acceleration settings that each took a render to show they were
wrong, though all three could be known at validate time:

1. **P2:** a `first_block` cache at 0.05 on 12 base steps skipped nothing
   (12.79 vs 12.89 min) and still changed the output (PSNR 25.6 dB).
2. **P5:** `taylorseer` (`cache_interval` 3, `max_order` 1, 30 steps,
   1344x768, 4 refs) passed validate and ran out of memory 2 min in (job
   a28aa8ffb931).
3. **P3:** `compile: {repeated_blocks: true}` on a group-offloaded int8
   transformer added 289 s on the first pass and saved about 0.6 s per pass
   after that. The worker released the compiled pipeline on every workflow
   change, so the cost was paid again each time.

## Verdict and scope

**Build smaller, in one stage.** #1 and #2 became validate warnings. The
same check picked up two related rules: `fullgraph` with a cache
(`ACCELERATION.md` stated it and nothing checked it) and a CUDA-only
attention backend on a server that isn't CUDA (#865's agreed follow-up).
#3 is docs only.

Corrections the plan made to the proposal:
- **No turbo-LoRA clause.** Every turbo schedule in the catalog is 9 steps
  or fewer, so `steps < 20` already covers it. Detecting turbo would mean
  matching filenames in the engine, which CLAUDE.md forbids.
- **Threshold caches only** (`first_block`, `mag`). `taylorseer` and
  `faster` skip on a fixed interval, and `text_kv` skips no steps.
- **TaylorSeer is not charged in the estimate.** `vram_estimate` errors
  are hard and `required_vram_gb` drives dispatch, so charging the state
  would turn a warning into a refusal. The engine also holds no
  architecture data to size it with. So #2 is a separate warning, and
  neither the code nor the docs give a state size.

## Decisions (Don's review of plan v1, 2026-10-14)

- **Q1.** #3 is docs only, no warning.
- **Q2.** `mag` is included in the short-schedule rule.
- **Q3.** The `fullgraph` + cache warning is included.
- **Q4.** The TaylorSeer warning always fires when TaylorSeer is set.
- **Changes:** add `attention_backend_unavailable`, keyed on a named set of
  backends rather than the `_hub` suffix. Shrink the #3 docs to one fact
  (compile is paid again when the worker releases the pipeline). Give no
  TaylorSeer state size. Keep 20 steps as the single threshold: nothing
  between 12 and 30 steps had been measured, and #875 measures 20.

## What was built

### Stage A, #887: `acceleration_warnings`

- **Engine:** `dw/acceleration_warnings.py`. It is registered as the last
  entry of `WARNING_CHECKS` in `dw/validation.py`, and it resolves values
  through variables and arguments. Strings use the existing
  `<path>: <code>: <message>` form, so no wire shape changed. The codes:
  - `cache_short_schedule`: a `first_block`/`mag` cache where the resolved
    `num_inference_steps` is below `CACHE_MIN_USEFUL_STEPS` (20).
  - `taylorseer_state_unestimated`: always fires on `type: taylorseer`, and
    names `max_order`. It adds "estimated at X of Y GB" only when the
    workflow declares a `vram_estimate` and the device capacity is known.
  - `compile_fullgraph_with_cache`: a component with `compile.fullgraph:
    true` in a pipeline that has a cache.
  - `attention_backend_unavailable`: a backend in
    `CUDA_ONLY_ATTENTION_BACKENDS` (the flash, flash-3, sage and xformers
    families plus `aiter_fa2_hub`, Hub kernels included), at pipeline or component
    level, on a known device that isn't CUDA. The device is compared with
    `get_device_type()`.
  - A value that can't be resolved, or an unknown device, keeps its
    warning silent. No warning ever refuses.
- **Save path:** `_write_validated` (`dw/server/routes/library.py`) runs
  only selected warning checks, not the whole registry. It now also runs
  `acceleration_warnings`, so `save_workflow` returns the same codes as
  `validate_workflow`. This was the tester's bounce (see below).
- **Map:** `docs/ARCHITECTURE.md`, *Validation, plan and cost*, has an
  *Acceleration warnings* row.
- **Docs:** `docs/ACCELERATION.md` names all four codes, gives the P5 OOM
  configuration with no state size, says the compile cost is paid again
  whenever the worker releases the pipeline, covers `fullgraph` with a
  cache, and has the backend warning in the MPS notes. It also removes the
  old line saying that validation doesn't check the backend.
- **Tests:** `tests/test_validation.py::TestAccelerationWarnings` covers
  every boundary through the real `validate` path, checks that no catalog
  workflow emits the three device-independent codes, and checks that none
  of the codes refuses.
  `tests/test_validate_arguments.py::test_save_workflow_carries_the_acceleration_warnings`
  covers the save route.

Deviations from the plan, all recorded on #887: the check lives in its own
module instead of inline in `validation.py`; `aiter_fa2_hub` is in the
CUDA-only set; a `for_each` step's members share one path, so each code
appears once per path.

## Verified

On mini-ai (mps), over MCP only: C-F446 to C-F453. The case that
`attention_backend_unavailable` stays silent on CUDA needs a CUDA server.
On mps it is covered only by unit tests with a patched device.

## Bounces per stage

- **Stage A, #887: 2.**
  - **Architecture review (1):** the new module had no row in
    `docs/ARCHITECTURE.md`. The row was added (`547a82c6`).
  - **Tester (1):** C-F451 step 2. `save_workflow` didn't return the
    acceleration warnings, because the plan wrongly assumed save runs the
    whole `WARNING_CHECKS` registry. Fixed in `bf448a2a` (`c96e9531`).

The stage cost isn't recorded, because the stage comments give no `usage:`
figures. The plan estimated $4-5.

## Deferred

- **#3 as a warning** (Q1). It is docs only. Reconsider if an agent keeps
  paying the compile cost across workflow changes.
- **Sizing TaylorSeer state** in the estimate. The engine has no
  architecture data, and the estimate drives dispatch and refusal.
- **A `mag.num_inference_steps` / pipeline-steps mismatch warning.** The
  schema says the two must match, and nothing checks it. It hasn't yet been
  shown to cost a render. Don decides whether it becomes its own issue.
- **The 20-step threshold** is a heuristic. #875's threshold experiments
  may move it, and it is a single constant.
