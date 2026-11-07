# The served worker applies dw's PyTorch settings; cuDNN determinism is the default (#1229)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-07). Plan v1 approved by Don on 2026-11-07, with Q2–Q4 at their
defaults; one stage, #1238, verified over MCP on lem the same day.

## The idea

The long-form bench (#1199, `bench/records/determinism.md`) found same-seed
output drifting on two-card jobs and blamed `cudnn_benchmark` autotuning.
With the autotuner off, the bench got 14/14 canonical. The idea was to flip
`dw/settings.py` to `cudnn_benchmark: false` + `cudnn_deterministic: true`,
after timing a conv-heavy family with benchmark on and off (R36).

## What the design found

The defaults the idea would flip never reached a served job. Only
`startup()` (`dw/__init__.py`) applied `enable_tf32`, `cudnn_benchmark`
and `cudnn_deterministic`. `startup()` runs in the `dw.serve` parent,
`dw.test` and `dw.validate`. Every REST, MCP and UI job runs in a spawned
worker that never called it. So served jobs ran on PyTorch's own defaults
(benchmark off, deterministic off, TF32 off), whatever `settings.json` said,
and the three keys ACCELERATION.md documented did nothing for them. Flipping
the defaults as filed would have changed only `dw.test`.

That also took away most of R36's timing question: served jobs, and every
catalog `cost` row, were already measured with benchmark off. What was left
to time was `deterministic` on against off, and verification covered that.

## Decisions

- **Q1 [record]:** why the bench's `settings.json` change seemed to fix
  the drift. Left to stage 1's flag report (see *What verification
  found*).
- **Q2:** `cudnn_deterministic: true` (default taken).
- **Q3:** `cudnn_benchmark: false` (default taken). Served jobs run as
  they already did.
- **Q4:** `enable_tf32: false` (default taken), so served output and cost
  rows stay as they were.
- **Out of scope:** `torch.use_deterministic_algorithms` /
  `CUBLAS_WORKSPACE_CONFIG`, a per-job or MCP override, re-measuring cost
  rows.

## What was built

**Stage 1, #1238.** Commits `44883ab4`, `31026818` and `9c25ba6f`, merged
to `develop` as `36181131`.

- **One function applies the flags.** `configure_torch(settings,
  device_type)` in `dw/__init__.py`. `startup()` calls it for the CLI.
  `WorkflowWorker.__init__` calls it with `get_device_type()` and
  reapplies it at the start of every job (a deviation from the plan: a
  library a job imports can change the flags). The cuDNN flags are set
  only on cuda. Matmul precision is set on every backend, to `"highest"`
  when TF32 is off.
- **New defaults** in `dw/settings.py`. `load_settings()` falls back to
  the class attributes, so the two can't disagree. An explicit
  `settings.json` key still wins.
- **Narration.** The first job a worker process finishes emits one `log`
  event, read back from `torch.backends` after the run, e.g. `torch:
  cudnn benchmark=false deterministic=true, tf32=false`. It uses the
  existing `RunContext.emit('log')` channel, with no protocol change.
- **Docs:** `docs/ACCELERATION.md` (*TF32 and cuDNN*), the reproducibility
  note in `docs/WORKFLOW_GUIDE.md`, and the plugin's
  `long-form/references/quality-and-speed.md` ("Determinism is the
  default"). `bench/records/determinism.md` got a dated correction at the
  top, and the record below it stays as written.
- **Tests:** `tests/test_torch_settings.py`.
- **Architecture review:** PASS. A map row for "PyTorch process flags" was
  suggested as optional and not added.

## What verification found (lem, develop @ 36181131)

- **Reproducible under load (C-F698).** A seeded SD 1.5 image came out
  byte-identical in four conditions: fresh worker on a quiet host, while
  an LTX job loaded on the other card, after `clear_memory`, and on the
  other card. A seeded LTX-2.5 clip was identical across cards and between
  cold and warm runs (frames and `pcm_sha256`).
- **Q1's answer (C-F699).** Both cards' fresh workers reported
  `benchmark=false deterministic=true, tf32=false`, read after SD 1.5 and
  LTX-2.5 pipelines had loaded. So nothing those pipelines import turns
  the autotuner on. The bench's `cudnn` and `cudnn2` batches can't have
  got their result through the settings keys. The likeliest explanation
  is timing (only two batches ran), but the earlier drift's cause stays
  unexplained beyond that.
- **No slowdown (C-F700, C-F701).** The SD 1.5 warm median was 3.594 s,
  against 3.618 s before the deploy. LTX-2.5 warm runs took 56.4 s and
  59.8 s, against the catalog row of 1.8 min. There was no pre-deploy
  decode reference.
- **No cost row moved (C-F702).** `validate_workflow` stayed clean, and no
  job raised a settings warning.
- **Not run:** C-F703, the optional H3 control, skipped for budget. H3 under
  load, where the drift was first seen, is not measured with the new
  defaults.

## Bounces per stage

- **#1238:** none. The tester found two defects in the cases, not the
  build: C-F698 assumed `clear_memory` restarts the worker (it doesn't),
  and C-F701's warm runs reused the warm-up's seed and would have hit the
  step cache. The first was filed as an amendment on the harness repo; the
  tester ran the second with fresh seeds.

Stage costs: not recorded in the stage comments, so left out.

## Follow-ups

- An H3 same-seed check under two-card load with the new defaults (C-F703,
  optional).
- Optional: a `docs/ARCHITECTURE.md` row for the PyTorch process flags
  (`dw/__init__.py: configure_torch`).
