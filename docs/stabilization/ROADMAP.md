# Stabilization roadmap

The single home for the 2026-09-28 architecture stabilization. Why it exists:
[ASSESSMENT.md](ASSESSMENT.md). Shared view of the same content:
https://claude.ai/code/artifact/4e21fcdc-ab14-4575-9b52-3805e6a66030

Each phase ends at a gate that passes or fails. A phase's detailed plan is
written when the previous gate passes, not before - the code each later
phase works on is what the earlier phases leave behind.

| Phase | Scope | Gate | Plan | Status |
| --- | --- | --- | --- | --- |
| 0 | Freeze, baseline metrics, fix B1-B8 | B1-B5, B7, B8 fixed with regression tests; B6 documented as a design limit; deployed to lem; `baseline.json` committed | [phase-0.md](phase-0.md) | done 2026-09-28 (`stabilization-gate-0`) |
| 1 | Metrics v2 first (see below); remove the REPL; one prepare pipeline; one admission service; `dw.run` becomes a thin client of `dw.serve` | Validation sees the definition the run sees; the server admits a request once (one `Workflow`, one expansion); every entry point reaches the worker through the server; ratchets re-baselined | [phase-1.md](phase-1.md) | done 2026-09-28 (`stabilization-gate-1`) |
| 2 | Seams in place: `references.py`, validation context + check registry, shared task rules, step cache, typed worker protocol | `validation_errors` is a registry loop; no prefix literals outside `references.py` | [phase-2.md](phase-2.md) (staged: 2a-2d) | done 2026-09-30 (`stabilization-gate-2`) |
| 3 | Structural moves: `app.py` routers + services, `LibraryPath`, split `result.py` / `pipeline.py`, one media + dsp module | No module over 1,000 lines, no function over 150; suite and lem smoke green | [phase-3.md](phase-3.md) (staged: 3a-3e) | 3a merged; 3b next |
| 4 | Context diet (CLAUDE.md <= 250 lines total) and guardrails installed | Guardrails live in dw CI and the harness; freeze lifted | written at gate 3 | - |

## Metrics

Two kinds, kept small on purpose.

**Ratchets.** These live in `scripts/arch_metrics.py`. Each is lower-is-better, and `--check` against `baseline.json` fails the build when one gets worse.

- Phase 0 set: modules, modules over 1,000 lines, functions over 150 lines, reference-prefix literals, test `patch("dw...")` targets, CLAUDE.md lines, duplicate blocks.
- Added at the start of Phase 1 ("metrics v2"), re-baselined in the same commit:
  - **Cyclomatic complexity:** functions above 15, by ruff's C901 (already a dev dependency). Baseline 21.
  - **Import cycles:** strongly connected components of more than one module in grimp's graph of `dw` + `dw_mcp`, lazy imports included, TYPE_CHECKING imports excluded. Baseline 6.
  - **Modules inside import cycles:** the sum of those components' sizes. Baseline 26, because one of the six is a 16-module knot (`arguments`, `result`, `pipeline`, `runs`, `step_cache`, `tasks`, ...) that could grow without the cycle count moving. It grew from 22 to 26 in Phase 0 (the `step_cache` → `pipeline` import). The target is 0 for both cycle ratchets, reached by Phases 2-3. Each ratchet only forbids getting worse.

**Gate reports.** These are produced at each phase gate by `scripts/arch_report.py`, which lands in Phase 1. They are written into the Gate reports section below. They are reports, never gates.

1. **Change coupling:** file pairs that change together in `git log`, measured as shared commits and the coupling degree, plus hotspots (churn × total complexity per file). This is the signal for a rule living in two places, like the #415 pair of `server/jobs.py` and `worker.py`. It is a short git-log script, because code-maat is a separate Java tool.
2. **Instability per package:** Ca, Ce and I = Ce / (Ca + Ce), from the same grimp graph. The engine core should be stable (low I) and the server volatile (high I).
3. **LCOM4:** on `Workflow`, `Pipeline`, `Result`, `JobManager` and the app factory. It is a before/after check that the Phase 3 splits improved cohesion. It is hand-rolled over `ast` (about 40 lines), since there is no maintained library for it; this is the one build-vs-buy exception.

Deliberately not adopted: function points, maintainability index, Halstead, coverage %, and comment density.

**Every gate report carries the full metrics table.** It has one column per gate, starting with "before Phase 0" (`3afd70e9`), so the trend reads left to right. The rows are:

- every ratchet;
- functions over cyclomatic complexity 15 and over 30;
- the complexity distribution (function count, median, mean, and how many functions exceed 10, 15, 20 and 30);
- the ten most complex functions, with file:line;
- SLOC by layer (engine, API = `dw/server` + `dw_mcp`, UI = `ui/src` without its tests; pygount code lines; for reference only), package instability, and LCOM4 on `Workflow`, `Pipeline`, `Result` and `JobManager` with their method counts. (`create_app` is a closure, not a class. It is measured when Phase 3 splits it into routers.)

All complexity numbers are ruff C901; gate 0's hand table used mccabe and is regenerated by `arch_report.py`.

`scripts/arch_report.py` (Phase 1) produces it for any commit or tag. Until then it is computed by hand, the way gate 0's was.

Each gate is tagged `stabilization-gate-N`, so any report can be recomputed for an earlier gate. Gate 0's reports are computed from its tag once `arch_report.py` exists.

## Gate reports

(Filled at each gate: the metrics diff against the previous gate, the top change-coupling pairs and hotspots, package instability, and LCOM4 for the tracked classes from Phase 3 on.)

### Gate 0 (2026-09-28, develop 6a88c746)

- Ratchets: every Phase 0 metric equal to `baseline.json` (`--check` exit 0). Values: 133 modules, 10 over 1,000 lines, 19 functions over 150 lines, 93 prefix literals, 285 test patch targets, 964 CLAUDE.md lines, 21 duplicate blocks.
- B2 on lem (RTX 3090), `templates/ltx2/two-stage` rerun with the same seed: 78.8 s before Phase 0, **0.77 s** after. A cold run is about 160-174 s throughout.
- Two gate-time follow-ups came from that real-GPU timing, not from the tests:
  - Task 11: a cached step defers its pipeline load until a step that actually runs borrows it.
  - Task 12: borrowed-pipeline cache keys are hashed from the definition as written, because `Pipeline.load` edits its definition in place.
  Mocks could not see either. A real-model timing stays in every gate.

#### Metrics

| Metric (lower is better) | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) |
| --- | --- | --- |
| Engine + MCP modules | 133 | 133 |
| Modules over 1,000 lines | 10 | 10 |
| Functions over 150 lines | 19 | 19 |
| Functions over cyclomatic complexity 15 | 21 | 21 |
| Functions over cyclomatic complexity 30 | 4 | 4 |
| Import cycles | 6 | 6 |
| Modules inside import cycles | 22 | 26 |
| Duplicate-code blocks, cross-file | 21 | 21 |
| Reference-prefix literals | 93 | 93 |
| Test `patch("dw...")` targets | 284 | 285 |
| CLAUDE.md lines, all files | 964 | 964 |

#### Complexity distribution (ruff C901)

|  | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) |
| --- | --- | --- |
| Functions | 1712 | 1719 |
| Median complexity | 2.0 | 2 |
| Mean complexity | 3.78 | 3.79 |
| Over 10 | 64 | 64 |
| Over 15 | 21 | 21 |
| Over 20 | 11 | 11 |
| Over 30 | 4 | 4 |

#### SLOC by layer (pygount code lines; reference only)

| Layer | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) |
| --- | --- | --- |
| Engine | 19980 | 20102 |
| API (dw/server + dw_mcp) | 6498 | 6525 |
| UI (ui/src, tests excluded) | 2266 | 2266 |

#### Package instability, I = Ce / (Ca + Ce)

| Package | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) |
| --- | --- | --- |
| dw (core) | Ca 31 / Ce 10 / I 0.24 | Ca 31 / Ce 11 / I 0.26 |
| dw.pipeline_processors | Ca 2 / Ce 4 / I 0.67 | Ca 3 / Ce 4 / I 0.57 |
| dw.server | Ca 1 / Ce 8 / I 0.89 | Ca 1 / Ce 8 / I 0.89 |
| dw.tasks | Ca 11 / Ce 20 / I 0.65 | Ca 11 / Ce 20 / I 0.65 |
| dw_mcp | Ca 1 / Ce 0 / I 0.0 | Ca 1 / Ce 0 / I 0.0 |

#### LCOM4 (components; 1 = cohesive)

| Class | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) |
| --- | --- | --- |
| Workflow | 1 (28 methods) | 1 (30 methods) |
| Pipeline | 1 (21 methods) | 1 (21 methods) |
| Result | 1 (15 methods) | 1 (15 methods) |
| JobManager | 1 (29 methods) | 1 (30 methods) |

#### Ten most complex functions at Gate 0

| Complexity | Function |
| --- | --- |
| 440 | dw/server/app.py:747 `create_app` |
| 79 | dw_mcp/server.py:65 `build_server` |
| 37 | dw/repl_commands.py:644 `_workflow_run` |
| 34 | dw/workflow.py:1279 `run` |
| 27 | dw/arguments.py:103 `realize_args` |
| 26 | dw/result.py:811 `save_artifact` |
| 24 | dw/locations.py:612 `_walk` |
| 24 | dw/server/app.py:1679 `_argument_reference_errors` |
| 22 | dw/media_info.py:19 `probe_media` |
| 22 | dw/tasks/loop_bed.py:397 `find_loop_bed` |

#### Change coupling at Gate 0 (commits since 2026-08-01)

| Shared commits | Degree % | File | File |
| --- | --- | --- | --- |
| 13 | 55 | dw/tasks/concat_videos.py | dw/tasks/dissolve_videos.py |
| 5 | 45 | dw/security.py | dw/type_helpers.py |
| 13 | 44 | dw/task_domains.py | dw/tasks/task.py |
| 26 | 37 | dw_mcp/catalog.py | dw_mcp/server.py |
| 44 | 33 | dw/server/app.py | dw_mcp/server.py |
| 23 | 33 | dw_mcp/media.py | dw_mcp/server.py |
| 7 | 32 | dw_mcp/authoring.py | dw_mcp/diagnose.py |
| 21 | 31 | dw_mcp/diagnose.py | dw_mcp/server.py |
| 8 | 30 | dw/tasks/concat_videos.py | dw/tasks/video_utils.py |
| 8 | 30 | dw_mcp/client.py | dw_mcp/media.py |

#### Hotspots at Gate 0 (churn x total complexity)

| Score | Commits | Complexity | File |
| --- | --- | --- | --- |
| 154400 | 160 | 965 | dw/server/app.py |
| 18537 | 111 | 167 | dw/workflow.py |
| 16848 | 108 | 156 | dw_mcp/server.py |
| 16576 | 56 | 296 | dw/pipeline_processors/pipeline.py |
| 11224 | 61 | 184 | dw/result.py |
| 10450 | 55 | 190 | dw/tasks/audio_utils.py |
| 10089 | 57 | 177 | dw/server/jobs.py |
| 4056 | 26 | 156 | dw/arguments.py |
| 3912 | 24 | 163 | dw/introspection.py |
| 3864 | 42 | 92 | dw/tasks/task.py |

Regenerated by `scripts/arch_report.py` at Phase 1 Task 2. The complexity figures are ruff C901 throughout. The hand table this replaced used mccabe with nested functions folded in, which counted 1,485 functions and scored `create_app` at 510.

### Gate 1 (2026-09-28, develop a113fa59)

- **Gate criteria.**
  - Validation, the run and the realized record share one fold stage and one expand stage (`Workflow._fold` / `_expand`). A parity test pins validation's expansion to the run's.
  - The server admits every validate, submit, rerun and enhance request once, through `dw/server/admission.py`. That is one `Workflow` and one fold per request, measured for a plain submit, validate with plan, a bound-acknowledgement submit and a rerun.
  - `python -m dw.run` is an HTTP client of `dw.serve`. The REPL is gone, so nothing but `dw.test` (the installation check) runs a workflow outside the server.
  - The worker still validates the definition it loads (ruling, Phase 2 item; done in stage 2c: the worker runs the admitted snapshot).
- **Real-model timings on lem** (RTX 3090, `templates/ltx2/two-stage`):
  - cold, through the new `dw.run` client over ssh with `seed=777`: exit 0, **195 s** wall, including the first worker spawn after the deploy restart;
  - then `validate_workflow` for the same seed reported `cached_steps: 3`, and the bound-acknowledgement rerun over MCP finished in **0.77 s** with every step reused.
- **Ratchets.** Every one equal or lower. Re-baselined at this gate, so the harness holds the gains.
- **Surface changes (release notes).**
  - `dw-repl` / `python -m dw.repl` removed; `docs/REPL_WORKER_GUIDE.md` became `docs/WORKER_GUIDE.md`.
  - `python -m dw.run` needs a running `dw.serve` and takes `--server`, `--workspace` (a server workspace name, no longer a directory) and `--token`. The flags `-o/--output_dir`, `--prompt-dir`, `--asset-dir`, `--output-layout`, `--trust-workflows` and `-l/--log_level` are gone; they are `dw.serve` settings. `httpx` is a base dependency.
  - Every plan fingerprint changes once, so an acknowledgement bound before this deploy gets one 409.
  - `workflow.json` records the folded variables: realized constants, resolved list entries and snapped values.
  - A job records the full warning set `/api/validate` reports, whether it came from submit, rerun or enhance.
  - A rerun rechecks its `asset:` / `prompt:` / `output:` references and answers 400 when one no longer resolves.
  - Validate's argument-error 400 body carries warnings.
  - An undeclared `constraint:` name is reported at its path.
  - A warning helper that fails is logged and never refuses a job.
- **Carried to Phase 2.**
  - The worker validating a job snapshot instead of re-reading the file (done, stage 2c).
  - The sub-workflow media deep copy in `recorded_variables`.
  - `dw.run` paging a truncated event tail (done, stage 2c).
  - The expansion memo is never invalidated.
  - `JobManager.submit`/`rerun` fall back loosely when no workflow name is given.

##### Metrics

| Metric (lower is better) | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) |
| --- | --- | --- | --- |
| Engine + MCP modules | 133 | 133 | 132 |
| Modules over 1,000 lines | 10 | 10 | 10 |
| Functions over 150 lines | 19 | 19 | 17 |
| Functions over cyclomatic complexity 15 | 21 | 21 | 19 |
| Functions over cyclomatic complexity 30 | 4 | 4 | 3 |
| Import cycles | 6 | 6 | 6 |
| Modules inside import cycles | 22 | 26 | 26 |
| Duplicate-code blocks, cross-file | 21 | 21 | 19 |
| Reference-prefix literals | 93 | 93 | 93 |
| Test `patch("dw...")` targets | 284 | 285 | 285 |
| CLAUDE.md lines, all files | 964 | 964 | 961 |

##### Complexity distribution (ruff C901)

|  | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) |
| --- | --- | --- | --- |
| Functions | 1712 | 1719 | 1685 |
| Median complexity | 2.0 | 2 | 2 |
| Mean complexity | 3.78 | 3.79 | 3.76 |
| Over 10 | 64 | 64 | 61 |
| Over 15 | 21 | 21 | 19 |
| Over 20 | 11 | 11 | 10 |
| Over 30 | 4 | 4 | 3 |

##### SLOC by layer (pygount code lines; reference only)

| Layer | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) |
| --- | --- | --- | --- |
| Engine | 19980 | 20102 | 19410 |
| API (dw/server + dw_mcp) | 6498 | 6525 | 6593 |
| UI (ui/src, tests excluded) | 2266 | 2266 | 2266 |

##### Package instability, I = Ce / (Ca + Ce)

| Package | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) |
| --- | --- | --- | --- |
| dw (core) | Ca 31 / Ce 10 / I 0.24 | Ca 31 / Ce 11 / I 0.26 | Ca 32 / Ce 12 / I 0.27 |
| dw.pipeline_processors | Ca 2 / Ce 4 / I 0.67 | Ca 3 / Ce 4 / I 0.57 | Ca 3 / Ce 4 / I 0.57 |
| dw.server | Ca 1 / Ce 8 / I 0.89 | Ca 1 / Ce 8 / I 0.89 | Ca 1 / Ce 9 / I 0.9 |
| dw.tasks | Ca 11 / Ce 20 / I 0.65 | Ca 11 / Ce 20 / I 0.65 | Ca 11 / Ce 20 / I 0.65 |
| dw_mcp | Ca 1 / Ce 0 / I 0.0 | Ca 1 / Ce 0 / I 0.0 | Ca 2 / Ce 0 / I 0.0 |

##### LCOM4 (components; 1 = cohesive)

| Class | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) |
| --- | --- | --- | --- |
| Workflow | 1 (28 methods) | 1 (30 methods) | 1 (34 methods) |
| Pipeline | 1 (21 methods) | 1 (21 methods) | 1 (21 methods) |
| Result | 1 (15 methods) | 1 (15 methods) | 1 (15 methods) |
| JobManager | 1 (29 methods) | 1 (30 methods) | 1 (29 methods) |

##### Ten most complex functions at Gate 1

| Complexity | Function |
| --- | --- |
| 416 | dw/server/app.py:740 `create_app` |
| 79 | dw_mcp/server.py:65 `build_server` |
| 34 | dw/workflow.py:1352 `run` |
| 27 | dw/arguments.py:103 `realize_args` |
| 26 | dw/result.py:811 `save_artifact` |
| 24 | dw/locations.py:612 `_walk` |
| 24 | dw/server/admission.py:242 `argument_reference_errors` |
| 22 | dw/media_info.py:19 `probe_media` |
| 22 | dw/tasks/loop_bed.py:397 `find_loop_bed` |
| 21 | dw/teacache.py:89 `_create_flux_teacache_forward` |

##### Change coupling at Gate 1 (commits since 2026-08-01)

| Shared commits | Degree % | File | File |
| --- | --- | --- | --- |
| 13 | 55 | dw/tasks/concat_videos.py | dw/tasks/dissolve_videos.py |
| 13 | 44 | dw/task_domains.py | dw/tasks/task.py |
| 5 | 43 | dw/security.py | dw/type_helpers.py |
| 26 | 37 | dw_mcp/catalog.py | dw_mcp/server.py |
| 44 | 33 | dw/server/app.py | dw_mcp/server.py |
| 23 | 33 | dw_mcp/media.py | dw_mcp/server.py |
| 7 | 32 | dw_mcp/authoring.py | dw_mcp/diagnose.py |
| 21 | 31 | dw_mcp/diagnose.py | dw_mcp/server.py |
| 8 | 30 | dw/tasks/concat_videos.py | dw/tasks/video_utils.py |
| 14 | 29 | dw/tasks/audio_utils.py | dw/tasks/task.py |

##### Hotspots at Gate 1 (churn x total complexity)

| Score | Commits | Complexity | File |
| --- | --- | --- | --- |
| 147420 | 162 | 910 | dw/server/app.py |
| 20240 | 115 | 176 | dw/workflow.py |
| 16848 | 108 | 156 | dw_mcp/server.py |
| 16576 | 56 | 296 | dw/pipeline_processors/pipeline.py |
| 11224 | 61 | 184 | dw/result.py |
| 10450 | 55 | 190 | dw/tasks/audio_utils.py |
| 10148 | 59 | 172 | dw/server/jobs.py |
| 4212 | 27 | 156 | dw/arguments.py |
| 3912 | 24 | 163 | dw/introspection.py |
| 3864 | 42 | 92 | dw/tasks/task.py |

### Gate 2 (2026-09-30, develop c5539dd1)

- **Gate criteria.**
  - `validation_errors` is a loop over one check registry (`dw/validation.py`), and a crashing check is an internal finding rather than a lost verdict.
  - No module but `dw/references.py` spells a reference prefix (`prefix_literals` 0).
  - The worker runs the definition admission checked: no re-read, no re-validate. Its messages are typed, and one dispatcher matches replies by request id.
  - Pipeline identity has one home (`dw/step_cache.py`), computed once per run. A borrow chain is part of it, which fixes a stale pipeline reuse and a stale step-cache hit.
- **Real-model timings on lem** (RTX 3090, deployed `c5539dd1`):
  - `templates/ltx2/two-stage`, cold, through `dw.run` over ssh with `seed=779`: exit 0, **189 s** wall, including the first worker spawn after the deploy restart. Gate 1 took 195 s.
  - `validate_workflow` for the same seed then reported `cached_steps: 3`, and the bound-acknowledgement rerun over MCP finished in **0.84 s** with every step reused. Gate 1 took 0.77 s.
  - A `for_each` check, an inline three-member SD 1.5 workflow with a seed: cold **13.8 s** with members `shot@apple` / `shot@pear` / `shot@plum`, and the rerun **0.84 s** with all three reused.
  - The catalog's `for_each` templates were not used. They set no seed, so they cannot show caching, and they run 19-42 min.
- **Ratchets.** Every one equal or lower; `baseline.json` was re-baselined at this gate (`test_dw_patch_targets` 285 → 284).
- **Surface changes (release notes).** Collected per stage in [phase-2.md](phase-2.md):

- **2a:** reference prefixes are spelled in backticks instead of quotes in nine MCP tool descriptions and in `compose_text`'s `parts` description (served by `GET /api/tasks/compose_text` and MCP `get_task`). The wording is unchanged.
- **2a:** `modules` 132 → 133 (`dw/references.py`, accepted by Don 2026-09-28).
- **2a follow-ups:**
  - `dw/workflow.py:747` could use `author_index`.
  - Prefix tests are spelled three ways.
  - Four import styles for `references`.
  - Prefixes inside f-strings, which the metric does not count.
- **2b, validation (B10):**
  - A crashing error check becomes one finding with a null path, naming the check and the exception type only: `check '<name>' failed (<ExcType>) - the server log has the detail`. The verdict is `valid: false`, and every other check still reports.
    - `/api/validate` already answered 200 `valid: false` on a crash; before, one crash replaced the whole verdict with a single generic line.
    - The 400 detail on `POST /api/jobs` and rerun, `Workflow.validate()`, `python -m dw.validate`, the `PUT /api/workflows/{name}` 400 and the worker's re-validate message all carry that sentence instead of the raw exception text.
  - When validation itself fails outright (context build, gates, argument checks), the submit and rerun 400 detail is `validation failed (<ExcType>) - the server log has the detail`.
  - A crashing warning source becomes one `internal: warning check '<name>' failed (<ExcType>) - the server log has the detail` warning and never refuses. The save route (`PUT /api/workflows/{name}`) can now return one where it used to drop the failure.
  - A crashing check inside a sub-workflow is reported at `steps[N].workflow.path`.
  - A 400 detail for any null-path error no longer starts with `None: `.
  - B10 is partly closed: `submit_job` still maps every exception to 400 (Phase 3, with the app.py routers).
- **2b, the four rules written twice (one home each in `dw/task_domains.py`):**
  - The run's frame-size error names every mismatched video by its real index. Before, it stopped at the first mismatch and always called the reference "video 0".
  - Validate's slice past-end warning uses the run's sample arithmetic, including #557's end rounding, so a verdict at the 10 ms threshold can change; its "X s requested" figure is computed in samples.
  - Validate's slice check honours a literal numeric `sample_rate` as a relabel, like the run (#180). A string `sample_rate` is ignored, and validation uses the file's own rate.
  - Validate refuses a literal `null` `threshold` or `index` on `select`, which the run already refused.
- **2b, media probing (B9):** validation reads header values and counts demuxed packets instead of decoding whole files, once per file per request. A `{"location": ...}` media entry is now probed like a plain path: a `dissolve_videos` input too short for its overlap is refused, and the slice past-end and shot-span warnings are raised, where before these entries were silently skipped.
- **2b, internal:** `dw.result_fps`, `dw.null_media` and `dw.select_validation` are deleted; their functions are in `dw.validation`. `unseeded_cache_warnings` moved to `dw.validation` (`dw.plan` re-exports it). `MEMBER_SEPARATOR` and `render_path` moved to `dw.references` (`dw.for_each` re-exports them). `tasks.select._THRESHOLD_RULES` is renamed `_THRESHOLD_TESTS`.
- **2b, metrics:** `modules` 133 → 131, `functions_over_150_lines` 17 → 16, `modules_in_import_cycles` 26 → 25.
- **2b follow-ups:** `/api/validate` logs a gate failure twice (`admit()` and `app.py`); temp-dir leaks in the dissolve and shot-span test helpers; the frame-size prefix sentence is still written by both callers; the dissolve run raises only the first shortfall; the slice region arithmetic is still in both `slice_audio` and `slice_preflight`.
- **2c, the worker runs what admission checked:**
  - A job carries the definition admission checked and runs it, even if its file is edited or deleted while the job waits. An edit reaches only jobs submitted after it; a rerun admits the file afresh, as before.
  - This covers the workflow's own definition only. Sub-workflow files and the assets, outputs and prompts a workflow references are still read when the step that needs them runs.
  - The worker no longer re-checks a job when it starts, so its re-validate failure message is gone. An asset, output, prompt or sub-workflow file that changes or disappears while a job waits now fails at the step that reads it, not at job start.
  - `probe_cache` (the plan's `cached_steps`) answers on the admitted definition.
- **2c, worker protocol (internal):**
  - The messages are typed (`dw/worker.py`); the queue still carries dicts of the old shapes.
  - `probe_id` became `request_id`. `memory_status`, `clear_memory` and `probe_cache` commands carry one, and their replies, including an `error` reply to one of them, echo it. `execute` and `probe_cache` carry `definition`, `file_spec` and `source` instead of `workflow_path` / `workflow` / `base_dir`.
  - A reply that answers no waiting request is discarded at DEBUG instead of logged as an unknown type, and a late error reply from a timed-out request no longer fails the next job.
  - A worker crash answers a waiting memory request at once, instead of after its timeout.
  - `ping`, `pong` and `shutdown_complete` are removed.
- **2c, `dw.run`:** `python -m dw.run` prints every event of a job whose tail ran past one page, and prints the history `note` once.
- **2c follow-ups:** `JobManager.definition()` still re-reads a path job's file (Phase 3); `dw/worker.py` is 12 lines under the 1000-line limit until Phase 3 moves the message types out; the snapshot parity test is close to tautological; `workflow_from_snapshot` passes `file_spec` through unnormalized when `workflow_dir` is None (server jobs always set it).
- **2d, borrow chains are part of pipeline identity (bug fix):**
  - A pipeline that reuses another step's components is identified by every pipeline definition in its reuse closure. The closure includes intermediate steps that pass a component on.
  - A step that `pipeline_reference`s such a pipeline is identified the same way.
  - A change anywhere up the chain now reloads the reusing pipeline and misses the step cache below it. Before, a stale resident pipeline, still holding the old shared component, was reused, and a stale cached result was republished.
  - Identical pass-through steps (for example `for_each` members that reuse and re-share one component) still share one pipeline. Two steps with the same own definition that borrow from different sources are now two pipelines.
  - One-time cost after the deploy: the reusing step misses and reloads once in `base-and-refiner` (`main`), `ltx2/generative-upscale` (`upscaled`), `ltx2/refine-clip` (`refine`) and `ltx2/two-stage` (`upscale`).
  - An elided step's `release_pipeline` carries to its predecessor only when their effective keys match.
- **2d, loading never edits the workflow's definition:**
  - `Pipeline` keeps its own container copy of the definition.
  - Embedded image metadata no longer carries a `"generator": "<torch._C.Generator …>"` string. Its `workflow` block keeps `loras` and `ip_adapter`, which a load used to pop, so "open as workflow" from the gallery works for adapter steps.
- **2d, media copies:**
  - Variables are resolved, substituted, recorded and realized without deep-copying media leaves.
  - A composed child copies only the arguments it declares, once, on entry, so an in-place write inside the child (`conform_artifact` restamping `fps`) never reaches the parent's result.
  - Python API only: an object a top-level caller passes to `Workflow.run` is the object the steps use, so an in-place write to it is visible to the caller. Server, MCP and CLI callers pass JSON and are unaffected.
- **2d, Python import surface:**
  - `dw.runs._run_lock_path` is renamed `run_lock_path`.
  - `pipeline_cache_key` and `step_pipeline_keys` moved from `dw.workflow` to `dw.step_cache`, and `component_names` from `dw.pipeline_processors.pipeline`. None is re-exported.
  - `step_pipeline_keys` returns effective keys.
  - A step missing from its run's key table raises an internal error rather than re-hashing.
- **2d, metrics:** `import_cycles` 6 → 5, `modules_in_import_cycles` 25 → 19.
- **2d follow-ups:**
  - The three scanners of `previous_result:` shapes.
  - `for_each._copy_leaf` and the step-cache snapshot still copy media leaves.
  - A child handed media through a substituted variable stores it in its `argument_template`, which is copied again by validation.
  - The elision carry assumes dict pipelines.

##### Metrics

| Metric (lower is better) | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) | Gate 2 (`stabilization-gate-2`) |
| --- | --- | --- | --- | --- |
| Engine + MCP modules | 133 | 133 | 132 | 131 |
| Modules over 1,000 lines | 10 | 10 | 10 | 10 |
| Functions over 150 lines | 19 | 19 | 17 | 16 |
| Functions over cyclomatic complexity 15 | 21 | 21 | 19 | 19 |
| Functions over cyclomatic complexity 30 | 4 | 4 | 3 | 3 |
| Import cycles | 6 | 6 | 6 | 5 |
| Modules inside import cycles | 22 | 26 | 26 | 19 |
| Duplicate-code blocks, cross-file | 21 | 21 | 19 | 5 |
| Reference-prefix literals | 100 | 100 | 100 | 0 |
| Test `patch("dw...")` targets | 284 | 285 | 285 | 284 |
| CLAUDE.md lines, all files | 964 | 964 | 961 | 961 |

##### Complexity distribution (ruff C901)

|  | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) | Gate 2 (`stabilization-gate-2`) |
| --- | --- | --- | --- | --- |
| Functions | 1712 | 1719 | 1685 | 1735 |
| Median complexity | 2.0 | 2 | 2 | 2 |
| Mean complexity | 3.78 | 3.79 | 3.76 | 3.71 |
| Over 10 | 64 | 64 | 61 | 62 |
| Over 15 | 21 | 21 | 19 | 19 |
| Over 20 | 11 | 11 | 10 | 10 |
| Over 30 | 4 | 4 | 3 | 3 |

##### SLOC by layer (pygount code lines; reference only)

| Layer | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) | Gate 2 (`stabilization-gate-2`) |
| --- | --- | --- | --- | --- |
| Engine | 19980 | 20102 | 19410 | 19970 |
| API (dw/server + dw_mcp) | 6498 | 6525 | 6593 | 6577 |
| UI (ui/src, tests excluded) | 2266 | 2266 | 2266 | 2266 |

##### Package instability, I = Ce / (Ca + Ce)

| Package | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) | Gate 2 (`stabilization-gate-2`) |
| --- | --- | --- | --- | --- |
| dw (core) | Ca 31 / Ce 10 / I 0.24 | Ca 31 / Ce 11 / I 0.26 | Ca 32 / Ce 12 / I 0.27 | Ca 33 / Ce 11 / I 0.25 |
| dw.pipeline_processors | Ca 2 / Ce 4 / I 0.67 | Ca 3 / Ce 4 / I 0.57 | Ca 3 / Ce 4 / I 0.57 | Ca 2 / Ce 4 / I 0.67 |
| dw.server | Ca 1 / Ce 8 / I 0.89 | Ca 1 / Ce 8 / I 0.89 | Ca 1 / Ce 9 / I 0.9 | Ca 1 / Ce 9 / I 0.9 |
| dw.tasks | Ca 11 / Ce 20 / I 0.65 | Ca 11 / Ce 20 / I 0.65 | Ca 11 / Ce 20 / I 0.65 | Ca 11 / Ce 21 / I 0.66 |
| dw_mcp | Ca 1 / Ce 0 / I 0.0 | Ca 1 / Ce 0 / I 0.0 | Ca 2 / Ce 0 / I 0.0 | Ca 2 / Ce 0 / I 0.0 |

##### LCOM4 (components; 1 = cohesive)

| Class | Before Phase 0 (`3afd70e9`) | Gate 0 (`stabilization-gate-0`) | Gate 1 (`stabilization-gate-1`) | Gate 2 (`stabilization-gate-2`) |
| --- | --- | --- | --- | --- |
| Workflow | 1 (28 methods) | 1 (30 methods) | 1 (34 methods) | 1 (39 methods) |
| Pipeline | 1 (21 methods) | 1 (21 methods) | 1 (21 methods) | 1 (21 methods) |
| Result | 1 (15 methods) | 1 (15 methods) | 1 (15 methods) | 1 (15 methods) |
| JobManager | 1 (29 methods) | 1 (30 methods) | 1 (29 methods) | 1 (31 methods) |

##### Ten most complex functions at Gate 2

| Complexity | Function |
| --- | --- |
| 415 | dw/server/app.py:741 `create_app` |
| 79 | dw_mcp/server.py:65 `build_server` |
| 35 | dw/workflow.py:1203 `run` |
| 27 | dw/arguments.py:104 `realize_args` |
| 26 | dw/result.py:811 `save_artifact` |
| 24 | dw/locations.py:597 `_walk` |
| 24 | dw/server/admission.py:191 `argument_reference_errors` |
| 22 | dw/media_info.py:19 `probe_media` |
| 22 | dw/tasks/loop_bed.py:397 `find_loop_bed` |
| 21 | dw/teacache.py:89 `_create_flux_teacache_forward` |

##### Change coupling at Gate 2 (commits since 2026-08-01)

| Shared commits | Degree % | File | File |
| --- | --- | --- | --- |
| 13 | 51 | dw/tasks/concat_videos.py | dw/tasks/dissolve_videos.py |
| 13 | 43 | dw/task_domains.py | dw/tasks/task.py |
| 5 | 43 | dw/security.py | dw/type_helpers.py |
| 26 | 36 | dw_mcp/catalog.py | dw_mcp/server.py |
| 45 | 33 | dw/server/app.py | dw_mcp/server.py |
| 23 | 33 | dw_mcp/media.py | dw_mcp/server.py |
| 7 | 32 | dw_mcp/authoring.py | dw_mcp/diagnose.py |
| 14 | 31 | dw/tasks/audio_utils.py | dw/tasks/concat_videos.py |
| 7 | 31 | dw/tasks/dissolve_videos.py | dw/tasks/video_utils.py |
| 21 | 30 | dw_mcp/diagnose.py | dw_mcp/server.py |

##### Hotspots at Gate 2 (churn x total complexity)

| Score | Commits | Complexity | File |
| --- | --- | --- | --- |
| 150728 | 166 | 908 | dw/server/app.py |
| 23168 | 128 | 181 | dw/workflow.py |
| 17582 | 59 | 298 | dw/pipeline_processors/pipeline.py |
| 17004 | 109 | 156 | dw_mcp/server.py |
| 11224 | 61 | 184 | dw/result.py |
| 11136 | 58 | 192 | dw/tasks/audio_utils.py |
| 10431 | 61 | 171 | dw/server/jobs.py |
| 4524 | 29 | 156 | dw/arguments.py |
| 4509 | 27 | 167 | dw/introspection.py |
| 3864 | 42 | 92 | dw/tasks/task.py |

## Working rules for the duration

- Hard freeze: no net-new features or functionality. The refactor may change
  surface (routes, tools, file layout) where consolidation requires it, and
  nothing more.
- Refactor work happens on `stabilization/phase-N` branches, merged to
  `develop` and deployed to lem at each gate.
- The harness keeps running testers and field-bug fixes (freeze prompt:
  [harness/stage-a-freeze.md](harness/stage-a-freeze.md)). It does not touch
  files listed in [hot-zone.txt](hot-zone.txt), which names what the current
  phase is restructuring.
- Metrics: `scripts/arch_metrics.py` (Phase 0, Task 1) is the one source of
  numbers for the phase gates here and for the harness's ratchet later.
- No new count-pinning tests; every new test fails before its fix.
