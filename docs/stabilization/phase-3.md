# Phase 3 Implementation Plan: structural moves

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every concept has one owner, and every owner fits in one file. No module over 1,000 lines, no function over 150, no import cycle.

**Architecture:** Five stages, each merged to `develop` on its own with its own hot zone, as in Phase 2. A stage's detailed tasks are written when the previous stage merges, on the code that stage left. Stage 3a is detailed below. The order is:
- cycles first, because they decide where the later splits can put code;
- the server next, as a move that doesn't change the surface;
- the one surface change (LibraryPath) on the smaller files that move leaves;
- media and DSP before the engine splits, so `result.py`'s audio code has a layer to land in.

**Tech Stack:** Python 3.10+, FastAPI `APIRouter`, pytest; metrics from `scripts/arch_metrics.py` and `scripts/arch_report.py`.

**Spec:** [ROADMAP.md](ROADMAP.md) (Phase 3 row: "No module over 1,000 lines, no function over 150; suite and lem smoke green"; Metrics: "The target is 0 for both cycle ratchets, reached by Phases 2-3") and [ASSESSMENT.md](ASSESSMENT.md). Items carried from gate 2 are listed under "Carried from gate 2".

## Where Phase 3 starts (gate 2, `c5539dd1`)

- **Modules over 1,000 lines (10):**

  | Module | Lines |
  | --- | --- |
  | `dw/server/app.py` | 4,521 |
  | `dw/pipeline_processors/pipeline.py` | 2,415 |
  | `dw/workflow.py` | 2,229 |
  | `dw/tasks/audio_utils.py` | 2,229 |
  | `dw/result.py` | 1,830 |
  | `dw/server/jobs.py` | 1,567 |
  | `dw_mcp/server.py` | 1,344 |
  | `dw/introspection.py` | 1,242 |
  | `dw/arguments.py` | 1,235 |
  | `dw/security.py` | 1,038 |

- **Functions over 150 lines (16, as the ratchet counts them):**

  | Function | Lines |
  | --- | --- |
  | `create_app` | 3,780 |
  | `build_server` | 1,279 |
  | `Workflow.run` | 566 |
  | `find_loop_bed` | 361 |
  | `Result.save_artifact` | 347 |
  | `concat_videos` | 332 |
  | `create_step_action` | 277 |
  | `serve.main` | 258 |
  | `plan.estimate` | 200 |
  | `teacache._create_flux_teacache_forward`, and its nested forward | 196 and 185 |
  | `probe_media` | 183 |
  | `attribute_voices` | 174 |
  | `gallery_frames` | 164 |
  | `worker._handle_execute` | 164 |
  | `join_into_song` | 153 |

- **Import cycles (5 components, 19 modules):**
  - an 11-module knot: `arguments`, `content_types`, `for_each`, `locations`, `media_frames`, `result`, `runs`, `shots`, `tasks.audio_utils`, `tasks.video_utils`, `variables`;
  - `introspection` ↔ `tasks.task`;
  - `security` ↔ `workspace`;
  - `server.app` ↔ `server.mcp_mount`;
  - `vram_estimate` ↔ `vram_inheritance`.

The surveys behind the stages are summarized in each stage's "What exists" block when the stage is detailed. Raw notes: the session scratchpad; the facts that matter are copied into this file.

## Stages

| Stage | Scope | Done when |
| --- | --- | --- |
| 3a | **Import cycles to zero.** Reference keys into `references`; audio format tables into `content_types`; the step value types (`AudioVideo`, `AudioTrack`) and the three helpers `result.py` borrows from `tasks` into one leaf module; the frame-grid helpers into `media_frames`; the four 2-module cycles each cut at their one edge. | `import_cycles` and `modules_in_import_cycles` are 0 and ratcheted there; `result.py` imports nothing from `dw.tasks` but `tasks.select` |
| 3b | **Server into routers and services, surface unchanged.** `create_app` becomes an app factory plus one `APIRouter` module per resource (jobs, workflows, prompts, gallery, assets, workspaces, models/system, memory/health, static). The closure helpers become module functions over `request.app.state`. `jobs.py` splits into history / job / manager. `serve.main` is cut up. `dw_mcp/server.py`'s tools move beside their handler modules, with docstrings byte-identical (the surface token budget test). The worker's message types move out of `worker.py`, and `_handle_execute` is cut up. B10's rest: `submit_job` answers 400 only for an admission refusal. | No module in `dw/server`, `dw_mcp`, `dw/serve.py` or `dw/worker.py` over 1,000 lines and no function there over 150; the route table (method, path, order, middleware order) is identical |
| 3c | **One `LibraryPath` and one library surface (breaking).** `WorkflowSource` generalizes into the one ordered, origin-tagged search path for workflows, prompts and assets. That covers: writable root first, shadowing, write-target selection, the read-only refusal, and serialization into the worker's environment with origins. It replaces the asset order written four times, the prompt order written twice, and the three origin mechanisms (survey D1-D11). The workflows, prompts and assets listings take one shape, and the UI, MCP and `dw.run` change in the same stage. Version bump. | Each library's search order is computed in one place, and the API and the worker read the same serialized path; the three listings share one field set |
| 3d | **One media I/O module and one DSP module.** One `av.open` decode/probe layer (the audio-to-float decode is written 5 times, the header-rate probe 7). A pure numpy/scipy `dsp` module out of `audio_utils.py`, with the task commands left as thin wrappers. `concat_videos` and `dissolve_videos` share one join. The long task functions are cut. Build-vs-buy deletions: the biquad's pure-Python fallback and its test, five `dbfs` copies, the second true-peak oversampler, `file_fps`. The teacache decision (below) lands here. | `av.open` appears only in the media module; `audio_utils.py` and the task modules under 1,000 lines and their functions under 150 |
| 3e | **Engine splits.** `result.py` keeps `Result` (writers, audio QC and output extraction move out). `pipeline.py` keeps `Pipeline` (placement, components, adapters, progress reporting move out). `Workflow.run` and `create_step_action` are cut into named phases, and the pipeline-ownership dicts become one object. `Workflow`'s validation block joins validation. `arguments.py` loses its media half. `introspection.py` loses type-reference checks and inert-argument warnings. `security.py` splits, with CodeQL re-modelled. The carried follow-ups that live in these files. | The Phase 3 gate criteria hold everywhere |

Gate 3 follows 3e, with the same checks as gate 2:
- the full metrics report, with a Gate 3 column and LCOM4 for `create_app`'s successor;
- a real-model timing on lem: the B2 cached rerun, one `for_each` run, and one run through each library (a `prompt:` and an `asset:` reference) after 3c;
- deploy, tag and re-baseline.

## Carried from gate 2

Each item goes to the stage whose files it touches:
- `dw/worker.py` message types move to their own module → 3b;
- `JobManager.definition()` re-reads a path job's file; a live job answers from its snapshot → 3b;
- `/api/validate` logs a gate failure twice → 3b;
- B10's rest (`submit_job` maps every exception to 400) → 3b;
- the three `previous_result` scanners → 3e;
- `for_each._copy_leaf` still copies media → 3e;
- the elision dict assumption → 3e;
- `result.py`'s import of `tasks.select.Selected` → 3e (outside the cycles; it is the last upward import).

## Release notes collected (for gate 3)

(Each stage adds its user-visible changes here at merge.)

## Global Constraints (all stages)

- Hard freeze: no net-new features or functionality. Surface may change only where the consolidation requires it (3c), and every change is listed for the gate's release notes.
- `scripts/arch_metrics.py --check docs/stabilization/baseline.json` passes at the end of every task, except for the `modules` rise a task's plan names (Decisions).
- Every new test fails before its fix. No count-pinning tests. Add no string `patch("dw...")` targets. A task that moves a patched name retargets the patch in the same commit (Decisions, no shims).
- Behavior-preserving tasks prove preservation with the existing suite: it passes unchanged, apart from the import paths and patch targets the move changes and the tests the task names.
- Tests: `venv/bin/python -m pytest -q -x -p no:cacheprovider` from the worktree root, plus `ruff check` and `ruff format --check` on `dw dw_mcp tests`. The worktree's `venv` is shared with the main checkout, so never run `pip install -e .` from the worktree.
- Never use `git stash`, in any form, including `git stash list`.
- Filesystem access keeps going through a `dw/security.py` validator. A validator that moves is re-modelled in `.github/codeql/` in the same commit (CLAUDE.md, Security Rules).

## Decisions (rulings, 2026-09-30)

- **The `modules` ratchet rises in Phase 3, by exactly the modules each stage names.**
  - Why: the phase's gate is about file size, and splitting a 4,521-line file is adding files. The ratchet exists to stop one-module-per-ticket drift, which a planned split is not.
  - How: each stage's plan lists its new modules. The stage's merge re-baselines `modules` to the count it names, with the list in the commit. A module no stage names is still a regression.
  - Expected: 3a +1 (`dw/media_types.py`); 3b about +14 (routers and services, the jobs split, the MCP tool modules, the worker protocol); 3c 0 (`LibraryPath` generalizes `workflow_sources.py`); 3d about +2 (the media and DSP modules, net of folding `media_audio` / `media_info` in); 3e about +10. That is roughly 131 → 158.
  - Cost if wrong: the ratchet is looser than it looks until Phase 4 re-sets it. Don accepts or rejects this before 3a starts.
- **No compatibility shims.** A moved name is imported from its new home by every caller, tests included.
  - Why: a re-export is a second path to the same name, which is the sprawl this phase removes. It is also dangerous: a `patch("dw.old.name")` against a re-export patches a name nothing looks up, so the test passes without testing anything.
  - Exception: a name the harness or a user entry point reaches by its old path (`dw.server.app.create_app`, patched by `dw.serve` tests at call time) stays where the lookup happens.
  - Cost if wrong: more churn in test imports per task (mechanical; `sed` plus the suite).
- **Breaking HTTP/MCP changes are confined to 3c,** where the three library listings take one shape. The gate 3 release bumps the minor version (0.6 → 0.7) through `scripts/release.sh`, which also moves `plugin.json`. 3b keeps every route path, method, status and body. Don confirms the version number at 3c.
- **Teacache: proposed for deletion in 3d.**
  - The case for:
    - `dw/teacache.py` (381 lines, the 196-line forward factory) is Flux-only.
    - No shipped workflow uses the `teacache` key.
    - diffusers ships `first_block`, `mag` and `taylorseer` cache hooks, which dw already routes, and `docs/ACCELERATION.md` already steers users to them.
  - What deleting it costs a user: the `rel_l1_thresh` knob and its Flux speed table. It is a breaking change to a documented workflow key, noted in the release notes. **Don decides before 3d.** If kept, 3d cuts the factory under 150 lines instead.
- **Build-vs-buy exceptions, kept on purpose:**
  - The compressor/limiter/gate and the true-peak look-ahead limiter with its 0.1 LU gain search stay hand-written.
  - `pedalboard`, the one candidate, is GPL-3.0 against this Apache-2.0 project, and it computes differently: tests pin the current numbers to 1e-5.
  - LUFS and true-peak already use `pyloudnorm` and `scipy`.
  - 3d relocates this code; it does not rewrite it.
- **B6 (the MCP mount shares one session) stays deferred.** The router split does not need it, and the single-user mount is by design. Revisit only if multi-agent use of one server becomes a requirement.

---

## Stage 3a: import cycles to zero

Work on branch `stabilization/phase-3a` in the worktree, from `develop` at `9e25c49a` or later.

**What exists (survey, 2026-09-30, at `9e25c49a`).** Every edge below was read in the code; line numbers are at that commit.

- **Reference keys live in `arguments.py`,** and four low modules import `arguments` only for them:
  - `FROM_FILE_KEY` (`arguments.py:52`), `FROM_PREVIOUS_RESULT_KEY` (55) and `FROM_ARGUMENTS_KEY` (59) are defined there.
  - Two aliases sit beside them: `PREVIOUS_RESULT_PREFIX = references.PREVIOUS_RESULT` (64) and `CONSTANT_PREFIX = references.CONSTANT` (69).
  - The modules that import them:
    - `shots.py:38` (`PREVIOUS_RESULT_PREFIX`);
    - `for_each.py:34`;
    - `variables.py:4`;
    - `vram_estimate.py:50`.
  - Other importers of the five names: `previous_results.py`, `workflow.py`, `validation.py`, `video_extensions.py`.
- **`content_types.py` imports upward:**
  - `from .for_each import MEMBER_SEPARATOR, render_path` (29). Both names are defined in `references.py`, and `for_each` re-exports them.
  - `from .result import AUDIO_FORMATS, MUXED_VIDEO_CONTENT_TYPE` (30). Its own docstring calls `result.py`'s table the source of truth for what it validates.
- **The step value types live in `result.py`:** `AudioVideo` (458-487) and `AudioTrack` (490-518), both plain classes with no `dw` dependency.
  - `tasks/video_utils.py:18` imports `AudioVideo` at module level.
  - `tasks/audio_utils.py:493` imports `AudioTrack` lazily.
- **`result.py` imports upward into `tasks`** for three helpers:
  - `_warn_on_rate_override` (`audio_utils.py:1906-1931`, uses only `emit_warning`), imported at `result.py:926`;
  - `AUDIO_FIT_TOLERANCE_SECONDS` (`video_utils.py:582`), imported at `result.py:1065`;
  - `_fit_audio_to_frames` and `_sample_axis` (`video_utils.py:585-637`, numpy/torch only), imported at `result.py:1238`.
  - `result.py:583` also imports `tasks.select.Selected`. That import is not in any cycle and is left for 3e.
- **`media_frames.py:16` imports five private grid helpers from `tasks/video_utils.py`:**
  - the helpers: `_compose_grid`, `_default_columns`, `_evenly_spaced_indices`, `_format_timestamp`, `_grid_tile` (about lines 275-337);
  - `video_utils.py:72` imports `frames_at` back, lazily;
  - `frame_grid` in `video_utils` is the helpers' only other caller.
- **`introspection` ↔ `tasks.task`:**
  - Task side: `Task._check_required_arguments` (`tasks/task.py:867-881`, called at 918, per iteration) lazily imports `missing_task_arguments` / `missing_task_argument_message`. Those call `describe_task`.
  - Introspection side: `introspection.py` imports the command registry from `task` lazily, in five places.
  - `dw/step.py` already sits above both. `Step.run` is what hands each iteration's arguments to the action.
- **`security` ↔ `workspace`:**
  - `validate_workspace_name` (`security.py:862-892`) lazily imports `RESERVED_WORKSPACE_NAMES` (`workspace.py:95`).
  - `workspace.py:465` and `486` lazily import `validate_workspace_name`.
  - CodeQL models `validate_workspace_name` in `dw.security` by name, in two places: `.github/codeql/dw-security/DwPathSanitizers.qll:59` and `.github/codeql/extensions/dw-models/models/dw-security.model.yml:19`. So it stays in `security.py`.
  - `dw/server/CLAUDE.md:22` explains the lazy import.
- **`vram_estimate` ↔ `vram_inheritance`:** `vram_inheritance.py:31-32` imports `KEY` and `vram_estimate_errors` at module level. `vram_estimate.py:155` and `200` lazily import `pipeline_identity` (`vram_inheritance.py:45`, plus its helpers).
- **`server.app` ↔ `server.mcp_mount`:**
  - `app.py:782` lazily imports `build_mcp_app`.
  - `mcp_mount.py:25` lazily imports `LOOPBACK_HOSTS` and `WILDCARD_HOSTS` (`app.py:680-683`).
  - `dw/serve.py:154` imports `LOOPBACK_HOSTS` from `server.app`, which pulls in FastAPI just to read a constant.
  - `dw/server/netinfo.py` imports only the stdlib.
  - `dw_mcp/client.py:14` keeps its own copy on purpose: `dw_mcp` must not import `dw`.

### Decisions (3a)

- **One new module, `dw/media_types.py`.** It holds:
  - the values steps hand each other: `AudioVideo`, `AudioTrack`;
  - the audio helpers `result.py` borrowed from `tasks`, which are about those values: `AUDIO_FIT_TOLERANCE_SECONDS`, `fit_codec_padding` (was `_fit_audio_to_frames`), `sample_axis` (was `_sample_axis`) and `warn_on_rate_override` (was `_warn_on_rate_override`).
  - Its imports: numpy, torch and `dw.events`, nothing else from `dw`.
  - Private names that cross modules become public.
  - `fit_codec_padding` is renamed rather than made public as `fit_audio_to_frames`, because `audio_utils.fit_audio_to_frames` exists with a different policy.
  - `modules` 131 → 132.
- **The reference keys move to `references.py`,** and the two aliases are deleted:
  - `FROM_FILE_KEY`, `FROM_PREVIOUS_RESULT_KEY` and `FROM_ARGUMENTS_KEY` move there;
  - callers of `PREVIOUS_RESULT_PREFIX` / `CONSTANT_PREFIX` use `references.PREVIOUS_RESULT` / `references.CONSTANT`.
- **`AUDIO_FORMATS`, `LOSSY_AUDIO_CONTENT_TYPES` and `MUXED_VIDEO_CONTENT_TYPE` move to `content_types.py`.** `result.py` imports them from there at module level, and its lazy `refuse_active_content_type` import becomes module level too.
- **The required-argument guard moves from `Task` to `Step`.** `Step.run` checks a `Task` action's arguments for each iteration, just before the action runs, with the same message. `Task._check_required_arguments` is deleted. `tasks/task.py` then imports nothing from `introspection`.
- **`validate_workspace_name(name, reserved=())`:** the reserved names become a parameter, which `workspace.py` passes. `security.py` no longer imports `workspace`. The CodeQL models name the function and its return value, so they are unchanged.
- **`pipeline_identity` and its helpers move into `vram_estimate.py`.** `vram_inheritance` imports them from there.
- **`LOOPBACK_HOSTS` and `WILDCARD_HOSTS` move into `dw/server/netinfo.py`.** `app.py`, `mcp_mount.py` and `serve.py` import them from there. `dw_mcp/client.py`'s copy stays, with its comment pointing at the new home.
- **Only one cut is strictly needed per cycle; 3a makes more on purpose.** The minimum is five edges, which a brute-force check on the survey graph confirmed. 3a also removes every low-module import of `arguments`, and every `result` → `tasks` import in the cycle. Why: the assessment names both ("result.py imports upward into tasks"), and 3d/3e need `result` and `arguments` above the layer they split into.

### Review Focus (3a)

1. **A moved name patched at its old path.**
   - Check: every `patch(...)` / `monkeypatch.setattr(...)` in `tests/` whose target is a name 3a moved now targets the module that looks the name up.
   - Reviewer: `git grep -n "_fit_audio_to_frames\|_warn_on_rate_override\|_sample_axis\|dw.result.AudioVideo\|dw.result.AudioTrack"` returns nothing in `tests/` or `dw/`.
2. **`rate_override_mismatch` still reaches the job's warnings, from both callers.**
   - Callers: the result save path (a declared `sample_rate` against an `AudioTrack` carrying another rate) and an audio task (`slice_audio` with a relabelling `sample_rate`).
   - Same `kind`, `command`, `file_rate`, `given_rate`.
   - The existing tests cover both; the reviewer confirms they still import the name from its new home and pass.
3. **A task step missing a required argument fails the same way.**
   - What: the same `ValueError` text (first letter capitalized), raised before the command runs.
   - Cases: a missing argument in the second of two `previous_result` iterations fails on that iteration, not before it runs the first; a list-template (non-dict) `arguments` is not checked.
   - Test: a committed `Step.run` test with a stub command that records whether it ran.
4. **Reserved workspace names are still refused** on `POST /api/workspaces` and on a `?workspace=` lookup, with the same message naming the reserved folders. The existing tests cover it; the reviewer confirms that a call to `validate_workspace_name` with no `reserved` still refuses a bad shape.
5. **`--mcp` host safety is unchanged.**
   - `client_base_url()` picks loopback for a wildcard or loopback bind and the bind host otherwise.
   - `dw.serve --mcp --host 0.0.0.0` with no token still exits 2.
   - `serve.py` no longer imports `dw.server.app` before `create_app`.

### Task 1: Reference keys and format tables to their owners

**Files:**
- Modify:
  - `dw/references.py`: add the three keys;
  - `dw/arguments.py`: remove the keys and the two aliases, and import the keys from `references`;
  - `dw/shots.py`, `dw/for_each.py`, `dw/variables.py`, `dw/vram_estimate.py`, `dw/previous_results.py`, `dw/workflow.py`, `dw/validation.py`, `dw/video_extensions.py`: import from `references`;
  - `dw/content_types.py`: take the three format tables and import `MEMBER_SEPARATOR` / `render_path` from `references`;
  - `dw/result.py`: import the tables and `refuse_active_content_type` from `content_types` at module level.
- Test: every test importing a moved name imports it from its new home.

**Interfaces:**
- Produces:
  - `references.FROM_FILE_KEY = "from_file"`;
  - `references.FROM_PREVIOUS_RESULT_KEY = "from_previous_result"`;
  - `references.FROM_ARGUMENTS_KEY = "from_arguments"`;
  - `content_types.AUDIO_FORMATS`, `content_types.LOSSY_AUDIO_CONTENT_TYPES`, `content_types.MUXED_VIDEO_CONTENT_TYPE`, with their values and comments moved verbatim.

- [ ] **Step 1: Move the keys.**
  - Cut the three `FROM_*_KEY` definitions and their comments from `arguments.py:52-59` and paste them into `references.py` after the prefix constants.
  - Delete `PREVIOUS_RESULT_PREFIX` and `CONSTANT_PREFIX` from `arguments.py`.
  - In every module in the list above, replace:
    - `from .arguments import FROM_…` with `from .references import FROM_…`;
    - `PREVIOUS_RESULT_PREFIX` with `references.PREVIOUS_RESULT`;
    - `CONSTANT_PREFIX` with `references.CONSTANT`.
  - In each module, follow whichever import style (`from . import references` or `from .references import …`) it already uses.
- [ ] **Step 2: Move the format tables.** Cut `AUDIO_FORMATS`, `LOSSY_AUDIO_CONTENT_TYPES` and `MUXED_VIDEO_CONTENT_TYPE`, with their comments, from `result.py` (about 406-438) and paste them into `content_types.py`. In `content_types.py`:
  - import `MEMBER_SEPARATOR` and `render_path` from `references`;
  - remove the `for_each` and `result` imports;
  - update the docstring sentence that names `result.py` as the table's source.

  In `result.py`, import the three names and `refuse_active_content_type` from `content_types` at module level, and remove the lazy import at 766.
- [ ] **Step 3: Update the tests.** `git grep -n "PREVIOUS_RESULT_PREFIX\|CONSTANT_PREFIX\|FROM_FILE_KEY\|FROM_PREVIOUS_RESULT_KEY\|FROM_ARGUMENTS_KEY\|AUDIO_FORMATS\|LOSSY_AUDIO_CONTENT_TYPES\|MUXED_VIDEO_CONTENT_TYPE" tests dw dw_mcp`. Fix every import to the new home.
- [ ] **Step 4: Verify.**
  - Run the full suite, `ruff check`, `ruff format --check` and `scripts/arch_metrics.py --check docs/stabilization/baseline.json`.
  - Record `import_cycles` / `modules_in_import_cycles` in the report. Expect the 11-module knot to shrink; `shots`, `for_each`, `variables` and `content_types` no longer reach `arguments` or `result`.
- [ ] **Step 5: Commit.** `refactor(references): reference keys and audio format tables move to their owners`

### Task 2: `dw/media_types.py`, and the frame-grid helpers into `media_frames`

**Files:**
- Create: `dw/media_types.py`.
- Modify:
  - `dw/result.py`: remove the two classes and the three lazy `tasks` imports;
  - `dw/tasks/video_utils.py`: remove the fit helpers and the grid helpers, and import both from their new homes;
  - `dw/tasks/audio_utils.py`: remove `_warn_on_rate_override`, and import `AudioTrack` and `warn_on_rate_override` at module level;
  - `dw/media_frames.py`: take the five grid helpers as its own, and drop the `tasks.video_utils` import;
  - every `dw/` module importing `AudioVideo` / `AudioTrack` from `result`;
  - tests.
- Test: `git grep -n "from dw.result import.*\(AudioVideo\|AudioTrack\)"` finds nothing. Tests import from `dw.media_types`.

**Interfaces:**
- Produces:
  - `dw.media_types.AudioVideo(frames, audio, sample_rate, fps=None, shots=None)` and `dw.media_types.AudioTrack(audio, sample_rate, source_mean_dbfs=None)`, both unchanged;
  - `AUDIO_FIT_TOLERANCE_SECONDS = 0.25`;
  - `fit_codec_padding(audio, frame_count, frame_rate, sample_rate)`, the old `_fit_audio_to_frames`, with the body unchanged;
  - `sample_axis(audio)`;
  - `warn_on_rate_override(command, actual_rate, given_rate)`.
- `media_frames` keeps the five helpers' names. They were private to `video_utils`, and they stay module-private in their new home, since `video_utils.frame_grid` is their only outside caller. Rename them public (`compose_grid`, `default_columns`, `evenly_spaced_indices`, `format_timestamp`, `grid_tile`) because two modules now use them.

- [ ] **Step 1: Create `dw/media_types.py`.**
  - Module docstring: "The values steps hand each other, and the audio rules that belong to them. A leaf: numpy, torch and events only, so the result writer and the tasks both sit above it."
  - Move the two classes verbatim, with their docstrings, from `result.py`.
  - Move `AUDIO_FIT_TOLERANCE_SECONDS`, `_fit_audio_to_frames` and `_sample_axis` from `video_utils.py`, and `_warn_on_rate_override` from `audio_utils.py`, applying the public names above.
  - Update the docstrings' cross-references to the new names.
- [ ] **Step 2: Repoint the callers.** `git grep -n "AudioVideo\|AudioTrack\|_fit_audio_to_frames\|_sample_axis\|_warn_on_rate_override\|AUDIO_FIT_TOLERANCE_SECONDS" dw dw_mcp`: every import comes from `dw.media_types`, at module level. `result.py` keeps `MODULAR_*_KEYS` and `_NO_PROPERTY`; they are about extracting pipeline output, which stays in `result`.
- [ ] **Step 3: Move the grid helpers.**
  - Cut the five helpers from `video_utils.py` (about 275-337) into `media_frames.py`, with their public names, and update `media_frames`' own call sites.
  - `video_utils.frame_grid` imports them from `media_frames` at module level. `video_utils.py:72`'s lazy `frames_at` import may become module level too.
- [ ] **Step 4: Update the tests.** Point imports and patch targets at the new homes (Review Focus 1). `tests/test_media_frames.py:466` has a comment naming `_default_columns`; update it.
- [ ] **Step 5: Verify.**
  - The suite, ruff, and `arch_metrics --check`. Expect `modules` 132, which regresses the baseline by the one named module; record it for Task 4.
  - `venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import arch_metrics as m; from pathlib import Path; print(m.import_graph(Path('.'))['cycles'])"` shows no component containing `result`, `content_types`, `shots`, `media_frames` or `tasks.video_utils`.
- [ ] **Step 6: Commit.** `refactor(media_types): step values and their audio rules in one leaf; grid helpers in media_frames`

### Task 3: The four two-module cycles

**Files:**
- Modify:
  - `dw/step.py`, `dw/tasks/task.py` (delete `_check_required_arguments` and its call);
  - `dw/security.py`, `dw/workspace.py`, `dw/server/CLAUDE.md:22`;
  - `dw/vram_estimate.py`, `dw/vram_inheritance.py`;
  - `dw/server/netinfo.py`, `dw/server/app.py`, `dw/server/mcp_mount.py`, `dw/serve.py`, and the `dw_mcp/client.py:14` comment.
- Test:
  - `tests/test_task_signature_errors.py:203`, rewritten against `Step.run`;
  - a new `Step.run` test (Review Focus 3);
  - the existing workspace, vram-inheritance and serve/mcp tests.

**Interfaces:**
- Produces: `validate_workspace_name(name: str, reserved=()) -> str`; `netinfo.LOOPBACK_HOSTS` and `netinfo.WILDCARD_HOSTS` (same values).
- Consumes: `introspection.missing_task_arguments(command, names)` and `introspection.missing_task_argument_message(command, missing)`, unchanged, now called from `dw/step.py`.

- [ ] **Step 1: Pin the guard's behaviour before moving it.**
  - In the test file that already exercises `Step.run`, add a test:
    - register a stub task command that records when it runs and requires `image`;
    - run a `Step` over two `previous_result` iterations whose second omits `image`;
    - assert that the first iteration ran, then a `ValueError` whose message starts `Task '<command>' requires 'image'`.
  - It passes on today's code; that is intended. This task preserves behaviour, so this is the characterization test that proves the move kept it. "Fails before its fix" applies to fixes, and 3a fixes nothing.
  - The cycle itself is guarded by the `import_cycles` ratchet (Task 4). Add no test that reads imports.
- [ ] **Step 2: Move the guard.**
  - In `Step.run`, where each iteration's realized arguments are handed to `step_action.run(...)`: if the action is a `Task` and the arguments are a dict, run the same check `Task._check_required_arguments` ran, with the same message capitalization, before the call.
  - Delete `_check_required_arguments` and its call at `task.py:918`.
  - Rewrite `tests/test_task_signature_errors.py:203` (non-dict `arguments` are not checked) against `Step.run`.
- [ ] **Step 3: `validate_workspace_name(name, reserved=())`.**
  - Delete the lazy import, and refuse `name in reserved` with the same message, joined from `reserved`.
  - `workspace.py`'s two callers pass `RESERVED_WORKSPACE_NAMES`.
  - Update `dw/server/CLAUDE.md:22` to say the reserved names are passed in.
  - Confirm that the two CodeQL files need no change: they name the function and its return value.
- [ ] **Step 4: `pipeline_identity` into `vram_estimate`.** Move `pipeline_identity` and every private helper only it uses from `vram_inheritance.py` into `vram_estimate.py`. Delete the two lazy imports in `vram_estimate` and import `pipeline_identity` in `vram_inheritance` from `vram_estimate`. Update the docstring reference at `vram_estimate.py:36`.
- [ ] **Step 5: Host sets into `netinfo`.**
  - Move `LOOPBACK_HOSTS` and `WILDCARD_HOSTS`, with their comments, from `app.py:680-683` into `dw/server/netinfo.py`.
  - `app.py` and `mcp_mount.py` import them from `netinfo`. `mcp_mount`'s import can move to module level: `netinfo` imports only the stdlib.
  - `serve.py:154` imports from `.server.netinfo`.
  - Update the `dw_mcp/client.py:14` comment to name `dw/server/netinfo.py`.
- [ ] **Step 6: Verify.** Run the suite, ruff, and `arch_metrics`. Expect `import_cycles` 0 and `modules_in_import_cycles` 0. If a cycle remains, print it with the one-liner from Task 2, Step 5, and cut the edge it names in this task. Report which edge it was.
- [ ] **Step 7: Commit.** `refactor: the last four import cycles cut at their one edge each`

### Task 4: Stage 3a merge

- [ ] **Step 1: Re-baseline.**
  - Run `venv/bin/python scripts/arch_metrics.py --write docs/stabilization/baseline.json`.
  - Confirm the diff against the committed baseline is exactly:
    - `modules` 131 → 132 (`dw/media_types.py`, named in this plan);
    - `import_cycles` 5 → 0;
    - `modules_in_import_cycles` 19 → 0;
    - anything that went down.
  - Any other rise is a finding, not a re-baseline.
- [ ] **Step 2: Docs.**
  - CLAUDE.md: find any sentence naming a moved symbol's old home (`git grep -n "result.py.*AudioVideo\|AudioVideo.*result.py\|_fit_audio_to_frames\|_warn_on_rate_override" CLAUDE.md dw/**/CLAUDE.md docs/`) and fix it.
  - Release notes: no user-visible change. Record "internal: import cycles removed" under this file's release notes.
- [ ] **Step 3: Hot zone.** Replace the 3a entries in `docs/stabilization/hot-zone.txt` with stage 3b's once 3b is detailed. Until then, the standing two entries only.
- [ ] **Step 4: Merge.**
  - Merge `stabilization/phase-3a` to `develop` with `--no-ff`, and push.
  - No lem deploy (deploys are at the gate).
  - Report the metrics table's changed rows to Don.
- [ ] **Step 5: Detail stage 3b** in this file, on the code 3a left, with its hot zone. Cross-check the design with Fable before Task 1 of 3b.

### Hot zone (3a)

While 3a runs, `docs/stabilization/hot-zone.txt` lists:

```
dw/references.py
dw/arguments.py
dw/shots.py
dw/for_each.py
dw/variables.py
dw/vram_estimate.py
dw/vram_inheritance.py
dw/content_types.py
dw/result.py
dw/media_types.py
dw/media_frames.py
dw/tasks/video_utils.py
dw/tasks/audio_utils.py
dw/tasks/task.py
dw/step.py
dw/introspection.py
dw/security.py
dw/workspace.py
dw/server/netinfo.py
dw/server/mcp_mount.py
dw/serve.py
```

Other files 3a touches only for an import line: `previous_results.py`, `workflow.py`, `validation.py`, `video_extensions.py`, `app.py`. They are not listed, so the harness keeps them. A conflicting harness edit to one of them is an import-line merge conflict, which is cheap.
