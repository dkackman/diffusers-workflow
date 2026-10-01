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

The four surveys behind the stages (server, engine, media/DSP, libraries) are committed in [phase-3-surveys/](phase-3-surveys/). They were taken at `9e25c49a`, so re-verify their line numbers before relying on them. Each stage's "What exists" block, written when the stage is detailed, copies in the facts it relies on after checking them in the code.

## Stages

| Stage | Scope | Done when |
| --- | --- | --- |
| 3a | **Import cycles to zero.** Reference keys into `references`; audio format tables into `content_types`; the step value types (`AudioVideo`, `AudioTrack`) and the three helpers `result.py` borrows from `tasks` into one leaf module; the frame-grid helpers into `media_frames`; the four 2-module cycles each cut at their one edge. | `import_cycles` and `modules_in_import_cycles` are 0 and ratcheted there; `result.py` imports nothing from `dw.tasks` but `tasks.select` |
| 3b | **Server into routers and services, surface unchanged.** `create_app` becomes an app factory plus one `APIRouter` module per resource (jobs, workflows, prompts, gallery, assets, workspaces, models/system, memory/health, static). The closure helpers become module functions over `request.app.state`. `jobs.py` splits into history / job / manager. `serve.main` is cut up. `dw_mcp/server.py`'s tools move beside their handler modules, with docstrings byte-identical (the surface token budget test). The worker's message types move out of `worker.py`, and `_handle_execute` is cut up. B10's rest: `submit_job` answers 400 only for an admission refusal. | No module in `dw/server`, `dw_mcp`, `dw/serve.py` or `dw/worker.py` over 1,000 lines and no function there over 150; the surface snapshot is unchanged: the same set of (method, path), the same order within each greedy `{name:path}` family, the same middleware order, the same OpenAPI document and the same MCP tool list |
| 3c | **One `LibraryPath` and one library surface (breaking).** `WorkflowSource` generalizes into the one ordered, origin-tagged search path for workflows, prompts and assets. That covers: writable root first, shadowing, write-target selection, the read-only refusal, and serialization into the worker's environment with origins. It replaces the asset order written four times, the prompt order written twice, and the three origin mechanisms (survey D1-D11). The workflows, prompts and assets listings take one shape, and the UI, MCP and `dw.run` change in the same stage. Version bump. | Each library's search order is computed in one place, and the API and the worker read the same serialized path; the three listings share one field set |
| 3d | **One media I/O module and one DSP module.** One `av.open` decode/probe layer (the audio-to-float decode is written 5 times, the header-rate probe 7). A pure numpy/scipy `dsp` module out of `audio_utils.py`, with the task commands left as thin wrappers. `concat_videos` and `dissolve_videos` share one join. The long task functions are cut. Build-vs-buy deletions: the biquad's pure-Python fallback and its test, five `dbfs` copies, the second true-peak oversampler, `file_fps`. The teacache decision (below) lands here. | `av.open` appears only in the media module; `audio_utils.py` and the task modules under 1,000 lines and their functions under 150 |
| 3e | **Engine splits.** `result.py` keeps `Result` (writers, audio QC and output extraction move out). `pipeline.py` keeps `Pipeline` (placement, components, adapters, progress reporting move out). `Workflow.run` and `create_step_action` are cut into named phases, and the pipeline-ownership dicts become one object. `Workflow`'s validation block joins validation. `arguments.py` loses its media half. `introspection.py` loses type-reference checks and inert-argument warnings. `security.py` splits, with CodeQL re-modelled. The carried follow-ups that live in these files. | The Phase 3 gate criteria hold everywhere |

Gate 3 follows 3e, with the same checks as gate 2:
- the full metrics report, with a Gate 3 column and LCOM4 for `create_app`'s successor;
- a real-model timing on lem: the B2 cached rerun, one `for_each` run, and one run through each library (a `prompt:` and an `asset:` reference) after 3c;
- deploy, tag and re-baseline.

## Carried from gate 2

From ROADMAP.md, Gate 2, the "follow-ups" lists. Each item goes to the stage whose files it touches:
- **3a:** the four import styles for `references`, normalized in the files 3a touches (the rest in 3e).
- **3b:**
  - B10's rest: `submit_job` maps every exception to 400.
  - `/api/validate` logs a gate failure twice (`admit()` and `app.py`).
  - `dw/worker.py` is 12 lines under the limit until its message types move out.
  - `JobManager.definition()` re-reads a path job's file; a live job answers from its snapshot.
  - The snapshot parity test is close to tautological.
  - `workflow_from_snapshot` passes `file_spec` through unnormalized when `workflow_dir` is None.
- **3d:**
  - The frame-size prefix sentence is still written by both callers.
  - The slice-region arithmetic is in both `slice_audio` and `slice_preflight`.
  - The dissolve run raises only the first shortfall. This is a rule-parity item, not a new check; if fixing it changes a message, the change goes in the release notes.
  - Temp-dir leaks in the dissolve and shot-span test helpers.
- **3e:**
  - `dw/workflow.py:747` could use `author_index`.
  - Prefix tests are spelled three ways.
  - Prefixes inside f-strings, which the metric does not count.
  - The three scanners of `previous_result:` shapes.
  - `for_each._copy_leaf` and the step-cache snapshot still copy media leaves.
  - A child handed media through a substituted variable stores it in its `argument_template`, which validation copies again.
  - The elision carry assumes dict pipelines.
  - `result.py`'s import of `tasks.select.Selected`: outside the cycles, and the last upward import.

## Release notes collected (for gate 3)

(Each stage adds its user-visible changes here at merge.)

- **3a (merged 2026-09-30):** no user-visible change.
  - Internal: import cycles 5 → 0 and modules inside cycles 19 → 0, both ratcheted at 0. `modules` went 131 → 132 (`dw/media_types.py`).
  - A task step missing a required argument is now refused by `Step.run`, not `Task.run`. The message is unchanged. It now comes before the step's task phase event, which a refused task never needed.
  - Merge notes:
    - `dw/server/app.py` lost its two host-set constants to `netinfo.py` without being in the 3a hot zone. No harness edit conflicted.
    - Task 3's "update `dw/server/CLAUDE.md:22`" step was moot: that line never explained the lazy import.
- **3b (merged 2026-09-30):** the HTTP routes, the MCP tools and their descriptions are unchanged. The surface snapshot is byte-identical from before 3b to its merge.
  - `POST /api/jobs` and `POST /api/jobs/{id}/rerun` answer **500** `"internal error - the server log has the detail"` when something fails after the request was admitted. They used to answer 400 for any exception (B10). A refused request is still 400.
  - `GET /api/jobs/{id}/workflow` (MCP `get_job_workflow`) and the job export answer with the definition the job was admitted with, when its workflow file has since moved, grown past the size limit or stopped parsing. They used to answer 404 / nothing. While the file can be read, they answer with the file, as before, which is also what a `new_seed` rerun draws its seed variable from.
  - `/api/validate` logs a gate failure once, not twice.
  - A worker command missing `arguments` or `output_dir` fails with "Workflow execution error" (was "Command processing error"). Admission always supplies both, so only a hand-built command sees it.
  - For developers:
    - `create_app` is a factory over `dw/server/routes/*`;
    - `app.dependency_overrides` does not reach the routers' routes;
    - the worker's messages live in `dw/worker_protocol.py`;
    - the MCP tools live in `dw_mcp/tools_*.py`.
  - Merge notes:
    - Outside the hot zone, 3b touched `dw/workflow.py` (`workflow_from_snapshot` makes `file_spec` absolute when `workflow_dir` is None), `docs/MCP.md` and `.github/copilot-instructions.md` (pointers), and the tests the plan directed.
    - `modules` went 132 → 151 (the 19 named in Decisions (3b)): the phase now projects to about 163.

## Global Constraints (all stages)

- Hard freeze: no net-new features or functionality. Surface may change only where the consolidation requires it (3c), and every change is listed for the gate's release notes.
- `scripts/arch_metrics.py --check docs/stabilization/baseline.json` passes at the end of every task, except for the `modules` rise a task's plan names (Decisions).
- Every new test fails before its fix. No count-pinning tests. Add no string `patch("dw...")` targets. A task that moves a patched name retargets the patch in the same commit (Decisions, no shims).
- Behavior-preserving tasks prove preservation with the existing suite: it passes unchanged, apart from the import paths and patch targets the move changes and the tests the task names.
- Tests: `venv/bin/python -m pytest -q -x -p no:cacheprovider` from the worktree root, plus `ruff check` and `ruff format --check` on `dw dw_mcp tests`. The worktree's `venv` is shared with the main checkout, so never run `pip install -e .` from the worktree.
- Never use `git stash`, in any form, including `git stash list`.
- A task that moves code without changing behavior is proved by characterization tests: tests that pass before and after the move. "Every new test fails before its fix" applies to fixes.
- Filesystem access keeps going through a `dw/security.py` validator. A validator that moves is re-modelled in `.github/codeql/` in the same commit (CLAUDE.md, Security Rules).

## Decisions (rulings, 2026-09-30)

- **The `modules` ratchet rises in Phase 3, by exactly the modules each stage names.**
  - Why: the phase's gate is about file size, and splitting a 4,521-line file is adding files. The ratchet exists to stop one-module-per-ticket drift, which a planned split is not.
  - How: each stage's plan lists its new modules. The stage's merge re-baselines `modules` to the count it names, with the list in the commit. A module no stage names is still a regression.
  - Expected: 3a +1 (`dw/media_types.py`); 3b about +14 (routers and services, the jobs split, the MCP tool modules, the worker protocol); 3c 0 (`LibraryPath` generalizes `workflow_sources.py`); 3d about +2 (the media and DSP modules, net of folding `media_audio` / `media_info` in); 3e about +10. That is roughly 131 → 158.
  - Cost if wrong: the ratchet is looser than it looks until Phase 4 re-sets it. **Accepted by Don, 2026-09-30.**
- **No compatibility shims.** A moved name is imported from its new home by every caller, tests included.
  - Why: a re-export is a second path to the same name, which is the sprawl this phase removes. It is also dangerous: a `patch("dw.old.name")` against a re-export patches a name nothing looks up, so the test passes without testing anything.
  - Exception: a name the harness or a user entry point reaches by its old path (`dw.server.app.create_app`, patched by `dw.serve` tests at call time) stays where the lookup happens.
  - Cost if wrong: more churn in test imports per task (mechanical; `sed` plus the suite).
- **Breaking HTTP/MCP changes are confined to 3c,** where the three library listings take one shape. The gate 3 release bumps the minor version (0.6 → 0.7) through `scripts/release.sh`, which also moves `plugin.json`. 3b keeps every route path, method, status and body. **0.7 confirmed by Don, 2026-09-30.**
- **Teacache: deleted in 3d (Don, 2026-09-30).**
  - The case for:
    - `dw/teacache.py` (381 lines, the 196-line forward factory) is Flux-only.
    - No shipped workflow uses the `teacache` key.
    - diffusers ships `first_block`, `mag` and `taylorseer` cache hooks, which dw already routes, and `docs/ACCELERATION.md` already steers users to them.
  - What deleting it costs a user: the `rel_l1_thresh` knob and its Flux speed table. It is a breaking change to a documented workflow key, noted in the release notes.
  - 3d deletes `dw/teacache.py`, `dw/teacache_models.json`, `tests/test_teacache.py`, the `pipeline.py` branch, the schema's `teacache` property, and the doc sections. A workflow that still sets `teacache` fails schema validation with the schema's own message, which the release note names.
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
  - This is safe because `Workflow.create_step_action` (`workflow.py:2228`) is the only place in `dw/` that constructs a `Task`, and `Step.run` is what runs it.
  - A `Task.run` called directly (tests only) no longer gets the friendly message. Python's `TypeError` surfaces instead.
- **`validate_workspace_name(name, *, reserved)`:** the reserved names become a required keyword-only parameter, which `workspace.py` passes.
  - It has no default: a security validator must not quietly do less when a caller forgets an argument. An omitted `reserved` is a `TypeError`.
  - Its only callers are `workspace.py:469` and `488`.
  - `security.py` no longer imports `workspace`. The CodeQL models name the function and its return value, so they are unchanged.
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
4. **Reserved workspace names are still refused** on `POST /api/workspaces` and on a `?workspace=` lookup, with the same message naming the reserved folders. The existing tests cover it. The reviewer confirms that every call site passes `reserved`, with `git grep -n "validate_workspace_name(" dw dw_mcp`: exactly the two in `workspace.py`.
5. **`--mcp` host safety is unchanged.**
   - `client_base_url()` picks loopback for a wildcard or loopback bind and the bind host otherwise.
   - `dw.serve --mcp --host 0.0.0.0` with no token still exits 2.
   - `serve.py` no longer imports `dw.server.app` before `create_app`.

### Before Task 1: the hot zone goes live

Once Don approves this plan, and before any 3a code changes:
- commit the 3a list below into `docs/stabilization/hot-zone.txt` on `develop`, with the plan docs;
- push;
- start the `stabilization/phase-3a` branch from that commit.

Phase 2 worked the same way: the harness must not edit `result.py` or `video_utils.py` in the middle of the stage.

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
- `media_frames` takes the five helpers under public names (`compose_grid`, `default_columns`, `evenly_spaced_indices`, `format_timestamp`, `grid_tile`), because two modules now use them: `media_frames` itself and `video_utils.frame_grid`.

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
- [ ] **Step 5: Verify, and re-baseline `modules`.**
  - Run the suite and ruff.
  - `arch_metrics --check` reports `modules` 132, a rise by the one module this plan names. Re-baseline it in this commit, as 2a did with `references.py`:
    - run `--write`;
    - confirm the diff is `modules` 131 → 132, plus anything that went down;
    - name `dw/media_types.py` in the commit message.
  - Every later task's `--check` is then green.
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
  - `tests/test_task_signature_errors.py`, class `TestTheRunTimeBackstop` (lines 185-203): both of its tests call `Task.run` / `_check_required_arguments` directly and are rewritten against `Step.run`. `tests/test_task.py`'s direct `Task.run` calls test unknown commands, not missing arguments, and stay;
  - a new `Step.run` test (Review Focus 3);
  - the existing workspace, vram-inheritance and serve/mcp tests.

**Interfaces:**
- Produces: `validate_workspace_name(name: str, *, reserved) -> str`; `netinfo.LOOPBACK_HOSTS` and `netinfo.WILDCARD_HOSTS` (same values).
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
  - Rewrite both tests in `TestTheRunTimeBackstop` against `Step.run`: the friendly message for `resample_audio` without `audio`, and no check for a non-dict `inputs` template.
- [ ] **Step 3: `validate_workspace_name(name, *, reserved)`.**
  - Delete the lazy import, and refuse `name in reserved` with the same message, joined from `reserved`. No default.
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
    - `import_cycles` 5 → 0;
    - `modules_in_import_cycles` 19 → 0;
    - anything that went down.
  - Any other rise is a finding, not a re-baseline.
- [ ] **Step 2: Docs.**
  - CLAUDE.md: find any sentence naming a moved symbol's old home (`git grep -n "result.py.*AudioVideo\|AudioVideo.*result.py\|_fit_audio_to_frames\|_warn_on_rate_override" CLAUDE.md dw/**/CLAUDE.md docs/`) and fix it.
  - Release notes: no user-visible change. Record "internal: import cycles removed" under this file's release notes.
- [ ] **Step 3: Hot zone.** Put `docs/stabilization/hot-zone.txt` back to the two standing entries. Stage 3b's list goes live when 3b is detailed and approved.
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
dw/previous_results.py
```

`previous_results.py` is listed because deleting the `PREVIOUS_RESULT_PREFIX` alias edits 12 lines there. Other files 3a touches only for an import line or two: `workflow.py`, `validation.py`, `video_extensions.py`, `app.py`. They are not listed, so the harness keeps them. A conflicting harness edit to one of them is a small merge conflict.

---

## Stage 3b: server into routers and services, surface unchanged

Work on branch `stabilization/phase-3b` in the worktree, from `develop` at `6ca92aa4` or later.

**What exists (survey [phase-3-surveys/server.md](phase-3-surveys/server.md), re-checked 2026-09-30 at `6ca92aa4`; line numbers there are 7 higher than now in `app.py`).**

- **`create_app`** (`dw/server/app.py`, 4,514 lines) holds:
  - 66 routes, 4 `@app.middleware("http")` functions, a manual `/mcp` route pair and the SPA mount;
  - 9 pydantic request models;
  - about 40 closure helpers.
- **Handlers close over little real state:**
  - `manager`, `downloads` and `updater`;
  - `token`, `host`, `port`, `wildcard_bind` and `allowed_hosts`;
  - `examples_dirs`, `ceiling_indexes` and the `mcp_*` trio.
  - Everything else is already on `app.state`, and tests read 15 of those `app.state` names.
- **Helpers shared across groups:**
  - `selected_workspace`, a `Depends` used by about 35 routes;
  - `_sources_for`;
  - `_admit`, which reaches the prompt roots and asset roots;
  - the output/asset resolution cluster, `_strip_output_prefix` … `_absolute_served_url`, used by gallery, media, assets, files and export.
- **Module-level code in `app.py` (about 740 lines):**
  - the catalog listing: `workflow_details`, `attach_observed`, `collect_prompt_references`, `catalog_name_for`, `resolve_*_workflow`, `prompt_details`, `_prune_missing` and the two detail caches;
  - the security helpers: `query_token_ok`, `_matched_route`, `ACTIVE_DOCUMENT_TYPES`;
  - the request models `AcknowledgedCost` / `JobRequest`.
- **Registration order matters:**
  - A greedy `{name:path}` GET must come after its `/download`, `/variables` and `/metadata`-style siblings.
  - The `/mcp` routes go before the SPA mount, and the SPA mount goes last.
  - Middleware order is the stack order.
  - `_matched_route` walks `request.app.router.routes` for `endpoint.query_token_ok`, which `include_router` preserves.
- **Tests reach `app.py` by module object, not by string:**
  - `app_module.build_plan` (5, in `tests/test_server.py`);
  - `app_module.video_shape` (2);
  - `"dw.server.app.local_addresses"` (1);
  - `app_module._prune_missing` (1);
  - `app_module.create_app` (3 files, patched as what `dw.serve.main` imports lazily).
  - 17 test files import `create_app`, and 3 import catalog helpers.
- **`jobs.py`** (1,567 lines) is `JobHistory` (97-513), `Job` (516-744) and `JobManager` (747-1567). `JobManager.definition()` (898) re-reads a path job's file. `submit_job` and `rerun_job` map every exception to 400 (`app.py:1221`, `1363`; B10's rest).
- **`dw/worker.py`** is 988 lines: the 2c message dataclasses plus the worker loop. `_handle_execute` (436) is 164 lines.
- **`dw/serve.py` `main`** is 258 lines. It runs in this order:
  - argparse;
  - workspace and directory derivation, and the env pins;
  - the `--mcp` token check;
  - trust;
  - the prompt dir;
  - the example libraries;
  - the uvicorn check;
  - `create_app`, which it imports lazily and on purpose: tests patch `dw.server.app.create_app`;
  - `uvicorn.run`.
- **`dw_mcp/server.py` `build_server`** is 1,279 lines.
  - It has 59 nested tool functions, registered by an inner `tool(fn, annotations)` helper, in 8 groups.
  - 657 of those lines are docstrings, and the docstrings are the agent-facing descriptions. `tests/test_mcp_server.py::test_the_tool_surface_fits_the_budget` counts their tokens.
  - `get_output_frames`, `get_output_image` and `get_output_audio` build MCP content in `server.py`.
  - The `mcp` SDK is an optional extra (`pyproject.toml`, `mcp = [...]`). `dw.run` imports `dw_mcp.client` without it, so only the modules that register tools may import the SDK.
  - `import dw_mcp.server` must stay torch-free (`tests/test_mcp_server.py`).

### Decisions (3b)

- **Proof of an unchanged surface is a snapshot diff, not a new test.**
  - Task 1 commits `scripts/surface_snapshot.py`. It writes a JSON file with:
    - the *set* of routes, sorted by (path, method), each with its name and endpoint `query_token_ok` flag. Cross-resource order is not preserved, and doesn't need to be: routes are grouped by resource, and a method mismatch makes Starlette keep looking (`Match.PARTIAL`). `_matched_route` accepts only `Match.FULL`, plus HEAD on a GET route, so it is method-aware too;
    - for each greedy family (routes of one method where one path is a `{name:path}` prefix of another: workflows, prompts, gallery, assets), the order of its members in `app.router.routes`, which must not change;
    - the tail entries in order: the `/mcp` route pair, then the SPA mount last. The snapshot is built with `mcp=True` and a real `ui_dir`, so the factory rewrite cannot drop or misorder them;
    - the middleware stack in order;
    - the full OpenAPI document from `create_app(...)` over a temp workspace;
    - the MCP tool list in registration order (name, description, input schema, annotations), from `build_server` over a stub client.
  - Every later 3b task snapshots at its base and at its head, and the diff must be empty. The only exception is an OpenAPI `operationId` or title that derives from a module path, which the task names.
  - 3c reuses the script: its diff *is* 3c's list of breaking changes.
  - The script lives in `scripts/`, which the module ratchet does not count.
- **Module layout (3b adds 19 modules; the Phase 3 estimate said about 14).** `modules` goes 132 → 151, which puts the phase at about 163 rather than 158. Don is told when 3b starts, before its hot zone goes live, together with the zone's scope: all of `dw/server/` and `dw_mcp/`.
- **`ROUTERS` order:** `jobs`, `system`, `library`, `media`, `gallery`, `assets`. The factory includes them, then adds the `/mcp` route pair, then includes the `files` router (`/outputs`, `/inputs`, `/exports`), then mounts the UI last. That is today's tail order, which the snapshot records from `/mcp` to the SPA mount. (Ruling after Task 1.)
  - `media` goes before `gallery`, so the `…/thumbnail` and `…/download` GETs keep sitting before `DELETE /api/gallery/{name:path}`, as they do today.
  - The snapshot's family orders prove the rest.

  | New module | Holds |
  | --- | --- |
  | `dw/server/routes/__init__.py` | `ROUTERS`: the routers in registration order |
  | `routes/jobs.py` | `/api/jobs*` and `/api/validate` (the validation plan helpers with it) |
  | `routes/library.py` | `/api/workflows*`, `/api/prompts*`, `/api/prompt-schema`, `/api/enhancers`, `/api/enhance`, `/api/workspaces*` (3c reworks exactly these) |
  | `routes/gallery.py` | `GET /api/gallery`, `POST /api/gallery/archive`, `DELETE /api/gallery/{name}` and their helpers |
  | `routes/media.py` | `/api/gallery/{name}/metadata\|assess\|audio\|frames\|thumbnail\|download` (`gallery_frames` cut under 150) |
  | `routes/assets.py` | `/api/uploads`, `/api/assets*` |
  | `routes/system.py` | pipelines, tasks, classes, schema, guides, models/downloads, diffusers, memory, health, server |
  | `routes/files.py` | `/outputs/{name}`, `/inputs/{name}`, `/exports/{job}.zip` |
  | `dw/server/deps.py` | FastAPI dependencies and per-request lookups: `selected_workspace`, `workspace_for`, `sources_for`, `ceiling_index` |
  | `dw/server/outputs.py` | output/asset resolution: strip prefix, output file, asset file, asset roots, served URLs, zip download |
  | `dw/server/catalog.py` | the module-level catalog listing now in `app.py`, and its two caches |
  | `dw/server/http_security.py` | the four middlewares as one `install_middleware(app, ...)`, plus `query_token_ok`, `_matched_route`, `ACTIVE_DOCUMENT_TYPES` |
  | `dw/server/job_history.py` | `JobHistory` |
  | `dw/server/job_record.py` | `Job` and the state/ack constants it uses |
  | `dw/worker_protocol.py` | the message dataclasses, `to_wire`/`from_wire`, `parse_reply` (the 2c ruling's own follow-up) |
  | `dw_mcp/tools_catalog.py`, `tools_media.py`, `tools_authoring.py`, `tools_jobs.py` | the 59 tools, grouped as the survey lists them |

  - Admission's request models (`AcknowledgedCost`, `JobRequest`) and `_admit` / the acknowledgement helpers join the existing `dw/server/admission.py`.
  - Each other request model moves to module level in its router, with the same class name, so the OpenAPI schema names are unchanged.
  - The overrun is the price of "no module over 1,000 lines" with one router per resource. Folding `gallery` and `media` together would sit near 1,000 lines.
- **Handlers read state from `request.app.state`, not a closure.**
  - `create_app` becomes a factory under 150 lines. It:
    - builds `JobManager` and the MCP app;
    - stores the closure-only values on `app.state` under new names (`downloads`, `updater`, `api_token`, `bind_host`, `bind_port`, `wildcard_bind`, `allowed_hosts`, `examples_dirs`, `ceiling_indexes`);
    - installs the middleware;
    - includes `ROUTERS` in order;
    - adds the `/mcp` routes and mounts the UI.
  - The 15 `app.state` names tests read today keep their names. The dead `app.state.workflow_sources` (survey D10) is deleted.
- **No shims (Phase 3 Decisions):**
  - The patches retarget to where the name is now looked up: `build_plan` → `routes.jobs`, `video_shape` → `routes.media`, `local_addresses` → `routes.system`, and `_prune_missing` → `catalog`.
  - The test imports of `collect_prompt_references`, `attach_observed`, `workflow_details`, `JobHistory`, `Job` and the job constants move to the new homes.
  - `create_app` stays in `dw/server/app.py`, so `app_module.create_app` and `dw.serve`'s lazy import keep working.
- **`dw_mcp` tools become methods, one class per group, holding `client`.**
  - The tools become `class CatalogTools: def __init__(self, client)` and friends, one tool per method, each with its docstring byte-identical.
  - `build_server` registers bound methods in today's order, through the same `_anticipated` wrapper and annotation constants.
  - Why methods and not nested functions: a nested function counts toward its enclosing function's length, so any registration function holding 19 docstring-heavy tools is over 150 lines again.
  - `run_workflow` and `wait_for_job` interpolate `MAX_WAIT_SECONDS` into their docstrings at registration. A bound method's `__doc__` is read-only, so format it on the class attribute, at class-body time or before binding. Never wrap the method in a new function: that changes the signature the SDK reads.
  - The MCP content builders for image, audio and frames move into `tools_media.py`.
  - The handler modules (`catalog.py`, `media.py`, …) stay SDK-free.
  - Verification: the snapshot's tool list diff is empty, and the token budget test passes unchanged.
- **B10's rest (the one behavior change in 3b; release note). The split is by phase, not by exception type.**
  - In `submit_job` and `rerun_job`, anything raised while resolving and admitting the request is a refusal: 400, as today. That covers resolving the workflow reference, `_admit(...)` and the acknowledgement check. 2b already turned a crashing check into a finding inside admission, so a crash there is not an admission error.
  - Anything raised after admission succeeded (`manager.submit`, `catalog_name_for`, `describe`) is logged with its traceback and answered 500 with `"internal error - the server log has the detail"`.
  - Two `try` blocks, no list of types. The review question for any call is whether it comes before or after admission.
  - The other `except Exception` → 400 sites in `app.py` (about 15) are out of 3b's scope on purpose. B10 names job submission; the library routes follow the same rule in 3c, and the rest in 3e.
- **The carried 2c items:**
  - `JobManager.definition()` answers a live job from `job.spec["definition"]` (the admitted snapshot). Only a restored job, which has no snapshot, reads the file.
  - Its answer must equal today's. While the file exists, `definition()` must equal `json.load` of the file, because `rerun` and `get_job_workflow` consume it. If admission normalizes anything into the snapshot, the implementer reports what, and `definition()` answers the file's form.
  - `workflow_from_snapshot` normalizes `file_spec` (`os.path.abspath`) when `workflow_dir` is None, the way it does when it is set.
  - The snapshot parity test compares the snapshot-built `Workflow` against a *second, independent* `workflow_from_file` read of the same path, instead of against the instance it was built from.
- **`/api/validate` logs a gate failure once.** `admit()` logs it, and the route stops logging it again.
- **`serve.main` is cut into** `build_parser()`, `configure_environment(args) -> ServeConfig` (a small dataclass of the derived dirs, token and layout), `check_bind_safety(args, token)` and `run(config)`. The order of the env pins, and the lazy `create_app` import, are unchanged.

### Review Focus (3b)

1. **Surface parity.** The route table, middleware order, OpenAPI document and MCP tool list are identical, by the snapshot diff attached to each task's report. A reviewer re-runs the script once at head for the task that moved the most routes.
2. **A patch that tests nothing.** No `monkeypatch.setattr(app_module, ...)` remains for a name that moved out of `app.py`, and no import of a moved name from its old module (`git grep` in each task's report).
3. **Two apps, one process.** Tests build many apps. Anything that was per-closure must now be per-`app.state`, never module-global:
   - `ceiling_indexes`, `downloads` and `updater` especially;
   - the two detail caches were already module-global and stay so.
   - A committed test builds two apps with different `examples_dirs` in one process and gets each app's own workflow listing. It passes today, because closure state is already per app, so it is a characterization test (Global Constraints).
4. **Query-token routes.** The five `@query_token_ok` GETs still accept `?token=`; every other `/api/` route still refuses it. Existing tests cover this and run against the router-built app:
   - `tests/test_security_auth.py:172-187` (download accepted, DELETE refused);
   - `tests/test_server.py:4637` (thumbnail);
   - `tests/test_server.py:1420` (events).
5. **B10.**
   - A request that makes admission raise `SecurityError` answers 400.
   - A request that makes `JobManager.submit` raise `RuntimeError` answers 500, and the log carries the traceback.
   - These are two committed tests, both failing before the fix. The 500 test is the one that fails.

### Task 1: The surface snapshot

**Files:** Create `scripts/surface_snapshot.py`.
- It builds `create_app(..., mcp=True, ui_dir=<a temp dir holding an index.html>)` over a temporary workspace, and `build_server(<stub client>)`. No worker is started: pass `job_manager=JobManager(..., worker_manager=<a stub>)`, the way `tests/test_server.py:243-246` injects `ScriptedWorkerManager`; the script cannot import tests, so it defines its own minimal stub.
- It writes the JSON described in Decisions, with sorted keys.
- `venv/bin/python scripts/surface_snapshot.py OUT.json` writes the file.
- Commit: `chore(scripts): surface_snapshot - route table, middleware, OpenAPI and MCP tools as one JSON`.
- Verify: two runs at the same commit are byte-identical.

### Task 2: `app.py`'s module-level code to its owners

- The catalog listing and its caches go to `dw/server/catalog.py`.
- `query_token_ok`, `_matched_route` and `ACTIVE_DOCUMENT_TYPES`, plus the four middlewares as `install_middleware(app)`, go to `dw/server/http_security.py`. The middlewares read the bind/token values from `app.state`.
- `AcknowledgedCost`, `JobRequest`, `_acknowledgement_form`, `_bound_plan_for`, `_check_bound_acknowledgement` and `_admit` go to `dw/server/admission.py`. They take the workspace and `request.app.state` explicitly.
- The closure-only values go onto `app.state` (Decisions).
- Retarget the `_prune_missing` patch and the catalog imports in tests.
- Snapshot diff empty. Commit.

### Task 3: `deps.py`, `outputs.py`, and the routers

This task moves the 66 routes into `dw/server/routes/*` in their original order, in two commits:
- the routers that need no output resolution: `system`, `library`, `jobs`;
- then `outputs.py` and the `gallery`, `media`, `assets` and `files` routers, with `gallery_frames` cut under 150 lines (its frame-selection, tile-encoding and response-building sections).

`create_app` becomes the factory.
- Retarget `build_plan`, `video_shape` and `local_addresses`.
- Add the two-apps test (Review Focus 3).
- Snapshot diff empty after each commit.
- `app.py` under 400 lines; no function in `dw/server` over 150.

### Task 4: `jobs.py` split, the snapshot follow-ups, and B10

- `JobHistory` goes to `job_history.py`. `Job` and the state/ack constants it uses go to `job_record.py`.
- `jobs.py` keeps `JobManager` and imports the rest.
- `definition()` answers a live job from its snapshot. Write a failing test first: delete the file after submit, then `definition()`, and the snapshot comes back.
- `workflow_from_snapshot` normalization, and the independent-read parity test.
- The B10 mapping, with its two tests (Review Focus 5).
- `/api/validate` logs once. Write a test with `caplog`: one record for one gate failure. It fails before the fix.
- Snapshot diff empty. B10 changes a response, not the surface, so the snapshot does not show it.
- Also test that `definition()` equals the file while the file exists (Decisions).

### Task 5: `dw/worker_protocol.py` and `_handle_execute`

- Move the message dataclasses, `to_wire`/`from_wire` and `parse_reply` from `worker.py` to `dw/worker_protocol.py`.
- `worker.py`, `worker_manager.py` and `jobs.py` import from there.
- Cut `_handle_execute` into named phases under 150 lines each: activate, build from snapshot, run, reply. Replies and their order are unchanged, and the existing worker tests prove it.
- `worker.py` must be under 900 lines afterwards.

### Task 6: `serve.main`

Cut `serve.main` as Decisions says. `tests/test_serve_main.py` and the two security tests that patch `create_app` pass unchanged.

### Task 7: `dw_mcp` tools out of `build_server`

- Move the 59 tools into the four `tools_*.py` modules as methods, as Decisions says.
- `build_server` under 150 lines, and `dw_mcp/server.py` under 400.
- Snapshot tool-list diff empty, the budget test unchanged, and the torch-free import test passes.

### Task 8: Stage 3b merge

- **Re-baseline.** `modules` 132 → 151, with the 19 modules named in the commit. Also re-baseline whatever went down. Any other rise is a finding.
- **Docs:**
  - `dw/server/CLAUDE.md` names the routers and services;
  - `dw_mcp/CLAUDE.md` names the tool modules and the byte-identical docstring rule;
  - CLAUDE.md's Server paragraph.
  - The claude_md_lines ratchet holds: replace text, don't add it.
- **Release notes:** B10's 500. The snapshot diff shows nothing else.
- **Hot zone** back to the standing entries. Merge `--no-ff` to develop, push, no deploy.
- **Next stage:** detail 3c, with the snapshot script as its break list, and cross-check with Fable.

### Hot zone (3b)

```
dw/server/
dw/serve.py
dw/worker.py
dw/worker_manager.py
dw/worker_protocol.py
dw_mcp/
scripts/surface_snapshot.py
```

`dw/workflow.py` gets one line (`workflow_from_snapshot`) and is not listed.
