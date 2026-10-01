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
  - From 3d: `arguments._with_frame_rate` has a dead `http(s)` branch; its one caller passes a validated local path.

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
- **3c (merged 2026-09-30): BREAKING, ships as 0.7.** The three library listings (`GET /api/workflows`, `/api/prompts`, `/api/assets`) now share one envelope. The snapshot diff against `cf1802cb` is the break list.
  - Removed fields: `workflow_dir`, `prompt_dir`, `asset_dir` (the writable root is the `libraries` entry with `writable: true` and `origin: "workspace"`), `sources` (workflows), `prompt_dirs`, `asset_dirs`, and `origins` (prompts; origin is now in `details[name]`).
  - Renamed: assets `libraries[].dir` is `root`. The MCP compact workflow listing's top-level `sources` is `libraries`.
  - Added to all three: `libraries: [{origin, root, writable}]`, the search path in order, and `shadowed: [{name, origin, shadowed_by}]`. A listing filter narrows `shadowed` with the entries. Every entry carries `origin` and `writable`: workflows and prompts in `details[name]` (prompts gained `writable`), assets in each entry. `workspace` is echoed by workflows and assets (prompts are shared and echo none). Workflows and prompts gained `shadowed`; assets' entries changed shape.
  - Unchanged: each listing's item key (`workflows`, `prompts`, `assets`), `details`, `folders`, sort order, and single-item reads with their `X-*-Origin` / `X-*-Writable` headers.
  - Deleting a read-only entry is one 403 for all three libraries: `'<name>' is in the read-only <origin> library (<root>); only the workspace's own <kind> can be deleted`. It was three messages.
  - D11: upload, keep and delete with no asset library all answer 409 `This workspace has no asset library`. Upload used to write into `outputs`. Delete's text was `This server has no asset library`.
  - B10 in the library routes: a workflow save whose validation gate itself crashes answers 500 `internal error - the server log has the detail` (was 400). An invalid workflow is still 400.
  - D5: a named workspace's `prompt:` resolves against the server's prompt library (`--prompt-dir`) during plan building, the workspace list and workspace creation, matching listing, saving and running. Creating a workspace no longer makes an empty `<root>/prompts`.
  - D2: `libraries` no longer lists an example prompt dir that does not exist. A missing examples root is dropped for every kind.
  - D7: an exact asset lookup no longer searches the example folders.
  - D8: a sub-workflow run from an examples folder is labeled and confined as the examples root, not the workspace's.
  - Accepted behaviour changes:
    - `prompt:name.json` now resolves.
    - An API prompt symlink escaping its root is skipped and the search continues (was 404). The engine still refuses it at run time.
    - Asset listing sort ties are broken by name.
  - Admission (`POST /api/validate` and the pre-queue check) now refuses an `asset:` whose workspace copy is a symlink pointing out of the library. Before, it passed when a later root held the name, and the run then failed. This is the same pattern as the prompt line above, and it now agrees with the worker.
  - A prompt save (`PUT /api/prompts`) on a server with no prompt library answers 409 `This server has no prompt library` (was a bare 500). Only a `create_app` caller can reach it; `dw.serve` always sets one.
  - Internal: `dw/workflow_sources.py` is `dw/library.py` (`WorkflowSource` is `LibraryRoot`, plus `LibraryPath`). `modules` is unchanged.

- **3d (merged 2026-09-30): one breaking change, ships in 0.7.**
  - **Breaking:** the `teacache` pipeline `configuration` key is removed. A workflow that still sets it fails validation: `Additional properties are not allowed ('teacache' was unexpected)`. Use `cache` instead (`first_block`, `mag`, `taylorseer`; docs/ACCELERATION.md).
  - The served acceleration guide (`list_guides` / `get_guide`) no longer has its "TeaCache" and "Cache vs TeaCache" sections, so `get_guide(..., section="TeaCache")` no longer resolves. WORKFLOW_GUIDE's caching paragraph describes one `cache` block.
  - `templates/step-caching.json`'s description (as `list_workflows` returns it) and the schema's `cache` description no longer mention teacache.
  - A `dissolve_videos` run with several inputs too short for their dissolves names all of them in one error, joined by `; `. With one short input the message is unchanged. Validation already listed them all.
  - Unchanged on purpose: every task command's arguments and description (the surface snapshot of every `describe_task` is byte-identical), every warning's text, and every pinned DSP number.
  - For developers (Python paths only; nothing on the API or MCP reaches them):
    - `dw.loudness`, `dw.media_audio`, `dw.media_info` and `dw.teacache` (with `teacache_models.json`) are gone. Their contents are in `dw.dsp` (pure numpy/scipy/pyloudnorm) and `dw.media` (the one module that calls `av.open`, enforced by `tests/test_media_layering.py`).
    - `video_utils.file_fps` is gone. `audio_utils.resample_waveform` is `dw.dsp.resample_waveform`.
    - `normalize_audio`, `compress_audio`, `filter_audio` and `analyze_audio` are in `dw.tasks.audio_dynamics`. The join helpers (`video_names`, `fit_audio_to_frames`, `bleed_join`, `match_levels`, and the steps concat and dissolve share) are in `dw.tasks.joins`.
    - Renamed: `_as_track` is `as_track`, `_waveform_and_rate` is `waveform_and_rate`, and `_as_number` is `coerce_number`.
  - Merge notes:
    - `modules` stays 151, two under the +2 projection. `modules_over_1000_lines` went 7 → 6, `functions_over_150_lines` 11 → 4, and `complex_functions` 17 → 12.
    - `_load_tracks_matching_rate` stayed in `audio_utils.py`, not `joins.py` as Decisions said. Its only callers are there, and `audio_utils` imports `joins`, so moving it would have made a cycle.
    - Found and left alone under the freeze, for after Phase 3: `gain_audio` still rounds a frame-based region's end the pre-#557 way, so it can come out one sample short; and `concat_videos` joins a track with no sample rate unresampled, a long-standing silent skip. `dissolve_videos` refuses that track, as before.
    - Outside the hot zone, 3d touched `plugins/dw/skills/series-episodes/SKILL.md` (its Sources line names the new homes) and `workflows/templates/step-caching.json` (its description).

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

---

## Stage 3c: one `LibraryPath`, and one library surface (breaking)

Work on branch `stabilization/phase-3c` in the worktree, from `develop` at `cf1802cb` or later.

**What exists (survey [phase-3-surveys/library.md](phase-3-surveys/library.md), re-located 2026-09-30 at `cf1802cb`: 3b moved the server half).**

There are three content libraries (workflows, prompts, assets), and each answers "which roots, in which order, who may write" its own way.

- **Engine side:**
  - `dw/workflow_sources.py`: `WorkflowSource(root, origin, writable)`, `workflow_sources()`, `find_workflow`, `listing`, `writable_source`, and `fallback_roots` (env `DW_WORKFLOW_PATH`). `resolve_sub_workflow` builds throwaway `WorkflowSource(root, EXAMPLES_ORIGIN, False)` objects as containers, mislabelling the writable root (D8).
  - `dw/assets.py` `asset_search_path` / `resolve_asset_reference` (env `DW_ASSET_PATH`).
  - `dw/prompts.py` `prompt_search_path` / `resolve_prompt_reference` (env `DW_PROMPT_PATH`). It is the same function as the asset one, written twice.
  - `dw/workspace.py` `library_fallbacks` / `set_library_fallbacks` / `example_libraries` / `discover_library`.
  - `resolve_asset_reference(ref, asset_dir=root)` still appends the env fallbacks, so "probe one root" probes several (D7).
- **Server side, after 3b:**
  - `dw/server/deps.py` `sources_for`, `prompt_roots`. `prompt_roots` has no isdir filter, which the engine's has (D2).
  - `dw/server/outputs.py` `asset_roots`, `resolution_roots`, `asset_roots_for_job`, `asset_in`, `common_assets`.
  - `dw/server/routes/assets.py` `_asset_origin`, which tags origin by directory equality.
  - `dw/server/admission.py` `over_roots`.
  - `dw/server/exports.py` `_copy_assets`.
  - `dw/locations.py` `media_roots`.
  - The asset order is written four times (D1). Origin tagging uses three mechanisms: workflows by attribute, prompts by `index == 0`, assets by directory equality (D4).
- **`dw/serve.py`** pins `DW_ASSET_DIR`, `DW_PROMPT_DIR` and the three `DW_*_PATH` tails for the worker. `create_app` recomputes `example_libraries` for the API, so the same order travels by two channels.
- **`named_workspace`** builds its prompts root as `<root>/prompts` and ignores `--prompt-dir` (D5). Plan building for a named workspace therefore resolves `prompt:` against a different library than listing, saving and running use.
- **`POST /api/uploads`** with no asset library writes into `outputs` (D11), where keep and delete answer 409.
- **The listings disagree:**

  | | `GET /api/workflows` | `GET /api/prompts` | `GET /api/assets` |
  | --- | --- | --- | --- |
  | writable dir | `workflow_dir` | `prompt_dir` | `asset_dir` |
  | roots | `sources: [{root, origin, writable}]` | `prompt_dirs: [str]` | `asset_dirs: [str]` and `libraries: [{origin, dir, writable}]` |
  | per-entry origin | `details[name].origin` | `origins{name: origin}` | in each entry |
  | per-entry writable | `details[name].writable` | absent | absent |
  | shadowed | absent | absent | `shadowed: [{…, shadowed_by}]` |
  | workspace echoed | yes | n/a (shared) | no |

  The three deletes of a read-only entry are all 403, with three different messages.
- **Consumers of those fields:**
  - `ui/src/lib/api.ts` (`workflow_dir` 255, `asset_dir`/`asset_dirs`/`libraries`/`shadowed` 482-487, `prompt_dir`/`prompt_dirs`/`origins` 581-587);
  - `ui/src/lib/types.ts` (`AssetLibrary`, `ShadowedAsset`, 302-315);
  - `ui/src/lib/pages/AssetsPage.svelte`;
  - `dw_mcp/assets.py` (`libraries[].dir`, 87-93), `dw_mcp/catalog.py` (the compact `sources`), `dw_mcp/prompts.py`;
  - tests: `test_server.py` (28 hits), `test_mcp_assets.py`, `test_library_sources.py`, `test_server_workspaces.py`, `test_catalog_shape.py`.
  - About 30 test sites set the `DW_*_DIR` / `DW_*_PATH` env vars.

### Decisions (3c)

- **`dw/workflow_sources.py` becomes `dw/library.py`** (a rename, so `modules` is unchanged). It holds:
  - `LibraryRoot(root, origin, writable)`, which was `WorkflowSource`;
  - `LibraryPath`: an ordered tuple of `LibraryRoot`s plus the library's `kind`;
  - the origin vocabulary `WORKSPACE` / `COMMON` / `EXAMPLES` / `BUILTIN`.

  The sub-workflow resolution, `builtin_root` and `catalog_root` move with it. Every import is repointed (no shims).
- **Layering.** `dw/library.py` imports nothing from `dw.server`. It may import `dw.workspace`, for the subdir constants and `example_libraries`; that is today's direction (`fallback_roots` already does it). `dw/workspace.py` never imports `dw/library.py` at module level, so there is no `Workspace.library()` convenience method.
  - The asset lister is server code (`dw/server/outputs.py` `iter_gallery_files`), so `LibraryPath.entries(lister)` takes the lister as a parameter. The server passes the media walk, and the engine passes `workflow_names` for the JSON libraries.
- **`LibraryPath` operations, the one implementation of each rule:**
  - `find(name) -> (path, LibraryRoot) | None`: front to back, each candidate confined with `validate_path(candidate, root.root)`, so a symlink that escapes is a miss;
  - `entries() -> (winners, shadowed)`;
  - `writable_root(shared=False) -> LibraryRoot | None`;
  - `require_writable(root)`, raising one `ReadOnlyLibraryError(name, root)`;
  - `roots()`.

  The per-kind differences are three small strategies, chosen by `kind`:
  - name → file: workflows and prompts add `.json`; assets take the name literally;
  - name validator: `validate_prompt_reference`, `validate_asset_reference`, or containment only;
  - lister: `workflow_names` for JSON libraries, the media walk for assets.

  Containment calls the same named validator each resolver calls today: `validate_workflow_path` for workflows, `validate_prompt_path` for prompts, and `validate_path(…, root)` with a non-None base for assets. The first two also enforce the extension, and CodeQL models all three by name.
- **One constructor per library, from one place:** `library_path(kind, workspace, examples_dirs, primary=None)`.
  - Assets: `[workspace assets (workspace, writable), common/assets (common, writable only for a shared write), example assets (examples, read-only)]`.
  - Prompts: `[the server's prompt dir (workspace, writable), example prompts (examples, read-only)]`.
  - Workflows: `[workspace workflows (workspace, writable), examples dirs (examples, read-only)]`, plus `builtin` when asked.
  - `primary` overrides the writable root. That is how a job's own `asset_dir` (the worker's `activate_asset_dir`, the export's job roots) gets its path.
  - Missing directories are dropped, the same way for every kind; that fixes D2.
- **One serializer for the worker:** `pin_library_paths(...)` in `dw.serve` and `library_path_from_env(kind, primary)` in the engine.
  - The env var names stay (`DW_ASSET_DIR`, `DW_PROMPT_DIR`, `DW_ASSET_PATH`, `DW_PROMPT_PATH`, `DW_WORKFLOW_PATH`), and so does their format (an `os.pathsep` list of read-only roots after the primary). About 30 test sites set them; the env is a test-pinned interface.
  - Origins are re-derived from the constructor (a root equal to the workspace root's `common/assets` is `common`, any other read-only root `examples`), so the worker's tags match the API's.
  - The env var names become constants in `dw/library.py` (D9).
- **Every consumer goes through `LibraryPath`:**
  - `resolve_asset_reference`, `resolve_prompt_reference`, `find_workflow`, `resolve_sub_workflow` (D8: a real `LibraryRoot` per root, correctly tagged);
  - `deps.sources_for` / `prompt_roots`, `outputs.asset_roots` / `resolution_roots` / `asset_roots_for_job` / `asset_in`, `_asset_origin`;
  - `admission.over_roots`, which becomes one `find` on the workspace's full path. Today it probes root by root through a resolver that re-appends the env tail: that is D7;
  - `exports._copy_assets`, `locations.media_roots`.

  After 3c, `git grep` finds no other loop over library roots.
- **D6 (`discover_library`'s precedence) stays.** It decides where the primary is when no server pinned one (the CLI, a test), and its answer is what feeds `primary`. Out of scope.
- **D5: a named workspace's prompts are the server's prompt library.** `named_workspace` takes the server's prompt dir (`app.state.prompt_dir`) instead of `<root>/prompts`. Prompts are shared by design (CLAUDE.md, "Workspaces on the server"), and `--prompt-dir` is the server's choice.
- **The one library surface (breaking; the release notes and 0.7 carry it).** The three listings share one envelope:
  - `libraries: [{origin, root, writable}]`, the search path in order. It replaces `sources`, `prompt_dirs`, `asset_dirs` and assets' `libraries[].dir`.
  - Every entry carries `origin` and `writable`: workflows in `details[name]` as today; prompts in `details[name]`, replacing the `origins` map; assets in each entry.
  - `shadowed: [{name, origin, shadowed_by}]` for all three.
  - `workspace`, echoed by workflows and assets. Prompts are shared and echo none.
  - Removed: `workflow_dir`, `prompt_dir`, `asset_dir`, `sources`, `prompt_dirs`, `asset_dirs`, `origins`. The writable root is the `libraries` entry with `writable: true` and `origin: "workspace"`.
  - Each listing keeps its item key (`workflows`, `prompts`, `assets`) and the rest of its body (`details`, `folders`, sort order). Single-item reads keep their raw body and `X-*-Origin` / `X-*-Writable` headers, because the editor saves what it reads.
  - The MCP compact listing renames its top-level `sources` to `libraries`.
- **Statuses, one rule each:**
  - Deleting a read-only entry: 403 from `ReadOnlyLibraryError`, one message, `"'<name>' is in the read-only <origin> library (<root>); only the workspace's own <kind> can be deleted"`.
  - No writable library: 409 everywhere. That includes `POST /api/uploads`, which stops writing into `outputs` (D11).
  - The library routes' other `except Exception → 400` sites follow B10's rule: a refusal is 400, a failure after the request was understood is 500.
- **The surface snapshot diff is 3c's break list.**
  - The routes return bare dicts with no `response_model`, so today's snapshot cannot see a body.
  - Task 1 therefore extends `scripts/surface_snapshot.py`: it GETs the three listings against a fixture workspace, holding one workspace workflow shadowing an example, one prompt plus an example prompt, and one asset each in the workspace, `common` and an example. It records the sorted key set of each body and of one entry per listing.
  - Task 3's report attaches the full diff against Task 1's snapshot. Every line in it is either in the release notes or a finding.

### Review Focus (3c)

1. **Order parity, engine and API.** For a workspace with an examples dir and a `common/assets`, the asset path the worker builds from the pinned env, and the one the API builds from `app.state`, are the same roots in the same order, with the same origins. A committed test builds both and compares them. The prompts and workflows get the same test.
2. **Shadowing.** An asset named in both the workspace and an example resolves to the workspace's copy (`asset:`, `/inputs`, the listing's winner). The example copy is listed under `shadowed`, with `shadowed_by: "workspace"`. The same holds for prompts and workflows, which gain `shadowed` in this stage.
3. **Read-only refusal.** Deleting a read-only entry answers 403 with the one message for all three libraries. Deleting a workspace entry still works. A shared (`common`) asset delete behaves as today.
4. **Confinement.** A symlink in any library that points outside its root is a miss in `find` and absent from `entries`. The existing symlink tests (`tests/test_security_symlinks.py`) pass, now reaching `LibraryPath`.
5. **The surface diff names every break.** The UI (`npm --prefix ui test`, plus `npm --prefix ui run check` if it exists) and the MCP tests pass against the new shapes. Nothing outside the listed fields changed.

### Task 1: `dw/library.py`, and the workflows on it

- Extend `scripts/surface_snapshot.py` with the listing body keys (Decisions), and snapshot the base before changing anything.
- Rename `dw/workflow_sources.py` to `dw/library.py`, with `WorkflowSource` → `LibraryRoot`.
- Add `LibraryPath` and `library_path(kind="workflows", …)`.
- Move the workflow consumers onto it: `find_workflow`, `listing`, `writable_source`, `resolve_sub_workflow` (D8), `deps.sources_for`, and `routes/library.py`'s workflow routes.
- Repoint every import.
- The workflow listing body is unchanged in this task, so the snapshot diff is empty.
- Tests: a `LibraryPath` unit test for `find` / `entries` / `writable_root` / `require_writable` over temp roots, and the D8 test (a sub-workflow resolved from the writable root is tagged `workspace`).

### Task 2: prompts and assets on `LibraryPath`, and the worker's path from one serializer

Two dispatches, each reviewed:
- **(a) Engine.**
  - The prompts and assets `library_path` constructors.
  - `assets.py`, `prompts.py` and `locations.media_roots` on them.
  - The env serializer and the env name constants (D9).
  - D2 and D7, each with a test that fails first.
- **(b) Server consumers.**
  - `deps.prompt_roots`, `outputs.asset_roots` / `resolution_roots` / `asset_roots_for_job` / `asset_in`, `_asset_origin`, `admission.over_roots` and `exports._copy_assets`.
  - D5, with a test that fails first.
  - Review Focus 1 and 2's committed tests (the parity test).

No listing shape changes yet, so the snapshot diff is empty. D11 and the status rules wait for Task 3.

### Task 3: the one library surface

Two dispatches:
- **(a)** The server, the MCP client and the Python tests, in one commit. `dw_mcp/assets.py` reads `libraries[].dir`, so `tests/test_mcp_assets.py` would be red between two commits.
- **(b)** The UI and its tests.
  - `ui/node_modules` is not in the worktree. `npm --prefix ui ci` is allowed: it is local to the checkout, unlike `pip install -e`.
  - The UI tests are `npm --prefix ui test` (vitest; `ui/src/lib/*.test.ts`, including `api.test.ts`) and `npm --prefix ui run check` (svelte-check plus tsc).

What changes:
- The new envelope for the three listings, the 403 message, 409 for an upload with no asset library (D11), and the B10 rule in the library routes.
- `ui/src/lib/api.ts`, `types.ts` and `AssetsPage.svelte` (plus any page reading `workflow_dir`, `prompt_dir`, `origins` or `sources`) move to the new fields. Run the UI tests.
- `dw_mcp/assets.py`, `catalog.py` and `prompts.py`, with their tests and any tool docstring that names a field.
  - A changed docstring changes the tool surface on purpose, and it appears in the snapshot diff.
  - The surface budget test may need its number updated only if the descriptions changed, and the report says by how much.
- The snapshot diff against `cf1802cb` is attached, with every line accounted for.

### Task 4: Stage 3c merge

- **Re-baseline:** `modules` is unchanged, since `library.py` replaces `workflow_sources.py`.
- **Docs:**
  - CLAUDE.md "Workflow sources" and "Workspaces on the server": replace text, don't add it;
  - `dw/server/CLAUDE.md`;
  - docs/SERVER.md and docs/MCP.md, for the listing shapes;
  - `ui/CLAUDE.md` if it names the fields.
- **Docs grep:** `git grep -n "prompt_dirs\|asset_dirs\|\"sources\"\|origins\|workflow_dir\|asset_dir\b" docs plugins ui/CLAUDE.md dw_mcp` finds no stale field name. An agent-facing guide or a plugin skill may quote one.
- **Release notes:**
  - every break, field by field, the 403 message, and D11's 409;
  - D5: a named workspace's `prompt:` resolves against the server's prompt library during plan building, matching listing, saving and running;
  - D2: `libraries` no longer lists an example prompt dir that does not exist.
- **Hot zone** back to the standing entries. Merge `--no-ff`, push, and do not deploy: the UI changed, and lem gets the build at gate 3, with an rsync.

### Hot zone (3c)

```
scripts/surface_snapshot.py
dw/library.py
dw/workflow_sources.py
dw/assets.py
dw/prompts.py
dw/workspace.py
dw/locations.py
dw/serve.py
dw/server/
dw_mcp/
ui/src/lib/api.ts
ui/src/lib/types.ts
ui/src/lib/pages/AssetsPage.svelte
```

---

## Stage 3d: one media I/O module and one DSP module

Work on branch `stabilization/phase-3d` in the worktree, from `develop` at `e3807f8a` or later.

**What exists (survey [phase-3-surveys/media.md](phase-3-surveys/media.md), re-checked 2026-09-30 at `e3807f8a`).**

- **A partial media layer already exists:**
  - `dw/media_audio.py` (266 lines);
  - `dw/media_info.py` (398), which holds `probe_media` (183 lines) and `probe_metadata`;
  - `dw/media_frames.py` (416);
  - `dw/loudness.py` (82), which uses `pyloudnorm` and `scipy`.
- **`av.open` is called in 6 modules:**
  - `media_audio` (5);
  - `media_frames` (`video_shape`, `_read_frames`);
  - `media_info` (`probe_media`, `probe_metadata`);
  - `tasks/video_utils.py` (`file_fps` :400, `_decode_audio_video` :477);
  - `tasks/assess.py` (`read_media` :123);
  - `pipeline_processors/chain.py` (`_decode_segment` :279).
- **The audio-to-float decode is written 5 times:**
  - `media_audio.extract_audio` (s16) and `decode_soundtrack` (flt);
  - `video_utils._decode_audio_video` (fltp);
  - `assess.read_media` (native dtype through `_as_float_samples`);
  - an inline copy in `probe_media`, through `media_info._as_frame_samples`, which `assess` imports privately.
- **`dw/tasks/audio_utils.py` is 2,202 lines.** By an AST pass:
  - about 500 lines are pure numpy/scipy, with no events and no argument checks: `as_channels_samples`, `slice_samples`, `equal_power_crossfade_join`, `_spectral_flatness`, `_harmonicity`, `crossfade_concat`, `level_dbfs`, `_equal_power_ramps`, `_matched_channels`, `_true_peak_envelope`, `_limiter_curve`, `_limit_at`, `_search_gain`, `_fade_curve`, `_follow_envelope`, `_time_constant_coef`, `_biquad_coefficients`, `_apply_biquad`, `_spectral_balance`, and the core of `resample_waveform`;
  - the other ~1,470 lines are task commands and their warning helpers.

  Moving the DSP out leaves about 1,650 lines, so the commands themselves must split to get under 1,000.
- **There are six level-to-dB helpers, with three silence conventions:**

  | Helper | At silence |
  | --- | --- |
  | `loudness._dbfs` | clamps to `SILENCE_DBFS` (the gallery's `peak_dbfs`) |
  | `result._peak_dbfs` | None |
  | `audio_utils.level_dbfs` | None |
  | `loop_bed._db` | None |
  | `assess._db` | None below its `_SILENCE` threshold, since it serializes to JSON |
  | `voice_attribution._dbfs` | `-inf`, since it is compared numerically |

  `assess` also has its own `_rms` and `_peak`.
- **There are two true-peak oversamplers:**
  - `loudness.true_peak_dbfs` (whole array);
  - `audio_utils._true_peak_envelope` (blocked, 2**18 samples with a 32-sample overlap, which bounds memory).

  Both use `scipy.signal.resample_poly` at 4x.
- **`_apply_biquad`** keeps a pure-Python fallback for a missing scipy. scipy is a declared dependency. `tests/test_audio_utils.py` pins the fallback against `lfilter` (`without_scipy`).
- **`file_fps`** (`video_utils.py:400`) is `container_fps` (`media_audio.py:53`) with every error swallowed to None. Its only callers are `arguments.py:1044` and `:1068`, which rely on the None.
- **`concat_videos` (332 lines) and `dissolve_videos`** repeat four blocks:
  - input loading;
  - the sample-rate reconcile (log, `sample_rate_mismatch` warning, resample; the text is identical except the command name);
  - level matching;
  - the final `fit_audio_to_frames`.

  `video_names` lives in `concat_videos.py` and is imported by `dissolve_videos` and `join_into_song`.
- **Long functions in scope** (lines, as the ratchet counts them): `probe_media` 183, `concat_videos` 332, `join_into_song` 153, `find_loop_bed` 361, `attribute_voices` 174, and teacache's `_create_flux_teacache_forward` 196 with its nested `teacache_forward` 185. That is 7 of the 11.
- **Teacache's footprint:**
  - `dw/teacache.py` (381), `dw/teacache_models.json` (shipped by the `*.json` package-data glob), `tests/test_teacache.py`;
  - `pipeline.py:17` and `:566-583`;
  - the schema's `teacache` property in `$defs/pipeline_configuration`, which is `additionalProperties: false`, so a leftover key is refused;
  - the phrase "Mutually exclusive with teacache" in the `cache` description (`workflow_schema.json:738`);
  - README.md:215 and :232, docs/ACCELERATION.md:126-182, docs/WORKFLOW_GUIDE.md:1465-1469, docs/TESTING.md:58, tests/README.md:31.

  No shipped workflow uses the key.
- **The task registry names each implementation by dotted path:** `task.py`'s `register_command(..., implementation="dw.tasks.audio_utils.fade_audio")`, 11 for audio_utils. `describe_task` imports that path only to read its signature and docstring. The path itself never reaches the API, so moving a command is invisible while its signature and docstring stay byte-identical.

### Decisions (3d)

- **Modules: 151 → 151.**

  | Task | Change | Count |
  | --- | --- | --- |
  | Task 1 | delete `dw/teacache.py` | −1 → 150 |
  | Task 2 | add `dw/dsp.py`, fold `dw/loudness.py` into it | net 0 → 150 |
  | Task 3 | `dw/media_audio.py` becomes `dw/media.py` (a rename); fold `media_info.py`'s probes into it and delete `media_info.py` | −1 → 149 |
  | Task 4 | split `audio_utils.py` into `dw/tasks/audio_dynamics.py` and `dw/tasks/joins.py` | +2 → 151 |

  The projection was +2, so this is two under it. No task rises above the 151 baseline, so the ratchet passes at every commit.
- **`dw/media.py` is the one module that opens a container.**
  - It holds media_audio's functions, plus `probe_media` and `probe_metadata` with their helpers.
  - Every other `av.open` moves into it as a decode or probe function: `media_frames`'s two, `video_utils._decode_audio_video`, `assess.read_media`'s decode, and `chain._decode_segment`.
  - `media_frames.py` stays as a consumer: contact sheets and seam tiles get their frames through `media`. Folding it in as well would put `media.py` over 1,000 lines.
  - One `to_float32(frame or samples)` replaces the five decode conversions, and one layout-name helper replaces the three spellings.
  - `media.py` returns arrays and rates, never step types. `video_utils.load_audio_video` still builds the `AudioVideo`, so `tests/test_shots.py`'s constructor-site list is unchanged.
  - `media.py` imports nothing from `dw.tasks` or `dw.result`. It may import `dw.media_types` (`fit_codec_padding`) and `dw.dsp`.
  - Each call keeps exactly one `av.open`. The decode-counting tests pin that (Review Focus 1).
- **`file_fps` is deleted.** `media.container_fps` survives and keeps raising. `arguments.py`'s two sites call it through one private `_declared_fps(path)` in `arguments.py`, which catches, logs at debug and returns None, exactly `file_fps`'s behaviour. 3e moves that half of `arguments.py` anyway.
- **`dw/dsp.py` is pure numpy/scipy/pyloudnorm.**
  - It imports nothing from `dw`, and it emits no events.
  - It takes `loudness.py` whole (`integrated_lufs`, `true_peak_dbfs`, `SILENCE_DBFS`, `MIN_LUFS_SECONDS`, `TRUE_PEAK_OVERSAMPLE`) and the pure functions listed above, each under its public name (the leading underscore dropped).
  - It holds the `LIMITER_*` constants, the periodicity thresholds, and `resample_waveform`. Resampling stays on PyAV's `AudioResampler`: that is an `av` frame API, not `av.open`, so it does not break the media rule.
  - `_normalize_limited` stays task-side, because it emits `target_lufs_capped` and `limiter_heavy`. Only the curve, the gain search and the envelope go to `dsp`.
  - Tests that read `audio_utils.LIMITER_*` or a private DSP name are repointed to `dsp`.
- **One `dsp.dbfs(amplitude, floor=None)`.**
  - Rule: None or ≤ 0 returns `floor`. Otherwise `20·log10(amplitude)`, clamped to `floor` when a floor is given.
  - It keeps each caller's silence convention:

    | Caller | Call |
    | --- | --- |
    | `loudness` (now in `dsp`) | `dbfs(v, floor=SILENCE_DBFS)` |
    | `result._peak_dbfs`, `level_dbfs`, `loop_bed` | `dbfs(v)`, so None |
    | `assess` | keeps its `_SILENCE` guard, then `dbfs(v)` |
    | `voice_attribution` | `dbfs(v, floor=-math.inf)` |

  - `assess._rms` and `_peak` become `dsp.rms` and `dsp.peak`. `result._peak_dbfs` keeps its tensor-to-numpy coercion; only its last line changes.
- **One true-peak oversampler.**
  - The blocked one survives as `dsp.true_peak_envelope`, and `true_peak_dbfs` becomes the max of it in dB.
  - First, a characterization test: the two agree within 0.01 dB on a short track, a track longer than 2**18 samples, a sine, and an impulse. It must pass before the old one is deleted.
- **The build-vs-buy list is exactly the four items in the stages table:**
  - the biquad fallback and its `without_scipy` test;
  - the dBFS copies;
  - the second true-peak oversampler;
  - `file_fps`.

  The survey's optional swaps (`scipy.signal.iirnotch`, `resample_poly` for resampling, `scipy.signal.correlate` in `_harmonicity`) are **not** made. They change numbers under a hard freeze, and the tests pin those numbers at 1e-5.
- **The audio commands split three ways.** Every command keeps its name, signature and docstring, and `task.py`'s `implementation=` strings are repointed.
  - **`dw/tasks/audio_utils.py`:** track plumbing (`_as_track`, `_as_number`, `_waveform_and_rate`, `_warn_on_rate_override`, `load_audio`, `_track_names`), `slice_audio` and its warnings, `gain_audio`, `resample_audio`, `fade_audio`, `crossfade_audio`, `mix_audio`, `loop_audio`.
  - **`dw/tasks/audio_dynamics.py`:** `normalize_audio` with `_normalize_limited`, `compress_audio`, `filter_audio`, `analyze_audio`.
  - **`dw/tasks/joins.py`:** what every joining command shares:
    - `video_names`, `fit_audio_to_frames`, `bleed_join`, `_declick_join`, `match_levels`, `warn_on_level_spread`, `_load_tracks_matching_rate`;
    - the concat/dissolve shared steps (Task 5).

    Its users are `concat_videos`, `dissolve_videos`, `join_into_song`, `chain.py` and `crossfade_audio`.
- **Patch targets move with their names, in the same commit, and the count does not rise:**
  - `dw.media_info.probe_media` (7, `tests/test_result.py`) becomes `dw.media.probe_media`, together with its lazy import at `result.py:195`;
  - `dw.media_audio.av.open` (2) and `dw.media_frames.av.open` (1) become `dw.media.av.open`;
  - `dw.tasks.concat_videos.load_audio_video` (2) is retargeted if the shared join loads the inputs.

  A patch whose target stopped being looked up passes vacuously, so the reviewer checks each one still intercepts.
- **What 3d touches in 3e's files:**
  - `pipeline.py` loses only the teacache import and branch;
  - `result.py` only the `probe_media` repoint and the last line of `_peak_dbfs`;
  - `arguments.py` only the two fps sites;
  - `workflow_schema.json` only the teacache property and the one phrase.

  All four are in the 3d hot zone so the harness keeps off them.
- **The carried items:**
  - **Frame-size sentence:** one producer, `task_domains.frame_size_error(command, sizes)`. Both `check_same_frame_size` and `video_size_errors` call it, and `tests/test_rule_parity.py` covers both. Its text is unchanged.
  - **Slice region:** one pure helper beside `frames_to_samples` in `task_domains.py`. It answers the requested region in samples, and `slice_audio` and `slice_preflight._requested_region` both call it.
  - **Dissolve shortfalls:** the run raises every shortfall in one `ValueError`, joined with `"; "`, as validation already reports them all. With one shortfall the message is unchanged. With several it changes, and the release note says so.
  - **Test temp dirs:** the dissolve and shot-span preflight test helpers use `tmp_path`, not `tempfile.mkdtemp()`.
- **The surface snapshot gains the task surface:** `list_tasks()`, `describe_task(c)` for every command, and the workflow schema. Task 1 snapshots the base first. Across 3d the only allowed diff is the teacache property and the phrase in the schema.

### Review Focus (3d)

1. **Decode once.** Every test that counts `av.open` or `InputContainer.decode` passes unchanged in meaning: `test_media_audio`, `test_media_frames`, `test_server`, `test_media_info`, `test_admission` (Phase 2b's B9). One call is one `av.open`.
2. **The numbers do not move.** The pinned values run unchanged:
   - compressor/limit/gate to 1e-5;
   - LUFS targets to 0.5 LU;
   - the limiter constants and curve;
   - exact-sample fades, crossfades and declicks;
   - spectral Parseval to 0.05 dB;
   - the plugin skills' pinned numbers.
3. **Silence per caller.** A silent track gives `peak_dbfs` at `SILENCE_DBFS` in the gallery, None in `assess` findings (a JSON body that serializes), None from `level_dbfs`, and `-inf` inside voice attribution.
4. **Teacache is refused, and nothing else is.** A pipeline `configuration` holding `teacache` fails validation with the schema's own message. `templates/step-caching.json` and every catalog workflow still validate.
5. **Join parity.** `concat_videos` and `dissolve_videos` give the same output, warnings (`sample_rate_mismatch` and the level-spread warning, word for word except the command name) and shots as before. `test_concat_videos`, `test_dissolve_videos`, `test_shots`, `test_rule_parity` and `test_assess` pass, and every retargeted patch still intercepts.

### Task 1: The task surface snapshot, and teacache deleted

- Extend `scripts/surface_snapshot.py` with `tasks` (`list_tasks()` and `describe_task(c)` for each command) and `workflow_schema`. Snapshot the base to `.superpowers/sdd/phase-3/3d-base.json` before changing anything.
- **Failing test first:** a definition whose pipeline `configuration` holds `"teacache": {"rel_l1_thresh": 0.4}` fails `validation_errors`. The test asserts the `additionalProperties` message that names `teacache`.
- Delete `dw/teacache.py`, `dw/teacache_models.json` and `tests/test_teacache.py`.
- In `pipeline.py`, remove the import and the branch (`:566-583`), so the method keeps only the attention-backend context.
- In the schema, remove the property and the "Mutually exclusive with teacache" phrase.
- **Docs:**
  - delete docs/ACCELERATION.md's TeaCache sections and its "Cache vs TeaCache" table, leaving its pointer to `first_block` and `mag`;
  - remove WORKFLOW_GUIDE.md:1465-1469;
  - drop TeaCache from README.md:215 and :232, docs/TESTING.md:58 and tests/README.md:31.
- The snapshot diff against the base is the schema lines only.
- Cheap model.

### Task 2: `dw/dsp.py`

- Create `dw/dsp.py` (Decisions), with `loudness.py` folded in and deleted. Repoint every importer: `audio_utils`, `join_into_song`, `media_info`, and the tests.
- The pure functions move verbatim under public names, and `audio_utils` imports them back for its commands.
- `dsp.dbfs` replaces the six copies, using the per-caller table (Decisions).
- `assess._rms` and `_peak` become `dsp.rms` and `dsp.peak`.
- **The true-peak characterization test lands first, then the merge.** It is a move, so it must pass before and after.
- Delete the biquad fallback and the `without_scipy` test.
- Review Focus 2 and 3 are the proof.
- The snapshot diff is empty.

### Task 3: `dw/media.py`

- `git mv dw/media_audio.py dw/media.py`. Fold `probe_media`, `probe_metadata` and their helpers in from `media_info.py`, and delete it.
- Move each remaining `av.open` in as a decode or probe function (Decisions).
- One `to_float32` and one layout-name helper.
- Delete `file_fps`; `arguments._declared_fps` takes its place.
- **Cut `probe_media` under 150 lines:**
  - `_stream_info(container)`;
  - an `_AudioAccumulator` (`add(frame)` and `finish()`, giving the levels, LUFS and envelope);
  - the decode loop.
- Retarget the patch targets in the same commit.
- **Add `tests/test_media_layering.py`.** It is an AST test that encodes the stage's invariant, not a count:
  - `av.open` appears only in `dw/media.py`;
  - `dw/dsp.py` imports no `dw` module;
  - `dw/media.py` imports nothing from `dw.tasks` or `dw.result`.

  It fails before the move, because of the `av.open` sites in `media_frames`, `video_utils`, `assess` and `chain`.
- Review Focus 1 is the proof.
- The snapshot diff is empty.

### Task 4: the audio commands split, and the two carried rules

- Split `audio_utils.py` into `audio_utils`, `audio_dynamics` and `joins` (Decisions). Repoint `task.py`'s `implementation=` strings and every importer: `chain.py`, `concat_videos`, `dissolve_videos`, `join_into_song`, `loop_bed`, `pair_audio`, `speech_generation`, `audio_transcription`, `voice_attribution`, `assess`, `arguments`, `assessment_rules`, and the tests.
- The private cross-module imports the survey lists become public names in the module that owns them:
  - `_waveform_and_rate`, `_as_number` and `_spectral_balance` are named in `audio_utils`;
  - `_spectral_flatness` and `_harmonicity` are named in `dsp`.
- Add the slice-region helper and the frame-size sentence's one producer (Decisions). Each gets a parity test: both callers give the same region for #557's rounding case, and both sizes messages come from the one function.
- Each of the three modules is under 1,000 lines, and none of their functions is over 150.
- The snapshot diff is empty: every task schema is byte-identical.

### Task 5: `concat_videos` and `dissolve_videos` on one join

- Move into `joins.py` the steps both commands repeat:
  - loading the inputs with their names;
  - reconciling sample rates (one `sample_rate_mismatch` producer, taking the command name);
  - matching levels or warning on the spread;
  - fitting the joined track to the frame grid at the written fps.
- **Cut `concat_videos` under 150 lines:**
  - `_shot_records_for`;
  - `_input_waveform` (the silence fill, or fit plus the drift warning);
  - `_join_seam` (the bleed or equal-power crossfade).
- The dissolve run raises every shortfall (Decisions). The test fails first, with two short inputs.
- Fix the temp-dir leaks in the dissolve and shot-span test helpers.
- Review Focus 5 is the proof.

### Task 6: the long task functions

- **`find_loop_bed` under 150 lines**, cut per the survey's outline:
  - `_coerce_arguments`;
  - `_search_window`;
  - `_tonal_scales`;
  - `_edge_ticks`;
  - `_scan_windows`, which returns the survivors and the rejected tally;
  - `_rank_and_loop`;
  - `_answer`.
- **`attribute_voices`:**
  - `_embed_references`;
  - `_attribute_lines`;
  - its closures `clip_duration` and `stem` become module functions.
- **`join_into_song`:** `_validated` and `_place_shots`.
- Behaviour is preserved, which the existing suite proves: `test_find_loop_bed`, `test_voice_attribution` (its six `patch` targets still intercept) and `test_join_into_song`.
- `voice_attribution`'s `_mono_16k` and the same load, resample and mono step in `speech_generation` and `audio_transcription` become one `dsp.to_mono_at(waveform, rate, target)`, but only if the three are the same arithmetic. If one differs, it is left alone, with a line in the report.

### Task 7: Stage 3d merge

- **Metrics:**
  - `modules` 151 (no change; the list is in Decisions);
  - `modules_over_1000_lines` 7 → 6;
  - `functions_over_150_lines` 11 → 4 (`save_artifact`, `estimate`, `Workflow.run` and `create_step_action` remain for 3e);
  - re-baseline everything that fell.
- **Docs:**
  - CLAUDE.md wherever it names a moved home: the audio headroom bullets name `_normalize_limited` and `LIMITER_*`, and the assessment bullet names `dw/tasks/assess.py`;
  - docs/TASKS.md's "no torchaudio dependency" line, if it names a file;
  - `dw/server/CLAUDE.md`, if it names `media_info` or `media_audio`.

  Replace text, don't add it.
- **Release notes:**
  - **Breaking:** the `teacache` pipeline configuration key is removed. A workflow that sets it fails validation with the message the Task 1 test asserts. The replacement is `cache` (`first_block`, `mag`, `taylorseer`), per docs/ACCELERATION.md.
  - A `dissolve_videos` run with several short inputs names all of them.
  - Internally: the media and DSP modules, and the commands' new homes.
- Hot zone back to the standing entries. Merge `--no-ff`, push, and do not deploy (lem gets 3d at gate 3).

### Hot zone (3d)

```
scripts/surface_snapshot.py
dw/dsp.py
dw/loudness.py
dw/media.py
dw/media_audio.py
dw/media_info.py
dw/media_frames.py
dw/teacache.py
dw/teacache_models.json
dw/workflow_schema.json
dw/pipeline_processors/pipeline.py
dw/pipeline_processors/chain.py
dw/result.py
dw/arguments.py
dw/task_domains.py
dw/slice_preflight.py
dw/video_size_errors.py
dw/dissolve_frame_errors.py
dw/shot_span_preflight.py
dw/validation.py
dw/assessment_rules.py
dw/tasks/task.py
dw/tasks/audio_utils.py
dw/tasks/audio_dynamics.py
dw/tasks/joins.py
dw/tasks/concat_videos.py
dw/tasks/dissolve_videos.py
dw/tasks/join_into_song.py
dw/tasks/loop_bed.py
dw/tasks/voice_attribution.py
dw/tasks/assess.py
dw/tasks/pair_audio.py
dw/tasks/video_utils.py
dw/tasks/speech_generation.py
dw/tasks/audio_transcription.py
dw/server/routes/media.py
dw/server/routes/gallery.py
docs/ACCELERATION.md
```

The image and text task modules stay open to the harness. `pipeline.py`, `result.py` and `arguments.py` are listed for the few lines 3d changes in each, which keeps a harness edit from colliding with them.

## Stage 3e: engine splits

Work on branch `stabilization/phase-3e` in the worktree, from `develop` at `b9c2096e` or later.

**What exists (three surveys at `b9c2096e`, in `.superpowers/sdd/phase-3/3e-survey-*.md`; the facts below were checked there).**

- **Sizes.** Six modules are over 1,000 lines and all of them are in 3e:

  | Module | Lines | Long or complex functions |
  | --- | --- | --- |
  | `dw/pipeline_processors/pipeline.py` | 2,394 | `load_component` 147 lines, complexity 14 |
  | `dw/workflow.py` | 2,239 | `Workflow.run` 574 lines, complexity 35; `create_step_action` 279, complexity 15 |
  | `dw/result.py` | 1,740 | `Result.save_artifact` 344 lines, complexity 26 |
  | `dw/introspection.py` | 1,242 | `_parse_docstring_args` complexity 16 |
  | `dw/arguments.py` | 1,233 | `realize_args` 132 lines, complexity 27 |
  | `dw/security.py` | 1,040 | none |

  `dw/plan.py` (886) holds the fourth long function, `estimate` (201 lines, complexity 16). The 3e files hold 5 of the 12 `complex_functions`.
- **`result.py` has four concerns besides `Result`:**
  - audio QC: `warn_without_headroom`, `warn_if_written_above_full_scale`, `warn_if_written_near_silent` and their constants and probe helper (66-338), plus `save_artifact`'s post-write block (927-1062);
  - byte-level writers: naming (`output_file_path`, `_dedupe_existing_path`), `flatten_alpha_for`, `frames_for_encoding`, `write_audio`, the waveform `normalize_audio` (not the task command), `as_audio_track`, `_as_stereo`, and the image metadata pair `_save_image_with_metadata` / `read_embedded_metadata`;
  - output extraction: `get_artifact_list`, `modular_artifacts` and their helpers (1378-1593), which import only `media_types`, numpy and torch;
  - `guess_extension`, a content-type fact that reads `content_types.AUDIO_FORMATS`.
- **`result.py`'s patch targets:**
  - 39 of its 41 string targets name `encode_video`, `export_to_video` and `is_av_available`. They are looked up by the video writes (`video_fps`, `conform_artifact(s)`, `save_audio_video`, and the `video`/`gif` dispatch), which stay in `Result`.
  - The other 2 name `warn_if_written_above_full_scale`, looked up at `result.py:1009`.
  - 17 `patch("dw.media.probe_media")` lines work only because `_probe_written_media` imports `probe_media` lazily (`result.py:194`).
  - 3 `monkeypatch.setattr(result_module, "emit_warning")` lines (`tests/test_result.py:571`, `:1788`, `:1809`) follow the dict-tensor skip, `_as_stereo` and `flatten_alpha_for`.
- **`Selected`** is used in `result.py` only by `Result.add_result` (502-506), through a lazy import from `dw.tasks.select`. It is lazy by habit, not for a cycle. `tests/test_result.py:52` is its only other importer.
- **`tests/test_shots.py:83`** pins `("dw/result.py", "pair_audio_with_frames")` in its constructor-site list.
- **`plan.py:37-38`** re-exports `unseeded_cache_warnings` from `dw.validation` ("so `from dw.plan import ...` keeps working"). That is a shim, and `tests/test_plan.py` is its only user.
- **`arguments.py`** is three groups: the core (`realize_args`, path, constant and prompt references, lazy frame commands; 371 lines), object construction (562) and media fetching (300).
  - 7 of its 8 string patch targets name media names (`safe_get` ×2, `_fetch_remote_video` ×4, `load_video` ×1). `load_type_from_name` stays.
  - Two `monkeypatch.setattr(arguments_module, "load_video")` lines (`tests/test_video_utils.py:608`, `tests/test_assess.py:803`) also follow `load_video`.
  - `dw/validation.py:31` imports the private `_names_no_media`.
  - `_with_frame_rate`'s `http(s)` branch (1045-1049) is dead: its one caller (1171) passes `validate_media_path`'s absolute, resolved path.
  - Two lazy imports are redundant: `shots_beside` from `.runs` (1039) and `PIL.Image` (806), both already imported at the top.
- **`introspection.py`:**
  - The type-reference checks (779-959) and `component_name_errors` (962-1037) are one group of 257 lines. The inert-argument warnings with `workflow_argument_warnings` (1040-1242) are another, of 203.
  - Both groups are imported by `dw/validation.py:36`, and `workflow_argument_warnings` also by `dw/server/routes/library.py:19`.
  - `_NAME_PATTERN` (23) is used by both groups.
  - The type-reference block imports `NON_TYPE_KEYS`, `fetch_constant`, `is_constant_reference` and `is_media_reference` lazily from `arguments`.
- **`security.py`:**
  - It imports nothing from `dw`, and every module that depends on it relies on that.
  - The trust gate (76-344, 269 lines) is its own concern. Its importers are `type_helpers`, `pipeline_processors/pipeline.py`, `pipeline_processors/remote.py`, `locations`, `serve`, `validate` and `server/routes/system.py`. `UntrustedWorkflowError` belongs to the exception family.
  - The name and reference validators are 413 lines (554-966).
- **CodeQL models validators in two places:**
  - `.github/codeql/dw-security/DwPathSanitizers.qll` matches by function name, except `ValidatorParameter`, which requires `getRelativePath() = "dw/security.py"` (line 135).
  - `.github/codeql/extensions/dw-models/models/dw-security.model.yml` matches by module `dw.security` (9 rows).
  - The trust gate defines no validator either model names.
- **`pipeline.py`** has the `Pipeline` core (757 lines) and output checks (86), plus five concerns:

  | Concern | Lines |
  | --- | --- |
  | placement | 353 |
  | components | 693 |
  | adapters | 158 |
  | progress reporting | 228 |
  | denoiser cache hooks | 91 |

  - Its patch targets are `patch("dw.pipeline_processors.pipeline.load_component")` ×2 (`test_modular_pipeline.py:198,217`) and 5 string `monkeypatch.setattr("dw.pipeline_processors.pipeline.<name>")` lines for `load_component`, `load_loras` and `empty_device_cache` (`test_pipeline_components.py:716-753`). They are looked up by `Pipeline.load`, `load_optional_component` and `_discard_failed_load`, which stay.
  - `tests/test_configuration_schema.py:21-24` scans `pipeline.py` and `config_objects.py` (`SOURCES`) for `configuration.get(...)` keys and asserts at `:111` that it finds `offload`, which only `loading_device` and `place_component` read.
  - `test_offload_placement.py:45-47` monkeypatches `get_device_type` for `place_component`.
  - `_MISSING` is shared by `_resolve_submodule` and `get_component`.
- **`workflow.py`** has four per-run ownership tables on `self`:
  - `_pipeline_keys_by_step`, also read by `dw/worker.py:540`;
  - `_running_pipeline_keys`;
  - `_deferred_pipelines`;
  - `_prior_step_keys`.

  Five `hasattr`/`getattr` guards exist because tests call `create_step_action` without `run`. `create_step_action` is called directly about 30 times in tests and spied with `patch.object`.
- **`workflow.py`'s memory reclaim:**
  - 19 string patch targets name `dw.workflow.empty_device_cache` and `dw.workflow.release_host_caches`. They are looked up at `run` 1732 (between steps), `_finish_release` 1861 and the superseded eviction (2080-2081).
  - `tests/test_pipeline_caching.py:263-275` asserts `call_count == steps + 1` across two of those sites.
- **`workflow.py`'s validation block** (607-966) is about 360 lines.
  - `dw/validation.py` (879) already calls `sub_workflow_errors` and `sub_workflow_argument_warnings` duck-typed, and promises never to import `dw.workflow`.
  - `validation_errors` has 105 test call sites in 29 files.
  - 14 `patch.object(dw.workflow, "get_device_type" / "device_capacity_gb")` lines serve `validation_context`: `test_vram_estimate.py` 278-331 ×10, `test_vram_inheritance.py:61-62`, `test_h3_vram_ceiling.py:27-28`.
- **`Workflow.run`** has eight phases:

  | Phase | Lines | What |
  | --- | --- | --- |
  | A | 1220-1264 | context |
  | B | 1266-1315 | prepare; `_owned_arguments` rebinds `arguments` for a composed child |
  | C | 1317-1391 | run directory; resets `started_at` and `run_id` |
  | D | 1393-1443 | step setup, with an early return for no steps |
  | E | 1446-1732 | the per-step loop |
  | F | 1734-1740 | `workflow_end` |
  | G | 1742-1761 | the three `except` clauses |
  | H | 1762-1777 | the manifest write and teardown |

  - **Event order:** the prepare warnings, `run_start` (only when the run owns its directory), `workflow_start`, then per step `step_start`, deferred loads, the action's `cached`/`loading` phases, the step's own events, `pipeline_released`, the save logs and warnings, `shot_name_collision`, `step_end`; finally `workflow_end`.
  - **The release frame:** the release drops `step_action` and the popped pipeline in the frame that then calls `_finish_release` (1576-1579), so `gc` and `empty_device_cache` can free it. No test pins this.
  - **`_run_dir` is never reset at the top of `run`.** A reused `Workflow` that fails in prepare writes a failed manifest into its previous run's directory. The server builds a fresh `Workflow` per job.
- **The carried items, as found:**
  - `workflow.py:747` is stale: the `author_index` idiom is at 660 and 732.
  - The elision carry is `elision._carry_release` (125-161), which hands `step_pipeline_keys` steps whose `pipeline` is only checked for truthiness.
  - A composed child's `argument_template` is written into its `workflow_definition` (`create_step_action` 2212-2214) and deep-copied again by `validate()` and `run()`.
  - The three `previous_result:` scanners are `step_cache.referenced_result_names`, `previous_results._collect_refs` and `previous_results._collect_reference_paths`:
    - The first recurses into a `from_previous_result` value, so `"previous_result:x"` written there yields both `"previous_result:x"` and `"x"`.
    - The other two record the value bare and do not recurse.
    - The third also skips a `from_previous_result` that is a `variable:` reference.
  - `for_each._copy_leaf` and the step-cache snapshot (`copy.deepcopy(step_data)`, `workflow.py:1102`) still copy media leaves.
  - Prefix spellings: 39 helper uses, about 55 `startswith`/`removeprefix` lines and 17 slicing lines across about 30 modules. Four f-strings build a prefix the metric cannot see: `introspection.py:1202`, `workflow.py:942`, `routes/assets.py:363` and `:433`.

### Decisions (3e)

- **Modules: 151 → 165 (+14), against the phase's projection of about +10 for 3e (about 158 overall, accepted 2026-09-30).**

  | Task | New modules | Count |
  | --- | --- | --- |
  | Task 1 | `dw/writers.py`, `dw/audio_qc.py`, `dw/output_extraction.py` | +3 → 154 |
  | Task 4 | `dw/argument_media.py` | +1 → 155 |
  | Task 5 | `dw/type_references.py`, `dw/argument_warnings.py` | +2 → 157 |
  | Task 6 | `dw/trust.py` | +1 → 158 |
  | Task 7 | `dw/pipeline_processors/placement.py`, `components.py`, `adapters.py`, `progress.py` | +4 → 162 |
  | Task 8 | `dw/pipeline_ownership.py` | +1 → 163 |
  | Task 9 | `dw/step_value_checks.py` | +1 → 164 |
  | Task 10 | `dw/workflow_run.py` | +1 → 165 |

  - **Why +4 over:** `pipeline.py` and `workflow.py` alone need +7, because no smaller set gets them under 1,000. The survey checked each merge:
    - adapters into components would be about 970 lines;
    - progress kept in `pipeline.py` would be about 1,085;
    - the ownership functions in `workflow_run.py` would be about 950.
  - **Already trimmed:** the optional `argument_objects.py` and `task_signatures.py` splits are not taken, and `security.py` loses one module's worth, not two.
  - **Ratchet:** each task's commit re-baselines `modules` to its row, so the ratchet holds after every task.
  - **Cost if wrong:** the phase ends at about 165, not 158. Phase 4 re-sets the ratchet either way.
  - **This is the first line of the 3e report to Don.**
- **Deviations from the stages table (line 71), recorded here and not edited there** (the precedent is 3d's `_load_tracks_matching_rate` note):
  - **`security.py` sheds only the trust gate (→ about 771 lines). The name validators stay, so neither CodeQL model changes.**
    - Why: moving the name validators would re-point five `.model.yml` rows and `ValidatorParameter`, and only a CodeQL run in CI can check that. The trust gate holds no modelled validator. Paths, names and references stay as one input-validation module, and trust is a separate concern.
    - Cost if wrong: `security.py` keeps two validator families. It is under the limit either way.
  - **Prefix spellings: 3e converts the spellings in the code it moves.** That covers `arguments`, `introspection` (including the f-string at 1202), `workflow` (including 942), `plan`, `previous_results`, `step_cache` and `for_each` where Task 11 touches them.
    - Phase 4 (guardrails) takes the other modules, together with a metric that counts `startswith`/`removeprefix` on a prefix and f-string prefix parts. The spelling only holds once a ratchet holds it.
    - Cost if wrong: the item lingers one more phase.
  - **Not done, because each changes behaviour under the freeze.** Each stays in the carried list for after Phase 3:
    - **`for_each._copy_leaf` sharing leaves:** siblings would see each other's in-place edits.
    - **The step-cache snapshot via `copy_containers`:** the "not copyable, so uncacheable" branch would go dead, and such a step would become cacheable.
    - **Sharing `resolve_sub_workflow_path` with `create_step_action`:** a missing builtin would raise `SubWorkflowNotFound` instead of a security error.
    - **`_run_dir`'s reset:** preserved exactly (Task 10).
- **Validation, Option B:**
  - The public validation methods stay on `Workflow` with one-line bodies that call `dw.validation`: `validation_context`, `validation_errors`, `validate`, `adapter_warnings`, `inherited_vram_warnings`, `slice_past_end_warnings`, `shot_span_warnings`, `null_variable_argument_warnings` and `sub_workflow_warnings`. The same goes for `cache_hits` (Task 10).
  - **A method that delegates without moving a name is not a shim.** No name has two import paths, and nothing patches a name that is not looked up.
  - The private bodies move, and their tests retarget.
  - `validation_errors` keeps two lines: building a context when none is given, and the `ValueError` for a `context` passed with conflicting `arguments`/`composing` (`workflow.py:777-800`). Both go into `validation.workflow_errors(workflow, arguments, composing, context)`, and the method becomes one call to it.
  - **Callers in `dw/` keep calling the `Workflow` methods.** Nothing in `dw/` imports `workflow_errors` or the other moved bodies directly. `tests/test_server.py:738,6156` monkeypatch `Workflow.validation_errors`, and a caller switched to the module function would leave them passing vacuously.
  - Cost if wrong: about 150 call sites would later move to module functions, mechanically.
- **`step_value_checks.py` reverses 2b's folding of `fps_errors`, `null_media_errors` and `select_errors`.**
  - 2b folded them to pay for creating `dw/validation.py` (Decisions (2b): "It pays for itself"), not on a one-module principle. Its invariant, one registry and one runner, stays in `validation.py`.
  - Moving the block in adds about 225 lines to `validation.py`, 1,105 without this move. The checker bodies (about 292 lines) are the seam that keeps the registry whole.
- **`ConstantError` moves to `dw/variables.py`,** below both `workflow` and `validation`.
- **A composed child is opened through `Workflow.open_sub_workflow(path)`**, which returns `(child, resolved)` using `resolve_sub_workflow_path` + `workflow_from_file`. This is how `validation` and `workflow_run` construct children without importing `dw.workflow`. They test a child with `isinstance(x, type(workflow))`; `dw/` and `tests/` define no `Workflow` subclass.
- **Patch targets move only with their lookup site, in the same commit, and `test_dw_patch_targets` stays 284:**
  - the 2 `warn_if_written_above_full_scale` targets go to `dw.audio_qc` (Task 2);
  - the 7 media targets go to `dw.argument_media` (Task 4);
  - the 19 memory targets go to `dw.pipeline_ownership` (Task 8). All three `empty_device_cache` sites and `_release_host_caches` must land in that one module, or `steps + 1` needs two patches.
  - The uncounted ones move too:
    - the 3 `emit_warning` monkeypatches (Tasks 1 and 2);
    - the 2 `load_video` monkeypatches (Task 4);
    - the `get_device_type` monkeypatch (Task 7);
    - the 14 `patch.object(dw.workflow, "get_device_type" / "device_capacity_gb")` (Task 9);
    - `workflow_module.realize_args` (`test_workflow_step_cache.py:152,165`; Task 10).
  - **A `patch.object` on a name the module still imports but no longer looks up passes vacuously.** The metric does not count these. The reviewer checks each one still intercepts, by breaking the patched function once and watching the test fail.
- **The 17 `dw.media.probe_media` patches need the lazy import kept verbatim** in `audio_qc._probe_written_media`. Hoisting it binds the original name, and those tests then probe real files.
- **The 39 encoder targets stay valid because the video writes stay in `Result`.** If any part of the video dispatch moves to `writers.py`, those patches silently stop intercepting and real encoders run.
- **`Selected` moves to `dw/media_types.py`,** beside `AudioVideo` and `AudioTrack`. `result.py` imports it at the top, so it has no upward import left.
- **One `previous_result:` walker, with exact parity per caller.**
  - `references.iter_previous_result_references(value, *, descend_into_from)` yields `(path, name, via)`.
  - `step_cache` calls it with `descend_into_from=True`, so `"previous_result:x"` written as a `from_previous_result` value still yields both spellings.
  - `previous_results` calls it with `False`, and its reference-error caller drops `variable:` values.
  - It lives in `references.py` because `previous_results` imports `step_cache`. `tests/test_references.py:78-86` asserts `references.py` has no import statements at all, so the walker uses no `typing`, `collections` or `dataclasses`.
  - Parity tests are written against the three old functions first and pass before the switch. The extra spelling is kept, not argued away: the freeze settles it.
- **The surface snapshot gains `catalog_validation`:** for every JSON under `workflows/` and `dw/workflows/`, `validation_errors()` and the warning checks that need no server state (no `ceiling_index`, no observed costs).
  - It is a same-machine comparison, because `validation_context` reads the device.
  - Task 1 runs it twice on the base and diffs the two before relying on it.
  - Across 3e the whole snapshot (routes, OpenAPI, MCP, tasks, schema, catalog validation) is byte-identical.
- **Sanctioned size fallbacks** (use one only if a module lands over 1,000, and say so in the report):
  - `arguments.py`: the object-construction group to `dw/argument_objects.py`, with `NON_TYPE_KEYS` to `type_helpers`. That adds +1 module.
  - `workflow.py`:
    - `workflow_output_subfolder` and `catalog_root_dir` to `dw/library.py`;
    - then the resident-reuse and fresh-load wrappers to `pipeline_ownership.py` (survey §2d).

    Neither adds a module.

### Review Focus (3e)

1. **Event order.** The `save_artifact` order is:
   1. `writing`;
   2. the rate-override warning;
   3. the pre-write `audio_no_headroom` warning;
   4. the flatten-alpha warning;
   5. `fps_mismatch` (up to three `video_fps` calls; keep them all);
   6. probe, then `joined_audio_short_after_mux`;
   7. `audio_clipped`;
   8. the held `audio_no_headroom`;
   9. `audio_near_silent`;
   10. `wrote`.

   The `run` order is the one in What exists. `test_events`, `test_phase_events`, `test_result`, `test_runs` and `test_shots` pass unchanged.
2. **The release frees before the save.** A new characterization test (Task 10, written on the base first) holds a `weakref` to a released pipeline and asserts it is dead when the step's `Result.save` runs. No helper may keep `step_action` or the popped pipeline alive across `finish_release`.
3. **Every moved patch still intercepts:**
   - the 19 memory targets, with `steps + 1`;
   - the 7 media targets and the 2 `load_video` monkeypatches;
   - the 2 `audio_qc` targets and the 17 `probe_media` patches;
   - the 14 `patch.object` device lines;
   - the `get_device_type` placement monkeypatch.

   A vacuous patch is a defect even when the suite is green.
4. **Composed runs.**
   - A child's rebound `arguments` reach `new_run_id` and every manifest write (`RunRecord.arguments`).
   - A child reads its handed arguments without a copy in its definition.
   - A parent-saved child conforms and does not save.
   - `test_sub_workflow*`, `test_runs` and `test_for_each` pass unchanged.
5. **Validation verdicts are byte-identical.** The `catalog_validation` snapshot does not move. Each `author_index` swap and the elision guard are neutral on valid input. A definition with a non-dict `pipeline` beside an elided step no longer raises `AttributeError` from the carry.

### Task 1: The catalog-validation snapshot, and `result.py`'s helpers out

- **Snapshot:** add `catalog_validation` to `scripts/surface_snapshot.py` (Decisions).
  - Run it twice on the base. The two must be identical before anything else; if they are not, stop and report what varies.
  - Then snapshot to `.superpowers/sdd/phase-3/3e-base.json`.
- **Create `dw/writers.py`:**
  - Contents: `_artifact_size`, `_file_size_mb`, `ALPHA_CONTENT_TYPES`, `flatten_alpha_for`, `frames_for_encoding`, `output_file_path`, `_dedupe_existing_path`, `AUDIO_WRITE_CHUNK_FRAMES`, `write_audio`, `normalize_audio` (the waveform one), `as_audio_track`, `_as_stereo`, `read_embedded_metadata`, and `embed_image_metadata(image, path, content_type, metadata)` (was the method `Result._save_image_with_metadata`).
  - `AUDIO_WRITE_ARGUMENTS` stays in `result.py` beside `get_audio_write_arguments`.
  - PIL and piexif stay lazy.
- **Create `dw/output_extraction.py`:** `MODULAR_*_KEYS`, `_frames_from_attributes`, `_audios_from_attribute`, `OUTPUT_FIELD_EXTRACTORS`, `get_artifact_list`, `output_field_names`, `modular_artifacts`, `first_item`, `frames_with_audio`, `pair_audio_with_frames`, `as_waveform_array`.
- **Create `dw/audio_qc.py`:** `HEADROOM_WARN_DBFS`, `_peak_dbfs`, `warn_without_headroom`, `CLIPPED_WARN_DBFS`, `_UNPROBED`, `_probe_written_media` (its lazy `from .media import probe_media` verbatim), `warn_if_written_above_full_scale`, `NEAR_SILENT_*`, `warn_if_written_near_silent`.
- **Other moves:** `guess_extension` to `dw/content_types.py`, and `Selected` to `dw/media_types.py` (Decisions).
- **None of the new modules imports `result`, `step`, `workflow`, `tasks.*` or `pipeline_processors.*`.**
- **Repoint every importer:**
  - `chain.py:39` (it then no longer imports `result`), `routes/media.py:35`, `tasks/select.py`;
  - the tests: `test_result.py:21-27`, `:52`, `:1123`, `:1670-1703`, `:2036-2266`; `test_result_output_naming.py:1`; `test_concat_videos.py:15`; `test_gather.py:176`; `test_server.py:1460`, `:1475`, `:1504`, `:2661`;
  - `test_shots.py:83`'s key, which becomes `"dw/output_extraction.py"`;
  - the `emit_warning` monkeypatches at `test_result.py:1788` and `:1809`, which go to `dw.writers`.
- **Comments that name a moved home:** `step_cache.py:265`, `tasks/joins.py:466`, `media.py:689`, `routes/gallery.py:102`, `tasks/audio_utils.py:113`, `tests/test_modular_output_properties.py:6`.
- **Modules 154.** The snapshot diff is empty.

### Task 2: `save_artifact` cut

- **Cut `Result.save_artifact` to about 60-80 lines and complexity about 8, in the shape of survey §3:**
  - module-level `_refuse_scalar_artifact`;
  - `_save_mapping_artifact`;
  - `_write_artifact_file`, dispatching to:
    - `_write_video_file` (stays in `Result`; the encoder lookups stay here);
    - `_write_audio_file`;
    - `writers.write_json_file`;
    - `writers.write_text_file` (the `ValueError` text unchanged);
    - `_write_saveable`.
  - The post-write block becomes `audio_qc.check_written_media(output_path, artifact, content_type, *, video_fps, consumed_by_normalizer, headroom_warned, predicted_peak_dbfs)` with `remeasure_shots_after_mux`, `written_peak_already_warned` and `warn_held_prediction`.
- **Preserve exactly:**
  - every early return. The batched-audio recursion returns from inside the `try` (check `is not None`, not truthiness), so a nested failure is still logged twice;
  - the flag reset after `writing` and before the `try`;
  - `warn_without_headroom`'s short-circuit on `_consumed_by_normalizer`;
  - `artifact` rebinding only inside `_write_saveable`, which returns nothing;
  - every `video_fps` call;
  - the order in Review Focus 1.
- **Retarget in this commit:** the 2 `patch("dw.result.warn_if_written_above_full_scale")` to `dw.audio_qc`, and the `test_result.py:571` monkeypatch if the dict branch's lookup moved.
- The existing suite is the proof. `result.py` lands under 1,000 lines, about 850.

### Task 3: `plan.estimate` cut

- **Cut `estimate` to about 60 lines and complexity about 8,** per survey §5:
  - `_own_price`, which runs before the observed early return;
  - `_read_child`, keeping the `(ValueError, AttributeError)` catch;
  - `_child_observed`, keeping its precondition and its `except Exception`;
  - `_child_catalog_price`, using the `_list_entries` alias;
  - a `_ChildTotals` accumulator;
  - `_rolled_up_estimate`.
- Keep the docstring whole.
- `cached_minutes` is computed from the rounded minutes before tempering.
- **Drop the `unseeded_cache_warnings` re-export (`plan.py:37-38`).** `tests/test_plan.py` imports it from `dw.validation`.
- `tests/test_plan.py` is the proof. Sonnet.

### Task 4: `arguments.py`'s media half

- **First, the removals (tests unchanged):**
  - `_with_frame_rate`'s conditional becomes `shots = shots_beside(location)`, and the "so a URL carries none" sentence goes;
  - the redundant lazy `shots_beside` and `PIL.Image` imports go.
- **Create `dw/argument_media.py`:**
  - Contents: `is_media_reference`, `fetch_media`, `fetch_image`, `fetch_video`, `_describe_value_source`, `fetch_image_with_context` and `fetch_video_with_context` (public now, since `realize_args` calls them), `_with_frame_rate`, `_declared_fps`, `_fetch_remote_video`.
  - Lazy imports stay lazy.
- **Repoint:**
  - `tasks/gather.py:6`, `tasks/task.py:574`, `video_extensions.py:24`, and `introspection`'s lazy `is_media_reference`;
  - `video_extensions.py:24-26`'s `PROMPT_PREFIX`, which it re-imports through `arguments`: repoint it to `references.PROMPT`;
  - the tests;
  - the 7 string patch targets and the 2 `load_video` monkeypatches.
- **`_names_no_media` becomes `names_no_media`** (`validation.py:31` imports it).
- **Move `NON_TYPE_KEYS` to `dw/type_helpers.py`** (importers `introspection.py:835`, `tests/test_examples.py:19`, `tests/test_workflow_trust.py:494`).
- **Cut `realize_args` to complexity 15 or less**, per survey §1.6:
  - `_realize_explicit_reference(value, base_dir) -> (value, handled)`, shared by the dict and list branches. When only the path resolution fired, it returns the resolved value for the later conventions.
  - `_realize_type_reference`;
  - `_realize_nested`, with its `ValueError` text unchanged;
  - `_realize_list`, keeping the `(step 'name')` re-raise and the `OMITTED` filter.
- **Prefix spellings in the code that stays** (`:287`, `:666`): use `references.ref_name` / `is_ref`.
- **Size:** `arguments.py` under 1,000 (about 933 before the cut adds helper lines). If it crosses, take the `argument_objects` fallback (Decisions) and say so.

### Task 5: `introspection.py`'s two check groups

- **`_NAME_PATTERN` becomes `CLASS_NAME_PATTERN`.**
- **Create `dw/type_references.py`:**
  - Contents: the type-reference block (779-959) and `component_name_errors`.
  - Its lazy imports become top-level: `type_helpers`, `security`, `arguments`, `argument_media`, `for_each`. Each was checked to be cycle-free.
  - `_is_type_key` reads `NON_TYPE_KEYS` from `type_helpers`.
- **Create `dw/argument_warnings.py`:** `_resolved_value`, the five `_inert_*`, `workflow_argument_warnings`. The f-string at 1202 builds its reference with `references.make_ref`.
- **Repoint:** `validation.py:36`, `routes/library.py:19`, `test_component_type_errors`, `test_component_name_errors`, `test_introspection`, `test_task_discovery`, `test_task_signature_errors`, `test_security_trust_gate:611,702`.
- **Cut `_parse_docstring_args` under complexity 16:** `_args_block_start` and `_open_entry`.
- Convert the moved code's prefix spellings (`:644`, `:1051-1052`, `:1141-1142`).
- `introspection.py` lands at about 782.

### Task 6: `dw/trust.py`

- **Move the trust gate (76-344) into `dw/trust.py`,** which imports only `security`. `UntrustedWorkflowError` stays in `security.py`.
- **Repoint:** `type_helpers.py:5`, `pipeline_processors/pipeline.py`, `pipeline_processors/remote.py`, `locations.py:45`, `serve.py`, `validate.py`, `routes/system.py:27`, and the tests (`test_security_trust_gate`, `test_workflow_trust`, and every test importing `TRUST_WORKFLOWS_ENV_VAR`).
- **No validator moves, so no CodeQL file changes.** Check that with `grep` over `.github/codeql/` for each moved name, and record the result in the report.
- **Docs:**
  - CLAUDE.md:253: entry points use `dw/security.py`'s validators, and the trust gate is `dw/trust.py`;
  - `docs/SECURITY.md` where it names the trust gate's home;
  - `docs/SECURITY_QUICKREF.md:15`'s import block;
  - `.github/copilot-instructions.md:94`.

  Replace text, don't add it.
- `security.py` lands at about 771. Cheap model.

### Task 7: `pipeline.py` split

- **Create four modules under `dw/pipeline_processors/`:**
  - `placement.py` (about 370): the placement group;
  - `components.py` (about 800): the components group plus the denoiser cache hooks, with `_MISSING` beside both users;
  - `adapters.py` (about 170);
  - `progress.py` (about 240).
- **Imports:**
  - `components` → `placement` is the only edge between them. `pipeline.py` keeps `Pipeline`, the component-name helpers and the output checks, about 845 lines.
  - Lazy imports stay lazy: `apply_group_offloading`, peft, `ComponentsManager`, sdnq, `SequentialPipelineBlocks`.
- **Patch targets:**
  - The 6 string patch targets and the `load_component` monkeypatch do not move.
  - The `get_device_type` monkeypatch (`test_offload_placement.py:45-47`) retargets to `placement`.
- **`reported_blocks` and `reported_progress_bars` move unchanged,** including their `finally` restores.
- **Add the four new modules to `tests/test_configuration_schema.py`'s `SOURCES`** in the same commit. `offload` is read only by `placement.py` after the move, so the scan fails loudly at `:111` without this, and its per-key checks go partly vacuous.
- **Repoint the tests** (survey §7's importer list).
- **Docs:** the CLAUDE.md bullets naming `apply_on_demand_placement`, `place_component` and `active_loras` by module, and `kernel_availability.py:14,59`'s comment.

### Task 8: `PipelineOwnership`, and `create_step_action` cut

- **Create `dw/pipeline_ownership.py`:**
  - `PipelineOwnership` (survey §2c): `prior`, `running`, `keys_by_step`, `deferred` (a `_Deferred` dataclass), and the methods `begin`, `load_key`, `record`, `key_for`, `defer`, `mark_released` and `superseded_key`. The `RuntimeError` text of `_load_key` stays byte-for-byte.
  - Module functions: `finish_release`, `evict_superseded`, `reclaim_after_step`, `_allocated_mb`, `_release_host_caches`.
  - It does not import `workflow`.
- **Wire it into `Workflow`:**
  - `Workflow.__init__` creates one, and `run` replaces it with `PipelineOwnership(prior_step_keys)`. Keep `running=None` until `begin`, and `prior_step_keys or {}`.
  - The `hasattr`/`getattr` guards go.
  - `dw/worker.py:540` and its docstring read `workflow.pipeline_ownership.keys_by_step`, keeping a `getattr` chain for the bare `StubWorkflow`.
  - About 31 test lines retarget.
- **Retarget the 19 memory patch targets to `dw.pipeline_ownership` in this commit.** `steps + 1` still holds.
- **Cut `create_step_action` under 150 lines.** It stays a `Workflow` method with its signature, as a dispatcher. The branches are:
  - `_pipeline_action`;
  - `_reuse_resident`;
  - `_reference_action`;
  - `_sub_workflow_action`, which keeps its own resolution (Decisions);
  - the task branch.

  The trust gate stays before `loading`.
- **`argument_template`:** the child keeps handed arguments on `_handed_arguments`. The property returns them when set, else the definition's `argument_template`. The child's one intended copy is still `_owned_arguments`. Do not edit `workflow_schema.json`.
- Opus.

### Task 9: The validation block joins `dw/validation.py`

- **Create `dw/step_value_checks.py`** with `fps_errors` (and `FPS_KEY`), `null_media_errors` and `select_errors`, plus their helpers.
  - Repoint `test_result_fps`, `test_select_validation`, `test_null_media`, `test_rule_parity`, `test_validation`, `test_workflow`, and `test_reference_sets` if it pins a moved name.
  - `validation.py:20`'s docstring follows.
- **Move into `validation.py`** (Option B; the public methods on `Workflow` keep one-line bodies):
  - `workflow_errors(workflow, arguments, composing, context)` (the body of `validation_errors`, including the context build and the conflict `ValueError`);
  - `workflow_context(workflow, arguments, composing, ceiling_index)`;
  - `run_warning_check`;
  - `undeclared_variable_errors`;
  - the `sub_workflow_errors` and `sub_workflow_argument_warnings` bodies, through `open_sub_workflow`.
- **`admission.py:158-159` and `routes/library.py:282,298` keep calling the `Workflow` methods** (Decisions, Option B).
- **`ConstantError` moves to `dw/variables.py`** (`tests/test_prepare_pipeline.py:9`).
- **Retarget the 14 `patch.object(dw.workflow, "get_device_type" / "device_capacity_gb")` lines to `dw.validation`.** Confirm each still intercepts: `workflow.py` keeps importing both for `_prepare_definition`, so a stale patch would pass.
- **`author_index`:** sites 660 and 732 become `references.author_index(source_indices, index)`, and 714 goes.
- The f-string at `workflow.py:942` builds its reference with `references.make_ref`.
- **Size:** `validation.py` lands at about 815. The catalog snapshot is the proof.

### Task 10: `Workflow.run` cut into `dw/workflow_run.py`

- **Characterization test first, on the base:** Review Focus 2's weakref test. It must pass before and after. If it fails on the base, report it and keep the current ordering exactly; do not fix it under the freeze.
- **Create `dw/workflow_run.py`** with:
  - `RunRecord` (status, run_id, started_at, arguments, seed, realized_name, annotations). `_write_run_manifest` takes it, and its lazy `__version__` import becomes top-level;
  - a `StepLoop`;
  - the phases `prepare_run`, `open_run`, `begin_steps`, `run_step`, `wire_child`, `release_step_pipeline`, `save_step`, `record_step`;
  - `prepare_definition`, `cache_lookup`, the `cache_hits` loop;
  - `selected_field`, `_relative_shots`, `release_unreferenced_results`.

  It does not import `workflow`.
- **What stays on `Workflow`:**
  - `run` stays a method, about 90 lines and complexity about 8, in the shape of survey §4. `cache_hits` stays a one-line method.
  - The `Workflow` attributes others read are still written on the instance: `manifest`, `_run_dir`, `_run_version`, `_elided_steps`, `_cache_enabled_this_run`.
- **Preserve:**
  - **`RunRecord.arguments` is updated when `_owned_arguments` rebinds them,** not only a local.
  - **`started_at` and `run_id` are set twice,** and a pre-open failure writes the first values.
  - **The release ordering:** measure `before`, drop `step_action` and the popped pipeline in `run_step`'s frame, then `finish_release`.
  - **`_run_dir` is not reset at the top of `run`.** This is a known quirk, preserved under the freeze; do not fix it.
  - Every `except` clause and the `finally`.
- **Retarget the tests:**
  - `_prepare_definition`'s 6 test calls;
  - `workflow_module.realize_args` (`test_workflow_step_cache.py:152,165`);
  - `release_unreferenced_results` (`test_workflow.py:10`, `test_previous_results.py:14`).
- **Size:** `workflow.py` under 1,000. Expect about 950-990 once Task 8's dispatcher and branch methods are counted, so plan the first fallback (`workflow_output_subfolder` and `catalog_root_dir` to `dw/library.py`, about 38 lines) from the start. Take the second only if still over (Decisions).
- Opus.

### Task 11: The elision guard and one `previous_result:` walker

- **Elision:**
  - **Failing test first:** `_carry_release` with a kept step whose `pipeline` is not a dict raises `AttributeError` today.
  - Fix: return early unless the predecessor and both pipelines are dicts, and hand `step_pipeline_keys` only dict pipelines.
- **The walker:**
  - **Parity tests first, against the three old functions:** bare and prefixed `from_previous_result`, a `variable:` `from_` value, nested lists, and other keys beside `from_`. They must pass on the base.
  - Then add `references.iter_previous_result_references` (Decisions), with no import statement (`test_references.py:78-86`), and switch the three callers. The parity tests keep passing, now against the callers.
  - Convert the `startswith`/slice spellings in the code the switch touches: `previous_results.py:39,42,197,251,252,406,407` and `for_each.py:190-202,235,255,323,325` where they read `previous_result:`.
- The catalog snapshot is unchanged.

### Task 12: Stage 3e merge

- **Metrics:**
  - `modules` 165 (the list is in Decisions);
  - `modules_over_1000_lines` 6 → 0;
  - `functions_over_150_lines` 4 → 0;
  - `complex_functions` 12 → 7 or fewer;
  - re-baseline everything that fell.
- **Docs:** CLAUDE.md wherever it names a moved home:
  - the two audio QC bullets: `dw/result.py` → `dw/audio_qc.py`;
  - `Workflow.cache_hits` / `_prepare_definition` / `_cache_lookup`;
  - `release_unreferenced_results`;
  - `_name_fault` (stays in `security.py`; check);
  - the `pipeline.py` homes.

  Also: `docs/SECURITY.md:204-208`'s table rows, `.github/copilot-instructions.md:25,27`, `dw/server/CLAUDE.md` if it names a moved home, `docs/stabilization/ROADMAP.md`'s function table, and the comments listed in survey §11. Replace text, don't add it.
- **Release notes:** no user-visible change. Internally, the new homes. Add the carried items moved to Phase 4 or to after Phase 3 (Decisions) to the carried list.
- **Finish:** hot zone back to the standing entries. Merge `--no-ff` and push. Gate 3 follows.

### Hot zone (3e)

```
scripts/surface_snapshot.py
dw/result.py
dw/writers.py
dw/audio_qc.py
dw/output_extraction.py
dw/content_types.py
dw/media_types.py
dw/tasks/select.py
dw/pipeline_processors/chain.py
dw/plan.py
dw/arguments.py
dw/argument_media.py
dw/argument_objects.py
dw/type_helpers.py
dw/tasks/gather.py
dw/tasks/task.py
dw/video_extensions.py
dw/introspection.py
dw/type_references.py
dw/argument_warnings.py
dw/security.py
dw/trust.py
dw/locations.py
dw/serve.py
dw/validate.py
dw/pipeline_processors/pipeline.py
dw/pipeline_processors/placement.py
dw/pipeline_processors/components.py
dw/pipeline_processors/adapters.py
dw/pipeline_processors/progress.py
dw/pipeline_processors/remote.py
dw/kernel_availability.py
dw/workflow.py
dw/workflow_run.py
dw/pipeline_ownership.py
dw/validation.py
dw/step_value_checks.py
dw/variables.py
dw/references.py
dw/elision.py
dw/step_cache.py
dw/previous_results.py
dw/for_each.py
dw/library.py
dw/step.py
dw/worker.py
dw/server/routes/media.py
dw/server/routes/library.py
dw/server/routes/system.py
```

The test files follow their modules. `dw/library.py` and `dw/argument_objects.py` are listed only for the sanctioned fallbacks.
