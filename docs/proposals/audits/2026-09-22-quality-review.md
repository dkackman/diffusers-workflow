# diffusers-workflow — independent quality review

**Reviewed:** `develop` @ `09ce397` (2026-09-22 18:22 CDT), a fresh clone. I also read the GitHub issues (#17–#367: 280 issues, 87 PRs) and the full git history (1,169 commits).
**Reviewer stance:** read-only. I did not call the live server, did not ssh, and did not run the test suite (it needs torch and diffusers-from-git). All paths below are relative to the repo root.

---

## 1. Executive summary

**Verdict.** The project is ambitious, and at the unit level it is carefully engineered. Many things are done properly:

- containment and trust gating
- a bound cost acknowledgement
- a graceful worker-crash path
- about 3,500 tests
- a real attempt to keep numbers in the docs honest by pinning them to diffusers symbols

The growth since the web UI (2026-08-29) and the MCP server (2026-09-01) has still outpaced the architecture:

- Engine, server and MCP code tripled in 3.5 weeks (13.9k → 42.9k LOC).
- The main server module grew **10×** (`dw/server/app.py` 431 → 4,226 lines).
- The share of commits that are fixes climbed every week:

  | Weeks | Fix share |
  |---|---|
  | Aug 29–Sep 11 | 17–18% |
  | Sep 12–16 | 39% |
  | Sep 17–22 | 57% |

The issue stream is not converging. The loop closes issues quickly (median 2.75 h from filing to verified close), but inflow keeps pace: 50 and 49 new issues on the last two days, and the open backlog is at its highest yet (34).

Most of that inflow belongs to a few **bug classes produced by the structure**, not to one-off defects:

- validation that is a hand-kept copy of execution
- an audio subsystem with no single value model
- a cost estimator that grows one special case per issue
- copy-pasted templates

Each fix closes one parameter combination, and the next session finds the neighbouring one. The code is decent. The main risk is that the process is very good at producing point fixes, which lets the project put off the few structural changes that would stop the flow.

**Top five recommendations, ranked by leverage:**

1. **Make validation and execution share one preparation pipeline** (substitute → constrain → expand → elide → realize). Put each static check next to the task, pipeline or writer it guards.
   - Today there are 18 validator passes in `Workflow.validation_errors` (`dw/workflow.py:603-720`), plus 2 more error sources and 9 warning sources inside the HTTP route (`dw/server/app.py:1658-1876`).
   - They check a separately prepared definition (`expanded_definition`, `workflow.py:439`, a hand copy of `_prepare_definition`, `workflow.py:798`).
   - `JobManager.submit` runs a **different, weaker** check that ignores the caller's arguments (`dw/server/jobs.py:818,831` → `loaded.validate()`).
   - This is the root of the largest code-bug class ("validates clean, dies at run time"): 50 of 272 loop-era issue titles mention `validate`.
2. **Fix the SSRF and `from_file` holes, and centralise containment** (security, cheap, urgent).
   - `requests.get` follows redirects after a one-time host check, and DNS is resolved twice (`dw/tasks/audio_utils.py:361`, `dw/tasks/video_utils.py:443`, diffusers `load_image` via `dw/arguments.py:943`).
   - Run-time `from_file` checks only the scheme and an unrooted path (`dw/arguments.py:848-876` → `security.validate_url`).
   - There are at least five separate "is inside root" predicates.
3. **Give audio one value type and one meaning for `sample_rate`, at the dispatch boundary.**
   - Today `sample_rate` means *relabel* in some commands, *resample* in others, and *silently relabel* in `pair_audio` (`dw/tasks/pair_audio.py:171-173`).
   - Clipping is warned about after the fact instead of being prevented.
   - About 75 of 272 issues are audio, level or mix issues.
   - Separately, decide whether a mastering toolkit (compressor, gate, EQ, spectral analysis in `audio_utils.py`) belongs in this engine at all, rather than in ffmpeg filters.
4. **Put the loop's output under CI, and add a durable job ledger.**
   - CI runs on push to `master` and on PRs only (`.github/workflows/ci.yml:3-8`). The loop merges to `develop` and deploys `develop` to lem, so nothing the agents ship is CI-checked until a release PR. `develop` → `master` PRs failed CI 6 times on 09-18/19, and one failing merge landed on `master` (#241).
   - Jobs are written to SQLite only when they finish, so a restart erases queued and running jobs (#300; `jobs.py:757-760, 1158-1163`).
5. **Stop growing the cost estimator by special case; redesign it as one fitted model with priors.**
   - 15+ issues in 10 days touched it (#91 → #93 → #154 → #242 → #252 → #255 → #267 → #268 → #275 → #301 → #312 → #315 → #319 → #330 → #341).
   - It now has six `basis` values, plus tempered, low_confidence, partial and unpriced overlays, plus parent/child roll-up rules (`dw/plan.py:282-402`).
   - It duplicates driver logic from `dw/server/observed_cost.py`, and the two copies already diverge.

   While you are at it, split `create_app` (a single 3,524-line closure) into routers.

---

## 2. Quality assessment by area

| Area | Grade | One-line reason |
|---|---|---|
| Core engine (pipeline loading, chains, for_each, step cache) | B | Capable and heavily featured. Long functions (`Workflow.run` 490 lines, `workflow.py:1010`); model-specific names leak into generic code. |
| Workflow JSON language | C+ | Powerful, but at least 11 string prefixes resolved in different phases, key-name magic, and variable types inferred from defaults. Sharp edges show up as issues (#338, #363, #365). |
| Static validation | C | 20+ hand-maintained validator passes that copy run-time logic. Submit runs a weaker check. Several validators fail open (`except Exception` → `[]`). |
| Cost/memory planning | C | Accreting heuristics, a duplicated driver logic, and a stream of follow-up issues. Well unit-tested, but the model itself is the problem. |
| HTTP server | C+ | Good per-request workspace handling and containment intent. One 3.5k-line closure; no response models; 400-for-everything with `str(e)`; small concurrency hazards. |
| Job manager | C | Clean crash handling, but in-memory queue and running jobs, no hang timeout, ad-hoc SQLite migrations, and the thinnest tests in the codebase (test/code 0.2). |
| MCP surface | B- | Mostly thin over HTTP, which is good. 58 tools and about 14k tokens of prose kept under a ratcheting budget test; tool prose kept in 3 places; parameter aliases instead of one name (#179); mounted mode shares one workspace across sessions (#298). |
| Tasks: audio | C- | No shared audio model; `sample_rate` semantics vary by command; a mastering stack with pure-Python per-sample loops on the single GPU worker. |
| Tasks: image/video | B- | Main task metadata comes from signatures (good), with three hand-kept side tables that drift (#185, #350, #366). Image processors test/code 0.2. |
| Security | B- (design) / C (current) | Serious and thoughtful, with the trust gate and path policies. Containment is not centralised, and there are two URL validators of different strength with a redirect bypass. |
| Tests | B- | Large and fast. Real behaviour tests for DSP, heavily mocked elsewhere. 17% of asserts pin wording; only 2 real generation tests; no property-based tests; not run on `develop`. |
| UI | B- | Reasonable size (20k LOC, 40 vitest files, 6 Playwright specs). Hand-written `types.ts` against untyped server dicts; e2e not in CI. |
| Docs | C+ | Plentiful and mostly accurate, but sprawling (14k lines), mixed with plans and proposals, and restated in code comments and tool docstrings. 395 inline `#NNN` references in 56 of 115 Python files. |
| Catalog/templates | C+ | Useful, but copy-paste siblings (`assemble-and-score` vs `dissolve-between-shots` are 95% identical), so fixes miss siblings. Descriptions are 2–4 KB of prose acting as agent instructions. |

---

## 3. Weaknesses, ranked by leverage

### W1. Validation is a parallel re-implementation of execution (redesign)

**What.** Static validation has three layers:

1. **Engine passes.** `Workflow.validation_errors` (`dw/workflow.py:603-720`) runs the JSON schema, then concatenates 18 error passes from 15 modules: previous_results, subfolders, reference_names, content_types, scalar_result, locations, reference_limits, adapter_compatibility, task_domains, select_validation, introspection ×2, variable_constraints ×2, vram_estimate, kernel_availability, sub_workflow and undeclared variables.
2. **Server-only passes.** The HTTP route adds `argument_errors`, a 107-line `_argument_reference_errors` closure (`app.py:1531`, which the CLI and REPL never run), 9 warning sources and 4 `_inert_*` helpers.
3. **A separate definition.** All of this runs on a definition prepared by `expanded_definition` (`workflow.py:439`), a hand copy of the run path's `_prepare_definition` (`workflow.py:798`). The copy skips `apply_constraints`, constraint-reference resolution, `realize_args` and elision.

**Evidence.**

- Nearly every validator's comment names the incident that created it: #89, #96, #136, #139, #140, #141, #155, #162, #168, #178, #212, #265, #345.
- The rules are duplicated, not shared. `select_validation.py:15` hard-codes `_RULES = {"argmax", …}` in parallel with `tasks/select.py:82-106`.
- `_UNRESOLVED_PREFIXES` is redefined 6 times in 3 different variants (`subfolders.py:30`, `content_types.py:29`, `kernel_availability.py:35`, `reference_names.py:34`, `reference_limits.py:48`, `adapter_compatibility.py:57`).
- `JobManager.submit` (`jobs.py:818,831`) calls `loaded.validate()`, which is `validation_errors()` **with no arguments**. So a `run_workflow` that skipped `validate_workflow` is queued without argument-aware checks.
- Validators fail open: `workflow.py:571` and `:733`, and the plan at `app.py:1858`.
- The issue record: #89, #96, #118, #123, #136, #141, #145, #155, #166, #168, #208, #213, #285, #287, #293, and today's #345, #347, #364, #365.

**Why it matters.** Every new task parameter, pipeline or template creates an unguarded gap until someone runs into it. The agent loop is good at finding these, so they turn into a steady flow of issues. Fail-open validators make this worse: a bug in the validator hides a bug in the workflow.

**What I'd do (redesign).**

1. Write one `prepare(definition, arguments) -> PreparedWorkflow` used by `run`, `validate` and `plan`.
2. Replace per-incident modules with a `static_check` hook declared next to each implementation. `register_command` already does this for argument signatures; extend it.
3. Make `submit` run the same argument-aware validation that `validate_workflow` runs.
4. Make validator crashes errors, not silence.
5. Consider validating the schema *after* substitution. Today 55 typed schema fields reject `variable:` and only 16 accept it, which is the whole #363 class.

### W2. Security: SSRF redirect and rebinding bypass, `from_file` policy gap, containment in five places (fix now, then refactor)

**Evidence.**

- **Redirect bypass.** `validate_media_url` (`dw/locations.py:287-321`) resolves the host and refuses internal addresses. The fetches that follow are plain `requests.get(validated_url, timeout=…)` (`audio_utils.py:361`, `video_utils.py:443`) and diffusers `load_image` (`arguments.py:943`, `tasks/gather.py:64`).
  - Redirects are followed by default, so a public URL that 302s to `169.254.169.254` or loopback gets through.
  - The host is resolved twice, so DNS rebinding works.
  - Responses have no size cap.
- **`from_file` at run time.** It goes through `validate_media_location` (`arguments.py:848-876`), which calls `security.validate_url` (scheme only, `security.py:327-351`) and `validate_path` with no roots.
  - The static pass does check literal `from_file` values (`locations.py:403-450`).
  - A value that arrives by variable override most likely reaches the loader under the weaker policy. That contradicts the module's own promise that "the loaders call the same functions at run time" (`locations.py:30-35`).
- **Containment predicates.** At least five: `security.validate_path` (prefix after realpath), `locations._within`, `workspace._is_within` (commonpath), `dw_mcp/media._confine`, and `dw_mcp/assets.py:72-100`. That is why #113, #114, #124 and #138 each needed a separate fix.
- **Error text.** `POST /api/jobs` maps *every* exception to 400 with `str(e)` (`app.py:1157-1160`; also `:1275`, `:1294`, `:2086`, `:2467`), which can leak absolute server paths despite the #247 and #310 hardening.

**What I'd do.**

1. One fetch helper for all remote media:
   - `allow_redirects=False`, or re-validate the host on every hop
   - pin the resolved IP for the connection
   - a byte cap
2. One `contain(path, roots)` function.
3. One URL validator. Delete `security.validate_url`'s direct uses (`plan.py:830`, `arguments.py:871`).
4. Map internal exceptions to 500 with a generic message.
5. Add each case to the security regression suite.

### W3. The audio subsystem has no single model, and has grown into a mastering suite (rethink scope, refactor core)

**Evidence.**

- `AudioTrack` exists only as a return type (`_as_track`, `audio_utils.py:378`). Inputs are `str | AudioVideo | ndarray`, normalised separately per command (`_waveform_and_rate` `:1344`, `_load_tracks_matching_rate` `:837`) and again in `pair_audio`, `concat_videos` and `dissolve_videos`.
- `sample_rate` means different things depending on the command:

  | Where | What `sample_rate` does |
  |---|---|
  | Single-track commands (`audio_utils.py:1355-1383`) | Relabels, with a warning |
  | `concat_videos` (`concat_videos.py:146-174`) | Resample target |
  | Mix/crossfade when pinned (`:858`) | Relabels every track |
  | `pair_audio` (`pair_audio.py:171-173`) | Relabels silently: the #180 bug class, still present |

- The issue chains show the same thing: #108 → #180 → #196 → #205 → #287 → #293 for sample rate. For headroom: #158 → #159 → #161 → #174 → #194 → #286 → #295 → #305 → #306 → #323 → #362, and #362 is still open today ("still clips after mp3 encode").
- Levels are warned about after encode (`result.py:116-160`, `:190-235`) rather than enforced at the write boundary.
- `audio_utils.py` grew 266 → 1,710 lines since 08-29 and has had 36 commits since 09-01. It now contains:
  - a compressor, limiter and gate with a pure-Python per-sample envelope loop (`:1494-1513`, about 8M iterations for a 3-minute track, run on the single FIFO GPU worker)
  - biquad EQ that depends on scipy only transitively, via controlnet-aux (`:1601-1631`)
  - spectral flatness and harmonicity analysis

**Why it matters.** About 75 of 272 loop-era issues fall in the audio/level/mix area, the largest category. The pattern is always "silently wrong, add a warning", which fixes one combination per issue.

**What I'd do.**

1. Coerce every audio input to one `AudioTrack(waveform float32 [C,N], rate, source)` at task dispatch.
2. Make `sample_rate` always mean "resample to"; the relabel escape hatch gets its own name.
3. Enforce a true-peak ceiling at the single write boundary. Measure *after* encode and correct, rather than warning.
4. Push DSP (EQ, dynamics, loudness, LUFS per #361) down to ffmpeg filter graphs through PyAV, which is already a dependency.
5. Before adding more of it (#349 CPU grade, #361 LUFS), decide deliberately how much post-production this project should own.

### W4. The loop's quality gate is the implementer's local pytest; `develop` isn't CI-checked, and "verified" is repro-shaped (process, high leverage)

**Evidence.**

- `ci.yml:3-8` triggers on `push: [master]` and on `pull_request` only. The loop merges straight to `develop` and deploys it.
- CI history: 6 failed `develop` → `master` PR runs on 09-18/19, and a failed push to `master` on 09-19 (#241).
- Playwright e2e is not in CI.
- Only 2 tests run a real generation, gated on an accelerator (`tests/test_worker.py:37-40`).
- #197 is the cautionary tale:
  1. The first fix called `_fit_audio_to_frames` on a CUDA tensor and broke **every** in-memory LTX-2 audio+video save.
  2. The tester caught that and bounced it.
  3. The second fix passed verification.
  4. The next day the original repro came back: the fit had been applied to a local variable, not to the artifact the concat reads.
  5. A fourth hand-off closed it.

  The tester verifies the filed repro. It does not verify the invariant.
- Bounce rate is low (11 of 184 handed-off issues bounced, 6%), and only 4 issues were formally reopened. But 63 issues cite an earlier issue in their title, 55 of them a `status:verified` one. By my hand classification, about 30 of those are "the fix was incomplete or missed a sibling" and about 20 are "the fix broke a suite expectation".

**What I'd do.**

1. Run CI on push to `develop`, and gate lem deploys on it: the deploy script checks the commit's status.
2. Add a nightly GPU smoke on lem: one real generation per family, driven by pytest rather than an agent.
3. For "silently wrong" classes, add property or fuzz tests over parameter combinations (Hypothesis on the audio tasks and `validate` ≡ `run` agreement). Point tests are the only kind the suite has today.
4. Ask the tester prompt to verify *the stated root-cause invariant* plus one sibling, not only the repro.

### W5. The cost/memory estimator grows by special case (redesign)

**Evidence.**

- `estimate` (`plan.py:402`) is 189 lines.
- Six `basis` values (`plan.py:282-287`), plus these rules and overlays:
  - tempered and low_confidence blending (#301, #319)
  - partial and unpriced (#242, #252)
  - parent roll-up with minimum runs (#268, #275)
  - a shifted driver forces unknown, at parent level (#267) and child level (#341)
  - child-observed only when the parent is unpriced (#315)
- `_declared_drivers` and `_driver_comparable` (`plan.py:354-377`) are self-described copies of `observed_cost.py:57/95`, and they already diverge: one buckets a list by length, the other JSON-dumps it.
- The host-memory projection has its own chain: #243 → #254 → #264 → #272 → #274 → #334 → #348. #348 is open: a shared model load is scaled linearly with list length.
- The regression suites needed at least 8 wording changes just to track estimator changes (#172, #276, #304, #307, #312, #320, #327, #330, #331).

**Why it matters.** This is the most-churned feature in the loop era, and each change breaks suite expectations downstream. The loop spends a lot of its budget here for a figure that is inherently approximate.

**What I'd do.**

1. Replace it with one explicit model: `minutes = f(drivers)`, fitted per workflow and device from history, with curated values as a Bayesian prior and a confidence interval.
2. Report `{estimate, low, high, n}` and drop the basis taxonomy.
3. Memory: model `fixed + per_entry × n`, not `per_entry × n`.
4. Put driver comparison in one module that both the engine and the server import.

### W6. `dw/server/app.py` is one 3,524-line closure; responses are untyped (refactor)

**Evidence.**

- `create_app` spans `app.py:703-4226`: 64 routes, about 25 helpers and 3 middlewares, all closures. There is no `APIRouter` and 0 `response_model`s.
- Business logic lives in route bodies:
  - validation orchestration (`validate_workflow` 219 lines, `:1658-1876`)
  - cost acknowledgement (`:985-1045`)
  - gallery indexing and orphan scanning (`:2678-2830`)
  - zip building (`:3275-3360`)
  - frame tiling and budgeting (`gallery_frames` 139 lines)
- A run request validates three times: the route, then `submit`, then the worker.
- Concurrency hazards:
  - `_workflow_detail_cache` and `_prompt_detail_cache` are module-global dicts mutated from threadpool routes without a lock (`:212`, `:226-236`, `:527`).
  - Workspace delete checks for queued jobs under `manager._lock`, releases it, then deletes (`:1973-1991`), which is a check-then-act race.
  - Chunked uploads bypass the size precheck and are read into memory whole (`:3517-3530`).
- The UI's `types.ts` (363 lines) and `api.ts` (673 lines) are hand-written against these untyped dicts.

**What I'd do.**

1. Split into routers (jobs, catalog, validate, workspaces, gallery, assets, models, system) and service modules the routers call.
2. Add pydantic response models, and generate the UI's TypeScript types from OpenAPI.
3. A mechanical refactor, well covered by the 8k lines of server tests. It is a good job for the implementer agent *if* it is scoped as "no behaviour change".

### W7. Job durability and worker hangs (inspect, then fix; small)

**Evidence.**

- The queue is only in memory (`jobs.py:757-760`). History rows are written on terminal state only (`:81-88`, `:1180-1186`).
- `shutdown()` (`:1158-1163`) never marks pending or running jobs, so #300 covers the running job too, not just queued ones.
- There is no timeout on a hung worker (`repl_worker.py:115-120`), and cancel is a message a hung worker can't read.
- The step cache lives only in memory, per process (`step_cache.py:39`; #244, #281).
- SQLite migrations are ten try/except `ALTER TABLE`s with no schema version (`jobs.py:94-164`).
- `jobs.py` has test/code 0.2, the lowest of any major module.

**What I'd do.**

1. Insert the job row at submit.
2. On startup, mark leftover `queued`/`running` rows as `interrupted`.
3. Add a watchdog: no event for N minutes while a phase is expected to report → kill and restart the worker, and fail the job.
4. Add a `user_version`-based migration.
5. Write the job-lifecycle tests.

### W8. The workflow language's sharp edges (rethink, incrementally)

**Evidence.**

- At least 11 string prefixes resolved in different phases: `variable:`, `constant:`, `item:`, `gather:`, `asset:`, `output:`, `prompt:`, `previous_result:`, `builtin:`, `constraint:`, `frame:`.
- On top of those, dict forms (`from_file`, `from_previous_result`, `{media_type, location}`) and key-name magic:
  - `_image`/`_video` suffixes auto-load media
  - `_type`/`_dtype`/`dtype` import Python objects, with exceptions `content_type` and `offload_type` (`arguments.py:26,136-171`)
  - a `{}` escape
  - an `EscapedString` class that exists only so a second realize pass doesn't undo the first (`arguments.py:29`)
- Because `realize_args` also runs over the variables block (`workflow.py:846`), a *variable's* name triggers media loading (#365).
- Variable types come from `type(default)` (`variables.py:277`), which produces #338 (silent int truncation) and #364 (null default).
- `null` means "drop the key" except under `MEDIA_SOURCE_KEYS` (`variables.py:98-112`).

**Why it matters.** Agents author this language. Every implicit rule is a trap that turns into an issue, then a warning, then docs prose, then a suite case.

**What I'd do.**

1. Add declared variable types: `{"type": "float", "default": 1, "enum": […]}`, with the inferred form kept as legacy.
2. Make type import and media loading explicit, e.g. `{"$type": "torch.bfloat16"}` and `{"$media": "asset:x.png"}`, instead of key-name inference. Deprecate the suffix magic behind a schema version.
3. Resolve all references in one documented phase order.

### W9. Model-family knowledge in generic engine code (refactor)

**Evidence.**

- `dw/adapter_compatibility.py` is entirely MiniMax-H3 (`H3_WORKFLOWS = {"ref2va","t2va","fl2va"}`, `:39-47`) and runs on every validation (`workflow.py:672`).
- `pipeline.py:1985` hard-codes `("transformer","transformer_ref")`.
- `prompt_weighting.py:293-315` checks Flux by name, and `teacache.py:291` keys on `FluxTransformer2DModel`.
- `workflow_schema.json:361,367` quotes H3 numbers.

**Why it matters.** The project's own rule (root `CLAUDE.md`) is "model knowledge lives [in skills and the catalog], never in engine code". Each new family will add more of these.

**What I'd do.** Add a per-family plugin hook (validators and component-partition hints) registered from a `families/` package, so the engine core stays generic.

### W10. Template duplication and prose-as-configuration (refactor)

**Evidence.**

- `workflows/templates/assemble-and-score.json` and `dissolve-between-shots.json` have identical step lists and are 95% line-similar.
- The sibling-missed-fix issues are the direct result: #129, #196, #205, #215, #227, #302, and #339/#342 open now for both.
- Template `description` fields reach 4 KB (`minimax/dialogue-short.json` 4,068 chars; 43 KB across 83 files). The five skills add 48 KB, and MCP tool descriptions add about 36 KB.
- The same guidance is restated in `docs/*.md`, tool docstrings, handler docstrings, template descriptions and SKILL.md. The MCP prose already drifts: the `run_workflow` handler says to poll `get_job_events` while the server instructions say to use `wait_for_job`.
- `tests/test_mcp_server.py:1450` keeps `SURFACE_BUDGET = 13_890` with 7 tokens of headroom under about 100 lines of changelog comments. Every docstring edit becomes a trimming exercise.

**What I'd do.**

1. Compose shared assembly through `builtin:` sub-workflows (the engine already supports them) instead of copying.
2. Make one source for each piece of guidance (the guides served by `get_guide`), and have tool docstrings point to it.
3. Replace the character budget with a real token count and per-tool caps, so the budget stops acting as a ratchet.

### W11. Comment and docstring archaeology (hygiene)

**Evidence.**

- 395 inline `#NNN` references across 56 of 115 Python files (app.py 47, result.py 35, workflow.py 34, audio_utils.py 33, plan.py 33), plus 293 in tests.
- app.py is about 30% comments and docstrings. In `validation_errors`, comments outnumber code.
- Docstrings tell incident stories ("reported succeeded with no warnings (#139)…") instead of stating the invariant.

**Why it matters.** This is the fingerprint of a fix-per-issue process. It makes the code harder to read and hides the actual contracts. The history belongs in git and the issues; comments should state *what must stay true*.

**What I'd do.** Leave existing comments alone, but change the implementer prompt's convention: a comment states the invariant, and the commit message carries the history.

### W12. Tests: numerous, but skewed (inspect)

**Evidence.**

- 3,522 tests and 55k LOC, about 1.29 lines of test per line of code.
- 716 mock/patch uses across 70 files.
- About 17% of asserts (984 of 5,787) are `"literal" in message` checks, plus 272 `match=` checks.
- Hard token budgets are pinned (`test_catalog_structure.py:402-451`, `test_mcp_server.py:1450`).
- Test/code ratio for the weakest modules:

  | Module | Test/code |
  |---|---|
  | `server/jobs.py` | 0.2 |
  | `image_utils.py` | 0.2 |
  | `pipeline.py` | 0.7 |
  | `workflow.py` | 0.7 |

- The DSP tests are real behaviour tests on synthetic waveforms, which is good.

**What I'd do.** Move wording pins to error *codes* (structured `{code, path, message}` errors are also better for agents). Add property tests (W4). Test the job lifecycle. Add a small, real-codec end-to-end encode test (PyAV on CPU is cheap).

---

## 4. Quality trajectory

### 4.1 Where the inflection is

- **Web UI and HTTP server:** first commit `d50ddce`, 2026-08-29, "feat(server,ui): HTTP server, introspection service, and Svelte SPA" (31 files, +3,314).
- **MCP:** design and spec on 2026-09-01 (`5459435`, `9e9df55`); package moved out of `dw` in `7d64a88` the same day.
- **Plugin skills:** 2026-09-08 (`4ace771`).
- **Agent loop on GitHub Issues:** from 2026-09-12 (issues #69+).

Before 2026-08 the repo had about 260 commits over 21 months. **August 1–28 was already a big engine push** (dw 7.3k → 13.9k LOC, tests 4.2k → 14.0k), so "before" is not a quiet baseline.

### 4.2 Size snapshots

I took the last `develop` commit before each date and counted lines with `git cat-file`. `dw` excludes `dw/community_pipelines` and `dw/workflows`.

| Date | dw | dw_mcp | ui/src (non-test) | ui tests | tests/ LOC | test fns | docs & *.md lines |
|---|---|---|---|---|---|---|---|
| 2026-08-01 | 7,264 | 0 | 0 | 0 | 4,201 | 236 | 4,008 |
| 2026-08-29 | 13,913 | 0 | 2,223 | 68 | 13,954 | 991 | 5,539 |
| 2026-09-05 | 18,953 | 1,478 | 10,240 | 1,639 | 24,385 | 1,627 | 19,440 |
| 2026-09-12 | 26,684 | 2,613 | 12,328 | 2,647 | 37,614 | 2,456 | 40,722 |
| 2026-09-18 | 35,162 | 3,484 | 13,879 | 4,285 | 49,640 | 3,226 | 49,403 |
| 2026-09-22 (HEAD) | 38,791 | 4,133 | 14,999 | 5,283 | 55,148 | 3,519 | 15,829 |

- **Code (dw + dw_mcp) grew 3.1×** in 24 days; Python tests grew 4.0×. The test:code LOC ratio rose from 1.00 to 1.29, and the UI went from 0.03 to 0.35.
- **Test volume kept up with code.** Test *effectiveness* did not keep up with the bug classes (W4, W12).
- **Docs** peaked around 49k lines on 09-18, including plans and proposals. They were cut to about 16k on 09-20 (`c5d4ac5` "cleanup docs"), which was a healthy correction.

Hotspot file growth, in lines:

| File | 08-29 | 09-05 | 09-12 | 09-17 | 09-22 |
|---|---|---|---|---|---|
| dw/server/app.py | 431 | 1,476 | 2,860 | 3,655 | **4,226** |
| dw/workflow.py | 507 | 733 | 1,209 | 1,705 | 1,820 |
| dw/pipeline_processors/pipeline.py | 1,678 | 1,797 | 2,007 | 2,289 | 2,293 |
| dw/result.py | 850 | 907 | 972 | 1,290 | 1,585 |
| dw/server/jobs.py | 296 | 700 | 1,174 | 1,360 | 1,551 |
| dw_mcp/server.py | — | 439 | 906 | 1,180 | 1,340 |
| dw/tasks/audio_utils.py | 266 | 579 | 804 | 1,463 | **1,710** |
| dw/plan.py | — | — | — | 587 | 889 |

`pipeline.py` has stabilised (+4 lines in the last week). Growth has moved to the **server, result writer, audio and planner**, which are the surfaces the agents drive.

**Most-touched files since 08-29:**

| File | Commits touching it |
|---|---|
| app.py | 122 |
| test_server.py | 102 |
| dw_mcp/server.py | 87 |
| docs/MCP.md | 79 |
| docs/SERVER.md | 72 |
| workflow.py | 68 |
| root CLAUDE.md | 67 |
| jobs.py | 53 |

Documentation files are among the hottest, because every surface change is restated in 2–4 prose locations.

### 4.3 Commit mix

Non-merge commits on `develop`, bucketed by committer date. "Fix" is a case-insensitive `^fix`, `^correct` or `^repair` subject match.

| Window | Commits | Fix | Fix % | feat/add | Code +/− | Test lines + | test/code added |
|---|---|---|---|---|---|---|---|
| ≤ 2026-07 | 253 | 12 | 4% | 31 | +7,320 / −2,613 | 4,093 | 0.56 |
| Aug 1–28 | 64 | 3 | 4% | 9 | +6,105 / −1,326 | 8,946 | 1.47 |
| Aug 29–Sep 4 | 180 | 32 | 17% | 67 | +21,806 / −3,972 | 12,934 | 0.59 |
| Sep 5–11 | 92 | 17 | 18% | 19 | +13,786 / −1,954 | 15,471 | 1.12 |
| Sep 12–16 | 189 | 75 | 39% | 48 | +12,094 / −1,989 | 13,850 | 1.15 |
| Sep 17–22 | 211 | 122 | **57%** | 34 | +11,314 / −1,927 | 11,589 | 1.02 |

Caveats:

- The loop names every issue-driven change `fix(...)`, including some enhancements, so the jump at 09-12 partly reflects that labelling convention.
- Even so, 57% fixes on a stable ~11–12k lines of code added per window means **new surface is still being added at a steady rate while the fix share climbs**. That is not a stabilising codebase.
- Deletions stay at 15–18% of additions throughout, so little is being consolidated.
- 484 of 665 commits since 08-29 carry a Claude co-author trailer.

### 4.4 Issue flow (loop era, #69+)

| Day | Created | Closed | Open at end of day |
|---|---|---|---|
| 09-12 | 18 | 16 | 2 |
| 09-13 | 39 | 32 | 9 |
| 09-14 | 31 | 34 | 6 |
| 09-16 | 13 | 14 | 4 |
| 09-17 | 24 | 9 | 19 |
| 09-18 | 23 | 35 | 7 |
| 09-19 | 16 | 11 | 12 |
| 09-20 | 9 | 13 | 8 |
| 09-21 | 50 | 37 | 21 |
| 09-22 | 49 | 36 | **34** |

- Median time from filing to a completed close is **2.75 h** (p90 16 h). The process is fast.
- There is no decline in inflow. Some of the last two days' spike is the external findings ledger (#338–#367) and a heavier regression cadence.
- Closed outcomes for #69+: 216 completed (176 with `status:verified`), 22 not planned (18 `wontfix`, 2 `duplicate`), 34 open.

**Rough categories by title keyword** (272 loop-era issues, first match wins; approximate):

| Category | Issues |
|---|---|
| Audio / level / mix | 75 |
| Regression-suite drift or process | 66 |
| Cost/memory estimation | 34 |
| MCP/API surface | 29 |
| Templates/catalog/skills | 25 |
| Validation gap (not caught above) | 17 |
| Other | 24 |

Separately, 50 titles mention `validate`, and 12 issues carry the `security` label.

**Process noise versus code signal.** About a quarter of the stream (the 66 drift/process issues) is the loop maintaining its own regression suites: expectations made stale by the loop's own fixes, and approval requests to edit them (for example #172, #173, #260, #263, #276, #304, #307, #320–#323, #327–#331). That is real cost, but it is not a code defect. The `regression` label (55 issues) mostly means "found by the regression agent", not "a verified fix regressed". Most real code regressions came from dependency drift:

- #169 and #232: transformers 5.17 broke all TTS
- #133: safety-checker black images

A handful came from fix-on-fix: #197, #150 (after #98), #161 (after #159), #330 (after #312), #334 (after #272).

### 4.5 Regressions and fix-on-fix

- **Hand-offs to verify** (timeline `labeled status:fixed-pending-verify`): 184 issues had at least one. 173 passed first time, 10 needed 2 hand-offs, and #197 needed 4.
- **Formally reopened:** 4 (#107, #184, #197, #214).
- **Follow-on issues:** 63 issues cite an earlier issue number in their *title*; 55 of those cite a `status:verified` one. By my hand classification, about 30 are incomplete fixes or missed siblings, and about 20 are suite drift caused by a fix. Examples of incomplete fixes:
  - #123 after #118, #129 after #126, #138 after #113, #145 after #96, #150 after #98
  - #161 after #159, #196 and #205 after #180, #227 after #199, #254 after #243
  - #275 after #268, #287/#293 after #108, #290/#291 after #288, #302 after #235
  - #319 after #301, #334 after #272, #341 after #267, #342 after #246, #345 after #285
- **Fix commits per issue:** 152 issues were referenced by fix commits since 09-12. 23 had 2 or more; #193 had 18 (a feature built through "fix" commits), #85 had 5, and #90/#150/#170/#197 had 4 each.

**Reading.** The first-pass verify rate (94%) overstates quality, because verification checks the filed repro. The real rework rate is better measured by follow-on issues: about 30 incomplete fixes out of roughly 180 fixed issues is **about 1 in 6**. Those follow-ons concentrate in exactly the structural areas above: validation, audio sample rate and headroom, the estimator, and sibling templates.

---

## 5. Broader perspective

**Product scope is drifting from "declarative diffusers workflow engine" toward "agent-operated video post-production studio".**

- Recent and open work: audio mastering (EQ, dynamics, LUFS), shot assembly, series/episode tooling, cost estimation, a CPU colour grade (#349), transcription, a contact-sheet vision tool (#193, 18 commits).
- Each of these is individually reasonable. Together, they put a DAW and an NLE on a single-GPU FIFO worker, with CPU-heavy DSP running in the process that owns the GPU.
- My suggestion is to draw a line on purpose: generation and the minimum assembly needed to deliver a shot stay in dw, and post-production goes to ffmpeg graphs or a separate tool. Make that call before the loop's discovery process makes it for you. The standing tester task explores the post-production workflow, so it will keep finding post-production gaps.

**Many implicit behaviours make the system hard for agents.**

- The MCP interface is heavily documented to compensate for implicit semantics: key-name magic, phase-dependent prefixes, warnings in place of errors.
- The tool surface costs about 14k tokens resident, and the loop spends effort keeping prose within budget.
- Making the semantics explicit (typed variables, explicit `$media`/`$type`, structured error codes, one verb per concept rather than the #179 aliases) would let much of that prose go away. Structured errors would also let the MCP client stop flattening 409 bodies into prose (`dw_mcp/client.py:364-451`).

**Warnings have become a substitute for errors or corrections.**

- There are about 15 `emit_warning` sites in audio and result writing alone, and many issues read "silent no-op, add a warning" (#288, #290, #291, #292, #294).
- Warnings accumulate, collide with suite assertions (#194, #258, #305, #309), and do not change outcomes.
- Prefer, in order: refuse at validation, then correct automatically, then warn. Use the last only when the first two are impossible.

**One process, many owners of state.** The worker owns models, step cache and memory stats; the server owns the queue, history and workspaces; and in mounted mode, the MCP client owns a workspace pin shared by every session (#298).

- A single-user posture is a legitimate decision (Don's note on #298).
- If so, make it explicit in code: refuse a second MCP session, or lock the pin. The current posture is to warn when it happens.

**The agent loop itself.**

- *Worked well:*
  - It found and closed ~220 real issues in 10 days.
  - The security suite found genuine holes (#112–#117, #124, #138).
  - The consumer-only tester is a genuinely independent check.
- *What it lacks:* something that works against the structural debt. Each session's scope is one issue, and the budgets reward the narrowest patch. Two options:
  - Periodically inject structural work, for example "W1 phase 1: single prepare pipeline, no behaviour change", scoped by you and verified by the full pytest suite plus the regression suites.
  - Have triage detect *clusters*: three or more issues in the same subsystem within a week trigger an `owner:don` design note instead of another point fix.
- *Suite-maintenance overhead* (about 66 issues) suggests the suites pin too much incidental output (warning counts, exact basis strings, exact wording). Pinning outcomes and error codes would make them sturdier.

**Operability.**

- Good: the deploy script (`scripts/deploy.sh`), a systemd unit, health checks, and `get_server_info` with environment details (#222).
- Missing:
  - job durability (W7)
  - a hang watchdog, since the phase-stall events are only advisory (#357)
  - persisting the step cache across restarts (#244)
  - visibility into downloads a job triggers (#343)
  - a schema version on `jobs.sqlite`

**Dependency risk.** The project tracks diffusers git HEAD and new transformers releases closely; #169, #232, #178 and the CI break in #224/#228 all came from upstream. A pinned, known-good lock for lem (with the updater as an explicit upgrade path) would separate "upstream moved" from "we broke it".

---

## 6. Method and caveats

**Code.**

- I read the hotspots directly and spot-checked the key claims:
  - `jobs.py:818/831` validates without arguments
  - `result.fps` is `"type": "integer"` in the schema
  - `app.py:1157-1160` maps every error to 400 with `str(e)`
  - `requests.get` has no redirect control
  - `validate_media_location` calls the scheme-only `validate_url`
  - the per-sample Python loop in `audio_utils.py:1494-1513`
- Three read-only review sub-agents covered: server/MCP; engine/validation/planning; tasks, security, tests, UI and docs. Function lengths and nesting came from `ast` scans.
- One sub-agent built the MCP server object from the existing dw-agent venv to count tool schema sizes. It imported code from there but modified nothing; I note it because you asked that those checkouts not be used.

**Issues.**

- `gh issue list` (all 280 issues) and GraphQL timelines (labeled, reopened and closed events, comment counts), plus full reads of #193, #197, #298 and #300 and targeted reads of others.
- Categories are keyword heuristics, and the fix-on-fix split is my hand classification of 63 title-linked follow-ons. Treat both as ±20%.

**History.**

- Commit mix and size numbers come from `git log --numstat` and `git cat-file` over `develop`.
- Some history was rewritten: many 2026-08-15 commits share a committer timestamp. I bucketed by committer date.
- "Fix %" depends on commit-message conventions, which changed when the loop started.

**Not done.**

- I did not run pytest (needs torch and diffusers-git) or the UI tests.
- I made no live server calls, so every run-time claim is from reading the code, not reproducing it.
- In particular, I infer the SSRF redirect bypass and the run-time `from_file` gap from the code. They should be confirmed with a test against a redirecting URL and a variable-supplied `from_file` in the security suite before and after fixing.

**Issues #338–#367** are mid-flight. I used them only as signal about the current state: they repeat the same classes (validation gaps #345, #347, #364, #365; audio levels #358, #361, #362; estimator #341, #348; get_task metadata #350, #366; language #338, #363).
