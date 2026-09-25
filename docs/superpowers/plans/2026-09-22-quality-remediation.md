# Quality Remediation Plan

**Source:** `docs/proposals/audits/2026-09-22-quality-review.md`. It is an independent, read-only review of `develop` @ `09ce397`, covering 280 issues and the git history since 2026-08-29. Weakness IDs (W1–W12) below refer to that document's section 3.

**Goal:** Stop the issue stream from re-growing out of the same structural bug classes. The loop fixes issues fast (median 2.75 h to a verified close), but about 1 in 6 fixes needs a follow-on issue. Fix-commit share rose from 17% to 57% in three weeks while new surface kept landing at a steady rate. Each phase below removes a *class* of bug rather than one instance.

**Principles**

- **No behaviour change first, behaviour change second.** Every redesign starts with a phase that the full pytest suite plus the regression suites prove is behaviour-preserving. Behaviour changes land after that, one per issue, each with a `breaking-change` label if it changes the MCP/HTTP interface.
- **Structural work is scoped by Don, executed by the loop.** Each phase becomes one GitHub issue, parked `owner:don` + `status:needs-approval` until Don approves its scope. Once approved, it hands to `owner:implementer` like any other issue. The tester verifies it against the *invariant* stated in the issue, not only a repro.
- **Freeze point fixes in a class once its redesign is approved.** New issues in that class are linked to the phase issue instead of patched individually, unless they are security or data-loss issues.
- **Order of preference for a wrong outcome:** refuse at validation, then correct automatically, then warn. Use a warning only when the first two are impossible.

---

## Phase 0 — Security fixes (now; small)

W2. Confirm each finding with a failing case in `regression-suite-security.md` (in the iterate repo) before the fix, and see it pass after.

- [ ] **Remote media fetch helper.** One function used by `audio_utils.py:361`, `video_utils.py:443`, and the `load_image` paths (`arguments.py:943`, `tasks/gather.py:64`). It should:
  - refuse redirects, or re-validate the host on every hop;
  - pin the resolved IP for the connection, so DNS rebinding can't swap it;
  - cap the response size.
- [ ] **Run-time `from_file`** goes through the same URL and root policy as the static pass (`locations.py:403-450`), not `security.validate_url` (scheme only) with an unrooted `validate_path` (`arguments.py:848-876`).
- [ ] **Error text.** Internal exceptions on `POST /api/jobs` and the other `str(e)` → 400 sites (`app.py:1157-1160, 1275, 1294, 2086, 2467`) become 500 with a generic message. Only known validation errors stay 400 with detail.
- [ ] **One `contain(path, roots)`** replaces `security.validate_path`, `locations._within`, `workspace._is_within`, `dw_mcp/media._confine` and `dw_mcp/assets.py:72-100`. This one is a refactor with no behaviour change; the four fixes above change behaviour.

**Done when:** the security suite has cases for a redirect to an internal address, a variable-supplied `from_file` outside the roots, and path leakage in 400 bodies. All of them pass.

## Phase 1 — Gates and durability (small; unblocks everything after it)

W4 and W7.

- [ ] **CI on `develop`.** Add `develop` to `push.branches` in `.github/workflows/ci.yml`. Also add the Playwright e2e job, or a smoke subset of it.
- [ ] **Deploy gate.** `scripts/deploy.sh` refuses a commit whose CI status on GitHub isn't `success`. It waits briefly if the status is pending, and `--force` exists for emergencies.
- [ ] **Nightly GPU smoke on lem.** Pytest-driven, not agent-driven: one real generation per model family, run from a systemd timer, with failures filed as issues.
- [ ] **Durable job ledger (#300):**
  - insert the job row at submit;
  - on startup, mark leftover `queued`/`running` rows as `interrupted`;
  - have `shutdown()` mark in-flight jobs;
  - replace the try/except `ALTER TABLE` chain (`jobs.py:94-164`) with a `PRAGMA user_version` migration.
- [ ] **Hung-worker watchdog (#357):**
  - if there has been no worker event for N minutes while a phase is expected to report, kill and restart the worker and fail the job with a clear reason;
  - cancel must work against a hung worker.
- [ ] **Job-lifecycle tests.** `server/jobs.py` has the lowest test/code ratio in the codebase (0.2).

**Done when:** a push to `develop` runs CI, and lem refuses to deploy a red commit. Restarting `dw-serve` with a queued job and a running job leaves both visible as `interrupted`.

## Phase 2 — One preparation pipeline for run, validate and plan (redesign; highest leverage)

W1. This is the root of the "validates clean, dies at run time" class: 50 issue titles mention `validate`. Currently open examples: #345, #347, #363, #364, #365.

- [ ] **2a (no behaviour change).** Extract `prepare(definition, arguments) -> PreparedWorkflow` from `_prepare_definition` (`workflow.py:798`). It runs substitute → constrain → expand → elide → realize.
  - Delete the hand copy `expanded_definition` (`workflow.py:439`).
  - `run`, `validate_workflow` and `plan` all consume `PreparedWorkflow`.
  - Proof: the full pytest suite, plus all four regression suites, pass unchanged.
- [ ] **2b.** `JobManager.submit` (`jobs.py:818,831`) runs the same argument-aware validation as `validate_workflow`, instead of `loaded.validate()` with no arguments. Remove the resulting triple validation on the run path (route → submit → worker).
- [ ] **2c.** Validators fail closed. A validator crash (`workflow.py:571, 733`, `app.py:1858`) becomes a validation error naming the validator, not an empty list.
- [ ] **2d.** Move checks next to what they guard.
  - Add a `static_check` hook to `register_command` (which already owns argument signatures), and to pipelines and writers.
  - Migrate the 18 per-incident validator modules onto it one at a time, deleting the duplicated rule tables as you go. Examples: `select_validation._RULES` vs `tasks/select.py`, and the six `_UNRESOLVED_PREFIXES` copies.
  - Move the server-only passes (`_argument_reference_errors`, `app.py:1531`) into the engine, so the CLI and REPL get them too.
- [ ] **2e.** Validate the JSON schema *after* variable substitution, so typed fields accept `variable:` references. That removes the #363 class: 55 typed fields reject `variable:` today.
- [ ] **2f.** Add a Hypothesis property test: for generated argument combinations over the catalog templates, `validate` is clean ⇒ `prepare` succeeds. This test is the invariant the tester verifies against.

## Phase 3 — Audio value model and scope decision

W3. Audio is the largest issue category (about 75 issues). Open: #358, #361, #362; #349 is adjacent.

- [ ] **Decision (Don): how much post-production dw owns.** The review recommends that generation, plus the minimum assembly needed to deliver a shot, stays in dw. Mastering-grade DSP should go to ffmpeg filter graphs through PyAV (already a dependency), or out of dw entirely. #349 (CPU grade) and #361 (LUFS) should wait for this decision.
- [ ] **3a (no behaviour change).** Coerce every audio input to one `AudioTrack(waveform: float32 [C,N], rate, source)` at task dispatch. Delete the per-command normalisers:
  - `_waveform_and_rate` (`audio_utils.py:1344`);
  - `_load_tracks_matching_rate` (`:837`);
  - the copies in `pair_audio`, `concat_videos` and `dissolve_videos`.
- [ ] **3b (`breaking-change`).** `sample_rate` always means *resample to*. The relabel behaviour gets its own explicit argument. Fix `pair_audio.py:171-173`, which relabels silently today.
- [ ] **3c.** Enforce a true-peak ceiling at the single audio write boundary: measure after encode and correct, instead of warning after the fact (`result.py:116-160, 190-235`). That closes the headroom chain (#158 … #362).
- [ ] **3d.** Depending on the scope decision, move EQ, dynamics and loudness to ffmpeg filter graphs. This replaces the pure-Python per-sample envelope loop (`audio_utils.py:1494-1513`) that runs on the GPU worker, and removes the transitive-only scipy dependency.

## Phase 4 — Cost and memory estimator as one fitted model

W5. The estimator is the most-churned feature: 15+ issues since #91, plus about 8 suite-wording issues that followed its changes.

- [ ] **4a.** Create one driver-comparison module that both `plan.py` and `server/observed_cost.py` import. It replaces `_declared_drivers`/`_driver_comparable` (`plan.py:354-377`), whose two copies already diverge on list bucketing.
- [ ] **4b.**
  - Replace the six `basis` values and their overlays with one model: `minutes = f(drivers)`, fitted per workflow and device from history, with curated catalog values as the prior.
  - Report `{estimate, low, high, n}`.
  - This is `breaking-change` for the `plan` payload. Update the regression suites to pin the *shape* and the bounds, not basis strings.
- [ ] **4c.** The host-memory model is `fixed + per_entry × n`. #348 (`f617c19`, landed after the review snapshot) moved to exactly this. Fold it into the unified model instead of keeping a separate chain.

## Phase 5 — Server decomposition and typed responses (mechanical refactor)

W6. Well suited to the implementer, because 8k lines of server tests cover it.

- [ ] **5a (no behaviour change).** Split `create_app` (`app.py:703-4226`) into `APIRouter`s: jobs, catalog, validate, workspaces, gallery, assets, models and system.
  - Move business logic out of route bodies into service modules: validation orchestration, cost acknowledgement, gallery indexing and orphan scan, zip building, frame tiling.
- [ ] **5b.** Add pydantic response models for every route, and generate the UI's `types.ts` from the OpenAPI schema in place of the 363 hand-written lines.
- [ ] **5c.** Fix the concurrency hazards found in the review:
  - the module-global detail caches mutated without a lock (`app.py:212, 226-236, 527`);
  - the check-then-act race in workspace delete (`:1973-1991`);
  - chunked uploads that bypass the size precheck (`:3517-3530`).
- [ ] **5d.** Structured errors, `{code, path, message}`, from validation and the API. Move test and suite assertions from wording to codes; about 17% of assertions pin message text. `dw_mcp/client.py:364-451` then stops flattening 409 bodies into prose.

## Phase 6 — Workflow language: explicit over implicit

W8. Every change here is `breaking-change` and gated on a workflow schema version, so existing workflows keep their meaning. Open: #338, #363 (via Phase 2e), #364, #365.

- [ ] **6a.** Declared variable types, `{"type": "float", "default": 1, "enum": [...]}`. The inferred-from-default form (`variables.py:277`) stays as a legacy mode. Fixes the #338 (truncation) and #364 (null default) class.
- [ ] **6b.** Explicit media and type references, `{"$media": "asset:x.png"}` and `{"$type": "torch.bfloat16"}`, replace inference from key names:
  - the `_image`/`_video` suffixes, and `_type`/`_dtype` with their exceptions (`arguments.py:26, 136-171`);
  - `realize_args` stops running over the variables block (`workflow.py:846`; #365).
  - Deprecate the inferred forms behind the schema version.
- [ ] **6c.** Document one resolution-phase order for all 11 prefixes, and enforce it in `prepare` (Phase 2).

## Phase 7 — Hygiene and consolidation (continuous, low risk)

- [ ] **W9. Model families out of generic code.** Add a `families/` package with a per-family hook for validators and component hints. Move MiniMax-H3 (`adapter_compatibility.py`, `pipeline.py:1985`), Flux (`prompt_weighting.py:293-315`, `teacache.py:291`) and the H3 numbers in `workflow_schema.json` into it.
- [ ] **W10. Templates.**
  - Compose shared assembly through `builtin:` sub-workflows, starting with `assemble-and-score` / `dissolve-between-shots` (95% identical; #339/#342 are the latest missed-sibling pair).
  - Make one source for each piece of guidance (`get_guide`), with tool docstrings pointing at it.
  - Replace `SURFACE_BUDGET`'s character ratchet (`test_mcp_server.py:1450`) with a token count and per-tool caps.
- [ ] **W10. Task metadata.** Derive the hand-kept side tables from signatures, or from one registry (#185, #350, #366).
- [ ] **W11. Comments.** New comments state the invariant; the history goes in the commit message. No sweeping rewrite of existing comments. Edit a comment when its code is next touched.
- [ ] **W12. Tests.** Add a real-codec CPU encode end-to-end test (PyAV). Add property tests for the audio tasks alongside 2f.
- [ ] **Dependencies.** A pinned, known-good lock on lem, with `update_diffusers` as the explicit upgrade path. That separates "upstream moved" (#169, #232) from "we broke it".

---

## Loop process changes (iterate repo, not this repo)

These go in the role prompts and drivers in `dkackman/iterate`, alongside this plan.

- [ ] **Cluster escalation in triage.** When three or more open or recently closed issues fall in one subsystem within seven days, triage parks the newest one `owner:don` + `status:needs-approval` with a `triage: cluster` comment naming the others and the phase above they belong to, instead of dispatching another point fix.
- [ ] **Tester verifies the invariant.** The implementer's hand-off states the root-cause invariant. The tester checks it plus one sibling combination, not only the filed repro (the #197 lesson).
- [ ] **Comment convention** (W11) goes into the implementer prompt.
- [ ] **Suites pin outcomes, not incidentals.** New cases assert outcomes and error codes, not warning counts, basis strings or exact wording. This targets the ~66 suite-maintenance issues.
- [ ] **Structural sessions.** Phase items run as ordinary issues with a "no behaviour change" scope line. Their verify step is the full pytest suite plus the regression suites, run by the regression agent, rather than one repro.

## Decisions needed from Don

1. **Post-production scope** (Phase 3), before #349 and #361 proceed.
2. **Single-user posture** (#298). Either make it explicit in code (refuse a second MCP session, or lock the workspace pin) or design for several sessions. Today it only warns.
3. **Phase order.** Recommended order: 0 → 1 → 2 → 5a → 3 → 4 → 6, with 7 running continuously. 5a is placed early because the router split makes 2b/2d and 4 easier to land.
4. **Freeze policy.** Whether new issues in a class under redesign are linked and held, or still point-fixed.

## Open issues mapped to phases (as of 2026-09-22)

| Phase | Issues |
|---|---|
| 1 | #300, #357, #244 |
| 2 | #345, #347, #363, #364, #365 |
| 3 | #349, #358, #361, #362 |
| 4 | #341, #348 |
| 5 | #298 (posture decision) |
| 6 | #338, #364, #365 |
| 7 | #339, #342, #350, #366 |
