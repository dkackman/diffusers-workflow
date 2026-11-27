# Release Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the defects the 2026-11-27 pre-release review found in the changes since v0.22.0, so regression runs against a tree with no known should-fix.

**Architecture:** No new modules and no new features. Each task is one defect: a test that fails on the current code, the smallest change that makes it pass, one commit. Tasks are grouped by area (preferences engine, server, tasks, installer, UI) and are independent unless a task says otherwise.

**Tech Stack:** Python 3.14 + pytest (`venv/bin/python`), Svelte 5 + TypeScript + vitest, Bash and PowerShell bootstrap scripts, GitHub Actions.

**Spec:** There is no separate spec. The review's findings are the requirements, and each task opens with the finding it answers (what goes wrong, and the trigger). Don's rulings of 2026-11-27 are in *Global Constraints*.

## Global Constraints

- Work on `develop` in this checkout. No worktree, no loop, no GitHub issues for these items.
- First commit of the session: the three uncommitted files `dw/preferences.py`, `dw/setup_cli.py`, `dw/setup_service.py`. `origin/develop` has their tests (commit `291ef3aa`) and is red without them.
- Every task: failing test first, then the fix, then `venv/bin/ruff format dw dw_mcp tests scripts` and `venv/bin/ruff check dw dw_mcp tests scripts`, then commit with `git commit -- <paths>`. Never `git add -A ui`.
- Python tests: `DW_DEVICE=cpu venv/bin/python -m pytest -n0 <file>`. UI tests: `cd ui && npx vitest run <file>`. UI changes also pass `npm run check`, `npm run lint` and `npx prettier --check src e2e`.
- Commit headlines use `refs`, never `fix: #N`.
- Neither ratchet may rise: no new module, no new `svelte-ignore`, no file pushed over its size ceiling. `dw/workflow.py` is at 1,123 lines and `dw/preferences.py` near 820; add as few lines there as the fix needs.
- Model knowledge stays out of engine code: no model's number or name in `dw/*.py`.
- UI rules (`ui/CLAUDE.md`): tests mock `api` with `src/lib/test/apiMock.ts`; component tests never call `vi.resetModules()`; copy follows `ui/GLOSSARY.md`.
- A server model change is followed by `venv/bin/python scripts/dump_openapi.py` and `cd ui && npm run gen:api`, committed together.
- Rulings (Don, 2026-11-27): a soft VRAM figure never skips a preference target (Task 1); the 3090 rounding is fixed (Task 2); the queued-job pin and the alpha install URL are fixed only if the fix is simple (Task 3 is; the URL is left alone, see *Not planned*).

## Review Focus

These are the conditions the fixes are most likely to get wrong, each pinned by a test in the task named.

1. A host whose card is smaller than every measured card (12 GB): each tier still resolves, and the answer says the fit is unchecked (Task 1).
2. A job admitted at one default tier and run after the default changes: the worker runs the admitted model (Task 3).
3. A real browser fires `hashchange` listeners with microtask turns between them: every filter resets on a bare link (Task 18).
4. A refresh that lands while the person is mid-gesture (viewing a take, dragging a row, editing a ladder): nothing they are looking at disappears and nothing is written to the wrong place (Tasks 19, 20, 21, 29).
5. Windows with Claude open during an upgrade: the script stops before touching the venv and says what to close (Task 16). This one cannot be run here; it is checked in the Windows run (Task 33).

---

## Phase 0: Land what is already written

### Task 0: Commit the three source fixes

**Files:** `dw/preferences.py`, `dw/setup_cli.py`, `dw/setup_service.py` (already edited, uncommitted)

- [ ] **Step 1:** Run `DW_DEVICE=cpu venv/bin/python -m pytest -q -n0 tests/test_setup_cli.py tests/test_setup_service.py tests/test_preferences.py tests/test_preferences_routes.py`. Expected: 189 passed.
- [ ] **Step 2:** `git commit -m "fix: preference target and default-tier refusals, hf login path, UTF-8 setup runner" -- dw/preferences.py dw/setup_cli.py dw/setup_service.py`

---

## Phase 1: Preferences engine

### Task 1: A soft VRAM figure does not skip a target

**Finding:** `card_need_gb` (`dw/preferences.py:325-345`) drops the `hard` flag that `vram_estimate.required_vram_gb` returns, and `skip_reason` skips on the figure either way. A soft figure is the size of the card a `cost` row was measured on, not a floor. On a 16 GB card the shipped draft rung (SD 1.5, measured on 24 GB) refuses; on 12 GB all three tiers refuse.

**Files:**
- Modify: `dw/preferences.py` (`card_need_gb`, `skip_reason`, `resolve`'s notes, the module docstring's *Fit* paragraph)
- Modify: `docs/WORKFLOW_GUIDE.md` (*Fit*)
- Test: `tests/test_preferences.py`

**Interfaces:**
- Produces: `card_need_gb(definition, arguments, device_type, child=None) -> tuple[float, bool] | None`, the `(gb, hard)` pair as `required_vram_gb` gives it. Check its other callers with `grep -rn card_need_gb dw tests` and update each.

- [ ] **Step 1: Write the failing tests.** Three cases against a target with a `cost` row measured on a 24 GB cuda card and no `vram_estimate`, and one with `vram_estimate` projecting 20 GB:
  - `test_a_soft_figure_does_not_skip`: host `Host("cuda", 12, 64)`; the soft target resolves, `skipped == []`.
  - `test_a_soft_figure_over_the_card_is_a_note`: same host; `resolution.notes` holds exactly one line: `"measured on a 24 GB card; this one has 12 GB, so the fit is not checked"`.
  - `test_a_hard_figure_still_skips`: host `Host("cuda", 12, 64)`; the `vram_estimate` target is skipped with the existing "needs 20 GB VRAM on cuda" reason and the next target in the rung is picked.
- [ ] **Step 2:** Run them; the first two fail with a refusal.
- [ ] **Step 3: Implement.** `skip_reason` skips only when `hard` is true. `resolve` appends the note above for the chosen target when the figure is soft and larger than `host.smallest_card_gb`. Existing tests that pinned a soft skip are rewritten to the new rule, not deleted.
- [ ] **Step 4:** Rewrite the *Fit* paragraphs in the module docstring and `docs/WORKFLOW_GUIDE.md`: a declared `vram_estimate` skips; a `cost` row's card is reported, never enforced.
- [ ] **Step 5:** Run `tests/test_preferences.py tests/test_preferences_routes.py tests/test_preference_defaults.py`; all pass. Commit.

### Task 2: The preference host rounds a card the way the pool does

**Finding:** `default_host` (`dw/preferences.py:168`) takes `device_capacity_gb()` raw, so a 3090 is 23.6 and a hard 24 GB figure does not fit; the message then reads "needs 24 GB ... the smallest card has 24 GB". The pool uses `math.ceil` (`dw/worker_manager.py:286`).

**Files:** Modify `dw/preferences.py` (`default_host`); Test `tests/test_preferences.py`

- [ ] **Step 1: Failing test** `test_default_host_rounds_the_card_up_like_the_pool`: patch `dw.device_capacity_gb` (with `patch.object`) to return `23.6`; `preferences.default_host().smallest_card_gb == 24`.
- [ ] **Step 2: Implement:** `math.ceil(card)` when `card` is truthy.
- [ ] **Step 3:** Pass, commit.

### Task 3: An untiered step replays its admission pin after the default tier changes

**Finding:** `Workflow.resolve_preferences` (`dw/workflow.py:579-590`) falls back to `session.preferences()["default_tier"]`, read from disk in the worker. If `PUT /api/preferences` changed it after admission, the pin key `path@tier` misses and the worker resolves fresh, running a model the plan never quoted. A rerun misses the same way.

**Files:** Modify `dw/preferences.py` (new helper), `dw/workflow.py` (one call); Test `tests/test_preferences.py`

**Interfaces:**
- Produces: `preferences.pinned_tier(pinned: dict, path: str) -> str | None`: the tier of the only key in `pinned` of the form `f"{path}@<tier>"`, or `None` when there are none or more than one.

- [ ] **Step 1: Failing tests.**
  - `test_pinned_tier_reads_the_only_pin_for_a_path`: `{"preference:k@standard": {...}}` gives `"standard"`; two tiers for the path give `None`; another path's pin gives `None`.
  - `test_a_default_tier_change_does_not_move_a_pinned_untiered_step`: admit a workflow with one untiered `preference:` step under a store whose default is `standard` (keep `session.resolved`); expand it again under `Session(pinned=<that map>, store=<same store with default_tier "draft">)`; the resolved target equals the admitted one and `resolution.pinned` is true.
- [ ] **Step 2: Implement.** In `resolve_preferences`, where the tier would come from the store default, use `preferences.pinned_tier(session.pinned, path)` first. Explicit and inherited tiers are unchanged.
- [ ] **Step 3:** Pass, commit.

### Task 4: The preferred template follows the server's default tier

**Finding:** `workflows/templates/preferred/text-to-image.json` defaults `tier` to `"standard"`, so the store's `default_tier` is unreachable through the one template that exists for it, while `plugins/dw/skills/choosing-a-model/SKILL.md` says "pass none, so the store's default applies".

**Decision (recommended, awaiting Don's yes):** fix the promise at the agent, not the engine. The template's `tier` also selects its `presets` bundle (canvas size), so a null default would need the fold to consult the store before presets apply, which is an engine change on the path every run takes. Instead the skill reads the server's default and passes it.

**Files:** Modify `plugins/dw/skills/choosing-a-model/SKILL.md`, `plugins/dw/README.md` if it repeats the sentence, `workflows/templates/preferred/README.md` if present; Test `tests/test_plugin_skills.py`

- [ ] **Step 1: Failing test** `test_choosing_a_model_passes_the_servers_default_tier`: the skill text contains `get_preferences` and `default_tier`, and no longer contains "pass none".
- [ ] **Step 2: Rewrite** the table row and the paragraph under it: when the user says nothing about quality, call `get_preferences`, read `default_tier`, and pass it as `tier`. State plainly that the template itself runs `standard` when no tier is passed.
- [ ] **Step 3:** Run `tests/test_plugin_skills.py`; pass; commit. Bump the plugin's skill pin if that test file requires it.

### Task 5: A failed preference step keeps its own reason

**Finding:** `_preference_errors` (`dw/workflow.py:600-602`, read at `:688`) is keyed by path and popped on any later success. Two steps naming the same `preference:` path at different tiers share one slot: a later success erases an earlier refusal, which then reads "resolves only inside a composing workflow's expansion".

**Files:** Modify `dw/workflow.py`; Test `tests/test_preferences.py`

- [ ] **Step 1: Failing test** `test_a_refused_step_keeps_its_reason_when_a_later_step_resolves`: steps `a` (`tier: best`, no target fits) then `b` (`tier: draft`, fits), same path; validation's error for step `a` contains the best rung's skip reason, not "resolves only inside".
- [ ] **Step 2: Implement.** Stop popping on success: a resolved step's path is rewritten, so the slot is only read for a step that failed. Keep the first error recorded for a path within one expansion, and clear the dict at the start of `resolve_preferences`.
- [ ] **Step 3:** Pass, commit.

### Task 6: A null component on one step does not hide another step's download

**Finding:** `dw/plan_downloads.py:289-295` unions per-repo "skip this component" sets across steps. One step loading repo X with `"safety_checker": null` hides the component for a second step that loads it.

**Files:** Modify `dw/plan_downloads.py`; Test: the existing plan-downloads test file (`grep -ln plan_downloads tests/`)

- [ ] **Step 1: Failing test** `test_a_component_one_step_nulls_and_another_loads_is_still_counted`: two steps on `org/sd`, the first with `"safety_checker": None`; `_collect_sources` gives no skip for `org/sd`.
- [ ] **Step 2: Implement:** intersect the skip sets per repo instead of taking their union.
- [ ] **Step 3:** Pass, commit.

### Task 7: A `builtin:` refusal names no install path

**Finding:** `dw/library.py:825` puts the packaged workflows directory's absolute path in an error that the preferences routes send as `detail`.

**Files:** Modify `dw/library.py`; Test `tests/test_preferences_routes.py`

- [ ] **Step 1: Failing test:** add `"builtin:a/b.json"` to the parameters of the existing `test_a_refused_target_names_no_server_path`, and assert the body does not contain `os.path.dirname(dw.__file__)`.
- [ ] **Step 2: Implement:** name the root by its label ("the packaged workflows") as the neighbouring refusals do.
- [ ] **Step 3:** Pass, commit.

### Task 8: Observed cost is not quoted across a change of resolved model

**Finding (inferred, verify first):** `dw/server/observed_cost.py:76-101` buckets history by workflow name and cost drivers. After a ladder or default-tier change, runs of the old target are quoted for the new one with `basis: observed`.

**Files:** Modify `dw/server/observed_cost.py`; Test: its existing test file

- [ ] **Step 1: Reproduce in a test** `test_history_from_another_preference_target_is_not_this_ones`: three recorded runs whose job spec `preferences` map resolves `preference:text_to_image@standard` to `models/a`; a plan whose resolution is `models/b` gets no observed figure. If it already passes, delete the test, record "not reproduced" in the commit message of Task 9 and skip this task.
- [ ] **Step 2: Implement:** add the sorted resolved targets (from the job's `preferences` map and the plan's session) to `_bucket_key`.
- [ ] **Step 3:** Pass, commit.

---

## Phase 2: Server

### Task 9: A failed update keeps refusing jobs until the restart

**Finding:** `dw/server/self_update.py:167` runs the dw upgrade, then the diffusers reinstall. If the second fails, status is `failed`, nothing restarts, and `submission_refusal` (`dw/server/app.py:286-291`) lifts because it only refuses while `running`. New workers then import the new engine under the old server. The same holds after "a job is running; restart dw-serve when it finishes".

**Files:** Modify `dw/server/self_update.py`, `dw/server/updater.py`, `dw/server/app.py`; Test: the existing self-update test file

**Interfaces:**
- Produces: `DwUpdater.status()` gains `"restart_required": bool`, true from the moment the first pip step has returned 0 until the process exits. Add the field to the status model in `dw/server/api_models_system.py`, then dump OpenAPI and regenerate the UI types.

- [ ] **Step 1: Failing tests.**
  - `test_a_failure_after_the_dw_step_still_refuses_jobs`: `run_commands` fake succeeds for the first command and fails the second; `status()["status"] == "failed"`, `status()["restart_required"] is True`, and `manager.submission_refusal()` is the text "dw was partly updated - restart dw-serve before submitting jobs".
  - `test_an_update_finished_under_a_running_job_refuses_new_jobs`: update succeeds while `is_busy` is true; `submission_refusal()` returns the same restart text.
  - `test_a_failure_before_anything_installed_lifts_the_refusal`: first command fails; `restart_required` false, refusal `None`.
- [ ] **Step 2: Implement.** `run_commands` is called per step so the updater knows the dw step landed. `submission_refusal` refuses when status is `running` or `restart_required`.
- [ ] **Step 3:** Dump OpenAPI, `npm run gen:api`; run the server and twin tests (`tests/test_api_contract.py tests/test_ui_twins.py`); commit all together.

### Task 10: A client that disconnects is not a traceback

**Finding:** the e2e log shows `RuntimeError: No response returned.` from `reject_foreign_origins` (`dw/server/http_security.py:103`) when the browser drops a request. Noise in a user's log, no wrong behaviour.

**Files:** Modify `dw/server/http_security.py`; Test: its existing test file

- [ ] **Step 1: Failing test** `test_a_dropped_request_is_not_an_error`: call the middleware with a `call_next` that raises `RuntimeError("No response returned.")` and a request whose `is_disconnected()` is true; it returns a response with status 499 and logs nothing at error level.
- [ ] **Step 2: Implement:** catch that `RuntimeError` only when the request is disconnected; re-raise otherwise.
- [ ] **Step 3:** Pass, commit.

---

## Phase 3: Tasks

### Task 11: Held frames are not excused as cuts when most neighbours are held

**Finding:** `held_measure` (`dw/tasks/assess.py:1021-1043`) uses the spike rule in `dw/tasks/scene_cuts.py:93-101`: a cut is a change of at least 6% that is at least 4x the median of its neighbours. When more than half the neighbours are held the median is 0, every new picture is a cut, and the guard removes the holds beside it. A clip on threes reports 0 held frames.

**Files:** Modify `dw/tasks/assess.py` only (scene-cut detection for `plan_cuts` is unchanged); Test: the existing held-frames tests (`grep -ln held_measure tests/`)

- [ ] **Step 1: Failing tests** on synthetic frames:
  - `test_a_clip_on_threes_reports_its_holds`: 60 frames, a new picture every third frame with at least 6% mean change; `held_frames == 40`, cuts `0`.
  - `test_a_half_rate_clip_reports_holds_at_head_and_tail`: 60 frames, every other one a repeat; `held_frames == 30`.
  - `test_a_real_cut_between_two_moving_shots_is_still_excused`: two moving shots joined; the frame at the join is a cut and not held.
- [ ] **Step 2: Implement:** in `held_measure`, take the median for the spike rule over neighbours that are not themselves held (change above the held threshold); with no such neighbours, nothing is a cut.
- [ ] **Step 3:** Pass, run `tests/test_assess.py tests/test_interpolate_frames.py`, commit.

### Task 12: Refill builds one flow grid, not one per held frame

**Finding:** `dw/tasks/interpolate_frames.py:371-377` caches a full-resolution context per `(ph, pw, timestep)`; a 120-frame hold keeps about 3 GB at 1080p until the call returns. The grid is the same for every timestep.

**Files:** Modify `dw/tasks/interpolate_frames.py`; Test `tests/test_interpolate_frames.py`

- [ ] **Step 1: Failing test** `test_refill_builds_one_grid_for_a_long_hold`: count calls to the grid builder while refilling a 39-frame hold; exactly 1.
- [ ] **Step 2: Implement:** cache the warp grid by `(ph, pw)`; build the timestep plane per call and do not keep it.
- [ ] **Step 3:** Pass (output frames unchanged against the existing tests), commit.

### Task 13: `paste_into_box` checks its record against the clip, and takes the fit's own frame count

**Findings:** (a) `dw/tasks/paste_box.py:121-139` never compares the record's `model_width`/`model_height` with the generated clip, so a 2x model output is pasted at the wrong scale with no error. (b) `:115-120` refuses a source whose frame count is not the record's, though `fit_to_model` itself cut or padded the source to `num_frames`, and the recipe in `docs/TASKS.md:1409-1421` passes the original clip.

**Files:** Modify `dw/tasks/paste_box.py`, `docs/TASKS.md`; Test `tests/test_paste_into_box.py`

- [ ] **Step 1: Failing tests.**
  - `test_a_generated_clip_of_another_size_is_refused`: record for 32x64, generated 64x128; raises with both sizes in the message.
  - `test_a_longer_source_is_cut_to_the_records_frames` and `test_a_shorter_source_is_padded_as_the_fit_padded_it`: 100-frame and 140-frame sources against a 121-frame record; output has 121 frames.
- [ ] **Step 2: Implement.** Refuse on a size mismatch. Bring the source to `model_frames` with the same cut/pad function `dw/tasks/fit.py` uses (import it; do not copy it).
- [ ] **Step 3:** Update the TASKS.md recipe's sentence about frame counts. Pass, commit.

### Task 14: `analyze_held_frames` gives no seconds for a video with no frame rate

**Finding:** `dw/tasks/assess.py:963` (`rate = fps or 24.0`) times clusters at an assumed 24 fps with nothing saying so. The repo rule is never to relabel to a fixed 24.

**Files:** Modify `dw/tasks/assess.py`; Test `tests/test_assess.py`

- [ ] **Step 1: Failing test** `test_held_clusters_without_a_frame_rate_are_in_frames`: 8 frames, `fps=None`; each cluster has `first_frame` and `last_frame` and no `start`/`end`; the result carries `"fps": None`. With `fps=12` the seconds are present as today.
- [ ] **Step 2: Implement.** Check how the findings' `at`/`range` are consumed (`dw/server/assess.py`, the UI's `findingPlace.ts`) before removing them; if a consumer requires a number, keep the frame index there and add `"unit": "frames"`.
- [ ] **Step 3:** Pass, commit.

---

## Phase 4: Installer and release job

None of these can be run for real on this Mac. Each has a dry-run or text test in `tests/test_bootstrap.py`; the real check is Task 33.

### Task 15: The bootstrap resolves across both indexes

**Finding:** `dw/install_scripts/bootstrap.sh:141` and `bootstrap.ps1:124` pass `--extra-index-url <pytorch index>` to `uv pip install`. uv resolves each package from the first index that carries it, so a PyTorch mirror that lags PyPI can make `numpy>=2.5.4` or `Pillow>=13.0.0` unsatisfiable on the cu130 and cpu indexes. `install.sh` uses pip, which merges indexes.

**Files:** Modify both bootstrap scripts; Test `tests/test_bootstrap.py`

- [ ] **Step 1: Reproduce** (network, public indexes only): `uv pip compile pyproject.toml --extra server --extra mcp --extra-index-url https://download.pytorch.org/whl/cpu --python-platform x86_64-unknown-linux-gnu -o /dev/null`, then again with `--index-strategy unsafe-best-match`. Record both outcomes in the commit message.
- [ ] **Step 2: Failing test** `test_a_torch_index_install_takes_the_best_match_across_indexes`: the dry-run command line for a non-`pypi` torch index contains `--index-strategy unsafe-best-match`, in both scripts.
- [ ] **Step 3: Implement** on the two `--extra-index-url` lines only. Pass, commit.

### Task 16: The Windows bootstrap stops cleanly

**Findings:** (a) `bootstrap.ps1:84-103` ends only the `dw-serve` task before uv touches the venv; with Claude open, `dw-mcp.exe` is running and uv cannot replace it. (b) `exit` at `:35,152,157` closes the user's window in the documented `& ([scriptblock]::Create((irm ...)))` form. (c) `schtasks /End ... 2>&1 | Out-Null` at `:100` under `$ErrorActionPreference = "Stop"` aborts on PowerShell 5.1 when the task is gone.

**Files:** Modify `dw/install_scripts/bootstrap.ps1`, `docs/INSTALL.md`; Test `tests/test_bootstrap.py`

- [ ] **Step 1: Failing text tests.**
  - `test_the_windows_script_never_exits_the_host`: the script contains no bare `exit` statement outside a comment.
  - `test_the_windows_upgrade_checks_for_processes_in_the_venv`: the script contains a `Get-Process` check on paths under the venv ahead of the first `uv pip` line.
  - `test_schtasks_end_cannot_abort_the_upgrade`: the `schtasks /End` line sits inside a block that sets `$ErrorActionPreference = "Continue"`.
- [ ] **Step 2: Implement.** Wrap the body in a function invoked at the end; failures `throw` a one-line message and success `return`s. Before the install step on an upgrade, list processes whose path is under the venv; if any, stop with "Close Claude Desktop and Claude Code, then run this again" and their names. Do not kill them.
- [ ] **Step 3:** Add the "close Claude before upgrading on Windows" line to `docs/INSTALL.md`'s upgrade section. Pass, commit.

### Task 17: PyPI publishes before the GitHub release

**Finding:** `.github/workflows/ci.yml:217,297`: `release` attaches the stamped bootstrap, and `pypi` runs after it. The stamped script installs `diffusers-workflow==<tag>`, so the one-liner fails from release creation until the upload lands, or for as long as the `pypi` environment waits.

**Decision (Don runs releases; confirm before this task):** `pypi` needs `wheel`; `release` needs `[wheel, pypi]`; `mcp-registry` keeps `needs: release`.

**Files:** Modify `.github/workflows/ci.yml`, `docs/RELEASING.md` (the job-order sentence)

- [ ] **Step 1:** Edit the two `needs:` lines and move any artifact download the `pypi` job took from `release` to take from `wheel`.
- [ ] **Step 2:** Validate the YAML: `venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`. Commit.

---

## Phase 5: UI

### Task 18: A bare link resets every filter

**Finding:** each `hashState` box (`ui/src/lib/router.svelte.ts:158-174`) has its own `hashchange` listener. A browser runs microtasks between listeners, so the first box's write effect writes the other boxes' old values back before they sync. Clicking "Jobs" in the sidebar with status and name filters set resets status only. `router.svelte.test.ts:547` passes because jsdom calls listeners in one turn.

**Files:** Modify `ui/src/lib/router.svelte.ts`; Test `ui/src/lib/router.svelte.test.ts`

- [ ] **Step 1: Failing test** `a hash with no query resets every box, one listener at a time`: capture the listeners `addEventListener('hashchange', ...)` receives, set two boxes non-initial, change the hash to the bare path, call each captured listener with `await Promise.resolve()` between; both boxes read their initial value and the hash has no query.
- [ ] **Step 2: Implement:** one module-level `hashchange` listener that syncs every registered box in a single synchronous pass; boxes register and unregister with it.
- [ ] **Step 3:** Run `router.svelte.test.ts` and `freshness.test.ts`; run `npx playwright test e2e/journeys/consistency.spec.ts` after `npm run build`. Commit.

### Task 19: Viewing a take keeps its row open

**Finding:** `ui/src/lib/request/RequestItem.svelte:88-91`: a row open only by default collapses when a refresh changes its state, unmounting `AttemptStrip` and its `MediaViewer` mid-view, with `open=` left in the hash.

**Files:** Modify `ui/src/lib/request/RequestItem.svelte` (and `ItemMore.svelte` / `AttemptStrip.svelte` for the callback); Test `ui/src/lib/request/RequestItem.test.ts`

- [ ] **Step 1: Failing test** `a row stays open while one of its takes is being viewed`: render a running item, open take 1, rerender with status `done`; the dialog is still in the document and the row's `aria-expanded` is `"true"`.
- [ ] **Step 2: Implement:** opening a take sets `chosen = true`, the way `onnote` does.
- [ ] **Step 3:** Pass, commit.

### Task 20: The Preferences edit lock releases when its key goes away

**Finding:** `ui/src/lib/pages/PreferencesPage.svelte:49,167-188`: `editing` is cleared only by the open card. If another tab removes the key, the card unmounts and every Edit and New preference stays disabled until the page is left.

**Files:** Modify `PreferencesPage.svelte`; Test: its test file

- [ ] **Step 1: Failing test** `the edit lock releases when the edited key is removed elsewhere`: open Edit on a server-only key, feed a read without that key; another card's Edit button is enabled and a notice reads "<key> was removed on the server; your edit was dropped".
- [ ] **Step 2: Implement:** derive whether `editing` is still among the shown keys; when it is not, clear it and raise the notice through `notify`.
- [ ] **Step 3:** Pass, commit.

### Task 21: Discard reopens on the server's ladder

**Finding:** `ui/src/lib/ladder/PreferenceCard.svelte:98-103,212-218`: when a read moves a key under a dirty draft, `seen` advances without adopting; Discard restores the old base and nothing re-adopts, so Edit reopens the old ladder as if clean.

**Files:** Modify `PreferenceCard.svelte`; Test `PreferenceCard.test.ts`

- [ ] **Step 1: Failing test** `discard after a remote change adopts the server's ladder`: dirty the draft, rerender with a new etag and ladder, Discard, Edit; the editor shows the new ladder and Save sends the new etag.
- [ ] **Step 2: Implement:** `discard()` adopts `entry` when its etag differs from the base's.
- [ ] **Step 3:** Pass, commit.

### Task 22: A dry-run answer never paints over a later draft

**Finding:** `ui/src/lib/ladder/dryRun.svelte.ts:54,84-87`: `sent` advances when the 300 ms timer fires, not when the draft changes, so an answer for draft N landing inside draft N+1's debounce window is shown against N+1. The header comment and the test "drops an answer older than the latest check sent" claim otherwise.

**Files:** Modify `dryRun.svelte.ts`; Test `dryRun.svelte.test.ts`

- [ ] **Step 1: Failing test** `an answer that lands during the next draft's debounce is dropped`: send draft 1, change the draft, resolve draft 1's answer before the timer fires; `answer` stays what it was (or null) and `targetErrors` is empty.
- [ ] **Step 2: Implement:** bump the sequence on every draft change, not on send. This file's `dryRun` function is already flagged long (70 lines); do not add to its length net.
- [ ] **Step 3:** Pass, commit.

### Task 23: The same workflow twice in a tier renders

**Finding:** `ui/src/lib/ladder/TierAnswer.svelte:60` keys skipped entries by workflow name, and `ChainEditor.svelte:68-70,91` looks state up by name. The server accepts one workflow twice in a rung (per device), which gives a duplicate-key error and the first entry's state on both chips.

**Files:** Modify `TierAnswer.svelte`, `ChainEditor.svelte`; Test: their test files

- [ ] **Step 1: Failing tests.** `TierAnswer`: two skipped entries for `qa/x` with different reasons both render. `ChainEditor`: a tier `[x(cuda), x(mps), y]` with skips for both `x` entries shows each chip its own reason.
- [ ] **Step 2: Implement:** key by index; in `shown()`, match the nth chip named `x` to the nth skip named `x`.
- [ ] **Step 3:** Pass, commit.

### Task 24: An oversized upload says so

**Finding:** `ui/src/lib/apiErrors.ts:173-207`: 413 is not a refusal, so a file over the server's limit reads "Something went wrong" and the server's message is dropped.

**Files:** Modify `apiErrors.ts`; Test `apiErrors.test.ts`

- [ ] **Step 1: Failing test** `a 413 is a refusal with the server's reason`: `describeError` of a 413 with detail "Upload too large: 300000000 > 209715200" gives kind `refused` and a title containing "too large".
- [ ] **Step 2: Implement:** add 413 to `REFUSED` with copy "This file is too large to upload", detail shown beneath.
- [ ] **Step 3:** Pass, commit.

### Task 25: The run form sends what the variable takes

**Findings:** in `ui/src/lib/workflow/typedArguments.ts:18` (a) a variable whose default is `null` is a text widget, so `["hello"]` typed for `templates/check-script`'s `lines` goes as a string and is refused; (b) a bare scalar typed for a list-default variable is sent as a number, where v0.22.0 sent `"5"` and the engine made `["5"]`.

**Files:** Modify `typedArguments.ts`; Test `typedArguments.test.ts`

- [ ] **Step 1: Failing tests.**
  - `a null default takes a typed list or object as JSON`: `["a"]` gives `['a']`; `{"k":1}` gives `{k: 1}`; `hello` and `5` stay strings.
  - `a list default takes a bare scalar as text`: default `[1]`, typed `5` gives `'5'`; typed `[5]` gives `[5]`.
- [ ] **Step 2: Implement:** for a null default, parse as JSON only when the trimmed text starts with `[` or `{` and parses; for a list default, send parsed JSON only when it is an array.
- [ ] **Step 3:** Pass, run `tests/test_ui_twins.py`, commit.

### Task 26: Run does not send a variable the workflow dropped

**Finding:** `ui/src/lib/workflow/runForm.ts:16-21`: `overrides` is seeded once and never filtered, so after a quiet reload removes a variable the form still sends it and the server answers "Unknown variable" with no field to show it under.

**Files:** Modify `runForm.ts`; Test `runForm.test.ts`

- [ ] **Step 1: Failing test** `an override for a variable no longer declared is not sent`: overrides `{foo: 'x', prompt: 'y'}` against a definition declaring only `prompt`; `runArguments` gives `{prompt: 'y'}`.
- [ ] **Step 2: Implement:** filter by declared names in `runArguments`. Pass, commit.

### Task 27: The Workflows filter matches the word being typed

**Finding:** `ui/src/lib/workflow/catalogSearch.ts:74-93` matches whole words only, so the live filter box shows an empty catalog at `f`, `fl`, `flu` and fills at `flux`.

**Files:** Modify `catalogSearch.ts`; Test `catalogSearch.test.ts`

- [ ] **Step 1: Failing test** `the last word of the query matches as a prefix`: `flu` finds the flux entries; `flux ima` ranks entries with both `flux` and a word starting `ima` first; `lux` finds nothing (still not a substring match).
- [ ] **Step 2: Implement:** earlier words stay whole-word; the final word matches a word's start. Update `catalogSearch.test.ts:83`'s description to say so. Pass, commit.

### Task 28: Editing a braced argument keeps its braces

**Finding:** `ui/src/lib/editor/commit.ts:20-30` against `:46-55`: `addedArgument` writes `{pil}` for a `*_type` key, but replacing the field's text stores bare `latent`, which the engine loads as a type and fails at load time after validation passed.

**Files:** Modify `commit.ts`; Test `commit.test.ts`

- [ ] **Step 1: Failing test** `editing a braced value keeps it braced`: stored `{pil}`, typed `latent`, stored value becomes `{latent}`; stored `torch.float16` (unbraced), typed `torch.float32`, stays unbraced; typing `{x}` explicitly is not double-braced.
- [ ] **Step 2: Implement** in `commitArgument`. Pass, commit.

### Task 29: Small guards

Three one-line defects, one commit each.

- [ ] **Go to on a stale finding** (`ui/src/lib/editor/editorJumps.svelte.ts:40`): test `a finding whose step is gone does nothing`: jump to index 3 of a 3-step document; no throw. Implement: return when `doc.steps[place.index]` is undefined.
- [ ] **A drag whose rows changed** (`ui/src/lib/request/reorder.svelte.ts:176-182`): test `a drop after the rows changed writes nothing`: pointer down, change `keys()`, pointer up; the move callback is not called. Implement: compare the keys captured at pointer-down with the current ones at pointer-up; on a difference, end the drag with no move.
- [ ] **Held-line wording** (`ui/src/lib/request/runPlan.ts:110-124`): tests replace the pins at `runPlan.test.ts:303-304`: review state `due` and not listening reads "Review time has passed. Running it now is your go-ahead."; `redirected` reads "Held: it will not start on its own. Running it now is your go-ahead."; a future deadline keeps "Waiting for your OK until HH:MM." Check the words against `ui/GLOSSARY.md`.

---

## Phase 6: Gates and what only Don can settle

### Task 30: Full gates

- [ ] `venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts`
- [ ] `DW_DEVICE=cpu venv/bin/python -m pytest -q` (expect more than 24,383 passed, 0 failed)
- [ ] `venv/bin/python -m pytest -m integration -n0` (MPS)
- [ ] `cd ui && npm run check && npm run lint && npx prettier --check src e2e && npm run build && npm run test && npx playwright test`
- [ ] Both ratchets, and confirm no metric rose against this morning's numbers (`modules` 291, `modules_over_size_ceiling` 6, `a11y_suppressions` 2).

### Task 31: Ratchets and the bundle budget (Don's rulings)

Not code until ruled. For each, the choice is raise the baseline or do the work:

- Architecture: `modules` 282 → 291 and six modules over 1,100 lines (`h3_windows.py` 1,352, `ledger.py` 1,206, `workflow_run.py` 1,129, `workflow.py` 1,123, `h3_rules.py` 1,114, `tasks/assess.py` 1,111).
- UI: `a11y_suppressions` 1 → 2 (`ChainEditor.svelte:156`, the focusable draggable chip).
- e2e: entry bundle 309.8 KB against 300 KB.

### Task 32: Un-accepting a take (Don's ruling)

`dw/ledger_views.py:69-75` and `dw/ledger.py:625`: writing `accepted: null` leaves the item reading `done`, because `accepted_basis` is carried forward. If the basis is meant to outlive the acceptance this is correct; if not, the fix is to clear `accepted_basis` when `accepted` becomes null, with a test in `tests/test_ledger_close.py`.

### Task 33: Fresh-machine runs (Don)

The installer has only ever been dry-run. Still owed from the installer hand-off of 2026-11-25: the one-liner on a clean Windows machine and on lem, and the Claude Desktop / Cowork check. Tasks 15 and 16 are verified there: a cu130 or cpu install resolves, an upgrade with Claude open stops with the "close Claude" message, and the script-block form leaves the window open.

---

## Not planned

- **The alpha install URL.** `releases/latest/download/...` skips pre-releases, so an alpha tag would leave the one-liner pointing at v0.22.0, which has no bootstrap. Nothing simple fixes it in code; it does not arise if the tag is `v0.23.0`.
- **Refill reporting `held_after == held_before`** for a hold replaced by a morph. It follows from how held is measured after Task 11; re-check once that lands, and file it only if it still reads wrong.
- **A target file with `"steps": "nope"` is stored by PUT.** The file is the user's own workflow and validate refuses it at use.
