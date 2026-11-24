# UI follow-ups after the quality program: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** close the 16 open UI follow-up issues (#1598-#1612, #1623, #1624, #1599). The two release-blockers, #1598 and #1600, go first. Then come the user-visible bugs the later phases left, then copy, tests and metric holes, then the timestamps pass. #1603 is investigated separately.

**Architecture:** Most of the work is UI: Svelte 5 components, the `resource` freshness layer, the request plan store, and `ui/scripts/arch-metrics.mjs`. There are four server touches, each additive or a fix:
- the keep_output 409 gains `code: 'exists'` (`dw/server/routes/assets.py`);
- the prompt and media opt-ins are fixed (`dw/server/outputs.py`, `dw/server/job_history.py`, `dw/security.py`);
- two fields are added if Don agrees (Question 3);
- timestamp fields for #1599 (Question 2).

One engine touch is in scope if it stays small: a lock around the first class import in `dw/introspection.py` (Task 23).

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config, Vite 8, monaco-editor 0.57. Server: FastAPI 0.143.0 / Starlette 1.10, pytest.

**Spec:** the 16 issues themselves (`gh issue view N --comments`) and `docs/superpowers/specs/2026-11-22-ui-quality-design.md` (P9, P11). The appendix below, the **item ledger**, lists every item in every issue. Each item is checked against the code at develop 62814175 and given a status and the task that takes it. Don closes the issues from the ledger.

**What the phases already did.** The phase plans' *Folded in* lists were checked against the code, not taken on trust. Two of them claimed more than the code shows:
- **#1600's "Undo behind busy"** is still open. `undoToast` checks its target before `action`'s `turn.after` waits (`lib/destructive.ts:42`, `lib/action.svelte.ts:80`).
- **#1601's WorkflowPage tilde** is still open (`pages/WorkflowPage.svelte:89`).

The other way round, #1601's "Schema h1, Don's call" is already done: 237b0023 renamed it to `Schema`.

**Out of scope:**
- New features beyond #1599.
- Raising a Jobs budget.
- Any new Python module unless Don approves one (Question 1).
- Anything the ledger marks done, duplicate or obsolete.

## Measured at the start (develop 62814175)

**UI metrics** (`cd ui && node scripts/arch-metrics.mjs`): `files_over_size_ceiling 0, complex_functions 13, long_functions 0, import_cycles 0, modules_in_import_cycles 0, prefix_literals 0, a11y_suppressions 1, raw_error_toasts 0, empty_outside_loadstate 0, unwatched_reads 0, window_dialogs 0, unguarded_mutations 0, banned_terms 0, raw_media 1, duplicate_blocks 16`. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only `complex_functions: 11 -> 13`.
- `raw_media` 1: `src/lib/job/ChainSegments.svelte:238 video`.
- `duplicate_blocks` 16, in files these tasks touch: `EditorPage:3`/`:223`, `PromptEditorPage:4`/`:170`, `VariablesForm:179`, `PipelineOptions:261`, `RequestPage:252`, `NoteList:190`, `PlanChanges:107`, `RequestHeader:187`, `WorkflowsPage:345`, `PromptsPage:126`, `LoadState:70`.

**Python** (`$PY scripts/arch_metrics.py`): `modules 281`, `claude_md_lines 129`, `modules_over_size_ceiling 5`, `complex_functions 2`, `duplicate_blocks 0`.

| File | Lines | Cap |
|---|---|---|
| `dw/server/app.py` | 292 | 292 |
| `dw/server/api_models.py` | 1105 | 1105 |
| `dw/server/job_history.py` | 1000 | 1000 (Question 1) |
| `dw/server/jobs.py` | 1139 | 1139 |
| `dw/server/routes/jobs.py` | 1033 | 1033 |
| `dw/server/routes/library.py` | 1000 | 990 after Task 7 |
| `dw/server/routes/assets.py` | 673 | 700 |
| `dw/server/routes/gallery.py` | 691 | 700 |
| `dw/server/outputs.py` | 855 | 900 |
| `dw/server/exports.py` | 506 | 510 |
| `dw/server/changes.py` | 255 | 255 |
| `dw/ledger_views.py` | 639 | 639 |
| `dw/security.py` | 922 | 930 |
| `dw/introspection.py` | 897 | 910 |
| `dw/plan.py` | 1011 | 1011 |

**UI sizes** (limit 600):

| File | Lines |
|---|---|
| `app.css` | 600 (full; Task 7 splits it) |
| `patterns.css` | 517 |
| `frames.css` | 111 |
| `api.ts` | 598 |
| `App` | 578 |
| `Sidebar` | 559 |
| `editorShell.svelte.ts` | 552 |
| `ModelsPage` | 543 |
| `OverviewPage` | 542 |
| `AssetsPage` | 520 |
| `EditorPage` | 502 |
| `StepEditor` | 495 |
| `JobPage` | 492 |
| `router.svelte.ts` | 483 |
| `WorkflowPage` | 470 |
| `PromptEditorPage` | 461 |
| `GalleryPage` | 456 |
| `RequestItem` | 450 |
| `MediaViewer` | 429 |
| `RequestPage` | 417 |
| `ItemNote` | 399 |
| `planStore.svelte.ts` | 380 |
| `VariablesForm` | 377 |
| `JobResults` | 374 |
| `EventLog` | 367 |
| `JobsPage` | 334 |
| `ComparePage` | 333 |
| `ReferenceField` | 324 |
| `MediaPicker` | 290 |
| `resource.svelte.ts` | 272 |
| `jobsListing.ts` | 228 |
| `OutputTile` | 222 |
| `destructive.ts` | 209 |
| `action.svelte.ts` | 111 |
| `format.ts` | 101 |
| `arch-metrics.mjs` | 901 |
| `eslint.config.js` | 185 |
| `index.html` | 112 |

**Prettier** (`npx prettier --check src e2e scripts`): 3 files, `src/lib/observed.test.ts`, `src/lib/ui/Segmented.test.ts`, `scripts/arch-metrics.test.ts`. **Ruff isort** on `ui/e2e/serve_fixture.py`: 1 error.

**Budgets** (`ui/e2e/performance.spec.ts`): entry 300 KB, Overview first load 420 KB, JSON view 1,400 KB (last measured 1,350.6 KB).

**Issues closable now:** #1611. Every item in it is done (2f5f72d8, 72453254, db946501).

## Questions for Don

Each task is written for the recommended answer.

1. **#1607, the `job_history.py` split.** The file is at its 1,000-line ceiling. Phase 4 reached that ceiling by compressing unrelated functions, and Task 8's `run_id` index and query reorder need lines in it. **Recommended:** approve one new module, `dw/server/job_prompts.py`, holding `submitted_prompts` and the prompt memo that `outputs.py` keeps today (`job_prompts`, `run_prompts`, `_PROMPT_MEMO`). `modules` would go to 282 in Task 8 and the baseline moves with it. The alternative is that Task 8 pays its lines inside the file. That repeats the pattern #1607 objects to, and the docstrings phase 4 shortened stay shortened.
2. **#1599's scope.** **Recommended:**
   - Lists show one relative "changed" time, with the exact time on hover and focus. Detail views show created and changed exactly.
   - The nouns are: workflow, prompt, asset, output, workspace, job, request, note, model.
   - LoRAs show a time only where their listing already has one.
   - The server adds times only through an opt-in listing parameter, `times=true`, on `/api/workflows`, `/api/prompts` and `/api/workspaces`. That is the `prompts=true`/`media=true` pattern, so MCP's listings stay byte for byte.
   - Workflows, Prompts and Assets offer **newest first**.
   - "by the agent" appears only on ledger notes and request revisions.
   - `formatDuration` gains a day unit (`1d 02h`), which closes #1609's day-unit item. The glossary row changes with it.

   The alternative is plain additive fields that MCP also returns.
3. **Two server fields the UI's links and plan line need** (#1600):
   - `WorkerInfo.workspace` (the running job's workspace, for `WorkerList`'s link);
   - `host_memory.inherited_from` in the plan (the example a host-RAM figure is inherited from).

   Both are additive, but `get_server` and the MCP validate tools return these models too. **Recommended:** add both as ordinary optional fields (`sometimes(...)`), because a new key is not a behaviour change and both carry facts MCP callers can use. The alternative is to leave both items open.
4. **Shift+Ctrl/Cmd+S and Shift+Ctrl/Cmd+Enter** act like the unshifted keys today (#1600). **Recommended:** intended, since Shift (and Caps Lock) should not turn Save off. Task 2 pins both with a test.
5. **"the shared library" or "the shared asset library"** (#1601). **Recommended:** "the shared asset library" in every sentence. "Shared library" is also what the prompt and LoRA libraries are, and the glossary row becomes that term.
6. **The viewer's open-in-new-tab link** (#1605). **Recommended:** don't restore it, because Download covers it. The ledger row closes as won't-do.
7. **The JSON view's 3.5% budget margin** (#1624). **Recommended:** accept it. There is no task, and a Monaco upgrade re-measures.
8. **The JSON editor's failure copy** is phase 7's fixed copy, and it names one cause for every failure (#1624). **Recommended:** `The JSON editor couldn't load. Reload the page; if that doesn't help, check that the server is running.` It mirrors the loading shell's line.

## Global constraints

**MCP**
- MCP behaviour is unchanged unless it is opt-in. Server changes are fixes (Tasks 1, 8) or follow Questions 2 and 3.
- The keep_output 409 keeps its message word for word, inside the `{code, message}` detail shape `dw/server/preconditions.py:59` already uses. Task 1 pins that MCP's `keep_output` error text is unchanged.

**OpenAPI**
- OpenAPI is regenerated only with the pinned overlay: `PYTHONPATH=/private/tmp/claude-501/-Users-don-src-dkackman-diffusers-workflow/c38aa355-7a1b-44f4-aee3-aaaef6ebc96d/scratchpad/p3/openapi-overlay $PY scripts/dump_openapi.py && cd ui && npm run gen:api`.
- Never hand-edit `ui/src/lib/generated/`.

**Environment**
- `PY=/Users/don/src/dkackman/diffusers-workflow/venv/bin/python`. The worktree has no venv.

**Size limits**
- Python `modules` stays 281. No task adds a module, and a test file is not one. Question 1's answer is the only exception. A task that finds it needs one stops and asks Don.
- `claude_md_lines` stays 129. `ui/CLAUDE.md` stays 12 lines: rewritten, never grown.
- The caps in the tables above hold. Measure every file a task touches with `wc -l` before and after.
- No page over 600 lines. `app.css` is full until Task 7 splits it, so new global rules go in `patterns.css` or the file Task 7 creates.
- No function over complexity 10 or 60 lines, in TS or Python.

**The UI metrics gate**
- Before Task 1, run `cd ui && node scripts/arch-metrics.mjs > "$SCRATCH/ui-fu.json"`.
- Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-fu.json"`, which prints nothing.
- `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only `complex_functions: 11 -> 13`.
- Every 0 stays 0. A task that lowers a count lowers that key in `baseline.json` in the same commit.
- Task 19 widens metrics. A newly counted hit is fixed in the same task, so the count it reports stays at its current value.

**Checks every task runs**
- UI, from `ui/`: `npm run check`, `npm run lint`, and the full `npm test` (vitest) before committing.
- e2e always serves the built UI: `npm run build && DW_E2E_PYTHON=$PY npx playwright test <the specs the task touches>`.
- After any copy change, also run `e2e/smoke.spec.ts`. Its test at `:36` is #1603; rerun it alone with `-g "introspected arguments"`.
- A task that changes a URL the page calls runs every spec that matches it (`grep -ln "<path>" e2e/*.ts e2e/journeys/*.ts`).
- Python: `$PY -m pytest <files> -q` with `DW_STRICT_RESPONSES=1`, then `$PY scripts/arch_metrics.py`.

**Fixtures and journeys**
- Never rely on sleeps or spec order: wait on a condition.
- A workspace or asset a spec creates gets a unique name and is deleted in `finally`.
- Assert only what holds whatever ran before.

**Replacing a component**
- When a component replaces or absorbs another, list each capability the old one had and pin each with a test. Task 10's missing-file helper lists its four users' capabilities.

**Budgets**
- Jobs budgets (entry 300 KB, Overview 420 KB, JSON view 1,400 KB) are raised only with Don. A task that goes over stops and reports.

**Staging and commits**
- Stage files by name. Never `git add -A ui` (Don edits `ui/package.json`; a dependency change is a question for Don). Never add or force-add `.superpowers/`. Never `git stash`.
- Headlines use a scope (`ui(editor): …`, `server(assets): …`, `docs(…)`), never `fix: #N`.
- Name issues and rows in the body as `refs #1600 (1600.3, 1600.16)`.
- Each commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Security**
- No `eval`, `exec` or `shell=True`.
- Paths go through `validate_path` / `validate_output_path`. Task 8 changes what `validate_path` refuses: it may accept `..` inside a segment, never as a segment. `tests/test_security.py` must still pass, and Task 8 adds traversal cases.

**Accessibility**
- WCAG contrast is guidance, never a gate. Names, focus, `aria-*` and live regions are in scope.

**Docs**
- `docs/SERVER.md`, `docs/ARCHITECTURE.md`, `ui/GLOSSARY.md` and `ui/CLAUDE.md` are edited by the task that changes the behaviour.

**Copy fixed by this plan.** Use it exactly. `<…>` is filled in, and `<a|b>` picks by count.
- **Editor, file gone** (a re-read finds the open file deleted): banner `<path> was deleted on the server. Save to write it back, or leave to discard your edits.`. The stale-write dialog for a gone file is titled `<path> was deleted` (not "changed").
- **Editor, listing failed:** an `ErrorLine` with `doing="list the workflows"` under the file bar.
- **describeError:** unchanged titles. The detail is kept whenever the title did not come from the message.
- **JobHeader Replace:** title `Replace the export?`, message as today, `confirmLabel: 'Replace'`, `danger: true`.
- **Notes:** a resumed draft over a newer note raises the existing `ConflictNotice` copy. The JobPage "Updated by an agent or another tab" line stays hidden for `CHANGED_MARK_MS` after Keep mine / Use theirs.
- **Router:** `#/ws/<name>//<section>` opens `<section>`. The Not-found crumb is plain text `Page not found`.
- **Jobs:** a move button while moving has `aria-busy="true"` and `aria-disabled="true"`. A banner's Rerun is busy only for its own batch.
- **Ledger and note deletes** (`destroyConfirm` with `place: 'ledger'`): `This removes it from the ledger.`. A request delete: `Its plan, notes and history are deleted from the ledger. Its jobs and their outputs are kept.`.
- **PromptsPage chip titles:** `written for <model> - click to filter, click again to show all` and `click to filter by this tag, click again to show all`.
- **Timestamps (Task 21):**
  - list `changed <relative>`, with the `title` and `datetime` exact (`formatExact`);
  - detail `Created <exact>` and `Changed <exact>`;
  - by the agent `changed <relative> by the agent`;
  - the duration day unit `1d 02h`.

## Review focus

1. **A document deleted while open.** The banner says "deleted", Save writes it back as a new file (`createOnly`), and no toast fires for a broken JSON draft. Pinned in Task 2: `a re-read that finds the file gone says it was deleted`, `a save over a deleted file creates it`, `a newer read never toasts a draft that does not parse`.
2. **An Undo pressed while its action is still running, then a workspace or request switch.** The revert checks its target again after the wait and says the `away` line. Pinned in Task 5: `an undo that waited re-checks its target`.
3. **A filename with `..` inside a segment** (`take..2.mp4`). It serves, gets its duration and keeps its prompt, while `../x`, `a/../b` and `%2e%2e` are still refused. Pinned in Task 8: `test_a_dotted_name_is_served`, `test_a_parent_segment_is_still_refused`.
4. **A row that changes again while marked, or is live and changed.** It flashes again and keeps its live bar. Pinned in Task 3: `a second change re-marks the row`, `a live row keeps its bar while marked`.
5. **The first class import under concurrent requests.** Several editors asking `describe` at once neither wedge nor serialise forever. Pinned in Task 23: `test_concurrent_first_imports_finish`.

---

## Release-blocker tasks (#1598, #1600)

### Task 1: Errors say what failed, and the questions that overwrite are titled

Closes 1600.3, 1600.15, 1600.16, 1600.36, 1600.37.

**Files:**
- Modify: `ui/src/lib/apiErrors.ts` (`describeError`), `ui/src/lib/confirm.svelte.ts` (`firstSentence`), `ui/src/lib/job/JobHeader.svelte`, `ui/src/lib/CachesPanel.svelte`, `ui/src/lib/pages/ModelsPage.svelte` (drop `danger` from the three trash triggers: their confirms are server-cache questions, `destructive.ts:157-159`), `ui/src/lib/gallery/saveAsAsset.ts` (match `code === 'exists'` only), `dw/server/routes/assets.py` (`keep_output_as_asset` 409 detail `{"code": "exists", "message": <today's string>}`)
- Test: `ui/src/lib/apiErrors.test.ts`, `ui/src/lib/confirm.test.ts` (or the file holding `firstSentence`'s tests: `grep -rln firstSentence src --include=*.test.ts`), `ui/src/lib/job/JobHeader.test.ts`, `ui/src/lib/CachesPanel.test.ts`, `ui/src/lib/gallery/saveAsAsset.test.ts`, `tests/test_assets_keep.py` (or the file holding keep tests: `grep -rln "assets/keep" tests`), `tests/test_mcp_assets.py` (the same way)

**Interfaces:**
- `describeError(e, what?)` keeps its return type. The rule becomes: `detail` is set when `message` is non-empty and differs from `title`. The old rule required `message.length > title.length`, and a title derived from the message (`classify` returns it equal) drops it.
- `firstSentence(message)` cuts at the first `. `, `! ` or `? ` followed by a capital letter, or at a newline. A message with none is whole.

- [ ] **Step 1: Write the failing tests.**
  - `apiErrors.test.ts`: `a short engine message keeps its detail` (`new Error("KeyError: 'x'")` gives title `Something went wrong`, detail `KeyError: 'x'`) and `a title taken from the message has no detail`.
  - `confirm.test.ts`: `a filename's dot does not end the title` (`Delete take.v2.png? This removes it on disk.` gives `Delete take.v2.png?`).
  - `JobHeader.test.ts`: `the Replace question is titled and in danger`.
  - `CachesPanel.test.ts`: `a cache trash button is not drawn in danger`.
  - `saveAsAsset.test.ts`: `an exists code asks; the wording alone does not`.
  - pytest: `test_keep_output_conflict_has_code_exists`, and `test_mcp_keep_output_conflict_text_is_unchanged` (the MCP error string equals today's).
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/apiErrors.test.ts src/lib/confirm.test.ts src/lib/job/JobHeader.test.ts src/lib/CachesPanel.test.ts src/lib/gallery/saveAsAsset.test.ts` and the two pytest files. Expected: FAIL.
- [ ] **Step 3: Implement.** If MCP's text changes because its client renders a dict detail differently, stop and report.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/journeys/find-and-reuse.spec.ts`, and `$PY -m pytest tests/test_security.py tests/test_api_contract.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(errors): a short engine message keeps its detail; overwrite questions are titled; keep_output says exists`.

### Task 2: The editors keep up with the file on the server

Closes 1598.1, 1598.2, 1598.3, 1598.7, 1598.18, 1598.19, 1600.35b, 1602.16b.

**Files:**
- Modify: `ui/src/lib/editorShell.svelte.ts` (`remote` gains `'gone'`; `#arrived` stops flushing through the toasting path), `ui/src/lib/ui/RemoteChanged.svelte` (the gone banner), `ui/src/lib/documentSave.ts` (the gone title), `ui/src/lib/pages/EditorPage.svelte`, `ui/src/lib/pages/PromptEditorPage.svelte` (an effect on `shell.remoteLoads` re-reads `enhancer.idea`; `remove()` checks busy after the warning read), `ui/src/lib/stepModes.svelte.ts` (`retain(names: string[])`, called on a remote load), `ui/src/lib/editor/StepEditor.svelte` (a `destroyed` flag its async class lookups check), `ui/src/lib/editor/workflowFile.ts`, `ui/src/lib/promptFile.ts` (one `withSavedEntry` helper in `workflowFile.ts`, which `withSavedPrompt` calls)
- Test: `ui/src/lib/editorShell.test.ts`, `ui/src/lib/pages/EditorPage.test.ts`, `ui/src/lib/pages/PromptEditorPage.test.ts`, `ui/src/lib/stepModes.test.ts` (create if none)

**Interfaces:**
- `RemoteRead` gains `gone?: true`. Each page's `read` returns `{ gone: true }` on a 404.
- `#arrived` sets `remote = 'gone'`. While `'gone'`, `preconditions(path)` returns `{ createOnly: true }`.
- `unsaved()` gains `{ quiet = false }`. `#arrived` calls `unsaved({ quiet: true })`, which flushes with `applyJson(raw, { quiet: true })`.
- `StepModes.retain(names)` keeps the modes of names still present and gives a new name its default.

- [ ] **Step 1: Write the failing tests.**
  - `editorShell.test.ts`: `a re-read that finds the file gone says it was deleted`, `a save over a deleted file creates it`, `a newer read never toasts a draft that does not parse`, `Shift+Ctrl+S saves and Shift+Ctrl+Enter runs` (Question 4).
  - `EditorPage.test.ts`: `a failed workflow listing says so`, `a remote reload keeps the steps' modes by name`.
  - `PromptEditorPage.test.ts`: `a remote reload refreshes the enhancer idea`, `a second Delete while the warning loads does nothing`.
  - `stepModes.test.ts`: `retain keeps present names and defaults new ones`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editorShell.test.ts src/lib/pages/EditorPage.test.ts src/lib/pages/PromptEditorPage.test.ts src/lib/stepModes.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `editorShell.svelte.ts` stays ≤ 580 lines, `EditorPage` ≤ 560 and `PromptEditorPage` ≤ 520.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/journeys/editor-safety.spec.ts e2e/journeys/agent-edits.spec.ts e2e/json-view.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): a deleted file is said, a newer read never toasts a draft, and remote reloads refresh what they own`.

### Task 3: Change marks show every change, beside the live bar

Closes 1598.4, 1598.5a-d, 1598.6, 1598.13, 1598.17, 1598.22c.

**Files:**
- Modify: `ui/src/lib/resource.svelte.ts` (`changeMarks`), `ui/src/app.css` (the `.row-changed` rule), `ui/src/frames.css`, `ui/src/lib/request/planStore.svelte.ts` (`#refused` passes the fresh read through `onread` as quiet), `ui/src/lib/request/announce.ts`
- Test: `ui/src/lib/resource.svelte.test.ts` (or the file holding resource tests: `grep -rln "from './resource.svelte'" src --include=*.test.ts`), `ui/src/lib/pages/RequestPage.test.ts`, `ui/src/lib/request/announce.test.ts`, `ui/src/lib/request/planStore.svelte.test.ts`, `ui/e2e/changed-mark.spec.ts`

**Interfaces:**
- `changeMarks().mark(keys)` clears to `NONE` and sets `keys` in the next microtask, so a re-marked row's animation restarts. Its timer is `poll(() => clear, CHANGED_MARK_MS, { immediate: false, once: true })`; if `poll` has no `once`, add it to `poll`, where the hidden-tab pause lives.
- The mark is drawn as an overlay (`::after`) on rows too, never a `box-shadow`, so `.row-live`'s inset bar stays. The tile and card overlay in `frames.css` becomes the only rule, and `app.css:481-484` goes.
- The overlay's colour is `var(--live)` over a 1px `var(--bg)` edge (`box-shadow: 1px 0 0 var(--bg)` on the `::after`), so it reads over a bright picture. Check by eye on the gallery in both themes (`node scripts/ux-capture.mjs`).
- `announce.ts`: a rename whose `was` is empty or null reads `The agent named <title>.`

- [ ] **Step 1: Write the failing tests.**
  - resource tests: `a second change re-marks the row`, `marks clear after CHANGED_MARK_MS`, `the mark timer waits while the tab is hidden`, `a question change drops the marks of the old question` (a page-level dependency change between two quiet reads).
  - `RequestPage.test.ts`: `an item an agent edited is marked, and the mark clears` (extends `:528`), `a stale retry's fresh read marks the agent's change`.
  - `announce.test.ts`: `naming an untitled item is said`.
  - `changed-mark.spec.ts`: `a live row keeps its bar while marked`, `a tile's mark is one bar`.
- [ ] **Step 2: Run** the vitest files. Expected: FAIL.
- [ ] **Step 3: Implement.** `app.css` must not grow.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/changed-mark.spec.ts e2e/journeys/agent-edits.spec.ts e2e/requests.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(freshness): a mark restarts on a second change, keeps the live bar and pauses with the tab`.

### Task 4: One read per question

Closes 1598.8, 1598.9, 1598.10, 1598.11, 1598.12, 1598.20.

**Files:**
- Modify: `ui/src/lib/pages/OverviewPage.svelte` (its `relistWorkspaces` resource becomes `lazy: true`, relying on the Sidebar's, which is always mounted), `ui/src/lib/pages/AssetsPage.svelte` (an upload adopts `[...held.assets, uploaded]`; a bulk delete adopts the held list minus the deleted), `ui/src/lib/pages/JobPage.svelte` (`ranRead` watches nothing and reloads when `job.status` or `job.realized` changes), `ui/src/lib/pages/RequestPage.svelte` (`const interval = $derived(refreshInterval(store.read))`, `alsoLive: () => interval`), `ui/src/lib/request/planStore.svelte.ts` (`refresh(isCurrent?: () => boolean)`: no `take` when it is false), `ui/src/lib/resource.svelte.ts` (the load receives `current: () => boolean` as its first argument; existing loads ignore it), `ui/src/lib/freshness.test.ts` (`HELPER_EXCEPTIONS` keyed `file:helper`)
- Test: `OverviewPage.test.ts`, `AssetsPage.test.ts`, `JobPage.test.ts`, `RequestPage.test.ts`, `planStore.svelte.test.ts`, `freshness.test.ts`

- [ ] **Step 1: Write the failing tests.**
  - `OverviewPage.test.ts`: `a workspaces move lists the workspaces once` (Sidebar and Overview mounted together).
  - `AssetsPage.test.ts`: `an upload marks nothing and keeps an agent's fresh mark`, `a bulk delete adopts`.
  - `JobPage.test.ts`: `a jobs move of another job reads no workflow`.
  - `RequestPage.test.ts`: `a read that keeps the interval does not re-arm the poll`.
  - `planStore.svelte.test.ts`: `an overtaken refresh is not taken`.
  - `freshness.test.ts`: `an exception names its helper`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate (`unwatched_reads` 0), `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/workspaces.spec.ts e2e/requests.spec.ts e2e/journeys/agent-edits.spec.ts e2e/journeys/find-and-reuse.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(freshness): one workspace listing, adopted uploads, and no read for a move that is not this job's`.

### Task 5: Notes and Undo never write over what changed meanwhile

Closes 1600.7, 1600.8, 1600.9, 1600.10, 1600.11, 1605.9.

**Files:**
- Modify: `ui/src/lib/destructive.ts` (`undoToast` re-checks `target` when the revert starts, not only when pressed), `ui/src/lib/action.svelte.ts` (`turn.after` re-runs the target check through a `ready?: () => boolean` argument), `ui/src/lib/noteDraft.svelte.ts` (stores `base`, the saved note's text, beside the kept text), `ui/src/lib/ui/NoteEditor.svelte` (Cancel `disabled={saving || pending}`; a resumed draft whose `base` differs from `saved()` raises the conflict notice), `ui/src/lib/pages/JobPage.svelte` (`noteResolvedAt`; `noteMoved` false within `CHANGED_MARK_MS` of it), `ui/src/lib/request/conflict.ts` and `planStore.svelte.ts` (`culprit` for `moved_on` picks the newest write whose diff names the item's state), `ui/src/lib/pages/GalleryPage.svelte` and `ui/src/lib/picks.svelte.ts` (a star Undo writes only when the file's star is still what this wrote; otherwise `Nothing undone: the star was changed since.`)
- Test: `destructive.test.ts`, `action.svelte.test.ts` (or the action tests' file), `noteDraft.svelte.test.ts`, `ui/NoteEditor.test.ts`, `JobPage.notes.test.ts`, `request/conflict.test.ts`, `GalleryPage.test.ts`

- [ ] **Step 1: Write the failing tests.**
  - `action.svelte.test.ts`: `an undo that waited re-checks its target`.
  - `noteDraft.svelte.test.ts`: `a kept draft remembers the note it answered`.
  - `NoteEditor.test.ts`: `a resumed draft over a newer note asks`, `Cancel waits for the Replace question`.
  - `JobPage.notes.test.ts`: `a resolved note does not say it was updated`.
  - `conflict.test.ts`: `moved_on names the write that moved the item`.
  - `GalleryPage.test.ts`: `a star Undo leaves another writer's star`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/requests.spec.ts e2e/journeys/review-a-request.spec.ts e2e/journeys/consistency.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(undo): an undo re-checks its target after it waits; drafts remember their note; a star Undo spares another's`.

### Task 6: Routes, Jobs buttons and the links that need the job's workspace

Closes 1600.20, 1600.21, 1600.25, 1600.26, 1600.28c, 1600.30c, 1600.31b.

**Files:**
- Modify:
  - `ui/src/lib/router.svelte.ts`: `scrolls` mirrored to `sessionStorage` key `dw-scrolls`, keyed by history entry index, newest `SCROLLS_KEPT`, wrapped in try/catch.
  - `ui/src/lib/routes.ts`: `parseView` drops empty parts after the workspace before matching.
  - `ui/src/lib/Breadcrumb.svelte`: a missing view's crumb is a `span`.
  - `ui/src/lib/assets/ShadowedAssets.svelte`: `GroupHeading disabled={filterActive}` with FolderGroups' title.
  - `ui/src/lib/job/JobRow.svelte` and `ui/src/lib/pages/JobsPage.svelte`: `moving.busy` passed as a `busyId`; `rerunning` keyed per batch, through a `busyBatch` state.
  - `ui/src/lib/WorkerList.svelte`: `jobHref(w.current_job, w.workspace ?? workspace.current)`.
  - `ui/src/lib/plan.ts`: `inherited from <inherited_from>`.
  - `dw/server/api_models.py`: `WorkerInfo.workspace`, `PlanHostMemory.inherited_from`, both `sometimes(...)`. Pay in docstrings to stay at 1105.
  - The worker-info and plan producers: `grep -rn "current_job" dw/server/*.py` and `grep -rn "host_memory" dw/plan.py`.
  - OpenAPI regenerated with the overlay.
- Test: `router.svelte.test.ts`, `routes.test.ts`, `Breadcrumb.test.ts` (create if none), `assets/ShadowedAssets.test.ts`, `JobsPage.test.ts`, `WorkerList.test.ts`, `plan.test.ts`, `tests/test_server.py` (worker info), `tests/test_plan*.py` (`grep -rln host_memory tests`), `tests/test_api_contract.py`

Done only per Question 3. If Don declines, the two field rows stay open and the rest ships.

- [ ] **Step 1: Write the failing tests.**
  - `router.svelte.test.ts`: `a reload restores the entry's scroll`, `storage that throws keeps scrolls in memory`.
  - `routes.test.ts`: `a doubled slash keeps the section`.
  - `Breadcrumb.test.ts`: `the not-found crumb is not a link`.
  - `ShadowedAssets.test.ts`: `the toggle is disabled while filtering`.
  - `JobsPage.test.ts`: `a moving job's buttons are busy`, `one batch's rerun leaves the other banners`.
  - `WorkerList.test.ts`: `a worker's job links its own workspace`.
  - `plan.test.ts`: `an inherited host-RAM figure names its example`.
  - pytest: `test_worker_info_names_the_jobs_workspace`, `test_inherited_host_memory_names_its_example`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement**, then regenerate OpenAPI.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/journeys/consistency.spec.ts e2e/restart-banner.spec.ts e2e/smoke.spec.ts`, `$PY -m pytest tests/test_api_contract.py tests/test_mcp_server.py -q`, and `wc -l dw/server/api_models.py` (≤ 1105). Expected: PASS.
- [ ] **Step 5: Commit** `ui(routes): reload keeps scroll, a doubled slash keeps its section; Jobs buttons say they are busy; links name the job's workspace`.

### Task 7: #1598's code health and its two missing fixtures

Closes 1598.14, 1598.16, 1598.21, 1598.22a, 1598.22b, 1598.23, 1598.24.

**Files:**
- Create: `ui/src/rows.css`, holding `app.css`'s row rules (`.row-live`, `.row-changed`, `.sr-only` and the rows block around `:470-500`), imported by `src/main.ts` beside `app.css`. A CSS file is not a Python module.
- Modify:
  - `ui/src/app.css` (≤ 560);
  - `dw/server/routes/library.py`: pay 10 lines from its own comments, never moving code out, ≤ 990;
  - `dw/server/exports.py`: `inline_json` uses `WORKFLOW_FILE_NAME`, `MANIFEST_FILE_NAME` and `JOB_FILE_NAME`; add `JOB_FILE_NAME` beside the other two if absent;
  - `dw/server/changes.py:19`, wrapped;
  - `tests/test_changes.py`: the swallowed-raise test asserts the warning was logged (`caplog`) and that the download finished;
  - `ui/e2e/journeys/agent-edits.spec.ts`: the models case re-seeds `e2e/tiny` in `finally` through the fixture's seeding helper, or `test.skip` with `e2e/tiny is not seeded` when it is absent;
  - `ui/e2e/serve_fixture.py`: one asset and one workflow with a proof picture, from `solid_png`.
- Test: `ui/e2e/changed-mark.spec.ts` gains `an asset tile's change mark shows` and `a workflow card's change mark shows`; `tests/test_exports.py` (`grep -rln inline_json tests`)

- [ ] **Step 1: Write the failing tests** named above, and `test_inline_json_reads_the_named_files`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/changed-mark.spec.ts e2e/journeys/agent-edits.spec.ts --repeat-each=2`, `$PY -m pytest tests/test_changes.py tests/test_server.py -q`, `$PY scripts/arch_metrics.py`, and `wc -l ui/src/app.css dw/server/routes/library.py`. Expected: PASS, with app.css ≤ 560 and library.py ≤ 990.
- [ ] **Step 5: Commit** `ui(css): rows get their own file; server: export names, a logged swallowed raise; e2e: a re-runnable models case and proof fixtures`.

## User-visible bugs

### Task 8: Prompt and media opt-ins are right for every name and every job

Closes 1607.1-1607.8.

**Files:**
- Create (Question 1, if approved): `dw/server/job_prompts.py`, holding `submitted_prompts` (moved from `job_history.py`), `job_prompts`, `run_prompts` and `_PROMPT_MEMO` (moved from `outputs.py`).
- Modify:
  - `dw/security.py`: `DANGEROUS_PATTERNS`' `\.\.` becomes a segment match, `(^|/)\.\.(/|$)`, and `validate_path`'s containment check still decides.
  - `dw/server/outputs.py` (`resolve_output_file`).
  - `dw/server/job_history.py`: `CREATE INDEX IF NOT EXISTS jobs_run_id ON jobs (run_id)`; the `WHERE` puts `run_id IN (...)` before `json_valid`.
  - The prompt memo: an invalid row memoises `None`; every read is inside `_PROMPT_MEMO_LOCK`; a run id shared by jobs takes the newest `created_at`.
  - `dw/server/routes/gallery.py`, `routes/jobs.py`, `routes/assets.py`: restore the phase-4 descriptions from `git show 78b66483^:<file>` where the cap allows.
  - OpenAPI regenerated.
- Test: `tests/test_security.py`, `tests/test_outputs.py` (`grep -rln resolve_output_file tests`), `tests/test_gallery_listing.py` (`grep -rln "prompts=true" tests`)

- [ ] **Step 1: Write the failing tests:**
  - `test_a_dotted_name_is_served` (`take..2.mp4` gives 200 and a `duration_seconds`, over HTTP and MCP `media=true`);
  - `test_a_parent_segment_is_still_refused` (`../x`, `a/../b`, `%2e%2e/x`, `..`);
  - `test_a_shared_run_id_takes_the_newest_job`;
  - `test_an_invalid_row_is_read_once`;
  - `test_gallery_prompts_cover_only_the_returned_page`;
  - `test_the_run_id_index_exists`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** Without Question 1's module, `job_history.py` stays ≤ 1000, paid in its own comments.
- [ ] **Step 4: Run** `$PY -m pytest tests/test_security.py tests/test_security_*.py tests/test_jobs_listing.py tests/test_api_contract.py tests/test_mcp_server.py -q`, `$PY scripts/arch_metrics.py` (`modules` 281, or 282 with the module), and `cd ui && npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/journeys/find-and-reuse.spec.ts e2e/security.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `server(outputs): a dotted file name is served; prompts are read once, under the lock, newest job first`.

### Task 9: The media viewer on touch, under full screen, and after a delete

Closes 1605.1, 1605.2, 1605.3, 1605.4, 1605.6, 1605.7, 1605.14, 1605.18.

**Files:**
- Modify: `ui/src/lib/ui/MediaViewer.svelte` (`setPointerCapture` on pointerdown; two-pointer pinch through `mediaViewer.ts`), `ui/src/lib/mediaViewer.ts` (`pinchZoom(start: [P, P], now: [P, P], zoom: number): number`), `ui/src/lib/confirm.svelte.ts` and the toast host (portal into `document.fullscreenElement ?? document.body`), the viewer's step (a neighbour that 404s shows `MissingFile` and drops the stale notice), `ui/src/lib/ui/Menu.svelte` or the More menu's owner (focus returns to the More button, or to the viewer's close button when the trigger has unmounted)
- Test: `ui/src/lib/ui/MediaViewer.test.ts` (an `afterEach` that resets `history.state`), `mediaViewer.test.ts`, `ui/e2e/gallery-findings.spec.ts`

- [ ] **Step 1: Write the failing tests:** `a pan ends on a pointerup outside the viewer`, `two pointers zoom by their spread`, `a confirm opened in full screen shows`, `stepping to a deleted neighbour says it is missing`, `More closes on Escape and returns focus`, `closing Use as input on a phone focuses the viewer`, `a workspace change closes the viewer`, and the `history.state` reset.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** `MediaViewer` ≤ 480 lines.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/gallery-findings.spec.ts e2e/journeys/find-and-reuse.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(viewer): pinch and pan on touch, dialogs in full screen, and a deleted neighbour said`.

### Task 10: Gallery, Assets and tiles: one missing-file helper, sorting and names that differ

Closes 1605.8, 1605.10, 1605.11, 1605.12a-b, 1605.13a-b, 1605.15, 1605.16, 1605.17, 1606.5b, 1606.6, 1606.7, 1606.8, 1606.9, 1606.13, 1610.1.

**Files:**
- Create: `ui/src/lib/missingFile.svelte.ts`, exporting `missingFile(opts: { open: () => string | null; known: () => boolean; names: () => readonly string[] }): { readonly missing: string | null; dismiss(): void }`. Its capabilities, pinned: a stray `open=` is said, a workspace change resets it, dismiss clears it, and a file that arrives later clears it. It replaces the four copies in `GalleryPage`, `AssetsPage`, `JobResults` and `ComparePage`.
- Modify:
  - `ui/src/lib/gallery/order.ts`: name sort through `Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })`.
  - `ui/src/lib/pages/AssetsPage.svelte`: `viewedOrigin` set in an `$effect`, not the derived; the `media=true` read limited to the newest 200 assets.
  - `ui/src/lib/names.ts`: `distinctLeaves(paths: string[]): Map<string, string>`, the shortest unique tail per path, used by `UseAsInput` and `OutputTile`'s caption and name.
  - `ui/src/lib/ui/RecipePanel.svelte` and `ui/src/lib/recipe.ts`: no Job link when the job read 404s; no job-load `ErrorLine` for a file with embedded metadata; `workflow_name` null falls back to the file's run, never the embedded id; `copyRows` typed without the cast.
  - `ui/src/lib/ui/OutputTile.svelte`: active, hover and picked distinct (picked keeps its ink edge, active an accent outline, hover the panel tint); no play badge on a shadowed tile; a note mark on asset tiles; an audio cover of 72 px.
- Test: `missingFile.svelte.test.ts`, `gallery/order.test.ts`, `names.test.ts`, `RecipePanel.test.ts`, `OutputTile.test.ts`, `AssetsPage.test.ts` (`open= under a q= that filters it out`), `GalleryPage.test.ts` (`Use as input's choose path`, `its dialog from More`, `the deep-linked folder is aria-expanded`)

- [ ] **Step 1: Write the failing tests:** each capability above, `take2 sorts before take10`, `two same-leaf workflows are told apart`, `a job that left history is not linked`, and `an embedded image shows no job error`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** The metrics gate's `duplicate_blocks` must not rise.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/gallery-findings.spec.ts e2e/journeys/find-and-reuse.spec.ts e2e/responsive.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(media): one missing-file helper; natural name sort; same-leaf names told apart; tile states that differ`.

### Task 11: The job page's results, errors and log say true things

Closes 1608.2-1608.13.

**Files:**
- Modify:
  - `ui/src/lib/job/JobProgress.svelte`: the h2 moves before the `details`, and the summary holds the count.
  - `ui/src/lib/job/JobViewer.svelte` and `JobResults.svelte`: the star is enabled from the listing when the metadata read fails; `MissingFile` for a stray `open=` when there are no files; the viewer's star follows the record on a quiet reload.
  - `ui/src/lib/job/jobError.ts`:
    - "Out of memory" is split into the card matcher (`CUDA|MPS|OutOfMemoryError`), with title `The card ran out of memory.`, and a host one (`Cannot allocate memory|MemoryError|Killed`), with title `The machine ran out of memory.`;
    - the gated matcher needs `GatedRepoError|gated repo`;
    - a 403 alone reads `The model repo refused access.`.
  - The step state: a step with a `step_done` event before the restart stays done.
  - `ui/src/lib/job/EventLog.svelte`: the kind column is blank when the row has no sentence of its own; a failed live read shows Refresh while running.
- Test: `JobPage.test.ts` (`the Error panel sits above Results`), `job/jobError.test.ts`, `job/EventLog.test.ts` (the unmount test's comment says what it pins, plus `a destroyed log does not read again`), `ui/e2e/job-event-log.spec.ts` (the row-height bound set to the measured ratio plus 10%, recorded in a comment)

- [ ] **Step 1: Write the failing tests:** each item, one test each, named for the behaviour.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** `JobPage` ≤ 520 lines.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/job-event-log.spec.ts e2e/chain-actions.spec.ts e2e/chain-segments.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(job): host memory is not the card's; a log row is said once; the star survives a failed read`.

### Task 12: Compare draws every kind and knows where you came from

Closes 1608.1, 1609.5, 1609.6.

**Files:**
- Modify: `ui/src/lib/pages/ComparePage.svelte` (audio and text cells through `MediaBody`, each opening the viewer; the synced transport pauses while the viewer is open), `ui/src/lib/routes.ts` (`compareHref(workspace, names, from?: { href: string; label: string })` carries `from` as `back=` in the hash; Compare's back link uses it, else the Gallery)
- Test: `ComparePage.test.ts`, `routes.test.ts`, `JobResults.test.ts` (its Compare link carries the job)

- [ ] **Step 1: Write the failing tests:** `from a job, Compare's back link returns to it`, `an audio cell opens in the viewer`, `the transport pauses under the viewer`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/gallery-findings.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(compare): every kind opens in the viewer, and the back link returns where you came from`.

### Task 13: The Jobs list's banners, filters and reads

Closes 1609.2, 1624.13, 1624.14, 1624.16.

**Files:**
- Modify:
  - `ui/src/lib/jobsListing.ts`: the live and finished reads go out together (`Promise.all`); `restartBatches` asks `GET /api/jobs/{id}/restart-batch` for a loaded tombstone's batch size, or uses the count the listing already carries (read `routes/jobs.py` first; no server change if the count is there). A batch beyond the page is found from the newest tombstone the server reports.
  - `ui/src/lib/pages/JobsPage.svelte`: no "Searching the newest" line when only `queued`/`running` are filtered.
  - The all-jobs proof: no proof slot for a job from another workspace.
- Test: `jobsListing.test.ts`, `JobsPage.test.ts`, `ui/e2e/restart-banner.spec.ts`

If counting a batch beyond the page needs a server parameter, stop and ask Don. That would be an opt-in change to `/api/jobs`.

- [ ] **Step 1: Write the failing tests:** `the two reads go out together`, `a batch partly beyond the page is counted whole`, `a live-only filter does not offer Search all`, `another workspace's job has no proof slot`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, and every spec that matches `/api/jobs` (`grep -ln "api/jobs" e2e/*.ts e2e/journeys/*.ts`). Expected: PASS.
- [ ] **Step 5: Commit** `ui(jobs): banners count the whole batch, live filters say nothing about paging, and reads go together`.

### Task 14: The run form, the reference field and Rerun

Closes 1612.2, 1612.5, 1612.8-1612.21.

**Files:**
- Modify:
  - `ui/src/lib/ui/ReferenceField.svelte` and `ui/src/lib/referenceField.ts`: an empty MIME type falls back to the extension; the blob URL is revoked on destroy; the late-upload toast reads `The upload finished after the field closed; nothing was written.` / `... after the field changed; nothing was written.` with sentence-case `Unsupported file type`; a pick previews at once; `previewOf` draws nothing for a wrong-kind `asset:`.
  - `ui/src/lib/editor/VariablesForm.svelte`: a default file reference shows its tile; a field keeps its first-rendered kind until the user edits; Advanced folds while a folded field has a value and shows `<n> set`.
  - `ui/src/lib/workflow/runForm.ts`: a prefill is not stored as a draft.
  - `ui/src/lib/plan.ts` `planSummary`: `<n> GB + unsized` when some downloads are unsized.
  - `ui/src/lib/workflow/EnhanceDialog.svelte`: Close cancels the server job through the existing cancel route.
  - `ui/src/lib/job/RerunWithChanges.svelte`:
    - a blank field sends `null` (read the rerun route for its empty value first);
    - New seed disables the seed field;
    - JSON is compared parsed.
  - `docs/ARCHITECTURE.md:249`: "a file field or a step argument's reference".
  - `ui/src/lib/freshness.test.ts`: the duplicate `RerunWithChanges` entry goes.
- Test: `ReferenceField.test.ts`, `referenceField.test.ts`, `VariablesForm.test.ts`, `runForm.test.ts`, `plan.test.ts`, `EnhanceDialog.test.ts`, `RerunWithChanges.test.ts`, `WorkflowPage.test.ts` (`Run keeps the draft`, `a workflow switch keeps each draft`, `Use writes the field and the draft`)

- [ ] **Step 1: Write the failing tests:** one per item, named for the behaviour.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** If the rerun route cannot clear a value, stop and ask Don; that is a server change.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/journeys/first-image.spec.ts e2e/journeys/chain-a-step.spec.ts e2e/expanding-text.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(run): uploads and previews keep to their field; Rerun can clear a value; prefills are not drafts`.

### Task 15: The workflow editor and the catalog

Closes 1612.1, 1612.3, 1612.6, 1612.7, 1612.22-1612.32, 1612.35, 1601.17b.

**Files:**
- Modify:
  - `ui/src/lib/editor/ValidationPanel.svelte`: warn tone when `placed.author` is non-empty; `PlanLines open` when a line is `warn`; "Fix N issues" counts only errors still in the latest answer.
  - `dw/server/routes/library.py` and `ui/src/lib/api.ts` `listWorkflows`: `inputs` is honoured under `view=compact`, paid in library.py's comments so it stays ≤ 990.
  - `ui/src/lib/workflow/catalogSearch.ts`: whole-word match, punctuation stripped, `:56` wrapped.
  - Variable names and steps:
    - `references.ts` exports `VARIABLE_NAME_FIRST`, twin-pinned to `validate_variable_name`'s pattern;
    - Enter while `isComposing` does nothing;
    - the name error has an id that the input's `aria-describedby` names;
    - `StepEditor` requires `workflow`;
    - the rename rewrite accepts `previous_result:\s*<name>`;
    - Undo anchors on the renamed neighbour by its new name.
  - `PipelineOptions` `OPTION_KEYS` is exported and twin-pinned.
  - Component slots show a loading state while a lookup is in flight.
  - `DocumentFileBar`'s `wf-description` becomes `doc-description`, and the copy reads `shown on the card` / `change the folder, file name or description`.
  - `EditorJumps` moves to `ui/src/lib/editor/editorJumps.svelte.ts`. That is a UI file, not a Python module.
  - A prompt reference shows its text as `title` in every arg path.
  - The boolean switch on non-boolean text shows the hint `Only true or false can be a boolean.`.
- Test: `ValidationPanel.test.ts`, `catalogSearch.test.ts`, `VariablesForm.test.ts`, `StepEditor.test.ts`, `StepList.test.ts`, `PipelineOptions.test.ts` (role-based combobox query), `LoraList.test.ts` (two rows, ids unique), `EditorPage.test.ts` (`the example banner goes after Save`), `tests/test_ui_twins.py` (`test_the_ui_variable_name_first_character_is_the_engines`, `test_the_ui_pipeline_option_keys_are_the_schemas`), `tests/test_library_listing.py` (`grep -rln "view=compact" tests`)

- [ ] **Step 1: Write the failing tests:** one per item.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** Check the LoRA scale field's width at 1440 px by eye.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/step-model.spec.ts e2e/lora-suggestions.spec.ts e2e/journeys/editor-safety.spec.ts`, `$PY -m pytest tests/test_ui_twins.py tests/test_api_contract.py -q`, and `wc -l dw/server/routes/library.py` (≤ 990). Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): warnings are not green, names follow the engine's rule, and the catalog matches words`.

### Task 16: Request rows, plan changes, notes and the ledger

Closes 1623.1-1623.4, 1623.6-1623.14.

**Files:**
- Modify:
  - `ui/src/lib/request/RequestItem.svelte`: the `why` line is clamped to one line with a `title`.
  - `ConflictNotice.svelte`: no **again** while the request is closed.
  - `announce.ts`: a reorder that puts a dependency below its dependant says `<title> now runs after <title> that waits on it.`.
  - `planDiff.ts`: seen-snapshots keep the 50 newest keys, `SEEN_KEPT = 50`.
  - `planChanges.ts`: a read with both writers splits them by `history[].by`.
  - `ItemNote.svelte`: `remove()` asks the unread question only when `answerOf` is null; a whitespace-only remainder closes the box.
  - The row a restored draft belongs to opens with it.
  - `dw/server/jobs.py` `request_jobs` docstring: the pointer is restored, paid within the file.
  - `dw/ledger_views.py`: `needs_counts(view)`.
- Test: `RequestItem.test.ts` (`blocked, reason and takes lines stack`), `ConflictNotice.test.ts`, `announce.test.ts`, `planDiff.test.ts`, `planChanges.test.ts`, `ItemNote.test.ts`, `RequestPage.edit.test.ts`, the hold tests (`the banner flips when the agent connects`, `the warning is an alert`, the fixture typed), `tests/test_ledger_views.py` (`revised`, `ready` and `accepted-missing` counts on a real ledger)

- [ ] **Step 1: Write the failing tests:** one per item.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** `wc -l dw/server/jobs.py dw/ledger_views.py` ≤ 1139 and ≤ 639.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/requests.spec.ts e2e/journeys/review-a-request.spec.ts`, and `$PY -m pytest tests/test_ledger_views.py tests/test_ledger*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): reasons fit their row, a closed request has no retry, and seen-snapshots are bounded`.

### Task 17: Delivery, the loading shell, the shell and the Status page

Closes 1624.1, 1624.2, 1624.3, 1624.5, 1624.6, 1624.7, 1624.8, 1624.12, 1624.17, 1624.19, 1624.20, 1624.21, 1624.23, 1624.26, 1624.27.

**Files:**
- Modify:
  - `dw/server/app.py`: `GZipMiddleware(..., exclude_content_types=...)` passed explicitly with Starlette's list, ≤ 292 lines.
  - `ui/index.html`: the theme script inside an IIFE; `min-height: 100dvh` with `100vh` before it as the fallback; a `<noscript>` line, `dw needs JavaScript. Turn it on for this page and reload.`, with the loading text hidden under `noscript`.
  - `ui/src/lib/editor/JsonEditor.svelte`: Question 8's copy.
  - `OverviewPage.svelte`: the first-run card renders once `workflows.status !== 'loading'`.
  - `Sidebar.svelte`: focus returns to the collapse button on re-expand.
  - `routes.ts` `switchHref(name, view, query?)`: keeps the query keys the target section reads and drops `open=`.
  - `ModelsPage.svelte` `repoIdHint`: a bare id is allowed; its hint is `A repo id is a name or org/name, like gpt2 or black-forest-labs/FLUX.1-dev.`.
  - `ServerPage.svelte` and `serverinfo.ts`: a loopback bind reached on a non-loopback page origin, or with a forwarded header the server reports, says `You reach this server through a tunnel or proxy; keep it bound to <bind_host>.`, and no rebind snippet. A plain `localhost` tunnel adds to the rebind panel `Reaching it through ssh -L is the safer way: see [How to use dw from another machine](…)`.
  - `docs/ARCHITECTURE.md`: the Status row gains "never advise rebinding a server reached through a proxy".
- Test: `tests/test_server_delivery.py` (`test_a_large_body_is_gzipped` with seeded jobs, `test_a_live_event_stream_is_not_compressed`, `test_the_exclusions_are_explicit`), `ui/scripts/loading-shell.test.ts` (`the theme script leaks no global`, `with no script the shell says so`), `ui/e2e/performance.spec.ts` (`the shell's failed line shows after its delay`, with the animation sped by `page.clock`), `FirstRunCard.test.ts` or `OverviewPage.test.ts`, `Sidebar.test.ts` (`a folded group while collapsed`, `re-expand returns focus`), `routes.test.ts`, `ModelsPage.test.ts`, `ServerPage.test.ts`

- [ ] **Step 1: Write the failing tests:** one per item.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/performance.spec.ts e2e/journeys/first-run.spec.ts e2e/chrome.spec.ts e2e/workspaces.spec.ts`, `$PY -m pytest tests/test_server_delivery.py tests/test_mcp_server.py -q`, and `wc -l dw/server/app.py` (≤ 292). Expected: PASS, with every budget held.
- [ ] **Step 5: Commit** `ui(shell): explicit compression rules, a shell without script, and a Status page that knows a tunnel`.

## Copy and polish

### Task 18: Copy and small visual inconsistencies

Closes 1601.1, 1601.2, 1601.3, 1601.4, 1601.8, 1601.11, 1601.12b, 1601.13, 1601.14, 1601.15.

**Files:**
- Modify:
  - `ui/src/lib/destructive.ts`: `DestroySubject.place?: 'disk' | 'ledger'`, and `workspaceConfirm` names the outputs count as "files in outputs/" and the starred count as "starred in the gallery".
  - `ItemNote.svelte` and `RequestMenu.svelte` (the fixed copy).
  - `assets/deletion.ts` and `ShadowedAssets.svelte`: "shared asset library" (Question 5).
  - `ui/GLOSSARY.md`.
  - `RequestHeader.svelte`: h1 `font-weight: 600`.
  - `PromptsPage.svelte`: titles.
  - `WorkflowPage.svelte:89`: `about 3 min`.
  - `format.ts` `formatApprox`: a comment on the 30-59 s case.
  - `MediaPicker.svelte:180`: the label unmuted.
  - `patterns.css:91`: `border-bottom`, with the last row's rule removed.
  - `PlanItems.svelte`: the local `.list` goes.
- Test: `destructive.test.ts`, `format.test.ts` (`45 s is under 1 min`), `RequestMenu.test.ts`, `ItemNote.test.ts`, `PromptsPage.test.ts`, `ui/scripts/copy-rules.test.ts`

- [ ] **Step 1: Write the failing tests:** `a ledger delete says the ledger`, `the workspace delete says which count is which`, `45 s is under 1 min`, `a chip's title says a second press shows all`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** Check the datatable and picker labels at 1440 px by eye.
- [ ] **Step 4: Run** `npm test`, the metrics gate (`banned_terms` 0), `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/requests.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(copy): ledger deletes say the ledger, one name for the shared asset library, and table rules that agree`.

## Tests and metric hardening

### Task 19: The metrics and lint rules see what they claim to

Closes 1601.19, 1601.21, 1602.1-1602.13, 1606.1-1606.4, 1624.10, 1624.11.

**Files:**
- Modify:
  - `ui/scripts/arch-metrics.mjs`:
    - `isActionCall` resolves to the `action.svelte` import;
    - `window_dialogs` covers `globalThis.`, `self.` and computed `['confirm']`;
    - `banned_terms` checks single-word literals, its detail is sorted, and a mustache literal is covered by a fixture;
    - `duplicate_blocks` counts shared lines, not runs;
    - `topLevel` is split to complexity ≤ 10;
    - `unguarded_mutations` follows `api.ts`'s exported writers (`setNote`), `downloadArchive` and `picks.svelte.ts`;
    - `raw_media` resolves a braced or conditional `this`, follows `{@html}` through a same-file variable, counts `track`, `embed`, `object`, `canvas` and a CSS `background-image: url(`, and counts a nested `<source>`/`<img>` once;
    - the stale `unguarded_mutations` comment is reworded.
  - `ui/eslint.config.js`:
    - no-raw-message also catches `e.message` after `instanceof`, `(e as Error).message`, `String(e)` and `${e}` in display code;
    - the delete rule reads template strings, `chooseDialog` labels and markup `confirmLabel=`;
    - `patterns: ['monaco-editor/*']` everywhere but `src/lib/monaco.ts` and `src/lib/monacoContribs.ts`.
  - `ui/scripts/copy-rules.test.ts`: whole-term glossary match, sample rows for `star-keep`, `latest-result` and `held-review`; the token rule ties definitions to `:root` and skips templated names; the design-rule test asserts it found at least one declaration.
  - `ui/e2e/performance.spec.ts` `isChunk`: names only `monaco` and `editor.api` chunks.
- Test: `ui/scripts/arch-metrics.test.ts`, `ui/scripts/lint-rules.test.ts`, `ui/scripts/copy-rules.test.ts`

- [ ] **Step 1: Write the failing tests:** one fixture per widened rule, each counting the form it now sees, plus `the banned detail is sorted`.
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/`. Expected: FAIL.
- [ ] **Step 3: Implement.** Any hit the wider rules find in `src/` is fixed in this task, so every metric that is 0 stays 0 and `raw_media` stays 1.
- [ ] **Step 4: Run** `npm test`, `npm run lint`, the metrics gate, and `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (only `complex_functions: 11 -> 13`). Expected: PASS.
- [ ] **Step 5: Commit** `ui(metrics): the rules see the forms they missed, and their details are stable`.

### Task 20: Missing tests, journey robustness and formatting

Closes 1601.20b, 1602.14, 1602.16a, 1602.17c, 1602.18, 1602.21, 1602.23-1602.26, 1606.11, 1609.10b, 1610.2b, 1610.5, 1610.6, 1610.7, 1610.8, 1612.33, 1612.34, 1624.18, 1624.22, 1624.24, 1624.25.

**Files:**
- Modify:
  - `dw/server/job_history.py` `restart_batch` docstring names `workflow_name`. The docstring is reworded in place so the line count stays ≤ 1000.
  - `docs/SERVER.md`: one sentence, `` `If-Match: *` always matches a note, since a note is never missing. ``
  - `ui/e2e/journeys/consistency.spec.ts`:
    - `toBeVisible`/`toBeEnabled` before each `dispatchEvent`, or a real click;
    - case 3 triggers the refresh through `feed.move`-style API bumps, not the poll;
    - case 4 uses the real banner, raised through the outage route `outage-and-token.spec.ts` uses.
  - `ui/e2e/journeys/chain-a-step.spec.ts`: the asset is deleted in `finally`.
  - `ui/e2e/journeys/first-run.spec.ts`: the remote-URL scan reads `body`; the template's presence is checked first, with a clear `test.skip`.
  - `ui/e2e/smoke.spec.ts:102`: `exact: true`.
  - Prettier: `npm run format` on the three flagged files only.
  - `ruff --select I --fix ui/e2e/serve_fixture.py`.
- Test: `tests/test_restart_tombstones.py` (`test_would_queue_names_the_workflow`); `GalleryPage.test.ts` and `AssetsPage.test.ts` (`a double press deletes the picked once`); `ui/PageHeader.test.ts` (`afterEach` resets `location.hash`); `ui/Segmented.test.ts` (`an icon option`, `size sm`); `ui/NoteEditor.test.ts` (rename to `a second Ctrl+Enter while the first asks sends one write`, plus `a second click while the first asks sends one write`); the mocks that build a `Listing` gain `workspace` (`grep -rn "Listing\b" src --include=*.test.ts`); `RequestItem.test.ts:732` asserts no `button` and no `a` in the slot; `ui/e2e/gallery-findings.spec.ts` (`Back closes the job page viewer`, `Back from Compare`); `PromptsPage` e2e (`a long prompt card is clamped to three lines`, by box height); `ui/e2e/responsive.spec.ts` (`the Overview's long job id is clipped`, `a row tile is 32 px`, by bounding box); `App.test.ts` (`several running jobs: the meter label counts them`)

- [ ] **Step 1: Write the tests.** Each must fail against a deliberately broken copy of what it pins: break it, see red, restore.
- [ ] **Step 2: Run** them. Expected: PASS after the restore.
- [ ] **Step 3: Run** `npm test`, the metrics gate, `npx prettier --check src e2e scripts` (clean), `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/journeys/ e2e/smoke.spec.ts e2e/responsive.spec.ts e2e/gallery-findings.spec.ts --repeat-each=2`, `$PY -m pytest tests/test_restart_tombstones.py -q`, and `wc -l dw/server/job_history.py` (≤ 1000). Expected: PASS.
- [ ] **Step 4: Commit** `test(ui): the pins the phases left soft, journeys that wait on conditions, and formatting`.

## Timestamps (#1599)

### Task 21: One rule and one live time

Closes 1599.1, 1599.2, 1609.3. Written for Question 2's recommendation.

**Files:**
- Create: `ui/src/lib/ui/Time.svelte`, with props `{ at: number; mode?: 'relative' | 'exact'; by?: 'agent' | null; label?: 'changed' | 'created' | null }`. It renders a `<time datetime>` with `title` set to `formatExact(at)`. A relative time is re-rendered by one shared minute ticker (`ui/src/lib/clock.svelte.ts`, `export const now: { readonly value: number }`, a `poll` at 30 s that pauses with the tab).
- Create: `docs/superpowers/specs/2026-11-24-timestamps-inventory.md`, listing, for each glossary noun, the times the server returns (field, route), the views that show them today, and the views the rule adds.
- Modify: `ui/src/lib/format.ts` (`formatDuration` from 86,400 s: `<d>d <hh>h`), `ui/GLOSSARY.md` (the Formats rows: relative in lists, exact in details, the duration's day form; *Before you write markup*: a time is `<Time>`)
- Test: `ui/src/lib/ui/Time.test.ts` (`a list time is relative with the exact time on hover`, `it ages without a reload`, `it sets datetime`, `by the agent is said`), `format.test.ts` (`26527 h reads in days`)

- [ ] **Step 1: Write the inventory.** Fill it from `ui/src/lib/generated/api-schema.ts` (`grep -n "mtime\|created_at\|finished_at\|last_" …`) and the views (`grep -rln "formatMtime\|formatDay\|formatExact" src`).
- [ ] **Step 2: Write the failing tests.**
- [ ] **Step 3: Run** them. Expected: FAIL.
- [ ] **Step 4: Implement.** Move the existing users (Jobs, the job header, requests, gallery and asset detail, Models, CachesPanel, `LastUsedCell`) onto `<Time>`, keeping each one's format.
- [ ] **Step 5: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/requests.spec.ts e2e/job-event-log.spec.ts`. Expected: PASS.
- [ ] **Step 6: Commit** `ui(time): one live time component and one rule for lists and details`.

### Task 22: Every noun says when it changed, sorted and held in place

Closes 1599.3, 1599.4, 1599.5, 1599.6.

**Files:**
- Modify:
  - Server: the `times=true` opt-in on `/api/workflows`, `/api/prompts` and `/api/workspaces`, each entry gaining `mtime` (and `created` where the filesystem gives a birth time; else absent). Files are `dw/server/routes/library.py` (≤ 990, paid in comments), the prompts route and the workspaces route (`grep -rn "def list_" dw/server/routes/*.py`), `dw/server/api_models_library.py` (765) and `api_models.py` (≤ 1105).
  - OpenAPI regenerated.
  - UI: `WorkflowsPage`, `WorkflowPage`, `PromptsPage`, `PromptEditorPage` (last saved), `AssetsPage` and `GalleryPage` tiles (the `OutputTile` `meta` slot), the Sidebar and Overview workspaces, `EditorPage` (last saved), notes (`written <time> by <the agent|you>` where the ledger says), and LoRAs where the listing has a time.
  - The newest-first option on Workflows, Prompts and Assets through `codec.oneOf(['name', 'newest'])`.
  - `ui/scripts/copy-rules.test.ts`: no `toLocale*` or `Intl.DateTimeFormat` outside `src/lib/format.ts`, with `EventLog.svelte` and `request/reviewText.ts` moved onto `format.ts`.
  - `docs/ARCHITECTURE.md` *UI* table, new row **Times**: owner `ui/src/lib/ui/Time.svelte`, `src/lib/format.ts`, `src/lib/clock.svelte.ts`; rule "A time is shown through `<Time>`: relative in a list with the exact time on hover, exact in a detail; nothing formats a date outside `format.ts`"; enforced by `Time.test.ts` and `copy-rules.test.ts`.
- Test: one page test per view (`<page> says when each <noun> changed`), `the newest-first sort`, `copy-rules.test.ts` (`no date is formatted outside format.ts`), pytest `test_times_are_opt_in` (no `mtime` without `times=true`; MCP's listings unchanged) for each route

- [ ] **Step 1: Write the failing tests.**
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement**, then regenerate OpenAPI.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/journeys/consistency.spec.ts e2e/journeys/find-and-reuse.spec.ts e2e/performance.spec.ts` (budgets held), `$PY -m pytest tests/test_api_contract.py tests/test_mcp_server.py <the route tests> -q`, and `wc -l` on every server file touched. Expected: PASS.
- [ ] **Step 5: Commit** `ui(time): every list says when each thing changed, sorts newest first, and dates stay in format.ts`.

## Investigation

### Task 23: #1603, the first-import wedge behind the smoke flake

Closes 1603.1-1603.4.

**What is known.**
- `load_allowed_class` (`dw/introspection.py:155-195`) calls `importlib.import_module(module_name or "diffusers")` and then `getattr` on diffusers' lazy module, with no lock.
- The routes that reach it (`dw/server/routes/system.py:128-132` `class_description`, and the pipeline describe route) are sync FastAPI handlers, so they run on the thread pool.
- The editor opening `models/flux-dev` asks several describes at once.
- A concurrent first import of a lazy diffusers submodule across threads can deadlock on CPython's per-module import locks.
- `ui/playwright.config.ts:33` already pins `workers: 1` for this reason, and the test already has `test.setTimeout(120_000)` (`smoke.spec.ts:39`). The flake persists only when the whole spec runs, so the concurrency is within one page's requests, not between workers.

**Files:**
- Modify: `dw/introspection.py` (a module-level `_IMPORT_LOCK = threading.RLock()` held around the `import_module` + `getattr` in `load_allowed_class` and the two other `import_module` calls at `:374` and `:631`; ≤ 910 lines)
- Test: `tests/test_introspection.py` (`grep -rln load_allowed_class tests`)

The fix is small and in scope only if Step 2 shows the stall inside an import. If the dump shows it elsewhere (the worker, a lock in `dw/server`), stop. Report the stack and make Step 5's mitigation the whole task.

- [ ] **Step 1: Reproduce.** `cd ui && npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts --repeat-each=3` (expect a timeout), then `-g "introspected arguments" --repeat-each=3` (expect a pass).
- [ ] **Step 2: Capture the stall.** Start the fixture server as `playwright.config.ts`'s `webServer` does, under `$PY -X faulthandler`. During the stall, run `kill -SIGABRT <pid>`, or `$PY -m py_spy dump --pid <pid>` if py-spy is installed; do not install it. Record whether the threads wait in `_bootstrap._lock_unlock_module` / `_ModuleLock.acquire` under `load_allowed_class`.
- [ ] **Step 3: Write the failing test** `test_concurrent_first_imports_finish`. Eight threads call `load_allowed_class` on four distinct pipeline names, in a fresh subprocess (`subprocess.run([sys.executable, "-c", …], timeout=120)`, no `shell=True`), and all return within the timeout. It must fail or hang without the lock in at least one of five runs; if it never does, keep it as a regression test and say so.
- [ ] **Step 4: Implement the lock.** Run `$PY -m pytest tests/test_introspection.py tests/test_server.py -q` and `$PY scripts/arch_metrics.py` (`modules` unchanged). Then run Step 1's full-spec command with `--repeat-each=5`. Expected: PASS every time.
- [ ] **Step 5: Mitigation, whatever Step 4 found.** `ui/e2e/serve_fixture.py` warms the import before it reports ready: one `load_allowed_class("FluxPipeline")` on the main thread before uvicorn starts. Keep the comment in `playwright.config.ts` naming #1603.
- [ ] **Step 6: Commit** `engine(introspection): one first import at a time, so concurrent describes cannot wedge`, with the stack dump's summary in the body. If the lock was not the fix, commit `test(e2e): warm the pipeline import before the fixture reports ready` instead.

---

## Phase exit

- Every ledger row is done, duplicate, obsolete, or answered by Don.
- Every metric that was 0 is 0, and nothing is above `$SCRATCH/ui-fu.json`.
- `modules` is 281 (or 282 with Question 1), `claude_md_lines` is 129, the size caps hold, and no page is over 600.
- The whole e2e suite passes twice in a row, `performance.spec.ts` and the smoke spec included.

## Appendix: item ledger

See the section below (also at `/private/tmp/claude-501/-Users-don-src-dkackman-diffusers-workflow/c38aa355-7a1b-44f4-aee3-aaaef6ebc96d/scratchpad/followups/item-ledger.md`).

### Item ledger (checked against develop 62814175)

Every checkbox or bullet in #1598-#1612, #1623 and #1624 has one row. An item with parts in different states is split into `a`/`b` rows. Status:
- **done**: the evidence is the commit, or the file:line at HEAD, that shows it.
- **open**: the code at HEAD still behaves the way the issue says (file:line given).
- **duplicate of X**: the same defect as row X; X's task fixes both.
- **obsolete**: the code it names is gone, or the item asks for no action (it was recorded only, or accepted by ruling).
- **needs Don**: a product call. The task named applies the answer.

Paths are relative to `ui/src/` unless they start with `dw/`, `tests/`, `docs/`, `ui/` or `scripts/` (`ui/scripts/`).

#### #1598: UI quality 2a/2b, smaller deferred findings (release-blocker)

No phase plan folded #1598, and no commit references it. `resource.svelte.ts` has had no commit since 2b (b581ea58).

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1598.1 | A document deleted elsewhere gets no notice; Save meets `stale_write` "changed" | open | `lib/editorShell.svelte.ts:444-462` `#arrived` has no gone path; `lib/documentSave.ts:19` | T2 |
| 1598.2 | The flush on a newer etag can toast an error for a broken JSON draft | open | `#arrived` → `unsaved()` → `flush()` (`editorShell.svelte.ts:335,455`) → `JsonEditor.svelte:55` `apply(false)` → `applyJson` toasts (`:356-369`) | T2 |
| 1598.3 | The enhancer idea and step modes are not refreshed on a remote reload / Load theirs | open | `pages/PromptEditorPage.svelte:135` sets `enhancer.idea` only in `open()`; `remoteLoads` is read only by `editor/editorCheck.svelte.ts:52` | T2 |
| 1598.4 | A stale 409 retry hides the agent's change (no row mark) | open | `request/planStore.svelte.ts:367-379` `#refused` takes the fresh read loud | T3 |
| 1598.5a | A row that changes again while marked does not flash again | open | `lib/resource.svelte.ts:110-116` re-sets the same keys; the CSS animation never restarts | T3 |
| 1598.5b | A row both changed and live loses its live bar | open | `app.css:474-484`: `.row-changed`'s `box-shadow` replaces `.row-live`'s | T3 |
| 1598.5c | Stale marks can survive a question change | open | `resource.svelte.ts:205-207` clears on `asking`, but no test pins a page-level question change | T3 |
| 1598.5d | A rename from an untitled item is not announced | open | `request/announce.ts:53-55` has no untitled case and no test | T3 |
| 1598.6 | The changed bar is faint over bright pictures | open | `frames.css:91-100`, a 3px `var(--live)` overlay | T3 |
| 1598.7 | EditorPage's workflow-listing failure is silent | open | `pages/EditorPage.svelte:95`: nothing reads `catalog.error` | T2 |
| 1598.8 | The workspace list is fetched twice per trigger | open | `lib/Sidebar.svelte:55`, `pages/OverviewPage.svelte:119` | T4 |
| 1598.9 | Asset uploads and bulk deletes reload loudly instead of `adopt` | open | `pages/AssetsPage.svelte:295,329,341` `list.reload()` | T4 |
| 1598.10 | An open job page costs one extra `GET /api/jobs/{id}/workflow` per `jobs` move | open | `pages/JobPage.svelte:54-67` `watch: ['jobs']` | T4 |
| 1598.11 | RequestPage's `alsoLive` re-arms its poll on every read | open | `pages/RequestPage.svelte:115`; `resource.svelte.ts:146-151` effect reads `store.read` | T4 |
| 1598.12 | An overtaken reply still reaches the plan store | open | `planStore.svelte.ts:150-156` `refresh()` calls `take` before the resource's `latestOnly` drops it (`resource.svelte.ts:225-233`); `newerOf` limits the harm | T4 |
| 1598.13 | The changed-mark timer is a bare `setTimeout`, not `poll` | open | `resource.svelte.ts:115` | T3 |
| 1598.14 | `app.css` is at the 600-line limit | open | `wc -l`: 600 | T7 |
| 1598.15 | `EditorPage.svelte` is at 591 of 600 | obsolete | 502 lines at HEAD | - |
| 1598.16 | `dw/server/routes/library.py` is at 1,000 lines | open | `wc -l`: 1000 | T7 |
| 1598.17 | The old inset `.row-changed` shadow still paints under the tile overlay | open | `app.css:481-484` applies to `.cellwrap`/`a.card`; `frames.css:87-100` does not reset it | T3 |
| 1598.18 | `withSaved` / `withSavedPrompt` are duplicated | open | `editor/workflowFile.ts:153`, `promptFile.ts:85` | T2 |
| 1598.19 | `StepEditor` has no unmount guard | open | `editor/StepEditor.svelte`: no destroyed flag | T2 |
| 1598.20 | `HELPER_EXCEPTIONS` in `freshness.test.ts` are per file | open | `lib/freshness.test.ts:131` | T4 |
| 1598.21 | `ExportSummary.inline_json` spells the file names as literals | open | `dw/server/exports.py:174-178` | T7 |
| 1598.22a | The swallowed-raise test asserts only that nothing raises | open | `tests/test_changes.py:497-511` | T7 |
| 1598.22b | One over-long docstring line in `dw/server/changes.py` | open | `dw/server/changes.py:19` (106 chars) | T7 |
| 1598.22c | The request page's mark test doesn't check that the mark clears | open | `pages/RequestPage.test.ts:528-543` | T3 |
| 1598.23 | The e2e models case runs only once per fixture server | open | `ui/e2e/journeys/agent-edits.spec.ts:271-283` deletes `e2e/tiny`, never re-seeds | T7 |
| 1598.24 | No fixture asset or workflow with a proof picture | open | `ui/e2e/serve_fixture.py` seeds none | T7 |

#### #1600: UI phase 3 follow-ups, user-visible gaps (release-blocker)

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1600.1 | Workspace delete is a dead end while the gallery cannot be read | done | a16a16d6; `lib/workspaceActions.ts:106-113` | - |
| 1600.2 | EditorPage `save()` can reject unhandled after a successful save | done | ebf6bce5; `pages/EditorPage.svelte:280-293` | - |
| 1600.3 | `describeError` drops the detail when the message is shorter than a fixed title | open | `lib/apiErrors.ts:214` `message.length > title.length` | T1 |
| 1600.4 | ItemNote's new-note box resends the already-sent part | done | 1d6b832c; `request/ItemNote.svelte:150-152` | - |
| 1600.5 | NoteDraft keeps "You have an unsaved draft." after the saved note matches | done | 1d6b832c; `lib/noteDraft.svelte.ts:59` `kept` is `$derived` | - |
| 1600.6 | N1: a mid-PUT switch then reopen shows "changed elsewhere" over the user's save | done | 1d6b832c (the same derived `kept`, phase 6 Task 13) | - |
| 1600.7 | M1: a kept draft does not remember the note it answered | open | `noteDraft.svelte.ts` stores no base text | T5 |
| 1600.8 | M3: JobPage's "Updated by an agent or another tab" after the box closes | open | `pages/JobPage.svelte:77` | T5 |
| 1600.9 | M6: NoteEditor's Cancel is not disabled on `pending` | open | `ui/NoteEditor.svelte:236-241` `disabled={saving}` | T5 |
| 1600.10 | The `moved_on` card can cite the wrong write | open | `planStore.svelte.ts:314` `culprit(basis.history, 0)` | T5 |
| 1600.11 | An Undo waiting behind busy checks its target only before it waits | open | `lib/destructive.ts:42` checks, then `lib/action.svelte.ts:80` `turn.after(revert)` waits | T5 |
| 1600.12 | Undo says nothing when the agent already reopened the request | done | `request/RequestMenu.svelte:68-69` (phase 6 Task 8) | - |
| 1600.13 | A step restored by Undo returns at its old raw index | done | dba80b4b; `editor/StepList.svelte:62` | - |
| 1600.14 | A step restored under a new name is not announced | done | dba80b4b; `StepList.svelte:79` "came back as" | - |
| 1600.15 | The keep_output 409 has no `code: 'exists'` | open | `dw/server/routes/assets.py:519-524`; `gallery/saveAsAsset.ts:36` matches "already exists" | T1 |
| 1600.16 | `firstSentence` cuts a dialog title at a filename's dot | open | `lib/confirm.svelte.ts:54-57` | T1 |
| 1600.17 | The asset listing GET runs on every save-as-asset press | done | phase 4 Task 6; `gallery/saveAsAsset.ts` lists after a 409 | - |
| 1600.18 | The empty Compare page flashes a skeleton; a failed read shows Retry | done | `pages/ComparePage.svelte:43,149` (phase 4 Task 15) | - |
| 1600.19 | The Models breadcrumb renders nearly invisible | obsolete | not reproduced: db946501's body (no route-specific rule; the capture caught the 150 ms fade-in) | - |
| 1600.20 | The Jobs move buttons have no busy cue | open | `job/JobRow.svelte:133-158`; `pages/JobsPage.svelte:191` | T6 |
| 1600.21 | One Rerun disables every restart banner | open | `JobsPage.svelte:258-261` one `rerunning.busy` | T6 |
| 1600.22 | Hand-edited filter values in the hash are not validated | done | `lib/router.svelte.ts:94` `codec.oneOf` (phase 4 Task 8) | - |
| 1600.23 | Roles are encoded as `%2C` | done | `router.svelte.ts:56-58` | - |
| 1600.24 | A Firefox scrollbar click is not a user scroll | done | `router.svelte.ts:251-258` `movedByUser` | - |
| 1600.25 | Reload no longer restores scroll | open | `router.svelte.ts:230-231` an in-memory `Map` | T6 |
| 1600.26 | `#/ws/studio//requests` drops `requests`; the Not-found crumb links to itself | open | `lib/routes.ts:96`; `lib/Breadcrumb.svelte:41` | T6 |
| 1600.27 | `join('/')` for the Compare route | done | phase 4 Task 15 (`compareHref`, `parseView` compares parts) | - |
| 1600.28a | Group toggle under a filter: FolderGroups | done | phase 4 Task 8 | - |
| 1600.28b | Group toggle under a filter: MediaPicker | duplicate of 1600.29 | | - |
| 1600.28c | Group toggle under a filter: ShadowedAssets | open | `assets/ShadowedAssets.svelte:48-56` toggles while filtering | T6 |
| 1600.29 | MediaPicker toggle in a search; one collapse key across tabs and workspaces | done | 46b1cdd9; `assets/MediaPicker.svelte:147` | - |
| 1600.30a | Job link uses `workspace.current`: GalleryDetail | done | phase 4 Task 5 (RecipePanel links the file's workspace) | - |
| 1600.30b | ... EnhancePanel | done | `editor/EnhancePanel.svelte:97` `enhancer.jobWorkspace` | - |
| 1600.30c | ... WorkerList (needs a server field) | open | `lib/WorkerList.svelte:37` | T6 |
| 1600.31a | The held chip reads "held · 02:05" | done | phase 6 Task 7 | - |
| 1600.31b | The host-RAM line cannot name its example (needs a server field) | open | `lib/plan.ts:85-94` "inherited from an example" | T6 |
| 1600.32 | Re-clicking the shape filter clears with no cue | done | b074794d | - |
| 1600.33 | The prompt editor's file area has no collapse | done | f9203a45 (one `DocumentFileBar`; `editor/DocumentFileBar.svelte:11-13,120`) | - |
| 1600.34 | The StepEditor field lost its 480px cap | done | dba80b4b | - |
| 1600.35a | Ctrl/Cmd+S under a dialog or mid-composition reaches the browser | done | ebf6bce5; `editorShell.svelte.ts:80-93` | - |
| 1600.35b | Shift+Ctrl/Cmd+S and Shift+Ctrl/Cmd+Enter act too: intended? | needs Don | `editorShell.svelte.ts:68-73` ignores Shift | T2 |
| 1600.36 | JobHeader's Replace confirm is neither `danger` nor titled | open | `job/JobHeader.svelte:129-132` | T1 |
| 1600.37 | The Models and Caches trash triggers carry `danger` over non-danger confirms | open | `CachesPanel.svelte:129,214`, `pages/ModelsPage.svelte:340`; `destructive.ts:157-159` | T1 |
| 1600.38 | "(run root)" | done | `job/JobResults.svelte:190` "(top folder)" | - |
| 1600.39 | `formatMinutes` says "about 80 min" | done | `lib/format.ts:74-79` (phase 6 Task 4) | - |

#### #1601: UI phase 3 follow-ups, wording and polish

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1601.1 | "This removes it on disk." on a ledger record or note | open | `destructive.ts:83`; used by `request/ItemNote.svelte:86`, `RequestMenu.svelte:120` | T18 |
| 1601.2 | "go from the ledger for good" | open | `RequestMenu.svelte:124` | T18 |
| 1601.3 | "the shared library" vs "the shared asset library" | needs Don | `assets/deletion.ts:13,26`, `ShadowedAssets.svelte:37` vs `destructive.ts:143` | T18 |
| 1601.4 | The outputs count and the starred count in the workspace delete may not match | open | `destructive.ts` `workspaceConfirm` | T18 |
| 1601.5 | The Schema page h1 is "Workflow Schema" | done | 237b0023; `pages/SchemaPage.svelte:27` | - |
| 1601.6 | ServerPage says "Status" twice | done | c97be26e | - |
| 1601.7 | The editor's back label is the leaf | done | 1daa1b4e | - |
| 1601.8 | RequestHeader's h1 lost `font-weight: 600` | open | `request/RequestHeader.svelte:205-210` sets no weight | T18 |
| 1601.9 | The meter's aria-label and the doubled "running" | done | db946501 | - |
| 1601.10 | The drawer-close focus brief | obsolete | the phase-3 brief is closed; the behaviour (focus the h1) stands | - |
| 1601.11 | The PromptsPage chip titles don't mention the second press | open | `pages/PromptsPage.svelte:123,134` | T18 |
| 1601.12a | Stale `~` comments: runPlan.ts, WorkflowsPage | done | d06237a1; no `~` in `WorkflowsPage.svelte` | - |
| 1601.12b | Stale `~` comment: WorkflowPage | open | `pages/WorkflowPage.svelte:89` "~3 min" | T18 |
| 1601.13 | `formatApprox`'s redundant `seconds < 60 \|\| minutes < 1` | open | `format.ts:85` | T18 |
| 1601.14 | Picker labels muted; datatable rule is `border-top` | open | `assets/MediaPicker.svelte:180`; `patterns.css:91` | T18 |
| 1601.15 | PlanItems' local `.list` padding duplicates the global rule | open | `request/PlanItems.svelte:415-417` vs `patterns.css:133-135` | T18 |
| 1601.16a | EnhancePanel's unformatted line | done | `prettier --check` passes | - |
| 1601.16b | JobHeader's unrelated reformat | obsolete | "leave or revert": left | - |
| 1601.17a | The shared file bar's `wf-folder`/`wf-savename` ids | done | 1daa1b4e (`doc-folder`/`doc-savename`) | - |
| 1601.17b | ... and its workflow wording | duplicate of 1612.30 | `DocumentFileBar.svelte:97-102,132` | T15 |
| 1601.18 | The form view lost "introspection-driven forms" | obsolete | recorded only, intended | - |
| 1601.19 | The `unguarded_mutations` header says "don't rename it to `function`" | open | `ui/scripts/arch-metrics.mjs:91` | T19 |
| 1601.20a | Over-wide test lines: RequestItem.test, RunDialog.test | done | `prettier --check` passes | - |
| 1601.20b | ... observed.test | duplicate of 1610.7 | | T20 |
| 1601.21 | The `banned_terms` detail order is unstable | open | `arch-metrics.mjs:657,854` unsorted | T19 |

#### #1602: UI phase 3 follow-ups, harden the checks and tests

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1602.1 | `isActionCall` matches any `action` identifier | open | `arch-metrics.mjs:376-379` | T19 |
| 1602.2 | `window_dialogs` misses `globalThis.confirm`, `self.confirm`, `window['confirm']` | open | `arch-metrics.mjs:371-374` | T19 |
| 1602.3 | `banned_terms` skips single-word literals, undocumented | open | `arch-metrics.mjs:585-586` `/\s/` guard | T19 |
| 1602.4 | Parenthesised copy in `{}` is not seen by `banned_terms` | open | `arch-metrics.mjs:583-600`; no fixture pins a mustache literal | T19 |
| 1602.5 | `duplicate_blocks` counts runs | open | `arch-metrics.mjs:70-73` | T19 |
| 1602.6 | The glossary check uses `includes()`; three terms lack sample rows | open | `ui/scripts/copy-rules.test.ts:53`; `lib/terms.ts:28,34,53` | T19 |
| 1602.7 | `topLevel` has complexity 11 | open | `arch-metrics.mjs:390` | T19 |
| 1602.8 | The no-raw-message lint rule covers one pattern | open | `ui/eslint.config.js` | T19 |
| 1602.9 | The delete lint rule sees only a literal `confirmLabel` | open | `ui/eslint.config.js` | T19 |
| 1602.10 | `unguarded_mutations` cannot see `setNote` | open | `lib/api.ts:544`; documented as a limit at `arch-metrics.mjs:87` | T19 |
| 1602.11 | The metric does not follow `downloadArchive` or `picks.svelte.ts`'s download | open | `arch-metrics.mjs` | T19 |
| 1602.12 | The token rule is loose | open | `ui/scripts/copy-rules.test.ts` | T19 |
| 1602.13 | The design-rule test passes vacuously | open | | T19 |
| 1602.14 | No test of `workflow_name` on a `would_queue` entry; `restart_batch` docstring omits it | open | `dw/server/job_history.py:384-391` | T20 |
| 1602.15 | No test of the no-meter labels | done | db946501; `App.test.ts:188` | - |
| 1602.16a | No double-press test on the bulk delete | open | only the single delete: `pages/GalleryPage.test.ts:1179` | T20 |
| 1602.16b | PromptEditorPage's busy check sits before an awaited fetch | open | `pages/PromptEditorPage.svelte:227-230` | T2 |
| 1602.17a | `WorkflowPage.test.ts`'s back-label regex accepts either label | done | `WorkflowPage.test.ts:139,144` | - |
| 1602.17b | The RequestPage loading test does not assert the title after load | done | `RequestPage.test.ts:239-241` (`loaded()` finds the title heading) | - |
| 1602.17c | `PageHeader.test.ts` leaves `location.hash` set | open | `ui/PageHeader.test.ts:14` only `cleanup` | T20 |
| 1602.18 | No test for Segmented `icon` or size `sm` | open | `ui/Segmented.test.ts:13,27` | T20 |
| 1602.19 | `editorKeys` tests: read-only, busy skip, unrelated key | done | ebf6bce5; `editorShell.test.ts:710-780` | - |
| 1602.20 | ItemNote's `danger: true` Cancel question is unpinned | done | 1d6b832c | - |
| 1602.21 | NoteEditor's "a second Save while the first asks" presses Ctrl+Enter | open | `ui/NoteEditor.test.ts:274` | T20 |
| 1602.22 | NoteList's double guard | done | 1d6b832c | - |
| 1602.23 | `If-Match: *` always passes for a note: say so in SERVER.md | open | `docs/SERVER.md:1558-1565` | T20 |
| 1602.24 | Journey `dispatchEvent` clicks skip actionability | open | `ui/e2e/journeys/consistency.spec.ts:65,69` | T20 |
| 1602.25 | Journey case 3 waits 15 s on the feed | open | `consistency.spec.ts:186` | T20 |
| 1602.26 | Journey case 4 uses a stand-in div | open | `consistency.spec.ts:213` | T20 |

#### #1603: Flaky e2e, smoke "editor opens a workflow with introspected arguments"

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1603.1 | Repro: the whole `e2e/smoke.spec.ts` times out | open | `ui/e2e/smoke.spec.ts:36-59` | T23 |
| 1603.2 | Repro: the test alone passes | open | | T23 |
| 1603.3 | Confirm the wedge worker- or server-side with a stack dump; serialise or warm the first import | open | `dw/introspection.py:175-178` imports with no lock, from FastAPI's thread pool (`dw/server/routes/system.py:128-132`) | T23 |
| 1603.4 | Until then, a longer timeout or a warm-up | open | the 120 s timeout exists (`smoke.spec.ts:39`) and still flakes (phase 7 plan); no warm-up | T23 |

#### #1605: UI phase 4 follow-ups, the media viewer, Gallery and Assets

No commit after the phase-4 merge (78b66483) touches `MediaViewer.svelte`, `mediaViewer.ts`, `gallery/order.ts`, `RecipePanel.svelte` or `recipe.ts`.

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1605.1 | Pan sticks after a `pointerup` outside the viewer | open | `ui/MediaViewer.svelte:258-281`, no pointer capture | T9 |
| 1605.2 | No pinch zoom on phones | open | `MediaViewer.svelte` | T9 |
| 1605.3 | Full screen hides portal confirms and toasts | open | `MediaViewer.svelte:168-174` | T9 |
| 1605.4 | Stepping after the open file is deleted elsewhere gives no message | open | | T9 |
| 1605.5 | Restore the open-in-new-tab link if missed | needs Don | | - |
| 1605.6 | The More menu's Escape and focus return aren't pinned | open | | T9 |
| 1605.7 | Phone focus return after Use as input may target an unmounted trigger | open | | T9 |
| 1605.8 | Name sort is case-sensitive and not numeric | open | `gallery/order.ts:82-85` | T10 |
| 1605.9 | Star Undo writes the prior state unconditionally | open | `pages/GalleryPage.svelte:203-207`; `picks.star` | T5 |
| 1605.10 | AssetsPage has a side effect inside the `viewed` derived | open | `pages/AssetsPage.svelte:232-236` | T10 |
| 1605.11 | Same-leaf workflows render identical Use as input buttons | open | `gallery/UseAsInput.svelte:59` `leafName` | T10 |
| 1605.12a | The RecipePanel Job row links a job that left history | open | `ui/RecipePanel.svelte` | T10 |
| 1605.12b | `copyRows`' cast drops the key literal type | open | `lib/recipe.ts:93` | T10 |
| 1605.13a | The job-load ErrorLine shows on embedded images | open | `RecipePanel.svelte:164` | T10 |
| 1605.13b | A null `workflow_name` falls back to the embedded id | open | `recipe.ts` | T10 |
| 1605.14 | Test: the workspace-change close | open | | T9 |
| 1605.15 | Test: `open=` where `q=` filters the asset out | open | | T10 |
| 1605.16 | Test: Use as input's choose path, and its dialog from More | open | | T10 |
| 1605.17 | Test: `aria-expanded` in the deep-link-to-a-shut-folder test | open | | T10 |
| 1605.18 | Test: history-state isolation between MediaViewer tests | open | | T9 |

#### #1606: UI phase 4 follow-ups, tiles and the raw_media metric

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1606.1 | `raw_media` misses a braced or conditional `svelte:element this` | open | `arch-metrics.mjs:77-78` (documented limit) | T19 |
| 1606.2 | ... `{@html}` held in a variable | open | `arch-metrics.mjs:80` | T19 |
| 1606.3 | ... `<track>`, `<embed>`, `<object>`, `<canvas>`, CSS `background-image` | open | `arch-metrics.mjs:76` | T19 |
| 1606.4 | ... double-counts nested media | open | | T19 |
| 1606.5a | The picker's current tile and hover look alike | done | phase 5 Task 4 (46b1cdd9) | - |
| 1606.5b | The active, hover and picked states look alike | open | `ui/OutputTile.svelte` | T10 |
| 1606.6 | A shadowed video shows a play badge | open | | T10 |
| 1606.7 | Asset tiles have no note mark | open | | T10 |
| 1606.8 | An audio result is cramped in a 48 px cover | open | | T10 |
| 1606.9 | Same-leaf tiles share a caption and accessible name | duplicate of 1605.11 | one helper names both | T10 |
| 1606.10 | A long job id spills past the Overview row | done | 2f5f72d8 | - |
| 1606.11 | `Listing.workspace` is missing from the test mocks | open | to be checked per mock in the task | T20 |
| 1606.12 | The Overview's jsdom computed-style check can't fail | done | 2f5f72d8 (`ui/e2e/responsive.spec.ts`) | - |
| 1606.13 | AssetsPage's background `media=true` read has no bound | open | `pages/AssetsPage.svelte` | T10 |
| 1606.14 | `api.ts` at exactly 600 lines | done | 598 at HEAD (phase 5 Task 1) | - |

#### #1607: UI phase 4 follow-ups, prompts=true / media=true on the server

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1607.1 | `validate_path` refuses a legal `take..2.mp4`; `resolve_output_file` too | open | `dw/security.py:43,118-121` (`\.\.` anywhere) | T8 |
| 1607.2 | Two jobs sharing a run id: `prompts=true` picks one arbitrarily | open | `dw/server/outputs.py:174-180` | T8 |
| 1607.3 | A row with invalid JSON is never memoised | open | `outputs.py:149-171`; `job_history.py:636` | T8 |
| 1607.4 | One prompt-memo read happens outside the lock | open | `outputs.py:168` | T8 |
| 1607.5 | `json_valid` first; `run_id` has no index | open | `job_history.py:634-636`; indexes only at `:309,313` | T8 |
| 1607.6 | `job_history.py` needs a real split (a new module) | needs Don | 1000 lines | T8 |
| 1607.7 | Restore OpenAPI descriptions shortened for lines | open | | T8 |
| 1607.8 | A gallery test that prompts cover only the returned page | open | only `tests/test_jobs_listing.py:131` | T8 |

#### #1608: UI phase 4 follow-ups, job page results, job errors, event log

No commit after 78b66483 touches `JobResults`, `JobProgress`, `JobError`, `jobError.ts` or `EventLog`.

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1608.1 | Compare's back link reads "← Gallery" when you arrive from a job | open | `pages/ComparePage.svelte:144` | T12 |
| 1608.2 | The Progress `h2` sits inside `<summary>` | open | `job/JobProgress.svelte:39-40` | T11 |
| 1608.3 | A failed metadata read leaves the star disabled | open | | T11 |
| 1608.4 | A job with no files shows no `MissingFile` for a stray `open=` | open | | T11 |
| 1608.5 | A star changed elsewhere loses to the viewer's state | open | | T11 |
| 1608.6 | Test: the Error panel above Results | open | | T11 |
| 1608.7 | "Out of memory" matches host-RAM and ffmpeg errors ("card") | open | `job/jobError.ts:16-18` | T11 |
| 1608.8 | The gated 403 matcher fires for private repos | open | `jobError.ts:32` | T11 |
| 1608.9 | A finished step can be marked red after a restart | open | | T11 |
| 1608.10 | The kind column repeats the text | open | `job/EventLog.svelte` | T11 |
| 1608.11 | A failed live read keeps its error until the next event | open | | T11 |
| 1608.12 | The unmount test's comment overstates it | open | | T11 |
| 1608.13 | The e2e row-height bound is loose | open | `ui/e2e/job-event-log.spec.ts` | T11 |

#### #1609: UI phase 4 follow-ups, Jobs list, Compare, request takes

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1609.1 | The search placeholder is clipped | done | c127dbd9 (the field's `title`) | - |
| 1609.2 | All-jobs view: other workspaces' jobs get a dashed proof slot | open | | T13 |
| 1609.3 | `formatDuration` has no day unit | open | `format.ts:63-69` | T21 |
| 1609.4 | `listWithPrompts` re-ask test; a second poll | done | c127dbd9 | - |
| 1609.5 | Compare's audio and text cells aren't drawn through `MediaBody` | open | `ComparePage.svelte` | T12 |
| 1609.6 | The synced transport keeps playing under the viewer | open | | T12 |
| 1609.7 | The empty copy appears a microtask late | obsolete | accepted, recorded only | - |
| 1609.8 | A deep link `open=` to a closed row shows nothing | done | 7076d8f4 | - |
| 1609.9 | A job linked to two items shares a viewer key | done | 7076d8f4; `request/AttemptStrip.test.ts:84` | - |
| 1609.10a | Tests: history wiring, Accepted, Accept stars | done | `AttemptStrip.test.ts:72,99`; `RequestPage.edit.test.ts:944` | - |
| 1609.10b | "A take with no file stays a slot" is only partly red | open | `request/RequestItem.test.ts:732` | T20 |

#### #1610: UI phase 4 follow-ups, shared helpers and tests

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1610.1 | The missing-file effect, `saving` state and duration reading are copied four times | open | `job/JobResults.svelte`, `ComparePage`, `AssetsPage`, `GalleryPage` | T10 |
| 1610.2a | Real-browser Back test: request takes | done | `ui/e2e/requests.spec.ts:443` | - |
| 1610.2b | ... job page viewer and Compare | open | | T20 |
| 1610.3 | OverviewPage's jsdom check can't fail | duplicate of 1606.12 | done there | - |
| 1610.4 | The primary-button pin: dirty, builtin, running | done | 1daa1b4e | - |
| 1610.5 | The prompt-card test doesn't cover the CSS clamp | open | | T20 |
| 1610.6 | The Schema heading test is a substring match | open | `ui/e2e/smoke.spec.ts:102` | T20 |
| 1610.7 | Prettier flags older files | open | `prettier --check` flags `observed.test.ts`, `ui/Segmented.test.ts`, `ui/scripts/arch-metrics.test.ts` | T20 |
| 1610.8 | Fixture import order in `serve_fixture.py` | open | `ruff --select I`: 1 error | T20 |

#### #1611: UI at 390 px

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1611.1 | The Overview is 446 px wide at 390 px | done | 2f5f72d8; `ui/PageHeader.svelte:117` | - |
| 1611.2 | The Overview Requests panel hides the first title | done | 72453254 | - |
| 1611.3 | Header popovers stay open after a fast tap | done | db946501; `App.svelte:172` | - |

#### #1612: UI phase 5 follow-ups, run form, reference field, catalog, editor

No commit after the phase-5 merge (fd57a4e8) touches `editor/`, `workflow/` (but `runCheck.svelte.ts`), `ReferenceField`, `RerunWithChanges` or `routes/library.py`.

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1612.1 | Green "Ready to run." beside warnings; warn lines in a closed fold | open | `editor/ValidationPanel.svelte`, `workflow/PlanLines.svelte` | T15 |
| 1612.2 | The ARCHITECTURE *Reference input* row is broader than the code | open | `docs/ARCHITECTURE.md:249` | T14 |
| 1612.3 | "Fix N issues" points at a fixed issue for 600 ms | open | | T15 |
| 1612.4 | "A job would be refused: 0 issues to fix." | done | d06237a1; `runCheck.svelte.ts:36-37` | - |
| 1612.5 | A default file reference shows only as placeholder | open | `editor/VariablesForm.svelte` | T14 |
| 1612.6 | `inputs=true` ignored under `view=compact`; the trait test | open | `dw/server/routes/library.py` | T15 |
| 1612.7 | Catalog search is substring; punctuation; long comment line | open | `workflow/catalogSearch.ts:56` | T15 |
| 1612.8 | `planSummary` shows a partial GB total | open | | T14 |
| 1612.9 | An empty MIME type is refused as the wrong kind | open | | T14 |
| 1612.10 | The blob URL leaks on destroy mid-upload | open | `ui/ReferenceField.svelte` | T14 |
| 1612.11 | The late-upload toast says "changed" when closed; lower-case copy | open | | T14 |
| 1612.12 | A pick's preview after unsaved typing waits for the upload | open | | T14 |
| 1612.13 | `previewOf` of a wrong-kind `asset:` draws its tile | open | `lib/referenceField.ts` | T14 |
| 1612.14 | A field flips to `ReferenceField` when the catalog loads late; duplicate freshness entry | open | `freshness.test.ts` | T14 |
| 1612.15 | A prefill is saved as a draft and shows "Restored" | open | | T14 |
| 1612.16 | The Advanced toggle cannot fold while a folded field holds a value | open | | T14 |
| 1612.17 | No test: Run keeps the draft; a workflow switch | open | | T14 |
| 1612.18 | Closing Enhance doesn't cancel the server job; no Use test | open | | T14 |
| 1612.19 | A field cleared to blank keeps the job's value | open | `job/RerunWithChanges.svelte` | T14 |
| 1612.20 | New seed plus an edited seed gives a 400 | open | | T14 |
| 1612.21 | Reformatted JSON counts as a change | open | | T14 |
| 1612.22 | The variable-name first-character rule is not twin-pinned | open | `tests/test_ui_twins.py:67` pins characters only | T15 |
| 1612.23 | Enter during IME adds a variable; no `aria-describedby` | open | | T15 |
| 1612.24 | `StepEditor`'s `others` from a `{}` default | open | | T15 |
| 1612.25 | `previous_result: a` (space) is not rewritten by a rename | open | | T15 |
| 1612.26 | Undo puts the step last when its neighbour was renamed | open | `StepList.svelte:62` | T15 |
| 1612.27 | `OPTION_KEYS` has no twin pin | open | `editor/PipelineOptions.svelte:44` | T15 |
| 1612.28 | Stale component slots during a lookup; brittle combobox selector | open | | T15 |
| 1612.29 | The LoRA label test uses one row; check the scale width | open | | T15 |
| 1612.30 | The `wf-description` id; no test the example banner goes after Save | open | `DocumentFileBar.svelte:97-102` | T15 |
| 1612.31 | `EditorJumps` lives in `editorCheck.svelte.ts` | open | | T15 |
| 1612.32 | A prompt reference in a step argument has no hover preview | open | | T15 |
| 1612.33 | The chain-a-step journey's shared asset can be left behind | open | `ui/e2e/journeys/chain-a-step.spec.ts` | T20 |
| 1612.34 | The row-tile size test pins 32 px by regex | open | | T20 |
| 1612.35 | Switching a text row to boolean leaves text with no explanation | open | | T15 |

#### #1623: UI phase 6 follow-ups, request rows, plan changes, notes, ledger

No commit after the phase-6 merge (f8690eb9) touches `request/`, `RequestPage`, `RequestsPage`, `dw/server/jobs.py` or `dw/ledger_views.py`.

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1623.1 | The take-reason line is not clamped and has no `title` | open | `request/RequestItem.svelte:99,166` | T16 |
| 1623.2 | No test pins blocked + reason + takes lines stacking | open | | T16 |
| 1623.3 | A ConflictNotice keeps **again** after the request closes | open | `request/ConflictNotice.svelte` | T16 |
| 1623.4 | Moving a dependency below its dependant is silent | open | | T16 |
| 1623.5 | By-eye check at 1440 and 1300 px | done | 2f5f72d8 (phase 7 Task 10) | - |
| 1623.6 | Seen-snapshots are never evicted | open | `request/planDiff.ts:149` | T16 |
| 1623.7 | A read with a user and an agent write lists both under the agent | open | | T16 |
| 1623.8 | `remove()` asks the unread question while `answerOf` shows an answer | open | `ItemNote.svelte` | T16 |
| 1623.9 | A restored draft reopens its box without keeping its row open | open | | T16 |
| 1623.10 | A whitespace-only remainder leaves an empty box open | open | `ItemNote.svelte:150-152` | T16 |
| 1623.11 | Hold tests: agent connect/disconnect, `role="status"`, `as never` | open | | T16 |
| 1623.12 | `request_jobs` docstring lost its `delete_job_run` pointer | open | `dw/server/jobs.py` | T16 |
| 1623.13 | `needs_counts(request_view)` shadows the module function | open | `dw/ledger_views.py:597` | T16 |
| 1623.14 | The real-ledger needs test omits revised, ready, accepted-missing | open | | T16 |
| 1623.15 | `needs=true` derives twice per row | obsolete | recorded only ("fine at today's sizes") | - |

#### #1624: UI phase 7 follow-ups, delivery, budgets, Jobs paging, shell

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1624.1 | Compression exclusions depend on Starlette's defaults | open | `dw/server/app.py:69`; `pyproject.toml:133` pins FastAPI only | T17 |
| 1624.2 | `test_a_gzipped_body_is_todays_body` does not prove compression | open | `tests/test_server_delivery.py` | T17 |
| 1624.3 | No test of a live event stream under compression | open | | T17 |
| 1624.4 | Compressed and identity bodies share one strong ETag | obsolete | matters only if a caching proxy is recommended; none is | - |
| 1624.5 | The theme script leaks `var theme` | open | `ui/index.html:90` | T17 |
| 1624.6 | The `max-height` keyframe has no real-browser test | open | | T17 |
| 1624.7 | The shell's copy is wrong with JavaScript off | open | `ui/index.html` | T17 |
| 1624.8 | `100vh` jumps on mobile | open | `ui/index.html:47` | T17 |
| 1624.9 | The JSON view is 1,350.6 KB of a 1,400 KB budget | needs Don | `ui/e2e/performance.spec.ts:14` | - |
| 1624.10 | The Monaco ESLint rule blocks only the bare import | open | `ui/eslint.config.js:31-36` (paths only) | T19 |
| 1624.11 | The budget allow-list accepts any `monaco-*` chunk | open | `performance.spec.ts:83-90` | T19 |
| 1624.12 | The JSON editor's failure copy names one cause | needs Don | phase 7's fixed copy | T17 |
| 1624.13 | Restart banners count loaded rows only | open | `pages/JobsPage.svelte:172` | T13 |
| 1624.14 | A `queued`/`running` filter on a partial page still says "Searching the newest" | open | | T13 |
| 1624.15 | Under Search all, older rows go stale | obsolete | accepted by ruling | - |
| 1624.16 | Reads are sequential | open | `lib/jobsListing.ts:163-166` | T13 |
| 1624.17 | The first-run link reads "Open Workflows" while loading | open | `pages/OverviewPage.svelte:152` | T17 |
| 1624.18 | The long job id's clipping is pinned by a regex | open | | T20 |
| 1624.19 | No test: a folded group while the sidebar is collapsed | open | | T17 |
| 1624.20 | A workspace switch drops harmless filters | open | `lib/routes.ts` `switchHref` | T17 |
| 1624.21 | Focus isn't restored when the sidebar is re-expanded | open | | T17 |
| 1624.22 | No test: the meter label with several jobs running | open | | T20 |
| 1624.23 | A bare invalid id gets the "org/name" hint | open | `pages/ModelsPage.svelte:15-24` | T17 |
| 1624.24 | The first-run journey's remote-URL check scans `main` only | open | `ui/e2e/journeys/first-run.spec.ts` | T20 |
| 1624.25 | The journey depends on `templates/text-to-image` | open | | T20 |
| 1624.26 | A `localhost:<port>` tunnel looks local, so Status offers the rebind snippet | open | `pages/ServerPage.svelte` | T17 |
| 1624.27 | Record "never advise rebinding through a proxy" in ARCHITECTURE | open | | T17 |

#### #1599: created / last-changed timestamps

| Row | Item | Status | Evidence | Task |
|---|---|---|---|---|
| 1599.1 | Inventory: each noun's server timestamps and the views that show them | open | | T21 |
| 1599.2 | One rule, one live `<Time>` component over `format.ts` | open | no `ui/Time*.svelte`; `format.ts:22-53` | T21 |
| 1599.3 | Fill the gaps; add missing times to the response models | open | | T22 |
| 1599.4 | Lists sort newest first by last change | open | | T22 |
| 1599.5 | Say who changed it where the server knows | open | | T22 |
| 1599.6 | A rule that no date is formatted outside the helper; an ARCHITECTURE row | open | `job/EventLog.svelte`, `request/reviewText.ts` call `toLocale*` | T22 |

#### Counts

| Issue | Rows | done | duplicate | obsolete | open | needs Don |
|---|---|---|---|---|---|---|
| #1598 | 29 | 0 | 0 | 1 | 28 | 0 |
| #1600 | 45 | 25 | 1 | 1 | 17 | 1 |
| #1601 | 25 | 8 | 2 | 3 | 11 | 1 |
| #1602 | 29 | 6 | 0 | 0 | 23 | 0 |
| #1603 | 4 | 0 | 0 | 0 | 4 | 0 |
| #1605 | 20 | 0 | 0 | 0 | 19 | 1 |
| #1606 | 15 | 4 | 1 | 0 | 10 | 0 |
| #1607 | 8 | 0 | 0 | 0 | 7 | 1 |
| #1608 | 13 | 0 | 0 | 0 | 13 | 0 |
| #1609 | 11 | 5 | 0 | 1 | 5 | 0 |
| #1610 | 9 | 2 | 1 | 0 | 6 | 0 |
| #1611 | 3 | 3 | 0 | 0 | 0 | 0 |
| #1612 | 35 | 1 | 0 | 0 | 34 | 0 |
| #1623 | 15 | 1 | 0 | 1 | 13 | 0 |
| #1624 | 27 | 0 | 0 | 2 | 23 | 2 |
| #1599 | 6 | 0 | 0 | 0 | 6 | 0 |
| **Total** | **294** | **55** | **5** | **9** | **219** | **6** |
