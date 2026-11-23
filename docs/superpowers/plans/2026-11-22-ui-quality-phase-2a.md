# UI quality phase 2a: connection, errors, loading, actions. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** after this phase, an outage or a missing token is one banner, not twenty "Failed to fetch"; every page recovers on its own; loading, empty and error never look alike; every error speaks the user's language; and a button that mutates can't be pressed twice.

**Architecture:** One `connection` store, fed by every API response and by the app shell's status poll, owns the server's state (`ok`, `offline`, `locked`) and an `epoch` that rises on reconnect and on an accepted token. `describeError()` maps any failure to a title and a remedy. Page data comes through `resource(load, { watch } | { live })`, which reloads quietly on its change kinds and on `connection.epoch`, and renders through `<LoadState>`. Mutating buttons go through `action()`. Phase 1's carry-overs come first.

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config; one small FastAPI change (`dw/server/deps.py`).

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`: principles P2, P3, P4 and P6, the building-blocks rows *Connection*, *Error copy*, *Page data*, *Actions* and *Viewing the UI as a whole*, the metrics `raw_error_toasts`, `empty_outside_loadstate` and `unwatched_reads`, and the survey items D8 and D13 (detail in `docs/superpowers/specs/2026-11-22-ui-duplication-survey.md` §8 and §13). The findings are SH-1, SH-2, SH-14, SH-15, XC-3..7, JB-1, JB-6, JB-7, JB-11 and GA-14 (`docs/superpowers/specs/2026-11-22-ui-audit.md`, Tier 1 rows 11-14 and 18, and 22).

**Why two plans.** The spec's phase 2 has two independent subsystems: this plan's shell (connection, errors, loading, actions), and change detection (the server's `writes()` declarations, the `MODELS`/`LORAS` kinds, the generated page table, changed-row marks, `DocumentEditor.watchRemote`). The second is `docs/superpowers/plans/2026-11-22-ui-quality-phase-2b.md`. **2a goes first**: 2b's UI half reads the `resource()` declarations this plan creates and builds its marks into them. 2b's server tasks (2b Tasks 1-2) depend on nothing here and may run in parallel on their own branch. The outage-and-token journey closes 2a, and the agent-edits journey closes 2b.

## Global constraints

- Server changes are behaviour-preserving for MCP (`dw_mcp/` is an HTTP client of the same routes). This plan's only server change is Task 5's `no_workspace` code, and the MCP message it produces stays byte-identical (pinned).
- Error details keep the `detail: {code, message, ...}` shape that `apiErrors.ts` (`errorCode`, `errorPayload`, `errorDetail`) already reads. No response model changes, so no OpenAPI regeneration in 2a.
- Security rules in `CLAUDE.md`: no new filesystem access, and no `eval`, `exec` or `shell=True`. `ux-capture.mjs` spawns the fixture server with an argument array, never a shell string.
- Commit messages use a scope, never `fix: #N`. Stage files explicitly; never `git add -A ui` (Don edits `ui/package.json` locally).
- UI checks run from `ui/`: `npm run check`, `npm run lint`, `npx vitest run <file>`. Python checks use `venv/bin/python -m pytest <file> -n0`.
- **Metrics gate: no metric rises.** develop is already red against `docs/stabilization/ui/baseline.json` (`complex_functions: 11 -> 14`, `prefix_literals: 0 -> 3`). Before Task 1, save the starting measurement: `node scripts/arch-metrics.mjs > "$SCRATCH/ui-start.json"`. Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-start.json"`, which must print nothing. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` may print only those two pre-existing lines.
- Python ratchet (`venv/bin/python scripts/arch_metrics.py`): **this plan adds no Python module** (`modules` stays at today's 281). `claude_md_lines` stays at 129: `ui/CLAUDE.md` rules are paid for by rewriting its existing lines, never by adding lines.
- No page file over 600 lines, and no function over complexity 10 or 60 lines. These pages are near the ceiling: `GalleryPage` 597, `ModelsPage` 596, `EditorPage` 593, `App.svelte` 579, `OverviewPage` 567. Logic a task adds to one of them moves into the `.ts` module the task names.
- WCAG contrast is guidance only, never a check.
- e2e runs on port 8971 (`playwright.config.ts`). Journeys live in `ui/e2e/journeys/`. `ux-capture.mjs` uses port 8972.
- A known flaky Python test that is not this phase's: `tests/test_ledger_job_link.py::TestDelete::test_a_request_made_again_under_the_id_takes_the_jobs_back` (a second-truncated `changed_at` against a float `created_at`). Don't chase it.
- Copy fixed by this plan (use it exactly):
  - offline banner: `"Can't reach the server. Retrying every few seconds."`, with the button **Retry now**
  - locked banner: `"This server needs an API token."`, with the button **Enter token**
  - token save: the toast `"Token accepted"`, or, inline in the popover, `"The server refused that token"` (the popover stays open)
  - load failure: `"Couldn't load <label>: <title>"`, then the remedy on its own line, and **Retry**. The editors keep phase 1's `"Couldn't open <label>: <title>"`.
  - refresh failure, with data kept: `"Couldn't refresh <label>: <title>"` and **Retry**
  - failure toast: `"Couldn't <doing>: <title>"`; its description is the remedy; it has the action **Copy details** when there is a detail
  - filtered empty: `"Nothing matches those filters."`, with the button **Clear filters**
  - cancel past 60 s: `"Cancel <workflow>? It has run for <formatDuration>; the work so far is lost."`, with **Cancel job** (danger) and **Keep running**
  - job finish toast: the existing `finishedText(job)`, with the action **Open**

## Review focus

1. **An outage while a list is on screen.** The rows stay (no blank, no "No prompts yet"), the banner shows, and when the server returns the page reloads once, keeping the filter text and the focus. Pinned in Task 10 (`keeps data when a quiet refresh fails`, `reloads quietly once when the epoch rises`) and Task 15's journey.
2. **A workspace switch while a load is in flight.** The old workspace's reply never shows in the new one. Pinned in Task 10 (`drops a reply overtaken by a workspace switch`).
3. **A wrong token.** The popover says it was refused, the banner stays `locked`, and nothing says "Saved". Pinned in Task 8 (`a refused token stays locked and says so`).
4. **Double-click on Run again or New seed, and a failure.** Exactly one job is queued; after a failure the button works again and the toast is mapped copy. Pinned in Task 6 (`refuses re-entry while busy`, `re-enables and reports after a rejection`).
5. **The server restarts under an open job page.** The page leaves "running" within one reconnect and shows the job as recorded, or that it is no longer on the server. Pinned in Task 13 (`a dropped stream re-reads the job and stops on a terminal one`).

---

### Task 1: Editor carry-overs: Start from asks about a broken draft; flush keeps IME and focus

Carry-overs 1 and 2 from phase 1.

**Files:**
- Modify: `ui/src/lib/pages/EditorPage.svelte` (`chooseStart`, ~l.252)
- Modify: `ui/src/lib/editorShell.svelte.ts` (`flush`, and `root`, which becomes a setter)
- Test: `ui/src/lib/editorShell.test.ts`, `ui/src/lib/pages/EditorPage.test.ts`

**Interfaces:**
- Produces:
  - `chooseStart` asks when `ed.unsaved()` (dirty **or** `jsonParseFailed`; it flushes first), with `danger: true`.
  - `DocumentEditor.root` is set through a setter that adds `compositionstart` and `compositionend` listeners on the element (and removes them from the old one). While composing, `flush()` does not blur the focused field; it still runs the registered flushers.
  - After a blur, `flush()` restores focus after the DOM update (`tick()`): to the same node if it is still connected; else to `root.querySelector` by the node's `id`, else by `[name="<name>"]`; else to nothing. `selectionStart` and `selectionEnd` come back for text fields. `flush()` stays synchronous: only the restore is deferred.

- [ ] **Step 1: Write the failing tests.** In `editorShell.test.ts`:
  - `flush does not blur a field mid-composition`: dispatch `compositionstart` on an input in `root`, then flush. No `blur` event fires, and the flushers still ran.
  - `flush restores focus to a field the commit re-rendered`: a harness whose blur replaces the input with a new one carrying the same `id`. After `await tick()`, `document.activeElement` is the new input, with the caret where it was.

  In `EditorPage.test.ts`: `Start from asks before discarding a JSON draft that does not parse`. Make the JSON draft unparseable, choose an example, and check that the dialog appears and that Cancel keeps the draft.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editorShell.test.ts src/lib/pages/EditorPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces. `EditorPage.svelte` must not grow: the one-line change replaces `ed.dirty`.
- [ ] **Step 4: Run** the two files and `npx vitest run src/lib/pages/PromptEditorPage.test.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): Start from asks on a broken draft; flush keeps IME and focus`.

### Task 2: Router carry-over: a second navigation while asking, and declined replaces

Carry-over 3. Both cases are fixed, not scoped out.

**Files:**
- Modify: `ui/src/lib/router.svelte.ts`
- Modify: `ui/src/lib/workspace.svelte.ts:82`, `ui/src/lib/pages/JobPage.svelte:89` (use `replaceRoute`)
- Modify: `ui/scripts/design-rules.test.ts` (a new rule)
- Test: `ui/src/lib/router.svelte.test.ts`

**Interfaces:**
- Produces:
  - `replaceRoute(hash: string): void` replaces the current entry with `hash` through the guards. Declined, the URL is put back with `history.replaceState(<committed state>, '', committed)` and no history traversal. Allowed, it settles at the committed index (no new index).
  - Every arriving entry is stamped on its `hashchange`, guarded or not: an unstamped entry gets `lastSeenIndex + 1`. So `undo()` always walks `committedIndex - indexOf()`, however many entries arrived while asking.
  - A `hashchange` while a guard is asking no longer returns early. It records the newest target and re-stamps it. When the guard answers: allowed settles on the entry the browser is on now (the user's latest choice), and declined walks back to the committed entry.
  - `parse()`'s legacy redirect uses `history.replaceState` + `settle()`, not `location.replace`, so it never raises a second `hashchange`.
  - The design rule `location.replace appears only in router.svelte.ts`.

- [ ] **Step 1: Write the failing tests:**
  - `a second navigation while a guard asks is not lost`: guard pending, navigate A then B, guard allows. `route.parts` is B and `location.hash` is B.
  - `declining after two navigations walks back to the committed entry`: `history.go` is called with `-2`, and the route is unchanged.
  - `a declined replaceRoute puts the committed hash back in place`: no `history.go`, `location.hash` is the committed hash, and `history.length` is unchanged.
  - In `design-rules.test.ts`: `location.replace appears only in the router`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/router.svelte.test.ts scripts/design-rules.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Keep `committed`, `committedIndex` and the `undoing` flag. Add `lastSeenIndex`, and a `replacing` flag set by `replaceRoute` before its `location.replace`.
- [ ] **Step 4: Run** the router tests, `npx vitest run src/lib/routes.test.ts src/lib/pages/JobPage.test.ts`, and `npx playwright test e2e/journeys/editor-safety.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(router): stamp every arrival, keep the latest target, guard replaces`.

### Task 3: `ux-capture.mjs`: the whole UI in one folder

The building-blocks row *Viewing the UI as a whole*. It is landed early so every later task's reviewer can capture before and after.

**Files:**
- Create: `ui/scripts/ux-capture.mjs`
- Modify: `ui/package.json` (only the `scripts` entry `"ux:capture": "node scripts/ux-capture.mjs"`; stage the file with `git add -p` so Don's local edits stay out)
- Test: `ui/scripts/ux-capture.test.ts`

**Interfaces:**
- Produces:
  - `node scripts/ux-capture.mjs [OUT]` starts `<python> ui/e2e/serve_fixture.py --port 8972` (the same interpreter choice as `playwright.config.ts`'s `e2ePython`, through `spawn(cmd, args)`) and waits for `/api/health`. It shoots every route in `light` and `dark` at 1440×900 and 390×844 into `OUT` (default `mkdtemp`), prints the folder, and stops the server even on failure.
  - `export function captureRoutes(): { name: string; hash: string }[]`: every `WS_SECTIONS` entry (`src/lib/routes.ts`) under `ws/default/`, plus `ws/default/workflows/e2e-card`, `ws/default/edit/e2e-card`, `ws/default/requests/e2e-launch`, `ws/default/jobs/e2erestart01`, every `SHARED_SECTIONS` entry under `shared/`, and every `SERVER_SECTIONS` entry under `server/`.
  - File names: `<name>-<theme>-<width>.png`.

- [ ] **Step 1: Write the failing test** `ui/scripts/ux-capture.test.ts` (node environment): `captureRoutes names every workspace section once and each fixture item route`. It compares against `WS_SECTIONS`.
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/ux-capture.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Export `captureRoutes` so the test needs no browser. Guard `main` with the `process.argv[1] === fileURLToPath(import.meta.url)` idiom from `arch-metrics.mjs`.
- [ ] **Step 4: Run** the test, then `npm run ux:capture`. Expected: PASS, and a folder holding 4 shots per route.
- [ ] **Step 5: Commit** `ui(scripts): ux-capture screenshots every route, both themes, two widths`.

### Task 4: Three new metrics, measured and baselined

The spec's mechanical enforcement for P2, P6 and P11 (the UI half).

**Files:**
- Modify: `ui/scripts/arch-metrics.mjs` (the header comment's counting rules, the `collector` rules, `metrics`)
- Modify: `docs/stabilization/ui/baseline.json` (add three keys; leave the existing values alone)
- Test: `ui/scripts/arch-metrics.test.ts`

**Interfaces:**
- Produces, as ESLint collector rules (so they parse `.svelte` the way the others do):
  - `raw_error_toasts`: a `notify.error(` call whose first argument's source text matches `/\.message\b|String\(\s*\w+\s*\)/`. **Measured today: 24.**
  - `empty_outside_loadstate`: an `ImportDeclaration` whose source ends in `/Empty.svelte`, in any file but `src/lib/ui/LoadState.svelte`. **Measured today: 7** (Assets, Gallery, Jobs, Overview, Prompts, Requests and Workflows pages).
  - `unwatched_reads`: in a `.svelte` file, a call `api.<m>(...)` where `m` matches `/^(get|list)/` or is in `READ_ALIASES`, and no ancestor is a call to `resource(` or `poll(`. **Measured today: 50** (53 calls, 3 of them lexically inside a `poll(` in `ServerPage` and `RequestPage`).
  - `export const READ_ALIASES = ['gallery', 'galleryMetadata', 'health', 'memory', 'diffusersStatus', 'server', 'eventLog', 'describeTask', 'describePipeline', 'describeClass']`: the `api` reads not named `get*`/`list*`. A new read named otherwise must be added here.
  - If a measured count differs from the numbers above, the measurement wins: write it to the baseline and say so in the commit message.

- [ ] **Step 1: Write the failing tests** with `measure(files)` over temp fixtures, in the style of the existing `measure` tests:
  - `counts a raw error toast and not a described one`: `notify.error(e.message)` and `notify.error(String(e))` count, and `notify.failure('save', e)` does not.
  - `counts an Empty import outside LoadState only`
  - `counts a read outside resource and poll, not inside`: `api.listJobs()` in a handler counts, while `resource(() => api.listJobs(), …)` and `poll(() => api.health(), 5000)` don't. A `.ts` file never counts.
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/arch-metrics.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement**, then record the measured values in the baseline under the three new keys.
- [ ] **Step 4: Run** the test, then the metrics gate from Global constraints. Expected: PASS, with nothing printed.
- [ ] **Step 5: Commit** `ui(metrics): raw_error_toasts, empty_outside_loadstate, unwatched_reads`.

### Task 5: `describeError`, `notify.failure`, and a named `no_workspace` refusal

P6, and carry-over 4.

**Files:**
- Modify: `ui/src/lib/apiErrors.ts`, `ui/src/lib/toast.ts`
- Modify: `ui/src/lib/ui/LoadFailed.svelte`
- Modify: `dw/server/deps.py` (a helper `no_such_workspace(name) -> HTTPException`), `dw/server/routes/library.py:212, 302`
- Modify: `dw_mcp/client.py:542` (the hint also applies when `detail.code == "no_workspace"`)
- Modify: `tests/test_workspace_copy.py:766`, `tests/test_jobs_listing.py:133` (read `detail["message"]`)
- Test: `ui/src/lib/apiErrors.test.ts`, `ui/src/lib/ui/LoadFailed.test.ts`, `tests/test_mcp_workspaces.py`, `tests/test_server.py`

**Interfaces:**
- Produces (TypeScript):
  - `type ErrorKind = 'offline' | 'locked' | 'forbidden' | 'no_workspace' | 'not_found' | 'refused' | 'server' | 'json' | 'unknown'`
  - `type Described = { kind: ErrorKind; title: string; remedy?: string; detail?: string; code?: string; status?: number }`
  - `describeError(e: unknown, what?: string): Described`, mapped in this order:

    | Input | kind | title | remedy |
    |---|---|---|---|
    | `TypeError` (a fetch that never got a response) | offline | `Can't reach the server` | `Check that dw.serve is running. This page reloads when it's back.` |
    | 401 | locked | `The server needs an API token` | `Enter it with the key button in the header.` |
    | 403 | forbidden | the message's first line | — |
    | 404, code `no_workspace` | no_workspace | `This workspace doesn't exist` | `Pick another one in the sidebar.` |
    | 404 | not_found | `` `${what ?? 'That'} wasn't found` `` | — |
    | 400, 409, 422 | refused | the message's first line, cut to 160 characters at a word | — |
    | ≥ 500 | server | `The server hit an error` | `Try again. The server log has the detail.` |
    | `SyntaxError` | json | `The JSON doesn't parse` | — |
    | anything else | unknown | `Something went wrong` | — |

    `detail` is the full message whenever it is longer than the title, and `code` is `ApiError.code`.
  - `notify.failure(doing: string, e: unknown, id?: string)`: `toast.error("Couldn't <doing>: <title>", { description: remedy, duration: ∞, id, action: detail ? { label: 'Copy details', onClick: copies detail } : undefined })`.
  - `LoadFailed` shows `"Couldn't open <label>: <title>"` and the remedy, and offers **Start a new one named <label>** only for `kind === 'not_found'` (an unknown workspace no longer offers it).
- Produces (HTTP): an unknown workspace answers 404 with `detail: {code: "no_workspace", message: "No such workspace: <name>"}` from all three raise sites. The MCP tool message is unchanged: `"No such workspace: <name> - list_workspaces shows what exists; use_workspace switches."`

- [ ] **Step 1: Write the failing tests:**
  - `apiErrors.test.ts`: one `describes …` case per table row. Also `cuts a long refusal to its first line and keeps the rest as detail`.
  - `LoadFailed.test.ts`: `offers to start a new one for a missing file, not for a missing workspace`.
  - `tests/test_server.py`: `test_an_unknown_workspace_is_a_named_404` (`detail.code == "no_workspace"`).
  - `tests/test_mcp_workspaces.py`: `test_a_coded_no_workspace_keeps_the_recovery_hint`, with the dict detail and the exact string above.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/apiErrors.test.ts src/lib/ui/LoadFailed.test.ts` and `venv/bin/python -m pytest tests/test_server.py tests/test_mcp_workspaces.py -n0 -k workspace`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `deps.py`, `workspace_for` and the two `library.py` sites raise `no_such_workspace(name)`.
- [ ] **Step 4: Run** the tests above, plus `venv/bin/python -m pytest tests/test_workspace_copy.py tests/test_jobs_listing.py tests/test_mcp_client.py tests/test_api_contract.py -n0`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(errors): describeError and notify.failure; server(workspaces): coded no_workspace 404`.

### Task 6: `action()` and `ActionButton`; the job page's actions

P4, JB-6 and JB-7.

**Files:**
- Create: `ui/src/lib/action.svelte.ts`, `ui/src/lib/ui/ActionButton.svelte`
- Modify: `ui/src/lib/job/JobHeader.svelte` (`rerun`, `exportJob`, the Cancel button ~l.279), `ui/src/lib/job/RerunWithChanges.svelte` (~l.62, 106), `ui/src/lib/job/ChainSegments.svelte:133`
- Test: `ui/src/lib/action.svelte.test.ts`, `ui/src/lib/job/JobHeader.test.ts`

**Interfaces:**
- Consumes: `notify.failure` (Task 5); `confirmDialog` (phase 1).
- Produces:
  - `type ConfirmSpec = { message: string; title?: string; confirmLabel: string; cancelLabel?: string; danger?: boolean }`
  - `action<A extends unknown[], R>(fn: (...args: A) => Promise<R>, opts: { doing: string | ((...args: A) => string); confirm?: (...args: A) => ConfirmSpec | null; success?: string | ((r: R) => string); undo?: (r: R) => { message: string; revert: () => Promise<unknown> } }): Action<A, R>`
  - `interface Action<A, R> { run(...args: A): Promise<R | undefined>; readonly busy: boolean }`. `run` returns `undefined` without calling `fn` while busy, when the confirm is declined, or on failure. It never rejects: a failure is `notify.failure(doing, e)`. `undo` shows `notify.undo(message, revert)`, and a failed revert is itself a `notify.failure`.
  - `ActionButton` props: `{ act: Action<A, R>; args?: A; children: Snippet; danger?: boolean; disabled?: boolean; title?: string; class?: string }`. It is disabled and `aria-busy` while `act.busy`.
  - In `JobHeader`: Run again, New seed, Export and Cancel are `ActionButton`s. Cancel's `confirm` returns the cancel copy (Global constraints) only when the job has run for more than 60 s (`now - job.started_at > 60`), else `null`. Cancel's `doing` is `` `cancel ${leaf}` ``, so a 404 reads `"Couldn't cancel e2e-card: That wasn't found"`.

- [ ] **Step 1: Write the failing tests.** In `action.svelte.test.ts`:
  - `refuses re-entry while busy`: two `run()` calls give one `fn` call.
  - `re-enables and reports after a rejection`: `busy` is false afterwards, and `notify.failure` was called with `doing`.
  - `a declined confirm never calls fn`
  - `success and undo toasts`

  In `JobHeader.test.ts`:
  - `a double click on Run again queues one job`
  - `cancel asks only past 60 seconds`
  - `a failed cancel is reported`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/action.svelte.test.ts src/lib/job/JobHeader.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** The four `notify.error(e instanceof Error …)` sites in these three files (`JobHeader.svelte:84,123`, `RerunWithChanges.svelte:111`, `ChainSegments.svelte:136`) go away with the change.
- [ ] **Step 4: Run** `npx vitest run src/lib/job` and the metrics gate. Expected: PASS, and `raw_error_toasts` is lower by 4 (24 to 20). Lower the baseline value to match.
- [ ] **Step 5: Commit** `ui(actions): action() and ActionButton; job actions guarded, cancel confirms past a minute`.

### Task 7: Every remaining raw error toast goes through `describeError`

P6. `raw_error_toasts` reaches 0.

**Files:**
- Modify: the remaining sites the metric lists: `assets/uploads.ts:159`, `editorShell.svelte.ts:185`, `gallery/GalleryDetail.svelte:131,136`, `pages/AssetsPage.svelte:220,305`, `pages/EditorPage.svelte:266,296,335,355`, `pages/GalleryPage.svelte:122,143`, `pages/JobsPage.svelte:186`, `pages/PromptEditorPage.svelte:109,215,253`, `request/ItemNote.svelte:163`, `restartBatches.ts:61,99`, `workspaceActions.ts:96`
- Test: the existing tests of those files, updated where they assert the old raw text

**Interfaces:**
- Consumes: `notify.failure`, `describeError` (Task 5).
- Produces:
  - Each site becomes `notify.failure('<verb phrase>', e)`, where the verb phrase names the thing: `'save <label>'`, `'delete <n> files'`, `'upload <name>'`, `'start over from <example>'`, `'run <label>'`. `uploads.ts`'s batch line lists `"<name>: <title>"` per file.
  - `editorShell`'s JSON parse toast reads `` `JSON: ${describeError(e).detail ?? describeError(e).title}` ``. Here the parser's position text is the useful part, and it is the user's own text, not engine text.

- [ ] **Step 1: Update the tests first.** Any test that asserts a raw message now asserts the mapped copy. For example, in `GalleryPage.test.ts`, a failed delete's toast reads `"Couldn't delete 2 files: …"`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages src/lib/gallery src/lib/assets src/lib/request/ItemNote.test.ts src/lib/restartBatches.test.ts`. Expected: FAIL where updated.
- [ ] **Step 3: Implement.** No page grows: each replacement is the same length or shorter.
- [ ] **Step 4: Run** the whole `npx vitest run`, `npm run lint` and the metrics gate. Expected: PASS, and `raw_error_toasts: 0`. Set it to 0 in the baseline, which from now on holds the rule (no ESLint rule is added: 2b's `no-restricted-syntax` block would replace it).
- [ ] **Step 5: Commit** `ui(errors): every error toast is described copy; raw_error_toasts reaches 0`.

### Task 8: The `connection` store, the banner and a token that takes effect

P3, SH-1, SH-2 (the header), XC-3 and XC-4.

**Files:**
- Create: `ui/src/lib/connection.svelte.ts`, `ui/src/lib/ui/ConnectionBanner.svelte`
- Modify: `ui/src/lib/apiRequest.ts` (`send` reports), `ui/src/lib/statusPoll.ts` (the status read moves here from `App.svelte`), `ui/src/App.svelte` (the banner, and the header word), `ui/src/lib/TokenPopover.svelte`, `ui/src/lib/workspace.svelte.ts` (reload the listing on an epoch rise), `ui/src/lib/api.ts` (`checkToken`)
- Test: `ui/src/lib/connection.svelte.test.ts`, `ui/src/lib/TokenPopover.test.ts`, `ui/src/lib/statusPoll.test.ts`, `ui/src/lib/ui/ConnectionBanner.test.ts`

**Interfaces:**
- Produces:
  - `type ConnectionState = 'ok' | 'offline' | 'locked' | 'unknown'`
  - `connection: { state: ConnectionState; epoch: number; health: HealthInfo | null; memory: MemoryInfo | null }`, a module `$state` object. It imports nothing from `api`, so there is no cycle.
  - `reportResponse(status: number)`: a 401 gives `locked`. Any other status gives `ok`, and `epoch++` when the state was `offline` or `locked` (leaving `unknown` doesn't raise it).
  - `reportUnreachable()` gives `offline`. `send()` calls it on a `TypeError` from `fetch` (an `AbortError` is not an outage), then rethrows.
  - `onReconnect(fn: () => void): () => void`: called on every epoch rise.
  - `statusPoll.ts`: `startStatusPoll(): () => void` keeps the current `pollStatus` cadence. It writes `connection.health` and `connection.memory` (settled apart, as today) and calls `setAttentionBackends`. `checkNow(): Promise<void>` reads them once (the banner's **Retry now**).
  - `api.checkToken(): Promise<void>`: an unscoped `GET /api/workspaces`.
  - `TokenPopover` save: `setApiToken(value)`, then `checkToken()`. A 2xx closes the popover and toasts `"Token accepted"`. The epoch rise comes from `send()`'s own `reportResponse`, since the state was `locked`, so no separate call is needed. A 401 shows `"The server refused that token"` inline and leaves the popover open. An offline result shows the offline title inline.
  - `workspace.svelte.ts` registers `onReconnect(() => { invalidateWorkspaces(); void loadWorkspaces() })`, so a `['default']` fallback from a locked start is replaced.
  - The `App.svelte` header shows `offline` or `locked` in place of `idle` when the state says so. `ConnectionBanner` props: `{ ontoken: () => void }`. It renders nothing in `ok` or `unknown`.

- [ ] **Step 1: Write the failing tests:**
  - `connection.svelte.test.ts`:
    - `a 401 locks and the next 2xx raises the epoch`
    - `leaving unknown does not raise the epoch`
    - `a fetch TypeError is offline and an AbortError is not`
    - `onReconnect is called once per rise`
  - `TokenPopover.test.ts`:
    - `an accepted token raises the epoch and closes`
    - `a refused token stays locked and says so`: no "Saved" or "Token accepted", and the popover is still open.
  - `statusPoll.test.ts`: `a failed health read leaves memory and health null and reports offline`.
  - `ConnectionBanner.test.ts`: `offline shows Retry now and locked shows Enter token`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/connection.svelte.test.ts src/lib/TokenPopover.test.ts src/lib/statusPoll.test.ts src/lib/ui/ConnectionBanner.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `App.svelte`'s `refreshStatus` and its `health`/`memory` state go; it reads `connection.health`. `App.svelte` must end shorter than it starts.
- [ ] **Step 4: Run** `npx vitest run`, `npm run check` and the metrics gate. Expected: PASS, and `unwatched_reads` is lower by 2 (`App.svelte`'s `health`/`memory` calls).
- [ ] **Step 5: Commit** `ui(connection): one connection state and banner; the token takes effect at once`.

### Task 9: `ServerHealthList`: one set of status rows

D8, SH-2 (the popover), SH-14 and XC-6.

**Files:**
- Create: `ui/src/lib/ServerHealthList.svelte`
- Modify: `ui/src/lib/StatusPopover.svelte`, `ui/src/lib/pages/ServerPage.svelte` (its own health/memory `poll` goes, and so does the token blame at ~l.71)
- Modify: `ui/src/app.css` (`dl.kv`, the grid both copies share; see survey §8)
- Test: `ui/src/lib/ServerHealthList.test.ts`, `ui/src/lib/StatusPopover.test.ts`, `ui/src/lib/pages/ServerPage.test.ts`

**Interfaces:**
- Consumes: `connection` (Task 8), `describeError` (Task 5).
- Produces:
  - `ServerHealthList` props: `{ onnavigate?: () => void }`. It reads `connection.state`, `connection.health` and `connection.memory`, and renders the Server, Worker(s), Queue and Memory rows as `dl.kv`.
  - With `state` `offline`, the Server row reads `unreachable` in `--bad`; with `locked`, `needs the API token`. In either case there is no worker row text: never "not started" without a health reading.
  - `ServerPage` shows `describeError(e, 'the server details')` for its `api.server()` failure. Only a 401 mentions the token.

- [ ] **Step 1: Write the failing tests:**
  - `unreachable is red and says nothing about the worker`
  - `locked names the token`
  - `a live reading shows the worker and the queue`
  - In `ServerPage.test.ts`: `a 500 does not blame the token`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ServerHealthList.test.ts src/lib/StatusPopover.test.ts src/lib/pages/ServerPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Both copies of the rows, and their `dl/dt/dd` CSS, are deleted.
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(status): ServerHealthList is the one set of server status rows`.

### Task 10: `resource()` and `<LoadState>`

P2, and the core of D13 and XC-7.

**Files:**
- Create: `ui/src/lib/resource.svelte.ts`, `ui/src/lib/ui/LoadState.svelte`
- Modify: `eslint.config.js` (`no-restricted-imports` of `**/Empty.svelte`)
- Test: `ui/src/lib/resource.svelte.test.ts`, `ui/src/lib/ui/LoadState.test.ts`

**Interfaces:**
- Consumes: `changes` (`changes.svelte.ts`), `poll`, `connection.epoch` (Task 8), `describeError`.
- Produces:
  - `type Freshness = { watch: readonly ChangeKind[]; alsoLive?: () => number | null; live?: never } | { live: { everyMs: number }; watch?: never; alsoLive?: never }`. Exactly one of `watch` and `live` is required by the type. `watch: []` declares data that never changes (a schema). `alsoLive` adds a timer at the interval it returns while it is non-null (the Models downloads, the request page).
  - `type ResourceOptions<T> = Freshness & { workspace?: () => string; merge?: (held: T | undefined, incoming: T) => T }`. `workspace` defaults to `workspace.current`, and `merge` to taking the incoming value.
  - `resource<T>(load: () => Promise<T>, options: ResourceOptions<T>): Resource<T>`. It is called during component init and registers its effects there.
  - `interface Resource<T> { readonly status: 'loading' | 'ready' | 'error'; readonly data: T | undefined; readonly error: unknown; readonly stale: boolean; reload(): Promise<void> }`
  - Rules:
    - Reactive values that `load` reads before its first `await` are dependencies. A change reloads with `status = 'loading'`, keeping `data` unless `workspace()` changed (then `data = undefined`).
    - A reply overtaken by a later load is dropped (the `latestOnly` pattern).
    - A **quiet** reload (a feed kind moved, `connection.epoch` rose, a live tick) never sets `loading`. On failure it keeps `data` and `status` and sets `stale = true`.
    - A non-quiet failure sets `status = 'error'`.
    - `reload()` is non-quiet.
    - Timers go through `poll` (paused while hidden), and the feed through `changes(workspace()).watch`.
  - `LoadState` props: `{ of: Resource<T>; label: string; empty?: boolean; filtered?: boolean; onclear?: () => void; emptyIcon?: Snippet; emptyText?: Snippet; skeleton?: Snippet; children: Snippet<[T]> }`. It renders the first that applies:
    1. `loading` with no data: `skeleton`, or `<Skeleton/>`.
    2. `error` with no data: the load-failure copy (Global constraints), with **Retry** calling `of.reload()`.
    3. `filtered && empty`: the filtered-empty copy, with **Clear filters** calling `onclear`.
    4. `empty`: `<Empty>` with `emptyIcon` and `emptyText`.
    5. Otherwise `children(data)`, with the refresh-failure line above it when `error` or `stale` holds and data is kept.
  - ESLint: `**/Empty.svelte` is a restricted import everywhere but `src/lib/ui/LoadState.svelte`. Flat config replaces a rule's options instead of merging them, so the pattern joins the existing bits-ui `no-restricted-imports` block, and a second block for `src/lib/ui/**` (ignoring `LoadState.svelte`) carries the Empty pattern alone. The rule fails until Tasks 11-12 land, so it is added **disabled** (`'off'`, with a comment naming Task 12) and switched on in Task 12.

- [ ] **Step 1: Write the failing tests.** In `resource.svelte.test.ts` (with `feedMock` and fake timers):
  - `loads once and is ready`
  - `drops a reply overtaken by a workspace switch`
  - `keeps data across a filter change, clears it across a workspace switch`
  - `keeps data when a quiet refresh fails` (`stale` is true and `status` is `ready`)
  - `reloads quietly once when the epoch rises`
  - `reloads quietly on a watched kind and not on another`
  - `live polls at its interval and pauses while hidden`
  - `alsoLive polls only while it returns a number`
  - `merge decides what is held`

  In `LoadState.test.ts`:
  - `loading is a skeleton, not empty`
  - `an error with no data offers Retry`
  - `filtered empty offers Clear filters`
  - `a failed refresh keeps the rows and says so`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/resource.svelte.test.ts src/lib/ui/LoadState.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Keep each function at complexity 10 or less: the trigger kinds dispatch to one `run({ quiet })`.
- [ ] **Step 4: Run** the tests, `npm run lint` and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(data): resource() with declared freshness, and LoadState`.

### Task 11: The library pages load through `resource`: Prompts, Workflows, Assets and Gallery

D13, XC-5 and XC-7 for these four pages, and GA-14.

**Files:**
- Modify: `ui/src/lib/pages/PromptsPage.svelte`, `WorkflowsPage.svelte`, `AssetsPage.svelte`, `GalleryPage.svelte`, `ui/src/lib/gallery/GalleryDetail.svelte` (metadata, ~l.80-95), `ui/src/lib/assets/AssetDetail.svelte` (~l.33-49)
- Test: each page's existing `*.test.ts`, plus `ui/src/lib/gallery/GalleryDetail.test.ts`

**Interfaces:**
- Consumes: `resource`, `LoadState` (Task 10).
- Produces:
  - Each page's loader, its `latestOnly`, its `$effect(void workspace.current)` and its `changes().watch` become one `resource`. They keep today's kinds: Prompts `['prompts']`; Workflows `['workflows']`, plus `['gallery']` for its proofs; Assets `['assets']`; Gallery `['gallery']`, with `merge` keeping `picks` as today.
  - Each page renders through `LoadState` with labels `'prompts'`, `'workflows'`, `'the asset library'` and `'the gallery'`. A page with a filter passes `filtered` and `onclear`.
  - `GalleryDetail`'s metadata is `resource(() => api.galleryMetadata(name), { watch: ['gallery'] })` under `LoadState label="what made this file"`. A failed read now shows the load-failure copy, not "no embedded metadata" (GA-14).
  - `AssetDetail` gets the same treatment, watching `['assets']`.

- [ ] **Step 1: Write the failing tests:**
  - In each page test: `shows a skeleton while loading, not the empty state`, and `a failed first load offers Retry and Retry loads`.
  - `Prompts filtered to nothing offers Clear filters`
  - `GalleryDetail.test.ts`: `a failed metadata read is not "no embedded metadata"`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/PromptsPage.test.ts src/lib/pages/WorkflowsPage.test.ts src/lib/pages/AssetsPage.test.ts src/lib/pages/GalleryPage.test.ts src/lib/gallery src/lib/assets`. Expected: FAIL.
- [ ] **Step 3: Implement.** `GalleryPage` (597 lines) must end shorter than it starts. If `merge` for picks needs more than a line, it goes in `ui/src/lib/gallery/order.ts` as `keepPicks(held, incoming)`.
- [ ] **Step 4: Run** the tests above, `npx playwright test e2e/freshness.spec.ts` and the metrics gate. Expected: PASS, with `unwatched_reads` and `empty_outside_loadstate` lower. Lower both in the baseline.
- [ ] **Step 5: Commit** `ui(pages): prompts, workflows, assets and gallery load through resource and LoadState`.

### Task 12: Requests, Jobs, Overview and Compare through `resource`

D13 for the remaining list pages. `empty_outside_loadstate` reaches 0.

**Files:**
- Modify: `ui/src/lib/pages/RequestsPage.svelte`, `JobsPage.svelte`, `OverviewPage.svelte`, `ComparePage.svelte`, `ui/src/lib/gallery/RequestFilter.svelte` (~l.32-44)
- Modify: `eslint.config.js` (turn Task 10's Empty rule on)
- Test: the four page tests, `ui/src/lib/gallery/RequestFilter.test.ts`

**Interfaces:**
- Consumes: Task 10.
- Produces:
  - Requests: `{ watch: ['requests'] }`.
  - Jobs: `{ live: { everyMs: 3000 } }` (live state, as today; JB-17's paging is phase 7). Its "clear filters" moves into `LoadState`'s filtered state.
  - Overview: one `resource` per panel, keeping today's six kinds (gallery, jobs, workflows, requests, assets, workspaces). Each panel has its own `LoadState`, so one failed panel doesn't blank the page.
  - Compare: `{ watch: ['gallery'] }`, with `merge` keeping each cell's metadata.
  - RequestFilter: `{ watch: ['requests'] }`.

- [ ] **Step 1: Write the failing tests:**
  - `JobsPage.test.ts`: `loading shows a skeleton, not "No jobs yet"`.
  - `OverviewPage.test.ts`: `one panel failing leaves the others`.
  - `RequestsPage.test.ts`: `recovers when the server returns`, by raising the epoch.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/RequestsPage.test.ts src/lib/pages/JobsPage.test.ts src/lib/pages/OverviewPage.test.ts src/lib/pages/ComparePage.test.ts src/lib/gallery/RequestFilter.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `OverviewPage` (567 lines) must end shorter.
- [ ] **Step 4: Run** `npx vitest run`, `npm run lint` (the Empty rule now on) and the metrics gate. Expected: PASS, and `empty_outside_loadstate: 0`. Set it to 0 in the baseline.
- [ ] **Step 5: Commit** `ui(pages): requests, jobs, overview and compare load through resource; Empty only via LoadState`.

### Task 13: The pages that didn't recover: Job, Workflow and Models; a dropped job stream

XC-5 for the pages the audit named, and JB-1.

**Files:**
- Modify: `ui/src/lib/api.ts` (`streamJobEvents` gains `onDrop`), `ui/src/lib/pages/JobPage.svelte`, `ui/src/lib/pages/WorkflowPage.svelte`, `ui/src/lib/pages/ModelsPage.svelte`
- Test: `ui/src/lib/api.test.ts`, `ui/src/lib/pages/JobPage.test.ts`, `ui/src/lib/pages/WorkflowPage.test.ts`, `ui/src/lib/pages/ModelsPage.test.ts`

**Interfaces:**
- Consumes: Tasks 5, 8 and 10.
- Produces:
  - `streamJobEvents(jobId, after, onEvent, onEnd, onDrop?: () => void)`: `source.onerror` calls `onDrop`. EventSource keeps its own reconnect, so the stream isn't closed there.
  - JobPage:
    - The job is `resource(() => api.getJob(jobId), { watch: ['jobs', 'gallery'] })` under `LoadState label="job <id>"`. `gallery` is there because a run note and the run's outputs move `gallery`, not `jobs`.
    - Its definition read is a second resource with `watch: []`.
    - `onDrop` calls `job.reload()`. When the reload answers a terminal status or `historical`, the stream is stopped and the page shows that record. A 404 shows the load-failure copy (`"Couldn't load job <id>: job <id> wasn't found"`).
    - An epoch rise reopens a closed stream for a job that is still running.
  - WorkflowPage: the definition is `{ watch: ['workflows'] }`, the proofs are `{ watch: ['gallery'] }`, and both render under `LoadState`.
  - ModelsPage:
    - The cache list is `{ watch: [] }` until 2b adds `models`.
    - Downloads are `{ watch: [], alsoLive: () => (active ? 2000 : null) }`.
    - Diffusers is `{ watch: [], alsoLive: () => (updating ? 2000 : null) }`.
    - Each reloads on an epoch rise, and each renders under `LoadState`. The page's two hand-made `poll`s go.

- [ ] **Step 1: Write the failing tests:**
  - `api.test.ts`: `onDrop is called on a stream error and the stream is left open`.
  - `JobPage.test.ts`:
    - `a dropped stream re-reads the job and stops on a terminal one`
    - `a job gone after a restart says so`
    - `the page recovers when the server returns`
  - `WorkflowPage.test.ts`: `a failed load offers Retry and recovers on reconnect`.
  - `ModelsPage.test.ts`: `recovers on reconnect`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/api.test.ts src/lib/pages/JobPage.test.ts src/lib/pages/WorkflowPage.test.ts src/lib/pages/ModelsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `ModelsPage` (596 lines) must end shorter.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/job-event-log.spec.ts` and the metrics gate. Expected: PASS. Lower `unwatched_reads` in the baseline.
- [ ] **Step 5: Commit** `ui(pages): job, workflow and models recover; a dropped job stream re-reads the job`.

### Task 14: Finish toasts for short jobs, with an Open button

SH-15 and JB-11.

**Files:**
- Modify: `ui/src/lib/jobWatch.ts` (adopted ids), `ui/src/lib/finishAlert.ts` (`followJobs`, moved from `App.svelte`'s `watchJobs`/`announce`), `ui/src/lib/api.ts` (`submitJob`, `rerunJob` and `enhance` adopt the id they return), `ui/src/App.svelte`
- Test: `ui/src/lib/jobWatch.test.ts`, `ui/src/lib/finishAlert.test.ts`

**Interfaces:**
- Consumes: `connection.health` (Task 8).
- Produces:
  - `adoptJob(id: string): void` and `nextWatch(watched, info, adopted?: ReadonlySet<string>)`. An adopted id counts as watched from the moment it was submitted, so a job that ends before the next poll still ends, once. An adopted id is dropped after it has been announced.
  - `followJobs(info: HealthInfo | null, ctx: { onItsPage: (id: string) => boolean }): Promise<void>` in `finishAlert.ts`. It owns the title state and the announce, and `App.svelte` calls it from an `$effect` on `connection.health`.
  - The success toast is `notify.news(finishedText(job), { label: 'Open', onClick: go to the job })`. The failure toast is `notify.error` with the same action, through a new optional `action` parameter on `notify.error`.

- [ ] **Step 1: Write the failing tests:**
  - `jobWatch.test.ts`: `an adopted job that finished before the next poll ends once`.
  - `finishAlert.test.ts`:
    - `announces with an Open action that goes to the job`
    - `does not announce on the job's own page`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/jobWatch.test.ts src/lib/finishAlert.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `App.svelte` loses `watchJobs`, `announce` and their two `api.getJob` calls.
- [ ] **Step 4: Run** `npx vitest run` and the metrics gate. Expected: PASS, and `unwatched_reads` is lower by 2.
- [ ] **Step 5: Commit** `ui(jobs): short jobs get their finish toast, with Open`.

### Task 15: The outage-and-token journey, and the rules for the next agent

The phase's journey gate, and P2, P3, P4 and P6 in `ui/CLAUDE.md`.

**Files:**
- Create: `ui/e2e/journeys/outage-and-token.spec.ts`
- Modify: `ui/CLAUDE.md` (rewrite lines; add none), `docs/ARCHITECTURE.md` (UI section: new rows *Connection state*, *Error copy*, *Page data*, *Actions*; update *Keeping UI views fresh*)

**Interfaces:**
- Consumes: everything above, and the fixture server on port 8971.

- [ ] **Step 1: Write the journey.** Each case is its own `test`:
  1. Open Prompts, then `page.route('**/api/**', r => r.abort())`. The offline banner shows within one status poll, and the prompt rows stay. Unroute. The banner goes, and the rows reload with no navigation (`framenavigated` count unchanged).
  2. Answer `/api/**` with 401 unless `Authorization: Bearer e2e-token` is sent. The banner reads locked, and the header reads `locked`. Click **Enter token**, type a wrong token, and the popover says it was refused. Type `e2e-token`: the toast says "Token accepted", and the Workflows page and the sidebar's workspace list fill with no reload.
  3. Job page: with the events route aborted, the page leaves "running" for the job's recorded state after a reload of its data.
  4. Run `e2e-card` from its workflow page: the finish toast appears with **Open**, and Open goes to the job.
  5. Double-click **Run again** on that job: exactly one new job appears in `/api/jobs`.
- [ ] **Step 2: Run** `cd ui && npx playwright test e2e/journeys/outage-and-token.spec.ts`, then the whole `npx playwright test`. Expected: PASS.
- [ ] **Step 3: Write the rules.** `ui/CLAUDE.md` keeps its line count.
  - Bullet 2's freshness sentence becomes: "Page data is a `resource(load, { watch } | { live })` (`src/lib/resource.svelte.ts`) rendered through `<LoadState>` (only it imports `Empty`); a mutating button is an `action()` / `ActionButton`; an error shown to the user is `describeError`'s copy (`notify.failure`), never `e.message`; the server's state is `connection`, never a page's."
  - The sentence it replaces ("every other view of server data reloads quietly … `changes(workspace).watch` …") goes, and so does the pointer to the comment table.

  Add the ARCHITECTURE rows. Each names its owner module, the rule and its test (`connection.svelte.test.ts`, `apiErrors.test.ts`, `resource.svelte.test.ts`, `action.svelte.test.ts`, and the journey).
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, the metrics gate, and `venv/bin/python scripts/arch_metrics.py` (`modules` 281, `claude_md_lines` 129). Expected: PASS.
- [ ] **Step 5: Commit** `ui(e2e): outage-and-token journey; rules for page data, actions and errors`.

---

## Phase exit

- Every 2a finding (SH-1, SH-2, SH-14, SH-15, XC-3..7, JB-1, JB-6, JB-7, JB-11 and GA-14) and carry-overs 1-4 are fixed and pinned by a test named above.
- `raw_error_toasts` and `empty_outside_loadstate` are 0 in `docs/stabilization/ui/baseline.json`. `unwatched_reads` is at its measured remainder, which 2b takes to 0.
- No metric is above `$SCRATCH/ui-start.json`, and there is no new Python module.
- `docs/superpowers/plans/2026-11-22-ui-quality-phase-2b.md` is checked against the code as it now stands (the remaining `unwatched_reads` sites) before it starts.
