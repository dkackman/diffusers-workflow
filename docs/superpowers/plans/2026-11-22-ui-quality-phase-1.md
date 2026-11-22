# UI quality phase 1: data safety. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** after this phase, nothing the user typed or uploaded is lost, and "Saved" appears only when the server actually wrote what is on screen.

**Architecture:** The server gains opt-in HTTP preconditions on workflow and prompt writes and on uploads. With no header, behaviour is unchanged, so MCP tools are unaffected. On the client, `DocumentEditor` gains four things:
- a flush registry, run before every action
- a load state, so a failed load can never be saved over the real file
- the ETag it last read
- a router-level leave guard

One `saveDocument()` turns 409s into one shared conflict dialog. Both editors and the asset upload use it, so the three behave the same.

**Tech stack:** FastAPI (`dw/server/routes/library.py`, `assets.py`), Svelte 5 runes, vitest with @testing-library/svelte, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md` (principles P1 and P5, the building-blocks table rows *Conditional writes*, *Save with preconditions* and *Editor flush, load state, leave guard*). Its finding IDs are listed in the appendix, and the full audit is in `docs/superpowers/specs/2026-11-22-ui-audit.md`.

## Global constraints

- Server changes are opt-in by request header. A request without `If-Match` / `If-None-Match` behaves exactly as today. MCP (`dw_mcp/`) is unchanged and its tests stay green.
- 409 refusals use the existing `detail: {code, message, ...}` shape that `ui/src/lib/apiErrors.ts` (`errorCode`, `errorPayload`) already reads. The codes are exactly `exists` and `stale_write`.
- ETag value: `"` + the first 32 hex characters of the SHA-256 of the file's bytes + `"`.
- Paths go through the existing validators (`resolve_writable_workflow`, `resolve_prompt_name`, `validate_output_path`). No new filesystem access (CLAUDE.md security rules, CodeQL model).
- Commit messages use a scope, never `fix: #N` (that form auto-closes issues). Stage files explicitly; never `git add -A ui`, because Don edits `ui/package.json` locally.
- UI checks run from `ui/`: `npm run check`, `npm run lint`, `npx vitest run <file>`, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json`. Python checks use `venv/bin/python -m pytest <file> -n0`.
- No page file may cross the 600-line ceiling (`ui/scripts/arch-metrics.mjs`), and no metric may rise. Logic added to a page moves into the module this plan names.
- Copy fixed by this plan (use it exactly):
  - exists: `"<label> already exists. Replace it?"`, with buttons **Replace** (danger) and **Cancel**
  - stale_write: `"<label> changed since you opened it, in another tab or by an agent. Replace that version with yours?"`, with buttons **Replace** (danger) and **Cancel**
  - leave: `"Leave without saving your changes to <label>?"`, with buttons **Leave** (danger) and **Keep editing**
  - failed load: `"Couldn't open <label>: <describeError title>"`, with buttons **Retry** and, for a 404, **Start a new one named <label>**

## Review focus

The failure modes most likely to bite, with the task whose tests pin each one:

1. **Typing, then saving within the debounce window.** Cmd+S 50 ms after the last keystroke must save the last keystroke. Pinned in Task 6 (`flushes a pending JSON edit before saving`) and in Task 10's journey.
2. **Back from a page with unsaved edits, then cancel.** The URL, the route and the page must all stay where they were, with no double history entry. Pinned in Task 4 (`restores the hash when a guard declines`).
3. **A save conflict, then Cancel.** The edits stay, the page stays dirty, and no "Saved" toast appears. Pinned in Task 5 (`cancelling a conflict leaves the document dirty`).
4. **Opening a read-only example and saving.** It writes a shadow copy into the workspace with create-only. If that copy already exists, the user is asked. Pinned in Task 2 (`If-None-Match ignores a read-only source of the same name`) and Task 5 (`a document from a read-only origin saves create-only`).
5. **Uploading a batch where some names already exist.** One dialog for the batch, and no file silently replaced. Pinned in Task 8 (`asks once for a batch with existing names`).

---

### Task 1: The confirm dialog queues, names itself and offers more than two choices

Covers XC-15, and the dialog parts of P5 that later tasks need.

**Files:**
- Modify: `ui/src/lib/confirm.svelte.ts`
- Modify: `ui/src/lib/ui/ConfirmDialog.svelte`
- Test: `ui/src/lib/ui/ConfirmDialog.test.ts`

**Interfaces:**
- Produces:
  - `type Choice<T extends string> = { label: string; value: T; danger?: boolean }`
  - `chooseDialog<T extends string>(message: string, choices: Choice<T>[], options?: { title?: string }): Promise<T | null>`. Resolves to `null` on Escape or an outside click.
  - `confirmDialog(message, { confirmLabel?, cancelLabel?, alert?, danger?, title? }): Promise<boolean>`, now a wrapper over `chooseDialog`.
  - A call made while a dialog is open waits its turn (FIFO) instead of replacing the open one.
  - The dialog's accessible name is `title`, or else the message's first sentence.

- [ ] **Step 1: Write the failing tests** in `ConfirmDialog.test.ts`:
  - `queues a second dialog until the first is answered`: two `confirmDialog` calls. Answer the first with the confirm button; the first promise is `true`. The second message is then shown, and Escape gives `false`.
  - `resolves a choice by value`: `chooseDialog('x', [{label:'Replace',value:'replace',danger:true},{label:'Cancel',value:'cancel'}])` resolves to `'replace'` on click.
  - `marks a danger choice`: the Replace button has class `danger`.
  - `is named by its title`: `getByRole('alertdialog', { name: 'Delete e2e-one.png?' })`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/ConfirmDialog.test.ts`. Expected: FAIL (`chooseDialog` is not exported).
- [ ] **Step 3: Implement.** `confirmState` holds `{ open, message, title, choices, alert }` and a queue of pending `{ args, resolve }`. `resolveChoice(value: string | null)` resolves the head and opens the next. The component renders the choices in order; the first non-danger choice whose value is `'cancel'` or `null` is the `AlertDialog.Cancel`. The danger class reuses the existing `.danger` button style in `app.css` (add it if absent, using `--bad`).
- [ ] **Step 4: Run** the same command. Expected: PASS. Then run `npx vitest run` for the whole UI suite: every existing `confirmDialog` caller still passes.
- [ ] **Step 5: Commit** `ui(confirm): queue dialogs, named titles, danger and multi-choice`.

### Task 2: The server: ETags and conditional writes for workflows and prompts

Covers WF-3, GA-2, XC-9 (server side) and Review focus 4.

**Files:**
- Create: `dw/server/preconditions.py`
- Modify: `dw/server/routes/library.py` (`get_workflow`, `save_workflow`, `patch_workflow`, the prompt `GET` at ~l.860, `save_prompt`)
- Test: `tests/test_server_preconditions.py`
- Docs: `docs/SERVER.md` (the workflow and prompt routes), and a `docs/ARCHITECTURE.md` row *Conditional writes*

**Interfaces:**
- Produces (Python):
  - `content_etag(path: str) -> Optional[str]`: `None` when the file does not exist.
  - `check_write_preconditions(request: Request, path: str, label: str) -> None` raises `HTTPException(409, detail={"code": "exists", "message": f"{label} already exists"})` when `If-None-Match: *` is set and `path` exists. It raises `HTTPException(409, detail={"code": "stale_write", "message": f"{label} changed since it was read", "etag": <current or None>})` when `If-Match` is set and differs from `content_etag(path)`.
- Produces (HTTP):
  - `GET /api/workflows/{name}` and `GET /api/prompts/{name}` send `ETag` for the file they read.
  - `PUT` and `PATCH` responses send `ETag` for the file they wrote.
  - Preconditions are checked against the **writable destination** path (the one `resolve_writable_workflow` / `resolve_prompt_name(..., allow_create=True)` returns), inside the write lock. Prompts gain a module-level `_PROMPT_WRITE_LOCK`.

- [ ] **Step 1: Write the failing tests** in `tests/test_server_preconditions.py`, using the server test client fixture from `tests/test_server.py`:
  - `test_get_workflow_sends_an_etag_that_matches_its_bytes`
  - `test_put_without_preconditions_still_overwrites` (unchanged behaviour)
  - `test_if_none_match_star_refuses_an_existing_workflow`: 409, `detail.code == "exists"`, and the file is unchanged on disk.
  - `test_if_none_match_ignores_a_read_only_source_of_the_same_name`: an examples-dir workflow `x` plus `PUT x` with `If-None-Match: *` gives 200 and writes the shadow copy.
  - `test_if_match_refuses_after_another_write`: GET (etag A), an unconditional PUT (etag B), then a PUT with `If-Match: A` gives 409 `stale_write` with `detail.etag == B`.
  - `test_if_match_on_a_deleted_file_is_stale_write_with_no_etag`
  - `test_put_and_patch_send_the_written_etag`
  - Then the same four precondition cases for prompts: `test_prompt_*`.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_server_preconditions.py -n0 -v`. Expected: FAIL (no ETag header).
- [ ] **Step 3: Implement** `preconditions.py`. It reads the file with `open(path, "rb")` on the path the caller already validated, so add no new path resolution there. Wire it into the five handlers. `save_workflow` and `save_prompt` take `response: Response` to set the header. `get_workflow` adds `ETag` to its existing `JSONResponse` headers.
- [ ] **Step 4: Run** the new tests, then `venv/bin/python -m pytest tests/test_server.py tests/test_api_contract.py tests/test_mcp_authoring.py tests/test_mcp_prompts.py tests/test_changes.py`. Expected: all PASS.
- [ ] **Step 5: Docs.** In `docs/SERVER.md`, add two sentences per route on `ETag`, `If-Match` and `If-None-Match`. Add an ARCHITECTURE row: concept *Conditional writes*; owner `dw/server/preconditions.py`; rule "a write with no precondition header behaves as before; `exists` / `stale_write` are the only codes"; enforced by `tests/test_server_preconditions.py`.
- [ ] **Step 6: Commit** `server(library): ETag and If-Match/If-None-Match on workflow and prompt writes`.

### Task 3: The server: create-only uploads

Covers GA-3 (server side).

**Files:**
- Modify: `dw/server/routes/assets.py` (`upload_media`, after `dest` is validated, ~l.240)
- Test: `tests/test_server_preconditions.py`

**Interfaces:**
- Consumes: `check_write_preconditions` from Task 2.
- Produces: `POST /api/uploads?asset_name=…` with `If-None-Match: *` gives 409 `exists` (label = the `asset:` reference) when `dest` exists. A request without the header is unchanged (MCP `upload_asset` is unaffected). A request with no `asset_name` keeps its random name, so the header has no effect there.

- [ ] **Step 1: Write the failing tests:**
  - `test_upload_if_none_match_refuses_an_existing_asset_name`: the original bytes are kept.
  - `test_upload_without_the_header_still_replaces`
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_server_preconditions.py -n0 -k upload`. Expected: FAIL.
- [ ] **Step 3: Implement.** Call `check_write_preconditions(request, dest, make_ref(ASSET, f"{UPLOADS_SUBDIR}/{name}"))` before the write.
- [ ] **Step 4: Run** the new tests and `venv/bin/python -m pytest tests/test_mcp_assets.py tests/test_server.py -k upload`. Expected: PASS.
- [ ] **Step 5: Commit** `server(assets): create-only uploads with If-None-Match`.

### Task 4: The router: leave guards

Covers XC-2, WF-4 and Review focus 2.

**Files:**
- Modify: `ui/src/lib/router.svelte.ts`
- Test: `ui/src/lib/router.svelte.test.ts`

**Interfaces:**
- Produces: `addLeaveGuard(guard: () => boolean | Promise<boolean>): () => void`. The guard returns `true` to allow leaving. The router asks the guards in registration order before applying any `hashchange` whose **path** differs from the current one (a query-only change made by `setHashQuery` is never guarded). While the guards are asked, the hash is put back with `history.replaceState`, so the address bar and route don't move. If every guard allows, the router navigates to the target once (no extra history entry beyond the user's own) and runs `parse()`, which is when `applyRouteWorkspace` happens. If any guard declines, the route stays as it was.

- [ ] **Step 1: Write the failing tests:**
  - `asks a guard before a path change and goes on when it allows`
  - `restores the hash when a guard declines`: `location.hash` and `route.parts` are unchanged, and `applyRouteWorkspace` was not called for the target.
  - `does not guard a query-only change`
  - `a removed guard is not asked`
  - `awaits an async guard`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/router.svelte.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Keep `let committed = location.hash`. Set a `navigatingPastGuards` flag so the router's own re-navigation is not guarded again.
- [ ] **Step 4: Run** the router tests and `npx vitest run src/lib/routes.test.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(router): leave guards asked before any route change`.

### Task 5: `DocumentEditor`: flush, load state, ETag; and `saveDocument`

Covers WF-1, WF-2, GA-1, XC-1, XC-9 and Review focus 3 and 4.

**Files:**
- Modify: `ui/src/lib/editorShell.svelte.ts`
- Create: `ui/src/lib/documentSave.ts`
- Modify: `ui/src/lib/api.ts` (`getWorkflow`, `getPrompt`, `saveWorkflow`, `savePrompt`, `uploadMedia`)
- Test: `ui/src/lib/editorShell.test.ts`, `ui/src/lib/documentSave.test.ts`, `ui/src/lib/api.test.ts`

**Interfaces:**
- Consumes: `chooseDialog`, `confirmDialog` (Task 1); `addLeaveGuard` (Task 4); the HTTP contract (Tasks 2–3).
- Produces in `api.ts`:
  - `type WritePreconditions = { ifMatch?: string; createOnly?: boolean }`
  - `getWorkflow(name)` gains `etag: string | null` on its result.
  - `getPrompt(name)` gains `etag: string | null` on `StoredPrompt`.
  - `saveWorkflow(name, workflow, pre?: WritePreconditions): Promise<{ saved: WorkflowSaved; etag: string | null }>`
  - `savePrompt(name, prompt, pre?): Promise<{ saved: PromptSaved; etag: string | null }>`
  - `uploadMedia(file, assetName?, shared?, pre?: WritePreconditions)`
  - Headers: `If-Match: <etag>`, `If-None-Match: *`.
- Produces on `DocumentEditor`:
  - `loadState: 'new' | 'loading' | 'loaded' | 'failed'`, `loadError: unknown`
  - `etag: string | null`, `loadedPath: string | null`
  - `beginLoad()`, `load(doc, { etag?, path?, writable? } = {})`, `failLoad(error: unknown)`, `startNew(doc)`
  - `readonly canWrite: boolean` (`loadState` is `'new'` or `'loaded'`)
  - `registerFlush(fn: () => void): () => void`, `flush(): void`
  - `preconditions(path: string): WritePreconditions`: `{ ifMatch: etag }` when `path === loadedPath` and the origin is writable, else `{ createOnly: true }`.
  - The constructor option `label: () => string` (the leave dialog's `<label>`). The constructor registers `addLeaveGuard` with the leave copy from Global constraints, and removes it when the component is destroyed (`$effect` teardown).
  - `setView(next)` calls `flush()` first. While `jsonParseFailed` it refuses to leave the JSON view and shows `notify.error('Fix the JSON before switching view', 'json-parse')`.
  - `applyJson(raw, { quiet = false } = {})`: a quiet parse failure sets `jsonParseFailed` without a toast.
- Produces in `documentSave.ts`:
  - `type SaveOutcome = 'saved' | 'cancelled'`
  - `saveDocument(ed: DocumentEditor, opts: { path: string; label: string; write: (pre: WritePreconditions) => Promise<{ etag: string | null }> }): Promise<SaveOutcome>`. It:
    1. calls `ed.flush()`
    2. throws `Error('not loaded')` unless `ed.canWrite`
    3. writes with `ed.preconditions(path)`
    4. on 409 `exists` or `stale_write`, asks with the copy from Global constraints, and on **Replace** writes again with `{}`
    5. on success calls `ed.markSaved()` and sets `ed.etag` and `ed.loadedPath = path`
    6. rethrows any other error
  - `freeName(name: string, taken: ReadonlySet<string>): string`: `"photo.jpg"` with `photo.jpg` taken gives `"photo-2.jpg"`, then `"photo-3.jpg"`, and so on.

- [ ] **Step 1: Write the failing tests.** In `editorShell.test.ts`:
  - `flush runs every registered flusher and a removed one is not run`
  - `a failed load cannot write`
  - `preconditions: if-match for the loaded writable path, create-only for a new path`
  - `a document from a read-only origin saves create-only`
  - `setView flushes, and refuses to leave JSON while it does not parse`
  - `a quiet parse failure does not toast`
  - `asks before leaving while dirty`: the guard returns the dialog's answer, and returns `true` immediately when not dirty.

  In `documentSave.test.ts`, with `api` and `chooseDialog` mocked:
  - `saves with the editor's preconditions and records the etag`
  - `replaces on an exists conflict when the user chooses Replace`
  - `cancelling a conflict leaves the document dirty`: no `markSaved`, and the outcome is `'cancelled'`.
  - `asks with the stale_write copy`
  - `rethrows a non-conflict error`
  - `refuses to save a document whose load failed`
  - `freeName counts past taken names`

  In `api.test.ts`:
  - `saveWorkflow sends If-Match and reads the ETag`
  - `createOnly sends If-None-Match: *`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editorShell.test.ts src/lib/documentSave.test.ts src/lib/api.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `saveWorkflow` and `savePrompt` switch from `request` to `fetchJson` so they can read `ETag`. `jsonInit` gains an optional `headers` parameter. Keep `editorShell.svelte.ts` under 600 lines.
- [ ] **Step 4: Run** the three files, then the whole `npx vitest run`, `npm run check` and the metrics check. Expected: PASS, and no metric rises.
- [ ] **Step 5: Commit** `ui(editor): flush, load state, etag and saveDocument with conflict dialogs`.

### Task 6: The JSON editor applies as you type

Covers WF-1, WF-2 and Review focus 1.

**Files:**
- Modify: `ui/src/lib/editor/JsonEditor.svelte`
- Modify: `ui/src/lib/editor/EditorBody.svelte` (or wherever `JsonEditor` is mounted with `onchange`; pass `register`)
- Test: `ui/src/lib/editor/JsonEditor.test.ts` (new; monaco is mocked as in `EditorPage.test.ts`, through `src/lib/pages/JsonEditorStub.svelte` or a fake `setupMonaco`)

**Interfaces:**
- Consumes: `DocumentEditor.registerFlush` and `applyJson(raw, { quiet })` (Task 5).
- Produces:
  - `JsonEditor` props gain `register?: (flush: () => void) => () => void`.
  - The module constant `JSON_APPLY_MS = 300`.
  - The editor calls `onchange(value, { quiet: true })` 300 ms after the last content change, and `onchange(value)` (not quiet) on blur and on flush.
  - An unchanged value is never re-applied.
  - `onchange`'s signature becomes `(value: string, options?: { quiet?: boolean }) => void`.

- [ ] **Step 1: Write the failing tests:**
  - `applies quietly 300 ms after typing stops`
  - `flushes a pending JSON edit before saving`: type, then call the registered flush at once, and `onchange` fires with the new text.
  - `does not re-apply an unchanged value`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/JsonEditor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** with `editor.onDidChangeModelContent` plus a timer. Clear the timer and unregister the flush in the existing teardown.
- [ ] **Step 4: Run** the test, then `npx vitest run src/lib/pages/EditorPage.test.ts src/lib/pages/PromptEditorPage.test.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): JSON view applies as you type and flushes on demand`.

### Task 7: Wire the workflow and prompt editors to the shared save

Covers WF-1..5, WF-16, XC-1, XC-12 (unknown editor names), GA-1, GA-2 and the prompt heading leftover.

**Files:**
- Modify: `ui/src/lib/pages/EditorPage.svelte`
- Modify: `ui/src/lib/pages/PromptEditorPage.svelte`
- Create: `ui/src/lib/ui/LoadFailed.svelte`, an inline panel. Phase 2's `LoadState` absorbs it.
- Modify: `ui/src/lib/editor/StepEditor.svelte:272` (the placeholder becomes `e.g. FluxPipeline`)
- Modify: `dw/workflow_schema.json` (`minLength: 1` on a pipeline step's `component_type` and `model_name`)
- Test: `ui/src/lib/pages/EditorPage.test.ts`, `ui/src/lib/pages/PromptEditorPage.test.ts`, `tests/test_schema*.py`, or whichever schema test file holds a step's required fields (find it with `grep -ln component_type tests/test_*schema*.py`)

**Interfaces:**
- Consumes: Tasks 4–6.
- Produces:
  - `LoadFailed` props: `{ label: string; error: unknown; onretry: () => void; oncreate?: () => void }`. It renders the failed-load copy from Global constraints. It shows **Start a new one named <label>** only when `oncreate` is set, which the editors do for a 404.
  - In both editors:
    - Save, Run and Validate (and their Cmd+S and Cmd+Enter shortcuts) are disabled while `!ed.canWrite`.
    - Every field uses `oninput`, not `onchange`.
    - Saving goes through `saveDocument`.
    - After the first save of a new document, the URL is replaced (`history.replaceState`) with the edit route for the saved path, and the heading shows the name.
    - The editor's old `confirmLeave` breadcrumb handler is removed; the router guard covers every exit.

- [ ] **Step 1: Write the failing tests.** In `EditorPage.test.ts`:
  - `a failed load shows Retry and disables Save`
  - `a 404 offers to start a new workflow with that name`
  - `Save after a load failure never calls saveWorkflow`
  - `a new workflow saved under an existing name asks before replacing`
  - `the URL moves to the saved name after the first save`
  - `Cmd+S saves text typed into the JSON view without a blur`

  Mirror the same cases in `PromptEditorPage.test.ts` (prompt text typed with no blur, overwrite asks, failed load, URL and heading after save).

  In the schema test: `test_a_pipeline_step_with_an_empty_component_type_is_invalid`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/EditorPage.test.ts src/lib/pages/PromptEditorPage.test.ts` and the schema test. Expected: FAIL.
- [ ] **Step 3: Implement.** If either page would pass 600 lines, move its load and save orchestration into `ui/src/lib/editor/workflowFile.ts` (`openWorkflow(ed, name)`, `saveWorkflowAt(ed, path)`) and `ui/src/lib/promptFile.ts` (`openPrompt`, `savePromptAt`). Each is a thin caller of `saveDocument`.
- [ ] **Step 4: Run** the UI suite, `npm run check`, `npm run lint`, the metrics check, and `venv/bin/python -m pytest tests/ -k "schema or validate" -q`. Expected: PASS. If a catalog workflow under `workflows/` or `dw/workflows/` now fails `python -m dw.validate`, stop and report it; don't loosen the rule.
- [ ] **Step 5: Commit** `ui(editors): shared save with conflicts, failed-load guard, URL after first save`.

### Task 8: Uploads keep their names and never replace silently

Covers GA-3, GA-4, GA-17 and Review focus 5.

**Files:**
- Create: `ui/src/lib/assets/uploads.ts`
- Modify: `ui/src/lib/pages/AssetsPage.svelte` (`askNames` / `sendAll`, ~l.266-310, and the drop target)
- Modify: `ui/src/lib/editor/MediaArgumentInput.svelte:86`
- Modify: `ui/src/lib/assetlib.svelte.ts` (export `UPLOAD_EXTENSIONS`)
- Modify: `tests/test_ui_twins.py` (`UPLOAD_EXTENSIONS` equals `dw/server/routes/assets.py`'s `ALLOWED_UPLOAD_EXTENSIONS`)
- Test: `ui/src/lib/assets/uploads.test.ts`, `ui/src/lib/pages/AssetsPage.test.ts`

**Interfaces:**
- Consumes: `uploadMedia(..., pre)` and `freeName` (Task 5), and `chooseDialog` (Task 1).
- Produces:
  - `UPLOAD_EXTENSIONS: readonly string[]` (lower-case, with the dot), and `isUploadable(file: File): boolean`.
  - `type ConflictChoice = 'replace' | 'keep-both' | 'skip'`
  - `uploadAll(files: File[], opts: { shared: boolean; folder?: string; taken: ReadonlySet<string> }): Promise<{ uploaded: Uploaded[]; skipped: string[]; failed: { name: string; error: unknown }[] }>`. It:
    - sends each file under its own name (`folder/` + `file.name`) with `{ createOnly: true }`
    - collects the 409 `exists` names and asks **once** with `chooseDialog`: `"<n> of <m> files already exist in the library."`, offering **Replace them** (danger), **Keep both** and **Skip them**
    - then replaces (no precondition), renames with `freeName`, or skips
  - Files failing `isUploadable` are never sent. The toast `"Skipped <names>: not an image, video, audio or LUT file"` names them.
  - A single upload uses the same function, so the same dialog with `n = m = 1`.

- [ ] **Step 1: Write the failing tests.** In `uploads.test.ts`:
  - `sends each file under its own name, create-only`
  - `asks once for a batch with existing names`
  - `keep both renames with freeName`
  - `skip leaves existing assets untouched`
  - `unsupported files are not sent`

  In `AssetsPage.test.ts`: `a dropped batch is uploaded under the files' names`.

  In `tests/test_ui_twins.py`: `test_the_ui_upload_extensions_match_the_server`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/assets/uploads.test.ts src/lib/pages/AssetsPage.test.ts` and `venv/bin/python -m pytest tests/test_ui_twins.py -n0 -k upload`. Expected: FAIL.
- [ ] **Step 3: Implement.** `AssetsPage`'s `askNames` single-file prompt is removed: the file's own name is the default. Renaming an asset is a later phase's control, not an upload step. `taken` comes from the current asset listing.
- [ ] **Step 4: Run** the tests above, plus `npx vitest run src/lib/editor/MediaArgumentInput.test.ts src/lib/assets`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(assets): uploads keep their names, ask before replacing, skip unsupported files`.

### Task 9: The gallery: protect starred files, and make Save-as-asset conflicts sane

Covers GA-11 (the delete half) and GA-13.

**Files:**
- Modify: `ui/src/lib/pages/GalleryPage.svelte` (`removePicked`, ~l.148)
- Modify: `ui/src/lib/gallery/GalleryDetail.svelte` (`keepAsAsset`, ~l.98)
- Test: `ui/src/lib/pages/GalleryPage.test.ts`, `ui/src/lib/gallery/GalleryDetail.test.ts`

**Interfaces:**
- Consumes: `chooseDialog` (Task 1).
- Produces:
  - When a selection to delete includes starred files, the delete asks: `"Delete <n> files? <s> of them are starred. This removes them on disk."`, offering **Delete all** (danger), **Delete the <n−s> unstarred** and **Cancel**. With no starred files it is a single danger confirm, `"Delete <n> files? This removes them on disk."`.
  - Keep-as-asset reads `ApiError.status === 409` (not the message text). A cancelled Replace does nothing, with no toast. The Replace prompt reads `"<asset:ref> already exists. Replace it?"`.

- [ ] **Step 1: Write the failing tests:**
  - `deleting a selection with starred files says how many and can spare them`
  - `cancelling Replace on keep-as-asset shows no error`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/GalleryPage.test.ts src/lib/gallery/GalleryDetail.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `GalleryPage.svelte` must stay at 600 lines or fewer. If it would grow, move `removePicked`'s wording and partition into `ui/src/lib/gallery/deletion.ts` (`deletionChoices(files, names)`).
- [ ] **Step 4: Run** the tests and the metrics check. Expected: PASS.
- [ ] **Step 5: Commit** `ui(gallery): spare starred files on delete; quiet cancel on asset replace`.

### Task 10: The journey gate, and the rules for the next agent

Covers all of the above end to end, and P1 in `ui/CLAUDE.md`.

**Files:**
- Create: `ui/e2e/journeys/editor-safety.spec.ts`
- Modify: `ui/CLAUDE.md` (*Rules for new UI*)
- Modify: `docs/ARCHITECTURE.md` (update the row *The editors' document*; add a row *Saving over something*)

**Interfaces:**
- Consumes: everything above, and the fixture server (`ui/e2e/serve_fixture.py`, port 8971 via `playwright.config.ts`).

- [ ] **Step 1: Write the journey spec.** Each case gets its own `test`, with the reload and disk check through the API:
  1. JSON view: type a new `id` and press Cmd+S at once. Reload, and the new id is there.
  2. JSON view: type and click **form**. The form shows the edit. With broken JSON, the view stays on JSON and shows the parse error.
  3. Edit, click sidebar **Jobs**, choose **Keep editing**: still in the editor, still dirty. Click **Jobs** again and choose **Leave**: on Jobs. Press Back: the editor shows the saved version.
  4. With `page.route('**/api/workflows/e2e-card', 500)`, open the editor: the **Retry** panel shows and Save is disabled. Remove the route and press Retry: the workflow loads.
  5. A new workflow named `e2e-card`, saved: the **Replace** dialog appears, and Cancel leaves the file intact.
  6. Two pages on one workflow. A saves, then B saves: B gets the stale_write dialog.
  7. Prompt editor: type text and press Cmd+S immediately. Reload, and the text is there.
  8. Assets: upload two files where one has an existing name. One dialog appears, and **Keep both** gives `name-2.ext`.
- [ ] **Step 2: Run** `cd ui && npx playwright test e2e/journeys/editor-safety.spec.ts`. Expected: PASS. Then run the whole `npx playwright test`: PASS.
- [ ] **Step 3: Write the rules.** Add to `ui/CLAUDE.md`:
  - "A document editor is a `DocumentEditor`; every input registers a flush or writes on `oninput`; saves go through `saveDocument`; nothing else writes a workflow, prompt or upload."
  - "A route change while dirty is the router's leave guard, never a per-link handler."
  - "A dialog is `chooseDialog`/`confirmDialog`; a destructive choice is `danger: true`."

  Update the ARCHITECTURE rows to name `documentSave.ts`, `preconditions.py` and the journey spec as enforcement.
- [ ] **Step 4: Run** `cd ui && npm run preflight`. Expected: every step passes.
- [ ] **Step 5: Commit** `ui(e2e): editor-safety journey; rules for document editors and dialogs`.

---

## Phase exit

- Every finding listed for phase 1 in the spec's phase table is fixed and pinned by a test named above.
- `npm run preflight` and `venv/bin/python -m pytest tests/ -q` pass.
- The arch-metrics baseline is not raised.
- Phase 2's plan (connection, loading, errors) is written next, against the code as it stands after this phase.
