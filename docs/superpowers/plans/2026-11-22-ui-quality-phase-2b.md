# UI quality phase 2b: change detection. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** what anyone changes (an agent, another tab, `dw.run`), every open view shows on its own. Nobody has to remember to wire it, because a new route or a new view that leaves it out fails a test (P11).

**Architecture:** On the server, a write route declares what it changes where it is defined: `@writes(WORKFLOWS)` or `@NO_CHANGE(reason)`. A route class applies the bump after a 2xx, so no handler bumps by hand, and a completeness test walks `app.routes`. Two new kinds, `models` and `loras`, cover the server-wide libraries. On the client, 2a's `resource()` becomes the only way to subscribe (ESLint). A test reads every page's freshness declaration as the page table. A quiet reload marks the rows it changed. An open document whose file changes underneath it says so (`DocumentEditor.watchRemote`).

**Tech stack:** FastAPI (`dw/server/changes.py`, the eight routers, `dw/hub_cache.py`, `dw/server/updater.py`), Svelte 5 runes, vitest, Playwright, ESLint flat config.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`: principle P11, the building-blocks rows *Change kinds declared at the route*, *Watched reads* and *Open documents that change underneath you*, and the three *Change detection* bullets of *Mechanical enforcement*. The findings are RQ-8 (the changed-row mark) and P11 (`docs/superpowers/specs/2026-11-22-ui-audit.md`, the Requests line "Agent edits arrive with no visible cue, and renames aren't announced").

**Order.** This plan follows `docs/superpowers/plans/2026-11-22-ui-quality-phase-2a.md`. Tasks 1-2 are server-only and depend on nothing in 2a; they may run in parallel with 2a on their own branch. Tasks 3-7 need 2a's `resource`, `LoadState` and `connection`. Before Task 5, re-measure `unwatched_reads`: the site list there is a forecast of what 2a leaves.

## Global constraints

- MCP behaviour is unchanged. Bumps are counters only. The one response-shape change is two new integer fields on `ChangeCounts` (`GET /api/workspaces/{name}/changes`), which `dw_mcp/` doesn't read. It goes through `dw/server/api_models_library.py`, then `venv/bin/python scripts/dump_openapi.py` (under the FastAPI/Pydantic that `constraints-openapi.txt` pins) and `cd ui && npm run gen:api`. Never hand-edit `ui/src/lib/generated/`. The suite runs with `DW_STRICT_RESPONSES=1` (`tests/conftest.py`), so a snapshot key with no model field fails at once.
- The bump comes after the response is built and only for 200-299. A raised exception or a 4xx/5xx bumps nothing, as today. A shared library (prompts, the shared asset library, and now models and LoRAs) moves every workspace (`bump_all`).
- Security rules in `CLAUDE.md`: no new filesystem access and no new path resolution. `Change.in_library` decides from `LibraryRoot.origin`, as `changed_assets` does today, never from a path.
- **No new Python module.** `writes`, `NO_CHANGE`, `Change` and `ChangeRoute` live in `dw/server/changes.py` (124 lines now; it must stay under the 1,000-line warning). `modules` stays at 281, and `claude_md_lines` at 129. `dw/server/routes/jobs.py` is 1,044 lines and `api_models.py` 1,105 (already over 1,100): neither may grow. Each decorator line added there is paid for by a hand bump removed.
- Commit messages use a scope, never `fix: #N`. Stage files explicitly; never `git add -A ui`.
- **The UI metrics gate** is 2a's: compare against `$SCRATCH/ui-start.json` (re-measured at this plan's start if run on a fresh branch). `unwatched_reads` must reach 0 in `docs/stabilization/ui/baseline.json` by the end.
- No page file over 600 lines, and no function over complexity 10 or 60 lines.
- e2e runs on port 8971, and journeys live in `ui/e2e/journeys/`. A "within two feed polls" assertion uses `timeout: 15_000` (`CHANGES_POLL_MS` is 5 s).
- The known flaky test `tests/test_ledger_job_link.py::TestDelete::test_a_request_made_again_under_the_id_takes_the_jobs_back` is not this phase's.
- Copy fixed by the spec (use it exactly):
  - an unedited document reloaded: the toast `"Updated by an agent or another tab"`
  - an edited one: the banner `"<label> changed since you opened it, in another tab or by an agent"`, with **Load theirs** and **Keep mine**. After **Keep mine**, the next save gets phase 1's `stale_write` dialog.

## Review focus

1. **A feed reload while the user is in the middle of something.** Typing in a list's filter, with a selection, scrolled down: the filter text, the focus, the selection and the scroll all stay, and only the changed rows are marked. Pinned in Task 3 (`a quiet reload keeps focus, filter and selection`) and Task 7's journey.
2. **An agent saves the workflow open with unsaved edits.** A banner, no reload. **Keep mine** and then Save gives the `stale_write` dialog, not a silent overwrite. Pinned in Task 6 (`an edited document gets the banner, and Keep mine leaves the conflict to save`).
3. **A 2xx that changed nothing.** A workspace copy with neither runs nor assets bumps nothing. A note set on a target counts once, under the target's kind only. Pinned in Task 1 (`test_a_copy_of_nothing_moves_nothing`, the `put_note` cases).
4. **A route added next month with no declaration.** The completeness test names it; a declared route with no case names it too. Pinned in Task 1 (`test_every_write_route_declares_what_it_changes`, `test_every_declaring_route_has_a_case`).
5. **A download that finishes while the Models page is open and idle.** The new model appears with no timer running, from the download manager's own bump. Pinned in Task 2 (`test_a_finished_download_moves_models`) and Task 5 (`ModelsPage reloads its cache list on a models move`).

---

### Task 1: The server declares what each write route changes

The spec's *Change kinds declared at the route*, and the server-side completeness test.

**Files:**
- Modify: `dw/server/changes.py`
- Modify: `dw/server/deps.py` (`selected_workspace` records `request.state.dw_workspace = ws.name`)
- Modify: every router's `APIRouter()`, which becomes `APIRouter(route_class=ChangeRoute)` (`dw/server/routes/{assets,files,gallery,jobs,library,loras,media,requests,system}.py`), and every POST, PUT, PATCH and DELETE handler (40 of them; the list is below)
- Test: `tests/test_changes.py`
- Docs: `docs/ARCHITECTURE.md` (the row *Workspace change counters*), and the docstring of `dw/server/changes.py`

**Interfaces:**
- Produces (Python, in `changes.py`):
  - `WORKSPACES = "workspaces"`: a declarable name for the server-wide counter. It is not in `KINDS`.
  - `Declared(NamedTuple)`: `kinds: tuple[str, ...]`, `shared: bool`, `by: str | None`, `reason: str | None`.
  - `writes(*kinds, shared=False, by=None)` is a decorator that sets `fn.writes = Declared(...)`. It raises `ValueError` for no kinds or an unknown one. `by` names who bumps instead of the route ("the job manager"): the route is declared, but `ChangeRoute` doesn't bump.
  - `NO_CHANGE(reason: str)` is a decorator that sets `Declared((), False, None, reason)`. An empty reason raises `ValueError`.
  - `class Change`, the request's pending bump, all methods chainable:
    - `only(*kinds)` narrows to a subset of what was declared. Outside it, it raises `ValueError`.
    - `in_workspace(name)`
    - `everywhere()` means `bump_all`.
    - `in_library(ws, root)` is `everywhere()` when `root.origin == COMMON_ORIGIN`, else `in_workspace(ws.name)`.
    - `nothing()`
  - `pending_change(request: Request) -> Change`, a FastAPI dependency (`change: Change = Depends(pending_change)`).
  - `class ChangeRoute(APIRoute)`: its `get_route_handler()` wraps the handler. It puts `Change(endpoint.writes)` on `request.state.dw_change` first. After `response = await handler(request)` with `200 <= response.status_code < 300`, unless `by` or `nothing()` applies, it bumps:
    - `WORKSPACES` with `bump_workspaces()`
    - a `shared` declaration, or `everywhere()`, with `bump_all(kinds)`
    - otherwise `bump(target, kinds)`, where the target is `in_workspace`'s name, else `request.state.dw_workspace`, else it raises `RuntimeError("<route name> declares <kinds> but named no workspace")`
  - Removed: `changed`, `changed_everywhere`, `changed_assets`. `Changes.forget` stays, and the workspace create and delete handlers still call it (an epoch reset, not a bump).
- The declarations. A route-name column of the case table follows each:

  | Route (handler) | Declaration | Handler narrows |
  |---|---|---|
  | `upload_media`, `keep_asset` | `writes(ASSETS)` | `everywhere()` when shared |
  | `delete_asset` | `writes(ASSETS)` | `in_library(ws, root)` |
  | `archive_assets`, `archive_outputs`, `gallery_compare` | `NO_CHANGE("builds a download or a view; writes nothing a page lists")` | |
  | `submit_job`, `rerun_job`, `rerun_restart_batch`, `move_job`, `cancel_job`, `enhance` | `writes(JOBS, by="the job manager")` | |
  | `delete_job_run` | `writes(GALLERY, JOBS)` | `in_workspace(<the job's own workspace>)`, as it computes today |
  | `export_job` | `NO_CHANGE("an export lands in exports/, which no page lists")` | |
  | `validate_workflow` | `NO_CHANGE("validation reads only")` | |
  | `delete_output`, `put_star` | `writes(GALLERY)` | |
  | `put_note` | `writes(ASSETS, GALLERY, WORKSPACES)` | `only(WORKSPACES)` for a workspace target, `only(ASSETS).in_library(...)` for an asset, else `only(GALLERY)` |
  | the five request routes | `writes(REQUESTS)` | `in_workspace(name)` |
  | `create_workspace`, `delete_workspace` | `writes(WORKSPACES)` | |
  | `copy_workspace` | `writes(GALLERY, ASSETS)` | `in_workspace(target.name).only(...)`, or `nothing()` when neither was copied |
  | `save_workflow`, `patch_workflow`, `delete_workflow` | `writes(WORKFLOWS)` | |
  | `save_prompt`, `delete_prompt` | `writes(PROMPTS, shared=True)` | |
  | the seven `system.py` writes and both `loras.py` writes | `NO_CHANGE("until the models and loras kinds (2b Task 2)")`, except `clear_memory`: `NO_CHANGE("memory is live state on the status poll")` | |

  Use the handler names exactly as they appear in the routers. Where one differs from the name in the table, the router's name wins, and the case key follows it.

- [ ] **Step 1: Write the failing tests** in `tests/test_changes.py`:
  - `test_every_write_route_declares_what_it_changes`: over `app.routes`, every `APIRoute` with a POST, PUT, PATCH or DELETE method has an `endpoint.writes`. The failure lists the names of the routes without one.
  - `test_every_write_route_is_a_change_route`
  - `test_every_declaring_route_has_a_case`: re-key `CASES` by route name (the handler). The routes with kinds and no `by` must equal `set(CASES) | set(UNCASED)`. `UNCASED` maps a route name to its reason, and starts empty.
  - `test_no_route_bumps_by_hand`: the source of `dw/server/routes/*.py` contains none of `changed(`, `changed_everywhere(`, `changed_assets(`, `.bump(`, `.bump_all(` or `.bump_workspaces(`.
  - `test_a_copy_of_nothing_moves_nothing`
  - `test_a_handler_that_raises_bumps_nothing`: a throwaway app with one `ChangeRoute` whose handler raises an `HTTPException(409)` after work.
  - `test_change_only_refuses_an_undeclared_kind`
  - `test_change_in_library_decides_from_the_roots_origin`: it replaces `test_changed_assets_decides_from_the_roots_origin`.
  - The existing `test_each_write_moves_exactly_its_kind` keeps every case, keyed by route name now.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_changes.py -n0 -v`. Expected: FAIL (no route declares anything).
- [ ] **Step 3: Implement.** Decorator order: `@router.put(...)` is outermost, then `@writes(...)`, so the attribute is on the function when it is registered (the same pattern as `query_token_ok`). Each `changed(...)` call is removed; where the bump target was conditional, the handler takes `change: Change = Depends(pending_change)` and narrows it.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_changes.py tests/test_server.py tests/test_api_contract.py tests/test_mcp_*.py tests/test_workspace_copy.py tests/test_ledger_job_link.py -q` and `venv/bin/python scripts/arch_metrics.py`. Expected: PASS, `modules` 281, and no size warning rises.
- [ ] **Step 5: Docs.** In the ARCHITECTURE row *Workspace change counters*:
  - The owner column becomes `writes`, `NO_CHANGE`, `Change`, `ChangeRoute` and `JobManager._changed`.
  - The rule gains "a write route declares its kinds or `NO_CHANGE` with a reason; the bump runs after a 2xx".
  - The enforcement column adds the three new tests.
- [ ] **Step 6: Commit** `server(changes): write routes declare their kinds; the bump runs after a 2xx`.

### Task 2: The `models` and `loras` kinds

The spec's new kinds: `MODELS` (download, delete, diffusers update) and `LORAS` (catalog save).

**Files:**
- Modify: `dw/server/changes.py` (`MODELS`, `LORAS`, `KINDS`, `SERVER_WIDE = (MODELS, LORAS)`)
- Modify: `dw/server/api_models_library.py` (`ChangeCounts.models`, `ChangeCounts.loras`)
- Modify: `dw/hub_cache.py` (`DownloadManager.changes = None`; a terminal download calls `changes.bump_all(MODELS)`), `dw/server/updater.py` (the same on a finished update), `dw/server/app.py` (wire both, as `manager.changes` is wired)
- Modify: `dw/server/routes/system.py`, `dw/server/routes/loras.py` (Task 1's placeholders become declarations)
- Modify: `ui/src/lib/generated/openapi.json`, `api-schema.ts` (regenerated only)
- Test: `tests/test_changes.py`, `tests/test_hub_cache.py` (or wherever `DownloadManager` is tested: `grep -ln DownloadManager tests/`)
- Docs: `docs/SERVER.md` (the `/changes` route's kinds), the ARCHITECTURE row

**Interfaces:**
- Produces:
  - `writes(MODELS)` on `start_download`, `cancel_download`, `delete_cached_model`, `clear_cache`, `delete_cache_entry` and `update_diffusers`. The Models page lists the hub cache and the downloads, and the Server page's `CachesPanel` the disk caches, so one kind covers what those pages show.
  - `writes(LORAS)` on `save_lora` and `delete_lora`.
  - A `SERVER_WIDE` kind is always applied with `bump_all`, so it needs no workspace.
  - The background bumps, `DownloadManager` on done, failed or cancelled and `DiffusersUpdater` on finish, never raise (the `JobManager._changed` rule).
  - The UI type `ChangeKind` gains `'models' | 'loras'` through the regenerated `ChangeCounts`, with no hand edit.

- [ ] **Step 1: Write the failing tests:**
  - New `CASES` entries `delete_cached_model` (monkeypatch `dw.server.routes.system.delete_model` to return 0), `clear_cache`, `save_lora` and `delete_lora`. Each expects `{"models": 1}` or `{"loras": 1}` in **two** workspaces.
  - `test_a_finished_download_moves_models`: a `DownloadManager(download_fn=…, info_fn=…)` fake, through `create_app(download_manager=…)`.
  - `start_download`, `cancel_download`, `delete_cache_entry` and `update_diffusers` each get a case if the fake makes them reachable. Otherwise they go in `UNCASED` with the reason ("reaches the network", "runs pip").
  - `test_the_route_answers_a_default_workspace` now expects `models` and `loras` in `kinds`.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_changes.py -n0`. Expected: FAIL.
- [ ] **Step 3: Implement**, then regenerate: `venv/bin/python scripts/dump_openapi.py && cd ui && npm run gen:api`.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_changes.py tests/test_api_contract.py tests/test_server.py -q` and `cd ui && npm run check && npx vitest run src/lib/changes.test.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `server(changes): models and loras kinds, bumped by routes and by the download and update managers`.

### Task 3: A quiet reload marks what changed

RQ-8, generalised.

**Files:**
- Modify: `ui/src/lib/resource.svelte.ts` (`rows`, `changed`, and `lazy` for Task 6)
- Modify: `ui/src/app.css` (`.row-changed`)
- Modify: `ui/src/lib/request/announce.ts` (renames)
- Modify: `ui/src/lib/pages/RequestsPage.svelte`, `ui/src/lib/request/RequestItem.svelte`, `ui/src/lib/pages/PromptsPage.svelte`, `WorkflowsPage.svelte`, `AssetsPage.svelte`, `GalleryPage.svelte`
- Test: `ui/src/lib/resource.svelte.test.ts`, `ui/src/lib/request/announce.test.ts`, `ui/src/lib/pages/PromptsPage.test.ts`, `ui/src/lib/pages/RequestsPage.test.ts`

**Interfaces:**
- Consumes: 2a's `resource`.
- Produces:
  - `ResourceOptions<T>` gains `rows?: (data: T) => Iterable<readonly [key: string, value: unknown]>` and `lazy?: boolean`. A lazy resource loads only on a watched move or an epoch rise, never at first.
  - `Resource<T>` gains `readonly changed: ReadonlySet<string>`: after a **quiet** reload, the keys whose value's `JSON.stringify` differs from the held one, or that are new. It is empty after a non-quiet load, and cleared `CHANGED_MARK_MS = 4000` after it was set. The constant is exported.
  - `.row-changed` in `app.css`: a left inset bar in `--live` that fades out over `CHANGED_MARK_MS`, with no animation under `prefers-reduced-motion`. `app.css` is already a `--live` file in `design-rules.test.ts`, so that test is unchanged.
  - The keys:
    - Requests list: the request id
    - Request page: the item key
    - Prompts and Workflows: the name
    - Assets and Gallery: the file name

    Each row or tile gets `class:row-changed={res.changed.has(key)}`.
  - `announce.ts`: a rename says `"The agent renamed <old> to <new>."`.

- [ ] **Step 1: Write the failing tests:**
  - `resource.svelte.test.ts`:
    - `a quiet reload reports the changed and new keys and clears them`
    - `a non-quiet load marks nothing`
    - `a lazy resource does not load until a watched kind moves`
  - `PromptsPage.test.ts`: `a quiet reload keeps focus, filter and selection`. Focus the filter, type `fo`, move `prompts`; the input is still focused with `fo`, and only the changed row has `row-changed`.
  - `RequestsPage.test.ts`: `a request an agent edited is marked`.
  - `announce.test.ts`: `announces a rename`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/resource.svelte.test.ts src/lib/request/announce.test.ts src/lib/pages/PromptsPage.test.ts src/lib/pages/RequestsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** The diff runs only on a quiet reload, over `rows`, and stays O(rows). `GalleryPage` must not cross 600 lines: its key function goes in `gallery/order.ts`.
- [ ] **Step 4: Run** `npx vitest run`, `npx playwright test e2e/freshness.spec.ts` and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(data): a quiet reload marks the rows it changed; renames are announced`.

### Task 4: One way to subscribe

The spec's ESLint rule, with the last two hand-wired watchers (`Sidebar`, `RequestPage`) moved onto `resource`.

**Files:**
- Modify: `ui/eslint.config.js`
- Modify: `ui/src/lib/Sidebar.svelte` (~l.57), `ui/src/lib/pages/RequestPage.svelte` (the closed-request watch and the timer, ~l.120-175), `ui/src/lib/request/planStore.svelte.ts` (`requestRoutes`)
- Test: `ui/src/lib/Sidebar.test.ts`, `ui/src/lib/pages/RequestPage.test.ts`

**Interfaces:**
- Consumes: Task 3's `resource`, and `refreshInterval` / `newerOf` (`request/refresh.ts`).
- Produces:
  - ESLint `no-restricted-syntax` with the selector `CallExpression[callee.property.name='watch'][callee.object.callee.name='changes']` and the message `"Subscribe through resource() (src/lib/resource.svelte.ts)"`. It covers `files: ['src/**/*.{ts,svelte}']` and `ignores: ['src/lib/resource.svelte.ts', 'src/lib/test/**', '**/*.test.ts']`.
  - Sidebar: `resource(() => loadWorkspaces(), { watch: ['workspaces'] })`, whose load invalidates the cached listing first.
  - RequestPage:
    - One `resource(() => store.refresh(), { watch: ['requests'], alsoLive: () => refreshInterval(store.read), merge: newerOf })`. This replaces both of the page's `$effect`s, the `changes().watch` and the `poll`.
    - `PlanStore` gains `refresh(): Promise<LedgerRequest>`, which reads through its `get`, keeps `newerOf` and calls `onread`.
    - `requestRoutes(ws: string, id: string): { get; put }` in `planStore.svelte.ts` builds the two api calls, so the page holds no `api.getRequest`.

- [ ] **Step 1: Write the failing tests:**
  - `RequestPage.test.ts`:
    - `an open request refreshes on its interval and a closed one on the feed`: the cases the old effects had, now through the one resource.
    - `a refresh answered after a save does not put the old revision back`
  - `Sidebar.test.ts`: `a workspace made elsewhere appears`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/RequestPage.test.ts src/lib/pages/RequestPage.edit.test.ts src/lib/Sidebar.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement**, then `npm run lint`. Expected: no `changes(...).watch` outside the owner.
- [ ] **Step 4: Run** `npx vitest run`, `npx playwright test e2e/requests.spec.ts e2e/workspaces.spec.ts` and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(data): resource is the one way to subscribe; sidebar and request page moved onto it`.

### Task 5: Every read declares its freshness: `unwatched_reads` reaches 0, and the page table becomes a test

The spec's UI-side change-detection enforcement.

**Files:**
- Create: `ui/src/lib/freshness.test.ts`
- Modify: `ui/src/lib/changes.svelte.ts` (the page-table comment goes; one line points to `freshness.test.ts`)
- Modify: the sites `unwatched_reads` still lists. **Re-measure first.** The forecast after 2a: `CachesPanel.svelte:25`, `editor/LorasEditor.svelte:53`, `editor/StepEditor.svelte:139`, `job/ChainSegments.svelte:68`, `job/JobResults.svelte:74,96`, `job/RerunWithChanges.svelte:82`, `pages/EditorPage.svelte:165,241`, `pages/PromptEditorPage.svelte:102,228`, `pages/SchemaPage.svelte:10`, `pages/ComparePage.svelte` (any left), and `pages/ModelsPage.svelte`'s cache list. Plus `editor/workflowFile.ts` and `promptFile.ts`, which receive the document loads.
- Test: `freshness.test.ts`, and the touched components' tests

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces. Each remaining site becomes one of three things:
  1. **A view of server data** becomes a `resource` with its kinds:
     - CachesPanel, and ModelsPage's cache list: `['models']`. ModelsPage's downloads become `{ watch: ['models'], alsoLive: … }`.
     - LorasEditor: `['loras']`.
     - JobResults' metadata: `['gallery']`.
     - The editors' workflow listing: `['workflows']`.
     - StepEditor's pipeline description and SchemaPage: `watch: []` (static).
  2. **A document an editor opens** becomes a call into `openWorkflow` / `openPrompt` in `editor/workflowFile.ts` / `promptFile.ts`. Its freshness is `watchRemote` (Task 6).
  3. **A read made by a click** (RerunWithChanges' definition, ChainSegments' segment) becomes a named function in a `.ts` module, called from the handler: `job/rerun.ts` (new) for RerunWithChanges, and the existing `lib/chainActions.ts` for ChainSegments.
  - `freshness.test.ts` (node environment):
    - It scans `src/App.svelte`, `src/lib/Sidebar.svelte`, `src/lib/pages/*.svelte` and every `.svelte` file containing `resource(`.
    - From each `resource(` call it reads the freshness literal: the `watch: [...]` kinds, `live: { everyMs: N }` and `alsoLive`.
    - It asserts the result equals `PAGE_TABLE`, a constant in the test written as the documented table (`{ 'pages/PromptsPage.svelte': [{ watch: ['prompts'] }], ... }`).
    - It fails, naming the file, when a page file imports `../api` or a `*lib.svelte` loader and has no `resource(`, unless it is in `DOCUMENT_PAGES = ['pages/EditorPage.svelte', 'pages/PromptEditorPage.svelte']`, which must call `watchRemote`.

- [ ] **Step 1: Write the failing test** `freshness.test.ts`, with `PAGE_TABLE` as the target. Then write the component tests for each converted site: `ModelsPage reloads its cache list on a models move`, `LorasEditor shows a LoRA saved elsewhere`, and `CachesPanel reloads on a models move`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/freshness.test.ts src/lib/pages/ModelsPage.test.ts src/lib/editor/LorasEditor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the conversions. `EditorPage` (593 lines) and `ModelsPage` (596 lines) must end shorter.
- [ ] **Step 4: Run** `npx vitest run`, `npm run lint` and the metrics gate. Expected: PASS, and `unwatched_reads: 0`. Set it to 0 in the baseline.
- [ ] **Step 5: Commit** `ui(data): every read declares its freshness; the page table is a test`.

### Task 6: `DocumentEditor.watchRemote`: an open document that changes underneath you

The spec's *Open documents that change underneath you*.

**Files:**
- Modify: `ui/src/lib/editorShell.svelte.ts` (`watchRemote`, `remote`, `takeTheirs`, `keepMine`)
- Create: `ui/src/lib/ui/RemoteChanged.svelte`
- Modify: `ui/src/lib/pages/EditorPage.svelte`, `ui/src/lib/pages/PromptEditorPage.svelte` (or their `workflowFile.ts` / `promptFile.ts`), `ui/src/lib/pages/WorkflowPage.svelte` (the notice)
- Test: `ui/src/lib/editorShell.test.ts`, `ui/src/lib/pages/EditorPage.test.ts`, `ui/src/lib/pages/PromptEditorPage.test.ts`, `ui/src/lib/pages/WorkflowPage.test.ts`

**Interfaces:**
- Consumes: Task 3's `lazy` resource, and phase 1's `load`, `etag` and `saveDocument`.
- Produces:
  - `watchRemote(opts: { kind: ChangeKind; read: () => Promise<{ doc: Record<string, any>; etag: string | null; writable: boolean }> }): void`. It is called during component init and creates a `lazy` resource that watches `[kind]`. It does nothing while `loadState` is not `'loaded'`, or while a save is in flight (`busy`).
  - On a move it re-reads the document (the GET, since the routes have no HEAD). If `etag` equals `ed.etag`, nothing happens.
    - Unedited (`!ed.unsaved()`): `ed.load(doc, { etag, path: ed.loadedPath, writable })` and the toast `"Updated by an agent or another tab"`.
    - Edited: `remote = 'diverged'`, with the newer read held.
  - `remote: 'same' | 'diverged'`
  - `takeTheirs()` loads the held read and sets `remote = 'same'`.
  - `keepMine()` sets `remote = 'same'` and leaves `etag` alone, so the next save sends the old `If-Match` and gets `stale_write`.
  - `RemoteChanged` props: `{ label: string; ontheirs: () => void; onmine: () => void }`, a non-modal `role="status"` banner with the spec's copy.
  - `WorkflowPage`: when its definition resource's `changed` holds the workflow's name, it shows the same `"Updated by an agent or another tab"` notice inline.

- [ ] **Step 1: Write the failing tests:**
  - `editorShell.test.ts`:
    - `an unedited document reloads on a move with a newer etag`
    - `an edited document gets the banner, and Keep mine leaves the conflict to save`
    - `the same etag does nothing`
    - `no reload while a save is in flight`
  - `EditorPage.test.ts` and `PromptEditorPage.test.ts`: `Load theirs replaces the edits`.
  - `WorkflowPage.test.ts`: `a definition changed elsewhere says so`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editorShell.test.ts src/lib/pages/EditorPage.test.ts src/lib/pages/PromptEditorPage.test.ts src/lib/pages/WorkflowPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `editorShell.svelte.ts` stays under 600 lines, and `EditorPage` doesn't grow (the wiring is one `watchRemote` call in `workflowFile.ts`'s open path).
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/freshness.test.ts` (the document pages now call `watchRemote`), `npx playwright test e2e/journeys/editor-safety.spec.ts` and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): an open document that changes underneath says so`.

### Task 7: The agent-edits journey, and the rules for the next agent

The spec's change-detection end-to-end gate, and P11 in `ui/CLAUDE.md`.

**Files:**
- Create: `ui/e2e/journeys/agent-edits.spec.ts`
- Modify: `ui/e2e/serve_fixture.py`: a scratch `HF_HUB_CACHE` seeded with one fake repo, `models--e2e--tiny` (a `snapshots/<rev>/config.json` and a `refs/main`), so a model delete runs against scratch, never the developer's cache. Set it before the server import, beside `DIFFUSERS_HELPER_ROOT`.
- Modify: `ui/CLAUDE.md` (rewrite lines; add none), `docs/ARCHITECTURE.md` (the rows *Keeping UI views fresh* and *The editors' document*), `docs/SERVER.md` (the declaration rule, for route authors)

**Interfaces:**
- Consumes: everything above, on port 8971.

- [ ] **Step 1: Write the journey.** One `test` per kind. Each opens the page, writes through `request` (the HTTP API, as an agent would), and expects the change on the page within `15_000` ms with `framenavigated` unchanged:
  1. Workflows: `PUT /api/workflows/e2e-agent`. The card appears and is marked `row-changed`.
  2. Editor open on `e2e-card`, unedited: `PATCH` its variables. The toast appears and the field shows the new value. Then edit a field and `PATCH` again: the banner appears, and **Keep mine** then Save gives the stale_write dialog.
  3. Prompts: `PUT /api/prompts/e2e-agent`.
  4. Assets: `POST /api/uploads?asset_name=e2e-agent.png`.
  5. Requests: `PUT` a request (the existing `freshness.spec.ts` case moves here, and that file is deleted).
  6. Job page of a finished fixture job: `PUT /api/notes` on its run. The *Run note* shows the text.
  7. Sidebar: `POST /api/workspaces {name: 'e2e-agent'}` appears, then `DELETE` it and it goes.
  8. Models: `DELETE /api/models?repo=e2e/tiny`, and the row goes.
  9. Editor on a workflow with a LoRA-capable step: `PUT /api/loras/e2e-agent`, and the LoRA list offers it.

  Clean up what each case made, so the other specs see the fixture as it was.
- [ ] **Step 2: Run** `cd ui && npx playwright test e2e/journeys/agent-edits.spec.ts`, then the whole `npx playwright test`. Expected: PASS.
- [ ] **Step 3: Write the rules.** `ui/CLAUDE.md` keeps its line count.
  - Bullet 2 (rewritten in 2a) gains: "`resource` is the only subscriber (ESLint), and `src/lib/freshness.test.ts` holds every page's declaration; an open document watches with `DocumentEditor.watchRemote`."
  - It is paid for by cutting bullet 2's "Every timer pauses while the tab is hidden …" sentence down to "timers are `poll`'s (`src/lib/poll.ts` says what they promise)".

  `docs/SERVER.md` gains one paragraph for route authors: "every POST, PUT, PATCH and DELETE route is `@writes(kind, …)` or `@NO_CHANGE(reason)` (`dw/server/changes.py`); `tests/test_changes.py` fails on a route with neither, and on a declaring route with no case." Update the ARCHITECTURE rows to name `freshness.test.ts`, `watchRemote` and the journey.
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, the metrics gate, `venv/bin/python -m pytest tests/ -q` and `venv/bin/python scripts/arch_metrics.py` (`modules` 281, `claude_md_lines` 129). Expected: PASS, apart from the known flaky test named in Global constraints.
- [ ] **Step 5: Commit** `ui(e2e): agent-edits journey; rules for declared writes and watched reads`.

---

## Phase exit

- P11 holds and is pinned on both sides: `test_every_write_route_declares_what_it_changes`, `test_every_declaring_route_has_a_case`, `freshness.test.ts`, the ESLint subscribe rule, and `agent-edits.spec.ts`.
- RQ-8's changed-row mark shows on Requests, the request page, Prompts, Workflows, Assets and Gallery.
- `unwatched_reads: 0`, `raw_error_toasts: 0` and `empty_outside_loadstate: 0` are in `docs/stabilization/ui/baseline.json`. No other metric is above `$SCRATCH/ui-start.json`, and there is no new Python module.
- Phase 3's plan is written next, against the code as it stands.
