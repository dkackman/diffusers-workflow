# Server pages and Preferences Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise the web UI's Server pages by question (Activity, Models, Connect, System; Schema in the top bar) and rebuild Preferences as an answer-first read view with a chain editor and a live dry run.

**Architecture:** UI-only moves for the page split (existing panels recomposed, router gains a sub-path for server views). Server gains an opt-in full resolution on `GET /api/preferences` (admission's gate check, `used_by`, `contract`) and a write-free `POST /api/preferences/{key}/check`. The Preferences UI reads the full GET at rest and posts the draft to the check route while editing.

**Tech Stack:** Svelte 5 + TypeScript (Vitest, Playwright), Python FastAPI (pytest), generated OpenAPI types.

**Spec:** `docs/superpowers/specs/2026-11-27-server-pages-preferences-design.md` (mockups in `docs/superpowers/specs/2026-11-27-mockups/`)

## Global Constraints

- Work on a branch `feat/server-pages-preferences` off `develop`; commits say `refs #1762` (never a headline that closes an issue).
- UI rules in `ui/CLAUDE.md` hold for every UI task: page data through `resource()` + `<LoadState>`; every mutating button an `action()`; destructive through `src/lib/destructive.ts`; failure copy via `describeError`/`ErrorLine`, never `e.message`; colour only through the listed tokens (`--good` ready, `--warn` downloads first / unsaved, `--bad` nothing fits or an error, `--select` the default tier); `PageHeader` for each page; copy per `ui/GLOSSARY.md` (banned terms stay at 0); tests mock `api` with `src/lib/test/apiMock.ts`; imports per `LAYER_ALLOWS`.
- Server types are generated: after any response-model change run `python scripts/dump_openapi.py` then `cd ui && npm run gen:api`; never hand-write them. `tests/test_api_contract.py` must pass.
- Every new `api` key is classified in `ui/scripts/arch-metrics.mjs` (`READ_ALIASES` or `NOT_READS`). Ratchets (`arch-metrics`, freshness, design-rules, copy-rules, e2e performance budgets) are held; raising one needs Don — stop and ask.
- Server inputs validated as the existing preference routes do (`_checked_key`, `preferences.check_tier`); never `eval`/`exec`/`shell=True`.
- Wording rules from the spec: a model after the pick is "fallback, not checked here"; never "if it fails, try the next"; an emptied tier reads "No models: requests at this tier are refused"; Save's footer says "Jobs already queued keep the model they were given."
- No "Download now" button (#1781). A missing download reads "downloads N GB on first run" (or "downloads its weights on first run" with no size) and links to `#/server/models/downloaded`.
- MCP `get_preferences` and the run form's `getPreferences({resolve: true})` answers must not change.
- Checks: Python `venv/bin/python -m pytest tests/ -n auto`; UI from `ui/`: `npm run check`, `npm run lint`, `npm test`, then `npm run build && npx playwright test`.

## Review Focus

1. **Hub unreachable on the full read** (offline box, HF down): the Preferences page still loads; tiers show their pick with no size note, never an error page. → test in Task 5.
2. **Out-of-order dry-run answers**: a slow check for an older draft arriving after a newer one must not repaint the chips. → test in Task 9.
3. **A workflow deleted from the catalog after its key was saved**: the resting view shows that tier's refusal on the card (`--bad`), not a blank row. → test in Task 8.
4. **A composition cycle or a self-reference** in the catalog while computing `used_by`: terminates, counts each workflow once. → test in Task 5.
5. **Old bookmarks** `#/server/status`, `#/server/preferences`, `#/jobs`, `#/server` (and with a `?query`) land on the new pages. → test in Task 2.

---

### Task 1: Schema link moves to the top bar

**Files:**
- Modify: `ui/src/App.svelte` (top-right nav, beside the `/docs` OpenAPI link ~line 327)
- Modify: `ui/src/lib/navItems.ts` (`SERVER_ITEMS` drops `schema`)
- Test: `ui/src/App.test.ts`, `ui/src/lib/Sidebar.test.ts`

**Interfaces:** Produces nothing later tasks use. `#/server/schema` stays routed.

- [ ] **Step 1: Write failing tests**
  - `App.test.ts` › `the top bar links the workflow schema beside OpenAPI`: a link with accessible name `workflow schema` and `href="#/server/schema"`, no `target` attribute, and it is the element immediately before the OpenAPI link (`aria-label="interactive API reference (OpenAPI)"`).
  - `Sidebar.test.ts` › `the server group has no schema row`: no link with `href="#/server/schema"` in the `navigation` aside.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/App.test.ts src/lib/Sidebar.test.ts` — expect both new tests FAIL.
- [ ] **Step 3: Implement.** An `<a class="plain helplink">` with `ListTree` (size 15), `title`/`aria-label` "workflow schema"; remove the schema entry from `SERVER_ITEMS`.
- [ ] **Step 4: Run** the same command — PASS.
- [ ] **Step 5: Commit** `feat(ui): schema link moves to the top bar (refs #1762)`

### Task 2: Server routes by question

**Files:**
- Modify: `ui/src/lib/routes.ts`
- Modify: `ui/src/lib/Breadcrumb.svelte`, `ui/src/lib/job/JobError.svelte`, `ui/src/lib/navItems.ts`
- Modify: `ui/src/App.svelte` (routing block only: temporary mapping, pages come in Tasks 3–4)
- Test: `ui/src/lib/routes.test.ts`, `ui/src/lib/Breadcrumb.test.ts`, `ui/src/lib/Sidebar.test.ts`

**Interfaces:**
- Produces:
  - `SERVER_SECTIONS = ['activity', 'models', 'connect', 'system', 'schema'] as const`
  - `SECTION_LABELS.server = { activity: 'Activity', models: 'Models', connect: 'Connect', system: 'System', schema: 'Schema' }`
  - `MODELS_TABS = ['preferences', 'downloaded'] as const`, `type ModelsTab`
  - `RouteView` server variant: `{ kind: 'server'; section: ServerSection; rest: string[] }`
  - `serverHref(section: ServerSection, ...rest: string[]): string`
  - `SERVER_ITEMS` order: Activity (`Activity` icon), Models (`Database`), Connect (`Plug`), System (`Server`).

- [ ] **Step 1: Write failing tests** in `routes.test.ts`:
  - `parseView(['server','models','downloaded'])` → `{kind:'server', section:'models', rest:['downloaded']}`
  - `parseView(['server','models'])` → `rest: []` (the page treats empty as `preferences`)
  - `parseView(['server','models','nope'])` → `kind: 'missing'`
  - `parseView(['server','activity','x'])` → `kind: 'missing'` (only models takes a sub-path)
  - `legacyRedirect(['server','status'], 'w')` → `['server','activity']`
  - `legacyRedirect(['server','preferences'], 'w')` → `['server','models','preferences']`
  - `legacyRedirect(['jobs'], 'w')` → `['server','activity']`; `legacyRedirect(['server'], 'w')` → `['server','activity']`
  - (Review Focus 5) the router keeps the query across a legacy redirect: `#/server/status?x=1` lands on `#/server/activity?x=1` (assert through whatever `router.svelte.test.ts` already uses for legacy-with-query; add one case there if routes.ts has no query input).
  - `serverHref('models', 'downloaded')` → `'#/server/models/downloaded'`
  - `Breadcrumb.test.ts`: the Server crumb links `#/server/activity`; on `#/server/models/downloaded` the crumbs are Server › Models.
  - `Sidebar.test.ts`: server rows in order Activity, Models, Connect, System; `#/server/models/downloaded` marks Models `aria-current="page"`.
- [ ] **Step 2: Run** `npx vitest run src/lib/routes.test.ts src/lib/Breadcrumb.test.ts src/lib/Sidebar.test.ts` — FAIL.
- [ ] **Step 3: Implement.** `parseView`: server group accepts `rest` only for `models` when it is one `ModelsTab`; `status`/`preferences` are no longer sections, so they fall through to `legacyRedirect`, which maps them (add a `case 'server':` branch reading `second`). Update the `jobs` and `server` legacy cases. `JobError`'s link → `serverHref('models','downloaded')`. In `App.svelte` map `activity` → `<StatusPage />`, `connect`/`system` → `<ServerPage />`, `models` → `<ModelsPage />` for now (replaced in Tasks 3–4).
- [ ] **Step 4: Run** the step-2 command, then `npm run check` — PASS.
- [ ] **Step 5: Commit** `feat(ui): server routes by question, old links redirect (refs #1762)`

### Task 3: Activity, Connect and System pages

**Files:**
- Create: `ui/src/lib/pages/ActivityPage.svelte`, `ConnectPage.svelte`, `SystemPage.svelte` (+ `.test.ts` each)
- Create: `ui/src/lib/SoftwareVersions.svelte` (+ test) — the dw and diffusers rows with their update controls, moved out of `ModelsPage.svelte`
- Create: `ui/src/lib/ActiveDownloads.svelte` (+ test) — in-flight downloads with cancel, moved out of `ModelsPage.svelte`
- Create: `ui/src/lib/serverAddress.svelte.ts` — the address/base derivation now inline in `ServerPage.svelte` (`addresses`, `address`, `proxied`, `base`), as `serverAddress(info: () => ServerInfo | null)` returning `{ addresses, selected, address, loopback, proxied, base }` getters
- Delete: `ui/src/lib/pages/StatusPage.svelte`, `ui/src/lib/pages/ServerPage.svelte` (move `ServerPage.test.ts` cases into the three new tests)
- Modify: `ui/src/App.svelte`, `ui/src/lib/pages/ModelsPage.svelte` (drops the update rows and in-flight downloads), `ui/src/lib/freshness.test.ts` (declarations for the new pages), `ui/e2e/smoke.spec.ts:89`, `ui/e2e/responsive.spec.ts:18`

**Interfaces:**
- Consumes: Task 2's routes.
- Produces: `<SoftwareVersions />`, `<ActiveDownloads />` (no props; each owns its `resource`, same `watch`/`alsoLive` as today's ModelsPage code).

Panel placement (spec §3):
- **Activity** — `PageHeader` "Activity", secondary `hostname · vversion · device`; `ServerHealthList`; `ActiveDownloads` (renders nothing when none active); an update in flight (`SoftwareVersions` shows its own running state, so Activity shows only a one-line "dw / diffusers updating…" link to System when either status is `running`); `JobsPage scope="all"`.
- **Connect** — Binding, Addresses, Authentication, MCP, Connect from another machine.
- **System** — Server (hostname, version, device), `SoftwareVersions`, Runtime, Directories, Storage, `CachesPanel`.

- [ ] **Step 1: Write failing tests** (one file per page, `apiMock`): each page renders its panel headings and none of the others' (e.g. `ConnectPage` has "Authentication" and no "Runtime"; `SystemPage` has "Runtime" and an update button for dw when `dwStatus().newer === true`; `ActivityPage` lists the queue and shows an active download's repo with a Cancel button). Carry over every existing `ServerPage.test.ts` assertion to the page that now owns its panel.
- [ ] **Step 2: Run** `npx vitest run src/lib/pages/ActivityPage.test.ts src/lib/pages/ConnectPage.test.ts src/lib/pages/SystemPage.test.ts src/lib/SoftwareVersions.test.ts src/lib/ActiveDownloads.test.ts` — FAIL.
- [ ] **Step 3: Implement** by moving markup and script, not rewriting it; route the three pages in `App.svelte`; update the two e2e specs to `#/server/activity`.
- [ ] **Step 4: Run** `npm test && npm run check && npm run lint` — PASS (freshness and arch-metrics included).
- [ ] **Step 5: Commit** `feat(ui): split Status into Activity, Connect and System (refs #1762)`

### Task 4: Models page with Preferences and Downloaded tabs

**Files:**
- Create: `ui/src/lib/pages/ModelsDownloaded.svelte` — today's `ModelsPage.svelte` body after Task 3 (download form, disk meter, cache list)
- Modify: `ui/src/lib/pages/ModelsPage.svelte` — becomes the tab shell
- Modify: `ui/src/lib/pages/PreferencesPage.svelte` — loses its own `PageHeader` (the shell owns it); body unchanged until Task 8
- Modify: `ui/src/App.svelte` (`models` passes `tab={view.rest[0] ?? 'preferences'}`)
- Test: `ui/src/lib/pages/ModelsPage.test.ts` (existing cases move to a `ModelsDownloaded.test.ts`)

**Interfaces:**
- Consumes: `ModelsTab`, `serverHref`.
- Produces: `ModelsPage` prop `tab: ModelsTab`.

- [ ] **Step 1: Write failing tests:** `ModelsPage` with `tab='preferences'` shows tab links "Preferences" (`aria-current="page"`, href `#/server/models/preferences`) and "Downloaded · N" (href `#/server/models/downloaded`, N = cached repo count from `listModels`), and renders the preferences content; with `tab='downloaded'` renders the download form (`aria-label="Hugging Face repo to download"`).
- [ ] **Step 2: Run** `npx vitest run src/lib/pages/ModelsPage.test.ts src/lib/pages/ModelsDownloaded.test.ts` — FAIL.
- [ ] **Step 3: Implement.** Tabs are links (route-driven, so back/forward work), styled like the editor's view switch; `PageHeader title="Models"`, cache summary and `DiskMeter` stay in the header.
- [ ] **Step 4: Run** `npm test && npm run check` — PASS.
- [ ] **Step 5: Commit** `feat(ui): Models holds Preferences and Downloaded as tabs (refs #1762)`

### Task 5: Full resolution on the preferences read (server)

**Files:**
- Create: `dw/preference_usage.py` — `preference_users(library) -> dict[str, list[str]]`
- Modify: `dw/server/routes/system.py` (`get_preferences`, `_entry`, new `_full_host`)
- Modify: `dw/server/api_models_library.py` (`PreferenceEntry`, `PreferenceContract`)
- Modify: `dw/preferences.py` only if `load_preferences()` does not already carry `contracts` through (then add it, untouched by the overlay)
- Test: `tests/test_preference_usage.py`, `tests/test_preferences_routes.py`
- Regenerate: `ui/src/lib/generated/*`

**Interfaces:**
- Produces (HTTP): `GET /api/preferences?resolve=true&full=true`. `full` is ignored unless `resolve` is true. With it, each entry also carries `used_by: list[str]` (sorted) and, when the shipped file has one, `contract: {required: list[str], optional: list[str]}`; resolutions are made with `host.gate_check = preferences.downloads_gate_check()`, as `dw/server/admission.py:230` does.
- Produces (Python): `preference_users(library) -> dict[str, list[str]]` — key → sorted workflow names that reach `preference:<key>` directly (a step's `workflow.path`, or the default of the variable a `variable:` path names) or through any chain of composed catalog workflows. `library` is what `dw.server.deps.sources_for(state, ws)` returns; read files from `library.entries()`.
- `resolve=true` without `full`, PUT and DELETE answers: unchanged (no gate check, no `used_by`, no `contract`).

- [ ] **Step 1: Write failing tests**
  - `test_preference_usage.py`:
    - `test_direct_path_counts`: catalog `a` with a step `{"workflow": {"path": "preference:k"}}` → `{"k": ["a"]}`.
    - `test_variable_default_counts`: path `"variable:model"`, variable default `"preference:k"` → counts.
    - `test_composed_parent_counts`: `b` composes `a` (step path `"a"`) → `{"k": ["a", "b"]}`.
    - (Review Focus 4) `test_cycle_terminates`: `a` composes `b`, `b` composes `a`, `b` names `preference:k` → `{"k": ["a", "b"]}`; and a self-composing workflow terminates.
    - `test_unreadable_json_is_skipped`: one file is invalid JSON → others still counted, no raise.
  - `test_preferences_routes.py`:
    - `test_full_read_adds_used_by_and_contract`: `?resolve=true&full=true` → `used_by` is a list; for `text_to_image`, `contract.required == ["prompt", "seed", "width", "height"]`.
    - `test_plain_resolve_is_unchanged`: `?resolve=true` has no `used_by`/`contract` key, and `downloads_gate_check` is not called (monkeypatch it to raise).
    - `test_full_read_runs_the_gate_check`: monkeypatch `preferences.downloads_gate_check` to return a check skipping `qa/a` with `"gated, no access to org/x"` → that tier's `skipped` carries it.
    - (Review Focus 1) `test_full_read_survives_hub_failure`: the gate check raises `OSError` for every target → 200, picks present, no notes.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_preference_usage.py tests/test_preferences_routes.py -v` — new tests FAIL.
- [ ] **Step 3: Implement.** `get_preferences` gains `full: bool = Query(False, description="With resolve: resolve as admission does (gated access, download notes) and add used_by and contract")`. `_entry(request, ws, store, key, resolve, full=False)`. `used_by` computed once per request, not per key. `resolve()` already swallows a raising `gate_check` (it logs and continues), so Review Focus 1 should pass without new code — confirm, don't add a second guard.
- [ ] **Step 4: Run** the step-2 command — PASS. Then `venv/bin/python scripts/dump_openapi.py && (cd ui && npm run gen:api)` and `venv/bin/python -m pytest tests/test_api_contract.py -v` — PASS.
- [ ] **Step 5: Commit** `feat(server): full preferences read with used_by and contract (refs #1762)`

### Task 6: The check route (server)

**Files:**
- Modify: `dw/preferences.py` — add `target_problems` and `check_draft`
- Modify: `dw/server/routes/system.py` — `POST /api/preferences/{key}/check`
- Modify: `dw/server/api_models_library.py` — `PreferenceCheckRequest`, `PreferenceTierCheck`, `PreferenceCheck`
- Test: `tests/test_preferences.py`, `tests/test_preferences_routes.py`
- Regenerate: `ui/src/lib/generated/*`

**Interfaces:**
- Produces (Python):
  - `target_problems(key, ladder, open_target) -> dict[str, str]` — target name → why a PUT would refuse it (not in the catalog; refused arguments; refused by security). Shape errors (not a dict, unknown tier, empty rung, extra target keys) still raise `PreferenceError`, worded as `check_ladder` words them; reuse `check_ladder`'s helpers so the two cannot drift.
  - `check_draft(key, ladder, store, host, open_target, contract=None) -> dict[str, dict]` — per tier present in `ladder`: `{"target", "skipped", "notes", "after": [names after the pick], "error", "target_errors": {name: message}, "warnings": {name: [message]}}`, keys omitted when empty. Resolution runs over `{**store, "preferences": {**store["preferences"], key: ladder}}` with `resolutions`'s semantics. `warnings` names each `contract["required"]` variable a target's workflow does not declare: `"does not take <var>"`.
- Produces (HTTP): `POST /api/preferences/{key}/check`, body `{"ladder": {...}, "full": false, "known_blocked": []}`, answers `{"tiers": {tier: PreferenceTierCheck}}`. Never writes; no `If-Match`; not `@writes`. A key not in the store is allowed. Shape error → 422 with the `PreferenceError` text. `full=false`: `host.gate_check` = `downloads_gate_check(lookup_sizes=False)` wrapped so a name in `known_blocked` is skipped with reason `"gated, no access (last full check)"` before the inner check runs. `full=true`: as Task 5's full read. Contract from the shipped file for `key`, if any.

- [ ] **Step 1: Write failing tests**
  - `test_preferences.py`:
    - `test_target_problems_names_each_stale_target`: ladder `{"draft": ["qa/a", "gone/x"], "best": ["gone/y"]}` → `{"gone/x": ..., "gone/y": ...}`, each message containing "not in the catalog".
    - `test_check_draft_lists_after_the_pick`: `{"standard": ["qa/a", "qa/b"]}` on a fitting host → `target == "qa/a"`, `after == ["qa/b"]`.
    - `test_check_draft_skips_known_blocked`: `qa/a` in `known_blocked` → skipped with reason `"gated, no access (last full check)"`, `target == "qa/b"`.
    - `test_check_draft_warns_on_contract`: contract `{"required": ["prompt", "width"]}`, `qa/a` declares only `prompt` → `warnings == {"qa/a": ["does not take width"]}`.
  - `test_preferences_routes.py`:
    - `test_check_writes_nothing`: POST a ladder for `text_to_image` → 200 and `_overlay()` unchanged (None).
    - `test_check_accepts_a_new_key`: key `brand_new` → 200.
    - `test_check_refuses_a_bad_shape`: `{"ladder": {"huge": ["qa/a"]}}` → 422 naming the tier.
    - `test_check_fast_makes_no_hub_call`: monkeypatch `huggingface_hub.HfApi` methods and `requests.head` (whatever `plan_downloads` calls with `lookup_sizes=True`) to raise → `full=false` answers 200. If `lookup_sizes=False` still reaches the Hub, fix that path here.
    - `test_check_validates_the_key`: key `../x` → 422 (as the PUT refuses it).
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_preferences.py tests/test_preferences_routes.py -v` — new tests FAIL.
- [ ] **Step 3: Implement** the two functions and the route.
- [ ] **Step 4: Run** step 2 — PASS; regenerate types and run `tests/test_api_contract.py` — PASS.
- [ ] **Step 5: Commit** `feat(server): dry-run check for an unsaved preference ladder (refs #1762)`

### Task 7: UI client, types and copy helpers

**Files:**
- Modify: `ui/src/lib/api.ts` — `getPreferences({ key?, resolve?, full? })`; `checkPreference(key: string, body: { ladder: unknown; full?: boolean; known_blocked?: string[] }): Promise<PreferenceCheck>`
- Modify: `ui/src/lib/types.ts` (re-export generated `PreferenceCheck`, `PreferenceTierCheck`)
- Modify: `ui/src/lib/preferences.ts` — `Rung` gains `notes: string[]`; `rungsFrom` fills it
- Create: `ui/src/lib/ladder/preferenceCopy.ts` — `keyLabel`, `reasonText`, `downloadNote`
- Modify: `ui/scripts/arch-metrics.mjs` — `checkPreference` in `NOT_READS` beside `validate` (same comment applies: a check a draft asks for, never a page's view)
- Modify: `ui/src/lib/test/apiMock.ts` — `checkPreference` default
- Test: `ui/src/lib/ladder/preferenceCopy.test.ts`, `ui/src/lib/preferences.test.ts`, `ui/src/lib/api.test.ts`

**Interfaces:**
- Produces:
  - `keyLabel(key: string): string` — `'text_to_image'` → `'Text to image'`; `'image-edit'` → `'Image edit'`
  - `reasonText(reason: string): string` — `'cuda-only'` → `'needs CUDA'`; `'mps-only'` → `'needs Apple MPS'`; `'cpu-only'` → `'CPU only'`; `'cuda/mps-only'` → `'needs CUDA or Apple MPS'`; `'needs 32 GB VRAM on cuda, the smallest card has 24 GB'` → unchanged; anything else unchanged
  - `downloadNote(notes: string[]): { text: string } | null` — `'will download 41.2 GB'` → `{ text: 'downloads 41.2 GB on first run' }`; `'will download its weights'` → `{ text: 'downloads its weights on first run' }`; `'not downloaded'` → same as weights; no download note → `null`

- [ ] **Step 1: Write failing tests** for each mapping above, `rungsFrom` carrying `notes`, and `api.getPreferences({resolve: true, full: true})` requesting `/api/preferences?resolve=true&full=true` and `checkPreference('k', {...})` POSTing JSON to `/api/preferences/k/check`.
- [ ] **Step 2: Run** `npx vitest run src/lib/ladder/preferenceCopy.test.ts src/lib/preferences.test.ts src/lib/api.test.ts` — FAIL.
- [ ] **Step 3: Implement.** Match the server's note strings exactly as Task 6's `full=false` path emits them; read `dw/preferences.py` `downloads_gate_check` for the wording.
- [ ] **Step 4: Run** step 2, then `npm test` (arch-metrics test included) — PASS.
- [ ] **Step 5: Commit** `feat(ui): preference check client and plain-language copy (refs #1762)`

### Task 8: Preferences read view

**Files:**
- Modify: `ui/src/lib/pages/PreferencesPage.svelte` — full read, intro, default-tier control, cards
- Create: `ui/src/lib/ladder/PreferenceCard.svelte` (read mode; hosts the editor from Task 9) + `PreferenceCard.test.ts`
- Create: `ui/src/lib/ladder/TierAnswer.svelte` — one tier row of the read view
- Delete (after Task 9 replaces its editing): nothing yet; `LadderCard.svelte` stays until Task 9
- Test: `ui/src/lib/pages/PreferencesPage.test.ts`

**Interfaces:**
- Consumes: Task 5's full read (`resource(() => api.getPreferences({ resolve: true, full: true }), { watch: ['workflows', 'models'] })`), Task 7 helpers.
- Produces: `PreferenceCard` props `{ key, entry: PreferenceEntry | null, defaultTier: string, catalog: readonly string[], onsaved, onreset, onreload, ondiscard? }` (same callbacks as `LadderCard`, so `PreferencesPage`'s adopt logic is kept).

What the page shows (spec §4; copy may be tightened, meanings may not):
- Intro: "When a workflow or agent asks for an image (or another kind of output) without naming a model, dw picks one from that preference's list. Each list has three tiers — Draft is fastest, Best is highest quality — and dw runs the first model in the tier that this machine can run."
- Default tier: `Segmented` labelled "Requests that don't name a tier run at", options Draft/Standard/Best, same `setDefaultTier` action and `StaleNotice` on 412.
- Card header: `keyLabel(key)` + `<code>{key}</code>`; chip "as shipped" or "changed on this server"; "used by N workflows" as a `Details` toggle listing each (link to `#/shared/examples/<name>` or the workspace's `workflows/<name>` — use whichever `WorkflowsPage` links use for a catalog entry); Edit button; for `source === 'server'`, "Reset to shipped" (existing action, confirm copy names that it restores the shipped list).
- Per tier (`TierAnswer`): tier name, a "default" tag on the default tier (`--select`), then one of: `--good` dot + `<code>target</code>` + "ready"; `--warn` dot + target + `downloadNote` text linking to `serverHref('models','downloaded')`; `--bad` + "Nothing fits this machine: <error>". Below: skipped targets struck, each with `reasonText`; then "then <names> (fallback, not checked here)" muted. A tier absent from the ladder is not drawn.

- [ ] **Step 1: Write failing tests**
  - `PreferenceCard.test.ts`: ready tier shows "ready"; a tier with notes `['will download 41.2 GB']` shows "downloads 41.2 GB on first run" linking `#/server/models/downloaded`; skipped `{workflow:'models/minimax-h3-image', reason:'cuda-only'}` shows the name struck and "needs CUDA"; a ladder `['models/z-image','models/z-image-sdnq']` picking the first shows "then models/z-image-sdnq (fallback, not checked here)"; the default tier row has the "default" tag; "used by 2 workflows" expands to both names.
  - (Review Focus 3) a resolution `{error: "Preference 'k' tier best names a target not in the catalog: 'gone/x'"}` renders "Nothing fits this machine" with that text, inside the card, not a page-level error.
  - `PreferencesPage.test.ts`: requests `getPreferences` with `{resolve: true, full: true}`; choosing Best in the default-tier control calls `setDefaultTier('best', etag)`.
- [ ] **Step 2: Run** `npx vitest run src/lib/ladder/PreferenceCard.test.ts src/lib/pages/PreferencesPage.test.ts` — FAIL.
- [ ] **Step 3: Implement.** Edit is a button that, until Task 9, swaps in the existing `LadderCard` editing UI, so nothing regresses mid-branch.
- [ ] **Step 4: Run** `npm test && npm run check && npm run lint` — PASS (design-rules: if `--select` is a new token use in a listed file, add the file to `scripts/design-rules.test.ts`'s list — that list is review's, not a ratchet).
- [ ] **Step 5: Commit** `feat(ui): Preferences read view answers what runs here (refs #1762)`

### Task 9: Chain editor with live dry run

**Files:**
- Create: `ui/src/lib/ladder/ChainEditor.svelte` (+ test) — the tiers as chips
- Create: `ui/src/lib/ladder/AddTarget.svelte` — the `+ add` popover (`Popover`, `Suggest`, device checkboxes)
- Create: `ui/src/lib/ladder/dryRun.svelte.ts` (+ test) — `dryRun(key: () => string, draft: () => Ladder, knownBlocked: () => string[])` returning `{ readonly answer: PreferenceCheck | null; readonly checking: boolean }`
- Modify: `ui/src/lib/ladder/ladderDraft.ts` — `blockedFrom(resolutions): string[]` (targets the last full read skipped with a reason starting `"gated"`)
- Modify: `ui/src/lib/ladder/PreferenceCard.svelte` — Edit mounts `ChainEditor`; Save/Discard/Reset live here
- Delete: `ui/src/lib/ladder/LadderCard.svelte` (+ its test, cases moved to `ChainEditor.test.ts` / `PreferenceCard.test.ts`)
- Modify: `ui/scripts/arch-metrics.mjs` `STORE_EXCEPTIONS` only if `dryRun`'s call is counted as an unwatched read — name the reason ("a draft's dry run, not page data")

**Interfaces:**
- Consumes: Task 7 `api.checkPreference`, `reasonText`, `downloadNote`; Task 8 `PreferenceCard`.
- Produces: `ChainEditor` props `{ draft: Ladder (bindable), catalog: readonly string[], answer: PreferenceCheck | null, contract?: {required: string[]; optional: string[]} }`.

Behaviour (spec §5):
- One card in edit at a time (`PreferencesPage` holds `editing: string | null`); an edited, unsaved draft registers `addLeaveGuard` while dirty, removed on save/discard/unmount.
- Chips left to right under "try in order →"; drag to reorder (native HTML drag and drop; no new dependency); Alt+ArrowLeft/Alt+ArrowRight on a focused chip moves it; up/down buttons remain; × removes. Chip shows target arguments as "with a, b" (read-only).
- Chip state from `answer.tiers[tier]`: the pick → "runs here" (`--good`); in `skipped` → struck + `reasonText`; in `after` → "fallback, not checked here"; in `target_errors` → `--bad` + the message; in `warnings` → `--warn` + each message. A target added in this edit (not in the ladder the edit started from) and shown as "runs here" also says "access checked on save". When the key has a `contract`, the editor's head says "Every model here must take: <required, comma-separated>".
- `dryRun`: 300 ms after the last draft change, `checkPreference(key, { ladder: sendable(draft), full: false, known_blocked })`; each call carries a sequence number and an answer older than the latest sent is dropped.
- Footer when dirty: "Unsaved changes · Jobs already queued keep the model they were given", Discard, Save. Save disabled while any `target_errors` or the dry run reports a shape error. Save → existing PUT with If-Match (412 `StaleNotice`, 422 `ErrorLine` as today) → then `onreload()` refetches the full read (Task 8's resource `refresh`).
- An emptied tier shows "No models: requests at this tier are refused" and is left out of `sendable` as today.
- **New preference** (spec §7): the page's "New preference" button opens a form — name (`keyProblem` as today), the line "A workflow uses it by naming preference:<name> as a step's workflow path.", and a first model for the default tier (`Suggest`) — and creates the card in edit mode, unsaved.

- [ ] **Step 1: Write failing tests**
  - `dryRun.svelte.test.ts` (fake timers): two draft changes 100 ms apart send one check after 300 ms; (Review Focus 2) with two checks in flight where the first resolves last, `answer` is the second's.
  - `ChainEditor.test.ts`: Alt+ArrowRight on the first chip of `['a','b']` makes the draft `['b','a']`; × removes; adding through the popover with device `cuda` appends `{workflow:'x', devices:['cuda']}`; an answer with `target_errors: {'gone/x': 'not in the catalog'}` marks that chip and the card's Save is disabled; `after: ['b']` shows "fallback, not checked here"; emptying a tier shows the refused copy.
  - `PreferenceCard.test.ts`: Edit then Save calls `setPreference(key, ladder, etag)` then `onreload`; a dirty draft registers a leave guard (mock `addLeaveGuard`) and Discard removes it; editing a second card while one is dirty is not possible (its Edit is disabled with title "Save or discard the open edit first").
  - `PreferencesPage.test.ts`: New preference with name `image_edit` and first model `qa/a` shows an unsaved card in edit mode; an invalid name shows `keyProblem`'s message.
- [ ] **Step 2: Run** `npx vitest run src/lib/ladder src/lib/pages/PreferencesPage.test.ts` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test && npm run check && npm run lint` — PASS.
- [ ] **Step 5: Commit** `feat(ui): chain editor with a live dry run for preferences (refs #1762)`

### Task 10: End-to-end, docs and the whole-branch check

**Files:**
- Modify/Create: `ui/e2e/preferences.spec.ts` (or extend the existing preferences e2e if one exists — `grep -rl preferences ui/e2e`)
- Modify: `docs/SERVER.md` (the web UI's Server pages), `ui/GLOSSARY.md` if a new term is introduced (Activity, Connect, System, "fallback"), `docs/ARCHITECTURE.md` row for the preferences routes if one names them
- Modify: `dw_mcp` nothing (MCP unchanged; confirm `tests/test_mcp*` pass)

- [ ] **Step 1: Write the e2e:** open `#/server/status` → lands on Activity; sidebar shows Activity, Models, Connect, System; top-bar schema link opens Schema; Models › Preferences shows the text_to_image card with a "ready" or "downloads … on first run" row; Edit, Alt+ArrowRight a chip, the chip states update within 2 s, Discard restores.
- [ ] **Step 2: Run** `cd ui && npm run build && npx playwright test e2e/preferences.spec.ts e2e/smoke.spec.ts e2e/responsive.spec.ts` — PASS (never `dw.serve` on the e2e port; `DW_E2E_PORT` if another checkout holds it).
- [ ] **Step 3: Update docs.**
- [ ] **Step 4: Full run:** `venv/bin/python -m pytest tests/ -n auto` and, from `ui/`, `npm run check && npm run lint && npm test && npx playwright test` — all PASS; report any ratchet that moved and stop for Don if one rose.
- [ ] **Step 5: Commit** `docs: server pages and preferences (refs #1762)`
