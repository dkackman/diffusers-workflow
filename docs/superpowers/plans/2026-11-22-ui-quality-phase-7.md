# UI quality phase 7: shell, first run, performance. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a newcomer who opens the UI, on this machine or on a GPU box across the network, sees something within a fraction of a second, is told what the app is and gets one click to a workflow that will run, never lands on an empty page with nowhere to go, can switch workspace and section from any sidebar state, and reads true things on the Status and Models pages; the app sends a fraction of today's bytes, and the next agent cannot quietly send them again.

**Architecture:** One server change, in `dw/server/app.py`: responses are gzip-compressed when the client accepts it (Starlette's `GZipMiddleware`, which already leaves event streams, media and ranged responses alone; `/mcp` is left out on top), and the UI mount becomes a `StaticFiles` subclass that marks the content-hashed `/assets/*` immutable and makes `index.html` revalidate. Compression is transport only: every body decodes to the bytes it is today, so MCP sees nothing new. The rest is UI: a loading shell inline in `index.html`, Monaco imported as the editor core plus the JSON language only, the Jobs list paged through the `limit`/`total` contract `/api/jobs` already has (no server change), a first-run card on the Overview, and the shell fixes. The budgets are held by `ui/e2e/performance.spec.ts`, which the full e2e run includes.

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config, Vite 8, monaco-editor 0.57 (already a dependency; no package change). Server: FastAPI 0.143.0 / Starlette 1.10, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`. This plan covers:
- the phase-7 row of *Phases*: SH-3, SH-4, SH-6..9, SH-16, SH-17, SH-19, SH-S1..S6, XC-8, XC-14, JB-8, JB-9, JB-12, JB-17; the building blocks the first-run card, gzip/caching, a loading shell, monaco trim and paged job history; the journey gate `first-run.spec.ts`;
- goal 5 (*A compelling first hour*) and the Non-goals line "server changes are limited to ... compression";
- decision 4 (SH-S1), found already done.

The audit (`docs/superpowers/specs/2026-11-22-ui-audit.md`: Tier 2 §C *First run*, §E *Server status* and *Remote performance*, Tier 3 *Shell* and *Jobs*) has the finding text. This is the program's last phase: *Open items no phase owns* at the end lists what stays open.

**Out of scope:**
- Splitting the main bundle by route (lazy pages). The entry chunk is 260 KB gzipped; the budget below holds it there, and a split is its own change.
- A narrower `/api/workflows` for the Overview: compression takes it from 282 KB to 62 KB (decision 4).
- Navigation chords or any global shortcut beyond `/` and `?` (Question 3).
- Every carry-over item not listed under *Folded in*.

**Folded in** (each in code a task already rewrites):
1. #1611 - The Overview is 446 px wide at 390 px (the h1's path line) (Task 10).
2. #1611 - Header popovers stay open after a fast tap on a link; close them on navigation (Task 9).
3. #1601 - The meter's aria-label omits the visible state word, and "running" shows twice with one job and a meter (Task 9).
4. #1601 - ServerPage's h1 "Status" over an h2 "Status" (Task 6).
5. #1602 - No test covers the no-meter `N running` and `status` labels (Task 9).
6. #1600 - The header breadcrumb on the Models page "renders nearly invisible": checked by eye in both themes, fixed if it differs from other pages, otherwise closed as not reproduced (Task 9).
7. #1600 - Workspace delete is a dead end while the gallery cannot be read (`workspaceActions.ts`, the Overview's Manage panel) (Task 11). #1600 is labelled release-blocker.
8. #1606 - A very long job id spills past the Overview row (Task 10).
9. #1606 and #1610 - The Overview's jsdom computed-style assertion can't fail; replaced by the real-browser overflow check in `e2e/responsive.spec.ts` (Task 10).
10. #1609 - The Jobs search placeholder is clipped by PageHeader's 220 px filter cap (a `title`) (Task 5).
11. #1609 - `listWithPrompts` has no test of the new-job re-ask branch, and the page test never runs a second poll (Task 5, which rewrites it).
12. #1623 - The Overview row was checked at 390 px only; Task 10's by-eye pass covers 1440 and 1300 px.

**Left alone** (no task touches the code): #1600's other items, #1601's other items (the Schema h1 is Don's product call), #1602's metric, lint and journey items, #1603 (the engine import wedge), #1605, #1606's other items, #1607 (all; this phase does not touch `job_history.py`), #1608, #1609's Compare, takes and day-unit items, #1610's other items, #1612, #1623's other items.

## Measured at the start (develop f8690eb9)

**UI metrics** (`cd ui && node scripts/arch-metrics.mjs`): `files_over_size_ceiling 0, complex_functions 13, long_functions 0, import_cycles 0, modules_in_import_cycles 0, prefix_literals 0, a11y_suppressions 1, raw_error_toasts 0, empty_outside_loadstate 0, unwatched_reads 0, window_dialogs 0, unguarded_mutations 0, banned_terms 0, raw_media 1, duplicate_blocks 16`. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only `complex_functions: 11 -> 13`.

**Python** (`$PY scripts/arch_metrics.py`): `modules 281`, `claude_md_lines 129`, `modules_over_size_ceiling 5`.

| File | Lines | Cap this phase |
|---|---|---|
| `dw/server/app.py` | 292 | 292 (Task 1 pays in its own comments) |
| `dw/server/http_security.py` | 180 | 180 (untouched) |
| `dw/server/api_models.py` | 1105 | 1105 (untouched) |
| `dw/server/job_history.py` | 1000 | 1000 (untouched) |
| `dw/server/jobs.py` | 1139 | 1139 (untouched) |
| `dw/server/routes/jobs.py` | 1033 | 1033 (untouched) |
| `dw/server/routes/system.py` | 599 | 599 (untouched) |

**UI sizes** (limit 600): `app.css` 600 (full; new global rules go in `src/patterns.css`, 517), `api.ts` 598, `Sidebar` 584, `App` 561, `OverviewPage` 526, `ModelsPage` 505, `ServerPage` 333, `CachesPanel` 275, `JobsPage` 270, `JsonEditor` 151, `monaco.ts` 125, `LinkedNotes` 94, `main.ts` 22.

**Bytes and time** (the fixture server, `ui/e2e/serve_fixture.py`, built UI, Chromium):

| What | Today |
|---|---|
| Entry chunk `index-*.js` | 813 KB raw, 260 KB gzipped |
| Entry CSS `index-*.css` | 112 KB raw, 21 KB gzipped |
| Overview, first load | 23 responses, 1,201 KB on the wire (nothing is compressed): JS 794, CSS 109, `/api/workflows` 282 (62 KB gzipped) |
| Jobs / Status / Gallery, first load | about 920 KB each |
| Overview at 5 Mbps, 50 ms | first paint 1,664 ms (a blank page until the bundle lands), idle 2,723 ms |
| JSON view, first open | 5,101 KB: `editor.api` 2,672, `monaco` 1,266, json worker 423, editor worker 297, codicon 150, CSS 182, schema 69; 2.8 s on loopback, about 8 s at 5 Mbps |
| Response headers on `/` and `/assets/*` | `etag` and `last-modified` only: no `content-encoding`, no `cache-control` |
| Polls in 20 s, steady | Overview and Gallery 12 (changes, memory, health × 4); Jobs and Status 18 (those plus `/api/jobs` × 6, every job, every 3 s) |

**Findings checked against the code:**

| ID | State today | Where |
|---|---|---|
| SH-3 | open: a loopback bind lists every interface's URL and builds the "Connect from another machine" snippets from them | `pages/ServerPage.svelte:101-145,197-253` |
| SH-4 | open: the document is 446 px wide at 390 px (the h1's secondary path line does not wrap) | `ui/PageHeader.svelte`, `pages/OverviewPage.svelte:139` |
| SH-6 | open: a new workspace's empty states are text with nowhere to go | `pages/OverviewPage.svelte:168,262,296`, `JobsPage.svelte:238`, `GalleryPage.svelte:363`, `AssetsPage.svelte:423` |
| SH-7 | open: no orientation; the first workflow cards are whatever sorts first | `pages/OverviewPage.svelte` |
| SH-8 | open: collapsed, the sidebar draws only the open workspace's dot | `Sidebar.svelte:178-225` (`{:else if !collapsed}`) |
| SH-9 | open: the new-workspace field has no Create or Cancel and stays open on a click away | `Sidebar.svelte:300-338` |
| SH-16 | partly done: the table scrolls itself and the form stacks at 640 px. The field has only a placeholder, and at 390 px the size column is off screen | `pages/ModelsPage.svelte:134-156,394-406` |
| SH-17 | open: `low_disk` warns on a disabled cache | `CachesPanel.svelte:141-148` |
| SH-19 | open: a shut workspace links to its overview whatever section is open | `Sidebar.svelte:209-224` |
| SH-S1 | done: `UPDATE_DIFFUSERS_CONFIRM` (phase 3, decision 4) | `pages/ModelsPage.svelte:57-66` |
| SH-S2 | open: "+ new" disappears with no reason when the server has no workspace root | `Sidebar.svelte:300` |
| SH-S3 | partly done: the button has an `aria-label`, `theme: <t> - click to change`; it does not say what a press does | `App.svelte:303-314` |
| SH-S4 | open: the copy says `dw-serve`; a checkout runs `python -m dw.serve` | `pages/ServerPage.svelte:174,251` |
| SH-S5 | open: the `claude mcp add` line always carries the token header | `serverinfo.ts:mcpAddCommand` |
| SH-S6 | open: `?` opens the help but nothing says so; no shortcut reaches a page's search | `App.svelte:94-102`, `KeyboardHelp.svelte` |
| XC-8 | open: no compression, no cache headers, no loading shell, Monaco imported whole (`import * as monaco from 'monaco-editor'`) | `dw/server/app.py:290`, `ui/index.html`, `ui/src/lib/monaco.ts:6` |
| XC-14 | done: the Status page polls `/api/jobs` once every 3 s (6 in 20 s, measured above) | `pages/StatusPage.svelte` |
| JB-8 | done: a queued row sits on the same grid as a finished one at 1440 and 390 px (phase 4's `JobRow`; checked with two injected queued jobs) | `job/JobRow.svelte` |
| JB-9 | open: on a dropped item's job, `LinkedNotes` draws its bordered panel around a `NoteList` that renders nothing | `request/LinkedNotes.svelte:71-84` |
| JB-12 | done: a queued row reads `waiting 3m 20s` (with `· cards busy` / `· host RAM` when held), not a log line | `job/JobRow.svelte:180-198` |
| JB-17 | open: the Jobs page reads every job, every 3 s | `pages/JobsPage.svelte:87-98`, `jobsListing.ts:listJobs` |

**Found already done:** SH-S1 (phase 3), XC-14 (phase 2a's `resource`), JB-8 and JB-12 (phase 4's `JobRow`).

**Decisions the spec left open** (taken here; questions are separate):
1. **Paged job history is the contract `/api/jobs` already has** (Question 2). The page asks for every queued and running job (`status=queued,running`) and the newest `JOBS_PAGE` (50) finished ones (`status=succeeded,failed,cancelled&limit=<shown>`), and shows `total`. **Show older** raises `shown` by 50. No server change, so MCP's `list_jobs` is untouched. Two requests, so a running job older than the page is never cut off.
2. **Compression is app-wide except `/mcp`**, with Starlette's own exclusions (event streams, media types, zip, a 206). JSON, HTML, JS, CSS and the font compress. Level 6, minimum 1,000 bytes.
3. **Caching:** `/assets/*` (Vite names them by content hash) `Cache-Control: public, max-age=31536000, immutable`; `/` and `/index.html` `Cache-Control: no-cache`, so an upgraded server's new bundle is picked up on the next load. `/outputs`, `/inputs` and the API keep what they send today.
4. **The Overview keeps reading `/api/workflows`** (it needs the names and `writable`, and the first-run card needs the template's place): 282 KB becomes 62 KB on the wire. A narrower listing would be a new server parameter for about 0.1 s at 5 Mbps.
5. **The loading shell is static HTML and CSS in `index.html`**, with the theme set by an inline script before first paint (moved from `main.ts`, so the shell and the app agree). A CSS-only delay turns it into a "didn't load" line after 10 s, which works when the script never runs.
6. **The Monaco import keeps every editor feature and only the JSON language**: `editor.api.js`, the editor contributions `monaco-editor/esm/vs/index.js` lists, and `languages/features/json/register.js`; never the language definitions, the LSP client, or the CSS, HTML and TypeScript language features. A test pins the contribution list to the installed package, so an upgrade that adds one names it.
7. **Budgets live in `ui/e2e/performance.spec.ts`** as named constants, measured on the wire with gzip: entry JS + CSS 300 KB (today 281), the Overview's first load 420 KB (today about 350 compressed), the JSON view 1,250 KB (today 1,350 compressed). A task that goes over stops and reports; nobody raises a budget without Don.
8. **Switching workspace keeps the section**, dropping what follows it (`jobs/<id>` becomes `jobs`); `edit` becomes `workflows` and `compare` becomes `gallery`, because both name another workspace's files. From a shared or server page it lands on the overview, as today.
9. **The new-workspace field closes on a click away only when it is empty**; a typed name stays until Cancel or Escape.
10. **The first-run card's link is found, not assumed**: `templates/text-to-image` where the listing has it (the workspace's own library opens its run page, a read-only library its Examples page), else the Examples page, else the Workflows page.
11. **`python -m dw.serve` is the command the UI prints**: it runs in a checkout and in an install alike.
12. **`Sidebar` splits**: the new-workspace form becomes `NewWorkspace.svelte`, so the sidebar falls under 560 lines with room for Tasks 8's changes.

## Questions for Don

Each task is written for the recommended answer.

1. **When does the first-run card show?** Recommended: on every workspace's Overview until **Got it** is pressed in this browser, then never again in it. The alternative is only while the server has no jobs at all, which a newcomer to a shared GPU box with history would never see. (Task 10.)
2. **Paged Jobs: how far does search reach?** Recommended: the page holds the live jobs plus the newest 50 finished, a filter searches what is loaded and says so (`Searching the newest 50 of 812 jobs.`), and **Search all** loads the rest on request. No server change, so MCP is untouched. The alternative is a server-side `q` parameter on `/api/jobs` (opt-in, additive), which searches the whole history without loading it. (Task 5.)
3. **Global shortcuts (SH-S6).** Recommended: a header button for the shortcut help, `?` named on the first-run card, and one new key, `/` to focus the page's search. No navigation chords (`g j` and the like). (Task 9.)

## Global constraints

- **MCP behaviour is unchanged.** The one server change is transport: a gzipped body decodes to today's bytes, `/mcp` is never compressed, and no route, parameter, status or key changes. `dw_mcp`'s HTTP client already decodes gzip; Task 1 pins `list_jobs` and `get_server` bodies byte for byte with and without `Accept-Encoding`. The Jobs paging uses `status` and `limit`, which exist; `dw_mcp/catalog.py:list_jobs` keeps sending what it sends. No OpenAPI change, so nothing under `ui/src/lib/generated/` is regenerated. (If one were: `PYTHONPATH=/private/tmp/claude-501/-Users-don-src-dkackman-diffusers-workflow/c38aa355-7a1b-44f4-aee3-aaaef6ebc96d/scratchpad/p3/openapi-overlay $PY scripts/dump_openapi.py && cd ui && npm run gen:api`; never hand-edit the generated files.)
- `PY=/Users/don/src/dkackman/diffusers-workflow/venv/bin/python` (the worktree has no venv).
- **Size limits.**
  - Python `modules` stays 281. No task adds a module (a test file is not one). A task that finds it needs one stops and asks Don.
  - `claude_md_lines` stays 129: `ui/CLAUDE.md` stays 12 lines, rewritten, never grown.
  - The caps in the table above hold. `app.py` pays for Task 1 in its own comments and docstrings, said in fewer lines, never losing a fact and never by moving code out. If it cannot, stop and ask Don.
  - `ui/src/lib/api.ts` ≤ 598 (this phase adds nothing to it). No page over 600 lines; `Sidebar` ≤ 560, `App` ≤ 590, `OverviewPage` ≤ 560. `app.css` is full.
  - No function over complexity 10 or 60 lines, in TS or Python.
- **The UI metrics gate.** Before Task 1: `cd ui && node scripts/arch-metrics.mjs > "$SCRATCH/ui-p7.json"`. Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-p7.json"`, which prints nothing; `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only `complex_functions: 11 -> 13`. Every 0 stays 0. A task that lowers a count lowers that key in `baseline.json` in the same commit.
- **Checks every task runs.** UI, from `ui/`: `npm run check`, `npm run lint`, the full `npm test` before committing; after any copy change `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts` (the test at `:36`, "editor opens a workflow with introspected arguments", is the known flake #1603: rerun it alone with `-g "introspected arguments"`). e2e always serves the built UI. A task that changes a URL the page calls runs the specs that match it (`grep -ln "api/jobs" e2e/*.ts e2e/journeys/*.ts` for Task 5). Python: `$PY -m pytest <files> -q` with `DW_STRICT_RESPONSES=1`, and `$PY scripts/arch_metrics.py`.
- **Staging and commits.** Stage files by name; never `git add -A ui` (Don edits `ui/package.json`; a dependency change is a question for Don, not a task), never add or force-add `.superpowers/`, never `git stash`. Headlines use a scope (`server(ui): …`, `ui(shell): …`, `ui(jobs): …`, `docs(…)`), never `fix: #N`; name an issue with `refs #N` in the body. Each commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Security.** No `eval`, `exec` or `shell=True`. The UI mount serves only files under the resolved UI directory: the subclass adds headers and never resolves a path itself (`StaticFiles` does, as today); Task 1 pins a traversal attempt. No new path or URL input.
- WCAG contrast is guidance, never a check. Names, focus and `aria-expanded` are in scope.
- **When a component replaces or absorbs another**, list each capability the old one had and pin each with a test (Tasks 2, 3 and 8 list theirs).
- **Fixtures and journeys** never rely on sleeps or spec order: wait on a condition; a workspace a spec creates gets a unique name and is deleted in `finally`; assert what holds whatever ran before.
- **Docs.** `docs/SERVER.md` and `docs/REMOTE.md` are edited by the task that changes the behaviour; ARCHITECTURE rows by Tasks 1, 3, 4 and 12.
- **Copy fixed by this plan.** Use it exactly; `<…>` is filled in, `<a|b>` picks by count, `[…]` is a link.
  - **Loading shell** (`index.html`): `Loading dw…`; after 10 s, in its place: `dw didn't load. Reload the page; if that doesn't help, check that the server is running.`
  - **JSON view that cannot load Monaco** (an upgraded server, a stale tab): `The JSON editor couldn't load. Reload the page to get the current version.`, the failure behind `<Details summary="What failed">`; the form view stays usable.
  - **Jobs paging**: under the list, while `shown < total`: `Showing the newest <shown> of <total> jobs.` with the button `Show <k> older` (`k = min(50, total - shown)`); while a status or search filter is on and `shown < total`, above the list: `Searching the newest <shown> of <total> jobs.` with the button `Search all <total>`. The search field's `title`: `search names, prompts, notes and ids`.
  - **Status page, loopback bind**: the Addresses panel `This server is bound to <bind_host>, so only this machine can reach it.` (no table); the Connect panel `Only this machine can reach this server. To use it from another, restart it bound to the network, with a token:`, the snippet `DW_API_TOKEN=<token> python -m dw.serve --host 0.0.0.0`, then [How to use dw from another machine](https://github.com/dkackman/diffusers-workflow/blob/master/docs/REMOTE.md).
  - **Status page, MCP not mounted**: `Not mounted. Restart the server with --mcp (python -m dw.serve --mcp) to serve the MCP tools at <path>.`; in the Connect panel `The MCP endpoint is not mounted, so there is no command to run yet. Restart the server with --mcp.`
  - **The `claude mcp add` line**: `claude mcp add --transport http dw <url>`, plus ` --header "Authorization: Bearer <token>"` and the note `Replace <token> with the server's API token.` only when `auth_required`.
  - **Status page panel**: the h2 `Status` becomes `Server`.
  - **Models download field**: `aria-label` `Hugging Face repo to download`, placeholder `org/name, e.g. black-forest-labs/FLUX.1-dev`; a value that is not one `/` between two non-empty, space-free parts shows `.hint` `A repo id is org/name, like black-forest-labs/FLUX.1-dev.` and Download stays disabled.
  - **Sidebar**: a collapsed shut workspace is a dot link titled `<name>`, `aria-label` `<name> workspace`. The new-workspace form's buttons `Create` (submit) and `Cancel`. With no workspace root, in place of `new`: muted `New workspaces need a workspace folder`, titled `Start the server with --workspace <folder> to add workspaces.`
  - **Header**: the theme button's `aria-label` and `title` `Theme: <system|light|dark>. Switch to <light|dark|system>.`; a help button (`Keyboard` icon) `aria-label` `Keyboard shortcuts`, `title` `Keyboard shortcuts (?)`; the help gains `<kbd>/</kbd>` `search this page's list`. The meter button's `aria-label` starts with its state word: `<idle|running>, server & worker status, <meter text>`; with the one-job `running` link showing, the meter shows no second `running`.
  - **First-run card**: heading `New here? Make your first picture`; `dw runs image, video and audio workflows on this server's GPU. Pick a workflow, press Run, and what it makes lands in the Gallery.`; an ordered list `Open text-to-image. It uses a small model that needs no Hugging Face login.` / `Type a prompt and press Run.` / `Watch the job, then find the picture in the Gallery.`; the primary link `Open text-to-image` (`Browse examples` when the listing has no `templates/text-to-image` but has examples, `Open Workflows` when it has neither); `Press ? for keyboard shortcuts.`; the button `Got it`, `aria-label` `Dismiss this introduction`.
  - **Empty states** (SH-6): Overview recent `Nothing generated yet. [Run a workflow] and what it makes lands here.`; Overview jobs `No jobs yet. [Run a workflow] to start one.`; Overview workflows `No workflows yet. [Create one in the editor], or [start from an example].`; Jobs `No jobs yet. [Pick a workflow] and run it.`; Gallery (no request filter) `Nothing generated yet. [Run a workflow] and what it makes lands here.`; Assets `Nothing in the asset library yet. Upload a file, or open one in the [Gallery] and Save as asset.` Every "Run a workflow" / "Pick a workflow" goes to the workspace's Workflows page.
  - **Workspace delete with no starred count**: the confirm's message ends `The starred count couldn't be read.` in place of the count.

## Review focus

1. **A browser holding the previous version.** After the GPU box is upgraded, a tab opened before it asks for a Monaco chunk that no longer exists, and a returning browser must not run a cached old `index.html`. `index.html` revalidates (`no-cache`); a failed Monaco import says so and the form view still works; the loading shell says the app didn't load when the bundle never arrives. Pinned in Task 1 (`index_html_revalidates`), Task 3 (`a JSON editor that cannot load Monaco says so`) and Task 2 (`the shell says the app did not load after 10 s`).
2. **Compression where it must not happen.** Live job events must still arrive one by one, a video seek must still get a 206 with the right bytes, a ZIP download must stay a ZIP, `/mcp` must stay as it is. Pinned in Task 1 (`test_event_stream_is_not_compressed`, `test_a_range_request_is_not_compressed`, `test_media_and_zip_are_not_compressed`, `test_mcp_is_never_compressed`).
3. **A history of thousands of jobs.** The search and status filter only see loaded rows; a running job created before the newest 50 finished ones; a restart batch partly beyond the page. The page says what it searched, a live job is never cut off, and `Search all` reaches the rest. Pinned in Task 5 (`a running job older than the page is listed`, `a filter over a partial page says what it searched`).
4. **Storage that throws** (a private window, blocked site data) under the first-run card's dismissal, the sidebar's collapse and the theme script. The card shows, `Got it` hides it for the session, nothing breaks, and the theme falls back to the system's. Pinned in Task 10 (`a dismissal that cannot be stored lasts the session`) and Task 2 (`the theme script survives storage that throws`).
5. **A server without the template the card names**: a named workspace whose library lacks `templates/text-to-image`, a server with no examples, a listing that fails. The card links the Examples page or the Workflows page, and a failed listing still shows the card with the Workflows link. Pinned in Task 10 (`the card finds the template where the listing has it`, `no template: the examples`, `no examples: the workflows`, `a failed listing still orients`).

---

### Task 1: Compression and caching for the UI (server)

XC-8's server half. Transport only.

**Files:**
- Modify: `dw/server/app.py` (`create_app`; a `UiFiles(StaticFiles)` subclass; `GZipMiddleware`), paying in its own comments
- Test: `tests/test_server_delivery.py` (new test file); `tests/test_security_web_headers.py`, `tests/test_server.py`, `tests/test_mcp_server.py` (unchanged; must pass)
- Docs: `docs/SERVER.md` (a short paragraph under the web UI: responses are gzipped when asked; `/assets/*` cached immutable, `index.html` revalidates); `docs/REMOTE.md` *Browser* (one sentence: the UI is compressed and cached, and a reload after an upgrade picks up the new version); ARCHITECTURE *Server* table, new row **UI delivery**: owner `dw/server/app.py`: `create_app`, `UiFiles`; rule "Responses are gzipped when the client accepts it, except `/mcp` and what Starlette leaves alone (event streams, media, zip, a 206); `/assets/*` is immutable for a year and `index.html` revalidates; the mount serves only the built UI directory"; enforced by `tests/test_server_delivery.py`

**Interfaces:**
- `class UiFiles(StaticFiles)` in `app.py`, overriding `file_response(self, full_path, stat_result, scope, status_code=200)`: calls the base, then sets `Cache-Control` by `scope["path"]`: under `/assets/` → `public, max-age=31536000, immutable`; anything else → `no-cache`. The mount line becomes `app.mount("/", UiFiles(directory=resolved_ui, html=True), name="ui")`.
- Compression: `app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)` wrapped so `/mcp` and `/mcp/*` pass through uncompressed: a nested ASGI callable `_compress(app_)` in `app.py` that routes by `scope["path"]` (no new module). It is added after `install_middleware(app)` so the security middlewares still answer first and their 400/401/403 bodies are compressed like any other.

- [ ] **Step 1: Write the failing tests** in `tests/test_server_delivery.py` (fixtures as in `tests/test_security_web_headers.py`; the ui dir holds `index.html` and `assets/app-abc123.js` of 5 KB):
  - `test_json_is_gzipped_when_asked` (`Accept-Encoding: gzip` on `GET /api/jobs` → `content-encoding: gzip`, `vary` names `Accept-Encoding`)
  - `test_a_gzipped_body_is_todays_body` (`GET /api/jobs`, `/api/server`, `/api/workflows` with and without gzip decode to equal bytes)
  - `test_assets_are_immutable` and `test_index_html_revalidates` (`/` and `/index.html` → `no-cache`, never `immutable`)
  - `test_event_stream_is_not_compressed` (`GET /api/jobs/{id}/events` on a finished job: no `content-encoding`)
  - `test_a_range_request_is_not_compressed` (`Range: bytes=0-3` on an output → 206, the 4 bytes, no `content-encoding`)
  - `test_media_and_zip_are_not_compressed` (a `.png` output and the zip download)
  - `test_mcp_is_never_compressed` (`create_app(mcp=True)`: a `POST /mcp` initialize with `Accept-Encoding: gzip` has no `content-encoding`)
  - `test_the_ui_mount_serves_nothing_outside_its_directory` (`/assets/../../<a file beside the ui dir>` and its `%2e%2e` spelling → 404)
  - `test_api_answers_keep_their_cache_headers` (`/api/jobs` carries no `immutable`)
- [ ] **Step 2: Run** `$PY -m pytest tests/test_server_delivery.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Pay the added lines in `app.py`'s comments; `wc -l dw/server/app.py` ≤ 292.
- [ ] **Step 4: Run** the file, `tests/test_security_web_headers.py`, `tests/test_security_auth.py`, `tests/test_server.py`, `tests/test_mcp_server.py`, `tests/test_api_contract.py`, `$PY scripts/arch_metrics.py` (`modules 281`), then `cd ui && npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/security.spec.ts e2e/job-event-log.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `server(ui): gzip what the client accepts, cache the hashed UI files`.

### Task 2: A loading shell that paints at once

XC-8's loading shell. The theme script moves from `main.ts` into `index.html`.

**Files:**
- Modify: `ui/index.html` (a `#boot` shell inside `#app`, an inline `<style>`, an inline theme script), `ui/src/main.ts` (clears `#app` before `mount`; its theme block goes)
- Test: `ui/scripts/loading-shell.test.ts` (new)

**Interfaces:**
- `index.html`: `<div id="app"><div id="boot" role="status"><span class="brand">dw</span><span class="loading">Loading dw…</span><span class="failed">…</span></div></div>`. The inline style gives `html` and `#boot` the `--bg`, `--ink` and `--muted` values `app.css` defines, light and dark, by `prefers-color-scheme` and by `[data-theme]`; `.failed` is hidden and shown by a 10 s `animation-delay` that also hides `.loading` (no script).
- The inline script is the block `main.ts` has today (read `localStorage` `dw-theme`, set `data-theme` for `light`/`dark`, swallow a throw), unchanged in behaviour.
- `main.ts`: `const target = document.getElementById('app')!; target.replaceChildren(); mount(App, { target })`.
- Capabilities of the moved theme block, each pinned: a stored `light`/`dark` applies before first paint; none or anything else leaves the OS preference; a throwing storage is ignored; `theme.ts`'s `cycleTheme` still writes the key the script reads.

- [ ] **Step 1: Write the failing tests** in `scripts/loading-shell.test.ts` (reads `index.html`, `src/app.css` and `src/lib/theme.ts` from disk; runs the inline script in jsdom with a stubbed storage):
  - `the shell's colours are app.css's, in both themes` (`--bg`, `--ink`, `--muted` for `:root`, the dark media block and `[data-theme='dark']`)
  - `the theme script applies a stored choice`, `ignores anything else`, `survives storage that throws`
  - `the theme script reads the key theme.ts writes` (`dw-theme` in both)
  - `the shell says the app did not load after 10 s` (the `.failed` text is the copy, its delay is `10s`)
  - `main clears the shell before mounting` (source check: `replaceChildren` precedes `mount(`)
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/loading-shell.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/chrome.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(shell): paint a loading shell before the bundle arrives`.

### Task 3: Monaco with the JSON language only

XC-8's monaco trim (decision 6).

**Files:**
- Create: `ui/src/lib/monacoContribs.ts` (the editor contributions, one side-effect import each)
- Modify: `ui/src/lib/monaco.ts` (imports), `ui/src/lib/editor/JsonEditor.svelte` (a failed import), `ui/eslint.config.js` (`restricted()` gains the bare `monaco-editor` path, `allowTypeImports: true`, in every block)
- Test: `ui/src/lib/monacoContribs.test.ts` (new), `ui/scripts/lint-rules.test.ts`, `ui/src/lib/editor/JsonEditor.test.ts` (or the file holding its tests: `grep -rln JsonEditor src --include=*.test.ts`), `ui/e2e/json-view.spec.ts` (new)
- Docs: ARCHITECTURE *UI* table, new row **Monaco**: owner `ui/src/lib/monaco.ts`, `ui/src/lib/monacoContribs.ts`; rule "Monaco is the editor core, its contributions and the JSON language; nothing imports `monaco-editor` whole"; enforced by `monacoContribs.test.ts`, the ESLint rule (`lint-rules.test.ts`), `e2e/performance.spec.ts`

**Interfaces:**
- `monaco.ts`: `import * as monaco from 'monaco-editor/editor/editor.api.js'`, `import './monacoContribs'`, the existing `jsonDefaults` and worker imports; nothing else from the package.
- `monacoContribs.ts`: the `./editor/contrib/*`, `./editor/browser/*`, `./editor/common/standaloneStrings.js` and codicon CSS imports `node_modules/monaco-editor/esm/vs/index.js` makes, rewritten as `monaco-editor/...` specifiers, in its order; `export const CONTRIBS: readonly string[]` lists the same specifiers for the test.
- `JsonEditor`: the dynamic import's rejection sets `failed = e`; it renders the copy with `<Details summary="What failed">` holding `describeError(e).detail ?? describeError(e).title` (never the raw message).
- ESLint: importing `monaco-editor` (bare) is an error with the message `Import Monaco through src/lib/monaco.ts, which loads the JSON language only.`; a type-only import stays allowed.
- Capabilities of the whole-package editor the JSON view keeps, each pinned in `e2e/json-view.spec.ts`: (1) a schema diagnostic on an unknown key (`.squiggly-error` under `"stepz"`); (2) completion from the schema (Ctrl+Space lists `steps`); (3) hover text from the schema's description; (4) find (Ctrl+F opens `.find-widget`); (5) folding (a fold control on `steps`); (6) both themes (`dw-dark` after the header's dark toggle); (7) the prompt editor's JSON view validates against the prompt schema; (8) Ctrl+S saves the edit (the app's handler, unchanged).

- [ ] **Step 1: Write the failing tests.**
  - `monacoContribs.test.ts`: `every contribution the package lists is imported` (parses `node_modules/monaco-editor/esm/vs/index.js`; a missing or extra specifier is named), `no language definition, LSP client or other language feature is imported` (source scan of `monaco.ts` and `monacoContribs.ts`).
  - `lint-rules.test.ts`: `importing monaco-editor whole is refused`, `a type import is allowed`.
  - JsonEditor test: `a JSON editor that cannot load Monaco says so` (the import mocked to reject; the copy shows, the form view's switch still works).
  - `e2e/json-view.spec.ts`: the eight capabilities, one test each, against the built UI.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/monacoContribs.test.ts scripts/lint-rules.test.ts` and the JsonEditor test. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, `npm run lint`, the metrics gate, `npm run build`, then `DW_E2E_PYTHON=$PY npx playwright test e2e/json-view.spec.ts e2e/smoke.spec.ts e2e/journeys/editor-safety.spec.ts e2e/expanding-text.spec.ts e2e/step-model.spec.ts e2e/lora-suggestions.spec.ts`. Record the JSON view's bytes for Task 4. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): load Monaco's editor core and the JSON language only`.

### Task 4: Budgets the next agent cannot miss

The performance budget (decision 7). Needs Tasks 1-3.

**Files:**
- Create: `ui/e2e/performance.spec.ts`
- Docs: ARCHITECTURE *UI* table, new row **Performance budget**: owner `ui/e2e/performance.spec.ts` (`ENTRY_BUDGET_KB`, `FIRST_LOAD_BUDGET_KB`, `JSON_VIEW_BUDGET_KB`); rule "The entry bundle, the Overview's first load and the JSON view stay within their budgets on the wire; a budget is raised only with Don"; enforced by that spec

**Interfaces:**
- Constants at the top of the spec, each with today's figure in a comment: `ENTRY_BUDGET_KB = 300`, `FIRST_LOAD_BUDGET_KB = 420`, `JSON_VIEW_BUDGET_KB = 1250`. Bytes are `(await response.request().sizes()).responseBodySize` (encoded) summed over the responses of the load, in a fresh context.
- If a measurement is over its budget after Tasks 1-3, stop and report the figure to the lead; do not raise the constant.

- [ ] **Step 1: Write the spec:**
  - `the UI's files are compressed and cached` (the entry JS: `content-encoding: gzip`, `cache-control` holds `immutable`; `/`: `no-cache`)
  - `the entry bundle stays within budget` (the `index-*.js` and `index-*.css` the page loads)
  - `a first load of the Overview stays within budget` (`/#/ws/default/overview` until `networkidle`)
  - `the JSON view stays within budget and loads only the JSON language` (bytes from pressing **JSON** until `.monaco-editor` and the schema marker; no request whose path names `ts.worker`, `css.worker`, `html.worker`, `typescript`, `cssMode` or `htmlMode`)
  - `the loading shell paints before the bundle` (the entry JS delayed 2 s with `page.route`; `#boot` shows `Loading dw…` on a background equal to `--bg`; then the app's h1)
- [ ] **Step 2: Run** `cd ui && npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/performance.spec.ts`. Expected: PASS (it fails before Tasks 1-3).
- [ ] **Step 3: Measure** the Overview at 5 Mbps / 50 ms (CDP `Network.emulateNetworkConditions`) for first paint and idle, and the JSON view's bytes, for the phase summary. Not a test.
- [ ] **Step 4: Run** `npm test`, the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(perf): budgets for the entry bundle, the first load and the JSON view`.

### Task 5: The Jobs list reads a page, not the whole history

JB-17 (decision 1); folds #1609's placeholder title and `listWithPrompts` tests.

**Files:**
- Modify: `ui/src/lib/jobsListing.ts` (`listJobs`, `listWithPrompts`, `JOBS_PAGE`, `mergeListings`), `ui/src/lib/pages/JobsPage.svelte` (the `shown` state, the two copy lines, the filter `title`)
- Test: `ui/src/lib/jobsListing.test.ts`, `ui/src/lib/pages/JobsPage.test.ts`
- Docs: `docs/SERVER.md` `GET /api/jobs` row: "No parameters means every job, which is what the web UI polls" becomes "No parameters means every job; the web UI asks for every queued and running job and the newest finished ones with `limit`"

**Interfaces:**
- `export const JOBS_PAGE = 50`.
- `listJobs(ws, scoped, prompts, opts?: { status?: string; limit?: number })` passes them through to `api.listJobs`.
- `export function mergeListings(live: JobList, done: JobList): { jobs: JobSummary[]; total: number; prompts?: Record<string, string> }`: jobs by `created_at` (oldest first, ids unique, live wins), `total = live.jobs.length + done.total`, prompts merged.
- `listWithPrompts(ws, scoped, known, limit)`: reads `status: 'queued,running'` and `status: 'succeeded,failed,cancelled', limit` in parallel, each with `prompts` only on the first read or when it names a job not in `known` (each read asked again on its own); returns `{ list, prompts }` as today, `list` the merge.
- `JobsPage`: `let shown = $state(JOBS_PAGE)`, read by the resource's load before its first await so a change re-reads; **Show older** `shown += JOBS_PAGE`; **Search all** `shown = total`; a filter change does not reset `shown`. Both lines are `.hint` with the button `quiet`. The all-jobs view (Status page) uses the same component, so it pages too.

- [ ] **Step 1: Write the failing tests.**
  - `jobsListing.test.ts`: `the poll asks for live jobs and the newest finished page`, `a running job older than the page is listed`, `a job in both answers is listed once`, `total counts the live and every finished job`, `prompts are asked for only for jobs not yet known` (each read), `a new job on the second poll re-asks for prompts` (#1609).
  - `JobsPage.test.ts`: `the page says how many of how many it shows`, `Show older reads the next 50`, `a filter over a partial page says what it searched`, `Search all reads every job`, `a second poll keeps the rows and the prompts` (#1609), `the search field's title says what it searches`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/jobsListing.test.ts src/lib/pages/JobsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/chain-actions.spec.ts e2e/chain-segments.spec.ts e2e/security.spec.ts e2e/job-event-log.spec.ts e2e/requests.spec.ts e2e/restart-banner.spec.ts e2e/journeys/outage-and-token.spec.ts e2e/journeys/first-image.spec.ts e2e/journeys/find-and-reuse.spec.ts e2e/smoke.spec.ts` (every spec that matches `/api/jobs` or reads the Jobs page; re-grep before running). Expected: PASS.
- [ ] **Step 5: Commit** `ui(jobs): read the live jobs and the newest finished page, not the whole history`.

### Task 6: The Status page tells the truth about reaching this server

SH-3, SH-S4, SH-S5; folds #1601's doubled "Status".

**Files:**
- Modify: `ui/src/lib/serverinfo.ts` (`mcpAddCommand`, `bindSnippet`), `ui/src/lib/pages/ServerPage.svelte`
- Test: `ui/src/lib/serverinfo.test.ts`, `ui/src/lib/pages/ServerPage.test.ts`
- Docs: `docs/REMOTE.md` *Browser*: the Status page shows the addresses only on a network bind

**Interfaces:**
- `mcpAddCommand(address, port, path = '/mcp', withToken = true): string`: the header part only when `withToken`.
- `export const REMOTE_GUIDE = 'https://github.com/dkackman/diffusers-workflow/blob/master/docs/REMOTE.md'` and `export function bindSnippet(): string` (`DW_API_TOKEN=<token> python -m dw.serve --host 0.0.0.0`, using `TOKEN_PLACEHOLDER`).
- `ServerPage`: when `isLoopbackBind(info.bind_host)` the Addresses panel holds the copy's sentence and no table, and the Connect panel the copy's lines, snippet with `CopyButton` and the link; otherwise as today. `mcpAddCommand(..., info.auth_required)`; the note only when `auth_required`. The first panel's h2 is `Server`.

- [ ] **Step 1: Write the failing tests.** `serverinfo.test.ts`: `the mcp line carries the token header only when a token is required`, `the bind snippet uses python -m dw.serve and the placeholder`. `ServerPage.test.ts`: `a loopback bind offers no network URL`, `a wildcard bind lists its addresses`, `an unmounted MCP says python -m dw.serve --mcp`, `no token required, no token note`, `one heading says Status`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/serverinfo.test.ts src/lib/pages/ServerPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks plus `e2e/chrome.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(status): offer network URLs only on a network bind, and the token only when one is needed`.

### Task 7: Models and caches on a phone, and a warning only where it applies

SH-16, SH-17.

**Files:**
- Modify: `ui/src/lib/pages/ModelsPage.svelte` (the field, `repoIdHint`, the narrow columns), `ui/src/lib/CachesPanel.svelte`
- Test: `ui/src/lib/pages/ModelsPage.test.ts`, `ui/src/lib/CachesPanel.test.ts`, `ui/e2e/responsive.spec.ts` (a 390 px check that the size column is in view)

**Interfaces:**
- `export function repoIdHint(value: string): string | null` in the page's `<script module>` (or `src/lib/format.ts` if the page has no module script): null for an empty value or `org/name` (one `/`, both parts non-empty, no whitespace); the copy's hint otherwise.
- Below 640 px the repo table hides its `files` and `last used` columns (`th`/`td` classes `wideonly`), so repo and size fit at 390 px.
- `CachesPanel`: the low-disk row only when `cache.enabled && cache.low_disk`.

- [ ] **Step 1: Write the failing tests.** `ModelsPage.test.ts`: `the download field is labelled`, `a value that is not org/name says so and cannot be sent`, `a repo id can be sent`. `CachesPanel.test.ts`: `a disabled cache does not warn about disk`, `an enabled cache under its floor warns`. `responsive.spec.ts`: `the models table shows size at 390 px`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/ModelsPage.test.ts src/lib/CachesPanel.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Check `#/server/models` at 390 and 1440 px by eye (`node scripts/ux-capture.mjs`).
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks plus `e2e/responsive.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(models): a labelled download field, the size column on a phone, no disk warning for a disabled cache`.

### Task 8: The sidebar switches workspace from any state, and keeps the section

SH-8, SH-9, SH-19, SH-S2 (decisions 8, 9, 12).

**Files:**
- Create: `ui/src/lib/NewWorkspace.svelte` (the form, its state and the `creating` action, moved from `Sidebar`)
- Modify: `ui/src/lib/Sidebar.svelte`, `ui/src/lib/routes.ts` (`switchHref`)
- Test: `ui/src/lib/Sidebar.test.ts`, `ui/src/lib/NewWorkspace.test.ts` (new), `ui/src/lib/routes.test.ts`, `ui/e2e/workspaces.spec.ts`

**Interfaces:**
- `export function switchHref(name: string, view: RouteView): string` in `routes.ts`: a `ws` view keeps its section (`edit` → `workflows`, `compare` → `gallery`) and drops `rest`; any other view → `wsHref(name, 'overview')`.
- `Sidebar`: a shut workspace links `switchHref(name, route.view)`; collapsed, it renders as `.dot.solo` inside an `a` with the copy's title and label. With no `workspace.root`, the muted line replaces `new`.
- `NewWorkspace.svelte` props `{ onclose?: () => void }`, rendered by `Sidebar` in place of the old form. Capabilities of the old form, each pinned in `NewWorkspace.test.ts`: (1) it opens with focus in the field; (2) Enter creates once, however often pressed while busy; (3) a refused name is said beside the field (`workspaceNameError`, then the server's `describeError` title) and the form stays open; (4) Escape closes it; (5) success closes it and lands on the new workspace's overview (`createWorkspaceAndGo`); (6) `aria-busy` while creating. New: **Create** submits, **Cancel** closes, a `focusout` leaving the form closes it only when the field is empty.
- `Sidebar` ends at or below 560 lines.

- [ ] **Step 1: Write the failing tests.**
  - `routes.test.ts`: `switching keeps the section`, `switching drops a job id`, `the editor switches to workflows`, `compare switches to the gallery`, `a shared page switches to the overview`.
  - `Sidebar.test.ts`: `collapsed, every workspace can be chosen`, `a shut workspace links to the same section`, `no workspace root says why there is no new`.
  - `NewWorkspace.test.ts`: the six kept capabilities, `Create creates`, `Cancel closes`, `a click away closes an empty field`, `a click away keeps a typed name`.
  - `workspaces.spec.ts`: `switching workspace from the gallery lands on the other gallery` (a unique workspace name, deleted in `finally`).
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/routes.test.ts src/lib/Sidebar.test.ts src/lib/NewWorkspace.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Check the collapsed sidebar and the open form at 1440 px by eye.
- [ ] **Step 4: Run** `npm test`, the metrics gate, `wc -l src/lib/Sidebar.svelte` (≤ 560), the e2e checks plus `e2e/workspaces.spec.ts e2e/chrome.spec.ts e2e/journeys/consistency.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(shell): switch workspace from a collapsed sidebar, keep the section, and name the new-workspace buttons`.

### Task 9: The header says what its controls do

SH-S3, SH-S6; folds #1611's popovers, #1601's meter label, #1602's label test and #1600's Models breadcrumb.

**Files:**
- Modify: `ui/src/App.svelte` (the theme button, the help button, `/`, closing popovers on navigation, the meter label), `ui/src/lib/theme.ts` (`nextTheme`), `ui/src/lib/KeyboardHelp.svelte`, `ui/src/lib/pageFrame.svelte.ts` (`focusPageFilter`)
- Test: `ui/src/App.test.ts` (or the file holding the shell's tests: `grep -rln "render(App" src`), `ui/src/lib/theme.test.ts` (or `token.test.ts`'s neighbour; create if none), `ui/src/lib/pageFrame.test.ts` (create if none)

**Interfaces:**
- `export function nextTheme(t: Theme): Theme` in `theme.ts`; `cycleTheme` uses it. The label is the copy's.
- `export function focusPageFilter(): boolean` in `pageFrame.svelte.ts`: focuses the first `input` inside `main .pagehead .filter` (or a `main .pagehead input.filter`), true when it found one.
- `App.onKeydown`: `/` outside an editable field with no overlay open (`overlayOpen()`) calls `focusPageFilter()` and prevents the default only when it returned true.
- The route effect that closes the drawer also sets `statusOpen`, `tokenOpen` and `helpOpen` false.
- The meter's `stateword` is not rendered while the one-job `running` link shows; the button's `aria-label` is the copy's.
- #1600's breadcrumb: compare `#/server/models` with `#/ws/default/jobs` in both themes (`node scripts/ux-capture.mjs`); if Models differs, fix the rule that makes it; if not, say so in the commit body (`refs #1600`).

- [ ] **Step 1: Write the failing tests.** `theme.test.ts`: `the theme button says what a press does` (each of the three). Shell tests: `a navigation closes the status and token popovers`, `the help button opens the shortcuts`, `slash focuses the page's search`, `slash in a field types a slash`, `the meter's label starts with its state word`, `one running job is said once`, `no meter: N running and status labels` (#1602). `KeyboardHelp`: `the help lists slash`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/theme.test.ts src/lib/pageFrame.test.ts` and the shell test file. Expected: FAIL.
- [ ] **Step 3: Implement.** `App.svelte` ≤ 590 lines.
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks plus `e2e/chrome.spec.ts e2e/journeys/consistency.spec.ts e2e/journeys/outage-and-token.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(shell): the header says what its controls do; slash searches, question mark helps`.

### Task 10: A first-run card, and an Overview that fits a phone

SH-7, SH-4 (decision 10); folds #1611's overflow, #1606's long id and its jsdom check, #1610's same check, #1623's by-eye pass.

**Files:**
- Create: `ui/src/lib/firstRun.ts`, `ui/src/lib/FirstRunCard.svelte`
- Modify: `ui/src/lib/pages/OverviewPage.svelte` (the card; the workflows resource keeps the listing), `ui/src/lib/ui/PageHeader.svelte` (the secondary line wraps), the Overview's `.jid` (clipped with an ellipsis, the whole id in `title`)
- Test: `ui/src/lib/firstRun.test.ts` (new), `ui/src/lib/FirstRunCard.test.ts` (new), `ui/src/lib/pages/OverviewPage.test.ts` (the jsdom computed-style test goes), `ui/e2e/responsive.spec.ts` (ROUTES gain `#/ws/default/overview`, `#/server/status`, `#/ws/default/requests`, `#/ws/default/assets`), `tests/test_ui_twins.py`

**Interfaces:**
- `firstRun.ts`: `export const FIRST_RUN_WORKFLOW = 'templates/text-to-image'`; `export const FIRST_RUN_KEY = 'first-run-dismissed'`; `export type FirstRunLink = { href: string; label: 'Open text-to-image' | 'Browse examples' | 'Open Workflows' }`; `export function firstRunLink(listing: WorkflowList | null, ws: string): FirstRunLink`: the template's own page when listed (`writable !== false` → `wsHref(ws, 'workflows', ...)`, else `sharedHref('examples', ...)`, as `WorkflowsPage` decides); otherwise the Examples page when any listed workflow is read-only; otherwise the workspace's Workflows page. A null listing (failed) gives the Workflows page.
- `FirstRunCard.svelte` props `{ link: FirstRunLink }`; shown while `storageGet(FIRST_RUN_KEY, false)` is false (Question 1, written for the recommended answer); **Got it** sets a component flag and `storageSet(FIRST_RUN_KEY, true)` (a throwing store keeps the flag). Rendered by the Overview above the note editor, the copy's text and structure, `section.panel` with `aria-labelledby` its heading.
- `OverviewPage`: the `workflows` resource returns `{ names, listing }`; `featured` reads `names`; the card gets `firstRunLink(workflows.data?.listing ?? null, name)`, and shows while the listing loads (`firstRunLink(null, …)` until it lands).
- `PageHeader`'s secondary line: `overflow-wrap: anywhere` (in its own style block).
- `tests/test_ui_twins.py::test_the_first_run_workflow_exists`: reads `FIRST_RUN_WORKFLOW` from `ui/src/lib/firstRun.ts` and asserts `workflows/<it>.json` exists.

- [ ] **Step 1: Write the failing tests.**
  - `firstRun.test.ts`: `the card finds the template where the listing has it` (workspace and read-only cases), `no template: the examples`, `no examples: the workflows`, `a failed listing still orients`.
  - `FirstRunCard.test.ts`: `the card says what dw is and how to start` (the copy), `Got it hides it and remembers`, `a dismissal that cannot be stored lasts the session`, `a dismissed card stays hidden`.
  - `OverviewPage.test.ts`: `the overview shows the card until it is dismissed`, `a long job id keeps its row`.
  - `responsive.spec.ts`: the four routes at every width (the Overview's is SH-4).
  - `test_ui_twins.py`: `test_the_first_run_workflow_exists`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/firstRun.test.ts src/lib/FirstRunCard.test.ts src/lib/pages/OverviewPage.test.ts` and `$PY -m pytest tests/test_ui_twins.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** `OverviewPage` ≤ 560 lines. Check the Overview at 1440, 1300 and 390 px in both themes by eye (`node scripts/ux-capture.mjs`), the requests rows included (#1623).
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks plus `e2e/responsive.spec.ts e2e/journeys/find-and-reuse.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(overview): a first-run card that leads to a workflow that runs, and a page that fits a phone`.

### Task 11: No dead ends: empty states, the agent-notes box, the workspace delete

SH-6, JB-9; folds #1600's workspace delete.

**Files:**
- Modify: `ui/src/lib/pages/OverviewPage.svelte`, `ui/src/lib/pages/JobsPage.svelte`, `ui/src/lib/pages/GalleryPage.svelte`, `ui/src/lib/pages/AssetsPage.svelte` (their `emptyText` snippets), `ui/src/lib/request/LinkedNotes.svelte`, `ui/src/lib/request/noteState.ts` (`noteListShows`), `ui/src/lib/request/NoteList.svelte` (uses it), `ui/src/lib/workspaceActions.ts` (`askToDelete`), `ui/src/lib/destructive.ts` (`workspaceConfirm` takes `starred: number | null`)
- Test: `ui/src/lib/pages/OverviewPage.test.ts`, `JobsPage.test.ts`, `GalleryPage.test.ts`, `AssetsPage.test.ts`, `ui/src/lib/request/noteState.test.ts`, `ui/src/lib/pages/JobPage.notes.test.ts`, `ui/src/lib/workspaceActions.test.ts`, `ui/src/lib/destructive.test.ts`

**Interfaces:**
- Each empty state is the copy's, its links `a` elements to `wsHref(ws, 'workflows')`, `wsHref(ws, 'edit')`, `sharedHref('examples')` or `wsHref(ws, 'gallery')`.
- `export function noteListShows(request: LedgerRequest, item: string | null, shown: number): boolean` in `noteState.ts`: `shown > 0` or the list is editable (the rule `NoteList.editable` holds today, moved here). `NoteList` and `LinkedNotes` both use it; `LinkedNotes` draws its `.agentnotes panel` only when it is true.
- `askToDelete`: a failed gallery read reports nothing and passes `starred = null`; `workspaceConfirm(name, contents, null)` ends its message with the copy's sentence.

- [ ] **Step 1: Write the failing tests.** Each page: `an empty <page> says where to go` (the link's href). `noteState.test.ts`: `a dropped item with no notes shows no list`, `a dropped item keeps its notes to read`. `JobPage.notes.test.ts`: `a dropped item's job has no empty agent-notes box`. `workspaceActions.test.ts`: `a gallery that cannot be read still asks, without the count`. `destructive.test.ts`: `an unknown starred count is said`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/ src/lib/request/noteState.test.ts src/lib/workspaceActions.test.ts src/lib/destructive.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate (`banned_terms` 0: "Save as asset" is the glossary's term), the e2e checks plus `e2e/requests.spec.ts e2e/workspaces.spec.ts e2e/journeys/consistency.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(shell): every empty page says where to go; no empty agent-notes box; delete asks even without the starred count`.

### Task 12: The first-run journey, the rules for the next agent, and the phase gate

**Files:**
- Create: `ui/e2e/journeys/first-run.spec.ts`
- Modify: `ui/GLOSSARY.md`, `ui/CLAUDE.md` (rewrite, no new line), `docs/ARCHITECTURE.md`, `docs/stabilization/ui/baseline.json` (only if a count fell)

**Interfaces:**
- `GLOSSARY.md` *Before you write markup* gains: the first-run orientation (`FirstRunCard`, `firstRunLink` in `src/lib/firstRun.ts`); Monaco (`src/lib/monaco.ts` only; the ESLint rule refuses the whole package); a workspace switch (`switchHref`, `src/lib/routes.ts`); a list longer than a page (`JOBS_PAGE`, `mergeListings`, `src/lib/jobsListing.ts`: show a page and say how many of how many).
- `ui/CLAUDE.md`'s checks line names `e2e/performance.spec.ts` ("its budgets are raised only with Don"), paid for by trimming the line's parenthesised remark. Still 12 lines.
- ARCHITECTURE: *UI*'s *Routes and their names* row adds `switchHref`; *Keeping UI views fresh* adds "the Jobs list reads its live jobs and the newest finished page (`jobsListing.ts`)"; *Page frame* adds `focusPageFilter`; the enforcement columns of the rows Tasks 1, 3 and 4 added name `first-run.spec.ts` where it pins them.

- [ ] **Step 1: Write the journey** `first-run.spec.ts` (a fresh context, no storage; the workspace it creates is `e2e-first-<random>`, deleted through the API in `finally`):
  1. Delay the entry JS 1 s with `page.route`; open `/`: `Loading dw…` shows, then the Overview's h1. The first-run card's heading reads `New here? Make your first picture`.
  2. Press `Open text-to-image`: the URL ends `/workflows/templates/text-to-image` and the run check reads `Ready to run.`. Go Back.
  3. Press `Got it`: the card is gone; reload: still gone.
  4. Press the header's `Keyboard shortcuts`: the dialog lists `/`; close it. On Jobs, press `/`: the search field has focus.
  5. Open the sidebar's new-workspace form, type the name, press `Create`: the new workspace's Overview shows; its recent-outputs empty state's `Run a workflow` link goes to its Workflows page.
  6. Open its Gallery; press `default` in the sidebar: the URL is `#/ws/default/gallery`.
  7. Collapse the sidebar; press the new workspace's dot (`<name> workspace`): its Gallery opens. Expand it again.
  8. Open Status: the Connect panel reads `Only this machine can reach this server.` and no `http://<address>:<port>` other than loopback is shown; the MCP panel names `python -m dw.serve --mcp`.
  9. At 390 px, the Overview's document is no wider than the viewport.
  10. The entry JS was served with `content-encoding: gzip` and an `immutable` cache header.
- [ ] **Step 2: Run** `cd ui && npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/journeys/first-run.spec.ts`. Expected: PASS. A failure names the task that left a gap; fix it there.
- [ ] **Step 3: Write the rules** as in Interfaces.
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, `DW_E2E_PYTHON=$PY npx playwright test` (the whole suite, `performance.spec.ts` included), the metrics gate, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (only `complex_functions: 11 -> 13`), `$PY scripts/arch_metrics.py` (`modules 281`, `claude_md_lines 129`), `wc -l dw/server/app.py dw/server/api_models.py dw/server/job_history.py dw/server/jobs.py dw/server/routes/jobs.py` (≤ 292, 1105, 1000, 1139, 1033), `wc -l ui/src/lib/api.ts ui/src/lib/Sidebar.svelte ui/src/App.svelte ui/src/lib/pages/OverviewPage.svelte` (≤ 598, 560, 590, 560), and `$PY -m pytest tests/test_server_delivery.py tests/test_ui_twins.py tests/test_security_web_headers.py tests/test_api_contract.py tests/test_mcp_server.py -q`. Expected: PASS; no page over 600.
- [ ] **Step 5: Commit** `ui(shell): the first-run journey, and the rules for the next agent`.

---

## Phase exit

- Every response body decodes to what it was; `/mcp`, event streams, media, zip and ranged answers are never compressed; MCP's replies are byte for byte today's.
- The entry bundle, the Overview's first load and the JSON view are within `performance.spec.ts`'s budgets, which the full e2e run holds.
- Every metric that was 0 is 0; nothing is above `$SCRATCH/ui-p7.json`; `modules` 281, `claude_md_lines` 129, the size caps hold, no page over 600.
- `first-run.spec.ts` holds the phase.

## Open items no phase owns

Phase 7 is the last phase of the program, so what is left alone above has no phase to go to. Each stays in its GitHub issue for triage outside the program: #1600's remaining items (the issue carries the release-blocker label, so its open entries need triage before a release), #1601's remaining items (the Schema h1 is Don's call), #1602's metric, lint, missing-test and journey items, #1603, #1605, #1606's remaining items, #1607 (the `job_history.py` split needs a new module and so Don's approval), #1608, #1609's Compare, takes and day-unit items, #1610's remaining items, #1612, and #1623's remaining items.
