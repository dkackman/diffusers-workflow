# UI quality phase 6: requests and agent collaboration. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a person who asked an agent for a multi-shot piece opens the request and can tell at a glance, on the list and on the page, what needs them and why; sees why a take failed without leaving the page; can leave a note without silently stopping a run they meant to approve; picks a take with a button that looks like a decision; cannot edit a closed request by accident; and can see what the agent changed, both while watching and since they last looked.

**Architecture:** Two opt-in server fields, the spec's building blocks for this phase: a failed or cancelled take's `reason` on a request read (`reasons=true`), and a list row's `needs` counts (`needs=true`). Without the parameter every reply is key for key what it is today, so MCP sees nothing new. On the UI side, what needs the person stays derived in one place, `planState` (`ui/src/lib/request/itemState.ts`), and the server's `needs` mirrors it, both pinned by one shared fixture (`tests/fixtures/request_needs_cases.json`). The header's needs line and the list row are the same sentence. The request run dialog adopts phase 5's `verdictOf`. What the agent changed is diffed in the UI (`planDiff`), live and against a per-browser "last seen" snapshot. `RequestItem.svelte` is split so its open half is its own component.

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config, Node 24 type stripping for `scripts/arch-metrics.mjs`. Server: FastAPI 0.143.0 / Pydantic 2.14.1 for the OpenAPI dump, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`. This plan covers:
- the phase-6 row of *Phases*: RQ-1..4, RQ-7..11, RQ-13..21, the building block "the server's take error and summary fields", and the journey gate `review-a-request.spec.ts`;
- decision 2 (*The hold and notes*, approved): keep the server's rule (a note on the item or on the request redirects that item's hold; a note on another item does not), warn before saving, keep **Run now…** in the banner;
- decision 5 (*Requests made by hand*, approved): requests are written by an agent, and no "New request" UI is built;
- the ★ vocabulary row **Accept**: accepting also stars the file (`ui/GLOSSARY.md` marks it phase 6);
- the phase-5 carry-over: the request run dialog adopts `verdictOf` (RQ-10).

The requests reviewer's full report (finding text, repro, screenshots under the session scratchpad's `ux-requests/shots/`) numbers its findings 1..21 = RQ-1..21. The audit (`docs/superpowers/specs/2026-11-22-ui-audit.md`, Tier 1 #21, Tier 2 §A *Requests*, Tier 3 *Requests*) has the short text.

**Out of scope:**
- RQ-5, RQ-6, RQ-12 (done in phases 3 and 4).
- A per-revision change record stored by the server (Question 2); a stored "resolved" state on notes (decision 7 below).
- Menu roles and arrow-key navigation in `ItemMenu` (a11y beyond names and focus; RQ-15 here is the clipping only).
- Every #1600-#1612 item not listed under *Folded in*.

**Folded in** (each in code the task already rewrites):
1. #1600 - The held chip reads "held · 02:05" (Task 7). Its pair, the host-RAM line naming its example, needs a plan field and lives in `plan.ts`: left alone.
2. #1600 - Undo says nothing when the agent already reopened the request (`RequestMenu.svelte`, Task 8).
3. #1600 - ItemNote's new-note box resends the already-sent part (Task 13).
4. #1600 - NoteDraft keeps "You have an unsaved draft." after the saved note already matches, and N1 (a "changed elsewhere" notice over the user's own save), one fix (Task 13).
5. #1600 - `formatMinutes` gives "about 80 min" where the glossary promises "about 1 h 20 min" (Task 4; every request figure uses it).
6. #1601 - The stale `~` comment in `ui/src/lib/request/runPlan.ts` (Task 11).
7. #1602 - ItemNote's `danger: true` on its Cancel question is unpinned (Task 13).
8. #1602 - `NoteList`'s double guard leaves the action's drop branch dead (Task 13).
9. #1609 - A deep link `open=` to a take whose row is closed shows nothing (Task 6: the row holding it opens).
10. #1609 - A job linked to two request items shares one viewer key across their strips (Task 6).
11. #1609 - Tests for the take viewer's history wiring and the "Accepted" state; the "Accept never stars" test becomes "Accept stars" (Task 6).
12. #1611 - The Overview Requests panel hides the first request's title at 390 px (Task 4: it renders the same attention line).
13. #1612 - "A job would be refused: 0 issues to fix." (Task 11: `verdictOf` falls back to `A job would be refused.`).
14. `prefix_literals` 3 → 0: the three are request code (`itemState.ts:308-309`, `fixtures.ts:65`), and Task 2 rewrites `revisedLine`'s helper to use `PROMPT` from `references.ts`.

## Measured at the start (develop fd57a4e8)

**UI metrics** (`cd ui && node scripts/arch-metrics.mjs`): `files_over_size_ceiling 0, complex_functions 13, long_functions 0, import_cycles 0, modules_in_import_cycles 0, prefix_literals 3, a11y_suppressions 1, raw_error_toasts 0, empty_outside_loadstate 0, unwatched_reads 0, window_dialogs 0, unguarded_mutations 0, banned_terms 0, raw_media 1, duplicate_blocks 16`. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only `complex_functions: 11 -> 13` and `prefix_literals: 0 -> 3`, both pre-existing.
- `prefix_literals` 3: `request/fixtures.ts:65`, `request/itemState.ts:308`, `:309`.
- `duplicate_blocks` 16; in files this phase rewrites: `RequestPage.svelte:233`, `NoteList.svelte:177`, `PlanChanges.svelte:85`, `RequestHeader.svelte:174`. A task that removes one lowers the count and `baseline.json`.

**Python** (`/Users/don/src/dkackman/diffusers-workflow/venv/bin/python scripts/arch_metrics.py`): `modules 281`, `claude_md_lines 129`, `modules_over_size_ceiling 5`.

| File | Lines | Cap this phase |
|---|---|---|
| `dw/server/api_models.py` | 1105 | 1105 (untouched) |
| `dw/server/job_history.py` | 1000 | 1000 (Task 1 adds a column to one SELECT on its existing lines) |
| `dw/server/routes/jobs.py` | 1033 | 1041 (untouched) |
| `dw/server/routes/library.py` | 1000 | 1000 (untouched) |
| `dw/server/routes/requests.py` | 388 | 388 |
| `dw/server/api_models_library.py` | 765 | 765 |
| `dw/ledger.py` | 1146 | 1146 (over the ceiling already; Task 3's parameter rides on existing lines) |
| `dw/server/jobs.py` | 1140 | 1140 (over the ceiling already) |
| `dw/ledger_views.py` | 510 | under the 1,000 warning |

**UI sizes** (limit 600): `app.css` 600, `api.ts` 598, `RequestItem` 550, `OverviewPage` 514, `RequestPage` 393, `PlanItems` 391, `RunDialog` 378, `itemState.ts` 374, `ItemNote` 368, `RequestHeader` 270, `AttemptStrip` 277, `NoteList` 256, `RequestsPage` 221, `PlanChanges` 157, `reviewText.ts` 110. `app.css` is full: a new global rule goes in `src/patterns.css` (511).

**Findings checked against the code:**

| ID | State today | Where |
|---|---|---|
| RQ-1 | open: an item's own notes and the request's notes redirect its hold (server rule, kept); no warning while writing; the redirected banner says "the agent will revise" whether or not it is connected; `runNowKeys` offers nothing on `redirected` | `request/reviewText.ts:60-92`, `request/ItemNote.svelte`, `dw/ledger_views.py:_review` |
| RQ-2 | open: `needsOf` counts every `failed` display, runnable or not | `request/itemState.ts:121-146` |
| RQ-3 | open: `LedgerAttempt` has no error; `request_jobs` does not select `error` | `dw/server/api_models_library.py:406`, `dw/server/job_history.py:593`, `request/AttemptStrip.svelte` |
| RQ-4 | open: one line, first match wins; `LedgerSummary` has no hold, choose or notes counts; idle wording is engine speak | `request/attention.ts:34-87`, `dw/ledger_views.py:summary_view` |
| RQ-7 | open: `itemMenu` checks only `dropped`; grips and note boxes stay on a closed request | `request/menuEntries.ts:73-87`, `request/PlanItems.svelte:278`, `request/NoteList.svelte:61-63` |
| RQ-8 | partly done: the changed-row mark and rename announcements (phase 2). Argument and workflow changes are not announced | `request/announce.ts:27-50` |
| RQ-9 | open: Plan changes shows `why` only | `request/PlanChanges.svelte`, `request/planChanges.ts` |
| RQ-10 | partly done: a submit refusal and an unchecked validation go behind `Details` (phase 2). An invalid validation's first engine message is the row's reason; a task-only workflow's quote is "no figure - task-only workflow (compose_text): …" in warn tone; the button reads "Run 1" | `request/runQueue.svelte.ts:155-157`, `request/RunDialog.svelte`, `plan.ts:describeEstimate` |
| RQ-11 | open: failed items only under *Will not run*; the lead says "N ready to run" counting revised items the needs line calls revised | `request/runPlan.ts:readyPlan`, `request/standing.ts:lead` |
| RQ-13 | open: an open row keeps an empty 96 px proof cell; the title squeezes at 1300 px | `request/RequestItem.svelte:226-240,355-360` |
| RQ-14 | open: every open row repeats "Note for the agent", "Add a note for the agent" and "Runs X with no arguments" | `request/NoteList.svelte:176-179`, `request/RequestItem.svelte:309` |
| RQ-15 | open: the widest entry runs into the popover's edge | `request/ItemMenu.svelte` `.entries`, `app.css:574` |
| RQ-16 | partly done: clearing the choice already has an Undo toast (phase 3). The button is `quiet` and reads as disabled | `request/AttemptStrip.svelte:145-158` |
| RQ-17 | open: revised rows start collapsed; the needs line is plain text | `request/itemState.ts:220`, `request/RequestHeader.svelte:118` |
| RQ-18 | open: a reorder above a dependency saves without comment | `request/PlanItems.svelte:203-218` |
| RQ-19 | done: the empty list says requests come from an agent (decision 5); nothing to build | `pages/RequestsPage.svelte:65-66` |
| RQ-20 | open: an answered note stays as an open note forever | `request/noteState.ts:noteLine`, `request/ItemNote.svelte` |
| RQ-21 | partly done: a text result is an `OutputTile` excerpt (phase 4). The filter chip shows the id; the `open` tag and the `ready` chip look alike | `gallery/RequestFilter.svelte:68`, `request/StateTag.svelte`, `.chip.ready` |

**Found already done:** RQ-19 (the empty state, decision 5); RQ-8's changed-row mark and rename announcement (phase 2); RQ-16's silent clear (phase 3's undo toast); RQ-21's text cover (phase 4's excerpt tile); RQ-10's submit refusals behind `Details` (phase 2).

**Decisions the spec left open** (taken here; questions are separate):
1. **Held replaces ready.** An item with a review hold counts as `held` ("waiting for your OK"), not `ready` or `revised`, so one item is never two needs. A failed item counts only when it can be run again from the page (dependencies done, a stored workflow); a cancelled one, as today, does not count.
2. **The hold warning is a line in the note box**, not a confirm: decision 2 says "warn before saving", and a modal on every note would punish the common case. The request's box counts every live item that is `holding` or `due`, because a request note redirects them all; an item's box counts only that item's hold, because the server redirects an item's hold only on its own notes or the request's (`ledger_views._hold`). (Corrected in the final fix wave: this first read "any note redirects".)
3. **Run now… stays offered** on a redirected hold, connected or not (decision 2).
4. **A closed request is read-only on the page**: no grips, moves, drop, Accept, retake or note boxes; existing notes stay readable. The server still accepts writes; the agent never hears a note on a closed request (`note_waiting` needs `open`).
5. **Failed items join the ready dialog unticked**, under their own heading, so retrying three failed shots is one dialog and never an accident.
6. **The run button counts jobs**: `Run <n> <job|jobs>`; the header button is `Run <n>…` and the dialog `Run what can run now`, so "ready" means only the `ready` display.
7. **A note's lifecycle stays UI-only**: an answered note shows the agent's reply (the `why` of its first write after the note) and a **Clear note** button that deletes it without asking, as an answered note's delete already does. No stored `resolved` field (that would be MCP-visible ledger state).
8. **`RequestItem` splits**: the open half becomes `ItemMore.svelte`, so the row file falls well under 600 and Tasks 5, 9 and 10 have room.

## Global constraints

- **MCP behaviour is unchanged.** The two server changes are additive and opt-in, and each task says why MCP output is unchanged:
  - Task 1: `reason` reaches an attempt only when `GET`/`PUT /api/workspaces/{name}/requests/{id}` is asked `reasons=true`. `dw_mcp/ledger.py`'s `get_request` and `set_request` send no such parameter, so their replies are key for key today's.
  - Task 3: `needs` reaches a list row only when `GET /api/workspaces/{name}/requests` is asked `needs=true`. `dw_mcp`'s list reads and the notes channel's `GET /api/requests` never send it, and `/api/requests` takes no such parameter.
  - OpenAPI is regenerated only with the pinned overlay: `PYTHONPATH=/private/tmp/claude-501/-Users-don-src-dkackman-diffusers-workflow/c38aa355-7a1b-44f4-aee3-aaaef6ebc96d/scratchpad/p3/openapi-overlay $PY scripts/dump_openapi.py && cd ui && npm run gen:api`, where `PY=/Users/don/src/dkackman/diffusers-workflow/venv/bin/python` (the worktree has no venv). If the overlay is gone, rebuild it: `$PY -m pip install --target <dir> -c constraints-openapi.txt fastapi pydantic`. Never hand-edit `ui/src/lib/generated/`.
- **Size limits.**
  - Python `modules` stays 281. No task adds a module (a test file or a JSON fixture is not one). A task that finds it needs a module stops and asks Don.
  - `claude_md_lines` stays 129: `ui/CLAUDE.md` stays 12 lines, rewritten, never grown.
  - The caps in the table above hold. A capped file a task adds to pays for it in the same file (a docstring or comment said in fewer lines; never by moving code out). If `api_models_library.py` or `routes/requests.py` cannot pay without losing a field description, stop and ask Don whether to raise that cap.
  - `ui/src/lib/api.ts` ≤ 598. No page over 600 lines; `app.css` is full, new global rules go in `src/patterns.css`.
  - No function over complexity 10 or 60 lines, in TS or Python.
- **The UI metrics gate.** Before Task 1: `cd ui && node scripts/arch-metrics.mjs > "$SCRATCH/ui-p6.json"` (any scratch folder). Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-p6.json"`, and it prints nothing. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only the pre-existing `complex_functions: 11 -> 13` (and `prefix_literals: 0 -> 3` until Task 2 clears it). Every count that is 0 stays 0. A task that lowers a count lowers that key in `baseline.json` in the same commit.
- **Checks every task runs.** UI, from `ui/`: `npm run check`, `npm run lint`, the full `npm test` (vitest) before committing, and after any copy change `npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/smoke.spec.ts e2e/requests.spec.ts` (the smoke test at `:36`, "editor opens a workflow with introspected arguments", is the known flake #1603). Python: `$PY -m pytest <files> -q` with `DW_STRICT_RESPONSES=1`, and `$PY scripts/arch_metrics.py`. e2e always serves the built UI.
- **Staging and commits.** Stage files by name; never `git add -A ui` (Don edits `ui/package.json`), never add or force-add `.superpowers/`, never `git stash`. Headlines use a scope (`server(requests): …`, `ui(requests): …`, `docs(…)`), never `fix: #N`; to name an issue write `refs #N` in the body. Each commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Security.** Server code validates with `dw/security.py` against a real base; no `eval`, `exec` or `shell=True`. Nothing in this phase adds a path or URL.
- WCAG contrast is guidance, never a check. Names, focus, `aria-pressed`/`aria-expanded` and a dialog's label are in scope.
- **When a component replaces or absorbs another**, list each capability the old one had and pin each with a test (Task 9 lists `RequestItem`'s). **Values sent to the engine**: the run dialog sends each item's stored `arguments` exactly as the ledger holds them; this phase adds no coercion.
- **Docs.** `docs/SERVER.md`'s *Requests* paragraphs and the endpoint list are edited by the task that changes the behaviour; ARCHITECTURE's *Requests ledger* row by Tasks 1 and 3, *Requests in the UI* by the task that changes the rule it states.
- **Copy fixed by this plan.** Use it exactly; `<…>` is filled in, `<a|b>` picks by count.
  - **Needs** (`needsLine`, `needsParts`, header and list): `Needs you: <parts>.`, parts in this order, joined by `, `: `<n> waiting for your OK`, `<n> ready to run`, `<n> failed`, `<n> revised since <its take|their takes>`, `<n> <take|takes> to choose`, `<n> accepted <take|takes> missing`. In the header each part is a button titled `Go to <title of the first such item>`.
  - **Standing lead**: with something to run and nothing in flight, `Nothing is running.` (the count moves to the needs line); the done tail is unchanged.
  - **A failed row that cannot run again yet**: `Failed. It can run again once <titles> <is|are> done.`
  - **Takes line**: failed or cancelled `<n> <take|takes>, none succeeded` (no longer "attempts").
  - **A take's reason**: under a failed or cancelled take, its `reason`, one line, the whole in `title`; the row's line `Take <n> failed: <reason>` or `Take <n> was cancelled: <reason>` (the newest such take).
  - **List row** (`attention`): with `needs`, the needs sentence above, tone `needs`; otherwise `running`, `queued`, `<n> planned, nothing running`, `all done, still open`, `nothing planned yet`; the figure `<n> <note|notes> waiting for the agent` when `notes_waiting` > 0.
  - **Request filter chip**: `request: <title>`, with an item `request: <title> › <item key>`; titled `<id> - open this request`; the id alone while the title is unknown.
  - **Hold**: chip holding `waiting for your OK · <MM:SS>`, redirected `on hold: note left`, due `review time up`. Banner redirected and connected: `You left a note after these were held, so they won't start on their own. The agent will read it and may revise them; run them here if they're right as they are.` Redirected and not connected: `You left a note after these were held, so they won't start on their own, and the agent isn't connected. Run them here, or wait for the agent.` Live news for redirected: `The held items won't start on their own: you left a note.` Note-box warning, while a hold the box's note would redirect is `holding` or `due` (the request's box: any live item; an item's box: that item): `Saving this note stops <1 item|<n> items> waiting for your OK from starting on their own.`
  - **Accept**: the button `Accept` (default button style) / `Accepted` (pressed, with the check); the toast `Accepted take <n> for <title> and starred it`, with Undo; a star that fails: `notify.failure('star the accepted take', e)`.
  - **Closed request**: above the plan `This request is <done|abandoned>. Reopen it to change the plan or leave a note.`, followed by a **Reopen** button (the same action as the header menu's **Reopen**, which stays); after Reopen the toast `Reopened "<title>"`; Undo over an already reopened request `"<title>" is already open`.
  - **Row chrome**: an item with no notes shows only the button `Add a note for the agent` (no heading); what it runs is one fold, summary `Runs <workflow>` plus ` with <n> <argument|arguments>` when it has any, body the arguments or `No arguments.`
  - **Reorder**: under a row placed above an item it depends on, `Runs after <titles> whatever its place.`; the live region `<title> moved to position <p> of <t>. It still runs after <titles>.`
  - **Run dialog**: heading `Run what can run now` (single: unchanged `Run <title>` / `Run another take of <title>`); header button `Run <n>…`, titled `Price and queue what can run now`; the spend button `Run <n> <job|jobs>` (busy `Queuing…`); a quote with no figure `No time estimate for this workflow.` with `<Details summary="Why there's no estimate">` holding the engine reason; a refused validation `verdictOf`'s text, then `Ask the agent to fix the item's workflow or arguments.`, the engine messages behind `<Details summary="Engine detail">`; a gated or missing model `verdictOf`'s text and remedy in warn tone (the row stays tickable); the failed group's heading `Failed before. Tick one to run it again.`; `verdictOf` with no counted issue `A job would be refused.`
  - **What the agent changed** (`planDiff` lines): `Added <title>`, `Dropped <title>`, `Restored <title>`, `Renamed <old> to <new>`, `Changed <title>: <its workflow|its arguments (<names>)|what it waits on>`; the panel `Since you last looked (r<a> to r<b>)` with the button **Dismiss**; announcements `The agent changed the arguments of <title>.` and `The agent changed the workflow of <title>.`
  - **Notes**: under an answered note `The agent answered in r<n>: <why>`; its button **Clear note**, titled `The agent has answered; remove it from the list`.

## Review focus

1. **A take whose error is long, multi-line or a restart tombstone** - a traceback, a 2 KB CUDA message, `server restart: …`. The reason is the first line, clipped (`row_reason_fields`, `REASON_MAX`), never the traceback; the tile clamps it to one line. Pinned in Task 1 (`a traceback gives its first line, clipped`, `a restart tombstone gives its reason`) and Task 5 (`a long reason is one line with the whole in its title`).
2. **A server older than this phase** (a remote GPU box not yet updated) answers without `needs` or `reason`. The list falls back to today's status line and ordering; a take without a reason shows none. Pinned in Task 4 (`a summary without needs keeps the status line`) and Task 5 (`a failed take with no reason shows no reason line`).
3. **Dependencies that point nowhere** - a failed item whose dependency was dropped or is missing, a reorder above a missing key. It is not runnable, not counted, and its line names the dependency's state. Pinned in Task 2's fixture case `failed-after-dropped` and Task 10 (`a dependency the plan no longer holds gives no hint`).
4. **An agent write landing mid-action** - the hold flips to redirected while a note draft is open, the run dialog is open when an item becomes held, a quiet read arrives during a drag. The warning and banner follow the latest read, the draft is kept, the dialog reads holds off the plan as it is now. Pinned in Task 7 (`the warning follows the latest read`) and Task 12 (`a diff is taken against the read the page held, not the snapshot`).
5. **The "last seen" snapshot going wrong** - storage that throws, a snapshot from a request deleted and recreated under the same id (its revision above the current one), a snapshot of another workspace's request. No diff is shown and nothing breaks; snapshots are keyed by workspace and id. Pinned in Task 12 (`storage that throws shows no diff`, `a stored revision above the current one is discarded`, `snapshots are per workspace`).

---

### Task 1: A failed take's reason, when asked (server)

RQ-3's server half. Additive and opt-in.

**Files:**
- Modify: `dw/server/job_history.py` (`request_jobs` selects `error`, on its existing lines), `dw/server/jobs.py` (`request_jobs`' live dict carries `error`; `_linked` adds `reason`), `dw/ledger_views.py` (`item_view`'s attempt), `dw/server/routes/requests.py` (`reasons` on `get_request` and `put_request`), `dw/server/api_models_library.py` (`LedgerAttempt.reason`)
- Modify: `ui/src/lib/generated/*` (regenerated only), `ui/src/lib/api.ts` (`getRequest` and `putRequest` send `reasons=true`)
- Test: `tests/test_ledger_take_reason.py` (new test file); `tests/test_api_contract.py`, `tests/test_ledger*.py` (unchanged; must pass)
- Docs: `docs/SERVER.md` (the parameter, in the endpoint list); ARCHITECTURE *Requests ledger* row: "a failed or cancelled take's `reason` (the job's `row_reason_fields` reason) is on an attempt only when the read asked `reasons=true`"

**Interfaces:**
- `jobs.py` `_linked(...)` adds `"reason": row_reason_fields(row["status"], row.get("error")).get("reason")` (None when none). The live dict passed to it carries `"error": job.error`. jobs.py stays ≤ 1140.
- `ledger_views.item_view`'s attempt dict gains `**({"reason": job["reason"]} if job.get("reason") else {})`.
- `routes/requests.py`: `_jobs_for(request, name, reasons=False)`; without `reasons` it hands each job dict with `reason` set to None, so no attempt carries one. `get_request(..., reasons: bool = False)` and `put_request(..., reasons: bool = False)` pass it. `list_requests` and `list_open_requests` are unchanged.
- `LedgerAttempt.reason: str = sometimes("Only with reasons=true: why a failed or cancelled job ended, the first line of its error.")`.
- `api.ts`: the two paths gain `?reasons=true`; the file stays ≤ 598.

- [ ] **Step 1: Write the failing tests** in `tests/test_ledger_take_reason.py` (fixtures as in `tests/test_ledger_page_actions.py`):
  - `test_a_failed_take_carries_its_reason_when_asked` (`error="CUDA out of memory.\nTraceback…"` → `reason == "CUDA out of memory."`)
  - `test_a_traceback_gives_its_first_line_clipped` (a 500-character first line ends with `…` at `REASON_MAX`)
  - `test_a_restart_tombstone_gives_its_reason` (cancelled, `server restart: …`)
  - `test_a_succeeded_take_has_no_reason`
  - `test_a_read_without_reasons_has_no_reason` (GET, the key absent from every attempt)
  - `test_a_put_without_reasons_has_no_reason` and `test_a_put_with_reasons_has_them`
  - `test_a_live_failed_job_gives_its_reason` (still in `JobManager.jobs`)
- [ ] **Step 2: Run** `$PY -m pytest tests/test_ledger_take_reason.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, regenerate OpenAPI (Global constraints), `cd ui && npm run gen:api`.
- [ ] **Step 4: Run** the file, `tests/test_api_contract.py`, `tests/test_ledger*.py`, `tests/test_mcp_server.py`, then `wc -l dw/server/job_history.py dw/server/jobs.py dw/server/routes/requests.py dw/server/api_models_library.py ui/src/lib/api.ts` (≤ 1000, 1140, 388, 765, 598), `$PY scripts/arch_metrics.py` (`modules 281`), `cd ui && npm run check`. Expected: PASS.
- [ ] **Step 5: Commit** `server(requests): a failed take says why, when the page asks`.

### Task 2: "Needs you" counts only what the person can act on

RQ-2, RQ-17's needs-line half, RQ-11's vocabulary half, and the rule Task 3 mirrors. Clears `prefix_literals`.

**Files:**
- Create: `tests/fixtures/request_needs_cases.json`, `ui/src/lib/request/needs.twin.test.ts`
- Modify: `ui/src/lib/request/itemState.ts` (`NeedsReason`, `needsOf`, `opensByDefault`, `takesLine`, `moved`), `ui/src/lib/request/standing.ts` (`NEEDS`, `needsParts`, `lead`), `ui/src/lib/request/RequestHeader.svelte`, `ui/src/lib/request/fixtures.ts`
- Test: `ui/src/lib/request/itemState.test.ts`, `ui/src/lib/request/standing.test.ts`, `ui/src/lib/request/RequestHeader.test.ts`
- Docs: ARCHITECTURE *Requests in the UI*: "which items need the user" now says "held replaces ready; a failed item only when it can run again"

**Interfaces:**
- `NeedsReason` gains `'held'`; `REASONS` order is `held, ready, failed, revised, choose, accepted_missing` (the copy's order).
- `needsOf(item, display, open, runnable, held)`: on an open live item, `held` when it has a review, else `ready` for the ready display and `revised` for the revised one; `failed` only when `runnable`; `choose` and `accepted_missing` as today.
- `opensByDefault` also opens a `revised` row.
- `export type NeedsPart = { reason: NeedsReason; text: string; key: string }` and `export function needsParts(plan: PlanState): NeedsPart[]` in `standing.ts` (`key` the first live item, in plan order, with that reason). `needsLine(needs)` keeps its signature (the list uses it) and the copy's order.
- `RequestHeader` renders `Needs you: ` then each part as `button.bare` calling `onselect(part.key)`, joined by `, `, then `.`.
- `standing.lead` drops the count (`Nothing is running.`); the failed-but-blocked line is `blockedLine(state: ItemState): string | null` in `itemState.ts`, drawn by `RequestItem` where `waits on` is drawn today.
- `moved` uses `PROMPT` from `../references`; `fixtures.ts` builds `` `${PROMPT}board-01` ``.
- The fixture: `{ "cases": [{ "name": string, "request": <a read-shape request: state, agent_wrote_at, notes, items with key, title, status, ready, depends_on, workflow, dropped, accepted, accepted_missing?, attempts[{job, status}], review?>, "needs": {held, ready, failed, revised, choose, accepted_missing}, "notes_waiting": number }] }`, with at least the cases `held-ready`, `failed-runnable`, `failed-blocked`, `failed-after-dropped`, `cancelled`, `revised-in-flight`, `two-takes-unaccepted`, `closed-request`, `note-waiting`. `needs.twin.test.ts` fills each item through `fixtures.ts`' helpers and checks `planState(request).needs` and `notesWaiting(request)`.

- [ ] **Step 1: Write the failing tests.**
  - `itemState.test.ts`: `a failed item waiting on a revised one is not counted`, `a held ready item counts as held, not ready`, `a revised row opens by default`, `a failed row says what it waits on`, `takes, not attempts, in the takes line`.
  - `standing.test.ts`: `the lead leaves the count to the needs line`, `needs parts name the first item of each reason`, `the parts come in the copy's order`.
  - `RequestHeader.test.ts`: `pressing a needs part goes to its first item`.
  - `needs.twin.test.ts`: `every shared case gives its needs`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate (`prefix_literals` 0), and the e2e checks. Expected: PASS; the baseline check prints only `complex_functions: 11 -> 13`.
- [ ] **Step 5: Commit** `ui(requests): needs you counts only what you can act on, and each part goes to its item`.

### Task 3: A list row's needs, when asked (server)

RQ-4's server half. Additive and opt-in; mirrors Task 2's rule.

**Files:**
- Modify: `dw/ledger_views.py` (`needs_counts`, `note_waiting`, `summary_view(..., needs=False)`), `dw/ledger.py` (`listing` and `list_requests` pass `needs`, on their existing lines), `dw/server/routes/requests.py` (`list_requests(..., needs: bool = False)`), `dw/server/api_models_library.py` (`LedgerSummary.needs`)
- Modify: `ui/src/lib/generated/*` (regenerated only), `ui/src/lib/api.ts` (`listRequests` sends `needs=true`)
- Test: `tests/test_ledger_needs.py` (new test file), `tests/test_ui_twins.py` (the note-waiting case set also checks `ledger_views.note_waiting`)
- Docs: `docs/SERVER.md` endpoint list; ARCHITECTURE *Requests ledger* row: "a list row's `needs` (by reason, and `notes_waiting`) only when asked `needs=true`, the UI's `planState` rule, pinned by `tests/fixtures/request_needs_cases.json`"

**Interfaces:**
- `ledger_views.needs_counts(request_view: dict) -> dict[str, int]`: over a read-shape request (Task 2's fixture shape), the keys `held, ready, failed, revised, choose, accepted_missing, notes_waiting`, every key present. The rule is Task 2's: open request, live items; `held` when a review applies, else `ready` (status planned and `ready`) and `revised` (status revised with no queued or running attempt); `failed` when status failed, not every attempt cancelled, every dependency done and a stored workflow; `choose` when not accepted and two or more succeeded; `accepted_missing`.
- `ledger_views.note_waiting(note, request) -> bool`: the rule `dw_mcp/ledger.py:note_waiting` states, pinned to the same `tests/fixtures/note_waiting_cases.json`.
- `summary_view(request, jobs=(), prompts=None, needs=False)`: with `needs`, builds each item's read fields without outputs (`_item_status`, `_ready`, `_review`, `_result`'s missing flag; no proof callable is called) and adds `"needs": needs_counts(...)`.
- `LedgerSummary.needs: dict[str, int] = sometimes("Only with needs=true: how many items want the user, by reason - held, ready, failed (can run again), revised, choose, accepted_missing - and notes_waiting.")`. `OpenRequestSummary` inherits it and never carries it.
- `ledger.py` stays ≤ 1146: the keyword rides on the existing signatures and calls.

- [ ] **Step 1: Write the failing tests** in `tests/test_ledger_needs.py`:
  - `test_every_shared_case_gives_its_needs` (reads the fixture, `needs_counts(case["request"])`)
  - `test_the_summary_needs_match_the_view` (a stored ledger with a hold, a failed item, a blocked failed item and two takes: `summary_view(..., needs=True)["needs"] == needs_counts(request_view(...))`)
  - `test_a_summary_reads_no_proof_for_needs` (a proof callable that raises)
  - `test_the_list_says_nothing_about_needs_unless_asked` and `test_the_open_requests_list_never_carries_needs`
  - `tests/test_ui_twins.py`: the note-waiting cases also hold for `ledger_views.note_waiting`.
- [ ] **Step 2: Run** `$PY -m pytest tests/test_ledger_needs.py tests/test_ui_twins.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, regenerate OpenAPI, `cd ui && npm run gen:api`.
- [ ] **Step 4: Run** the files, `tests/test_api_contract.py`, `tests/test_ledger*.py`, `tests/test_mcp_server.py`, `wc -l dw/ledger.py dw/server/routes/requests.py dw/server/api_models_library.py ui/src/lib/api.ts` (≤ 1146, 388, 765, 598), `$PY scripts/arch_metrics.py`. Expected: PASS.
- [ ] **Step 5: Commit** `server(requests): a list row says what needs the user, when asked`.

### Task 4: The Requests list says everything that needs you

RQ-4's UI half and RQ-21's filter chip; folds #1611's Overview title and #1600's `formatMinutes`.

**Files:**
- Modify: `ui/src/lib/request/attention.ts`, `ui/src/lib/pages/RequestsPage.svelte`, `ui/src/lib/pages/OverviewPage.svelte` (the requests panel row), `ui/src/lib/gallery/RequestFilter.svelte`, `ui/src/lib/format.ts` (`formatMinutes`)
- Test: `ui/src/lib/request/attention.test.ts`, `ui/src/lib/pages/RequestsPage.test.ts`, `ui/src/lib/pages/OverviewPage.test.ts`, `ui/src/lib/gallery/RequestFilter.test.ts` (or the file that holds its tests: `grep -rln RequestFilter src --include=*.test.ts`), `ui/src/lib/format.test.ts`
- Docs: `docs/SERVER.md` *Requests* first paragraph (what a row's line says, the order)

**Interfaces:**
- Consumes: Task 3's `LedgerSummary.needs` (optional in the type), Task 2's `needsLine`.
- `AttentionKind` gains `'needs'`; `attention(summary)`: with `needs` holding any count above 0, `{ kind: 'needs', text: needsLine(needs) }`; otherwise the Global constraints fallbacks (the `idle` line becomes `<n> planned, nothing running`, revised counted with planned). A summary without `needs` keeps today's failed-first line.
- `needsYou(summary)`: with `needs`, any count but `notes_waiting` above 0; without, today's rule. `rank`: `needs.held > 0` → 0, other needs → 1, in flight → 2, else 3.
- `RequestsPage` draws `notes_waiting` as the copy's figure in `.figures`. The Overview row keeps its title visible at 390 px (title on its own line above the attention line).
- `RequestFilter` reads the title from its existing `requests` resource.
- `formatMinutes(minutes)`: under 1 `under 1 min`; under 60 as today; from 60 `formatApprox(minutes * 60)`'s hour form (`about 1 h 20 min`).

- [ ] **Step 1: Write the failing tests.**
  - `attention.test.ts`: `a row with needs says the needs sentence`, `held requests come first`, `a summary without needs keeps the status line`, `idle reads planned, nothing running`.
  - `RequestsPage.test.ts`: `a row names its waiting notes`.
  - `OverviewPage.test.ts`: `a request row keeps its title beside a long attention line`.
  - `RequestFilter` test: `the chip names the request's title`, `the id stands in until the title is known`.
  - `format.test.ts`: `formatMinutes past an hour reads hours and minutes`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/attention.test.ts src/lib/pages/RequestsPage.test.ts src/lib/pages/OverviewPage.test.ts src/lib/format.test.ts` and the RequestFilter test. Expected: FAIL.
- [ ] **Step 3: Implement.** Check the list and the Overview at 1440, 1300 and 390 px by eye (`node scripts/ux-capture.mjs`).
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks plus `e2e/responsive.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): the list says everything that needs you, held first`.

### Task 5: Why a take failed, on its tile and its row

RQ-3's UI half.

**Files:**
- Modify: `ui/src/lib/request/AttemptStrip.svelte`, `ui/src/lib/request/itemState.ts` (`failedLine`), `ui/src/lib/request/RequestItem.svelte`
- Test: `ui/src/lib/request/itemState.test.ts`, `ui/src/lib/request/AttemptStrip.test.ts` (new) or `RequestItem.test.ts`

**Interfaces:**
- Consumes: Task 1's `LedgerAttempt.reason` (optional).
- `export function failedLine(item: LedgerItem): string | null` in `itemState.ts`: the newest failed or cancelled attempt with a `reason`, as the copy's `Take <n> failed: <reason>` / `Take <n> was cancelled: <reason>`; null when none has one.
- `AttemptStrip`: under a failed or cancelled tile, `<span class="reason bad" title={reason}>` clamped to one line.
- `RequestItem` draws `failedLine` as a `.sub` line of a failed or cancelled row (collapsed or open).

- [ ] **Step 1: Write the failing tests.** `itemState.test.ts`: `the newest failed take's reason is the row's line`, `a cancelled take says cancelled`, `a failed take with no reason shows no reason line`. Strip: `a failed tile shows its reason`, `a long reason is one line with the whole in its title`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks (the fixture's failed takes carry `e2e: this attempt failed`). Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): a failed take says why on its tile and its row`.

### Task 6: Accept looks like a decision, and stars the file

RQ-16 and the ★ **Accept** row; folds #1609's three take items. Undo semantics follow Question 3 (written here for the recommended answer).

**Files:**
- Modify: `ui/src/lib/request/AttemptStrip.svelte` (the button, the viewer key, the deep link), `ui/src/lib/request/PlanItems.svelte` (`accept`), `ui/src/lib/request/RequestItem.svelte` (opens for a deep-linked take)
- Test: `ui/src/lib/request/AttemptStrip.test.ts`, `ui/src/lib/pages/RequestPage.edit.test.ts`, `ui/e2e/requests.spec.ts` (Back from the take viewer, #1610's request-takes case)
- Docs: `ui/GLOSSARY.md` *Accept* row drops "(phase 6)"; `docs/SERVER.md` the Accept sentence

**Interfaces:**
- The Accept button drops `quiet`: the default button look; `Accepted` keeps `aria-pressed` and the check.
- `PlanItems.accept(row, take, job)`: when `job` is not null and the store answers `saved`, `api.setStarred(take.attempt.output.name, true)` through `action()` (a take with no output stars nothing); the toast is the copy's; its Undo takes back the acceptance only (Question 3). A star that fails reports through `notify.failure` and leaves the acceptance.
- The viewer key is `<item key>/<job>/<name>`, so one job linked to two items opens in its own strip. `RequestItem` opens itself when the hash's `open=` names one of its takes (`openInHash()` starts with `<item key>/`).

- [ ] **Step 1: Write the failing tests.**
  - `AttemptStrip.test.ts`: `Accept is a default button, Accepted is pressed`, `a job linked to two items keeps a viewer per strip`, `the viewer's history opens and closes with the hash`, `the Accepted state shows the check`.
  - `RequestPage.edit.test.ts`: `Accept stars the take's file`, `a take with no file is accepted and stars nothing`, `a failed star leaves the acceptance`, `Undo takes back the acceptance and keeps the star`, `a deep link to a take opens its row`.
  - `requests.spec.ts`: `Back closes the take viewer`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/AttemptStrip.test.ts src/lib/pages/RequestPage.edit.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): Accept looks like a decision and stars the take's file`.

### Task 7: The hold - warn before a note stops it, and keep Run now

RQ-1 (decision 2); folds #1600's held chip.

**Files:**
- Modify: `ui/src/lib/request/reviewText.ts` (`bannerText`, `runNowKeys`, `chipText`, `holdNews`, `holdsStopped`), `ui/src/lib/request/NoteList.svelte`, `ui/src/lib/request/ItemNote.svelte` (`stops` prop)
- Test: `ui/src/lib/request/reviewText.test.ts`, `ui/src/lib/request/ItemNote.test.ts`, `ui/src/lib/request/NoteList.test.ts`, `ui/src/lib/pages/RequestPage.test.ts`
- Docs: `docs/SERVER.md` the hold paragraph (redirect wording, Run now in every state but due-and-connected); ARCHITECTURE *Requests in the UI* ("**Run now…** only opens that dialog" stays; add "offered while holding, redirected, or due with the agent not connected")

**Interfaces:**
- `bannerText(group, listening, left)`: redirected splits on `listening` (Global constraints copy); the rest unchanged.
- `runNowKeys(group, listening)`: offered when `holding`, `redirected`, or `due && !listening`.
- `chipText` and `holdNews`: the copy.
- `export function holdsStopped(request: LedgerRequest): number`: live items whose `review.state` is `holding` or `due`.
- `NoteList` passes `stops={holdsStopped(request)}` to every `ItemNote` (the request's and each item's); `ItemNote` shows the warning line while editing and `stops > 0`, above the listening line.

- [ ] **Step 1: Write the failing tests.**
  - `reviewText.test.ts`: `a redirected hold says the truth when the agent isn't connected`, `a redirected hold still offers Run now`, `the held chip says waiting for your OK`, `holdsStopped counts holding and due, not redirected`.
  - `ItemNote.test.ts`: `the box warns that saving stops the held items`, `no hold, no warning`.
  - `NoteList.test.ts` / `RequestPage.test.ts`: `the warning follows the latest read` (a read that redirects the hold removes the line; the draft stays).
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/ src/lib/pages/RequestPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): a note warns before it stops a hold, and Run now stays`.

### Task 8: A closed request is read-only

RQ-7 (decision 4); folds #1600's "already open".

**Files:**
- Modify: `ui/src/lib/request/menuEntries.ts` (`MenuContext.open`), `ui/src/lib/request/PlanItems.svelte` (no handle, no Accept, no row action, the closed line), `ui/src/lib/request/NoteList.svelte` (`editable`), `ui/src/lib/request/RequestMenu.svelte` (the Undo)
- Test: `ui/src/lib/request/menuEntries.test.ts`, `ui/src/lib/pages/RequestPage.edit.test.ts`, `ui/src/lib/request/NoteList.test.ts`, `ui/src/lib/request/RequestMenu.test.ts`
- Docs: `docs/SERVER.md` the header-menu sentence ("a closed request shows its plan read-only")

**Interfaces:**
- `MenuContext` gains `open: boolean`; `itemMenu` on a closed request keeps only the links.
- `PlanItems`: with `request.state !== 'open'` it passes no `handle`, no `onaccept`/`onkeep`, no `action`, and shows the copy's line above the list (`role="note"`).
- `NoteList.editable` is also false on a closed request; saved notes still render.
- `RequestMenu`'s Undo of **Mark done**/**Abandon**: when the store reads the request already `open`, `notify.success('"<title>" is already open')`.
- **Reopening (Don, 2026-11-24: a closed request must be reopenable).** The closed line carries a **Reopen** button that writes `state: open` through the same `planStore` state intent the header menu's **Reopen** uses (one code path, not a second writer); the header menu keeps **Reopen** and **Delete…** on a closed request; after a reopen the grips, Accept, row actions and note boxes come back without a reload, and the toast reads `Reopened "<title>"`. A refused reopen (stale revision) says so as the menu's does.

- [ ] **Step 1: Write the failing tests.** `menuEntries.test.ts`: `a closed request's row menu has only links`. `RequestPage.edit.test.ts`: `a closed request has no grips, Accept, Run or drop`, `the closed line names its state`. `NoteList.test.ts`: `a closed request keeps its notes and offers no box`. `RequestMenu.test.ts`: `Undo over a request the agent reopened says it is already open`. `RequestPage.edit.test.ts`: `the closed line's Reopen opens the request and brings the editing back`, `the header menu still offers Reopen on a closed request`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/ src/lib/pages/RequestPage.edit.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): a closed request is read-only until reopened`.

### Task 9: Rows at every width, with less repeated chrome

RQ-13, RQ-14, RQ-15 (the clipping), RQ-21's tag/chip look. Splits `RequestItem` (decision 8).

**Files:**
- Create: `ui/src/lib/request/ItemMore.svelte`
- Modify: `ui/src/lib/request/RequestItem.svelte`, `ui/src/lib/request/NoteList.svelte` (no heading without notes), `ui/src/lib/request/ItemMenu.svelte` (`.entries`), `ui/src/lib/request/StateTag.svelte`
- Test: `ui/src/lib/request/RequestItem.test.ts`, `ui/src/lib/request/ItemMore.test.ts` (new), `ui/src/lib/request/NoteList.test.ts`, `ui/e2e/requests.spec.ts` (the menu's box)

**Interfaces:**
- `ItemMore.svelte` props `{ row: ItemState; request: LedgerRequest; workspace: string; open: boolean; ongo; onreload; onaccept?; onmove?; pending: LedgerNoteRef[] (bindable) }` and `export function add(seed?)` passing to `NoteList.add`. It holds what `RequestItem`'s `.more` holds today. Capabilities to keep, each pinned in `ItemMore.test.ts` or the existing `RequestItem.test.ts`:
  1. the block is `hidden`, never removed, so drafts survive a close;
  2. `AttemptStrip` renders only while open and the item has takes, with `onaccept`/`onnote` absent on a dropped item;
  3. `after:` and `needed by:` name each item with its state, and a press calls `ongo`;
  4. the menu's "Add a note for the agent" and "Note this take" open the row and focus the new note's box with the take's reference;
  5. `pendingRefs` stays bound across reads;
  6. "For the whole request instead" hands the draft to the header;
  7. what it runs, folded (copy).
- `RequestItem` keeps the row and ends at or below 450 lines. When open with takes, the proof cell collapses to 0 (`.proofcell.under` takes no track) so the title starts at the grip. Under a 900 px container the chip and figure move under the title.
- `NoteList` without saved notes renders no `.label`, only the box's button.
- `ItemMenu` `.entries` sizes to its widest entry (`width: max-content; min-width: 100%`), so no entry touches the edge.
- `StateTag` `open`: ink text on the `--line` border, so it no longer matches `.chip.ready`'s ink outline.

- [ ] **Step 1: Write the failing tests.** `ItemMore.test.ts`: the seven capabilities. `RequestItem.test.ts`: `an open row's proof cell takes no room`. `NoteList.test.ts`: `no notes, no heading`. `requests.spec.ts`: `the item menu's widest entry fits inside it` (bounding boxes) and `a row at 1300 px keeps its title on one line`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/`. Expected: FAIL.
- [ ] **Step 3: Implement.** Check 1440, 1300 and 390 px by eye.
- [ ] **Step 4: Run** `npm test`, the metrics gate (`duplicate_blocks` not above 16; lower `baseline.json` if `NoteList:177` went), `wc -l src/lib/request/RequestItem.svelte` (≤ 450), the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): rows fit at every width and say each thing once`.

### Task 10: Reorder says when a dependency wins

RQ-18.

**Files:**
- Modify: `ui/src/lib/request/reorderMath.ts` (`runsAfter`), `ui/src/lib/request/PlanItems.svelte` (`moved`, the hint), `ui/src/lib/request/RequestItem.svelte` (a `hint` line)
- Test: `ui/src/lib/request/reorderMath.test.ts`, `ui/src/lib/pages/RequestPage.edit.test.ts`

**Interfaces:**
- `export function runsAfter(plan: PlanState): Record<string, ItemRef[]>` in `reorderMath.ts`: for each live item, its dependencies placed after it in plan order; keys the plan does not hold, or holds as dropped, are skipped.
- `PlanItems` passes each row's list to `RequestItem` (`hint` line, copy) and adds the copy's second sentence to `moved`'s live text.

- [ ] **Step 1: Write the failing tests.** `reorderMath.test.ts`: `an item above its dependency runs after it`, `an item below its dependency has nothing`, `a dependency the plan no longer holds gives no hint`. `RequestPage.edit.test.ts`: `moving an item above its dependency says it still runs after it`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/reorderMath.test.ts src/lib/pages/RequestPage.edit.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): a row above its dependency says it still runs after it`.

### Task 11: The run dialog in plain words, and failed items can join

RQ-10 and RQ-11's dialog half (decisions 5, 6); adopts `verdictOf`; folds #1612's "0 issues" and #1601's `runPlan.ts` tilde.

**Files:**
- Modify: `ui/src/lib/request/runPlan.ts` (`readyPlan`, `Candidate.optional`, `quoteLine`), `ui/src/lib/request/runQueue.svelte.ts` (`#priceRow`), `ui/src/lib/request/RunDialog.svelte`, `ui/src/lib/runCheck.svelte.ts` (`verdictOf`'s zero-issue text), `ui/src/lib/pages/RequestPage.svelte` (the header button)
- Test: `ui/src/lib/request/runPlan.test.ts`, `ui/src/lib/request/runQueue.svelte.test.ts`, `ui/src/lib/request/RunDialog.test.ts`, `ui/src/lib/runCheck.test.ts`
- Docs: `docs/SERVER.md` the run paragraph (`Run <n>…`, `Run <n> <job|jobs>`, the failed group); GLOSSARY *Whether a run will work* row names the run dialog too

**Interfaces:**
- `Candidate` gains `optional?: boolean`. `readyPlan(plan)` adds each live `failed` row that is `runnable` as an optional candidate, after the others, and no longer lists it under skipped.
- `RunQueue` starts an optional row unticked; pricing is the same.
- `#priceRow`: an invalid result sets `row.error` to `verdictOf(result, null, placeFindings(result, [])).text + ' Ask the agent to fix the item\'s workflow or arguments.'` and `row.detail` to the errors' and `error`'s text joined by `\n`. A valid result whose `verdictOf` is gated or not found keeps `priced` and sets `row.verdict: Verdict` (new `RunRow` field) for the warn line.
- `export function quoteLine(estimate: PlanEstimate | null): { text: string; tone: 'plain' | 'warn'; detail?: string }`: a figure as `describeEstimate`; no figure `No time estimate for this workflow.` (plain) with `detail` the engine reason.
- `verdictOf` with `issues === 0` on an invalid answer: `A job would be refused.`
- `RunDialog`: the copy's heading, the failed group under its heading, `row.verdict` text and remedy, `quoteLine` with `Details`, the spend button. `RequestPage`'s header button reads `Run <n>…` with its title.

- [ ] **Step 1: Write the failing tests.**
  - `runPlan.test.ts`: `failed items that can run join the ready plan, optional`, `a failed item that cannot run stays skipped`, `no figure reads no time estimate with the reason as detail`.
  - `runQueue.svelte.test.ts`: `an optional row starts unticked`, `a refused validation reads the verdict, the engine text behind detail`, `a gated model is priced with its verdict`.
  - `RunDialog.test.ts`: `the button counts jobs`, `failed items sit under their heading, unticked`.
  - `runCheck.test.ts`: `a refusal with no counted issue reads A job would be refused.`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/runPlan.test.ts src/lib/request/runQueue.svelte.test.ts src/lib/request/RunDialog.test.ts src/lib/runCheck.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): the run dialog speaks plainly and can retry failed items`.

### Task 12: What the agent changed, live and since you last looked

RQ-8's leftover and RQ-9 (written for Question 2's recommended answer; if Don picks the server record, this task is rewritten before it starts).

**Files:**
- Create: `ui/src/lib/request/planDiff.ts`, `ui/src/lib/request/planDiff.test.ts`
- Modify: `ui/src/lib/request/announce.ts`, `ui/src/lib/request/PlanChanges.svelte`, `ui/src/lib/pages/RequestPage.svelte` (`onread` keeps the diffs; the snapshot)
- Test: `ui/src/lib/request/announce.test.ts`, `ui/src/lib/pages/RequestPage.test.ts`
- Docs: ARCHITECTURE *Requests in the UI*: "what changed between two reads is `planDiff`; Plan changes shows it under the write it came with while the page was open, and since the last visit from a per-browser snapshot (`request-seen:<workspace>:<id>`), a convenience that may be empty"

**Interfaces:**
- `export type PlanChange = { kind: 'added' | 'dropped' | 'restored' | 'renamed' | 'changed'; key: string; title: string; was?: string; what?: 'workflow' | 'arguments' | 'depends_on'; names?: string[] }`
- `export function planDiff(before: Snapshot | LedgerRequest, after: LedgerRequest): PlanChange[]` and `export function lineOf(change: PlanChange): string` (the copy).
- `export type Snapshot = { revision: number; items: Record<string, { title: string; workflow: string | null; dropped: boolean; depends_on: string[]; args: Record<string, string> }> }`, `export function snapshotOf(request: LedgerRequest): Snapshot` (each argument as `fingerprint(value)`: its JSON when ≤ 120 characters, else `#` + an FNV-1a 32-bit hex of it), `readSnapshot(workspace, id)` / `writeSnapshot(workspace, id, snap)` through `storage.ts`, each try/caught.
- `announce` adds the two copy sentences for a `changed` workflow or arguments, after the rename.
- `PlanChanges` gains `diffs: Record<number, PlanChange[]>` (by the revision a read brought) and `since: { from: number; to: number; changes: PlanChange[] } | null` with `ondismiss`. `RequestPage` fills `diffs` in `onread` from the previous read the store held (agent writes only), computes `since` on the first read from the stored snapshot (dropped when its revision is above the read's), and writes the snapshot on that read and on **Dismiss**.

- [ ] **Step 1: Write the failing tests.** `planDiff.test.ts`: `names added, dropped, restored and renamed items`, `names the arguments that changed`, `a long argument compares by fingerprint`, `nothing changed, no lines`. `announce.test.ts`: `says an argument change after a rename`. `RequestPage.test.ts`: `a write read while open lists its changes under it`, `since you last looked lists the changes from the snapshot`, `a stored revision above the current one is discarded`, `storage that throws shows no diff`, `snapshots are per workspace`, `a diff is taken against the read the page held, not the snapshot`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/planDiff.test.ts src/lib/request/announce.test.ts src/lib/pages/RequestPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `RequestPage` stays at or below 420 lines (logic in `planDiff.ts`).
- [ ] **Step 4: Run** `npm test`, the metrics gate (lower `duplicate_blocks` if `PlanChanges:85` or `RequestPage:233` went), the e2e checks plus `e2e/journeys/agent-edits.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): plan changes say what changed, and what changed since you last looked`.

### Task 13: A note the agent answered shows its answer and can be cleared

RQ-20 (decision 7); folds #1600's ItemNote resend, NoteDraft `kept` and N1, and #1602's ItemNote and NoteList test items.

**Files:**
- Modify: `ui/src/lib/request/noteState.ts` (`answerOf`), `ui/src/lib/request/ItemNote.svelte`, `ui/src/lib/request/NoteList.svelte`, `ui/src/lib/noteDraft.svelte.ts` (`kept`)
- Test: `ui/src/lib/request/noteState.test.ts`, `ui/src/lib/request/ItemNote.test.ts`, `ui/src/lib/request/NoteList.test.ts`, `ui/src/lib/noteDraft.test.ts` (or the file that holds NoteDraft's tests), `ui/src/lib/ui/NoteEditor.test.ts` (NoteDraft's other user, unchanged behaviour)

**Interfaces:**
- `export function answerOf(note: LedgerNote, request: LedgerRequest): { revision: number; why: string } | null`: the first agent history entry after the note (`agentWritesSince`'s first).
- `ItemNote`: an answered saved note shows the copy's answer line under its text, and its delete button reads **Clear note** (no confirm, as today for an answered note).
- The new-note box, when a save went out while typing continued and the box starts with the sent text, keeps only the trimmed remainder.
- `NoteDraft.kept` is a `$derived` against `saved()`, so a draft equal to the saved note is not "unsaved", and a reopen after a mid-save switch raises no "changed elsewhere" notice over the person's own save.
- `NoteList` keeps one guard against a second write (the action's), and the dead branch goes.

- [ ] **Step 1: Write the failing tests.** `noteState.test.ts`: `the answer is the agent's first write after the note`, `no write after it, no answer`. `ItemNote.test.ts`: `an answered note shows the answer and Clear note`, `a second save after typing on sends only the new words`, `Cancel's discard question is danger`. `noteDraft` test: `a draft equal to the saved note is not kept`. `NoteList.test.ts`: `a second Save while one is out sends one write`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/ src/lib/noteDraft.test.ts src/lib/ui/NoteEditor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test`, the metrics gate, the e2e checks. Expected: PASS.
- [ ] **Step 5: Commit** `ui(notes): an answered note shows the agent's answer and can be cleared`.

### Task 14: The journey, the rules for the next agent, and the phase gate

**Files:**
- Create: `ui/e2e/journeys/review-a-request.spec.ts`
- Modify: `ui/e2e/serve_fixture.py` (request `e2e-review`), `ui/e2e/requests.spec.ts` (list counts and order, if the third request moves them), `ui/GLOSSARY.md`, `ui/CLAUDE.md` (rewrite, no new line), `docs/ARCHITECTURE.md`, `docs/stabilization/ui/baseline.json`

**Interfaces:**
- The fixture adds request `e2e-review`, "E2E review", in the default workspace, written `by="agent"`, with its own history rows: `hero` "Hero still" done with two succeeded takes, none accepted; `plate` "Background plate" failed once, `error` `"CUDA out of memory. Tried to allocate 2.00 GiB\nTraceback (most recent call last): …"`, workflow `e2e-card`; `comp` "Composite" failed once, depending on `plate`; `title` "Title card" on `e2e-card`, held for an hour.
- `GLOSSARY.md` *Before you write markup* gains: what a request wants from the person (`needsParts`, `needsLine` in `src/lib/request/standing.ts`); what the agent changed (`planDiff`, `src/lib/request/planDiff.ts`). *Words* gains **Needs you** (the sentence; held replaces ready) and **take** keeps "attempt" out of request copy.
- `ui/CLAUDE.md` bullet 1's "what a row shows is derived in `src/lib/request/itemState.ts`" becomes "what a row shows and what needs the person is derived in `src/lib/request/itemState.ts`, which the server's `needs=true` mirrors", paid for by trimming its parenthesised ARCHITECTURE pointer. Still 12 lines.

- [ ] **Step 1: Write the journey** `review-a-request.spec.ts`:
  1. On Requests, the `E2E review` row reads `Needs you: 1 waiting for your OK, 1 failed, 1 take to choose.` and is listed first; at 390 px its title is visible.
  2. Open it: the header's needs line has the same parts; press `1 failed`: Background plate's title has focus, its row is open and reads `Take 1 failed: CUDA out of memory. Tried to allocate 2.00 GiB`; Composite reads `Failed. It can run again once Background plate is done.`
  3. Open the request's note box: the warning reads `Saving this note stops 1 item waiting for your OK from starting on their own.`; save `warmer, please`: the banner reads the not-connected redirected copy, **Run now…** is still there, Title card's chip reads `on hold: note left`.
  4. Press **Run now…**: the dialog's Title card row reads `No time estimate for this workflow.`; the button reads `Run 1 job`; Cancel.
  5. On Hero still, press **Accept** on take 2: the toast reads `Accepted take 2 for Hero still and starred it`; `GET /api/gallery` lists that file starred.
  6. As the agent, `PUT` the plan renaming Hero still to `Hero still (wide)` and changing Background plate's `line`: within two feed polls the live region says the rename and `The agent changed the arguments of Background plate.`, and the newest Plan changes entry lists both lines.
  7. Focus Composite's grip and press Home: its row reads `Runs after Background plate whatever its place.`
  8. Open an item menu: its widest entry fits inside it.
  9. **Mark done**: grips, Accept, row actions and note boxes are gone and the closed line shows; Undo brings them back. **Mark done** again, then press the closed line's **Reopen**: the toast reads `Reopened "E2E review"` and the grips, Accept and note boxes are back; reload: the request is still open.
  10. Reload: no `Since you last looked` panel (the snapshot is current).
- [ ] **Step 2: Run** `cd ui && npm run build && DW_E2E_PYTHON=$PY npx playwright test e2e/journeys/review-a-request.spec.ts`. Expected: PASS. A failure names the task that left a gap; fix it there.
- [ ] **Step 3: Write the rules** as in Interfaces; ARCHITECTURE's *Requests in the UI* enforcement column adds `review-a-request.spec.ts`, `needs.twin.test.ts` and `planDiff.test.ts`, and *Requests ledger*'s adds `tests/test_ledger_take_reason.py` and `tests/test_ledger_needs.py`.
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, `DW_E2E_PYTHON=$PY npx playwright test`, the metrics gate, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (only `complex_functions: 11 -> 13`), `$PY scripts/arch_metrics.py` (`modules 281`, `claude_md_lines 129`), the `wc -l` caps of Tasks 1 and 3 plus `dw/server/api_models.py dw/server/routes/jobs.py dw/server/routes/library.py` (≤ 1105, 1041, 1000), and `$PY -m pytest tests/test_ui_twins.py tests/test_api_contract.py tests/test_ledger_take_reason.py tests/test_ledger_needs.py tests/test_ledger*.py tests/test_mcp_server.py -q`. Expected: PASS; no page over 600.
- [ ] **Step 5: Commit** `ui(requests): the review-a-request journey, and the rules for the next agent`.

---

## Phase exit

- `GET`/`PUT` one request and `GET` the list answer as before unless asked `reasons=true` / `needs=true`; `/api/requests` is unchanged; MCP's replies are key for key today's.
- `prefix_literals` is 0; every metric that was 0 is 0; `duplicate_blocks` is at or below 16; nothing is above `$SCRATCH/ui-p6.json`.
- `modules` 281, `claude_md_lines` 129, the size caps hold, no page over 600, `RequestItem` at or below 450.
- The UI's needs rule and the server's are one fixture; ARCHITECTURE says so.
- `review-a-request.spec.ts` holds the phase.
