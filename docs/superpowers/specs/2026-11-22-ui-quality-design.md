# UI quality: design for a trustworthy, consistent, agent-maintainable web UI

**Date:** 2026-11-22
**Source:** the six-reviewer Playwright UX audit of 2026-11-22 (`docs/superpowers/specs/2026-11-22-ui-audit.md`). Its findings are listed in the appendix under the IDs below.
**Status:** draft for Don's review

## Why

The backend and the MCP surface carry most of the project's engineering: owner modules, twin tests, the architecture map, ratchets. The UI has some of that already: the overlay wrappers, `DocumentEditor`, the change feed, the response contract and the arch ratchet. But the audit found that the UI's main failures come from what it lacks in that respect:

- **Data loss.** Each page solves the same problems (saving, loading, errors, confirming, connection state) on its own, so each has its own gaps. Out of that came eight ways to lose work, several of them reporting "Saved" while saving nothing.
- **Inconsistency.** Similar concepts behave differently from place to place:
  - "Keep" means two things, and "Accept", "Star" and "Keep" all sound like "I like this one" but do three different things.
  - One run of a workflow has six names.
  - A file viewer exists in four forms.
  - Some pages recover after an outage and others don't.
  - Some deletes ask first, some offer undo, and some do neither.
- **Leaked internals.** The engine's internals reach the user in its own words: calibration math, `overwrite=true`, raw JSON events, "no session is listening".

This design fixes the findings by first building the shared pieces they all lack. After that, each fix is small, and the next agent to touch a page inherits correct behaviour instead of reinventing it.

## Goals

1. **Nothing the user did is lost, and nothing reports success that didn't happen.**
2. **One concept, one behaviour, one name.** The same kind of thing (a file, a destructive action, a load, an error, a reference input) looks and acts the same on every page.
3. **Plain language first, engine detail on request.** Every verdict leads with one sentence and a remedy. Engine text sits behind a disclosure, never in a toast.
4. **Agent-maintainable.** Every rule above is a shared primitive with tests and, where it can be checked mechanically, a lint or ratchet rule. An agent that writes a new page the wrong way fails a check; it doesn't need to have read this document.
5. **A compelling first hour.** A new user can tell what the app is, run something, find the result, understand it and reuse it, without the docs.

## Non-goals

- A visual redesign or a new component library. The look in `app.css` stays.
- WCAG contrast as a gate. Contrast stays guidance (Don's ruling). Keyboard and screen-reader basics are in scope.
- New engine features. Server changes are limited to what a UI fix needs: conditional writes, a failed take's error, a few summary fields, compression.
- Changing MCP tool behaviour. Every server change is opt-in by the caller, so agents see no difference.

## Design principles

These go into `ui/CLAUDE.md` as rules.

| # | Principle | What it means in code |
|---|---|---|
| P1 | **A write never loses work, and only real success says "Saved".** | The editors flush every pending input before any action. Saves carry preconditions (create-only, or "unchanged since I read it"). Leaving with unsaved edits always asks. |
| P2 | **Loading, empty and error are different states.** | Page data comes through `resource()` and renders through `<LoadState>`. `Empty` is importable only by `LoadState`. |
| P3 | **The connection is app state, not a page's problem.** | One `connection` store, one banner. Pages reload on reconnect and on token change without code of their own. |
| P4 | **Actions are uniform.** | Every button that mutates goes through `action()`. That gives it a busy guard (no double submit), mapped error copy and the confirm-or-undo policy. |
| P5 | **Reversible means undo; irreversible means a named, styled confirm.** | The policy table below. `ConfirmDialog` gains `title`, `danger` and a subject name. |
| P6 | **Errors speak the user's language.** | `describeError()` maps status, code and fetch failures to a title, a remedy and an optional detail. Raw `e.message` in a toast is ratcheted to zero. |
| P7 | **Show the proof, and show it the same way.** | One `MediaViewer`, one `OutputTile`, one `RecipePanel` ("what made this"), used by the gallery, assets, the job page, request takes and compare. |
| P8 | **One reference input.** | A media or reference argument is edited through `ReferenceField` everywhere: run form, editor, rerun dialog. It handles upload, browse, `previous_result:` and kind filtering. |
| P9 | **One vocabulary.** | `ui/GLOSSARY.md` defines every user-facing noun and verb. A copy ratchet fails on banned synonyms in templates. |
| P10 | **Every route is a page.** | `PageHeader` sets `document.title`, moves focus to its `h1` after navigation and renders the breadcrumb from the sidebar's labels. |

### Destructive-action policy (P5)

| Action kind | Example | Behaviour |
|---|---|---|
| Reversible on the server, or cheap to redo | star/unstar, drop a request item, remove a step or variable in the editor, accept/unaccept a take, mark a request done | Act at once, with an **Undo** toast (`notify.undo`) |
| Destroys a user file | delete an output, asset, prompt, workflow or workspace | A confirm with a `danger` button that names the thing and states the count, including how many are starred |
| Spends GPU time or changes the server | Run, a cancel more than 60 s into a run, Update diffusers, clear a cache | A confirm that says what it costs or changes; Run keeps the priced dialog it has today |
| Overwrites | save over an existing name, upload over an existing asset, save over a newer version | A conflict dialog offering **Replace**, **Keep both** (where a name can be suggested) or **Cancel** |

## Architecture: the shared building blocks

All of these live in `ui/src/lib/` (cross-page modules) or `ui/src/lib/ui/` (components). Each gets a row in `docs/ARCHITECTURE.md`'s UI table, naming its owner module and the test that holds it.

| Building block | Module | Interface (outline; the phase plans pin the exact signatures) | Replaces |
|---|---|---|---|
| Conditional writes (server) | `dw/server/preconditions.py` | `ETag` on GET and PUT of workflows and prompts. `If-None-Match: *` refuses with 409 `exists`; `If-Match: <etag>` refuses with 409 `stale_write` carrying the current `etag`. Uploads honour `If-None-Match: *`. With no header, behaviour is unchanged, so MCP sees nothing new. | silent overwrite |
| Save with preconditions | `ui/src/lib/documentSave.ts` | `saveDocument(ed, ...)` flushes, writes with the editor's preconditions and turns a 409 into the shared conflict dialog. Both editors and the asset upload use it. | each page's own save |
| Editor flush, load state, leave guard | `ui/src/lib/editorShell.svelte.ts` (`DocumentEditor`), `ui/src/lib/router.svelte.ts` (`addLeaveGuard`) | `registerFlush`, `flush()`, `loadState`, `etag`, `canWrite`. The router asks its guards before any hash change. | blur-only apply, breadcrumb-only guard |
| Connection | `ui/src/lib/connection.svelte.ts`, `ui/src/lib/ui/ConnectionBanner.svelte` | `connection.state: 'ok' \| 'offline' \| 'locked' \| 'unknown'` and `connection.epoch` (rises on reconnect and on an accepted token). Fed by `apiRequest.send()` and the status poll. | per-page "Failed to fetch", header "idle" |
| Error copy | `ui/src/lib/apiErrors.ts`: `describeError` | `(e: unknown) => { title, remedy?, detail?, code? }` | `notify.error(e.message)` everywhere |
| Page data | `ui/src/lib/resource.svelte.ts`, `ui/src/lib/ui/LoadState.svelte` | `resource(load, { watch: ChangeKind[] })` gives `{ status, data, error, reload }`. It reloads on feed kinds and on `connection.epoch`, and keeps the last data when a refresh fails. | ad-hoc `$effect` loaders, empty-while-loading |
| Actions | `ui/src/lib/action.svelte.ts`, `ui/src/lib/ui/ActionButton.svelte` | `action(fn, { confirm?, undo?, success? })` gives `{ run, busy }`. It refuses re-entry while busy and reports through `describeError`. | double-submit, unhandled rejections |
| Disclosure of engine text | `ui/src/lib/ui/Details.svelte` | a one-line `summary` plus slotted engine detail, with a copy button | 40-line toasts, raw paragraphs |
| Media | `ui/src/lib/ui/MediaViewer.svelte`, `OutputTile.svelte`, `RecipePanel.svelte` | A viewer fitted to the window, with zoom, previous/next, swipe and fullscreen, and the open file in the URL. One tile per kind (image, video poster, audio duration, text excerpt). | GalleryDetail's docked panel, the 120 px take tiles, the asset grid |
| Reference input | `ui/src/lib/ui/ReferenceField.svelte` | Kind-aware suggestions (earlier steps' `previous_result:` first), upload, browse, a preview thumbnail, and a prompt preview | VariablesForm text fields, MediaArgumentInput |
| Page frame | `ui/src/lib/ui/PageHeader.svelte` | title, `document.title`, focus on navigation, breadcrumb from the shared `SECTION_LABELS` | the Breadcrumb's internal keys, a static tab title |
| Vocabulary | `ui/GLOSSARY.md`, `ui/src/lib/terms.ts`, `ui/scripts/copy-rules.test.ts` | the terms, banned synonyms with their replacements, and the duration format (`about 3 min`) | six names for one run |
| Viewing the UI as a whole | `ui/scripts/ux-capture.mjs` | Starts the fixture server and screenshots every route in both themes at 1440 and 390 px into a folder, so an agent or reviewer can look at the whole UI in one pass. | ad-hoc audit scripts |

### Mechanical enforcement (agent maintainability)

New metrics are added to `ui/scripts/arch-metrics.mjs`, so the ratchet tightens as each phase lands. Each starts at its current count and must reach 0 by the phase that names it.

- `raw_error_toasts`: `notify.error(` whose argument is `e.message`, `String(e)` or `.message`. Reaches 0 in phase 2.
- `window_dialogs`: `window.prompt` / `window.confirm` / `alert(`. Reaches 0 in phase 3.
- `empty_outside_loadstate`: imports of `Empty.svelte` from anywhere but `LoadState.svelte`, held by ESLint `no-restricted-imports`. Phase 2.
- `unguarded_mutations`: an `onclick={() => api.<mutating verb>` written inline instead of through `action()`. Reaches 0 in phase 3.
- `banned_terms`: the copy rules over `.svelte` templates and `notify` strings (`ui/scripts/copy-rules.test.ts`). Phase 3.
- The page files near the 600-line ceiling (`App.svelte`, Gallery, Models, Assets, Overview, Editor) are split by responsibility as each phase touches them: the page keeps loading and layout, and the logic moves to tested `.ts` modules.

Playwright **journey specs** pin the end-user stories the audit used, as regression gates: `e2e/journeys/*.spec.ts`. There is one per phase: editor safety, outage and token, first run, find-and-reuse, chain a step, review a request.

## Vocabulary (decisions for Don, marked ★)

| Concept | Term | Notes |
|---|---|---|
| One execution of a workflow | **job** | "take" only inside a request, meaning a job made for an item. "attempt", "result (latest)" and bare "run" as a noun are dropped. ★ |
| Version badge (v1/v2) | **v2**, with the tooltip "2nd job of this workflow in this folder" | It stays because `dw/runs.py` owns the number. |
| Workflow name | its **file name** (`e2e-card`) | The JSON `id` is shown only as secondary text on the job page. ★ |
| Favourite a file | **Star** | |
| Copy an output into the asset library | **Save as asset** | Replaces "Keep as asset", so "keep" no longer means two things. ★ |
| Choose a take as the item's result | **Accept** | Accepting also stars the file. ★ |
| Notes | **Your note** (the person's own) / **Note for the agent** | |
| Waiting for review | **Waiting for your OK** | The hold text says "the agent will run it at 4:44 PM if it's connected". |
| Starter workflows | **Examples** | Everywhere: the sidebar, "start from" and the read-only banner. |
| GPU | **card** stays in the queue text; the hardware line reads "GPU" | ★ |
| Approximate durations | **about 3 min** | No leading `~`, which reads as a minus sign in the monospace font. |

## Phases

Each phase is its own plan: `docs/superpowers/plans/2026-11-22-ui-quality-phase-N.md`. Only phase 1 is written now. Each later plan is written when the phase before it lands, against the code as it is then. Phases 4–7 depend only on phases 1–3 and can run in parallel.

| Phase | Theme | Building blocks | Findings (appendix IDs) | Journey gate |
|---|---|---|---|---|
| 1 | **Data safety** | conditional writes, `documentSave`, editor flush, load state and leave guard, confirm queue | WF-1..5, WF-16, XC-1, XC-2, XC-9, XC-15, GA-1..4, GA-11 (delete half), GA-13, GA-17 | `editor-safety.spec.ts` |
| 2 | **Connection, loading, errors** | `connection`, `ConnectionBanner`, `describeError`, `resource`/`LoadState`, `action()`/`ActionButton`, `ux-capture.mjs` | SH-1, SH-2, SH-14, XC-3..7, JB-1, JB-6, JB-7, JB-11, SH-15, GA-14 | `outage-and-token.spec.ts` |
| 3 | **Consistency and vocabulary** | `PageHeader`, `Details`, confirm/undo policy, glossary and copy ratchet, `SECTION_LABELS` | SH-10, SH-11, SH-13, SH-18, XC-10..13, XC-16, RQ-6, RQ-12, WF-15, WF-18, the glossary rows | the copy ratchet at 0 |
| 4 | **Results: find, understand, reuse** | `MediaViewer`, `OutputTile`, `RecipePanel` | GA-5..10, GA-11 (bulk star), GA-12, GA-15, GA-16, GA-L*, JB-2..4, JB-10, JB-14..16, RQ-5, SH-12, SH-5 | `find-and-reuse.spec.ts` |
| 5 | **Creating: run form, catalog, editor** | `ReferenceField` | WF-6..14, WF-17, WF-19..24, JB-5, JB-13 | `chain-a-step.spec.ts`, `first-image.spec.ts` |
| 6 | **Requests and agent collaboration** | the server's take error and summary fields | RQ-1..4, RQ-7..11, RQ-13..21 | `review-a-request.spec.ts` |
| 7 | **Shell, first run, performance** | the first-run card, gzip/caching, a loading shell, monaco trim, paged job history | SH-3, SH-4, SH-6..9, SH-16, SH-17, SH-19, SH-S*, XC-8, XC-14, JB-8, JB-9, JB-12, JB-17 | `first-run.spec.ts` |

## Decisions for Don

1. ★ The vocabulary rows above.
2. **The hold and notes (RQ-1).** Today any note cancels a held item's auto-run (`dw/ledger_views.py:_review`, the #1521 design). Proposal: keep the rule, warn before saving, and keep **Run now…** in the banner. Narrowing the rule to notes about the held item would change the agent contract and is a separate ruling.
3. **Uploads with no name (GA-4).** The UI will send each file's own name, create-only. The server's random-name default stays for callers that send no name. That leaves MCP unchanged.
4. **Update from GitHub (SH-S1).** It gets a confirm ("changes the server's Python environment and restarts the idle worker"). It doesn't move off the Models page.
5. **Requests made by hand (RQ-19).** Proposal: say plainly that requests are written by an agent, and build no "New request" UI. A by-hand request would be its own feature.
6. **Gate Run on a refused check (WF-8).** Proposal: when the run check says the run will be refused, Run becomes **Fix N issues**, which scrolls to them. The server stays the authority: the UI disables Run only on the check's own "refused" verdict.

## Appendix: findings by ID

The full consolidated text and repro steps are in `docs/superpowers/specs/2026-11-22-ui-audit.md` (its screenshots were session scratch and are not kept). Short form:

**SH (shell):**
- SH-1: the token takes no effect until reload
- SH-2: the header and popover are wrong without a token
- SH-3: loopback bind offers LAN URLs
- SH-4: the 390px overview overflows
- SH-5: workspace switch requests the old workspace's thumbnails
- SH-6: new workspace dead-end empty states
- SH-7: no first-run orientation
- SH-8: the collapsed sidebar can't switch workspaces
- SH-9: the new-workspace field has no buttons
- SH-10: the delete-workspace confirm wording
- SH-11: the Manage panel is all disabled controls
- SH-12: overview tiles link to the gallery root; jobs panel ids
- SH-13: the header meter has no state word; icon-only controls
- SH-14: popover and status page duplicates
- SH-15: short jobs get no finish toast
- SH-16: Models download field and mobile columns
- SH-17: the disabled quantized cache warns
- SH-18: the breadcrumb uses internal keys; bad section not corrected
- SH-19: switching workspace keeps no section
- SH-S1: Update from GitHub has no confirm
- SH-S2: "+ new" hidden with no reason
- SH-S3: the theme cycle is unlabelled
- SH-S4: `dw-serve` vs `python -m dw.serve`
- SH-S5: MCP snippet always carries the token header
- SH-S6: no global shortcuts; `?` not advertised

**WF (workflows and editor):**
- WF-1: JSON Cmd+S saves the old document
- WF-2: switching from JSON view drops the edit
- WF-3: new-name save overwrites
- WF-4: the sidebar leave loses edits
- WF-5: the blank starter saves unrunnable
- WF-6: no `previous_result:` in media fields
- WF-7: renaming a step breaks references
- WF-8: Run is enabled when refused; unreadable refusal
- WF-9: misleading seed warning
- WF-10: gated model "admitted"
- WF-11: no upload or browse in the run form; kind filter
- WF-12: run-form values lost
- WF-13: catalog substring search, count, URL
- WF-14: cards "N vars"
- WF-15: cost wording; the `~` glyph
- WF-16: URL stays `/edit` after first save
- WF-17: variable Enter, name validation, types
- WF-18: no undo for removing a step or variable
- WF-19: component slots are static
- WF-20: LoRA rows unlabelled
- WF-21: engine text on the run page; no Essentials/Advanced
- WF-22: no read-only banner
- WF-23: validation errors not linked
- WF-24: no Enhance on the run form

**JB (jobs):**
- JB-1: stuck "running" after restart (SSE)
- JB-2: results below the fold
- JB-3: errors buried and not actionable; failed step not red
- JB-4: list rows indistinguishable; filter
- JB-5: rerun-with-changes not prefilled; new seed; types
- JB-6: double-click rerun
- JB-7: cancel has no handling or confirm
- JB-8: queued row breaks the grid
- JB-9: empty agent-notes box
- JB-10: workflow stale after finish; seed chip
- JB-11: toasts miss short jobs; no Open
- JB-12: queued text reads as a log
- JB-13: author warnings shown to the person running it
- JB-14: event log raw JSON, ETA seconds, whole-job estimate
- JB-15: layout leftovers
- JB-16: missing actions (compare, keep, delete, bulk, grouping)
- JB-17: unbounded jobs poll

**GA (gallery, assets, prompts):**
- GA-1: prompt Cmd+S
- GA-2: new prompt overwrites
- GA-3: upload overwrites
- GA-4: batch upload hex names
- GA-5: folder collapse no-op
- GA-6: tall image covers the grid
- GA-7: small viewer
- GA-8: recipe hidden and forgotten
- GA-9: search by path only; no sort or date
- GA-10: keep means two things; no reuse path
- GA-11: bulk delete ignores stars; no bulk star
- GA-12: the asset grid loads originals
- GA-13: cancelling Replace errors
- GA-14: failed metadata looks like none
- GA-15: video holds focus
- GA-16: Compare URL and empty state
- GA-17: unsupported drops are uploaded
- GA-L*: the low-severity list (crops, audio/text tiles, star wrap, filter width, Compare tools, asset detail, prompt cards, single-delete confirm, picker gaps)

**RQ (requests):**
- RQ-1: a note cancels the hold silently
- RQ-2: needs-you counts blocked failures
- RQ-3: no failure reason on takes
- RQ-4: the list hides attention
- RQ-5: takes too small to compare
- RQ-6: Mark done and Abandon have no confirm or undo
- RQ-7: closed request still editable
- RQ-8: agent edits arrive invisibly
- RQ-9: plan changes show no diff
- RQ-10: the run dialog shows engine text
- RQ-11: failed items excluded from "ready"; vocabulary
- RQ-12: `~` glyph
- RQ-13: mid-width row layout
- RQ-14: repeated note chrome
- RQ-15: the item menu clips
- RQ-16: Accept looks disabled
- RQ-17: revised rows collapsed; needs line not clickable
- RQ-18: reorder ignores dependencies
- RQ-19: no by-hand request
- RQ-20: note lifecycle
- RQ-21: minor items (text cover, filter chip id, chip styles)

**XC (cross-cutting):**
- XC-1: save after a failed load overwrites
- XC-2: the leave guard is breadcrumb-only
- XC-3: no global offline or token state
- XC-4: token not verified or reloaded
- XC-5: pages don't recover
- XC-6: Status page blames the token
- XC-7: loading shown as empty
- XC-8: compression, monaco, loading shell
- XC-9: two-tab overwrite
- XC-10: a11y (skip link, titles and focus, aria-pressed, drawer focus, dialog names)
- XC-11: Back loses scroll and gallery filters
- XC-12: unknown routes
- XC-13: the finish toast names the id; no Open
- XC-14: polling (Status page double poll)
- XC-15: a second confirm hangs the first
- XC-16: empty Compare and Examples copy
