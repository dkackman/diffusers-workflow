# Review Hold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** An agent can hold planned request items for the user's review until a deadline; the request page shows the hold and a countdown; a note left before the deadline redirects the agent; silence lets the agent run the items at the deadline (#1521).

**Architecture:** The hold is two additive item fields in `requests.json` (`dw/ledger.py`). Its state (`holding`, `redirected`, `due`) is derived on the server at read time (`dw/ledger_views.py`), so the UI and the notes channel read one answer and there is no twin rule. Nothing on the server runs a job at the deadline: the notes channel process (`dw_mcp/ledger.py`) announces a `review_due` event and the agent session runs the items through `run_workflow` with the cost it already quoted. The request page's existing priced Run dialog (shipped in 0.21.0, 7d091d81) is the person's own way to run them early.

**Tech Stack:** Python 3 / FastAPI / pydantic (server), `dw_mcp` (MCP + channel), Svelte 5 + TypeScript + vitest + Playwright (UI), pytest.

**Spec:** issue #1521, plan v1 and Don's approval of its six defaults (2026-11-21). This file is the binding detail.

## Global Constraints

- No new Python module (the module baseline stays 278), no new REST route, no new MCP tool or tool parameter. No server timer, thread or startup hook that submits a job.
- `dw/ledger.py` must stay at or under 1100 lines (it is 1062). `dw/server/api_models.py` is not touched. UI source files stay under 600 lines. If a ceiling cannot be met, report BLOCKED; do not split a file or add a module.
- Item fields, exact names: `review_until` (writable), `review_set_at` (server-stamped, ignored on write), `review` (derived, ignored on write).
- `MIN_REVIEW_SECONDS = 60`, `MAX_REVIEW_HOURS = 24`, both constants in `dw/ledger.py`.
- `review` is `{until, state, seconds_left}`; `state` is exactly one of `holding`, `redirected`, `due`. It is present only when the item's derived status is `planned`, the request's state is `open`, and `review_until` is set; otherwise the key is absent.
- State order, first that holds: `redirected` (a note on the item, or one of the request's own notes, has `at >= review_set_at`), `due` (now >= `review_until`), `holding`. Only a note redirects.
- `seconds_left` is a whole number, never negative, computed on the server at read time.
- Channel event kind, exact: `review_due`. Its `meta` values are all strings.
- The page never starts a held item on its own. The only control that spends GPU time is the existing Run dialog.
- Commands run from the worktree root with the main checkout's interpreter: `/Users/don/src/dkackman/diffusers-workflow/venv/bin/python -m pytest tests/<file> -q -n0`. Before each Python commit: `venv-python -m ruff format --check dw dw_mcp tests scripts` and `venv-python -m ruff check dw dw_mcp tests scripts`. UI: `cd ui && npx vitest run <path>`, and `npm run check && npm run lint` before each UI commit.
- Stage files by name (`git add <path>`); never `git add -A`. Commit messages follow the repo's `feat(scope): ...` style and must not contain `#1521` after `fix:` or `closes` wording (use `refs #1521`).
- Prose in docs, skills, UI strings and docstrings: plain sentences, no em dashes, matching the surrounding file.

## Review Focus

1. An agent answers a note with `set_request` but echoes the old, now past `review_until`: the item must not become runnable without review. The echo is accepted (unchanged value), `review_set_at` is kept, and the state stays `redirected` because the note's `at` is still >= `review_set_at`. Test in Task 1.
2. A plan read with GET and written back unchanged by the UI (`by: user`) keeps `review_until` and `review_set_at` exactly. Test in Task 1.
3. A request whose revision and `updated_at` do not change must still produce `review_due` once its deadline passes. Test in Task 3.
4. A held item whose dependency is unfinished at the deadline is announced when it becomes ready, not before. Test in Task 3.
5. A hold already due when the channel connects is announced as stale, with the confirm-before-spending wording. Test in Task 3.

---

### Task 1: Ledger fields, derived `review`, REST models, generated types

**Files:**
- Modify: `dw/ledger.py`, `dw/ledger_views.py`, `dw/server/api_models_library.py`
- Modify (generated): the OpenAPI document and `ui/src/lib/api` types, regenerated with `python scripts/dump_openapi.py` and `cd ui && npm run gen:api` (run `npm ci` in `ui/` first if `ui/node_modules` is missing; find the exact script names in `ui/package.json` and `docs/RELEASING.md` if these differ)
- Modify: `ui/e2e/serve_fixture.py` so one planned, ready item of the fixture's plan carries `review_until` one hour ahead (Task 2's e2e reads it)
- Create: `tests/test_ledger_review.py`
- Modify: an existing REST test file for the requests routes (the style of `tests/test_ledger_notes_api.py`) for one round trip

**Write side (`dw/ledger.py`):**
- Add `review_until`, `review_set_at`, `review` to `_ITEM_FIELDS`.
- `_item(...)` stores `review_until` and `review_set_at`:
  - `raw["review_until"]` absent or `None` stores both as `None`.
  - Otherwise it must be a string that `datetime.fromisoformat` parses and that carries a UTC offset; anything else is refused. Normalize to UTC, whole seconds, `isoformat()` (the form `_now()` writes).
  - If the normalized value equals `previous.get("review_until")`: keep it and keep `previous.get("review_set_at")`, whatever the clock says.
  - If it differs: it must be at least `MIN_REVIEW_SECONDS` after `at` and at most `MAX_REVIEW_HOURS` after `at`; store it and set `review_set_at = at` (the write's own stamp, already passed to `_item`).
  - `raw["review_set_at"]` and `raw["review"]` are never read.
- Refusals are `LedgerError` with `status=422` and codes `invalid_review_until` (not a string, unparseable, no offset), `review_too_soon`, `review_too_far`. Messages name the item key and the bound, in the file's existing voice.
- Dropped items keep whatever they had stored; no special handling.

**Read side (`dw/ledger_views.py`):**
- `request_view(request, jobs=(), now=None)`: `now` is an aware `datetime` for tests, defaulting to the current UTC time. `item_view` gains what it needs to compute `review` (pass the request's own notes, the request state and `now`; keep its existing positional callers working).
- `review` per the Global Constraints. A note counts when its `at` parses and is `>=` `review_set_at`; compare as datetimes, not strings. Notes of dropped items do not exist on a live item, so only the item's own notes and the request's own notes are read. A missing `review_set_at` on a stored hold (hand-edited file) means no note redirects.
- `summary_view` is unchanged.

**API models (`dw/server/api_models_library.py`):**
- `LedgerReview` (`until: str`, `state: Literal["holding","redirected","due"]`, `seconds_left: int`), and on `LedgerItem`: `review_until: str | None`, `review_set_at: str | None`, and `review` declared with the file's `sometimes(...)` helper (absent when no hold applies). Descriptions say what each is in one sentence.
- `RequestWrite.items` description names `review_until`.

**Tests (`tests/test_ledger_review.py`), each a separate test:**
- setting a future `review_until` stamps `review_set_at` to the write's time and reads back `review.state == "holding"` with `seconds_left` matching an injected `now`;
- an input with a non-UTC offset is stored normalized to UTC;
- unchanged echo keeps `review_set_at` (write twice, second with a later clock);
- a changed value restamps `review_set_at`;
- refusals: not a string, unparseable, no offset, 59 s ahead, 24 h + 1 s ahead, each with its code and status 422;
- exactly 60 s ahead and exactly 24 h ahead are accepted;
- `null` clears both fields and `review` is absent;
- `redirected` by an item note saved after the hold; by a request note saved after the hold; not by a note saved before the hold;
- deleting the redirecting note returns the state to `holding`;
- `due` with injected `now` at and after the deadline; `seconds_left == 0`;
- `review` absent when the item has a queued job, is dropped, or the request is `done`;
- Review Focus 1 and 2, as written above;
- a client-sent `review_set_at` or `review` is ignored, not stored and not refused.
- REST: `PUT` then `GET` a request with a hold through the test client and assert the three fields on the item; a 59 s hold answers 422 with code `review_too_soon`.

**Gate:** the new tests, plus `tests/test_ledger*.py`, the response-contract and OpenAPI-pin tests (find them with `grep -rl openapi tests | head`), and the architecture ratchet test (`grep -rl "modules" tests/test_arch* scripts | head`) pass; ruff clean; `wc -l dw/ledger.py` <= 1100. In `ui/`: `npm run check` passes with the regenerated types.

**Commit:** one or two commits, e.g. `feat(ledger): a planned item can be held for review until a deadline (refs #1521)`.

---

### Task 2: The request page shows the hold and its countdown

**Files:**
- Create: `ui/src/lib/request/review.svelte.ts`, `ui/src/lib/request/review.svelte.test.ts`, `ui/src/lib/request/reviewText.ts`, `ui/src/lib/request/reviewText.test.ts`
- Modify: `ui/src/lib/request/itemState.ts`, `RequestHeader.svelte`, `RequestItem.svelte`, `runPlan.ts`, `RunDialog.svelte`, the request page that wires them, their tests, `ui/src/lib/request/fixtures.ts` / `fakeLedger.ts` as needed, `ui/e2e/requests.spec.ts`
- Read first: `ui/CLAUDE.md`, `listening.svelte.ts` (the pattern to copy for a shared, visibility-aware timer), `announce.ts`, `noteState.ts`, `runPlan.ts`, `scripts/design-rules.test.ts`

**Behaviour:**
- `review.svelte.ts`: one shared one-second ticker, running only while a hold is on screen and the tab is visible (the `hold()` pattern of `NotesChannel`). A countdown is anchored to the moment the request was read plus the server's `seconds_left`; the browser's wall clock is never compared with `review.until`. When a countdown reaches zero the page re-reads the request once (through the page's existing refresh path) so the server's `due` state replaces the local guess.
- `reviewText.ts`: pure functions returning the strings below, so they are tested without a component.
- `itemState.ts`: each live row exposes its `review` (or null); the plan state exposes the held rows. The state shown is always the server's `review.state`.
- `RequestHeader.svelte`: a hold banner next to the notes, shown when any live row has a `review`. One banner for the request, worded by the most urgent state present (`due`, then `holding`, then `redirected`) and by `listening` from `listening.svelte.ts`. `N` counts the rows in that state; the minutes are the sum of those rows' `estimate_minutes`, omitted when none has one. Exact strings:
  - holding, listening: `N items held for your review (about M min by the agent's estimate). A session is listening: they run at HH:MM (in MM:SS) unless you leave a note.` followed by a `Run now…` button
  - holding, not listening: `N items held for your review. No session is listening, so nothing will run at HH:MM. Run them here, or leave a note.` followed by `Run now…`
  - redirected: `You left a note after these were held: the agent will revise before running.`
  - due, listening: `Review time is up: the agent should be starting these.`
  - due, not listening: `Review time passed with no session listening: nothing ran.` followed by `Run now…`
  - Singular forms where N is 1 (`1 item held`, `it runs at`, `Run it here`). `HH:MM` is the deadline in the browser's locale time; `MM:SS` becomes `H:MM:SS` past an hour.
- `Run now…` opens the existing Run dialog for the held rows that are `ready`; it is absent when none is ready.
- `RequestItem.svelte`: a chip on a held row: `held · MM:SS`, `held · note left`, or `review time up`.
- `runPlan.ts` / `RunDialog.svelte`: a row in the dialog for a held item carries the line `Held for review until HH:MM. Running it now is your go-ahead.`; when its state is `due` and a session is listening, the line is `Review time is up and a session is listening: it may be starting this now.` in the warn tone.
- Accessibility: the banner's state changes are announced once through `announce.ts`; the ticking digits are not in a live region. Follow whatever `scripts/design-rules.test.ts` requires for a live region and for the button.
- Nothing on the job page or the gallery changes.

**Tests:**
- vitest: countdown anchoring and the zero-crossing refresh under fake timers; the ticker stops while the tab is hidden and when the last holder releases; every banner string above, singular and plural, with and without estimates; chips; the dialog lines; one announcement per state change and none per tick; `Run now…` absent when no held row is ready.
- e2e (`ui/e2e/requests.spec.ts`, using the fixture hold from Task 1): the held item shows the banner and a running countdown; `Run now…` opens the dialog with the held line; saving a note on the held item turns the banner to the redirected string after the refresh.
- Do not use `vi.resetModules` in Svelte component tests.

**Gate:** `npx vitest run` (whole UI suite), `npm run check`, `npm run lint`, the UI architecture metrics script if `ui/package.json` has one, and the e2e spec `npx playwright test e2e/requests.spec.ts` (if Playwright cannot start in this environment, say so in the report with the error; do not skip silently).

**Commit:** `feat(ui): the request page shows items held for review, with a countdown and a run-now (refs #1521)`.

---

### Task 3: The notes channel announces a due hold

**Files:**
- Modify: `dw_mcp/ledger.py`, `dw_mcp/server.py` (`NOTES_INSTRUCTIONS` only), `dw_mcp/tools_authoring.py` (`set_request` docstring only)
- Modify: `tests/test_mcp_notes_channel.py` (or the file that tests `NoteWatch`; find it with `grep -rl NoteWatch tests`), `tests/test_mcp_server.py` only if a pinned text needs its expectation updated
- Read first: `dw_mcp/CLAUDE.md`

**Behaviour (`dw_mcp/ledger.py`):**
- `due_reviews(request)`: the live items of an open request whose `review.state == "due"` and whose `ready` is true, in plan order, as `{request, revision, item, title, until, estimate_minutes}`.
- `review_event(due, workspace, stale=False)`: one event for one request's due items.
  - Content, fresh: a head line `Review time is up on request "<id>" with no note from the user. Held items now due:`, one line per item `- <key>: <title> (about <M> min)` (the bracket omitted without an estimate), then: `Read the request with get_request("<id>", workspace="<ws>"). Run only the items still ready and without a new note, each with run_workflow(request="<id>/<key>", acknowledged_cost=<the plan you quoted>). If a run is refused for a changed plan, or validate_workflow now quotes a figure well off what you told the user, do not run it: set a new review_until with the new quote in the why.`
  - Content, stale: the head line is `Review time passed on request "<id>" before this session connected. Held items:` and the last paragraph is `These holds were set before this session, so the user's go-ahead may be stale. Tell the user what is waiting and confirm before spending GPU time.`
  - `meta`: `kind="review_due"`, `workspace`, `request`, `revision`, `items` (comma-joined keys), `until` (the earliest), and `stale="1"` only when stale. All values strings.
- `NoteWatch`:
  - takes an injectable monotonic clock (`clock=time.monotonic`) for tests;
  - the per-request stamp becomes `(revision, updated_at, ready)` using the listing row's `ready` count;
  - after reading a request it records when its earliest `holding` item's `seconds_left` runs out (on the monotonic clock); a request is re-read when its stamp changed **or** that moment has passed;
  - due items are announced once per `(item, until)`; a new `review_until` on the same item is announced again; the bookkeeping for a request is dropped when it leaves the open list, as for notes;
  - on the first poll that reaches the server, due items go out as one stale `review_event` per request, after the notes digest; later polls send fresh ones, after any note events of the same poll.
- `NOTES_INSTRUCTIONS`: add one or two sentences: a `kind="review_due"` event means a hold the agent set has reached its deadline with no note, and the agent runs the named items now under the usual cost gate, unless the event says the hold is stale, in which case it confirms first. Keep any length limit the tests pin.
- `set_request` docstring: add `review_until` to the item fields and one clause: `review_until: an ISO time until which a planned item is held for the user's review`. The existing guide pointer stays last. `SURFACE_BUDGET` and `CLIENT_TEXT_LIMIT` tests must pass; if the budget fails, tighten this docstring's own wording, and report what you cut.

**Tests:** fires once when due and ready; not while `holding`; not when `redirected`; Review Focus 3 (stamp unchanged, clock advanced past the recorded moment, the request is re-read and the event sent); Review Focus 4 (due but not ready sends nothing; the listing's `ready` count rises, the event is sent); a new `review_until` fires again; Review Focus 5 (first poll, stale wording and `stale="1"`); notes and due items in one poll keep notes first; every `meta` value is a `str`; a request that closes drops its bookkeeping. Plus the existing channel, server-text and guide-pointer tests (`tests/test_mcp_server.py`, `tests/test_guide_pointers.py`).

**Gate:** those tests and ruff.

**Commit:** `feat(mcp): the notes channel says when a review hold is due (refs #1521)`.

---

### Task 4: Skill, guides, map and release note

**Files:**
- Modify: `plugins/dw/skills/script-to-video/references/request.md`, `tests/test_plugin_skills.py` (`TestSkillRequests` pins for the new section)
- Modify: `docs/WORKSPACES.md` (Requests), `docs/WORKFLOW_GUIDE.md` (the cost-gate step that says to quote `plan.estimate`), `docs/MCP.md` (notes channel), `docs/SERVER.md` (request item fields), `docs/ARCHITECTURE.md`, `docs/RELEASING.md`
- Read first: `plugins/dw/README.md` (how skills are pinned and versioned; bump what it says to bump), the surrounding text of each section edited

**Content:**
- `request.md`, a new section `## Hold items for the user's review`, between *Run each item* and *Accept the keeper*:
  - when: the plan is written and priced, and the user wants to look at it on the request page before it runs;
  - the user's go-ahead comes first: quote the total in chat and get a yes to "these run at T unless you leave a note". Silence on the page confirms that yes; it never replaces it;
  - hold only when the dw-notes channel is connected in this session (its instructions are in your context); without it nothing will tell you the deadline passed, so run on the user's word as before;
  - set the hold with `set_request`: each held item gets `review_until`, an ISO time with an offset, 15 minutes ahead unless the user says otherwise (the server refuses under 60 seconds or over 24 hours), and the `why` says so;
  - then end the turn. Do not block in `wait_for_job` on something else; the event is what wakes the session;
  - on a note event: do the revision, write the plan, and give the revised items a new `review_until` if the user should look again;
  - on a `review_due` event: `get_request`, run only the named items that are still `ready` with no newer note, each with `run_workflow(request=..., acknowledged_cost=...)` for the plan you quoted. A refusal for a changed plan, or a quote well off what you told the user (more than a quarter), means a new hold with the new figure, not a run;
  - an item held behind a dependency is announced when the dependency finishes; you run it then. The server never starts a job for you;
  - a `review_due` event marked stale is from before this session: confirm with the user first.
  Update the *When a note arrives mid-session* section only where it must mention the new event kind. Add pins in `TestSkillRequests` for the section's load-bearing phrases, in that class's existing style (a pinned phrase must not straddle a line wrap).
- `docs/WORKSPACES.md` Requests: the three fields, the three states and their order, the two bounds, that only a note redirects, and that the server runs nothing at the deadline.
- `docs/WORKFLOW_GUIDE.md`: a short paragraph at the cost-gate step: a hold defers the run the user approved; no note by the deadline permits exactly the run that was quoted and nothing else; pointer to the WORKSPACES section.
- `docs/MCP.md` notes channel: the `review_due` kind, its `meta`, the stale form, and that the page shows whether a session is listening.
- `docs/SERVER.md`: the item fields on the request routes, and the request page's hold banner in the web UI section if that section describes the page.
- `docs/ARCHITECTURE.md`: update the rows that own the requests ledger, the notes channel and the request page's run dialog to name the hold, its owning modules and the tests that enforce it. Also fix the row that says the UI never sends `acknowledged_cost` (the Run dialog and the restart banner do; say which surfaces send it).
- `docs/RELEASING.md`: under the next release's notes, a user-facing paragraph and a `- REST:` line naming `review_until`, `review_set_at` and `review` on the request item, in the style of the 0.21.0 entries.

**Gate:** `tests/test_plugin_skills.py`, `tests/test_guide_pointers.py`, any docs tests that pin these files (`grep -rl "WORKSPACES.md\|ARCHITECTURE.md" tests | head`), ruff on `tests`.

**Commit:** `docs: the review hold in the request skill, the guides, the map and the release notes (refs #1521)`.
