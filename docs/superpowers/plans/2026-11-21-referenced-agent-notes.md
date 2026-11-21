# Referenced Notes for the Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace each request's and item's single `comment` with a list of user notes, each optionally referencing a linked job and one of its output files, writable from the request page, the job page and the gallery, and announced with the reference by the notes channel.

**Architecture:** Notes stay in `requests.json` (`dw/ledger.py`). New ledger functions add, edit and delete one note under the existing `_writing` lock; the whole-plan PUT stops carrying notes and copies the stored ones forward. A new `agent_wrote_at` request field replaces the history scan in the waiting rule. New HTTP routes under `/api/workspaces/{name}/request-notes` feed three UI surfaces; `dw_mcp` reads `notes` and the channel dedupes by note id.

**Tech Stack:** Python 3 / FastAPI / pydantic (server), `dw_mcp` (MCP + channel), Svelte 5 + TypeScript + vitest (UI), pytest.

**Spec:** `docs/superpowers/specs/2026-11-21-referenced-agent-notes-design.md`

## Global Constraints

- Notes are the user's: no `dw_mcp` tool writes them, and every whole-plan PUT (agent or user) keeps the stored notes whatever it sends.
- Note text: `clean_note` rules (`dw/notes.py`), 4000 characters.
- `MAX_NOTES = 20` live notes per target (an item, or the request itself).
- Note id: `"n<k>"`, `k` from a per-request counter `next_note` (stored, starts at 1, never reused, even after a delete).
- A note is `{id, text, created, at, ref?}`: `created` is set once at add; `at` changes with the text; the waiting rule reads `at`.
- `LISTENING_WINDOW = 30` seconds; the UI asks every 15 seconds. Save labels: **Send to agent** when a session is listening, **Save note** when not; saving is never disabled.
- `ref` is `{job, output?, kind?}`; `job` required when `ref` is present; `kind` is stored by the server, never taken from the caller. `ref` never changes after the note is written.
- References name a job id and a manifest output name (relative, as `proof_output` returns it), never an absolute path.
- Every surface labels the new note "Note for the agent". The existing run/file note (`set_note`, `notes.json`) is untouched.
- No compatibility step: `_load` converts a stored `comment` to `notes` on read; `comment` is never written again.
- Run `pytest tests/<file> -q` for Python, `cd ui && npx vitest run <path>` for UI, and `cd ui && npm run check && npm run lint` before each UI commit.

## Review Focus

1. A note referencing a take whose run was later deleted: the view shows `ref.gone: true`, the page greys the chip, and the channel event says "(that job's run no longer exists)". Test in Task 2 and Task 5.
2. A request-page tab with an old revision saves the plan after a note was added from the gallery: the note survives. Test in Task 1.
3. Saving a note's text unchanged keeps its `at`, so it is not re-announced. Test in Task 2.
4. A `ref.output` that is absolute, contains `..`, or is not in that job's manifest is refused with 422 `invalid_note_ref`. Test in Task 3.
5. A note on a closed request is accepted and stored but never waits or gets announced. Test in Task 4.

---

### Task 1: Ledger stores notes; whole-plan PUT keeps them; `agent_wrote_at`

**Files:**
- Modify: `dw/ledger.py` (`_ITEM_FIELDS` :88, `_load` :198, `_item` :829, `_BODY_FIELDS` :715, `_check_body` :720, `put_request` :927, `request_view` :405; remove `_comment` :604)
- Create: `tests/test_ledger_notes.py`
- Modify: `tests/test_ledger.py` (comment tests move to the new file or are deleted where the rule no longer exists)

**Interfaces:**
- Produces: stored item and request carry `notes: list[dict]`; request carries `next_note: int` and `agent_wrote_at: str | None`. `request_view` returns `notes` on the request and every item (never `comment`), and `agent_wrote_at`.
- Produces: `_upgrade(request: dict) -> dict` in `dw/ledger.py`, called by `_load` per request.

- [ ] **Step 1: Write failing tests in `tests/test_ledger_notes.py`** (reuse `plan`, `item`, `write` from `tests/test_ledger.py`)

```python
def test_a_stored_comment_is_read_as_one_note(tmp_path):
    # write a requests.json by hand with item a's comment {text:"warmer", at:T}
    got = ledger.get_request(str(tmp_path), "cats")
    assert got["items"][0]["notes"] == [
        {"id": "n1", "text": "warmer", "created": T, "at": T}
    ]
    assert "comment" not in got["items"][0]
    assert got["next_note"] == 2  # request's own comment, if any, is n1 first

def test_agent_wrote_at_is_filled_from_history_on_read(tmp_path): ...
    # history with agent entry at T1, user at T2 -> agent_wrote_at == T1

def test_an_agent_write_sets_agent_wrote_at(tmp_path): ...
def test_a_user_write_leaves_agent_wrote_at(tmp_path): ...

def test_a_plan_write_keeps_stored_notes_whatever_it_sends(tmp_path):
    # store a note on item a, then write(plan(..., items=[item("a", notes=[])]), by="user")
    # and by="agent" -> notes unchanged both times

def test_a_plan_write_carrying_comment_is_refused(tmp_path):
    # "comment" is no longer a body or item field -> LedgerError invalid_request
```

- [ ] **Step 2: Run** `pytest tests/test_ledger_notes.py -q` — expect failures (`KeyError: 'notes'`, no `agent_wrote_at`).

- [ ] **Step 3: Implement in `dw/ledger.py`**
  - `_upgrade(request)`: when `comment` is present on the request or an item, move it to `notes` (request first, then items in order, ids `n1`, `n2`, …), set `next_note`; fill `agent_wrote_at` from the latest `by == "agent"` history entry when the key is absent. Idempotent.
  - `_ITEM_FIELDS`: replace `"comment"` with `"notes"` (accepted and ignored, so a GET body can be PUT back). `_BODY_FIELDS`/`_check_body`: drop `comment`; accept and ignore `notes`, `next_note`, `agent_wrote_at`.
  - `_item(...)`: copy `previous.get("notes", [])`. `put_request`: copy the request's stored `notes` and `next_note`; set `agent_wrote_at = at` when `by == "agent"`, else keep the stored one.
  - `request_view`: `notes` default `[]` on request and items; include `agent_wrote_at`.

- [ ] **Step 4: Update `tests/test_ledger.py`**: delete tests of `_comment` behaviour that no longer exists (agent/user comment writes); keep everything else passing.

- [ ] **Step 5: Run** `pytest tests/test_ledger.py tests/test_ledger_notes.py tests/test_ledger_integrity.py tests/test_ledger_job_link.py -q` — expect PASS.

- [ ] **Step 6: Commit** `feat(ledger): notes list per item and request, agent_wrote_at; plan writes keep notes`

---

### Task 2: Ledger note writes and ref view

**Files:**
- Modify: `dw/ledger.py` (new functions after `put_request`; `item_view` :357)
- Test: `tests/test_ledger_notes.py`

**Interfaces:**
- Consumes: Task 1's stored shape.
- Produces:
  - `MAX_NOTES = 20`
  - `add_note(root: str, request_id: str, *, item: str | None, text: str, ref: dict | None, check_ref: Callable[[str, str | None, dict], dict]) -> dict` returns `{"note": <note>, "revision": int}`. `check_ref(request_id, item, ref)` returns the ref to store (`{job, output?, kind?}`) or raises `LedgerError(code="invalid_note_ref", status=422)`; the ledger reads no job store.
  - `edit_note(root, request_id, note_id: str, text: str) -> dict` same return.
  - `delete_note(root, request_id, note_id: str) -> dict` returns `{"revision": int}`.
  - Errors: `unknown_request` 404, `unknown_note` 404, `unknown_item` 422, `item_dropped` 422, `too_many_notes` 422.
  - `item_view` / `request_view` enrich each note's `ref` with `attempt: int | None` (1-based position of `ref.job` in the item's `attempts`) and `gone: True` when the job is not linked any more or its `run_exists` is false (key omitted otherwise).

- [ ] **Step 1: Write failing tests**

```python
def test_add_note_on_an_item_with_a_ref(tmp_path):
    out = ledger.add_note(root, "cats", item="a", text="lipsync off",
                          ref={"job": "j1", "output": "w/r1/final/a.mp4"},
                          check_ref=lambda rid, it, ref: {**ref, "kind": "video"})
    assert out["note"]["id"] == "n1" and out["note"]["ref"]["kind"] == "video"
    assert out["revision"] == 2
    got = ledger.get_request(root, "cats")
    assert got["history"][-1]["by"] == "user"
    assert got["history"][-1]["why"] == "note added"

def test_add_note_without_item_goes_on_the_request(tmp_path): ...
def test_note_ids_are_never_reused_after_delete(tmp_path): ...  # n1 deleted, next is n2
def test_edit_note_with_same_text_keeps_at(tmp_path): ...        # Review Focus 3
def test_edit_note_with_new_text_gets_new_at(tmp_path): ...      # and created is unchanged
def test_add_note_sets_created_equal_to_at(tmp_path): ...
def test_the_21st_note_on_a_target_is_refused(tmp_path): ...     # code too_many_notes
def test_a_note_on_a_dropped_item_is_refused(tmp_path): ...      # code item_dropped
def test_a_failed_check_ref_stores_nothing(tmp_path): ...        # revision unchanged
def test_view_marks_ref_attempt_and_gone(tmp_path):
    # jobs_for returns j1 (run_exists False), j2 (True) on item a
    # note ref j2 -> attempt 2, no "gone"; note ref j1 -> attempt 1, gone True;
    # note ref j9 (not linked) -> attempt None, gone True        # Review Focus 1
```

- [ ] **Step 2: Run** `pytest tests/test_ledger_notes.py -q` — expect FAIL (`AttributeError: add_note`).

- [ ] **Step 3: Implement** the three functions in `dw/ledger.py`, each inside `with _writing(root):` — `_load`, find the request, apply, bump `revision`, append a history entry via a shared helper (`why` is exactly `"note added"`, `"note edited"` or `"note removed"`, `by: "user"`, no goal/state keys), apply the `MAX_FILE_BYTES` check as `put_request` does, `write_json_atomic`. Text goes through `clean_note`; a cleared (None) text on edit is refused (use delete).

- [ ] **Step 4: Run** `pytest tests/test_ledger_notes.py tests/test_ledger.py -q` — expect PASS.

- [ ] **Step 5: Commit** `feat(ledger): add, edit and delete one note, with ref attempt and gone in the view`

---

### Task 3: HTTP routes, ref checking, API models, gallery link

**Files:**
- Modify: `dw/server/routes/library.py` (new routes next to `put_request` :1016)
- Modify: `dw/server/api_models_library.py` (replace `LedgerComment` :356 with `LedgerNote`, `LedgerNoteRef`; `LedgerItem.comment` :399 and `LedgerRequest.comment` :448 become `notes`; drop `RequestWrite.comment` :511; add `NoteAdd`, `NoteEdit`, `NoteWritten`)
- Modify: `dw/server/api_models.py` (exports :68, :124; `JobRef` :986 untouched; gallery metadata model gains `linked_jobs`)
- Modify: `dw/server/job_history.py` (new `jobs_publishing_file`)
- Modify: `dw/server/routes/media.py:106` (gallery metadata)
- Create: `tests/test_ledger_notes_api.py`
- Modify: `tests/test_api_contract.py` if it lists routes the UI calls

**Interfaces:**
- Consumes: Task 2's `add_note`, `edit_note`, `delete_note`.
- Produces (routes, all user-only — they take no `by` and record `by: "user"`; no `dw_mcp` tool calls them):
  - `POST /api/workspaces/{name}/request-notes` body `NoteAdd {request: str, item: str | None, text: str, ref: {job: str, output: str | None} | None}` → `NoteWritten {note: LedgerNote, revision: int}`
  - `PUT /api/workspaces/{name}/request-notes` body `NoteEdit {request, note, text}` → `NoteWritten`
  - `DELETE /api/workspaces/{name}/request-notes?request=&note=` → `{revision: int}`
  - (A separate path, because `/requests/{request_id:path}` would swallow a `/notes` suffix.)
- Produces: `JobHistory.jobs_publishing_file(file_name: str, workspace: str) -> list[dict]` — every job whose manifest lists the file (`reused` entries included) and that has `request` set, as `{id, request, item}`, newest first.
- Produces: gallery metadata field `linked_jobs: list[{id, request, item, attempt}]` (`attempt` from the order of `manager.request_jobs(workspace, request)` filtered to that item, 1-based).

- [ ] **Step 1: Write failing tests in `tests/test_ledger_notes_api.py`** (use the `TestClient` + `ScriptedWorkerManager` setup from `tests/test_ledger.py`; run one scripted job linked to `cats/a`)

```python
def test_post_a_note_with_a_ref_to_a_linked_job_output(client): ...
    # 200; note.ref == {job, output: <proof name>, kind: "image"}
def test_post_a_ref_to_a_job_not_linked_to_the_item_is_422(client): ...   # invalid_note_ref
def test_post_a_ref_output_not_in_the_manifest_is_422(client): ...
def test_post_a_ref_output_absolute_or_dotdot_is_422(client): ...         # Review Focus 4
def test_put_and_delete_a_note(client): ...
def test_a_plan_put_does_not_change_notes(client): ...                    # body with notes:[] -> notes kept
def test_gallery_metadata_lists_linked_jobs_including_reuse(client): ...  # second job reusing the file
```

- [ ] **Step 2: Run** `pytest tests/test_ledger_notes_api.py -q` — expect FAIL (404 on the new route).

- [ ] **Step 3: Implement** the routes. `check_ref` (route-level closure): the job must be in `manager.request_jobs(name, request)` with `item` equal to the note's item (for a request-level note, any item of that request); when `output` is given it must be one of the names `outputs._shown` yields for `history.manifest_of(job)` (or the live job's manifest), and `kind` is that name's kind. Map `LedgerError` as `put_request` does. Gallery metadata adds `linked_jobs` from `jobs_publishing_file`.

- [ ] **Step 4: Run** `pytest tests/test_ledger_notes_api.py tests/test_ledger.py tests/test_api_contract.py tests/test_ui_twins.py -q` — expect PASS (update `UI_READ_ROUTES` if the contract test requires the new routes).

- [ ] **Step 5: Commit** `feat(server): request-notes routes with ref checks; gallery metadata names linked jobs`

---

### Task 4: Waiting rule per note (Python and TS twins)

**Files:**
- Modify: `dw_mcp/ledger.py` (`note_waiting` :122, `waiting_notes` :141)
- Modify: `ui/src/lib/request/noteState.ts`
- Modify: `tests/fixtures/note_waiting_cases.json` (rewrite every case to `notes` / `agent_wrote_at`; add cases)
- Modify: `ui/src/lib/request/noteState.test.ts`, `tests/test_ui_twins.py` (the parametrized test reads note ids)

**Interfaces:**
- Produces (Python): `note_waiting(note: dict, request: dict) -> bool` — False unless `request["state"] == "open"`; True when `agent_wrote_at` is null or unparseable `at`; else `note.at > agent_wrote_at`. The item-dropped check stays in `waiting_notes`.
- Produces (Python): `waiting_notes(request) -> list[{request, revision, item, title, note, text, at, ref}]` — request's notes first, then each live item's, in stored order; `item` is `""` for the request's own.
- Produces (TS): `noteWaiting(note: LedgerNote | null, request: Pick<LedgerRequest,'state'|'agent_wrote_at'>): boolean`; `noteLine(note, request, now)` decides `answered` from `agent_wrote_at` and keeps the history detail text when agent entries after `note.at` are present, else says `The agent has updated the plan since (${formatMtime(agent_wrote_at)}).`; `notesWaiting(request)` counts every waiting note.
- Fixture `waiting` becomes a list of note ids.

- [ ] **Step 1: Rewrite the fixture and add cases**: "two notes on one item, agent wrote between them" → only the later waits; "history holds no agent entry but agent_wrote_at is later" → not waiting; "closed request with a note" → none (Review Focus 5); "dropped item's note" → none.

- [ ] **Step 2: Run** `pytest tests/test_ui_twins.py -q` and `cd ui && npx vitest run src/lib/request/noteState.test.ts` — expect FAIL.

- [ ] **Step 3: Implement** both twins to the interfaces above.

- [ ] **Step 4: Run** both commands again — expect PASS.

- [ ] **Step 5: Commit** `feat: the waiting rule is per note and reads agent_wrote_at (Python and UI twins)`

---

### Task 5: Channel events and MCP docs

**Files:**
- Modify: `dw_mcp/ledger.py` (`_target` :173, `note_event` :185, `waiting_event` :216, `NoteWatch` :255-310)
- Modify: `dw_mcp/tools_authoring.py:102` (tool docs: `notes` replaces `comment`; read and act, never write)
- Modify: `tests/test_mcp_notes_channel.py`, `tests/test_mcp_notes_stdio.py`, `tests/test_mcp_server.py:1875-1906`

**Interfaces:**
- Consumes: Task 4's `waiting_notes` dicts.
- Produces: `NoteWatch` dedupe key `(note id, at)`. `note_event` first line, with a ref:
  `Note from the user on request "<id>", item "<key>" (<title>), about take <attempt> of <n> (job <job>), file "<output>" (<kind>), saved <UTC>:` — `file …` omitted without `output`; `take <attempt> of <n>` becomes `job <job>` alone when `attempt` is null; append ` (that job's run no longer exists)` after the job when `gone`. `saved <UTC>` is `created`; when `at` differs it reads `saved <created UTC>, edited <at UTC>`. `meta` adds `note`, `created`, and `job` / `output` when present.
- Produces: the `DwClient` used by the channel sends `X-DW-Notes-Channel: 1` on its request-list read (`get_request(client, workspace=name)` in `NoteWatch.poll`) and on no other call.
- `waiting_event` lines carry the same reference phrase.

- [ ] **Step 1: Write failing tests**: the list poll carries the `X-DW-Notes-Channel` header and a per-request read does not; event text with ref + output, ref without output, gone ref, no ref (unchanged wording), edited note (`saved …, edited …`); `meta` keys; two notes on one item announced separately; an edited note (new `at`) re-announced; same-text edit not re-announced.

- [ ] **Step 2: Run** `pytest tests/test_mcp_notes_channel.py -q` — expect FAIL.

- [ ] **Step 3: Implement**; the take count `n` is `len(item["attempts"])`, so `waiting_notes` also carries `attempts` count for item notes.

- [ ] **Step 4: Run** `pytest tests/test_mcp_notes_channel.py tests/test_mcp_notes_stdio.py tests/test_mcp_server.py -q` — expect PASS.

- [ ] **Step 5: Commit** `feat(mcp): channel events name the note's take and file; tool docs read notes`

---

### Task 5b: Server knows whether a channel is listening

**Files:**
- Create: `dw/server/listening.py`
- Modify: `dw/server/routes/library.py` (`list_requests` :982 records a marked read; new route), `dw/server/app.py` (one `Listening` on `app.state`), `dw/server/api_models_library.py` (`NotesChannelState`)
- Test: `tests/test_notes_listening.py`

**Interfaces:**
- Produces: `class Listening` in `dw/server/listening.py` with `LISTENING_WINDOW = 30`, `seen(workspace: str) -> None`, `state(workspace: str) -> dict` returning `{"listening": bool, "last_seen": str | None}` (ISO UTC, seconds). Thread-safe (a `threading.Lock`), in memory only; it takes a `clock` callable for tests.
- Produces: `GET /api/workspaces/{name}/notes-channel` → `NotesChannelState {listening, last_seen}`. `list_requests` calls `seen(name)` only when the request has `X-DW-Notes-Channel: 1`.

- [ ] **Step 1: Write failing tests**: no poll → `{listening: False, last_seen: None}`; a marked list read → listening True; 31 s later (fake clock) → False with `last_seen` kept; an unmarked list read records nothing; one workspace's poll doesn't mark another.
- [ ] **Step 2: Run** `pytest tests/test_notes_listening.py -q` — expect FAIL.
- [ ] **Step 3: Implement** to the interfaces above.
- [ ] **Step 4: Run** `pytest tests/test_notes_listening.py tests/test_api_contract.py tests/test_ui_twins.py -q` — expect PASS.
- [ ] **Step 5: Commit** `feat(server): say whether a notes channel is polling a workspace`

---

### Task 6: UI — types, API client, request page note list, "Note this take"

**Files:**
- Modify: `ui/src/lib/types.ts:90` (`LedgerComment` → `LedgerNote {id, text, at, ref?: LedgerNoteRef}`, `LedgerNoteRef {job, output?, kind?, attempt?, gone?}`; `comment` → `notes` on item and request; `agent_wrote_at`)
- Modify: `ui/src/lib/api.ts` (`addNote`, `editNote`, `deleteNote`)
- Modify: `ui/src/lib/apiRequest.ts:133` (drop `comment` from `RequestWriteBody`), `ui/src/lib/request/intents.ts:248-343` (remove the `note` intent), `ui/src/lib/request/planStore.svelte.ts` (no notes in writes), the fake ledger used by tests (`fakeLedger.ts`)
- Create: `ui/src/lib/request/NoteList.svelte`, `ui/src/lib/request/NoteRef.svelte`
- Modify: `ui/src/lib/request/ItemNote.svelte` (one note: edit / delete, or a new one when `note` is null; takes an optional `ref` to show and remove before saving), `RequestItem.svelte:222-231`, `RequestHeader.svelte:91-94`, `AttemptStrip.svelte`, `menuEntries.ts:85`, `draft.ts`, `ui/src/lib/pages/RequestPage.svelte:123`
- Test: `NoteList.test.ts`, updated `ItemNote.test.ts`, `RequestItem.test.ts`, `RequestHeader.test.ts`, `RequestPage.test.ts`, `RequestPage.edit.test.ts`, `menuEntries.test.ts`, `draft.test.ts`

**Interfaces:**
- Consumes: Task 3 routes; Task 4 `noteLine`, `noteWaiting`.
- Produces: `api.addNote(ws, {request, item?, text, ref?}) → Promise<{note, revision}>`, `api.editNote(ws, {request, note, text})`, `api.deleteNote(ws, request, note) → Promise<{revision}>`.
- Produces: `<NoteList notes={LedgerNote[]} request item={string|null} workspace about filter?={(n)=>boolean} pendingRef?={LedgerNoteRef|null} onchanged />` — every note (or those `filter` keeps) with its `noteLine`, then an "Add a note for the agent" box; `pendingRef` prefills the box's ref chip. Calls the API itself and then `onchanged()` (the request page reloads the plan).
- Produces: `<NoteRef ref workspace />` chip "take {attempt} · {basename(output)}" linking to the job page (no output) or the gallery file (with output); greyed with title "this run no longer exists" when `gone`.
- `draftKey(workspace, request, item, refKey)` where `refKey` is `job` or `job/output` or `""`.
- Produces: `api.notesChannel(ws) → Promise<{listening, last_seen}>`; `ui/src/lib/request/listening.svelte.ts` exports `listening(workspace)`: a shared reactive `{listening: boolean}` per workspace, polled every 15 s while at least one `NoteList` using it is mounted, plus `refresh()` that `NoteList` awaits just before a save. A failed read counts as not listening.
- `ItemNote`'s save button reads **Send to agent** / **Save note** from it; `noteLine(note, request, now, listening)` gains the listening argument and, for a waiting note on an open request, returns "An agent session is listening; it will see this within about 10 seconds." or "No agent session is listening. It reads this the next time it updates the plan." Each saved note shows `Saved <created>` and `· edited <at>` when they differ.

- [ ] **Step 1: Write failing tests**: save button label follows `listening` and saving works in both states; the waiting line follows `listening`; an edited note shows both times; NoteList renders two notes with their lines and refs; adding calls `api.addNote` with `pendingRef`; removing the chip before save sends no `ref`; AttemptStrip "Note this take" opens the box with `{job, output: attempt.output.name}`; RequestPage plan save sends no notes; draft keys differ per ref; label text is "Note for the agent".

- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request src/lib/pages` — expect FAIL.

- [ ] **Step 3: Implement** to the interfaces above, keeping `ItemNote`'s draft, Esc and refusal behaviour.

- [ ] **Step 4: Run** `cd ui && npx vitest run && npm run check && npm run lint` — expect PASS.

- [ ] **Step 5: Commit** `feat(ui): request page holds a list of notes for the agent, with take references`

---

### Task 7: UI — job page and gallery note boxes

**Files:**
- Modify: `ui/src/lib/pages/JobPage.svelte` (panel near `JobHeader`'s request link), `ui/src/lib/job/JobResults.svelte` (per-output "Note this file")
- Modify: `ui/src/lib/gallery/GalleryDetail.svelte`, gallery metadata type in `ui/src/lib/types.ts`
- Test: `ui/src/lib/job/JobResults.test.ts`, `ui/src/lib/gallery/GalleryDetail.test.ts`, a new `ui/src/lib/pages/JobPage.notes.test.ts`

**Interfaces:**
- Consumes: Task 6's `NoteList`, `NoteRef`, `api.getRequest`.
- Job page: when the job has `request` and `item`, load the request (JobHeader already does; share that read) and render `NoteList` with `filter = note.ref?.job === job.id` and `pendingRef = {job: job.id}`; "Note this file" on an output sets `pendingRef = {job, output: name}`. Nothing renders for an unlinked job.
- Gallery detail: with one `linked_jobs` entry, `NoteList` with `filter = ref.output === file.name` and `pendingRef = {job, output: file.name}`; with several, a take picker ("take {attempt} of item {item}") chooses the entry first; hidden when `linked_jobs` is empty.

- [ ] **Step 1: Write failing tests**: unlinked job shows no panel; linked job shows only its notes and adds with `{job}`; "Note this file" adds with `{job, output}`; gallery with no linked job shows nothing; with two linked jobs shows the picker and posts the chosen job.

- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/job src/lib/gallery src/lib/pages/JobPage.notes.test.ts` — expect FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** `cd ui && npx vitest run && npm run check && npm run lint` — expect PASS.

- [ ] **Step 5: Commit** `feat(ui): notes for the agent from the job page and the gallery`

---

### Task 8: Docs

**Files:**
- Modify: `docs/MCP.md` ("The notes channel": event example with a reference and edit time; `note`/`created`/`job`/`output` meta; dedupe by note id; the listening header and what "listening" can't know - a session without the development-channels flag)
- Modify: `docs/WORKSPACES.md` (Requests: notes are a list, where they can be written, references, user-only)
- Modify: the 0.21.0 release notes file the notes channel entry is in (one line for referenced notes)

- [ ] **Step 1:** Update the three files; the event example matches Task 5's exact first-line format.
- [ ] **Step 2: Run** `pytest -q -x` (whole Python suite) and `cd ui && npx vitest run` — expect PASS.
- [ ] **Step 3: Commit** `docs: referenced notes for the agent`
