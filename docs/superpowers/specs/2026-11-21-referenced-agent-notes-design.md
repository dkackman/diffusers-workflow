# Referenced notes for the agent

Status: draft, 2026-11-21. Follows the notes channel (`dw-mcp --notes-channel`,
develop `6df5c458`).

## Intent

A **note for the agent** is the user's instruction about a request or one of
its items. Today it says *which item* but not *which take or file*: an item
with three attempts, each writing a video, an audio track and a few stills,
gets one note, and "the lipsync is off" leaves the agent to guess which
output the user was looking at.

The user wants to write a note **where they are looking at the thing** - the
request page, a job page, a gallery file - and have the note carry a
reference to that specific thing, so the agent (in `get_request` and in the
channel event) knows exactly what the note is about.

## Goals

- A note can carry a reference to a specific job, and optionally one output
  file of that job.
- Notes can be written from three places: the request page, the job page and
  the gallery file view - each only for jobs linked to a request item.
- An item (and the request itself) can hold several notes, so a second note
  does not overwrite the first.
- The agent sees the reference in `get_request` and in the channel event, in
  words it can act on (item, take number, job id, file name and kind).
- The page still tells the user, per note, whether the agent has updated the
  plan since it was saved.

## Non-goals (this iteration)

- Notes on jobs not linked to a request (one-off runs). They have no plan
  for the agent to answer in. If they turn out to matter, the follow-up is a
  per-workspace notes store with an explicit "handled" mark; nothing here
  should block that.
- Agent-written notes, replies or threads. Notes stay the user's alone.
- Merging with the existing run/file note (`set_note`, `notes.json`).

## Anti-goals

- An agent's write can change or delete a user's note. The ledger rule
  "the comment is the user's" must hold for every note.
- The two kinds of note get confused in the UI. The existing run/file note
  is a label both sides write; the note for the agent is an instruction.
  Every surface labels the new one "Note for the agent".
- The channel leaks server filesystem paths. A reference names a job id and
  a manifest output name (the gallery name, relative to the workspace
  outputs), never an absolute path.
- Saving a note from the job page or gallery loses a concurrent plan edit.

## Today (what this changes)

- Ledger (`dw/ledger.py`): one `comment {text, at}` per item (`_ITEM_FIELDS`)
  and per request, written only `by == "user"` (`_comment`, :604). No schema
  version in `requests.json`.
- Saved by a whole-plan PUT with the read revision
  (`dw/server/routes/library.py:1016`); the UI's `PlanStore` retries on
  `stale_revision`.
- Waiting rule: a note waits while the request is open and no `by == "agent"`
  history entry is later than its `at`. Python `note_waiting`
  (`dw_mcp/ledger.py:122`) and TS `noteWaiting` (`ui/src/lib/request/noteState.ts`)
  are twins held equal by `tests/fixtures/note_waiting_cases.json`.
- Channel: `NoteWatch` dedupes by `(item key, at)`; `note_event` /
  `waiting_event` build the text.
- Jobs carry `request` / `item` when linked (`job_record.py:194`). The job
  page already shows the `request › item` link (`JobHeader.svelte`). The
  gallery detail's job ref is only `{id, status}` (`JobRef`,
  `api_models.py:986`).
- An output file is identified by its manifest name
  (`<workflow>/<run_id>/<subfolder>/<file>`); its kind comes from
  `output_kinds`.

## Approaches considered

1. **Whole-plan PUT from every surface.** Keep the one write path; the job
   page and gallery load the plan, add the note and PUT it. Simple server,
   but two more surfaces each need a plan store with revision retry, and a
   note from the gallery races any plan edit open in another tab.
2. **Dedicated note endpoints on the request** (chosen). `POST`, `PUT` and
   `DELETE` on `/api/workspaces/{ws}/requests/{id}/notes[/{note_id}]`,
   user-only. The server does the read-modify-write under the ledger lock,
   bumps the revision and adds a `by: "user"` history entry, so no surface
   handles revisions for a note. The whole-plan PUT keeps carrying notes
   back unchanged (an agent omitting them keeps them, as now).
3. **Separate notes store** (like `notes.json`) keyed by target. Covers
   one-off jobs, but the "agent has answered" signal then needs its own
   mechanism. Deferred - see Non-goals.

Picked 2: the user asked for notes in several places, and a dedicated write
keeps each surface to one call with no revision handling. (Approach was the
designer's call; the user approved "notes in several places, each carrying a
reference, inside the request ledger, linked jobs only".)

## Design

### Data

Each item and each request gets `notes: [Note]`, replacing `comment`:

```json
{
  "id": "n3",
  "text": "lipsync drifts after 2s - reroll this one",
  "at": "2026-11-21T13:02:11Z",
  "ref": {"job": "j_8f2c", "output": "lighthouse/run_41/final/shot3.mp4"}
}
```

- `id`: short, unique within the request, assigned by the server, never
  reused.
- `created`: when the note was first saved; set once, never changed (user's
  addition, 2026-11-21). `at` is the last text change and is what the
  waiting rule reads. Upgrade on read sets `created = at`. The UI shows
  "Saved <created>", plus "· edited <at>" when they differ; the channel
  event carries both.
- `ref`: optional. `job` is required when `ref` is present; `output` is
  optional (a note about the take as a whole).
- Validation on write: `ref.job` must be linked to this request (and, for an
  item note, to this item); `ref.output` must be in that job's manifest.
  A ref that fails is refused (`400`), not stored.
- A ref can go stale later (job deleted, run purged). It is kept as written;
  the views add `ref.gone: true` and the UI shows the reference greyed out.
- Cap: a working cap of live notes per target (builder's call, around 20),
  inside the existing `MAX_FILE_BYTES`.

**Upgrade on read.** `_load` turns a stored `comment {text, at}` into
`notes: [{id: "n1", text, at}]`. The next write stores the new shape.
`requests.json` has no version field and this does not add one; the
`comment` key is simply never written again. No compatibility step is
needed: each server has its own workspaces, and two servers on different
versions never share a `requests.json` (user's decision, 2026-11-21). A
rollback to a pre-notes server would drop notes written since; that is
accepted.

### Writes

- `POST .../notes` `{item?, text, ref?}` adds a note; returns the note and
  the new revision. `item` absent means the request's own notes.
- `PUT .../notes/{id}` `{text}` edits text only; a new text gets a new `at`
  (so it waits, and is announced, again). Ref is fixed once written.
- `DELETE .../notes/{id}` removes it.
- All three refuse an agent caller with the existing "the note is the
  user's" error. How the server tells user from agent stays as it is for
  the whole-plan PUT.
- The note endpoints are the **only** way to change a note (designer's
  call). A whole-plan PUT, from the user or the agent, keeps the stored
  notes whatever it sends: `notes` drops out of `_ITEM_FIELDS` and
  `RequestWrite`, and the server copies the stored notes onto the written
  plan. This stops a request-page tab's PUT from deleting a note just added
  from the gallery. The UI's existing request-page note intent moves to the
  new endpoints, and `PlanStore` / `fakeLedger.ts` stop sending notes.
- A note on a dropped item is refused, as now.

### Reads

- `get_request` and `GET .../requests/{id}`: each item and the request carry
  `notes`, each note's `ref` enriched with what the agent needs to act
  without another call: `attempt` (1-based take number within the item),
  `kind` of the output, and `gone` when stale.
- MCP tool docs (`tools_authoring.py`) say `notes` replaces `comment`:
  read them, act on them, never write them.

### Waiting rule

Applied **per note**: a note waits while the request is open, its item is
not dropped, and the request's last agent write is not later than its `at`.

The last agent write is a new stored request field, `agent_wrote_at`, set
on every `by: "agent"` write and never trimmed (designer's call). Today the
rule searches `history`, which keeps only the first entry plus the latest
99. Cheap note writes from three surfaces could push the agent's entry out
and make answered notes wait, and be re-announced, again. On upgrade,
`agent_wrote_at` is filled from the latest agent entry in `history`. Note
writes still add `by: "user"` history entries.

Python and TS twins both change to read `agent_wrote_at`; the shared
fixture gains cases for several notes on one target with different `at`s
and for a history with no agent entry left in it.

### Channel

- `NoteWatch` dedupes by `(request, note id, at)`.
- `note_event` text adds the reference when there is one:

  ```
  Note from the user on request "lighthouse-short", item "shot3" (Shot 3: lamp room),
  about take 2 of 3 (job j_8f2c), file "lighthouse/run_41/final/shot3.mp4" (video),
  saved 2026-11-21 13:02 UTC:
  ```

  and, when stale, "(that job's run no longer exists)".
- `meta` adds `note`, `job` and `output` when present (identifier keys,
  string values, as the channel contract needs).
- `waiting_event` lists each waiting note with its reference on its line.

### Is a session listening?

User's addition, 2026-11-21: the note's save button says whether an agent
session will hear it now.

- The channel process marks its request-list poll with a header,
  `X-DW-Notes-Channel: 1`. The server keeps, in memory, the last time a
  marked poll read each workspace's list. Nothing is stored on disk, and a
  restart forgets it until the next poll.
- `GET /api/workspaces/{name}/notes-channel` returns
  `{listening: bool, last_seen: str | null}`. `listening` means a marked
  poll in the last `LISTENING_WINDOW = 30` seconds, three channel
  intervals at the default 10.
- What it can't know: a session started without the development-channels
  flag still runs the poller, and Claude Code drops its events. So
  "listening" means "a session has the channel set up", and the docs say
  so.
- The header is only a hint for the page. Anyone holding the token could
  send it; the worst outcome is a wrong label.
- **The button is relabelled, never disabled.** When a session is
  listening, the save button reads **Send to agent** and the line under a
  waiting note says "An agent session is listening; it will see this within
  about 10 seconds." When none is, it reads **Save note** and says "No agent
  session is listening. It reads this the next time it updates the plan."
  Saving works either way.
- The page asks every 15 seconds while a note box is on screen, and once
  more just before a save, so the label is fresh at the moment it counts.

### UI

- **Request page, item row** (`RequestItem.svelte`, `ItemNote.svelte`): the
  single note box becomes a list of the item's notes, each with its
  waiting/answered line and, when it has a ref, a chip ("take 2 ·
  shot3.mp4") linking to the job or file. An "Add a note for the agent"
  box below. The attempt strip gets a per-take "Note this take" action
  that opens the box with the ref prefilled (job + that take's proof
  output); the chip shows it and can be removed before saving.
- **Request header**: the request's own notes, same list, no ref picker.
- **Job page** (`JobPage.svelte`): when the job has `request`/`item`, a
  "Note for the agent" panel near the `request › item` link, showing that
  item's notes about this job and an add box with `ref.job` set. Each
  output in the manifest list gets "Note this file", which sets
  `ref.output`. Not shown for unlinked jobs.
- **Gallery detail** (`GalleryDetail.svelte`): a file can be published by
  more than one job. A step-cache hit republishes an earlier run's file, and
  `job_for_file` returns only the job that first wrote it. So the metadata
  gains `linked_jobs`: every job whose manifest lists this file, `reused`
  entries included, that is linked to a request item, as `{id, request,
  item, attempt}` (designer's call). With one entry, the add box sets
  `ref = {job, output: <this file>}`; with several, it asks which take the
  note is about. It is hidden when no linked job published the file.
- `ref.output` validation accepts any manifest entry of the job, `reused`
  ones included, since a take's proof output can be one.
- Drafts keep the existing per-tab sessionStorage behaviour; the draft key
  includes the ref so two half-written notes on different files don't
  collide.
- Wording everywhere: "Note for the agent". The existing run/file note
  keeps its current label and place.

### Tests

- Ledger: upgrade on read; add/edit/delete; agent refusal on endpoints and
  whole-plan PUT; ref validation (job not linked, output not in manifest,
  dropped item); stale ref view; caps.
- Twin fixture: per-note waiting cases.
- Channel: dedupe by note id; event text and meta with and without ref,
  stale ref; digest with several notes on one item.
- UI: list rendering, add from attempt strip / job page / gallery, chip
  removal, draft keys.

## Left to the builder

- Exact cap numbers and note id format.
- Endpoint response shapes beyond "the note and the new revision".
- Layout details of the note list and chips, within the existing request
  page and job page styles.
- Whether the job page shows all of the item's notes or only those about
  this job (spec says "about this job"; showing all with a filter is fine).
