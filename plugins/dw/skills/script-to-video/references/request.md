# The request: one ledger for a multi-job ask

Part of the `script-to-video` skill, and used the same way by `storyboard`,
`long-form` and `series-episodes`. The shot document
(`references/shot-document.md`) says *what* each shot is. The request says
which jobs were made for the ask, which one is the keeper, what is left and
what it cost. It lives in the workspace's `requests.json`, so it outlives the
session and shows in the UI. The fields and the server's rules are in
`get_guide("workspaces", section="Requests")`.

## When to open one

Open a request when the ask takes **three or more jobs**: a shot list, a board
set, an episode, a cast sheet plus shots. A single image, or one or two
takes, gets no request: `run_workflow`'s `note` is enough.

Open it at the step that writes the shot list, right after
`save_prompt("<project>/shots", ...)`, before any job runs.

## Open it

One `set_request` call, `revision=0`:

- `request`: the project's name as a key, `^[a-z0-9][a-z0-9-]{0,63}$`:
  lower-case it and turn `_` and `.` into `-` (`gilded_echoes` becomes
  `gilded-echoes`).
- `prompt`: the user's ask, **verbatim**. It is write-once: it can't be
  changed later.
- `title`: a few words. `goal`: what done looks like, in one or two lines.
- `why`: one line, at most 200 characters ("opened from the shot list").
- `items`: one per job the plan needs, in order:

| Item | `key` | `arguments` point at |
| --- | --- | --- |
| a shot | `shot-NN` | `prompt:<project>/shot-NN` |
| a board | `board-NN` | `prompt:<project>/board-NN` |
| a cast sheet (portrait, voice) | `cast-<name>` | `prompt:<project>/<name>` |
| the score or bed | `score` | the score's saved caption |
| the cut | `cut` | none; `depends_on` every shot |

Each item carries `key`, `title`, `workflow` (the template it runs, such as
`templates/minimax/dialogue-short`), `arguments` (what the run passes, each
text as a `prompt:` reference, never the prompt text itself), `depends_on`
(other keys: a shot on its cast, the cut on the shots) and
`estimate_minutes` (the template's `plan.estimate.minutes`, or null). `NN` is the
shot document's `id`, so item `shot-03` is shot object `03`. The shot
document stays the creative source: a changed shot is edited there and
recompiled, and the item is not touched.

The call returns the request at revision 1.

## Run each item

Pass `request="<request>/<key>"` on every `run_workflow` for that item,
with its `note` as before. A rerun (`rerun_job`) keeps the link by itself.
`get_request("<request>")` then shows each item's `status`, its
`attempts` (job, status, note, minutes), its `result` and the request's
`spent_minutes` and `remaining_minutes`. Quote what is left from
`remaining_minutes` when the user asks.

## Hold items for the user's review

Use this when the plan is written and priced and the user wants to look at it
on the request page before it runs. A hold defers a run the user already
approved; it never stands in for the approval.

1. Quote the total in chat and get a yes to "these run at T unless you leave
   a note". Silence on the page confirms that yes; it never replaces it.
2. Hold only when the dw-notes channel is connected in this session (its
   instructions are in your context). Without it nothing tells you the
   deadline passed, so run on the user's word as before.
3. Set the hold with `set_request`: each held item gets `review_until`, an
   ISO time with an offset, 15 minutes ahead unless the user says otherwise
   (the server refuses under 60 seconds or over 24 hours), and the `why`
   says so. Then end the turn. Do not block in `wait_for_job` on something
   else; the channel event is what wakes the session.

What wakes you:

- **A note.** Do the revision, write the plan, and give the revised items a
  new `review_until` if the user should look again. A note redirects the
  hold; nothing else does.
- **A `review_due` event.** `get_request`, then run only the named items that
  are still `ready` with no newer note, each with
  `run_workflow(request=..., acknowledged_cost=...)` for the plan you quoted.
  A refusal for a changed plan, or a quote well off what you told the user
  (more than a quarter), means a new hold with the new figure, not a run.
- **A dependency finishing.** An item held behind a dependency is announced
  when the dependency finishes; you run it then. The server never starts a
  job for you.
- **A `review_due` event marked stale** is from before this session: confirm
  with the user first.

## Accept the keeper

The keeper is accepted in the request, **not with `set_note`**:

1. `get_request("<request>")`: the plan and its current `revision`.
2. `set_request` with the same `request`, `title` and `goal`, that
   `revision`, every item as read (one left out is removed), the kept
   item's `accepted: {"job": "<job id>"}`, and a `why` ("shot-03: take 2,
   the turn lands").

The derived fields that came back on the read are ignored on the write, and
a user's `notes` are kept whether you send them back or leave them out. A
`stale_revision` refusal means someone wrote since your read: read again,
merge, write.

With nothing accepted, an item's result is its latest succeeded job.

## Before a revision pass

`get_request("<request>")` first, and read each item's `notes` and
their `at`: they are what the user typed on the request page, job page or
gallery, and they are the brief for that item's next attempt, beside the beat
review. A note's `refs` say which take (`attempt`, `job`) and file (`output`,
`kind`) it is about; one with only an item is about a shot not yet rendered,
and the request's own notes may name several items. A note newer than the
agent's last `set_request` is not yet answered.

## When a note arrives mid-session

A `<channel source="dw-notes" ...>` event is a note the user has just
saved, delivered by dw's notes channel; its text lists the items, takes and
files it is about, and its `request`, `note` and `items` attributes name the
target (`item` too, for a note written on an item). Do the revision pass above for that item now:
`get_request` first (the note may have changed since), then the rerun or
fix it asks for, under the usual cost gate. Finish with a `set_request`
whose `why` says what you did - the user watches the request page, not this
session, and that write is what marks the note answered. A
`kind="notes_waiting"` event lists notes from before the session: say what
is waiting and confirm before spending on them. A `kind="review_due"` event
is the hold above running out.

## Changing the plan

A shot added, split or cut changes the shot document first, then the
request: `set_request` with the new item list and a `why`. An item with jobs
that is left out is kept as `dropped`; one with none is removed. Close the
request with `state="done"` once the cut is delivered, or `"abandoned"` when
the user drops the ask.
