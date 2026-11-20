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
`estimate_minutes` (the template's `plan.estimate`, or null). `NN` is the
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

## Accept the keeper

The keeper is accepted in the request, **not with `set_note`**:

1. `get_request("<request>")`: the plan and its current `revision`.
2. `set_request` with the same `request`, `title` and `goal`, that
   `revision`, every item as read (one left out is removed), the kept
   item's `accepted: {"job": "<job id>"}`, and a `why` ("shot-03: take 2,
   the turn lands").

The derived fields that came back on the read are ignored on the write, and
a user's `comment` is kept whether you send it back or leave it out. A
`stale_revision` refusal means someone wrote since your read: read again,
merge, write.

With nothing accepted, an item's result is its latest succeeded job.

## Before a revision pass

`get_request("<request>")` first, and read each item's `comment` and its
`at`: it is what the user typed on the request page, and it is the brief for
that item's next attempt, beside the beat review. A comment newer than the
item's last attempt is not yet answered.

## Changing the plan

A shot added, split or cut changes the shot document first, then the
request: `set_request` with the new item list and a `why`. An item with jobs
that is left out is kept as `dropped`; one with none is removed. Close the
request with `state="done"` once the cut is delivered, or `"abandoned"` when
the user drops the ask.
