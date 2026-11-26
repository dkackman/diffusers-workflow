# Workspaces

A workspace is the directory your own content lives in: the workflows you
write, the prompt library they reference, the assets they read, and the files
they generate.

```
<workspace>/
  workflows/    your workflows
  prompts/      the stored prompt library ('prompt:' references)
  assets/       input media
  outputs/      generated files
```

It keeps day-to-day work out of a checkout of this repository: the repo's
`workflows/` is a corpus of examples to read, copy and run, and generated
media does not belong in a source tree.

## Which directory is used

The workspace is resolved by `dw.serve`. `python -m dw.run` is a client of a
running `dw.serve`, so it resolves no directory of its own: its jobs land in
the server's workspace. First match wins:

1. `--workspace <dir>` on `dw.serve`
2. the `DW_WORKSPACE` environment variable
3. `"workspace"` in `~/.diffusers_helper/settings.json`
4. the working directory, when it holds any of `workflows/`, `prompts/` or `outputs/`
5. `~/diffusers-workspace`

Rule 4 means a checkout serves itself: the repository root holds all three
folders. Rules 1-3 are *explicit*; rules 4 and 5 are inferred, which matters
for the prompt library (below).

Resolving creates nothing. `dw.serve` creates the workspace's `workflows/`
(unless `--workflow-dir` was given), `assets/` and the shared
`common/assets/` at start, and a run creates its output directory.

`dw.run --workspace NAME` is a different thing: it names one of the server's
workspaces (the `default` one, or a named one beside it), not a directory. A
path there is refused as a workspace name rather than resolved.

## An installed copy

An installed copy ([INSTALL.md](INSTALL.md)) has no checkout, so rule 4 rarely
applies. Its server is started by the service `dw setup` installs, from the
service manager's working directory (`/` on macOS, your home on Linux), so the
workspace is `~/diffusers-workspace` unless your home holds `workflows/`,
`prompts/` or `outputs/`. The service does not see your shell's `DW_WORKSPACE`;
to put the workspace elsewhere, set `"workspace"` in
`~/.diffusers_helper/settings.json` and restart the service
([SERVER.md](SERVER.md#the-service-dw-setup-installs)). Outputs, prompts and
assets live under the workspace as on any other server; the models are in the
Hugging Face cache as usual.

The package carries no `workflows/` catalog, so the workflow library starts
empty apart from dw's built-ins. To add the examples, copy them from the
release's source archive into the workspace, where they become your own
(writable) copies under the same names (`templates/text-to-image`):

```bash
v=0.23.0   # your dw version; `dw setup --status` prints it
curl -L https://github.com/dkackman/diffusers-workflow/archive/refs/tags/v$v.tar.gz | tar xz
mkdir -p ~/diffusers-workspace/{workflows,prompts,assets}
cp -R diffusers-workflow-$v/workflows/. ~/diffusers-workspace/workflows/
cp -R diffusers-workflow-$v/prompts/. ~/diffusers-workspace/prompts/
cp -R diffusers-workflow-$v/assets/. ~/diffusers-workspace/assets/
```

## Overriding one folder

`--workflow-dir`, `--prompt-dir`, `--asset-dir` and `--output-dir` on
`dw.serve` each override one folder of the workspace. `dw.run` takes none of
them; every job it queues writes where the server does.

```bash
python -m dw.serve --workspace ~/studio --output-dir /mnt/big-disk/renders
python -m dw.serve --prompt-dir ~/shared-prompts
```

`--output-dir` is the usual one: video work fills disks.

## Working in a workspace

```bash
mkdir -p ~/studio/{workflows,prompts,assets,outputs}
export DW_WORKSPACE=~/studio

# or, standing, in ~/.diffusers_helper/settings.json
#   { "workspace": "/home/you/studio" }

python -m dw.serve                       # serves ~/studio
python -m dw.run ~/studio/workflows/x.json   # queued on that server
```

An example from a checkout still runs by path, and writes into the server's
workspace's outputs; `dw.run` itself does not read `DW_WORKSPACE`:

```bash
python -m dw.run ~/src/diffusers-workflow/workflows/templates/text-to-image.json
```

A file the server cannot read is sent to it inline, and its relative paths
then resolve against the server's `workflows/`.

The server reports what it resolved at `GET /api/server`, under
`directories`: `workspace` alongside the `workflows`, `prompts`, `assets`
and `outputs` paths.

## The prompt library

`dw.serve` resolves the prompt library once at start and pins it for every
job, first match wins:

1. `--prompt-dir`, then `DW_PROMPT_DIR`
2. the workspace's `prompts/`, when the workspace is explicit (rules 1-3),
   even if it does not exist yet
3. `./prompts` in the working directory
4. the nearest `prompts/` at or above the workflow directory
5. the workspace's `prompts/`

So a standing `DW_WORKSPACE` or `"workspace"` setting stops a checkout's
`./prompts` being found; an inferred workspace does not. See
[Prompt References](WORKFLOW_GUIDE.md#prompt-references).

## Assets

`assets/` is the input-media library. A workflow argument written as
`asset:name.ext` (or `asset:folder/name.ext`) resolves to that file's path,
rooted at the library rather than at the workflow file. `--asset-dir`
overrides the folder. Browser uploads land in `assets/uploads/` and come back
as `asset:uploads/<name>`, served for preview under `/inputs/` (the UI's own
bundles own `/assets/`).

A generated file becomes an input the same way: **Keep as asset** in the
gallery (`POST /api/assets/keep`, `keep_output` over MCP) links or copies it
out of `outputs/` into `assets/` under a name you choose, so a later workflow
carries `asset:<name>` rather than a run id that pruning would break. The copy
stays inside the workspace — a hard link where the filesystem allows one, so
keeping one frame of a large render costs no second copy of it. See
[Asset References](WORKFLOW_GUIDE.md#asset-references).

## Where workflows are read from, and written to

The server reads workflows from a search path and writes them to exactly one
place — the front:

```
<workspace>/workflows/    yours, writable — every save lands here
<--examples-dir>          read-only, repeatable
```

A name found in an earlier root shadows the same name in a later one, so a
workspace copy of an example is the one that runs. Reads — listing, opening,
downloading, validating, running — span the whole path. Saves and deletes do
not: `PUT` always writes into the writable root, and deleting something from a
read-only root is refused with a 403 that says where it came from.

That makes "open an example, change it, save" do the obvious thing: the copy
lands in your library and shadows the example from then on, and the example
itself is never touched. It is also what stops an agent's saves landing in a
checkout — point `--workflow-dir` (or `--workspace`) at your own directory and
the repository's workflows at `--examples-dir`:

```bash
python -m dw.serve --workspace ~/studio --examples-dir ~/src/diffusers-workflow/workflows
```

`GET /api/workflows` reports the path as `libraries` (`[{origin, root,
writable}]`, in search order) and tags every entry in `details` with its
`origin` and `writable`, which is how the UI knows to hide delete and how an
MCP client can tell what it may change. The writable root is the `libraries`
entry with `writable: true` and `origin: "workspace"`; a copy an example's
name hides is listed under `shadowed`.

### The prompts and assets an examples tree brings with it

An example workflow references the prompts and media that live beside its
tree, not the ones in your workspace, so each `--examples-dir` puts those on
the back of the two libraries as well: the `prompts/` and `assets/` folders
beside the directory (or inside it, if that is where they are). Both libraries
work exactly like the workflow path — your workspace's own is searched first
and a name there shadows the example's, reads span everything, and writes only
ever reach your own:

```
<workspace>/prompts/      yours, writable — every save lands here
<--examples-dir>/../prompts   read-only

<workspace>/assets/       yours, writable — uploads and "keep as asset" land here
<--examples-dir>/../assets    read-only
```

So the command above makes `workflows/models/flux-dev.json`'s
`"prompt:flux/biomechanical_daffodil"` resolve out of the checkout, without
copying the prompt library into the workspace. `GET /api/prompts` and
`GET /api/assets` report the roots as `libraries`, the same envelope as the
workflow listing, and tag each entry with its `origin` and `writable`;
deleting a prompt that came from a read-only library is refused with a 403,
and saving one writes a copy into your workspace the way saving an example
workflow does.

The packaged workflows in `dw/workflows/` are *not* on the path: they are
what a `builtin:` sub-workflow step names (`GET /api/builtin-workflows` lists
them), not workflows to browse or run on their own.

## Runs

Each execution writes its own directory under the output folder, named by the
workflow and the run:

```
outputs/
  ltx2/Gyre/
    20260905-181530-a1b2c3d4/
      Gyre-still.0-0.0.png
      Gyre-video.1-0.0.mp4
      manifest.json
      workflow.json
```

The folder is the workflow's identity — its path under a `workflows/` tree
when it has one, its file name otherwise, its `id` for an inline definition.
The run id is a timestamp plus a short digest of what actually ran, so two
runs of the same workflow sort by time and a rerun of an edited workflow is
visibly different; a second run of the same spec in the same second takes a
counter rather than sharing a directory.

`workflow.json` is the realized workflow — the definition with this run's
arguments, seed and stored prompts pinned into it, so the directory reproduces
itself. `manifest.json` points at it and lists which prompts were inlined, with
`workflow.prompt_digests` beside them: `{prompt name: sha256 of the text it
inlined}`, the text actually run even if the prompt was saved again while the job
waited in the queue.

`manifest.json` records the run beside what it made — status, seed, arguments,
device, dw version, and each step's files, named relative to the directory so
it keeps describing itself if you move or copy it. It is written even when a
run fails part way, since the files it did write are on disk either way. A
sub-workflow is part of its parent's run: it writes into the same directory and
rolls up into the same manifest.

A pipeline step that ran also carries `timing` in its `manifest.json` entry
(and in the same entry of the job's manifest from the API or MCP `get_job`):
load, encode, denoise, decode and save seconds. A reused, task or composed
step has none. See [Step timing](#step-timing).

An unchanged rerun still reuses the step cache: it writes no new files and its
manifest reports the earlier run's, marked `"reused": true`. The cache is
validated against the output *root* a run writes into, which is the pinned
workspace's own `outputs/` - so it is per workspace, not per workflow alone.
A run in one workspace does not make `validate_workflow`'s
`plan.cached_steps` come back nonzero for a matching run sitting in another
workspace, and deleting a workspace drops its cache entries along with its
`outputs/` directory. The cache itself is also per *process*: entries are
held in memory by the running server, not read back from `outputs/`, so a
`dw.serve` restart empties it even though every run directory is still on
disk - a `plan.cached_steps` of 0 right after a restart is expected, not a
lost run.

For the same reason, two `dw.serve` processes on the same workspace share no
cache entries: a run on one server is a cache miss on the other, even though
both read the same `outputs/`. To avoid recomputing, either use one server
with `--devices cuda:0,cuda:1`, whose affinity sends a rerun back to the
card that holds it warm ([Dispatch](SERVER.md#dispatch)), or, with two
servers, rerun each job on the server that ran it.

A later workflow names what an earlier run made with an `output:` reference —
`output:ltx2/Gyre/latest/Gyre-still.0-0.0.png` — so a multi-stage pipeline no
longer needs files copied back by hand. See
[Output References](WORKFLOW_GUIDE.md#output-references).

Each step's entry in `manifest.json` records its `role`, `final` or
`intermediate` (and `role_declared` when the step's `result.role` stated it),
from that one run's data flow. The gallery reads across runs as well: a file a
later run in the same workspace read - by an `output:` reference, or by an
`asset:` that `keep_output` kept from it - lists as `intermediate`, so a
project's shots, narration and score stop reading as finals once the assembly
that cuts them has run. Nothing is rewritten; a role `result.role` declared
`final` stays final, and a run from before roles stays unknown. See
`dw/roles.py`.

To keep the previous layout — everything at the output root, with only a
`workflows/`-mirroring subfolder — use `--output-layout flat`, `DW_OUTPUT_LAYOUT=flat`,
or `"output_layout": "flat"` in settings. Scripts that glob the output directory
are the reason to.

### Step timing

A manifest entry for a pipeline step that ran - in `GET /api/jobs/{id}`, in
MCP `get_job`, and on that step's `step_end` event - carries `timing`: where
the step's time went. It is measured in the worker on its monotonic clock, not
taken from event timestamps, because a job served from history keeps only its
last 200 events, and a long job's load and encode events are gone by then. Every `_s` is seconds to
0.1. A field whose mark never happened is null, never guessed.

| Field | Meaning |
|---|---|
| `load_s` | Loading or fetching the step's pipeline, cached or not |
| `load_cached` | `true` if the pipeline came from the worker's cache, `false` if any part loaded fresh, null if the step loaded nothing |
| `encode_s` | Call start to the first denoiser forward (or a modular pipeline's denoise bar opening): prompt and reference encoding and preparation |
| `denoise_s` | First denoiser forward to the last pass's end. The whole call when the pipeline reports no passes; `encode_s` and `decode_s` are then null |
| `passes` | Denoise passes the pipeline actually ran: one per iteration of its denoise loop, so one per scheduler timestep. This is not always `num_inference_steps`: SD 1.5's default PNDM scheduler runs 21 passes at 20 steps (N+1 from 2 steps up). `denoise_s` ≈ `first_pass_s + (passes − 1) × pass_s` |
| `first_pass_s` | The first pass, which carries any compile warm-up |
| `pass_s` | Median of passes 2..n; null with fewer than two passes |
| `compiled` | The denoiser runs through `torch.compile` |
| `decode_s` | Last pass's end to the call's return: VAE decode and post-processing |
| `save_s` | Writing the step's files |
| `segments` | Pipeline calls summed into this timing. A chain step sums its segments into one timing with `segments: N`; a step that iterates counts each call. 1 for a plain step |

A step that set a `cache` also carries `timing.cache`: `{type, threshold, forwards, computed, skipped, hit_rate}`. `forwards` counts denoiser forwards, not steps - classifier-free guidance runs two per step (cond and uncond), so it is a whole multiple of `passes` - and `computed + skipped == forwards`; `hit_rate` is `skipped / forwards` to 3 places. A chain sums the counts over its segments. Counted for `first_block` and `mag` (`mag` also reports `num_inference_steps`, `max_skip_steps`, `retention_ratio`, `calibrate`); `taylorseer` (with `cache_interval`, `max_order`), `faster` and `text_kv` keep no per-forward decision in diffusers, so they report `type` and their config keys with `threshold` and every count null. A step with no cache has no `cache` key. If the installed diffusers moved the hook state dw reads, the counts are null and one `warning` event with `kind: "cache_stats_unavailable"` is emitted; the run never fails for it. The same fields arrive live as a `cache_stats` event ([SERVER.md](SERVER.md#progress-events)). See [Reading the skip counts](ACCELERATION.md#reading-the-skip-counts).

A step served from the step cache (`reused: true`) has no `timing`, nor does a
task step or a composed (sub-workflow) step. A failed or cancelled step has no
manifest entry, so no timing. Job summaries and history are unchanged.

### Host-memory history

Each finished job's history row records a `memory_key`: a digest of the
run's pipeline identities (`component_type`, `model_name`, `workflow` of
every pipeline step) and every `quantization_config`, `offload` and
`group_offload` value, with the caller's arguments resolved. The
workflow's name is not in it, so a renamed or inline copy of a template
shares the template's history, and a different `weights_dtype` or offload
is a different key. From it the server projects a run's host-RAM peak:
`observed` (the largest job peak RSS among this machine's successful runs
with the same key), else `fit` (the list-run projection from the
workflow's own history), else no figure. A workflow's declared
`host_ram_gb` is a floor under that figure (basis `declared` when it is the
larger). Rows recorded before the column
existed have no key and count as no data. Validate reports it as
`plan.host_memory`.

## Requests

A workspace keeps a ledger of what was asked of it: `requests.json` at the
workspace root, holding `{"requests": {<id>: <request>}}`. A request is the
user's `prompt` (set when the request is created, never changed after: a
change is refused with 422 `prompt_is_write_once`), a `title`, a `goal`
saying what done looks like, a `state` (`open`, `done`, `abandoned`), the
user's `notes` on the whole of it, and an ordered plan of `items`. Only
the server writes the file.

Read it with `get_request(request=None)` (MCP) or `GET
/api/workspaces/{name}/requests[/{id}]`; write one request's whole plan with
`set_request` or `PUT /api/workspaces/{name}/requests/{id}`; remove one with
`delete_request` or `DELETE /api/workspaces/{name}/requests/{id}?revision=N`:

- **Revisions.** Every write names the `revision` it was read at, or 0 to
  create. A write against any other revision is refused with 409
  `stale_revision`, carrying `current_revision`: read again, merge, write.
  Each write adds a history entry with `by` (`agent` over MCP, `user` from
  the UI), its `why` (one line, at most 200 characters) and the time, plus
  the `goal` and the `state` when that write set or changed them. A plan
  write after the first also carries `changed`: by item key, which of
  `added`, `dropped`, `restored`, `renamed`, `workflow`, `arguments` and
  `depends_on` it changed. Only the newest 10 entries that changed an item
  keep it.
- **State.** Any write may set `state`, the user's as well as the agent's:
  `open` to `done` or `abandoned`, one closed state to the other, and back
  to `open`. Closing changes nothing else. A closed request takes no new
  job (422 `request_closed`) until it is reopened, and stays writable.
- **Items** are `{key, title, depends_on, estimate_minutes, workflow,
  arguments, accepted, notes, review_until}`, matched across writes by `key`;
  the order written is the order kept. An item left out of a write is deleted,
  or kept as `dropped` (moved to the end) when jobs link to it; writing a
  dropped key again, not dropped, brings it back with everything it held. Each
  item reads back with derived fields (below). A field not listed is refused on
  write, apart from the derived ones and `notes`, which are ignored - so a plan
  can be written back as it was read.
- **`purge`** on a write lists keys of stored items the plan leaves out
  that are to be removed for good instead of kept as dropped. Their jobs
  keep their history rows, their link and their files; a key written again
  later reads those jobs as its own. Naming an item the plan still lists,
  or one that is not stored, is refused.
- **`depends_on`** names other keys of the same request, at most 16, never
  the item itself, and never in a cycle. An item others still depend on
  cannot be dropped.
- **`accepted`** names a succeeded job linked to that request and item: `{"job": "<id>"}` or the bare job id, stored as `{job, run_id}`. It can be moved to another such job, or cleared with null.
- **`notes`**, on an item and on the request, are the user's: a list, so a
  second note does not replace the first. A note is `{id, text, created, at,
  refs}`: `id` is assigned by the server (`n1`, `n2`, ...) and never reused,
  `text` is at most 4000 characters, `created` is when it was first saved and
  never changes, and `at` is the last text change. At most 20 notes live on
  one item or on the request.
  A note carries `refs`, the things it is about: each is `{item, job?,
  output?}` and reads back with the output's `kind`, the take's `attempt`
  (1-based, oldest first) and `gone: true` once the item was dropped or the
  job's run no longer exists. A reference with only `item` is a planned shot
  that may have no job yet, so a note can act before a render. `job` must be
  linked to that item, and `output` must be a file in that job's manifest,
  named as the gallery names it. An item's note references only that item; the
  request's own note may reference several items, at most 16 references in
  all. A bad reference refuses the whole note (422 `invalid_note_ref`). A
  reference is fixed once written and is kept as written if it goes stale.
  Notes are written only by the user and only through `POST`, `PUT` and
  `DELETE /api/workspaces/{name}/request-notes`, which edit the text of one
  note (references stay) or remove it, add a revision and a `user` history
  entry, and need no revision from the caller. A whole-plan write, from the
  user or the agent, ignores any `notes` it sends and keeps the stored ones,
  so a plan read earlier cannot delete a note added since. The agent reads them on its
  next `get_request`, before it writes again. A request saved before notes
  were a list reads its old `comment` as the first note.
  A note waits while the request is open, its item is not dropped and the
  request's last agent write (`agent_wrote_at`) is not later than the note's
  `at`; the agent's next `set_request` answers every note waiting then.
  Nothing is sent when one is saved, unless a Claude Code session is open
  with [the notes channel](MCP.md#the-notes-channel), which tells that
  session about each note as it is saved, with its references.
- **Review hold.** An agent can hold a planned item for the user to look at
  before it runs (a `revised` item can be held too). An item carries three fields: `review_until`, an ISO time
  with an offset that the agent writes; `review_set_at`, the time the server
  stamped when that value was set (ignored on write; writing the same
  `review_until` back keeps it, so a plan the UI reads and writes unchanged
  does not restart the hold); and `review`, derived on each read and ignored
  on write. A new `review_until` must fall at least 60 seconds
  (`MIN_REVIEW_SECONDS`) and at most 24 hours (`MAX_REVIEW_HOURS`) ahead,
  or the write is refused with 422 `review_too_soon` or `review_too_far`;
  a value that is not an ISO time with an offset is 422
  `invalid_review_until`. `review` is `{until, state, seconds_left}` and is
  present only while the item's `status` is `planned` or `revised`, the request is `open`
  and `review_until` is set. `state` is the first that holds: `redirected`
  (a note on the item, or one of the request's own notes, was saved at or
  after `review_set_at`), `due` (the deadline has passed), else `holding`.
  `seconds_left` is a whole number, never below 0. Only a note redirects:
  rewriting the plan, even with the old `review_until` echoed back, does
  not clear a redirect; only a new `review_until` starts the hold again.
  A `review_until` left out of an item, or null, clears its hold, so a plan
  rewrite must send each held item's `review_until` back as read. The
  server runs nothing at the deadline: with
  [the notes channel](MCP.md#the-notes-channel) open, a `review_due` event
  tells the agent session, and the agent runs what it quoted. With no such
  session listening, a due item simply stays `ready` until someone runs it.
- **Keys** (request ids and item keys) match `^[a-z0-9][a-z0-9-]{0,63}$`.
- **Caps.** Titles 200 characters, prompt and goal 20000, note 4000;
  `estimate_minutes` 0 to 10000; `arguments` at most 64, scalar values
  only, names as variable names, strings at most 20000. No text carries
  control characters. A workspace holds 100 **open** requests and a request
  200 **live** items: what is finished does not count against either.
  Closed requests have their own cap of 1000 and a request's dropped items
  theirs of 200 (409 `too_many_dropped_items`), and the whole file about
  2 MB (413 `ledger_too_large`). Each refusal says what frees room: close
  or delete a request (409 `ledger_full`), `purge` dropped items. A request
  already open, or already closed, stays writable at its cap.
- **Deleting** removes only the ledger entry, at the `revision` read (409
  `stale_revision` otherwise; 404 `unknown_request`). Every linked job keeps
  its history row, its `request`/`item` and its files, so the job and
  gallery filters still find them, and a request written again under the
  same id reads those jobs as its own. Refused with 409
  `request_has_active_jobs` (naming them in `jobs`) while a linked job is
  queued or running.

**Linking jobs.** `run_workflow(request="<request-id>/<item-key>")` (or
`request` and `item` in the `POST /api/jobs` body) links the job to that
item. The client refuses anything but exactly one `/` before any call; the
server refuses 422, before anything queues, a half that isn't a valid key, or
a link with `unknown_request`, `request_closed`, `unknown_item` or
`item_dropped`. A rerun inherits the link; if it no longer admits, the rerun
still queues and keeps the link, with a `rerun_ledger_link_<code>` warning. `get_job` and `list_jobs` show `request`/`item` on a linked job, and
`list_jobs(request=...)` and `list_gallery(request=...)` (`?request=&item=`
over REST) keep one request's, or one item's, jobs and files.

An item's `workflow` and `arguments` are stored, not run: nothing in the
ledger queues a job. Running an item is an ordinary submit of those two
values with the link, through the same admission as any other job - the
arguments are checked against the workflow then, and a bound cost
acknowledgement is checked against the plan `validate` answered.

**Derived, never stored.** Each read works the item out from its linked
jobs. `status` is the first that applies: `dropped`, `done` (a linked job
succeeded), `running`, `queued`, `failed` (every linked job failed or was
cancelled), `planned` (none). `revised` replaces `done` when the plan moved
after the take: no succeeded take is current and no acceptance was recorded
since. A take is current when the item's `changed_at` is not later than the
take's `created_at`, or the item's `workflow` and `arguments` equal the
take's as submitted, and every `prompt:` reference in the item's current
`arguments` resolves to text whose digest equals the take's recorded
`workflow.prompt_digests` entry. Only a prompt's text counts: saving the same
text, or changing only its description or tags, changes nothing. A take from
before digests were recorded is never revised by a prompt edit, a prompt that no
longer resolves counts as unchanged, and references inside the saved workflow's
definition do not count. A revised item keeps its `result` and `attempts`,
derives `revised_because` (`["arguments"]`, may include `workflow`, or
`prompt:<name>`), counts in
`remaining_minutes` again, and is held and run like a planned one; its
dependents are not ready, since revised is not done. A dropped item is never
revised, and a take run with other arguments but no plan write does not
revise the item. The server stamps `changed_at` (when the item is created or
its `workflow` or `arguments` differ from the stored ones; title, estimate,
`depends_on` and review edits do not move it) and `accepted_basis` `{at}`;
both are ignored on write, so a GET then PUT changes nothing, and an item
stored before the stamp (null `changed_at`) is never revised until its plan
changes. A write that names an `accepted` take (any linked succeeded take,
the accepted one included) while the stored item reads revised, leaving
`workflow` and `arguments` as they were, records `accepted_basis`: the item
reads `done` again, keeping that take as its result, until the plan changes
again. Because writes are whole plans, writing back a revised item that
already carries `accepted` counts as re-accepting it. The other ways back
to `done`: a new take that succeeds, or writing the arguments (and workflow)
back to equal a take's. A re-accept also records `prompts: {name: digest}` in
`accepted_basis`, and holds until a prompt's text changes again; that basis
judges the accepted take only, so a newer take run on the current text still
makes the item `done` (the accepted take stays its result). A prompt-driven
revision is no ledger write: `save_prompt` adds no revision or history entry
and wakes no notes-channel event, so an agent sees it on its next
`get_request`. The server reads each distinct prompt once per view, memoized
by (path, mtime). `ready` is true when a job could be queued
for the item now: it is `planned` or `revised`, stores a `workflow`, every item it
depends on is `done`, and the request is open. `attempts` lists the jobs
oldest first as `{job, status, note?, minutes, output}`; `result` is the
accepted job's `{job, run_id, output}`, else the latest succeeded job whose
run still exists, else null; `accepted_missing: true` appears when the
accepted job's run is gone. `output` is the one file that stands for the
job, `{name, kind}` or null: the first image or video of its `final`
entries, else their first sound or text, falling back on its other entries;
`name` is the file as the job's manifest and the gallery name it (relative
to the workspace's outputs, the name `GET /api/gallery/{name}/thumbnail`
takes) and `kind` is `image`, `video`, `audio` or `text`. It is null for a
job that wrote nothing the gallery shows, or whose run directory is gone.
`spent_minutes` sums the attempts' minutes, per item and per request, and
the request's `remaining_minutes` sums the estimates of items neither done
nor dropped (a revised item counts). A list row carries the counts by status (`revised` appears when any), `ready` (how many
items are), and `cover`: up to three `output`s of the latest results,
newest first. To keep a take, read `get_request`, then write the plan back
with `set_request` and that item's `accepted: {"job": "<id>"}`.

**The file.** Every write holds a file lock beside the ledger
(`.requests.json.lock`), so two servers on one workspace cannot lose each
other's write; one that waits more than 10 seconds for it is refused with
503 `ledger_busy`. The file is replaced atomically. A `requests.json` that
is there but is not a ledger - unreadable, not JSON, or JSON of another
shape - is never replaced: the list answers with `unreadable` saying why,
and every other read and write, and any new link, is 409
`ledger_unreadable` until the file is fixed or removed.

A server started from loose directories has a default workspace with no
root folder: its ledger reads as empty and a write is refused with 409
`no_workspace_root`.

## Several workspaces on one server

Everything above describes one workspace. `dw.serve` goes one step
further: the workspace root can hold several, and a client picks which one
it is working in - `dw.run --workspace NAME` among them.

```
<workspace root>/
  workflows/  assets/  outputs/    <- the 'default' workspace
  prompts/                         <- shared by all of them
  loras/                           <- the LoRA catalog, shared by all of them
  common/assets/                   <- shared by all of them
  studio/
    workflows/  assets/  outputs/  <- the 'studio' workspace
  scratch/
    workflows/  assets/  outputs/  <- the 'scratch' workspace
```

The root's own three folders are the workspace named `default`. A named
workspace is a sibling directory holding the same three folders and no
`prompts/`: there is one prompt library, because a `prompt:` reference
duplicated per workspace would resolve to different text depending on where
a workflow was saved.

Reserved names, which no workspace can take: `workflows`, `prompts`,
`assets`, `outputs`, `exports` (where `POST /api/jobs/{id}/export` gathers a
job, `<root>/exports/<job id>/`), `common`, `loras` (the LoRA catalog,
shared like `prompts/`), and `requests.json` and `workspace.json` (the
default workspace keeps its ledger and its note as files of those names at
the root, where a workspace would be a folder). They are reserved in any
letter case - `Requests.json` is the ledger file on macOS and Windows - and
a folder at the root under one of them in another case is never listed as a
workspace. A pre-existing workspace named `loras` stops being listed, and
the server logs a warning at start naming its directory.

A new workspace cannot take a name the root already holds in any letter
case, whether that entry is a workspace, a plain folder or a file
(`409`): `Shots` beside `shots` would be one folder on a case-insensitive
disk, and a folder that is not a workspace is never taken over.

**The shared asset library.** `common/assets/` holds assets that belong to
no single workspace, such as a recurring cast's portraits and voice clips.
It sits on every workspace's asset search path behind that workspace's own
library, so:

- `asset:cast/priya.png` resolves in the workspace first, then in the shared
  library, then in any read-only examples library — a workspace's own name
  still shadows a shared one
- `GET /api/assets` spans all of them and tags each entry's `origin`:
  `workspace`, `common`, or `examples`
- writes still land in the workspace unless they say otherwise:
  `POST /api/uploads?shared=true`, `POST /api/assets/keep` with
  `"shared": true`, and over MCP `upload_asset(..., shared=True)` /
  `keep_output(..., shared=True)`

It holds assets only. A prompt is already shared, and workflows and outputs
belong to the work that made them.

Two agents can share the server's GPUs without sharing a namespace: each
takes a workspace, and neither can save over the other's workflows or delete
the other's renders. The queue is shared and the engine runs one job per GPU
(`dw.serve --devices`), so their jobs run side by side when each fits a free
card, and queue otherwise.

**How a client picks one.** Every scoped route takes an optional
`?workspace=<name>`; omitting it means `default`. `GET /api/jobs` spans every
workspace unless one is named.

| Client | How |
| --- | --- |
| Web UI | The sidebar lists every workspace; the selected one is named in the hash (`#/ws/<name>/...`), so a link and a reload both land where they say. The choice is remembered in `localStorage` as a fallback for a route that names none (Shared, Server), and Server → Status adds a filter over the all-workspaces queue — job history spans every workspace there and says which one each job ran in |
| MCP | `list_workspaces`, then `use_workspace(name)`: a session default, so switching is one visible step in the transcript. Most tools also take `workspace=` per call; over a shared `/mcp` mount see [REMOTE.md](REMOTE.md#claude-code-no-local-install) |
| HTTP | `?workspace=` on the route, or `"workspace"` in a `POST /api/jobs` body |
| Web UI (create/delete) | The sidebar's `+ new` creates one; a workspace's own Overview page deletes it (disabled for `default`) |

A job carries its own workflow, asset and output directories, so it stays in
the workspace it was submitted from however many others the server serves
while it runs — including through a rerun, and when its files are served back
from history.

**Creating and deleting.** `POST /api/workspaces` (`create_workspace` over
MCP) makes one; creating does not switch to it (over MCP, unless
`use=true`). Deleting removes everything in
it, so it refuses until acknowledged and answers first with what it would
remove — file counts and bytes per folder. The default workspace cannot be
deleted (it holds the shared prompt library, and there has to be somewhere to
work), nor can one with jobs still queued or running.

A workspace is a **namespace, not a security boundary**. The API token is
all-or-nothing: anything that can reach the server can name any workspace on
it. Use them to keep work apart, not to keep it private.

## Copying between workspaces

A run is self-contained on disk, so copying its folder into another
workspace's `outputs/` is what makes it resolvable there. `copy_to_workspace`
(MCP) and `POST /api/workspaces/{destination}/copy` ([SERVER.md](SERVER.md))
do that for runs, named `<workflow>/<run id>` or, for all of a workflow's runs,
a bare `<workflow>`, and for assets, named as after `asset:` (`cast/iris.png`,
or a folder such as `proj/` for every file under it). An asset keeps its
library-relative path, so `uploads/` stays `uploads/` and an `asset:` reference
means the same file in both workspaces. Its note and its kept-shots entry (the
folder's `notes.json` and `manifest.json`, including where it was kept from)
are merged into the destination folder's own sidecars, which are written
fresh: naming a sidecar directly is a 400, since it travels with its files.

Media is hard-linked, so a copy costs almost no disk, falling back to a real
copy across devices. JSON sidecars (the manifest, `workflow.json`, notes,
any `.json`) are always copied, never linked: they are edited in place, and a
link would carry an edit in one workspace into the other. Nothing follows or
creates a symlink.

The copy is all or nothing: every entry is checked before anything is written.
A run whose run id the destination already has is the same run, so it is
skipped and listed as `already_present`, and repeating a copy changes nothing.
A run whose job is still queued or running is refused, since its manifest is
not final. An asset the destination already holds with the same bytes is
skipped as `already_present` too; a different file at that path is a conflict,
never replaced. Every conflict, runs and assets alike, is named in one 409
before anything is written.

Because a copied asset is a hard link, nothing edits media in place: an upload
over an existing name writes a new file and renames it into place, and
`keep_output` and `delete_asset` unlink first, so an edit in one workspace
never reaches the other.

A run keeps its `vN` unless the destination already uses that number for the
workflow. Then this request's runs of that workflow are renumbered after the
destination's highest, in run-id order, under the run lock so a concurrent run
cannot take the same number. `renumbered` reports each `from` and `to`. Pinned
`output:` references name run ids, so none of them break.

The reply carries warnings, not refusals: `unresolved_output_refs` (a run's
`output:` input that does not resolve in the destination), `absolute_paths`
(manifest paths that still point into the source workspace, reported relative
to the source workspace's root) and `no_jobs`. A copied run has no job record, so it appears in
the gallery and resolves as an `output:` reference, but not in `list_jobs`.

A move is a copy, then a delete of the source runs (`delete_output`, or
`DELETE` on the gallery runs) and assets (`delete_asset`); there is no move that could lose a run halfway.

## Running a sweep

A sweep answers one quality or speed question by rendering the same request
under several configurations, changing one factor at a time, and keeping the
results together. There is no sweep tool: every step is one call to a tool
dw already has.

1. **One workspace per experiment.** `create_workspace` a workspace for it,
   then `use_workspace`. Put the question and the factor list in its
   workspace note, so the experiment explains itself to the next session:
   `set_note("workspace:<name>", "Q: does steps change the face? factors: steps 4/8/12")`.

2. **A base workflow with every factor as a variable.** If a catalog
   template already exposes the factor as a variable (`get_workflow(name,
   variables_only=true)`), run the template. Otherwise `get_workflow` it,
   add a variable for each factor, put `variable:<name>` where the value
   was, and `save_workflow` the copy into the experiment's workspace.
   `text-to-image` fixes `num_inference_steps` at 25, so a steps sweep saves a copy
   with `"steps": 25` under `variables` and `"num_inference_steps":
   "variable:steps"` in the step. A factor that is a dtype or a quantization
   name keeps its braces in the default, `"{nf4}"`, so it reaches the
   pipeline as a string ([Types and escaping](WORKFLOW_GUIDE.md#types-and-escaping)).
   Keep the template's `cost` block in the copy, or `plan.estimate` has no
   basis to quote from.

3. **Pin the seed, change one factor per variant.** Every variant passes the
   same `seed` in `arguments`, and differs from the base in one factor only.
   A factor can change the cost, so `validate_workflow` each variant with its
   `arguments`, quote each one's `plan.estimate`, and get one go-ahead
   for the whole list.

4. **One `run_workflow` per variant, with a `note`.** The note's first line
   is the overrides, `steps=8` or `steps=20 attention=sage`, so a listing
   that clips notes still shows which variant is which. Call `wait_for_job`
   until the variant finishes before submitting the next. The server holds a
   second large variant until the host can hold it beside the first (see
   [Dispatch](SERVER.md#dispatch)), so waiting is for ordering, not for memory.

5. **Only the latest run of a step is cached.** The step cache holds one
   entry per workflow step: its most recent run. A variant whose step and
   seed match that run, such as the same variant submitted twice in a row,
   is served from the cache: its manifest marks the step `"reused": true`
   and lists that run's files ([Runs](#runs)). Going back to an earlier
   variant after another one has run renders it again. Each card's worker
   keeps its own cache, and a server restart empties them. Either way a
   re-render is expected, not a failed variant.

6. **The results table.** `list_jobs(workspace=<name>)` gives each variant's
   job id, status and note. `get_job` on each gives its arguments, step
   `timing` ([Step timing](#step-timing)) and files.

7. **Assess each output with `set_note`.** Put the note on the output file
   (its gallery name, as `list_gallery` gives it). The first line is the
   score and who judged it, then one `key: value` line per facet:

   ```
   score 4/5 · judged_by claude-opus-5-5
   faces: sharp, eyes aligned
   staging: matches the brief
   defects: none
   ```

   `get_gallery_metadata` returns the whole note. `list_gallery` and
   `list_jobs` clip a note to its first 200 characters, so keep the
   first line short: it is what a listing shows.

8. **Keep the winner.** `keep_output` the winning file under a stable
   `asset:` name, and put the conclusion (the answer, and the variant that
   gave it) in the workspace note.
