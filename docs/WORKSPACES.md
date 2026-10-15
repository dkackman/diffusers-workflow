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

This exists so that day-to-day work does not have to live inside a checkout of
this repository. The examples under the repo's `workflows/` are still examples
— a corpus to read, copy and run — but they are not where your own workflows
belong, and generated media does not belong in a source tree at all.

## Which directory is used

The workspace is resolved by `dw.serve`. `python -m dw.run` is a client of a
running `dw.serve`, so it resolves no directory of its own: its jobs land in
the server's workspace. First match wins:

1. `--workspace <dir>` on `dw.serve`
2. the `DW_WORKSPACE` environment variable
3. `"workspace"` in `~/.diffusers_helper/settings.json`
4. the working directory, when it holds any of `workflows/`, `prompts/` or `outputs/`
5. `~/diffusers-workspace`

Rule 4 is why nothing changes when you work from a checkout: the repository
root holds all three, so it resolves to itself and every default lands exactly
where it always has. Only a working directory with none of those folders falls
through to the home workspace.

Nothing is created just by resolving. A command that is about to write creates
what it needs — `dw.serve` creates the workspace's `workflows/` so the UI has
somewhere to save, and a run creates its output directory.

`dw.run --workspace NAME` is a different thing: it names one of the server's
workspaces (the `default` one, or a named one beside it), not a directory. A
path there is refused as a workspace name rather than resolved.

## Overriding one folder

The per-directory flags are `dw.serve`'s, and each overrides exactly one
folder of the workspace:

```bash
python -m dw.serve --workspace ~/studio --output-dir /mnt/big-disk/renders
python -m dw.serve --prompt-dir ~/shared-prompts
```

`dw.run` takes none of them - every job it queues writes where the server
does.

`--output-dir` is the one people reach for most: video work fills disks, and
the outputs folder is the one worth putting on another volume.

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
`directories.workspace` alongside the three folder paths.

## The prompt library

`prompt:` references resolve to the workspace's `prompts/` when a workspace was
named explicitly (rules 1–3 above, or `DW_PROMPT_DIR`, which still wins over
everything). A workspace that was merely inferred from the working directory
does not preempt the older discovery — `./prompts`, then the nearest `prompts/`
above the workflow file — so a repository workflow keeps reaching the library
it lives beside. When the workspace is explicit, its `prompts/` becomes the library
even if it does not exist yet, so a checkout's `./prompts` is no longer found once
a standing workspace setting (like `DW_WORKSPACE` or `"workspace"` in settings.json)
is in place; `--prompt-dir` and `DW_PROMPT_DIR` still override it. This follows
the "explicit wins" rule. See [Prompt References](WORKFLOW_GUIDE.md#prompt-references).

## Assets

`assets/` is the input-media library. A workflow argument written as
`asset:name.ext` (or `asset:folder/name.ext`) resolves to that file's path,
rooted at the library rather than at the workflow file — so a workflow and the
media it reads no longer have to sit in the same folder. `--asset-dir` and
`DW_ASSET_DIR` override the folder, and browser uploads land in
`assets/uploads/`, coming back as `asset:uploads/<name>` (and served for
preview under `/inputs/`, since the SPA's own bundles own `/assets/`).

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

The packaged workflows in `dw/workflows/` are deliberately *not* on the path.
They are the pieces a `builtin:` sub-workflow step names, resolved by the
engine where that step is read — not workflows to browse or run on their own.

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
itself. `manifest.json` points at it and lists which prompts were inlined.

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

A later workflow names what an earlier run made with an `output:` reference —
`output:ltx2/Gyre/latest/Gyre-still.0-0.0.png` — so a multi-stage pipeline no
longer needs files copied back by hand. See
[Output References](WORKFLOW_GUIDE.md#output-references).

To keep the previous layout — everything at the output root, with only a
`workflows/`-mirroring subfolder — use `--output-layout flat`, `DW_OUTPUT_LAYOUT=flat`,
or `"output_layout": "flat"` in settings. Scripts that glob the output directory
are the reason to.

### Step timing

A manifest entry for a pipeline step that ran - in `GET /api/jobs/{id}`, in
MCP `get_job`, and on that step's `step_end` event - carries `timing`: where
the step's time went. It is measured in the worker on its monotonic clock, not
taken from event timestamps, because the job keeps only the last 200 events and
the load and encode events are gone from a long job. Every `_s` is seconds to
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
workflow's own history), else no figure. Rows recorded before the column
existed have no key and count as no data. Validate reports it as
`plan.host_memory`.

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

The root's own three folders are the workspace named `default`, so a server
that has never heard of named workspaces behaves exactly as it did. A named
workspace is a sibling directory holding the same three folders — and *not* a
`prompts/`, because there is one prompt library: `prompt:` is shared by
reference, and a prompt duplicated per workspace would resolve to different
text depending on where a workflow happened to be saved. `workflows`,
`prompts`, `assets` and `outputs` are reserved names for that reason.

Three more names are reserved beside `workflows`, `prompts`, `assets` and
`outputs`: `exports`, `common` and `loras`. `loras/` is the LoRA catalog,
shared by every workspace like `prompts/`; a workspace already named `loras`
stops being listed on upgrade, and the server logs a warning at start naming
its directory. `POST /api/jobs/{id}/export` gathers one
finished job into `<root>/exports/<job id>/`, and that folder is never mistaken
for a workspace.

**The shared asset library.** `common/assets/` is the one place an asset can
live that belongs to no single workspace. Assets are otherwise per workspace,
which is right for the inputs of one piece of work and wrong for a recurring
cast: a character's portrait and voice clip uploaded while making episode one
were invisible from the workspace episode four was made in, and the only way
through was to copy the files in. It sits on every workspace's asset search
path behind that workspace's own library, so:

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

This is what lets two agents share the server's GPUs without sharing a
namespace: each takes a workspace, and neither can save over the other's
workflows or delete the other's renders. The queue is shared across
workspaces and the engine runs one job per GPU (`dw.serve --devices`), so two
agents' jobs run side by side when there is a free card each fits, and queue
otherwise.

**How a client picks one.** Every scoped route takes an optional
`?workspace=<name>`; omitting it means `default`, which is why every
pre-workspace call still means what it meant.

| Client | How |
| --- | --- |
| Web UI | The sidebar lists every workspace; the selected one is named in the hash (`#/ws/<name>/...`), so a link and a reload both land where they say. The choice is remembered in `localStorage` as a fallback for a route that names none (Shared, Server), and Server → Status adds a filter over the all-workspaces queue — job history spans every workspace there and says which one each job ran in |
| MCP | `list_workspaces`, then `use_workspace(name)`. It is a session default rather than an argument on each call, so switching is one visible step in the transcript instead of a flag that can be forgotten on the call where it mattered |
| HTTP | `?workspace=` on the route, or `"workspace"` in a `POST /api/jobs` body |
| Web UI (create/delete) | The sidebar's `+ new` creates one; a workspace's own Overview page deletes it (disabled for `default`) |

A job carries its own workflow, asset and output directories, so it stays in
the workspace it was submitted from however many others the server serves
while it runs — including through a rerun, and when its files are served back
from history.

**Creating and deleting.** `POST /api/workspaces` (`create_workspace` over
MCP) makes one; creating does not switch to it. Deleting removes everything in
it, so it refuses until acknowledged and answers first with what it would
remove — file counts and bytes per folder. The default workspace cannot be
deleted (it holds the shared prompt library, and there has to be somewhere to
work), nor can one with jobs still queued.

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
   until the variant finishes before submitting the next. Dispatch is not
   yet aware of host RAM (#869), so two large variants on two cards can run
   the host out of memory together.

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

## Where this is going

The resolver, the workflow search path with writes confined to the writable
root, run directories with an on-disk manifest, `asset:` and `output:`
references, and server-side named workspaces are all implemented. A further
stage was designed but deliberately not built: a client-side workspace (a
laptop directory, under version control) that mirrors into a read-only
server workspace, so an agent could author offline and only push at submit
time. It stayed on the drawing board because source control of creative work
is not this project's job — that is already handled on the client, by the
user, with the tools they already use — which is what makes a mirroring
layer unnecessary rather than merely speculative.
