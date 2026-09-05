# Proposal: workspaces on the server, and client workspaces that mirror into them

Status: draft / scoping only — no implementation.
Follows [workspaces.md](workspaces.md), whose stages one through five and
stage six level one are implemented. Replaces the first draft of level two,
which made a client-side workspace the system of record and submitted runs as
anonymous inline definitions.

## What changed since that draft

The first draft asked the server to run work it had no record of. An agent
would author into a laptop workspace, and at submit time its assets would be
pushed and its definition rewritten and sent inline. Two problems with that,
one practical and one conceptual:

- **The web UI cannot see any of it.** The gallery, editor and prompts pages
  read the server's directories. Work authored through MCP would exist only in
  the agent's transcript and in a job's recorded definition — invisible to the
  browser the same user has open on the same machine. For a single user
  working through both surfaces, that splits their material in two.
- **Rewriting at submit time is a translation layer** — inlining prompts,
  rewriting asset references — that exists only because the server has nowhere
  to put a workflow that is not its own.

Giving the server named workspaces removes both. The agent's work lands
somewhere real, the UI can browse it, and a run is submitted by name like any
other rather than as a rewritten copy.

## The ownership model

One owner per class of artifact, not per workspace:

| Artifact | Canonical | Flows |
|---|---|---|
| Workflows, prompts, assets | The client, when one is mirroring | client → server |
| Outputs, run manifests, job history | The server | server → client, on request |

Nothing is canonical in two places, so there is no merge to get wrong. A
client workspace stays a plain directory under the user's control — in their
project, under version control, backed up with everything else — and the
server holds a materialized copy plus everything the GPU produced.

A user with no client workspace at all (the web UI alone, or `dw-mcp` on the
same box) is unaffected: their server workspace is owned by the server, and
is writable from every surface exactly as today.

### The one real conflict, and how it is settled

A mirrored workspace is **read-only on the server**. If the UI could edit it,
the next mirror would silently clobber that edit.

Stage four already has the vocabulary: `GET /api/workflows` tags every entry
with `origin` and `writable`, the UI hides delete and marks read-only sources,
and a save resolves to the writable root instead of overwriting. A mirrored
workspace reports `origin: "mirror"`, `writable: false`, and "save" from the
UI writes a copy into a server-owned workspace — the same gesture that already
works for an example. No new concept, and the UI change is a listing that
already carries the flag.

## Workspaces on the server

```
<workspaces root>/
  default/          server-owned, writable
  studio-don/       server-owned
  laptop-mirror/    mirrored from a client, read-only here
    workflows/ prompts/ assets/ outputs/
```

One server process, workspaces as a dimension of a request — not one process
per workspace. A second server process would duplicate the model cache in host
RAM, which on this hardware is the expensive resource, and buy no concurrency:
jobs serialize on the one GPU anyway.

### Routes

- `GET /api/workspaces` — name, owner (`server` / `mirror`), writable, sizes,
  which is the default
- `POST /api/workspaces` — create one
- `DELETE /api/workspaces/{name}` — remove it. Deletes generated work, so it
  needs the gallery's bulk-delete care: a count and a confirmation, never a
  quiet success
- Every existing route gains an optional workspace selector, defaulting to the
  server's configured default, so nothing that works today changes shape

### The plumbing this needs

Two of the four roots already travel with a job; two do not:

| Root | Today |
|---|---|
| `output_dir` | already per-job on the worker protocol (`dw/worker.py`, `dw/server/jobs.py`) |
| `workflow_dir` | already per-job — stage four made confinement travel with the job |
| `prompt_dir`, `asset_dir` | process-wide, pinned into the environment at startup (`dw/serve.py`) and inherited by the spawned worker |

So the work is moving the prompt and asset roots from environment pinning to
per-job values. Stage five set the precedent: `activate_output_root` is a
contextvar the run activates and `resolve_output_reference` reads, and
`get_prompt_dir`/`get_asset_dir` would take the same shape — the environment
variable staying as the fallback for the CLI and REPL, which have one
workspace per process and always will.

What stays process-wide and should: the model cache (keyed by what a pipeline
loads, so two workspaces naming the same model share it — which is the whole
point of the persistent worker) and the step cache (keyed by workflow id, step
and output root, so entries from different workspaces cannot collide).

`jobs.sqlite` needs a workspace column, or history stops making sense the
moment there are two.

## Mirroring

### Direction and trigger

One way, client → server, for authored content only. Two triggers:

- **On save.** `save_workflow` / `save_prompt` write locally and push. The
  common case costs nothing extra and keeps the UI in step with the agent.
- **`sync_workspace()` explicitly**, for reconciliation — first use, a
  workspace edited outside the agent, or after working offline.

### What crosses, and how change is detected

Workflows and prompts are small JSON: push the ones whose content differs.
Assets are not, so the mirror compares digests and pushes only what changed.

**This supersedes the previous draft's content-addressed upload naming.** A
mirror has to preserve names: `asset:gyre/frames/iris.png` must mean the same
file on both sides, or every reference in every mirrored workflow needs
rewriting — the translation layer this design exists to remove. So the upload
route grows a destination name (validated by the existing asset-name validator
and confined to the library, like every other client-supplied path), and the
digest is used for *change detection*, not for naming.

The browser's file picker keeps generated names; it has no name worth
preserving and no mirror to keep consistent.

### What the mirror does not do

- **It does not delete.** A file gone locally stays on the server unless the
  user asks (`sync_workspace(prune=True)`). Deleting generated work as a side
  effect of a sync is not a thing to do by default.
- **It does not pull.** The server never writes back into a client workspace;
  that is what makes the ownership model hold.

### Running a mirrored workflow

By name, in the mirrored workspace, like any other run. No inline rewriting,
no prompt inlining, no `base_dir` question — because the mirror already put
the workflow, its prompts and its assets on the server under the names its
references use. This is the simplification that server-side workspaces buy,
and it is most of why the design changed.

## Results

The server is canonical for what it generated. `fetch_run(job_id)` copies a
run's files and its manifest into the client workspace's
`outputs/<identity>/<run id>/`, and that copy is a cache: deleting it loses
nothing, and `output:` references resolve on whichever side is running.

## What this does not do

- It does not make `dw_mcp` run workflows. One engine, on the machine with the
  GPU.
- It does not make a workspace a security boundary. The API token is
  all-or-nothing: anyone who can reach the API can reach every workspace.
  Workspaces are a namespace — for keeping one person's work separate from
  another's, not for keeping it private from them. This must be stated plainly
  in the docs, or someone will assume otherwise.
- It does not change the trust model: a mirrored workflow is a stored workflow
  like any other, and `--trust-workflows` governs it the same way.

## Staging

1. **Workspaces on the server, one at a time.** The registry, the CRUD routes,
   the per-job prompt and asset roots, the workspace column in history. The
   default workspace behaves exactly as the single workspace does today.
2. **The UI catches up.** A workspace switcher; the gallery, workflows and
   prompts pages scoped to the selection.
3. **MCP selects a workspace** — per call, defaulting to one per session. This
   alone covers the remote-agent case without any client workspace at all:
   an agent gets its own namespace on the box and the user can see it.
4. **Client workspaces mirror.** `dw-mcp --workspace`, push on save,
   `sync_workspace`, the named upload destination, read-only marking of a
   mirrored workspace.
5. **`fetch_run`.** Optional, and last.

Stage 3 is the natural stopping point if mirroring turns out not to be worth
it — it is useful on its own, which is a good property for the stage before
the speculative one.

## Open questions

- **Does a mirror own the whole workspace, or a directory within it?** Whole
  workspace is simpler to reason about and to mark read-only. Per-directory
  would let an agent mirror workflows while using the server's shared prompt
  library — appealing, and probably a later refinement rather than a first
  cut.
- **What is the default workspace called, and where does the root live?**
  `~/diffusers-workspace` is one workspace today; the multi-workspace root
  could be that directory's parent, or a `workspaces/` inside it. The upgrade
  path for an existing single workspace has to be answered before the first
  route is written.
- **How does a client name its mirror?** Explicitly (`--mirror-as
  laptop-don`) is predictable; derived from the hostname is convenient and
  collides the first time someone has two checkouts.
- **Should the UI offer "copy this into my workspace"** as a first-class
  gesture between workspaces, the way stage four's save-a-copy works between
  sources? It is the same operation one level up.
