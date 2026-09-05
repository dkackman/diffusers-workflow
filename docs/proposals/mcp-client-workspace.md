# Proposal: a client-side workspace for MCP

Status: draft / scoping only — no implementation.
Follows [workspaces.md](workspaces.md), whose stages one through five and
stage six level one are implemented. This is level two.

## Problem

`dw_mcp` is an HTTP client of a running `dw.serve`, so it has no workspace of
its own: it inherits the server's. Everything an agent authors — workflows,
prompts, uploaded assets — lives on the machine with the GPU, under that
server's directories.

For a single-user box that is fine, and it is what the workspace work already
made safe (saves land in the workspace's `workflows/`, never in a checkout).
It stops being fine when the agent and the GPU are different machines, which
is the case `dw-mcp --url` and `dw.serve --mcp` exist to serve:

- Work an agent authors on a laptop is stored on someone else's box, mixed in
  with everyone else's, and is gone when that box is reimaged.
- The agent cannot keep its material under version control, or in the
  directory the rest of the project lives in.
- Two agents against one server share one namespace. A workflow named
  `Gyre` is *the* `Gyre`.

The goal is the one the original proposal stated: **the server becomes a GPU,
and the workspace is the user's.**

## What level one already supplies

- `upload_asset` reads a file on the machine the MCP server runs on and pushes
  its bytes. That is the primitive the original proposal called out as
  missing, and it is the hard one.
- `asset:` references are rooted at the server's asset library, *not* at the
  workflow file. An inline workflow whose inputs are `asset:` references needs
  no `base_dir` at all — which sidesteps the confinement problem the original
  proposal flagged as level two's real design work. `POST /api/jobs` already
  accepts an inline definition, and the engine resolves `asset:` against the
  server's library wherever the definition came from.
- `list_assets` lets the client see what the server already holds before
  pushing anything.
- Validation is already free and server-side (`validate_workflow`), so a
  local workspace needs no local schema validation and `dw_mcp` keeps its
  no-`dw.*`-imports boundary.

So level two is mostly *bookkeeping*: keep the files locally, and translate a
local definition into one the server can run.

## Proposed shape

### 1. `dw-mcp --workspace <dir>`

The same resolver the engine uses would be wrong here — it lives in `dw/`, and
`dw_mcp` may not import it (that boundary is what keeps torch out of a pure
HTTP client). `dw_mcp` gets its own small resolver over the same shape and the
same environment variable:

```
<workspace>/
  workflows/   authored here, never uploaded as files
  prompts/     authored here
  assets/      local input media
  outputs/     what `fetch_run` brings back (stage 3 below)
```

Without `--workspace`/`DW_WORKSPACE`, every tool behaves exactly as it does
today — server-side only. This is opt-in; nothing changes for a local setup.

### 2. Local as a *location*, not a second set of tools

The tool surface is already ~35 tools; doubling it would cost more than it
buys. Instead the workflow tools take a location:

```
list_workflows(where="server" | "local" | "both")   default: "both" when a
                                                    workspace is configured,
                                                    "server" otherwise
get_workflow(name, where=...)
save_workflow(name, workflow, where=...)            default: "local" when
                                                    configured
delete_workflow(name, where=...)
```

Every listing entry already carries `origin` and `writable` from stage four;
a local entry reports `origin: "local"`. That keeps one vocabulary for "where
did this come from and may I change it" across both sides.

The default of `save_workflow` is the one real judgment call. Defaulting to
`local` when a workspace is configured is what makes the feature do what it
was asked for without the agent having to remember a parameter — and an agent
that means "put this on the server for the web UI to see" can say so.

### 3. Running a local workflow: sync, then submit inline

`run_workflow(workflow_path="local:<name>")` — or `where="local"` — does:

1. Read the local definition.
2. Walk it for `asset:` references. For each, find the file in the local
   `assets/`, and make sure the server has it (below). Rewrite the reference
   to the server-side name.
3. Walk it for `prompt:` references and **inline the text** from the local
   prompt library, rather than pushing prompts into the server's. A run of a
   local workflow should not leave anything in the server's library, and the
   submitted definition is a copy already; inlining also means the job's
   recorded definition reproduces exactly.
4. Submit the rewritten definition inline to `POST /api/jobs`. No `base_dir`
   is needed or sent, because nothing in the definition resolves relative to
   a file any more.
5. Report what it uploaded alongside the job id, so the agent (and the user
   reading the transcript) can see the side effects.

Two shapes must be refused, clearly rather than mysteriously:

- **Plain relative paths** (`"image": "assets/iris.png"`). These resolve
  against the workflow file's directory, which does not exist on the server.
  The error should name the fix: use an `asset:` reference.
- **Sub-workflow steps naming a local file.** Syncing those means recursively
  uploading workflows, which is a second design; refuse for now and say so.
  `builtin:` sub-workflows are fine — they live in the engine.

### 4. The server change: content-addressed uploads

`POST /api/uploads` names what it stores `<uuid>.<ext>`. That makes every
upload distinct, so a sync would re-push a 40MB reference video on every run.

Name uploads by a digest of their content instead — `uploads/<sha256[:16]>.<ext>`:

- Re-uploading the same bytes is idempotent and returns the same reference,
  so the sync's "does the server have it?" is answerable from `list_assets`
  alone, with no new route and no protocol negotiation.
- The browser's file picker gets the same dedupe for free.
- Nothing is lost that is not already lost: the stored name is a generated
  one today.

Two files with identical content become one asset. That is correct for a
content-addressed store and worth stating in the docs.

The upload timeout needs attention: `dw-mcp --timeout` defaults to 30s for
*any one request*, which a 200MB push over a home connection will exceed. The
upload path should use a separate, larger budget rather than making users
raise the timeout for every call.

### 5. Bringing results back (optional, last)

`fetch_run(job_id)` writes the job's files and its manifest into the local
workspace's `outputs/<identity>/<run id>/`, using the manifest stage three
already writes. The local workspace then holds the whole record — what was
run, and what came out — and `output:` references would resolve locally for
whatever comes next.

This is genuinely optional: `get_output_image` and `download_output` already
cover "show me" and "save this one file". It matters for the case where the
agent's machine is where the work is kept.

## What this does not do

- It does not make `dw_mcp` run workflows. There is still exactly one engine,
  on the machine with the GPU.
- It does not replicate the server's catalog locally, or sync in the
  background. Sync happens at submit time, for one workflow, and is reported.
- It does not change trust: an inline definition is untrusted on the server
  unless it was started with `--trust-workflows`, so a locally-authored
  workflow reaching for `pre_load_modules` is refused there. That is the
  correct default and the error already says so.

## Alternatives considered

- **Mount the workspace over the network** (NFS/SMB) and point `dw.serve` at
  it. Zero code, and legitimately the right answer on a LAN with a file
  server. It fails the laptop-and-a-remote-GPU case this is for, and makes
  the engine's path confinement depend on a mount staying up.
- **Run `dw.serve` locally, pointing at a remote GPU.** Not possible: the
  engine and the GPU are the same process tree by design.
- **Do nothing.** Defensible while every user is single-box. The cost is that
  `--url`/`--mcp` remote use stays second-class: authored work accumulates on
  the server, under one namespace.

## Open questions

- **Name collisions.** A local `Gyre` and a server `Gyre` are different
  workflows. `where="both"` has to present that unambiguously — probably by
  reporting both with their origins rather than silently shadowing, which is
  the opposite of what the server-side search path does within one catalog.
- **Does the local workspace want its own search path**, with the repo's
  examples on it read-only, mirroring stage four? Probably yes eventually,
  and it should reuse the same `origin`/`writable` vocabulary.
- **Digest length.** 16 hex characters of SHA-256 is ~64 bits: collision-safe
  for a personal library by a wide margin, short enough to read. Worth
  confirming rather than assuming.
- **Should `fetch_run` be automatic** on job completion when a workspace is
  configured? Convenient, but it turns a poll into a large download the agent
  did not ask for. Probably explicit.
