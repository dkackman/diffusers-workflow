# Proposal: workspaces — decoupling content from the repo (and from the server)

Status: draft / design only — no implementation.

## Problem

The project grew CLI-first, where "outputs next to the code" was correct and
free. A workflow file, the assets it reads, the prompts it references and the
files it writes all resolved relative to the checkout, and that was the whole
story. The web UI inherited that model and mostly gets away with it — it is
single-user and usually started from the repo root. The MCP server is where it
stops working: an agent authoring a workflow writes it into the repo's example
corpus, cannot supply input bytes at all, and has no notion of "my stuff"
versus "the examples that ship with the project".

The examples use case is still valid — `workflows/` as a runnable corpus is one
of the better things about the project. What is missing is a second place for
everything that is *not* an example.

## Current behavior

Four roots, four different resolution rules, three of them anchored to the
process working directory:

| Content | Where it resolves | How |
|---|---|---|
| Workflow files | CLI: any path given on the command line. Server: `--workflow-dir`, default `./workflows`. Packaged: `builtin:name.json` → `dw/workflows/` | `workflow_from_file` / `resolve_workflow_reference` (`dw/server/app.py:192`) |
| Input assets | The **workflow file's own directory** | `resolve_relative_path(path, base_dir)` (`dw/arguments.py:769`); `base_dir` is `dirname(file_spec)` |
| Stored prompts | `DW_PROMPT_DIR` → `./prompts` → nearest `prompts/` walking up from the workflow file | `get_prompt_dir` (`dw/prompts.py:30`) |
| Generated output | `-o/--output-dir`, default `./outputs`, plus a subfolder mirroring the workflow's position under the nearest directory literally named `workflows` | `workflow_output_subfolder` (`dw/workflow.py:104`), `Workflow.effective_output_dir` |
| Settings, job history, logs | `DIFFUSERS_HELPER_ROOT` → `~/.diffusers_helper/` | `dw/settings.py` |

Only the last row is already decoupled. Only prompts has an environment
variable and a discovery walk; there is no `DW_WORKFLOW_DIR` or
`DW_OUTPUT_DIR`, so "start it from the repo root" is load-bearing for
everything else.

### What that costs today

- **Output layout is coupled to repo layout.** `workflow_output_subfolder`
  keys the output subfolder off a path segment literally named `workflows`.
  Move a workflow out of that tree and its outputs silently flatten into the
  output root. The *filesystem shape of the checkout* is the grouping key.
- **Assets must live beside the workflow.** Because `base_dir` is the workflow
  file's directory, any workflow with inputs drags a sibling `assets/` folder
  into the repo — hence `.gitignore`'s `workflows/*/assets/`, `local_inputs/`,
  `/*.mp4`, `/*.wav`. Generated media is being gitignored *inside the source
  tree* rather than kept out of it.
- **The output→input loop is a manual copy.** The GYRE workflows reference
  `assets/Gyre-score_choir.10-0.0.wav` and `assets/Gyre-still_1_iris.0-0.0.jpg`
  — files a previous run wrote to `outputs/` and a human copied back into the
  repo. Multi-stage work (the marmot passes, `outputs/marmot_pass1..3`) is
  exactly the day-to-day case, and the engine has no vocabulary for "the thing
  the last stage made".
- **Inputs and outputs share a directory on the server.** `POST /api/uploads`
  writes to `<output_dir>/uploads` (`dw/server/app.py:1233`) because the output
  dir is the only writable root the server knows about.
- **History and output can disagree.** `jobs.sqlite` lives under
  `~/.diffusers_helper/` and stores manifest paths *relative to* whatever
  `--output-dir` the server was started with (`dw/server/jobs.py:594`). Start
  it from a different cwd and old rows point at nothing. A CLI run records no
  manifest at all.
- **MCP writes into git.** `save_workflow` → `PUT /api/workflows/{name}` →
  the server's `workflow_dir`, which defaults to the repo's example corpus.
  There is no writable/read-only distinction, and no upload tool at all
  (`dw_mcp/CLAUDE.md` notes `POST /api/uploads` is deliberately uncovered), so
  an agent can author a workflow it has no way to supply inputs for.
- **`workflow_dir` is doing two jobs.** It is both "where workflows are kept"
  and the server's *confinement boundary* for `workflow_path`, inline
  `base_dir`, and sub-workflow steps. Any change to the first has to preserve
  the second.

### What is already right

The mechanism is mostly there; the defaults and the discovery rules are what
bind it to the repo.

- `--workflow-dir` / `--output-dir` / `--prompt-dir` already exist as explicit
  parameters. Nothing in the engine *requires* the repo.
- `builtin:` is already the precedent for "packaged, read-only, reachable by a
  scheme rather than a path".
- `prompt:name` is already a library reference rooted at a library rather than
  at the workflow file — the exact shape assets need, including its validator
  (`validate_prompt_reference`) and its no-double-resolution rule.
- `DIFFUSERS_HELPER_ROOT` already establishes a user-data root outside the
  checkout.
- Path validation is centralized in `dw/security.py`, so a new root is one
  containment base, not a new class of check.

## Proposal

### 1. Name the container: a workspace

One directory holding everything a user generates or curates:

```
<workspace>/
  workflows/     # mine, writable
  prompts/       # the existing prompt library
  assets/        # input media, incl. uploads/
  outputs/       # runs
```

A single resolver (`dw/workspace.py`) replaces three ad-hoc rules:

1. `--workspace` flag
2. `DW_WORKSPACE`
3. `workspace` in `~/.diffusers_helper/settings.json`
4. the working directory, *if it looks like a workspace* — contains any of
   `workflows/`, `prompts/`, `outputs/`
5. `~/diffusers-workflow/`, created on demand

Rule 4 is what keeps this backward compatible: a repo checkout looks like a
workspace, so running from the repo root behaves exactly as it does now. Only
a bare working directory falls through to the home workspace. The existing
per-slot flags stay and continue to override individual roots, so no current
invocation changes meaning. `get_prompt_dir`'s walk-up becomes a deprecated
fallback rather than the primary rule.

### 2. A workflow search path, writes to the front

Rather than one `workflow_dir`, an ordered list:

```
workspace/workflows/   (writable)
<repo>/workflows/      (read-only, if present)
dw/workflows/          (read-only, packaged — today's builtin:)
```

Reads resolve front-to-back; **writes only ever go to the first entry**. That
alone fixes the MCP-writes-into-git problem while keeping the example corpus
first-class and browsable. `GET /api/workflows` gains `origin` and `writable`
per entry; the UI and MCP grey out save/delete on non-writable ones, and
"save as" from an example into the workspace becomes the natural gesture.
`examples:Foo.json` joins `builtin:` as an explicit scheme.

Confinement generalizes cleanly: the boundary becomes *the workspace root plus
the read-only roots* instead of a single `workflow_dir`. Server-side runs stay
confined; CLI runs of an arbitrary file stay unconfined, as today
(`workflow_from_file`'s docstring already states that a locally-run file is not
a trust boundary).

### 3. Run directories, and output layout from workflow identity

Replace the `workflow_output_subfolder` path-segment hack with the workflow's
own identity — its name, or its path relative to the search-path root it came
from — and give each run its own directory:

```
outputs/<workflow-identity>/<run-id>/
  <files...>
  manifest.json      # spec, arguments, seeds, resolved model ids, saved files
```

This is the change that makes multi-stage work tractable: intermediates, final
outputs and provenance are colocated, prunable as a unit, and addressable.
`manifest.json` also means a CLI run leaves a record, and `jobs.sqlite` demotes
from sole record to index — history becomes rebuildable from disk.

Costs to plan for: `dw/step_cache.py` compares `entry["output_dir"]`; the
gallery scans `output_dir` recursively and would want to group by run; the UI
builds `/outputs/<name>` URLs from paths relative to `output_dir`. A flat mode
should stay available for CLI users who prefer today's layout.

### 4. First-class assets, and references that close the loop

Two new reference prefixes, built on the `prompt:` machinery rather than beside
it:

- `asset:name` / `asset:folder/name` — rooted at `workspace/assets/`, not at
  the workflow file. A workflow stops needing to live next to its inputs, which
  is what currently drags media into the source tree.
- `output:<run-id>/<file>` (or a `latest` selector per workflow) — reference a
  previous run's product directly. This removes the manual copy-back that the
  GYRE and marmot workflows document.

Both resolve in `realize_args` where `prompt:` does, reuse
`validate_prompt_reference`-style validation, and inherit the existing rule
that a resolved value may not itself begin with a reference prefix.

`POST /api/uploads` moves to `workspace/assets/uploads/`. Inputs and outputs
stop sharing a directory, and — because the destination is no longer "the
output dir" — an MCP `upload_asset` tool becomes reasonable to add.

### 5. Decoupling MCP from the server

Two levels, worth doing in order:

**Level 1 — workspace-aware client.** `get_server_info` reports the workspace,
its roots and the writable flags. `save_workflow` targets the workspace, never
an example root. Add `upload_asset` so an agent can supply input bytes. MCP
stays a pure HTTP client of a running `dw.serve`; the boundary
`dw_mcp/CLAUDE.md` describes (no `dw.*` imports, no torch) is untouched.

**Level 2 — client-side workspace.** `dw-mcp --workspace <dir>` keeps
workflows, prompts and assets *on the agent's machine*. Authoring, schema
validation and library management are local; only runs go to the remote
`dw.serve`, submitted inline (`POST /api/jobs` with `workflow` + `base_dir`,
already supported) with assets pushed content-addressed — hash-named, skipped
if the server already has them. This is the literal answer to "not everything
should be on the server": the server becomes a GPU, and the workspace is the
user's.

Level 2 has one real snag worth deciding early: inline submission's `base_dir`
is confined to the server's workflow root, so a locally-authored workflow's
asset references have to be rewritten to server-side asset ids at submit time.
That is tractable but it is the design work, not a detail.

Trust interacts here too: `--trust-workflows` is process-wide today. Per-root
trust ("my workspace is trusted, the shared example corpus and anything an
MCP client submits is not") is a better-shaped knob once roots are named
things, and it is the knob that lets a server usefully run untrusted while an
owner's own workflows still use `pre_load_modules`.

### 6. Multiple workspaces

`.gitignore` already carries `local_projects/`, so this is happening
informally. `dw workspace new|list|use`, `--workspace` on every entry point, a
switcher in the UI later. One active workspace per process — no ambiguity about
which root a bare name resolved against.

## Staging

Each stage is independently shippable and the first is behavior-neutral.

1. **`dw/workspace.py` + `--workspace`/`DW_WORKSPACE`.** One resolver, all
   entry points consume it, defaults unchanged in a repo checkout. Nothing
   moves yet.
2. **Assets root and `asset:` references.** Move uploads out of the output
   directory. Workflows can stop hoarding sibling `assets/` folders.
3. **Run directories and on-disk `manifest.json`.** Output layout derives from
   workflow identity instead of file position. Touches step cache, gallery
   grouping and the UI's output URLs — the biggest single stage.
4. **Workflow search path, writable-first, `examples:`.** `origin`/`writable`
   through the API into the UI and MCP. This is the stage that stops agents
   writing into git.
5. **`output:` references.** Closes the multi-stage loop.
6. **MCP level 1** (done), **then level 2** - scoped separately in
   [server-workspaces.md](server-workspaces.md), which reframes it: named
   workspaces on the server, with a client workspace mirroring into one,
   rather than a client-side system of record submitting inline. Level 1
   had already removed the `base_dir` confinement problem this document
   called level 2's real design work.

## Open questions

- Does the repo's top-level `workflows/` stay as an example root on the search
  path, or migrate wholesale into the package next to `dw/workflows/` so a
  pip install has the same corpus a checkout does? The second is cleaner and is
  a bigger move.
- Run id: job id for server runs, but CLI runs have none. Timestamp plus a
  short hash of the resolved spec would serve both and dedupes reruns.
- Should `outputs/` be inside the workspace at all, or a peer with its own
  setting? Video work fills disks, and the outputs root is the one people most
  plausibly want on a different volume.
