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

First match wins:

1. `--workspace <dir>` on `dw.run`, `dw.serve` (`config set workspace=` in the REPL)
2. the `DW_WORKSPACE` environment variable
3. `"workspace"` in `~/.diffusers_helper/settings.json`
4. the working directory, when it holds any of `workflows/`, `prompts/` or `outputs/`
5. `~/diffusers-workspace`

Rule 4 is why nothing changes when you work from a checkout: the repository
root holds all three, so it resolves to itself and every default lands exactly
where it always has. Only a working directory with none of those folders falls
through to the home workspace.

Nothing is created just by resolving. A command that is about to write creates
what it needs — `dw.run` creates its output directory, `dw.serve` creates the
workspace's `workflows/` so the UI has somewhere to save.

## Overriding one folder

The existing per-directory flags still work and each overrides exactly one
folder of the workspace:

```bash
python -m dw.run workflows/sd15.json -o /mnt/big-disk/renders
python -m dw.serve --workspace ~/studio --output-dir /mnt/big-disk/renders
python -m dw.run some.json --prompt-dir ~/shared-prompts
```

`--output-dir` is the one people reach for most: video work fills disks, and
the outputs folder is the one worth putting on another volume.

## Working in a workspace

```bash
mkdir -p ~/studio/{workflows,prompts,assets,outputs}
export DW_WORKSPACE=~/studio

# or, standing, in ~/.diffusers_helper/settings.json
#   { "workspace": "/home/you/studio" }

python -m dw.serve                       # serves ~/studio
python -m dw.run ~/studio/workflows/x.json
```

An example from a checkout still runs by path, and writes into the workspace's
outputs:

```bash
DW_WORKSPACE=~/studio python -m dw.run ~/src/diffusers-workflow/workflows/sd15.json
```

The server reports what it resolved at `GET /api/server`, under
`directories.workspace` alongside the three folder paths.

## The prompt library

`prompt:` references resolve to the workspace's `prompts/` when a workspace was
named explicitly (rules 1–3 above, or `DW_PROMPT_DIR`, which still wins over
everything). A workspace that was merely inferred from the working directory
does not preempt the older discovery — `./prompts`, then the nearest `prompts/`
above the workflow file — so a repository workflow keeps reaching the library
it lives beside. See [Prompt References](WORKFLOW_GUIDE.md#prompt-references).

## Assets

`assets/` is the input-media library. A workflow argument written as
`asset:name.ext` (or `asset:folder/name.ext`) resolves to that file's path,
rooted at the library rather than at the workflow file — so a workflow and the
media it reads no longer have to sit in the same folder. `--asset-dir` and
`DW_ASSET_DIR` override the folder, and browser uploads land in
`assets/uploads/`, coming back as `asset:uploads/<name>` (and served for
preview under `/inputs/`, since the SPA's own bundles own `/assets/`). See
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

`GET /api/workflows` reports the path as `sources` and tags every entry with
its `origin` and `writable`, which is how the UI knows to hide delete and how
an MCP client can tell what it may change.

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
```

The folder is the workflow's identity — its path under a `workflows/` tree
when it has one, its file name otherwise, its `id` for an inline definition.
The run id is a timestamp plus a short digest of what actually ran, so two
runs of the same workflow sort by time and a rerun of an edited workflow is
visibly different; a second run of the same spec in the same second takes a
counter rather than sharing a directory.

`manifest.json` records the run beside what it made — status, seed, arguments,
device, dw version, and each step's files, named relative to the directory so
it keeps describing itself if you move or copy it. It is written even when a
run fails part way, since the files it did write are on disk either way. A
sub-workflow is part of its parent's run: it writes into the same directory and
rolls up into the same manifest.

An unchanged rerun still reuses the step cache: it writes no new files and its
manifest reports the earlier run's, marked `"reused": true`.

A later workflow names what an earlier run made with an `output:` reference —
`output:ltx2/Gyre/latest/Gyre-still.0-0.0.png` — so a multi-stage pipeline no
longer needs files copied back by hand. See
[Output References](WORKFLOW_GUIDE.md#output-references).

To keep the previous layout — everything at the output root, with only a
`workflows/`-mirroring subfolder — use `--output-layout flat`, `DW_OUTPUT_LAYOUT=flat`,
or `"output_layout": "flat"` in settings. Scripts that glob the output directory
are the reason to.

## Where this is going

Workspaces are the first stage of the design in
[proposals/workspaces.md](proposals/workspaces.md): a workflow search path with
writes confined to the workspace, run directories with an on-disk manifest,
`asset:` and `output:` references, and an MCP client that can keep its
workspace on its own machine. Only the resolver and its wiring are implemented
today; everything still lives where it did.
