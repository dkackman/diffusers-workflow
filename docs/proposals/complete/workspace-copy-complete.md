# Copy runs and assets between workspaces: `copy_to_workspace` (#853)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-13). Plan v2 was approved by Don on 2026-10-13. Stage A, #859, and
stage B, #860, shipped the same day, at `develop` @ `b8a26e26` and
`d084ea6a`, and were verified on mini-ai (mps).

## The idea

A field report: an agent needed to move a project's work (the Soviet-anthem
assets and runs) from the `default` workspace into a workspace of its own.
No tool did it (`move_job` only reorders the queue), so the agent
re-uploaded every file and deleted the originals by hand. A re-upload loses
provenance: the file becomes an `uploads/` asset with no manifest and no
`output:` reference.

## Verdict and scope (Don, 2026-10-13)

**Build, in two stages.** Don's scope ruling came before the plan:

1. **Copy only, no move.** A move is a copy followed by `delete_output` /
   `delete_asset`, which already require `acknowledged_cost`. So there is no
   new destructive path, and no jobs-DB `workspace`/`run_dir` rows are
   rewritten.
2. **In scope:** asset-library entries and whole runs
   (`<workflow>/<run id>/` with manifests). Saved workflows and prompts are
   out of scope. Prompts are already shared by the whole root.
3. **`output:` references** are workspace-relative and are never
   rewritten. A copied manifest that names a run not in the destination
   gives a warning, not a refusal.
4. **Confinement:** both workspaces are under one root, checked with the
   existing validators. Copying into the shared library stays with
   `keep_output(shared=true)`.

This also settled #854's deferred question about root placement for
uploads: a copied asset keeps its library-relative path exactly, so
`uploads/x` stays `uploads/x`.

## Decisions (Don's review of plan v1, 2026-10-13)

- **Q1. Hard-link or copy?** Hard-link **media only**, falling back to
  `copy2`. JSON sidecars (`manifest.json`, `workflow.json`, `notes.json`,
  `job.json`, any `.json`) are always copied, because `record_kept_shots`
  and `write_realized_workflow` write in place, so a linked sidecar would
  carry an edit across workspaces.
- **Q2.** No `acknowledged_cost` on the copy: it never overwrites.
- **Q3.** A colliding `vN` is renumbered after the destination's highest,
  in run-id order, and the reply maps old to new.
- **Q4.** `SURFACE_BUDGET` was raised by the measured size, with a dated
  note, once per stage.
- **Q5.** Copied runs get no jobs-DB rows. They appear in `list_gallery`
  but not in `list_jobs`, and the reply and `WORKSPACES.md` say so.

## What was built

### Stage A, #859: copy runs

- **Engine:** `dw/workspace_copy.py`.
  - Both workspaces resolve through `named_workspace`. Every path gets
    `validate_path` with that side's folder as the base. A symlink
    anywhere on a source path or inside a run is refused, and a
    `ConfiguredWorkspace` (no root) is refused with 409.
  - All or nothing: every entry is checked before anything is written. A
    run id the destination already holds goes to `already_present` and is
    skipped, so a repeat copy is idempotent. A run whose job is queued or
    running is refused by name (409).
  - Media is hard-linked, falling back to a copy, and `.json` is always
    copied. Warnings: `unresolved_output_refs`, `absolute_paths`, and the
    no-jobs note. Afterwards it calls `forget_workspace_usage()`.
- **Run versions:** after the architecture review, numbering moved to
  `dw/runs.py: copy_runs_in(source_identity_dir, identity_dir, run_ids,
  place)`. It runs under `run_lock_path`, next to `record_run_versions`.
- **Run-entry shape:** `security.validate_output_prefix`, which shares one
  `_OUTPUT_SEGMENT` with `OUTPUT_REFERENCE_PATTERN`. An entry with an empty
  segment (`wf/`, `wf//<run>`) is refused, not normalized.
- **Route:** `POST /api/workspaces/{destination:path}/copy` in
  `dw/server/routes/library.py`, with body `{source, runs, assets}`.
- **MCP:** `copy_to_workspace(destination, runs, assets, source=None)`, a
  thin client of the route, registered `WRITES`.
- **Outside the plan:** `set_note` was registered as an MCP tool. The
  notes feature (`a6483383`) had written it, documented it and named it in
  the server instructions, but never registered it, and the acceptance
  case for note independence needed it. Tool count: 62 → 64.
- **Outside the plan:** `api_path` now encodes `.`/`..` segments
  (`%2E`), so httpx no longer drops them. The fix is in the shared
  `path_segment`, so it covers every MCP tool that builds a path.

### Stage B, #860: copy assets

- `copy_between` replaced `copy_runs` as the engine entry point. `runs` and
  `assets` are both optional, but at least one is required (400 if
  neither is given).
- An asset entry is a library-relative file or a folder (`proj/` or
  `proj`), and an `asset:` prefix is accepted. `copied.assets` lists each
  file copied, expanded from a folder.
- Each file's entries in its folder's `notes.json` and `manifest.json`
  (kept shots) are merged into fresh destination sidecars.
  `provenance.workspace` keeps naming the source. A sidecar can't be named
  directly (400), and a folder copy doesn't copy sidecars as files.
- Identical bytes at the path go into `already_present`. Different bytes
  are a conflict, and they refuse the whole request, runs included (409,
  `detail.conflicts`).
- **Outside the plan:** the plan's risk said "copy, don't link, any path
  that edits media in place". The build found one such writer:
  `upload_media` wrote over an existing name with `open(path, "wb")`. It
  now writes a temp file and `os.replace`s it, so linking stays safe. This
  also fixes an existing bug, where an upload over a `keep_output` link
  changed the linked original.

### Tests and docs

- `tests/test_workspace_copy.py`, on the real engine and route with no
  mocks of the copy. It covers sidecar independence, the hard-link check
  (same inode when `os.link` succeeds), traversal, symlinks, conflicts and
  renumbering.
- `SURFACE_BUDGET` was raised once per stage, plus once for `set_note`. It
  ended at 14_700.
- Docs: `docs/WORKSPACES.md` ("Copying between workspaces"),
  `docs/MCP.md`, `docs/SERVER.md`, `docs/ARCHITECTURE.md` (the "Workspace
  copy" row), and the regenerated OpenAPI documents.

## Verified

On mini-ai (mps), over MCP only. Stage A: C-F395 to C-F402 and SE-F047.
Stage B: C-F403 to C-F405 and SE-F048. Nothing here needs CUDA: it is all
filesystem work.

## Bounces per stage

- **Stage A, #859: 2.**
  - **Architecture review (1):** three second-owner findings: the
    reference prefix was spelled locally, run-version numbering lived
    outside `dw/runs.py`, and the output name shape was restated. All three
    were fixed by calling the owners.
  - **Tester (1):**
    - C-F398 couldn't run, because no MCP tool could change a note
      (`set_note` was unregistered).
    - SE-F047: `wf/` was normalized instead of refused, and bad
      destination names missed the route (a bare 404, or a 405 when httpx
      dropped `.`/`..`) instead of getting "No such workspace".
  - The tester also filed three case-text problems as harnest#104; they
    weren't counted.
- **Stage B, #860: 0.** The tester didn't count one SE-F048 arm:
  `asset:proj/x.wav` was accepted, which the plan allows. It was filed as a
  case amendment, harnest#106.

Cost per stage isn't recorded: the stage comments name no `usage:`
figures. The plan estimated $8–12 for both.

## Deferred

- **A UI for copying.** It wasn't wanted for the field report. Build it if
  copying from the web UI is ever asked for.
- **Jobs-DB rows for copied runs** (Q5). `get_job` and `job_provenance`
  are empty for a copied run. Revisit if an agent needs job provenance on a
  copy.
- **Saved workflows and prompts.** Out of scope by Don's ruling.
- **The `absolute_paths` warning on inline runs.** Every inline run warns
  `workflows/__inline__.json`. It's accurate, but the tester called it
  noise. Quiet it if it misleads an agent.
