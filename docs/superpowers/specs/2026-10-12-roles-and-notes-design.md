# Roles and notes for runs, outputs, assets and workspaces

Date: 2026-10-12. Status: approved in conversation, awaiting spec review.

## Goal

The gallery is hard to navigate because nothing tells the deliverable from
the scratch files. The `final` / `intermediate` subfolder convention works
for the catalog templates (121 and 37 uses across all 65), but the engine
treats no folder name specially. As a result, a hand-built workflow lands
its files flat, or in whatever folder the agent picked. And nothing records
*why* a file exists: an agent asked to "experiment with x, y, z and keep a
note of the settings with each output" has nowhere to put the note.

This adds two things:

1. **Roles.** Every file in a run is `final` or `intermediate`, and the
   files a run read are its `input`s. A role is decided by default, with
   nothing for the agent to do. In the UI roles are the main way to
   navigate, and folders become a detail.
2. **Notes.** Optional plain text on a run, on a file in a run, on an asset
   and on a workspace. It can be set at submission or afterwards, by the UI
   or by an agent.

Free-form tags are deliberately out of scope. They will be revisited if
roles turn out not to be enough categorization.

## Build order

1. Roles: engine recording, server inference for older runs, API and MCP
   fields, gallery and job UI.
2. Notes: the four stores, submission and edit paths, API and MCP, UI.

Each stage ships on its own and passes the full suite.

## 1. Roles

### Values

- `final`: a file the run produced as a deliverable.
- `intermediate`: a file the run produced so that a later step could
  consume it.
- `input`: an `asset:` or `output:` file the run's arguments resolved to.
  An input is never copied. It is a reference recorded in the run's
  manifest.

### Deciding a step's role

The role is decided per step, and every file the step wrote gets it. The
first rule that holds wins:

1. **Explicit.** The step's `result.role` is `"final"` or
   `"intermediate"`. This is a new optional property in
   `dw/workflow_schema.json` under the result block, next to `save` and
   `subfolder`, and it may be a `variable:` reference like `subfolder`.
2. **Convention.** The step's resolved `result.subfolder` is `final` or
   `intermediate`, or its first path segment is one of them
   (`final/act-1`).
3. **Inferred.**
   - A step whose files a later step *that writes media* consumes is
     `intermediate`. Media means a gallery media kind: image, video or
     audio.
   - Every other step is `final`.
   - Consuming means a `previous_result:<step>` reference anywhere in the
     later step's arguments, or in its task or pipeline inputs.
   - A consumer that writes only text or JSON does not demote what it
     reads. This covers transcripts, `check_script`, `plan_cuts` and
     assessment probes. It is why `check_script` on a finished video
     leaves the video `final`.

A flat-layout file with no manifest has no role. It is listed with
`role: null` and counted as final by the UI filter, so nothing disappears
from view.

### Sub-workflows and `for_each`

- A composing step's role applies to the files the composed child wrote
  into the parent's run.
- A `for_each` step has one role for all its expansions.
- The child's own manifest, if it has its own run, keeps its own roles.

### Inputs

As a run opens, the engine records every `asset:` and `output:` reference
the substituted arguments resolved to, as `inputs: [{reference, kind}]` in
`manifest.json`. Duplicates are removed and the order follows first use.

### Where roles are stored

- **New runs.** `write_run_manifest` (`dw/workflow_run.py:209`) adds
  `role` to each `steps[]` entry when it writes the terminal manifest,
  alongside `inputs`.
- **Older runs.** A manifest whose steps carry no `role` is inferred on the
  server from the realized `workflow.json` beside it, using the same
  function. The rules live in one module (`dw/roles.py`) that both the
  engine and the server import, and that module names no model. The result
  is cached per run directory, keyed by the manifest's mtime, next to the
  gallery's existing run metadata lookups.

### API and MCP

- `GalleryFile` (`dw/server/api_models.py:781`) gains `role`.
- `GalleryMetadata` (`:843`) gains `role` and `inputs`.
- `GET /api/gallery` gains an optional `role` query parameter, repeatable,
  taking `final` and `intermediate`. Leaving it out lists every file.
- MCP `list_gallery` gains the same `role` filter, and its compact
  projection gains `role`. MCP's default stays every role, because
  diagnosis needs the intermediates.
- `get_gallery_metadata` returns `role` and `inputs`.
- The OpenAPI document is regenerated (`scripts/dump_openapi.py`, then
  `npm run gen:api`), and `scripts/surface_snapshot.py` is updated.

### UI

**Gallery** (`ui/src/lib/pages/GalleryPage.svelte`):

- **Role chips are the main filter,** in the top bar beside the workflow
  filter: **Final** and **Intermediate**, multi-select. Final is on by
  default. The UI's default differs from MCP's on purpose.
- **Every tile carries its role as a coloured tag,** ahead of the file
  name and visible at thumbnail size. Final is prominent and intermediate
  is muted.
- **The subfolder filter leaves the top bar.** The folder path moves to a
  small "Location" line at the bottom of the detail panel.
- Inside a run, finals sort before intermediates.

**Job results** (`ui/src/lib/pages/JobPage.svelte`):

- Grouped into sections: **Final** first and expanded, then
  **Intermediate**, collapsed when the run has any finals, then **Made
  from**, the input references tagged `input`.

**Detail panel** (`ui/src/lib/gallery/GalleryDetail.svelte`), top to
bottom:

1. Role tag.
2. Note (section 2).
3. A collapsed **Settings** line with the run's recorded arguments and
   seed, which the manifest already holds.
4. **Made from**, the inputs, each one clickable.
5. Location.

**Assets page:** every asset is an input, so assets get no role tag. A kept
file shows a "kept from <workflow> v<n>" line from the provenance
`record_kept_shots` already writes (`dw/runs.py:784`, read by
`kept_provenance` `:872`).

## 2. Notes

### What a note is

- Optional plain text, at most 4,000 characters. Longer is refused with a
  400.
- Rendered as text, never as HTML or markdown.
- An empty string or null clears the note.

### Stores

| On | Stored in | Shape |
|---|---|---|
| Run | `notes.json` in the run directory | `{"run": "..."}` |
| File in a run | the same `notes.json` | `{"files": {"<path relative to the run dir>": "..."}}` |
| Asset | the existing directory sidecar `manifest.json` that `record_kept_shots` writes | a top-level `notes` map keyed by file name |
| Workspace | new `workspace.json` at the workspace root | `{"notes": "..."}` |

Why `notes.json` and not `manifest.json`: the engine rewrites the whole
manifest on every write while a run is going (`write_run_manifest`), so a
note written into it by the server could be lost. A separate file keeps the
engine's manifest single-writer.

Every write is atomic, using a temp file and replace, as `write_manifest`
does (`dw/runs.py:683`). Concurrent edits to the same `notes.json` are
serialized by a per-path lock in the server. The engine writes
`notes.json` only once, when the run opens, before the server can address
the run directory.

### The effective note

A file's effective note is its own entry in `files`, or else the run's
note.

### Bookkeeping

- `notes.json` joins `RUN_SIDECARS` (`dw/server/routes/gallery.py:277`) and
  `RUN_BOOKKEEPING_FILES` (`dw/server/outputs.py:54`). As a result:
  - a run left holding only notes is pruned when its last output is
    deleted;
  - it is still detected as an orphan;
  - `notes.json` is never listed as media;
  - `dw/server/exports.py` copies it into the export zip beside the
    manifest.
- Deleting one output removes that file's entry from `notes.json`.
- `workspace.json` joins the entries `delete_workspace` tolerates
  (`_foreign_entries`, `dw/workspace.py:374`).
- Deleting an asset (`dw/server/routes/assets.py:470`) removes that file's
  entries from its directory sidecar: its notes, provenance and shots. This
  fixes an existing leak, where a later upload under the same name
  inherited stale provenance.

### Setting a note

**At submission.** `JobRequest` (`dw/server/admission.py:383`) and MCP
`run_workflow` gain an optional `note`.

- It is stored in a new `note` column on the sqlite `jobs` table, added by
  the existing `ALTER TABLE ADD COLUMN` migration path
  (`dw/server/job_history.py:61-133`).
- It is passed to the worker in the job spec, and the engine writes it to
  `notes.json` when the run opens.
- `note` joins `RERUN_SPEC_KEYS` (`dw/server/job_record.py:31`), so
  `rerun_job` carries it forward.

**At upload or keep.**

- `POST /api/uploads` and MCP `upload_asset` gain `note`. Upload writes
  the directory sidecar when a note is given.
- `KeepRequest` and MCP `keep_output` gain `note`. Without one, keep copies
  the source file's effective note.

**Afterwards.** One route, `PUT /api/notes`, with body
`{target, note}`. `target` is one of:

- a gallery file name: sets that file's note;
- a run directory, `<identity>/<run_id>`: sets the run note;
- `asset:<name>`;
- `workspace:<name>`.

The target is resolved by the same validated resolution the delete and
metadata routes use (`dw/server/routes/gallery.py:346`,
`dw/server/routes/media.py:69`, `validate_asset_reference`). No path
handling is new, and every write lands on a path that has been through
`validate_output_path` against its root.

The route answers with the target's effective note and `note_source`.
Notes are free metadata, so they need no `acknowledged_cost`.

MCP gains one tool, `set_note(target, note)`, over that route.

### Reading notes

- `GalleryFile` gains `note`, the effective one.
- `GalleryMetadata` gains `note`, `run_note` and `note_source`, which is
  `"file"` or `"run"`.
- `AssetFile` (`api_models.py:864`) and `WorkspaceInfo` (`:627`) gain
  `note`.
- `JobSummary` (`:345`) and `JobDetail` (`:419`) gain `note`.
- In MCP, the compact projections of `list_gallery`, `list_assets`,
  `list_workspaces` (`WORKSPACE_SUMMARY_FIELDS`, `dw_mcp/workspaces.py:21`)
  and `list_jobs` / `get_job` include `note`.
  - In a compact projection the note is clipped to 200 characters with
    `note_truncated: true`.
  - The detail tools (`get_gallery_metadata`, `get_job`) return it in full.
- New response keys are declared with `sometimes(...)` where optional, as
  strict responses require.

### UI

**Workspace overview** (`ui/src/lib/pages/OverviewPage.svelte:128`):

- The note is shown under the heading. It is edited in place with a
  pencil, a textarea and Save/Cancel.
- With no note, the place shows a muted "Add a note".
- `loadWorkspaces` (`ui/src/lib/workspace.svelte.ts:72`) keeps `note`.

**Gallery and asset detail:**

- The note sits under the role tag, edited the same way.
- When the note shown is the run's, it is labelled "run note", and the
  editor offers "edit for this file" or "edit run note".

**Gallery tiles:** a small note icon when the effective note is set,
showing the first line on hover.

**Job page:** the run note sits under the job header, and can be edited.

**One editor component.** `NoteEditor.svelte` is shared by all four
places. It renders the note as text, never with `{@html}`.

### Agent guidance

A short paragraph goes in the MCP server instructions
(`dw_mcp`) and in `docs/WORKFLOW_GUIDE.md`'s *Authoring a workflow from an
agent* section, saying:

- When running a set of variations, pass `note` on each `run_workflow`
  saying what that run tests.
- Use `set_note` to mark a keeper.
- `result.role` is available when the default inference is wrong for a
  step.

`docs/SERVER.md` and `docs/MCP.md` document the new route, fields and tool.
`docs/ARCHITECTURE.md` gains a row for `dw/roles.py`.

## Testing

**Python:**

- Role inference:
  - each of the three rules and their precedence;
  - the text and JSON consumer exception;
  - `for_each`;
  - sub-workflow composing steps;
  - a `variable:` role or subfolder;
  - older-run inference matches what a new run records for the same
    workflow;
  - a flat file gives `null`.
- `inputs` recording for `asset:` and `output:` references, with
  duplicates removed.
- Each note store: write, read, clear, the 4,000-character limit, atomic
  writes, and the effective-note fallback.
- Bookkeeping:
  - prune when only notes remain;
  - orphan detection;
  - the export includes `notes.json`;
  - deleting a single output removes its note entry;
  - deleting an asset clears its sidecar entry;
  - a workspace holding `workspace.json` deletes.
- `PUT /api/notes` security:
  - traversal in each target form is refused;
  - an unknown target gives a 404;
  - a non-writable library gives a 409, as keep and delete answer.
- Submission: `note` reaches the sqlite row and `notes.json`, and survives
  a rerun.
- MCP: `set_note`, the `note` parameters, the compact clipping and
  `note_truncated`.

**UI (Vitest):**

- Role chips, the default Final filter and tile tags.
- Job grouping.
- `NoteEditor`: save, cancel, clear, file versus run, and text-only
  rendering.
- The note in each of the four places.

**Snapshots:** OpenAPI regenerated, and `scripts/surface_snapshot.py`
updated for the MCP tool and parameters.

## Out of scope

- Free-form tags.
- Search over notes.
- Notes on prompts and saved workflows, which already have `description`.
- Moving notes with a cross-workspace move (#853).
