# Design: making media-less run directories visible

Status: **proposal**. Written for #170 (filed by the regression agent, model
`opus` via provider `anthropic`), triaged and drafted by the implementer
agent, model `sonnet` via provider `anthropic`, 2026-09-15.

## The ask, as filed

`list_workspaces().usage` reports files/bytes a consumer cannot otherwise
enumerate: `list_gallery`, `list_assets` and `list_workflows` all report the
workspace empty while `usage` counts 44 files / 36,276 bytes in it. #170's own
investigation (repeated here, verified against the code) traces this to run
directories that hold a `manifest.json` and `workflow.json` but no media - a
run whose output was deleted before #134 shipped the "delete by run directory
name" form of `delete_output`, or a run that failed before writing anything
and was never swept. There is no MCP call that names one of these directories,
so there is also none that can delete it. #170 is explicit that #134 itself is
not at fault: per-job residue measured at zero across four fresh jobs in the
same session, including a failed one cleaned up by its `<identity>/<run id>`
name.

## The cheap half, checked first

The triage comment on #170 asked whether `workspace_usage` counting
`manifest.json`/`workflow.json` at all is intended, before treating the gap as
a design problem. It is. `workspace_usage` (`dw/workspace.py`) walks every
file under a workspace's own folders with `_tree_usage` - no extension filter,
no media check - because its stated purpose is "how much disk a workspace
occupies," the number a client offers alongside "delete this workspace." The
gallery, by contrast, deliberately filters to `MEDIA_KINDS`
(`_iter_gallery_files`, `dw/server/app.py`) because it exists to show a
consumer their generated files, not their bookkeeping. Two different
questions, two different counts, both working as designed. `usage` is not the
bug; the bug is that nothing else can account for the gap it reveals.

## Why this needs a decision and not an edit

Every existing way to inspect a workspace's outputs - `list_gallery`,
`GET /api/gallery`, the job page's manifest - is keyed to *media files*: a
`manifest.json`/`workflow.json`-only directory has none, by construction
(`_iter_gallery_files` only yields names whose extension is in
`MEDIA_KINDS`). Making an orphan visible means adding a concept a consumer
has to learn - "a run directory can exist with nothing to show you" - on a
surface that has never needed it before. That is squarely the kind of change
guardrail 7 asks to be proposed rather than edited in: new MCP surface,
however small.

There is also a real design question underneath it, not just a listing
gap: **should an orphan be swept automatically, surfaced for a human/agent
decision, or both?** A media-less run directory is not always junk - a run
that failed mid-step and left partial intermediate files with no
`content_type` result would look identical to one that failed before writing
anything, and a workflow that legitimately writes no media (a pure `utility`
shape - e.g. a text-analysis step whose result is JSON, not a saved file) is
indistinguishable from an orphan by this test alone. Deciding that is worth
getting right once rather than shipping a listing tool whose shape has to
change later.

## Options

### A. `list_gallery` gains an "orphans" mode

Add a boolean, e.g. `list_gallery(only_orphans=True)`, that walks the output
tree the same way `_iter_gallery_files` does but inverts the filter: instead
of yielding media files, it yields run directories (identity + run id, via
`split_run_path`/`is_run_id`) that hold `manifest.json` or `workflow.json`
and **no** file whose extension is in `MEDIA_KINDS` anywhere under them. Each
entry would carry `name` (the `<identity>/<run id>` `delete_output` already
accepts), `mtime` (from the manifest, for sorting/staleness), and `reason`
(`"no-media"` vs `"failed"`, read off the manifest's own status field if it
has one - worth checking before committing to the field name). This reuses
the existing filter/pagination shape consumers already know, and pairs
directly with the `delete_output` form #134 added - no new tool, one new
parameter, one documented reading.

Cost: `list_gallery`'s docstring and the tool description both grow a
second meaning ("files, or absence of files"), and the REST mirror
(`GET /api/gallery?only_orphans=true`) needs the same walk done a second way
(gallery's normal walk is `os.walk` + extension filter; this one is
`os.walk` + *directory* filter) - not free, but contained to one function
`_iter_orphan_runs` beside `_iter_gallery_files`.

### B. A dedicated tool, e.g. `list_orphaned_runs`

Same underlying walk, its own MCP tool and REST route
(`GET /api/gallery/orphans` or `GET /api/outputs/orphans`) rather than a mode
of an existing one. Clearer name, no double meaning on `list_gallery`, but
one more tool for a consumer to discover and one more entry in every list of
"the gallery-adjacent tools" (`get_guide`, the plugin skills, `dw/CLAUDE.md`
if it ever enumerates them). Given how rare this call is expected to be -
once per regression sweep, essentially never for a normal workflow session -
the discoverability cost of a whole tool seems disproportionate next to
option A's one parameter.

### C. Sweep automatically, no listing at all

Fold orphan cleanup into something that already runs unattended: a
`forget_workspace_usage`-adjacent pass at server startup, or opportunistically
whenever a job finishes in a workspace (walk that workspace's outputs,
`shutil.rmtree` any run directory with no media, same "climb and rmdir while
empty" logic `delete_output`/`_sweep_run_directory` already have). This closes
the backlog with no new MCP surface at all, but removes the option to inspect
before deleting - the same objection the `wontfix` reasoning below would raise
if a consumer's own retained-but-empty run (a deliberate `utility`-shape
workflow, see above) got swept as if it were junk. Also weakens the
regression sweep's own assertion further: "the workspace is empty" would
become true by a mechanism the sweep did not invoke and cannot verify was
*this run's* doing.

## Recommendation

**A**, with the "no-media" test scoped to a run directory holding *only*
`manifest.json`/`workflow.json` (or either alone) and nothing else - not
"no `MEDIA_KINDS` file," which would misclassify a legitimate no-media
utility run as an orphan. That distinction needs one query against the
manifest: does this step list contain any `result.content_type`.
Concretely:

- `list_gallery(only_orphans=False)` (default unchanged), and
  `list_gallery(only_orphans=True)` returns run directories matching the
  refined test above, each `{name, mtime}` (`name` is exactly what
  `delete_output` already accepts - the pairing is the point). No new
  `delete_output` shape is needed; #134 already covers removal once a
  consumer has the name.
- `GET /api/gallery?only_orphans=true` mirrors it.
- No automatic sweep (rejects C) - orphan cleanup stays an explicit,
  attributable action, consistent with how `delete_output` already works for
  everything else in a workspace.
- The regression agent's own sweep step gains a call it did not have before:
  `list_gallery(only_orphans=True)` then `delete_output` per name, closing
  the loop #170 opened.

## Open questions for Don

1. Does the "no `content_type` anywhere in the manifest" test correctly
   distinguish an orphan from a legitimate no-media run, or is there a
   workflow shape it still misclassifies?
2. Is `only_orphans` the right name, or should this be a `subfolder`-style
   third value (`list_gallery(mode="orphans")`) to leave room for another
   listing mode later without stacking booleans?
3. Should the backlog already on `lem` (44/48/72/12/4 files across the
   workspaces #170 measured) be cleared by hand once this ships, or left for
   whoever next runs the regression sweep against that workspace to clear via
   the new call - i.e. is a manual `lem` cleanup in scope for this ticket or
   a separate housekeeping pass?
