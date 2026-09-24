# Design: hierarchical workspace names

Status: **declined 2026-09-23** (issue #375, closed not planned). Written for
issue #167 and researched 2026-09-16 by claude-sonnet-5 via anthropic.

Why it was declined: plan v2 on #375 weighed value against build cost,
inertia and reversibility. `lem` held 20 workspaces, 14 of them unrelated
projects, and the `QA-EP*` clutter behind the ask was down to one. Against
that, the change loosens `WORKSPACE_NAME_PATTERN`, a deliberate security
boundary, and it cannot be backed out without a breaking change once
grouped names reach job records. Don: "don't build".

The design below still has two errors, corrected in #375's plan. Read that
plan before reviving this:
- `DELETE /api/workspaces/{name}` needs `{name:path}`.
- Reserved names must be checked per segment: whole-string, `outputs/x`
  passes.

Revisit if a workflow starts making per-episode workspaces again, or the
list reaches about 40 with real prefix families. If only the look is
wanted, a display-only prefix grouping in the UI sidebar gets most of it
with no engine change.

## The ask, as filed

Don's own QA workflow makes one workspace per test episode -
`QA`, `QA-EP1`, `QA-EP2` - and they pile up flat at the workspace root with
no relationship visible between them. The ask is for a workspace name to
carry structure - something like `QA/EP1` - so related workspaces group
under a common prefix the way subfolders already group files inside one
workspace's outputs (`docs/proposals/output-folders.md`) or entries inside
the workflow and prompt libraries.

There is no reproduction here - nothing is broken. This is an ergonomics
request: the catalog and every workflow in it work today with workspaces
exactly as flat as they are.

## Why this needs a design and not an edit

A workspace name is load-bearing in three places that all assume it is
exactly one path segment:

- **The security boundary.** `WORKSPACE_NAME_PATTERN = r"^[\w][\w.-]*\Z"`
  (`dw/security.py:694`) refuses any separator on purpose - the comment
  above it says a workspace is "one folder under the workspace root", so
  `..`, a hidden name, and anything with `/` in it are excluded before the
  name is ever joined onto the root. Allowing a separator is not a regex
  tweak; it is loosening the exact rule that keeps `named_workspace` safe to
  call with untrusted input.
- **The listing.** `workspace_names` (`dw/workspace.py:439`) is a single
  non-recursive `os.listdir(workspace.root)`, keeping only entries that
  `_holds_a_workspace` (all three of `NAMED_SUBDIRS` present). A workspace
  called `QA/EP1` would have to be discovered by walking - and walking
  raises the question of how deep, and whether an intermediate node like
  `QA` is itself a listable, usable workspace or just a grouping label.
- **Deletion's safety check.** `delete_workspace` refuses to `rmtree` a
  directory holding anything besides `NAMED_SUBDIRS` + `exports`
  (`_foreign_entries`) - which is exactly what a directory holding *child*
  workspaces would look like: `QA/` would contain `EP1/`, not
  `workflows/`/`assets`/`outputs`, and the current check would call that
  foreign and refuse to delete it (correctly, today, since nothing makes
  `QA/` itself a workspace) or would need to learn a new legitimate shape.

This is the same shape of change #69 (output/job folders) was held for:
a position-addressed name (there: `<workflow identity>/<run id>/<file>`;
here: `<root>/<name>`) breaks every reader that assumes a fixed number of
segments once a segment can be inserted. Four readers assume one segment
today: `workspace_names`, `named_workspace`, the two MCP/REST workspace
routes, and the UI. That is worth deciding on paper before code moves.

### What a child inherits, and from whom

`named_workspace` already answers one version of this question for a
*flat* name: a named workspace points its `prompts_root` and
`common_root` at the true root's `prompts/` and `common/`, not at itself -
a stored prompt is shared by reference, so `prompt:scenic` must resolve to
the same text regardless of which workspace asked (`dw/workspace.py:88-90`).
Nesting reopens that question one level down: does `QA/EP1` share `QA`'s
own prompts and common assets, or only the true root's? The ticket doesn't
ask for scoped sharing, and inventing it now would be solving a problem
nobody has filed - but the proposal has to say so explicitly, because
silence here is where scope creeps in during implementation.

### True nesting vs. a shallow naming convention

Two designs satisfy "workspaces group under a prefix":

1. **True nesting.** `QA` becomes a real, listable, usable workspace in its
   own right, and `EP1` is a child of it with its own three folders.
   `workspace_names` returns a tree or a set of `/`-joined paths;
   `delete_workspace("QA")` has to decide whether it cascades to `EP1` or
   refuses while children exist.
2. **Shallow sub-workspace convention.** A workspace name may contain
   exactly one `/`, so `QA/EP1` is one flat workspace (one set of three
   folders, one entry in `workspace_names`) whose *name* happens to have a
   `/` in it and whose folder on disk is nested one level
   (`<root>/QA/EP1/{workflows,assets,outputs}`). `QA` alone is never a
   workspace and is never listed, created, or deleted - it is purely a
   grouping label baked into sibling names, the same precedent the
   one-level `subfolder` field settled on for run outputs rather than
   arbitrary output-tree nesting.

**Recommendation: option 2.** It is the smaller change - one more
character class in `WORKSPACE_NAME_PATTERN`, one join in `named_workspace`
- and it answers "what does `QA` inherit from its parent" by making the
question not arise: there is no parent workspace, only a naming
convention. It also matches what the issue actually asked for: grouping in
a listing/UI, not a new addressable resource. True nesting (option 1) is a
bigger, genuinely different feature - listable intermediate nodes, cascade-
or-refuse delete semantics, inheritance rules for prompts/common assets at
each level - and nothing in the ask requires it. If Don wants real nesting
later, that is its own proposal built on top of this one, not a reason to
build it now.

## The design (option 2)

- **Name grammar.** `WORKSPACE_NAME_PATTERN` becomes
  `r"^[\w][\w.-]*(/[\w][\w.-]*)*\Z"` - one or more segments, each obeying
  today's per-segment rule, joined by a single `/`. This is the exact
  pattern `output-folders.md` already uses for a `subfolder` segment
  (`OUTPUT_REFERENCE_PATTERN`'s per-segment rule), so it is a precedent
  already reviewed and shipped, not a new security surface. `..` stays
  refused because no segment may start with `.` doubled as a whole segment
  under `[\w][\w.-]*`; a leading, trailing, or doubled `/` is refused by
  requiring each segment to start with `[\w]`. A depth cap of 2 (one `/`,
  matching the ask's own example) rather than unbounded depth - unbounded
  nesting is exactly the option-1 territory this proposal is declining.
- **Reserved names** apply per full name, not per segment - `QA/outputs` is
  refused the same way `outputs` is today, since a segment named after one
  of a workspace's own subfolders is exactly the collision
  `RESERVED_WORKSPACE_NAMES` exists to prevent.
- **On disk**, `named_workspace(workspace, "QA/EP1")` joins the segments
  the way it joins one today: `os.path.join(workspace.root, "QA", "EP1")`.
  `os.path.join` with a validated, separator-free-per-segment name is safe
  the same way it is safe for one segment - `validate_workspace_name`
  is the barrier, exactly as `dw/security.py`'s CodeQL modeling already
  expects for a workspace name.
- **`workspace_names`.** Still one call, but `os.walk` bounded to depth 2
  under `workspace.root` instead of `os.listdir`: a workspace is any
  directory at depth 1 that `_holds_a_workspace`, or any directory at depth
  2 under a depth-1 grouping directory that does. A depth-1 directory that
  itself holds `NAMED_SUBDIRS` is a workspace named by its own segment (no
  change from today); a depth-1 directory that does *not* hold
  `NAMED_SUBDIRS` but has depth-2 children that do is purely a grouping
  prefix and contributes no entry of its own - `QA/` never appears in the
  listing, only `QA/EP1` and `QA/EP2` do. This is what keeps `QA` from
  becoming an addressable resource by accident.
- **Delete.** `delete_workspace(workspace, "QA/EP1")` behaves exactly as it
  does today, `rmtree`-ing `<root>/QA/EP1` after the same foreign-entries
  check. If that leaves `<root>/QA/` empty, it is removed too (nothing
  should keep an empty grouping directory around); if `QA` still holds
  other children, it is left alone. There is no `delete_workspace(root,
  "QA")` - `QA` alone was never a workspace name in `workspace_names`, so
  the existing `if name not in workspace_names(workspace): raise
  FileNotFoundError` already refuses it with no new code.
- **MCP/REST surface: no shape change.** `create_workspace`,
  `delete_workspace`, `list_workspaces` (REST: `POST/DELETE/GET
  /api/workspaces`, MCP: `create_workspace`/`delete_workspace`/
  `list_workspaces`) already take a bare `name: str` - a name with a `/` in
  it is still a string, so no request/response schema changes, no
  `breaking-change` label. The only behavior change is that `name` may now
  validate where it previously raised 400.
- **UI.** `ui/src/lib/workspace.svelte.ts` and the workspace switcher on
  `ServerPage.svelte` read `workspace_names`/`list_workspaces` and today
  render one flat list; grouping `QA/EP1`, `QA/EP2` under a `QA` heading in
  that list (split the name on `/`, group by the prefix) is a presentation
  change with no new API. Creating a child (`ServerPage.svelte`'s add-
  workspace flow) needs its text field to accept a `/`، which is a one-line
  relaxation of whatever client-side check mirrors
  `WORKSPACE_NAME_PATTERN`, if any exists there today - worth confirming
  during implementation rather than assuming.

## What this proposal is not deciding

- **True nesting** (option 1 above) - an addressable, listable, deletable
  `QA` workspace with its own folders and children. Out of scope; a
  separate proposal if ever wanted.
- **Per-level inheritance** of prompts/common assets - not needed under
  option 2, since there is no parent workspace to inherit from.
- **A depth greater than 2.** If a future ask wants `QA/EP1/take3`, that is
  a new proposal with its own reason, not an extension assumed here.

## Rollout

No migration: every existing workspace name is a valid one-segment name
under the new pattern, `workspace_names`'s depth-2 walk returns exactly what
`os.listdir` returned before for a root with no grouped names, and no
existing `output:`/`asset:`/`prompt:` reference or job record encodes a
workspace name, so nothing stored needs rewriting.
