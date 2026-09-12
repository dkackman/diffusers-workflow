# Proposal: folders for a job's outputs

Status: **proposed, not implemented**. Written for MCP feedback ticket T016.

## The ask, as filed

> Jobs and workspace outputs support folders one level deep - the same
> shape, look, and behaviour as the folder support that already exists for
> the workflow and prompt libraries. The intent is separation of
> intermediate outputs from final ones. Folders carry through all three
> surfaces: MCP tools, REST API, and UI. MCP consumers are steered toward
> using them for the intermediate/final distinction.
>
> Actual: everything a job produces sits at one level, so a finished episode
> is indistinguishable from the twenty scratch files that went into it
> without reading names.

The report is accurate. A `dialogue-short` run writes its five shot videos,
its portrait stills, its sliced audio beds and its one assembled episode
into a single run directory, distinguished only by the step name embedded in
each file name. The consumer of that run - an agent deciding what to show
the user, or a person opening the gallery - has to know the workflow to know
which file is the deliverable.

## Why this is a proposal and not an edit

A destination folder is a new field in the workflow schema, which means it
is a concept every consumer has to learn and every surface has to carry. But
the reason it is a proposal rather than an edit is narrower than that: **the
output tree already has a layer with the meaning this wants.**

A run writes to `<output root>/<workflow identity>/<run id>/`, and three
separate things read that shape by position:

- `strip_run_id` (`dw/runs.py`) turns `ltx2/Gyre/20260905-181530-a1b2c3d4/
  still-0.png` into the gallery folder `ltx2/Gyre`, which is how a workflow
  run fifty times is one entry in the folder filter rather than fifty.
- `output:` references are `<workflow identity>/<run id>/<file>`, with
  `latest` allowed in the run-id position (`resolve_output_reference`).
- The manifest, and `workflow.json` beside it, sit at the root of the run
  directory and describe everything in it.

Adding a folder *inside* a run directory puts a segment between the run id
and the file name, and every one of those three readers has to be told which
segment is which. Getting that wrong is not a cosmetic bug: `strip_run_id`
would start reporting `ltx2/Gyre/final` and `ltx2/Gyre/intermediate` as two
unrelated workflows in the gallery filter, and an `output:` reference that
worked yesterday would resolve to nothing. That is the part that needs
deciding before anything is written.

There is also a question the ticket settles one way and the existing code
settles the other, covered under [Open questions](#open-questions): the
workflow and prompt libraries do not actually enforce one level.

## The shape

A step's `result` block names the folder its files are written into:

```json
{
  "name": "assemble_episode",
  "task": { "command": "concat_videos", "arguments": { "...": "..." } },
  "result": {
    "content_type": "video/mp4",
    "folder": "final"
  }
}
```

- `folder` is optional. Without it, files are written where they are written
  today: at the root of the run directory. **Every existing workflow keeps
  working unchanged, and a workflow that never sets `folder` is
  indistinguishable from one written before this existed.**
- The value is a relative path validated the way every other
  workflow-supplied path segment is - through `validate_output_path` against
  the run directory, so `../` and absolute paths are refused rather than
  escaping into another run.
- It may carry a `variable:` reference, so a caller can route a run's
  outputs without editing the workflow.
- It is per step, not per workflow: the point is that one run's steps land
  in different places.

The convention the tooling steers toward is two names, `final` and
`intermediate` - see [Steering](#steering-consumers-toward-it). The engine
does not know those names or treat them specially; a workflow that wants
`shots/` and `audio/` gets them.

## What each surface does with it

### The run directory and the manifest

```
outputs/dialogue-short/20260911-205805-a1b2c3d4/
  manifest.json
  workflow.json
  intermediate/dialogue_short-shot_a.0-0.0.mp4
  intermediate/dialogue_short-slice_a.0-0.0.wav
  final/dialogue_short-assemble.0-0.0.mp4
```

The manifest already records each step's files as paths relative to the run
directory (`manifest_relative_files`), so a foldered file appears as
`final/dialogue_short-assemble.0-0.0.mp4` with no schema change to the
manifest at all. Each manifest entry additionally carries `folder` (the
empty string for an unfoldered step), so a consumer can group without
parsing paths.

### `get_job` and `list_jobs`

`get_job`'s manifest gains the `folder` on each entry, which is additive. On
top of that, a finished job's reply carries:

```json
"outputs": { "final": ["final/...mp4"], "intermediate": ["intermediate/...mp4", "..."] }
```

- a grouping of the manifest's files by folder, so "what did this produce"
  is answerable without walking the manifest. A job whose workflow uses no
  folders reports `{"": [...]}`, which is the honest answer rather than a
  guess at which file is the deliverable.

`list_jobs` stays a listing: it gains nothing per job, because a row that
carried file lists is what T018 just finished cutting down. The `total`/
`truncated` shape is untouched.

### The gallery listing (`GET /api/gallery`, MCP `list_gallery`)

This is the surface where the decision above bites. The rule:

- `folder` on a gallery entry stays **the workflow identity**, exactly as
  today - `ltx2/Gyre`, not `ltx2/Gyre/final`. The folder filter continues to
  group a workflow's runs, which is what it is for.
- A new field, `group` (`final`, `intermediate`, or `""`), carries the
  in-run folder, and `GET /api/gallery` takes an optional `?group=` filter
  beside the existing `?folder=`.
- `strip_run_id` is extended to drop everything after the run id rather than
  just the file name: `ltx2/Gyre/<run>/final/x.mp4` -> `ltx2/Gyre`, and the
  discarded segments become the `group`. Today's paths are unaffected
  because they have nothing after the run id.

This keeps the two axes separate - *which workflow* and *which part of the
run* - rather than multiplying them into one filter list that grows by a
factor of two for every workflow.

### `asset:` and `previous_result:`

- **`previous_result:` is unaffected.** It names a step, and results are
  passed in memory between steps; where a step's files landed on disk has
  never been part of it. A foldered step is read back exactly as it is now.
- **`output:` gains one segment.** The name is
  `<workflow identity>/<run id>/<file>`, where `<file>` is already allowed
  to be a relative path - `_resolve_segments` walks segments and
  `validate_output_reference` permits `/`. So
  `output:dialogue-short/latest/final/episode.mp4` works with **no change to
  the resolver**, and the `latest` search (newest run that holds the file)
  keeps working because it is a per-run existence check on the whole
  remainder of the path.
- **`asset:` is unaffected.** The asset library is a separate tree and
  already takes nested names (`asset:gyre/frames/web.mp4`).
- **`keep_output`** takes the output's relative name, which now may contain
  the folder, and writes into the asset library under `asset_name` as
  before. The output's folder is not copied into the asset name - the two
  trees mean different things, and a promoted file is by definition final.

### The web UI

One change, mapped onto what the gallery already has: beside the existing
folder filter (workflow identity), a second, smaller control for `group`,
defaulting to showing everything. A run that used no folders shows exactly
what it shows today. The job page groups the manifest it already renders
under the same headings.

The editor's form for a `result` block gains `folder` the way it gains every
other schema field - from the schema, with no editor-specific work.

### Breaking changes

**None over MCP or REST.** Everything above is additive: a new optional
workflow field, new keys on existing replies, a new optional query
parameter. Two things a scripted consumer should know:

1. A manifest file name may now contain a `/` where it previously could not.
   Anything that assumed a manifest entry was a bare file name - splitting
   on `/` and taking the last part, or joining it to a directory by hand -
   keeps working, but a consumer that wants the folder should read the
   `folder` field rather than parsing the path.
2. `gallery` entries gain `group`; `folder` keeps its current meaning. A
   consumer filtering on `folder` sees no change.

## Steering consumers toward it

The ticket is explicit that this only pays off if the MCP consumer uses it,
and a schema field nobody sets is worth nothing. Four places, in the order
an agent meets them:

1. **The templates.** Every multi-step template in `workflows/templates/`
   marks its deliverable step `"folder": "final"` and its scratch steps
   `"folder": "intermediate"`. This is the one that matters most: agents
   compose by copying a template, so the convention propagates whether or
   not anyone reads a description.
2. **The authoring guide** (`docs/WORKFLOW_GUIDE.md`, the `Authoring a
   workflow from an agent` section, and its CLAUDE.md mirror) states the
   two-name convention and the rule: a step whose output the user will be
   shown is `final`, everything else is `intermediate`.
3. **Tool descriptions.** `run_workflow` says nothing new (it does not
   author). `get_job` names the `outputs` grouping, and `save_workflow`'s
   description states the convention for a workflow being stored for reuse.
4. **The plugin skills** (`plugins/dw/skills/*`) each state it for their
   family's templates, which is where a composing agent is already reading.

## What this does not do

- **No nesting policy for the libraries.** The workflow, prompt and asset
  libraries are untouched.
- **No automatic classification.** The engine does not guess which step is
  final. A workflow that says nothing gets today's behaviour, and the
  reports say `""` rather than inventing an answer.
- **No retention policy.** "Prune the intermediates, keep the finals" is an
  obvious next thing to want and is not part of this: it needs a decision
  about what happens to an `output:` reference pointing into a pruned
  folder, and that is its own proposal.
- **No move-after-the-fact.** There is no "mark this output final" call on a
  finished job. The folder is decided by the workflow, at write time.
- **No change to file naming.** Names stay
  `{workflow}-{step}.{i}-{j}.{k}.ext`; the folder is a prefix, not a
  replacement for the step name in the name.

## Phasing

1. **The engine.** `folder` in the schema and in `Result.save`, path
   validation against the run directory, `folder` on manifest entries,
   `strip_run_id` extended. Tests: an unfoldered workflow writes exactly
   what it writes today; a foldered one lands where it says; `../` is
   refused; `output:.../latest/final/x.mp4` resolves.
2. **The server surfaces.** `outputs` grouping on `get_job`, `group` on
   gallery entries, `?group=` filter, MCP pass-through, docs.
3. **The steering.** Templates marked up, guide and skills updated, the
   catalog re-audited so `list_workflows` traits still hold.
4. **The UI.** The group control on the gallery, grouped manifest on the job
   page.

Stages 1 and 2 are the ticket; 3 is what makes it used; 4 is what makes it
visible to a person rather than an agent.

## Open questions

1. **One level, or a path?** The ticket asks for one level deep, "to stay
   consistent with the workflow and prompt libraries". Those libraries are
   in fact not one level deep - both walk the tree (`workflow_names`,
   the prompt library's `prompt:folder/name`) and a name is a relative path
   of any depth; `templates/minimax/reference-to-video` is a real catalog
   name with two levels. So consistency with them means *not* enforcing a
   depth limit. My recommendation: allow a path, validate it, and make the
   *convention* one level (`final`, `intermediate`) in the templates and the
   guide - enforcement would be a rule with nothing behind it, and it would
   block `shots/act-1` on a workflow that wants it. This is the one place
   where the proposal as written diverges from the ticket, and it is Don's
   call.
2. **Should `get_job`'s `outputs` grouping exist at all**, or is the
   `folder` field on each manifest entry enough? The grouping is
   redundant-but-cheap; it exists because it answers "what did this
   produce" in one read, which is the question the ticket is actually
   about.
3. **`final` as a default for a one-step workflow?** A `shot`-shape
   workflow has exactly one output and it is the deliverable. Defaulting it
   to `final` would make the common case right for free - but it would also
   mean the same workflow's files move on upgrade, which breaks any stored
   `output:` reference to them. Recommendation: no default, ever.
