# Design: subfolders for a run's outputs

Status: **designed, not implemented**. Written for MCP feedback ticket T016;
reviewed against the code on 2026-09-12. Decisions taken in that review are
marked *decided*.

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

## Why this needed a design and not an edit

A destination folder is a new field in the workflow schema, so every
consumer has to learn it and every surface has to carry it. But the reason
it needed deciding is narrower: **the output tree already has a layer with
the meaning this wants**, and three things read that tree by position.

A run writes to `<output root>/<workflow identity>/<run id>/`:

- `strip_run_id` (`dw/runs.py`) turns `ltx2/Gyre/20260905-181530-a1b2c3d4/
  still-0.png` into the gallery folder `ltx2/Gyre`, which is how a workflow
  run fifty times is one entry in the folder filter rather than fifty. It
  does this by testing whether the *last* directory segment is a run id.
- `output:` references are `<workflow identity>/<run id>/<file>`, with
  `latest` allowed in the run-id position (`resolve_output_reference`).
- The manifest, and `workflow.json` beside it, sit at the root of the run
  directory and describe everything in it.

A folder *inside* a run directory puts a segment between the run id and the
file name. As the code stands, `strip_run_id` on
`ltx2/Gyre/<run>/final/x.mp4` returns `ltx2/Gyre/<run>/final` - the run id
is no longer the last segment, so the gallery filter would gain one entry
per run. That is the part that had to be decided before anything was
written. The other two readers turn out to need nothing (below).

There is also a question the ticket settles one way and the existing code
settles the other: the workflow and prompt libraries do not enforce one
level, so "consistent with them" means not enforcing one here either.

## The field

A step's `result` block names the subfolder of the run directory its files
are written into:

```json
{
  "name": "assemble_episode",
  "task": { "command": "concat_videos", "arguments": { "...": "..." } },
  "result": {
    "content_type": "video/mp4",
    "subfolder": "final"
  }
}
```

- **Named `subfolder`, on every surface** (*decided*). It is literally a
  subfolder of the run directory. The word `folder` was rejected because a
  gallery entry's `folder` already means the workflow identity, and an
  author who set `"folder": "final"` would have read back
  `folder: "ltx2/Gyre"` beside some other key; `group` was rejected because
  `for_each` documentation already uses "group" for what `gather:` reads.
  One word, same meaning, in the schema, the manifests, the gallery and the
  query string.
- Optional. Without it, files are written where they are written today: at
  the root of the run directory. **Every existing workflow keeps working
  unchanged, and a workflow that never sets `subfolder` writes
  byte-identical paths to one written before this existed.**
- A relative path of any depth (*decided*; see the first open question,
  now closed). Validated through `validate_output_path` against the run
  directory, so `../`, absolute paths and symlink escapes are refused
  rather than reaching another run. `\` is refused like `/../`.
- Substitution applies as it does to the rest of the step: `variable:` so a
  caller can route a run's outputs without editing the workflow, and
  `item:` inside a `for_each` template so members can land in their own
  places (`"subfolder": "item:subfolder"`). This is the case that argues
  against a depth limit - `shots/act-1` is a natural thing to want.
- Per step, not per workflow: the point is that one run's steps land in
  different places.
- **No default, ever** (*decided*). A one-step workflow's output is by
  definition the deliverable, but defaulting it to `final` would move
  every such workflow's files on upgrade and silently stop any stored
  `output:` reference from advancing.

The convention the tooling steers toward is two names, `final` and
`intermediate` - see [Steering](#steering-consumers-toward-it). The engine
does not know those names or treat them specially; a workflow that wants
`shots/` and `audio/` gets them.

### `file_base_name` may no longer contain a separator

`"file_base_name": "final/"` passes `validate_string_input` and
`validate_output_path` today and then fails at `open()` because the
directory does not exist - a latent crash, not a feature. With a real
placement field it becomes a validation error naming `subfolder` as the way
to do it. No shipped workflow (`workflows/`, `dw/workflows/`, `plugins/`)
uses a separator there.

## What each surface does with it

### The engine (`dw/result.py`, `dw/workflow.py`, `dw/runs.py`)

- `Result.save(output_dir, base_name)` reads `subfolder` from the result
  definition, joins it onto `output_dir`, validates the joined directory
  with `validate_output_path(joined, output_dir)`, creates it, and saves
  into it. Nothing else in `save` changes: names stay
  `{workflow}-{step}.{i}-{j}.{k}.ext`, deduplication stays per path.
- The manifest entry `workflow.py` appends per step gains
  `"subfolder": <value or "">`. A step-cache hit keeps the *definition's*
  subfolder while its files stay the earlier run's absolute paths, exactly
  as `reused` entries work now. Sub-workflow entries roll up as they do
  now; a sub-workflow's subfolder is relative to the run directory it
  inherits, which is the right reading.
- `dw/runs.py` gains `split_run_path(relative) -> (identity, run_id,
  subfolder)`: locate the first directory segment matching
  `RUN_ID_PATTERN` (`^\d{8}-\d{6}-[0-9a-f]{8}(-\d+)?$`, specific enough
  that no identity segment matches it); identity is everything before,
  subfolder everything after. `strip_run_id` becomes a thin wrapper
  returning the identity, so its callers and tests hold. A path with no
  run id (flat layout) returns its directory as identity and `""` as
  subfolder, as today.

### The two manifests

There are two, and they record files differently:

- `manifest.json` in the run directory records files relative to the *run
  directory* (`manifest_relative_files`), so a foldered file appears as
  `final/dialogue_short-assemble.0-0.0.mp4`.
- The job's manifest in `jobs.sqlite`, returned by `get_job`, records files
  relative to the *output root* (`_relative_output_names`), so the same
  file appears as `dialogue-short/<run>/final/dialogue_short-assemble.0-0.0.mp4`.
  `job_for_file` matches on the tail of that name, so attribution of a
  foldered file keeps working.

Both already carry directory segments, so neither needs a schema change.
Both gain `subfolder` on each step entry so a consumer groups without
parsing paths.

```text
outputs/dialogue-short/20260911-205805-a1b2c3d4/
  manifest.json
  workflow.json
  intermediate/dialogue_short-shot_a.0-0.0.mp4
  intermediate/dialogue_short-slice_a.0-0.0.wav
  final/dialogue_short-assemble.0-0.0.mp4
```

### `get_job` and `list_jobs`

`get_job`'s manifest entries carry `subfolder`. **Nothing else is added**
(*decided*): an earlier draft bucketed the files into an `outputs` map,
which would have needed `""` as a JSON key for unfoldered steps and added
a second way to read what the entries already say. The tool description
carries the reading instead: *a step's `subfolder` says what kind of
output it is; by convention the deliverable is `final`*.

`list_jobs` stays a listing and gains nothing per job - a row that carried
file lists is what T018 just finished cutting down.

### The gallery (`GET /api/gallery`, MCP `list_gallery`)

- `folder` on an entry stays **the workflow identity**, exactly as today -
  `ltx2/Gyre`, not `ltx2/Gyre/final`. The folder filter continues to group
  a workflow's runs, which is what it is for.
- Each entry gains `subfolder` (`final`, `intermediate`, `""`, or whatever
  the workflow wrote), from `split_run_path`.
- `GET /api/gallery` takes an optional `?subfolder=` beside `?folder=`; the
  reply's `folders` list is unchanged and gains a sibling `subfolders`
  (distinct values over the whole tree, for the UI control).
- MCP `list_gallery(limit, subfolder=None)`. Today it takes only `limit`,
  so this is a new parameter, not a mirror of an existing one.

The two axes - *which workflow* and *which part of the run* - stay
separate rather than multiplying into one filter list.

**Flat layout.** With `--output-layout flat` there is no run id to anchor
on, so `subfolder` is `""` and a foldered file's directory - `ltx/final` -
appears in the workflow filter. Accepted: flat is the legacy layout for
callers whose scripts glob the output directory, and the on-disk placement
is still honoured, which is what such a caller wants.

### `output:`, `asset:`, `previous_result:`, `keep_output` - no change

Verified against the code:

- **`previous_result:`** names a step and results pass in memory; where a
  step's files landed has never been part of it.
- **`output:`** names are `<workflow identity>/<run id>/<file>` where
  `<file>` is already a relative path: `_resolve_segments` walks every
  remaining segment and `validate_output_reference` permits `/`. So
  `output:dialogue-short/latest/final/episode.mp4` resolves with no change,
  and `latest` (newest run *holding the file*) keeps working because it is a
  per-run existence check on the whole remainder.
- **`asset:`** is a separate tree that already takes nested names.
- **`keep_output`** / `POST /api/assets/keep` take the output's relative
  name, which may now contain the subfolder, and write into the asset
  library under `asset_name` as before. The subfolder is not copied into
  the asset name - a promoted file is by definition final.
- `delete_output`, `get_output_image`, `download_output`, `/outputs/`
  serving all take the whole remainder as a path.

### The web UI

- Gallery: a second, smaller filter for `subfolder` beside the folder
  filter, fed by `subfolders`, defaulting to everything. `grouping.ts` is
  unchanged - the server still supplies `folder`.
- Job page: the manifest it already renders is grouped under headings by
  `subfolder` when any entry has a non-empty one; unchanged otherwise.
- Editor: `result.subfolder` appears from the schema, no editor work.

### Breaking changes

**None over MCP or REST, and none to existing workflows or workspaces.**
Everything is additive: a new optional workflow field, new keys on existing
replies, a new optional query parameter, no migration of `jobs.sqlite` or
of run directories. Three things a consumer should know:

1. A manifest file name may now contain a `/` after the run id where it
   previously could not. A consumer that wants the subfolder should read
   the `subfolder` field rather than parse the path.
2. `file_base_name` containing a separator is refused at validation. It
   failed at write time before, so no working workflow changes behaviour.
3. **The steering stage moves template outputs.** Once a template marks
   steps `final`/`intermediate`, its files land in `<run>/final/x.mp4`.
   A user workflow chained off a *template's* output by run path
   (`output:<template identity>/latest/x.mp4`) keeps resolving - `latest`
   picks the newest run holding the file - but stops advancing past the
   last pre-change run. `keep_output` is the supported way to make a
   generated file a stable input. Adding `subfolder` also changes a step's
   cache key, so the first run of a marked-up seeded template regenerates
   rather than hitting the step cache. Both go in the release note beside
   the `shots` list change.

## Steering consumers toward it

The ticket is explicit that this only pays off if the MCP consumer uses it.
Four places, in the order an agent meets them:

1. **The templates** (*decided*: `workflows/templates/**` with more than
   one saving step, and the templates the plugin skills name). The
   deliverable step gets `"subfolder": "final"`, scratch steps
   `"subfolder": "intermediate"`. Agents compose by copying a template, so
   the convention propagates whether or not anyone reads a description. A
   test asserts every multi-step template's saving steps all carry a
   subfolder, so the convention cannot drift.
2. **The authoring guide** (`docs/WORKFLOW_GUIDE.md`, `Authoring a workflow
   from an agent`, and its CLAUDE.md mirror - the two change together)
   states the rule: a step whose output the user will be shown is `final`,
   everything else is `intermediate`.
3. **Tool descriptions.** `get_job` states the reading; `save_workflow`
   states the convention for a workflow stored for reuse; `list_gallery`
   names the parameter. `run_workflow` says nothing new.
4. **The plugin skills** (`plugins/dw/skills/*`) each state it for their
   family's templates.

**No validation nudge** (*decided*): a warning on a multi-step workflow
with no subfolders would fire on every existing example and on workflows
whose outputs are all legitimately final.

## What this does not do

- No nesting policy for the workflow, prompt or asset libraries.
- No automatic classification: the engine does not guess which step is
  final. A workflow that says nothing gets today's behaviour and reports
  `""`.
- No retention policy. "Prune the intermediates, keep the finals" is the
  obvious next want and needs its own decision about `output:` references
  into a pruned subfolder.
- No move-after-the-fact: no "mark this output final" call on a finished
  job. Placement is decided by the workflow, at write time.
- No change to file naming.

## Tests

- **Engine.** An unfoldered workflow writes the same paths as today; a
  foldered one lands in `<run>/final/`; `../final`, an absolute path and a
  symlink escape are refused; a separator in `file_base_name` is refused;
  a `for_each` member with an `item:` subfolder lands per member;
  `manifest.json` and the job manifest both carry the segment and the
  `subfolder` field; a reused entry carries the definition's subfolder;
  `output:.../latest/final/x` resolves; `split_run_path` on
  `a/b/<run>/final/x`, on a `-2` counter run id, and on a flat path.
- **Server.** Gallery entries carry `subfolder`; `?subfolder=` filters;
  `subfolders` lists distinct values; `job_for_file` attributes a foldered
  file; MCP `list_gallery` passes the parameter through.
- **UI.** Gallery subfolder filter and job-page grouping (vitest).
- **Templates.** Every multi-step template's saving steps carry a
  subfolder.

## Phasing

Four stages, each its own PR to `develop`:

1. **Engine.** Schema, `Result.save`, `file_base_name` check, manifest
   field, `split_run_path`/`strip_run_id`, engine tests.
2. **Server and MCP.** Gallery `subfolder`/`?subfolder=`/`subfolders`,
   `get_job` entries, MCP parameter and descriptions, SERVER/MCP/GUIDE
   docs and the CLAUDE.md mirror.
3. **Steering.** Templates marked up with the drift test, skills updated,
   catalog re-audited so `list_workflows` traits still hold, release note.
4. **UI.** Gallery control, job-page grouping, editor picks the field up
   from the schema.

Stages 1 and 2 are the ticket; 3 is what makes it used; 4 is what makes it
visible to a person rather than an agent.

## Open questions - closed

1. **One level, or a path?** A path. The libraries walk the tree
   (`templates/minimax/reference-to-video` is a real two-level catalog
   name), so consistency with them means no depth limit; the convention is
   one level.
2. **`get_job` grouping?** No; the per-entry field is enough.
3. **`final` as a default for one-step workflows?** No default, ever.
4. **Field name?** `subfolder`, on every surface.
5. **Steering scope?** Multi-step templates and the plugin skills'
   templates.
6. **Validation nudge?** No.
