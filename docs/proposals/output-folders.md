# Design: subfolders for a run's outputs

Status: **stage 1 (engine) implemented; stages 2-4 pending**. Written for MCP feedback ticket T016;
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
  now closed). Its *shape* is checked by a segment pattern,
  `^[\w][\w.-]*(/[\w][\w.-]*)*\Z` - the same rule a segment of an
  `output:` reference obeys (`OUTPUT_REFERENCE_PATTERN`), so every
  subfolder the engine writes is one a later workflow can name. That
  refuses `..`, a leading or trailing `/`, an empty segment, a segment
  beginning with `.` or `-`, and `\` - which matters because
  `DANGEROUS_PATTERNS` (`dw/security.py`) does not list a backslash, so
  `"final\\x"` would otherwise be one directory on POSIX and two on
  Windows. Its *containment* is then checked by `validate_output_path`
  against the run directory, which is what guards symlinks. One ceiling
  follows from the alignment: `OUTPUT_REFERENCE_PATTERN` allows seven
  segments in total, so a subfolder deep enough to push
  `<identity>/<run>/<subfolder>/<file>` past seven is writable but not
  `output:`-addressable. Documented, not enforced.
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

### Where the checks live

Schema validation runs *before* substitution, so a schema `pattern` on
`subfolder` would reject the `"variable:dest"` and `"item:subfolder"` this
design wants. The schema therefore gets a `description` only. The shape
check runs in two places:

- **Statically**, as a post-expansion pass in `validation_errors`
  (`dw/workflow.py`), beside `previous_result_reference_errors` - a new
  `subfolder_errors` in a small module of its own, reporting the JSON path
  of the offending `result` block, so `dw.validate` and `POST /api/validate`
  name it before anything is queued. It runs on the substituted, expanded
  definition, so an `item:`-driven subfolder is checked per member.
- **At run time**, where the step's target directory is computed (below),
  for a definition that reached the engine without validation.

### `file_base_name` may no longer contain a separator

`"file_base_name": "final/"` passes `validate_string_input` and
`validate_output_path` today and then fails at `open()` because the
directory does not exist - a latent crash, not a feature. With a real
placement field it becomes an error naming `subfolder` as the way to do
it, checked in the same two places. No shipped workflow (`workflows/`,
`dw/workflows/`, `plugins/`) uses a separator there.

## What each surface does with it

### The engine (`dw/result.py`, `dw/workflow.py`, `dw/runs.py`)

- **The step's target directory is computed once, in `workflow.py`**, not
  inside `Result.save`. A new `Workflow.step_output_dir(step_definition)`
  returns `effective_output_dir` joined with the step's `subfolder`
  (shape-checked, then `validate_output_path(joined, effective_output_dir)`,
  then created), or `effective_output_dir` itself when there is none. It is
  handed to *both* `Result.save` as its `output_dir` and to
  `create_step_action` as the pipeline's `output_dir` (`workflow.py`
  ~1003/1032/962). The second matters: a chain pipeline's `save_segments`
  spill (`dw/pipeline_processors/chain.py`, `SegmentSpill`) writes through
  the pipeline's `output_dir`, so without this a `keep_segments: true`
  step's segments would land at the run root while its video landed in
  `final/`. `Result.save` itself is unchanged apart from refusing a
  separator in `file_base_name`: names stay
  `{workflow}-{step}.{i}-{j}.{k}.ext`, deduplication stays per path.
- The manifest entry `workflow.py` appends per step gains
  `"subfolder": <value or "">`, and so does the `step_end` event it emits
  - the job page groups from `step_end` events while a job runs and
  confirms from the manifest at the end (`ui/src/lib/results.ts`), so a
  field only on the manifest would appear only once the job finished. A
  step-cache hit keeps the *definition's* subfolder while its files stay
  the earlier run's absolute paths, exactly as `reused` entries work now;
  the cache compares the whole step snapshot including `result`, so a
  changed subfolder misses.
- **Sub-workflows.** A `workflow` step's own `result` block governs only
  what the *parent* saves from the child's return value (`dw/step.py`); the
  child's steps save through their own `result` blocks into the inherited
  run directory, and the parent rolls their entries up verbatim. So a
  parent's `subfolder` never prefixes the child's steps - each step places
  its own files. This is the right reading (a child is a step list, not a
  folder) and needs no code, but it is stated because the alternative is
  plausible.
- `dw/runs.py` gains `split_run_path(relative) -> (identity, run_id,
  subfolder)`: locate the first directory segment matching
  `RUN_ID_PATTERN` (`^\d{8}-\d{6}-[0-9a-f]{8}(-\d+)?$`); identity is
  everything before, subfolder everything after. A workflow *file* named
  in that shape would produce a matching identity segment - unsupported,
  not impossible. `strip_run_id` becomes a thin wrapper returning the
  identity, so its callers and tests hold. A path with no run id (flat
  layout) returns its directory as identity and `""` as subfolder, as
  today.

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
4. `$defs/result` accepts additional properties, so a workflow that already
   carries a stray `subfolder` key validates today and is ignored; after
   stage 1 it moves files. One line in the release note.

## Steering consumers toward it

The ticket is explicit that this only pays off if the MCP consumer uses it.
Four places, in the order an agent meets them:

1. **The templates** (*decided*: `workflows/templates/**`, and the
   templates the plugin skills name). A *saving step* is one whose
   `result` sets `content_type` and does not set `save: false`; a template
   is in scope when it has **two or more** saving steps. The deliverable
   step gets `"subfolder": "final"`, scratch steps
   `"subfolder": "intermediate"`; a template whose saving steps are all
   deliverables (`image-processors`, `lora-styles` - a set of variants,
   each final) marks them all `final`. A template whose saving steps are
   all `workflow` steps over `builtin:` children (`compose-workflows`,
   `sub-workflow`) is exempt: the files come from the child's steps, and
   marking a builtin's step `final` would presume a role it does not have
   - the builtins in `dw/workflows/` stay unmarked. Agents compose by
   copying a template, so the convention propagates whether or not anyone
   reads a description. A test asserts the rule as stated here - every
   in-scope, non-exempt template's saving steps all carry a subfolder - so
   the convention cannot drift. `workflows/models/**` is out of scope for
   this pass, though `list_workflows` returns those beside the templates;
   marking them up is a follow-up once the convention has held in the
   templates.
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
  foldered one lands in `<run>/final/`; `../final`, `/final`, `final/`,
  `final//x`, `.hidden`, `final\x` and an absolute path are refused by
  the shape check, a symlink escape by containment; `subfolder_errors`
  reports the JSON path for a bad literal and for a bad `item:`-driven
  member; a separator in `file_base_name` is refused statically and at
  run time; a `for_each` member with an `item:` subfolder lands per
  member; a chain step with `keep_segments` puts its segments in the
  subfolder; `manifest.json`, the job manifest and the `step_end` event
  all carry the `subfolder` field; a reused entry carries the definition's
  subfolder; a parent's subfolder on a `workflow` step does not move the
  child's files; `output:.../latest/final/x` resolves; `split_run_path`
  on `a/b/<run>/final/x`, on a `-2` counter run id, and on a flat path.
- **Server.** Gallery entries carry `subfolder`; `?subfolder=` filters;
  `subfolders` lists distinct values; `job_for_file` attributes a foldered
  file; MCP `list_gallery` passes the parameter through.
- **UI.** Gallery subfolder filter and job-page grouping (vitest).
- **Templates.** Every template with two or more saving steps, not exempt
  as a `builtin:` composition, has a subfolder on each saving step.

## Phasing

Four stages, each its own PR to `develop`:

1. **Engine.** Schema description, `subfolder_errors` in
   `validation_errors`, `step_output_dir` feeding `Result.save` and
   `create_step_action`, `file_base_name` check, `subfolder` on the
   manifest entry and the `step_end` event, `split_run_path`/`strip_run_id`,
   engine tests. `get_job` entries carry the field from this stage on -
   the server spreads the entry as recorded.
2. **Server and MCP.** Gallery `subfolder`/`?subfolder=`/`subfolders`,
   MCP `list_gallery` parameter, `get_job`/`save_workflow`/`list_gallery`
   descriptions, SERVER/MCP/GUIDE docs and the CLAUDE.md mirror.
3. **Steering.** Templates marked up with the drift test, skills updated,
   release note. Nothing in the catalog derivation reads `subfolder`
   (`catalog_shape.py` reads `result.content_type` only), so shapes and
   traits cannot move; one side effect is welcome - an `item:subfolder`
   reference makes `subfolder` a derived entry field in a list-driven
   workflow's `lists`.
4. **UI.** `subfolder` on the `ManifestEntry` and `JobEvent` types,
   gallery control, job-page grouping (live from `step_end`, confirmed
   from the manifest), editor picks the field up from the schema.

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
