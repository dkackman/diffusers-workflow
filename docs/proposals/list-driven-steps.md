# Proposal: list-driven steps (`for_each`)

Status: **stages 1 and 2 implemented** (expansion pass, validation, schema,
docs - `dw/for_each.py`; `music-video` and `dialogue-short` on a `shots`
list, 2026-09-11); stage 3 (catalog per-entry cost and entry shape) not
started. Written for MCP feedback ticket T003.

## The ask, as filed

> `music-video` and `dialogue-short`: each generates one step per shot. A
> `shots: [...]` variable should fan out generation steps, one per list entry.
> Step count is fixed in the template; adding a shot means editing the
> template.

Both templates are written the long way. `music-video` has `shot_1_wide_open`,
`shot_2_closeup`, `shot_3_room`, `shot_4_finale`, each a near-copy of the one
before, and `slice_1` … `slice_4` beside them cutting the audio bed into
matching pieces. `dialogue-short` has five shot steps and two portrait steps.
A five-shot episode is the template; a six-shot episode is a different file.

The consumer side already moved this way: `assemble-and-score` takes a list,
and `compose_text(parts=[...])` (T002) assembles text written once. What is
missing is the generation side - a step that runs once per entry.

## Why this is a proposal and not an edit

Fanning a step out per list entry forces a question the ticket does not
settle: **how does the downstream step name the group?**

`previous_result:shot` today means "that step's result". A result holding N
artifacts drives the cartesian iteration the engine already has -
`dw/previous_results.py:get_iterations`, four images x three masks = twelve
runs. So a fanned-out group read back through `previous_result:` would run
the concat once *per shot* rather than once *over all of them*. Gathering and
iterating are two different meanings and the reference syntax has one
spelling. That, plus what the expanded steps are called, what the step cache
keys on, and what the manifest and the realized workflow record, is more
than a syntax addition.

## The shape

```json
{
  "name": "shot",
  "for_each": "variable:shots",
  "pipeline": {
    "arguments": {
      "prompt": "item:prompt",
      "references": [
        { "reference_type": "variable:image_reference_type",
          "from_previous_result": "draw_singer" }
      ]
    }
  }
}
```

with

```json
"shots": [
  { "name": "wide_open", "prompt": "the band walks on, wide" },
  { "name": "closeup",   "prompt": "closeup on the singer" }
]
```

and downstream

```json
{ "name": "edit",
  "task": { "command": "concat_videos", "arguments": { "videos": "gather:shot" } } }
```

- `for_each` names a list - a `variable:` reference, or a literal list. One
  step per entry, in list order.
- `item:` inside the step is that entry: `item:prompt` is one field, bare
  `item:` is the whole entry (a list of plain strings is the common case).
  A field holds whatever the entry wrote there - a string, a number, a list
  or an object - and the pass splices it in as definition text. This is not
  optional: `dialogue-short`'s shots reference both characters in shots 1
  and 5 and one character in shots 2-4, so a shot's `references` list has to
  come from the entry whole (`"references": "item:references"`), and a
  scalar-only `item:` could not write that template. An entry may therefore
  carry `from_previous_result`, `asset:` and `prompt:` strings; they become
  ordinary references in the expanded step and are checked as such (below).
- **Same-key siblings.** Inside a `for_each` over a list, a
  `previous_result:`/`from_previous_result` naming *another* `for_each` step
  over the same list resolves to the member with the same key:
  `"from_previous_result": "slice"` inside `shot` becomes
  `slice@wide_open` inside `shot@wide_open`. This is the one pairing the
  engine needs and the reason it is not a zip (below): `music-video` slices
  the audio bed in one task step and generates the shot in a pipeline step,
  so shot *i* reads slice *i* across two steps, and "put both in one entry"
  cannot express it. The two groups must name the same list (the same
  `variable:` or an identical literal); a same-key reference into a group
  over a different list is an error.
- `gather:shot` is the list of every member's artifacts, in order, as one
  value. It is a *list-valued* reference, which is the distinction
  `previous_result:` cannot make.

### Recommendation: expansion is a source transform, not a runtime concept

Run the expansion as a pass over the definition immediately after
`replace_variables` in `Workflow.run` (`dw/workflow.py`), before the run id is
computed and before the step loop starts. The pass:

1. replaces each `for_each` step with N ordinary steps, named `shot@wide_open`
   … (below);
2. substitutes `item:` references inside each copy with that entry's values,
   and rewrites a same-key sibling reference to the member with this key;
3. rewrites every `gather:shot` into the explicit list of
   `["previous_result:shot@wide_open", "previous_result:shot@closeup"]` -
   exactly what a hand-written template contains today.

After the pass the definition is an ordinary workflow. Nothing below it
changes: no new reference kind at run time, no change to what
`previous_result:` means, no change to the cartesian rule, and the step cache,
the manifest, the event stream and `pipeline_reference` all keep working
because they are looking at ordinary steps. **This is the argument for the
whole design** - the alternative, teaching `get_previous_results` a second
meaning, touches every consumer of a result and leaves two spellings that
differ only by whether a group happens to be behind them.

Consequences worth stating:

- A `gather:` written *inside* a list splices into it
  (`["previous_result:intro", "gather:shot"]` is one flat list), because
  textual expansion is the natural reading. A `gather:` naming a step that is
  not a `for_each` group is an error, not a one-element list - that is a typo
  every time.
- Pipeline loading is already shared: `pipeline_cache_key` hashes what a
  pipeline *loads*, excluding arguments and seed, so N identical expanded
  pipeline blocks hit one loaded model within the run. The expansion needs no
  `pipeline_reference` rewriting to get the reuse the hand-written template
  gets by hand (the templates' `pipeline_reference` shots become full
  pipeline blocks in the expansion, and `pipeline_reference` inside a
  `for_each` step is meaningless). `release_pipeline` on a fanned step would
  drop the model after the first member and reload it for the second: the
  pass carries it onto the **last member only**, and `release_models` the
  same way.
- `realize_args(variables)` runs on the variables *before* substitution
  (`dw/arguments.py`, the `isinstance(v, list)` branch hands a list to
  `resolve_path_references`). Stage 1 must confirm that an `asset:` inside an
  entry object comes out as a path and not as loaded media, or `item:` would
  splice a PIL object into the definition. If it loads, exempt the list a
  `for_each` names and let the expanded step realize it as any step does.

### Expanded names: the entry's own name before its index

`shot@wide_open` when the entry is an object carrying a `name`, else
`shot@0`, `shot@1`.

The index alone is tempting and wrong for the cache. Entries are keyed
`(workflow id, step name)` in `dw/step_cache.py`, so inserting a shot in the
middle shifts every later index onto a different entry's content. The cache
validates against the step's data as well, so the result is correct - it is a
miss - but every shot after an insertion regenerates, which on a six-shot H3
episode is most of an hour of GPU to add one shot in the middle. A name
carried by the entry survives insertion, and it also makes the manifest, the
event stream and the gallery read in shot names rather than in ordinals.

`@` rather than dots, underscores or brackets: `.` is already the property
separator in a reference (`previous_result:segment.mask`), `_` and `-`
collide with a hand-written `shot_1`, and `:` is the reference-prefix
separator. Brackets were the first draft and are wrong for a reason that
only shows up later: the name lands in a filename
(`{workflow_id}-{step_name}.{i}`, `dw/workflow.py`), and `[...]` is a shell
glob class - `ls *shot[wide_open]*` matches one character - as well as a
character every gallery and MCP URL would have to encode. `@` is none of
those. A step name is otherwise an unrestricted string, so `@` has to be
reserved - a hand-written step named `shot@0` becomes an error.

**Entry names are validated.** The name reaches the filesystem and the
step-cache key, so it must match the variable-name pattern
(`validate_variable_name`, `^[a-zA-Z_][a-zA-Z0-9_-]*$`) and be unique within
its list - two entries named `closeup` would otherwise expand to two steps
with one name, one cache entry and one clobbered output file. Both are
errors from the pass, reported with the entry's position.

### Validation

The static reference check shipped for T005
(`previous_result_reference_errors`) runs on the unexpanded definition, where
`shot@wide_open` does not exist yet. `validate_workflow` should run the same
expansion first and check the expanded steps. It expands **the list the run
will use**: `POST /api/validate` already takes the caller's `arguments`
(`argument_errors` folds them the way `set_variables` does), so the
pre-flight expands the folded list, not the declared default - otherwise it
checks a step set the caller is not about to run. Without arguments the
default is the list, and an empty or absent one validates as zero steps and
a `gather:` naming nothing. Order becomes: schema -> substitute -> expand ->
reference check. Schema validation stays where it is, against the template
step, and the schema gains `for_each` on a step plus the `item:`/`gather:`
prefixes wherever a reference is allowed.

Two errors deserve their own message rather than the generic "names no
earlier step": `previous_result:shot` where `shot` is a `for_each` group
("use `gather:shot`, or a same-key reference from inside a group over the
same list"), and `gather:` naming a step that is not a group. References an
entry carries (`from_previous_result` inside `item:references`) are checked
in the expanded step like any other, which is where they exist.

### The realized workflow and the manifest

`dw/realize.py` folds the run's arguments into the variable defaults, so a
realized list-driven workflow carries the actual `shots` list and reproduces
the same expansion deterministically. **Keep `for_each` in the realized file**
rather than writing the expanded steps: it reproduces the run exactly and stays
legible. The manifest names the expanded steps (`shot@wide_open`), because the
manifest records what ran. That the two name steps differently is deliberate
and should be documented where the manifest is.

### Limits and cost

Each entry is a full generation, so the expansion needs a ceiling of its own,
well below `MAX_ITERATIONS = 10000`, which guards a different thing. The
ceiling and the step cache's bound have to be stated together: the cache
holds `DEFAULT_MAX_ENTRIES = 50` (`dw/step_cache.py`, LRU), and a run whose
expanded steps exceed it evicts its own earlier members - the insertion case
the naming scheme exists for stops hitting, silently. `music-video` expands
to shots + slices + five fixed steps, so the cache is full at about twenty
shots. Either the ceiling is set so a maximal run fits (32 entries, under
any of the templates) or the cache bound is raised alongside it; the
proposal picks **32 and leaves the cache alone** unless a real template
needs more (raised to 128 in stage 3).

The larger consequence is that a list-driven template's cost is set by the
caller: `list_workflows` quotes a per-workflow figure today, and a six-entry
`shots` list is six times the shot cost. The catalog entry should carry the
per-entry cost and the variable it multiplies by, **and the shape of an
entry** - which fields it takes, which are required, what `name` is for -
because an agent composing over MCP sees only the catalog and cannot
otherwise author the list. The MCP guidance (`docs/WORKFLOW_GUIDE.md`,
"Authoring a workflow from an agent") should say to quote `cost x len(list)`
before running.

### What this does not do

It does not give the engine a "zip". `for_each` runs over **one** list; a
pairing - shot *i* with its audio slice *i* - is expressed by putting both in
the same entry:

```json
{ "name": "wide_open", "prompt": "...", "start_frame": 0 }
```

That is the same answer the guide already gives for the cartesian rule
("write it as one step per pair"), and it is why the entry should be an object
rather than a string in any template that has more than a prompt to vary. The
same-key sibling rule is the one place two *steps* are paired, and it pairs
them by name over one list rather than zipping two. Two `for_each` lists on
one step, or a `for_each` over a cartesian product, is out of scope and
probably always should be.

## Phasing

1. **Expansion pass + `item:` + `gather:`** with structured `item:` values
   and the same-key sibling rule, entry-name validation, schema, validation
   ordering (arguments folded, the two directed errors), the 32-entry
   ceiling, `release_pipeline`/`release_models` on the last member, and the
   reserved `@`. No template changes. This is self-contained and testable
   without a GPU: the pass is a pure function from definition to definition,
   so its tests are ordinary unit tests - and the two tests that matter most
   are the two templates rewritten by hand in the test suite and expanded
   back to what they contain today.
2. **`music-video` and `dialogue-short` rewritten** onto a `shots` list. Both
   are breaking for scripted callers - the per-shot variables
   (`shot_2_deflect`, `shot_3_react`, …) become entries in one list - and
   T005 has already broken those names once this week, so the two should land
   together or the second should wait for a deliberate version bump of the
   templates. The `dw:minimax-h3` plugin skill quotes those variables and
   `tests/test_plugin_skills.py` pins what it quotes, so both move with the
   templates.
3. **Cost and entry-shape reporting** for a list-driven entry, in the catalog
   and in the plugin skills that quote it.

Stage 1 is the bulk of the work and carries the risk; stages 2 and 3 are
mechanical once it exists.

## Decided at review

- **No loop index inside an entry.** Nothing in the two templates needs
  "shot 3 of 6" in a prompt. Adding `item:@index` later is compatible; making
  it available now costs a second special name.
- **`for_each` does not accept a number** (`"for_each": 4`). It reads well
  and it is one line in the pass, but it gives entries no names, which is
  exactly the cache problem above.
- **`dialogue-short`'s two-speaker structure** (which character speaks in
  which shot, T010's voice references) is carried by each entry's
  `references` list, spliced whole by `item:references` - a `"speaker": "a"`
  flag was the first idea and cannot produce a references list whose length
  varies by shot. The template question stage 2 has to get right is how much
  of that list the entry writes and how much the template fixes.

## Notes for stage 2

Observations from the stage-1 review, recorded so the template rewrite does
not have to rediscover them:

- **`pipeline_reference.reference_name` naming a `for_each` group gets no
  directed error.** Both templates use `pipeline_reference` on their shot
  steps, and a reference to a group's name resolves to nothing helpful - the
  expansion only rewrites `previous_result:` and `from_previous_result`.
  Rewriting a shot step onto `for_each` means either pointing every member's
  `pipeline_reference` at a step outside the group, or giving that key the
  same treatment.
- **An entry field named `image`/`*_image`/`*_video` is realized before
  expansion.** `realize_args` runs over the variables, so such a field is a
  loaded PIL image or a decoded frame list by the time `item:image` splices
  it into a member. It works, it is untested, and it is why the expansion
  copies a leaf only inside a member and falls back to the object itself
  when it cannot be copied.
- **A `for_each` over an empty list expands to zero steps.** A workflow whose
  only step is that one returns before `workflow_start`, so the run produces
  no events and no manifest entries. Whether that is an error or an empty
  success is a stage-2 decision.
- **`expanded_definition` skips `realize_constants`.** A list defaulted to a
  `constant:` name fails validation (the list arrives as the `constant:`
  string) and then runs fine, since `Workflow.run` does realize constants. A
  template that wants a constant default needs that pass in the validator.

Resolved in stage 2: the `pipeline_reference` question went away - every
member is the full pipeline block, and the identity-keyed pipeline cache
reuses the loaded model exactly as the reference did (a test holds every
member to one `pipeline_cache_key`). A group's `pipeline_reference` still
gets no directed error; nothing bundled uses one now. Entries that name
other variables (`variable:character_a_voice`) are resolved by
`resolve_variable_values` before `realize_args`, which is what let the
optional voices move into the entries. The empty-list and
`realize_constants` questions are still open and belong to stage 3 with the
catalog work.

## Stage 3: cost and entry shape (design)

Written after stage 2 merged (2026-09-11). Stage 2 left an agent composing
over MCP with two gaps, both in what the catalog tells it: how much a
list-driven run costs when the list is the agent's, and what an entry of
that list has to contain. The stopgap in the skill and the guide - divide
the listing's total by the default entry count - runs high because the
total includes the fixed steps (portraits, the song), and nothing in the
listing says which fields an entry takes. The rules below follow the
catalog's standing principles: cost is measured, never derived; structure is
derived, never declared twice.

### Cost: a measured per-entry figure beside the measured total

A `cost` entry gains an optional `per_entry` block:

```json
"cost": [
  {
    "device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 42,
    "per_entry": {"variable": "shots", "minutes": 7.2, "entries": 5}
  }
]
```

- `minutes` stays what it is: the measured total for the workflow as
  shipped, so every existing consumer and test is untouched.
- `per_entry.minutes` is the measured cost of one entry - the sum of every
  member the entry produces (`slice@x` plus `shot@x` for music-video), taken
  from the run's step timings, averaged over the default list.
- `per_entry.entries` is the length of the default list the total was
  measured with, so the fixed part is recoverable: `fixed = minutes -
  per_entry.minutes × entries`, and a run over N entries quotes `fixed +
  per_entry.minutes × N`. Recording `entries` rather than `fixed` keeps
  both numbers on the file as they were measured; a later edit of the
  default list that forgets the cost block fails a test that compares
  `entries` to `len(variables[variable])`.
- Schema: `per_entry` is `{variable: string, minutes: number ≥ 0, entries:
  integer ≥ 1}`, `additionalProperties: false`; `variable` must name a
  list-valued variable that some step's `for_each` reads (test).

The listing carries `cost` through unchanged (it already does), so
`per_entry` reaches `list_workflows` in both views with no projection
change. The skill and the guide replace the divide-by-default stopgap with
the formula; `tests/test_plugin_skills.py` pins that the H3 skill quotes
`per_entry`. Both templates need a measured run on lem for the numbers -
the step timings T021 put in the job's progress block are the source, so
one run per template gives every figure.

### Entry shape: derived from what the steps read

Nothing has to be declared. A `for_each` step reads its entry through
`item:` strings, so the fields an entry needs are exactly the set of
`item:<field>` references across every step over that variable, plus
`name`. `derive_catalog_metadata` gains a `lists` block:

```json
"lists": {
  "shots": {
    "fields": ["name", "num_frames", "prompt", "references"],
    "steps": ["shot"],
    "entries": 5
  }
}
```

- `fields`: sorted union of `item:` fields over every step whose `for_each`
  is `variable:<name>`, with `name` first. `item:` alone (the whole entry,
  used when entries are scalars) yields `fields: null` and the entry is a
  value, not an object.
- `steps`: the `for_each` steps over that list, so an agent knows
  `shot@<name>` and `slice@<name>` are what the manifest will show.
- `entries`: the default list's length, which is what the cost block's
  `entries` is checked against.

`lists` joins `COMPACT_FIELDS`: it is what authoring the argument needs,
and it is small. A workflow with no `for_each` has `lists: {}` and the
compact view drops the key when empty, so listings that never had it do
not grow.

The meaning of each field stays where meaning lives today - the
description and the template's default entries. `GET
/api/workflows/{name}/variables` is the call that shows those defaults,
and its 200-character preview only cuts *string* variables; a `shots` list
comes back whole, ten kilobytes of prompts for dialogue-short. Stage 3
extends the preview into list and dict values (`shots[0].prompt` named in
`truncated`), so an agent sees the shape of every entry at a glance and
asks for `full=true` when it wants the text.

### Validation: unknown entry keys

A missing field is already a `ForEachError` at the entry's path. An extra
key - `num_frame` for `num_frames` - is ignored in silence today, and the
member then runs with the template's value for the field the caller meant
to override. Stage 3 reports an entry key no step reads as a validation
*warning* (not an error: a caller may carry an annotation field on
purpose), at the entry's path, naming the fields the list takes. The
`lists.fields` derivation is the same computation, so this is one function
used twice.

### Decisions carried from stage 2, ruled here

- **An empty `for_each` list is a validation error**, not an empty success.
  A workflow that expands to no generating step is never what a caller
  asked for, and the run producing no manifest and no events looks like a
  server fault. `expand_for_each` raises `ForEachError` at the step's path
  with "for_each over an empty list would run no steps"; the schema's
  `minItems: 1` covers the literal form.
- **`expanded_definition` runs `realize_constants`** before substitution,
  the way `Workflow.run` does, so a list defaulted to a `constant:` name
  validates as it runs. It is a name lookup, no download, so the pre-flight
  stays free.
- **The step cache's bound has to clear a maximal run.** 32 entries over
  two groups plus fixed steps is ~70 members; `DEFAULT_MAX_ENTRIES = 50`
  (`dw/step_cache.py`) evicts a run's own earlier members before it ends,
  which silently defeats the insertion case the naming scheme exists for.
  Raised to 128 (`dw/step_cache.py`) - the proposal's original "leave it
  alone unless a real template needs more" no longer holds once the
  ceiling is 32.
- **The editor's flow view draws no edges for `gather:` or `item:`**
  (`ui/src/lib/flow.ts` maps only `previous_result:`), so both templates
  render their editor step as an orphan. An edge from each `for_each` step
  to the step that gathers it, and between paired groups over one list,
  is the whole fix; it is UI-only and can ship with or after the rest.

### Order of work

1. `lists` derivation + `COMPACT_FIELDS` + unknown-key warning (pure
   functions, unit-tested, no GPU).
2. `per_entry` schema, the two catalog tests (variable named exists; entries
   matches the default list), the variables-endpoint nested preview.
3. Skill and guide wording, pinned by test; MCP server instructions mention
   `lists` and `per_entry` in the `list_workflows` sentence.
4. Empty-list error, `realize_constants` in validation, cache bound.
5. Measure both templates on lem (one run each) and fill in `per_entry`.
6. Flow-view edges.

Steps 1-4 are self-contained and testable here; step 5 is the only one
that needs the GPU box, and the skill's formula degrades honestly without
it (no `per_entry` means "quote the total and say the list is the
default's").
