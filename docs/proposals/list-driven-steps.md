# Proposal: list-driven steps (`for_each`)

Status: **needs approval**. Written for MCP feedback ticket T003; nothing here
is implemented.

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
- `gather:shot` is the list of every member's artifacts, in order, as one
  value. It is a *list-valued* reference, which is the distinction
  `previous_result:` cannot make.

### Recommendation: expansion is a source transform, not a runtime concept

Run the expansion as a pass over the definition immediately after
`replace_variables` in `Workflow.run` (`dw/workflow.py`), before the run id is
computed and before the step loop starts. The pass:

1. replaces each `for_each` step with N ordinary steps, named `shot[wide_open]`
   … (below);
2. substitutes `item:` references inside each copy with that entry's values;
3. rewrites every `gather:shot` into the explicit list of
   `["previous_result:shot[wide_open]", "previous_result:shot[closeup]"]` -
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
  gets by hand. `release_pipeline` on a fanned step is the exception: it would
  drop the model after the first member and reload it for the second. Refuse
  it on a `for_each` step, or honour it only on the last member.

### Expanded names: the entry's own name before its index

`shot[wide_open]` when the entry is an object carrying a `name`, else
`shot[0]`, `shot[1]`.

The index alone is tempting and wrong for the cache. Entries are keyed
`(workflow id, step name)` in `dw/step_cache.py`, so inserting a shot in the
middle shifts every later index onto a different entry's content. The cache
validates against the step's data as well, so the result is correct - it is a
miss - but every shot after an insertion regenerates, which on a six-shot H3
episode is most of an hour of GPU to add one shot in the middle. A name
carried by the entry survives insertion, and it also makes the manifest, the
event stream and the gallery read in shot names rather than in ordinals.

Brackets rather than dots or underscores: `.` is already the property
separator in a reference (`previous_result:segment.mask`) and `_` collides
with a hand-written `shot_1`. A step name is otherwise an unrestricted string,
so the bracket form has to be reserved - a hand-written step named `shot[0]`
becomes an error.

### Validation

The static reference check shipped for T005
(`previous_result_reference_errors`) runs on the unexpanded definition, where
`shot[wide_open]` does not exist yet. `validate_workflow` should run the same
expansion first, using the variables' declared defaults, and check the
expanded steps - which also means an empty or absent default list validates
as zero steps and a `gather:` naming nothing. Order becomes: schema ->
substitute -> expand -> reference check. Schema validation stays where it is,
against the template step, and the schema gains `for_each` on a step plus the
`item:`/`gather:` prefixes wherever a reference is allowed.

### The realized workflow and the manifest

`dw/realize.py` folds the run's arguments into the variable defaults, so a
realized list-driven workflow carries the actual `shots` list and reproduces
the same expansion deterministically. **Keep `for_each` in the realized file**
rather than writing the expanded steps: it reproduces the run exactly and stays
legible. The manifest names the expanded steps (`shot[wide_open]`), because the
manifest records what ran. That the two name steps differently is deliberate
and should be documented where the manifest is.

### Limits and cost

Each entry is a full generation, so the expansion needs a ceiling of its own -
something like 64 steps, refused outright above it, well below
`MAX_ITERATIONS = 10000`, which guards a different thing. The larger
consequence is that a list-driven template's cost is set by the caller:
`list_workflows` quotes a per-workflow figure today, and a six-entry `shots`
list is six times the shot cost. The catalog entry should carry the per-entry
cost and the variable it multiplies by, and the MCP guidance
(`docs/WORKFLOW_GUIDE.md`, "Authoring a workflow from an agent") should say to
quote `cost x len(list)` before running.

### What this does not do

It does not give the engine a "zip". `for_each` runs over **one** list; a
pairing - shot *i* with its audio slice *i* - is expressed by putting both in
the same entry:

```json
{ "name": "wide_open", "prompt": "...", "start_frame": 0 }
```

That is the same answer the guide already gives for the cartesian rule
("write it as one step per pair"), and it is why the entry should be an object
rather than a string in any template that has more than a prompt to vary. Two
`for_each` lists on one step, or a `for_each` over a cartesian product, is out
of scope and probably always should be.

## Phasing

1. **Expansion pass + `item:` + `gather:`**, schema, validation ordering, the
   64-step ceiling, and the reserved bracket name. No template changes. This is
   self-contained and testable without a GPU: the pass is a pure function from
   definition to definition, so its tests are ordinary unit tests.
2. **`music-video` and `dialogue-short` rewritten** onto a `shots` list. Both
   are breaking for scripted callers - the per-shot variables
   (`shot_2_deflect`, `shot_3_react`, …) become entries in one list - and
   T005 has already broken those names once this week, so the two should land
   together or the second should wait for a deliberate version bump of the
   templates.
3. **Cost reporting** for a list-driven entry, in the catalog and in the
   plugin skills that quote it.

Stage 1 is the bulk of the work and carries the risk; stages 2 and 3 are
mechanical once it exists.

## Open questions for the approval

- **Is the loop index needed inside an entry?** Nothing in the two templates
  needs "shot 3 of 6" in a prompt, so v1 exposes no index. Adding `item:@index`
  later is compatible; making it available now costs a second special name.
- **Should `for_each` accept a number** (`"for_each": 4`) for the case where
  only the count varies? It reads well and it is one line in the pass, but it
  gives entries no names, which is exactly the cache problem above.
- **`dialogue-short`'s two-speaker structure** (which character speaks in which
  shot, T010's voice references) becomes a field on each entry
  (`"speaker": "a"`). That is a template question, not an engine one, but it is
  the thing stage 2 has to get right.
