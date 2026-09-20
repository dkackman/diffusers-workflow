# Proposal: score and select - deterministic N→1 reduction between a fan-out and an expensive stage

Status: proposed, 2026-09-13. Raised by an external reviewer's forum reply and
filed as #119. Written by Don's session, model `opus` via provider `anthropic`.
No code changes yet.

## The problem

The README's own headline shape is *4 Flux stills → choose one → LTX video*.
Today the choice lives at the agent boundary: job A fans out, the agent looks
at the gallery, `keep_output`s the winner as an asset, and job B references
`asset:name`. That is deliberate and it is the right boundary when "best" is a
visual or subjective judgment - the agent is doing the one thing it is good at
and the engine does not need a control-flow language.

What the engine cannot express is the *deterministic* version of the same
pattern:

```
fan out N candidates
  → score each one cheaply
  → keep the argmax (or the first above a threshold, or index k)
  → spend the expensive stage on that one only
```

Nothing in the workflow language reduces a list to one member by a rule, and
nothing emits a number per candidate for a rule to read. So a workflow that
wants "generate four, upscale the sharpest" is two jobs and an agent in
between, even though no judgment is involved - and the rule that made the
choice is recorded nowhere, so the recipe does not replay.

The reviewer's split is the right one and this proposal keeps it:

| Kind of choice | Where it lives |
|---|---|
| Semantic / subjective ("which composition is best") | The agent, as today: `keep_output` + `asset:` |
| Deterministic N→1 (`argmax`, threshold, index) | **A reducer task inside the workflow - this proposal** |
| Conditional execution (skip steps, retry a batch) | Not proposed. See *Not in scope* |

## Why a reducer and not `when`

Every question that makes general branching hard is a question about a step
that *did not run*: what `previous_result:A` means when A was skipped, whether
a skipped step has a manifest or cache entry, what the cache key is after a
branch. A reducer has none of them. Every step runs; one of them returns a
shorter list. `previous_result:` never has to mean "maybe absent", the manifest
records every step as it does now, and the step cache is untouched.

That last point is where the feature pays for itself. The cache keys on
`(workflow id, step name)` under a seed, so on a seeded workflow the candidates
and their scores are cached after the first run. Change only the rule or the
threshold and the rerun regenerates nothing upstream - it re-selects and then
spends the expensive stage. The reviewer asked whether "changing only the
selector can reuse upstream cached candidates"; with a reducer that holds by
construction.

## What the engine already gives us, and the one thing it does not

The surrounding machinery exists:

- `for_each` expands a template step into named members (`still@a`,
  `still@b`, …) with their own cache lines.
- `gather:<group>` is the list of every member's result and **splices inside
  a list**, so `["gather:still"]` reaches a task as N artifacts in *one*
  iteration.
- Inside a member, a reference to another group over the same list resolves
  to the member with the same key - a `judge` group with the same `for_each`
  as `still` reads `previous_result:still` as `still@<its own key>`.
- Utility tasks are where non-pipeline transformations go, and a task's
  result flows downstream like any other.

The thing it does not give us: there is **no whole-list reference for a
single step's artifacts**. Every `previous_result:` reference is expanded by
the cartesian pass in `dw/previous_results.py` - one iteration per artifact,
even when the reference sits inside a list. A step that made four images with
`num_images_per_prompt: 4` cannot hand all four to one task call. So a
reducer over that step is impossible without a new reference kind.

This proposal therefore makes the fan-out **a `for_each` group**, which is the
better shape anyway: members are named, cached individually, and the winner
carries an entry name rather than an index that shifts when the list is
edited. A whole-list reference for a single step (`all:<step>`, say) is listed
under *Later* rather than designed here.

## Design

### 1. `select` - the reducer task

A registered task command (`dw/tasks/task.py`, `register_command`) with a
closed rule set:

```json
{
  "name": "pick",
  "task": {
    "command": "select",
    "arguments": {
      "candidates": ["gather:still"],
      "scores": ["gather:judge"],
      "rule": "argmax"
    }
  }
}
```

| `rule` | reads | returns |
|---|---|---|
| `argmax` / `argmin` | `scores` | the candidate at the best score |
| `first_above` / `first_below` | `scores`, `threshold` | the first candidate meeting the threshold, in list order; a validation error at run time if none does (see *No candidate passes*) |
| `index` | `index` (an int, or `previous_result:` naming a step that returned one) | that candidate |

Contract:

- `candidates` and `scores` are parallel lists of the same length; a mismatch
  is an error naming both lengths. `scores` are numbers; a score that is a
  string is parsed once (`"0.82"`) and otherwise refused - a judge that
  returns prose is a judge misconfigured, not something to fuzzy-match.
- Returns **one** artifact, so `previous_result:pick` downstream is a single
  iteration. No cartesian expansion.
- The step's `step_end` event and manifest entry carry `selected` - the
  winning position, the winning member's entry name when the candidates came
  from a `for_each` group (`still@b`), and the score - so the choice is
  replayable from the manifest and visible on the job page. The realized
  `workflow.json` already pins `output:.../latest` for the same reason; the
  winner is the same kind of fact.
- Ties go to the first in list order, stated in the docs.

Ten-ish lines of reduction; the work is the contract and the recording.

### 2. `judge` - the first scorer

`select` reduces over nothing without a number per candidate. The cheapest
first scorer reuses what `image_to_text` already loads - a vision-language
model - and asks it for a number against a rubric:

```json
{
  "name": "judge",
  "for_each": "variable:candidates",
  "task": {
    "command": "judge",
    "arguments": {
      "image": "previous_result:still",
      "rubric": "variable:rubric",
      "scale": [0, 10]
    }
  }
}
```

- Same `for_each` as `still`, so `previous_result:still` inside `judge@a`
  is `still@a` by the existing same-list rule. `gather:judge` then pairs with
  `gather:still` by position.
- The task prompts the VLM with the rubric and the scale, parses the reply
  to one number, and returns that number as its artifact. A reply that does
  not parse is an error naming the step and the raw reply - not a zero, which
  would silently lose.
- `model_name` defaults to `image_to_text`'s default and is overridable the
  same way. The model is pinned per family in the catalog's cost entry like
  any other.
- A VLM judge is *deterministic enough* for a seeded workflow only if the
  model's generation is seeded too; the task threads the step seed through.

A second scorer - an aesthetic predictor or CLIP score against the prompt -
is the natural follow-on and needs no change to `select`. It is not in this
proposal's first stage because the judge covers the rubric-shaped cases
("sharpest", "closest to the prompt", "no text artifacts") with a model the
engine already carries.

### 3. One catalog template

`templates/flux/best-of-n-to-video` (name to taste): `still` fanned over a
`candidates` list of seeds, `judge` over the same list, `pick`, then the
LTX-2.5 image-to-video template as a `builtin:` sub-workflow on
`previous_result:pick`. Marks `still`/`judge` `intermediate` and the video
`final` per the subfolder rule. This gives the plugin skills something to name
and pins the shape in `tests/test_plugin_skills.py` the way every other
template's numbers are.

### 4. Validation

- `select` is a task command, so `validate_workflow` already checks its
  argument names against the implementation's signature. Add: `rule` is in
  the closed set; `threshold` present iff the rule is a threshold rule;
  `index` present iff `rule: index`.
- `candidates` and `scores` must both be `gather:` references (or both plain
  lists) - two different groups is allowed, two different *lengths* is caught
  at run time only, since list length is a run-time fact.

### No candidate passes

`first_above` with nothing above the threshold is the one place this design
touches the conditional-execution question, and the answer is: it is an
error. The run fails at `pick`, the manifest holds every candidate and every
score, and the agent - which is the search loop today - decides whether to
draw a new seed (`rerun_job(new_seed=True)`) or lower the bar. That keeps
"spend more GPU time on another batch" where the reviewer put it: a search
policy, not failure recovery, and the agent's call.

## Testing

- `select`: each rule on a fixed candidate/score list; the tie rule; length
  mismatch; a non-numeric score; `index` from a `previous_result:`.
- `judge`: the parse path with a stubbed model reply - a clean number, a
  number in prose, a reply with no number (error, raw reply in the message).
- Manifest: a run of the template writes `selected` on `pick` with the
  member name, and `get_job` / the job page show it.
- Cache: on a seeded run, changing only `rule` reuses `still@*` and `judge@*`
  (`reused: true`) and regenerates only `pick` and the video.
- Plugin: the template's numbers pinned like the others.

## Not in scope

- `when` / conditional steps / skipped-step semantics. Nothing in the catalog
  needs them, and the reviewer's analysis of what they cost is the argument
  for not paying it yet.
- Retry of any kind. A transient-failure retry, if it ever comes, is a
  `JobManager` concern, not a workflow-language one.
- A whole-list reference for a single step's artifacts (`all:<step>`). Real,
  small, and separable - but the `for_each` shape is better for this use and
  the new reference kind would need its own reference-checker and cache
  treatment. Listed under *Later*.
- Semantic selection. Stays with the agent.

## Later

- `all:<step>` so a `num_images_per_prompt` fan-out can feed `select`
  without a `for_each`.
- A second scorer (aesthetic / CLIP).
- `select` returning the top *k* rather than one (`rule: top_k`) - a
  shortlist for the agent to choose from is a plausible hybrid of the two
  boundaries, and it is the same reducer with a different return shape.

## Open question for whoever picks this up

Whether `judge` should be a new command or a mode of `image_to_text`
(`"parse": "number"`). A separate command keeps the contract obvious in the
catalog listing and the guides; a mode avoids a second model-loading path.
Leaning towards a separate command that shares the loader.
