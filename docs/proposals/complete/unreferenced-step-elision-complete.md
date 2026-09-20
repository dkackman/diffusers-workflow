# Proposal: a step nothing references does not run

Status: awaiting approval. Raised by #109 (tester agent, model `opus` via
provider `anthropic`); written by the implementer agent, same model and
provider.

## The case that raised it

`templates/minimax/dialogue-short` draws its two characters with Z-Image and
then references those portraits from every shot. An episode can just as well
be cast from portraits that already exist - a shot entry's subject reference
takes `from_file: "asset:cast/priya.jpg"` exactly as its voice references do,
and it works today. What does not work is the consequence: the two
`draw_character_a` / `draw_character_b` steps still run, and their output is
discarded. Job `48000580aec1` spent roughly 55 seconds and two model loads on
portraits nothing in the run looked at.

The recurring cast is the headline use of this template. Paying for it every
episode is the wrong default, and no argument the caller can pass avoids it.

The documentation half of #109 is shipped: the template's description and the
`dw:minimax-h3` skill now say a shot reference may be a file, and say plainly
that the draw steps still run. This proposal is the part that needs a
decision.

## What is being proposed

At run time, before the first step executes, drop any step whose result no
later step references and which saves nothing.

A step is *referenced* when a later step (after `for_each` expansion and
variable substitution) names it through any of:

- `previous_result:<step>` or `from_previous_result: <step>`
- `gather:<step>`
- a `pipeline_reference` naming its pipeline
- `shared_components` / `reused_components` keyed on it

A step is *kept regardless* when:

- it declares a `result` with `save` not false - it is a deliverable, and a
  workflow whose whole point is writing three images references nothing
- it is the last step
- it carries `release_pipeline` / `release_models` - dropping it would leak
  the memory it was there to free. (Better: move the release onto the step
  that now runs in its place, the way `expand_for_each` already moves it onto
  the last member. That is the more useful rule but the more delicate one.)

Elision is transitive: dropping a step can make the step it referenced
unreferenced in turn, so it iterates to a fixed point.

## Why it is a decision rather than an edit

1. **It is a new engine property every workflow inherits.** A step that runs
   today and stops running tomorrow is a behaviour change across the whole
   catalog, not one template. The `save`-not-false carve-out is what keeps
   that from being destructive, and it is exactly the kind of rule that is
   right in the common case and surprising in some particular one.
2. **It interacts with the step cache and with `plan`.** `plan.estimate`,
   `plan.steps` and the cost acknowledgement all count steps; eliding changes
   the number a caller acknowledged. The plan would have to be computed after
   elision, which means `POST /api/validate` has to do the elision too.
3. **It is observable in the manifest.** A run that used to write
   `intermediate/…draw_character_a.0-0.0.jpg` stops writing it. Anything
   built on an `output:` reference to a now-elided step breaks.
4. **Silent elision is its own trap.** If a reference is misspelled, the step
   feeding it becomes unreferenced and quietly vanishes, and the failure moves
   from "previous result not found" to "the picture is wrong". The static
   `previous_result` check (`dw/previous_results.py`) already refuses an
   unresolvable reference, which contains this - but only for references the
   definition spells literally.

## The cheaper alternative, for comparison

Template-local, no engine change: give `dialogue-short` the variables
`character_a_portrait` / `character_b_portrait`, defaulting to null, and have
each shot's subject reference read `from_file: "variable:character_a_portrait"`
the way the voices already read `variable:character_a_voice`. A reference
whose file is null is dropped from the list, so the default run is unchanged.

This is the idiom a reader of the variable list would expect to find, and it
is what #109's option (2) asks for. It does **not** save the portrait cost on
its own: the draw steps still run, and the shots would then carry both a
`from_previous_result` entry and a `from_file` entry for the same subject,
one of which has to drop. There is no mechanism for dropping the
`from_previous_result` one - the null-file rule only covers `from_file`. So
option (2) delivers a better-shaped argument surface and no saving, unless it
is paired either with this proposal or with a narrower "a reference whose
`from_previous_result` names an elided step is dropped" rule.

That is the trade to decide: the general engine property, the narrow template
change, or both.

## Recommendation

Take the general rule, with these guardrails:

- keep any step that saves, is last, or releases
- report every elided step as a run warning naming it and why, so a
  misspelled reference shows up as "draw_character_a was skipped: nothing
  references it" rather than as a silently different picture
- compute `plan` after elision, so the quoted cost is the cost
- record the elided steps in the manifest, so a run says what it did not do

Without the warning this is a trap; with it, it is the property the tester
asked for and a real saving on every recurring-cast episode.

## Not in scope

Conditional execution (`when:`) is a different feature and a much larger one -
#118 has just closed the step object specifically so an invented `when` is a
hard error rather than a silent no-op. Elision is static: it depends only on
what the realized workflow references, never on a value produced at run time.
