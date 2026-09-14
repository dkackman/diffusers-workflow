# A declared bound that reaches a list entry

Status: proposed, 2026-09-14. Raised by #145, which revisits limit 2 of
`preflight-argument-bounds.md` (accepted, dkackman, 2026-09-14). Written by
the implementer agent, model `opus` via provider `anthropic`.

## The problem

`variable_constraints` (#96) refuses `num_frames: 61` on
`templates/minimax/music-video`, where the frame count is a top-level
variable. The same 61 is accepted, quoted at 8.4 minutes and queued on
`templates/minimax/dialogue-short`, where the frame count is a field of a
`shots` entry:

```
validate_workflow("templates/minimax/dialogue-short",
                  arguments={"shots": [{"name": "probe", "num_frames": 61, ...}]})
-> {"valid": true, "errors": [], "warnings": [],
    "plan": {"estimate": {"minutes": 8.4, "basis": "derived"}}}
```

61 is the value `validate_workflow`'s own tool description cites as the thing
#96 stopped happening ("61 used to validate and then fail 138 s into the run,
after the weights were loaded"). It still does - it just has to be spelled as
a list entry. Three consequences, all in the ticket:

1. The refusal is missing. The run loads H3, enters the text encoder and
   fails on the pipeline's own check, which is the two minutes of GPU the
   feature exists to not spend.
2. The **snap warning** is missing. `shots[1].num_frames: 130` validates
   `warnings: []` and then generates 141 frames. As a top-level variable
   that comes back as a notice naming 141.
3. The **rule is unreadable**. `get_workflow(..., variables_only=true)` and
   the compact `list_workflows` entry report no `constraints` block for
   `dialogue-short` at all, while `lists` tells the caller that a `shots`
   entry carries `num_frames`. So the discovery surface names the field and
   says nothing about what it may hold - the exact gap #96 named as the half
   that stops the next consumer picking 61.

This was accepted knowingly: limit 2 records that `dialogue-short` "cannot
declare the rule per entry, and relies on the run-time check as before."
What the limit did not weigh is that `dialogue-short` is not a corner - it is
the only H3 template a caller composes a multi-shot deliverable with, the one
place per-shot length is *meant* to vary, and therefore the place a frame
count is most likely to be typed by hand. Today it is also the only
list-driven workflow in the catalog whose entries carry a constrained field
(`music-video`'s entries carry `prompt` and `start_frame` only), so the whole
exposure is one field of one template - and so is the whole cost of closing it.

## Why this is a decision and not an edit

Whatever closes this changes what a `variable_constraints` key *means*, which
is the part a consumer reads and a template author writes. It also reopens a
limit you accepted yesterday. Picking the shape is yours; the mechanics are
small either way.

## Option A - the key stays a name, matched wherever that name is declared (recommended)

A constraint is keyed by a plain variable name today. Keep that, and match the
name in both places a value by that name can sit: a top-level variable, and a
key of the same name in an entry of a list-valued variable.

```json
"variable_constraints": {
    "num_frames": {"modulus": 17, "remainder": 5, "min_frames": 124,
                   "max_frames": 345, "snap": "up", "reason": "..."}
}
```

`dialogue-short` declares exactly that block - the same six lines
`music-video` already carries, no new syntax at all - and its per-entry
counts are checked.

Mechanics, all in `dw/variable_constraints.py` plus reporting:

- `constraint_errors` / `constraint_warnings` additionally walk each
  list-valued variable whose entries are objects, and check any entry key
  that matches a declared constraint name. The path is where the caller
  wrote it: `arguments.shots[0].num_frames`, or
  `variables.shots[0].num_frames` for a stored default. Every existing
  caller - `validation_errors`, `POST /api/validate`, the pre-queue check in
  `POST /api/jobs`, `validate_workflow` - gets it with no change.
- `apply_constraints` already runs on the run's `variables` dict *before*
  substitution (`_prepare_definition`), so a `shots` list is sitting right
  there: it rounds an entry field in place and `emit_warning`s the same
  notice, and raises for one no rule can reach. The rounded value then
  propagates through `item:` like any other.
- The catalog reports it twice: the existing `constraints` block (which is
  keyed by name, so it needs nothing) plus the field's entry in `lists`, so
  a caller reading what a `shots` entry carries reads the rule beside the
  field name rather than having to cross-reference.

What it costs: a name means one rule for the whole workflow. A template that
wanted `num_frames` bounded one way as a variable and another way inside an
entry cannot say so. No template wants that - a bound is a property of the
model the value is handed to, and both spellings are handed to the same
pipeline - and a template that did want it should use two names.

The honest risk: an entry field that happens to share a constrained name but
feeds something else would be checked against a rule that does not apply to
it. In a file the same author writes end to end, with the only live case
being `num_frames -> num_frames`, that is a naming mistake the constraint
would usefully surface rather than a trap. `tests/test_variable_constraints.py`
already sweeps the catalog; the sweep would gain "every entry field a
constraint matches feeds the pipeline argument that constraint is about."

## Option B - a path-shaped key

```json
"variable_constraints": {"shots[].num_frames": {...}}
```

Explicit, and it answers the collision A accepts. It is also the second
dialect you ruled out: a key that is sometimes a name and sometimes a path,
a `constraints` block a consumer can no longer index by variable name, and a
grammar (`[]`, nesting, how deep) to define and validate. For one field of one
template.

## Option C - report the limit instead of closing it

Leave enforcement alone and make the discovery surface say the bound does not
reach entry fields - e.g. `dialogue-short` declares the constraint for
documentation and the catalog marks it advisory. This makes the rule readable
(consequence 3) and leaves 1 and 2 exactly as they are: a documented bound
nothing checks is the drift `preflight-argument-bounds.md` rejected Option C
for. Worth naming only to reject.

## Option D - hoist `num_frames` to a top-level variable

`dialogue-short` drops the per-entry field and bounds one workflow-wide count.
No engine change at all. It also deletes the feature the template is written
around - the tag entry runs 141 frames where the others run 124, because
length is per-shot in a cut sequence - so the bound would be enforced by
removing the thing being bounded.

## Recommendation

A. It declares nothing new, it is one block added to one template, and it
makes the constraint follow the value rather than the spelling - which is what
a caller reading `lists` already assumes. If the collision risk reads worse to
you than the dialect does, B does the same job for more grammar.

## Not in scope

- A constraint on a field of a *nested* structure that is not a list entry
  (`references[].something`). Nothing in the catalog has one.
- Limits 1 and 3 of `preflight-argument-bounds.md` (a constraint that depends
  on another variable; the run-time check as backstop) stand unchanged.

*Implementer agent, model `opus` via provider `anthropic`.*
