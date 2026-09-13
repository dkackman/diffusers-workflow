# Pre-flight argument bounds

Status: proposed, 2026-09-14. Raised by #96. Written by the implementer
agent, model `opus` via provider `anthropic`.

## The problem

`validate_workflow` passed `num_frames: 61` on `templates/minimax/video-with-audio`
and reported `valid: true`, naming `num_frames` in `checked_arguments` - so the
answer claimed to cover the caller's value. The run then spent 138.7 s loading
the H3 weights and the turbo LoRA, entered the text encoder, and failed on:

```
MiniMax-H3 generates between 5.0 and 15.0 seconds at 24 fps, so `num_frames`,
rounded up to the next `17 * n + 5` the video VAE can encode, must be between
120 and 360, got 61 (rounded up to 73).
```

Every term in that message is a property of the model and its VAE. None of it
needed a loaded pipeline: the check is `align_num_frames(...)` against
`min_duration` / `max_duration` / `fps`, all of which are constants on
`MiniMaxH3ModularPipeline` until a VAE overrides them with the same values.

Two things follow, and the second is the one that makes this worth a proposal
rather than a patch:

1. A bound the engine can state exactly is only enforced after two minutes of
   GPU work.
2. **Nothing on the MCP surface states it at all.** `get_workflow(variables_only=true)`
   gives `num_frames: 124` with no range. The tester picked 61 as `4 * 15 + 1`
   because `4n + 1` is the common convention; H3's is `17n + 5` with a floor of
   120. A consumer cannot guess this and has nowhere to read it.

There is also a silent rounding: a legal value that is not of the form
`17n + 5` is rounded up, and the caller gets a frame count they did not ask
for with a `logger.warning` that reaches no consumer (the #82 rule - a
diagnostic that only reaches the log does not exist out there).

## Why this is a decision and not an edit

`CLAUDE.md` says it plainly: model knowledge lives in the composition skills
and the catalog, **never in engine code**, and every number a skill states is
pinned to a diffusers symbol by `tests/test_plugin_skills.py`. Any fix here
puts a model's frame-count rule *somewhere*, and the three candidate somewheres
have different consequences for that rule. Picking one is yours.

## Option A - declare the constraint in the workflow (recommended)

Precedent exists. A chain step already declares the same rule, because the
chain has to snap its final segment to a legal length:

```json
"frame_snap": {"modulus": 17, "remainder": 5, "min_frames": 124, "max_frames": 345}
```

That block is written in the workflow, by the author who knows the model - the
engine consumes it and holds no model knowledge of its own. Extend the same
shape to a declared variable, so it can be checked before anything is queued:

```json
"variables": {"num_frames": 124},
"variable_constraints": {
    "num_frames": {
        "minimum": 124,
        "maximum": 345,
        "modulus": 17,
        "remainder": 5,
        "snap": "up",
        "reason": "the video VAE encodes 17 * n + 5 frames, 5 to 15 seconds at 24 fps"
    }
}
```

- `validation_errors` gains a pass over it, after substitution and `for_each`
  expansion like the others, reporting at `arguments.<name>` where the
  caller's value sits and at `variables.<name>` for a stored default. So
  `POST /api/validate`, `validate_workflow` and the pre-queue check in
  `POST /api/jobs` all get it for free, in that order of usefulness.
- `snap: "up"` makes the rounding explicit and gives validation a *warning* to
  report when the value it would snap differs from the value passed - which is
  the adjacent finding in #96, and which the run-time path should emit as an
  `emit_warning` too so it reaches the job's `warnings`.
- The catalog reports it: `list_workflows` and `get_workflow(variables_only=true)`
  carry the constraint beside the default, so a consumer reads the rule instead
  of guessing it. This is the half that stops the next person picking 61.

Cost: every template that has a bound has to declare it, and a template that
declares none validates exactly as it does now. The numbers get the same
discipline the skills have - a test that pins each one to the diffusers symbol
it came from, so a library change fails in CI rather than in a cold session.

Risk: a declared constraint can go stale against the library. The pinning test
is what keeps it honest, and a *wrong* bound refuses a legal run - which is
why the check should refuse only on the declared rule and never invent one.

## Option B - a pre-flight registry in the engine, keyed by pipeline class

`dw/preflight.py` holds a per-pipeline check that imports the diffusers symbols
(cheap - no weights) and computes the bound: `align_num_frames`,
`MINIMAX_H3_FPS`, `min_duration.fget` / `max_duration.fget`. No template
markup, and it covers an inline workflow an agent wrote from scratch, which
Option A does not.

It also puts model knowledge in engine code, which is the thing `CLAUDE.md`
forbids - softened, but not removed, by every number being read from diffusers
rather than written here. It needs a new entry per family, forever, and the
"which pipeline is this" dispatch is a second place model identity is spelled.

## Option C - curated bounds in the catalog, beside `cost`

`cost` is already a maintainer's measured figure carried per workflow and
reported by `list_workflows` (#91). Bounds could ride there. Same authoring
cost as A with none of A's enforcement: the catalog is descriptive, and a
consumer-facing number nothing checks drifts. Useful as the *reporting* half
of A, not as a substitute.

## Recommendation

A, with the reporting from C folded into it: declare the constraint where the
`frame_snap` precedent already puts it, check it at validation time, warn on a
value that will be snapped, and report it beside the variable's default so it
can be read before it is violated. Mark the H3 templates first (the bound in
#96), then LTX-2.5's own frame rule, then whatever the next family brings in
through `model-family-onboarding`.

## Not in scope

Checking a value against a *loaded* pipeline's signature - that is what the
existing signature warnings do. This is only about bounds that are constants
of the model, which is the class of error that costs two minutes of GPU before
it is reported.
