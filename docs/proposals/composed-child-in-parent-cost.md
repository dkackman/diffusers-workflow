# A composed step already inside the parent's cost row (#1773)

Status: proposed, awaiting approval.

## Problem

A `cost` entry is measured at the workflow's variable defaults, a whole run.
`templates/minimax/dialogue-short`, `storyboard` and `music-video` had their
rows set while they drew portraits and boards with inline Z-Image Turbo steps.
`268c931e` (#1722) turned those steps into composed `portrait_workflow` steps
and left the rows alone. The estimate adds every composed child to the parent's
own figure, so the drawing is now counted twice.

CUDA quotes with no run history:

| Template, defaults | Before #1722 | Now |
|---|---|---|
| dialogue-short | 21.2 | 24.4 |
| storyboard | 10.1 | 14.3 |
| music-video | 35 | 36.6 |

The same defaults also report `differs: ["num_images_per_prompt"]` and
`low_confidence: true`, from pricing the child for the step's arguments.

## Proposal

A `cost` entry may name the composed steps whose time its `minutes` already
holds, by the child that was measured:

```json
"cost": [{"device": "cuda", "name": "RTX 3090", "minutes": 21.2,
          "includes": {"models/z-image": ["draw_character_a", "draw_character_b"]}}]
```

- A named step that resolves to that child adds nothing: no minutes, no
  `scaled` clause, no `differs`.
- A named step that resolves to any other child (an override, or a different
  `preference:`) is priced as today. The quote is then high by the covered
  child's share, which the entry does not state.
- A step not named is priced as today.
- The covered time scales with the parent's drivers, as it did before #1722.
- A catalog test refuses a name that is not a composing step of the workflow.

The three templates declare it. Their defaults quote 21.2, 10.1 and 35 again,
and the #1773 calls quote 7.8 (12 steps, one 175-frame shot) and 10.0 (9 steps,
175 + 124 frames).

Touches `dw/workflow_schema.json`, `dw/plan.py` / `dw/plan_pricing.py`, the
three templates, `docs/WORKFLOW_GUIDE.md`. No new module.

## Alternative: restate the rows to own work only

Data only: subtract today's `models/z-image` figure per composed step (about
1.58 min each at 768x768; dialogue-short 21.2 becomes about 18.0).

- `list_workflows` would show less than a whole run costs.
- The subtracted share is derived, not measured.
- Observed history times whole runs, so it would be tempered toward a figure
  that covers only part of one.
- The #1773 calls would quote about 8.2 and 10.1 (one portrait drawn).
