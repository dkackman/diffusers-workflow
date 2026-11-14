# Per-preset cost factor on a template's `cost` row (#1239)

## Problem
`scale_curated` (`dw/cost_scaling.py`) scales width/height/steps/frames in plain proportion,
with no fixed term for model load and decode. A preset tier that raises the canvas
(minimax `best`: 1344x768 vs 960x544) is therefore quoted at 1.98x `standard`
(dialogue-short 38.6 -> 76.2 min). The bench (#1218) puts `best` at 1.3-1.7x (1.32x on
dialogue-short, about 51 min). A user approving a `best` run sees about 50% too much.

## Options
1. **Per-preset factor on the `cost` row** (the issue's suggestion; a schema change).
   A `cost` entry gains e.g. `"preset_factors": {"tier": {"best": 1.32}}`, the measured
   ratio to the default tier. `_driver_scaling` uses it in place of the canvas axis
   when the run's preset matches, and names it in `scaled`. Model knowledge stays in the
   workflow, per CLAUDE.md. Cost: a new schema field, a rule authors must learn.
2. **Fixed-term scaling rule** (engine change): `minutes = fixed + variable * ratio`,
   with `fixed` declared per cost row. Fixes every canvas move, not only presets, but
   needs a measured `fixed` per template and changes every derived quote.
3. **Do nothing**: tier order is already right (#1236); only the magnitude is high.

## Recommendation
Option 1, limited to preset tiers. Default if approved: add `preset_factors` to the cost
schema, populate it for the minimax templates from the #1218 bench, with tests that a
`best` quote lands within 1.3-1.7x `standard`.
