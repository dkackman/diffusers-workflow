# Quality, speed and cost for long pieces

Part of the `long-form` skill. Always quote `plan.estimate` from
`validate_workflow`. These figures are here to explain a quote and to plan
a piece, not to replace the estimate.

## Measured costs (RTX 3090, 24 GB)

Unless a row says otherwise, these are H3 chains at the template defaults:
960x544, 5 steps with the 4-step turbo LoRA, int4, sparse attention.

| What | Wall time | Source |
| --- | --- | --- |
| `templates/minimax/chained-segments` cost row, guide, 3 x 124 frames | 13.5 min | the template's `cost` (guide seam A/B, C-F311, `docs/proposals/complete/h3-guides-complete.md`); last_frame measured 9.6 |
| 10-segment take (about 43 s), check on, `retries: 0` | 20.6 min (S1), 18.2 min (S2) | `bench/records/check.md` |
| 6-segment take, one segment re-rolled | 14.4 min | `bench/records/check.md` |
| 2-minute take, guide, no counter | 52 min | `bench/records/counters.md`, *Round 3* |
| 2-minute take, `shot: static` (anchor 0.25) | 69 min (+32%) | same |
| 2-minute take, last_frame + `look_hold` + anchor 0.25 (a knob) | 48 min | same |
| 2-minute take, `shot: static_long` (keystone slot `last`) | 57-62 min | `bench/records/structural.md`, *Keystone slot*; plain keystone 53 min |
| `templates/minimax/repair-segment`, one segment | 8.1 min | `bench/records/repair.md` |

- **Per output minute.** From the two-minute walls, that is about 26 min per
  minute for a guide chain and about 35 min for a static one. A short chain
  costs more per minute, because the model load and the first-pass stall
  are fixed costs.
- **The cost row is a 3-segment figure.** The catalog template itself, run
  for 2 minutes with `shot: static_long`, took 59.2 min
  (`bench/records/tiers.md`). Quote `plan.estimate` and scale by length from
  that.
- **Determinism is free.** `cudnn_benchmark: false` with
  `cudnn_deterministic: true` cost nothing measurable on H3, and makes
  same-seed runs on a two-card server repeat bit for bit
  (`bench/records/determinism.md`).

## The tier dial

The design names three tiers for every family:

- **draft**: approve blocking, timing, head count and line delivery. It is
  the cheapest path whose staging carries to the final.
- **standard**: the default for a deliverable, chosen quality-first.
- **best**: the highest measured quality, at any cost.

Defaults favour quality. A speedup with no marked loss becomes a default.

- **The `tier` argument** (measured, `bench/records/tiers.md`) is on
  `minimax/chained-segments`, the three `chain-*` templates,
  `dialogue-short` and `music-video`. `tier: draft` uses fewer steps on
  the same canvas and weights: 4 steps for a guide chain, 5 for the Ref2VA
  ones. It is 28-44% cheaper and keeps the standard take's staging at the
  same seed, so approve blocking and timing on it, never texture.
  `standard` is the default. `tier: best` is 1344x768 on int8 weights, the
  cleanest take measured, at 1.3-1.7x standard's wall time. It re-stages,
  needs about 65 GB of host RAM per job, and runs one at a time on a 128 GB
  box. On a static shot, `weights_dtype: "{int8}"` alone at standard is
  cleaner for 0-5% more time (2-minute `static_long`: seam look 1.50 to
  0.66). Offer it when the host runs one H3 job at a time; it is not the
  default. Any argument you pass still wins over the tier. LTX chains have no
  tier: their schedule is fixed, and the step up is `ltx2/upscale-clip`.
- **LTX-2.5 parity** (Task 23, `bench/records/ltx.md`): measured.
  `last_frame` stays the `templates/ltx2/chained-segments` default. Cap an
  LTX take at about 15-20 s (3-4 segments), then cut and start the next
  shot from a fresh still of the subject. `templates/ltx2/extend-clip` is a
  knob at about 2.1x the wall time, for a subject that stays in frame.
  `look_hold` on LTX wins on numbers, is invisible by eye and is safe on a
  moving camera; whether it becomes the default is Don's call.
- **Windowed refine seam pass**: measured. It wins as a seam pass on every
  shot type, at about 36-40 min per minute of take, and is not a burn fix
  (`bench/records/structural.md`, *Step 4*). It is experimental and
  flag-gated; do not offer it.

## What does not buy quality on a long take

- **More counters on a moving shot.** Every guide counter lost seam look on
  the arc (`bench/records/counters.md`).
- **Re-rolling texture drift.** The re-roll inherits the carry
  (`bench/records/check.md`).
- **A longer static chain on `shot: static`.** Cut at about 30 s, or use
  `shot: static_long`. A two-minute static take is unusable on every counter
  in `bench/records/counters.md` (*Round 3*); keystone slot `last` holds it
  for two minutes on both seeds, with the pose easing back every 4.25 s
  (`bench/records/structural.md`, *Keystone slot*; `levers.md`).

## Apple Silicon

`templates/minimax/chained-segments` at its defaults is refused on a 24 GB
Mac: about 18.3 GB is projected against a 17.8 GB working set. 768x448 passes
under guide continuity. `num_frames` cannot be lowered below 124
(`docs/ACCELERATION.md`). The bench ran on CUDA only. The design carries the
levers' ordering to MPS, but not the minutes.
