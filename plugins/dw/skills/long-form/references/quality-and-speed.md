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
| `templates/minimax/repair-segment`, one segment | 8.1 min | `bench/records/repair.md` |

- **Per output minute.** From the two-minute walls, that is about 26 min per
  minute for a guide chain and about 35 min for a static one. A short chain
  costs more per minute, because the model load and the first-pass stall
  are fixed costs.
- **The cost row and the bench disagree.** The cost row is an interim figure
  (ruling R34) that predates the bench's settings. Task 20 re-measures it.
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

- **Speed levers and tiers** (Tasks 19-20): *pending measurement*. Which
  settings each tier maps to, and what each step down saves per minute, are
  not measured yet. Until they are, run the template defaults and quote
  `plan.estimate`.
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
- **A longer static chain.** Cut at about 30 s. A two-minute static take is
  unusable on every counter in `bench/records/counters.md` (*Round 3*). The
  exception under review is `keystone: segment_1`: it resets the burn at two
  minutes on both seeds, at no wall cost, with a visible mid-shot pop
  (`bench/records/structural.md`, *Keystone confirmation*). Until Don rules
  on it, cut at about 30 s (`levers.md`, *Under review*).

## Apple Silicon

`templates/minimax/chained-segments` at its defaults is refused on a 24 GB
Mac: about 18.3 GB is projected against a 17.8 GB working set. 768x448 passes
under guide continuity. `num_frames` cannot be lowered below 124
(`docs/ACCELERATION.md`). The bench ran on CUDA only. The design carries the
levers' ordering to MPS, but not the minutes.
