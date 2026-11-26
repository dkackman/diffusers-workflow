# Watching an H3 run

Part of the `minimax-h3` skill: what a healthy run looks like while it is
running, and the dead-air fix after it finishes.

## Silence is no hang

- `denoise_step` is null at first, through the encode and into the first
  pass, which takes the most time (the measured figures are in the skill).
- The block cache makes later steps uneven, so two-minute gaps are healthy.
  A slow first step at a new shape is healthy too: Triton autotunes once per
  shape per box (about 6 min on a 3090). `dialogue-short`'s quote already
  includes it: its `scaled` names `first pass at N shape(s) this card hasn't
  run` when the size, `weights_dtype`, `attention_backend` or a shot's frame
  count is new here, or the `triton` cache was cleared since it ran.
- `phase_stall` in `get_job_events` describes the wait; it does not mean a
  fault. Judge progress by `denoise_step`.
- A cancelled H3 job runs on to its next step boundary before it stops. If
  none comes within 300 s (30 s in a load or task), its worker is killed and
  the next job reloads the models.

## Dialogue gaps need room tone

A `shot_dead_air` finding means a gap in a voiced shot is digital silence.
Run `find_loop_bed` on the `output:` cut, then
`slice_audio`->`loop_audio`->`mix_audio` its pick at its `gain`.
A `shot_bed_drop` finding means a voiced shot's floor sits more than 20 dB
under a neighbour's bed (a storm, a crowd): run `find_loop_bed` with its
`max_bin_dbfs`/`max_mean_dbfs`/`target_bed_dbfs` raised to find the loud bed,
then rerun `assemble-and-score` with the pick's `start_seconds`,
`duration_seconds` and `gain` as `bed_start_seconds`, `bed_seconds` and
`bed_gain`, unless the cut is a deliberate scene change.

## Held frames

Run `assess_output(name, probe="analyze_held_frames")` on every take. H3 shows
single repeated frames on a 4-frame spacing, so a few holds are normal. 13 or
more `held_frames`, or a cluster, means the take visibly stutters there: look
at the cluster's time with `get_output_frames`, and if it shows, repair it:
`interpolate_frames` with `{"video": "output:<take>", "refill": "held"}`
replaces the held frames with interpolated ones, keeping the length, fps and
soundtrack. Its `log` event gives `held_before` and `held_after`; re-assess
the result. Reroll the take or re-plan the shot (for example, less fast
motion) instead when the holds sit against a cut or at the clip's end, which
refill leaves held, or when the motion itself is wrong, not just stuttering.
A long hold in slow motion can survive a refill and still count in
`held_after`, because the motion it lost is under ε per interpolated step;
when `held_after` is still 13 or more, reroll.
