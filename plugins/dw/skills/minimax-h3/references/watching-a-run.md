# Watching an H3 run

Part of the `minimax-h3` skill: what a healthy run looks like while it is
running, and the dead-air fix after it finishes.

## Silence is no hang

- `denoise_step` is null at first, through the encode and into the first
  pass, which takes the most time (the measured figures are in the skill).
- The block cache makes later steps uneven, so two-minute gaps are healthy.
  A slow first step at a new size is healthy too: autotune runs once per
  box.
- `phase_stall` in `get_job_events` describes the wait; it does not mean a
  fault. Judge progress by `denoise_step`.
- A cancelled H3 job runs on to its next step boundary before it stops.

## Dialogue gaps need room tone

A `shot_dead_air` finding means a gap in a voiced shot is digital silence.
Run `find_loop_bed` on the `output:` cut, then
`slice_audio`->`loop_audio`->`mix_audio` its pick at its `gain`.

## Held frames

Run `assess_output(name, probe="analyze_held_frames")` on every take. H3 shows
single repeated frames on a 4-frame spacing, so a few holds are normal. 13 or
more `held_frames`, or a cluster, means the take visibly stutters there: look
at the cluster's time with `get_output_frames`, and if it shows, reroll the
take or re-plan the shot (for example, less fast motion).
