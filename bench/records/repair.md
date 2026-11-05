# Repair in place and the capstone gate (Task 14 Steps 3-4)

2026-11-05. Bench server lem:8766 at `082e93ce`. Arms: `bench/arms/repair.json`;
record: `repair.json`.

## Step 3: repair segment 4 of the check bench's S3 take

The take is Task 13's `chk-s3-seg4-broken` (`fdea6eba4a33`, `check.md`), run
with `keep_segments`. Its segment 4 is broken - a cut to a night harbour 26
frames in - though the check passed the re-roll, so the shot carries no
`check_failed`; it was repaired as the segment the eye flags.
`templates/minimax/repair-segment` with `chain_take` the take, `segment: 4`,
segment 4's own S3 prompt (the street, the smile, its line), seed 4 and the
template's defaults, which are the chain's settings (960x544, 5 steps, int4,
turbo LoRA, sparse attention). Job `46d9c4f4b1d7`, 8.1 min on cuda:0:
`render 175 frames, head guide 22 at 0, tail guide 39 at 136; take frames
328..441 of 634 replaced`.

### Seams

`analyze_seams` on the take before and after. The repaired shots are recorded
as chain seams (`segment 4 (repaired)`, trim 22 and 12, `crossfade_ms` 3), so
the probe measures them like the chain's own.

| seam | before: jump / look / level dB | after: jump / look / level dB |
|---|---|---|
| 1 (seg 1-2) | 1.75 / 1.57 / 8.81 | 1.74 / 1.51 / 8.81 |
| 2 (seg 2-3) | 1.81 / 1.54 / 1.50 | 1.79 / 1.49 / 1.50 |
| **3 (seg 3 - repaired 4)** | 0.42 / 1.36 / 2.89 | **1.57 / 1.11 / 0.79** |
| **4 (repaired 4 - seg 5)** | 0.13 / 0.59 / 4.43 | **0.29 / 1.27 / 0.35** |
| 5 (seg 5-6) | 1.94 / 2.58 / 0.66 | 1.94 / 2.57 / 0.12 |

The chain's other seams (1, 2, 5) average jump 1.82, look 1.86, level 3.48 dB.
Against them, under `combined_floor()` at the S3 guide row (jump 0.332, look
0.438, level 6.204 dB):

- seam 3: jump 1.57 (tie, inside 0.332), look 1.11 (better by 0.75), level
  0.79 dB (tie);
- seam 4: jump 0.29 (better by 1.53), look 1.27 (better by 0.59), level
  0.35 dB (tie). Its frame step is 1.54 grey levels against 4.4-8.0 at the
  chain's seams: the tail guide is segment 5's own frames, so the rejoin is
  nearly invisible.

Both repaired seams are as good as or better than the chain's own. The
"before" seams 3 and 4 read low only because the harbour's fast pan raised
segment 4's typical step to 20; they were not good seams. Nothing outside
the span moved: seams 1, 2 and 5 are unchanged apart from encode noise
(level step at seam 5 0.66 -> 0.12 dB, inside the floor).

### By eye: the repair cannot undo a break that has already propagated

Take frames 328-~400 are now the street, the woman smiling and walking, as
asked. From ~404 the render dissolves to the harbour, and the take stays on
it until segment 5's own cut back to the street at 449. The cause is the
geometry, not the render: the tail guide is segment 5's frames 12-50 (take
442-480), and segment 5 opened on the harbour, because its guide was broken
segment 4's tail; its first 19 kept frames are harbour. The repair is held to
end on that guide, so it has to reach the harbour. `bench/cuts.py` on the
repaired take: segment 4 14.5 (at 404), segment 5 17.0 (at 449).

So a broken segment in a guide chain breaks the next segment's opening too,
and repairing segment k alone then cannot clean the take. Two consequences:

- The check has to stop the break at segment k, before its tail becomes a
  carry - which is what `cut_ratio` (`d37c3192`, `check.md`) adds.
- A repair of a break that reached segment k+1 needs segments k and k+1
  re-rendered together, between k-1 and k+2 (22 + 204 + 39 = 265 frames
  plus the grid overrun, 11-12 s, inside H3's 345-frame ceiling, so one
  render - but the template takes one segment), or the take re-run from
  segment k. Not built.

### Peak VRAM

`nvidia-smi` sampled every 0.5 s (`~/dw-bench-t13/vram.csv` on lem). The
repair ran alone on cuda:0, starting from an emptied card (637 MiB after the
worker's full cleanup).

| phase | peak (nvidia-smi) |
|---|---|
| load (SDNQ quantizes on cuda) 09:29:30-09:31:20 | 19,377 MiB = 18.92 GiB |
| render, 175 frames + 61 guide frames, 960x544 | 15,319 MiB = 14.96 GiB |
| for reference, the S3 chain's segments (124 + 22) | 13,479 MiB = 13.16 GiB |

The template's estimate is `16.3 + 42.84 x 960 x 544 x 175 / 2^30` = 19.95 GiB.
It holds: 1.0 GiB over the whole job's peak (the load transient, which the
base term carries for every H3 template) and 5.0 GiB over the render's. The
derived voxel rate is borne out at the margin: the repair's render peak sits
1.80 GiB over a chain segment's for 90 more frames of rows on the same
canvas, 41 bytes a voxel against the 42.84 the template folds the guides
into. No change to the estimate.

## Step 4: the capstone gate

**The gate is open.** Plan: spike #1187's late capstone "if Task 11 left S2
(moving camera) without a winning counter". It did, and the R35 re-score kept
it (`counters.md`: S2 "none - every guide counter loses seam look; both
last_frame counters freeze the arc"; weakened, not flipped). The S2 failure
the capstone targets is real by eye (the sky blows out to white and the scene
invents palms by the last segment), and the check bench shows it as texture
(segments 7-10 fall to 0.48-0.32 of segment 1's fine band, `check.md`).

**No existing condition-row feature expresses it.** A guide must start on the
17-frame grid and end by `num_frames` (`guide_end_problem`), and every
condition row is in the forward from the first step; the keystone replaces the
keyframe; the latent anchor matches statistics after the loop. A capstone
needs a row past the render's end and a step gate.

**The smallest engine change, built as a spike** (commit `45115d6a`, behind
`DW_EXPERIMENTAL_REFINE_WINDOWS`, which the bench server runs with; refused
without it; no new module):

- a guide may carry `past_end: true` (one frame, no audio, timed one latent
  past the render's last) and `enter_below_sigma` (`h3_guides`,
  `h3_guide_steps`, which records the gated rows);
- the windowed denoiser (`h3_windows`, already installed under the flag)
  leaves the gated rows out of the forward while the step's sigma (1 - t) is
  at or above the gate - the sequence less those rows, as a window is - and
  lets them in below it;
- chain key `experimental_capstone: {enter_below_sigma, from_end: 5}`: segment
  1's frame `num_frames - from_end` (119 at 124 frames) is appended to every
  later segment's guides.

Tests: `tests/test_h3_capstone.py` (27) - the guide keys and refusals, the
row's time one latent past the end, a gated forward equal to the forward of
the sequence without the rows (on the tiny transformer), the rows back in
below the gate, the chain feeding frame 119 to segments 2+, and an end-to-end
tiny-graph render where the gate changes the result.

**Not yet run: it needs a deploy.** Arms ready in `bench/arms/capstone.json`
(`s2-arc.json` gained a `capstone` variable, null by default): S2 at seeds 1166
and 2, gates 0.9 (the last two of H3's four turbo forwards, sigma 0.857 and
0.667) and 0.7 (the last only), about 18 min each, read against
`ctl-guide-s2-*` (`counters-controls.json`) under the S2 guide floor, by
eye for the arc. Build it only if it wins S2.
