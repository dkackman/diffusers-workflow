# Per-segment check: catch rate and false positives (Task 13 Step 5)

2026-11-05. Bench server lem:8766 at `082e93ce`, cuda:0/cuda:1, cuDNN
deterministic. Arms: `bench/arms/check.json`; record: `check.json` (copied
from lem's `~/dw-bench-t13`). The bench workflows gained three knobs, all off
by default, so every control is unchanged: `check`, `retries` and
`keep_segments` (chain `save_segments` + `keep_segments`).

| arm | scene | check | retries | job | wall |
|---|---|---|---|---|---|
| chk-s3-seg4-broken | S3, 6 segments, segment 4's prompt replaced | true | 1 | `fdea6eba4a33` | 14.4 min |
| chk-s1-fp | S1, 10 segments | true | 0 | `7b76a3796335` | 20.6 min |
| chk-s2-fp | S2, 10 segments | true | 0 | `604e115324b7` | 18.2 min |

Default ceilings: jump 4.0, look 5.5, texture 2.0 (either way).

## The made-to-fail segment (S3)

Segment 4's prompt opens on the keyframe and then "at 1.00 seconds cuts hard
to a deserted harbour at midnight in pouring rain ... the woman is gone",
with a stranger's shouted line (the five others are the S3 bench prompts).

- **Attempt 1 was caught.** `Chain segment 4/6: check failed (look_ratio
  15.1) - re-rolling, attempt 2 of 2 at seed 1001169` (job log 08:33:31).
- **Attempt 2 passed the check, and is broken.** The run record has
  segment 4 at `attempts: 2`, `seed: 1001169` and no `check_failed`, so no
  warning. By eye (contact sheets of take frames 320-532) the kept attempt
  holds the street for its first 26 kept frames, then cuts to the
  neon harbour at take frame 354 (render frame 48, about 2 s in;
  `bench/cuts.py`'s largest step) and
  stays there to the end of the segment.
- **Segment 5 is broken too, and also passed.** Its guide was the harbour
  tail, so it opens on the harbour and cuts back to the street at take frame
  449 (19 kept frames in), with a different woman (white top, different
  face) for the rest of the take.

So the check caught 1 of the 3 broken renders: the one whose break sat at
the seam. It missed both breaks that sat inside a segment (26 and 19 frames past
the seam). That
is the check's design, not its thresholds: `jump_ratio` reads the one frame
step across the seam and `look_ratio` a 4-frame window either side
(`dw/tasks/assess.py` `_seam_video`, `LOOK_WINDOW`). analyze_seams on the
kept take sees nothing either: seam jump max 1.93, look max 2.58 (all five
seams under the ceilings; seams 3 and 4 read 0.42 and 0.13 because the
harbour's fast pan raised segment 4's typical step to 20).

### Fix: `cut_ratio` (commit `d37c3192`, needs a deploy)

A fourth measure: the segment's largest frame step over its own floored P90
step - `_seam_video`'s arithmetic on a seam inside the segment.
`bench/cuts.py` measured it on decoded takes:

| set | segments | P50 | P99 | max |
|---|---|---|---|---|
| 47 bench takes (controls, noise floor, round 1; S1-S6) | 282 | 1.22 | 2.62 | 2.98 |
| this take, segment 4 (cut to the harbour) | 1 | | | 4.16 |
| this take, segment 5 (cut back) | 1 | | | 17.24 |

`CUT_RATIO_MAX = 4.0`, the seam's own ceiling, sits between the good
segments' 2.98 and the harbour cut's 4.16. The margin on the low side is
thin (the harbour pan inflated segment 4's own typical step); a cut into a
static scene reads far higher, like segment 5's 17.2. Tests:
`tests/test_chain_check.py` (`test_a_cut_inside_the_segment_fails_on_cut_ratio`,
`test_steady_motion_inside_the_segment_passes_cut_ratio`,
`test_cut_ratio_null_skips_it`). To confirm on the GPU after the deploy:
rerun `chk-s3-seg4-broken` - attempt 2 should now fail on `cut_ratio`, be
kept as the closest and flagged with `check_failed` and a warning.

## False positives (S1, S2 with the check on)

`retries: 0`, so the check measures and flags but renders nothing extra. Both
takes reproduce the noise-floor runs to the third decimal (S1 jump max 1.09,
look max 2.61, fine texture 1.118; S2 2.09, 2.71, 3.558), so turning the
check on leaves the render alone.

| scene | checked segments | jump fails | look fails | texture fails | cut_ratio (offline) |
|---|---|---|---|---|---|
| S1 | 9 | 0 | 0 | 3: segments 8, 9, 10 at 2.02, 2.22, 2.43 | 0 (max 0.99) |
| S2 | 9 | 0 | 0 | 3: segments 7, 9, 10 at 0.48, 0.39, 0.32 | 0 (max 2.21) |
| S3 (segments 2, 3, 5, 6) | 4 | 0 | 0 | 0 | segment 5 (a real cut) |

- **Jump and look: no false positives** in 22 good seams. The highest reads
  were 2.09 (jump) and 2.71 (look), about half the ceilings. No retune.
- **Texture: 6 of 18 segments flagged, all late in a 10-segment take.** S1
  crosses 2.0 at segment 8 (30.7 s), the point counters.md already names
  ("a static take longer than about 30 s should be cut into shots"); S2's
  fine band falls below 0.5 from segment 7, where the eye notes record the
  sky blowing out to white. These are the drift the ceiling exists to catch,
  not good segments failing, so the ceiling stands at 2.0.
- **But a texture failure cannot be fixed by a re-roll**: the drift is in
  the carry, and the re-roll renders from the same carry. At the templates'
  `retries: 1` each of those segments renders twice for nothing (S1: 3 extra
  segments, about +30% wall time) and is then flagged anyway. Proposal, not
  committed: let `texture_ratio` flag and warn without re-rolling (only
  jump, look and cut drive a re-roll), or re-roll on texture only when the
  previous segment passed it. That is a change of the check's policy, so it
  is for the controller to rule on.

## `cut_ratio` on the GPU (2026-11-05, deploy `155ae56c`)

Arms `bench/arms/check-cut-ratio.json`, record `check-cut-ratio.json`: the
`chk-s3-seg4-broken` arm again (new workflow id suffix, so no step-cache
hit), and the same arm at `retries: 3`.

| arm | job | card | retries | wall | segment 1 |
|---|---|---|---|---|---|
| chk-s3-seg4-broken-cut | `6668a913863c` | cuda:1 | 1 | 15.9 min | `24b18bc0/d21b04b0`, identical to the original run `fdea6eba4a33` |
| chk-s3-seg4-broken-r3 | `ccce7adc9aaf` | cuda:0 | 3 | 23.7 min | `806ef3ca/3ba72965`, drifted (the card had just run the 120 s refine) |

Both differ from the S3 canonical segment 1 (`289012f1`); the original check
arm did too, so the rerun is read against the original, which it matches.

**`cut_ratio` catches the cut 26 frames in.** On the rerun:

- attempt 1 of segment 4 fails on look 15.1 *and* cut 5.92;
- attempt 2 (seed 1001169), the one that passed before, now fails on
  **`cut_ratio 4.15`** (the offline `bench/cuts.py` read 4.16), is kept as
  the closest, and the run record's segment 4 shot carries `attempts: 2`,
  `seed: 1001169`, `check_failed: ["cut_ratio"]`;
- segment 5's cut back to the street fails on `cut_ratio 17.25` on both
  attempts (kept attempt 1, `check_failed: ["cut_ratio"]`);
- the job carries one `chain_check_failed` warning per segment, naming the
  measure and value: "Chain segment 4/6 failed its check on every attempt
  (cut_ratio 4.15); kept attempt 2 of 2, the closest" (and 5/6 at 17.25).

So all three broken renders are now caught, and segments 1-3 and 6 still
pass (no false positive). The margin is thin, as predicted: 4.15 against a
4.0 ceiling, because the harbour's fast pan lifts the segment's own typical
step.

**The retry budget does not find a clean attempt, here or at `retries: 3`.**
Segment 4's prompt asks for the cut, so every seed renders it: attempts 1-3
read cut 5.26 (+ look 15.26), 4.48, 5.21 (+ look 13.96), and attempt 4 failed
too (the log names only the measures that trigger a re-roll); kept attempt 2
at 4.48. Segment 5 opens on the kept harbour tail, so every seed breaks:
attempts 1-3 read cut 7.47, look 6.84, cut 7.91, attempt 4 failed; kept
attempt 1. The budget is
spent (+3 segments, +8 min against `retries: 1`) and the take is flagged,
which is what the check can do: a re-roll fixes a seed's accident, not a
prompt that asks for the break, and not a carry that already holds it
(`repair.md`, Step 3). (The `retries: 3` run is drifted at segment 1, so
its numbers are not comparable to the rerun's; its attempt 2 at the same
seed reads 4.48, not 4.15.)
