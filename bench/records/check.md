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
