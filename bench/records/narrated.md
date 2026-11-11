# Narrated take: drift counters and sync, measured (narrated-long-take plan)

Measured 2026-11-09. Can the lip-sync measure (`dw/tasks/lip_sync.py`) tell
a segment whose mouth moves with its own audio from one whose audio is wrong,
well enough to be a per-segment chain check? The plan's rule: if at least 90%
of good segments score above every bad one at the same segment, the measure
stays in the chain check as built; otherwise it leaves the chain check, the
task stays, and sync is judged by eye. This used no new GPU time.

## Sync calibration

### Method

- **Footage.** The 70 kept segments of lem workspace `constitution-walk-2`:
  MiniMax H3 dialogue, one speaker to camera, 1344x768, int8 base, 20 steps,
  24 fps, 187 or 209 frames each.
- **Measure.** `lip_sync._mouth_signal` (the mean absolute grey change inside
  the mouth box cut from YuNet's mouth corners) against `lip_sync._envelope`
  (the audio's RMS per frame's span of samples), Pearson r at every
  whole-frame lag within ±24 frames (±1 s), on CPU.
- **Good.** Each segment with its own audio.
- **Bad.** Each segment with (a) its own audio 0.5 s late (12 frames) and
  (b) another segment's audio (seven segments on in the list, padded or cut
  to length).
- Raw per-segment numbers: the controller's scratchpad `calib/calib.json`
  (`r0`, `near_max` for |lag| <= 3, `best_r`, `best_lag`, per case).

### Separation

A segment separates when its good score is above its bad score for that same
segment. The rule needs 90% (63 of 70).

| Score | Good above late | Good above other audio |
|---|---|---|
| Best r within ±0.4 s (as built) | 40/70 | 60/70 |
| Best r within ±3 frames of lag 0 | 61/70 | 64/70 |
| r at lag 0 | 56/70 | 62/70 |

Across segments, no threshold separates one segment on its own: the 10th
percentile of good (best r within ±3 frames) is 0.138, the 90th of late is
0.27. Medians: good 0.288, late 0.145, other audio 0.100.

### Where the best lag sits

The best lag within ±1 s is the clearer sign. Good segments cluster at 0 to
+2 frames (34/70 at 0..+2, 40/70 at -2..+4). With the audio 0.5 s late, the
best lag moves to +12 to +14 frames (39/70 there), and only 5/70 land within
-2..+4.

### Verdict

No score reaches the 90% rule against the late audio, and no threshold
separates a single segment, so lip sync is not a per-segment chain check
(ruling R9): `lip_sync_min` is gone from the chain's `check`, and the chain
records no sync value. `measure_lip_sync` stays as a coarse take-level
indicator: `correlation` is the best r within ±3 frames of lag 0 (the best
row above), and `lag_seconds` is the best lag within `max_lag_seconds`
(default 1 s), which read over a whole take shows a consistent offset. One
segment's correlation is not a verdict. Judge sync by eye.

## CP1: control vs candidate, seed 1, 12 segments (~86 s)

Bench `bench/workflows/narrated-take.json` on lem (develop 036cb279,
`DW_EXPERIMENTAL_REFINE_WINDOWS=1`), workspace `constitution-walk-3`, scene
`s3-dialogue-walk`, speaker still `asset:bench/constitution-speaker.jpg`,
held narration `asset:bench/constitution-narration-retimed.wav` (298.45 s,
43 planned segments), `stop_after: 12`, tier `best` (1344x768, int8, 20
steps, no LoRA), 192/170 frames per segment, guide 22.

### Control (guide carry, no counters) - job d5584b91816d

Run `BenchNarratedTake/20261109-225944-af94be0b`. Wall 233 min, 19.4 min
per segment. No re-rolls; `texture_ratio` (warn-only) flagged segments 9-12
(2.12, 2.25, 2.51, 2.39).

| Segment | starts (s) | face_drop |
| --- | --- | --- |
| 2 | 8.0 | 0.003 |
| 3 | 15.1 | 0.032 |
| 4 | 22.2 | 0.043 |
| 5 | 29.3 | 0.057 |
| 6 | 36.3 | 0.146 |
| 7 | 43.4 | 0.114 |
| 8 | 50.5 | 0.111 |
| 9 | 57.6 | 0.140 |
| 10 | 64.7 | 0.148 |
| 11 | 71.8 | 0.154 |
| 12 | 78.9 | 0.207 |

Frame saturation (HSV mean): 68.3 at 0 s, 70.7 at 15 s, 75.6 at 30 s,
77.8 at 45 s, 78.8 at 60 s, 81.4 at 75 s, 81.0 at 85 s. Brightness flat
(111.6-115.5).

Eye: no resets, no change of size in frame, speaking throughout; the
narration's cadence carries (the halting pauses of constitution-walk-2 are
gone). The face is etched by 30 s, glossy and caricatured by 60 s, crusted
by 85 s; the backdrop turns painterly with repeating trees from about 60 s.
Defect 1 (the smoothing/waxing drift) reproduces at the same lengths as
`bench/records/drill.md` ask 1; `face_drop` steps up at segment 6 (36 s),
the first point an eye calls it.

### Candidate (latent carry, latent keystone every 3, capstone 0.7) - job 576bad7a2138

Run `BenchNarratedTake/20261110-025240-db6a16a5`, same arguments as the
control plus `experimental_latent_carry: true`, `keystone: segment_1`,
`keystone_slot: latent`, `experimental_capstone: {enter_below_sigma: 0.7}`,
`anchor_every: 3`. Wall 224 min, 18.6 min per segment (4% under the
control). No re-rolls; `texture_ratio` flagged nothing.

| Segment | control face_drop | candidate face_drop |
| --- | --- | --- |
| 2 | 0.003 | 0.001 |
| 3 | 0.032 | 0.038 |
| 4 | 0.043 | 0.023 |
| 5 | 0.057 | 0.037 |
| 6 | 0.146 | 0.094 |
| 7 | 0.114 | 0.105 |
| 8 | 0.111 | 0.102 |
| 9 | 0.140 | 0.113 |
| 10 | 0.148 | 0.141 |
| 11 | 0.154 | 0.195 |
| 12 | 0.207 | 0.205 |

| Saturation at | 0 s | 15 s | 30 s | 45 s | 60 s | 75 s | 85 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 68.3 | 70.7 | 75.6 | 77.8 | 78.8 | 81.4 | 81.0 |
| candidate | 68.4 | 67.6 | 68.0 | 66.3 | 67.7 | 66.4 | 65.7 |

Seams (`analyze_seams`, no findings on either): candidate jump_ratio
0.63-1.53 (seams 10 and 11 at 1.53/1.51), lightness steps up to 6.1;
control 0.49-1.12, lightness steps up to 3.6. By eye the candidate's worst
seams hold framing and pose with a slight brightness lift, no pop.

Eye: the candidate removes the colour burn outright and holds the face
natural to about 60 s, where the control is already glossy. It does not
hold to 85 s: by segment 11 the face glazes as the control's did at 60 s,
and `face_drop` meets the control's at the end (0.205 against 0.207).
The backdrop drifts in both (the colonnade gives way to rows of trees).

### CP1 reading

The candidate moves the face drift from about 30-36 s to about 70 s and
removes the saturation burn, at no extra wall time; neither arm holds a
face to 85 s. Seed 1 only.

### Static long (keystone segment_1, slot last) - job 7f52bc5b09b6

Seed 1, 12 segments, 282 min wall. `face_drop` stays within ±0.015 of
segment 1 throughout; saturation 68 to 75 to 73; `cut_ratio` re-rolled
segments 2 and 3. Eye: face and backdrop hold to 85 s, but she stands:
every segment eases back to the opening frame, so there is no walk.

### Face repair on the candidate - jobs 44be41a64140, d56817f8fa5d

`templates/minimax/face-repair` on candidate segments 6 and 11 (about 7.4
min each). It softens the glaze but has no reference to aim at: the face
comes back blurrier and, on segment 11, older. Dropped as a fix.

### Don's CP1 verdict (2026-11-10)

The candidate's glaze is distracting by 40 s; the control degrades
markedly by 40 s; the static long take holds the face throughout. A
walk with the static take's face quality wins, at a higher budget. Arms
D (keystone `last` every 3 segments: pulls her back to the walk's start)
and A (candidate anchors every segment: its anchors barely moved
`face_drop`) were dropped before running. Next: the moving keystone
(`docs/proposals/narrated-long-take.md`, Amendment 1).

## Moving keystone: keyframes

### v1: Klein edit per keyframe (dropped)

Twelve `templates/flux3/image-edit` runs on the speaker still (seed 1110),
one per segment end, each asking for the background "d metres further down
the avenue" (d = 7 to 84). About 1 min each on cuda:0.

- Identity holds: SFace similarity to the still 0.907-0.931.
- Framing is consistent between keyframes but pulled back from the still:
  face width 274 px against the still's 361 (0.76x), so segment 1 would
  read as a dolly-out.
- The recession stalls: keyframes 1-3 move the colonnade back, 4-12 are
  near-identical. Distance asked for in words stops working past ~28 m.

### v2: one plate, widening crops, the matted speaker (approved)

1. **Plate.** `templates/flux3/image-edit` at 2048x1168 on the still
   shrunk to half size in a grey border, asked to outpaint and remove her
   (seeds 1110, 1111; jobs 39f1ab8bdc05, d9d5d922b1cf). Seed 1111 kept: a
   centred colonnade at the end of a tree-lined avenue.
2. **Crops.** Keyframe k (0-12) is a 1.75:1 crop of the plate whose width
   grows geometrically from 1155 px to the plate's 2048 (1.77x over the
   walk, ~4.9% per segment), the building's base held at 56% of the
   height and its centre easing from 72% of the width to the middle,
   resized to 1344x768 with a 3 px Gaussian blur for depth of field.
3. **Speaker.** `remove_background` (RMBG-2.0) on the still (job
   7ed5e7747fdb), pasted onto every crop at the same pixels.

Her face is the still's own pixels in every keyframe: size, position and
identity cannot drift. The backdrop recedes steadily from keyframe 1 to
12. Matte edges (hair, earrings, collar) hold at full size.

Tried first and dropped: `templates/flux3/multi-reference` composites of
the still onto crops 1, 6 and 12 (seed 1110; jobs 302d42a0f24c,
95514085cb39, f0a4fd947702). The backdrop followed the crops and identity
held (0.913, 0.926, 0.930), but face scale wandered at 0.95x, 0.77x and
0.90x: a zoom pump at every segment end.

The chain starts from keyframe 0 (her on the tightest crop), not the
still, whose closer, different building would swap partway through
segment 1. Assets: `asset:uploads/bench/keyframes-v2/kf00.jpg` to
`kf12.jpg`. Don approved the series 2026-11-10.

Watch at CP1: every keyframe holds the same expression and pose, so
each segment lands on the same face. If that reads as a tic, vary the
expression per keyframe.

## CP1: moving keystone, seed 1 - job 514b463a855f

`shot: walk_keyframed`, opening on keyframe 0, keyframes 1-12 as each
segment's last image, otherwise the control's arguments. 12 segments,
219 min wall, no re-rolls.

| Segment | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `face_drop` | 0.006 | 0.007 | 0.020 | 0.014 | 0.008 | 0.010 | 0.013 | 0.020 | 0.010 | 0.032 | 0.041 |

Against CP1's arms at segment 12: control 0.207, candidate 0.205.

| Saturation at | 0 s | 15 s | 30 s | 45 s | 60 s | 75 s | 85 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moving keystone | 75.5 | 74.8 | 78.9 | 78.7 | 79.2 | 78.5 | 80.3 |

(Keyframe 0's plate starts warmer than the still, 75.5 against 68.)

Seams: jump_ratio mean 0.60, max 0.83; look_ratio max 0.89; audio level
step max 2.1 dB - the cleanest seams of any CP1 arm.

Eye: the backdrop recedes as the keyframes do (colonnade large at 0 s,
behind her head by 60 s, an avenue of trees at 85 s), with no pull-back.
Face crops at 30/60/85 s match the static long take's at the same
times (both carry the same etched skin texture from about 30 s), and
show none of the candidate's glaze. Whether it reads as a walk is
Don's call on the take.
