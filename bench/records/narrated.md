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
(`docs/proposals/complete/narrated-long-take-complete.md`, Amendment 1).

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

## CP1: moving keystone, seed 2 - job e7917ab0cab4

Same arm, seed 2. 210 min wall, no re-rolls.

| Segment | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `face_drop` | -0.002 | 0.023 | 0.006 | 0.012 | 0.023 | 0.018 | 0.003 | 0.003 | 0.014 | 0.014 | 0.032 |

| Saturation at | 1 s | 15 s | 30 s | 45 s | 60 s | 75 s | 85 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moving keystone s2 | 75.0 | 75.1 | 77.5 | 79.7 | 74.7 | 76.6 | 78.5 |

Seams: jump_ratio mean 0.37, max 0.60; look_ratio max 0.63.

Eye: face crops at 1/30/85 s match seed 1's. One sway inside segment 9
(about 58-61 s): she drifts right of centre and the colonnade comes out
from behind her, sharper, then she is back on her mark by 62 s as the
segment lands on its keyframe. Both seeds replicate the face hold.

### Don's verdict on the moving keystone (2026-11-11)

Faces good on both seeds, "maybe some slight burn". No walking, only sway. Seed 2 drops
lip sync on occasion; seed 1 is better. Don chose a slow lateral camera arc over a walk.

Mouth stillness (YuNet mouth box, grey diff per frame against audio RMS, half-second
bins, flagged when speech > 0.7x median and mouth < 0.45x median): seed 2 had 12 flagged
bins, 7 of them within 0.8 s of a segment end - the closed-mouth keyframes pull the
mouth shut as each segment lands.

## Keyframes v3: camera arc, mid-word mouths, graded plate

One plate, graded to the still's background (Reinhard LAB), cropped along a lateral arc
(left 105 -> 505 px over 12 segments), the matted speaker pasted with a 65 px drift.
Keyframes 1-12 cycle four Klein open-mouth edits of the still, aligned on the YuNet
landmarks. Arms: `bench/arms/narrated-arc.json`, `narrated-arc-latent.json`.

## Arc, seed 1 - jobs f2775de46a93 (arc), 7f9f6f106524 (arc + latent carry + capstone 0.7)

| | moving keystone s1 | arc | arc + latent |
|---|---|---|---|
| face_drop, segments 2-12 max | 0.041 | 0.068 | 0.049 |
| saturation 0/15/30/45/60/75/85 s | 76.0/75.3/79.4/79.2/79.7/79.0/80.8 | 73.8/74.8/78.2/80.9/79.3/77.5/77.2 | 73.8/74.7/76.0/78.4/77.8/78.5/76.5 |
| still-mouth bins | 7 | 3 | 2 |
| seam_jump max / mean | 0.83 / 0.60 | 1.09 / 0.65 | 1.22 / 0.88 |
| texture fine, last vs first segment | 0.148 | 0.002 | 0.009 |

Reading: the arc reads (building and trees cross behind her), no reset, lip sync better.
The face is **more** burned than the moving keystone from about 30 s: deeper, sharper
folds and warmer skin (`arc-faces.jpg`). The texture measure is whole-frame and missed
it. Cause: the Klein open-mouth edits aged the whole face (`kf-faces-v2-v3.jpg`), and
the chain follows its keyframes. Seam scores include the camera move.

## Keyframes v4: the original face, only the mouth blended in

The v3 series with each variant reduced to a feathered mouth ellipse (centre the YuNet
mouth midpoint, radii 0.95 x 0.62 mouth widths, blur 0.14) over the original matte,
colour-shifted to the original on a ring around the ellipse. Arm:
`bench/arms/narrated-arc4-latent.json`.

## Path assessment (2026-11-12)

At 86 s the chain holds narration and pose; burn is not fixed, and identity now rests on
a hand-built keyframe per segment that no agent can build through MCP. Nothing is
tested past 86 s. diffusers 0.43.0 ships Kandinsky S2V 1.0 (reference frame + speech
track, 5 s chunks with FramePack memory, MIT). Don approved: spike S2V on the same still
and narration, run v4 once on H3, then stop keyframe iteration either way.

## Kandinsky S2V 1.0 spike (2026-11-12) - Lite distilled, 86 s, seed 1

Standalone script on lem (`~/spikes/ks2v`, diffusers 0.43.0 beside the server's 0.42.1),
`kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers`, 16 steps, keyframes-v2 kf00 +
the retimed narration, auto CPU offload, cuda:0. 768x512, 25 fps, peak 17.0 GiB, about
2.6 min per 5 s chunk (86 s in about 50 min).

| at 0.5/15/30/45/60/75/85 s | moving keystone s1 | arc + latent s1 | S2V Lite distilled |
|---|---|---|---|
| SFace similarity to first frame | 1/.95/.90/.93/.92/.91/.91 | 1/.91/.90/.87/.91/.90/.90 | 1/.90/.80/.76/.87/.79/.71 |
| saturation | 75.4 -> 80.8 | 73.5 -> 76.5 | 70.9 -> 91.0 |
| face-crop Laplacian variance | 554-826 | 552-995 | 60-238 |
| still-mouth bins | 7 | 2 | 5 |

Reading: fails. Identity drifts to a different, darker face by 85 s, the earrings go,
saturation climbs by 20, and the face is soft throughout (`ks2v-vs-h3-faces.jpg`). The
narration is unbroken and there is no reset, but H3 with the moving keystone holds the
face far better. Lite base (50 steps, guidance 3.0) and Pro (21B, would need 4-bit on
24 GB) are untested.

## Arc v4 + latent carry, seed 1 - job 3147f5720689

| at 0.5/15/30/45/60/75/85 s | moving keystone s1 | arc v3 + latent | arc v4 + latent |
|---|---|---|---|
| saturation | 75.4/75.3/79.4/79.2/79.7/79.0/80.8 | 73.5/74.7/76.0/78.4/77.8/78.5/76.5 | 73.4/73.5/76.0/76.3/75.4/77.0/76.3 |
| SFace similarity to first frame | 1/.95/.90/.93/.92/.91/.91 | 1/.91/.90/.87/.91/.90/.90 | 1/.93/.71/.92/.91/.79/.85 |
| face-crop Laplacian variance | 603/610/700/595/826/554/696 | 614/552/773/955/789/969/995 | 621/558/708/756/944/986/1118 |

face_drop per segment 0.010-0.074 (max at segment 11); still-mouth bins 2, neither near
a segment end; seam_jump max 1.08, mean 0.79; texture fine 0.052.

Reading: the mouth-only keyframes removed the orange, aged cast of v3 and saturation is
the flattest of any arm; lip sync is the best of any arm. The face still sharpens
through the take (crop sharpness nearly doubles by 85 s, folds deepen from about 45 s;
`arc4-faces.jpg`). Moving keystone s1 (v2 keyframes, no latent carry) does not sharpen
this way. Which of latent carry, the graded and blurred plate, or the arc causes it is
not isolated. Per R20, keyframe iteration stops here.

## Arc v4 without latent carry, seed 1 - job ddbac3a46144

The isolating run for #1312: the arc v4 arm with `experimental_latent_carry` off
(`bench/arms/narrated-arc4.json`), everything else as job 3147f5720689. 246 min, two
re-rolls (segments 2 and 6).

Face crop, every sixth frame, per 10 s window (median Laplacian variance of the crop at
256 px / mean HSV saturation of the crop / median SFace similarity to the first frame).
The crop is sized differently from the table above, so compare within this table only.

| window | moving keystone s1 | arc v4 + latent | arc v4, no latent |
|---|---|---|---|
| 0-10 s | 452 / 99.8 / .92 | 477 / 101.3 / .92 | 472 / 101.8 / .92 |
| 10-20 s | 508 / 102.3 / .91 | 489 / 104.4 / .90 | 439 / 104.7 / .90 |
| 20-30 s | 536 / 104.1 / .90 | 533 / 103.7 / .88 | 410 / 107.6 / .88 |
| 30-40 s | 512 / 104.5 / .89 | 482 / 109.2 / .85 | 435 / 106.0 / .89 |
| 40-50 s | 492 / 103.8 / .90 | 604 / 105.0 / .88 | 468 / 104.4 / .89 |
| 50-60 s | 626 / 104.0 / .90 | 680 / 106.3 / .89 | 519 / 107.0 / .89 |
| 60-70 s | 566 / 102.7 / .90 | 808 / 106.7 / .90 | 462 / 106.0 / .89 |
| 70-80 s | 451 / 101.7 / .89 | 710 / 107.4 / .89 | 407 / 104.7 / .89 |
| 80-90 s | 469 / 103.1 / .87 | 856 / 104.7 / .87 | 390 / 102.8 / .89 |

Whole-frame saturation at 0.5/15/30/45/60/75/85 s: 73.3/73.7/77.3/76.6/76.6/75.7/75.0.
face_drop per segment 0.026-0.075 (max at segment 7). Still-mouth bins 3 (moving
keystone s1: 7; v4 + latent: 2). seam_jump max 0.89, mean 0.59 (v4 + latent: 1.08 /
0.79); texture fine 0.050. Lowest single-frame SFace similarity 0.51 (60-70 s window);
v4 + latent dips to 0.21, moving keystone s1 to 0.75.

Reading: latent carry is what over-sharpens the face. With it off, crop sharpness stays
between 390 and 519 for the whole take, where the same keyframes with latent carry climb
from 477 to 856. The plate, the arc and the v4 keyframes are cleared. The v4 mouth-only
keyframes keep their lip-sync gain without latent carry (3 still-mouth bins against 7
for the v2 keyframes). On these measures this is the cleanest 86 s arm, and it is the
shipped `walk_keyframed` configuration with v4-style keyframes. One seed; Don has not
viewed it (`cp1-arc4-s1.mp4`, `arc4-nolatent-faces.jpg`).

## Masked handoff probe, seed 101-112 - job 64a392dcebe1 (refs #1322)

A test of the handoff ComfyUI-Continuity calls "masked": the previous segment's tail is
written into the new segment's own latent and held under a mask, where the guide chain
passes it as extra condition rows. No engine change - twelve unchained fl2va steps in one
workflow, each with `source_video` (the previous tail, 22 frames, padded to 192),
`mask_video` (22 black frames, then white), `mask_grow` 0, `refine_strength` 1.0, the
tail's first frame as `image`, and the narration slice as `hold_audio`. Plain `dialogue`
conditions otherwise: no keystone, no keyframes, no latent carry. The tail goes through a
VAE encode, not as raw latents. Compared with the `dialogue` control, seed 1
(`20261109-225944-af94be0b`); the two differ in seed as well as handoff.

Drift split into the step across each seam (1 s after minus 1 s before, summed over 11
seams) and the slope inside each segment:

| measure | control: total / seams / in-segment | masked: total / seams / in-segment |
|---|---|---|
| luma | +0.3 / -0.6 / +0.9 | +2.8 / +11.6 / -8.7 |
| saturation | +12.5 / +18.4 / -5.9 | +17.6 / +20.5 / -2.8 |
| face saturation | +21.3 / +33.8 / -12.5 | +24.1 / +39.4 / -15.3 |
| face sharpness | +648 / +624 / +24 | -109 / -145 / +36 |

Face crop per 10 s window (median sharpness / mean saturation / median SFace similarity
to the first frame), same script as the table above:

| window | control | masked |
|---|---|---|
| 0-10 s | 204 / 101.3 / .92 | 169 / 97.1 / .92 |
| 20-30 s | 280 / 109.2 / .89 | 207 / 105.9 / .84 |
| 40-50 s | 546 / 110.6 / .78 | 219 / 110.6 / .71 |
| 60-70 s | 830 / 117.2 / .79 | 203 / 119.6 / .71 |
| 80-90 s | 986 / 118.8 / .74 | 190 / 118.2 / .60 |

Reading. The masked handoff removes the over-sharpening: face-crop sharpness stays
between 156 and 262 for 86 s where the control climbs from 204 to 986, and the crops show
smooth skin at 85 s where the control is etched. It does not remove the colour drift:
saturation still steps up at every seam by the same amount, and brightness now steps up
too. Identity holds less well (similarity .60 at the end against .74), with nothing but
the tail to anchor it. Still-mouth bins 4, all in the first 10 s.

Seam defects of this construction. The mask rounds to whole latent cells, so 18 of the 22
tail frames were held and frames 18-21 were regenerated; the workflow's own join (trim
22) therefore shows a one-frame jump of 3-4x the median frame difference at every seam.
A local join at 18 brings the seam peak to 0.9-3.1x (control 0.9-1.8x), still a visible
bump at about half the seams, and the first seam is followed by ~0.4 s of reframing.
14 min per segment against 18-19 for the guide chain; 165 min for the take.

One seed, different from the control's; Don has not viewed it
(`masked-handoff-s1.mp4`, `masked-vs-control-faces.jpg`).

## Levelled latent carry, seed 1 - job b800065fb772 (refs #1322)

`experimental_latent_carry: "levelled"` (d2967163) on the plain `dialogue` take: each
carried tail's video latents are pulled back, per channel, to the mean and spread of
segment 1's tail before the next segment reads them. No keystone, no keyframes. Same
seed and conditions as the control (`20261109-225944-af94be0b`), so segment 1 is the
same render. 192 min for 12 segments, no re-rolls.

| measure | control: total / seams / in-segment | levelled: total / seams / in-segment |
|---|---|---|
| luma | +0.3 / -0.6 / +0.9 | +4.1 / -4.3 / +8.4 |
| saturation | +12.5 / +18.4 / -5.9 | -2.6 / +6.6 / -9.2 |
| face saturation | +21.3 / +33.8 / -12.5 | +9.2 / +13.7 / -4.5 |
| face sharpness | +648 / +624 / +24 | -170 / -252 / +83 |

| window | control | levelled |
|---|---|---|
| 0-10 s | 204 / 101.3 / .92 | 219 / 100.5 / .93 |
| 20-30 s | 280 / 109.2 / .89 | 239 / 105.2 / .89 |
| 40-50 s | 546 / 110.6 / .78 | 225 / 103.5 / .83 |
| 60-70 s | 830 / 117.2 / .79 | 174 / 107.7 / .81 |
| 80-90 s | 986 / 118.8 / .74 | 107 / 105.5 / .76 |

What the level pulled back grew through the take: the tail's channel means sat 0.061
from the level at segment 2 and 0.191 at segment 12, its spread 1.03x and 1.07x. Each
segment drifts further inside itself even though it starts from a levelled tail.

Reading. The burn is gone: whole-frame saturation ends where it began, face saturation
rises less than half as far, and the face is never etched. The cost is the other
direction: from about 60 s the picture softens (face sharpness 219 to 107) and by 85 s
it is slightly hazy and pale. Identity drifts as in the control (similarity .76 against
.74; face_drop reaches 0.195 at segment 12) - levelling does not hold the face, only the
tone. Still-mouth bins 1. Seam peak 0.8-1.6x the median frame difference, as smooth as
the control.

One seed; Don has not viewed it (`levelled-carry-s1.mp4`,
`levelled-vs-control-faces.jpg`). Untried: a partial pull (the spread left alone, or
half strength) against the late softening, and levelled with the keystone or v4
keyframes for identity.
