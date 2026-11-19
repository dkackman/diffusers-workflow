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
tone. Still-mouth bins 1. Seam peak 0.7-1.7x the median frame difference, as smooth as
the control.

One seed; Don has not viewed it (`levelled-carry-s1.mp4`,
`levelled-vs-control-faces.jpg`). Untried: a partial pull (the spread left alone, or
half strength) against the late softening, and levelled with the keystone or v4
keyframes for identity.

### Don's verdict (2026-11-13)

On `levelled-carry-s1.mp4`: "she becomes progressively glossy". Rejected; the stills and
the face-sharpness measure read the gloss as softening and missed it. Don closed the
experiment here. The untried combinations above were not run.

## bf16 transformer, seed 1 - gate db33868da007, arm af390360c19b (#1319 stage 1, #1346)

`bench/workflows/narrated-take-bf16.json` is `narrated-take.json` with the transformer's
SDNQ int8 config removed. The transformer loads in bf16 under the same block group
offload; the text encoder and VAE stay int8 SDNQ. Every other argument matches the CP1
control `d5584b91816d` (shot `dialogue`, seed 1, `stop_after` 12, tier `best`, no
counters). Both jobs were submitted inline because the server can't reach `bench/` by
path, so `get_job_workflow` shows what ran.

### Gate (stop_after 1) - job db33868da007

- Wall time 951 s: load 94.6, encode 29, denoise 738.3 (19 passes, ~37 s each), decode 70.9.
- Host RAM: `get_job`'s `host_memory_job_peak_committed_mb` is 37,124 MB. Committed
  memory leaves out file-backed pages, so it misses the bf16 transformer, which is
  mmap'd from its checkpoint. The worker process's peak RSS on lem was 108,325 MB,
  reached at load. In-run RSS was ~98 GB, ~63 GB of it file-backed. `host_ram_gb` was
  set to 110 from the RSS figure.
- Peak reserved VRAM 22,142 MB.
- No admission refusal, no OOM, under 45 min, so the full arm ran.

### Cost

| | int8 control d5584b91816d | bf16 af390360c19b |
|---|---|---|
| wall, 12 segments | 233.0 min | 223.7 min |
| denoise per pass | 54.6 s | 54.5 s |
| host committed peak | 68,747 MB | 37,994 MB |

The bf16 arm loaded nothing (`load_cached`: the worker still held the gate's pipeline),
so its wall time carries no load; the control's load was 58 s. Its passes ran at the
control's pace, not the gate's ~37 s. Committed memory leaves out file-backed pages, and
the bf16 transformer is mmap'd from its checkpoint, so 37,994 MB does not mean bf16 used
less host RAM. The host figure for bf16 is the gate's process peak RSS, 108,325 MB.

### face_drop per segment (chain check; already relative to segment 1)

| segment | int8 | bf16 |
|---|---|---|
| 2 | 0.003 | 0.011 |
| 3 | 0.032 | 0.019 |
| 4 | 0.043 | 0.027 |
| 5 | 0.057 | 0.060 |
| 6 | 0.146 | 0.101 |
| 7 | 0.114 | 0.097 |
| 8 | 0.111 | 0.183 |
| 9 | 0.140 | 0.178 |
| 10 | 0.148 | 0.176 |
| 11 | 0.154 | 0.212 |
| 12 | 0.207 | 0.225 |

Mean over segments 7-12, where CP1's burn shows: int8 0.146, bf16 0.179. Both arms fail
`texture_ratio` on segments 9-12 at the same size (int8 2.12 / 2.25 / 2.51 / 2.39, bf16
2.17 / 2.32 / 2.49 / 2.30, from each job's warnings).

### Face crop at the CP1 windows (`bench/face_windows.py`)

Sharpness (median Laplacian variance of the face crop), saturation (mean HSV S of the
crop) and SFace similarity to the face at 0.5 s. This script's crop and scale differ
from the earlier sections' scratch measure (the control reads 90 here against 204
above at 0-10 s), so only the two columns in this table compare with each other.
Growth is read from each arm's own 0-10 s window, because different weights give a
different segment 1.

| window | int8: sharp / sat / sim | bf16: sharp / sat / sim |
|---|---|---|
| 0-10 s | 90 / 101.7 / .92 | 103 / 101.5 / .94 |
| 20-30 s | 167 / 109.6 / .90 | 120 / 115.2 / .90 |
| 40-50 s | 283 / 111.9 / .80 | 252 / 118.5 / .80 |
| 60-70 s | 573 / 118.4 / .80 | 700 / 127.4 / .76 |
| 80-90 s | 807 / 120.4 / .74 | 894 / 120.4 / .72 |
| growth 0-10 to 80-90 | 9.0x / +18.7 / -.18 | 8.7x / +18.9 / -.22 |

Face strip (0.5 / 40 / 60 / 85 s, int8 row above bf16): `asset:bench/1346-bf16-face-strip.jpg`
in workspace `constitution-walk-3`, written by `bench/face_windows.py` from the two takes
(`output:BenchNarratedTake/20261109-225944-af94be0b/final/BenchNarratedTake-take.1-0.0.mp4`,
`output:BenchNarratedTakeBf16/20261114-081739-b46f591b/final/BenchNarratedTakeBf16-take.1-0.0.mp4`).
Whether the bf16 row looks less burnt is Don's call (plan Q3).

### Verdict

**Precision is not a burn lever at int8 to bf16** (seed 1). The bf16 arm burns at
the control's rate: face_drop reaches 0.225 against 0.207 at segment 12 (mean over 7-12
0.179 against 0.146), sharpness grows 8.7x against 9.0x, saturation +18.9 against +18.7,
and identity falls .22 against .18. Only the sharpness growth is lower in bf16, by an
amount one seed can't separate from noise. The rest is level or worse. A lever would
have shown as a clear drop, as int4 to int8 did (`tiers.md`: crust 1.6x instead of 4.5x).
None showed, so this is not a marginal result and needs no second seed. The burn left
at int8 is not quantization error. A higher-precision tier would not remove it, and
#1319's TP trigger (a) is not met.

## Seam restore (ref2va tail redraw), seed 1 - early ccb20cd03ffa, arm da4a93845ba9 (refs #1325, #1351)

Before each seam k, the restore redraws segment k's frames 68-191 (124 frames, on H3's
17n+5 grid) as ref2va. Each redraw runs `refine_strength` 0.3 at shift *s* = 12, which is
ComfyUI *d* = 0.3 / (12 - 11*0.3) = 0.0345. It takes the still as reference, segment
k+1's prompt, the matching 124-frame narration as `hold_audio`, and seed 1000+k. Restore
frames 102-123 become segment k+1's 22-frame guide, and frame 102 becomes its image.
Segment k keeps its own unrestored tail in the output. The control is reused, not re-run:
`aed61997052a`, `bench/seam-restore-control`, 5912 s.

- **Control take:** `bench/seam-restore-control/20261114-181113-99c25c22/final/SeamRestoreControl-take.22-0.0.mp4`
- **Arm take:** `bench/seam-restore/20261115-200029-ff99d30b/final/SeamRestoreProbe-take.37-0.0.mp4`

Both are 1042 frames, 6 segments, with audio. lem was on diffusers `0.44.0.dev0` @
`43e30028` (`get_diffusers_state` before the early job: `{"status":"idle",
"version":"0.44.0.dev0","commit":"43e30028af5120d8b0868cb6975b9fad39cc4fab"}`). Each
arm job ran alone on lem, after `clear_memory` on both cards.

The construction matches the control. The restore workflow, diffed against the control
with `release_pipeline` stripped, differs only in three places:
- the `rtailsrc`/`raud`/`restore`/`rtail` steps, which replace the control's `tail` steps;
- `key k`'s and `seg k`'s guide source (`rtail k-1` instead of `tail k-1`);
- the description.

Variables, seeds, prompts, still, audio slices, size and schedule are identical.

Two failed jobs came first. Neither is part of the arm:
- **`dd3e9db5f675`** (plan v3): refused in `restore01`, 874 s in. Its 120-frame source was
  off the 17n+5 grid (#1372). Host peak 66.3 GB, and it never denoised a restore.
- **`782ceef4f82e`** (plan v5): seg01 finished, 192 frames in 903 s. restore01 completed
  all 19 passes (+1072 s to +2181 s). The job was then SIGKILLed by the OOM killer about
  41 s into `decode.video`. Host committed memory peaked at 124 151.9 MB, against 128 GB
  on lem, because seg01's fl2va and restore01's ref2va pipelines were resident together.
  Plan v6 added `release_pipeline: true` to every segment and restore step.

**Gate** (early job `ccb20cd03ffa`, run `20261115-190311-7fd38e7b`, take
`bench/seam-restore-early/20261115-190311-7fd38e7b/final/SeamRestoreProbeEarly-take.9-0.0.mp4`,
362 frames, 47.6 min, host peak 66 058.8 MB). The gate compares seg02 against the
control take's frames 192-361:

| measure | control | restore |
|---|---|---|
| seg02 face-sharpness rise (seam + in-segment) | +4.6 (-16.1 + 20.7) | -9.5 (-22.4 + 12.9) |
| seg02 face-saturation rise | -3.3 (-0.6 - 2.6) | +0.4 (+3.6 - 3.1) |
| seam-1 score (peak / median) | 1.37 | 1.40 (1.02x the control) |

The gate passed: the arm's rise is below the control's, and its seam score is under
1.5x the control's. The full job ran.

Early-job timing (`get_job(ccb20cd03ffa)` step timing). `load_cached` was false on every
step, as `release_pipeline` intends.

| step | load | encode | denoise | per pass | decode |
|---|---|---|---|---|---|
| seg01 | 57.9 s | 33.7 s | 680.4 s | 34.3 s | 60.2 s |
| restore01 | 57.0 s | 93.7 s | 746.2 s | 39.2 s | 39.5 s |
| seg02 | 56.7 s | 48.4 s | 892.2 s | 46.9 s | 59.5 s |

- **Loads** took 171.6 s in total, 57.0 s of it for the restore.
- **Restore denoise** took 746.2 s (one restore, restore01).

**Full job `da4a93845ba9`** (run `20261115-200029-ff99d30b`, cuda:1). 209.2 min wall
time against the control's 98.5 min. Host peak 67 473.1 MB, against the control's
71 057.0 MB.

Measurements: every frame was measured, using YuNet's largest face. The drift split
uses 1 s endpoint windows, the `bench/driver.py` `_split` rule.

| measure | control: total / seams / in-segment | restore: total / seams / in-segment |
|---|---|---|
| luma | -0.5 / -0.8 / +0.3 | -3.0 / -9.0 / +5.9 |
| saturation | +8.8 / +9.4 / -0.7 | +17.0 / +21.4 / -4.4 |
| face saturation | +8.3 / +11.8 / -3.5 | +15.1 / +29.8 / -14.7 |
| face sharpness | +105 / +32 / +73 | +251 / +137 / +114 |

Face-sharpness step at each seam, from seg k's last 1 s to seg k+1's first 1 s:
- control: -16.1, -1.3, +30.8, +22.8, -3.9
- restore: -22.4, -1.9, +63.7, +26.3, +71.0

Face windows: median sharpness / mean saturation / SFace similarity to the 0.5 s face,
sampled every 0.5 s. The take is 43 s, so the windows are 10 s back to back.

| window | control | restore |
|---|---|---|
| 0-10 s | 94 / 99.3 / .94 | 91 / 100.3 / .94 |
| 10-20 s | 100 / 101.1 / .92 | 90 / 106.5 / .92 |
| 20-30 s | 156 / 104.6 / .89 | 171 / 114.1 / .90 |
| 30-40 s | 211 / 107.2 / .83 | 259 / 114.6 / .83 |
| 40-43 s | 331 / 106.8 / .78 | 407 / 116.6 / .79 |

Seam jumps. The join jump is the frame difference from (seg k, 191) to (seg k+1, 22)
over the median frame difference (control 6.33, restore 6.07). The detail step is face
median sharpness and mean saturation over the 6 frames either side of the join.

| seam | control: jump / sharp / sat | restore: jump / sharp / sat |
|---|---|---|
| 1 | 1.18x / 192 -> 77 / 99.4 -> 97.8 | 1.21x / 189 -> 78 / 98.9 -> 101.6 |
| 2 | 1.19x / 70 -> 138 / 97.7 -> 104.3 | 1.22x / 66 -> 120 / 102.4 -> 113.0 |
| 3 | 1.46x / 96 -> 100 / 102.7 -> 106.3 | 1.45x / 92 -> 126 / 110.1 -> 118.9 |
| 4 | 1.55x / 108 -> 160 / 106.3 -> 109.2 | 1.22x / 142 -> 194 / 117.6 -> 121.9 |
| 5 | 1.59x / 206 -> 321 / 103.7 -> 105.8 | 0.84x / 399 -> 482 / 113.8 -> 118.0 |

The peak frame difference within -2..+3 frames of each join gives the same ratios. The
one exception is control seam 1, where it is 1.32x at frame 195.

How far the redraw moved: `rtail k` (restore frames 102-123) against seg k's frames
170-191.

| seam | mean abs diff | face sharpness change | SFace similarity |
|---|---|---|---|
| 1 | 2.8 | -20.2 | .950 |
| 2 | 2.8 | -2.0 | .949 |
| 3 | 2.8 | -0.5 | .974 |
| 4 | 2.9 | +10.5 | .963 |
| 5 | 3.2 | +23.0 | .949 |

Still-mouth bins: 0 control, 0 restore. The mouth box here is approximated from YuNet's
mouth-corner landmarks (1.5 x 1.0 corner widths). It is not `lip_sync`'s box, so these
counts don't compare with the other sections'.

Full-job timing (`get_job(da4a93845ba9)` step timing). `load_cached` was false on every step, as
`release_pipeline` intends.

| step | load | encode | denoise | per pass | decode |
|---|---|---|---|---|---|
| seg01 | 47.4 s | 19.5 s | 743.8 s | 39.3 s | 59.8 s |
| restore01 | 57.6 s | 95.8 s | 859.7 s | 45.2 s | 40.6 s |
| seg02 | 51.4 s | 38.1 s | 1038.7 s | 54.7 s | 58.0 s |
| restore02 | 57.1 s | 96.8 s | 867.4 s | 45.2 s | 41.6 s |
| seg03 | 51.2 s | 45.5 s | 1039.7 s | 54.7 s | 58.0 s |
| restore03 | 58.6 s | 99.9 s | 863.1 s | 45.2 s | 43.1 s |
| seg04 | 205.5 s | 34.5 s | 1035.4 s | 54.5 s | 63.3 s |
| restore04 | 56.1 s | 101.6 s | 865.0 s | 45.3 s | 42.2 s |
| seg05 | 213.8 s | 34.0 s | 1036.6 s | 54.6 s | 59.2 s |
| restore05 | 57.9 s | 102.1 s | 861.3 s | 45.2 s | 39.8 s |
| seg06 | 57.1 s | 40.5 s | 1037.4 s | 54.6 s | 65.9 s |

- **Loads** took 913.7 s in total, 287.3 s of it for the restores.
- **Restore denoise** took 4316.5 s, 72 min, which is a third of the job.
- **Segment passes** ran at 54.6 s against the control's 46.9 s and the early job's 46.9 s
  (seg02). Nothing in the step timing explains this.

The probe used about 5.1 h of lem time against the 4 h approved: early 47.6 min, full
209.2 min, and about 52 min on the failed jobs.

The only ruling this record states is the gate's. Whether the restore helps is Don's
viewing verdict, on the two take paths above. One seed.

## Cut probe at the #1311 keeper's 41.0 s cut (#1385 stage A, #1388)

The keeper is GettysburgJoin `be80b3c3` (job 86e43a45af77, workspace `drill-1311-2`, left
untouched). It joins passage A (job 48c68b1e1c72, run `45d023c8`, `speaker.jpg`) to passage B
(job 50b48953037f, run `5760a9e4`, `gettysburg/speaker-angle-b.jpg`) at 41.0 s. Everything
below ran in workspace `bench`, one H3 job at a time.

The measure is `bench/cut_windows.py` (new): `bench/face_windows.py`'s measure, which is
YuNet's largest face, the Laplacian variance of its gray crop, and SFace cosine, taken in
1 s windows sampled every 0.25 s. Each window is the median. "SFace across the cut" is
the cosine between the mean features of the 1 s on each side. A clip's numbers sit a
few percent off the source's because every arm is re-encoded.

### The keeper

`cut_windows.py pair A B` on the two source takes:

- **A climbs to its end.** It starts at about 360, sits at about 650-800 over 31-38 s,
  and reaches 1059 and **1141** in its last 2 s.
- **B opens at 276**, a step of **0.24x (-76%)** at the cut.
- **B holds at about 200-280 for its first 17 s.** It rises to about 600-900 after 45 s.
  Its only window at or above A's end level is one spike at 54-55 s (1228), so it never
  reaches that level for more than a second.
- **SFace across the cut is 0.913.**

This corrects `bench/records/drill.md`'s "Resets at the cut: none" (see the correction
under it there). `detail_ratio` 0.86 is a whole-frame measure. The face crop loses about
three quarters of its detail at the cut and does not regain it for the length of B's
first segment.

### The arms

Each arm is a 16 s clip: A's last 8 s (frames 792-983), then 8 s of B or its replacement.
The cut is at 8.0 s.

| Arm | What follows the cut | Job | File (workspace `bench`) |
|---|---|---|---|
| C | the keeper's own cut, trimmed (join frames 792-1175) | 359bca51bb43 | `BenchCutProbeCJ/20261116-160223-dcd1316b/final/arm-c-0.0.mp4` |
| J | B's first 8 s in four 2 s chunks, `sharpen` 1.1 / 0.75 / 0.45 / 0.2, rejoined | 359bca51bb43 | `BenchCutProbeCJ/20261116-160223-dcd1316b/final/arm-j-0.0.mp4` |
| S | B's segment 1 re-rendered from angle B sharpened to A's end level | 88dc63d36f81 (render), f522b8e3c84d (clip) | `BenchCutProbeSL/20261116-163339-fc4b4c3f/final/arm-s-0.0.mp4` |
| L | B's segment 1 re-rendered from A's frame 980 through flux3 image-edit | 094d3eee5576 (render), f522b8e3c84d (clip) | `BenchCutProbeSL/20261116-163339-fc4b4c3f/final/arm-l-0.0.mp4` |

The inputs:

- **C and J** come from `bench/workflows/cut-probe-cj.json`. J uses only `trim_video`,
  `sharpen` and `concat_videos`.
  - B's chunks rejoin as one continuous take (`continuous_joins`, `seam_fade_ms` 0), so
    B's audio passes through, and that head then joins A's tail with `trim_frames` 0.
  - The amounts were calibrated locally with Pillow's UnsharpMask (radius 2, the
    `sharpen` task's filter) on B's 1 s windows. The first is matched to A's last 1 s.
- **The S still** is `bench/workflows/cut-probe-stills.json` (job 6ae8e4f404d1, run
  `BenchCutProbeStills/20261116-160341-815995d8`). It is `arm-s-still-0.0.png`: angle B
  through `sharpen` 0.36, which measures **1119** against 634 unsharpened (-2% from 1141).
- **The L frame** is `arm-l-frame-0.0.png` from the same job. It is A's frame 980
  (40.83 s), the sharpest face in A's last 2 s, at 1558 (that span's median is 1091).
  - It went through `templates/flux3/image-edit` with 7a326418cd63's instruction and
    seed 1311 (job 314c4f2dee50,
    `templates/flux3/image-edit/20261116-160404-0035fcca/final/Flux3KleinImageEdit-edit.1-0.0.jpg`).
  - The edited still measures **2085**. The gate skips the render only below 799 (30%
    under 1141), so **the gate passed** and L was rendered.
- **The S and L renders** are `templates/minimax/narrated-take` with B's job's arguments
  (narration, script, scene, speaker, `shot` `standing`, seed 1311). Only `image`
  changed, and `stop_after` is 1. The renders are
  `templates/minimax/narrated-take/20261116-160645-788405be/final/MiniMaxH3NarratedTake-take.1-0.0.mp4`
  (S, 13.7 min) and `.../20261116-162036-dbdd2de4/final/...` (L, 12.3 min).
- **The S and L clips** come from `bench/workflows/cut-probe-sl.json`.

### Sharpness and identity at the cut

Measured with `cut_windows.py clip <arm> 8`:

| Arm | last 1 s before | first 1 s after | step | SFace across the cut |
|---|---|---|---|---|
| C | 1079 | 251 | 0.23x (-77%) | 0.902 |
| J | 1098 | 1091 | 0.99x (-1%) | 0.921 |
| S | 1111 | 437 | 0.39x (-61%) | 0.913 |
| L | 1104 | 949 | 0.86x (-14%) | 0.942 |

Per second after the cut (8-16 s of the clip):

| Arm | 8-9 | 9-10 | 10-11 | 11-12 | 12-13 | 13-14 | 14-15 | 15-16 |
|---|---|---|---|---|---|---|---|---|
| C | 251 | 215 | 196 | 253 | 194 | 252 | 212 | 192 |
| J | 1091 | 949 | 596 | 733 | 393 | 490 | 294 | 258 |
| S | 437 | 353 | 297 | 409 | 312 | 436 | 342 | 304 |
| L | 949 | 778 | 756 | 880 | 795 | 930 | 857 | 704 |

Before the cut, all four clips read 684, 608, 535, 637, 589, 720, then about 1000 and
about 1100: it is the same A tail.

### Does B's opening keep the added detail through 8 s?

- **S: partly, at a lower level.** The still was raised to 1119, but the render's first
  second is 466 (source segment) and its 8 s settle at about 300-450 (median about 350).
  That is about 1.4x B's 200-280 over the same seconds, but a third of the still's level.
  H3 drops most of the added sharpening in the first second and keeps the rest to 8 s.
- **L: yes, at about three quarters of A's end level.** The source segment reads 984
  first, then 726-963 to 8 s, without a downward trend until the last second (726). The
  edited still was 2085, so the render keeps about 40-45% of it. That is still 3-4x B's
  opening.
- **J holds by construction.** It is a filter, not a render. Its step-down (1091 down to
  258) is the four amounts tapering into B's own level.

### Limit

S and L re-render only B's first segment (`stop_after` 1, plan Q3). They show how B would
open from a new start image. They do not show B's later segments, which chain from that
opening and are not rendered here. A full-B re-render, and whether the opening's level
carries through B's own chain, is outside this probe. Each arm is one seed (1311).

This record gives no ruling on the arms. Which reads best at the cut is Don's viewing
verdict (plan Q2), on the four clips above.

## #1399 turbo version vs burn (#1403)

_Bench by the lead agent, model `claude-opus-5-5` via `anthropic`, 2026-11-17._

The question: do the newer lightx2v turbo files burn less on an 8-segment H3 chain than
the ones the templates ship? Each new file runs on the #1326 probe it replaces, at the
same seed (42), prompt, size (960x544), segment count (8) and check settings (check on,
retries 0). Only `lora_weight_name` moves. The two probe workflows are copied unchanged
into `bench/workflows/probe-1399-*.json`. The controls are #1326's own runs, reused and
re-measured.

| Arm | Probe | Turbo file | Steps (evals) | Shift video/audio | Partition | Job | Workspace |
|---|---|---|---|---|---|---|---|
| A0 | chained-segments (cs-0) | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16` | 5 (4) | 6/3 | fl2va | `5fa5a09b083f` | probe-1326 |
| A1 | chained-segments | `minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16` | 5 (4) | 6/3 | fl2va | `372b4229b0a1` | probe-1399 |
| B0 | chain-video-continuity (cvc-0) | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16` | 9 (8) | 12/3 | ref2va | `fecc614dc513` | probe-1326 |
| B1 | chain-video-continuity | `minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16` | 9 (8) | 12/3 | ref2va | `8f7d30d7434e` | probe-1399 |
| C | B1 + #1151 anti-burn 0.5 | - | - | - | ref2va | did not run | - |

- **Shifts.** ModelTC's spec page lists no shift for v1.3 or v1.1, so each arm keeps its
  predecessor's: fl2va 6/3 and ref2va 12/3.
- **Partitions.** `validate_workflow` refuses the ref2v v1.1 file on the fl2va probe
  (cross-partition). That is as designed, so no ref2v file ran on fl2va.
- **Runs.** One H3 job at a time on lem. Each job's `note` names its arm.

### Build

`get_diffusers_state` at the bench:

    {"status": "idle", "version": "0.44.0.dev0",
     "commit": "1dabc292d8413638cf0666cca6d7a6c2c765befa", "revert": false}

The controls ran on 2026-11-14 on lem develop @ 0a60c514. Their job records do not hold
the diffusers commit. Every `get_diffusers_state` recorded in that week (the move-seeds
audit and the seam-restore record of 2026-11-15) reads `43e30028`, so the controls very
likely ran on `43e30028`, not `1dabc292`.

**The new arms and their controls therefore ran on different diffusers builds.** A
difference smaller than this tolerance is not attributed to the turbo file:
- creep: ±15%;
- per-seam lightness step: ±0.3;
- mean |L|: ±0.1.

That tolerance is also the noise floor of one seed per arm.

The measures are deterministic per file: measuring the same file again gives the same
numbers. The spread between runs comes from the seed and the build, not from the
measure.

### The face measure has no face to measure

The plan's burn measure is face-crop sharpness creep and SFace (YuNet + SFace,
`bench/face_windows.py`'s measure, per 1 s window in `bench/face_creep.py`).

The #1326 subject is a toy astronaut with a gold-mirrored visor, and it has no face.
YuNet finds a "face" in:
- A0: 2 of 140 samples;
- A1: 31 of 140;
- B0: 38 of 164;
- B1: 17 of 164.

Those hits are on the visor and suit and sit at a scale of 20 to 900. No arm has a
detection in both its first and its last 1 s window. **Face creep and SFace, first to
last, are unmeasurable on every arm.**

In their place, `bench/region_creep.py` (new) takes the same Laplacian variance of the gray
frame, per 1 s window (median of samples every 0.25 s), on two regions:
- a fixed centre box (x 35-65%, y 5-60%);
- the whole frame.

The box does not track the subject. A's astronaut sits left of centre, and B's framing is
tighter. The whole frame is the steadier of the two. Neither reads chroma noise. B0's
background ends in coloured speckle that a viewer reads as burn, while its gray Laplacian
falls.

### Measurements

| | A0 | A1 | B0 | B1 |
|---|---|---|---|---|
| Face creep / SFace first to last | no face | no face | no face | no face |
| Whole-frame sharpness, first 1 s to last full 1 s | 687 -> 1956 (x2.85) | 640 -> 1904 (x2.97) | 158 -> 105 (x0.67) | 111 -> 163 (x1.47) |
| Centre-box sharpness | 1418 -> 1701 (x1.20) | 1312 -> 2242 (x1.71) | 466 -> 129 (x0.28) | 354 -> 389 (x1.10) |
| `segment_looks.detail`, seg 1 first to seg 8 last | 5.584 -> 8.078 (x1.45) | 5.711 -> 8.391 (x1.47) | 4.549 -> 6.396 (x1.41) | 4.721 -> 6.209 (x1.32) |
| Lightness, start to end | 81.00 -> 81.01 | 80.24 -> 73.59 | 82.02 -> 68.64 | 71.32 -> 55.30 |
| Saturation, start to end | 13.68 -> 18.52 | 13.14 -> 15.78 | 17.17 -> 23.82 | 13.19 -> 17.86 |
| Per-seam lightness step, seams 1-7 | 0.63, 0.21, 0.19, -0.46, 0.10, -0.67, -1.38 | 0.54, 0.12, 0.54, -0.50, 0.47, -0.43, -0.39 | -2.15, -1.28, -2.05, 0.61, -0.76, -1.04, -0.93 | -4.54, -3.80, -12.72, 7.70, -2.83, -1.83, -1.14 |
| Mean seam \|L\| | 0.52 | 0.43 | 1.26 | **4.94** |
| `chain_check` fails | 0 | 0 | 7 (segs 2-8, jump_ratio) | 2 (segs 4-5, jump_ratio + look_ratio) |
| Wall time (denoise) | 14.6 min (496 s) | 15.2 min (493 s) | 62.6 min (3224 s) | 53.8 min (2580 s) |
| Transformer passes | 32 | 32 | 64 | 64 |

- **Audio.** `analyze_seams` warns on seam level steps above 3 dB:
  - A0: seams 3, 6 and 7;
  - A1: seams 1-3;
  - B0: seams 2, 4 and 6 (8.0, 7.1 and 14.1 dB);
  - B1: seams 3, 4, 6 and 7 (5.8, 7.4, 5.5 and 4.6 dB).
- **A1's audio level.** It sits about 20 dB above cs-0's.
- **B1's seam jumps.** B1's seam `jump_ratio` is lower than B0's at 5 of 7 seams (1.7 to
  2.4 against B0's 7.8 to 15.1).
- **B1's lightness.** It opens 11 points darker than B0 and loses 16 more by the end.
  Segment 4 drops to 52 (seam 3, -12.7), and segment 5 recovers 7.7.

### Reading

The bar is "burns less": creep at most 0.7x its own control, with SFace and seams no worse.

- **A1 (fl2v v1.3) does not burn less than A0.**
  - Its creep is no lower on either region (whole frame 1.04x A0's, centre box 1.43x),
    and its detail climb matches (x1.47 against x1.45).
  - Its seams are within tolerance (mean |L| 0.43 against 0.52).
  - It darkens by 6.6 where A0 holds lightness.
  - It does not meet the bar.
- **B1 (ref2v v1.1) fails the bar on seams, whatever its creep.**
  - Its mean seam |L| is 4.94 against B0's 1.26, nearly four times as large, and far
    outside tolerance. The bar asks seams no worse.
  - It does pass `chain_check` more often (2 fails against 7), and its seam frame jumps
    are smaller.
  - Its detail climbs less (x1.32 against x1.41).
  - Its whole-frame Laplacian rises (x1.47) where B0's falls (x0.67). That says more
    about B0's softening than about B1's burn.
- **No file met the bar, so no catalog entry is added.**

### Arm C did not run

The gate has two conditions:
1. B1's face-sharpness creep is at most 0.7x A0's;
2. B1's `chain_check` fails are no worse than B0's.

- **Condition 2 is met:** B1 has 2 fails, B0 has 7.
- **Condition 1 cannot be evaluated as written,** because no arm has a measurable face.
  - The stand-ins split. On the whole frame, B1's x1.47 is 0.52x A0's x2.85, which opens
    the gate. On the centre box, B1's x1.10 is 0.92x A0's x1.20, which shuts it.
  - A gate that cannot be read is not met, so C, about 55 min of lem, did not run.
  - `validate_workflow` was not tried for anti-burn on ref2va. Its catalog entry lists
    fl2va only.
- **C could not have met the bar either way.** It is B1 plus anti-burn, so it would also
  have to bring the seams from 4.94 back to B0's 1.26.

### Limits

- **One seed** per arm.
- **Different diffusers builds** for the new arms and the controls (see *Build*).
- **The burn measure is a stand-in.** It is the whole-frame and fixed-box gray Laplacian,
  not the face measure, and it misses chroma noise.
- **Any future turbo-version bench needs a different subject.** It needs a subject with a


## Kandinsky S2V Pro 4-bit probe (2026-11-16 to 11-18, #1339) - Pro distilled, 20 s gate, seed 1

**Result: fails the 20 s gate on all three thresholds, so the 86 s take was not run.**

### Path taken

This used the plan's fallback for the waveform case, outside the dw server. S2V's encoder
step takes `audio` as a 16 kHz mono waveform. dw's argument realization decodes only
image and video media (`dw/argument_media.py`), so an `asset:` wav reaches the pipeline as
a path. **Onboarding needs an `audio` media branch.** `bench/arms/ks2v-pro-4bit.json` (workflow `bench/workflows/ks2v-pro-4bit.json`) is
the dw shape, and it validates. The probe that ran is `bench/ks2v_pro_probe.py`, run from
`lem:~/spikes/ks2v` on **the spike venv's interpreter** (`~/spikes/ks2v/venv/bin/python`,
the 2026-11-12 standalone venv), not the dw server's venv.

- Inputs: `kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers`, 16 steps, 768x512,
  25 fps, seed 1, cuda:0, with these files from workspace `constitution-walk-3`:
  - the still: keyframes-v2 kf00, `asset:uploads/bench/keyframes-v2/kf00.jpg`;
  - the narration: `asset:bench/constitution-narration-retimed.wav`.
- diffusers 0.43.0, torch 2.14.1+cu130, in the spike venv (`RECORD` lines in the logs
  below). The dw server on lem reports its own diffusers separately (0.44.0.dev0 at
  e84bda1c on 2026-11-18). It didn't run this probe.
- Backend that loaded: **SDNQ int4**, on the first try both times. bnb nf4 was never needed.

### Attempt 1: 4-bit, encoders on model CPU offload (2026-11-16)

GPU OOM in `KandinskyS2VDistilledCoreDenoiseStep`, chunk 1: 21.02 GiB allocated by PyTorch
and 23.43 GiB in use on a 23.56 GiB card, while trying to allocate 862 MiB. The automatic
retry with group offload crashed in the probe script itself. It called
`enable_group_offload` on the Qwen2.5-Omni thinker, which has no such method (fixed in
a732352d). Log: `lem:~/spikes/ks2v/pro-gate-20s.log`.

**Pro distilled at 4-bit does not fit 24 GB without offloading the transformer.**

### Attempt 2: 4-bit, transformer group-offloaded (2026-11-18)

Only the transformer was group-offloaded (block level, with a stream). The encoders, CLIP
and the VAE stayed on model CPU offload. Log: `lem:~/spikes/ks2v/pro-gate-20s-go.log`.
Output: `lem:~/spikes/ks2v/pro-gate-20s-go-s1.mp4`, 500 frames at 25 fps (20.00 s) with a
20.00 s AAC narration track.

- **Peak VRAM: 17.1 GiB allocated, 17.6 GiB reserved.**
- **Host memory: about 48 GB RSS for the probe.** The pinned transformer and the 7B thinker
  pushed lem into swap. At 04:52 UTC the kernel OOM killer took the dw server's worker
  (pid 1304878, `dw-serve.service`), not the probe. sshd took minutes to answer while it
  thrashed.
- **Sampling took 143.9 min for 4 chunks, 36.0 min per chunk on average.** By chunk:
  - chunk 1: 89.3 min, with step 3 alone taking 75 min of swap thrash;
  - chunk 2: 20.9 min;
  - chunk 3: 17.1 min;
  - chunk 4: 12.5 min, at 46.7 s a step once the memory pressure cleared.

  Even the steady rate is about 5x Lite distilled's 2.6 min per chunk. At that rate an
  86 s take (18 chunks) would need about 3.7 h with lem's memory otherwise idle.

### Face table at the gate

Measured with `bench/ks2v_gate.py`, YuNet 2023mar and SFace 2021dec, all four takes in the
same script. SFace is the cosine to each take's own 0.5 s face. Saturation is the mean HSV S
of the YuNet face crop. The Laplacian is the variance of that crop in grey, at native scale.
Still-mouth bins are half-second bins over 0-20 s where the narration's RMS is above 0.7x
its median and the mouth box's frame diff is below 0.45x its median.

This script's tight crop reads saturation and Laplacian on a different scale from the
2026-11-12 table, so those two columns compare only within this table. SFace reproduces
the 2026-11-12 row to within 0.02. Re-run over 0.5/15/.../85 s, it gives keystone
1/.95/.90/.93/.92/.90/.90 and Lite 1/.90/.81/.76/.86/.80/.73.

| at 0.5/5/10/15/19.5 s | moving keystone s1 | arc + latent s1 | S2V Lite distilled | **S2V Pro distilled, 4-bit** |
|---|---|---|---|---|
| SFace similarity to first frame | 1/.87/.93/.95/.95 | 1/.91/.85/.91/.92 | 1/.85/.77/.90/.85 | **1/.76/.69/.73/.72** |
| SFace similarity to kf00 | .93/.86/.92/.95/.94 | .94/.92/.84/.88/.90 | .94/.81/.74/.87/.80 | **.91/.77/.68/.73/.69** |
| saturation | 105.3/104.8/107.1/100.1/109.7 | 102.6/109.1/115.2/109.5/107.4 | 93.9/88.8/105.2/100.9/107.1 | **92.0/93.2/98.4/97.4/97.4** |
| face-crop Laplacian variance | 312/208/210/308/305 | 327/285/206/257/375 | 159/66/33/49/51 | **114/96/61/71/94** |
| still-mouth bins (0-20 s) | 0 of 40 | 0 of 40 | 0 of 41 | 0 of 41 |

Sources:
- moving keystone s1: run `20261110-230854-abc647e9`;
- arc + latent s1: run `20261112-040003-f9eb4c89`;
- Lite distilled: `lem:~/spikes/ks2v/lite-distill-86s-s1.mp4`.

The two H3 runs are under `constitution-walk-3/outputs`; each uses its `take.1-0.0.mp4`.

The kf00 row comes from the same script run with `--ref` set to kf00 (2026-11-18). That
run reproduced every other row exactly. kf00 is the reference `face-drift-check` uses. The
first-frame row's reference, the Pro take's frame 12 (0.5 s), is also on the server as
`asset:bench/ks2v-pro/pro-gate-20s-ref-0.5s.png`.

Pro files on the server, workspace `constitution-walk-3`:
- the take: `asset:bench/ks2v-pro/pro-gate-20s-go-s1.mp4`;
- the contact sheet: `asset:bench/ks2v-pro/pro-gate-20s-faces.jpg`.

### Gate decision

| threshold | Pro at 20 s | |
|---|---|---|
| SFace ≥ 0.88 | 0.72 at 19.5 s, and never above 0.76 after 0.5 s (against kf00: 0.69, never above 0.77) | **fails** |
| saturation rise ≤ 5 | 92.0 → 97.4, +5.4 | **fails** |
| Laplacian within the H3 arms' range (206-375, both H3 arms over the same samples) | 61-114 | **fails** |

**The gate fails, so the run stopped at 20 s, as the plan says.** Under Don's 2026-11-17
ruling there is no third attempt and no further download. The Pro-distill weights stay in
lem's default HF cache (`~/.cache/huggingface`); deleting them is Don's call.

### Contact sheet

`lem:~/spikes/ks2v/pro-gate-20s-faces.jpg` has one row per take, at 0.5/5/10/15/19.5 s.

Reading: Pro starts close to the still. By 5 s the face is rounder and smoother, with
waxy skin and softer folds, and by 10 s it reads as a different woman. It does not come
back by 19.5 s. Both H3 rows hold the face and its folds. Lite drifts too, but less than
Pro over the same 20 s. The earrings, scarf and background hold in every row.

Pro at 4-bit is softer than Lite distilled at the start (114 against 159 at 0.5 s) and
loses identity faster. Whether 4-bit SDNQ caused this or Pro does it anyway isn't
isolated: an unquantized Pro (21B) needs about 42 GB for its transformer alone.

## LongCat-Video-Avatar 1.6 INT8 (2026-11-19, #1465)

It fits a 3090 only with block streaming, at 255 s per extra window. It fails the 30 s
gate on SFace to kf00 (0.80 at 15 s) and on 256 px sharpness (254 against 410 at
20-30 s). See `bench/records/longcat-avatar.md`.

## Keystone attention: step 1, the block-numbering check (#1391 stage B, #1394)

Status: **closed, gate failed.** Don ruled on 2026-11-18 that step 2 doesn't run and
the gate does. The gate failed, and on 2026-11-19, after viewing both takes, he stopped
the bench there: no full arms (see *Keystone attention: early gate* below).

The method is roycho87's brightened-keystone test (#1391 issuecomment-6922271175), after
NerdNotes test 19 (CC BY-SA 4.0, concepts only). See `docs/ATTRIBUTIONS.md`.

### Setup

- Build: diffusers `0.44.0.dev0` @ `e84bda1c` (`get_diffusers_state`), not the control's
  `43e30028` (Don's ruling, 2026-11-18).
  Workspace `bench`, one H3 job at a time.
- Shared segment 1 and the test keystones: `f835a2292345`
  (`bench/workflows/keystone-probe-seg01.json`, 1524 s, dense). Its files are under
  `KeystoneProbeSeg01/20261118-124246-9b6fdb6d/`: the segment is
  `segments/KeystoneProbeSeg01-seg01.1-0.0.mp4` and the keystones are
  `keystones/KeystoneProbeSeg01-{normal.2,bright.3,sharp.4}-0.0.png`. Mean luma: normal
  101.99, bright 132.44 (×1.299), sharp 101.99 (Laplacian variance 116 → 198).
- Every run renders segment 2 only, with dense attention, seed 2, and segment 1's
  22-frame tail as the guide. `keystone-probe-noarg.json` is `keystone-probe.json`
  without the argument. The argument runs differ only in `blocks`, at `balance` 20.
- Measure: `bench/keystone_luma.py`, mean BT.601 luma (0-255) of the new frames 22-191,
  plus their HSV saturation.

### Results

| Job | Keystone | `blocks` | Card | Wall | s/pass | frame 0 | new frames | vs normal | new sat |
|---|---|---|---|---|---|---|---|---|---|
| `53e01c397101` | normal | (no argument) | cuda:1 | 2045 s (cold) | 98.8 | 103.63 | **101.049** | - | 70.98 |
| `4444aa27a7dc` | bright | (no argument) | cuda:1 | 1977 s | 99.1 | 104.00 | 101.182 | **+0.13%** | 70.87 |
| `b686b8a0a40b` | bright | `0-29` | cuda:0 | 1930 s (cold) | 92.6 | 132.62 | 101.627 | **+0.57%** | 70.58 |
| `ad1e332586f1` | bright | `30-39` | cuda:0 | 1808 s | 90.4 | 109.48 | 109.526 | **+8.39%** | 72.13 |

`53e01c397101` re-runs `43045ca7c88a`, which lem's crash cancelled ("server restart").
Every take is `bench/keystone-probe{,-noarg}/<run>/segments/*-seg02.2-0.0.mp4`: runs
`20261119-003654-fca9788e`, `20261119-011109-be51bfbc`, `20261119-014436-ed1d2859` and
`20261119-021658-05c9b3fa`.

24-frame window means:

- normal, no argument: 101.56 102.72 100.41 100.82 100.06 101.55 101.03 100.62
- bright, no argument: 101.69 102.86 100.45 101.02 100.48 101.73 101.07 100.55
- bright, `0-29`: 104.62 102.30 101.12 103.16 100.99 99.91 101.90 101.92
- bright, `30-39`: 103.26 109.30 109.06 110.29 109.80 107.93 110.29 110.46

### Reading

- **The numbering matches.** `0-29` lands near no argument (+0.57% against +0.13%), and
  `30-39` lifts clearly (+8.39%). That is the shape roycho87 reported (+4.4% / +4.1% /
  +13.8%). diffusers' indices 0-49 behave as ComfyUI's `diffusion_model.blocks[i]`,
  so the arms may go ahead on this numbering.
- **Our plain keystone carries less on its own.** With no argument, the bright keystone
  moves the new frames by +0.13% where roycho87 saw +4.4%. This chain's 22-frame
  carried guide may dominate the single keyframe row more than his setup's did. The
  absolute lifts are smaller than his, but the ratio between the groups matches.
- **The early blocks copy the keystone into frame 0 only.** With `0-29`, frame 0 reaches
  132.6, the bright keystone's own level, and the new frames barely move. With `30-39`,
  the lift spreads over the whole segment: windows 2-8 hold at about 108-110.
- **The LSE op's cost isn't separable from these runs.** The argument runs landed on
  cuda:0 and the no-argument runs on cuda:1. The argument runs were faster per pass
  (90-93 s against 99 s), so the difference between cards outweighs the op here. The
  arms' cost comparison has to pin both arms to one card or compare like-for-like
  cards.

## Keystone attention: early gate (#1391 stage B, #1394)

Don's ruling (2026-11-18): no step 2. The numbering check passed, so the arms use
roycho87's 13 blocks, `31-33,36-44,47`, at `balance` 20, the plan's fallback when no
group is picked. The gate ran as #1325's did: seg01-02 of both arms, one H3 job at a
time, with `free -g` and `get_health` checked before each. lem stayed up through both.

### Setup

- Build: diffusers `0.44.0.dev0` @ `e84bda1c` (`get_diffusers_state`), workspace `bench`.
- Control: `bench/workflows/keystone-attention-control-early.json`. Every segment's
  image is seg01's frame 0 (key01), the 22-frame tail is the guide, and attention is dense.
- Attention arm: `bench/workflows/keystone-attention-early.json` is the control plus
  `experimental_keystone_attention` (`blocks` `31-33,36-44,47`, `balance` 20) on seg02.
  Nothing else differs: seeds, prompts, audio slices, size and schedule are the same.
  The validate warning confirmed the argument: "runs balance 20 on blocks
  31-33,36-44,47".
- **The two seg01s match.** seg01 carries no argument in either arm. Across the two
  cards its frames 0-169 differ by a mean |diff| of 0.024 (0-255). Most frames are
  bit-identical, and the mean PSNR is 335 dB. So seg01 is a fair yardstick for the cards.

### Jobs

| Job | Arm | Card | Wall | Host peak |
|---|---|---|---|---|
| `2817a04202ae` | control | cuda:0 | 3232 s | 67 923 MB |
| `29923da18475` | attention | cuda:1 | ~3758 s | 67 906 MB |

Step timing (`get_job`). seg02 reused the pipeline seg01 had loaded.

| Arm | Step | Load | Encode | Denoise | Per pass | Decode |
|---|---|---|---|---|---|---|
| control | seg01 | 43.0 s | 38.1 s | 1347.1 s | 70.9 s | 57.8 s |
| control | seg02 | cached | 44.0 s | 1632.1 s | 85.9 s | 54.7 s |
| attention | seg01 | 46.3 s | 22.7 s | 1546.2 s | 81.5 s | 58.3 s |
| attention | seg02 | cached | 34.4 s | 1975.9 s | 103.8 s | 57.1 s |

**The LSE op's cost** is about +5% per pass. It is an estimate: the dispatcher put the
arms on different cards (`run_workflow` has no device pin), so the jobs couldn't share
one. Per pass, the identical seg01 ran 81.5 s on cuda:1 against 70.9 s on cuda:0, so
cuda:1 is 1.149x slower. seg02's raw ratio is 103.8 / 85.9 = 1.208, and correcting for
the card leaves 1.208 / 1.149 = 1.05.

Takes, in workspace `bench`:
- control: `keystone-attention-control-early/20261119-033059-82643275/final/KeystoneAttentionControlEarly-take.6-0.0.mp4`
- attention: `keystone-attention-early/20261119-042526-08aab8e0/final/KeystoneAttentionEarly-take.6-0.0.mp4`

The segments are under `segments/` in the same runs (`*-seg01.1-0.0.mp4`,
`*-seg02.5-0.0.mp4`).

### Measures

How they were measured:
- Every frame of each 362-frame take. YuNet 2023mar's largest face was found on all 362.
- Face sharpness is the Laplacian variance of the gray face crop. Saturation is the HSV
  mean. Luma is BT.601 gray.
- seg01 is frames 0-191 and seg02 is frames 192-361, matching `analyze_seams`'
  `between_frames`.
- The split uses 1 s (24-frame) endpoint means, the `bench/driver.py` `_split` rule.
- The seg02 rise is the seam step (seg02's first 1 s minus seg01's last 1 s) plus
  seg02's in-segment slope.

| Measure | Control: total / seams / in-segment | Attention: total / seams / in-segment |
|---|---|---|
| luma | +1.58 / +2.50 / -0.92 | +1.43 / +1.54 / -0.11 |
| saturation | +1.79 / +1.09 / +0.70 | +2.44 / +2.30 / +0.14 |
| face saturation | -2.23 / -0.51 / -1.71 | -2.75 / -1.01 / -1.74 |
| face sharpness | -63.6 / -5.2 / -58.4 | -54.2 / +18.1 / -72.4 |

**Gate:**

| Measure | Control | Attention |
|---|---|---|
| seg02 face-sharpness rise (seam + in-segment) | **-16.9** (-5.2 - 11.7) | **-7.7** (+18.1 - 25.8) |
| seg02 face-sharpness, first 1 s → last 1 s | 103.6 → 92.0 | 127.1 → 101.3 |
| seg02 face-saturation rise | -3.5 (-0.5 - 3.0) | -3.5 (-1.0 - 2.5) |
| seam-1 score (`analyze_seams` `jump_ratio`) | 1.09 | 1.19 (**1.09x** the control) |
| seam-1 `look_ratio` (contrast) | 2.50 | 2.91 |
| SFace to the 0.5 s face: seg02 first 1 s / last 1 s | .908 / .907 | .916 / .894 |

Face-sharpness medians per 1 s window, frames 0-359. seg01 is windows 0-7 (frames
0-191) and seg02 starts at window 8 (frame 192).

- control: 160 113 111 89 96 92 144 103 100 106 102 99 108 131 82
- attention: 160 113 111 89 96 92 143 102 108 98 103 99 108 120 98

### Verdict

**The gate fails on its letter.** The seam score passes (1.09x, under 1.5x). But the
attention arm's seg02 face-sharpness rise, -7.7, is higher than the control's, -16.9,
and the gate asks for no worse. Following Don's ruling, the full arms did not run.

How to read it:
- **Neither arm creeps sharper here.** Face sharpness falls through both seg02s. The
  "rise" is a smaller fall in the attention arm, not over-sharpening. The gate's
  direction comes from #1325, where the control rose +4.6, and doesn't separate
  "creeps sharper" from "keeps more of the keystone's detail".
- **The op clearly does something.** seg02's new frames differ from the control's by a
  mean |diff| of 17.8 (PSNR 17.7 dB): pose and framing change, not just grain. seg02
  opens at face sharpness 127, closer to the keystone's 156 (seg01's first 1 s), than
  the control's 104. That is the +18 seam step. It then decays faster (-25.8 against
  -11.7) and ends a little sharper, 101 against 92.
- **The luma seam step is smaller** (+1.5 against +2.5). Identity drifts a little more
  by seg02's end (SFace .894 against .907).
- **The frames don't show a clear win.** Face crops at 0.5, 8.5, 11.0 and 14.9 s
  (`get_output_frames`, crop `[372, 0, 600, 768]`) look alike in texture and identity,
  with neither crusted. They differ in pose and framing: at 11.0 s the attention arm's
  speaker is nearer the centre, closer to the keystone's framing.
- Ruling point 4 applies: the gate fails, but the op visibly acts. That is the case for
  going back to step 2 with a narrower set of groups, which is Don's call.
- A positive result here would still be a dense-attention result, while the shipped
  templates run sparse (Q5).

### Don's verdict (2026-11-19)

**Gate failed. Stop here.** Don watched both takes and saw no clear win. Neither step 2
with narrower groups nor the full arms runs, and no more lem time goes to this. Stage A's
`experimental_keystone_attention` stays experimental and behind the flag. The chain key
and the moving-shot arm stay cut (plan v4's non-goals): nothing here brings them back.
