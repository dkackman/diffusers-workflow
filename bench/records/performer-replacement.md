# Performer replacement gate (Task 16, #1194)

**Verdict: works.** A ref2va regional edit at `refine_strength` 1.0 inside a per-frame person
mask replaces the performer on an S4 cut. The new face matches the replacement portrait
better than a fresh ref2va render of the same portrait does. Outside the mask the source
holds, and the soundtrack is the source's. The one flaw is a background seam: the edit
regenerates a band of about 64-80 px around the person, and that band can redraw nearby
set geometry (here the couch back) so it no longer lines up with the kept source. No
engine change was needed. Recommendation: file the follow-up feature below.

Run 2026-11-07 on lem's bench (H3 on `:8766` behind the int8 queue, masking and face
checks on `:8767`).

## Setup

- **Cut:** S4 `deflect` (the close-up of the shaved-head man in the grey flannel shirt
  reading a magazine; 960x544, 124 frames, 24 fps, dialogue). Source run
  `BenchS4Montage/20261104-040255-9813688b`, file
  `intermediate/BenchS4Montage-shot@deflect.3-0.0.mp4`.
- **Replacement:** `asset:bench/t26-stranger.jpg`, a bearded man in round glasses, a
  mustard corduroy jacket and a green t-shirt, on a grey studio backdrop.
- **Mask** (`bench/workflows/t16-person-mask.json`): `video_frames`, then `segment`
  (GroundingDINO base + SAM2 hiera-large, both Apache-2.0; prompt `person`, threshold 0.3).
  `segment` already runs per frame on a video (`per_frame`), so the catalog gives a
  per-frame person mask today with no new code. Job `9fb7e66a2211`, 53 s on cuda:1.
  - Every frame has a person: coverage 0.27-0.35 of the frame (mean 0.31), union 0.38.
  - Mean frame-to-frame flip is 0.55% of pixels (max 6.1%, at the page turn).
  - The mask holds the whole body and leaves out the magazine. RMBG-2.0 (`remove_background`)
    was not used: it is CC BY-NC.
- **Edit** (`bench/workflows/t16-performer-swap.json`): S4's own ref2va shot pipeline
  (int4, ref2v turbo 8-step LoRA, 9 steps, `video_shift` 12, sparse attention).
  - Inputs: the cut as `source_video` and `hold_audio`, the mask as `mask_video`,
    `mask_grow` 1, `refine_strength` 1.0, and the portrait as `<Picture 1>`.
  - Prompt: the cut's own prompt with `<Subject 1>` rewritten as the portrait's man, plus
    "the grey studio backdrop of `<Picture 1>` is not used".
- **Control:** the same step with the source, mask and `refine_strength` dropped. This is
  a fresh ref2va render of the portrait in the same scene, and it is the ceiling a plain
  generation reaches.

Arms: `bench/arms/t16-performer-replacement.json`.

## Numbers

| Run | Job | Wall | face_drift vs portrait, median (min) |
|---|---|---|---|
| Source cut (baseline) | face check `d1e61cba18ef` | - | 0.056 (0.013) |
| Swap, seed 1194 | `0aa481a58b58` (face `70f69429f43a`) | 6.2 min (cold load) | **0.759** (0.700) |
| Swap, seed 2002 | `a78e8dbf981d` (face `70e0e3e71303`) | 4.0 min | **0.728** (0.662) |
| Fresh ref2va control, seed 1194 | `aec28b177ec6` (face `27b86eecde21`) | 3.4 min | 0.683 (0.579) |

The face_drift threshold is 0.363. All 16 sampled frames found a face on every run.

Source against edit by region (a scratch numpy script, not committed: PSNR, 7 px windowed luma SSIM and mean abs RGB diff, with the regions taken per frame from that frame's mask):

| Region | PSNR s1194 / s2002 | SSIM (7 px window) | Mean abs diff |
|---|---|---|---|
| Inside the mask | 10.8 / 10.9 dB | 0.22 / 0.23 | 60 / 59 |
| 1-32 px outside | 12.7 / 13.3 dB | 0.34 / 0.35 | 41 / 38 |
| 33-64 px outside | 18.0 / 18.6 dB | 0.61 / 0.64 | 20 / 19 |
| More than 96 px outside | **37.5 / 37.5 dB** | 0.964 / 0.964 | 2.8 / 2.8 |

- **Where the change stops.** Mean abs diff by distance from the union mask, in 16 px bands:
  29, 26, 19, 11, 5, 3, then 3. The edit stops about 64-80 px past the person's
  silhouette, which is the patch cell plus `mask_grow` 1. Past that the residual of about
  3 is the VAE round trip.
- **Audio.** The swap's track correlates 0.9988 with the source at -0.03 dB, so
  `hold_audio` held it. The control's track does not match the source (corr -0.06).
- **Temporal stability.** Mean frame-to-frame luma change in the 1-64 px band is 1.5-1.6,
  against 0.52 in the source. More than 64 px out it is 0.12-0.20, against 0.16-0.26 in
  the source. The band moves more because the new body moves there, not because it
  flickers (frames 40-47 by eye show no shimmer).

## By eye (frames 0, 62 and 123; full-resolution crops at the shoulder and the head)

- **Identity.** On both seeds the man is the portrait's man: round wire glasses, a grey and
  dark beard, greying swept hair, the mustard corduroy jacket and the green t-shirt.
  Nothing of the bald source performer is left. Pose, framing, the page turn and the
  magazine position follow the source. The mouth is open on the line at frame 62 and
  closed at 0 and 123, as in the source, so lip timing to the held dialogue looks right at
  these three moments. This was not checked frame by frame.
- **Outside the mask.** The brick wall, curtain, window and couch arm away from him are the
  source's. The control instead renders a different set, with a stair railing and a
  different couch, which shows how much the mask is holding.
- **Seam.** The brick wall continues through the regenerated band behind the head with no
  visible line: bricks and mortar line up on both seeds. The couch back does not. Beside
  the right shoulder the regenerated band redraws the back cushion lower, or as a
  different cushion, and its top edge steps where it meets the kept couch. The step is
  steady across frames, not flicker, but a viewer would see it on a still.
- **Inside the grown cells.** The new man is slimmer than the source performer. The space
  he no longer fills, which the mask still covers, is filled with plausible couch and wall.
  The magazine cover lies inside the grown cells and is redrawn: new garbled masthead text
  on both seeds.

## What this answers for #1194

- The edit itself needs nothing new. ref2va with `source_video` and `mask_video` already
  works, and outside the mask is kept by the engine, so Directors-Cut's `composite_part`
  step is built in.
- The identity holds at strength 1.0 because it comes from the reference, as the issue
  predicted. It scores above a fresh render, since the source's pose and framing keep the
  face frontal.
- The per-frame mask is already in the catalog for a single-person cut (`segment` over
  `video_frames`).
- What is missing is the multi-person and multi-cut machinery, not the swap:
  - A per-person track. `segment` detects every frame independently and merges all
    matches, so with two people it cannot say which person a mask belongs to.
  - Cut detection, to split a multi-shot source into cuts.

## Follow-up to file (owner: Don)

**feature: performer replacement, the per-cut loop around a masked ref2va edit (#1194).**
Stages:

1. **Cut-detection task.** Find hard cuts in a source clip and return shot boundaries in
   dw's `shots` form, so `assess_output`, `for_each` and the chain tools can read them. A
   content or adaptive detector in the PySceneDetect style is BSD-3; check its licence
   first.
2. **Person-track task.** Per-person, per-frame masks across a cut: SAM2 video propagation
   from a first-frame `segment` box per person, under the same Apache-2.0 weights. It
   returns one mask video per tracked person, plus a contact sheet so a caller can say
   which person is "Subject N".
3. **Template** `templates/minimax/replace-performer`. `for_each` over cuts: trim the cut
   to the `17n + 5` grid, track, run the ref2va regional edit (`refine_strength` 1.0, the
   portrait as `<Picture k>`, `hold_audio` the cut), then `concat_videos` and `pair_audio`
   with the source track. Each cut gets its own seed, so a bad cut can be re-rolled alone.
4. **The seam (bench it in the template's stage).** Try a tighter edit and see whether it
   stops the band redrawing set geometry (the couch-back step here), at the cost of a
   harder edge:
   - `mask_grow` 0;
   - a mask eroded by part of a cell before it is pooled.
   Measure the seam with the 16 px distance bands above and the crops at the shoulder.

Out of scope here, and still open in #1194: the VLM cast-listing prompt, two-person cuts,
the community swap LoRA (unsourced) and FastH3 Omni (#980).

## The seam: mask_grow 0 vs 1 (#1374)

**Result: `mask_grow` 0 does not pass the rule, so `templates/minimax/replace-performer`
keeps `mask_grow` 1.** mask_grow 0 confines the change to about 32 px instead of about
80 px. Its edge is harder, though (a `step` about 2.2-2.5 times mask_grow 1's on both
seeds), and the couch back beside the right shoulder is still redrawn. Do not run the
eroded-mask arm here: it is #1382.

Run 2026-11-16 on lem's main server (`:8765`, workspace `bench`, cuda:0, develop
`0de4c8f3`). The old bench server and its workspace were gone, so the deflect cut and
its mask were copied in as `asset:bench/t16/deflect-cut.mp4` and
`asset:bench/t16/deflect-mask.mp4` (from `BenchS4Montage/20261104-040255-9813688b` and
job `9fb7e66a2211`). Everything else matches `t16-swap-g0-*`.

The mask_grow 1 arms are the 2026-11-07 runs above, on that day's code; only the
mask_grow 0 arms were run now.

| Arm | Job | Run | Wall |
|---|---|---|---|
| g0, seed 1194 | `b4a85961ac8c` | `BenchT16PerformerSwap/20261116-014355-c095e171` | 4.1 min |
| g0, seed 2002 | `15a5044427a6` | `BenchT16PerformerSwap/20261116-014824-d971d9c5` | 4.0 min (warm) |
| g1, seed 1194 | `0aa481a58b58` | `20261107-021420-b3b99216` (old bench) | 6.2 min |
| g1, seed 2002 | `a78e8dbf981d` | `20261107-022021-f8596e06` (old bench) | 4.0 min |

`python bench/seam_bands.py deflect-cut.mp4 ARM.mp4 deflect-mask.mp4` gives the mean
abs diff from the source in 16 px bands outside the union mask:

| Arm | inside | 1-16 | 17-32 | 33-48 | 49-64 | 65-80 | 81-96 | **step** |
|---|---|---|---|---|---|---|---|---|
| g1 s1194 | 58.58 | 30.17 | 25.80 | 19.02 | 10.99 | 5.18 | 3.12 | **4.37** |
| g1 s2002 | 57.11 | 27.63 | 23.17 | 18.04 | 9.89 | 4.63 | 2.98 | **4.45** |
| g0 s1194 | 56.17 | 15.24 | 5.42 | 3.13 | 2.97 | 2.90 | 2.91 | **9.81** |
| g0 s2002 | 55.29 | 17.19 | 5.91 | 3.11 | 2.94 | - | - | **11.29** |

- Frame-to-frame luma change in the 1-16 px band is 1.97 / 2.17 for g0 and 3.08 / 3.11
  for g1, against 1.24 in the source.
- Full frame against the source, g0: 15.22 dB (SSIM 0.695) for s1194 and 15.36 dB for
  s2002.

**By eye** (shoulder crop `[560, 200, 400, 260]`, frames 0 and 62):
- g0 still redraws the couch back beside the right shoulder. On s1194 it becomes a
  separate raised cushion with a notch where it meets the kept couch. On s2002 it is a
  raised cushion corner.
- So the step moves closer to the silhouette instead of going away.
- Identity, pose, the page turn and mouth timing follow the source, as at mask_grow 1.

**Rule** (plan v2/v3, #1194): mask_grow 0 becomes the default only if it removes the
couch-back step with no hard edge. That is judged by the shoulder crops, plus a `step`
no larger than mask_grow 1's on both seeds. It fails both tests. The eroded-mask arm is
#1382, in the next section.

## The seam: an eroded mask (#1382)

**Result: erosion does not pass the rule, so `templates/minimax/replace-performer` keeps
`mask_grow` 1. No engine erode is proposed.** Eroding the person mask by 16 px before H3
pools it, at `mask_grow` 0, halves mask_grow 1's hard-edge `step`. But the jump across the
mask edge stays 6-7 times g1's step, and the crops show a ghost of the source silhouette.

Run 2026-11-17 (UTC) on lem's main server (`:8765`, workspace `bench`, cuda:0, develop
`44370b83`). How the mask was made:
- `python bench/erode_mask.py deflect-mask.mp4 deflect-mask-e16.mp4 --px 16` removed
  0.2259 of the mask's pixels, and the file kept the original's 124 frames at 960x544, 24 fps.
- The file was placed beside its siblings as `asset:bench/t16/deflect-mask-e16.mp4`.
  `upload_asset` files a named upload under `uploads/`, so it was copied into the
  workspace's assets directory instead.
- The arms are `t16-swap-e16-*`. They are the g0 arms with `mask_video` set to the
  eroded mask; `source_video` names the same asset as #1374.

| Arm | Job | Run | Wall |
|---|---|---|---|
| e16, seed 1194 | `38bc1fead1fe` | `BenchT16PerformerSwap/20261117-032650-64e9f1ff` | 4.1 min |
| e16, seed 2002 | `b75eda2bd8a8` | `BenchT16PerformerSwap/20261117-033123-2e1e5588` | 4.0 min (warm) |

**Cells.** e16 changes what H3 regenerates, so the N=24 retry the plan held in reserve
was not needed.
- The engine's `regenerated_cells`, from the `Regional edit` log line of 18870 patch
  cells: **g0 7491, e16 5959**.
- Both numbers were predicted offline with `h3_mask_cells` before the run, and the
  eroded file read back to the same 5959.
- For comparison, e24 would have regenerated 5291 cells (0.3255 of the pixels removed).

`python bench/seam_bands.py deflect-cut.mp4 ARM.mp4 deflect-mask.mp4 --inner 2` uses the
**original** mask for every arm. The g1 files were still on lem, under the old bench
workspace.
- `in 17-32` and `in 1-16` are the bands just inside the union mask (-2, -1).
- `inside` is the rest of the inside, so it is not #1374's `inside` column.
- `jump` is the band -1 diff minus the band 1 diff.

| Arm | inside | in 17-32 | in 1-16 | 1-16 | 17-32 | 33-48 | **step** | **jump** |
|---|---|---|---|---|---|---|---|---|
| g1 s1194 | 54.53 | 74.95 | 66.04 | 30.17 | 25.80 | 19.02 | **4.37** | **35.87** |
| g1 s2002 | 51.18 | 77.23 | 71.81 | 27.63 | 23.17 | 18.04 | **4.45** | **44.18** |
| g0 s1194 | 51.96 | 74.71 | 62.49 | 15.24 | 5.42 | 3.13 | **9.81** | **47.25** |
| g0 s2002 | 50.52 | 71.97 | 66.64 | 17.19 | 5.91 | 3.11 | **11.29** | **49.45** |
| e16 s1194 | 47.71 | 55.97 | 33.74 | 5.24 | 3.20 | 2.98 | **2.04** | **28.50** |
| e16 s2002 | 48.45 | 58.96 | 34.71 | 5.34 | 3.19 | 2.92 | **2.16** | **29.37** |

- e16 stays near the source floor (about 2.9) from 17 px outward: the change is confined
  to about 16 px outside the original silhouette.
- Frame-to-frame luma change in the 1-16 px band is 1.12 / 1.14 for e16, against 1.24 in
  the source.

**By eye** (shoulder crop `[560, 200, 400, 260]`, frames 0 and 62):
- **A ghost on both seeds.** A pale, translucent streak runs down the old silhouette, from
  the shoulder to the magazine. The eroded-away rim is kept source, and it still holds the
  outline of the original performer, where the new man's shoulder does not reach.
- **The couch back on s2002.** Beside the shoulder it is still redrawn as a separate
  cushion. On s1194 the couch top line runs on, but the streak crosses it.
- **The rest.** Identity, pose, the page turn and mouth timing follow the source, as at
  g0 and g1.

**Rule** (plan v2, #1382): erosion works only if all four tests pass.

| Test | Result |
|---|---|
| The couch-back step is gone | **Fails**: still redrawn on s2002 |
| `step` is no larger than g1's on both seeds | **Passes**: 2.04 ≤ 4.37 and 2.16 ≤ 4.45 |
| No jump between bands -1 and 1 is larger than g1's band 1-2 step | **Fails**: 28.50 and 29.37 against 4.37 and 4.45 (g1's own jumps are 35.87 and 44.18) |
| No ghost of the source silhouette | **Fails** |

**Conclusion: keep `mask_grow` 1.** A cleaner edge on the outside is paid for with a
visible ghost on the inside. An engine erode would ship the same trade, so none is
proposed.

## The performance on a two-person, two-cut clip (#1374, ep62)

Don's watch item: in the tester's C-F887 run, the swapped person was redrawn standing
square to the camera in the portrait's pose, and the source performance was lost.

**It reproduces, and the cause is the prompt, not the edit.** At `refine_strength` 1.0
nothing of the person survives inside the mask. Pose, action and framing come from the
prompt and the kept context.

- **Prep:** `replace-performer-cuts` job `14e01110c193` on `ep62-episode.mp4` (248 frames).
  - It returned two cuts, 0+124 and 124+124, and they are right. Cut-1 is a two-shot:
    a standing young man in a vest and a seated bald man at a table. Cut-2 is one
    push-in on the bald man, from his hands to his face.
  - The mask covers both people in cut-1 ("person").
- **Run 1:** job `3cd3b2b66fec`, seed 7. The prompt said "<Subject 1> sits on an
  olive-green couch".
  - Both masked people were redrawn seated, square to the camera, in the portrait's
    pose. The standing man's performance and cut-2's push-in were lost: the C-F887
    failure.
  - The kept area held. On cut-1, `seam_bands` gives 3.2 at 81-96 px and 2.9 at 113-128,
    the same VAE floor as deflect. So the edit, trims and pad fed the right frames, and
    the wrong pose came from the prompt.
- **Run 2:** job `1e170e5fab79`, seed 7, same cuts and mask. The prompt names the
  bald man as the one replaced, keeps the vest man as `<Subject 2>` "as in
  `<Video 1>`", and describes the two-shot's action.
  - Cut-1 keeps the performance. The vest man stands, bends to the table and holds out
    the takeout carton, as in the source. The portrait's man sits at the table in the
    bald man's place, turned toward him.
  - Cut-2 does not. The one prompt described cut-1's two-shot, so the single-person
    push-in came back as a wide two-shot.
- **Fix in the template:** a cut may carry its own `prompt`, read with the same
  `{"item", "default"}` form as `seed`. The skill now says to describe each cut's pose
  and framing.

## A mask-free head swap against replace-performer (#1411, #1414)

Arm 1 and the face baseline were run 2026-11-17 on lem cuda:0 (RTX 3090), develop @
4bb60015 (stage B's whole-render guide). Arm 3 was re-run the same day on cuda:1,
develop @ 086aa8eb, with the plan's scene reference. Nothing on arm 3's path (ref2va,
the whole-render guide, the LoRA loader) changed in between. Written by claude-opus-5-5
(anthropic).

**Single seed.** Every number below is seed 1194 only: one run per arm, with no spread.

### Setup

- **Shot.** #1194's deflect cut (`asset:bench/t16/deflect-cut.mp4`, 960x544, 124
  frames, 24 fps). It is a copy of the #1194 fixture BenchS4Montage/20261104-040255-9813688b
  `intermediate/BenchS4Montage-shot@deflect.3-0.0.mp4`. One person is replaced by the
  man in the portrait `asset:bench/t26-stranger.jpg`.
- **Arm 1:** `templates/minimax/replace-performer`.
  - The cut, its person mask `asset:bench/t16/deflect-mask.mp4`, one cut of 124
    frames, and the portrait as the reference.
  - The template's own ref2va settings: turbo LoRA, 8 passes.
- **Arm 3:** `bench/workflows/t16-recast-shot.json`, ref2va with no mask.
  - **The scene reference, as `<Picture 1>`.** This is frame 0 of the cut (grab job
    `86a7c413ce45`), edited with `templates/minimax-h3-image/multi-reference` (job
    `3afa8dbb3be0`, seed 1194, 1920x1088) to show the portrait's man in the performer's
    place. It is kept as `asset:bench/t16/recast-scene-ref.png`. The edit dressed him in
    the portrait's mustard jacket and green tee.
  - The whole clip goes in as one full-length video guide at frame 0 (`<Video 1>`).
  - The head-swap LoRA (`loras/minimax-h3/head-swap.json`) at 1.0, in place of turbo.
  - 20 steps, CFG 1, video_shift 12, turbo off; int4 with sparse attention. The audio
    is kept with `hold_audio`.
  - The prompt is the LoRA author's structured `head_swap:` template (the repo's
    `docs/head-swap.md`), with the portrait's facial details filled in.
- **Face hold** is scored against the portrait on both arms.
- Arms: `bench/arms/t16-recast-shot.json`. Arm 1 was run by template name over MCP.
  `bench/workflows/t16-recast-replace-performer.json` is a snapshot of the template at
  4bb60015, with that run's arguments as its defaults, because `bench/driver.py` reads
  `bench/workflows/` only.

### Numbers

| Arm | Job | Wall (quoted) | Face hold, SFace median (min) | PSNR, more than 96 px outside the person | Seam band needed | Audio corr (gain) |
|---|---|---|---|---|---|---|
| Source cut (baseline) | face `66b2175675da` | - | 0.056 (0.013) | - | - | - |
| 1: replace-performer | `9c0fd3314b01` (face `36575d4889d7`) | 4.2 min (6.2) | **0.747** (0.695) | **37.5 dB** | yes: the mask edge, step 4.71 | 0.9991 (-0.02 dB) |
| 3: head-swap LoRA + frame-0 guide + scene reference | `7d1c0393d177` (face `54231da0b116`) | 26.3 min (17.9) | **0.543** (0.499) | **23.1 dB** | no: no mask, no edge (step 2.10) | 0.9991 (-0.02 dB) |

- **Face hold.** Each face check is `templates/face-drift-check` with 16 frames sampled
  and threshold 0.363. All 16 frames found a face on both arms, and neither arm drew a
  `face_drift` finding.
  - Arm 1's 0.747 is in line with the earlier masked swap at this seed (0.759).
- **PSNR by region.** From a scratch numpy script (not committed). Regions are taken
  from the union of the person mask over every frame, the same mask `seam_bands` uses.
  Arm 1 and the earlier portrait run reproduce to 0.1 dB from the same script.

  | Region | Arm 1 | Arm 3 |
  |---|---|---|
  | Inside the mask | 10.8 dB | 12.0 dB |
  | 1-32 px outside | 15.7 dB | 21.7 dB |
  | More than 96 px outside | 37.5 dB | 23.1 dB |

- **Where the change stops** (`bench/seam_bands.py`, 16 px bands, mean abs diff).
  - **Arm 1:** 58.5 inside, then 30.7, 26.0, 18.9, 11.2 and 5.2, settling at about 2.9
    from 81-96 px. That is the same edge as the earlier deflect runs, then the VAE floor.
  - **Arm 3:** 48.8 inside, 14.0 at 1-16 px, then 10-14 out to 256 px and 4-9
    beyond.
  - There is no edge to seam in arm 3, but nothing in the frame is left as it was. The
    background is re-rendered at roughly 3-4x the VAE floor. Flicker outside 16 px tracks
    the source's band for band (0.02-0.29 against 0.04-0.33), so this is a steady
    redraw, not shimmer.
- **Audio.** Both tracks correlate 0.9991 with the source, so `hold_audio` held on both.
- **Cost.** Arm 3's quote was inherited (17.9 min); it ran 26.3 min. Its 20 passes
  took 65.2 s each after a first pass of 316.2 s, plus 23.0 s of decode and 56.7 s of
  encode. Arm 1 ran 4.2 min against its 6.2 min quote.

### By eye (frames 0, 62 and 123 of each output, for Don)

Outputs (workspace `bench`):
- Arm 1: `templates/minimax/replace-performer/20261117-193659-838443b8/final/MiniMaxH3ReplacePerformer-replaced.5-0.0.mp4`
- Arm 3: `BenchT16RecastShot/20261117-223645-41345183/final/recast-0.0.mp4`

- **Arm 1.** A whole-person swap: the portrait's man in his mustard jacket and green tee,
  with glasses and a beard, in the source pose and page turn.
- **Arm 3.** Also a whole-person swap, now taken from the scene reference: the jacket,
  the green tee, the glasses, the grey beard. The couch, the brick wall, the magazine and
  the page turn are kept, and the swap holds through the shot.
  - The face reads as a cousin of the portrait rather than the same man: fuller in the
    cheeks, with a heavier mustache. The 0.543 agrees.
  - The room is redrawn rather than passed through, so the brick and couch texture
    differ slightly from the source at every frame.

### Against the gate for stage D

| Criterion | Arm 3 against arm 1 | Result |
|---|---|---|
| Face hold within 0.03 of arm 1's | 0.543 against 0.747 (-0.204) | **fail** |
| Untouched regions within 2 dB of arm 1's | 23.1 against 37.5 dB (-14.4) | **fail** |
| No seam band needed | no mask, so no edge | pass |
| Don prefers arm 3 on the frames | - | **pending (Don)** |

**Arm 3 fails the gate on the measured criteria, on one seed.** A preference from Don
cannot carry it past two measured failures. The head-swap LoRA's catalog entry stays
`rejected`, now with this evidence instead of "not tried".

**Deviations from the plan:**
- **Re-pin.** The LoRA's pinned revision `849fb3d4` no longer exists: the repo's history
  was rewritten, and the first arm-3 run (`0514e2ebbea1`) failed on the missing file.
  The entry and the workflow are re-pinned to `61213345`, the commit that last wrote the
  file (310,177,072 bytes, which passed the Hub's scans).
- **A supplementary portrait run, outside the gate.** Before the re-run, arm 3 was run
  once with the face portrait itself as `<Picture 1>` (the LoRA author's documented
  input) instead of the scene reference: job `bcca0dababf5` (face `a7a8496dd0eb`), on
  cuda:0 at 4bb60015, 19.6 min against a 9.1 min quote. It swapped the head only and kept
  the source's grey flannel body. Its face hold was 0.653 (min 0.574). Its PSNR was
  17.6 / 25.6 / 29.5 dB inside / 1-32 px / more than 96 px, with a step of 2.20 and an
  even redraw of about 8 outside 16 px. That run still fails both measured criteria
  (-0.094 face, -8.0 dB), so the verdict does not depend on which picture is used. It is
  kept for the comparison only. The gate rows above are the plan's arm, the scene
  reference.

## track_people: Grounding-DINO against SAM 3 (#1397 stage B, #1405)

The mini-ai arms were run 2026-11-18 on mps (Apple M5 Pro), develop @ c10048cd, in
workspace `bench`, one after another on one worker. The lem column waits for the lem loop:
lem's hold on the transformers 5.21 upgrade has not been lifted. Written by
claude-opus-5-5 (anthropic).

### Setup

- **Workflow.** `bench/workflows/track-people-backend-ab.json`: one `track_people` step
  with prompt `person`, `max_people` 4, and the backend and threshold set per arm.
- **Arms.** `bench/arms/track-people-backend-ab.json`. Each arm uses its backend's own
  default threshold (`DEFAULT_THRESHOLDS`): DINO 0.3, SAM 3 0.5. The scales differ, so
  equal numbers would not mean an equal bar.
- **Fixtures.**
  - **ep65 dissolve.** `asset:qa-cast/ep65-shot-kf-priya-hal.mp4`: 121 frames, one shot.
    The first man is alone at frame 0. The grey-jacket man enters at the left edge around
    frame 10 and has taken the frame by about 2.9 s, when the first man is gone.
  - **ep62 pair.** `asset:qa-cast/ep62-episode.mp4`: 248 frames, 2 cuts. A two-shot,
    then a push-in on one man.
  - **ep49.** `asset:qa-cast/ep49-episode.mp4`: 372 frames, 3 cuts, one figure per cut.
- **Measures.**
  - *Coverage* is read from the contact sheet and an 8-frame sheet of every `person-N`
    mask.
  - *Late entrant* asks whether someone who appears after a cut's first frame gets a
    mask.
  - *Holds* asks whether each `person-N` stays one figure through ep65's transition.
  - *Wall* is the job's finish time minus its start time.
  - *Peak reserved* is the highest `gpu_memory_reserved_mb` in the job's memory events.
    The worker keeps models cached, so from the first sam3 job on, each reading includes
    both backends' resident weights (about 5.3 GB allocated). Read the numbers as
    host-level, not per-backend.

### The 12 cells

| Fixture | Arm | Host | Job | Threshold | People (per cut) | Coverage | Late entrant | Holds | Wall | Peak reserved |
|---|---|---|---|---|---|---|---|---|---|---|
| ep65 dissolve | dino | mini-ai | `7be915401780` | 0.3 | 1 | Full body of the first man until he leaves (about 2.9 s). Nothing after that. | **missed**: the grey-jacket man has no mask | yes, person-1 is only the first man | 94 s | 9.1 GB |
| ep65 dissolve | sam3 | mini-ai | `fc4db3d061db` | 0.5 | 2 | person-1 matches DINO's. person-2 runs from the edge sliver at frame 10 to the full figure at 5.0 s. | **caught**: person-2, first_frame 10, score 0.626 | yes, one figure per mask through the transition | 320 s | 10.7 GB |
| ep62 pair | dino | mini-ai | `d27f080d8433` | 0.3 | 2 / 1 | Both men in the two-shot, and the push-in's man. Tight edges. | none in the clip | yes | 222 s | 13.9 GB |
| ep62 pair | sam3 | mini-ai | `9b71cf7e89de` | 0.5 | 2 / 1 | The same figures and boxes within 3 px. The contact sheet is indistinguishable from DINO's. | none in the clip | yes | 608 s | 12.1 GB |
| ep49 | dino | mini-ai | `8b26b9a32774` | 0.3 | 2 / 1 / 1 | Each cut's figure, plus a small left-edge detection in shot 1 (score 0.40, 0.8% of the frame). | none in the clip | yes | 315 s | 13.9 GB |
| ep49 | sam3 | mini-ai | `184d68f81c12` | 0.5 | 2 / 1 / 3 | DINO's figures plus the same shot-1 edge detection (0.70). Shot 3 adds two **false entrants**: person-2 at frame 20 (cabinet edge, score 0.501) and person-3 at frame 46 (a photo on the fridge, 0.545). | the two false entrants above | yes | 917 s | 12.4 GB |
| ep65 dissolve | dino | lem | pending (lem loop) | 0.3 | | | | | | |
| ep65 dissolve | sam3 | lem | pending (lem loop) | 0.5 | | | | | | |
| ep62 pair | dino | lem | pending (lem loop) | 0.3 | | | | | | |
| ep62 pair | sam3 | lem | pending (lem loop) | 0.5 | | | | | | |
| ep49 | dino | lem | pending (lem loop) | 0.3 | | | | | | |
| ep49 | sam3 | lem | pending (lem loop) | 0.5 | | | | | | |

### Reading

- **Late entrants.** SAM 3 is the only backend that sees someone who arrives mid-cut.
  DINO detects on each cut's first frame and then propagates, so the ep65 entrant gets no
  mask. A performer replacement on that clip could not target him.
- **Where nobody arrives, the two agree.** On ep62 and ep49 the backends find the same
  people with boxes within a few pixels. SAM 3's scores sit near 0.98, DINO's at
  0.80-0.88.
- **SAM 3's re-detection catches false entrants as well as real ones.** Both ep49 extras
  are background, scored just above the 0.5 floor (0.501 and 0.545). A floor of 0.55
  would have dropped both and kept ep65's real entrant (0.626). That margin rests on
  three scores from one run each, which is too thin to move the default.
- **Cost on mps.** SAM 3 takes 2.7x (ep62) to 3.4x (ep65) DINO's wall time. Peak
  reserved memory is close: within about 2 GB either way once both are cached.
- **The extra threshold run was not used.** The plan allows one more sam3 threshold if
  the first misses the late entrant. 0.5 caught him, so it was not needed.

**Recommendation (mini-ai, pending lem's column): keep SAM 3 as an option (`backend:
"sam3"`) for clips where someone enters mid-cut, and keep Grounding-DINO as the default.
SAM 3 matches DINO on every other cut, but costs about 3x the time on mps and adds
near-floor false entrants that need checking on the contact sheet.**
