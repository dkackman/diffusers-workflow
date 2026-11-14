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
