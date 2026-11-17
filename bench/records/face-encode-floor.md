# Face encode floor - #1418 stage A (#1420)

How far a face score moves between the in-memory clip a job scores and the
saved file a later job scores, split by cause. No engine change; the numbers
behind `face-encode-floor.json`.

## Setup

- Server: mini-ai (mps, Apple Silicon), develop @ ad4871e1. `analyze_face_drift`
  runs YuNet + SFace on the CPU, so these numbers don't depend on the accelerator.
- Workflows: `bench/workflows/probe-1418-face-encode-floor-prep.json` (one per
  clip: a `video_frames` decode saved as dw's mp4, the reference saved as png
  through `strip_exif`, and that png's pixels saved as jpg at PIL's defaults,
  q75 4:2:0), then `bench/workflows/probe-1418-face-encode-floor.json` (one per
  clip: four arms reading the prep run's files back as `output:`).
- Arms, all with the same reference face, `samples_per_shot` 16 and the default threshold:
  - **FM/RP**: in-memory frames (this job's decode, held as a clip by `trim_video`) against the png
  - **FM/RJ**: in-memory frames against the jpg
  - **FF/RP**: the saved mp4 against the png
  - **FF/RJ**: the saved mp4 against the jpg
- Δ = later arm minus FM/RP, as median / min. The pairs are video encode (FF/RP),
  reference encode (FM/RJ) and the #1400 combination (FF/RJ). The floor is the
  largest |Δ| over the clips.

## Runs

| Clip | Shape | Prep job | Arms job (run) |
|---|---|---|---|
| ep63-shot-ltx-priya (priya ref) | 960x544, 121 f | b7b30ea87fd3 | 6e8242acc204 (20261117-144112-6f39f0b8) |
| ep83-shot-refined-priya (priya ref) | 1024x576, 121 f | b8f6c6f82923 | db45288c3781 (20261117-144452-b81262c4) |
| ep64-shot-ltx-hal (hal ref) | 960x544, 121 f | 972200f30470 | 3a9a3a1eb26e (20261117-144457-b9f3fc75) |
| ep89-shot-ltx-hal-cont (hal ref) | 960x544, 57 f | a3812c2e83ca | 6d022184883c (20261117-144501-071e2285) |
| priya-hal-two-shot (priya ref) | 960x544, 121 f | b3899920277c | 447449bb16ba (20261117-144504-d11bbf3b) |

Arms runs are under `BenchProbe1418FaceEncodeFloor/`, prep runs under
`BenchProbe1418FaceEncodeFloorPrep/`.

## Scores (median / min)

Every arm sampled 16 frames and found a face in all 16.

| Clip | FM/RP | FM/RJ | FF/RP | FF/RJ |
|---|---|---|---|---|
| ep63 | 0.597 / 0.496 | 0.598 / 0.497 | 0.601 / 0.480 | 0.603 / 0.481 |
| ep83 | 0.176 / 0.091 | 0.176 / 0.093 | 0.192 / 0.067 | 0.192 / 0.069 |
| ep64 | 0.559 / 0.486 | 0.558 / 0.487 | 0.573 / 0.456 | 0.573 / 0.456 |
| ep89 | 0.200 / 0.137 | 0.200 / 0.137 | 0.194 / 0.140 | 0.194 / 0.140 |
| two-shot | 0.210 / 0.177 | 0.209 / 0.176 | 0.216 / 0.160 | 0.215 / 0.160 |

ep83, ep89 and the two-shot score low and raise a `face_drift` warn in every
arm, the same in all four. Faces were found, so they are scored here, not
excluded. In the two-shot the largest face can be Hal's, not Priya's.

## Δ against FM/RP (median / min)

| Clip | Video encode (FF/RP) | Reference encode (FM/RJ) | Combination (FF/RJ) |
|---|---|---|---|
| ep63 | +0.004 / −0.016 | +0.001 / +0.001 | +0.006 / −0.015 |
| ep83 | +0.016 / −0.024 | 0.000 / +0.002 | +0.016 / −0.022 |
| ep64 | +0.014 / −0.030 | −0.001 / +0.001 | +0.014 / −0.030 |
| ep89 | −0.006 / +0.003 | 0.000 / 0.000 | −0.006 / +0.003 |
| two-shot | +0.006 / −0.017 | −0.001 / −0.001 | +0.005 / −0.017 |
| **floor (max \|Δ\|)** | **0.016 / 0.030** | **0.001 / 0.002** | **0.016 / 0.030** |

- **Median floor: 0.016.** Nearly all of it is the video encode. The reference
  saved as jpg moves the median by 0.001 at most.
- **Min floor: 0.030.** The saved file's minimum runs lower than the in-memory
  one in four of the five clips: one bad encoded frame pulls the minimum down
  while the median rises.
- **Is the median floor ≥ 0.02** (the trigger for the chain `face_drop`
  follow-up)? **No**, 0.016.
- **Is it ≥ 0.03** (which brings option (b) back)? **No.**

## Repeat gap

ep63 was run twice more with fresh inline workflow ids (the same steps; only
the description text differs):

- `BenchProbe1418FaceEncodeFloorR1`: job b80dc7123c69, run 20261117-144549-fc26687b
- `BenchProbe1418FaceEncodeFloorR2`: job b0067d92b97d, run 20261117-144553-2f1de1d8

All eight numbers match the first ep63 run, so the **repeat gap is 0.000**, and
the floor needs no repeat term.

## Source-4 differences (listed apart)

- `frames_sampled` (16) and `num_frames` are equal between FM and FF in every clip.
- `shots_source` differs. FM is `artifact`: one shot `video 1` from `trim_video`,
  with `start`/`end` null because the in-memory decode carries no fps. FF is
  `none`: shot `whole`, `start`/`end` in seconds, and the note "no shot
  boundaries found; the whole file was measured as one shot". Sampling is by
  frame index, so the same indices are scored in both.

## Excluded

None. No clip had a frame without a face.

## Bound

This is a **lower bound**. Each saved file is a re-encode of a clip that had
already been encoded once, decoded back to frames, which is one extra libx264
pass over frames that already went through the codec. An H3 clip scored in-job
against its own first save on lem, decoder output that was never encoded before,
would give the **upper bound**.

## Re-measure

An FF arm is deterministic (repeat gap 0.000). Run
`probe-1418-face-encode-floor.json` with a clip's prep-run files as
`saved_video` / `reference_png` / `reference_jpg`, and the `ff_rp` / `ff_rj`
median and min should match the table exactly.
