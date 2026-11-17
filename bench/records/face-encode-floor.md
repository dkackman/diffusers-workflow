# Face encode floor - #1418 stage A (#1420)

How far a face score moves between the in-memory clip a job scores and the
saved file a later job scores, split by cause. No engine change; the numbers
behind `face-encode-floor.json`.

## Setup

- Server: mini-ai (mps, Apple Silicon), develop @ 9be01f4e. `analyze_face_drift`
  runs YuNet + SFace on the CPU, so these numbers don't depend on the accelerator.
- Workflows: `bench/workflows/probe-1418-face-encode-floor-prep.json` (one per
  clip: a `video_frames` decode saved as dw's mp4, the reference saved as png
  through `strip_exif`, and that png's pixels saved as jpg at PIL's defaults,
  q75 4:2:0), then `bench/workflows/probe-1418-face-encode-floor.json` (one per
  clip: six arms reading the prep run's files back as `output:`).
- Arms, all with the same reference face, `samples_per_shot` 16 and the default threshold:
  - **FM/RP**: in-memory frames (this job's decode, held as a clip by `trim_video`) against the png
  - **FM/RJ**: in-memory frames against the jpg
  - **FD/RP**: the saved mp4 decoded back into memory by `video_frames` and held
    by `trim_video` exactly as FM is, against the png
  - **FD/RJ**: the same against the jpg
  - **FF/RP**, **FF/RJ**: the saved mp4 read directly as `output:`. Listed apart
    (source 4, below), not used for the floor.
- Δ = later arm minus FM/RP, as median / min. The floor pairs are video encode
  (FD/RP), reference encode (FM/RJ) and the #1400 combination (FD/RJ). Every
  floor pair has an equal shot record: `shots_source` (`artifact`), shot
  (`video 1`), `start_frame` 0, `num_frames` and `frames_sampled` 16. The floor is
  the largest |Δ| over the clips.

## Runs

| Clip | Shape | Prep job | Arms job (run) |
|---|---|---|---|
| ep63-shot-ltx-priya (priya ref) | 960x544, 121 f | b7b30ea87fd3 | 04101689daea (20261117-150235-e418912a) |
| ep83-shot-refined-priya (priya ref) | 1024x576, 121 f | b8f6c6f82923 | b7e19db6542e (20261117-150350-474c9aa0) |
| ep64-shot-ltx-hal (hal ref) | 960x544, 121 f | 972200f30470 | e748779dd2bd (20261117-150356-c90bdf07) |
| ep89-shot-ltx-hal-cont (hal ref) | 960x544, 57 f | a3812c2e83ca | 80a76fb3c411 (20261117-150401-54b3514f) |
| priya-hal-two-shot (priya ref) | 960x544, 121 f | b3899920277c | aded186053e3 (20261117-150405-dba6f19b) |

Arms runs are under `BenchProbe1418FaceEncodeFloor/`, prep runs under
`BenchProbe1418FaceEncodeFloorPrep/`. The first build's four-arm runs
(6e8242acc204, db45288c3781, 3a9a3a1eb26e, 6d022184883c, 447449bb16ba) are
superseded: they took the floor from FM vs FF, whose shot records differ
(bounce C-F992). Their FM and FF numbers equal the re-run's exactly.

## Scores (median / min)

Every arm sampled 16 frames and found a face in all 16.

| Clip | FM/RP | FM/RJ | FD/RP | FD/RJ | FF/RP | FF/RJ |
|---|---|---|---|---|---|---|
| ep63 | 0.597 / 0.496 | 0.598 / 0.497 | 0.601 / 0.480 | 0.603 / 0.481 | 0.601 / 0.480 | 0.603 / 0.481 |
| ep83 | 0.176 / 0.091 | 0.176 / 0.093 | 0.192 / 0.067 | 0.192 / 0.069 | 0.192 / 0.067 | 0.192 / 0.069 |
| ep64 | 0.559 / 0.486 | 0.558 / 0.487 | 0.573 / 0.456 | 0.573 / 0.456 | 0.573 / 0.456 | 0.573 / 0.456 |
| ep89 | 0.200 / 0.137 | 0.200 / 0.137 | 0.194 / 0.140 | 0.194 / 0.140 | 0.194 / 0.140 | 0.194 / 0.140 |
| two-shot | 0.210 / 0.177 | 0.209 / 0.176 | 0.216 / 0.160 | 0.215 / 0.160 | 0.216 / 0.160 | 0.215 / 0.160 |

ep83, ep89 and the two-shot score low and raise a `face_drift` warn in every
arm, the same in all six. Faces were found, so they are scored here, not
excluded. In the two-shot the largest face can be Hal's, not Priya's.

## Δ against FM/RP (median / min)

| Clip | Video encode (FD/RP) | Reference encode (FM/RJ) | Combination (FD/RJ) |
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

ep63 was run twice more with fresh inline workflow ids (the same six-arm steps;
only the description text differs):

- `BenchProbe1418FaceEncodeFloorR3`: job 860c33a9568c, run 20261117-150432-80bad5f7
- `BenchProbe1418FaceEncodeFloorR4`: job 89c03ef33402, run 20261117-150437-1a06a645

All twelve numbers match the ep63 run, so the **repeat gap is 0.000**, and the
floor needs no repeat term. (The first build's four-arm repeats, b80dc7123c69 and
b0067d92b97d, also gave 0.000.)

## Source-4 differences (listed apart)

The FF arms read the same saved mp4 as FD, but as a file, so they carry a
different shot record: `shots_source` `none`, shot `whole`, `start`/`end` in
seconds (0.0–5.042; ep89 0.0–2.375), and the note "no shot boundaries found;
the whole file was measured as one shot". FM and FD carry `artifact`: one shot
`video 1` from `trim_video`, with `start`/`end` null because the in-memory decode
carries no fps. `start_frame` 0, `num_frames` and `frames_sampled` 16 are equal
in all six arms.

Same pixels, two shot records: FF minus FD, per clip and reference (median / min).

| Clip | FF/RP − FD/RP | FF/RJ − FD/RJ |
|---|---|---|
| ep63 | 0.000 / 0.000 | 0.000 / 0.000 |
| ep83 | 0.000 / 0.000 | 0.000 / 0.000 |
| ep64 | 0.000 / 0.000 | 0.000 / 0.000 |
| ep89 | 0.000 / 0.000 | 0.000 / 0.000 |
| two-shot | 0.000 / 0.000 | 0.000 / 0.000 |

So the FM/FF shot-record difference is a label only: it samples the same frames
and gives the same scores.

## Excluded

None. No clip had a frame without a face.

## Bound

This is a **lower bound**. Each saved file is a re-encode of a clip that had
already been encoded once, decoded back to frames, which is one extra libx264
pass over frames that already went through the codec. An H3 clip scored in-job
against its own first save on lem, decoder output that was never encoded before,
would give the **upper bound**.

#1400 saw the in-job and saved scores of one H3 clip differ by up to 0.045 on a
minimum and 0.015 on a median (`docs/proposals/audits/2026-11-face-once-sheet.md`,
S/1002). This lower bound's 0.030 min is below that 0.045, which is what a
re-encode of already-encoded frames is expected to give; it does not contradict
the H3 number. Its 0.016 median floor is about the audit's 0.015.

## Re-measure

The arms are deterministic (repeat gap 0.000). Run
`probe-1418-face-encode-floor.json` with a clip's prep-run files as
`saved_video` / `reference_png` / `reference_jpg`, and every arm's median and min
(FD and FF included) should match the table exactly.
