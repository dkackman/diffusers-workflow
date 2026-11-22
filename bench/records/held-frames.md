# Held frames in Don's H3 takes: is 12-shown-as-24 judder real demand? (#1527, stage #1551)

lem, CPU only, 2026-11-22. The probe ran at commit `1587a690` (`bench/held_frames.py`) against lem's main checkout at
`f5e2b5fb` (`dw.tasks.assess.read_media`, `dw.tasks.scene_cuts.content_cuts`), numpy 2.5.4 and OpenCV 5.0.0. The
measurements are in `held-frames.json` next to this file.

## Verdict: the gate after A is met, so B proceeds

The plan's gate says B and C are cut if no take has a cluster, or if no take has 7 or more held frames once static shots and cuts
are excluded. Both conditions are cleared:

- **79** demand H3 takes have a cluster at ε 1.0.
- **10** takes that are not near-static have 7 or more held frames at ε 1.0. The highest is a `soviet-anthem` fanfare
  take: 35 held frames, in 7 clusters across its 9.4 s.

Held frames are rare overall. **510 of 649** takes hold nothing at ε 1.0, and the median is 0 in every mode and schedule.
They concentrate in **5-step** takes (`single` 5: 6 of 102 takes reach 7 held) and in a few 9-step and 12-step ones.
Whether a viewer sees them as judder is the Q3 `taste` question for Don.

## Twos or fours: twos, often one every 4th frame

- **Twos.** Of the 149 hold runs bounded by moving frames on both sides, **139 have length 1**: one frame shown twice.
  - Only 3 have length 3 (a frame shown four times), and 1 has length 2.
  - The 6 runs of 49 and 78 frames are still stretches inside near-static joined takes, not holds.
- **The recurring spacing is 4 frames.** Where a take holds more than once, the most common distance between successive
  holds is 4 frames: 68 of the spacings, against 4 at 2 frames.
  - It comes mostly from the `h3-quality` H3Ladder and H3Q1 takes (4 to 8 each) and from the two `known-issues-voiced`
    takes (10 holds each, 7 of them 4 frames after the one before).
  - So the pattern is one repeated frame per 4-frame group. That would fit H3's temporal latent stride if it is 4: a
    hypothesis, not checked here. It is not a 12 fps cadence.

**For stage C:** the repeat to synthesize away is a single frame, and it recurs on a 4-frame stride. That matches a hold per
latent frame, not every other frame. A 2-frame stride is the wrong period for these takes. This is a measurement, not a
cause: whether the 4-frame group comes from the VAE decode or the sampler is not settled here.

**Chain segments** hold in pairs 2 frames apart, about every 44 to 50 frames: at 1.83-1.92 s, 3.67-3.75 s and
5.75-5.83 s in `narrated-arc-latent` segment 001, with no bounded run. The frame between each pair has d ≥ ε but below
2.0, so the picture slows almost to a stop for three frames rather than repeating one. 22 of the 203 20-step segments have
a cluster, and none reaches 7 held frames. This is a note for B, not a claim.

## Method

    # on lem, from any directory; nothing is written under ~/diffusers-workspace
    PYTHONPATH=~/diffusers-workflow ~/diffusers-workflow/venv/bin/python bench/held_frames.py \
        scan ~/diffusers-workspace --out /tmp/held-frames.json --jobs 16
    python bench/held_frames.py report /tmp/held-frames.json      # the tables below, verbatim

- **Corpus:** every video under `outputs/` in the plan's 12 demand workspaces. `default` is `~/diffusers-workspace/outputs`.
  `qa-*`, `regression-*`, `drill-*` and `probe-*` were not scanned.
- **H3 or not:** a video is H3 when its run's realized `workflow.json`, or a sub-workflow the manifest names, loads
  `MiniMaxAI/MiniMax-H3`. 918 videos are listed and 649 of them are H3. The JSON lists the other 269 with `h3: false`.
- **The measure** is the plan's, and the module docstring states it:
  - d_i is the mean absolute difference of 64x36 grey thumbnails;
  - a frame is held when d_i < ε and a neighbouring d ≥ 2.0;
  - frames within one frame of a `content_cuts` cut are excluded;
  - a cluster is at least 2 holds whose gaps are 4 frames or fewer.
- **Mode:**
  - `chain segment`: a `.segment-NNN` file;
  - `joined`: an H3 output stitched from more than one shot carrying `segment_file`;
  - `cuts`: a task output of more than one shot without segments;
  - `derived`: a task output of one shot, such as a trim, a mux or an upscale;
  - `single`: an H3 pipeline output;
  - `no manifest`: a run that has only `workflow.json` (12 `shedding-skin-quartet` trailer shots).
- **Steps:** `num_inference_steps` of the H3 step, with the run's arguments applied.
- **Rerun:** a second scan reproduced the first line for line. The only difference was the 12 no-manifest takes, which the
  second scan added.

## Reading the numbers

- **Static share.** A near-static take, with fewer than 10% of its counted frames moving (d ≥ 2), still "holds" the
  still frames either side of a rare jump. The gate table marks these takes and does not count them.
  - The 10% threshold is the probe's choice, not the plan's.
  - Without that exclusion, the count of takes with 7 or more held frames rises from 10 to 16. Either way the verdict
    does not change.
- **The rule counts only a run's edges.** For a hold of length 3, only the frames either side are marked held, because a
  middle frame has still neighbours. That is why the run lengths, not the held count, answer the twos-or-fours question.

## Tables

### Corpus: videos per workspace

| Workspace | Demand | H3 videos | Other videos | Measured | Errors |
|---|---|---|---|---|---|
| known-issues | yes | 40 | 17 | 40 | 0 |
| h3-quality | yes | 167 | 7 | 167 | 0 |
| constitution-walk | yes | 9 | 16 | 9 | 0 |
| constitution-walk-2 | yes | 93 | 27 | 93 | 0 |
| constitution-walk-3 | yes | 164 | 0 | 164 | 0 |
| inner-space | yes | 11 | 3 | 11 | 0 |
| soviet-anthem | yes | 20 | 0 | 20 | 0 |
| russian-anthem | yes | 22 | 0 | 22 | 0 |
| shedding-skin-quartet | yes | 18 | 2 | 18 | 0 |
| acorn-wars | yes | 14 | 4 | 14 | 0 |
| sci0fi | yes | 3 | 13 | 3 | 0 |
| default | yes | 88 | 180 | 88 | 0 |

### Held frames per mode and schedule (demand H3 takes, n=649)

| Mode | Steps | Takes | held=0 @0.5 | held=0 @1.0 | held=0 @1.5 | median held @1.0 | max held @1.0 | takes with a cluster @1.0 | takes >= 7 held @1.0 | d p50 median |
|---|---|---|---|---|---|---|---|---|---|---|
| chain segment | 20 | 203 | 181 | 172 | 65 | 0 | 6 | 22 | 0 | 2.31 |
| chain segment | 5 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 1.62 |
| chain segment | 9 | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0.35 |
| cuts | 20 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 3.10 |
| cuts | 5 | 8 | 6 | 5 | 0 | 0 | 8 | 3 | 1 | 1.36 |
| derived | 12 | 30 | 27 | 25 | 20 | 0 | 4 | 4 | 0 | 0.31 |
| derived | 20 | 2 | 0 | 0 | 0 | 4 | 4 | 2 | 0 | 0.30 |
| derived | 30 | 26 | 17 | 17 | 15 | 0 | 5 | 8 | 0 | 0.33 |
| derived | 5 | 42 | 42 | 37 | 14 | 0 | 2 | 0 | 0 | 1.58 |
| derived | 8 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.24 |
| derived | 9 | 11 | 10 | 9 | 6 | 0 | 21 | 1 | 1 | 0.67 |
| joined | 20 | 34 | 26 | 21 | 0 | 0 | 18 | 7 | 5 | 2.45 |
| joined | 5 | 1 | 0 | 0 | 0 | 4 | 4 | 1 | 0 | 1.68 |
| joined | 9 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.46 |
| no manifest | 9 | 12 | 12 | 11 | 7 | 0 | 1 | 0 | 0 | 2.39 |
| single | 12 | 36 | 30 | 27 | 22 | 0 | 11 | 6 | 2 | 0.33 |
| single | 20 | 24 | 22 | 19 | 2 | 0 | 4 | 2 | 0 | 2.46 |
| single | 30 | 27 | 18 | 18 | 16 | 0 | 5 | 8 | 0 | 0.33 |
| single | 5 | 102 | 90 | 68 | 24 | 0 | 35 | 11 | 6 | 1.42 |
| single | 8 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.24 |
| single | 9 | 80 | 75 | 71 | 51 | 0 | 20 | 4 | 1 | 1.03 |

### Hold run lengths (demand H3 takes, eps 1.0)

A run of 1 is a frame shown twice, 3 is a frame shown four times (runs bounded by moving frames on both sides).

| Run length | Runs |
|---|---|
| 1 | 139 |
| 2 | 1 |
| 3 | 3 |
| 49 | 4 |
| 78 | 2 |

| Spacing between run starts (frames) | Count (top 6 per take) |
|---|---|
| 2 | 4 |
| 4 | 68 |
| 5 | 1 |
| 8 | 2 |
| 9 | 2 |
| 10 | 1 |
| 12 | 1 |
| 14 | 1 |
| 24 | 5 |
| 43 | 1 |
| 48 | 1 |
| 49 | 1 |
| 57 | 1 |
| 64 | 2 |
| 100 | 1 |
| 102 | 1 |
| 118 | 1 |
| 639 | 1 |

### Highest-count takes (eps 1.0)

| Workspace | Gallery name | Mode | Steps | Frames | held @0.5/1.0/1.5 | Clusters (start-end s: held) | d p10/p50/p90 |
|---|---|---|---|---|---|---|---|
| soviet-anthem | `SovietAnthemV4Fanfare/20261012-023309-62e08190/final/SovietAnthemV4Fanfare-first_and_last_frame_to_video_audio.0-0.0.mp4` | single | 5 | 226 | 12/35/48 | 0.79-1.38: 5, 1.88-2.67: 7, 3.38-3.46: 2, 3.88-4.38: 4 (+3) | 0.232/1.1/2.679 |
| known-issues | `known-issues-voiced/20260911-134446-f00a7b44/KnownIssuesVoicedShort-episode.7-0.0.mp4` | derived | 9 | 637 | 19/21/33 | 10.46-11.29: 6, 11.79-12.46: 5, 12.96-14.12: 8, 14.46-14.54: 2 | 0.251/0.76/2.668 |
| known-issues | `known-issues-voiced/20260911-134446-f00a7b44/KnownIssuesVoicedShort-shot_3_howie_incredulous.4-0.0.mp4` | single | 9 | 124 | 18/20/22 | 0.12-0.79: 5, 1.46-2.12: 5, 2.62-3.79: 8, 4.12-4.21: 2 | 0.265/1.252/4.721 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc4-latent/20261112-141747-0484225a/final/BenchNarratedTake-narrated-arc4-latent-take.1-0.0.mp4` | joined | 20 | 2062 | 11/18/19 | 9.83-9.92: 2, 11.67-11.75: 2, 13.75-13.83: 2, 22.12-22.21: 2 (+5) | 0.213/0.431/0.836 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc/20261111-233536-161c8b80/final/BenchNarratedTake-narrated-arc-take.1-0.0.mp4` | joined | 20 | 2062 | 9/13/17 | 7.75-7.83: 2, 9.83-9.92: 2, 11.71-11.79: 2, 77.04-77.12: 2 (+1) | 0.293/0.537/1.076 |

### Nearest the 6 and 12 held marks (eps 1.0)

| Workspace | Gallery name | Mode | Steps | Frames | held @0.5/1.0/1.5 | Clusters (start-end s: held) | d p10/p50/p90 |
|---|---|---|---|---|---|---|---|
| constitution-walk-3 | `BenchNarratedTake-narrated-arc-latent/20261112-040003-f9eb4c89/final/BenchNarratedTake-narrated-arc-latent-take.0.segment-001.mp4` | chain segment | 20 | 170 | 6/6/6 | 1.83-1.92: 2, 3.67-3.75: 2, 5.75-5.83: 2 | 0.233/0.464/0.977 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc-latent/20261112-040003-f9eb4c89/final/BenchNarratedTake-narrated-arc-latent-take.1-0.0.mp4` | joined | 20 | 2062 | 9/12/24 | 7.75-7.83: 2, 9.83-9.92: 2, 11.67-11.75: 2, 13.75-13.83: 2 (+1) | 0.303/0.681/1.911 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc4-latent/20261112-141747-0484225a/final/BenchNarratedTake-narrated-arc4-latent-take.0.segment-001.mp4` | chain segment | 20 | 170 | 6/6/6 | 1.83-1.92: 2, 3.67-3.75: 2, 5.75-5.83: 2 | 0.162/0.344/0.811 |

### The gate: takes with >= 7 held frames (eps 1.0)

A take whose moving share (counted frames with d >= 2) is under 0.1 is a near-static shot: its holds are the still frames either side of a rare jump, the static case the gate excludes.

| Workspace | Gallery name | Mode | Steps | Held | Moving share | Static | Run lengths | Spacing (frames: count) |
|---|---|---|---|---|---|---|---|---|
| soviet-anthem | `SovietAnthemV4Fanfare/20261012-023309-62e08190/final/SovietAnthemV4Fanfare-first_and_last_frame_to_video_audio.0-0.0.mp4` | single | 5 | 35 | 0.204 | no | 1: 9, 3: 1 | 2: 2, 4: 1, 8: 1, 10: 1, 14: 1, 48: 1 |
| known-issues | `known-issues-voiced/20260911-134446-f00a7b44/KnownIssuesVoicedShort-episode.7-0.0.mp4` | derived | 9 | 21 | 0.144 | no | 1: 10 | 4: 7, 24: 2 |
| known-issues | `known-issues-voiced/20260911-134446-f00a7b44/KnownIssuesVoicedShort-shot_3_howie_incredulous.4-0.0.mp4` | single | 9 | 20 | 0.350 | no | 1: 10 | 4: 7, 24: 2 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc4-latent/20261112-141747-0484225a/final/BenchNarratedTake-narrated-arc4-latent-take.1-0.0.mp4` | joined | 20 | 18 | 0.005 | yes | 49: 1, 78: 1 | 639: 1 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc/20261111-233536-161c8b80/final/BenchNarratedTake-narrated-arc-take.1-0.0.mp4` | joined | 20 | 13 | 0.036 | yes | none | none |
| soviet-anthem | `SovietAnthemV3B2/20261012-021844-a65b100a/final/SovietAnthemV3B2-first_and_last_frame_to_video_audio.0-0.0.mp4` | single | 5 | 13 | 0.101 | no | 1: 2 | 24: 1 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc-latent/20261112-040003-f9eb4c89/final/BenchNarratedTake-narrated-arc-latent-take.1-0.0.mp4` | joined | 20 | 12 | 0.092 | yes | 49: 1 | none |
| default | `templates/minimax/repair-segment/20261107-223531-8ef2779c/intermediate/MiniMaxH3RepairSegment-repair.1-0.0.mp4` | single | 5 | 11 | 0.059 | yes | none | none |
| sci0fi | `templates/minimax/image-to-video/20261024-214604-9dc9e5d6/final/MiniMaxH3I2V-keyframe_to_video_audio.0-0.0.mp4` | single | 12 | 11 | 0.824 | no | 1: 3 | 49: 1, 57: 1 |
| constitution-walk-3 | `BenchNarratedTake-narrated-arc4/20261112-183800-fcda7b21/final/BenchNarratedTake-narrated-arc4-take.1-0.0.mp4` | joined | 20 | 9 | 0.002 | yes | none | none |
| default | `templates/minimax/repair-segment/20261107-150537-d70aa012/final/MiniMaxH3RepairSegment-splice.2-0.0.mp4` | cuts | 5 | 8 | 0.389 | no | 1: 3 | 2: 1, 64: 1 |
| default | `templates/minimax/repair-segment/20261107-150537-d70aa012/intermediate/MiniMaxH3RepairSegment-repair.1-0.0.mp4` | single | 5 | 8 | 0.672 | no | 1: 6 | 2: 1, 4: 1, 9: 1, 12: 1, 64: 1 |
| constitution-walk-3 | `BenchNarratedTake/20261110-064139-3c390ada/final/BenchNarratedTake-take.1-0.0.mp4` | joined | 20 | 7 | 0.157 | no | none | none |
| soviet-anthem | `SovietAnthemV6Opener_s11/20261012-030806-6fc144eb/final/SovietAnthemV6Opener_s11-first_and_last_frame_to_video_audio.0-0.0.mp4` | single | 5 | 7 | 0.093 | yes | 1: 4 | 5: 1, 8: 1, 43: 1 |
| default | `templates/minimax/video-with-audio/20261011-134814-a32a1f67/final/MiniMaxH3-text_to_video_audio.0-0.0.mp4` | single | 5 | 7 | 0.712 | no | 1: 3 | 9: 1, 102: 1 |
| sci0fi | `templates/minimax/image-to-video/20261024-220757-9b155972/final/MiniMaxH3I2V-keyframe_to_video_audio.0-0.0.mp4` | single | 12 | 7 | 0.804 | no | 1: 3 | 100: 1, 118: 1 |

Gate (eps 1.0): 79 takes have a cluster; 10 non-static takes have >= 7 held frames.
