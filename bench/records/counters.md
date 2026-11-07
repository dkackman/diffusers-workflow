# Counter round (Task 11)

Round 1 ran every counter alone against today's chain at 6 segments (ruling
R27), on S1 static, S2 arc and S3 dialogue walk, at seed 1166. Records:
`bench/records/counters-controls.json` (10 controls: guide on S1-S3 and
last_frame on S1-S2, each at seeds 1166 and 2) and
`bench/records/counters-round1.json` (25 arms). All 35 succeeded on the bench
server (lem, port 8766).

Round 2 (all on S1, plus one S2 diagnostic) is in
`bench/records/counters-round2-confirm.json` (6),
`counters-round2-combos.json` (8, merged from two driver runs split by a power
loss), `counters-round2-long.json` (2 two-minute takes) and
`counters-round2-long-moments.json` (a measure-only pass on the two-minute
takes at eight moments, with every seam). All 16 succeeded. The results and
the **summary table Tasks 15 and 20 build from** are at the end, under
[Round 2 results](#round-2-results) and [Defaults by shot
type](#defaults-by-shot-type). Every verdict was re-scored on 2026-11-05
under ruling R35 (same-seed reruns are not noise-free, and segment-1 drift
labels): see [Re-scored under R35](#re-scored-under-r35-2026-11-05).
Round 3 (anchor 0.25 at two minutes) is at the end, under [Round 3 (2
min)](#round-3-2-min-2026-11-05).

## How the verdicts are read

- **Control.** Each arm is read against its own scene's control at its own
  continuity, at seed 1166 (`ctl-guide-*` or `ctl-lastframe-*`).
- **Floor.** The noise floor is seed variance (ruling R26):
  `noise_floor(control@1166, control@2)` for each scene and continuity. It
  comes from two seeds only, so it is thin. A measure where the two seeds
  happened to agree has a floor near 0, and any difference counts on it
  (S1 last_frame `jump max` is 0.000, for example).
  *(Superseded 2026-11-05, ruling R35: same-seed reruns are not noise-free,
  so the floor is now `combined_floor(seed floor, rerun gap)`, the larger of
  the seed floor and the same-seed rerun gap per measure (`bench/verdict.py`).
  The floors and the rerun gap are in `noise-floor.md`, *Correction (2026-11-05,
  ruling R35)*; the verdicts here were re-scored on it, see* Re-scored under
  R35 *below.)*
- **Measures.** `verdict()` reads every measure in the record except
  `wall_minutes`. Every measure is lower-is-better.
- **Primary measure per scene.**
  - S1: `texture_fine_seg_last`.
  - S2: `seam_look_mean` and `seam_jump_mean`, one verdict each.
  - S3: `seam_jump_mean`, `seam_look_mean` and `texture_fine_seg_last`, one
    verdict each.
- **Two verdict columns.**
  - *All measures* is the design's rule as written.
  - *Picture only* drops the two `seam_level_step_db` (soundtrack) measures.
    S1 and S2 carry room tone and garden ambience, not a held voice. S1's level
    step is large and noisy (7.7 vs 5.2 dB between the two control seeds;
    9.9 dB mean and 36.6 dB max at 10 segments in `noise-floor.md`). Almost
    every S1 guide arm raises it by 3-7 dB, so under the rule as written no
    S1 arm can win on picture. Whether it should count is a question for Don
    (`for-don.md`).
- **What `mixed` means here.** `verdict()` says `mixed` both when the primary
  wins but another measure is worse past its floor, and when the primary is
  inside its floor but another measure is worse. The cells say which:
  **bold** is worse than the control past the floor, _italic_ is better past
  it, and plain is inside the floor.
- **Backoff arms (R29).** `guide_backoff 17` takes 17 frames off every
  hand-over, so the take is 549 frames, not 634. The arm's moments sit at
  the same segments as the control's (62/234/498 against 62/277/583). Its
  seam means are over the same 5 seams and its texture is at the same last
  segment, so it is read segment by segment against the control.
- **last_frame moments.** last_frame takes run 734 frames, with moments at
  62/307/673.
- **The eye check.** Frames came from `GET /api/gallery/{name}/frames?at=<the
  arm's moments>` on the bench server, at the moments in each arm file. They
  were looked at for every control, every arm that wins or is mixed, and
  every arm whose texture number moved a lot. S2's last_frame arms also got
  an 8-frame contact sheet (`count=8`). To rerun this, call the same route on
  each output named in the tables.

## S1 static

Primary: fine-band texture drift at the last moment. Seams second.

### Guide continuity

| arm | verdict (all measures) | verdict (picture only) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor (seed 1166 vs 2) | | | 0.058 | 0.050 | 0.120 | 0.140 | 2.432 | 2.300 | 0.081 | 0.032 | 0.008 | | |
| ctl-guide-s1-s1166 | control | | 0.854 | 0.970 | 1.464 | 2.130 | 7.654 | 16.580 | 0.536 | 0.176 | 0.016 | `b40e855eb553` | `BenchS1Static/20261104-060443-90c37bc2/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| ctl-guide-s1-s2 | control | | 0.796 | 0.920 | 1.344 | 2.270 | 5.222 | 18.880 | 0.455 | 0.144 | 0.008 | `894911aed6c7` | `BenchS1Static/20261104-062828-0642442c/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor25-s1 | tex: mixed | tex: mixed | **0.928** | **1.150** | _1.128_ | _1.980_ | **12.160** | **23.540** | _0.367_ | _0.124_ | 0.015 | `0a40c41af97c` | `BenchS1Static/20261104-054934-fdc9b551/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor50-s1 | tex: mixed | tex: mixed | **1.048** | **1.320** | _1.050_ | _1.710_ | **13.224** | **20.340** | _0.270_ | _0.116_ | **0.028** | `fa17a2c51e33` | `BenchS1Static/20261104-061515-5949edcc/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor75-s1 | tex: mixed | tex: mixed | **1.300** | **1.960** | **1.752** | **3.080** | **15.258** | **23.770** | _0.260_ | _0.103_ | **0.045** | `f17bce198a2d` | `BenchS1Static/20261104-063901-253f19de/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-backoff17-s1 | tex: mixed | tex: mixed | 0.886 | 0.970 | 1.522 | **2.590** | **10.744** | **22.350** | 0.462 | 0.167 | 0.014 | `c594521e59d1` | `BenchS1Static/20261104-070045-f9d2dc73/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-keystone-s1 | tex: mixed | tex: mixed | 0.850 | **1.040** | 1.550 | _1.930_ | **14.556** | **23.260** | _0.373_ | _0.116_ | 0.020 | `1e1b6116ce26` | `BenchS1Static/20261104-071947-19ca76a4/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lookhold-s1 | tex: mixed | tex: mixed | 0.896 | **1.040** | _1.310_ | 2.110 | **11.678** | **23.210** | 0.464 | 0.145 | _0.007_ | `751b43732fb9` | `BenchS1Static/20261104-074252-80046177/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-antiburn-s1 | tex: mixed | tex: mixed | _0.724_ | _0.880_ | **2.646** | **3.210** | 5.458 | _13.260_ | _0.046_ | _0.114_ | **0.074** | `049d652e1648` | `BenchS1Static/20261104-080411-0301f0c0/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |

### last_frame continuity

| arm | verdict (all measures) | verdict (picture only) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor (seed 1166 vs 2) | | | 0.022 | 0.000 | 0.144 | 0.010 | 0.702 | 0.540 | 0.036 | 0.087 | 0.003 | | |
| ctl-lastframe-s1-s1166 | control | | 1.096 | 1.180 | 1.264 | 1.790 | 3.372 | 8.270 | 0.569 | 0.366 | 0.009 | `3a73cb0f5d28` | `BenchS1Static/20261104-065215-5b43dda0/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| ctl-lastframe-s1-s2 | control | | 1.074 | 1.180 | 1.408 | 1.800 | 4.074 | 7.730 | 0.605 | 0.453 | 0.006 | `f053abe1ce5f` | `BenchS1Static/20261104-071117-ae50341d/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lf-carrypick-s1 | tex: mixed | tex: wins | _0.994_ | _1.120_ | 1.252 | _1.730_ | **6.796** | **10.150** | _0.533_ | 0.330 | _0.006_ | `ae932366ea0e` | `BenchLfS1Static/20261104-081730-5241ac41/final/BenchLfS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lf-lookhold-s1 | tex: mixed | tex: wins | _0.788_ | _0.890_ | _0.634_ | _0.970_ | **4.324** | _6.780_ | _0.491_ | 0.304 | _0.005_ | `be3f574df3db` | `BenchS1Static/20261104-083621-b688e963/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` |

### Eye notes (S1)

- **Control, guide (both seeds).** Clean at frame 62. By 583 the stripes and
  the wall have burned into a crawling, mottled, over-sharpened texture. The
  black circle's edge has gone ragged and the face is harder and more
  saturated. Seed 2 looks the same. The numbers are measuring this burn.
- **anchor 0.25.** At 583 the stripes are visibly cleaner than the control's,
  though the wall is still mottled. The face holds identity. Less burn than
  the control.
- **anchor 0.5.** The cleanest of the guide arms at 583. The stripes are
  nearly flat paint and the wall's mottling is reduced. Identity holds, with
  a slight warm shift. The eye agrees with the texture number.
- **anchor 0.75.** The stripes are clean at 583, but she has drifted: a
  narrower face, lighter hair and a different expression. Seams are worse
  too. Not carried.
- **guide_backoff 17.** At 498 it burns like the control. No visible gain.
- **keystone.** At 583 it burns a little less than the control (the stripes
  are mottled, the wall still crawls). A modest gain that matches its texture
  number.
- **look_hold (guide).** At 583 it burns like the control. Its look gain is
  colour steadiness at the seams, not the burn.
- **Anti-burn LoRA.** The low texture number is a lie. The background stays
  clean, but the face smears to a soft, ghosted blur by 277, and by 583 the
  identity is broken (the features dissolve). This is the #1175 trap: the
  measure is low because detail is gone.
- **Control, last_frame.** It burns more than guide by 673. The teal square
  carries dark blotches.
- **last_frame + carry_pick sharpest 4.** Looks the same as its control: the
  same burn and the same blotched square. Its texture gain sits on the floor
  (0.0360 against 0.0357).
- **last_frame + look_hold.** Still burns at 673 (wall crawl, a blotch in the
  square), at about the control's level. Colour across seams is steadier and
  identity holds. The win is in the seams, not the burn.

## S2 moving arc

Primary: seam look and seam jump means. Texture second.

### Guide continuity

| arm | verdict (all measures) | verdict (picture only) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor (seed 1166 vs 2) | | | 0.090 | 0.230 | 0.068 | 0.270 | 2.790 | 5.540 | 0.605 | 1.210 | 1.660 | | |
| ctl-guide-s2-s1166 | control | | 1.362 | 2.090 | 1.286 | 1.740 | 1.462 | 3.430 | 3.084 | 2.293 | 1.430 | `dedd341c9108` | `BenchS2Arc/20261104-073022-cd97c6a1/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| ctl-guide-s2-s2 | control | | 1.272 | 1.860 | 1.218 | 2.010 | 4.252 | 8.970 | 3.689 | 3.504 | 3.090 | `01146e925c94` | `BenchS2Arc/20261104-075344-d76c369a/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor25-s2 | look: loses / jump: mixed | look: loses / jump: mixed | _1.228_ | _1.640_ | **2.032** | **3.600** | 1.364 | 3.150 | _0.340_ | _0.169_ | 0.124 | `4350da6ac745` | `BenchS2Arc/20261104-085551-68fe5c04/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor50-s2 | look: loses / jump: loses | look: loses / jump: loses | **1.496** | **2.370** | **3.214** | **5.620** | 0.988 | 3.050 | 3.056 | 2.424 | 2.401 | `3408f7cf9fa9` | `BenchS2Arc/20261104-091939-682f41d1/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor75-s2 | look: loses / jump: loses | look: loses / jump: loses | **1.666** | **2.730** | **3.500** | **5.540** | 1.152 | 3.200 | _2.313_ | 1.443 | 1.054 | `6fb052f46229` | `BenchS2Arc/20261104-093256-ab1bf1a9/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-backoff17-s2 | look: loses / jump: mixed | look: loses / jump: mixed | 1.370 | 1.890 | **1.430** | **2.610** | 1.126 | 1.990 | 2.873 | 1.534 | 0.000 | `ea394296a02d` | `BenchS2Arc/20261104-094613-88f4b225/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-keystone-s2 | look: loses / jump: mixed | look: loses / jump: mixed | 1.356 | 2.000 | **1.404** | **2.090** | 1.718 | 3.850 | 2.819 | 1.390 | 0.692 | `f774bdbcc1cc` | `BenchS2Arc/20261104-095641-03fe99ed/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lookhold-s2 | look: loses / jump: mixed | look: loses / jump: mixed | 1.400 | 2.030 | **1.562** | **2.970** | 1.362 | 2.760 | 2.978 | 2.035 | 0.705 | `64dc87a2ed3a` | `BenchS2Arc/20261104-100709-3398e6c0/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-antiburn-s2 | look: loses / jump: mixed | look: loses / jump: mixed | _1.248_ | _1.690_ | **1.364** | **2.470** | 2.674 | 5.110 | 3.440 | 2.920 | 1.399 | `9124d417577d` | `BenchS2Arc/20261104-101739-05b1c816/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |

### last_frame continuity

| arm | verdict (all measures) | verdict (picture only) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor (seed 1166 vs 2) | | | 0.082 | 0.180 | 0.284 | 0.070 | 3.116 | 6.720 | 0.170 | 0.066 | 0.214 | | |
| ctl-lastframe-s2-s1166 | control | | 1.258 | 1.770 | 1.310 | 1.930 | 5.018 | 9.810 | 3.429 | 3.641 | 3.684 | `b235c35b72c1` | `BenchS2Arc/20261104-080501-f258d5fa/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| ctl-lastframe-s2-s2 | control | | 1.340 | 1.590 | 1.026 | 1.860 | 1.902 | 3.090 | 3.599 | 3.707 | 3.470 | `765fc66286bd` | `BenchS2Arc/20261104-082803-71d81bfc/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lf-carrypick-s2 | look: mixed / jump: loses | look: mixed / jump: loses | **1.362** | 1.850 | _0.968_ | _1.230_ | 2.958 | 4.410 | _2.421_ | _1.144_ | _0.691_ | `9713480482bc` | `BenchLfS2Arc/20261104-103052-7d0e533a/final/BenchLfS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lf-lookhold-s2 | look: wins / jump: tie | look: wins / jump: tie | 1.340 | 1.840 | _0.402_ | _0.510_ | 3.784 | 7.330 | _2.406_ | _0.972_ | _0.595_ | `242358078691` | `BenchS2Arc/20261104-103927-af5f9b3e/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` |

### Eye notes (S2)

- **Control, guide.** The arc completes: it goes round to the statue's back
  by about 22 s. By the last segment the sky is blowing out to white and the
  scene has invented palms and a tower. Seed 2 is the same, with a white-out
  sky at 583.
- **anchor 0.25, 0.5, 0.75.** The arc completes. At 583 the sky is
  posterised into blotchy cloud bands (worse at 0.5 and 0.75), and the
  statue's coat crawls. The low texture at 0.25 is a flat, banded sky, not a
  clean one. The look loss at the seams is real.
- **backoff, keystone, look_hold (guide).** The arc completes, with
  end-of-take sky banding similar to the control's. No visible improvement.
- **Anti-burn LoRA.** At 583 the sky blows out to a white patch and the
  statue smears into a soft blob. Broken.
- **Control, last_frame.** The arc is slow but real: it reaches a side view
  by about 26-30 s. Clean sky.
- **last_frame + carry_pick and last_frame + look_hold: frozen motion.** On
  the 8-frame contact sheet both stay on the front view for the whole take.
  The statue faces the camera at 30 s as it does at 0 s, and the camera never
  goes round. Their look and texture wins are the numbers of a camera that
  did not move. **Rejected by eye.** One seed-2 run of last_frame + look_hold
  is in round 2, to see whether the freeze repeats.

## S3 dialogue walk

Primary: seam jump, seam look, and texture. Guide only (the brief gave S3 no
last_frame arms).

| arm | verdict (all measures) | verdict (picture only) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor (seed 1166 vs 2) | | | 0.332 | 0.140 | 0.410 | 0.070 | 0.378 | 2.590 | 0.226 | 0.112 | 0.111 | | |
| ctl-guide-s3-s1166 | control | | 1.524 | 1.830 | 1.470 | 1.710 | 2.968 | 8.810 | 0.067 | 0.114 | 0.188 | `c0b92d0d0cd4` | `BenchS3DialogueWalk/20261104-084517-6db94fb9/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| ctl-guide-s3-s2 | control | | 1.192 | 1.690 | 1.060 | 1.640 | 2.590 | 6.220 | 0.294 | 0.226 | 0.299 | `64b4a78c74a2` | `BenchS3DialogueWalk/20261104-090907-773b20e5/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor25-s3 | jump: mixed / look: mixed / tex: mixed | jump: mixed / look: mixed / tex: mixed | 1.642 | **2.040** | 1.484 | **1.800** | 3.230 | 8.810 | 0.103 | 0.165 | **0.499** | `7c1aa8f8b38d` | `BenchS3DialogueWalk/20261104-105035-5a02f9f8/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor50-s3 | jump: mixed / look: mixed / tex: mixed | jump: mixed / look: mixed / tex: mixed | 1.554 | **2.090** | 1.404 | **2.200** | **3.896** | 8.810 | 0.053 | 0.013 | **0.379** | `fef6b9aea7c9` | `BenchS3DialogueWalk/20261104-110353-90e5c4ef/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-anchor75-s3 | jump: mixed / look: loses / tex: mixed | jump: mixed / look: loses / tex: mixed | 1.666 | **2.300** | **1.922** | **2.820** | 3.050 | 8.810 | 0.076 | 0.120 | **0.451** | `fbf286d9f192` | `BenchS3DialogueWalk/20261104-111709-bec10029/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-backoff17-s3 | jump: mixed / look: mixed / tex: loses | jump: mixed / look: mixed / tex: loses | 1.496 | **2.040** | 1.124 | 1.690 | **3.646** | 6.470 | **0.337** | **0.472** | _0.036_ | `5873332b54fc` | `BenchS3DialogueWalk/20261104-113024-86593c80/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-keystone-s3 | jump: mixed / look: mixed / tex: mixed | jump: mixed / look: mixed / tex: mixed | 1.414 | **2.080** | 1.182 | **2.030** | **3.660** | 7.150 | 0.106 | 0.187 | 0.222 | `a2bd5e2545db` | `BenchS3DialogueWalk/20261104-114056-8aecac36/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-lookhold-s3 | jump: tie / look: tie / tex: tie | jump: tie / look: tie / tex: tie | 1.592 | 1.830 | 1.432 | _1.610_ | 3.028 | 8.810 | 0.055 | 0.109 | 0.244 | `a9a134a9acf8` | `BenchS3DialogueWalk/20261104-115128-bf419738/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |
| r1-antiburn-s3 | jump: loses / look: mixed / tex: loses | jump: loses / look: mixed / tex: loses | **2.046** | **2.390** | 1.744 | **2.010** | **4.954** | 6.280 | **0.671** | **0.876** | **0.519** | `c50ec5961418` | `BenchS3DialogueWalk/20261104-120204-7fe540b4/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` |

### Eye notes (S3)

- **Control (both seeds).** From 277 on, the sky at the top centre of the
  street is a flat white blown-out blob. By 583 the face and coat are
  posterised and plasticky. Seed 2 has the same blob, and at 583 the face is
  burned as well.
- **anchor 0.25 and 0.75.** At 583 they look the same as the control.
- **anchor 0.5.** Slightly softer, with less burn on the face at 583. Inside
  the floor on every primary.
- **backoff, keystone, look_hold.** At 583 the burn and the white blob are at
  the control's level. Keystone's face is, if anything, more contrasty.
- **Anti-burn LoRA.** Heavy blur from 277 on: the walker is out of focus and
  smeared. Broken.

## Continuity: last_frame against guide

| scene | verdict (all) | verdict (picture) | worse past floor | better past floor |
|---|---|---|---|---|
| s1-static | tex: mixed | tex: mixed | jump, jump max, tex mid | look, look max, level dB, level dB max |
| s2-arc | look: mixed / jump: mixed | look: mixed / jump: mixed | level dB, tex mid, tex coarse | jump, jump max |

| arm | verdict (all) | verdict (picture) | worse past floor | better past floor |
|---|---|---|---|---|
| r1-lf-carrypick-s1 | tex: mixed | tex: mixed | jump, jump max, tex mid | look, look max, level dB max, tex coarse |
| r1-lf-lookhold-s1 | tex: mixed | tex: mixed | tex mid | jump, jump max, look, look max, level dB, level dB max, tex coarse |
| r1-lf-carrypick-s2 | look: wins / jump: tie | look: wins / jump: tie |  | jump max, look, look max, tex fine |
| r1-lf-lookhold-s2 | look: wins / jump: tie | look: wins / jump: tie |  | jump max, look, look max, tex fine, tex mid |

**Controls.** As controls, the two continuities are mixed on both scenes:

- S1: last_frame has the better seam look and the worse seam jump, and burns
  a little more by eye.
- S2: last_frame has the better jump, the worse mid and coarse texture, and a
  slower arc.

**last_frame + look_hold.**

- S1: against the guide control it ties on texture (inside the floor) and is
  better on every seam measure. By eye it is about as burned.
- S2: its numbers beat the guide control, but the camera froze.

**Verdict: no default change from round 1.** last_frame is not shown better
than guide as the default. The S1 last_frame + look_hold result goes to
round 2 for confirmation.

## Round-1 winners

**Strict winners (verdict `wins`, confirmed by eye): none on any scene.**

- **S2.** The only `wins` the rule returned is `r1-lf-lookhold-s2`, and its
  arc is frozen.
- **S1.** `r1-lf-lookhold-s1` and `r1-lf-carrypick-s1` win on picture only.
  The carry pick wins on the floor exactly, and the eye sees no change.

The design asks for winners confirmed on a second seed. Round 1 is one seed
against a two-seed floor, so the candidates below go to round 2 as
**provisional winners**: arms whose primary beats the control past the floor,
whose eye check agrees, and whose only losses are on secondary measures.

| scene | provisional winners for round 2 | why carried | why not a strict win |
|---|---|---|---|
| S1 | `anchor_strength 0.25` (guide) | texture 0.367 vs 0.536 (floor 0.081); look better; the eye sees less burn | seam jump 0.928 vs 0.854 (floor 0.058); level dB |
| S1 | `anchor_strength 0.5` (guide) | texture 0.270; look better; the cleanest by eye | seam jump 1.048 (worse past the floor); coarse texture; level dB |
| S1 | `keystone segment_1` (guide) | texture 0.373; look max better; a modest gain by eye | jump max 1.04 vs 0.97 (floor 0.05); level dB |
| S1 | `look_hold` on last_frame | wins every picture measure against the last_frame control; ties texture and beats every seam against guide | level dB mean 4.32 vs 3.37 (floor 0.70); the eye sees the same burn |
| S2 | none | every guide counter loses seam look; both last_frame counters freeze the arc | |
| S3 | none | every primary is inside S3's wide floor (jump 0.33, look 0.41, texture 0.23); no visible gain | |

## Status per idea (round 1)

- **#1190 anchor strength:** mixed. It helps S1 (texture, and the eye sees
  less burn at 0.25 and 0.5; 0.75 drifts identity). It loses S2 (seam look)
  and is a tie on S3.
- **#1189 hand-over backoff (17):** loses everywhere. S1 texture is inside
  the floor, S2 seam look loses, and S3 texture loses.
- **#1187 keystone:** mixed. S1 texture wins with a small jump max cost. It
  loses S2 look and ties S3.
- **#1032 sharpest of last 4:** loses everywhere. On S1 it ties exactly on the
  floor and the eye sees no change. On S2 it freezes the arc.
- **#1191 look_hold:** mixed.
  - Guide: S1 look better with texture tied; S2 loses; S3 ties.
  - last_frame: S1 wins on picture, with burn unchanged by eye; S2 freezes
    the arc.
- **Anti-burn LoRA (`Imagineer13/H3-Anti-Burn-LoRA`, 0.6, every segment):**
  loses everywhere. Its low texture numbers are blur: the face smears on S1,
  the sky blows out and the statue smears on S2, and S3 is out of focus.
- **last_frame against guide as the default continuity:** mixed. The
  controls are mixed on both scenes. last_frame + look_hold beats guide on
  S1 seams but froze S2. Taste: needs Don if seed 2 confirms S1
  (`for-don.md`).

## Round 2

There are three arm files, in priority order. All are S1, since S2 and S3
have no candidate. Every arm differs from its control only in the counters
named in `tests/test_bench_workflows.py` (`ROUND2`, which checks the
effective variables).

1. **`bench/arms/counters-round2-confirm.json`** (6 runs):
   - `ctl-guide-s1-s1166-rerun`: the round-1 guide control again, past the
     step cache (`-r2` id). The workflows gained a null `duration_seconds`
     knob, so this checks that the round-1 controls still stand bit for bit.
   - anchor 0.25, anchor 0.5, keystone and last_frame + look_hold on S1 at
     seed 2, read against `ctl-*-s1-s2`.
   - last_frame + look_hold on S2 at seed 2: a diagnostic for the frozen
     arc, not a candidate.
2. **`bench/arms/counters-round2-combos.json`** (8 runs): the pairwise
   combinations of the S1 candidates that can run together, at seeds 1166
   and 2. Running both seeds now means a combination can be confirmed
   without a round 3.
   - anchor 0.25 + keystone and anchor 0.5 + keystone (guide).
   - last_frame + look_hold + anchor 0.25, and last_frame + look_hold +
     anchor 0.5. Anchor accepts last_frame.
   - Keystone needs guide and look_hold's keyframe parts need last_frame, so
     keystone + look_hold is a different continuity from both controls and
     is not run.
3. **`bench/arms/counters-round2-long.json`** (2 runs): two minutes
   (`duration_seconds: 120`, 29 segments, 2880 frames, moments
   62/1501/2868) for the guide control and for anchor 0.5, the best single
   candidate on S1's primary and by eye.
   - The bench workflows gained `duration_seconds` as a variable, null by
     default, so a chain keeps `segments`.
   - The 2-minute arms are read against each other. A 6-segment control
     cannot judge a 29-segment take.

## Round 2 results

### The rerun did not reproduce, so the guide floor is wider

`ctl-guide-s1-s1166-rerun` (job `fd34bc937628`) is the round-1 guide control
again, at the same seed, past the step cache. It is **not** bit-identical to
`ctl-guide-s1-s1166` (`b40e855eb553`): seam look 1.656 against 1.464, level
step 11.62 against 7.65 dB, fine texture 0.520 against 0.536, jump max 0.96
against 0.97. Seam jump mean is identical (0.854).

- It is the render, not the measure. Measuring `b40e855eb553` again today
  gives its round-1 numbers to the third decimal.
- By eye the two runs look the same at 62/277/583.
- The two-minute control (`8d1614ced27a`, same seed) does reproduce round 1
  over its first six segments: texture 0.536 at frame 583, seams 1-5 jump
  0.85, look 1.46, level 7.7 dB. So only the `-r2` rerun drifted. Its cause
  is not known (the distinct workflow id, the null `lora` entry or the card
  are candidates).

The floor no longer has a 0 for same-seed reruns, so from here on:

- **Guide S1 floor** = the largest pairwise difference among the three guide
  controls (1166, its rerun, seed 2): jump 0.058, jump max 0.050, **look
  0.312**, look max 0.140, **level dB 6.394**, level dB max 3.650, tex fine
  0.081, tex mid 0.038, tex coarse 0.009.
- **last_frame S1 floor** = the larger of its two-seed floor and the guide
  rerun noise: jump 0.022, jump max 0.010, look 0.192, look max 0.050, level
  dB 3.962, level dB max 3.650, tex fine 0.036, tex mid 0.087, tex coarse
  0.009.
- **Across continuities** (an lf arm read against a guide control, or a guide
  arm): the larger of the two.

Two round-1 readings change under this floor. guide `look_hold`'s seam look
gain on S1 (1.31 against 1.46) is now inside the floor: a tie. And the S1
level step (mean) is inside the floor for every S1 arm in round 1 and round 2,
which answers the level-step question below.

### Seed-2 confirmation

Cells: **bold** is worse than the control past the floor, _italic_ is better
past it. Primary is fine texture at the last moment.

| arm | against | verdict (all) | verdict (picture) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1-anchor25-s1 | ctl-guide-s1-s1166 | mixed | mixed | **0.928** | **1.150** | _1.128_ | _1.980_ | 12.160 | **23.540** | _0.367_ | _0.124_ | 0.015 | `0a40c41af97c` |
| r2-anchor25-s1-seed2 | ctl-guide-s1-s2 | mixed | mixed | 0.812 | 0.950 | 1.092 | _1.880_ | 10.426 | 18.240 | _0.325_ | 0.130 | **0.031** | `93dd359dc11e` |
| r1-anchor50-s1 | ctl-guide-s1-s1166 | mixed | mixed | **1.048** | **1.320** | _1.050_ | _1.710_ | 13.224 | **20.340** | _0.270_ | _0.116_ | **0.028** | `fa17a2c51e33` |
| r2-anchor50-s1-seed2 | ctl-guide-s1-s2 | mixed | mixed | **0.904** | **1.060** | 1.212 | _1.720_ | 6.550 | 16.230 | _0.272_ | 0.121 | **0.030** | `075ed6de83f7` |
| r1-keystone-s1 | ctl-guide-s1-s1166 | mixed | mixed | 0.850 | **1.040** | 1.550 | _1.930_ | **14.556** | **23.260** | _0.373_ | _0.116_ | 0.020 | `1e1b6116ce26` |
| r2-keystone-s1-seed2 | ctl-guide-s1-s2 | mixed | mixed | 0.800 | 0.900 | 1.536 | **2.710** | 10.352 | 20.130 | 0.448 | 0.123 | 0.010 | `ab59d503fd4d` |
| r1-lf-lookhold-s1 | ctl-lastframe-s1-s1166 | wins | wins | _0.788_ | _0.890_ | _0.634_ | _0.970_ | 4.324 | 6.780 | _0.491_ | 0.304 | 0.005 | `be3f574df3db` |
| r2-lf-lookhold-s1-seed2 | ctl-lastframe-s1-s2 | wins | wins | _0.806_ | _1.020_ | _0.838_ | _1.060_ | 3.664 | 8.880 | _0.550_ | 0.411 | 0.001 | `b7ca6da7d075` |
| r1-lf-lookhold-s1 | ctl-guide-s1-s1166 | mixed | mixed | _0.788_ | _0.890_ | _0.634_ | _0.970_ | 4.324 | _6.780_ | 0.491 | **0.304** | _0.005_ | `be3f574df3db` |
| r2-lf-lookhold-s1-seed2 | ctl-guide-s1-s2 | loses | loses | 0.806 | **1.020** | _0.838_ | _1.060_ | 3.664 | _8.880_ | **0.550** | **0.411** | 0.001 | `b7ca6da7d075` |

Per provisional winner:

- **anchor 0.25: holds.** Texture beats the control past the floor on both
  seeds (0.367 vs 0.536, 0.325 vs 0.455). The secondary costs do not repeat:
  the jump cost is seed 1166 only (seed 2: 0.812 vs 0.796, inside), and the
  coarse-texture cost is seed 2 only (0.031 vs 0.008; on 1166 it is 0.015 vs
  0.016). No measure is worse past the floor on both seeds.
- **anchor 0.5: holds on texture, and so does its jump cost.** Texture 0.272
  vs 0.455 on seed 2, but jump is again worse past the floor (0.904 vs 0.796,
  after 1.048 vs 0.854), and so are jump max and coarse texture, on both
  seeds.
- **keystone: does not hold.** Seed-2 texture is 0.448 against 0.455, inside
  the floor, and look max is worse (2.71 vs 2.27). **Not a winner.**
- **last_frame + look_hold: holds against its own control, not against
  guide.** It is a strict `wins` against the last_frame control on both seeds
  (every measure, soundtrack included, under the widened floor). Against the
  guide control it loses on seed 2: fine texture 0.550 vs 0.455 and mid
  texture 0.411 vs 0.144, both past the floor.
- **S2 last_frame + look_hold, seed 2 (`a3d6a044fd6b`): the camera is held
  back again.** It "wins" seam look against the last_frame control (0.386 vs
  1.026) and texture collapses to 0.83 (control 3.60). See the eye notes.

### Combinations

Each combination is read against the guide control, against the best single
counter of its own anchor strength, and (for the last_frame sets) against
last_frame + look_hold alone.

| arm | against | verdict (all) | verdict (picture) | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2-anchor25-keystone-s1-s1166 | ctl-guide-s1-s1166 | mixed | mixed | 0.898 | **1.160** | _1.092_ | _1.610_ | 10.750 | 15.170 | _0.309_ | _0.118_ | **0.031** | `8accd0c37f91` |
| r2-anchor25-keystone-s1-s1166 | r1-anchor25-s1 | mixed | mixed | 0.898 | 1.160 | 1.092 | _1.610_ | 10.750 | _15.170_ | 0.309 | 0.118 | **0.031** | `8accd0c37f91` |
| r2-anchor25-keystone-s1-s2 | ctl-guide-s1-s2 | mixed | mixed | 0.832 | **1.020** | 1.186 | _1.790_ | **14.018** | **27.160** | _0.311_ | 0.113 | **0.031** | `e34a5aebaab4` |
| r2-anchor25-keystone-s1-s2 | r2-anchor25-s1-seed2 | mixed | mixed | 0.832 | **1.020** | 1.186 | 1.790 | 14.018 | **27.160** | 0.311 | 0.113 | 0.031 | `e34a5aebaab4` |
| r2-anchor50-keystone-s1-s1166 | ctl-guide-s1-s1166 | mixed | mixed | **1.296** | **1.740** | **2.038** | **2.890** | 5.590 | 14.460 | _0.293_ | _0.128_ | **0.039** | `e5c7b013538f` |
| r2-anchor50-keystone-s1-s1166 | r1-anchor50-s1 | mixed | mixed | **1.296** | **1.740** | **2.038** | **2.890** | _5.590_ | _14.460_ | 0.293 | 0.128 | **0.039** | `e5c7b013538f` |
| r2-anchor50-keystone-s1-s2 | ctl-guide-s1-s2 | mixed | mixed | **1.068** | **1.590** | **1.672** | **2.710** | 10.810 | **24.100** | _0.246_ | 0.107 | **0.041** | `08e32162985c` |
| r2-anchor50-keystone-s1-s2 | r2-anchor50-s1-seed2 | mixed | mixed | **1.068** | **1.590** | **1.672** | **2.710** | 10.810 | **24.100** | 0.246 | 0.107 | **0.041** | `08e32162985c` |
| r2-lf-lookhold-anchor25-s1-s1166 | ctl-lastframe-s1-s1166 | wins | wins | _0.786_ | _0.960_ | _0.626_ | _0.850_ | 4.090 | 9.440 | _0.331_ | _0.208_ | 0.010 | `835379468ec3` |
| r2-lf-lookhold-anchor25-s1-s1166 | ctl-guide-s1-s1166 | wins | wins | _0.786_ | 0.960 | _0.626_ | _0.850_ | 4.090 | _9.440_ | _0.331_ | 0.208 | 0.010 | `835379468ec3` |
| r2-lf-lookhold-anchor25-s1-s1166 | r1-anchor25-s1 | tie | tie | _0.786_ | _0.960_ | _0.626_ | _0.850_ | _4.090_ | _9.440_ | 0.331 | 0.208 | 0.010 | `835379468ec3` |
| r2-lf-lookhold-anchor25-s1-s1166 | r1-lf-lookhold-s1 | mixed | mixed | 0.786 | **0.960** | 0.626 | _0.850_ | 4.090 | 9.440 | _0.331_ | _0.208_ | 0.010 | `835379468ec3` |
| r2-lf-lookhold-anchor25-s1-s2 | ctl-lastframe-s1-s2 | wins | wins | _0.740_ | _0.830_ | _0.760_ | _0.900_ | 4.086 | 8.180 | _0.297_ | _0.234_ | 0.015 | `41355297a083` |
| r2-lf-lookhold-anchor25-s1-s2 | ctl-guide-s1-s2 | mixed | mixed | 0.740 | _0.830_ | _0.760_ | _0.900_ | 4.086 | _8.180_ | _0.297_ | **0.234** | 0.015 | `41355297a083` |
| r2-lf-lookhold-anchor25-s1-s2 | r2-anchor25-s1-seed2 | mixed | mixed | _0.740_ | _0.830_ | _0.760_ | _0.900_ | 4.086 | _8.180_ | 0.297 | **0.234** | _0.015_ | `41355297a083` |
| r2-lf-lookhold-anchor25-s1-s2 | r2-lf-lookhold-s1-seed2 | mixed | mixed | _0.740_ | _0.830_ | 0.760 | _0.900_ | 4.086 | 8.180 | _0.297_ | _0.234_ | **0.015** | `41355297a083` |
| r2-lf-lookhold-anchor50-s1-s1166 | ctl-lastframe-s1-s1166 | wins | wins | _0.752_ | _0.930_ | _0.766_ | _1.190_ | 5.018 | 7.770 | _0.183_ | _0.178_ | 0.011 | `d8b17df7dfb9` |
| r2-lf-lookhold-anchor50-s1-s1166 | ctl-guide-s1-s1166 | wins | wins | _0.752_ | 0.930 | _0.766_ | _1.190_ | 5.018 | _7.770_ | _0.183_ | 0.178 | 0.011 | `d8b17df7dfb9` |
| r2-lf-lookhold-anchor50-s1-s1166 | r1-anchor50-s1 | wins | wins | _0.752_ | _0.930_ | 0.766 | _1.190_ | _5.018_ | _7.770_ | _0.183_ | 0.178 | _0.011_ | `d8b17df7dfb9` |
| r2-lf-lookhold-anchor50-s1-s1166 | r1-lf-lookhold-s1 | mixed | mixed | _0.752_ | **0.930** | 0.766 | **1.190** | 5.018 | 7.770 | _0.183_ | _0.178_ | 0.011 | `d8b17df7dfb9` |
| r2-lf-lookhold-anchor50-s1-s2 | ctl-lastframe-s1-s2 | mixed | mixed | _0.774_ | _0.880_ | _0.886_ | _1.250_ | 2.460 | 4.340 | _0.236_ | _0.245_ | **0.040** | `c8bfadf286ce` |
| r2-lf-lookhold-anchor50-s1-s2 | ctl-guide-s1-s2 | mixed | mixed | 0.774 | 0.880 | _0.886_ | _1.250_ | 2.460 | _4.340_ | _0.236_ | **0.245** | **0.040** | `c8bfadf286ce` |
| r2-lf-lookhold-anchor50-s1-s2 | r2-anchor50-s1-seed2 | mixed | mixed | _0.774_ | _0.880_ | _0.886_ | _1.250_ | 2.460 | _4.340_ | 0.236 | **0.245** | **0.040** | `c8bfadf286ce` |
| r2-lf-lookhold-anchor50-s1-s2 | r2-lf-lookhold-s1-seed2 | mixed | mixed | _0.774_ | _0.880_ | 0.886 | **1.250** | 2.460 | _4.340_ | _0.236_ | _0.245_ | **0.040** | `c8bfadf286ce` |

What the combinations show:

- **Keystone adds nothing to anchor 0.25, and hurts anchor 0.5.** With anchor
  0.25, texture is inside the floor of anchor 0.25 alone on both seeds (0.309
  vs 0.367, 0.311 vs 0.325), and jump max and coarse texture get worse. With
  anchor 0.5, every seam measure is worse than anchor 0.5 alone on both seeds
  (jump 1.296 vs 1.048, 1.068 vs 0.904; look 2.04 vs 1.05, 1.67 vs 1.21).
- **last_frame + look_hold + anchor is the best-measuring set on S1.** It
  beats the guide control on texture past the floor on all four runs, and on
  seam look on all four. Against anchor alone (guide) its seams are better on
  both seeds, and its texture is inside the floor (0.331 vs 0.367, 0.297 vs
  0.325 at 0.25; 0.183 vs 0.270, past the floor, and 0.236 vs 0.272 at 0.5).
  Its one repeated cost against guide is mid-band texture (0.208/0.234 at
  0.25, 0.178/0.245 at 0.5, against the guide controls' 0.176/0.144); every
  last_frame run carries a higher mid band (the bare last_frame controls are
  at 0.37/0.45).
- **The anchor improves last_frame + look_hold.** Against look_hold alone,
  both anchor strengths cut the texture past the floor on both seeds (to
  0.18-0.33 from 0.49-0.55), and by eye they take the burn off the face that
  look_hold alone leaves (below).
- **Anchor strength under last_frame:** 0.5 measures lower on texture at seed
  1166 (0.183 vs 0.331) and about the same at seed 2 (0.236 vs 0.297, inside
  the floor); 0.25 has the better seam look on both seeds (0.626/0.760 vs
  0.766/0.886).

### Two minutes: the anchor's gain holds on the numbers and fails by eye

`ctl-guide-s1-120s` (`8d1614ced27a`) against `r2-anchor50-s1-120s`
(`107859a31bc1`): 29 segments, 2880 frames, seed 1166, both guide. There is no
two-minute floor; the 6-segment guide floor is the only scale.

Texture drift on the S1 crop against frame 62, from the measure-only pass
(`counters-round2-long-moments.json`: `POST /api/gallery/compare` with
`texture_bands`, crop `[0,0,240,150]`, the same route and crop as the
driver, at eight moments; the driver's own measure compares the first and
last moment only):

| frame (s) | 300 (12.5) | 583 (24.3) | 1000 (41.7) | 1501 (62.5) | 2000 (83.3) | 2500 (104.2) | 2868 (119.5) |
|---|---|---|---|---|---|---|---|
| control fine | 0.180 | 0.536 | 1.108 | 1.399 | 1.646 | 1.658 | 1.706 |
| anchor 0.5 fine | 0.137 | 0.270 | 0.360 | 0.602 | 0.737 | 0.759 | 1.067 |
| control mid | 0.045 | 0.176 | 0.406 | 0.716 | 1.071 | 1.156 | 1.171 |
| anchor 0.5 mid | 0.038 | 0.116 | 0.162 | 0.227 | 0.307 | 0.273 | 0.462 |

Seams, by stretch of the take (`analyze_seams`, every seam):

| seams | control jump (max) | anchor jump (max) | control look (max) | anchor look (max) |
|---|---|---|---|---|
| 1-5 | 0.85 (0.97) | 1.05 (1.32) | 1.46 (2.13) | 1.05 (1.71) |
| 6-14 | 1.18 (1.64) | 0.97 (1.28) | 2.03 (3.26) | 1.19 (2.36) |
| 15-23 | 1.25 (1.94) | 0.72 (0.92) | 1.67 (3.75) | 1.27 (1.66) |
| 24-28 | 2.28 (**6.56**) | 1.13 (1.46) | 4.38 (**11.51**) | 1.51 (2.65) |
| all 28 | 1.34 (6.56) | 0.93 (1.46) | 2.23 (11.51) | 1.25 (2.65) |

On the numbers the anchor's gain **grows** with length:

- Its texture drift is half the control's at 24 s and a third at 42 s, and
  stays 0.6-0.9 lower to the end.
- The seam jump cost seen at 6 segments reverses after seam 5. The control's
  seams get worse with length, and at seam 26 it has a 6.56 jump and 11.5
  look, past the chain check's 4.0 ceiling. The anchor's worst seam is 1.46.

**By eye both takes are broken well before two minutes, in opposite ways**
(face crops at 583/800/1000/1250/1501, full frames at 62/1000/1501/2000/2868):

- **Control:** the burn of round 1 keeps building. At 33 s the face is
  cracking into crust. By 42 s it is a posterised, cut-out face. From 62 s
  the whole frame is a mosaic of crusted shards. She stays recognisably the
  same person, harsh and mottled. Seam 26 is a visible reset: the mosaic
  jumps to a lighter, cleaner backdrop for a segment.
- **Anchor 0.5:** at 24 s it is the cleaner take. At 33 s the face starts to
  soften, and by 42 s it is smooth and waxy. At 52 s the features are
  dissolving and the eyes have gone. By 62 s the face is a blurred ghost, and
  at 83-120 s it is a white mask with dark holes, over a soft, blocky
  backdrop. **Identity is lost.** Its low texture number after about 40 s is
  blur, the #1175 trap again, as with the anti-burn LoRA.

So the anchor's gain is real up to about 6-8 segments (25-33 s) and turns
into a different failure past that. **No counter tested makes a two-minute
static take usable.** This is the gap section 5 of the design names as the
gate for the second group (carry without the pixel round trip: #1150, #1044,
#1030) and for the per-segment check and re-roll. Until then a static shot
longer than about 30 s should be cut, not chained.

### Eye notes (round 2)

Frames at each arm's moments (62/277/583 guide, 62/307/673 last_frame), plus
crops of the top-left quarter (`crop=0,0,480,272`) at the last moment to see
the circle and stripes at full resolution.

- **Control rerun:** indistinguishable by eye from round 1's control at all
  three moments.
- **Seed-2 control:** the same burn as seed 1166 at 583: a crusted circle and
  mottled stripes.
- **anchor 0.25, seed 2:** stripes visibly cleaner and the circle's edge
  cleaner than the control's. Identity holds. Agrees with the number.
- **anchor 0.5, seed 2:** about as clean as 0.25 by eye; the circle keeps
  some crust. Identity holds at 24 s.
- **keystone, seed 2:** burns like the control. Agrees with the tie.
- **anchor + keystone (both strengths, both seeds):** about as clean as
  anchor alone. Keystone adds no visible gain, and identity holds.
- **last_frame + look_hold, seed 2:** still burns at 673 (crusted circle,
  heavy stripes), and the **face is blotched and burned**. As in round 1, its
  win is in the seams, not the burn.
- **last_frame + look_hold + anchor 0.25 (both seeds):** the stripes are
  cleaner than last_frame + look_hold's and the face is clean. The circle
  keeps some crust.
- **last_frame + look_hold + anchor 0.5 (both seeds):** the cleanest S1 frames
  in the bench at the last moment, at 673, 90 frames later than the guide
  arms' 583: the stripes are nearly flat paint and the face is clean. The
  circle still has crust, and the wall at the left is mottled at seed 1166.
  Identity holds.
- **S2 last_frame + look_hold, seed 2 (8-frame contact sheet,
  `count=8`):** not a full freeze this time, but the camera is held back. It
  stays on the front view to 13 s and reaches about a three-quarter view by
  30 s. The plain last_frame control at seed 2 goes round to the statue's
  back by 26-30 s. Round 1's seed-1166 run never left the front. **The
  motion loss repeats on both seeds**, so the "win" on seam look is still a
  camera that moved less.

### The two open questions

**Anchor 0.5 looks better but measures worse on seam jump.** Seed 2 repeats
the jump cost (0.904 vs 0.796, floor 0.058), and its texture lead over 0.25
is inside the floor on seed 2 (0.272 vs 0.325), where by eye the two look
alike. At two minutes 0.5 dissolves the face from about 50 s. **Firm
recommendation: anchor 0.25**, not 0.5, as the static default. It keeps the
texture gain on both seeds, its jump cost does not repeat, and it is the
lower identity risk (0.75 drifted identity in round 1; 0.5 loses it at
length). 0.25 has not been run at two minutes.

**Should the level step count on ambience-only scenes?** The same-seed rerun
moved S1's level step from 7.65 to 11.62 dB, and the spread over the three
guide controls is 6.4 dB. Under that floor no S1 arm's level-step mean is
worse past it. Verdicts with and without it are now the same in every round-2
row. **Firm recommendation: do not
count `seam_level_step_db_*` on scenes without a held voice or music bed
(S1, S2). Read them on picture.** Count it on S3, S4 and S6.

### Verdict per idea

| idea | verdict | numbers |
|---|---|---|
| #1190 anchor strength | **default for static shots, 0.25** (guide, `anchor: segment_1`, `anchor_strength: 0.25`), chains up to about 6-8 segments. Not for moving camera or dialogue walk. | S1 texture 0.367 vs 0.536 (`0a40c41af97c`) and 0.325 vs 0.455 (`93dd359dc11e`), floor 0.081. S2 look loses at every strength (2.03-3.50 vs 1.29, floor 0.07). S3 inside the floor. 0.5 dissolves the face at length (`107859a31bc1`). |
| #1189 hand-over backoff | **cut** | S1 texture 0.462 vs 0.536 (inside 0.081); S2 look 1.43 vs 1.29 (floor 0.07, loses); S3 texture 0.337 vs 0.067 (floor 0.23, loses). Round 1 only; nothing to carry. |
| #1187 keystone as keyframe | **cut** as a short-take counter. *(Superseded 2026-11-06: at two minutes the keystone at slot `last` is the static long-take default, `shot: static_long` on `templates/minimax/chained-segments`, Don's "not a loop", 43a0fb87/d555a373; see `structural.md` *Keystone slot*.)* | Seed 2 texture 0.448 vs 0.455 (tie, `ab59d503fd4d`); with anchor 0.25 no gain over anchor alone (0.311 vs 0.325); with anchor 0.5 every seam worse (jump 1.30/1.07 vs 1.05/0.90, `e5c7b013538f`, `08e32162985c`); S2 look loses (1.40 vs 1.29). |
| #1032 sharpest of last 4 (`carry_pick`) | **cut** | S1 texture 0.533 vs 0.569 on a 0.036 floor (no visible change, `ae932366ea0e`); S2 freezes the arc (`9713480482bc`). |
| #1191 `look_hold` on guide (creep + texture parts) | **cut** | S1 look 1.31 vs 1.46 now inside the 0.31 floor, texture tied; S2 look loses (1.56 vs 1.29); S3 tie. |
| #1191 `look_hold` on last_frame (all four parts) | **available knob, no default** | Wins against the last_frame control on both seeds (`be3f574df3db`, `b7ca6da7d075`) but loses texture to guide on seed 2 (0.550 vs 0.455) and holds the camera back on S2 on both seeds (`242358078691`, `a3d6a044fd6b`). Its value is as part of the static set below (Don's call). |
| #1033 / #1067 seam grade | **available knob, no default** (with #1191 on last_frame) | The seam grade is look_hold's `pickup` part, which only runs on last_frame. It was benched inside the last_frame look_hold arms, not alone; its numbers are that row's. |
| Anti-burn LoRA (0.6, every segment) | **cut** | Low texture is blur: S1 face dissolves by 277 (`049d652e1648`); S2 statue smears (`9124d417577d`); S3 jump 2.05 vs 1.52 and texture 0.671 vs 0.067 lose (`c50ec5961418`). |
| Continuity: guide | **default for every shot type** | last_frame alone is mixed against guide on S1 and S2 (round 1). |
| Continuity: last_frame | **available knob, no default**; the static alternative (last_frame + look_hold + anchor) is **Don's call** | Best-measuring S1 set (`835379468ec3`, `41355297a083`, `d8b17df7dfb9`, `c8bfadf286ce`), but mid texture worse than guide on both seeds, untested at length, and look_hold holds a moving camera back. |

## Defaults by shot type

Tasks 15 and 20 build the template defaults from this table. "Build as" is
the exact chain setting. A row marked **Don's call** is built as its "build
as" (the recommendation) unless Don overrules it in `for-don.md`.

| shot type | build as | evidence (jobs) | status |
|---|---|---|---|
| **static** (locked-off camera, subject in front; S1), up to about 30 s (6-8 H3 segments) | `continuity: guide`, `anchor: segment_1`, `anchor_strength: 0.25`. Nothing else. | anchor 0.25 at seed 1166 `0a40c41af97c` and seed 2 `93dd359dc11e`, against guide controls `b40e855eb553`, `fd34bc937628`, `894911aed6c7` | **Don's call, recommended.** The texture gain holds on two seeds and by eye, but the strict rule says `mixed`: a secondary measure is worse past the floor on each seed (jump on 1166, coarse texture on seed 2), though never the same one twice. |
| static, alternative set | `continuity: last_frame`, `look_hold: true`, `anchor: segment_1`, `anchor_strength: 0.25`. Offered as a knob, not the default. | `835379468ec3`, `41355297a083` (0.25); `d8b17df7dfb9`, `c8bfadf286ce` (0.5); look_hold alone `be3f574df3db`, `b7ca6da7d075` | **Don's call, recommended: not the default.** It measures best on S1 and looks cleanest, but it changes the continuity, carries a mid-texture cost on both seeds, has not run at two minutes, and its look_hold holds a moving camera back. |
| static, longer than about 30 s | Cut it into shots of about 30 s or less. The chain gets no counter beyond the static row. | two-minute takes `8d1614ced27a` (control), `107859a31bc1` (anchor 0.5) | **Settled:** no counter set tested gives a usable two-minute static take (the control burns, anchor 0.5 dissolves the face). This is the drift gap that opens the design's second group. *(Superseded 2026-11-06: a static take past about 30 s is built as `shot: static_long` (guide, keystone `segment_1` at slot `last`, anchor off), which holds 120 s on two seeds; `structural.md` *Keystone slot*.)* |
| **moving camera** (arc, dolly; S2) | `continuity: guide`, no counters. Never `last_frame` with `look_hold` or `carry_pick`. | guide controls `dedd341c9108`, `01146e925c94`; every round-1 S2 arm; look_hold motion loss `242358078691`, `a3d6a044fd6b` | **Settled.** Every guide counter loses seam look; both last_frame counters hold the camera back, on both seeds for look_hold. |
| **dialogue walk** (speaking subject in motion; S3) | `continuity: guide`, no counters. | guide controls `c0b92d0d0cd4`, `64b4a78c74a2`; every round-1 S3 arm | **Settled.** Every counter is inside S3's wide floor or loses, and none looks better. |

## Re-scored under R35 (2026-11-05)

Everything above is kept as it was written. This section re-reads every
round-1 and round-2 verdict under ruling R35, which supersedes R26:

- **The floor.** Same-seed reruns are not noise-free. The floor is
  `combined_floor(seed floor, rerun_gap(pairs))` (`bench/verdict.py`): the
  larger, per measure, of the two-seed floor above and the same-seed rerun
  gap of the 8 round-2 combination pairs that two drivers both ran. A read
  across continuities takes the larger of the two continuities' floors. For
  the rerun gap and the floors per scene, see `noise-floor.md`, *Correction
  (2026-11-05, ruling R35)*. On every scene the seam floors are now jump
  0.124 (S3 0.332), look 0.438, look max 0.790, and level step 6.2 dB mean
  and 8.8 dB max. The S1 guide fine-texture floor stays 0.081.
- **The labels.** `bench/segment1.py` hashes segment 1 of each record's
  output (decoded video frames 0-59 and AAC frames 0-129) and labels it
  `clean` (the canonical fingerprint for its scene and seed) or `drifted`.
  The canonical entries are in `bench/records/segment1-canonical.json`,
  taken from cold-loaded jobs where one exists (S1 and S2 at seed 1166, S1
  at seed 2), and otherwise from the agreeing majority. The fingerprints
  are in `bench/records/segment1-fingerprints.json`. They were taken on lem,
  read-only, with lem's ffmpeg 7.0.2. The video halves match the
  nondeterminism report's (`d59a1259`, `ed5094c1`). The audio halves differ
  from the report's because a Mac ffmpeg decodes AAC differently.
- **Confounded.** A row is confounded when its two sides open on different
  segment-1 outputs: one drifted and one clean, or two different
  alternates. Under R35 a confounded verdict is re-scored, and re-run only
  if the re-score flips it.
- **Marks.**
  - *unchanged*: the same verdict in both columns, with every measure on
    the same side of the floor.
  - *weakened*: the same verdict, but some measure that was past the floor
    is now inside it.
  - **flipped**: the verdict changes in either column (all measures, or
    picture only).

  A flip of a row is not always a flip of the call that row fed. The calls
  are under [Calls under R35](#calls-under-r35).

### Segment-1 labels (rounds 1 and 2)

`batch 1` is the first, overwritten driver run of the round-2 combinations
(`counters-round2-combos-batch1.json`), measured again for the rerun gap.

| arm | job | key | segment 1 | label |
|---|---|---|---|---|
| ctl-guide-s1-s1166 | `b40e855eb553` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| ctl-guide-s1-s2 | `894911aed6c7` | s1-static@2 | `f47df963/f88334a9` | clean |
| ctl-lastframe-s1-s1166 | `3a73cb0f5d28` | s1-static@1166 | `56e0831e/ac39b399` | drifted |
| ctl-lastframe-s1-s2 | `f053abe1ce5f` | s1-static@2 | `4302cf65/03cb1fa0` | drifted |
| ctl-guide-s2-s1166 | `dedd341c9108` | s2-arc@1166 | `ed5094c1/0a819c94` | clean |
| ctl-guide-s2-s2 | `01146e925c94` | s2-arc@2 | `14a80618/b4d1d486` | clean |
| ctl-lastframe-s2-s1166 | `b235c35b72c1` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| ctl-lastframe-s2-s2 | `765fc66286bd` | s2-arc@2 | `130afe0c/f90613fc` | drifted |
| ctl-guide-s3-s1166 | `c0b92d0d0cd4` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| ctl-guide-s3-s2 | `64b4a78c74a2` | s3-dialogue-walk@2 | `31765efa/48363338` | clean |
| r1-anchor25-s1 | `0a40c41af97c` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r1-anchor50-s1 | `fa17a2c51e33` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r1-anchor75-s1 | `f17bce198a2d` | s1-static@1166 | `56e0831e/ac39b399` | drifted |
| r1-backoff17-s1 | `c594521e59d1` | s1-static@1166 | `fe993ff4/fb5cdcbf` | drifted |
| r1-keystone-s1 | `1e1b6116ce26` | s1-static@1166 | `fe993ff4/fb5cdcbf` | drifted |
| r1-lookhold-s1 | `751b43732fb9` | s1-static@1166 | `fe993ff4/fb5cdcbf` | drifted |
| r1-antiburn-s1 | `049d652e1648` | s1-static@1166+lora | `a76a366c/ff5c4416` | clean |
| r1-lf-carrypick-s1 | `ae932366ea0e` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r1-lf-lookhold-s1 | `be3f574df3db` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r1-anchor25-s2 | `4350da6ac745` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| r1-anchor50-s2 | `3408f7cf9fa9` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| r1-anchor75-s2 | `6fb052f46229` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| r1-backoff17-s2 | `ea394296a02d` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| r1-keystone-s2 | `f774bdbcc1cc` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| r1-lookhold-s2 | `64dc87a2ed3a` | s2-arc@1166 | `57cbd211/a1d44644` | drifted |
| r1-antiburn-s2 | `9124d417577d` | s2-arc@1166+lora | `c7b4068d/33f00579` | clean |
| r1-lf-carrypick-s2 | `9713480482bc` | s2-arc@1166 | `e6e70a03/0a819c94` | drifted |
| r1-lf-lookhold-s2 | `242358078691` | s2-arc@1166 | `ed5094c1/0a819c94` | clean |
| r1-anchor25-s3 | `7c1aa8f8b38d` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| r1-anchor50-s3 | `fef6b9aea7c9` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| r1-anchor75-s3 | `fbf286d9f192` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| r1-backoff17-s3 | `5873332b54fc` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| r1-keystone-s3 | `a2bd5e2545db` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| r1-lookhold-s3 | `a9a134a9acf8` | s3-dialogue-walk@1166 | `289012f1/d21b04b0` | clean |
| r1-antiburn-s3 | `c50ec5961418` | s3-dialogue-walk@1166+lora | `bfcb3f23/949a7e65` | clean |
| ctl-guide-s1-s1166-rerun | `fd34bc937628` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r2-anchor25-s1-seed2 | `93dd359dc11e` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-anchor50-s1-seed2 | `075ed6de83f7` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-keystone-s1-seed2 | `ab59d503fd4d` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-lf-lookhold-s1-seed2 | `b7ca6da7d075` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-lf-lookhold-s2-seed2 | `a3d6a044fd6b` | s2-arc@2 | `6fbb63c2/f6555387` | drifted |
| r2-anchor25-keystone-s1-s1166 | `8accd0c37f91` | s1-static@1166 | `abd3dabc/d5900069` | drifted |
| r2-anchor50-keystone-s1-s1166 | `e5c7b013538f` | s1-static@1166 | `cd167eee/d5900069` | drifted |
| r2-lf-lookhold-anchor25-s1-s1166 | `835379468ec3` | s1-static@1166 | `cd167eee/d5900069` | drifted |
| r2-lf-lookhold-anchor50-s1-s1166 | `d8b17df7dfb9` | s1-static@1166 | `df2c8907/6b31365e` | drifted |
| r2-anchor25-keystone-s1-s2 | `e34a5aebaab4` | s1-static@2 | `40991534/43283722` | drifted |
| r2-anchor50-keystone-s1-s2 | `08e32162985c` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-lf-lookhold-anchor25-s1-s2 | `41355297a083` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-lf-lookhold-anchor50-s1-s2 | `c8bfadf286ce` | s1-static@2 | `f47df963/f88334a9` | clean |
| r2-anchor25-keystone-s1-s1166 | `7c1c1fb14c8c` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r2-anchor50-keystone-s1-s1166 | `9331f354cc4b` | s1-static@1166 | `abd3dabc/d5900069` | drifted |
| r2-lf-lookhold-anchor25-s1-s1166 | `8010fb212bb3` | s1-static@1166 | `abd3dabc/d5900069` | drifted |
| r2-lf-lookhold-anchor50-s1-s1166 | `a58834fed81a` | s1-static@1166 | `abd3dabc/d5900069` | drifted |
| r2-anchor25-keystone-s1-s2 | `36c91319afbc` | s1-static@2 | `86544f72/4c5a2d39` | drifted |
| r2-anchor50-keystone-s1-s2 | `5a60b829547e` | s1-static@2 | `86544f72/4c5a2d39` | drifted |
| r2-lf-lookhold-anchor25-s1-s2 | `2059d7bcf69b` | s1-static@2 | `86544f72/4c5a2d39` | drifted |
| r2-lf-lookhold-anchor50-s1-s2 | `daa36f09dc3f` | s1-static@2 | `86544f72/4c5a2d39` | drifted |
| ctl-guide-s1-120s | `8d1614ced27a` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r2-anchor50-s1-120s | `107859a31bc1` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |

Summary:

- **S1, seed 1166.** Clean: the guide control and its rerun, anchor 0.25,
  anchor 0.5, both lf arms, both 120 s takes, and batch 1's anchor
  0.25 + keystone. Drifted:
  - anchor 0.75 and the last_frame control (`56e0831e`);
  - backoff, keystone and guide look_hold (`fe993ff4`);
  - all four recorded combinations, and three of batch 1's
    (`abd3dabc`, `cd167eee`, `df2c8907`).
- **S1, seed 2.** Clean: the guide control, all four round-2 confirms, and
  the three combinations run after the reboot. Drifted: the last_frame
  control (`4302cf65`), anchor 0.25 + keystone (`40991534`), and all four
  of batch 1's (`86544f72`).
- **S2, seed 1166.** Clean: the guide control, last_frame + look_hold, and
  the anti-burn LoRA run. Drifted: the last_frame control and **all six
  round-1 guide arms**, which share `57cbd211`; last_frame + carry_pick
  (`e6e70a03`).
- **S2, seed 2.** Clean: the guide control. Drifted: the last_frame control
  (`130afe0c`) and the round-2 last_frame + look_hold diagnostic
  (`6fbb63c2`).
- **S3.** Every round-1 record is clean.
- **Anti-burn LoRA arms.** Each is a cold load, labelled against its own
  `+lora` entry (the LoRA changes segment 1 by design).

### Verdicts re-scored

Every row of the round-1 tables, the continuity table, the seed-2
confirmation table and the combinations table, plus the round-2 S2
diagnostic and batch 1's clean anchor 0.25 + keystone (the only clean
seed-1166 combination). The primary measures are those in *How the
verdicts are read*. Italic names in the last two columns are measures
where the arm is better.

| round | arm | against | segment 1 (arm / against) | verdict before | verdict under R35 | mark | now inside the floor | past the floor under R35 (worse; _better_) |
|---|---|---|---|---|---|---|---|---|
| 1 | r1-anchor25-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: wins | **flipped** | jump, jump max, level, level max, _look_, _look max_ | -; _tex fine_, _tex mid_ |
| 1 | r1-anchor50-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | jump max, level, level max, _look_, _look max_ | jump, tex coarse; _tex fine_, _tex mid_ |
| 1 | r1-anchor75-s1 | ctl-guide-s1-s1166 | drifted `56e0831e` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | look, level max | jump, jump max, look max, level, tex coarse; _tex fine_, _tex mid_ |
| 1 | r1-backoff17-s1 | ctl-guide-s1-s1166 | drifted `fe993ff4` / clean (**confounded**) | tex fine: mixed | tex fine: tie | **flipped** | look max, level, level max | -; - |
| 1 | r1-keystone-s1 | ctl-guide-s1-s1166 | drifted `fe993ff4` / clean (**confounded**) | tex fine: mixed | tex fine: mixed (pic wins) | **flipped** | jump max, level max, _look max_ | level; _tex fine_, _tex mid_ |
| 1 | r1-lookhold-s1 | ctl-guide-s1-s1166 | drifted `fe993ff4` / clean (**confounded**) | tex fine: mixed | tex fine: tie | **flipped** | jump max, level, level max, _look_, _tex coarse_ | -; - |
| 1 | r1-antiburn-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump max_, _level max_ | look, look max, tex coarse; _jump_, _tex fine_, _tex mid_ |
| 1 | r1-lf-carrypick-s1 | ctl-lastframe-s1-s1166 | clean / drifted `56e0831e` (**confounded**) | tex fine: mixed (pic wins) | tex fine: tie | **flipped** | level, level max, _jump_, _jump max_, _look max_, _tex fine_, _tex coarse_ | -; - |
| 1 | r1-lf-lookhold-s1 | ctl-lastframe-s1-s1166 | clean / drifted `56e0831e` (**confounded**) | tex fine: mixed (pic wins) | tex fine: wins | **flipped** | level, _jump max_, _level max_, _tex coarse_ | -; _jump_, _look_, _look max_, _tex fine_ |
| 1 | r1-anchor25-s2 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: loses / jump: mixed | look: loses / jump: mixed | unchanged | - | look, look max; _jump_, _jump max_, _tex fine_, _tex mid_ |
| 1 | r1-anchor50-s2 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: loses / jump: loses | look: loses / jump: loses | weakened | jump max | jump, look, look max; - |
| 1 | r1-anchor75-s2 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: loses / jump: loses | look: loses / jump: loses | unchanged | - | jump, jump max, look, look max; _tex fine_ |
| 1 | r1-backoff17-s2 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: loses / jump: mixed | look: mixed / jump: mixed | **flipped** | look | look max; - |
| 1 | r1-keystone-s2 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: loses / jump: mixed | look: tie / jump: tie | **flipped** | look, look max | -; - |
| 1 | r1-lookhold-s2 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: loses / jump: mixed | look: mixed / jump: mixed | **flipped** | look | look max; - |
| 1 | r1-antiburn-s2 | ctl-guide-s2-s1166 | clean / clean | look: loses / jump: mixed | look: tie / jump: tie | **flipped** | look, look max, _jump_, _jump max_ | -; - |
| 1 | r1-lf-carrypick-s2 | ctl-lastframe-s2-s1166 | drifted `e6e70a03` / drifted `57cbd211` (**confounded**) | look: mixed / jump: loses | look: tie / jump: tie | **flipped** | jump, _look_, _look max_ | -; _tex fine_, _tex mid_, _tex coarse_ |
| 1 | r1-lf-lookhold-s2 | ctl-lastframe-s2-s1166 | clean / drifted `57cbd211` (**confounded**) | look: wins / jump: tie | look: wins / jump: tie | unchanged | - | -; _look_, _look max_, _tex fine_, _tex mid_, _tex coarse_ |
| 1 | r1-anchor25-s3 | ctl-guide-s3-s1166 | clean / clean | jump: mixed / look: mixed / tex fine: mixed | jump: mixed / look: mixed / tex fine: mixed | weakened | jump max, look max | tex coarse; - |
| 1 | r1-anchor50-s3 | ctl-guide-s3-s1166 | clean / clean | jump: mixed / look: mixed / tex fine: mixed | jump: mixed / look: mixed / tex fine: mixed | weakened | jump max, look max, level | tex coarse; - |
| 1 | r1-anchor75-s3 | ctl-guide-s3-s1166 | clean / clean | jump: mixed / look: loses / tex fine: mixed | jump: mixed / look: loses / tex fine: mixed | unchanged | - | jump max, look, look max, tex coarse; - |
| 1 | r1-backoff17-s3 | ctl-guide-s3-s1166 | clean / clean | jump: mixed / look: mixed / tex fine: loses | jump: mixed / look: mixed / tex fine: loses | weakened | jump max, level | tex fine, tex mid; _tex coarse_ |
| 1 | r1-keystone-s3 | ctl-guide-s3-s1166 | clean / clean | jump: mixed / look: mixed / tex fine: mixed | jump: tie / look: tie / tex fine: tie | **flipped** | jump max, look max, level | -; - |
| 1 | r1-lookhold-s3 | ctl-guide-s3-s1166 | clean / clean | jump: tie / look: tie / tex fine: tie | jump: tie / look: tie / tex fine: tie | weakened | _look max_ | -; - |
| 1 | r1-antiburn-s3 | ctl-guide-s3-s1166 | clean / clean | jump: loses / look: mixed / tex fine: loses | jump: loses / look: mixed / tex fine: loses | weakened | look max, level | jump, jump max, tex fine, tex mid, tex coarse; - |
| 1 | ctl-lastframe-s1-s1166 | ctl-guide-s1-s1166 | drifted `56e0831e` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | jump max, _look_, _look max_, _level_, _level max_ | jump, tex mid; - |
| 1 | ctl-lastframe-s2-s1166 | ctl-guide-s2-s1166 | drifted `57cbd211` / clean (**confounded**) | look: mixed / jump: mixed | look: mixed / jump: mixed | weakened | level, _jump_, _jump max_ | tex mid, tex coarse; - |
| 1 | r1-lf-carrypick-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | jump max, _look_, _look max_, _level max_ | jump, tex mid; _tex coarse_ |
| 1 | r1-lf-lookhold-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump_, _jump max_, _level_ | tex mid; _look_, _look max_, _level max_, _tex coarse_ |
| 1 | r1-lf-carrypick-s2 | ctl-guide-s2-s1166 | drifted `e6e70a03` / clean (**confounded**) | look: wins / jump: tie | look: tie / jump: tie | **flipped** | _jump max_, _look_, _look max_ | -; _tex fine_ |
| 1 | r1-lf-lookhold-s2 | ctl-guide-s2-s1166 | clean / clean | look: wins / jump: tie | look: wins / jump: tie | weakened | _jump max_ | -; _look_, _look max_, _tex fine_, _tex mid_ |
| 2 | r1-anchor25-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: wins | **flipped** | jump, jump max, level max, _look_, _look max_ | -; _tex fine_, _tex mid_ |
| 2 | r2-anchor25-s1-seed2 | ctl-guide-s1-s2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _look max_ | tex coarse; _tex fine_ |
| 2 | r1-anchor50-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | jump max, level max, _look_, _look max_ | jump, tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-anchor50-s1-seed2 | ctl-guide-s1-s2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | jump, jump max, _look max_ | tex coarse; _tex fine_ |
| 2 | r1-keystone-s1 | ctl-guide-s1-s1166 | drifted `fe993ff4` / clean (**confounded**) | tex fine: mixed | tex fine: mixed (pic wins) | **flipped** | jump max, level max, _look max_ | level; _tex fine_, _tex mid_ |
| 2 | r2-keystone-s1-seed2 | ctl-guide-s1-s2 | clean / clean | tex fine: mixed | tex fine: tie | **flipped** | look max | -; - |
| 2 | r1-lf-lookhold-s1 | ctl-lastframe-s1-s1166 | clean / drifted `56e0831e` (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump max_ | -; _jump_, _look_, _look max_, _tex fine_ |
| 2 | r2-lf-lookhold-s1-seed2 | ctl-lastframe-s1-s2 | clean / drifted `4302cf65` (**confounded**) | tex fine: wins | tex fine: tie | **flipped** | _jump max_, _look max_, _tex fine_ | -; _jump_, _look_ |
| 2 | r1-lf-lookhold-s1 | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump_, _jump max_ | tex mid; _look_, _look max_, _level max_, _tex coarse_ |
| 2 | r2-lf-lookhold-s1-seed2 | ctl-guide-s1-s2 | clean / clean | tex fine: loses | tex fine: loses | weakened | jump max | tex fine, tex mid; _look_, _look max_, _level max_ |
| 2 | r2-anchor25-keystone-s1-s1166 | ctl-guide-s1-s1166 | drifted `abd3dabc` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | jump max, _look_, _look max_ | tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-anchor25-keystone-s1-s1166 | r1-anchor25-s1 | drifted `abd3dabc` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | _look max_, _level max_ | tex coarse; - |
| 2 | r2-anchor25-keystone-s1-s2 | ctl-guide-s1-s2 | drifted `40991534` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | jump max, level max, _look max_ | level, tex coarse; _tex fine_ |
| 2 | r2-anchor25-keystone-s1-s2 | r2-anchor25-s1-seed2 | drifted `40991534` / clean (**confounded**) | tex fine: mixed | tex fine: mixed (pic tie) | **flipped** | jump max | level max; - |
| 2 | r2-anchor50-keystone-s1-s1166 | ctl-guide-s1-s1166 | drifted `cd167eee` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | look max | jump, jump max, look, tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-anchor50-keystone-s1-s1166 | r1-anchor50-s1 | drifted `cd167eee` / clean (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | jump max, _level max_ | jump, look, look max, tex coarse; _level_ |
| 2 | r2-anchor50-keystone-s1-s2 | ctl-guide-s1-s2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | look, look max, level max | jump, jump max, tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-anchor50-keystone-s1-s2 | r2-anchor50-s1-seed2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | level max | jump, jump max, look, look max, tex coarse; - |
| 2 | r2-lf-lookhold-anchor25-s1-s1166 | ctl-lastframe-s1-s1166 | drifted `cd167eee` / drifted `56e0831e` (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump max_ | -; _jump_, _look_, _look max_, _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor25-s1-s1166 | ctl-guide-s1-s1166 | drifted `cd167eee` / clean (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump_, _level max_ | -; _look_, _look max_, _tex fine_ |
| 2 | r2-lf-lookhold-anchor25-s1-s1166 | r1-anchor25-s1 | drifted `cd167eee` / clean (**confounded**) | tex fine: tie | tex fine: tie | weakened | _jump max_ | -; _jump_, _look_, _look max_, _level_, _level max_ |
| 2 | r2-lf-lookhold-anchor25-s1-s1166 | r1-lf-lookhold-s1 | drifted `cd167eee` / clean (**confounded**) | tex fine: mixed | tex fine: wins | **flipped** | jump max, _look max_ | -; _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor25-s1-s2 | ctl-lastframe-s1-s2 | clean / drifted `4302cf65` (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump max_ | -; _jump_, _look_, _look max_, _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor25-s1-s2 | ctl-guide-s1-s2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump max_ | tex mid; _look_, _look max_, _level max_, _tex fine_ |
| 2 | r2-lf-lookhold-anchor25-s1-s2 | r2-anchor25-s1-seed2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump_, _jump max_, _look_ | tex mid; _look max_, _level_, _level max_, _tex coarse_ |
| 2 | r2-lf-lookhold-anchor25-s1-s2 | r2-lf-lookhold-s1-seed2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump_, _jump max_, _look max_ | tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor50-s1-s1166 | ctl-lastframe-s1-s1166 | drifted `df2c8907` / drifted `56e0831e` (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump max_, _look max_ | -; _jump_, _look_, _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor50-s1-s1166 | ctl-guide-s1-s1166 | drifted `df2c8907` / clean (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump_ | -; _look_, _look max_, _level max_, _tex fine_ |
| 2 | r2-lf-lookhold-anchor50-s1-s1166 | r1-anchor50-s1 | drifted `df2c8907` / clean (**confounded**) | tex fine: wins | tex fine: wins | weakened | _jump max_, _look max_ | -; _jump_, _level_, _level max_, _tex fine_, _tex coarse_ |
| 2 | r2-lf-lookhold-anchor50-s1-s1166 | r1-lf-lookhold-s1 | drifted `df2c8907` / clean (**confounded**) | tex fine: mixed | tex fine: wins | **flipped** | jump max, look max, _jump_ | -; _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor50-s1-s2 | ctl-lastframe-s1-s2 | clean / drifted `4302cf65` (**confounded**) | tex fine: mixed | tex fine: mixed | weakened | _jump max_, _look max_ | tex coarse; _jump_, _look_, _tex fine_, _tex mid_ |
| 2 | r2-lf-lookhold-anchor50-s1-s2 | ctl-guide-s1-s2 | clean / clean | tex fine: mixed | tex fine: mixed | unchanged | - | tex mid, tex coarse; _look_, _look max_, _level max_, _tex fine_ |
| 2 | r2-lf-lookhold-anchor50-s1-s2 | r2-anchor50-s1-seed2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | _jump max_, _look_, _look max_ | tex mid, tex coarse; _jump_, _level max_ |
| 2 | r2-lf-lookhold-anchor50-s1-s2 | r2-lf-lookhold-s1-seed2 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | look max, _jump_, _jump max_, _level max_ | tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-anchor25-keystone-s1-s1166, batch 1 (`7c1c1fb14c8c`) | ctl-guide-s1-s1166 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | look max | jump, jump max, tex coarse; _tex fine_, _tex mid_ |
| 2 | r2-anchor25-keystone-s1-s1166, batch 1 (`7c1c1fb14c8c`) | r1-anchor25-s1 | clean / clean | tex fine: mixed | tex fine: mixed | weakened | jump, look, look max, _level max_ | jump max, tex coarse; _level_, _tex fine_ |
| 2 | r2-lf-lookhold-s2-seed2 | ctl-lastframe-s2-s2 | drifted `6fbb63c2` / drifted `130afe0c` (**confounded**) | look: wins / jump: tie | look: wins / jump: tie | unchanged | - | -; _look_, _look max_, _tex fine_, _tex mid_, _tex coarse_ |

Count: 68 rows. 6 unchanged, 42 weakened, 20 flipped. 15 of the flips are
on confounded rows. The 5 clean flips (anchor 0.25 on both readings, and
keystone at S3, at S1 seed 2, and anti-burn at S2) are final under the
floor: a same-seed rerun cannot settle them.

### Calls under R35

- **S2 has no candidate: weakened, not flipped.** No S2 arm wins a primary
  under R35, so there is still no candidate. What changes is the reason.
  - Anchor 0.25, 0.5 and 0.75 still lose seam look by 0.75-2.2, past the
    0.438 floor.
  - Backoff and guide look_hold move from *loses* to *mixed*. Their
    seam-look means are inside the floor, and only look max is worse past
    it (2.61 and 2.97 against 1.74, floor 0.79).
  - Keystone moves to *tie* on both primaries.
  - Anti-burn is clean on both sides and also ties on the numbers; by eye
    it is broken.

  So "every guide counter loses seam look" holds for the anchor only. All
  six guide arms ran in one drifted worker state (`57cbd211`) against a
  clean control, but no flip makes any of them a win, and the eye notes
  saw no improvement. No rerun is needed for the call. The guide and
  last_frame S2 settings in *Defaults by shot type* stand.
- **Round-1 S1 anchor 0.75: unchanged call (not carried), weakened row.**
  This row is confounded: the arm opens on `56e0831e` and the control is
  clean. It is still *mixed*, with jump, jump max, look max, level and
  coarse texture worse past the floor. Only the seam-look cost is now inside
  it. Identity drift by eye stands. No rerun.
- **Round-1 S1 backoff 17: unchanged call (cut).** The row flips from *mixed*
  to *tie* (confounded, `fe993ff4`). The costs fall inside the floor, but
  the texture gain is still inside it too (0.462 against 0.536, floor
  0.081). S3 is clean and still loses texture (0.337 against 0.067, floor
  0.226). On S2 the row moves from *loses* to *mixed*. Backoff wins nowhere.
- **Round-1 S1 keystone: unchanged call (cut).** The 1166 row flips on
  picture only, from *mixed* to *wins* (confounded, `fe993ff4`), because its
  jump max cost is now inside the floor. The cut rested on seed 2, and seed
  2 is clean on both sides and now a plain *tie* (0.448 against 0.455).
  A winner must hold on both seeds, so the call does not move. S3 (clean)
  is now *tie* on every primary, and S2 is *tie* as well.
  - **Keystone with anchor 0.25: weakened.** Batch 1's clean 1166 run
    (`7c1c1fb14c8c`) reads better than the recorded drifted one. Its
    texture beats anchor 0.25 alone by 0.084, past the 0.081 floor by
    0.003, but jump max and coarse texture are worse past the floor
    (*mixed*). Its seam look is 1.53, about the control's 1.46, not the 1.09
    the drifted record showed. So "keystone adds nothing to anchor 0.25"
    becomes "a texture gain at the edge of the floor, paid for in seams".
    The cut stands.
- **Round-1 S1 guide look_hold: unchanged call (cut).** The row flips from
  *mixed* to *tie* (confounded). On S2 it moves from *loses* to *mixed* on
  look max, and on S3 it is still a tie.
- **The last_frame controls: all four are drifted.** S1 at 1166 is
  `56e0831e`, S1 at seed 2 is `4302cf65`, S2 at 1166 is `57cbd211`, and S2
  at seed 2 is `130afe0c`. Every row read against one is confounded:
  - **last_frame + look_hold, S1, against its own control: undetermined.**
    At 1166 the row goes from *mixed* to *wins*, because the level cost is
    now inside the floor. At seed 2 it goes from *wins* to *tie*: texture
    is 0.550 against 0.605, inside the new 0.068 floor. So the claim "a
    strict win against the last_frame control on both seeds" no longer
    holds as measured, and only clean controls can settle it. The call in
    *Defaults by shot type* (a knob, not the default) does not rest on this
    claim. It rests on the reads against guide, which are clean and
    unchanged (*mixed* at 1166 on mid texture, *loses* at seed 2), and on
    the S2 motion loss by eye. **The call stands.**
  - **last_frame + carry_pick, S1: flipped toward the cut.** It goes from
    *mixed* (picture *wins*) to *tie*. The cut stands.
  - **Continuity, last_frame against guide (the controls): unchanged.** On
    S1, last_frame's seam-look and level advantages are now inside the
    floor. It is still worse past the floor on jump and mid texture. On S2,
    its jump advantage is now inside the floor, and its mid and coarse
    texture costs remain. last_frame is still not better than guide as the
    default.
  - **S2 last_frame arms.** carry_pick flips to *tie* on both primaries.
    look_hold is unchanged (*wins* look), and so is the seed-2 diagnostic.
    Both were rejected by eye for holding the camera back, and that stands.
  - **The lf + look_hold + anchor set: unchanged against guide.** It beats
    the guide control on texture and seam look on all four runs. Against
    look_hold alone, both 1166 rows (confounded) flip from *mixed* to
    *wins*, while seed 2 (clean) is still *mixed*. It stays Don's call and
    not the default.
- **Anchor 0.25 (the static default): strengthened.** Both sides are clean
  on both seeds. At 1166 the row flips from *mixed* to a strict *wins*: the
  jump cost (0.928 against 0.854) is inside the 0.124 floor. At seed 2 the
  row is still *mixed*, on coarse texture only (0.031 against 0.008, floor
  0.010).
- **Anchor 0.5: weakened.** The seed-2 jump cost (0.904 against 0.796) is
  now inside the floor, so the jump cost repeats at 1166 only. Coarse
  texture is still worse past the floor on both seeds. The recommendation
  of 0.25 over 0.5 still stands on three points:
  - the face dissolves at two minutes (clean, `107859a31bc1`);
  - 0.5's texture lead over 0.25 at seed 2 is inside the floor;
  - the coarse-texture cost.
- **Anti-burn LoRA: unchanged call (cut).** On S2 it flips to *tie* (both
  sides clean). It still loses S3 (clean), and by eye it blurs every scene.
- **S3: unchanged (no counters).** Every record is clean. Keystone moves
  from *mixed* to *tie*, and nothing wins.
- **Two-minute takes: unchanged.** Both are clean (`d59a1259`) and are read
  against each other.
- **The level-step question: nearly moot.** With a 6.2/8.8 dB floor, only
  two re-scored rows still differ between the all-measures and the
  picture-only column:
  - r1-keystone-s1 (level 14.6 against 7.7 dB);
  - anchor 0.25 + keystone at seed 2 against anchor 0.25 alone (level
    max).

  Neither decides a call, so the recommendation to read S1 and S2 on
  picture changes nothing.

### Reruns

**Minimum set: 2 runs.** These are the only call that is undetermined
(last_frame + look_hold against its own control) and the only control
reads R35 cannot score:

1. `ctl-lastframe-s1-s1166` (`bench/arms/counters-controls.json`)
2. `ctl-lastframe-s1-s2`

How to run them:
- Give each a fresh `workflow_id_suffix` (for example `-r35`) so the step
  cache cannot serve it.
- Run each as the first job after a model load (a fresh bench server, or a
  canary control just before), and confirm `clean` with `bench/segment1.py`
  before reading it.
- They take about 9 minutes each, plus about 2 minutes of load.

These two runs re-score, against a clean control, every row now read
against a drifted one:
- r1-lf-lookhold-s1 and r2-lf-lookhold-s1-seed2 (the undetermined claim);
- r1-lf-carrypick-s1;
- the four lf combinations against the last_frame control;
- the S1 continuity row.

**Not needed for any call** (a confounded row flipped, but the call it fed
is decided elsewhere):
- r1-keystone-s1 (picture *wins*; the cut rests on the clean seed-2 tie);
- r1-backoff17-s1 and r1-lookhold-s1 (to *tie*; cut);
- r1-backoff17-s2, r1-keystone-s2 and r1-lookhold-s2 (no S2 win either
  way);
- r1-lf-carrypick-s2 and the S2 last_frame controls (rejected by eye);
- r2-anchor25-keystone-s1-s2 (picture *tie* against anchor alone; cut).

A clean rerun of r1-keystone-s1 is the cheapest to add (one run) if the
seed-1166 keystone row should read clean as well.

### Clean last_frame controls (2026-11-05)

The minimum set ran (`bench/arms/lastframe-ctl-clean-rerun.json`,
`bench/records/lastframe-ctl-clean-rerun.json`), each the first job after a
bench server restart on cuda:0 alone, with a fresh workflow-id suffix. Both
are **clean** by `bench/segment1.py`:

| arm | job | segment 1 | jump | jump max | look | look max | level dB | level dB max | tex fine | tex mid | tex coarse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl-lastframe-s1-s1166-clean | `88250351f1fe` | `d59a1259/b2c2c63e` clean | 1.036 | 1.190 | 1.254 | 1.730 | 4.508 | 9.350 | 0.522 | 0.342 | 0.002 |
| ctl-lastframe-s1-s2-clean | `194ee59b559d` | `f47df963/f88334a9` clean | 1.072 | 1.270 | 1.340 | 1.760 | 4.698 | 13.860 | 0.593 | 0.448 | 0.011 |

Against the drifted originals the clean controls moved fine texture by
0.047 (1166) and 0.012 (seed 2), inside the rerun gap. The S1 last_frame
floor, `combined_floor(noise_floor(clean 1166, clean 2), rerun gap)`, is
jump 0.124, jump max 0.430, look 0.438, look max 0.790, level 6.204 / 8.800
dB, tex fine **0.071**, tex mid 0.106, tex coarse 0.010.

| arm | against (clean) | tex fine arm / ctl | verdict (all) | verdict (picture) | past the floor (worse; _better_) |
|---|---|---|---|---|---|
| r1-lf-lookhold-s1 | ctl-lastframe-s1-s1166-clean | 0.491 / 0.522 | tie | tie | -; _jump, look_ |
| r2-lf-lookhold-s1-seed2 | ctl-lastframe-s1-s2-clean | 0.550 / 0.593 | tie | tie | -; _jump, look_ |
| r1-lf-carrypick-s1 | ctl-lastframe-s1-s1166-clean | 0.533 / 0.522 | tie | tie | -; - |
| r2-lf-lookhold-anchor25-s1-s1166 (drifted `cd167eee`) | ctl-lastframe-s1-s1166-clean | 0.331 / 0.522 | wins | wins | -; _jump, look, look max, tex fine, tex mid_ |
| r2-lf-lookhold-anchor50-s1-s1166 (drifted `df2c8907`) | ctl-lastframe-s1-s1166-clean | 0.183 / 0.522 | wins | wins | -; _jump, look, tex fine, tex mid_ |
| r2-lf-lookhold-anchor25-s1-s2 | ctl-lastframe-s1-s2-clean | 0.297 / 0.593 | wins | wins | -; _jump, jump max, look, look max, tex fine, tex mid_ |
| r2-lf-lookhold-anchor50-s1-s2 | ctl-lastframe-s1-s2-clean | 0.236 / 0.593 | mixed | mixed | tex coarse; _jump, look, level max, tex fine, tex mid_ |
| continuity: ctl-lastframe-s1-s1166-clean | ctl-guide-s1-s1166 (cross floor, tex fine 0.081) | 0.522 / 0.536 | mixed | mixed | jump, tex mid; _tex coarse_ |
| continuity: ctl-lastframe-s1-s2-clean | ctl-guide-s1-s2 (cross floor) | 0.593 / 0.455 | loses | loses | jump, tex fine, tex mid; - |

- **"last_frame + look_hold beats its own control on both seeds": settled,
  no.** On the primary (fine texture) it is a **tie on both seeds** (gains
  0.031 and 0.043 against a 0.071 floor). What it does beat, past the floor
  on both seeds, is seam jump and seam look mean: look_hold smooths the
  last_frame hand-over, not the burn. The call in *Defaults by shot type*
  (a knob, not the default) stands, now with clean evidence.
- **last_frame + carry_pick, S1: tie, clean on both sides.** The cut stands.
- **The lf + look_hold + anchor set** beats the clean last_frame control on
  texture at both seeds (*wins* three of four; anchor 0.5 at seed 2 is
  *mixed* on coarse texture). The anchor carries the gain, as against guide.
  Still Don's call, not the default.
- **Continuity (last_frame against guide), clean on both sides:** *mixed* at
  1166 and *loses* at seed 2. Guide stays the default.

## Round 3 (2 min), 2026-11-05

Round 3 asks one question: does Don's static default (guide, `anchor:
segment_1`, `anchor_strength: 0.25`) hold at two minutes? Two S1 runs, both
seed 1166, 120 s, 2880 frames, on cuda:0
(`bench/arms/counters-round3-long.json`, `bench/records/counters-round3-long.json`):

- `r3-anchor25-s1-120s` (`25f6bd1da507`): guide + anchor 0.25. 28 seams.
- `r3-lf-lookhold-anchor25-s1-120s` (`9877bbfd85a8`): last_frame + look_hold +
  anchor 0.25 (the static alternative set). 23 seams (last_frame segments
  are about 5.1 s, not 4.25 s).

The control is round 2's `ctl-guide-s1-120s` (`8d1614ced27a`), with
`r2-anchor50-s1-120s` (`107859a31bc1`) beside it. The eight-moment texture
and per-seam pass is `bench/records/counters-round3-long-moments.json`. It
was taken in-process on lem with the deployed `dw.media_compare.compare`
and `dw.server.assess.assess` (the code behind the compare and assess
routes), not through the server. It reproduces round 2's moments record
and the driver's measures exactly.

### Segment-1 labels

| arm | job | key | segment 1 | label |
|---|---|---|---|---|
| r3-anchor25-s1-120s | `25f6bd1da507` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r3-lf-lookhold-anchor25-s1-120s | `9877bbfd85a8` | s1-static@1166 | `91deb574/008eae4c` | **drifted** (a new alternate) |
| ctl-guide-s1-120s | `8d1614ced27a` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |
| r2-anchor50-s1-120s | `107859a31bc1` | s1-static@1166 | `d59a1259/b2c2c63e` | clean |

So anchor 0.25 against the control is a clean read. Every read of the
last_frame run is confounded.

### Floor

`combined_floor(seed floor, rerun gap)`, the R35 S1 guide row: jump
0.124/0.430, look 0.438/0.790, level 6.2/8.8 dB, texture 0.081/0.032/0.010.
For reads against the last_frame run, the S1 last_frame seed floor is
folded in as well, which only lifts mid texture to 0.087. These are
6-segment floors. **There is no two-minute floor**, and every two-minute
run is a single seed.

### Texture over time

Drift against frame 62 on the S1 crop `[0,0,240,150]` (the circle and the
top of the stripes), `|ln(band / band@62)|`:

| frame (s) | 300 (12.5) | 583 (24.3) | 1000 (41.7) | 1501 (62.5) | 2000 (83.3) | 2500 (104.2) | 2868 (119.5) |
|---|---|---|---|---|---|---|---|
| control fine | 0.180 | 0.536 | 1.108 | 1.399 | 1.646 | 1.658 | 1.706 |
| **anchor 0.25 fine** | 0.153 | 0.367 | 0.497 | 0.771 | 0.931 | 1.012 | 1.108 |
| lf + look_hold + anchor 0.25 fine | 0.144 | 0.253 | 0.478 | 0.575 | 0.408 | 0.697 | 0.930 |
| anchor 0.5 fine | 0.137 | 0.270 | 0.360 | 0.602 | 0.737 | 0.759 | 1.067 |
| control mid | 0.045 | 0.176 | 0.406 | 0.716 | 1.071 | 1.156 | 1.171 |
| **anchor 0.25 mid** | 0.046 | 0.124 | 0.179 | 0.271 | 0.319 | 0.356 | 0.430 |
| lf + look_hold + anchor 0.25 mid | 0.072 | 0.185 | 0.251 | 0.368 | 0.351 | 0.411 | 0.637 |
| anchor 0.5 mid | 0.038 | 0.116 | 0.162 | 0.227 | 0.307 | 0.273 | 0.462 |

On the numbers, anchor 0.25 both **delays and reduces** the burn:

- **Delay.** The control reaches 0.54 fine drift at 24 s (segment 6). Anchor
  0.25 reaches the same level between 42 and 62 s (segments 10-14), so it
  roughly doubles the clean length.
- **Reduction.** It ends at 1.11 fine and 0.43 mid, against the control's 1.71
  and 1.17. Mid stays at about a third of the control's from 42 s on.
- **Against 0.5.** On fine texture, 0.25 runs 0.1-0.2 above 0.5 from 24 to
  104 s. At the end the two are inside the floor (1.108 against 1.067).
  Mid texture is close all the way.

### Seams

By stretch of the take (`analyze_seams`, every seam, mean (max)):

| seconds | control jump | anchor 0.25 jump | lf+lh+a0.25 jump | control look | anchor 0.25 look | lf+lh+a0.25 look |
|---|---|---|---|---|---|---|
| 0-25 | 0.85 (0.97) | 0.93 (1.15) | 0.72 (0.78) | 1.46 (2.13) | 1.13 (1.98) | 0.71 (0.83) |
| 25-60 | 1.17 (1.64) | 0.88 (1.29) | 0.83 (0.99) | 1.88 (2.61) | 0.78 (2.37) | 0.77 (1.03) |
| 60-95 | 1.29 (1.94) | 0.80 (0.96) | 1.33 (**4.24**) | 1.71 (3.75) | 0.68 (1.53) | 1.53 (**6.36**) |
| 95-120 | 2.06 (**6.56**) | 1.26 (1.86) | 1.09 (1.27) | 4.12 (**11.51**) | 1.81 (2.70) | 2.00 (4.71) |

- Anchor 0.25's worst seam is the very last one, at 119.9 s (jump 1.86, look
  2.70). It never passes the chain check's 4.0.
- The last_frame run is the best set to about 60 s. Then it has one hard
  seam at 76.3 s (jump 4.24, look 6.36), and its look degrades through the
  last 25 s.
- Level step: anchor 0.25 is 5.8 dB mean (24.3 max) against the control's
  5.9 (36.5); last_frame is 5.5 (15.0). Every level mean is inside the
  6.2 dB floor. Anchor 0.25's loud steps come in the first minute (12.2 and
  9.9 dB means), as on every S1 guide arm.

### Verdicts (`verdict()`, primary `texture_fine_seg_last`)

| read | verdict (all / picture) | past the floor |
|---|---|---|
| anchor 0.25 vs control (clean, clean) | **wins / wins** | better on all 8 picture measures and on level max; level mean inside |
| lf+lh+a0.25 vs control (drifted, clean) | wins / wins (confounded) | better on all 8 picture measures and on level max |
| lf+lh+a0.25 vs anchor 0.25 (drifted, clean) | mixed / mixed (confounded) | fine better (0.930 vs 1.108); **worse** jump max (+2.38), look max (+3.66), mid (+0.21), coarse (+0.05) |
| anchor 0.25 vs anchor 0.5 (clean, clean) | tie / tie | only coarse texture better (0.019 vs 0.089) |

**Motion.** S1 is locked off, and the only motion measure the bench takes is
the within-segment frame delta at each seam (`typical_delta`). It is 2.0-2.4 on
every run, with no stall and no surge. By eye the last_frame run's subject
turns her head at 83 s; the guide runs stay frontal.

### Eye check: face and backdrop

Face crops (`crop=240:240:360:30`) at 24/42/52/62/83/104/112/119.5 s, and
full frames at the same moments:

- **Control:** as in round 2. The backdrop crusts from 24 s and the face
  cracks from 42 s. From 83 s she is a mottled mosaic, though still the
  same person.
- **Anchor 0.25:** clean at 24 s. At 42 s the backdrop is visibly crusting
  (the stripes and the circle). The face is waxy, with heavy blush, but holds.
  At 52 s it softens, and at 62 s it is pale and flattened. From 83 s it is
  washed out, and by 104-120 s it is a pale mask, with a white patch over the
  forehead and faint features. **This is the anchor 0.5 failure, about 20 s
  later and milder**: the eyes, nose and mouth are still there at 120 s,
  where 0.5's eyes had gone by 62-83 s. The backdrop is a full mosaic by
  62 s. The texture number after about 60 s is part blur (#1175 trap),
  as with 0.5.
- **last_frame + look_hold + anchor 0.25:** the best face hold of any
  two-minute run. The face is natural to 62 s and still present at 83 s
  (head turned). It smears from 104 s and is dissolving into crust at
  120 s. The backdrop burns like anchor 0.25's. It opens on a drifted segment 1
  (the woman at 24 s is visibly a different take from the guide runs').

Export for Don (the last 20 s of each, 100-120 s, re-encoded with x264 at
CRF 16): `~/Desktop/dw-bench-r3/` on the Mac:
`r3-guide-anchor25-last20s.mp4`, `r3-lastframe-lookhold-anchor25-last20s.mp4`
and `r2-guide-control-last20s.mp4`.

### The call

- **Guide + anchor 0.25 still beats the guide control at two minutes.** This
  is a strict *wins* on a clean read, with every picture measure better by
  3-23x its floor, and the margin grows with length. The burn is delayed, not
  removed:
  - visible backdrop crust from segment 10 (about 42 s);
  - the face softens from segment 13 (52 s);
  - the face is washed out from segment 20 (83 s).

  The control burns from segment 6 (24 s).
- **The static default stands, and so does the cut rule.** A two-minute
  static take is still not usable on any arm. Anchor 0.25 moves the usable
  length from about 30 s to about 35-40 s at best, which is not enough to
  change "cut static shots at about 30 s (6-8 segments)". The bench's
  conclusion is unchanged: the long take needs the second group (#1150,
  #1044, #1030) or the per-segment check and re-roll.
- **0.25 is better than 0.5 at length.** The numbers tie (coarse aside), but
  by eye 0.25 keeps the face about 20 s longer and never loses the eyes. This
  confirms the round-2 recommendation.
- **last_frame + look_hold + anchor 0.25 against guide + anchor 0.25: not
  settled, and confounded.** It holds the face longer and ends lower on fine
  texture. It is worse past the floor on mid and coarse texture and on the
  worst seam (4.24 jump at 76 s, past the 4.0 check). It is also 21 min
  faster (48 against 69 min). It opens on a drifted segment 1 (`91deb574`), so
  none of these reads is clean, and the mid-texture cost repeats round 2's
  on both seeds. The *Defaults by shot type* row (a knob, not the default)
  stands.
- **Minimum clean rerun: one run.** Run `r3-lf-lookhold-anchor25-s1-120s` with
  a fresh `workflow_id_suffix`, as the first job after a cold model load, and
  confirm it `clean` (`d59a1259`) with `bench/segment1.py` before reading it.
  This takes about 48 min plus the load. No rerun is needed for the anchor 0.25
  call. Both sides are clean, and every margin is far past the 6-segment
  floor. A seed-2 two-minute pair would give the first two-minute floor if
  one is wanted.
- **Cost:** anchor 0.25 took 69 min, against the control's 52 min (+32%).
  Round 2's anchor 0.5 took 68 min.
