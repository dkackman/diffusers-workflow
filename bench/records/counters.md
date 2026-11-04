# Counter round (Task 11)

Round 1 ran every counter alone against today's chain at 6 segments (ruling
R27), on S1 static, S2 arc and S3 dialogue walk, at seed 1166. Records:
`bench/records/counters-controls.json` (10 controls: guide on S1-S3 and
last_frame on S1-S2, each at seeds 1166 and 2) and
`bench/records/counters-round1.json` (25 arms). All 35 succeeded on the bench
server (lem, port 8766).

## How the verdicts are read

- **Control.** Each arm is read against its own scene's control at its own
  continuity, at seed 1166 (`ctl-guide-*` or `ctl-lastframe-*`).
- **Floor.** The noise floor is seed variance (ruling R26):
  `noise_floor(control@1166, control@2)` for each scene and continuity. It
  comes from two seeds only, so it is thin. A measure where the two seeds
  happened to agree has a floor near 0, and any difference counts on it
  (S1 last_frame `jump max` is 0.000, for example).
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

## Intended defaults

None yet. Defaults move after round 2.
