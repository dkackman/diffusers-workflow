# Noise floor and cheap-setting checks

Task 4 of the long-form media plan. Every scene's baseline arm was run twice on
lem (bench server, both cards) at seed 1166; the second run carries a distinct
inline workflow id (`-noise-b`) so the step cache cannot serve it. The floor
for a measure is the absolute difference of the two runs (`bench/verdict.py`
`noise_floor`). Records: `bench/records/noise-floor.json` (12 runs),
`bench/records/cheap-checks.json` (6 runs). Every measure is lower-is-better;
`texture_*_seg_last` is `|ln(last/first)|` of the crop's band energy.

## Reading the floor

- **The engine is deterministic for a seeded run.** S1, S2, S4, S5 and S6 give
  bit-identical picture measures across the two runs, so their floor is exactly
  0 on every seam and texture measure (S5 and S6 differ by 0.005-0.02 dB in the
  seam level step, an audio-encode wobble). Only S3, whose picture follows a
  spoken line per segment, moves between runs.
- **A floor of 0 means the verdict has no tie zone.** `verdict()` calls any
  difference past 0 a win or a loss. That is right for "did this arm change the
  result", and says nothing about how much another seed would move the same
  measure; the floor measures reproducibility of a run, not seed-to-seed
  variance. A bench arm that must hold across seeds needs a second seed, which
  this task did not run.
- **S3 is the only non-zero floor**: `texture_coarse_seg_last` 0.23 (0.21 vs
  0.44) and `seam_jump_mean` 0.27 are the large ones; use them for S3 verdicts.
- The two S4 runs ran side by side on the two cards and agree exactly.

## S4 rerun

The first S4 pair (jobs `0d08eccff9dc`, `14a8b42f73f0`) failed at step 11 of 15,
the `soundtrack` `slice_audio` step, with "slice_audio needs 'sample_rate' with
a raw waveform". Cause: the bench workflow (not the engine). `music_bed`'s
result hands over a raw Music 3 waveform that carries no rate of its own (the
result's `sample_rate: 44100` only labels the file at save time, and
`save: false` here), and `slice_audio` was given no `sample_rate`. The
`templates/minimax/music.json` and `music-video.json` templates pass
`sample_rate` beside a raw waveform for the same reason. Fix: `"sample_rate":
"variable:sample_rate"` on the `soundtrack` step (commit e716fd2c). The pair was
rerun with fresh ids (`-noise-a2`, `-noise-b2`) so the first pair's cached shots
did not serve it. Both succeeded in about 20 minutes each, not the 70 the
launch estimated.

## Crop check (ruling R10)

Frames were read through `GET /api/gallery/{name}/frames` at each scene's
moments, on a job of that scene, with the crop drawn over them. A crop must
stay clear of every subject at each moment, keep a 128 px short side (the
coarse band's longest wavelength; a smaller crop returns no coarse value), and
stay off the frame's border.

| scene | crop before | decision | why |
|---|---|---|---|
| S1, S2 | `[0,0,240,150]` | unchanged (already verified under #1175) | |
| S3 | `[0,0,240,150]` | unchanged, now verified | shop fronts, top left; the walker stays in the middle third in all three moments and a 12-frame contact sheet |
| S4 | `[0,0,240,150]` | `[500,16,160,130]` | the seated man's vest and the standing man's shoulder reach the old crop's lower right corner at frame 62; the new crop is wall in all three shots (a sconce at frame 62, brick at 186 and 434) |
| S5 | `[0,0,240,150]` | unchanged, now verified | sky above the horizon; the cyclist never leaves the lower half over a 12-frame contact sheet. Plain sky holds little fine texture, so S5's fine-band drift is a weak signal |
| S6 | `[0,0,240,150]` | `[690,340,190,135]` | the old crop held the page border and, at frame 240, the lighthouse finial. A smaller interior crop was tried and returned no coarse band; the new one is lower right, clear of the lighthouse and the climber in all three frames (at frame 240 the beam's faint haze touches its top edge, which is lighting, not subject) |

All six scenes are `crop_verified: true`. S3, S4, S5 and S6 were measured again
with `driver.py measure` (S4's runs were measured with the new crop once
they existed); S1 and S2 crops did not move, so their records stand.

## Tables

One table per scene: both runs, the floor (`noise_floor(a, b)`), job ids,
output names, and wall minutes per output minute. S1/S2 are 10 segments
(1042 frames), S3 6 segments (634), S4 four shots under one bed (496),
S5 LTX (359), S6 narration (288). Wall time includes model load.

### S1 static (`s1-static`)

| measure | run a | run b | floor |
|---|---|---|---|
| seam_jump_mean | 0.9311 | 0.9311 | 0.0000 |
| seam_jump_max | 1.0900 | 1.0900 | 0.0000 |
| seam_look_mean | 1.5289 | 1.5289 | 0.0000 |
| seam_look_max | 2.6100 | 2.6100 | 0.0000 |
| seam_level_step_db_mean | 9.8567 | 9.8567 | 0.0000 |
| seam_level_step_db_max | 36.5600 | 36.5600 | 0.0000 |
| texture_fine_seg_last | 1.1179 | 1.1179 | 0.0000 |
| texture_mid_seg_last | 0.4015 | 0.4015 | 0.0000 |
| texture_coarse_seg_last | 0.0488 | 0.0488 | 0.0000 |

| | run a | run b |
|---|---|---|
| job | `00e4000a3ffb` | `b4e31bb17593` |
| output | `BenchS1Static/20261104-000916-390b57bc/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4` | `BenchS1Static-noise-b/20261104-004113-f663789a/final/BenchS1Static-noise-b-chained_keyframe_to_video_audio.0-0.0.mp4` |
| wall min | 21.1 | 18.1 |
| output min | 0.72 | 0.72 |
| wall min per output min | 29.1 | 25.0 |

### S2 arc (`s2-arc`)

| measure | run a | run b | floor |
|---|---|---|---|
| seam_jump_mean | 1.1944 | 1.1944 | 0.0000 |
| seam_jump_max | 2.0900 | 2.0900 | 0.0000 |
| seam_look_mean | 1.2844 | 1.2844 | 0.0000 |
| seam_look_max | 2.7100 | 2.7100 | 0.0000 |
| seam_level_step_db_mean | 2.0356 | 2.0356 | 0.0000 |
| seam_level_step_db_max | 5.5400 | 5.5400 | 0.0000 |
| texture_fine_seg_last | 3.5576 | 3.5576 | 0.0000 |
| texture_mid_seg_last | 2.8636 | 2.8636 | 0.0000 |
| texture_coarse_seg_last | 1.9747 | 1.9747 | 0.0000 |

| | run a | run b |
|---|---|---|
| job | `04978dc6f338` | `51d7b09236f6` |
| output | `BenchS2Arc/20261104-010950-2f5ecfe0/final/BenchS2Arc-chained_keyframe_to_video_audio.0-0.0.mp4` | `BenchS2Arc-noise-b/20261104-012953-430a6737/final/BenchS2Arc-noise-b-chained_keyframe_to_video_audio.0-0.0.mp4` |
| wall min | 20.0 | 17.8 |
| output min | 0.72 | 0.72 |
| wall min per output min | 27.7 | 24.7 |

### S3 dialogue walk (`s3-dialogue-walk`)

| measure | run a | run b | floor |
|---|---|---|---|
| seam_jump_mean | 1.4780 | 1.2120 | 0.2660 |
| seam_jump_max | 1.8300 | 1.6000 | 0.2300 |
| seam_look_mean | 1.3780 | 1.3240 | 0.0540 |
| seam_look_max | 1.6100 | 1.8700 | 0.2600 |
| seam_level_step_db_mean | 2.7700 | 3.0000 | 0.2300 |
| seam_level_step_db_max | 8.8100 | 8.6500 | 0.1600 |
| texture_fine_seg_last | 0.1751 | 0.1782 | 0.0031 |
| texture_mid_seg_last | 0.2261 | 0.2076 | 0.0185 |
| texture_coarse_seg_last | 0.2075 | 0.4396 | 0.2320 |

| | run a | run b |
|---|---|---|
| job | `3c76d1966485` | `d2095670cd89` |
| output | `BenchS3DialogueWalk/20261104-015817-5942f052/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4` | `BenchS3DialogueWalk-noise-b/20261104-021940-da114475/final/BenchS3DialogueWalk-noise-b-chained_keyframe_to_video_audio.0-0.0.mp4` |
| wall min | 10.8 | 10.5 |
| output min | 0.44 | 0.44 |
| wall min per output min | 24.5 | 24.0 |

### S4 montage (`s4-montage`)

| measure | run a | run b | floor |
|---|---|---|---|
| seam_jump_mean | 20.4767 | 20.4767 | 0.0000 |
| seam_jump_max | 22.6900 | 22.6900 | 0.0000 |
| seam_look_mean | 13.4733 | 13.4733 | 0.0000 |
| seam_look_max | 18.2400 | 18.2400 | 0.0000 |
| seam_level_step_db_mean | 3.4967 | 3.4967 | 0.0000 |
| seam_level_step_db_max | 4.4200 | 4.4200 | 0.0000 |
| texture_fine_seg_last | 1.7141 | 1.7141 | 0.0000 |
| texture_mid_seg_last | 0.5798 | 0.5798 | 0.0000 |
| texture_coarse_seg_last | 1.2143 | 1.2143 | 0.0000 |

| | run a | run b |
|---|---|---|
| job | `568a3da301c4` | `511b37c9bd70` |
| output | `BenchS4Montage-noise-a2/20261104-024801-94372b51/final/BenchS4Montage-noise-a2-film.14-0.0.mp4` | `BenchS4Montage-noise-b2/20261104-024801-6c692da9/final/BenchS4Montage-noise-b2-film.14-0.0.mp4` |
| wall min | 19.7 | 20.8 |
| output min | 0.34 | 0.34 |
| wall min per output min | 57.2 | 60.3 |

### S5 LTX action (`s5-ltx-action`)

| measure | run a | run b | floor |
|---|---|---|---|
| seam_jump_mean | 1.2450 | 1.2450 | 0.0000 |
| seam_jump_max | 1.3600 | 1.3600 | 0.0000 |
| seam_look_mean | 2.0850 | 2.0850 | 0.0000 |
| seam_look_max | 2.4300 | 2.4300 | 0.0000 |
| seam_level_step_db_mean | 2.1450 | 2.1400 | 0.0050 |
| seam_level_step_db_max | 2.8500 | 2.8400 | 0.0100 |
| texture_fine_seg_last | 0.5457 | 0.5457 | 0.0000 |
| texture_mid_seg_last | 0.4867 | 0.4867 | 0.0000 |
| texture_coarse_seg_last | 1.5417 | 1.5417 | 0.0000 |

| | run a | run b |
|---|---|---|
| job | `a91df8d3684c` | `813a4ec70859` |
| output | `BenchS5LtxAction/20261104-005131-8fc124d3/final/BenchS5LtxAction-chained_image_to_video.0-0.0.mp4` | `BenchS5LtxAction-noise-b/20261104-005443-a7939f5d/final/BenchS5LtxAction-noise-b-chained_image_to_video.0-0.0.mp4` |
| wall min | 3.2 | 2.0 |
| output min | 0.25 | 0.25 |
| wall min per output min | 12.8 | 7.9 |

### S6 narration (`s6-narration`)

| measure | run a | run b | floor |
|---|---|---|---|
| seam_jump_mean | 40.2000 | 40.2000 | 0.0000 |
| seam_jump_max | 45.3300 | 45.3300 | 0.0000 |
| seam_look_mean | 70.6350 | 70.6350 | 0.0000 |
| seam_look_max | 82.7500 | 82.7500 | 0.0000 |
| seam_level_step_db_mean | 7.7900 | 7.8000 | 0.0100 |
| seam_level_step_db_max | 13.2600 | 13.2800 | 0.0200 |
| texture_fine_seg_last | 1.4011 | 1.4011 | 0.0000 |
| texture_mid_seg_last | 1.2213 | 1.2213 | 0.0000 |
| texture_coarse_seg_last | 0.8531 | 0.8531 | 0.0000 |

| | run a | run b |
|---|---|---|
| job | `54240901f86a` | `5670fac7bc34` |
| output | `BenchS6Narration/20261104-005645-a4669737/final/BenchS6Narration-film.9-0.0.mp4` | `BenchS6Narration-noise-b/20261104-005815-12f0dc21/final/BenchS6Narration-noise-b-film.9-0.0.mp4` |
| wall min | 1.6 | 1.3 |
| output min | 0.20 | 0.20 |
| wall min per output min | 7.9 | 6.6 |

Wall minutes per output minute, across the scenes: S1 25-29, S2 25-28, S3 24,
S4 57-60 (four H3 shots plus a music bed), S5 8-13, S6 7-8. The first run of a
pair in S5 and S6 paid model load; the second did not.

## Cheap-setting check (6 segments, #1175 ordering)

Arms A (control), B (`guide_noise` 0.4) and C (`chain.anchor: "segment_1"`) at
6 segments, moments frame 62 / 277 / 583, against #1175's 10-segment ordering
(`docs/proposals/complete/h3-guide-chain-drift-complete.md`, Stage 3). Jobs in
`bench/records/cheap-checks.json`.

| scene | measure | A | B | C | #1175 (10 segments) |
|---|---|---|---|---|---|
| S1 static | `texture_fine_seg_last` | 0.503 | 0.854 | 0.190 | A 1.00, B 1.47 (rougher), C 0.40 |
| S1 static | `seam_look_mean` | 1.622 | 1.392 | 1.404 | A 1.67, B 1.11, C 1.27 |
| S2 arc | `seam_jump_mean` | 1.362 | 1.166 | 1.854 | A 1.13, B 1.09, C 2.20 |
| S2 arc | `seam_look_mean` | 1.286 | 1.694 | 4.270 | A 0.98, B 1.07, C 3.62 |

`verdict(A, arm, floor, primary)` with the S1 and S2 floors (all 0, so any
difference counts):

| scene | arm | primary | verdict | reading |
|---|---|---|---|---|
| S1 | B | `texture_fine_seg_last` | loses | B is rougher, as in #1175 |
| S1 | B | `seam_look_mean` | mixed | wins the look, loses texture |
| S1 | C | `texture_fine_seg_last` | mixed | C beats A on texture, worse elsewhere (seam jump 1.16 vs 0.88) |
| S1 | C | `seam_look_mean` | mixed | |
| S2 | B | `seam_jump_mean` | mixed | best jump of the three, look slightly worse than A |
| S2 | B | `seam_look_mean` | loses | 1.69 vs 1.29, as B 1.07 vs A 0.98 was slightly worse at 10 segments |
| S2 | C | `seam_jump_mean` | loses | worst |
| S2 | C | `seam_look_mean` | loses | worst by a wide margin |

**Verdict: 6 segments reproduce #1175's ordering on every row**: on the static
scene C beats A on texture and B is rougher than both, and B and C both
improve on A's look; on the arc, C is worst on both seam measures and B has the
best jump, with a look slightly worse than A. The bench may use 6 segments.
The values are not the 10-segment values (the texture drift is smaller at 6
segments, as it has fewer segments to compound over), so a 6-segment bench
result is a ranking, not a number to compare with #1175's tables. The 960x544
size and the draft tier were not separately checked; the 6-segment arms ran at
the bench workflows' 960x544 and draft settings, and the ordering held there.

## Chain check thresholds (ruling R19)

`dw/pipeline_processors/chain_check.py` measures each segment with
`check_segment`: the seam's `jump_ratio` and `look_ratio` against the segment
before, and the whole-frame fine-band luma against segment 1's. The 28 seams of
the unchecked S1, S2 and S3 noise-run chains (S1 and S2 run a, S3 runs a and b;
S1 and S2 run b are bit-identical to run a) were measured with that function on
the decoded output, each segment's kept frames against the one before:

| measure | P50 | P99 | max | set |
|---|---|---|---|---|
| `jump_ratio` | 1.10 | 2.02 | 2.08 | `JUMP_RATIO_MAX` 25 -> 4.0 |
| `look_ratio` | 1.28 | 2.67 | 2.71 | `LOOK_RATIO_MAX` 12 -> 5.5 |
| `texture_ratio` (either way) | | 2.99 | 3.13 | kept 2.0, provisional |

The margin is about a factor of 2 on the P99 (jump 2.02 -> 4.0; look 2.67 -> 5.5, rounded up to half a point). 4.0 stays
below an intended cut's 9.9-20.3 and above a dissolve's 1.8; 5.5 stays well
under the 19.2 a saturation-1.6 grade reads.

The texture ceiling is not set from these runs, for two reasons. The two
same-seed runs of S1 and S2 are identical, so they show no spread to take a
floor from. And the texture ratio of an unchecked chain is the drift the check
exists to catch (1.07 rising to 2.43 over S1's ten segments, 1.04 falling to
0.32 over S2's), so its percentile measures the defect and a ceiling set at it
would pass the defect. It needs runs of a checked chain, or a repeat over
several seeds, to set.

The three scenes are one subject type each and one seed; the P99 of 28 values
is the maximum, in effect, so a faster-moving or darker scene may read above
these ceilings and be re-rolled on a good segment. `check` is opt-in per chain,
and a workflow can name its own ceilings.
