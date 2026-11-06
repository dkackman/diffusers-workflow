# Speed levers (Task 19)

Run 2026-11-05 on the bench server (lem, two RTX 3090s), deployed at
`a2034533`, then `6a89744a` (bench JSON only). Arms:
`bench/arms/selflift.json` and `bench/arms/tst.json`. Records:
`bench/records/selflift.json`, `tst.json`, and the moments pass
`t17-t19-moments.json` (PSNR/SSIM, raw texture, motion and audio, taken
in-process on lem).

**Short of Step 2's protocol.** Every arm ran 2 seeds, not 3. SelfLift ran at
1024x576 (544 does not halve to a 64 px grid), so the S1 6-segment floor
(measured at 960x544) is the nearest floor there is, not an exact one.

PSNR bands (workflow guide): about 44 dB is rerun noise, 23-27 dB is changed
numerics with the same shot, and about 19 dB is a re-staged shot. On the
moving S2 arc, even sparse against dense attention, a change that is
lossless by eye, reads 9-13 dB. So PSNR is read on S1 only.

## SelfLift (#1195)

### Verdict

- **Not lossless, and lift step 2 is broken.**
  - `lift_step 2` paints brush-stroke artifacts on every take, both scenes and
    both seeds. There are smeared stripes, ragged edges on the circle and the
    square, white feather streaks over the S2 trees, and streaks across the
    face and sweater from the first second. Its "texture win" on S1 is that
    damage. Frame-to-frame motion on the static shot doubles (0.47 to 1.20,
    and 0.64 to 1.15).
  - `lift_step 1` looks clean, but it re-stages the shot (a different head
    pose). S1 PSNR is 18.0 and 21.7 dB at frame 62, below the 23-27 dB band.
    Fine texture drift is worse past the floor on both seeds (0.58 to 0.70,
    and 0.63 to 0.73; floor 0.081).
- **The saving is below the 15-25% expected.**
  - Lift step 1 saves 11% of chain denoise and about 6% of the job's wall.
  - Lift step 2 saves 22% of denoise and about 14% of wall, but it is unusable.
  - On a single segment the saving is larger: lift step 1 saves 13-19% of
    denoise, and lift step 2 saves 33%. A chain's later segments carry
    full-canvas guide rows, so they gain less than segment 1.
- **As a dial step:** lift step 1 buys about 6% of wall for a re-staged shot
  and +0.1 fine drift. That is too small to be worth a step. The draft tier has
  bigger levers (steps, canvas). Recommendation: close #1195 with these
  numbers.

### Wall

Denoise per segment, in s, read from the job event log (`denoise.denoise` to
`decoding`). Steady state is segments 3-6, where no autotune stall remains.

| arm | card | seg 1 | seg 2 | seg 3-6 (each) | denoise total | job wall |
|---|---|---|---|---|---|---|
| ctl, seed 2 | cuda:0 | 53.5 | 69.8 | 69.7 | 402.8 | 11.50 min |
| lift 1, seed 2 | cuda:0 | 50.3 | 62.3 | 62.0 (**-11%**) | 360.8 | 10.84 min (**-6%**) |
| lift 2, seed 2 | cuda:1 | 39.4 | 50.7 | 50.3 | 291.1 | 9.03 min (cuda:1) |
| ctl, seed 1166 | cuda:0 | 176.9 | 211.0 | 69.7 | 666.5 | 15.90 min (about 11.5 without its stall) |
| lift 1, seed 1166 | cuda:0 | 50.3 | 61.8 | 61.9 | 359.8 | 10.76 min |
| lift 2, seed 1166 | cuda:0 | 105.0 | 117.0 | 54.5 (**-22%**) | 440.0 | 12.11 min (about 9.9 without its stall, **-14%**) |

- The first S1 job at the new canvas, and the first lift-2 job, each paid a
  first-call autotune stall of about 130-260 s in segments 1-2. Half-canvas
  shapes autotune too.
- On a single S2 segment, denoise was 63.1 and 58.4 s for the control, 50.9
  and 50.6 s for lift 1, and 41.7 and 39.4 s for lift 2. Job wall was
  1.69/1.58, 1.46/1.58 and 1.31/1.28 min.

### Quality

Verdicts are on the picture, primary `texture_fine_seg_last`, floor S1 guide
(the level step is not counted on S1/S2).

| arm | job | control | PSNR @62/277/583 | jump | look | tex fine | tex mid | tex coarse | motion | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 lift 1 @1166 | `874f0e4ac0e8` | `8b21a85f52e8` | 18.0 / 22.2 / 13.8 | 0.794→0.812 | 1.300→_1.754_ | 0.580→_0.702_ | 0.184→_0.490_ | 0.014→_0.346_ | 0.47→0.58 | loses |
| S1 lift 1 @2 | `96e48beb2dd3` | `09e78427dacf` | 21.7 / 17.4 / 18.4 | 0.898→0.794 | 1.938→**1.468** | 0.629→_0.734_ | 0.169→0.158 | 0.009→_0.020_ | 0.64→0.55 | loses |
| S1 lift 2 @1166 | `98dc7b8c3a12` | `8b21a85f52e8` | 16.7 / 16.8 / 14.9 | 0.794→_1.072_ | 1.300→1.734 | (0.580→0.078: artifacts) | 0.184→0.100 | 0.014→_0.268_ | 0.47→_1.20_ | broken by eye |
| S1 lift 2 @2 | `2a77f4abc476` | `09e78427dacf` | 17.3 / 14.7 / 14.2 | 0.898→_1.168_ | 1.938→2.168 | (0.629→0.086: artifacts) | 0.169→0.173 | 0.009→_0.124_ | 0.64→_1.15_ | broken by eye |
| S2 1 seg lift 1 @1166 / @2 | `b35afd39ed00` / `41d6746a43dd` | `4b79f1280670` / `6cbd7727f378` | (S2: not read) | | | inside floor | inside floor | inside floor | | tie |
| S2 1 seg lift 2 @1166 / @2 | `9925b2488c98` / `6edd44546d16` | `4b79f1280670` / `6cbd7727f378` | (S2: not read) | | | 1.012→1.194 / 0.541→_2.222_ | | | | broken by eye |

- **Audio.** S1/S2 tracks are room tone and ambience at -50 to -87 dB, so no
  timbre read is possible on these scenes. Nothing goes to `for-don.md`.
- **Segment-1 labels** do not apply. The canvas has no canonical entry, and
  the lift changes segment 1 by design.

## TST (#1196)

### Verdict

- **No quality effect, measured or visible, and it costs dense attention.**
  - tau 0.2 per head against dense attention: S1 PSNR is 31-35 dB, above the
    same-shot band and nearly a rerun. Every texture measure is inside the
    floor on both scenes and seeds.
  - By eye, S1 is indistinguishable from dense. S2 differs only in the clouds
    and far trees, as any numerics change on a moving shot does.
- **Cost.**
  - TST adds 5-6% denoise over dense.
  - Dense attention adds 22-28% over the sparse default.
  - So TST costs about +32% denoise over today's default, for nothing seen.
- **tau 0 is a no-op.** Its output is bit-identical to dense (same frames,
  same segment-1 fingerprint `e1959ec1/d4b8f719`), and it still costs 3% (the
  processor's overhead).
- `per_head: false` and other tau values were not in the arms.
- Recommendation: close #1196 with these numbers. Reopen only if a tau sweep is
  wanted, and that is a Don call: the spike found no gain at tau 0.2.

### Table

Denoise in s for one segment (the event log). Seed 1166 ran on cuda:1 and
seed 2 on cuda:0, so compare within a seed.

| scene, seed | sparse | dense | tst tau 0.2 | tau 0 | tst vs dense | dense vs sparse | PSNR tst vs dense @20/62/110 | PSNR dense vs sparse |
|---|---|---|---|---|---|---|---|---|
| S1 @1166 | 53.0 | 67.8 | 70.9 | 69.8 | +4.6% | +28% | 32.0 / 34.8 / 34.7 | 28.3 / 28.4 / 28.6 |
| S1 @2 | 49.4 | 60.5 | 64.1 | | +6.0% | +22% | 31.2 / 33.6 / 31.5 | 24.0 / 20.8 / 24.0 |
| S2 @1166 | 53.1 | 66.8 | 70.6 | | +5.7% | +26% | 23.3 / 19.1 / 18.5 | 13.2 / 11.5 / 11.9 |
| S2 @2 | 49.2 | 60.4 | 63.9 | | +5.8% | +23% | 23.4 / 20.6 / 18.4 | 11.4 / 9.6 / 9.3 |

Jobs (sparse, dense, tst, in that order):

- S1 @1166: `a32149365b7c`, `db14f86eb1bd`, `6dba38159cc7`; tau 0 is
  `20db977af359`.
- S1 @2: `dd38d2050097`, `1a172450268f`, `87b8dd26a67a`.
- S2 @1166: `7c3a90fc4c90`, `d0e2aad67720`, `80ed710df33f`.
- S2 @2: `6c644ea1d6c2`, `19e189043889`, `f7a673dae443`.

The two dense S1 jobs include a cold load of 121-135 s, which is why their job
wall reads 4.1-4.3 min.

Texture against dense is inside the floor on all four pairs: S1 fine
0.008→0.013 and 0.016→0.006; S2 fine 0.519→0.473 and 0.408→0.537. Motion is
unchanged (0.42→0.42, 0.50→0.51, 3.09→3.04, 4.11→4.23).

## Host-RAM gate (deployed `c441f92e..6495878e`)

### Verdict

**Confirmed: two H3 jobs ran at once, one per card, with no failure.** Each
H3 job commits 42-51 GiB. Two at once commit about 91 GiB of lem's 125.7 GiB,
and the sampled free host memory never fell below 31 GiB.

### Concurrent windows (UTC, 2026-11-05, from the job records)

| job | arm | card | start | finish | committed peak |
|---|---|---|---|---|---|
| `05a3a55b043c` | ks-pixel, 120 s | cuda:0 | 20:44:54 | 21:53:36 | 45.6 GiB |
| `4766787e793d` | ks-lc, 120 s | cuda:1 | 20:45:22 | 21:50:47 | 45.5 GiB |
| `d93adb33249e` | lc S2 @1166 | cuda:0 | 20:32:07 | 20:44:54 | 43.4 GiB |
| `d901c8c82784` | lc S2 @2 | cuda:1 | 20:32:07 | 20:45:22 | 43.3 GiB |
| `2c1fd3ab7869` | anchor 0.25 rerun, 120 s | cuda:1 | 21:50:47 | 23:00:43 | 46.0 GiB |
| `536548e14dd5`, `1b8edfdba405`, then SelfLift | S2 controls | cuda:0 | 21:53:36 | ... | 42.8, 41.7 GiB |

- The two 120 s takes overlapped for 65 min. From 20:32 to 00:51 both cards
  were busy, with 40 jobs on two cards. Every job succeeded.
- **Committed figure:** `host_memory_job_peak_committed_mb` in the bench
  server's job history (`~/.diffusers_helper/jobs.sqlite`, read-only). Other
  H3 jobs: `20ce09d60040` 50.5 GiB, `40bbb40c14ec` 50.7 GiB, `fc0f91663277`
  47.0 GiB, `bf9ee32e040f` 47.3 GiB.
- The old RSS basis put the same jobs at 49-120 GiB (`fc0f91663277`: 120.3
  GiB job-peak RSS). That is the mmapped checkpoint page cache, and it is what
  serialised the queue before.
- The lowest `host_memory_available_mb` in the memory events during the
  overlap is 31.2-31.9 GiB. Those events are sampled at phase changes, so the
  true peaks may be a little higher.
- **Card speed, not contention:** cuda:1 runs about 10% slower per pass than
  cuda:0 on every job, alone or paired (17.4 s against 15.7 s for a 120 s
  pass). Round 2's `8d1614ced27a`, on cuda:1 a day earlier, reads 17.4 s too.
  So the overlap costs nothing measurable. Compare wall times within one card.
