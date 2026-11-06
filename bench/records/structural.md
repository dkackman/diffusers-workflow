# Structural round (Task 17)

## Step 4: windowed refine

Task 17 Step 4 asks one question. Does a low-sigma refine of a finished chain,
run in overlapping time windows and fused at every step, smooth the seams and
drift? If it does, is it the seam-smoothing pass for every chain?

### Setup

- **Workflow.** `bench/workflows/refine-windows.json` has four steps:
  - `encode_h3_latents` reads the take one 17-frame clip at a time.
  - An H3 t2va `refine_strength` pass runs with `experimental_refine_windows`
    and `hold_audio` set to the take's own mp4.
  - `decode_h3_latents` trims the take back to its own length.
  - `pair_audio` puts the take's own track back.
- **Settings.** The refine runs 5 sigma points (4 evaluations) at shift 6, with
  the fl2v turbo 4-step LoRA, int4 SDNQ and sparse attention: the settings the
  chains were made with. The encode and decode steps set `release_models`
  (see *Failures* below).
- **Windows.** `h3_windows` has no default window, so the windows are the
  spike's protocol value: `{frames: 40, overlap: 12}` latents (about 6.6 s per
  window, 2 s overlap), pyramid blend, `positions: "true"`. A 26 s take is 187
  latents in 7 windows. One `positions: "rebased"` arm used `{frames: 42,
  overlap: 12}`. Rebasing needs a stride on the 5-latent clip grid, and
  `frames ≡ 2 (mod 5)` puts the pulled-back last window on it for any 5m + 2
  take.
- **Sources.** All sources are clean by `bench/segment1.py`, seed 1166, 6
  segments, 634 frames (26.4 s):

  | scene | source | job |
  |---|---|---|
  | S1 static | guide + anchor 0.25 (the static default) | `0a40c41af97c` |
  | S2 arc | guide control (no counters, per *Defaults by shot type*) | `dedd341c9108` |
  | S3 dialogue walk | guide control (no counters) | `c0b92d0d0cd4` |

- **Measures** (`bench/refine_pairs.py`, in-process on lem). A refined take is
  decoded fresh, so its run records no shots. The script reads both files and
  measures the refined take at the source's own seams:
  - `analyze_seams` jump, look and level step;
  - texture drift on the scene crop against frame 62, at frames 62, 150, 277,
    400, 583 and 630 ("seg_last" is 630);
  - thumbnail frame-delta motion;
  - PSNR and SSIM of the refined take against its source.
- **Records.**
  - Driver records: `t17-refine-r030.json` and `t17-refine-r020-r040.json`.
  - Pair measures: `t17-pairs-r030.json` and `t17-pairs-r020-r040.json`.
- **Floor.** `combined_floor(seed floor, rerun gap)` uses each scene's guide
  row from `noise-floor.md`. The refine runs on a fixed source, so this floor
  is conservative.

### Results

Each cell gives the source value, then the refined value. **Bold** is better
past the floor. Nothing was worse past the floor on any arm.

| arm | verdict | jump | jump max | look | look max | tex fine | tex mid | motion | PSNR @630 |
|---|---|---|---|---|---|---|---|---|---|
| S1 0.2 | tex tie / jump **wins** / look tie | 0.928→**0.566** | 1.15→0.72 | 1.128→1.114 | 1.98→1.41 | 0.346→0.351 | 0.116→0.107 | −17% | 29.6 |
| S1 0.3 | tex tie / jump **wins** / look tie | 0.928→**0.586** | 1.15→0.87 | 1.128→1.238 | 1.98→1.64 | 0.346→0.366 | 0.116→0.115 | −18% | 28.2 |
| S1 0.3 rebased | tex tie / jump **wins** / look tie | 0.928→**0.516** | 1.15→**0.71** | 1.128→1.014 | 1.98→1.21 | 0.346→0.381 | 0.116→0.115 | −19% | 28.2 |
| S1 0.4 | tex tie / jump **wins** / look tie | 0.928→**0.602** | 1.15→1.00 | 1.128→1.372 | 1.98→1.95 | 0.346→0.368 | 0.116→0.127 | −17% | 26.8 |
| S2 0.2 | look **wins** / jump **wins** | 1.362→**1.128** | 2.09→1.67 | 1.286→**0.808** | 1.74→1.19 | 4.43→4.14 | 4.40→4.03 | +1% | 30.1 |
| S2 0.3 | look **wins** / jump **wins** | 1.362→**1.114** | 2.09→**1.64** | 1.286→**0.766** | 1.74→1.14 | 4.43→4.26 | 4.40→4.19 | +3% | 27.8 |
| S2 0.4 | look **wins** / jump **wins** | 1.362→**1.102** | 2.09→**1.65** | 1.286→**0.720** | 1.74→1.12 | 4.43→4.10 | 4.40→4.02 | +3% | 26.7 |
| S3 0.2 | jump **wins** / look tie / tex tie | 1.524→**1.192** | 1.83→1.45 | 1.470→1.242 | 1.71→1.57 | 0.031→0.145 | 0.511→0.490 | 0% | 29.6 |
| S3 0.3 | jump **wins** / look tie / tex tie | 1.524→**1.100** | 1.83→**1.36** | 1.470→1.324 | 1.71→1.62 | 0.031→0.129 | 0.511→0.495 | +2% | 28.0 |
| S3 0.4 | jump **wins** / look tie / tex tie | 1.524→**1.000** | 1.83→**1.20** | 1.470→1.342 | 1.71→1.70 | 0.031→0.109 | 0.511→0.491 | +4% | 26.8 |

- **Level step.** The level step does not change (S1 12.2→12.6 dB, S2 and S3
  identical). The track is held, and the change of about 0.4 dB is the
  re-encode.
- **Drift against segment 1.** Every texture series is inside the floor at
  every moment. On S1 the fine drift is 0.367→0.352 at 583 and 0.346→0.366 at
  630, and the face crop is likewise unchanged. By eye at frame 583, the 0.3
  refine keeps the source's crusted stripes, circle and wall. **The refine
  smooths the seams. It does not undo the burn.**
- **Motion.**
  - S2 and S3 keep their motion (0 to +4%). The S2 arc completes, with the
    same framing at 62, 300 and 583.
  - S1's frame-to-frame delta falls by 17-19% at every strength. S1 is locked
    off, so this delta is breathing, blinks and the burn's crawl. That the drop
    does not depend on strength suggests the refine removes shimmer rather
    than motion. This has not been confirmed by eye in motion: it is in the
    S1 export.
- **Positions.** `rebased` reads a little better than `true` on S1 (jump max
  and look past or near the floor). Both are within noise of each other. For
  a take past 15 s, `rebased` keeps every window's media inside H3's trained
  span, so it is the setting for the long run.
- **Strength.** 0.2-0.4 all give the same seam verdicts. The seam gains grow
  slightly with strength (S3 jump 1.19→1.10→1.00), and fidelity falls
  (PSNR ~29.6→28.0→26.8). On S1, 0.4 starts to give the look back (1.37). **0.3
  is the pick**, with 0.2 as the conservative setting.

### Cost and VRAM (lem, RTX 3090, cuda:0)

- **Wall time.** A 26.4 s take takes 17-20 min:
  - load: 110-115 s;
  - encode: about 14 s;
  - denoise: 330 s, which is 4 passes of 83 s over 7 windows, plus a first-pass
    sparse/autotune stall of up to 115 s on the first job after a load;
  - decode: 2 × about 97 s, the step's own decode and `decode_h3_latents`.
  That is **about 40 min per minute of take** at this length.
- **Per window.** The denoise is about 2.9 s per window per pass.
- **At 120 s, projected.** 852 latents is 28 windows (rebased 42/12). That
  projects to about 22 min of denoise, about 15 min of decode and the 2 min
  load: **about 20 min per minute of take** once the fixed costs spread.
- **Peak VRAM.** nvidia-smi, sampled at 1 s, reads 22.9-23.5 GB on a 24 GB
  card for every arm. Most of that is the allocator's cache around the two
  float32 decodes. The windowed denoise itself is small (one 40-latent window
  at a time).

### 120 s (S1, `r3-anchor25-s1-120s`)

Run 2026-11-05 on the bench server at `155ae56c` (streamed decode deployed):
`t17-s1-120s-r030-rebased` (`2f6c0073d2f0`, cuda:0), 0.3, `{42, 12}`
rebased, over `r3-anchor25-s1-120s` (`25f6bd1da507`, clean). Arms
`bench/arms/t17-refine-120s.json`; records `t17-refine-120s.json` and
`t17-pairs-120s.json` (`refine_pairs.py` at the round-3 moments 62, 300,
583, 1000, 1501, 2000, 2500, 2868). The refined take is a fresh decode, so it
has no segment-1 label of its own; its source is clean. Floor: the S1 guide
row, as above (no two-minute floor exists).

| measure | source | refined | verdict |
|---|---|---|---|
| seam jump mean / max | 0.943 / 1.86 | **0.734** / 1.93 | mean **wins** (0.21 > 0.124); max tie |
| seam look mean / max | 1.032 / 2.70 | 1.156 / 2.96 | tie (inside 0.438 / 0.790) |
| level dB mean / max | 5.83 / 24.3 | 5.96 / 24.9 | tie (held track, re-encode) |
| tex fine / mid / coarse @2868 | 1.108 / 0.430 / 0.019 | 1.093 / 0.406 / 0.004 | tie |
| face fine @2868 | 0.040 | 0.086 | tie |
| motion | 0.636 | 0.528 | -17% |
| PSNR vs source | | 31.1 @62 -> 26.0 @1000 -> 23.5-24.2 from 2000 | |

- **Does it move the burn from about 42 s on? No.** The fine drift series
  against frame 62 runs 0.497 / 0.771 / 0.931 / 1.012 / 1.108 (source) and
  0.539 / 0.838 / 0.949 / 1.003 / 1.093 (refined) at 42 / 62 / 83 / 104 /
  119.5 s: inside the floor at every moment, mid likewise. By eye (stills at
  65, 70, 75 and 110 s, and the export) the two sides are the same picture:
  the same mosaic crust on the stripes and circle, the same pale, flattened
  face and white forehead patch at 110 s. The PSNR falls from 31 to about
  24 dB as the burn grows, so the refine re-renders the burned crust more
  freely, but it re-renders it as crust.
- **Do the seams win as at 26 s? Partly.** Jump mean wins past the floor
  over the whole take (0.94 -> 0.73), and by stretch it holds through 95 s
  (0-25 s 0.93 -> 0.53, 25-60 s 0.88 -> 0.55, 60-95 s 0.80 -> 0.56, max
  0.96 -> 0.75 there). In the last 25 s, where the source is most burned, it
  no longer helps: jump 1.26 -> 1.41 and look 1.81 -> 2.10 mean, 2.70 -> 2.96
  max, both inside the floor. Look ties overall, where at 26 s it tied too
  (S1). Nothing is worse past the floor.
- **Motion** falls 17%, as at 26 s (-17 to -19%), so the drop does not grow
  with length: it is the same shimmer removal, still owed an eye check.
- **Cost.** 71.5 min for 2 min of take, **about 36 min per minute**, not
  the 20 projected: encode 10.7 min; refine 33.3 min (load 2.0, encode 0.7,
  denoise 23.3 = 4 passes of 5.8 min over 28 windows, as projected, and its
  own decode 7.3); `decode_h3_latents` 27.1 min; mux 0.3. The two decodes
  are 34 min, half the job: dropping the step's redundant internal decode
  saves 7, and the float32 streamed decode is the next cost to look at.
- **Peak VRAM** (nvidia-smi at 1 s, cuda:0, `~/t17-120s-vram.csv` on lem):
  **16,519 MiB (16.1 GiB)**, during the refine's load; denoise 12.1 GB,
  decode 14.5 GB. The streamed decode takes the 120 s job well under the
  22.9-23.5 GB the 26 s jobs read with the whole-take decode.
- **Export** for Don: `~/Desktop/dw-bench-t17/t17-s1-120s-anchor25_LEFT-source_RIGHT-refine030-rebased-60-80s.mp4`
  (60-80 s, the burn region, x264 CRF 16, refined track).

**Verdict at 120 s:** the windowed refine is a seam pass at length too (jump
wins to 95 s), it does not touch the burn, and it costs about 36 min per
minute of take, half of it decode.

### Failures on the way, and their causes

- The first 4 arms failed with "Adapter name(s) {'turbo'} not in the list of
  present adapters". This hides the real cause. The worker log shows a **CUDA
  OOM while loading the transformer or VAE**, which diffusers'
  `load_components` swallows: it logs the OOM and goes on without the
  component. The LoRA then found no transformer.
- The OOM came from the `encode_h3_latents` float32 VAE. The task model cache
  kept it on cuda:0 into the refine's load.
- `release_models: true` on the encode and decode steps fixed it.
- An engine follow-up is worth filing: fail the load when `load_components`
  dropped a component the step needs, so the error names the OOM.

### Verdict

| scene | strength | wins? |
|---|---|---|
| S1 static | 0.2 / 0.3 / 0.4 | **Seams yes** (jump past the floor), drift no (tie), nothing worse. The burn stays. |
| S2 arc | 0.2 / 0.3 / 0.4 | **Yes**: look and jump both win past the floor, motion kept, nothing worse. |
| S3 dialogue walk | 0.2 / 0.3 / 0.4 | **Seams yes** (jump past the floor), look and texture tie, motion and the held voice kept. |

**It is the seam-smoothing pass for every chain shot type measured:**

- the default is `refine_strength` 0.3, with `{frames: 40-42, overlap: 12}`
  windows and `rebased` positions past 15 s;
- it never lost a measure, and it costs about 40 min per minute of take at
  26 s and about 36 min per minute at 120 s (half of it decode);
- **it is not a burn fix**: drift against segment 1 is unchanged on every
  scene and strength.

Promoting it out of the experimental flag needs four things, as Step 4's "if
it wins" says:

- the 120 s confirmation (done: seams yes to 95 s, burn untouched; see *120 s*);
- an eye check of S1's motion drop;
- dropping the step's redundant internal decode when `videos` is not in the
  output;
- the TDD feature stages.

### Exports for Don

The exports are in `~/Desktop/dw-bench-t17/` on the Mac: 20 s side by side,
4-24 s, x264 CRF 16, source on the left and refined on the right, with the
refined (held) track.

- `t17-s2-guide_LEFT-source_RIGHT-refine030-20s.mp4`: S2 at 0.3.
- `t17-s1-6seg-anchor25_LEFT-source_RIGHT-refine030-rebased-20s.mp4`: S1 at
  6 segments, 0.3 rebased.
- `t17-s1-120s-anchor25_LEFT-source_RIGHT-refine030-rebased-60-80s.mp4`: S1
  at 120 s, 0.3 rebased, 60-80 s (the burn region).

## Steps 2-3: latent carry (#1150, #1044, #1030)

### Verdict

- **Latent carry is not a default, and not a burn fix.** On the static scene it
  lowers the backdrop crust, but the take still burns. On the moving scene it
  makes the seams worse on both seeds.
- **The softness is the anchor's, not the carry's.** Without the anchor, the
  latent-carry take is crusty and harsh, like the pixel control. It is not soft.
  Don's "soft and masky" on `fc0f91663277` comes from anchor 0.25. The pixel
  anchor-0.25 take is just as soft and masky at the same moments.
- **The carry alone (keystone pair, R39) ties on every picture measure.** By
  eye its face holds better at 104-120 s. That needs Don's eye (export below).
- **The carry drops the late music bed on seed 1166, and only there.** It keeps
  S2's garden ambience on both seeds. The bed is a property of the seed-1166
  pixel chain, not of the carry. Seed 2's pixel take never grows one.
- **New finding: keystone + anchor 0.25 resets the burn at two minutes.** Both
  keystone takes crust from about 40 s. Then the backdrop snaps back clean:
  pixel between 79 and 81 s, latent carry by about 104 s. Fine drift at 120 s
  is 0.03-0.06, against anchor 0.25's 1.11. This is one seed. Round 2's
  keystone verdict ("does not hold") was read at 24 s, before any reset.
- **Cost.** The carry skips the pixel re-encode. A two-minute take encodes in
  4.7-7.7 min less (465 s against 757 s, 281 s against 776 s with the
  keystone), so it is about 5% faster overall (65.8 against 69.1 min).

### What the A/B removes

As R39 says, the keystone pair is #1044's latent-carry arm, not #1150's
VRAM-saving arm 2. Against the pixel guide it removes the VAE encode of the
guide frames, the posterior sampling, and the fp16 rounding of the round trip.
On fl2va every later segment's keyframe image is still a pixel round trip
(`carry.frames[0]`). The keystone pair pins that keyframe to the same fixed
pixels in both arms, so the carry is the only difference.

### Floors

These are the first two-minute floors. Both pairs are anchor 0.25, guide, S1,
120 s:

- seed floor: `25f6bd1da507` (seed 1166) against `bf9ee32e040f` (seed 2), both
  clean;
- rerun gap: `25f6bd1da507` against its same-seed rerun `2c1fd3ab7869`.

| floor | jump | jump max | look | look max | tex fine | tex mid | tex coarse | face fine |
|---|---|---|---|---|---|---|---|---|
| 120 s seed floor | 0.044 | 0.430 | 0.263 | 0.040 | 0.034 | 0.159 | 0.002 | 0.427 |
| 120 s rerun gap | 0.044 | 0.350 | 0.162 | 0.640 | 0.075 | 0.066 | 0.118 | 0.638 |
| S1 guide (6 seg) | 0.124 | 0.430 | 0.438 | 0.790 | 0.081 | 0.032 | 0.010 | |
| **combined (used)** | **0.124** | **0.430** | **0.438** | **0.790** | **0.081** | **0.159** | **0.118** | **0.638** |

- The 120 s floors fit inside the 6-segment row for seams and fine texture.
- Mid and coarse texture need the wider 120 s values.
- The face crop is too noisy to read at two minutes. The rerun alone moves it by
  0.64 (`2c1fd3ab7869`'s face blurs out, while `25f6bd1da507`'s does not), so
  the face is read by eye.
- S2 uses the S2 guide row (fine 0.605, mid 1.21, coarse 1.66), widened by
  the r36 reruns (`536548e14dd5` against `dedd341c9108`, which is bit-identical
  in motion, and `1b8edfdba405` against `01146e925c94`).
- The level step is shown but not counted on S1/S2 (Don's ruling).

### Segment-1 labels (`bench/segment1.py`)

Clean: `fc0f91663277`, `05a3a55b043c`, `4766787e793d`, `20ce09d60040`,
`bf9ee32e040f`, `d93adb33249e`, `d901c8c82784`, `536548e14dd5`, `1b8edfdba405`.
Drifted:

- `40bbb40c14ec` (lc anchor 0.25, seed 2): `5e870262/8698ff4d`. Its read is
  confounded.
- `2c1fd3ab7869` (the anchor-0.25 rerun, cuda:1): `c65cc0b6/b2c2c63e`, video
  half only. It is kept in the rerun gap, as R35 does.

### S1, 120 s (static)

Measures at the end of the take, and seams over all 28. **Bold** is better
than its control past the floor; _italic_ is worse.

| arm | job | control | jump | jump max | look | look max | tex fine | tex mid | tex coarse | face fine | level dB (not counted) | verdict (picture) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guide control | `8d1614ced27a` | | 1.341 | 6.56 | 2.233 | 11.51 | 1.706 | 1.171 | 0.088 | 0.711 | 5.9 | |
| lc, no anchor | `20ce09d60040` | `8d1614ced27a` | **0.932** | **1.49** | **1.016** | **2.49** | **1.105** | **0.740** | 0.147 | 0.746 | 12.1 | **wins** |
| anchor 0.25 | `25f6bd1da507` | | 0.943 | 1.86 | 1.032 | 2.70 | 1.108 | 0.430 | 0.019 | 0.040 | 5.8 | |
| lc + anchor 0.25 | `fc0f91663277` | `25f6bd1da507` | 0.842 | **1.17** | 1.175 | 2.13 | **0.643** | 0.334 | _0.216_ | 0.214 | 16.0 | mixed |
| anchor 0.25, seed 2 | `bf9ee32e040f` | | 0.987 | 1.43 | 1.295 | 2.66 | 1.142 | 0.590 | 0.017 | 0.467 | 10.5 | |
| lc + anchor 0.25, seed 2 (drifted) | `40bbb40c14ec` | `bf9ee32e040f` | **0.831** | 1.03 | 1.021 | **1.82** | **0.453** | **0.216** | _0.224_ | 0.048 | 9.8 | mixed (confounded) |
| keystone + anchor 0.25, pixel | `05a3a55b043c` | `25f6bd1da507` | 0.922 | 1.59 | 1.134 | 2.50 | **0.029** | **0.016** | 0.016 | 0.226 | 9.7 | **wins** |
| keystone + anchor 0.25, lc | `4766787e793d` | `05a3a55b043c` | 0.813 | **1.13** | 1.314 | 2.08 | 0.059 | 0.009 | 0.028 | 0.106 | 14.0 | tie |

Fine texture drift against frame 62, over the take:

| s | 12.5 | 24 | 42 | 62 | 83 | 104 | 119.5 |
|---|---|---|---|---|---|---|---|
| guide control | 0.180 | 0.536 | 1.108 | 1.399 | 1.646 | 1.658 | 1.706 |
| lc, no anchor | 0.132 | 0.225 | 0.547 | 0.997 | 1.207 | 0.930 | 1.105 |
| anchor 0.25 | 0.153 | 0.367 | 0.497 | 0.771 | 0.931 | 1.012 | 1.108 |
| lc + anchor 0.25 | 0.114 | 0.107 | 0.151 | 0.330 | 0.442 | 0.572 | 0.643 |
| keystone + anchor 0.25, pixel | 0.159 | 0.283 | 0.425 | 0.458 | 0.157 | 0.089 | 0.029 |
| keystone + anchor 0.25, lc | 0.085 | 0.118 | 0.065 | 0.044 | 0.076 | 0.053 | 0.059 |

The keystone-lc row is low throughout on the circle crop, yet the full frame
crusts from 42 to about 100 s. Read that row with the stills, not alone.

### By eye

Face crops and full frames were taken at 24, 42, 62, 83, 104 and 119.5 s, with
extra frames at 52-100 s for the keystone pair.

- **lc, no anchor (`20ce09d60040`).** It burns like the control: a painted,
  cracked face from 62 s, a high-contrast posterised face with white patches
  from 104 s, and a crusted backdrop. It is never soft. Its backdrop mosaic at
  120 s is finer than the control's shards, which matches the texture win.
- **lc + anchor 0.25 against anchor 0.25 (seed 1166).** Both are soft. The
  carry's backdrop is visibly less crusted at 42 s. From 83 s both faces are
  pale and washed out. At 104-120 s the carry's face is ghosted (a doubled,
  smeared outline), and the pixel take's is a white mask. Neither is usable.
  The carry adds no softness of its own: the pixel take is as soft.
- **Seed 2 (confounded).** The carry's face is present to 83 s and smeared
  after. The pixel control is a white mask with dark eye holes from 83 s.
- **Keystone pair.**
  - Both crust from about 42 s. The pixel take's crust is coarser.
  - Pixel: the backdrop is clean again from 81 s (it is still crusted at 79 s).
    No seam scores high there (look 0.93 at 81.7 s). From 104 s the face has
    streaks and the hair has turned dark and glossy.
  - Latent carry: crusted, though milder, until about 100 s, then clean by
    104 s. Its face at 104-120 s is the clearest of any two-minute take. The
    eyes, nose and mouth are sharp, and the face is a little plastic. The hair
    darkens in this take too.
  - Identity drifts in both: darker hair, a slightly different face from
    segment 1.

### Audio

Mean level per 20 s window (dBFS, from 5 s RMS windows):

| take | job | 0-20 | 20-40 | 40-60 | 60-80 | 80-100 | 100-120 |
|---|---|---|---|---|---|---|---|
| guide control | `8d1614ced27a` | -64 | -63 | -32 | -20 | -11 | -14 |
| lc, no anchor | `20ce09d60040` | -53 | -52 | -60 | -63 | -67 | -81 |
| anchor 0.25 | `25f6bd1da507` | -61 | -58 | -44 | -42 | -45 | -49 |
| anchor 0.25 rerun | `2c1fd3ab7869` | -63 | -52 | -47 | -45 | -36 | -20 |
| lc + anchor 0.25 | `fc0f91663277` | -52 | -63 | -55 | -68 | -64 | -67 |
| anchor 0.25, seed 2 | `bf9ee32e040f` | -63 | -62 | -58 | -72 | -62 | -74 |
| lc + anchor 0.25, seed 2 | `40bbb40c14ec` | -59 | -54 | -44 | -55 | -63 | -55 |
| keystone, pixel | `05a3a55b043c` | -60 | -62 | -64 | -79 | -72 | -56 |
| keystone, lc | `4766787e793d` | -62 | -62 | -70 | -79 | -55 | -58 |

- At seed 1166, without the keystone, all three pixel takes grow a late bed:
  -11 to -49 dB. Both carry takes stay at -52 to -81 dB. The bed has almost no
  energy above 6 kHz (0-1%), which fits a tonal music bed rather than hiss.
- Seed 2's pixel take has no bed at all, and its carry take is louder than it
  is. With the keystone, neither side has one.
- So the bed is something the pixel chain grows at one seed. The carry avoids
  it there, but "the carry suppresses the bed" does not hold in general.
- **S2 keeps its ambience.** Per segment, lc against control:
  - seed 1166: -50.4/-52.0/-54.1/-54.0/-53.9/-56.8 against
    -50.4/-52.1/-55.4/-55.6/-55.2/-57.1;
  - seed 2: -45.1/-43.9/-40.3/-45.6/-57.5/-56.9 against
    -45.1/-44.2/-40.2/-48.6/-61.3/-54.3.

  That is within 1-4 dB, segment by segment, on both seeds.

### S2, 6 segments (arc)

| arm | job | control | jump | jump max | look | look max | tex fine | tex mid | tex coarse | motion | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| guide control @1166 | `536548e14dd5` | | 1.362 | 2.09 | 1.286 | 1.74 | 3.084 | 2.293 | 1.430 | 4.16 | |
| lc @1166 | `d93adb33249e` | `536548e14dd5` | _1.664_ | 2.45 | _1.960_ | _2.75_ | 3.625 | _3.647_ | _3.536_ | 4.07 | mixed (seams worse) |
| guide control @2 | `1b8edfdba405` | | 1.306 | 1.83 | 1.418 | 1.77 | 3.518 | 2.994 | 2.545 | 4.44 | |
| lc @2 | `d901c8c82784` | `1b8edfdba405` | _1.480_ | 2.16 | _2.140_ | _3.54_ | 3.525 | 3.518 | 3.186 | 4.35 | mixed (seams worse) |

- The rule says *mixed*: the primary (fine texture) ties inside S2's wide floor.
  Plainly read, the carry **loses on the seams on both seeds**: look mean
  +0.67 and +0.72 against a 0.438 floor, and jump mean past 0.124 on both.
- Motion is kept (4.07 against 4.16, and 4.35 against 4.44), so the camera
  still moves.
- The S2 carry takes are 12.8-13.3 min, against 10.8-10.9 min for the
  controls. Most of the gap is a fresh model load (160-167 s) in both carry
  jobs. Without it they take 10.1-10.5 min, about the same as the controls.

### The call (Step 3)

- **#1044 (latent carry, extra rows): not built as a default.** On the static
  scene it wins texture at the cost of coarse texture. It does not stop the
  burn, and it ties when isolated. On the moving scene it loses the seams. Keep
  the scratch flag for the keystone round below; the second ksl seed decides it.
  If Don's eye does not prefer the carry's face there, close it with these
  numbers.
- **#1150 (the VRAM-saving arm 2) and #1030 (decode-once):** not measured by
  this A/B. The only saving seen is the skipped re-encode, about 5 min per
  two-minute take. That is not worth a feature on its own.
- **The keystone reset is the lead to chase.** It is one seed, and it is the
  first sign of a two-minute static take that comes back clean.

### Next round for Steps 5-7 (#1188), recommended

- **Run the keystone confirmation first. It is the smallest round that can
  change the plan.** Two arms at seed 2, S1, 120 s, on two cards at once:
  - `ks-pixel-anchor25-s1-120s-s2`;
  - `ks-lc-anchor25-s1-120s-s2`.

  That is about 2 x 68 min, 2.3 GPU-hours, or 70 min of wall. Optionally add
  a third arm, keystone without the anchor at seed 1166 (about 52 min, 0.9
  GPU-hours), to separate the keystone's reset from anchor 0.25's softness.
  In total that is about 3.2 GPU-hours, or about 2 h on two cards.
- **Hold the Step 5 build (hybrid windows) until that round reads.**
  - Step 4's windowed refine already gives the seam pass, and it does not touch
    the burn.
  - If the chain keystone resets the burn at seed 2 as well, Step 6's
    keystone/capstone question is answered in chain form. Step 5's remaining
    value is then the seamless blended finish: a new block, plus at least
    36 min per minute of take. That is a weak case.
  - If the reset does not repeat, hybrid windows with a finished window-1
    keystone (Steps 5-6) is the remaining structural burn lever, and it is worth
    the build.
- **Latent carry:** do not open a follow-up feature. The ksl seed-2 arm above
  decides whether the scratch flag stays (face hold by eye) or #1044 closes.

### Exports for Don

Exports are in `~/Desktop/dw-bench-t17/`, side by side, x264 CRF 16:

- `t17-s1-120s-keystone_LEFT-pixel_RIGHT-latentcarry-75-120s.mp4`
  (`05a3a55b043c` | `4766787e793d`, the right take's track). It shows the reset
  at about 80 s on the left, and the carry's face at the end.
- `t17-s1-120s-noanchor_LEFT-pixel-control_RIGHT-latentcarry-40-70s.mp4`
  (`8d1614ced27a` | `20ce09d60040`, the right take's track). It shows that
  without the anchor the carry burns rather than softens.

Records: `latent-carry.json` (the a and b drivers merged),
`latent-carry-controls.json`, `latent-carry-controls-r36.json` (arms
`bench/arms/latent-carry-controls-r36.json`), `latent-carry-noanchor.json`,
and the moments pass `t17-t19-moments.json`. The moments pass was taken
in-process on lem from the bench checkout at `6a89744a`: raw texture bands,
face crop, seams, motion, per-segment audio, and PSNR.
