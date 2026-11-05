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

### The 120 s question (S1, `r3-anchor25-s1-120s`): blocked

**The question.** Does the refine undo the burn from about 42 s on?

The run is blocked, but the 26 s data already answer the question in kind.
At every strength the refine leaves texture drift against segment 1 inside
the floor, and by eye it keeps the crust. A low-sigma refine (0.2-0.4)
re-denoises from a burned picture and keeps its structure. At 0.4 the PSNR
against the burned source is still about 27 dB. So the expectation at 120 s is
seam smoothing only. The burn after 42 s is the latent carry family's problem
(#1150, #1044, #1030), not this pass's.

**Why it is blocked.** A 120 s take cannot be decoded on a 24 GB card by the
stock video decode. The decode keeps every clip on the device, concatenates
them, and converts the whole take to float32. At 960x544 that is about
9 + 9 + 18 GB, and `decode_h3_latents` uses a float32 VAE. The fix is
`fb6a9e11` (`streamed_decode`), which is bit-identical to the whole decode.
The bench server still runs 082e93ce, so the fix is not deployed. Once it is,
run `t17-s1-120s-r030-rebased` with `source:
output:BenchS1Static/20261104-233558-8c6ae697/final/BenchS1Static-chained_keyframe_to_video_audio.0-0.0.mp4`,
`window_frames 42`, `window_overlap 12`, `window_positions rebased`, and the
S1 prompt. Measure it with `refine_pairs.py` at the round-3 moments (62, 300,
583, 1000, 1501, 2000, 2500, 2868). It should take about 40 min.

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
  26 s, falling toward about 20 min per minute at length;
- **it is not a burn fix**: drift against segment 1 is unchanged on every
  scene and strength.

Promoting it out of the experimental flag needs four things, as Step 4's "if
it wins" says:

- the 120 s confirmation;
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
  6 segments, 0.3 rebased. This stands in for the 120 s pair, which is blocked
  until the decode fix is deployed.
