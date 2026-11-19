# H3 stochastic sampling: turbo video and H3-Image (#1438, stage #1459)

Does `MiniMaxH3Scheduler(stochastic_sampling=True)`, a re-noise on every step, render
cleaner H3 turbo video (lightx2v 4-step LoRA), reduce texture creep on a chain, or clean
speckle from H3-Image?

**Answer: no arm is adopted.**
- The flag takes. It renders turbo video that is clean, not broken.
- Its re-noise is unseeded: two flagged runs at seed 42 differ as much as flag-off differs
  from flag-on, while two flag-off runs are bit-identical.
- On a 3-segment chain it does not reduce creep. It creeps more than the control, and
  within that noise.
- H3-Image shows no speckle at 2048² or at 16 MP, so the image arms had nothing to fix
  and did not run.

Run 2026-11-19 by claude-opus-5-5 (anthropic):
- the main `dw.serve` on lem, as deployed from develop (this stage changes no engine code);
- workspace `bench`;
- diffusers 0.44.0.dev0 @ `e84bda1c`.

## Setup

- **The scheduler block.** Every probe workflow rebuilds the video scheduler from config,
  so the flag reaches `__init__`:
  `"scheduler": {"configuration": {"scheduler_type": "MiniMaxH3Scheduler"},
  "from_config_args": {"stochastic_sampling": "variable:stochastic_sampling",
  "euler_update_form": "variable:euler_update_form"}, "shift": "variable:video_shift"}`.
  - The control arms go through the same rebuild with the flag `false`, so the flag is
    the only difference between arms.
  - `audio_scheduler` stays shift-only (`{"shift": "variable:audio_shift"}`).
  - The H3-Image copy has the same block without `shift`.
- **Variables reach `from_config_args`.** No fixed per-arm copies were needed.
- **The workflows:**
  - `bench/workflows/probe-1438-video-with-audio.json` (from
    `templates/minimax/video-with-audio.json`): 960x544, 124 frames, 5 steps, video
    shift 6, audio shift 3, seed 42.
  - `bench/workflows/probe-1438-chained-segments.json` (from
    `templates/minimax/chained-segments.json`): 3 segments, seed 42, the template's
    default prompt and still (the glass figure), `continuity` `guide`, `shot` `moving`.
  - `bench/workflows/probe-1438-h3-image.json` (from
    `templates/minimax-h3-image/text-to-image.json`): seed 1317, size `"2048"`.
- **Arms:** `bench/arms/probe-1438.json`.

### Each checkpoint's update form

Read from `scheduler/scheduler_config.json` on lem:

| Checkpoint | `euler_update_form` | Consequence |
|---|---|---|
| `MiniMaxAI/MiniMax-H3` | not declared (`{shift: 12.0, ...}`), so the class default `blend` | The flag needs no other change. **No V0b (blend only) arm**: blend is already what runs. |
| MiniMax-H3-Image | `velocity` (also `shift_provided_sigmas: false`) | The flag would force `blend` too, hence the I1 (blend only) / I2 (blend + flag) split. Not run: see I0. |

The probe workflows' `euler_update_form` variable defaults to each checkpoint's own form:
- `blend` for video;
- `velocity` for H3-Image.

## Proof that the flag took

- **The fail-fast arm (FF) was refused at load.** It set `stochastic_sampling` true with
  `euler_update_form` `velocity`. The job (c811b290cab1) failed in 9 s with:
  "`stochastic_sampling` replaces the Euler update, so `euler_update_form` has no effect
  with it; leave it at 'blend'". That is `scheduling_minimax_h3.py:110-113`, raised from
  `__init__` through `components.py:515` (`from_config`). So the key reaches the
  constructor.
- **The scheduler load is logged, but only in the server log.**
  - The server journal on lem (`journalctl --user -u dw-serve`) has `INFO:dw:Loading
    scheduler: <class '...MiniMaxH3Scheduler'>`, then `Setting scheduler shift: 6` and
    `Setting audio_scheduler shift: 3`, for every video arm (06:08:08 V0, 06:09:38 V1,
    06:11:03 V1b).
  - That line does **not** reach `get_job_events`. The job's events stop at the
    `loading` phase, the LoRA, the attention backend and the denoise block names.
  - The worker's dw logger lines for the scheduler are not forwarded as job `log` events.
    This is a gap in what MCP shows, not in the run.
- **A change to the flag alone forces a reload.** V1 ran straight after V0 on the same
  card with only the flag changed, and reports `load_cached: false`. `from_config_args`
  is part of the pipeline cache key, so the flag is in the loaded pipeline.
- **The installed build has the flag.** `get_diffusers_state`: 0.44.0.dev0 @ `e84bda1c`.
  Its `scheduling_minimax_h3.py` declares `stochastic_sampling: bool = False` (`:102`),
  with the blend-only check at `:110`.
- **The realized workflows carry it.** `get_job_workflow` on each arm shows the
  `scheduler` block above, with its own `stochastic_sampling` and `euler_update_form`
  variables, and `audio_scheduler` as shift only.
- **No step was reused.** No arm's manifest entry is marked `reused`. The repeat arms
  (V1b, V0b-extra) ran under the `-b` workflow id (`Probe1438VideoWithAudio-b`) to miss
  the step cache.

## Runs

| Arm | Job | Card | Wall | Timing | Output (workspace `bench`) |
|---|---|---|---|---|---|
| FF `1438-FF-flag-with-velocity` | c811b290cab1 | cuda:1 | 9 s | failed at load (expected) | none |
| V0 `1438-V0-flag-off` | 1fdfb1fb9b00 | cuda:1 | 88 s | load 4.7, encode 12.9, denoise 48.0, decode 20.3 s; 4 passes | `probe-1438-video-with-audio/20261119-060805-986f9d0a/final/Probe1438VideoWithAudio-text_to_video_audio.0-0.0.mp4` |
| V1 `1438-V1-flag-on` | e31891b65173 | cuda:1 | 84 s | `load_cached: false`; denoise 48.0 s | `probe-1438-video-with-audio/20261119-060932-aa96f052/final/Probe1438VideoWithAudio-text_to_video_audio.0-0.0.mp4` |
| V1b `1438-V1b-flag-on-again` | 59ac324fe47b | cuda:0 | 89 s | | `probe-1438-video-with-audio-b/20261119-061059-ce16afb2/final/Probe1438VideoWithAudio-b-text_to_video_audio.0-0.0.mp4` |
| V0b-extra (not in the arms file) | 449a0eadbd7e | cuda:0 | 85 s | load 4.6, encode 12.2, denoise 45.0, decode 19.5 s | `probe-1438-video-with-audio-b/20261119-062116-456254be/final/Probe1438VideoWithAudio-b-text_to_video_audio.0-0.0.mp4` |
| C0 `1438-C0-chain-flag-off` | d5ec6eadb81c | cuda:0 | 329 s | 3 segments: shots at frames 0, 124, 226 | `probe-1438-chained-segments/20261119-061547-8b5c1059/final/Probe1438ChainedSegments-chained_keyframe_to_video_audio.0-0.0.mp4` |
| C1 `1438-C1-chain-flag-on` | 60148e0df482 | cuda:1 | 332 s | denoise 195.1, decode 62.0 s | `probe-1438-chained-segments/20261119-061551-d87635ad/final/Probe1438ChainedSegments-chained_keyframe_to_video_audio.0-0.0.mp4` |
| I0 `1438-I0-as-is` | 85538c37cea0 | cuda:0 | 197 s | load 56.9, encode 37.4, denoise 92.4 s; 6 passes; 2048x2048 | `probe-1438-h3-image/20261119-061228-8d988866/final/Probe1438H3Image-text_to_image.0-0.0.png` |
| I0-16 `1438-I0-16-as-is-16mp` | 37d6b88a8ba3 | cuda:0 | 207 s | load 63.2, encode 29.2, denoise 98.7 s; 6 passes; 4096x4096; host peak 62 GB | `probe-1438-h3-image/20261119-062242-0c61bb0a/final/Probe1438H3Image-text_to_image.0-0.0.png` |
| I1, I2, I2b | not run | | | gate unmet: no speckle in I0 or I0-16 | |

- **About V0b-extra.** It is V0 repeated (flag off, seed 42) under the `-b` id. It is the
  control for the seeding check, so that "V1 and V1b differ" can't be put down to the card
  or the cache.
- **GPU time.** Total job wall is about 23.7 min, inside the 60 min cap. C0 and C1 ran in
  parallel on the two cards.

## Seeding

`compare_outputs`, full frame, at `frame:0`, `frame:62` and `frame:123`:

| Pair | PSNR mean (min-max) | SSIM | Mean abs diff |
|---|---|---|---|
| V0 vs V0b-extra (both flag off, cuda:1 vs cuda:0) | identical | 1.0 | 0.0 |
| V0 vs V1 (flag off vs on) | 12.74 dB (11.77-13.68) | 0.380 | 42.37 |
| V1 vs V1b (both flag on, seed 42) | 12.44 dB (12.07-12.87) | 0.383 | 43.58 |

**V1 and V1b differ.** Flag off reproduces bit for bit, even across cards. Flag on does
not reproduce at the same seed: the re-noise draws from the unseeded global RNG
(the H3 pipeline's denoise loop in diffusers calls `scheduler.step` without a
`generator`; dw doesn't call it).

V1 against V1b differs as much as V0 against V1. Both pairs keep the composition (a fox
in a sunlit snowy pine wood, a close-up by frame 123) and differ in all the detail.

So on a full-frame metric, one flagged draw cannot be told apart from the flag's own
noise. The comparison that remains is by texture and by eye, below.

## Texture: single shot

`compare_outputs(texture_bands=true)`. Values are luma (Y) and chroma (R-G, B-G) RMS
energy, fine 2-8 px / mid 8-32 px / coarse 32-128 px.

**Full frame, V0 baseline, V1 / V0 ratios:**

| Frame | Y | R-G | B-G |
|---|---|---|---|
| 0 | 0.975 / 1.009 / 1.148 | 0.945 / 0.944 / 1.022 | 0.970 / 0.997 / 1.175 |
| 62 | 1.122 / 1.036 / 1.147 | 0.858 / 0.917 / 1.056 | 0.903 / 0.957 / 1.015 |
| 123 | 1.261 / 1.213 / 1.222 | 1.021 / 0.981 / 0.986 | 1.135 / 1.258 / 1.002 |

V0's own bands at frame 0:
- Y 7.1434 / 23.2216 / 34.9612;
- R-G 2.2736 / 6.7797 / 10.9765;
- B-G 4.1409 / 12.3839 / 15.7416.

**Fox crop `[330, 60, 320, 320]` at `frame:123`** (PSNR 12.93, SSIM 0.422):

| | Y | R-G | B-G |
|---|---|---|---|
| V0 | 5.0203 / 12.1761 / 28.1259 | 1.5928 / 4.4612 / 15.2338 | 2.3102 / 7.2588 / 25.0504 |
| V1 | 6.5021 / 16.3345 / 37.6622 | 1.4875 / 3.9258 / 13.2633 | 2.49 / 8.2144 / 27.7361 |
| V1 / V0 | 1.295 / 1.342 / 1.339 | 0.934 / 0.880 / 0.871 | 1.078 / 1.132 / 1.107 |

**By eye (1:1 crop):**
- V1's fox is clean: crisper fur, no grain and no chroma speckle. Its chroma bands sit at
  or below V0's.
- V1's higher luma energy is fur and edge detail, not noise.
- "Clean or better" held, so the chain pair ran.
- Every luma ratio here is within the V1-against-V1b spread. The extra detail is one draw,
  not a measured effect.

## Chain (C0 vs C1)

These are #1399's measures:
- `bench/region_creep.py`, run on lem against the saved files: Laplacian variance of the
  gray frame, median per 1 s window, last full window over the first;
- `assess_output(probe="analyze_seams")`: its per-seam `lightness_step`, and their mean
  absolute value.

Tolerances are #1399's: creep ±15%, seam L ±0.3, mean |L| ±0.1.

| | C0 (flag off) | C1 (flag on) | C1 against C0 |
|---|---|---|---|
| Whole-frame sharpness, first 1 s to last full 1 s | 111 -> 202 (x1.82) | 117 -> 251 (x2.15) | +18%, **outside** ±15%: more creep |
| Centre-box sharpness | 151 -> 166 (x1.10) | 160 -> 293 (x1.83) | +66%, **outside**: more creep |
| Seam lightness step, seams 1-2 | -1.30, +2.17 | -1.43, -2.02 | seam 1 within ±0.3; seam 2 flips sign, outside |
| Mean seam \|L\| | 1.74 | 1.73 | within ±0.1 |
| `analyze_seams` `jump_ratio`, seams 1-2 | 1.15, 1.57 | 1.26, 1.34 | |
| `analyze_seams` findings | none | seam 1 `seam_level_step` 5.44 dB (warn, >3 dB) | |
| `segment_looks.detail`, segment 1 first to segment 3 last | 6.234 -> 7.417 | 6.148 -> 7.221 | |

**`compare_outputs(texture_bands=true)`**, full frame, C0 baseline, at `frame:0`,
`frame:123`, `frame:225` and `frame:327` (the last frame of each segment, and frame 0):
- PSNR 24.50 dB (16.15-33.82), SSIM 0.747.
- Luma fine band:
  - C0: 5.93 -> 7.81 -> 8.96 -> 9.44;
  - C1: 5.95 -> 7.95 -> 8.80 -> 9.91.
- Every C1/C0 band ratio is within 0.92-1.09.

By eye, both chains climb in detail together, segment over segment. The flag doesn't stop
it.

**Single-shot creep, for scale** (region_creep, whole frame / centre box):
- V0: x0.70 / x1.19;
- V1: x1.02 / x1.20;
- V1b: x0.64 / x0.80.

The two flagged draws spread by 40% on the whole frame, wider than the ±15% tolerance.
C1's extra creep is therefore inside the flag's own draw-to-draw noise. It is not a
measured harm, but it is no reduction either.

## H3-Image

- **I0** at 2048x2048 (the lighthouse keeper on the gallery, seed 1317): viewed full at
  512 and at 1:1 crops. Clean, with no speckle. The face is soft.
- **I0-16** at the `1:1` preset, 4096x4096: it fit on the 3090 (host peak 62 GB, denoise
  98.7 s). Viewed at 512 and at 1:1:
  - crop `[1760, 2300, 512, 512]` (the keeper);
  - crop `[2900, 1200, 512, 512]` (the sky).
  - **No speckle.** The sky is smooth. The face and coat carry a crisp, slightly
    over-sharpened texture, not coloured noise.
- **Gate unmet.** The plan runs I1/I2/I2b only if I0 or I0-16 shows speckle at 1:1.
  Neither does, so they did not run. The arms file keeps them for a later run with a scene
  that does speckle. No `probe-1438-h3-image-b` copy was saved.

## Verdict per arm

**Any adoption needs a seeded path first.** The re-noise must take the step's generator,
either upstream in diffusers or in a dw wrapper. Without one, a flagged template doesn't
reproduce at its seed, which V1 against V1b shows.

| Arm | Verdict | Why |
|---|---|---|
| V1 (flag on, single turbo shot) | **Drop** (not a template variable, not documented) | It renders clean, but the extra detail is within its own draw-to-draw noise, and it is unseeded. Revisit only with a seeded path, and with several draws per arm. |
| C1 (flag on, chain) | **Drop** | Creep is no lower than C0: x2.15 against x1.82 whole frame, x1.83 against x1.10 centre box. The bar was "burns less", and this fails it. Mean seam \|L\| is unchanged. |
| I1 / I2 (H3-Image blend, blend + flag) | **Not run, no verdict** | No speckle to clean at 2048² or 16 MP on this scene. |
| FF | (control) | Did its job: the flag reaches the constructor. |

## Deviations from the plan

- **No V0b (blend only) arm.** The video checkpoint is already `blend`.
- **V0b-extra was added.** It is a flag-off repeat at seed 42 under the `-b` id, and the
  control that makes the seeding result read cleanly.
- **The runs went through MCP, not `bench/driver.py`.** Each `run_workflow` went to the
  main server's workspace `bench`, with a note naming its arm. The arms file's `scene` is
  `s1-static` on every arm, only to satisfy the loader. The probe workflows use their
  templates' own prompts and stills, and no scene's.
- **The scheduler log line is not in `get_job_events`.** It appears only in the server
  journal (above). Proposed follow-up: forward the worker's scheduler-load log line as a
  job `log` event.
- **No extra flagged draw (V1c).** V0b-extra being bit-identical already isolates the
  flag's noise.

## Follow-ups (Don decides; not built here)

- **A seeded re-noise path.** Not needed now, since no arm won.
- **Stray scheduler keys pass validation silently.** As plan v1 proposed.
- **The scheduler-load line is missing from job events.** Above.
