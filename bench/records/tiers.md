# Tiers on the long-form templates (Task 20)

Run 2026-11-06 on the bench server (lem, RTX 3090s) at `279086b1`. From 15:45
UTC it ran on cuda:0 only, at Don's request. From 17:35 a second server on
:8767 (cuda:1, same workspace) took half of the arms. Arms:
`bench/arms/tiers.json`, `tiers-b.json`, `tiers-c.json`, `tiers-int8.json` and
`tiers-template.json`. Wall times are cuda:0 unless a row says cuda:1, which
runs about 10% slower. Every arm is seed 1166 unless named `-s2`. S1-S3 run 6
segments (634 frames, 26.4 s). S4 is the 4-shot montage.

## What the tiers are

The `tier` variable (`draft`, `standard`, `best`; default `standard`) moves
canvas, steps and weights together through `presets`
(`dw/presets.py`). Any variable the caller passes still wins. `tier` is
independent of `shot` on `minimax/chained-segments`. `shot` picks the
drift counters, `tier` the render settings, and the two bundles share no
variable.

| template | draft | standard (unchanged default) | best |
|---|---|---|---|
| `minimax/chained-segments` (FL2VA, 4-step v1.2 LoRA) | 960x544, **4 steps**, int4 | 960x544, 5 steps, int4 | **1344x768**, 5 steps, **int8** |
| `minimax/chain-video-continuity`, `minimax/chain-matched-to-audio`, `minimax/chain-matched-and-aligned`, `minimax/dialogue-short`, `minimax/music-video` (Ref2VA, 8-step v1.0 LoRA) | 960x544, **5 steps**, int4 | 960x544, 9 steps, int4 | **1344x768**, 9 steps, **int8** |

Wall time against standard, from the 26 s scenes, as a guide for a take
of any length:

| template family | draft | standard | best |
|---|---|---|---|
| FL2VA chain (S1, S2) | 0.66-0.72x | 1x (S1 14.1 min for 26 s; 2-minute `static_long` take 59.2 min) | 1.68x on S1 (about 100 min for a 2-minute take, estimated, not run) |
| Ref2VA shots (S4) | 0.56x | 1x (S4 20.3 min) | 1.32x |

**Cost rule (Step 2): passes.** `standard` is today's default, and a 2-minute
take at `standard` with `shot: static_long` ran **59.2 min** on one card
(`aa80458ab4f7`, below), well under the 3 h line. `best` is about 100 min
for the same take. It needs about 65 GB of host RAM per job, so a 128 GB
box runs one such job at a time (*Host RAM*).

Not tiered:

- **`ltx2/chained-segments`, `ltx2/extend-clip`: skipped.** LTX-2.5's
  distilled transformer runs a fixed 8-sigma schedule with no step knob
  (`docs/RECIPES_24GB.md`). Only `uint4` fits with the transformer resident
  on a 24 GB card, so no weights step exists either. Canvas is the only dial
  left. It is the caller's aspect choice (768x768 and 768x448 by default), and
  the recipe spends headroom on the two-stage refine rather than on base
  resolution. The quality step past the default is a separate pass
  (`ltx2/upscale-clip`, `two-stage`), not a setting of the chain. Takes are
  capped at about 15-20 s (`ltx.md`), so a draft would save minutes, not
  hours.
- **Kandinsky 6:** out of long-form scope (ruling).
- **Attention, cache, check and retries** are not in any bundle. Sparse
  top-k 0.15 is already the default at every tier (`speed.md`). The
  `first_block` and MagCache caches are measured only on the base model's
  20-30-step schedule (#862), not on these 4- and 8-step LoRAs. The chain
  check's re-roll cannot fix texture drift (`check.md`). These stay knobs.

## Step 1: quality per minute

### FL2VA chain (`minimax/chained-segments` settings, bench S1/S2/S3)

Records `bench/records/tiers.json` (the driver records, failures included, and the staging/crust pass), copied from `run1/tiers.out.json`, `run2.out.json`, `cuda1.out.json`,
`i8*.out.json` (lem `~/bench-t20`). Measures are the driver's (lower is
better). Staging carry is PSNR/SSIM against the same scene and seed's
standard control at frames 62, 277 and 583, both frames resized to 480x272.
PSNR bands from `speed.md`: about 44 dB is rerun noise, 23-27 dB the same
shot with changed numerics, and 19 dB or less a re-staged shot. On the S2 arc,
even a lossless change reads 9-13 dB. Crust is the Laplacian variance of the
whole frame at native size. It grows with the burn, so read its rise across
the take within one canvas only.

| arm | job | jump | look | level dB | tex fine | tex mid | tex coarse | crust f62 / f583 | carry PSNR f62 / f277 / f583 | wall min |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 standard (ctl) | `f5bb2add1911` | 0.854 | 1.464 | 7.65 | 0.536 | 0.176 | 0.016 | 195 / 879 | - | 14.1 |
| S1 4 steps (draft) | `cc6e23eb555a` | 0.960 | 1.694 | 10.82 | 0.519 | 0.195 | 0.017 | 203 / 793 | **32.3 / 27.1 / 21.0** | **9.3** |
| S1 672x384 | `f9d87fbc0d2f` | 0.912 | 1.562 | 10.91 | 0.330 | 0.135 | 0.008 | (canvas) | 19.0 / 19.1 / 17.3 | 11.1 |
| S1 int8 | `74c4feddb02b` | 0.888 | **0.590** | 4.36 | **0.269** | 0.047 | 0.017 | **141 / 232** | 17.0 / 17.9 / 15.4 | 14.6 |
| S1 1344x768 | `1c5014a737ab` | 0.730 | 1.436 | 5.20 | 1.008 | 0.487 | 0.048 | (canvas) 66 / 362 | 19.3 / 19.1 / 18.3 | 26.6 |
| S1 1344x768 int8 (best) | `6810b23c2e35` | **0.628** | **0.442** | 13.32 | 0.359 | 0.272 | 0.001 | (canvas) **54 / 84** | 17.3 / 18.7 / 15.0 | 23.7 |
| S1 1344x768 8-step v1.0 @9 | `856b243ef5fd` (cuda:1) | 1.012 | 2.480 | 1.45 | 1.026 | 0.362 | 0.101 | (canvas) 73 / 636 | 18.4 / 18.2 / 15.2 | 38.8 |
| S1 standard, seed 2 | `9badf299b28e` | 0.792 | 1.354 | 10.55 | 0.463 | 0.145 | 0.016 | 165 / 804 | - | 12.9 |
| S1 672x384, seed 2 | `170650f97b54` | 0.856 | 1.594 | 5.82 | 0.199 | 0.119 | 0.043 | (canvas) | 19.8 / 21.7 / 17.4 | 7.5 |
| S1 1344x768, seed 2 | `bca819290484` | 0.746 | 1.312 | 5.91 | 1.086 | 0.481 | 0.059 | (canvas) 77 / 502 | 20.7 / 18.5 / 17.3 | 19.6 |
| S2 standard (ctl) | `e649195caf6c` | 1.362 | 1.286 | 1.46 | 3.084 | 2.293 | 1.430 | - | - | 13.1 |
| S2 4 steps (draft) | `d14a8877f205` | 1.362 | 2.144 | 3.25 | 3.050 | 1.889 | 1.022 | - | **21.6** / 12.6 / 10.5 | **9.4** |
| S2 672x384 | `12bc51e5dbe7` | 1.602 | 2.222 | 3.89 | 3.468 | 2.557 | 1.501 | - | 12.7 / 11.7 / 9.7 | 9.3 |
| S2 1344x768 | `4b3d50baadbf` | 1.128 | 1.598 | 3.72 | 2.602 | 1.316 | 0.214 | - | 16.2 / 11.6 / 9.7 | 23.2 |
| S2 standard, seed 2 | `96a443e3aa38` | 1.394 | 1.452 | 2.01 | 2.862 | 1.606 | 0.523 | - | - | 13.7 |
| S2 int8, seed 2 | `a6c8c306fe92` (cuda:1) | 1.392 | **1.060** | 6.13 | 2.827 | 2.551 | 3.655 | - | - | 14.9 |
| S2 4 steps int8 | `3c57f30e4091` (cuda:1) | 1.440 | 1.238 | 2.00 | 1.403 | 0.419 | 0.854 | - | - | 12.6 |
| S3 standard (ctl) | `1b406282af1a` | 1.524 | 1.470 | 2.97 | 0.067 | 0.114 | 0.188 | 80 / 107 | - | 10.9 |
| S3 672x384 | `f566f7ff5971` | 1.568 | 1.840 | 3.98 | 0.837 | 0.721 | 0.500 | (canvas) | 15.3 / 13.4 / 10.4 | 7.9 |
| S3 1344x768 | `27cdc5df3a40` | 1.244 | 1.402 | 3.64 | 0.462 | 0.190 | 0.080 | (canvas) | 15.5 / 12.3 / 10.4 | 20.2 |
| S3 standard, seed 2 | `81006f364dae` | 1.192 | 1.060 | 2.59 | 0.294 | 0.226 | 0.299 | - | - | 13.7 |

Int8 confirmation (`bench/arms/tiers-int8.json`). Run one at a time on :8766
(`i8e`), after the two-server OOM (*Host RAM*). Controls are the rows above.

| arm | job | jump | look | level dB | tex fine | tex mid | tex coarse | wall min | against its control |
|---|---|---|---|---|---|---|---|---|---|
| S1 int8, seed 2 | `ebfd064fa9de` | 0.738 | **0.906** | 3.52 | **0.336** | 0.011 | 0.014 | 12.3 | wins look (1.354) and fine texture (0.463) |
| S1 1344x768 int8, seed 2 | `00fddd52dd59` | 0.732 | **0.484** | 3.98 | 0.406 | 0.234 | 0.036 | 22.8 | wins look; best of seed 2 again |
| S2 int8 | `f2dc3661076c` | _1.770_ | 1.574 | 3.42 | 2.894 | 2.317 | 1.128 | 12.3 | **jump worse** (1.362), look inside the gap |
| S2 int8, seed 2 | `a6c8c306fe92` (cuda:1) | 1.392 | **1.060** | 6.13 | 2.827 | 2.551 | _3.655_ | 14.9 | wins look, coarse worse: mixed |
| S2 1344x768 int8 | `5c3dd03000a1` | _1.468_ | **0.998** | 3.80 | 3.418 | 3.552 | 3.372 | 20.5 | wins look, jump and mid/coarse worse: mixed |
| S3 int8 | `167656fcf86d` | _1.860_ | **0.990** | 2.35 | _0.262_ | 0.367 | 0.012 | 12.4 | jump and fine worse, look wins: mixed |
| S3 int8, seed 2 | `3242ffebdc08` | _1.474_ | 1.010 | 1.49 | 0.244 | 0.319 | 0.003 | 12.3 | jump worse (1.192), rest inside: mixed |

**Two minutes, `static_long` (keystone slot last), int8** on the S1 bench
copy, the same chain as the template run below (`t20-int8-static-long-120s-s1166`):

| arm | job | jump | look | level dB | tex fine | tex mid | tex coarse | crust f62 / f1501 / f2868 | wall min |
|---|---|---|---|---|---|---|---|---|---|
| int4 (the template run, below) | `aa80458ab4f7` | 0.722 | 1.499 | 7.05 | 0.121 | 0.015 | 0.007 | 195 / 351 / 394 | 59.2 |
| int8 | `513f5ddd8d76` | **0.544** | **0.658** | 3.99 | **0.053** | 0.025 | 0.019 | 147 / 186 / 198 | 62.2 (+5%) |

By eye at 62, 1501 and 2868 frames both two-minute takes hold the wall and
the face, since the keystone slot does most of the work. The int8 take is
the cleaner of the two at every moment: smoother stripes, less grain on the
white ground and the circle.

**Int8 verdict.** On the static scene it wins on both seeds, at 26 s and at
two minutes, for 0-5% more wall time. On the arc and the dialogue walk it is
mixed on every arm: seam look improves, and seam jump gets worse on three of
four arms, past the 0.124 rerun gap. So it is a static-shot lever, not a
general one.

By eye (frames 62 and 583, S1 contact sheet; frame 583 on S2 and S3):

- **Draft (4 steps)** is the standard take: same framing, pose and face at
  frame 62 (PSNR 32 dB, closer than the "same shot" band), and the same crust
  by 24 s. It diverges later in the chain, as any change does once seams
  compound (21 dB at 24 s). On the S2 arc segment 1 carries (21.6 dB against
  a 9-13 dB lossless floor) and the later segments go their own way. It
  saves 28-34%. **This is the draft:** the only cheap point whose staging
  carries to `standard`.
- **672x384** re-stages (19 dB S1, 12.7 dB S2) and looks worse: a blotchy
  face and a crusted black circle on S1 at 24 s, a washed-out sky on S2, a
  gaunter face that loses identity on S3. It saves 21-42%. Rejected: it is
  neither a preview of the standard take nor a usable take.
- **int8** removes most of the S1 burn. At 24 s the wall's stripes are
  smooth, the white ground is clean and the face is natural, where int4 has
  crusted everything. The measures agree: seam look 1.46 to 0.59, fine
  texture drift 0.54 to 0.27, and crust rises 1.6x across the take
  against 4.5x. Same wall time (14.6 against 14.1 min). It re-stages
  (17 dB), so it is a different take, not a final of an int4 draft.
- **1344x768** alone keeps the int4 crust (crust 66 to 362, 5.5x) at 1.9x
  the wall time. With int8 it is the cleanest S1 take measured: crust flat
  (54 to 84), seam look 0.44, jump 0.63, at 1.68x the wall time (int8 runs
  faster than int4 at this canvas: 23.7 against 26.6 min). On S2 and S3
  1344x768 sharpens the scene and holds S3's face. **This is best.**
- **The 8-step v1.0 768p file at 9 steps** loses to the 4-step file at
  the same canvas on every seam measure. Its crust reaches 636 by 24 s, and it
  costs 2.7x standard. Rejected.

### Ref2VA shots (`dialogue-short`/`music-video` settings, bench S4)

S4 is four `dialogue-short` shots cut together, so its seams are cuts and
its seam measures are not drift (all arms read jump 19-21 and look 12-14). Read
the wall time and the eye check. PSNR against the control is 12-15 dB for
every arm, including the 5-step one. Talking close-ups move too much for a
frame match to show staging.

| arm | job | tex fine | tex mid | tex coarse | wall min |
|---|---|---|---|---|---|
| S4 standard (ctl, 9 steps int4) | `6d33a88d24de` | 1.714 | 0.580 | 1.214 | 20.3 |
| S4 5 steps (draft) | `cad009e3c50b` | 2.184 | 1.145 | 1.771 | **11.4** |
| S4 672x384 | `217149738709` | 1.812 | 0.619 | 1.078 | 14.5 |
| S4 int8 | `26e3fef72e97` | 1.869 | 0.684 | 0.973 | 18.3 |
| S4 1344x768 | `518d49656426` | 0.095 | 0.410 | 0.075 | 25.8 |
| S4 1344x768 int8 (best) | `413a48627655` | 0.009 | 0.308 | 0.145 | 26.8 |

By eye (frames 62 and 186): every arm gives the same close-up of the same
man with the same book and couch. Framing, pose and the read hold at 5 steps,
so the draft keeps the shot. 672x384 frames him tighter, a re-staged shot
for a 29% saving: rejected. The 1344x768 arms are the sharpest. Their
crop texture reads 0.01-0.10 against 1.7, though across cuts that is a
shot-to-shot difference, not drift. 1344x768 is the Ref2VA
LoRA's own canvas, and the H3 ramp's TV final (L7) is 1344x768 int8 too
(`plugins/dw/skills/minimax-h3/references/quality-and-speed.md`).
**best = 1344x768 int8, draft = 5 steps.**

## The template run (`templates/minimax/chained-segments`, `shot: static_long`, 120 s)

`bench/workflows/t20-template-static-long.json` composes the catalog
template by its catalog name, with S1's still and prompt, `shot:
static_long`, `duration_seconds: 120` and seed 1166. Everything else is the
template's own default.

| job | jump | jump max | look | look max | level dB | tex fine | tex mid | tex coarse | wall min |
|---|---|---|---|---|---|---|---|---|---|
| `aa80458ab4f7` | 0.722 | 1.44 | 1.499 | 2.67 | 7.05 | 0.121 | 0.015 | 0.007 | **59.2** |

The template path works end to end, inside the expected 57-62 min
(`structural.md`, *Keystone slot*). Fine-texture drift over two minutes is
0.12, against 0.54 for a 26 s standard chain with no keystone. A
step-cache-hit duplicate of this arm is `1f782a579fbb` (0.3 min).

## Host RAM

An int8 H3 worker holds about 63-69 GB of anonymous memory (the kernel's
OOM records, 22:35-23:18 UTC). An int4 one holds about 42-51 GB
(`speed.md`). On lem (125 GB), two int8 jobs on two servers at once
OOM-killed each other seven times: `d9e376286817`, `bc828c7ffe51`,
`9fb01b149f00`, `78440455b4f2`, `6c72b6242412`, `fe9149eb0028` and
`d79a0ff70ccf`. Those arms reran one at a time (`i8e`). One dw.serve
with both cards has a single RAM gate, which waits instead of failing. The two
bench servers each had their own gate. So on a 128 GB two-card box, `best`
runs one job at a time.

## Choices

- **standard stays today's default** on every template: int4, 960x544,
  the template's LoRA and steps. The cost rows stand: `chained-segments`
  13.5 min, `dialogue-short` 21.2, `music-video` 35. The `tier` variable
  joins `cost_drivers` (the observed-cost buckets), and the curated cost
  counts as measured at the default bundle (`docs/WORKFLOW_GUIDE.md`,
  *Presets*).
- **int8 at standard: Don's call, not made here.** On static shots it is the
  largest quality gain measured, on both seeds and at two minutes, at 0-5%
  more wall time. On moving and dialogue shots it is mixed: seam jump gets
  worse. It costs about 40% more host RAM (63-69 GB against 42-51), which
  halves lem's H3 throughput and may not fit a 64 GB host at all. It also
  re-stages the shot. The recommendation, if Don wants it, is int8 on
  `shot: static` and `static_long` only. That needs a rule for a bundle
  value both selectors set: `tier` sets `weights_dtype` too, and today the
  two bundles share no variable. Until then `tier: best` or
  `weights_dtype: "{int8}"` is the knob.
- **draft = 4 steps (FL2VA) / 5 steps (Ref2VA)**, same canvas and weights:
  28-44% off, staging carries.
- **best = 1344x768 int8**: the cleanest static take on both seeds, and the
  sharpest frame on every scene (S2 S3 S4). On the arc it is mixed by
  measure: look improves and jump worsens, the int8 pattern above. It costs
  1.3-1.7x standard's wall time and runs one at a time on a 128 GB host.

## Known issue found on the way

Composing a template with a braced `*_dtype` argument loses the braces:
the wrapper passed `weights_dtype: "{int8}"` to
`templates/minimax/chained-segments`, and the run failed at load with "module
diffusers has no attribute int8" (`1fbbc89cf0eb`) after validation passed.
`EscapedString` (`dw/arguments.py`) does not survive the hop into a child
workflow's variables. The int8 long take ran on the bench copy of S1 instead
(`t20-int8-static-long-120s-s1166`).
