# Temporal flicker probe: does a flow-compensated temporal IIR calm H3 shimmer without smearing motion? (#1747, stage #1749)

mini-ai (Apple M5 Pro, 15 CPU threads), CPU only, 2026-11-26. `bench/temporal_flicker.py` on OpenCV 5.0.0, run
from mini-ai's venv. The numbers are in `temporal-flicker.json` next to this file. *Written by the lead agent,
claude-opus-5-5 via anthropic.*

## Verdict: the numbers show no gain

The plan's gate counts a gain when flicker goes down and the gradient energy on moving edges stays within a few % of the
input. The script fixes those thresholds at a flicker ratio of 0.9 or less and a gradient ratio of 0.95 or more. On
this clip the filter **lowers flicker by 21%, but it costs 26% of the moving gradient energy**, and 14% of it on strong
edges even after the grain is blurred out. That is smear, not just grain removed. **The gate fails on the numbers.**
Don's look at the pair (Q2, `taste`) is still owed. Stage B is built only if both say yes.

## The clip

- **Clip:** `~/diffusers-workspace/outputs/probe-h3-mps-prequantized/20261119-031100-f58d30b1/final/probe-h3-mps-prequantized-text_to_video_audio.0-0.0.mp4`
  on mini-ai (sha256 `2e1ecc63…e160`).
  - MiniMax H3 text-to-video-audio, made on mps on 2026-11-19 from the `minimax/fox_dawn_context_ir` prompt: "a medium
    tracking shot moves alongside a red fox at ground level through a snow-covered pine forest at dawn".
  - 124 frames, 960x544, 24 fps.
- **Why this clip:** it is the only H3 video on mini-ai's disk. Every other video there is LTX or a bench crop, and the
  first-and-last-frame run has an empty `final/`. It was not picked for visible grain crawl, because none of the
  choices could be.
- **The clip's shape matters.** A tracking shot moves the whole frame. Only **0.08%** of pixels (about 420 a frame) have
  input flow under 0.25 px, and **98.9%** move more than 1 px. So:
  - the flicker number is measured on very few pixels;
  - nearly every pixel goes through the warp, which is the filter's hardest case.

  A locked-off H3 shot with grain on a still background would test the filter's best case. None is on mini-ai.

## Settings and time

- **Strengths:** the plan's defaults, luma 0.5, chroma 0.5, `occlusion_tolerance` 1.5 px.
- **Flow:** DIS `PRESET_FAST` at half scale, computed twice per pair (forward and backward check) in each of two passes.
- **CPU time:** **2.5 s wall, 3.8 s CPU** for the filter on 124 frames at **960x544** (RGB, 24 fps), or 0.02 s a frame. The plan listed CPU time as a
  risk; on this hardware it isn't one. The full-resolution flow that the measures use comes on top and isn't counted.

## Before and after

| measure | before | after | after/before |
|---|---|---|---|
| flicker: mean abs dY on the static mask (0-255) | 2.7478 | 2.1713 | **0.790** |
| moving gradient energy (Sobel gx²+gy², flow > 1 px) | 8718.33 | 6458.62 | **0.741** |
| diagnostic: strong moving edges, grain blurred out (σ 1.5, top 10% of input gradient) | 23970.04 | 20673.83 | 0.863 |

### The two regions

Both regions are chosen per pixel and per neighbour pair (t-1, t), from the **input** clip only, and the same masks are
used for before and after. Neither is a hand-drawn box. For each pair, `motion_masks()` runs DIS flow (`PRESET_FAST`) at
**full resolution** on the input's grey frames t and t-1 and takes the flow magnitude per pixel:

- **Static mask:** magnitude **under 0.25 px** (`STATIC_PX`). It covers 0.08% of the pixels, about 420 a frame,
  scattered where the tracking shot's background flow happens to be near zero.
- **Moving mask (the "moving subject"):** magnitude **over 1 px** (`MOVING_PX`). It covers 98.9% of the pixels, so on
  this tracking shot it is the fox *and* the passing forest, not the fox alone. Pixels between 0.25 and 1 px are in
  neither.
- **Strong moving edges** (diagnostic only): moving-mask pixels in the top 10% of the input frame's σ 1.5-blurred
  gradient.

The masks are not written out; `motion_masks()` in `bench/temporal_flicker.py` regenerates them from the clip.

- **Flicker** compares frame-to-frame luma on the pixels that the *input's* full-resolution flow calls static. The same
  mask is used for before and after.
- **Moving gradient energy** is the plan's motion-blur measure. It counts grain as detail, so removing grain from a
  moving area lowers it too.
- **The diagnostic** is extra, outside the gate. It blurs both clips at σ 1.5 to take the grain out, then measures only
  the strongest moving edges. It falls 14%, so the moving edges themselves soften. The forward-backward check at 1.5 px
  doesn't hold back enough of the warped blend on a tracking shot.

The measures are taken on the frames in memory, not on the re-encoded files. A rerun gave the same numbers to every
digit (see "Rerun result" below).

## The pair for Don

All three files are on mini-ai in `~/diffusers-workspace/bench/assets/temporal-flicker-1749/`. That is the `bench`
workspace's assets, so `list_assets` in the `bench` workspace lists them as `temporal-flicker-1749/…`.

- `before.mp4`: the input frames.
- `after.mp4`: the filtered frames.
- `pair.mp4`: before on the left and after on the right, side by side, 1920x544.

Before and after are encoded the same way (x264, crf 12, yuv420p), so the encoder treats both alike. None of them has
audio.

What to look for:
- **Shimmer:** on the snow and the pine needles, does the right half crawl less?
- **Smear:** on the fox's fur and the trunks passing by, is the right half softer, or does it trail?

## Rerun

On mini-ai, from a checkout with this script:

    venv/bin/python bench/temporal_flicker.py run \
      ~/diffusers-workspace/outputs/probe-h3-mps-prequantized/20261119-031100-f58d30b1/final/probe-h3-mps-prequantized-text_to_video_audio.0-0.0.mp4 \
      --out-dir /tmp/temporal-flicker-rerun
    venv/bin/python bench/temporal_flicker.py report bench/records/temporal-flicker.json

The first command prints the same table and writes its own `temporal-flicker.json`; wall and CPU time vary from run to
run, and nothing else should.

### Rerun result

Rerun 2026-11-26 on mini-ai, checkout at `develop` `6c1fb691`, the first command above with `--out-dir
/tmp/temporal-flicker-rerun-1749`:

| measure | first run before / after / ratio | rerun before / after / ratio |
|---|---|---|
| flicker (static mean abs dY) | 2.7478 / 2.1713 / 0.7902 | 2.7478 / 2.1713 / 0.7902 |
| moving gradient energy | 8718.3328 / 6458.6159 / 0.7408 | 8718.3328 / 6458.6159 / 0.7408 |
| strong moving edges (diagnostic) | 23970.0427 / 20673.8321 / 0.8625 | 23970.0427 / 20673.8321 / 0.8625 |
| static / moving share | 0.0008 / 0.9892 | 0.0008 / 0.9892 |
| filter time | 2.5 s wall, 3.8 s CPU | 2.4 s wall, 3.8 s CPU |
| gain | no | no |

Every measure matches to all printed digits; only wall time moved, by 0.1 s.
