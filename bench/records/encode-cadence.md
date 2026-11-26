# Encode cadence probe: do x264's B-frames put a periodic quality dip in dw's H.264 output on crisp content? (#1748, stage #1752)

mini-ai (Apple M5 Pro), CPU only, 2026-11-26. `bench/encode_cadence.py` on PyAV 19.0.2 (libavcodec 63.1.102),
imageio-ffmpeg 0.8.1 (ffmpeg 9.0), diffusers 0.44.0.dev0, OpenCV 5.0.0, from mini-ai's venv. The numbers, with every
frame's PSNR, are in `encode-cadence.json` next to this file. *Written by the lead agent, claude-opus-5-5 via
anthropic.*

## Verdict: not confirmed

Q1's line (#1748, ruled 2026-11-26) is a B-frame PSNR at least **1.0 dB** below the P-frames, or a period-N dip that
deep, on a real crisp clip, on either of dw's two current encode paths. **No source crossed it.** On the crispest real
clip the B-frames are **0.02 dB above** the P-frames on the audio path (0.01 dB above on the silent path). The
strongest period has a depth of **0.05 dB**. The synthetic hard-stroke pan, which the plan expected to overstate the
effect, shows a 0.19 dB gap. The largest gap anywhere is the soft H3 forest clip's silent path, at 0.75 dB.

The threshold doesn't decide stage B or a close-out (Q1). This record goes back to Don either way.

## Reproduce

    venv/bin/python bench/encode_cadence.py run --out-dir /tmp/encode-cadence-1752 \
      --clip ~/diffusers-workspace/outputs/ltx2-turtle-cartoon/20261002-235615-a5532268/final/ltx2-turtle-cartoon-chained_image_to_video.0-0.0.mp4 \
      --clip ~/diffusers-workspace/outputs/probe-h3-mps-prequantized/20261119-031100-f58d30b1/final/probe-h3-mps-prequantized-text_to_video_audio.0-0.0.mp4
    venv/bin/python bench/encode_cadence.py report /tmp/encode-cadence-1752/encode-cadence.json

Run it from the checkout's root on mini-ai. Each encode of the 1439-frame clip took about 19 s with its decode and
measures; the others took about 2 s. Two runs on mini-ai
gave the same tables to the last digit. The synthetic table was also identical on a second Mac (minid).

## The sources

- **synthetic**: the script's deterministic line-art pan, 96 frames at 640x368 and 24 fps. It has flat palette fills
  (circles, polygons) outlined in 3 px black, with loose 1-3 px ink strokes on top, all anti-aliased. The window pans
  3 px right per frame. This is the crisp case the issue describes, with no VAE softness.
- **clip1, the real crisp clip Q1 applies to**: `ltx2-turtle-cartoon/20261002-235615-a5532268/final/ltx2-turtle-cartoon-chained_image_to_video.0-0.0.mp4`
  on mini-ai (sha256 `b76eca1b…9939`). It is LTX-2 chained image-to-video, made on mps on 2026-10-02: a stylized
  turtle on white stone steps with a slow crane. It runs 1439 frames at 960x544 and 24 fps.
  - **Why this clip:** it is the most graphic video on mini-ai's disk, with flat white steps and hard shadow edges.
    There is no FLUX 3 Klein or H3 2D line-art video on mini-ai. A clip with true ink lines would be a harder test.
    The synthetic source covers that case.
- **clip2**: the H3 fox tracking shot from `records/temporal-flicker.md` (sha256 `2e1ecc63…e160`), 124 frames at
  960x544. It is soft, grainy footage and is not crisp. It is a contrast case only.

A real clip is already H.264. Its decoded frames are the source, so the probe measures the loss of **one more encode**.
That is what dw adds when it saves frames a pipeline produced.

## Results

`B gap` = mean PSNR(P) − mean PSNR(B); positive means the B-frames are worse. `period` and `depth`: the strongest
period from 2 to 16 frames in the per-frame PSNR, after a 9-frame running mean is removed, and the spread of its
folded phase means (`period_component` in the script). PSNR is on RGB; SSIM is on BT.601 luma.

**synthetic** (96 frames, 640x368)

| encode | PSNR mean | PSNR min | SSIM mean | SSIM min | P PSNR | B PSNR | B gap dB | period | depth dB | I/P/B | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a audio path | 36.807 | 36.318 | 0.99804 | 0.99773 | 36.899 | 36.708 | 0.191 | 2.0 | 0.098 | 1/46/49 | 66,351 |
| b silent path | 36.568 | 36.223 | 0.99762 | 0.99740 | 36.660 | 36.525 | 0.135 | 2.0 | 0.188 | 1/27/68 | 49,156 |
| c bf=0 | 36.354 | 36.097 | 0.99811 | 0.99740 | 36.346 | — | — | 2.0 | 0.021 | 1/95/0 | 96,121 |
| d crf 18 | 38.366 | 37.756 | 0.99922 | 0.99903 | 38.556 | 38.171 | 0.385 | 2.0 | 0.219 | 1/46/49 | 80,712 |

**clip1, turtle cartoon** (1439 frames, 960x544): the real crisp clip

| encode | PSNR mean | PSNR min | SSIM mean | SSIM min | P PSNR | B PSNR | B gap dB | period | depth dB | I/P/B | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a audio path | 42.664 | 39.686 | 0.98398 | 0.97469 | 42.632 | 42.651 | −0.019 | 2.0 | 0.045 | 6/443/990 | 6,117,956 |
| b silent path | 41.017 | 38.771 | 0.97909 | 0.97003 | 40.994 | 41.008 | −0.014 | 2.0 | 0.053 | 6/442/991 | 4,774,968 |
| c bf=0 | 41.287 | 39.811 | 0.97993 | 0.97300 | 41.270 | — | — | 6.95 | 0.035 | 6/1433/0 | 7,253,558 |
| d crf 18 | 44.400 | 41.760 | 0.98809 | 0.98248 | 44.317 | 44.419 | −0.102 | 2.0 | 0.060 | 6/443/990 | 9,472,812 |

**clip2, H3 fox** (124 frames, 960x544): soft, for contrast

| encode | PSNR mean | PSNR min | SSIM mean | SSIM min | P PSNR | B PSNR | B gap dB | period | depth dB | I/P/B | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a audio path | 36.829 | 33.409 | 0.96734 | 0.94199 | 37.026 | 36.657 | 0.369 | 4.13 | 0.316 | 2/32/90 | 2,275,136 |
| b silent path | 34.710 | 31.749 | 0.94924 | 0.91785 | 35.205 | 34.457 | 0.748 | 4.13 | 0.488 | 2/32/90 | 1,654,286 |
| c bf=0 | 35.820 | 32.873 | 0.95932 | 0.93651 | 35.745 | — | — | 6.2 | 0.342 | 2/122/0 | 2,588,862 |
| d crf 18 | 39.826 | 36.948 | 0.98221 | 0.97243 | 39.720 | 39.807 | −0.087 | 2.03 | 0.047 | 2/32/90 | 3,480,284 |

## What the numbers say beyond the line

- **`bf=0` buys nothing at the same crf.** It makes every file larger (1.14-1.45x) and lowers mean PSNR (0.45-1.38 dB
  below the default). That is because CRF holds quality per frame type, not per bit. Turning B-frames off doesn't
  give their bits to the other frames.
- **crf 18 is the knob that moves quality.** Against the default crf 23 it raises mean PSNR by 1.56, 1.74 and 3.0 dB
  (synthetic, turtle, fox) and minimum PSNR by 1.4-3.5 dB, for 1.22-1.55x the bytes. At crf 18 the B gap goes to
  zero or below on both real clips.
- **The silent path is the weaker of dw's two encoders.** `export_to_video` at imageio's `quality=5` comes in
  **1.65 dB** (turtle) and **2.12 dB** (fox) below the audio path's libx264 defaults, at 0.73-0.78x the bytes. On the
  synthetic source the difference is 0.24 dB. This is a gap between dw's two current outputs, not a cadence. It is
  out of this stage's scope (Q2: no default change).
- **The period figure is not a cadence.** The strongest period is mostly 2.0 frames, at depths of 0.05-0.22 dB. That
  is x264's alternating P/B pattern showing up at noise level, not a visible pulse. The fox's 4.13-frame period
  (0.32-0.49 dB) is its GOP structure on grainy content. It disappears at crf 18.
