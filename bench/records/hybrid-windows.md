# Hybrid windows: the 60 s narrated take in one generation (#1571)

Feature #1188, plan v1, stage S3. The question is whether one `experimental_hybrid_windows` generation can hold the narrated take's face for 60 s as well as the v4-keyframes chain does. The chain arm is a standing speaker on a lateral camera arc: only the backdrop moves. The gate for an arm: SFace drift ≤ the chain arm's to 60 s, no burn, and Don sees her walk (Q4, his eye).

*Note, 2026-11-23 (#1596):* Don watched the chain reference (job ddbac3a46144) for #1571 Q4 and she does not walk. Its 0.070 face drift held because the pose is pinned, not because she moved. The gate stands as Don ruled it; #1188 plan v2 has already decoupled it from the 0.070.

**Verdict: no arm passes the measured half of the gate.**
- Arm a (blended only) doesn't burn, but its face drifts more than twice as far as the chain's: 0.172 and 0.160 against 0.070.
- Arms b and c (ordered phase, k = 3) burn from about 40 s at both seeds. The face crop goes crunchy and distorted, and its 256 px Laplacian rises about 3x. They drift further still: 0.174 to 0.317.
- Whether she walks in any arm is Don's call on the exports (Q4).

## Setup

- **Workflow:** `bench/workflows/hybrid-windows.json`; arms in `bench/arms/hybrid-windows.json`.
- **Generation:** H3 t2va at 960x544. Turbo LoRA, 5 steps, int8 weights, sparse attention, video shift 6, audio shift 3.
- **Audio:** the constitution narration's first 60.42 s, held (`hold_audio` from `slice_audio`).
- **Prompt:** the narrated take's scene, speaker and soundscape, with the Preamble and Amendments One to Three as dialogue lines.
- **Windows:** `{frames: 42, overlap: 12, blend: pyramid, positions: rebased}`.
- **Length:** 1450 frames (60.42 s, 427 latents). Rebased windows must start on the 5-latent clip grid. 1441 frames puts the last window at latent 319, so 1450 (17m+5) is the nearest valid length to 60 s.

| Arm | `switch_step` | `handoff` |
|---|---|---|
| a | 0 (blended on every step) | - |
| b | 3 | `finished` |
| c | 3 | `x0` |

Each arm ran at seeds 1166 and 2.
- **Server:** the bench server on lem (`DW_EXPERIMENTAL_HYBRID_WINDOWS=1`, `BENCH_DEVICES=cuda:0,cuda:1`), worktree at bbde7947, with the main server stopped.
- **Chain arm:** job ddbac3a46144, `BenchNarratedTake-narrated-arc4` run `20261112-183800-fcda7b21` (`records/narrated.md`). It is 12 segments, 85.9 s, measured to 60 s.

**What differs from the chain (read the comparison with these in mind).**
- The chain arm is 1344x768 at 20 steps without turbo, at seed 1 only.
- Every chain segment is re-anchored to a keyframe still. The hybrid key refuses images, keyframes and references, so these takes have no picture anchor at all.
- Part of the drift gap may come from the missing anchor rather than from windowing. That gap is what S4 (anchors in every window) exists to close.

## Measures

**`face_drop`, as the chain check computes it** (`bench/face_drop_spans.py`). The take is cut into 7.2 s spans, the chain's segment length. Each span's score is the median of 8 sampled frames' SFace similarity to frame 0, and the drop is span 1's median minus span k's. All values are measured on the saved files: the chain's in-job 0.075 isn't comparable with a file score (#1420), so the chain is re-measured here the same way.

| Take | span medians (0-60 s) | `face_drop` max |
|---|---|---|
| chain (v4 keyframes) | .924 .901 .884 .886 .855 .893 .898 .904 .888 | **0.070** |
| a, s1166 | .893 .775 .746 .741 .723 .720 .765 .765 .793 | 0.172 |
| a, s2 | .854 .725 .716 .749 .746 .709 .745 .693 .713 | 0.160 |
| b, s1166 | .908 .841 .708 .806 .776 .764 .678 .669 .641 | 0.267 |
| b, s2 | .887 .880 .810 .813 .800 .755 .748 .747 .713 | 0.174 |
| c, s1166 | .908 .857 .764 .779 .761 .738 .712 .667 .591 | 0.317 |
| c, s2 | .897 .889 .817 .810 .827 .751 .772 .705 .698 | 0.199 |

- Arm a drops at once, by 0.12-0.13 in the second span, then holds roughly level.
- Arms b and c hold the first span or two (the first window is finished whole) and then fall steadily to the end.

**10 s windows** (`bench/ks2v_gate.py --crop256 --windows`; box crop, every sixth frame). Each cell gives 256 px Laplacian / saturation / SFace to the 0.5 s face.

| Take | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 |
|---|---|---|---|---|---|---|
| chain | 469 / 103.7 / .94 | 438 / 106.6 / .93 | 408 / 109.5 / .90 | 431 / 107.9 / .91 | 467 / 106.2 / .91 | 526 / 108.8 / .92 |
| a s1166 | 53 / 128.3 / .85 | 52 / 128.0 / .78 | 71 / 130.8 / .76 | 40 / 132.7 / .76 | 53 / 134.9 / .76 | 42 / 130.6 / .78 |
| a s2 | 50 / 134.5 / .82 | 53 / 127.4 / .78 | 62 / 133.5 / .78 | 52 / 135.7 / .74 | 60 / 130.6 / .73 | 42 / 133.2 / .70 |
| b s1166 | 46 / 126.9 / .90 | 74 / 132.5 / .73 | 82 / 135.2 / .76 | 97 / 132.0 / .76 | 132 / 131.0 / .66 | 146 / 133.2 / .63 |
| b s2 | 55 / 138.2 / .88 | 57 / 140.3 / .84 | 66 / 144.9 / .80 | 74 / 142.6 / .76 | 102 / 139.7 / .71 | 124 / 144.6 / .68 |
| c s1166 | 47 / 126.8 / .90 | 67 / 132.5 / .78 | 76 / 136.8 / .75 | 87 / 132.2 / .71 | 118 / 130.4 / .67 | 159 / 130.3 / .59 |
| c s2 | 55 / 138.0 / .89 | 60 / 139.2 / .82 | 72 / 143.2 / .80 | 87 / 139.7 / .78 | 95 / 135.7 / .71 | 132 / 138.6 / .67 |

- **Sharpness.** The chain's absolute Laplacian isn't comparable: it is a different resolution and step count, and turbo is soft. What carries the signal is the trend within each take.
  - Arm a is flat (40-71).
  - Arms b and c climb 2.4-3x from the first window to the last. The frame sheets show this as the burn: deep, over-drawn creases, and mouth and teeth distorting by 50 s.
- **Saturation.** It sits 20-35 above the chain from the first window, which is the turbo config's look. It doesn't climb within a take.
- **Still-mouth bins:** 0 of 120 in every take, so the held narration keeps her mouth moving throughout.

**Picture (frame sheets at 0.5/20/40/50/59.5 s).**
- Arm a keeps the same woman in front of the same building for the whole minute. Background props shift, and expression and pose vary.
- Arms b and c move: the building recedes and benches and lamp posts pass. That reads like a walk, or a camera dolly, or a scene re-drawn per window, and stills can't tell which. The burn arrives with it.

**Cost.**

| Arm | s1166 wall | s2 wall | peak committed host RAM (job) |
|---|---|---|---|
| a | 24.6 min (first job, includes load) | 19.8 min | 68.4 / 70.3 GB |
| b | 22.3 min | 20.2 min | 68.7 / 70.3 GB |
| c | 21.7 min | 19.4 min | 67.5 / 69.2 GB |

- **Host RAM.** The highest system-wide used RAM, sampled every 20 s across the whole run, was 78.3 GB of lem's 125.7 GB, with both bench workers resident and the main server stopped. That leaves about 47 GB of headroom. The plan cites #1325 for the worker's host-RAM hold, but #1325 is the closed seam-restore idea and names no limit, so the figure here is against the host's total.
- **Concurrency.** The six jobs ran one at a time even with two workers up: the dispatcher's host-RAM gate held the second job back. So a 60 s take costs about 20 min of one card at this config.

## Exports

These are on lem in `~/bench-workspace/exports/hybrid-windows/`:
- `a-blended-s1166.mp4`, `a-blended-s2.mp4`;
- `b-finished-k3-s1166.mp4`, `b-finished-k3-s2.mp4`;
- `c-x0-k3-s1166.mp4`, `c-x0-k3-s2.mp4`;
- `sheet-s1166.jpg`, `sheet-s2.jpg` (rows a, b, c; columns 0.5/20/40/50/59.5 s).

## What it settles

- **The ordered phase as built (k = 3 of 5, either handoff) fails:** it burns late at both seeds.
- **Blended-only windows hold the look but not the face.** The identity loss happens within the first 15 s and doesn't compound.
- Per the plan, S4 (anchors in every window) is built only if an arm *walks* but burns or drifts late. Arms b and c are that case if Don sees a walk in them. If he sees one only in arm a, its drift is the early, flat kind that an anchor also targets. That ruling, and so whether S4 is built, is Don's (Q4).
