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

# S5: the keystone in every window (#1691)

Feature #1188, plan v3, stage S5. This re-benches the same take with S4's `anchor: "keystone"`: window 1 is finished first, and its first-frame latent is laid into every window's forward as a condition row. S3's a and b stand as the unanchored references and were not re-run. Arm c is dropped.

Bench by lead `claude-opus-5-5` via `anthropic`, 2026-11-27.

**Verdict: arm a+k passes the measured half of the gate at both seeds. b+k fails it.**
- **a+k (blended, keystone).** Its `face_drop` max is 0.028 and 0.038, under 0.075, so it passes on the numbers. It shows no burn: sharpness and saturation are flat for the whole minute. S3's arm a, the same run without the keystone, measured 0.172 and 0.160.
- **b+k (ordered, k = 3, `finished`, keystone).** Its `face_drop` max is 0.105 and 0.111, over 0.10, so it fails. It also still burns from about 40 s, as b did in S3. The keystone holds the face better through the middle of the take, but the ordered phase still over-draws it late.
- **What is left for a+k is Don's eye:** whether she walks, and whether the scene changes at the seams. The stills put the same building behind her at 59.5 s as at 0.5 s, as S3's arm a did, so the keystone may be holding the background, and perhaps the pose, in place. Stills can't settle that. The clip can.

## Setup

- The workflow is `bench/workflows/hybrid-windows.json`. It gains an `anchor` variable, default `"none"`, so S3's arms run as before. The arms are in `bench/arms/hybrid-windows-keystone.json`.
- Everything else is S3's: the take, 1450 frames, the windows `{42, 12, pyramid, rebased}`, turbo at 5 steps, the held narration, and seeds 1166 and 2.

| Arm | `switch_step` | `handoff` | `anchor` |
|---|---|---|---|
| a+k | 0 | - | `keystone` |
| b+k | 3 | `finished` | `keystone` |

- **Server:** the bench server on lem (`DW_EXPERIMENTAL_HYBRID_WINDOWS=1`, `BENCH_DEVICES=cuda:0,cuda:1`). The worktree was at 84e5cd80 (`develop` ae89e852 plus the arms), and the main server was stopped (Don, 2026-11-27).
- **Jobs:** `be019d8db81b` (a+k s1166), `f887380adc1f` (a+k s2), `56ab4e3c03f4` (b+k s1166), `bef46c561049` (b+k s2).

## Measures

**`face_drop`** (`bench/face_drop_spans.py`, the saved files, 7.2 s spans). Re-measuring S3's a s1166 export gives 0.172 again, so the numbers below compare directly with S3's table.

| Take | span medians (0-60 s) | `face_drop` max |
|---|---|---|
| a+k, s1166 | .932 .928 .904 .919 .933 .921 .938 .940 .914 | **0.028** |
| a+k, s2 | .921 .918 .883 .909 .924 .930 .921 .911 .915 | **0.038** |
| b+k, s1166 | .909 .915 .804 .859 .882 .869 .892 .843 .835 | 0.105 |
| b+k, s2 | .887 .914 .776 .885 .880 .857 .863 .845 .828 | 0.111 |
| *(S3) a, s1166* | .893 .775 .746 .741 .723 .720 .765 .765 .793 | 0.172 |
| *(S3) b, s1166* | .908 .841 .708 .806 .776 .764 .678 .669 .641 | 0.267 |

- **a+k holds within 0.04 of its first span for the whole minute.** It doesn't take S3's 0.12 drop in span 2, and it doesn't slide either. The first span's median is also higher than a's (.93 against .89).
- **Both of b+k's maxima sit in span 3** (14.4-21.6 s), the first windows after the finished window 1, and recover afterwards. With the keystone, b's steady slide to 0.64 becomes a dip and a slow fall to 0.83. That is still over 0.10 at both seeds.

**10 s windows** (`bench/ks2v_gate.py --crop256 --windows`; box crop, every sixth frame). Each cell gives 256 px Laplacian / saturation / SFace to the 0.5 s face.

| Take | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 |
|---|---|---|---|---|---|---|
| a+k s1166 | 71 / 132.8 / .90 | 81 / 134.3 / .89 | 74 / 134.4 / .89 | 68 / 132.3 / .90 | 86 / 130.2 / .90 | 76 / 131.2 / .89 |
| a+k s2 | 68 / 140.8 / .91 | 72 / 143.7 / .90 | 76 / 143.1 / .90 | 72 / 141.7 / .90 | 74 / 139.8 / .90 | 71 / 140.9 / .91 |
| b+k s1166 | 46 / 127.7 / .90 | 80 / 133.9 / .84 | 83 / 139.5 / .85 | 90 / 137.5 / .83 | 118 / 140.3 / .84 | 148 / 144.0 / .78 |
| b+k s2 | 55 / 138.9 / .89 | 64 / 142.1 / .81 | 73 / 149.0 / .82 | 87 / 148.9 / .81 | 112 / 146.8 / .80 | 131 / 148.2 / .78 |

- **Burn.**
  - a+k's sharpness is flat (68-86) and so is its saturation. No burn.
  - b+k climbs 2.4x to 3.2x from the first window to the last, S3's b shape. The frame sheets show the same over-drawn creases and distorted mouth from 40 s.
  - b+k's saturation also rises about 10 over the take, at both seeds. a+k's doesn't.
- **a+k's absolute sharpness** (about 75) is higher than S3's a (about 50), but it doesn't climb within the take. It reads as a crisper face, not a burning one, but the gloss is for Don's eye: the measures have missed it before (#1324).
- **Still-mouth bins:** 0 of 120 in every take.

**Picture.** The sheets are full frames at 0.5/20/40/50/59.5 s, in rows a (S3), a+k, b+k and b (S3).
- **a+k** keeps the same woman, the same building and the same path for the minute. Benches and a lamp post appear at the frame edges later. Like S3's a, the stills can't tell whether she walks. The mean luma difference of each sampled frame from 0.5 s is 25-52 for a+k against 38-50 for a: a little closer to the first frame. That fits a keystone holding the background. Whether it also holds her in place is the question for Don's eye.
- **b+k** moves like S3's b: the building recedes, and trees, lamps and benches pass. The burn arrives with the movement.

**Cost.**

| Arm | s1166 wall | s2 wall |
|---|---|---|
| a+k | 27.6 min (first job, includes load) | 20.8 min |
| b+k | 23.2 min | 20.7 min |

- a+k's pre-pass (window 1 alone, all steps) adds about 1 min against S3's a (19.8 min at s2). b+k matches S3's b to within half a minute, as the plan expected: its pre-pass is window 1.
- The four jobs ran one at a time, as in S3, held by the host-RAM gate. The whole bench took about 92 min.

## Exports

These are on lem in `~/bench-workspace/exports/hybrid-windows/`, beside S3's:
- `ak-blended-key-s1166.mp4`, `ak-blended-key-s2.mp4`;
- `bk-finished-k3-key-s1166.mp4`, `bk-finished-k3-key-s2.mp4`;
- `sheet-keystone-full-s1166.jpg`, `sheet-keystone-full-s2.jpg` (rows a, a+k, b+k, b; full frames);
- `sheet-keystone.jpg` (face crops from `ks2v_gate.py`).

To compare a with a+k, watch `a-blended-s1166.mp4` beside `ak-blended-key-s1166.mp4`.

## What it settles

- **The keystone closes arm a's early drift:** 0.172 / 0.160 becomes 0.028 / 0.038, with no burn. On the measures, a+k is the first windowed take in this repo to hold the face for 60 s.
- **The keystone doesn't fix the ordered phase's late burn.** b+k still burns and fails the drift tier. Per the plan, the ordered phase is the suspect there, and it is cut from any promotion.
- **a+k's pass now rests on Don's eye (Q4 again):** she walks, and there is no scene change at the seams. If she walks, the next plan version is a promotion of a+k. If the keystone has pinned her in place, it is a close-out that names #1645.
