# H3 guide memory on a 3090 (#694 stage A, #778)

_Measured by `claude-opus-5-5` (anthropic) on lem's RTX 3090 (24 GB), 2026-10-08, `develop` at c0d188e1._

These are measurements only, with no code. They feed the guide term in #694 stage B.

## Setup

- **Workflow:** an H3 t2va step from the catalog template: group-offloaded transformer, int4 SDNQ, the Turbo LoRA and 8 steps, seed 42. Every run is **124 frames**. Workflows are `w778/f124-<canvas>-<n>x<len>.json` in lem's workspace.
- **Canvases:** 960×544 ("544") and 1344×768 ("768").
- **Guides:**
  - 22 and 39 frames: `output:GuideClips778/20261008-163337-a0e3501b/final/g22-0.0.mp4` and `g39-0.0.mp4`.
  - 124 frames: `asset:qa-guides/vwa-seed42-pre648.mp4`.
  - Placement: 2×22 at frames 0 and 34; 4×22 at frames 0, 34, 68 and 102; 4×124 all at frame 0.
- **Peak:** `nvidia-smi --query-gpu=timestamp,index,memory.used -lms 250`, windowed to each job's card and run time. This peak is the **denoise-and-decode phase**: the maximum after the card first falls back to about 2 GiB.
  - Each window opens with a guide-independent burst of 17.5–21 GiB that lasts about 60 s. The torch pool is still 40 MB at that point (the job's own `memory` event), so the burst isn't the step's allocation, and it is excluded.
  - The job events place the counted peak inside denoising. In 768-0x0 the pool reaches 17.0 GiB at the first denoise step and 18.6 GiB by step 4, and it stays there through decode.

## Deviations from the plan

1. **The metric is `memory.used`, not `max_memory_reserved`.** The engine records only per-phase `gpu_memory_reserved_mb` snapshots, with no running peak (nothing in `dw/` reads `max_memory_*`). `memory.used` is the reserved pool plus the CUDA context, about 0.7–1 GiB. It can't go above the card's 23.55 GiB, so two 768 rows are **censored**: they filled the card and completed only by thrashing the allocator. Phase 2 took 25 and 33 minutes, against 7 minutes for 768-0x0.
2. **Every run is 124 frames, not "near the top the plain projection allows".** At the near-top lengths (345 frames at 544, 226 at 768), even the 0-guide runs filled the card, so no guide delta could be read. Those runs were superseded and are listed at the end.

## Table

The peak is GiB of `memory.used` during denoise and decode. The guide voxels are Σ(guide frames) × W × H. The plain projection is the template's current `16.3 + 28.71 B × W·H·124`.

| run | job id | guides | guide voxels (M) | peak | result |
|---|---|---|---|---|---|
| 544 0 | 06c6a3efeab2 | — | 0 | 12.40 | ok |
| 544 1×22 | 1bb0a27d0080 | 22 | 11.5 | 13.02 | ok |
| 544 2×22 | 47b8a7edee48 | 22+22 | 23.0 | 14.93 | ok |
| 544 4×22 | 06b9afe51053 | 4×22 | 46.0 | 17.17 | ok |
| 544 1×124 | cf3d8c007835 | 124 | 64.8 | 18.24 | ok |
| 544 4×124 | 61a52b780209 | 4×124 | 259.0 | **OOM** | first denoise step: 18.02 GiB allocated plus a 5.08 GiB request, so it needs ≥ 23.1 |
| 544 1×39 | 97c3cee833dc | 39 | 20.4 | 14.86 | ok |
| 544 1×39 audio | 59fb23b7b803 | 39, `"audio": true` | 20.4 | 13.99 | ok |
| 768 0 | a91a3067b3e8 | — | 0 | 18.57 | ok |
| 768 1×22 | 6790607d855d | 22 | 22.7 | 18.83 | ok |
| 768 2×22 | 2f0d88c1532f | 22+22 | 45.4 | 20.89 | ok |
| 768 4×22 | fccc4e1c50c6 | 4×22 | 90.8 | ≥ 23.55 (card full) | ok, but phase 2 took 25 min |
| 768 1×124 | 78b98d536505 | 124 | 128.0 | ≥ 23.55 (card full) | ok, but phase 2 took 33 min |
| 768 4×124 | 24efcc4da549 | 4×124 | 512.0 | **OOM** | rotary in attention: 21.10 GiB allocated plus a 1.88 GiB request, so it needs ≥ 23.0 |

**Guide audio:** 1×39 with audio peaked 0.87 GiB *below* the same run without it. Guide audio has no measurable cost at this resolution, so it gets no key of its own (the plan's `gb_per_guide_audio` isn't needed).

## Fit

### Least squares on the delta over each canvas's 0-guide row (the plan's form)

`Δpeak = bytes_per_guide_voxel × guide_voxels + gb_per_guide × guides`, on the 7 uncensored guided rows:

- `bytes_per_guide_voxel` = 80.2, `gb_per_guide` = 0.18
- residuals (GiB):

  | row | residual |
  |---|---|
  | 544 1×22 | −0.42 |
  | 544 2×22 | +0.45 |
  | 544 4×22 | +0.60 |
  | 544 1×124 | +0.82 |
  | 544 1×39 | +0.76 |
  | 768 1×22 | −1.62 |
  | 768 2×22 | −1.43 |

**This fit does not reproduce the rows within 0.5 GB,** and no two-term additive form will. The pool grows in coarse steps: for example 768 1×22 adds 0.26 GiB over 768 0, while 544 1×22 adds 0.62. The same voxels cost much less at 768, whose 0-guide pool already holds the slack.

The fit also refuses runs that completed:
- It predicts 26.1 GiB for 768 4×22 and 28.3 GiB for 768 1×124, yet both completed.
- A voxel-only fit gives 88.9 B, with the same 768 misses.

### What the estimate actually needs: guide voxels at the template's own `bytes_per_voxel`

The estimate isn't a peak fit. Its plain projection is a ceiling: 5.6 GiB above the measured 544 peak and 1.15 GiB above the 768 peak. The useful question is therefore whether a guide term on top of that ceiling separates the runs that fit the card from those that don't.

Take `required = plain projection + 28.71 B × guide_voxels`: every guide voxel costs what a target voxel costs, and `gb_per_guide` is 0. On this table:

| run | projected | measured | margin | > 24 (flagged)? |
|---|---|---|---|---|
| 544 0 | 18.03 | 12.40 | +5.63 | no |
| 544 1×22 | 18.34 | 13.02 | +5.32 | no |
| 544 2×22 | 18.65 | 14.93 | +3.72 | no |
| 544 4×22 | 19.26 | 17.17 | +2.09 | no |
| 544 1×124 | 19.76 | 18.24 | +1.52 | no |
| 544 1×39 | 18.58 | 14.86 | +3.72 | no |
| 544 4×124 | **24.96** | OOM | — | **yes** |
| 768 0 | 19.72 | 18.57 | +1.15 | no |
| 768 1×22 | 20.33 | 18.83 | +1.50 | no |
| 768 2×22 | 20.94 | 20.89 | +0.05 | no |
| 768 4×22 | 22.15 | ≥ 23.55 (card full) | — | no (it completed) |
| 768 1×124 | 23.14 | ≥ 23.55 (card full) | — | no (it completed) |
| 768 4×124 | **33.41** | OOM | — | **yes** |

- Every uncensored row projects at or above its measured peak.
- Both OOMs project above 24 GiB.
- Every run that completed projects below 24 GiB.

This one number separates the two outcomes at every measured point. It is also the plan's own prior: about 25 B per guide voxel, "close to the 28.71 the plain term already uses".

The range that separates them on this data is 27.7–35.9 B per guide voxel:
- below 27.7, 768 2×22 projects under its measured peak;
- above 35.9, 768 1×124 is refused although it completed (above 50.6, 768 4×22 is too; above 99, 544 1×124 is too).

28.71 sits at the bottom of that range, which leaves the most room before over-refusal. Its 0.05 GiB margin on 768 2×22 is thin, but it is set against a pool peak, not an allocation.

## Exit

At 4 × 124 frames on the largest canvas, the guide term adds 13.7 GiB at 28.71 B, and 38.9 GiB at the least-squares fit (80.2 B plus 0.18 GiB per guide). Either is far more than 0.5 GB, so **the feature continues to stage B.**

## Superseded runs (near-top lengths, not used in the fit)

- Completed: 8c9e1e56946b, f359659af72d, 1a038f33ddb8, 10945d493503, f861562f9d65.
- Cancelled after the card proved saturated with no guides: c99be8c4b6c5, 99d668557254, c6aadbf84e99, 2e3ff4378dd5, dc5ee798b5e3, 5cd28af27fc6, c82054f77522, 8757baef94f6, 064e2d5af70c.
