# H3 guides in the VRAM estimate (#694)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-08). Plan v1 was designed on #694 and approved with its defaults; the
Stage A measurements forced a plan v2, which Don approved. It was built as
stages #778 (A) and #779 (B).

## The problem

`dw/vram_estimate.py` counted `references` and the voxel variables only. Up to
`GUIDE_LIMIT` (4) H3 guides, each VAE-encoded on the target canvas and spliced
in as condition rows, added memory `required_gb` never counted, and so did the
guide a `continuity: "guide"` chain appends to every segment after the first.
`WORKFLOW_GUIDE.md` already called the limit of 4 "a VRAM limit" that nothing
checked. `validate_workflow`, the run-time backstop and the worker-pool
dispatcher all under-projected a guided step. The one earlier measurement
(`h3-guides-complete.md`) put 4 × 124-frame guides at about 23.5 GiB at 544p,
with none of it projected.

## Verdict: build smaller

The issue asked for terms for guides, `hold_audio` and `refine_strength`. The
plan cut the last two on the code:
- **`hold_audio` adds no rows.** It is cut or padded to the target's own
  `num_audio_latents` and overwrites rows already allocated. Its only marginal
  cost is one audio-VAE encode, freed before denoising.
- **`refine_strength` is not a second pass.** It is its own step restarting
  the schedule at σ = strength, and its `width × height × num_frames` is
  projected like any other step's.

Two more corrections to the issue:
- **The term is per guide voxel, not per guide frame.** Guides are encoded on
  the target canvas, so their cost scales with `width × height` too.
- **The acceptance said "warning", but a declared estimate refuses.** A
  template's own `vram_estimate` is an error at validate, a `ValueError` at run
  and a hard need at dispatch; only an inherited one warns. The guide term
  follows whichever path it sits on.

## Plan v1 → v2

v1 planned two keys, `gb_per_guide` (per clip) and `bytes_per_guide_voxel`,
possibly a third, `gb_per_guide_audio`, and asked Stage A for a fit within
0.5 GB of each measured row. The measurements broke that:
- No two-term additive fit reproduces the rows: the least-squares fit (80.2 B
  plus 0.18 GiB per clip) misses the 768p rows by about 1.5 GiB. The pool grows
  in coarse steps, and the 768p 0-guide pool already holds slack.
- Guide audio has no measurable cost (1 × 39 with audio peaked 0.87 GiB
  *below* the same run without it).
- What the estimate needs is not a peak fit but a separation: the plain
  projection is a ceiling 1–5.6 GiB above the 0-guide peaks. Charging guide
  voxels at the template's own 28.71 B puts both OOMs above 24 GiB, every
  completed run below it, and every uncensored peak at or under its projection.

So v2 has one key, `bytes_per_guide_voxel`, at 28.71, and Stage A's acceptance
became that separation check. Scope stayed at two stages.

## What was built

### Stage A (#778): measure guide memory on lem (docs only)

14 H3 t2va runs on lem's RTX 3090 at 960×544 and 1344×768, all at 124 frames:
0, 1, 2 and 4 guides at 22 frames, 1 and 4 at 124, and 1 × 39 with and without
`"audio": true`. The record, with job ids, the table, the least-squares fit and
its residuals, the separation check and the exit verdict, stays where the code
cites it: [`../h3-guide-vram.md`](../h3-guide-vram.md) (merged at `acde06e9`).

The separating range on this data is 27.7–35.9 B per guide voxel; 28.71 sits at
its bottom, leaving the most room before over-refusal. Exit verdict: not met.
The term adds 13.7 GiB at 4 × 124 on 1344×768, so Stage B was needed.

Deviations from v1's measurement plan, both written into v2:
- **124 frames, not near-top lengths.** At the near-top lengths (345 at 544,
  226 at 768) even the 0-guide runs filled the card, so no guide delta could be
  read. Those runs are listed as superseded.
- **`nvidia-smi memory.used`, not `max_memory_reserved`.** The engine records
  no running peak. `memory.used` can't exceed 23.55 GiB, so two 768 rows
  (4 × 22, 1 × 124) are censored: they filled the card and completed only by
  thrashing the allocator (25 and 33 minutes of phase 2, against 7).

### Stage B (#779): the guide term through validate, run and dispatch

- **Schema:** optional `vram_estimate.bytes_per_guide_voxel` (number ≥ 0) in
  `dw/workflow_schema.json`. Absent adds nothing, so every existing projection
  is unchanged.
- **Engine:** `required_gb(estimate, values, references=0, guides=())` adds
  `bytes_per_guide_voxel × Σ snapped guide frames × canvas / 2^30`, the canvas
  being every voxel variable but `num_frames`. `_projections` counts the step's
  non-null `guides` plus one guide of `guide_frames` (22 or 39) when
  `chain.continuity == "guide"` and `segments != 1`; the counting is
  `dw/guides.py: guide_lengths`.
- **Probe at all three call sites:** validate (`ValidationContext.probe`),
  admission (`_vram_need`, the context's probe) and run (`apply_vram_estimate`
  with a header-only `probe_metadata`), so all three answer the same number.
  A clip no probe can count (a `previous_result:`, an unreadable header) is
  charged at the step's `num_frames`, and the message says "worst case".
- **Message:** names the guides, e.g. `… with 3 guides (22+124+22 frames; 1 not
  probeable before the run, so charged at num_frames - the worst case) projects
  to …`; the `Declared ceiling:` formula names the guide term.
- **Templates:** `bytes_per_guide_voxel: 28.71` on `video-with-audio`,
  `video-with-audio-768p`, `shots-batch` and `enhance-prompt`; ref2va templates
  unchanged (guides are refused there). `minimax/chained-segments.json` gained a
  `cost` entry (RTX 3090, 10.0 min, from job `d0bba0358b70`) and a
  `vram_estimate` (16.3 / 28.71 / 28.71).
- **Tests:** `tests/test_vram_guides.py` (21 tests), `_ESTIMATE_FIELDS` gains
  the key, and the every-template-validates-clean test covers chained-segments.
- **Docs:** `WORKFLOW_GUIDE.md` (the formula paragraph and the guide `count`
  row), the ARCHITECTURE *VRAM projection* row, and one clause on the
  `minimax-h3` skill's guides line.
- **Deploy:** `develop` @ `bc0b19b5` on lem.

## Deviations from the plan

- **`chained-segments` is an `fl2va` identity, not t2va.** No other fl2va
  template declares an estimate, so it became the *inherited* ceiling for
  hand-built fl2va workflows, which now get `vram_projection_inherited`
  warnings where they got none before. Its numbers are the H3 t2va ones.
- **`enhance-prompt`** also carries the key: it has an H3 t2va step with its
  own estimate, and `_ESTIMATE_FIELDS` requires one identity's templates to agree.
- **fl2va templates other than chained-segments** (`image-to-video`,
  `first-and-last-frame`, `last-frame-only`) still declare no estimate, so the
  plan's "every t2va/fl2va template" is in practice the t2va ones plus
  chained-segments.

## Deferred

- **`hold_audio` and `refine_strength` terms.** Cut on the code evidence.
  - **Reopen trigger:** a hold or refine run that OOMs while its plain
    projection passed.
- **A per-clip or guide-audio term.** Nothing was measured; the dropped keys
  are refused as unknown.
- **Calibration** rests on 14 points on one box. The thinnest margin is
  768 2 × 22 (projection 20.94 against a 20.89 GiB pool peak), and both
  censored 768 rows peaked above their projections: the term puts them on the
  right side of 24 GiB but doesn't bound their peaks. Bracketing runs between
  the completed and OOM rows would sharpen it.
  - **Reopen trigger:** a guided run that OOMs under a clean projection, or a
    template refusal of a guided run shown to fit.
- **The run-time asset-path seam.** At run time an `asset:` guide passed
  through a variable is an absolute path; if the probe-path check refused it,
  run would charge worst case where validate didn't. Both go through
  `resolve_probe_path`, and the tester saw agreement, but that is the seam to
  watch.
- **MPS:** the numbers stay CUDA-measured.

## Stages, cost and bounces

| stage | shipped | bounces |
|---|---|---|
| A #778 | `develop` @ `acde06e9` (docs only) | 1 re-plan stop (the measurements broke v1's 0.5 GB fit; plan v2) and 1 tester bounce: the record wasn't readable over MCP, so the re-hand-off quoted it verbatim and added the thin-margin and censored-row caveats |
| B #779 | `develop` @ `bc0b19b5` (commit `65d9d4d5`) | 0 (M-F118 to M-F125 passed over five verify sessions) |

The stage comments name no `usage:` figures, so cost is not recorded. The plan
estimated $7 to $11.

Acceptance cases: M-F117 to M-F125 (plan v2) in
`regression-suite-model-specific.md`. The v1 cases M-F104 to M-F112 assert the
dropped `gb_per_guide` and the 0.5 GB fit; dkackman/harnest#92 asks the
curator to retire them.
