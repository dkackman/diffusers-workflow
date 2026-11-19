# Per-shot audio drift figures (#1442)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18 from Adudeguyman/ComfyUI-H3-Project-Suite
1.5.0 (GPL-3.0, concepts only, README and `docs/HOW_IT_WORKS.md` read at 015db2c, no
code). It had two parts. **A:** audio drift numbers per chain segment (the high end and
the hiss under quiet moments). **B:** clip 1's sound passed to every later clip as an
audio anchor. Plan v1 is the plan comment on #1442, and it still holds the full sweep.
This file records what was built, what was measured, and what stayed cut.

## Verdict: build smaller, A only, report-only

- Most of A had already shipped. `analyze_shots` already reported per-shot loudness,
  peak, crest and the low/mid/high band levels (high is 4-20 kHz).
- What was missing was a per-shot high-band **share** and a measured **noise floor**.
- B was deferred. Only ref2va chains can carry an audio reference. `chained-segments`
  (fl2va `guide`, H3-invented sound) is the template most likely to drift, and B can't
  reach it.

## Plan history

- **v1 (approved 2026-11-18, batch lane, Q1-Q3 at defaults).**
  - Q1 [none]: the field names `high_share` and `noise_floor_dbfs`.
  - Q2 [none]: the floor is the 10th percentile of 50 ms RMS windows, `null` under
    0.5 s.
  - Q3 [none]: B stays deferred under its trigger. The close-out files B as a new
    `idea` if stage 1's record meets the trigger, and records "no defect found" if it
    doesn't.

## What was built

### Stage 1 (#1464): `high_share` and `noise_floor_dbfs` per shot (server)

Merges on `develop`: 0e540407 (5e48bf8d) and a8b8f6af (ddf562b4, the
architecture-review fix).

- **`high_share`** (`dw/tasks/assess.py`, `shots_answer`). The high band's share of
  the shot's own `spectral_balance`, via `_shares_from_bands`, which is factored out of
  `_band_shares`, so the seam probe's `spectral_shift` is unchanged. It is rounded to
  4 places, and `null` where the balance is `null`.
- **`noise_floor_dbfs`** (`_noise_floor`). The nearest-rank 10th percentile of the
  shot's 50 ms window levels.
  - `null` under 10 windows (0.5 s), or when the percentile lands on a digitally
    silent window. Never NaN.
  - Capped at the shot's `rms_dbfs`. This was a deviation: on a steady level, rounding
    could otherwise put the floor about 0.01 dB above it.
- **One window helper**, `_window_levels`, shared by `_dead_air` and `_noise_floor`.
  `_dead_air` answers are pinned unchanged.
- **Tests:** `TestHighShare`, `TestNoiseFloor`, `TestDeadAirSharesTheWindows` and the
  steady-tone cap test in `tests/test_assess.py`.
- **Docs:**
  - Two rows in the `analyze_shots` table, `docs/TASKS.md`, with the caveat that in
    dense dialogue the floor may follow speech tails.
  - The "rising together" sentence went in
    `plugins/dw/skills/minimax-h3/references/chains.md` (*Drift*), not `SKILL.md`.
    That was a deviation: `SKILL.md` was at its 12288-byte cap.
- **Unchanged:** no rule, probe, finding code, MCP parameter, REST route or syntax.

**Bounces (1):** the architecture review bounced the first hand-off because
`dw/tasks/assess.py` had grown past the module-size ceiling (1,200 lines). The fix
moved code out (`dw/dsp.py`, `dw/tasks/image_ops.py`) and brought it to 1,091 lines.
The tester then passed C-F1002 to C-F1010 on the first verify, against develop @
2ce2776c.

Stage cost isn't recorded, because the stage comments don't name `usage:` figures. It
used no GPU time beyond the verify's `assess_output` calls.

## What was measured

The bench S3 dialogue chain (`bench` `s3-dialogue-walk/20261107-204402-f54d3126`: 6
segments of generated dialogue on a guide chain). Shots come from the run manifest. Two
calls gave identical values.

| Segment | Frames | rms_dbfs | high_share | noise_floor_dbfs |
|---|---|---|---|---|
| 1 | 0–124 | -26.59 | 0.0289 | -69.58 |
| 2 | 124–226 | -17.62 | 0.0539 | -78.67 |
| 3 | 226–328 | -12.59 | 0.0440 | -61.16 |
| 4 | 328–430 | -11.19 | 0.0633 | -22.87 |
| 5 | 430–532 | -14.79 | 0.0481 | -51.10 |
| 6 | 532–634 | -15.35 | 0.0910 | -59.20 |

- `high_share` roughly triples from segment 1 to 6, but not monotonically.
- The floor doesn't rise with it. The exception is segment 4, which is dense and
  continuous, so its floor sits only 11.7 dB under its level.
- Under the skill's rule (both figures rising together), **this chain doesn't show the
  dulling-voice signature: no defect found.**
- **Not taken:** the S1 table and the rerun gap (a second run of the same scene). The
  verify's acceptance named only S3. The trigger fails on S3 even before a rerun gap,
  so neither was needed for the decision.
- The trend is seeded as `regression-perf/C-F1010.jsonl` on the harness repo.

## Cut, and what brings each back

- **Idea B, the audio anchor.** It isn't filed, because its trigger wasn't met (Q3's
  default).
  - It comes back only if, on a generated-voice chain, `high_share` and
    `noise_floor_dbfs` rise together across segments by more than they move between
    two runs of the same seed.
  - It would then be its own `idea`, and it must say how it reaches fl2va guide
    chains, or accept ref2va-only.
  - It would need new `chain` syntax, so removing it later would be breaking.
- **A drift rule or finding.** Comes back once two or more generated-voice chains show
  a trend past the rerun gap.
- **A spectral centroid ("brightness") field.** Comes back if `high_share` reads flat
  while a listener hears dulling.
- **A bench driver measure and its `bench/README.md` row.** Left out to keep stage 1
  one session. It comes back with a bench that tracks audio drift.

## What stays

The two per-shot fields on `analyze_shots`, report-only. They are cheap to remove if
nothing comes to read them.
