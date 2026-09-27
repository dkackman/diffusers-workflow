# A true-peak limit mode for `normalize_audio` (#474)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-09-27). Designed on #474 (split from #467). Plan v1 was approved with its
defaults to Q1-Q5. Plan v2 re-planned stage 2 after a bounce and was approved
with Q6's default (a). Built as stages #496 (1) and #497 (2).

## The problem

`normalize_audio(target_lufs=T, peak_dbfs=C)` applies one static gain,
`min(T - measured, C - sample_peak)`. When a single transient sets the peak,
the track can't reach T and the step warns `target_lufs_capped`. An example is
the studio-audience laugh in #467: a 22.7 dB crest held Rollback at -20.7 LUFS
with a -3 dBFS peak. #467 fixed loudness *matching* by matching downward.
This feature handles the other case: a series or a song that should sit
louder than its most dynamic material allows, such as the -14 to -16 LUFS
that streaming targets. The #486 field session got there only by leaving dw
for ffmpeg `alimiter`, and lost its provenance on the way.

## Verdict and what was decided

The verdict was **build smaller**: one `limit` boolean on `normalize_audio`,
with fixed timings and no knobs.
- Q1: a mode on the existing task, not a `limit_audio` task.
- Q2: `compress_audio(mode="limit")` is left alone and documented.
- Q3: timings and thresholds are fixed.
- Q4: with `limit`, `peak_dbfs` is a true-peak ceiling.
- Q5: only `assemble-and-score` exposes it.
- Q6: the ceiling is held on the mix, not the muxed film.

## What was built

### Stage 1 (#496): `normalize_audio(limit=true)`

The signature is now `normalize_audio(audio, peak_dbfs=-1.0, target_lufs=None,
limit=False, sample_rate=None)`, in `dw/tasks/audio_utils.py`. `limit=False`
is today's path unchanged. A non-bool `limit` is refused.

With `limit=True`:
- **Ceiling.** `peak_dbfs` is a true-peak ceiling: 4x `resample_poly`,
  linked across channels, and oversampled in blocks so that a long track
  never holds 4x of itself.
- **Gain.** It starts at `target_lufs - measured`, or at the peak-normalising
  gain when there is no target, with no ceiling cap. The makeup gain comes
  only from the target, so the limiter can't auto-level the way `alimiter`
  does by default.
- **Limiter** (`_limiter_curve`, `_limit_at`). Vectorised, with no
  per-sample Python loop. Look-ahead 5 ms, hold 20 ms, release 150 ms per
  6 dB (`LIMITER_*`). The look-ahead is applied to the gain curve and never
  to the signal, so there is no delay and sync with picture is untouched.
- **Gain search** (`_search_gain`). Limiting lowers loudness by an amount
  that depends on how dense the material is. The plan said to make one
  correction pass; stage 1 bounced on that (below), so the gain is
  **searched for** instead:
  - it starts at the target's own gain;
  - it tries the 12 dB cap next;
  - it closes by regula falsi to within 0.1 LU (`LIMITER_TOLERANCE_LU`), in
    at most 8 limiting passes.
- **Caps and warnings.**
  - Reduction stops at 12 dB (`LIMITER_MAX_REDUCTION_DB`).
  - A final result further short of the target than the tolerance, whether
    at the cap or not, warns `target_lufs_capped` with `limited: true` and
    `shortfall_lu`, both on the warning event.
  - `limiter_heavy` fires past 6 dB of reduction.
- **Reporting.** It extends the #392 log event: `constraint: "limiter"`,
  `gain_db`, `max_gain_reduction_db`, `limited_fraction`,
  `output_true_peak_dbfs`, `output_lufs`.

Also in stage 1:
- `scipy` is declared in `pyproject.toml`. It was already imported by
  `dw/loudness.py` but came in only transitively.
- docs/TASKS.md has the `target_lufs` and `limit` rows, and a note that
  `compress_audio`'s `limit` mode is sample-peak with no look-ahead.
- The CLAUDE.md `normalize_audio` bullet was updated.

### Stage 2 (#497): `assemble-and-score` and the skills

- **The template.** `workflows/templates/assemble-and-score.json` has a
  `limit` variable, default `false`, passed to `balanced` beside #467's
  `target_lufs` (`peak_dbfs: -3.0` unchanged).
- **The description (v2).** It says the ceiling holds on the mix `balanced`
  writes, not on the film: the AAC mux can land up to about 1 dB above it on
  limited material. The measured case was -2.54 dBTP, which is still under
  the -1 dBTP ceiling of streaming delivery.
- **The skills.**
  - `series-episodes`: match downward by default. Pass `limit: true` beside
    the same `target_lufs` only when the series should sit louder than its
    most dynamic episode allows. `limiter_heavy` means lower the target.
  - `minimax-music3`: one line saying the same for a song master.

## Measured on the fixture

The fixture is `asset:qa-cast/ep11-coldopen.mp4`, at -17.43 LUFS and
-0.96 dBTP. Runs at `target_lufs: -16`, `peak_dbfs: -3`:
- **Without `limit`:** capped at about -19.4/-19.5 LUFS (shortfall about
  3.4 LU), `constraint: "peak_ceiling"`.
- **`limit: true`, the task alone:** -15.96 LUFS and -2.999 dBTP.
  `max_gain_reduction_db` 4.46, `limited_fraction` 0.162.
- **`limit: true`, the `assemble-and-score` film:**
  - the `balanced` mix is at -3.0 dBTP / -16.0 LUFS;
  - the AAC-muxed film is at -16.01 LUFS / **-2.537 dBTP**, an encode
    overshoot of 0.46 dB;
  - an unlimited mix at a -3.0 sample peak came out of the same mux at
    -3.10 dBTP.

## Deferred, and why

- **Holding the ceiling on the encoded film** (Q6 (b)/(c)). No fixed margin
  can be known to be enough, because the overshoot depends on the material.
  A trim after the encode would lift the non-goal "changing how the encoded
  file is checked" and would be a stage of its own. Don chose (a): the film
  gate is ≤ -2.0 dBTP with no `audio_clipped`.
- **`warn_if_written_above_full_scale` reads sample peak** (`dw/result.py`),
  although `probe_media` measures `true_peak_dbfs`. This is a proposed
  follow-up that was not filed.
- **`compress_audio(mode="limit")`** is unchanged (Q2). It is sample-peak,
  has no look-ahead, and nothing uses it. Routing it through the new limiter,
  or removing it, would each be an issue of its own.
- **`limit` on other templates** (Q5): `dissolve-between-shots` and
  `minimax/music-video` need `target_lufs` exposed first.
- **Tunable timings** (Q3). A field report of pumping would be what brings
  them back.

## Cost and bounces

The stage comments record no usage figures, so no cost is given here. The
plan estimated $4-6 for stage 1, $2-3 for stage 2, and about $0.5 for the v2
change.

| stage | bounces | why |
|---|---|---|
| #496 | 1 | C-F133 `heavy` (-12 LUFS target) landed 2.06 LU short with no `target_lufs_capped`. The plan's single correction pass didn't converge on dense material. Fixed by the gain search. |
| #497 | 1 | C-F135 asserted ≤ -2.9 dBTP on the film, and the AAC mux gave -2.54. This was the plan's gap, not a build miss. Re-planned as v2 (Q6), and verified against C-F150 and C-F136. |

Cases: C-F132, C-F133, C-F134 (stage 1), C-F150, C-F136 (stage 2). C-F135
asserts the v1 film gate; dkackman/harnest#25 asks to retire it in favour of
C-F150.
