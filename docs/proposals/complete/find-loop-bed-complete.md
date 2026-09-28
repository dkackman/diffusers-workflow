# `find_loop_bed`: ranked room-tone loop windows (#218)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-09-27). Plan v1 approved by Don with every default (Q1-Q5) on
2026-09-27; stages #544 (A) and #545 (B), both verified.

## The report

Known Issues episodes 7 and 8 (2026-09-25) needed a room-tone bed under H3
dialogue cuts, and the session found one by downloading the WAVs and
running numpy locally, because the metadata envelope has only 1 s bins
(#464, closed as a duplicate). #465 shows that H3 dialogue shots hold
0.5-2 s at -70 to -84 dBFS *inside* the shot, so every dialogue cut needs a
bed. Picking the window has three traps a level check misses:

1. **Near-programme material**: speech about 30 dB down reads as quiet but
   is audible once it repeats.
2. **Lap-rate modulation**: the loop's repeat rate beats against the
   source's own level movement, which one pass through the source can't show.
3. **Ticks**: a 1 ms transient is invisible to 50 ms RMS and recurs once per
   lap. Episode 8's first pick held one 19.7 dB over the median 1 ms peak.

The field thresholds (every 50 ms bin at or below -55 dBFS, mean at or below
-60, 1 ms spike margin at or below 12 dB, ranked on the *looped* result's
50 ms ripple) became the task's defaults.

## Decisions (plan v1, approved as written)

- **Q1: not an assessment probe.** It is a search, not a check of a cut.
  `register_command` already had the `assessment` flag (from #485, default
  False), so no registry change was needed. The three-probe test is
  unchanged.
- **Q2: rank by looped `ripple_db` alone.** `envelope_peak_db` is a reading,
  with a `lap_modulation` warning above -15 dB. No combined score.
- **Q3: shots in stage B.**
- **Q4: task only.** No `assess_output` probe and no sync route, so the
  one-step workflow queues behind a running GPU job.
- **Q5: no in-workflow wiring into `slice_audio`.** Candidates carry
  `start_seconds`/`duration_seconds` (`slice_audio`'s names) and `gain`
  (`mix_audio`'s linear multiplier), so the agent copies the numbers.
- **No separate tremolo estimator.** The envelope spectrum of the looped
  result captures the beat whatever causes it.

## What was built

**Stage A, #544** (server): `dw/tasks/loop_bed.py`, registered with
`returns="json"`; domain entries in `dw/task_domains.py`; an FFT
`_harmonicity` (numerically equal to the old `numpy.correlate`, `bleed_join`
unchanged); `decode_soundtrack` in `dw/media_audio.py`; a `docs/TASKS.md`
section, with the room-tone recipe now starting from the task.

- Arguments: `audio` (path, `AudioTrack` or `AudioVideo`),
  `start_seconds`/`end_seconds`, `min_seconds` 0.5, `max_seconds` 2.0,
  `max_bin_dbfs` -55, `max_mean_dbfs` -60, `max_spike_db` 12, `crossfade_ms`
  250 (`loop_audio`'s default), `loop_seconds` 10, `target_bed_dbfs` -60,
  `max_candidates` 5.
- Output: `source`, `criteria`, ranked `candidates` (each with its level,
  spike, flatness/harmonicity and `looped` readings, `gain_db`/`gain`,
  `warnings`), `rejected` counts, and one `no_loop_bed` finding (with
  `rejected_by`) when nothing survives. No candidates is an answer, not a
  failed job.
- Deviations from the plan, as built:
  - A video's soundtrack is read as float32 from the audio stream alone
    (`decode_soundtrack`), not through `extract_audio`, whose s16 output
    requantizes -70 to -85 dBFS room tone near 16-bit's floor.
  - `rejected` is a true tally: every (start, length) window on the grid is
    counted once, under the first rule it fails (too_loud, silent, spike,
    tonal).
  - The tonal test (#198's flatness < 0.3 or harmonicity >= 0.45, thresholds
    unchanged) runs over every 0.1 s and 0.2 s block inside a window, not
    the whole window: pauses between syllables dilute intermittent speech
    below the threshold. A candidate reports its worst block's readings.
  - Flatness is measured over the band the source occupies
    (`_occupied_rate`), the #198 mechanism: an upsampled source's empty top
    band otherwise reads as tonal whatever the material.
  - The spike check also reads 5 ms just outside each end of a window, so a
    window ending on a click is rejected.
  - The looped measurement runs on a fixed pool of 200 non-overlapping
    survivors (`LOOPED_POOL`), so `max_candidates` only truncates the final
    ranking and a smaller value returns a prefix of a larger one.
  - `criteria` also echoes `min_seconds`, `max_seconds` and
    `target_bed_dbfs`.

**Stage B, #545** (server + plugin): `shots` (the probes'
`{name, start_frame, num_frames}` shape, optionally with samples) and `fps`.
They resolve as the argument (`"argument"`), then shots a `previous_result:`
video carries (`"artifact"`), then the run manifest beside the file
(`"manifest"`), else `null`. With shots, no candidate crosses a boundary,
each names its `shot`, `source.shots` places each shot in seconds, and
`rejected.shot_boundary` counts the windows a boundary cut through (present
only when shots resolved). Without shots the output is stage A's. #465's
`shot_dead_air` message now names `find_loop_bed`, and so do the
`minimax-h3` and `series-episodes` skills (no new numbers).

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #544 | 2 | 1: `max_candidates` cut the pool before the looped ranking (C-F163); a window ending on the click; `rejected` not a tally and whole-window tonality missing a voice under the bed (C-F161). 2: C-F161 still failed; the upsampled fixture's empty band read as tonal, fixed with occupied-band flatness and 0.1 s blocks. |
| B, #545 | 0 | |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
The plan estimated about $7-11 total.

Open against the suite, not the code: harnest#37 (relax C-F164's
`rejected` equality across a lossy re-encode) and the tester's C-F168
fixture amendment (assemble-and-score normalizes the film to about -24 dBFS,
so the case's bed is no longer quiet). Both cases keep their `pending:` lines
until those are ruled on.

## Deferred

- **A sync route or `assess_output` probe** (Q4). Back if the queue wait
  behind a GPU job hurts in practice.
- **Wiring a candidate into `slice_audio`** inside one workflow (Q5), which
  would be a syntax change.
- **A combined ranking score** (Q2). The ranking is fitted to two field
  cases; every reading is returned so an agent can re-rank.
- **Thresholds for louder rooms.** The defaults come from one programme (H3
  dialogue). An ambience-heavy LTX shot may reject everything at -55/-60;
  the arguments and the `no_loop_bed` finding's `rejected_by` are the
  mitigation until a second programme gives numbers.
