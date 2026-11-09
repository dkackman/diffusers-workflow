# Narrated take: drift counters and sync, measured (narrated-long-take plan)

Measured 2026-11-09. Can the lip-sync measure (`dw/tasks/lip_sync.py`) tell
a segment whose mouth moves with its own audio from one whose audio is wrong,
well enough to be a per-segment chain check? The plan's rule: if at least 90%
of good segments score above every bad one at the same segment, the measure
stays in the chain check as built; otherwise it leaves the chain check, the
task stays, and sync is judged by eye. This used no new GPU time.

## Sync calibration

### Method

- **Footage.** The 70 kept segments of lem workspace `constitution-walk-2`:
  MiniMax H3 dialogue, one speaker to camera, 1344x768, int8 base, 20 steps,
  24 fps, 187 or 209 frames each.
- **Measure.** `lip_sync._mouth_signal` (the mean absolute grey change inside
  the mouth box cut from YuNet's mouth corners) against `lip_sync._envelope`
  (the audio's RMS per frame's span of samples), Pearson r at every
  whole-frame lag within ±24 frames (±1 s), on CPU.
- **Good.** Each segment with its own audio.
- **Bad.** Each segment with (a) its own audio 0.5 s late (12 frames) and
  (b) another segment's audio (seven segments on in the list, padded or cut
  to length).
- Raw per-segment numbers: the controller's scratchpad `calib/calib.json`
  (`r0`, `near_max` for |lag| <= 3, `best_r`, `best_lag`, per case).

### Separation

A segment separates when its good score is above its bad score for that same
segment. The rule needs 90% (63 of 70).

| Score | Good above late | Good above other audio |
|---|---|---|
| Best r within ±0.4 s (as built) | 40/70 | 60/70 |
| Best r within ±3 frames of lag 0 | 61/70 | 64/70 |
| r at lag 0 | 56/70 | 62/70 |

Across segments, no threshold separates one segment on its own: the 10th
percentile of good (best r within ±3 frames) is 0.138, the 90th of late is
0.27. Medians: good 0.288, late 0.145, other audio 0.100.

### Where the best lag sits

The best lag within ±1 s is the clearer sign. Good segments cluster at 0 to
+2 frames (34/70 at 0..+2, 40/70 at -2..+4). With the audio 0.5 s late, the
best lag moves to +12 to +14 frames (39/70 there), and only 5/70 land within
-2..+4.

### Verdict

No score reaches the 90% rule against the late audio, and no threshold
separates a single segment, so lip sync is not a per-segment chain check
(ruling R9): `lip_sync_min` is gone from the chain's `check`, and the chain
records no sync value. `measure_lip_sync` stays as a coarse take-level
indicator: `correlation` is the best r within ±3 frames of lag 0 (the best
row above), and `lag_seconds` is the best lag within `max_lag_seconds`
(default 1 s), which read over a whole take shows a consistent offset. One
segment's correlation is not a verdict. Judge sync by eye.
