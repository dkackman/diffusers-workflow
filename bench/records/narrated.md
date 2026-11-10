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

## CP1: control vs candidate, seed 1, 12 segments (~86 s)

Bench `bench/workflows/narrated-take.json` on lem (develop 036cb279,
`DW_EXPERIMENTAL_REFINE_WINDOWS=1`), workspace `constitution-walk-3`, scene
`s3-dialogue-walk`, speaker still `asset:bench/constitution-speaker.jpg`,
held narration `asset:bench/constitution-narration-retimed.wav` (298.45 s,
43 planned segments), `stop_after: 12`, tier `best` (1344x768, int8, 20
steps, no LoRA), 192/170 frames per segment, guide 22.

### Control (guide carry, no counters) - job d5584b91816d

Run `BenchNarratedTake/20261109-225944-af94be0b`. Wall 233 min, 19.4 min
per segment. No re-rolls; `texture_ratio` (warn-only) flagged segments 9-12
(2.12, 2.25, 2.51, 2.39).

| Segment | starts (s) | face_drop |
| --- | --- | --- |
| 2 | 8.0 | 0.003 |
| 3 | 15.1 | 0.032 |
| 4 | 22.2 | 0.043 |
| 5 | 29.3 | 0.057 |
| 6 | 36.3 | 0.146 |
| 7 | 43.4 | 0.114 |
| 8 | 50.5 | 0.111 |
| 9 | 57.6 | 0.140 |
| 10 | 64.7 | 0.148 |
| 11 | 71.8 | 0.154 |
| 12 | 78.9 | 0.207 |

Frame saturation (HSV mean): 68.3 at 0 s, 70.7 at 15 s, 75.6 at 30 s,
77.8 at 45 s, 78.8 at 60 s, 81.4 at 75 s, 81.0 at 85 s. Brightness flat
(111.6-115.5).

Eye: no resets, no change of size in frame, speaking throughout; the
narration's cadence carries (the halting pauses of constitution-walk-2 are
gone). The face is etched by 30 s, glossy and caricatured by 60 s, crusted
by 85 s; the backdrop turns painterly with repeating trees from about 60 s.
Defect 1 (the smoothing/waxing drift) reproduces at the same lengths as
`bench/records/drill.md` ask 1; `face_drop` steps up at segment 6 (36 s),
the first point an eye calls it.

### Candidate (latent carry, latent keystone every 3, capstone 0.7) - job 576bad7a2138

Run `BenchNarratedTake/20261110-025240-db6a16a5`, same arguments as the
control plus `experimental_latent_carry: true`, `keystone: segment_1`,
`keystone_slot: latent`, `experimental_capstone: {enter_below_sigma: 0.7}`,
`anchor_every: 3`. Wall 224 min, 18.6 min per segment (4% under the
control). No re-rolls; `texture_ratio` flagged nothing.

| Segment | control face_drop | candidate face_drop |
| --- | --- | --- |
| 2 | 0.003 | 0.001 |
| 3 | 0.032 | 0.038 |
| 4 | 0.043 | 0.023 |
| 5 | 0.057 | 0.037 |
| 6 | 0.146 | 0.094 |
| 7 | 0.114 | 0.105 |
| 8 | 0.111 | 0.102 |
| 9 | 0.140 | 0.113 |
| 10 | 0.148 | 0.141 |
| 11 | 0.154 | 0.195 |
| 12 | 0.207 | 0.205 |

| Saturation at | 0 s | 15 s | 30 s | 45 s | 60 s | 75 s | 85 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 68.3 | 70.7 | 75.6 | 77.8 | 78.8 | 81.4 | 81.0 |
| candidate | 68.4 | 67.6 | 68.0 | 66.3 | 67.7 | 66.4 | 65.7 |

Seams (`analyze_seams`, no findings on either): candidate jump_ratio
0.63-1.53 (seams 10 and 11 at 1.53/1.51), lightness steps up to 6.1;
control 0.49-1.12, lightness steps up to 3.6. By eye the candidate's worst
seams hold framing and pose with a slight brightness lift, no pop.

Eye: the candidate removes the colour burn outright and holds the face
natural to about 60 s, where the control is already glossy. It does not
hold to 85 s: by segment 11 the face glazes as the control's did at 60 s,
and `face_drop` meets the control's at the end (0.205 against 0.207).
The backdrop drifts in both (the colonnade gives way to rows of trees).

### CP1 reading

The candidate moves the face drift from about 30-36 s to about 70 s and
removes the saturation burn, at no extra wall time; neither arm holds a
face to 85 s. Seed 1 only.
