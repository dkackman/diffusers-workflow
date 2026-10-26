# Look measures in `analyze_seams` (#1031)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-26). Designed on #1031 from an idea taken from harlixay7/Kinoloom
(MIT, commit 45cfe50, `docs/research/SEAMLESS_CHAINS*.md`; only the idea, none
of its ComfyUI code). Plan v1 was approved with its defaults to Q1-Q3. Built
as one stage, #1042 (A).

## The problem

`analyze_seams` saw a seam's picture through one number, `jump_ratio`: the
largest grey frame-to-frame change across the seam against the shots' typical
change. A chain join that keeps the framing but shifts saturation or
brightness over a frame or two scores low there. Grey change is small, and
the 64x36 grey assess thumbnails (`dw/media.py`) carry no colour at all.
Kinoloom measured saturation +10 to +19 at the joins of its pixel-continuation
chains. #1033 (colour-match chained segments) can't be judged without an
instrument for this, and #1034 (re-roll a failing seam) would read it too.

## Verdict and what was decided

The verdict was **build smaller**: per-seam measures only, with no rules.
- Q1: no rules (`info` or otherwise) in this stage. A threshold set now would
  come from Kinoloom's broken chains, not ours.
- Q2: one `look_ratio` plus `look_ratio_measure`, not three ratios.
- Q3: take drift (new `analyze_shots` fields) is cut.

Corrections the plan made to the idea:
- The measures can't be computed from what `analyze_seams` already read.
  Saturation needs colour the reader dropped, and a high-frequency ratio at
  64x36 is noise. They are taken during the existing single decode.
- "Apparent age from InsightFace riding on `face_track`" doesn't fit:
  `face_track` is YuNet (boxes only), InsightFace isn't a dependency, and its
  packaged models are licensed for non-commercial research. Cut.

## What was built

### Stage A (#1042): look stats and seam look fields

- **Look stats at decode.** `read_thumbnails_and_track` (`dw/media.py`)
  takes an optional `look` reducer, run on a 128x72 rgb24 copy of each frame
  through its own `VideoReformatter`; only the reducer's result is kept.
  `look_stats` in `dw/tasks/assess.py` is the shared reducer, on the 0-255
  scale:
  - lightness: mean Rec.709 luma (`image_ops.luma`);
  - contrast: luma standard deviation;
  - saturation: mean of per-pixel max(RGB) - min(RGB);
  - detail: mean |luma - 3x3 box blur|, the blur from
    `scipy.ndimage.uniform_filter(size=3, mode="nearest")`.
- `Media.looks` is an (frames, 4) float32 array, or None. The grey
  thumbnails are unchanged, so `jump_ratio` and `analyze_shots` are too.
  The in-memory path (`_thumb` / `_in_memory_thumbs`) calls the same
  `look_stats`, so a file and its in-memory original agree within rounding.
- **Seam fields.** `_seam_look`, beside `_seam_video`, adds to every seam
  record: `saturation_step`, `lightness_step`, `contrast_step` (signed,
  after - before), `look_ratio`, `look_ratio_measure` and `detail_ratio`.
  - Windows: the last 4 frames before the seam (before a dissolve's fade)
    against the first 4 after it, clipped to the shots.
  - Local rate (Kinoloom's "local-rate rule"): the 90th percentile of the
    change from one 4-frame mean to the next over each shot's span, the
    larger shot's counting, floored at 1.0. A shot under 8 frames gives no
    rate, so the floor applies. `look_ratio` is the largest |step| over its
    own measure's rate.
  - `detail_ratio` is null when the before window's detail is below 0.5.
  - A seam on media with no picture carries the six fields as null; an
    audio-only file has no seams, so nothing appears.
  - Numbers are rounded to 2 places, like the other seam fields.
- **No new rule, finding, or `rules_applied` / `rules_skipped` entry.** No
  MCP, REST, schema, UI or plugin change; the `assess_output` description is
  untouched, so the MCP surface budget isn't spent.
- **Tests:** `tests/test_seam_look.py`, 15 cases on real mp4s and in-memory
  clips (known-colour stats and path agreement, saturation and darkening
  cuts, a mid-ramp cut for the local rate, a blurred copy, grey, frameless,
  single-shot and dissolve inputs, no look rule names).
- **Docs:** six rows in the `docs/TASKS.md` `analyze_seams` table with the
  "measures, not rules; not perceptual" note, and one sentence in the
  `docs/WORKFLOW_GUIDE.md` assessment section.

## Measured on lem

From the tester's verification of C-F538-C-F544 (develop @ 9411935e), one
seam at frame 124 of joins over `ep3-shot1-incident.mp4`:

| seam | reading |
|---|---|
| same clip joined to itself | saturation_step 0.64, look_ratio 0.64, detail_ratio 1.0 |
| copy graded saturation 1.6 | saturation_step +19.21, look_ratio 19.21 (30x the control) |
| the same, order reversed | saturation_step -17.55 |
| copy darkened (exposure -1) | lightness_step -47.5, look_ratio 47.5 |
| copy softened (clarity -1) | detail_ratio 0.49 |
| greyscale clips | saturation_step 0.0 |

The +19 saturation join is the size Kinoloom reported at its chain joins.
A `dissolve-between-shots` output and all 7 seams of `ep53-episode.mp4`
carry the fields. `analyze_seams` on `ep54-episode.mp4` took about 2 s, and
`long-film-4089.mp4`'s default probes about 8 s, well under the MCP
client's 30 s. The build measured the reader's added cost at about 0.25 ms
per frame once the look copy got its own reformatter (a naive second
`reformat` rebuilt the scaler each frame, about 1.5 ms).

## Deferred, and why

- **Rules** (e.g. an `info` `seam_look_jump` on `look_ratio`). Comes back
  when lem measurements on real chains (#1033's before and after) show a
  `look_ratio` value that separates seams Don calls visible from ones he
  doesn't: a threshold from our footage, not Kinoloom's.
- **Take drift** (`analyze_shots` fields, first second against last). Comes
  back with a take whose seams all score low but whose end visibly looks
  different from its start, reported from real driving.
- **Apparent age / identity drift.** Comes back with a field report of a
  lead's face visibly ageing or changing identity across a chain, plus a
  decision to take on a face-attribute model and its licence.
- **Plugin.** Teaching `script-to-video` to read the fields is #1034's.
- **Local rate over MCP.** No tool grades a ramp over time, so the mid-ramp
  behaviour is checked only by the unit test, not by a suite case.

## Cost and bounces

The stage comments name no `usage:` figures, so cost is left out. The plan
estimated $4-6 for the one build session.

- **Stage A (#1042):** one architecture-review bounce, no tester bounces.
  The review found the 3x3 box blur hand-rolled (pad and nine shifted
  slices) where `scipy.ndimage.uniform_filter` already does it; fixed in
  `cee80db4`, same numbers. Verified first time on all seven cases
  (C-F538-C-F544).
