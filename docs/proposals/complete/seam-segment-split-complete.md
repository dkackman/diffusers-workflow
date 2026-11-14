# Seam/segment split of chain drift (#1324)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-14). Designed on #1324 from an idea taken from roadmaus/ComfyUI-Continuity
(MIT, read 2026-11-13: `docs/timeline.md`, changelog entries and module
docstrings; the concept only, none of its code). Plan v1 was approved at its
defaults to Q1-Q3 (ruling 2026-11-13, by the lead session under Don's
delegation). Built as two stages, #1349 (A) and #1350 (B).

## The problem

The bench couldn't say whether a chained take's drift is added at the seams,
where a handoff change could remove it, or inside the segments, where it
can't, and it couldn't say it in numbers anyone could repeat. The split had
been run by hand three times on the narrated bench with a scratch script
outside the repo (Don's table on #1324; `bench/records/narrated.md`, the
masked handoff #1322 and levelled latent carry), and each time it was the
deciding measure.

## Verdict and what was decided

The verdict was **build smaller**: whole-frame measures (saturation,
lightness, contrast, detail) only.
- Q1: 1 s windows (`round(fps)` frames), to match the records and Don's
  script, not the 4-frame `LOOK_WINDOW` of the seam step fields.
- Q2: face-crop measures (face saturation, face sharpness) stay out. They
  remain in the scratch script.
- Q3: a server field plus driver keys, so the numbers can be repeated over
  `assess_output`, rather than an in-process bench script.

Correction the plan made to the idea: the in-segment number is the change
between a segment's first window and its last, not a fitted slope. With
endpoint windows the parts sum exactly (end-to-end = Σ seam steps +
Σ in-segment changes), so each share is a true fraction of the whole.

**Limit of the measure** (kept as Don's ruling asked): the split says where
a measure changes, not whether the change is the one Don sees. Don's
2026-11-13 "progressively glossy" verdict on levelled carry was missed by
the face-sharpness measure and the stills. It informs a viewing verdict and
does not replace one.

## What was built

### Stage A (#1349): `segment_looks` on `analyze_seams`

- `dw/tasks/assess.py` `_segment_looks` adds an answer-level `segment_looks`
  list to `seams_answer`, one entry per shot in order:
  `{shot, between_frames: [first, last], first: {lightness, contrast,
  saturation, detail}, last: {...}}`, from `media.looks` (the 128x72 look
  copy #1031 added).
  - A segment runs from after its `overlap_frames` (and any seam fade) to
    the frame before the next shot starts, or the take's end for the last.
    `between_frames` is inclusive.
  - Windows are `round(fps)` frames clipped to the segment; a segment
    shorter than two windows splits in half; a one-frame segment uses that
    frame for both ends; a segment with no new frames gets
    `between_frames`, `first` and `last` all `null`.
  - `[]` for media with no picture or fewer than two shots. Values are
    rounded to 3 places.
  - A measure only: no rule or finding reads it; the seam fields are
    unchanged. No MCP, REST or schema change.
- Tests: `TestSegmentLooks` in `tests/test_seam_look.py`, 11 cases
  (exact telescoping on known steps and ramps, halving, overlap and
  `dissolve_videos` fade, frameless, audio-only, single shot, a real
  decoded take).
- Docs: the `analyze_seams` entry in `docs/TASKS.md` and the docstring's
  return line.

### Stage B (#1350): the bench driver's `split_*` keys

- `bench/driver.py` `_split`: `measure()` reads `segment_looks` from the
  `analyze_seams` reply it already fetches (stdlib only, no media) and adds,
  for each of saturation, lightness, contrast and detail:
  `split_<m>_total`, `_seams` (Σ next.first − this.last), `_segments`
  (Σ last − first), `_seam_share` (seams / (|seams| + |segments|), left out
  at 0) and `_seam_sign` (fraction of seams agreeing in sign with the seam
  total; left out when that total is 0).
  - A segment missing a measure is bridged (previous valid `last` to next
    valid `first`), so the sum still telescopes. A missing, `null` or empty
    `segment_looks` adds nothing.
- Tests: three new cases in `tests/test_bench_driver.py`.
- Docs: a `## Measures` table in `bench/README.md`; definitions in
  `bench/records/noise-floor.md` (marked signed, not lower-is-better, floor
  unmeasured); a ComfyUI-Continuity line (MIT, concept only) in
  `docs/ATTRIBUTIONS.md`.

## Measured on lem

From the tester's verification of C-F835-C-F839 (develop @ 694e8a8e):
- `ep53-episode.mp4` (8 manifest shots, 12-frame overlaps on 6-8): 8
  entries; shots 6-8 start at `start_frame + 12`. The one clear seam
  (seam 5, `saturation_step` −7.6) has saturation difference −6.71, the
  same sign. Three small seams (|step| ≤ 1.77, difference < 0.4) disagree:
  the 1 s against 4-frame window difference, not a finding.
- Synthetic alternating 4-shot take (ungraded / saturation 1.6): seam
  differences +19.3, −17.6, +19.27 against `saturation_step` +19.21,
  −17.71, +19.19; the in-memory path agrees within 1.3.
- `ep54-episode.mp4` and `ep53` together answered in about 11 s.

Stage B's runtime check (keys in `report`, saturation signs matching the
table on #1324) has not been run: it falls to the first bench session that
uses the keys, recorded in `bench/records/narrated.md`.

## Deferred, and why

- **Face-crop measures** (Q2). Come back when a handoff A/B needs face
  saturation or sharpness repeatable, either as a fixed `face_crop` on
  `Scene` read through compare, or as a `crop` parameter on `analyze_seams`
  (new MCP surface).
- **Rules on the split.** None: it is a measure, and no threshold has been
  measured on our footage.
- **`bench/verdict.py` and signed keys.** Its docstring says every measure
  is lower-is-better and `verdict()` counts every shared key, so passing a
  run's whole `measures` (now with signed `split_*` keys) by hand could give
  a spurious `mixed`. Nothing in the driver calls it; exclude `split_*` there
  when a session first does.

## Cost and bounces

The stage comments name no `usage:` figures, so cost is left out. The plan
estimated $3-5 for stage A and $2-3 for stage B.

- **Stage A (#1349):** no bounces. Its first deploy waited on a busy lem and
  was redone by a later session. Verified first time on C-F835-C-F839.
- **Stage B (#1350):** no bounces. No MCP surface, so the tester handed it to
  a docs review, which found it right and closed it.
