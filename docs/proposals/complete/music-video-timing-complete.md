# Music-video timing stack: beats, cut planner, pad-then-trim shots (#600)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-06). Designed on #600 from a comparative review of
vrgamegirl19/comfyui-vrgamedevgirl. That review was concept only: their
license is incompatible with Apache-2.0, so nothing was copied. Plan v1 was
approved for stages A-C (2026-10-05) and recorded as v2. Plan v3 re-shaped
stage C after Don's decision on #627, was approved (2026-10-06) with Q6 and Q7
at their defaults, and was recorded as v4. Built as stages #625 (A), #626 (B)
and #627 (C).

## The problem

`templates/minimax/music-video` cut every shot at one fixed `num_frames`
(124), from `start_frame`s set by hand. `transcribe_audio` returned word
timestamps (#483), but nothing turned timestamps or beats into a `shots`
list. #484/#485 came from a 24-shot H3 musical placed by hand, and #142
hardcoded its soundtrack slice to 4 × 124 frames.

## Verdict and what was decided

The verdict was **build smaller**: stages A-C, with D deferred.
- **No runtime `for_each`.** `for_each` takes only `variable:`, so planning
  is its own run. The agent reads the cut list, writes one prompt per shot,
  and passes the list as `shots`.
- **No librosa.** Beats come from numpy/scipy in `dw/dsp.py`.
- **No exact-audio trim on the deliverable.** `music-video` lays the whole
  song back over the edit with `pair_audio`.
- **Q1-Q3 (2026-10-05):** numpy/scipy tracker; planning is a separate run;
  the top-level `num_frames` is removed (`breaking-change`).
- **#627 decision (2026-10-06):** no engine change. Every `shots` entry
  carries all three length fields, required. `music-video` drops its snap,
  so an off-grid entry is refused at its JSON path instead of being rounded.
- **Q6:** `slice_audio` gains an optional `lead_frames`, so `start_frame`
  keeps meaning the cut start everywhere (not taken: a `slice_frame` field
  emitted by `plan_cuts`).
- **Q7:** `lead_s` defaults to 0.5 s (12 frames at 24 fps) in
  `music-video-cuts`.

## What was built

### Stage A (#625): `analyze_beats`
- Beat tracking on numpy/scipy (`dw/dsp.py`, task in `dw/tasks/beats.py`):
  beat times and BPM, `min_bpm`/`max_bpm`, an `onset` mode, and a warning
  (never a crash) on silence and room tone.
- Calibration: one anchor plus `tempo_bpm` gives an exact `grid` (both anchor
  forms, `beat_index`), and two or more anchors warp the nearest detected
  beats. A min at or above the max, and an anchor past the end, are refused.

### Stage B (#626): `plan_cuts` and `templates/minimax/music-video-cuts`
- `plan_cuts` (`dw/tasks/cuts.py`) takes timestamped transcript chunks and
  optional reference lyrics. It keeps the user's text and line order
  verbatim, places and warns about unaligned lines, and segments by line,
  stanza or beat.
- Options: `include_instrumental_gaps`/`min_gap_seconds` (intro, break and
  outro scenes), min/max scene length, a vocal tail pad, and
  `snap_to_beats` (either `beats` form).
- It returns one dict artifact whose `shots` tile `[0, duration)`
  contiguously. Frames are rounded once, so Σ `cut_frames` =
  `round(duration × fps)`.
- A bare-string transcript is refused statically and when chained, and the
  refusal names `timestamps: "segment"`.
- `music-video-cuts` runs transcribe → (beats) → `plan_cuts` on CPU from an
  `asset:` song.

### Stage C (#627): pad-then-trim shots (`breaking-change`)
- **`plan_cuts` grid:** optional `modulus`, `remainder`, `min_frames`,
  `max_frames` and `lead_s`. Without them, B's behaviour is unchanged.
  - `lead_frames = min(round(lead_s × fps), start_frame)`, so the first lead is
    0.
  - `num_frames` is the smallest grid value at or above `lead_frames +
    cut_frames`, and at least `min_frames`. The slack past the cut is a
    trimmed tail handle.
  - A shot that would pass `max_frames` is split (on a beat if one is
    available) and warned about.
  - The result gains `render_frames` = Σ `num_frames`, so the cost can be
    quoted.
  - The grid arithmetic reuses `dw/variable_constraints.py`'s alignment.
    The architecture bounce on #627 is why (see Bounces).
- **`slice_audio(lead_frames=)`:** optional, default 0, frame form only. The
  slice starts at `start_frame - lead_frames`. A negative start is refused
  statically (`dw/slice_preflight.py`, `task_domains.py`) when the values are
  literal, and at run time otherwise. It is also refused together with
  `start_seconds`.
- **`trim_video(video, start_frame, num_frames)`** in a new module,
  `dw/tasks/trim.py`. It keeps `[start, start+n)` frames, cuts the clip's
  audio to the same span at the audio's own rate, and writes shots metadata
  through `dw/shots.py`. A range past the end, and a zero or negative count,
  are refused.
- **`music-video`:**
  - The entries are `{name, prompt, start_frame, num_frames, lead_frames,
    cut_frames}`, all required.
  - The default four shots spell out 124/0/124 and render as before.
  - `song` is `{from_previous_result: write_song}`, so `song: "asset:…"`
    skips Music 3.
  - The top-level `num_frames` is gone. Passing it is an unknown-variable
    refusal.
  - The 17n+5 constraint (124-345) applies per entry with no snap, so 130 is
    refused at `arguments.shots[i].num_frames`.
  - Steps: `slice` → `shot` → a new `trim` for_each → `edit` gathers `trim`
    → `pair_audio`.
  - `tests/test_variable_constraints.py` names it as the one MiniMax
    template without a snap.
- **`music-video-cuts`:** passes H3's grid (17, 5, 124, 345) and `lead_s`
  (0.5 s) to `plan_cuts`.
  - Its shots, with a `prompt` added to each, validate as `music-video`
    input.
  - The only notice is the unread-field warning naming `lyric`/`kind`. The
    skill says to drop those two fields.
- **Docs:** the `minimax-h3` skill and `references/cuts.md` (plan, read,
  prompt, render), the other three skills' mentions, TASKS.md,
  WORKFLOW_GUIDE.md, RECIPES_24GB.md, ARCHITECTURE.md and the minimax
  templates README.

### Real run (C-F327, with Don's go-ahead)
On lem (RTX 3090), the estimate was 14.5 min and the run took 15.8 min (job
`61dd91084fac`):
- Two H3 shots rendered at 124 frames each and were trimmed to 48 and 72.
- The deliverable is 120 frames (5.0 s) with `media.shots` spans 0-48 and
  48-72.
- `assess_output` reported no sync or seam findings.

## Deferred

- **Stage D: per-shot `kind` (vocal/instrumental/broll) and `singer`.** Not
  approved. It needs an engine change (Q4: step-declared entry defaults or
  optional `for_each` fields), a `cast` map, and the skill's B-roll section.
  Now that C has cut a real music video, D goes back to Don as its own idea,
  with Q4's mechanism and what that run showed.
- **Q4: optional or defaulted `for_each` entry fields.** It is still open, so
  hand-written `music-video` lists must carry all three length fields.
- **Exact per-shot audio in the deliverable.** It is a non-goal: the
  unbroken song is laid back over the edit.
- **Stems (#604)** become an optional input to `music-video-cuts` once they
  land. **#598 (audio-hold)** is orthogonal.

## Fix-forward: #665, the render rule in the skill

The feature's final check failed one docs bullet, C-F328's: neither
`minimax-h3` `SKILL.md` nor `references/cuts.md` stated how to size an
entry's `num_frames` when `lead_frames + cut_frames` is off the grid. The
rule lived only in the tasks guide's `plan_cuts` section. It was a build
miss in stage C's skill text, not a gap in the plan, so it was filed as
fix-forward stage #665 with C-F328 as its acceptance. #665 added the rule,
`num_frames = max(124, next 17n+5 >= lead + cut)` (split past 345), to both
files with worked numbers (commit `6803a199`, merged as `f0f0411a`). It was
plugin-only, so no deploy, and C-F328 passed in full on re-verify.

## Cost

The stage comments carry no `usage:` figures, so no cost is recorded. The
plan estimated about $7 for C.

## Bounces

| Stage | Bounces | What |
|---|---|---|
| A #625 | 3 tester, then a reopen | C-F208's room-tone arm read as a confident pulse. Bounce 2's fix (`cb04ee98`, a periodicity/salience gate) regressed a real song. The stage was reopened and refixed (`cbf614a6`), and C-F208 arm 2 was rewritten onto a true noise bed. |
| B #626 | 1 tester | C-F218: the plan stopped at the last lyric instead of the song's end. Smaller misses in C-F213, C-F215 and C-F212. One hand-off was also parked with Don because a dirty tree in the shared checkout (from stage A) tripped the gate. |
| C #627 | 1 architecture, 1 tester | Architecture: `cuts.py` re-derived the frame grid that `dw/variable_constraints.py` owns. Tester: C-F220, `trim_video` wrote no shots metadata on a clip with no shots record. |
| C fix-forward #665 | none | Filed from the final check's C-F328 failure (render rule missing from the skill); verified first time. |
