# `minimax/face-repair`: low-strength H3 redraw of the tracked face crop, audio held (#1092)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 approved 2026-10-31 under Don's delegation, with Q1–Q4
at the plan's defaults; plan v2 folded those answers in with no scope
change. Three stages, #1097, #1098 and #1099, all verified 2026-10-31. It
is the H3 sibling of `face-repair-complete.md` (#599), whose
`crop_face_track` and `paste_face_track` tasks it reuses, and its middle
step is the H3 video-to-video with the audio held from
`h3-audio-hold-refine-complete.md` / `h3-regional-edit-complete.md` (#1062).

The concept came from r34vtraining/Longshot_Studio (its Face refine,
assessed 2026-10-31). That repo has **no license**, so nothing was copied:
the close-up prompt, the gate and the strength are ours, tuned on lem.

## The report

The only face pass was LTX (`templates/ltx2/face-repair`). An LTX redraw of
an H3 talking face can't see the H3 audio, so it could soften the lip sync
that made the take worth keeping. H3 with the shot's own track held is
conditioned on the sound that drove the mouth.

## Verdict

**Build, probe first.** Most of the mechanism existed already, so the open
question was whether H3 beats LTX at this job. Stage A answered it on GPU
with no code before anything was built (the #664 lesson: an H3 refine
template that failed its quality gate and was reverted).

## What the plan found against the issue

- **Audio alignment.** `crop_face_track` pads the crops to the grid with
  reflected frames at both ends, and `hold_audio` starts at sample 0, so
  reusing the task unchanged would have the sound lead the mouth by up to
  8 frames. Fix: `crops` carries the source track, silence-padded by
  `pad_before` / `pad_after` frames, so `hold_audio:
  previous_result:crop.crops` lines up frame for frame.
- **Length and rate.** H3 takes 124–345 frames at 24 fps. The crop step
  refuses anything else before a pipeline loads.
- **Strength scale.** The issue's ComfyUI "denoise 0.2/0.3/0.4" is
  unshifted. At `video_shift` 6 it is about `refine_strength`
  0.6 / 0.72 / 0.8 in dw (s = 6u / (1 + 5u)).
- **Sync metric.** `measure_sync` compares two audio tracks, and the
  pasted result's audio *is* the source's, so it always reads 0. Mouth sync
  is judged on frames (`compare_outputs` face grids, `get_output_frames`).
- **Template shape.** A separate template, not a `model` switch on the LTX
  one: the pipeline block differs wholesale, and the LTX template has 10+
  consumers.

## Decisions

- **Q1–Q4 (approval, 2026-10-31):** silence-padded track at both ends (tail
  only as a fallback); the probe picks one mode; the stage A session picks
  the takes; Don owns the gate.
- **Gate ruling on the probe (Don, 2026-10-31):** build, **t2va at
  `refine_strength` 0.6**.
  - ref2va with a cast picture was only marginally better (face SSIM 0.732
    vs 0.708) at about 4x the denoise time, so it doesn't ship. The
    template description says it is a slightly better, slower option.
  - 0.72 drifts identity; 0.8 distorts frames.
  - **Face gate 0.2 / 0.3** in the H3 template only. The task defaults and
    the LTX template's 0.03 / 0.06 are unchanged.

## What was built

**Stage A, #1097: the probe (GPU, no code).** Workspace `probe-1092` on
lem, seed 1092, two small-face takes from `KnownIssuesRestorePointHD`
(`tag`, the man says "We have hope."; `cold_open`, the same man silent).
Arms: LTX `ltx2/face-repair` at strength 2; H3 t2va refine at 0.6 / 0.72 /
0.8; ref2va at 0.6 with `asset:cast/pat.jpg` as `<Picture 1>`; the #1062
regional edit with a static face-box mask at 0.85. Result:
- H3 (both modes) keeps the man's face; LTX at strength 2 replaces him with
  a younger, rounder man with a smaller moustache, on both takes.
- H3 mouth sync is as good as LTX's.
- The masked full-frame regional edit was the worst arm (too few patch
  cells on a small face, as the plan expected).

**Stage B, #1098: `crop_face_track` audio and window** (`db71fc1a`, merged
as `3d8bc206`).
- When the clip has sound, `crops` carries it: `pad_before` frames of
  silence, the clip's sound, `pad_after` frames of silence, with each
  boundary rounded from frame times so the track is exactly
  `crop_frames / fps` long (`padded_audio`, `crop_soundtrack` in
  `dw/tasks/face_track.py`). A silent clip gives `crops` with no audio and
  no warning.
- Optional `min_frames`, `max_frames` and `fps` (±0.01). The window is
  checked against the **padded** crop count, and the message names both
  counts. `min_frames` ≤ `max_frames` is a validate-time problem in
  `dw/task_problems.py`.
- `docs/TASKS.md` documents the three arguments, the padded soundtrack, and
  that the grid defaults are LTX's, with H3's 17/5/32.
- The LTX template's `crops` intermediate now also carries sound; its
  final output and step-cache keys are unchanged.

**Stage C, #1099: template and skill** (`develop` @ `b110e7cd`).
- `workflows/templates/minimax/face-repair.json`, shape `shot`, traits
  `needs-input-media`, `has-audio`: `crop` (17/5/32, window 124–345,
  fps 24, gate 0.2/0.3) → `refine` (H3 t2va, `refine_strength` 0.6,
  `source_video` and `hold_audio` both `previous_result:crop.crops`, our
  own close-up prompt) → `pasted` (`paste_face_track`, final).
- Cost measured on lem: 2.2 min (job 38da34a526e1, 127 s from a fresh
  load).
- A minimax README row, a `TEMPLATE_PIPELINE_KEYS` row, `COMPACT_BUDGET`
  raised to 12_250 with a dated line, and one `minimax-h3` skill line
  within its byte cap.

## Deviations from the plan

- **The probe had to open the face gate.** These faces are about 0.087 of
  the frame, so at the LTX gate (0.03/0.06) every arm's paste was a no-op
  (PSNR 37.5 vs the source). That became Don's 0.2/0.3 ruling.
- **The probe's takes were 124 frames**, not trimmed to 17n+5 by hand;
  124 is already on the grid.
- **The padded-head risk (Q1) was checked on a built clip.** Every
  small-face take on lem is exactly 124 frames, so the tester appended one
  frame to make 125 (padded to 141 crops, job a4ad2c6752a7). The mouth on
  frames 0–6 matched the source and the unpadded arm frame for frame, so
  tail-only padding was **not** needed. Not a pass condition, but seen in
  both arms: the head frames' brow reads slightly more furrowed and the
  skin slightly redder than the source's. Identity holds.

## Bounces per stage

None. Each of #1097, #1098 and #1099 passed its first verify, and both
code stages passed architecture review.

## Cost

The stage comments name no `usage:` figures, so per-stage spend isn't
recorded here. The one figure on file: the probe spent about $6 against its
$1.5–2.5 estimate, mostly on reruns after the gated no-ops.

## Deferred and left open

- **ref2va mode** (cast picture as `<Picture 1>`). Back if a take loses
  identity under t2va at 0.6.
- **Tail-only padding.** Back if a padded run shows the head silence
  closing the mouth on the first real frames.
- **Takes under 124 frames** can't use this template; they use
  `ltx2/face-repair`.
- **Multi-face shots**, as in #599: `crop_face_track` tracks one face.
- **Composition with #1065** (the H3 refiner LoRA) was not tried.
