# A locked H3 frame that lip-syncs to a song (#852)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-14). Plan v0 (the issue body's *Proposed*) approved by Don on
2026-10-13 and capped at four H3 renders on lem; three stages: #877 (A) and
#878 (B) verified, #879 (C) closed not planned by its own condition.

## The report

A field report: for a music-video finale, an agent had to choose between a
still camera and real lip-sync. `first-and-last-frame` (FL2VA) pins the
frames and holds a wide two-shot, but takes no audio. `reference-to-video` /
`music-video` (Ref2VA) lip-syncs to the song, but the camera drifts on wide
two-shots. It rendered the finale as solos.

MiniMax's Ref2VA prompt guide lists `keyframe completion` as a task type
(an image as the target video's first frame, keyframe or last frame), and
task types combine with ` + `. So `[keyframe completion + audio reuse]` is
a valid Ref2VA task. Whether Ref2VA, which treats the keyframe as a
reference anchor, holds a locked two-shot as FL2VA does was untested.

## Decisions

- **Don's ruling (2026-10-13):** approved as proposed, capped at four H3
  renders on lem, all on one wide two-shot and one ~8 s song slice, each
  with its `plan.estimate` quoted first. Either outcome becomes a
  `minimax-h3` skill line; a template variant only if the recipe holds and
  needs a shape of its own.
- **Routing (2026-10-14):** first routed to the tester, whose handoff
  session couldn't run it; Don re-routed it to the lead as an approved
  experiment.

## What was found (stage A, #877)

Same still (`asset:h3-keyframe-audio-two-shot.jpg`) and 8 s duet slice
(`asset:h3-keyframe-audio-duet-8s.wav`, 192 frames at 24 fps, 960x544) for
all four. The man sings 0-2.8 s, the woman 4.4-7.3 s.

| # | Job | Variant | Time | Camera hold | Lip-sync |
|---|-----|---------|------|-------------|----------|
| 1 | `03c511a226bd` | Ref2VA, `<Picture 1>` first frame + `<Audio 1>` (`hold_audio`) | 13.3 min | Holds, static all 8 s; frame 0 re-creates the still (a touch wider), not its pixels | Right singer on each line |
| 2 | `d2f11181111b` | The same plus `<Picture 2>` last frame | 19.3 min | Holds, but the last frame does not land on the still | Right; no better than run 1 |
| 3 | `d8afb9f2d9e1` | FL2VA baseline, no audio | 6.2 min | Exact: first and last frames pixel for pixel | n/a, self-generated soundtrack |
| 4 | `97b8d792cdf2` | `music-video` baseline | ~9.6 min | Static, but in its own framing, not the still | Right, same quality as run 1 |

**Verdict: holds, first frame only.** The keyframe is honoured as a
composition, not as pixels; only FL2VA gives pixel-exact frames, and it
can't carry the song. A last frame costs 45% more and doesn't land. Run 1
carries the keeper note. Every job was kept on lem.

Quoted estimates: runs 1 and 2 at 12.0 min (basis `inherited` from
`reference-to-video`, low confidence); runs 3 and 4 came back `unknown`
(no catalog cost or observed runs for those shapes at one shot).

## What was built

**Stage B, #878** (plugin), `develop` @ `fba35f24`:

- One `minimax-h3` skill line under *Which shape is the request*, after
  *Pinned to a picture*: "A locked frame that lip-syncs to a song (FL2VA
  takes no audio): Ref2VA `[keyframe completion + audio reuse]`,
  `<Picture 1>` the first frame, `<Audio 1>` the song, held with
  `hold_audio`", plus the framing-not-pixels and last-frame limits.
- Room under the 12,288 B cap came from rewording, with no rule dropped
  (12,259 B after).
- `tests/test_plugin_skills.py`:
  `test_the_locked_frame_lip_sync_recipe_names_real_task_types` pins the
  task string, "FL2VA takes no audio" and `hold_audio` in the skill, both
  task types in `dw/workflows/h3_context_ir.json`, and `hold_audio` on
  `music-video`'s ref2va step.
- No plugin version bump: the plugin's version is the engine's and the
  release script bumps both (`test_the_plugin_version_is_the_engine_version`).

**Stage C, #879: not built.** The recipe lives entirely in the prompt: in
Ref2VA a keyframe is an image reference the prompt names as the first
frame. `templates/minimax/music-video` already takes it with arguments: the
still as `singer_reference` (or an entry's `singer`), the song as `song`,
the task in the entry's `prompt`, and the slice held by `hold_audio`. That
configuration validated on lem with no warnings, eliding `draw_singer` and
`write_song`. A new template or per-entry option would duplicate it.

## Bounces per stage

| Stage | Bounces |
|---|---|
| A, #877 | 0 |
| B, #878 | 0 |
| C, #879 | n/a (not built) |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
GPU time on lem: 48.4 min across the four renders, plus prep jobs
`b215107ef20a`, `a3fe84296132` and `f40f7f4ac95d`.

## Deferred

- **Pointing the skill line at `music-video`.** The line names no template;
  a short follow-up could name `templates/minimax/music-video` with the
  still as `singer_reference`, if it fits under the cap. Left for Don to
  route (#879).
- **C-F409 to C-F411** (stage C's cases) are to be retired by the curator,
  since #879 closed not planned.
- **Catalog cost for one-shot FL2VA and `music-video`**: both quoted
  `unknown` here. Runs 3 and 4 are now observed runs on lem.
