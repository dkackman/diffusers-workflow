# Finishing a cut in one job, and placing a score on a beat (#1622)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v1 on #1622 is the full design. This file records what
shipped, what was left out, and why.

**Result:** the plugin now shows how to cut, score, grade and grain in one job
by composing `templates/assemble-and-score` instead of copying its steps, and
how to set `score_start_frame` so a score's swell lands on a story beat. Plugin
docs and tests only: no engine, MCP, REST, catalog or workflow-syntax change.

## The problem

From #1523's field report ("Whitefish Point", lem, guidance points 3 and 4).
No template chains assemble, grade and export, so the agent hand-copied
`assemble-and-score`'s steps, the `grades.md` `apply_lut`/`film_grain` steps
and `export_timeline` into its own `fitz-light/cut`. That hand copy is also
where #1613 came from. Nothing said how to offset a score so its arc meets a
beat either: the agent read the envelope peaks and picked
`score_start_frame` 192 by hand.

The issue asked for a finishing template, or optional `grade`/`export`
arguments on `assemble-and-score`, plus a placement paragraph.

## Verdict: build smaller

The gap was knowledge, not machinery. A `workflow` step already composes a
catalog template by name (`docs/WORKFLOW_GUIDE.md`), and `apply_lut` and
`film_grain` take the composed AudioVideo and keep its audio
(`dw/tasks/finish_handlers.py` `_load_media`). So the plan wrote a recipe.

## What shipped (stage 1, #1700)

- **Finish the cut in one job**, a new section in
  `plugins/dw/skills/long-form/references/run-and-repair.md`, just before
  *Export the cut for an editor*:
  - step `cut` composes `templates/assemble-and-score` by catalog name, with
    `match_levels: "lufs"` as `audio.md` recommends, saved to `intermediate`;
  - step `graded` runs `apply_lut` on `previous_result:cut` with the
    `teal-orange` palette and strength from `cinematography/references/grades.md`;
  - step `grain` runs `film_grain` with teal-orange's grain values, saved to
    `final`;
  - it says to name the template rather than copy it ("a copy stops tracking
    the original").
- **Export as a second job.** #1613 now refuses an in-memory `previous_result:`
  for `export_timeline` at validate, so the recipe and *Export the cut* both say
  export runs as its own job with `{"video": "output:…"}` naming the grained
  file in `final`.
- **Placing the score on a beat**, a bullet in
  `long-form/references/audio.md` after the `assemble-and-score` entry. Read
  `get_gallery_metadata(envelope=true)` at a fine `interval_seconds`, take
  `t_peak = i × interval_seconds` at the swell's bin, and with the beat at cut
  frame `B` set `score_start_frame = round(t_peak × fps) − B`, counted at the
  template's `fps` the way its `slice_audio` step reads it. A negative result
  means the swell can't land that late (pick another beat or score), and the
  score must run at least `score_start_frame + total_frames` frames (else
  `slice_past_end`).
- **Pointers** from `cinematography/references/grades.md` (*Applying one*),
  `long-form/SKILL.md` §5, `script-to-video/SKILL.md` §6 and `audio.md`.
- **Tests:** `TestFinishTheCut` in `tests/test_plugin_skills.py` loads the
  recipe's fenced JSON against a copy of the real `assemble-and-score` in a tmp
  catalog and asserts no validation errors and no `sub_workflow_warnings` (an
  undeclared argument). It also pins the `teal-orange` look, the
  name-don't-copy and second-job wording, the four pointers, and the placement
  rule against the template's `slice_audio` arguments.

Deviations the build recorded: no manual plugin version bump (the plugin
version tracks the engine's and `scripts/release.sh` bumps both), and nearby
lines in `long-form/SKILL.md` §5 and `script-to-video/SKILL.md` §6 were
reworded to fit their byte budgets, with meaning unchanged.

## Left out, and what would bring it back

- **A finishing template.** It would be a third near-copy of the assemble
  chain unless it composed `assemble-and-score`, in which case it is this
  recipe plus a catalog entry, a `COMPACT_BUDGET` raise and a measured `cost`
  row. Trigger: a second field report of a session hand-copying the chain or
  failing to compose it after this recipe shipped. It would then compose
  `assemble-and-score`, not copy it.
- **`grade`/`export` arguments on `assemble-and-score`.** Steps can't be
  conditional, and `apply_lut` refuses with no lut and no palette, so an
  optional grade means grading every cut by default or an engine change. It
  would also spend `FILTERED_BUDGET` headroom and need the same change in
  `dissolve-between-shots`.
- **Export as the recipe's last step.** Not possible while #1613's ruling
  refuses `previous_result:` for `export_timeline`; the recipe follows that
  ruling.
- **Peak detection in `get_gallery_metadata`** was a non-goal: the envelope
  stays per-bin levels.

Don's delegate approved plan v1 at the default on 2026-11-25 (Q1: the recipe in
`run-and-repair.md`).

## Known limits

- The rule finds the loudest bin, not a musical downbeat. A broad swell gives
  a broad bin; the paragraph says to use a fine `interval_seconds` near it.
- Proposed follow-up, not filed: `assemble-and-score`'s description doesn't
  name `match_levels: lufs`, which `concat_videos` supports and the recipe now
  uses.

## Stages, bounces and cost

| Stage | Bounces | Notes |
|---|---|---|
| #1700 finishing recipe and placement paragraph | 0 | Built as 69b36350, merged to `develop` as ca1ef1e6, `deploy: plugin`. Tester verified on the first pass. |

The stage comments don't name `usage:` figures, so cost isn't recorded. The
plan estimated $2–3.

Verification (develop @ ca1ef1e6, lem, workspace `regression-complete`):
C-F1312, C-F1313 and C-F1314 pass. In C-F1313 the cut and the final were each
248 frames with identical `pcm_sha256` audio, frame 60 visibly regraded, and
the second-job export gave an `.otioz` with both shots in order. C-F1315
(placement) passes with an adapted fixture: at `gain_db` 24 the swell stood
only about 3 dB over the ep11 bed's ticks, so it was rerun at 40 dB, and both
beats (B = 120, B = 200) peaked within the required bin with 17–19 dB margin.
The fixture amendment is dkackman/harnest#303.
