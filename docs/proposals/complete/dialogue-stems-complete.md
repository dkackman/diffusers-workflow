# Dialogue/rest stems: htdemucs `vocals` plus a three-track re-mix (#1091)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 approved 2026-10-31 under Don's delegation, with Q1–Q4
at the plan's defaults; plan v2 folded those answers in with no scope
change. One stage, #1096, verified 2026-10-31 (mini-ai, then lem for the
kept stems). Stages B–C (a dedicated separator) were conditional and were
not built.

The concept came from r34vtraining/Longshot_Studio (its `build_stems`,
MelBand RoFormer dialogue/sfx stems, assessed 2026-10-31). That repo has
**no license**, so nothing was copied.

## The report

A short film's dialogue can't be levelled, cleaned or replaced separately
from its sound bed, because the world track is one stem. The issue asked for
a speech-vs-rest separator (MelBand RoFormer or a DnR-trained model) in place
of htdemucs, which was trained on songs.

## Verdict

**Build smaller: measure first.** The evidence for a new model was thin (no
field report, no film workspace using `separate_stems`, the nearest past
problem #340 solved by ducking), and a new separator brings a dependency
outside diffusers, a pickle boundary (most RoFormer checkpoints are `.ckpt`)
and unverified weight licences. So stage A measured htdemucs as a
dialogue/rest splitter with tasks that already exist, and a new separator was
gated on Don hearing it fail.

## What the plan found against the issue

- **A `model` argument on `separate_stems` is the wrong shape.** Result keys
  are never checked statically (`previous_result:step.key` resolves at run
  time), so `stems.vocals` on a dialogue model would validate clean and fail
  mid-run; and the `stacked_vocals` check indexes `stems["vocals"]`. Had a
  separator been built, it would have been its own command,
  `separate_dialogue`, with fixed `dialogue` / `rest` results.
- **`match_levels` is a concat/dissolve argument, not a task**, so the
  issue's "recut-bed-match_levels pass" is the skill's ducking/bed advice.
- **`dialogue-short` has no audio pass**: it gains nothing until a scoring
  step follows it. The split point is `assemble-and-score`'s world track.
- **A doc inconsistency:** `docs/TASKS.md` said htdemucs came from HF
  `adefossez/HTDemucs` in one place and from outside the HF cache in another.

## Decisions

- **Q1–Q4 (approval, 2026-10-31):** Don judges stage A by listening, with
  the numbers beside the verdict; no `dialogue_gain` template variable;
  pickles refused; vendor the minimal module, licence permitting (the last
  two applied only to stage B).
- **Listening verdict (Don, 2026-10-31): good enough.** Don listened to the
  mix, `vocals` and rest for button, tag and deflect. htdemucs stands; B–C
  are not built, and #1091 ends with stage A.

## What was built

**Stage A, #1096: the A/B record and the recipe** (plugin and docs only).
- The record, `audits/2026-11-dialogue-stems.md` (lem, workspace
  `dialogue-stems-1096`): three H3 dialogue shots from `known-issues`, each
  mixed with a Music 3 score at gain 0.5, then `separate_stems`, with
  bed-alone and dry-alone control separations. Findings:
  - `vocals` matches the dry shot within 0.1 dB rms and 0.2 dB peak;
  - the score puts -62 to -64 dBFS rms into `vocals`, 30–36 dB under the
    bed;
  - the rest is the score **plus the shot's own SFX and foley**, which
    htdemucs routes to `drums` / `other`;
  - transcribing `vocals` gave the line on 3/3 shots, where whisper-base
    looped on the dry audio on 2/3;
  - no `stacked_vocals` on single-speaker dialogue;
  - a speechless shot gives a near-silent `vocals` stem and an invented
    transcript, so check the stem's level, not the transcript.
  The 15 stems are kept as `asset:dialogue-stems-<mix>-<stem>.wav` there.
- `plugins/dw/skills/series-episodes/SKILL.md`, the *match_levels* bullet:
  besides ducking, give dialogue its own gain in a saved copy of
  `assemble-and-score` — `separate_stems` on `world_faded`,
  `previous_result:<step>.vocals` as the dialogue, a `mix_audio` of
  `drums`/`bass`/`other` as the rest, `mixed` layering
  `[soundtrack_resampled, dialogue, rest]` with one gain each, and
  `balanced` kept after it. It carries finding 3's caveat (keep the original
  score file to re-mix or replace the score) and the ambience caveat below.
- `docs/TASKS.md`: the *Stem separation* section names the dialogue/rest
  use, and *Voice attribution* now gives demucs' own downloader
  (`dl.fbaipublicfiles.com`, torch hub cache) as the weights' source.

## Deviations from the plan

- **Control separations instead of a difference.** `mix_audio` refuses
  negative gains, so "vocals − dry" can't be rendered; the bed and the dry
  shot were separated alone to measure leak each way.
- **`acorn-wars` s04 replaced by `tag`.** s04 held no speech; it is kept in
  the record as the speechless negative.
- **The split point is `world_faded`**, not `world`: it is the step that
  feeds `mixed`, and splitting after the fade keeps the fade on every stem.

## Bounces per stage

**#1096: one.** The first mini-ai verify failed C-F557 and C-F559:
- The tester's chain stopped at `mix_audio`, and a 1.4× dialogue gain on a
  stem peaking at -1.5 dBFS clipped to +1.4. Fix: the bullet says to keep
  `balanced` (`normalize_audio`, -3 dBFS peak) after `mixed`; re-run, it
  lands at -3.00.
- The fixture `asset:qa-cast/ep20-score.wav` is a near-silent ambience bed,
  not a music score, and htdemucs sends it almost whole to `vocals`, leaving
  a near-silent rest. That is the fixture, not the recipe: the bullet gained
  a caveat (a quiet, unpitched bed lands mostly in `vocals`; a shot with
  little foley leaves a near-silent rest), and the tester filed the suite
  amendment as dkackman/harnest#163. C-F557 and C-F559 stay `pending:
  #1096` until it is ruled on.

C-F562 (the kept stems) ran on lem, where the workspace lives.

## Cost

The stage comments name no `usage:` figures, so per-stage spend isn't
recorded here.

## Deferred and left open

- **A dedicated dialogue separator (`separate_dialogue`, stages B–C).**
  Back if a film session hears foley or score leak in `vocals` that matters,
  or needs a clean score from an already-scored mix. The re-plan must first
  find a checkpoint with a licence that permits Don's use and safetensors
  weights at a pinned revision.
- **A three-way dialogue/music/effects split** (BandIt-style). Back if the
  foley-in-the-rest caveat becomes the problem.
- **C-F557 / C-F559's fixture**: harnest#163.
