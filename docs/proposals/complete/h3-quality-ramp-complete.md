# H3 quality ramp: draft -> final tiers (#862)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-15). Don approved plan v2 on 2026-10-14. Its one stage, #882,
shipped on 2026-10-15 and is at `develop` @ `426b764b`. It was verified on
lem (CUDA) against the `regression-complete` workspace.

## The idea

Nothing in the plugin told an agent how to spend GPU time on quality. Every
H3 render before Don's quality matrix used int4 + a turbo LoRA, because the
templates default to that, and Known Issues faces smeared (2-2.5/5). The
matrix on lem (2x RTX 3090) showed a large, predictable trade-off. The base
model (no LoRA) at int8 gave photographic faces (4/5) at 2-4x the time.
Within one staging family, a cheap draft carried its staging to the final
at the same seed. The issue proposed a cross-cutting `quality-ramp` skill:
which tier to draft at, what to approve there, and what never to change
between tiers.

## Verdict and scope

**Build smaller.** No new skill and no new file. The ramp is a `## Ramp`
section in the existing `plugins/dw/skills/minimax-h3/references/quality-and-speed.md`.
It expands that file's "Draft then final" bullet. #863 had already put the
ladder, the speed levers and the composition invariants there, so the
section adds only the process around them.

Why not a top-level skill: only H3 has a measured ladder, and `minimax-h3`
already loads on every H3 request. A new skill would cost up to 12 KB per
trigger and rows in the README, CLAUDE.md and the skill test.

Corrections the plan made to the proposal:
- **TV path L1 -> L7, not L0 -> L7.** L0 changes the canvas. L0 stays as a
  throwaway sketch. Plan v2 first said L0's staging doesn't carry. #861 then
  measured L0 -> L7 holding on 3 of 3 seeds, so Don corrected the build:
  L0 is for blocking only because it smears faces, not because its staging
  moves.
- **No pin test.** The numbers and the advice sit in one file.
- **The base rung doesn't clear `lora_disabled`.** The warning is
  unconditional on a null LoRA (`dw/adapter_compatibility.py`). The section
  calls it expected on both cinema tiers.

## Decisions (Don's review of plan v1, 2026-10-14)

- **Q1.** A section in the existing `quality-and-speed.md`, not a new file
  or skill.
- **Q2.** Wait for #861's seed runs before stating carryover claims.
- **Q3.** Quote `plan.estimate` once #866 lands, and the ladder until then.
- **Q4.** Ship L7 as the TV final, marked "not confirmed by ear or in
  playback".
- **Ruling on dkackman/harnest#116:** #865 landed first, so P1 is L5 plus
  the template's `cache` argument, and `sage_hub` is an `attention_backend`
  argument on every H3 template. The plan's saved-copy path was dropped.

## What was built

### Stage 1, #882: the `## Ramp` section

- **`quality-and-speed.md` `## Ramp`:**
  1. *Pick the path*, a table by target. TV: L1 (~5.4 min) -> L7 (~7.2 min);
     only `weights_dtype` changes. Cinema: B3 (~12.9 min) -> L5 (~29.5 min)
     or P1 (~23.1 min, + `cache` `{"type": "first_block", "threshold": 0.05}`);
     only `num_inference_steps` changes, plus `cache` for P1. Close-up: ship
     C1. Each rung's arguments are spelled out. L0 is a sketch, B4 is never
     a cinema draft (2 of 4 seeds).
  2. *What carries*: a pointer to the *Quality ladder* rules, plus
     identical seed/canvas/prompt/references and "blocking carries, props
     sometimes don't", each with its #861 seed count.
  3. *What to approve* at draft and at final; re-roll only at the draft.
  4. *How to judge*: face crops and a contact sheet via
     `get_output_frames`, `check-script`, a 1-5 score with `set_note`, and
     what Claude can't judge (audio by ear, motion in playback).
  5. *Cost*: `plan.estimate` (#866), but the rung's ladder minutes when the
     estimate is `low_confidence`, which every `sage_hub` rung is.
- **`dialogue-short` at 1344x768** (the bounce fix): the default `shots[4]`
  (`tag`) is 141 frames with 4 references, over the 124-frame ceiling. The
  section says to copy `variables.shots`, set `shots[4].num_frames` to 124
  (or drop a reference), and pass the list unchanged at both tiers.
- **`minimax-h3/SKILL.md`:** the "Smeary faces" line reworded to cover
  draft -> final tiers, with no new line (12,278 of 12,288 bytes).
- **`script-to-video/SKILL.md`:** a one-line pointer for H3 shots in the
  test-render passage.
- **Deviations:** no plugin version bump, because `plugin.json`'s version
  must equal `pyproject.toml`'s and `scripts/release.sh` moves both. The
  README and CLAUDE.md are unchanged.

## Verified

Cases C-F414 to C-F419, in `regression-suite-complete.md`; all six now
pass, with no GPU spend (`validate_workflow` only). C-F416 to C-F418
validate B3/L5/P1 on `dialogue-short` with the 124-frame `shots`, each with
one `lora_disabled` warning per shot. The tester also checked the "drop a
reference" option: a 3-reference `tag` at 141 frames validates.

Whether a session *follows* the section isn't tested over MCP. Don judges
that on his next H3 episode.

## Bounces per stage

- **Stage 1, #882: 1.** C-F416, C-F417 and C-F418 failed: the cinema path
  sent `dialogue-short` to 1344x768, where its default 141-frame `tag` shot
  is refused by the VRAM ceiling. Fixed in `6834d687` (`426b764b`) with the
  `dialogue-short` paragraph above. Docs only; the template's default was
  left alone (960x544 fits it).

Stage costs aren't recorded, because the stage comments give no `usage:`
figures. The plan estimated about $2.

## Deferred

- **MagCache as the cinema-final cache.** After #882 was built, #875 and
  #927 measured M2 (MagCache 0.06 with calibrated H3 ratios, 18.1 min,
  -39% against L5), and Don chose it over `first_block` 0.05; `first_block`
  0.08 (P6, 19.5 min) is the fallback. The section still names P1 as the
  cache final, and the file's closing "Open:" line still lists
  `first_block` at 20 steps, P1 on a close-up or a second seed, and
  MagCache calibration as open. All are answered on #862's body. Updating
  the section needs the 29 ratios as a `cache` argument, so it is a
  follow-up edit with its own check, not a close-out doc fix.
- **A top-level `quality-ramp` skill.** What brings it back: a measured
  ladder for a second family (LTX-2.5, Flux). The family-agnostic half
  (tier rules, approval per tier, judging) would then move up into
  `skills/quality-ramp/`, and each family keeps its own ladder.
- **The 12-step base floor** stays advice (B6 at 8 steps duplicated a
  character; 10 was never tried). An engine rule is #868's territory.
- **L1 = L7 on more seeds.** The TV pair itself is measured on one seed;
  the section says so.
