# H3 AISkills claims: style-skills pointer and three probes on lem (#1059)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 approved by Don on 2026-10-30 ("approve with
defaults", Q1–Q4) as "build smaller"; three stages, #1070 (style pointer),
#1071 (probes), #1072 (rules), all verified. The probe record is
`docs/proposals/audits/2026-10-h3-aiskills-probes.md`; this doc is the
design record around it.

## The idea

einhorn13/AISkills publishes two H3 prompting guides (`MiniMax_H3_prompting_v20.md`,
`MiniMax_H3_prompting_deep_v3.md`). The repo has no license, so nothing was
copied; only its claims were tested. Most of v20 repeats MiniMax's own
`h3-prompt-writing` skill, which `minimax-h3` already points at. The issue
proposed two things:

- **Part 1:** point the skill at MiniMax's eight style packs in
  `MiniMax-AI/MiniMax-H3/skills/`, with the deep guide's "name the medium,
  then its physical traits" line.
- **Part 2:** probe seven claims on lem before any became a rule.

## Decisions

- **Plan v1 verdict: build smaller.** Probe 3 of the 7 claims: P2 (a
  timbre-only audio reference replays its source words), P3 (a style image
  pulls in its person and composition), P5 (screen side plus anatomical side
  holds handedness). These touch a shipped template (`voice-timbre-reference`)
  or #484's field-report classes.
- **Deferred (Q2 default):** P1 (FL2VA anime at ~12 unique fps), *comes back
  on* a cel/anime request or choppy motion in a field report; P4 (POV breaks
  on an outside description), *comes back on* a POV request or field report.
- **Dropped:** P6 (ongoing motion vs settle with a new last frame per
  segment), because dw chains carry the last frame forward as the next first
  frame and pin no end image (#985 claim 2), *comes back on* keyframed chains
  or a seam snap in a field report. P7 (positive targets beat negations) was
  already settled by the 2026-10 dialogue probes, rows 10b and 12.
- **Q1:** a new `references/style.md`, with `SKILL.md`'s Sources wording
  trimmed to pay for the link. **Q3:** the medium line ships unprobed,
  attributed to MiniMax's packs. **Q4:** ~60 GPU minutes, both P2 clips.

## What the probes found

From the audit doc (2 seeds per arm, 1001 and 2002; held only when both
broken seeds show the effect and neither followed seed does). Configuration:
960x544, 124 frames, int4 sdnq (VAE int8), 24 fps, turbo LoRAs per template.

| # | Claim | Verdict |
| --- | --- | --- |
| P2 | timbre-only audio reference replays its source words | **refuted**: on all four takes `check-script` heard only the new line, no `line_mismatch`, no source word from either clip |
| P3 | a style image pulls in its person and composition | **refuted, weakly**: no person, pose or window on any seed, but no watercolour carried over either; H3 ignored the reference under this phrasing |
| P5 | bare anatomical side misses; "on screen right" holds it | **refuted**: all four clips put the left hand on screen right |

## What was built

**Stage 1, #1070** (plugin only): `e745bca9`, develop `7735379d`.
- New `plugins/dw/skills/minimax-h3/references/style.md`: the eight packs
  by bare name, one line each on the request it suits; the per-pack install
  `npx skills add MiniMax-AI/MiniMax-H3 --skill <name>`, the wildcard
  described and never advised.
- The medium line was kept: the build read three upstream packs
  (`papercraft-stop-motion-explainer`, `paper-collage-explainer-generator`,
  `handdrawn-live-video-generator`) and each prompts that way. It is
  attributed to them, marked unprobed on dw, and stated not to override
  the guide-clip rule (a style prompt does not restyle a guide).
- `SKILL.md` links it from *Prompts* (12,286 bytes, under the 12,288 cap);
  `plugins/dw/README.md` agrees on the install form and pack count.

**Stage 2, #1071** (docs only): `7495862f`, develop `e7cc1545`. The audit doc;
12 probe jobs, 2 reference images and 4 `check-script` runs in workspace
`qa-h3-aiskills-probes` on lem. All succeeded; nothing is marked not run.

**Stage 3, #1072** (plugin only): `69d18760`, develop `24eb678d`. All three
verdicts were refuted, so no rule landed and `voice-timbre-reference` is
unchanged:
- P2: a bullet in `references/dialogue.md`'s *Refuted on both seeds* list.
- P3: a new *Probed and not a rule* section in `references/per-character.md`,
  keeping the "H3 ignored the reference" caveat.
- P5: the same section in `references/staging.md`, after #484's rules,
  with the probe's scope limit (one person facing camera, one prop, static).
- `SKILL.md` got no new bytes.

## Method deviations (from the audit doc)

- Reference images came from `templates/text-to-image` (cached, 512px), not
  FLUX 3, which needed a 40.3 GB download on lem.
- P3's prompt used H3's image token `<Picture 1>` where the plan wrote
  `<Image 1>`.
- P2's second clip came from the template's own Bark `voice` step with new
  words, so the clips differ in words, not voice.
- P5 ran `video-with-audio`'s defaults (4-step v1.2 fl2v turbo, 5 steps,
  shift 6), unlike the 2026-10 dialogue probes' 8-step settings.

## Bounces per stage

| Stage | Bounces | Cause |
| --- | --- | --- |
| 1, #1070 | 0 | The tester passed M-F189–M-F192 and sent the byte cap and test run to docs review, which closed it. The 7 local test failures are `TestKandinsky6Skill`/`TestFlux3KleinSkill` needing a newer diffusers than the agents' Mac has, unrelated. |
| 2, #1071 | 1 | M-F193: the ticket carried a summary of the audit doc, not its configuration. Cleared by posting the doc's header, configuration and verdict rows verbatim; nothing re-run. |
| 3, #1072 | 0 | |

No stage comment names `usage:` figures, so per-stage session cost is not
recorded. The plan estimated ~$10 (stage 1 ~$2, stage 2 ~$5, stage 3 ~$3)
and ~60 lem GPU minutes; the audit doc records no GPU total.

## Not built, and why

- **Any rule from P2, P3 or P5:** all refuted. P3's refutation is weak (the
  reference had no visible effect at all); *comes back if* a field report
  shows a style reference's subject leaking, or a phrasing that makes H3
  take a style reference's medium.
- **A `voice-timbre-reference` prompt change:** gated on P2 holding.
- **P1, P4 (deferred) and P6, P7 (dropped):** triggers above.
- **The guide's style recipes, A/B/C ratings and ComfyUI embeddings:**
  unlicensed and untested by us, or not applicable to diffusers.
- **A frame-cadence check** (what P1 would need): not built; P1 is deferred.

## Standing cost

The style-pack names are bare and untested; a MiniMax rename shows up only
when someone installs one.
