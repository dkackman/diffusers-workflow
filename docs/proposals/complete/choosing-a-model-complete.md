# Choosing a model: preference-first routing (#1627, stage A)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v2 on #1627 is the full design. This file records what
shipped, what was deferred, and why.

**Result:** an agent asked for a picture with no family, workflow or model named
now runs `templates/preferred/text-to-image`, which composes
`preference:text_to_image` (#1626) at the tier asked for. SD 1.5
(`templates/text-to-image`) is the smoke test only. A new plugin skill,
`choosing-a-model`, and a matching *Choosing a model* section in
`docs/AGENT_RULES.md` carry the rule. The UI half of the issue (rung display,
preferences editor, "make a better one") is deferred to a next plan.

## The problem

Once #1626 shipped a `preference:` reference and #1625 a shipped
`text_to_image` ladder, nothing told an agent to use them: with no family skill
firing, agents fell back to SD 1.5, and the docs taught SD 1.5 as the first
picture. #1621 (field report) showed agents guessing a tier per shot. The issue
also asked for a UI tier picker, a Settings preferences editor, and a
"make a better one" step after the first-run picture.

## Verdict: build smaller

Don approved with the delegate's recommendations (2026-11-25): build stage A
only, then bring the next plan (B-D) back to him. Stages B and C need #1626's
Q1 (per-tier host resolution on `GET /api/preferences`), which is ruled on
#1626, not here.

Design corrections the plan made against the issue:
- AGENT_RULES can't take a free-standing section (each must copy a skill,
  `tests/test_plugin_skills.py`), so the rule became a new skill plus its
  AGENT_RULES section.
- The tier picker is the existing `tier` preset dropdown (`VariablesForm`),
  annotated, not a second control.
- "Make a better one" belongs on the job page, not the first-run card, which has
  no success state.

## What shipped (stage A, #1724)

Built on `feat/1627-A-choosing-a-model` (`8aebf8cf` template, `5c311a26`
skill), merged to `develop` at `7be55d62`; `deploy: server` and `deploy: plugin`.

- **`plugins/dw/skills/choosing-a-model/SKILL.md`** (4.6 KiB). Rules: a named
  family, workflow or model wins; "does it work" runs the SD 1.5 smoke test;
  anything else runs `templates/preferred/text-to-image` at the asked tier, or
  with no tier so the store's default applies; before spending, read the
  `preference:text_to_image (tier …) -> …` line in `validate_workflow`'s
  `plan.preferences` and tell the user what will run. It does not name
  `get_preferences` (#1723 was not verified when it was written).
- **`docs/AGENT_RULES.md`** *Choosing a model* section naming the skill; the
  `agent-rules` summary in `dw/server/guides.py` names it too.
- **`workflows/templates/preferred/text-to-image.json`**: `prompt`, `seed`,
  `width`, `height`, and `tier` (`draft|standard|best`, default `standard`);
  one step composing `preference:text_to_image`. Standard resolves to
  `models/z-image`, draft to `templates/text-to-image`; `tier=ultra` is refused
  at `arguments.tier`.
- **Docs:** the skill is named in `plugins/dw/README.md` and `CLAUDE.md`. The
  root README's first image is now "4. Check it works" (smoke test) followed by
  "5. Make a real picture" (the preferred template); `CLAUDE.md`'s `dw.run`
  comment and system-test line read SD 1.5 as the smoke test.
- **Tests:** `tests/test_preferred_template.py` (new, runs the real resolver);
  `tests/test_plugin_skills.py` AGENT_RULES map, pins and the server-first
  count (11); `test_server_guides.py`, `test_step_cache.py`,
  `test_catalog_structure.py`.

Deviations the build recorded:
- `tier` is a `presets` selector that also sets size: 512x512 at draft (SD 1.5
  native), 1024x1024 at standard and best. An explicit `width`/`height` wins.
- The template passes `num_images_per_prompt: 1`, because z-image defaults to
  4. MiniMax-H3-Image declares no such variable, so `tier=best` on CUDA
  validates with a "declares no variable … dropped" warning; the run still
  makes one image.
- The template declares `"shape": "image"`; derived, a composed step reads as
  `image-set`.
- `tests/test_examples.py` skips `preference:` paths in its
  reference-resolves check. `COMPACT_BUDGET` rose 15_875 → 15_975 (measured
  15_951) for the new catalog entry.
- The plugin version was not bumped by hand (`scripts/release.sh` does it).

## Deferred, and what brings it back

- **Stage B, rung display beside the `tier` dropdown** (each rung's resolved
  target, cost, greyed skips with reasons). Needs `GET /api/preferences` to
  carry per-tier host resolution (#1626 Q1) and #1626 stage 2 (#1723)
  verified. Brought back by the next plan on #1762 (parked, blocked by #1723).
- **Stage C, Settings → Preferences editor** (per key/tier target lists,
  shipped vs server badge, reset-to-shipped, add key, etag saves with
  `ConflictNotice`). Needs #1723's REST routes and harnest#254 (ui-check PUT
  stubbing). Same trigger.
- **Stage D, "Make a better one"** as a job-page promotion on a completed
  `templates/text-to-image` job, submitting `templates/preferred/text-to-image`
  with the same prompt and seed. Its template shipped here; only the promotion
  and the first-run card copy remain.
- **`get_preferences` in the skill.** Added once #1723 is verified.
- **The H3 *Ramp* re-point** (#1621 item 1), cut (Q5): #1625 cut the video
  keys, and the Ramp's rungs are argument bundles on reference-to-video /
  dialogue-short, not i2v/t2v targets. Brought back by `image_to_video` /
  `text_to_video` keys shipping; then it is one plugin-only stage.

## Known limits

- A skill with no family trigger may not fire on every no-family request;
  the AGENT_RULES section is the backstop.
- `tier=null` is refused as `'None' is not a 'tier' value` (Python repr, not
  `null`): cosmetic, noted by the tester.

## Stages, bounces and cost

| Stage | Bounces | Notes |
|---|---|---|
| #1724 `choosing-a-model` skill and `templates/preferred/text-to-image` | 0 | Merged to `develop` at `7be55d62`; architecture review pass; tester verified on the first pass. |

The stage comments don't name `usage:` figures, so cost isn't recorded. The
plan estimated $2-3.

Verification (develop @ 7be55d62, lem, RTX 3090, workspace
`regression-complete`): C-F1423 to C-F1428 pass. Draft ran
`77d1d08089c7` (`@draft -> templates/text-to-image`), default ran
`8bba5549fd0c` (`@standard -> models/z-image`), one image each; `tier=best`
resolved `models/minimax-h3-image`; draft with 768x512 was applied, since SD
1.5 now declares `width`/`height` (#1625).
