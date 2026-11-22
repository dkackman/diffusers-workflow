# Intent-keyed index: result → template + pattern + bench evidence (#1519)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-22). Don filed the idea in November 2026. An agent that starts from a *result*
("change the environment, keep the motion", "hold this face across shots") had no
single place to look up which template and prompt pattern gets it, or whether that has
been measured. The know-how was spread over about 10 skill reference files. Don raised
it to `priority:2` for the announcement release (#1523), which needs one place where a
reader can see the bench evidence behind each claim. Plan v1 is the plan comment on
#1519. This file records what was built, what was cut and what stays.

## Verdict: build smaller (two stages)

Two plugin reference files and one test, covering H3 and LTX-2.5 only. Don approved
plan v1 as written on 2026-11-21, with every question at its default:

- **Q1:** the index lives in each family skill as `references/intents.md`, shipped
  with the plugin. It is not a skill of its own and not a `get_guide` section.
- **Q2:** no `list_workflows` intent filter.
- **Q3:** the session that lands a bench record adds its row, and the test enforces
  that. There is no curator role.
- **Q4:** the evidence rule is strict. A `confirmed` or `refuted` row cites a repo file
  under `bench/records/` or `docs/proposals/`. A result measured only in an issue
  thread is `untested`, with the issue named.

## What was built

**The row format.** `intent | template | pattern | evidence | status`.
- **intent:** the result, in the user's words.
- **template:** a full backticked catalog name, or empty for a prompt-only pattern.
- **pattern:** a backticked file plus `§ <heading>` in the owning skill, which may be
  another skill such as `long-form`. The row never restates a pattern or a number.
- **evidence:** repo paths, or for an `untested` row, the source.
- **status:** `confirmed`, `refuted` or `untested`. A refuted row stays, so the same
  idea isn't tried again.

**Files.**
- `plugins/dw/skills/minimax-h3/references/intents.md`: 22 rows from stage 1, plus 13
  appended in stage 2 to cover the H3 records stage 1 didn't cite. Row 1 is the
  environment/weather swap with its motion kept, `untested`, "community: Reddit, via
  #1518". #1518 promotes it when its record lands.
- `plugins/dw/skills/ltx-2.5/references/intents.md`: 20 rows. They cover chaining past
  one clip, extend, voice steering, repair, look_hold, carry_pick, audio dropout, alpha
  matte, two-stage, reference sheets and extend-clip's `clip` input.
- In each `SKILL.md`, under *Which shape is the request*, the line "Starting from a
  result rather than a template: `references/intents.md`."
- `plugins/dw/README.md`, one sentence naming `references/intents.md` as the per-family
  index.
- `bench/README.md` *Records*: a new record owes its `intents.md` row in the same
  change.

**`tests/test_plugin_intents.py`** reads every `plugins/dw/skills/*/references/intents.md`.
Rows are parametrized, and every failure message names the row
(`<skill> intents.md:<line> ('<intent>')`). The test checks:
1. five columns, a status in the closed set, and a template cell that resolves under
   `workflows/`;
2. a confirmed or refuted row cites a `bench/records/` or `docs/proposals/` path, and
   every cited path exists;
3. the pattern file exists and contains the named heading;
4. every `bench/records/*.md` is cited by some row or is on `EXEMPT` with a one-line
   reason. `test_exempt_record_exists_with_a_reason` keeps that list honest: every
   entry needs a reason and an existing file, and can't also be cited. It holds seven
   records: h3-image, longcat-avatar, face-encode-floor, noise-floor, determinism,
   for-don and h3-hybrid.

## Deviations found in the build

- **Freeing bytes in the H3 `SKILL.md`** (12 KiB cap): the performer-limits clause
  can't move, because `test_replace_performer.py` pins it in SKILL.md (C-F891). The
  bytes came instead from two pointers whose content is in the file they name. No rule
  or number changed. The file went from 12288 to 12253 bytes.
- **Two "issue-only" seed rows have records.** Odd shots, then bridges (#1094) cites
  `complete/h3-odd-then-bridge-complete.md`. `mask_grow` 0 (#1374) cites
  `bench/records/performer-replacement.md`. Both are `refuted`, so Q4's strict rule
  cost fewer rows than the plan expected.
- **Exempt list:** three records beyond the plan's named four. `determinism` is
  infrastructure. `for-don` is a digest whose results are cited at their own records.
  `h3-hybrid` is an engine experiment with no template, and its verdict is still
  pending Don.

## Where the records and the skills disagree (flagged, left for follow-up)

None of these were changed in this feature.
- **Dialogue drops the turbo LoRA** (H3 `SKILL.md` § Hard rules, "#984 B0"): no repo
  record holds the measurement, so the row is `untested`.
- **The LTX voice F0 numbers** in `ltx-2.5/SKILL.md` § Prompts come from commit
  `ebe8d215`, which has no record, so both voice rows are `untested`.
- **The draft tier's saving:** `long-form/references/quality-and-speed.md` says 28-44%.
  `bench/records/tiers.md` gives both 28-34% (:135) and 28-44% (:236).
- **Three refuted records have no skill line:** `h3-stochastic-sampling`, `speed` and
  `ref-prep-blur`. Their rows point at the nearest heading. Two of the records say
  outright that no skill line should land.
- **LTX `carry_pick`** is refuted in `bench/records/ltx.md`, but every `carry_pick` line
  in the skills is H3's. The row points at `long-form/references/levers.md` § Measured
  and cut.
- **Left out of the index:** `music-video` holding audio for lip sync (#795/#808) and
  the H3 staging rules (#484). Both are issue-only.

## Stages

| Stage | What | Bounces |
|---|---|---|
| #1555 | H3 index, format test checks 1-3, link line, README | 0 |
| #1556 | LTX-2.5 index, check 4 + `EXEMPT`, 13 H3 rows, bench README | 0 |

The tester verified the served plugin text for each stage. A docs review then passed
the repo-side items: every cited path exists, sampled records state their rows'
outcomes (9 rows for stage 1, 10 for stage 2), and the test logic was read. The stage
comments give no `usage:` figures, so cost isn't recorded. The plan estimated about $7.
Neither stage used GPU time.

## Not built, and what reopens it

- **`list_workflows(intent=…)`:** reopens if a `field-report` shows a session picked
  the wrong template while the index held the right row.
- **A `get_guide` copy:** reopens if a non-plugin MCP client becomes a real user.
- **Other families:** each one, once it has two or more records to cite.
- **A typed-decision pre-render check** against `refuted` rows (Don's comment on
  #1519): this needs enough labelled rows to replay offline, and a self-hosted model.
- **A cross-family "which family for this result" index:** that stays
  `script-to-video`'s decision tree.
