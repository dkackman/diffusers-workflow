# H3 templates expose `attention_backend` and `cache` (#865)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-14). Don approved plan v1 on 2026-10-14. Stage A, #883, and stage
B, #884, shipped the same day. The last of them is at `develop` @
`340e31d1`. Both were verified on lem (CUDA), and stage A also on mini-ai
(mps).

## The idea

Don's H3 quality matrix on lem (2x RTX 3090 24 GB, workspace `h3-quality`)
had to rebuild workflows with `tools/ladder.py` to reach the cinema
configuration. `attention_backend` existed on only four H3 templates, and
no template exposed `cache`. Both knobs were measured wins: `sage_hub` was
23-33% faster with no quality loss, and `first_block` 0.05 saved 22% at 30
base steps. The issue also proposed a `quality` tier variable that would
set the coupled tier settings (canvas, weights, LoRA, steps, shifts,
cache, attention) together.

## Verdict and scope

**Build smaller.** Part 1 shipped: `attention_backend` and `cache` are now
variables on every H3 template. Part 2, the tier presets, was cut. Once
part 1 landed, every tier can be reached with plain `arguments`, so the
tier table belongs in the `minimax-h3` skill as ready-to-pass argument
sets (#862).

Corrections the plan made to the proposal:
- **`cache` couldn't be a variable.** The raw workflow is checked against
  the schema before substitution, and the schema required `cache` to be an
  object. So the schema had to widen, and the substituted value had to be
  checked at validate time, or `{"type":"bogus"}` would pass validate and
  fail at load.
- **The engine already supported H3 caching.** `MiniMaxH3TransformerBlock`
  was already registered in `dw/cache_blocks.json`.
- **Scope is the H3 transformer steps only.** Z-Image steps, Music3's
  `write_song`, `music.json`, `music-video-cuts.json` and
  `compose-workflows.json` get nothing.
- **sage_hub is CUDA-only, and validation didn't check it.** That became
  #868's `attention_backend_unavailable` warning (see
  `acceleration-warnings-complete.md`).

## Decisions (Don's review of plan v1, 2026-10-14)

- **Q1.** Cut the presets. The tiers live in the skill as argument sets.
- **Q2.** `cache` is a whole-object variable, not a `cache_threshold`
  number. #867's P2 found that installing the `first_block` hook changes
  the output even when it skips nothing (PSNR 25.6 dB at 12 steps). A
  threshold of 0 therefore isn't off. Only `null`, which drops the key,
  turns the cache off.
- **Q3.** Template defaults are unchanged: int4 + turbo LoRA, and both new
  variables `null`. This also answers the issue's int8 question: int8 is a
  tier choice the skill states, not a template default.
- **Notes:** stage B's check joins the expanded-definition checks in
  `dw/validation.py`, so MCP `validate_workflow`, run-time validation and
  `for_each` members all get it. Stage B also replaces #863's "edit the
  JSON" text in `docs/RECIPES_24GB.md` and the skill's
  `quality-and-speed.md` with the argument.

The issue's open question about automatically reducing `num_frames` or refs
at 1344x768 is answered no: the VRAM ceiling warning already reports it,
and silently changing the shot would be worse.

## What was built

### Stage A, #883: `attention_backend` on every H3 template

- **Templates:** 15 templates gained an `attention_backend` variable
  (default `null`, no constraint), wired at the H3 step's transformer
  component in the same form as `video-with-audio.json`. They are
  `first-and-last-frame`, `last-frame-only`, `chained-segments`,
  `composable-references`, the three `chain-*` templates, `enhance-prompt`,
  `enhance-prompt-with-image`, `voice-timbre-reference`,
  `generated-subject-reference`, `storyboard`, and the `shot` `for_each`
  step of `dialogue-short`, `shots-batch` and `music-video`. The four
  templates that already had the variable are unchanged, which makes 19 in
  total.
- **Tests:** in `tests/test_catalog_structure.py`, every template with a
  `MiniMaxAI/MiniMax-H3` step must wire the variable, and only H3 steps may.
  Two `replace_variables` tests run on `dialogue-short`.
- **Docs:** `docs/ACCELERATION.md`, `docs/WORKFLOW_GUIDE.md`,
  `workflows/templates/minimax/README.md`, and the `minimax-h3` skill
  ("CUDA: `attention_backend=sage_hub`; null elsewhere"). `COMPACT_BUDGET`
  went from 10_700 to 10_800.
- **Deviation:** the ref2va templates wire the backend at
  `components.transformer_ref`, as `reference-to-video.json` already did,
  not at `components.transformer`.

### Stage B, #884: `cache` as a whole-object variable

- **Schema:** `pipeline_configuration.cache` in `dw/workflow_schema.json`
  is `["object","string"]` with `"pattern": "^variable:"`, so a literal
  string such as `"first_block"` still fails the raw schema.
- **Engine:** `cache_errors` in `dw/step_value_checks.py` is registered in
  `ERROR_CHECKS` and checks the substituted, `for_each`-expanded value
  against the schema's cache object shape. It reads that shape from the
  schema, and narrows number-or-string fields (today, `threshold`) to
  number, because every reference has been substituted by then. A
  substituted `null` drops the key, so no cache is built.
- **Templates:** all 19 H3 templates gained `cache` (default `null`), wired
  in the H3 step's `pipeline.configuration` only.
- **Tests:** `tests/test_validation.py::TestCacheVariable` and
  `TestCacheVariableOnAForEachTemplate`; catalog wiring tests;
  `test_turbo_schedules_carry_no_block_cache` now checks the effective
  default. `COMPACT_BUDGET` went from 10_800 to 10_850.
- **Docs:** the cache section of `WORKFLOW_GUIDE.md`, the H3 row in
  `RECIPES_24GB.md`, `ACCELERATION.md`, the minimax templates README, and
  the skill's `quality-and-speed.md` (use the cache only at about 20 or
  more base steps, never with the turbo LoRAs; `null` is off). The
  `docs/ARCHITECTURE.md` **Step-value checks** row names `cache_errors`.
- **Deviations:** an invalid cache on a `for_each` step is reported once,
  naming the member, not once per member. `step-caching.json` was added to
  `LEGITIMATE_MENTIONS`. No `breaking-change` label, because the schema only
  widens.

## Verified

Cases C-F420 to C-F429. The CUDA runs on lem: `dialogue-short` with
`sage_hub` (job `4637d362563a`, about 28 min against a 38.6 min estimate),
and a 30-step no-LoRA `dialogue-short` with `first_block` 0.05 at 960x544
(job `a104407184ad`, about 67 min).

Suite follow-ups the tester left open:
- C-F420, C-F422 and C-F423 say `components.transformer` and expect a
  substituted key in `get_job_workflow`, which keeps step references
  unsubstituted. The tester sent the wording to a suite amendment.
- C-F428 at 1344x768 is refused by the VRAM gate on a 3090 and passed at
  960x544. The resize is dkackman/harnest#114.
- C-F417 (pending #882) expects `attention_backend` and `cache` to be
  refused on `dialogue-short`, which no longer holds. Its amendment is
  owed by whoever verifies #865.
- No runtime event names the attention backend actually applied. The
  tester is filing that observability gap separately.

## Bounces per stage

- **Stage A, #883: 0.**
- **Stage B, #884: 2.**
  - **Architecture review (1):** the `docs/ARCHITECTURE.md` Step-value
    checks row didn't name `cache_errors`. Fixed at `92c3952f`.
  - **Tester (1):** C-F425 / C-F426 value 7. A string `threshold` passed
    validate, because the check reused the schema's `["number","string"]`
    type. Fixed in `e07fbf38` (`340e31d1`).

Stage costs aren't recorded, because the stage comments give no `usage:`
figures. The plan estimated $3-4 for A and $4-6 for B.

## Deferred

- **Tier presets** (a `presets` block, Q1). What would bring them back: a
  `field-report` showing a session mis-coupled a tier even with the
  skill's table, or a second family with a measured ladder that wants the
  same mechanism.
- **The tier table as argument sets in `minimax-h3`** is #862's work.
- **Per-device resolution of `attention_backend`.** The default stays
  `null`. #868 warns about a CUDA-only backend on a server that isn't CUDA.
