# Reference-picture preparation on H3 Ref2VA: crop and blur the backdrop (#1439)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). #1439 asked whether preparing an H3 reference picture, the way the
Fantastic MiniMax-H3 PromptBuilder's RefMod "Batch Masking" does (a SAM subject mask, a
crop with a 1.75x margin, the backdrop blurred outside the mask), stops the picture's
backdrop and framing bleeding into a `templates/minimax/reference-to-video` clip. Plan
v2, the plan comment on #1439, was approved on 2026-11-18 with Don's amended gate. It
was built as one bench-only stage, #1460. This file records what was measured.

## Verdict: build smaller, bench only; result: drop

It was a bench probe with no engine code, task, template step or MCP/REST/syntax
surface. It had a bleed gate before the crop and blur arms, and no two-character pair.
The base rate on record is about 1 in 25 three-reference shots
(`plugins/dw/skills/minimax-h3/references/staging.md`), so the arms could only be told
apart if arm 1 bled reliably.

**Result: not reproduced at one reference; drop.** None of the four arm-1 clips bled, so
by the gate arms 2 and 3 did not run, and no line went into `per-character.md`.

## Plan history

- **v1:** three arms (as-is, square crop, crop + blur), a two-picture bleed gate, and an
  optional two-character pair.
- **v2 (approved 2026-11-18, Don's ruling on Q1-Q4):**
  - Q1 [spend]: keep the gate, but start arm 1 by re-running the #484/#487 known-bleed
    shot if its inputs survive on lem. The v1 gate is the fallback. Cap: about 1.5 h of
    lem GPU.
  - Q2 [spend]: no two-character pair. It stays a follow-up.
  - Q3 [none]: mask from `segment` (Grounding-DINO + SAM2). The record names the
    still-image SAM 3 gap.
  - Q4 [none]: a square crop through `recenter_crop`, bounded by #1193 (head and
    shoulders, not a face crop).
- **Corrections the plan made to the issue:**
  - SAM 3 exists only as `track_people(backend="sam3")`, a video task. A still mask comes
    from `segment`.
  - No task combination can blur outside a mask. Arm 3 needed a bench script, and the
    record names five missing tasks.

## What was built

### Stage A (#1460): the probe and the record (bench only)

Merge `44126e05` on `develop`: commits `84467523` (the script and the probe workflow)
and `c32ec17f` (the record). No deploy: lem stayed at `603649f1`.

- **`bench/blur_outside_mask.py IMAGE MASK OUT`:** `--grow 12` (a square maximum
  filter), `--feather 6`, `--blur 20`. It binarizes the mask at 128 and composites
  `alpha*sharp + (1-alpha)*blurred`. It uses PIL, NumPy and scipy only, and has no unit
  tests (bench-only, per the plan). It was built but never run, because the gate failed.
- **`bench/workflows/probe-1439-refprep.json`:** composes the template at its defaults,
  with `prompt`, `subject` (the arm) and `seed`.
- **Record:** `bench/records/ref-prep-blur.md` + `.json`.

**Arm-1 path: the fallback.** The #487 leak shot lived in workspace `encanto-uprising`,
which no longer exists on lem, and #484 names no job id. Two FLUX 3 Klein pictures with
busy backdrops (`asset:bench/1439/grocer.jpg`, a grocery shop; `market.jpg`, a neon
night market) ran with a prompt that moves the subject to an empty beach at dusk, on
seeds 14391 and 14392.

| Picture | Seed | Job | Bleed | SFace median / min |
|---|---|---|---|---|
| grocer | 14391 | `bee9c7bccd1c` | none | 0.211 / 0.120 |
| grocer | 14392 | `57c092b9464e` | none | 0.179 / 0.058 |
| market | 14391 | `432ef37f69cd` | none | 0.222 / 0.042 |
| market | 14392 | `c58224e7a012` | none | 0.191 / 0.049 |

The SFace numbers are low because the face is small in a wide shot. They are a baseline
only, since there are no arms to compare them against.

**Caveats that bound the drop:**
- The prompt was wide and full-figure, and said outright that the picture's setting and
  framing were not used. That may suppress framing bleed by itself.
- The pictures were generated, not production stills.
- Two pictures at two seeds is a small sample.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1460 | 0 | Session cost not recorded on the stage; the plan estimated $4-6. GPU: about 16 min on lem (4 clips 14.9 min, 2 pictures ~1 min, 4 scorings ~0.2 min) against the ~90 min cap |

The tester verified M-F434 to M-F440 over MCP against lem `develop @ 603649f1`. M-F435
was n/a because no arm-2/3 assets exist. The record's prose was out of the tester's
reach over MCP.

## Deferred and open

- **The known-leak shot can't be re-run.** Its workspace is gone, so whether the #487
  bleed reproduces on the current build is unknown.
- **The two-character pair** (Q2), where one reference bleeding into another is more
  likely. It is not built, and Don decides.
- **A closer, setting-silent prompt:** re-run arm 1 chest-up without disclaiming the
  picture's setting before writing the question off.
- **Five missing tasks,** needed only if a follow-up reproduces bleed and a
  template preparation step is wanted:
  1. a still-image SAM 3 mask;
  2. mask grow and feather;
  3. a gaussian blur;
  4. a composite through a mask;
  5. a margin crop to a mask box.
- **RefMods bundles** stay with the cast sheet (#1400). `stack_pictures` is a separate
  question.
