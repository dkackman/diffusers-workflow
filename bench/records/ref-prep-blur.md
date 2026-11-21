# Reference prep for H3 ref2va: crop and blur the backdrop (#1439, stage #1460)

Does preparing an H3 reference picture stop its backdrop and its framing bleeding into a
`templates/minimax/reference-to-video` clip? The three arms:
- **Arm 1:** the picture as it is.
- **Arm 2:** a square crop on the person.
- **Arm 3:** the arm-2 crop with its backdrop blurred.

**Answer: not reproduced at one reference; drop.**
- The primary known-leak shot was gone, so the fallback ran: two busy-backdrop pictures,
  two seeds each.
- None of the four arm-1 clips shows any bleed: no backdrop and no framing.
- By the plan's gate, arms 2 and 3 did not run, and no skill line lands.

Run 2026-11-19 by claude-opus-5-5 (anthropic):
- the main `dw.serve` on lem (2x RTX 3090), develop @ `603649f1` (this stage changes no
  engine code);
- workspace `bench`.

Data: `bench/records/ref-prep-blur.json`.

## Which arm-1 path ran

**Fallback.** The plan's primary arm 1 was the known-leak shot from #487. That shot lived
in workspace `encanto-uprising`, which no longer exists on lem, and #484 item 5 names no
job id. So the fallback ran:
- busy-backdrop pictures on the template at its defaults;
- a prompt that moves the subject to an empty beach at dusk;
- two seeds (14391, 14392) shared by every arm;
- a second picture after the first showed no bleed.

## Setup

**Probe workflow.** `bench/workflows/probe-1439-refprep.json` composes the catalog
template by name. Its variables are `prompt`, `subject` (the arm) and `seed`.
- **Template defaults:** 960x544, 124 frames at 24 fps, 9 steps, int4, turbo LoRA, and the
  template's voice reference.
- **Run inline:** the probe file is new on this branch and not on lem's `603649f1`
  checkout, so it ran as an `inline_workflow` with the same JSON. The scoring ran by its
  catalog name, `bench/t26-face-check`.

**Pictures.** Both are FLUX 3 Klein (`templates/flux3/text-to-image`, 1024x1024),
generated for this probe:

| Picture | Asset | Seed / job | Backdrop |
|---|---|---|---|
| grocer | `asset:bench/1439/grocer.jpg` | 1439 / `fd1ebe8f261f` | Grocery shop: shelves, cereal boxes, cans, a neon OPEN sign. A man in a mustard jacket, chest-up. |
| market | `asset:bench/1439/market.jpg` | 1440 / `bff2f1550c70` | Neon night market: signs, lanterns, scooters, a crowd. A woman with an auburn bob and a teal scarf. |

**Prompts.** Each prompt describes its subject's face, hair and clothes. It states in
`retention_analysis` that the setting, framing and background of `<Picture 1>` are not
used. It then puts the subject on an empty beach at dusk in a wide, static, full-figure
shot.
- grocer: the probe's default `prompt`;
- market: the same prompt with the woman's description, passed as the `prompt` argument
  (the full text is in the JSON).

**Bleed.** Judged by eye at frames 0, 12, 24, 48 and 62 (mid-clip). Each clip is labelled
none, framing, backdrop or both.

**SFace.** `analyze_face_drift` with `samples_per_shot: 16`, through
`bench/workflows/t26-face-check.json`, scored against the original picture.

## Results (arm 1)

| Picture | Seed | Clip job | Run | Bleed | SFace median / min |
|---|---|---|---|---|---|
| grocer | 14391 | `bee9c7bccd1c` | `20261119-065039-c0b45ff9` | none | 0.211 / 0.120 |
| grocer | 14392 | `57c092b9464e` | `20261119-065047-15a48259` | none | 0.179 / 0.058 |
| market | 14391 | `432ef37f69cd` | `20261119-065556-5f3ef3d2` | none | 0.222 / 0.042 |
| market | 14392 | `c58224e7a012` | `20261119-065606-3b8df9ff` | none | 0.191 / 0.049 |

**What the frames show.**
- **All four clips:** from frame 0, a clean empty beach at dusk, with surf, wet sand and an
  orange-violet sky. The subject is full-figure and small in the frame, walking toward
  camera.
- **No backdrop bleed:** no shelves, cans, signs, lanterns, neon or crowd.
- **No framing bleed:** no chest-up opening frame.
- **Identity:** the clothes carry over (mustard jacket; teal scarf and charcoal coat), and
  the hair reads right.

**About the SFace numbers.** They are low (0.18–0.22 median, below the 0.363 drift
threshold) because the face is small in a wide shot. They are arm 1's baseline only.
With no arm 2 or 3 there is nothing to compare them against.

**Warning.** Job `432ef37f69cd` warned that its audio peaks at +0.78 dBFS (clips). This
is unrelated to the reference.

## Arms 2 and 3: not run

**The gate is unmet.** Neither picture showed bleed on either seed, so the record stops
at "not reproduced at one reference; drop":
- no segmentation;
- no crop;
- no blur;
- no arm-2/3 jobs.

**Settings for a later run.** The arms are ready:
- **Segmenter:** the `segment` task with prompt `"person"` (GDINO + SAM2, both on lem).
  Segment the full picture for the bbox, then segment the crop again for the blur mask.
- **Crop (arm 2):** a square `recenter_crop` on the mask bbox's centre, with a side of
  1.75x the subject box's longer side, capped at the full picture. Not computed here.
- **Blur (arm 3):** `bench/blur_outside_mask.py IMAGE MASK OUT`:
  - grow 12 px (a 25-px square maximum filter);
  - feather sigma 6;
  - backdrop gaussian sigma 20;
  - a composite of `alpha * sharp + (1 - alpha) * blurred`.

  The mask must be the crop's own size.

## Verdict

**Drop.** The rule's "drop" branch applies: no bleed at one reference, so there is nothing
for a crop or a blur to remove. No skill line goes into
`plugins/dw/skills/minimax-h3/references/per-character.md`.

**Caveats that bound the claim:**
- **Prompt.** The prompt is wide and full-figure, and says outright that the picture's
  setting and framing are not used. That may itself suppress framing bleed. The #487 leak
  may have needed a closer framing or a prompt that leaves the setting unstated.
- **Pictures.** The pictures are generated, not production stills.
- **Voice.** The market prompt's voice line still describes the template's male voice
  reference for a woman, a confound for the audio only.
- **Sample size.** Two pictures by two seeds is a small sample.

## The five tasks this needed that dw lacks

To prepare a reference inside a workflow rather than with a bench script, dw would need:
1. a still-image SAM 3 mask (`segment` gives SAM2 via GDINO today);
2. mask grow and feather;
3. a gaussian blur;
4. a composite through a mask (sharp over blurred);
5. a margin crop to a mask box (a `recenter_crop` sized from a mask's bbox).

## GPU time

About 16 min of lem GPU against the ~90 min cap:
- 2 pictures, about 1 min;
- 4 clips, 14.9 min (4.0, 3.6, 3.7 and 3.55 min);
- 4 SFace scorings, about 0.2 min.

## Follow-ups (Don decides; not built here)

- **The two-character pair.** Put two references in one ref2va clip, where one reference's
  backdrop or face bleeding into the other is more likely. That is where crop and blur
  would have something to fix.
- **A closer, setting-silent prompt.** Re-run arm 1 with a chest-up prompt that doesn't
  disclaim the picture's setting before writing the question off.
- **The five missing tasks above,** if either follow-up reproduces bleed.
