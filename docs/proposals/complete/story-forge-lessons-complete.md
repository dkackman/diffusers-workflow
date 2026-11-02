# Story-forge's production lessons, checked on lem (#1123)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-01). Plan v1 on #1123 was approved by Don on 2026-11-01 with every
default (Q1–Q4) and one amendment of his own: the L1 89-frame remedy cell.
v2 folded those in with no change of scope. It was built as three stages:
#1126 (stage A), #1127 (stage B) and #1128 (stage C).

The measurements, cell by cell with job ids, are in
`docs/proposals/audits/2026-11-story-forge-lessons.md`. This record does not
repeat them.

## The idea

story-forge (https://github.com/nicedreamzapp/story-forge, MIT) has shipped
several films on LTX-2.5 and wrote down what broke in its `LESSONS.json`. The
idea was to put the lessons that hold into the dw composition skills. It ran
LTX-2.5 through MLX (`dgrauet/ltx-2-mlx`) on Apple Silicon, not diffusers, so
the plan required each lesson to be measured on our pipeline before any text
landed.

## Verdict

**Build smaller.** Measure lessons 1, 2, 3, 4 and 6 on lem with A/B renders
in workspace `probe-1123`, land those that hold as skill text, and land
lesson 5 as unmeasured guidance. Cut lesson 7 and hand lesson 8 to #1122.
No engine, MCP, REST or syntax change, and no template default change.

**Verdict rule:** a lesson holds when its failure shows in at least half the
naive renders and its remedy removes it in at least half of the remedy
renders. N = 2 seeds per cell (Q1).

## What the plan found against the idea

1. Lesson 2 can't edit the caption spec's "camera movement is expected and
   good", because a test pins the spec verbatim to the pipeline's system
   prompt. A static rule would have gone in the skill's *Prompts* section.
2. Lesson 1 collides with dw's measured 401-frame single-pass
   `image-to-video` advice, so a 401-frame cell was measured too. Don added
   the 89-frame cell, so the measurement could tell a haze onset fixed in
   time apart from a tail that degrades.
3. Lesson 4 on Qwen-Image had nowhere to land (no template or skill).
   Klein and Z-Image were measured instead.

## Results

| # | Lesson | Verdict (at N = 2) | Landed |
| --- | --- | --- | --- |
| L1 | Haze after 2–3.5 s; render at cut length plus about 0.5 s | not reproduced: 0 of 10 naive renders. No fixed-time onset, and no degraded tail at 89, 121, 241 or 401 frames | nothing; no follow-up for the 401-frame default (Q2's trigger didn't fire) |
| L2 | i2v invents camera moves; static wording | not reproduced: 0 of 6 clear moves | nothing; `ltx-2.5` *Prompts* and `caption-spec.md` unchanged |
| L3 | Bodies in contact fuse | not reproduced: LTX 2 of 6 (mild, handshakes), H3 0 of 2. LTX followed the "just before contact" restaging in none of 6 renders | nothing |
| L4 | ALL-CAPS words render as signage | not reproduced: 0 of 6 Z-Image, 0 of 6 Klein | nothing |
| L6 | Impacts break physics | not reproduced: 1 of 4 | nothing |
| L5 | Wide shots drift hardest on identity | not measured (Q4) | guidance in `script-to-video` step 3 and `series-episodes` step 0, worded "not a measured rule" and citing the findings doc |
| L7 | Coverage: 3+ setups per line | cut | nothing |
| L8 | Write the beat for the viewer | handed to #1122 | #1122 shipped `beat` and `must_not` in `script-to-video`'s shot sheet, plus `references/beat-review.md` |

The main finding is negative: **story-forge's LTX-2.5 lessons do not transfer
from MLX to our diffusers pipeline**, at least not at N = 2. The two
lessons that would have contradicted existing dw advice (L1 against 401-frame
i2v, L2 against the caption spec) were exactly the ones that didn't
reproduce. Copying the lessons in unverified (the rejected alternative (a))
would have put wrong rules in two skills.

## What was built

### Stage A (#1126): L1 and L2 probes
24 L1 renders (t2v 241/121/89 frames on 3 prompts, i2v 241/401/89 on one
still) and 18 L2 renders (3 stills × 3 camera wordings × 2 seeds). It wrote
the findings doc's L1 and L2 sections. Docs only.

### Stage B (#1127): L3, L4 and L6 probes
12 LTX contact renders plus 2 H3 `dialogue-short` hug renders, 8 impact
renders, and 24 L4 stills (12 Z-Image, 12 Klein). Docs only.

### Stage C (#1128): landing
Lesson 5's guidance in two skills, and two short sections in the findings
doc ("Lesson 5" and "What landed"). `TestStoryForgeLessons` in
`tests/test_plugin_skills.py` pins the L5 wording and where it sits. It also
pins that **no** rule exists for a lesson that wasn't reproduced: there is no
`ltx-2.5/references/staging.md`, and none of the remedy phrases appear in the
LTX, H3, Klein, script-to-video or series-episodes skills. Plugin-only.

**Deviation:** the plan's `ltx-2.5/references/staging.md` wasn't created.
Its only content would have been the L3 and L6 rules, and neither held.

## Bounces per stage

| Stage | Bounces | Why |
| --- | --- | --- |
| A (#1126) | 1 | The hand-off didn't reproduce the findings doc's cell table and job ids. The tester can't read the checkout, and C-F603/C-F605 need the table. It was re-handed with the table and verified. |
| B (#1127) | 1 | Same missing table, plus the Klein half of L4 hadn't run: it needed a 40 GB model download. Don ruled it in (`spend`), the 12 Klein stills ran, and it was verified. |
| C (#1128) | 0 | Verified by reading. No landed LTX or image rule meant no verification render. |

Session cost: the stage comments name no `usage:` figures, so none are
recorded here.

## Found on the way, not fixed here

- **`templates/flux3/text-to-image` OOMs at load on a 24 GB 3090** against
  the current `main` revision of `black-forest-labs/flux-3-image-klein` (job
  49459dc14c17), though the template's description says about 19 GB. The
  modular pipeline has no `offload: "model"` (job 5ec3b2dcb9f5). The probes
  ran an inline copy with the text encoder leaf-level group-offloaded, which
  costs about 8–12 min of text encode per still. No issue was filed for it
  at close-out.
- A lesson for the hand-off protocol: a measurement stage's hand-off has to
  paste the findings doc's cell table, since the tester can't read the
  checkout. That caused both bounces.

## What brings it back

- **Any of L1–L4, L6:** a field report from a dw session that shows the
  failure on our pipeline, or a larger-N probe. The findings doc's cells and
  stored `probe-1123/*` prompts are the starting point.
- **L7:** a field report where a cut lacked a usable alternative take.
- **L4 on Qwen-Image:** a Qwen-Image template.
