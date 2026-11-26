# Continuity grammar: sides, eyelines, screen direction and light side across a cut (#1685)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v1 approved on 2026-11-25 by the plan-review delegate (every
question at its default) as "build smaller": a probe, then skill text. Two
stages, #1704 (probe, lem) and #1705 (skill text), both verified. The probe
record is `bench/records/continuity-grammar.md` (PRELIMINARY). The skill text
is `plugins/dw/skills/cinematography/references/continuity.md`. This doc is
the design record around them.

## The idea

janishar/helmstudio (Apache-2.0, alpha, read 2026-11-25), in its
`cinematic-filmmaker` and `animator` skills, states thirteen craft rules
that our skills didn't: the 180-degree rule, matched eyelines, screen
direction, cut on action, light direction held, "and then" means split,
holds at the head and tail, no music in shot prompts, voice-over input holding
only the words, design for animation, one style line, an expression sheet made
as an edit, and "no text" on sheets. None of them is measured in helmstudio,
so they were treated as hypotheses. Concepts only, no code.

## Decisions

- **Verdict: build smaller.** The sweep found none of the 13 stated as a
  rule. The real gap was *across a cut*: our placement rules (#484) are
  measured inside one group shot. Probe R1+R2 (sides, eyelines) and R5
  (light side) on FLUX 3 Klein, R3 (screen direction) on MiniMax-H3. A rule
  counts only when both seeds show it.
- **Q1 (spend):** about 60 min of lem for the three probes. **Q2 (scope):**
  cut R4 and R6-R13, and give each cut's reason in the note. **Q3
  (default):** note only, no `docs/AGENT_RULES.md` line. **Q4 (scope):** the
  music-in-shot-prompt audit (F1/F2) filed as #1706.
- **Cut, with reasons:**
  - R4 is an editing choice.
  - R6, R9 and R11 we already have.
  - R7 is measured against: H3's opening silence is a defect, and a static
    shot pushes in.
  - R8 is stated, and its template audit is #1706.
  - R10 is a taste rule with no demand.
  - R12 clashes with the face-once audit and with Klein's failed reverse edit.
  - R13 is measured against: "no lettering" gave pseudo-lettering on both
    seeds on H3.
- **Phrasing.** The probe used a subject's side of frame and its facing, not
  an anatomical side. "On screen right" for a body side was already refuted
  (P5, #1059).

## What the probe found (PRELIMINARY)

Workspace `probe-1685`, seeds 1685 and 1686 per arm, 12 jobs, all
succeeded. The workflow defaults are the control, and the treatment prompts
are in `bench/records/continuity-grammar-arms.json`.

| Probe | Family | Control jobs | Treatment jobs | Verdict |
|---|---|---|---|---|
| P-A sides, eyelines (R1, R2) | Klein | `01bc26f4b8c7`, `47784abf5969` | `f82e48f25759`, `32aecd104095` | **confirmed**: the singles hold side and facing on both seeds, and the control flips |
| P-B light side (R5) | Klein | `eb69644d79e6`, `e1268b23047e` | `4ae49d950d1b`, `2619ab70f376` | **unsettled**: the lamp effect showed on seed 1685 only, and the face key is soft |
| P-C screen direction (R3) | H3 | `7bb47e82a72f`, `0f412617cacb` | `f636fe850808`, `405ccc8e2fcc` | **confirmed**: left to right in both shots of both seeds, and the control walks toward the camera |

Caveat: the treatment singles name the off-screen partner's side, and Klein
drew them as over-the-shoulder shots. The shoulder figure in card `b` is not
June, so side is held but identity is not. LTX-2.5 and Kandinsky 6 were
not probed.

## What was built

- **#1704 (stage 1):** `bench/workflows/continuity-{side,light,direction}.json`,
  `bench/records/continuity-grammar.md` with its arms file, and an
  `EXEMPT` entry for the record in `tests/test_plugin_intents.py`.
- **#1705 (stage 2):**
  - `cinematography/references/continuity.md`, linked from
    `cinematography/SKILL.md` in one line. Its sections are:
    - Sides and eyelines (Klein), with the over-the-shoulder caveat;
    - Screen direction (H3);
    - Families not probed;
    - Probed and not a rule (R5, unsettled);
    - Cut without a probe.
  - Pointer lines in `storyboard/references/coverage-set.md` and, because R1
    held, a **Sides** bullet in `series-episodes/references/sitcom-multicam.md`.
  - A new `flux3-klein/references/intents.md`, since Klein had no index
    before, with R1 and R2 `confirmed`. H3's `intents.md` gets R3
    `confirmed`. R5 has no row while it is unsettled.
  - Pins in `tests/test_plugin_skills.py::TestContinuityNote`. The
    medium-close-up line, the closed axes and the shot keys are unchanged.

## Deferred

- **Don's look at P-B (R5)**, the taste question on #1704 (Q1). If he sees
  the light side held, a `confirmed` Klein intents row and a rule go in, and
  PRELIMINARY lifts. Until then the note says "unsettled, so not a rule".
- **Identity in the over-the-shoulder single.** Naming the partner's side
  drew a stranger's shoulder. This was not separated from the side effect.
- **LTX-2.5 and Kandinsky 6** arms. *Comes back on* a field report of a
  flipped side or direction on that family.
- **P-C audio** was near-silent (about -55 dBFS), and one control shot pushed
  in despite "The camera does not move." Both were seen, outside the probe.
- **F1/F2**, music in shot prompts: #1706.

## Cost and bounces

- **#1704:** about 22-23 lem GPU-minutes against the ~60 approved. Two
  bounces:
  1. `status:needs-info`: the record lives in the repo, and the tester needed
     its text quoted on the issue. This is the same bounce as #1581.
  2. C-F1337: a control read said "frame right" where the still shows frame
     left. C-F1339: R5's taste question wasn't in the Decision-for-Don block.
- **#1705:** one bounce. C-F1343: Klein's R5 intents row read `refuted`
  against an unsettled verdict. The row was dropped.
- Session dollar costs were not reported on the stages, so none are given.
