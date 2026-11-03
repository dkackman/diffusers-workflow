# H3 dialogue pacing: split `<d>` blocks and body actions between them (#1180)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-02). Plan v2 approved by Don on 2026-11-02 (non-turbo, no arm D,
Q2–Q5 at their defaults) as "build small"; two stages, #1184 (probe) and
#1185 (skill lines), both verified. The probe record is
`docs/proposals/audits/2026-11-h3-dialogue-pacing-probes.md`; this doc is
the design record around it.

## The idea

A post on X about **Seedance 2.5** dialogue (author and URL not recorded)
listed five tips for natural delivery: careful punctuation, short lines with
pauses, body actions as punctuation between lines, anchor gestures, and
splitting the generation. The 2026-10 dialogue probes (row 1) had measured
that H3 doesn't pace a short line across the clip: 2–3.4 s of silence, then
5–6 words/s. The only lever was sizing the line to the clip. The question
was whether splitting the line into `<d>` segments, with or without a body
action between them, spreads the speech instead.

## Decisions

- **Line:** 8 words (`I left the light on for you, Daniel.`, row 1's r150
  line, Q5), not the issue's ~12, which row 1 showed already mostly fills a
  124-frame clip.
- **Non-turbo (Don's ruling):** the skill's dialogue settings
  (`lora_model_name: null`, 12 steps, shifts 12/3, int4), because dialogue
  shots drop the turbo LoRA. Arm A was re-run as the control rather than
  compared with row 1's turbo numbers.
- **Arms:** A one block, B two blocks with a sentence break, C B plus
  "He sniffs and looks down at the table." between them. Arm D (anchor
  gesture) dropped (Q1).
- **Listening (Q2):** verdicts rest on measures only; no "more natural"
  claim without Don's listening verdict.
- **check-script (Q3):** if a rule landed, SKILL.md says to pass only the
  `<d>` contents as `lines`.
- **Enhancer (Q4):** not changed here; proposed as its own `idea` at
  close-out (see *Follow-ups*).
- **Out of scope:** punctuation alone (rows 6/6b), tip 5 (already in the
  skill), other models, turbo runs, engine or check-script code.

## What the probe found

Two jobs on lem (develop @ 05ffa7ea, diffusers 0.42.1), `150751a7522c`
(seed 1001) and `900ea1581d22` (seed 2002), all three arms per job,
~15 GPU min in total against a 50 min cap.

| # | Rule | Verdict |
| --- | --- | --- |
| 1 | Split paces the line | **confirmed**, with a caveat: lead 0.70 / 0.64 s and 2.23 / 2.26 words/s delivered, against A's 2.14 / 2.72 s and 4.71 / 5.41. Segment 1 is still spoken at 6.25 / 6.6 words/s; the pacing is a ~1.9 s silent gap between blocks and a slower segment 2. |
| 2 | An action between segments paces it | **refuted**: timing matches B, but H3 doesn't stage the action in the break; the look-down lands as segment 2 starts (window edge on s1001, after it on s2002). |
| 3 | The action prose is spoken | **no**, 0 of 6 clips. |

No listening verdict from Don was recorded, so neither the doc nor the skill
makes a naturalness claim. The A/C clips for listening are listed in the
audit doc's *For Don: listening*.

## What was built

- **Stage A (#1184):** the audit doc above. Docs only.
- **Stage B (#1185):** `plugins/dw/skills/minimax-h3/references/dialogue.md`
  gained the split technique under *How long a line can be* (arm B prompt,
  its numbers, the caveat, the configuration limit), a line that prose
  between blocks isn't spoken and doesn't count toward line length, the
  "short line" bullet now offering "split the line", and a refuted bullet
  for C under *Probed and not a rule*. SKILL.md's check-script line says
  `lines` are each `<d>` block's contents, never the prose between blocks
  (11,933 bytes, under the 12,288 cap). No plugin version bump: the plugin's
  version is the engine's (`plugins/dw/README.md`), and harnest#182 amended
  C-F692 to match. No `prompt-lint.md` note (optional).

## Bounces per stage

- **#1184:** two. First, the hand-off didn't carry what C-F688–C-F690
  needed to reproduce the per-clip table; second, C-F690's action-placement
  call for C s1001 was revised to a borderline yes and the C verdict to
  1 of 2 seeds. Both fixes were docs only; no GPU re-run.
- **#1185:** none on the build. C-F692 couldn't pass as first written (it
  expected a version bump, and the tester's shell had no `wc`); Don amended
  the case (harnest#182), measured the byte cap himself, and closed the
  stage.

Stage costs: not recorded in the stage comments, so left out.

## Follow-ups

- **Enhancer `idea`** (Q4): `dw/workflows/h3_context_ir.json` writes one
  `<d>` block per line. Filed as #1186: whether the enhancer
  should split a short line into two blocks at a sentence break.
- **Not settled:** whether the split sounds more natural (Don's listening
  call); other clip lengths, turbo, and more than one speaker were not probed.
