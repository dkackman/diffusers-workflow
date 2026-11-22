# H3 enhancer: split a short line at an existing sentence break (#1186)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-22). Plan v2 approved by Don on 2026-11-21 on its defaults (Q1
[spend] run the probe; Q2 [scope] the enhancer edit stays cut) as "build
smaller". One stage, #1543 (probe), verified. The probe record is
`docs/proposals/audits/2026-11-h3-existing-break-split-probe.md`; this doc is
the design record around it.

## The idea

#1180's close-out (Q4) owed an enhancer idea: `dw/workflows/h3_context_ir.json`
writes one `<d>` block per line, and #1184 had confirmed that splitting a short
line into two blocks removes H3's silent lead-in. Should the enhancer split a
line that is short for the clip at a sentence break?

## Decisions

- **Plan v1 (2026-11-02): defer**, ruled by Don. Three triggers would reopen
  it: (1) a `field-report` of a lead-in or rushed short line in an
  enhancer-written H3 prompt; (2) Don's listening verdict that #1184's split
  sounds better; (3) a probe confirming that a split at an **existing**
  sentence break removes the lead-in on both seeds.
- **Plan v2 (2026-11-21):** Don unparked the issue and asked whether the
  body's finding was trigger 3. It wasn't: #1184's arm B **inserted** a
  period (`I left the light on.` / `For you, Daniel.`), a form the enhancer
  can't write under its "preserve the user's wording and punctuation" rule.
  Triggers 1 and 2 were also unmet. Verdict: build the missing probe only.
- **Cut:** the enhancer instruction edit (v1's stage B). It returns as plan v3
  only if the probe confirms on both seeds and trigger 1 or 2 fires, or Don
  asks for it.

## What the probe found

Two jobs on lem in `qa-h3-existing-break`, at #1184's non-turbo configuration
(t2va, 960x544, 124 frames, int4, 12 steps, shifts 12/3, no LoRA):
`faa4eb4f8b00` (seed 1001) and `f2238f7e5357` (seed 2002), ~10.3 GPU min in
total against the ~35 min approved.

- **A2 (control):** `<d>[English] I left the light on. It's late, Daniel.</d>`
- **B2 (split):** `<d>[English] I left the light on.</d> <d>[English] It's late, Daniel.</d>`

| Seed | A2 lead | B2 lead | B2 delivered w/s | Gap at the period (A2 / B2) | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1001 | 0.40 | 0.40 | 2.03 | 2.16 / 2.26 | control not reproduced |
| 2002 | 0.28 | 0.28 | 1.91 | 2.24 / 2.20 | control not reproduced |

**Overall: control not reproduced.** A line that already holds two sentences
opens early and pauses ~2.2 s at its own period inside one block; splitting it
changes timing by ~0.1 s at most. Trigger 3 is **not met**: neither a
confirmation nor a refutation of the split. Read plainly, the enhancer's
one-block form already behaves for this kind of line, so the probe gives the
enhancer edit no case on this evidence.

## What was built

- **Stage A (#1543):** the audit doc above and a row in `todo.md`. Docs only;
  no code, enhancer, skill or test change.

## Bounces per stage

- **#1543:** one. The first hand-off didn't carry the audit's per-clip table,
  which C-F1159/C-F1160 read from the hand-off. The re-hand-off copied it in;
  no doc or GPU change.

Stage costs: no `usage:` figures in the stage comments, so left out beyond
the GPU minutes above.

## Deferred and follow-ups

- **Enhancer edit:** not built (Q2). Triggers 1 and 2 still stand; trigger 3
  is unmet and the probe gave it no support.
- **Skill text** (`plugins/dw/skills/minimax-h3/references/dialogue.md`,
  *How long a line can be*): two edits the audit names and doesn't make. The
  lead-in bullet was measured on one-sentence lines, and the split bullet's
  "its sentence break" was an inserted break.
- **Not settled:** whether either split sounds more natural (Don's listening:
  the audit's *For Don: listening*); other line lengths, clip lengths, turbo
  and two speakers.
