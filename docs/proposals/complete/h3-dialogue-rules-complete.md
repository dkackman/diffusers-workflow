# H3 dialogue rules: probes, `references/dialogue.md`, no speech budget (#608)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-07). Plan v2 approved by Don on 2026-10-05 as "build smaller", for
stages A1, A2 and B only; three stages, #640 (A1), #641 (A2), #642 (B), all
verified. The probe record is
`docs/proposals/audits/2026-10-h3-dialogue-probes.md`; this doc is the
design record around it.

## The idea

A comparative review of another H3 workflow repo (concept only: its license
is incompatible with Apache-2.0, so nothing was copied) proposed:

- **A.** Deterministic `validate_workflow` warnings, declared per H3
  template: a dialogue word budget (~2.6 words/s plus ~0.4 s per `<breath>`,
  ~1 s tail, because "H3 stretches dialogue and clips the last word"), a
  ~7,000-character prompt budget, and a positive-only negation lint.
- **B.** A `minimax-h3/references/dialogue.md` with nine empirical rules on
  voices, lip-sync, tags, extras, off-screen naming, silent scenes and
  continuations.
- **C.** A `minimax-music3` rule: transcribe every instrumental for
  accidental vocals.

None of these appears in MiniMax's guides, so every rule was probed on
`lem` before it could go into a skill.

## Decisions

- **Plan v1/v2 verdict: build smaller.** Probe first (A1, A2), write
  `dialogue.md` with confirmed rules only (B), and build the speech-budget
  warning (C1: a `speech` constraint kind in `variable_constraints`; C2:
  declare it on ~14 H3 templates) **only if** A1 showed H3 clips over-budget
  dialogue. Cut up front, each with a revival trigger: the negation lint, the
  `<breath>` budget term, and the 7,000-character budget unless an encoder
  limit existed.
- **Q2:** a garbled readable-text verdict would be recorded in `dialogue.md`
  and the enhancer fix filed separately. **Q3:** no inheritance into custom
  workflows. **Q4:** 2 seeds per arm. **Q5:** ~2.5 h of lem GPU, raised by
  Don on 2026-10-06 to ~3.25 h for A1+A2.
- **Probe 8 (continuation rule):** Don accepted it as *not run* (GPU
  budget); its prompt design is kept in the audit doc's *Not run* section.
- **Probe 1's 345-frame arm:** Don accepted it as recorded (seed 1001 only,
  2.2 and 4.0 words/s), option 3 on #640.
- **Probe 1's timings:** Don chose to wait for #661 (word-timestamp fix in
  `transcribe_audio`) and re-transcribe, so the tail figure is checkable
  over MCP (job ae52011e2f05).

## What the probes found

From the audit doc's verdict table (2 seeds per arm; confirmed only when
both broken seeds show the effect and neither followed seed does):

| # | Rule | Verdict |
| --- | --- | --- |
| 1 | ~2.6 words/s, ~1 s tail, H3 stretches and clips the last word | **refuted**: no word lost from 1.55 to 4.06 written words/s; tail 0.09–0.88 s, median ~0.37 s |
| 2 | ≤ 1 male + 1 female speaker | **refuted** |
| 3 | soft/whisper words flip a male voice female | **refuted** |
| 4 | each silent person needs a lips-closed sentence | **refuted** |
| 5 | a phone/radio holder lip-syncs it | **refuted** |
| 6 | `<pause>`/`<softer>` spoken; bare `…` invents words | **refuted** |
| 6b | `<breath>` is a silent beat | not spoken; silent beat **inconclusive** |
| 7 | ~7,000-character prompt budget | **refuted** from source: the encoder doesn't truncate |
| 8 | continuation from rest, `<breath>`, no words in warm-up | **not run** |
| 9 | ≤ 3 in-focus extras or faces clone | **inconclusive** (clones on one seed of two) |
| 10 | naming an off-screen thing pulls it in | **confirmed** |
| 10b | a negated noun renders the thing | **refuted** |
| 11 | voice words in a silent scene invent speech | **inconclusive**: H3 didn't render the soundscape |
| 12 | quoted sign text renders garbled | **legible** (refuted); a "blank" sign invents pseudo-text |
| 13 | Music3 instrumentals grow vocals | **worth a rule**: 1 of 4 runs, judged from the vocal stem |

## What was built

**A1, #640** (docs only): the audit doc, rows 1–8 and 6b. Develop
a73b486a → c7efd5a2 → 4d1bb7dd → ba8f784d → 1c830507 (the post-#661
re-transcription).

**A2, #641** (docs only): rows 9–13. Develop f2a4d77f, corrected at
bb5a17fd.

**B, #642** (plugin only), develop 8909eb69:
- `plugins/dw/skills/minimax-h3/references/dialogue.md`: the one confirmed
  rule (row 10, don't name what the shot shouldn't show); probe 1's rate and
  tail as prose; the encoder-doesn't-truncate finding; the legible-text
  verdict with the blank-sign side finding; and a *Probed and not a rule*
  section listing every refuted, inconclusive and not-run row as a finding,
  not an instruction. Every statement cites its audit row.
- One line in `minimax-h3/SKILL.md` *Prompts* linking it (11,640 bytes,
  under the 12,288 cap).
- One sentence on `minimax-music3/SKILL.md`'s instrumental bullet: an
  instrumental can still grow a voice; check the `vocals` stem of
  `separate_stems` with `analyze_audio` against the mix, never a transcript
  of the mix (11,972 bytes).
- `h3_context_ir.json` untouched: row 12 came back legible, so Q2 had
  nothing to fix.
- No plugin version bump: the README pins the plugin version to the
  engine's, and only the release script moves it.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A1, #640 | 2 | (1) the #608 table lacked job ids on every row and `<breath>` had no verdict of its own; (2) C-F272: probe 1's rate wasn't reproducible from the cited transcripts, because `transcribe_audio` folded leading silence into the first word. Fixed by #661 and a re-transcription (ae52011e2f05). The 345-frame clause of C-F271 was amended (harnest#71). |
| A2, #641 | 1 | Row 11 used a verdict outside the three allowed words; row 13 rested on whole-mix transcripts that were hallucinations. Row 11 became inconclusive; row 13 was re-decided from htdemucs vocal stems (1bf9f38af007). |
| B, #642 | 0 | C-F280's "transcribe every instrumental" wording predates row 13's finding; the tester filed the amendment as harnest#75. |

No stage comment names `usage:` figures, so per-stage session cost is not
recorded. The plan estimated ~$11–14 for A1, A2 and B.

**GPU:** A1 ~127 min, A2 ~74 min (63 H3, 11 Music3): ~201 min against the
raised ~195 min cap. A2 overran by ~7 min, from splitting batches (each
reloads H3) and splitting by seed; the audit doc records it.

## Not built, and why

- **C1/C2, the `speech` constraint kind and its declaration on the H3
  templates.** The plan's gate was A1 confirming that H3 clips over-budget
  dialogue; A1 refuted it at every rate tried, so per plan v2 C was not
  filed and the measured rate went into `dialogue.md` as prose instead. The
  design (schema `oneOf`, a `WARNING_CHECKS` entry reading expanded per-entry
  prompts, a terse catalog rendering) is kept in the plan comment on #608.
  *Comes back if* a field report or a later probe shows H3 losing words at a
  rate a template's default `num_frames` invites, or a new H3 checkpoint
  changes the delivery.
- **The 7,000-character prompt budget.** Refuted from source (row 7).
  *Comes back if* a long prompt is seen losing its tail.
- **The negation lint.** Row 10b refuted its revival condition.
- **The `<breath>` budget term.** Row 6b: never spoken, and no silent beat
  it creates is distinguishable from an ordinary sentence break.
- **The continuation rule (probe 8).** Not run; design in the audit doc.
  *Comes back if* a `chain-video-continuity` run shows words in the
  warm-up prefix or a jolt at the seam.
- **Rules 9 and 11** stay unsettled: a third seed, or (for 11) a
  soundscape H3 actually renders, would decide them.

## Overlap left as it was

#609 (script-adherence probe) had not landed when A1 ran; probe 1 used
`transcribe_audio` word timestamps. #610 (voice bible) and #488 (lip-sync
target check) are not pre-empted: the refuted speaker rules in
`dialogue.md` are findings, not guidance.
