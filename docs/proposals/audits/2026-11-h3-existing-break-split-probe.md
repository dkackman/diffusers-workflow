# MiniMax-H3 dialogue pacing: a split at an existing sentence break (#1186 stage A, #1543)

Probe date: 2026-11-22. Server `lem`, develop @ 906513b4.
Agent: claude-opus-5-5 (anthropic). Workspace `qa-h3-existing-break`.
No engine or skill file was changed. This document only records results.

**Why this probe.** #1184 (`2026-11-h3-dialogue-pacing-probes.md`) confirmed
that splitting a short line into two `<d>` blocks removes H3's silent lead-in.
Its split *inserted* a sentence break (`… on.</d> <d>[English] For you, …`),
which the prompt enhancer can't write without changing the user's punctuation.
#1186's trigger 3 asks whether a split at a break the line **already has** does
the same. This probe answers that.

The configuration is #1184's, unchanged: H3 T2VA at 960x544, in the skill's
dialogue configuration (no turbo LoRA):
- int4 sdnq (`weights_dtype` `{int4}`), `lora_model_name: null`;
- 12 steps, video shift 12 and audio shift 3;
- sparse attention (tau 0.5, top-k 0.15), no cache;
- 124 frames (5.167 s at 24 fps), one speaker.

The two arms, one prompt each, identical except for the `<d>` blocks:

- **A2 (control)**: `<d>[English] I left the light on. It's late, Daniel.</d>`.
  One block that holds two existing sentences, 8 words.
- **B2 (split)**: `<d>[English] I left the light on.</d> <d>[English] It's late, Daniel.</d>`.
  The same words and punctuation, split at the existing period.

The shared text around them is #1184's: `Task: T2VA. Duration: 5.17 seconds. One
man sits at a kitchen table, facing the camera, in warm evening light. He speaks
to the camera in a mid-pitched baritone voice, a neutral American accent and a
natural speaking rate: <d>…</d> overall_soundscape: a quiet kitchen, a faint
refrigerator hum. non_diegetic_music: N/A`

Each arm ran on seeds 1001 and 2002, both arms in one job per seed.

**The bar** (#1186 plan v2, stage A; #1184's):
- B2 **confirms** if its lead is under 1.0 s **and** its delivered rate is
  ≤ 4.5 words/s on **both** seeds.
- A2 has to show the lead-in (≥ 1.5 s) on both seeds for the comparison to
  count. If it doesn't, the result is **control not reproduced**, not a
  confirmation.
- A failed job or a missing seed is "not run", and no verdict is called on one
  seed.

Delivered rate is #1184's definition: words divided by the time from the first
word's start to the last word's end. Tail is 5.167 s minus the end of "Daniel."
Segment 1 is "I left the light on." (5 words) and segment 2 is "It's late,
Daniel." (3 words), in both arms. In A2 the "segments" are the two sentences
inside the one block.

## Verdict

| Seed | Job | A2 lead (needs ≥ 1.5) | B2 lead (needs < 1.0) | B2 delivered (needs ≤ 4.5) | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1001 | faa4eb4f8b00 | 0.40 | 0.40 | 2.03 | **control not reproduced** |
| 2002 | f2238f7e5357 | 0.28 | 0.28 | 1.91 | **control not reproduced** |

**Overall: control not reproduced** on both seeds. Trigger 3 is **not met**:
this is not a confirmation, and it is not a refutation either.

B2 by itself clears the lead and rate numbers on both seeds. It doesn't count,
because A2 has no lead-in to remove. **The one-block line opened as early as
the split one.** On each seed A2 and B2 start within 0.01 s of each other, put
the same silence between the two sentences (2.16 vs 2.26 s, 2.24 vs 2.20 s),
and end within 0.04 s. H3 already paced the line at its own period, inside one
`<d>` block, so splitting it there changed nothing measurable.

This is the outcome the plan's *Risks* named: a line that already holds two
sentences carries its own pause and opens sooner. For this kind of line, the
enhancer's one-block form already behaves.

## Measurements

Times are in seconds from clip start. **Lead** is the onset of the first
spoken word.

Every clip's transcript is `words_a2` / `words_b2` in the job named:
`H3ExistingBreakSplitProbe/<run>/final/H3ExistingBreakSplitProbe-words_{a2.2,b2.3}-0.json`.
Every transcript is word for word the written line, `I left the light on.
It's late, Daniel.`. No word was added, dropped or reordered, and there is no
stray early word in any clip, so each lead has one reading.

| Arm | Seed | Job | Run | Lead | Seg 1 (5 words) | Seg 1 w/s | Gap at the break | Seg 2 (3 words) | Seg 2 w/s | Delivered w/s | Tail | `suspect` chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 1001 | faa4eb4f8b00 | 20261122-061434-e34a99ea | 0.40 | 0.40–1.38 | 5.10 | 2.16 (1.38–3.54) | 3.54–4.34 | 3.75 | 2.03 (8 / 3.94) | 0.83 | all 8 `silent`; counted as speech (envelope) |
| B2 | 1001 | faa4eb4f8b00 | 20261122-061434-e34a99ea | 0.40 | 0.40–1.28 | 5.68 | 2.26 (1.28–3.54) | 3.54–4.34 | 3.75 | 2.03 (8 / 3.94) | 0.83 | all 8 `silent`; counted as speech (envelope) |
| A2 | 2002 | f2238f7e5357 | 20261122-061941-8e25e6b8 | 0.28 | 0.28–1.10 | 6.10 | 2.24 (1.10–3.34) | 3.34–4.42 | 2.78 | 1.93 (8 / 4.14) | 0.75 | none |
| B2 | 2002 | f2238f7e5357 | 20261122-061941-8e25e6b8 | 0.28 | 0.28–1.14 | 5.81 | 2.20 (1.14–3.34) | 3.34–4.46 | 2.68 | 1.91 (8 / 4.18) | 0.71 | none |

Every value is read from the transcript: lead is the first chunk's `start`,
segment 1 runs from "I"'s `start` to "on."'s `end`, the gap runs from "on."'s
`end` to "It's"'s `start`, segment 2 runs from "It's"'s `start` to "Daniel."'s
`end`, and tail is 5.167 s minus "Daniel."'s `end`. Rates are words divided by
those spans, to two decimals.

**Segment rates.** As in #1184, the first sentence is fast (5.1–6.1 words/s)
in both arms. Unlike #1184's A, A2 is not one fast burst after a silence. Its
pause sits at the period, where B2's block break puts it.

## For Don: listening

No "more natural" claim is made here. That is a listening verdict, and it's
Don's. The clips to compare, A2 against B2:

| Seed | Job | Run | A2 (one block) | B2 (split at the period) |
| --- | --- | --- | --- | --- |
| 1001 | faa4eb4f8b00 | 20261122-061434-e34a99ea | `final/a2_control-0.0.mp4` | `final/b2_split-0.0.mp4` |
| 2002 | f2238f7e5357 | 20261122-061941-8e25e6b8 | `final/a2_control-0.0.mp4` | `final/b2_split-0.0.mp4` |

The paths are under `H3ExistingBreakSplitProbe/<run>/` in workspace
`qa-h3-existing-break`. By the numbers the two arms of a seed should sound
alike in timing. Whether they sound alike in delivery is for the listen.

## Not run

None. Both arms ran on both seeds.

## What this says about the skill text

B2 **doesn't contradict** `plugins/dw/skills/minimax-h3/references/dialogue.md`
*How long a line can be*. A split at a sentence break still opens early, and
it still puts a silent gap of about 2 s between the blocks. No skill text is
changed here. Two follow-ups for that section, both for a later edit and
neither made on one probe:

1. **The lead-in bullet is narrower than it reads.** "A short line is not
   stretched to fill the clip. H3 holds silence first (2–3.4 s of lead-in for
   8–10 words in 5.17 s)" was measured on single-sentence lines. Here an 8-word
   line holding two sentences opened at 0.28–0.40 s with no split. A follow-up
   edit could scope the bullet to one-sentence lines and add that a line with
   its own sentence break paces itself at the break.
2. **The split bullet calls #1184's break "its sentence break".** That break was
   inserted (`… on for you, Daniel.` became `… on.` + `For you, Daniel.`). A
   follow-up edit could say "split at an inserted sentence break" and point at
   this audit for the existing-break case.

## Method notes and deviations

- **Jobs.** faa4eb4f8b00 (seed 1001, 311 s) and f2238f7e5357 (seed 2002,
  303 s). Both ran on cuda:1 (RTX 3090) and succeeded. Workflow id
  `H3ExistingBreakSplitProbe`, which `get_job_workflow` returns.
- **Workflow.** This is #1184's `H3DialoguePacingProbe` (recovered with
  `get_job_workflow 150751a7522c`) with only these changes: the `shots` list
  holds the two arms A2 and B2 in place of A, B and C; there is one
  `transcribe_audio` step per arm (`words_a2`, `words_b2`,
  `timestamps: "word"`); and the top level has `"seed": "variable:seed"`.
  #1184's realized copy carries a literal top-level seed, so the shape is the
  same. Each job set the seed with the `seed` argument. The pipeline
  configuration, quantization, offload, attention and generation arguments are
  #1184's.
- **Whisper "silent" suspects.** The seed-1001 clips are quiet (mean −43.1 and
  −42.0 dBFS, peaks −21.9 and −20.8), and the job warns that each clip is
  "quiet overall". Every word in both seed-1001 transcripts is flagged
  `suspect: "silent"`. The RMS envelope at 0.1 s bins (`get_gallery_metadata`,
  `envelope=true`) shows speech at each flagged word: −43 to −32 dBFS over
  0.4–1.2 s, a floor near −78 dBFS over 1.4–3.4 s, and −48 to −36 dBFS over
  3.5–4.3 s, in both clips. It agrees with the transcript's lead, gap and tail
  within 0.1 s. So every `silent` chunk is counted as speech. The seed-2002
  transcripts flag no chunk, and their envelopes weren't read.
- **No long chunk.** Unlike #1184's B, Whisper doesn't stretch any word across
  the gap here, so segment-2 onsets are read straight from the transcript and
  aren't marked "about".
- **No frames were judged.** The bar is about audio timing, and this probe
  stages no action, so the clips weren't sampled for mouth movement.
