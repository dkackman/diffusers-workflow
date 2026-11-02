# MiniMax-H3 dialogue pacing: record-only probes (#1180 stage A, #1184)

Probe date: 2026-11-02. Server `lem`, develop @ 05ffa7ea, diffusers 0.42.1.
Agent: claude-opus-5-5 (anthropic). Workspace `qa-h3-dialogue-pacing`.
No engine or skill file was changed. This document only records results.

Every probe ran on H3 T2VA at 960x544, in the skill's dialogue configuration
(`minimax-h3` *Hard rules*: dialogue shots drop the turbo LoRA):
- int4 sdnq (`weights_dtype` `{int4}`), `lora_model_name: null`, no turbo LoRA;
- 12 steps, video shift 12 and audio shift 3;
- sparse attention (tau 0.5, top-k 0.15), no cache;
- 124 frames (5.167 s at 24 fps).

**Arm A is a non-turbo re-run of row 1's r150 line** (2026-10 probes, *Probe 1*,
job be681f38d5a8). It is not the same configuration: row 1 ran turbo at 9 steps.
Its numbers are this run's control, not row 1's.

The three arms, one prompt each, identical except for the `<d>` blocks:

- **A (control)**: `<d>[English] I left the light on for you, Daniel.</d>`
- **B (split)**: `<d>[English] I left the light on.</d> <d>[English] For you, Daniel.</d>`
- **C (split + action)**: `<d>[English] I left the light on.</d> He sniffs and looks down at the table. <d>[English] For you, Daniel.</d>`

The shared text around them: `Task: T2VA. Duration: 5.17 seconds. One man sits at a
kitchen table, facing the camera, in warm evening light. He speaks to the camera in a
mid-pitched baritone voice, a neutral American accent and a natural speaking rate:
<d>…</d> overall_soundscape: a quiet kitchen, a faint refrigerator hum.
non_diegetic_music: N/A`

Each arm ran on seeds 1001 and 2002, all three arms in one job per seed.

**The bar** (plan v2, stage A):
- "Split paces the line" is **confirmed** when both B seeds have a lead under
  1.0 s and a delivered rate ≤ 4.5 words/s, and neither A seed does.
- "An action between segments paces it" is **confirmed** by the same test on C
  against A, plus the action rendered at its break on both C seeds.
- "The action prose is spoken" is a **warning** if it shows on even one seed.
- If A shows no lead-in, B and C are inconclusive.

Delivered rate is the 2026-10 definition: words divided by the time from the
first word's start to the last word's end. Per-segment rates are given too.
Tail is 5.167 s minus the end of "Daniel."

## Verdicts

| # | Rule | Verdict | Jobs | Observation |
| --- | --- | --- | --- | --- |
| 1 | "Split paces the line." | **confirmed** by the bar, with a caveat | 150751a7522c, 900ea1581d22 | A has a lead-in on both seeds (2.14 s, 2.72 s) and is spoken at 4.71 and 5.41 words/s, so it fails the bar. B leads at 0.70 s and 0.64 s, with delivered rates of 2.23 and 2.26 words/s, so it passes. **Caveat:** the split doesn't slow the words. Segment 1 is still spoken at 6.25 and 6.6 words/s. The rate falls because H3 puts a 1.9 s silent gap between the segments, not because the speech is slower. See *Arm B*. |
| 2 | "An action between segments paces it." | **refuted** (the action condition fails) | 150751a7522c, 900ea1581d22 | Lead and rate pass on both C seeds (lead 0.44 s and 0.66 s, delivered 1.93 and 2.16 words/s), and the break is a 2.1–2.5 s silence. But the action doesn't render at its break on either seed. The mouth stays closed through the break and no sniff is visible. The look-down comes after segment 2 has started, at about 4.0–5.0 s on s1001 and 4.5–5.0 s on s2002. See *Arm C*. |
| 3 | "The action prose is spoken." | **no**: 0 of 6 clips | 150751a7522c, 900ea1581d22 | None of "sniffs", "looks", "down" or "table" appears in any transcript, and none of the arms' words was added or dropped. No warning. |

## Measurements

Times are in seconds from clip start. **Lead** is the onset of the first
spoken word. Onsets and gaps come from the RMS envelope, which agrees with the
Whisper word timestamps within 0.1 s at each onset (see *Method notes*).
Transcripts are `H3DialoguePacingProbe-words_{a.3,b.4,c.5}-0.json` in each run.

| Arm | Seed | Lead | Gap at the break | Seg 1 w/s | Seg 2 w/s | Delivered w/s | Tail | Action word | Mouth moves in silence | Action at its break |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1001 | 2.14 | n/a (no break) | n/a | n/a | 4.71 (8 / 1.70) | 1.33 | none | **possibly**: lips parted or open at 0.0–1.5 s in the silent lead-in | n/a |
| A | 2002 | 2.72 | n/a | n/a | n/a | 5.41 (8 / 1.48) | 0.97 | none | no: closed or neutral at 0.0–2.5 s | n/a |
| B | 1001 | 0.70 | about 1.9 (1.6–3.5) | 6.25 | about 4.1 | 2.23 | 0.89 | none | no: closed or neutral through the gap | n/a |
| B | 2002 | 0.64 | about 1.9 (1.5–3.4) | 6.6 | about 3.8 | 2.26 | 0.99 | none | no: closed through the gap | n/a |
| C | 1001 | 0.44 | about 2.5 (1.3–3.8) | 6.25 | about 3.8 | 1.93 | 0.59 | none | no: closed through the break | **no**: no sniff; head and gaze drop at 4.0–5.0 s, after the break |
| C | 2002 | 0.66 | about 2.1 (1.5–3.6) | 5.8 | 4.05 | 2.16 | 0.81 | none | no: closed or smiling through the break | **no**: head down at 4.5–5.0 s, after the break |

### Arm A: the control

The lead-in that row 1 found under turbo holds without it. Both seeds open on
2.1–2.7 s of silence and then speak the 8-word line in about 1.5–1.7 s. The
tails (0.97–1.33 s) are longer than row 1's turbo tails. On s1001 the lips
are parted, and may move, in the silent lead-in. At 0.5 s sampling that is
ambiguous: it could be a held open mouth rather than speech-like movement.
On s2002 the mouth is closed until speech starts.

### Arm B: the split

Both seeds start segment 1 within 0.7 s, speak it fast (about 6.3–6.6 words/s),
leave a silent gap of about 1.9 s, and speak segment 2 more slowly (about
3.8–4.1 words/s). The mouth is closed through the gap on both seeds. The
pacing the split buys is the gap and a slower second segment. The first
segment stays at the fast rate A uses for the whole line.

### Arm C: the split and an action

The timing matches B: an early lead, a fast first segment, a silent break of
2.1–2.5 s and a slower second segment. The action prose doesn't fill the break
on screen. On both seeds the man sits still with his mouth closed through
it. The look-down renders late, overlapping or following segment 2. On
s1001, a quiet sound at 2.4–2.9 s (−51 to −58 dBFS, against speech at about
−40 dBFS) sits in the break. It could be a breath or a sniff, but nothing
visible goes with it, and it isn't transcribed. So C passes the lead and rate
test but fails the "action rendered at its break" condition on both seeds.

### What this says about the rules

- A split into two `<d>` blocks removes the silent lead-in on both seeds and
  puts a pause between the blocks. By the bar's delivered-rate definition, that
  paces the line. It doesn't slow the first block's words.
- Action prose between the blocks gives no pacing beyond what the split already
  gives, and H3 doesn't stage the action in the break. Here it rendered late.
- Action prose between `<d>` blocks isn't spoken (0 of 6 clips, counting A and
  B, which have none, as controls).

## For Don: listening

No "more natural" claim is made here. That is a listening verdict, and it's
Don's. The clips to compare, A against C:

| Seed | Job | Run | A (control) | C (split + action) |
| --- | --- | --- | --- | --- |
| 1001 | 150751a7522c | 20261102-230056-b618eb3c | `final/a_control-0.0.mp4` | `final/c_split_action-0.0.mp4` |
| 2002 | 900ea1581d22 | 20261102-230859-f997d065 | `final/a_control-0.0.mp4` | `final/c_split_action-0.0.mp4` |

B is in the same runs, as `final/b_split-0.0.mp4`.

## Not run

None. All three arms ran on both seeds.

## Method notes and deviations

- **Jobs.** 150751a7522c (seed 1001, 443 s) and 900ea1581d22 (seed 2002,
  435 s). Both ran on cuda:0 (RTX 3090) and succeeded. Workflow id
  `H3DialoguePacingProbe`, recoverable with `get_job_workflow`.
- **The row-1 workspace is gone.** `qa-h3-dialogue-probes` no longer exists,
  and `get_job_workflow be681f38d5a8` returns no definition. So arm A's prompt
  was rebuilt from the 2026-10 record's description of the r150 line. It is
  not a byte-for-byte copy of row 1's prompt.
- **Batching.** The plan names `qa/h3-probe-batch`. Instead, the arms ran as an
  equivalent inline workflow: `templates/minimax/shots-batch`'s pipeline
  configuration without the LoRA, one `shot` step over the three arms
  (`release_pipeline: true`), then one `transcribe_audio` step per arm
  (`words_a`, `words_b`, `words_c`, `timestamps: "word"`). The generation
  arguments are the ones listed at the top of this document.
- **Whisper "silent" suspects.** The clips are quiet (mean −40 to −43 dBFS),
  and Whisper flagged most words `suspect: "silent"`. The envelope shows speech
  at each flagged word, so the words were kept. Lead and tail can be re-read
  from each transcript's first-word start and last-word end within 0.1 s.
- **Long "for" chunks.** In B, Whisper stretches the word "for" across the
  gap (1.50–3.68 s on s1001, 1.40–3.52 s on s2002). The segment-2 onset and
  the gap are taken from the envelope, where the silence is unambiguous.
  That is why the segment-2 rates are marked "about".
- **Frames.** Sampled at 0.5 s steps (`get_output_frames`), so mouth and action
  judgements are coarse. A movement shorter than 0.5 s could be missed.
