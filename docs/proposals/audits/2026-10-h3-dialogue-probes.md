# MiniMax-H3 dialogue rules: record-only probes (#608 stage A1, #640)

Probe date: 2026-10-06. Server `lem`, develop @ 082313e9, diffusers 0.41.0.dev0.
Agent: claude-opus-5-5 (anthropic). Workspace `qa-h3-dialogue-probes`.
No engine or skill file was changed. This document only records results.

Every probe ran on H3 T2VA turbo at 960x544:
- int4 sdnq, with the `lightx2v/Minimax-h3-Turbo` LoRA
  (`minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors`);
- 9 steps, video shift 12 and audio shift 3;
- 124 frames, unless a row says otherwise.

Each arm ran on seeds 1001 and 2002. A rule is **confirmed** only when both
seeds of the rule-broken arm show the effect and neither seed of the
rule-followed arm does. It is **refuted** when the broken arm doesn't show
the effect, and **inconclusive** when the seeds disagree.

## Verdicts

| # | Rule (as stated on #608) | Verdict | Jobs | Observation |
| --- | --- | --- | --- | --- |
| 1 | "about 2.6 words/s … leaving about 1 s of tail. H3 stretches dialogue to fill the clip and clips the last word." | **refuted** (clipping and stretching). The tail is real but usually smaller than 1 s. | be681f38d5a8, 3541c1b0a79c | No word was lost at any written rate from 1.55 to 4.06 words/s at 124 frames (both seeds), or at 2.2 and 4.0 words/s at 345 frames. Short lines aren't stretched: they get a 2–3.4 s silent lead-in and are then spoken fast (5–6 words/s). Tails ran 0.12–1.22 s, median about 0.4 s. See *Probe 1* below. |
| 2 | "At most one male and one female speaker per scene, or the voices mix." | **not run**: GPU budget | — | See *Not run*. |
| 3 | "'soft / gentle / whisper / lullaby' in a male line's delivery can flip the voice female." | **refuted** | 6f588db1f043, 3a29df186aac | The voice stayed male on both arms and both seeds, with median pitch 85–103 Hz (a female voice sits around 165–255 Hz). In seed 1001 the soft-delivery wording made the delivery breathier (fewer voiced frames, a wider pitch spread) but not female. See *Probe 3*. |
| 4 | "Every silent on-screen person needs their own 'lips stay pressed together' sentence." | **not run**: GPU budget | — | See *Not run*. |
| 5 | "A voice-only source (a radio, a phone) gets lip-synced by whoever is holding it." | **not run**: GPU budget | — | See *Not run*. |
| 6 | "`<pause>` and `<softer>` are spoken aloud; a bare '...' produces invented words." | **refuted** | 6f588db1f043, 3a29df186aac | Neither seed spoke `pause`, `softer` or `breath`, and the bare `…` produced no invented words. Both seeds spoke the line as written. Seed 1001 put a 0.56 s gap where `<breath>` sat. See *Probe 6*. |
| 7 | "Prompt length budget: about 7,000 characters." (meant as the encoder's `max_sequence_length`) | **refuted** (from source, no GPU) | — | The H3 text encoder doesn't truncate. See *Probe 7*. |
| 8 | "A continuation should start from rest, open on `<breath>`, and begin no new words inside the discarded warm-up prefix." | **not run**: GPU budget | — | See *Not run*. |

## Probe 1: rate, tail and clipping

Prompt: one man at a kitchen table, facing the camera. Delivery: "a
mid-pitched baritone voice, a neutral American accent and a natural speaking
rate". The line is the only thing that changes, and every version ends on
"Daniel", so the last word is easy to check.

| Entry | Frames (s) | Words | Written words/s | Line |
| --- | --- | --- | --- | --- |
| r150 | 124 (5.17) | 8 | 1.55 | I left the light on for you, Daniel. |
| r200 | 124 | 10 | 1.94 | I left the porch light on, come home soon, Daniel. |
| r260 | 124 | 13 | 2.52 | I left the porch light on last night, so come home soon, Daniel. |
| r320 | 124 | 17 | 3.29 | I left the porch light on all last night, so please come home and call me, Daniel. |
| r400 | 124 | 21 | 4.06 | I left the porch light on last night, so please come home before the storm rolls in and call me, Daniel. |
| l200 | 345 (14.38) | 32 | 2.23 | (two sentences, ends "…so please come home soon, Daniel.") |
| l400 | 345 | 57 | 3.97 | (five sentences, ends "…and call me, Daniel.") |

Measured from each clip's audio track. *Lead* is the silence before the
first word. *Tail* is the silence after the last. *Delivered* is words
divided by the time from the first word to the last.

| Run | Lead (s) | Tail (s) | Delivered words/s | Pauses ≥ 0.3 s | Every word, incl. "Daniel"? |
| --- | --- | --- | --- | --- | --- |
| r150 s1001 | 3.35 | 0.52 | 6.15 | none | yes |
| r200 s1001 | 2.10 | 1.22 | 5.41 | none | yes |
| r260 s1001 | 0.40 | 0.67 | 3.17 | 0.5, 0.6 | yes |
| r320 s1001 | 0.25 | 0.67 | 4.00 | 0.95 | yes |
| r400 s1001 | 0.25 | 0.37 | 4.62 | none | yes |
| l200 s1001 (345 fr) | 0.70 | 0.87 | 2.50 | 0.9, 1.15, 2.1, 1.35 | yes |
| l400 s1001 (345 fr) | 0.25 | 0.37 | 4.15 | six | yes |
| r150 s2002 | 3.15 | 0.47 | 5.16 | none | yes |
| r200 s2002 | 2.15 | 0.12 | 3.45 | 0.9 | yes |
| r260 s2002 | 1.00 | 0.17 | 3.25 | 1.3 | yes |
| r320 s2002 | 0.25 | 0.22 | 3.62 | 1.0 | yes |
| r400 s2002 | 0.25 | 0.12 | 4.37 | none | yes |

What this says about the rule:
- **Clipping: refuted.** All 12 clips speak every word, up to 4.06 written
  words/s at 124 frames and 3.97 at 345 frames. At high rates H3 speeds the
  delivery up (4.4–4.6 words/s) rather than dropping words.
- **"Stretches dialogue to fill the clip": refuted.** A short line isn't
  slowed down. H3 holds silence for 2–3.4 s, then speaks the line fast. A
  mid-length line (2.5 words/s) fills the clip with pauses of 0.5–1.3 s
  between phrases. 2.6 words/s isn't a threshold where anything changes. It
  is roughly where the silent lead-in disappears and phrase pauses start.
- **Tail: smaller than the rule assumes.** It was 0.12–1.22 s, with a median
  of about 0.4 s. Above about 3 written words/s the tail was 0.1–0.7 s, yet
  the last word always finished. A "leave about 1 s" budget is a safety
  margin, not something the model guarantees.
- **Words/s for C1/C2:** H3 delivered 3.2–4.6 words/s whenever the line was
  long enough to fill the clip. Lines up to about 4 written words/s fit
  without loss.

Seed 1001 r150 also showed a lip-sync problem. The man's mouth moves during
the silent 3.35 s lead-in (frames at 1–3 s), before any audio. A short line
in a long clip may desync lips from voice. That belongs to a lip-sync probe,
not this rule.

## Probe 3: soft-delivery words on a male voice

Prompt: a bearded man beside a crib in a dim nursery, "an adult male with a
deep bass voice", saying "Go to sleep now, little one. The storm has passed,
and everyone is safe." The arms change only the delivery clause:
- **broken:** "speaking in a soft, gentle whisper, like a lullaby";
- **followed:** "speaking slowly and quietly, low and steady in his deep
  male register".

Pitch was estimated by autocorrelation over 40 ms windows (60–400 Hz), on
frames within 20 dB of the clip's peak:

| Run | Median F0 (Hz) | IQR (Hz) | Voiced share of loud frames |
| --- | --- | --- | --- |
| broken s1001 | 103 | 86–141 | 47% |
| followed s1001 | 94 | 87–105 | 54% |
| broken s2002 | 85 | 82–88 | 66% |
| followed s2002 | 85 | 80–89 | 68% |

All four are in the adult male range. The rule's effect (the voice flipping
female) didn't appear on either broken seed, so the rule is refuted for
these words on a voice already described as male and deep. The probe
doesn't cover a male line whose voice description is unspecified. The rule
may come from such prompts.

Jobs: 6f588db1f043 (run 20261006-104355-033b4212, seed 1001) and
3a29df186aac (run 20261006-105807-b68bc035, seed 2002), entries
`p3_broken_s*` and `p3_followed_s*`.

## Probe 6: control tags and a bare ellipsis inside `<d>`

Prompt: a woman in a hallway, facing the camera. The arms:
- **broken:** `<d>[English] I waited up all night. <pause> You never called.
  <softer> I was so worried… <breath> Just come home.</d>`
- **followed:** the same line with the tags removed and the `…` replaced by a
  full stop.

| Run | Transcript |
| --- | --- |
| broken s1001 | I waited up all night, you never called. I was so worried. Just come home. |
| broken s2002 | I waited up all night, you never called, I was so worried, just come home. |
| followed s1001 | I waited up all night, you never called. I was so worried. Just come home. |
| followed s2002 | I waited up all night. You never called. I was so worried. Just come home. |

Neither broken seed spoke a tag word or added a word, so the rule is
refuted on this line. The only trace the tags left was in timing. In seed
1001 there is a 0.56 s gap after "worried", where `<breath>` sat. Whisper
heard no breath sound there, and in seed 2002 the gap is closed. No tag
reliably produced a pause.

One limit: Whisper-base could drop a very soft spoken tag word. The finding
is "not transcribed", not "proved silent".

Jobs: 6f588db1f043 and 3a29df186aac, entries `p6_broken_s*` and
`p6_followed_s*`.

## Probe 7: prompt length

Resolved by reading the encoder source on `lem`. A GPU run would show
nothing a fixed truncation would not.

In diffusers 0.41.0.dev0, `diffusers/modular_pipelines/minimax_h3/encoders.py`:
- line 194 (text-to-video):
  `token_ids = components.tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"]`;
- line 298 (the image-conditioned path) and line 578 (ref2va) tokenize the
  same way.

The token ids are passed whole to `get_qwen3vl_prompt_embeds`, which builds
`torch.tensor([token_ids])` with an all-ones attention mask and takes hidden
layer 50 of the Qwen3-VL encoder. The module has no `max_sequence_length`
argument and no truncation anywhere.

So no encoder limit makes 7,000 characters a hard ceiling: a longer prompt
reaches the transformer whole. Any budget would be about quality (attention
spread over a long prompt) or memory, not truncation. This probe didn't
measure either.

## Not run

Probes 2, 4, 5 and 8 weren't run. Don approved 2.5 GPU hours for A1 and A2
together, and A1's share was about 90 minutes. Probe 1 used 59 minutes at
measured cost:
- the 7-entry batch took 42 min;
- each 124-frame entry takes about 3.4 min.

Probes 3 and 6 took 28 minutes (87 in all). At two seeds per arm, the four
skipped probes need about:
- 3 probes × 2 arms × 2 seeds × 3.4 min ≈ 41 min for probes 2, 4 and 5;
- 2 chains × 2 segments × 2 seeds ≈ 30 min for probe 8.

That is about 70–90 GPU minutes in total. They are prompt designs, ready to
run:
- **2:** two men and one woman exchanging lines, against one man and one
  woman. Judged on whether a line is spoken in the wrong voice.
- **4:** a speaker plus a silent listener in shot, with and without "his lips
  stay pressed together" for the listener.
- **5:** a phone call heard on speaker, held by an on-screen person, with and
  without a sentence keeping the holder's lips closed.
- **8:** one 2-segment `chain-video-continuity` chain per arm. One arm starts
  the second segment mid-word, with no `<breath>`; the other starts from
  rest, opens on `<breath>`, and has no words in the warm-up prefix.

## Method notes and deviations

- **Batching.** Probes ran through one inline workflow,
  `qa/h3-probe-batch`, saved in the probe workspace. It uses the same H3
  T2VA turbo step as the shots-batch template (`for_each` over
  `variable:shots`, with `release_pipeline`), plus a `transcribe_audio` step
  (`timestamps: "word"`) on each shot. A batch uses a single seed, so each
  seed is its own job.
- **Timing comes from the audio, not Whisper.** Whisper-base word timestamps
  always start the first word at 0.0 and stretch early words across the
  silent lead-in. In r150 it put "I" at 0–3.44 s, but the audio is silent
  until 3.35 s. Lead, tail and pauses were measured from the WAV instead:
  - RMS in 50 ms windows;
  - a window counts as speech when it is within 25 dB of the clip's peak.

  Whisper's text was used only to check that every word was spoken.
- **Probe 1 at 345 frames** ran two rates (2.2 and 4.0 words/s) on seed
  1001 only, not all five rates on both seeds. 345 frames costs about three
  times as much as 124, and the 124-frame sweep had already shown no
  clipping.
- **Probe 7** was answered from the encoder source rather than a GPU run
  (see above).
