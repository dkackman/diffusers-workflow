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
| 1 | "about 2.6 words/s … leaving about 1 s of tail. H3 stretches dialogue to fill the clip and clips the last word." | **refuted** (clipping and stretching). The tail is real but usually smaller than 1 s. | be681f38d5a8, 3541c1b0a79c; timings re-read in ae52011e2f05 | No word was lost at any written rate from 1.55 to 4.06 words/s at 124 frames (both seeds), or at 2.2 and 4.0 words/s at 345 frames. Short lines aren't stretched: they get a 2–3.4 s silent lead-in and are then spoken fast (5–6 words/s). Tails ran 0.09–0.88 s, median about 0.37 s, by word timestamps after #661. See *Probe 1* below. |
| 2 | "At most one male and one female speaker per scene, or the voices mix." | **refuted** | 04a238ab66d4, 0934c1b76fef | With two men and one woman, both seeds gave each line its own voice: the woman at 222–229 Hz, the two men clearly apart by speaker embedding (cosine about 0, against 0.52–0.55 when one man speaks two lines). The lines were spoken in the right order by the right person. See *Probe 2*. |
| 3 | "'soft / gentle / whisper / lullaby' in a male line's delivery can flip the voice female." | **refuted** | 6f588db1f043, 3a29df186aac | The voice stayed male on both arms and both seeds, with median pitch 85–103 Hz (a female voice sits around 165–255 Hz). In seed 1001 the soft-delivery wording made the delivery breathier (fewer voiced frames, a wider pitch spread) but not female. See *Probe 3*. |
| 4 | "Every silent on-screen person needs their own 'lips stay pressed together' sentence." | **refuted** | 04a238ab66d4, 0934c1b76fef | Without the sentence, the silent listener kept her mouth closed on both seeds while the speaker talked. See *Probe 4*. |
| 5 | "A voice-only source (a radio, a phone) gets lip-synced by whoever is holding it." | **refuted** | 04a238ab66d4, 0934c1b76fef | Without the sentence, the man holding the speakerphone kept his lips closed through the caller's line on both seeds. See *Probe 5*. |
| 6 | "`<pause>` and `<softer>` are spoken aloud; a bare '...' produces invented words." | **refuted** | 6f588db1f043, 3a29df186aac | Neither seed spoke `pause`, `softer` or `breath`, and the bare `…` produced no invented words. Both seeds spoke the line as written. See *Probe 6*. |
| 6b | "`<breath>` is honoured as a silent beat." (the condition for the budget's `<breath>` term coming back) | **not spoken aloud**: refuted. **Silent beat: inconclusive** | 6f588db1f043, 3a29df186aac | `<breath>` is never spoken and no breath sound was transcribed. Both broken seeds pause where it sat (1.00 s and 0.75 s), but both followed seeds, with no tag, pause at the same sentence break (0.70 s and 0.30 s). The tag may lengthen the pause, but it doesn't create one. The `<breath>` budget term stays cut. See *Probe 6*. |
| 7 | "Prompt length budget: about 7,000 characters." (meant as the encoder's `max_sequence_length`) | **refuted** (from source, no GPU) | — | The H3 text encoder doesn't truncate. See *Probe 7*. |
| 8 | "A continuation should start from rest, open on `<breath>`, and begin no new words inside the discarded warm-up prefix." | **not run**: GPU budget (Don's decision) | — | Its prompt design is kept in *Not run*. |

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

Measured from each clip's Whisper word timestamps (`whisper-base`,
`timestamps: "word"`). The clips were re-transcribed after #661 trimmed word
bounds to the waveform, in job **ae52011e2f05**: one `transcribe_audio` step
per clip, on its `output:` mp4. Each transcript is
`output:H3ProbeRetranscribe/20261007-021100-f34e5281/final/<step>-0.json`,
and the step is named after the clip (`p1_r150_124` … `p1_l400_345`, with
`_s2` for seed 2002). *Lead* is the first word's start. *Tail* is the clip
length (5.167 s, or 14.375 s at 345 frames) minus the end of "Daniel.".
*Delivered* is words divided by the time from the first word's start to the
last word's end.

| Run | Lead (s) | Tail (s) | Delivered words/s | Every word, incl. "Daniel"? | Envelope lead / tail (s) | Envelope pauses ≥ 0.3 s |
| --- | --- | --- | --- | --- | --- | --- |
| r150 s1001 | 3.38 | 0.47 | 6.06 | yes | 3.35 / 0.52 | none |
| r200 s1001 | 2.12 | 0.73 | 4.31 | yes | 2.10 / 1.22 | none |
| r260 s1001 | 0.38 | 0.69 | 3.17 | yes | 0.40 / 0.67 | 0.5, 0.6 |
| r320 s1001 | 0.26 | 0.49 | 3.85 | yes | 0.25 / 0.67 | 0.95 |
| r400 s1001 | 0.26 | 0.21 | 4.47 | yes | 0.25 / 0.37 | none |
| l200 s1001 (345 fr) | 0.70 | 0.88 | 2.50 | yes | 0.70 / 0.87 | 0.9, 1.15, 2.1, 1.35 |
| l400 s1001 (345 fr) | 0.24 | 0.28 | 4.11 | yes | 0.25 / 0.37 | six |
| r150 s2002 | 1.94 ("I"); 3.24 ("left") | 0.49 | 2.92 (5.56 from "left") | yes | 3.15 / 0.47 | none |
| r200 s2002 | 2.16 ("I"); 3.24 ("left") | 0.09 | 3.42 (5.43 from "left") | yes | 2.15 / 0.12 | 0.9 |
| r260 s2002 | 1.00 | 0.17 | 3.25 | yes | 1.00 / 0.17 | 1.3 |
| r320 s2002 | 0.28 | 0.11 | 3.56 | yes | 0.25 / 0.22 | 1.0 |
| r400 s2002 | 0.28 | 0.11 | 4.39 | yes | 0.25 / 0.12 | none |

The last two columns are the first hand-off's measurements from the WAV
envelope (*Method notes*). They are kept as a cross-check, and the
transcript figures are the ones C2 declares, since they can be re-read over
MCP. The two methods agree to within 0.1 s except in three places:
- **r200 s1001's tail** is 0.73 s by transcript and 1.22 s by envelope. The
  1.22 was the only tail over 0.9 s, so the corrected range is narrower.
- **Seed 2002 r150 and r200** start with an isolated "I" (1.94–1.96 s and
  2.16–2.26 s), then a gap, then the rest of the line from "left" at 3.24 s.
  In r200 the envelope saw the "I" too (lead 2.15). In r150 it did not (lead
  3.15), so that "I" is quiet, more than 25 dB below the peak. Rates are
  given both ways. From "left", both lines are spoken at 5.4–5.6 words/s,
  like seed 1001's r150.
- **Some inner words still span a pause** after #661: "please" at
  10.62–12.26 s in l200, "come" at 2.66–4.20 s in r260 s2002, and "please" at
  2.26–3.54 s in r320 s2002. Pauses are therefore still taken from the
  envelope. #661 trims each word's bounds to speech, so lead and tail (the
  outer edges) are correct.

What this says about the rule:
- **Clipping: refuted.** All 12 clips speak every word, up to 4.06 written
  words/s at 124 frames and 3.97 at 345 frames. At high rates H3 speeds the
  delivery up (4.1–4.5 words/s) rather than dropping words.
- **"Stretches dialogue to fill the clip": refuted.** A short line isn't
  slowed down. H3 holds silence for 2–3.4 s, then speaks the line fast. A
  mid-length line (2.5 words/s) fills the clip with pauses of 0.5–1.3 s
  between phrases. 2.6 words/s isn't a threshold where anything changes. It
  is roughly where the silent lead-in disappears and phrase pauses start.
- **Tail: smaller than the rule assumes.** It was 0.09–0.88 s, with a median
  of about 0.37 s (job ae52011e2f05). Above about 3 written words/s the tail
  was 0.11–0.49 s, yet
  the last word always finished. A "leave about 1 s" budget is a safety
  margin, not something the model guarantees.
- **Words/s for C1/C2:** H3 delivered 3.2–4.5 words/s whenever the line was
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
refuted on this line.

**`<breath>` (row 6b).** It sat between "worried…" and "Just". The silence
there was measured the same way as probe 1 (50 ms RMS windows, speech
within 25 dB of the clip's peak), on all four clips:

| Run | Silence before "Just" | Whisper's "worried" / "Just" (s) |
| --- | --- | --- |
| broken s1001 | 1.00 s, from 3.25 s | 2.66–3.56 / 4.12 |
| followed s1001 | 0.70 s, from 3.55 s | 2.88–3.80 / 3.80 |
| broken s2002 | 0.75 s, from 3.55 s | 2.98–3.84 / 3.84 |
| followed s2002 | 0.30 s, from 3.60 s | 3.00–3.78 / 3.78 |

Both broken seeds pause there, but so do both followed seeds, which have no
tag, at the same sentence break. The broken pause is 0.30–0.45 s longer on
each seed, so the tag may lengthen a pause H3 already takes. It doesn't
create a beat that wouldn't otherwise be there, and no breath sound was
transcribed. By the confirm bar (neither followed seed shows the effect),
"honoured as a silent beat" is **inconclusive**, so the plan's `<breath>`
budget term stays cut. (An earlier draft of this doc gave 0.56 s for seed
1001 and called seed 2002's gap closed. That came from a looser reading.
The table above replaces it.)

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

## Probe 2: three speakers in one scene

Prompt: three people at a kitchen table, each line in its own `<d>`:
- an older man, "a deep gravelly bass": "The bakery closes early today.";
- a woman, "a bright high soprano": "Then we leave before noon.";
- a young man, "a light tenor": "I will drive us."

The **followed** arm removes the young man, and the older man says line 3.

Each line was cut out of the audio at Whisper's end-of-line word ("today",
"noon", "us"). Per line: median pitch (as in *Probe 3*) and a speechbrain
ECAPA speaker embedding, compared by cosine.

| Run | F0 L1 / L2 / L3 (Hz) | cos L1–L2 | cos L1–L3 | cos L2–L3 |
| --- | --- | --- | --- | --- |
| broken s1001 | 91 / 222 / 130 | 0.08 | 0.08 | -0.03 |
| broken s2002 | 144 / 229 / 144 | 0.14 | -0.00 | -0.01 |
| followed s1001 | 98 / 184 / 103 | ~0 | **0.55** | ~0 |
| followed s2002 | 91 / 211 / 108 | 0.12 | **0.52** | 0.02 |

The followed arm is the method's control: the same man speaking lines 1
and 3 scores 0.52–0.55, while different people score about 0. On the
broken arm every pair is about 0, so the three lines are three voices on
both seeds. In seed 2002 the two men share a median pitch (144 Hz) but not
a voice. The frames of broken s1001 show the old man, the woman and the
young man gesturing and speaking in turn, in line order. Every transcript
was exact. The rule's effect (voices mixing) didn't appear.

## Probe 4: a silent listener in shot

Prompt: two women on a park bench. The older woman ("a warm low alto")
says "Your grandfather proposed to me on this very bench, fifty years
ago." The young woman listens and nods. The **followed** arm adds "The
young woman stays silent the whole time; her lips stay pressed together."

On both broken seeds, frames from 0.5 s to 5 s show the young woman with a
closed-mouth smile throughout, while the older woman's mouth moves with
the line. The listener never lip-syncs, so the rule's effect is absent
without the sentence. Every transcript was exact.

## Probe 5: a voice from a phone

Prompt: a man at an office desk holds a phone on speaker. A woman's voice
from the phone (off-screen, "thin and tinny") says "The flight has been
delayed, so I will land around midnight." The **followed** arm adds "The
man only listens and stays silent; his lips stay pressed together the
whole time."

On both broken seeds the man's lips stay closed from 0.8 s to 4.8 s while
the line plays. He frowns and looks at the phone; he doesn't mouth the
words. Every transcript was exact. Seed 2002's broken clip was quiet
(mean -40 dBFS, peak -19), which fits a voice coming from a phone, and
was still transcribed in full.

Probes 2, 4 and 5 ran as jobs 04a238ab66d4 (run 20261006-123950-799ef10c,
seed 1001) and 0934c1b76fef (run 20261006-130008-42cf313b, seed 2002),
entries `p2_*`, `p4_*` and `p5_*`. For probes 4 and 5 the verdict rests on
the broken arm, which lacked the effect on both seeds; the followed arm
can't change a refutation, so its frames were not reviewed in detail.

## Not run

Probe 8 wasn't run, by Don's decision on #640 (accept it as not run;
run probes 2, 4 and 5 instead). GPU spent on A1, at measured cost:
- probe 1: 59 min (its 7-entry batch took 42 min);
- probes 3 and 6: 28 min;
- probes 2, 4 and 5: 40 min (two 6-entry batches, about 20 min each).

That is about 127 minutes in all, inside the raised A1+A2 cap of about
3.25 hours. Probe 8 would need about 30 more (2 chains × 2 segments × 2
seeds). Its prompt design, ready to run:
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
- **Timing: word timestamps after #661, the envelope as a cross-check.** At
  the first hand-off, Whisper-base word timestamps always started the first
  word at 0.0 and stretched early words across the silent lead-in. In r150
  it put "I" at 0–3.44 s, but the audio is silent until 3.35 s. So lead,
  tail and pauses were measured from the WAV instead:
  - RMS in 50 ms windows;
  - a window counts as speech when it is within 25 dB of the clip's peak.

  #661 then trimmed `transcribe_audio`'s word bounds to the waveform. On
  Don's decision (#640, option 1), probe 1's clips were re-transcribed
  through `output:` references, in job ae52011e2f05 (Whisper only, no H3
  run). Probe 1's lead, tail and delivered rate now come from those
  transcripts, so C2's tail figure can be checked over MCP. The envelope
  figures stay alongside them. Pauses still come from the envelope, because
  a word can still span a pause (*Probe 1*). Job f07d705442ed in the same
  workspace was a first, malformed attempt at the re-transcription (a
  `for_each` over the mp4 references, which `transcribe_audio` refused for
  having no `sample_rate`). It produced nothing and isn't a probe.
- **Probe 1 at 345 frames** ran two rates (2.2 and 4.0 words/s) on seed
  1001 only, not all five rates on both seeds. 345 frames costs about three
  times as much as 124, and the 124-frame sweep had already shown no
  clipping. Don accepted this reduced arm as recorded on 2026-10-06 (#640,
  option 3): the clipping verdict rests on the 124-frame sweep (5 rates x 2
  seeds) plus these two one-seed 345-frame checks, with no further GPU.
- **Probe 7** was answered from the encoder source rather than a GPU run
  (see above).
- **Probes 2, 4 and 5** ran after the first hand-off, on Don's approval of
  about 41 more GPU minutes (#640). Probe 2's voice comparison uses
  speechbrain's ECAPA speaker embedding (`spkrec-ecapa-voxceleb`), with
  lines split at Whisper's word end times and line 1 starting at the
  audio onset, since Whisper's first-word start was unreliable before #661 (above).
