# MiniMax-H3 claims from WanGP's field notes: record-only probes (#985 stage A, #988)

Probe date: 2026-10-25. Server `lem`, develop @ b80dec77.
Agent: claude-opus-5-5 (anthropic). Workspace `qa-h3-wangp-probes`.
No engine or skill file was changed. This document only records results.

Pinned configuration (Don, at plan approval), the same for every cited job:
- `weights_dtype` **int8**, with the turbo LoRA **on**. Each job's `note`
  states both.
- Ref2VA (P1, P4 arm b): `transformer_ref`, LoRA
  `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`, 9 steps, video
  shift 12, audio shift 3, `sparse` attention.
- FL2VA (P4 arm a): `transformer`, LoRA
  `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`, 5 steps, shift 6.
- 960x544, 124 frames (5.17 s at 24 fps). Every shot holds its source slice
  with `hold_audio`.

Each arm ran on seeds 1001 and 2002. A claim is **confirmed** only when both
seeds of the claim-broken arm show the effect and neither seed of the
claim-followed arm does. It is **refuted** when the broken arm doesn't show
the effect, and **inconclusive** when the seeds disagree. Claim 4 has no
broken arm (it asks whether a route works, not whether a rule matters). It is
confirmed when both FL2VA seeds keep frame 0 and their mouths follow the
track, and refuted when either fails.

With `hold_audio`, the output soundtrack *is* the source slice. So the
transcripts and `measure_sync` can't tell the arms apart: they measure audio
against audio, not lips. The mouth frames decide every verdict here.

## Verdicts

| # | Claim (as stated on #985) | Verdict | Jobs | Observation |
| --- | --- | --- | --- | --- |
| 1 | "Lip-sync without `<d>`": for lip-sync to supplied audio, leave the words out of `<d>` and ask for natural lip movements synchronized to the track. | **refuted** | sung: 3d9a4e7377fa, 6d7f6faf8e60 (a); d2384deebb50, f27b7a7af223 (b). spoken: c029d63a15e8, c0763cc81d6b (a); 8451e9fd9289, e0790f00b4c2 (b) | With the track held, the words in `<d>` did no harm. On both tracks and both seeds, arm a's mouth opens and shuts at the same instants as arm b's, and the identity holds. See *Probe 1*. |
| 4 | "FL2VA with a held soundtrack" lip-syncs. Our skill says "FL2VA takes no audio" (SKILL.md:31). | **confirmed** | 52195bad071b, f4fe60dcca4d (FL2VA); 422f22a7df71, 8e42ec21cf9a (Ref2VA keyframe) | FL2VA plus `hold_audio` keeps the input picture pixel for pixel at frame 0. On both seeds its mouth follows the held song with the same open and shut pattern as the audio-referenced Ref2VA shots. #852's Ref2VA keyframe route also lip-syncs, but frame 0 keeps only the framing: the arms hang instead of being folded, and the shot is wider. See *Probe 4*. |

Not run: none. Claims 2 and 3 were dropped and deferred by the plan, not
probed.

### `measure_sync` figures

| Track | Jobs | `lag_seconds` | `confidence` | `reference_seconds` | Findings |
| --- | --- | --- | --- | --- | --- |
| sung (la-vela) | all 4 P1 sung, all 4 P4 | 0.0 | 0.175 | 118.751 | `sync_low_confidence` (warn, threshold 0.3) |
| spoken (hal-voice) | all 4 P1 spoken | 0.0 | 0.756 | 2.499 | none |

Lag 0.0 is expected, because the output audio is the held slice. The sung
figure looks unreliable: `audio_seconds` is 5.1667 but `reference_seconds`
is 118.751. The step's `reference` is `previous_result:slice` (a 5.17 s
slice), so `measure_sync` seems to have read something song-length, possibly
the whole source track. The spoken reference length of 2.499 s is about the
speech span, which suggests the trailing silence was trimmed. This is a
possible `measure_sync` or reference-resolution bug, recorded here for
follow-up. It doesn't affect a verdict.

## Probe 1: lip-sync with and without `<d>`

Workflow `probes/wangp-p1-ref2va` has four steps:
1. `slice_audio`: the track at `start_seconds`, `num_frames` 124 at 24 fps.
2. Ref2VA shot: portrait `asset:cast/tomas.jpg` as `<Picture 1>` and the
   slice as `<Audio 1>`, held with `hold_audio: previous_result:slice`.
   Task `[reference generation + audio reuse]`.
3. `measure_sync`.
4. `transcribe_audio` (whisper-small, word timestamps).

One person in frame.

- **Sung track:** `asset:song/la-vela.mp3` from 42.5 s, at 44100 Hz.
- **Spoken track:** `asset:uploads/cast/hal-voice.wav` from 0 s (see
  *Deviations*).
- **Arm a (broken):** the prompt names Tomás, asks for "natural lip
  movements synchronized" to `<Audio 1>`, and spells the words in
  `<d>[English] Pass it to your sister, pass it to your brother, ya no más
  silencio, let the whole hill ring</d>` (the spoken track's line in the
  spoken jobs). The action is limited to singing or speaking: "his arms stay
  folded and his body stays still".
- **Arm b (followed):** the same prompt without the `<d>` sentence.

| Track | Arm | Seed | Job | Run |
| --- | --- | --- | --- | --- |
| sung | b | 1001 | d2384deebb50 | 20261025-150427-fadb957c |
| sung | b | 2002 | f27b7a7af223 | 20261025-150827-df1f8eec |
| sung | a | 1001 | 3d9a4e7377fa | 20261025-151226-b0698795 |
| sung | a | 2002 | 6d7f6faf8e60 | 20261025-151629-4cca84db |
| spoken | b | 1001 | 8451e9fd9289 | 20261025-152027-06d1d37c |
| spoken | b | 2002 | e0790f00b4c2 | 20261025-152655-347effd2 |
| spoken | a | 1001 | c029d63a15e8 | 20261025-153053-a09b5b8d |
| spoken | a | 2002 | c0763cc81d6b | 20261025-153452-401141ed |

**Transcripts.** All eight match their source, as they must under the hold:
- sung: "Pass it to your sister, pass it to your brother Yeah, no more
  silence, you'll let the whole heal ring";
- spoken: "only God and Gary know how to configure the F5." The speech ends
  at about 2.5 s.

**Mouth frames:**
- **Sung, every job:** shut at 0.2 s and 1.1 s; open at 2.45 s, 2.8 s and
  3.3 s, with teeth or a smile; an "o" at 4.6 s. The pattern is the same on
  both arms and both seeds.
- **Spoken, every job:** open at 0.3 s and 2.4 s, then shut through the
  silent tail at 3.5 s and 4.6 s, on both arms and both seeds.

Arm a showed no desync, no extra mouthing and no identity drift, so the
claim is refuted. H3 lip-syncs a held track the same whether or not its
words are in `<d>`.

Limit: the frames were read at 4 to 6 instants per clip, not continuously.
Whisper's word timestamps on sung material are approximate, so this records
the mouth following the track's phrasing, not per-phoneme accuracy.

## Probe 4: FL2VA with a held song

The picture for both arms is frame 0 of d2384deebb50's shot (P1 arm b,
seed 1001): Tomás, arms folded, against a plain backdrop.

The slice is la-vela from 42.5 s at 44100 Hz. P1 had no winner, so the
prompt is in arm b style, as the plan directs.

- **Arm a:** workflow `probes/wangp-p4-fl2va`, with the steps `get_first_frame`,
  `slice_audio`, an fl2va shot with the picture as first frame and
  `hold_audio`, then `measure_sync` and `transcribe_audio`.
- **Arm b:** workflow `probes/wangp-p4-keyframe-ref2va`, #852's route: Ref2VA
  `[keyframe completion + audio reuse]`, with the picture as `<Picture 1>`
  (the first frame) and the slice as `<Audio 1>`, held.

| Arm | Seed | Job | Run | Frame 0 | Mouth |
| --- | --- | --- | --- | --- | --- |
| a (FL2VA) | 1001 | 52195bad071b | 20261025-153850-7c27a9b9 | the picture, pixel for pixel | shut at 1.1 s, open with teeth at 2.8 s, smiling and open at 3.3 s, "o" at 4.6 s |
| a (FL2VA) | 2002 | f4fe60dcca4d | 20261025-161250-1a83c8b5 | the picture, pixel for pixel | same pattern at the same instants |
| b (Ref2VA keyframe) | 1001 | 422f22a7df71 | 20261025-160108-f44246e6 | framing only: arms hang, shot wider | same pattern |
| b (Ref2VA keyframe) | 2002 | 8e42ec21cf9a | 20261025-160742-f53aecc0 | framing only, as seed 1001 | same pattern |

**Mouth motion.** The FL2VA mouth moves with the held track, at the same
instants as in the audio-referenced Ref2VA shots of P1 and of arm b, across
two seeds. A self-driven mouth wouldn't repeat across seeds and task types,
so `hold_audio` drives FL2VA's lips. #808 is the warning sign the plan cites:
there, a song held with image conditioning opened the mouth at only 3 of 6
onsets. This time no checked instant was missed, within the limit stated in
*Probe 1*.

**Side effect.** Seed 2002's FL2VA shot shows faint blotches on the cheeks
through the clip, the int8 smear the skill's `quality-and-speed.md` covers.

**Cost:**

| Arm | Load | Denoise | Decode |
| --- | --- | --- | --- |
| FL2VA (f4fe60dcca4d), 4 passes | 72.6 s | 49.7 s | 18.9 s |
| Ref2VA keyframe | 72.5 s | about 233 s | about 20 s |

The locked-frame lip-sync route that keeps the pixels is also about 4.7
times faster to denoise.

## Deviations

- **Spoken slice relabelled.** The spoken slice ran with `sample_rate` 48000,
  but `hal-voice.wav` is 44100 Hz, so the line plays about 8.8% fast and high.
  Both arms got the identical input, so the comparison stands.
- **OOM failures, rerun.** Three P4 jobs died by SIGKILL from host-RAM
  contention. Two workers (cached FL2VA at about 60 GB plus Ref2VA at about
  95 GB) exceeded lem's 128 GB. After `clear_memory`, each was rerun alone
  with the same bound fingerprint:

  | Failed | Rerun |
  | --- | --- |
  | db506b4f1049 | f4fe60dcca4d |
  | 11eb727cd72d | 422f22a7df71 |
  | 3c097799d825 | 8e42ec21cf9a |

  The failed ids are superseded and not cited as results.
- **Excluded jobs:** 423bcbe3c40f (wiring check, before the relabel was
  settled) and dbcdd254f897 (transcription of the source for prep).

## What stage B does with this (#985 plan v2)

- **Claim 1 refuted:** a one-line "tested and refuted" note in
  `references/dialogue.md`. No enhancer change.
- **Claim 4 confirmed:** SKILL.md:31-34 makes FL2VA plus `hold_audio` the
  locked-frame lip-sync route, which keeps the pixels. Ref2VA
  `[keyframe completion + audio reuse]` becomes the fallback. One line goes
  in `WORKFLOW_GUIDE.md:1740`.
