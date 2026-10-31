# htdemucs as a dialogue/rest splitter: A/B record (#1091 stage A, #1096)

Probe date: 2026-10-31 (server clock). Server `lem`, develop @ 1cc889fa.
Agent: claude-opus-5-5 (anthropic). Workspace `dialogue-stems-1096`.
No engine or skill file was changed. This document only records results.
The listening verdict ("good enough" or not) is Don's: the agent could not
listen to the stems. The leak notes below are read from level measurements
and control separations, and each one says so.

## The question

An H3 shot comes with its dialogue, room tone and foley baked into one track.
Once a score goes under it, there is no clean dialogue left to retime, duck or
re-mix. Can `separate_stems` (htdemucs) take a shot + score mix apart again,
into dialogue (`vocals`) and everything else (`drums` + `bass` + `other`, the
"rest")? The bar is that the `vocals` stem matches the dry shot and that
transcribing it gives the shot's line.

## The recipe

This is a mix, then `separate_stems`, then `mix_audio` over the three non-vocal
stems to make the rest. Only these four steps are the recipe. The measurement
workflow that ran it (below) adds levels, transcripts and controls around them.

```json
{
    "id": "DialogueStems",
    "variables": {
        "shot": "output:KnownIssuesRestorePointA/20261013-013302-bdc190ac/intermediate/KnownIssuesRestorePointA-shot@deflect.1-0.0.mp4",
        "score": "output:templates/minimax/music/20260922-134436-b40b67b6/final/MiniMaxMusic-balanced.1-0.0.mp3",
        "score_start": 8,
        "shot_seconds": 5.17,
        "score_gain": 0.5
    },
    "steps": [
        {
            "name": "bed",
            "task": {
                "command": "slice_audio",
                "arguments": {
                    "audio": "variable:score",
                    "start_seconds": "variable:score_start",
                    "duration_seconds": "variable:shot_seconds"
                }
            }
        },
        {
            "name": "mix",
            "task": {
                "command": "mix_audio",
                "arguments": {
                    "audios": ["variable:shot", "previous_result:bed"],
                    "gains": [1, "variable:score_gain"]
                }
            },
            "result": { "content_type": "audio/wav", "subfolder": "intermediate", "file_base_name": "mix" }
        },
        {
            "name": "stems",
            "task": {
                "command": "separate_stems",
                "arguments": { "audio": "previous_result:mix" }
            },
            "result": { "content_type": "audio/wav", "subfolder": "final", "file_base_name": "stems" }
        },
        {
            "name": "rest",
            "task": {
                "command": "mix_audio",
                "arguments": {
                    "audios": ["previous_result:stems.drums", "previous_result:stems.bass", "previous_result:stems.other"]
                }
            },
            "result": { "content_type": "audio/wav", "subfolder": "final", "file_base_name": "rest" }
        }
    ]
}
```

Each run returned `stems-0.0-{vocals,drums,bass,other}.wav` (44.1 kHz stereo)
and `rest-0.0.wav`. `mix_audio` warns that it resampled, because the H3 shots
are 32 kHz and the score is 44.1 kHz. That warning is expected. **No run raised
`stacked_vocals`**: these are single-speaker dialogue shots.

The measurement runs added the following steps:
- `analyze_audio` on the dry shot, the `vocals` stem, the rest, the bed, and
  the bed at its mix gain (`bed_scaled`);
- `transcribe_audio` (whisper-base) on the dry shot, the `vocals` stem and
  the bed;
- two controls, each separating one input on its own:
  - `ctl_bed` separates the gain-scaled bed alone, and its `vocals` stem is
    the score's leak into dialogue;
  - `ctl_dry` separates the dry shot alone, and its drums + bass + other is
    the shot's own non-speech content, which lands in the rest.

The controls stand in for a subtraction: `mix_audio` refuses negative gains, so
"vocals − dry" can't be rendered directly.

Score: Music 3 `MiniMaxMusic-balanced.1` (42.25 s), from `acorn-wars`. Score
gain was 0.5 in every mix. Levels are dBFS (`analyze_audio`, mean over channels
where it reports per channel).

## Mix 1: deflect

The shot is `KnownIssuesRestorePointA-shot@deflect.1` (5.17 s), with the score
from 8 s. Job `98c9dea3612f`, run `DialogueStems/20261031-172140-2a115543`.

| Signal | peak | rms | low | mid | high |
| --- | --- | --- | --- | --- | --- |
| dry shot | -1.29 | -21.72 | -31.40 | -22.67 | -43.03 |
| `vocals` stem | -1.24 | -21.71 | -31.19 | -22.67 | -43.01 |
| rest | -20.10 | -34.20 | | | |
| bed at mix gain | -18.68 | -32.81 | | | |
| control: bed-alone `vocals` | -60.77 | -62.79 | | | |
| control: dry-alone rest | -22.40 | -60.43 | | | |

The vocals are within 0.05 dB of the dry shot in peak, 0.01 dB in rms, and 0.2
dB in every band.

| Transcript of | Text |
| --- | --- |
| dry shot | "Of course, we have backups every night since 1994." then an "AHHHH…" loop (1824 chars) |
| `vocals` stem | "Of course, we have backups every night since 1994. Yeah!" |
| bed alone | "Thank you very much for watching." (Whisper's hallucination on music) |

The `drums` and `bass` stems are near-silent and warn so: this section of the
score has no drums or bass.

## Mix 2: button

The shot is `KnownIssuesRestorePointB-shot@button.0` (5.17 s), with the score
from 20 s. Job `af3f52108e3b`, run `DialogueStems/20261031-172221-2a0e0902`.

| Signal | peak | rms | low | mid | high |
| --- | --- | --- | --- | --- | --- |
| dry shot | -0.80 | -23.01 | -31.44 | -24.56 | -40.52 |
| `vocals` stem | -0.94 | -23.06 | -31.41 | -24.61 | -40.70 |
| rest | -12.55 | -28.05 | | | |
| bed at mix gain | -12.60 | -27.52 | | | |
| control: bed-alone `vocals` | -58.91 | -63.69 | | | |
| control: dry-alone rest | -12.95 | -46.49 | | | |

In the dry-alone control, the shot's own non-speech audio lands in `drums`
(mean -46.75, peak -12.85). It is transient SFX or foley, not speech.

| Transcript of | Text |
| --- | --- |
| dry shot | "We don't test them. That's how you lose them." |
| `vocals` stem | "We don't test them. That's how you lose them." |
| bed alone | "Thank you very much." (hallucination) |

## Mix 3: tag

The shot is `KnownIssuesRestorePointB-shot@tag.1` (5.875 s), with the score
from 30 s. Job `0e7e6bb039ef`, run `DialogueStems/20261031-172507-2722c9f3`.

| Signal | peak | rms | low | mid | high |
| --- | --- | --- | --- | --- | --- |
| dry shot | -1.38 | -23.26 | -33.63 | -23.93 | -47.44 |
| `vocals` stem | -1.21 | -23.35 | -33.56 | -24.01 | -48.25 |
| rest | -11.05 | -31.28 | | | |
| bed at mix gain | -10.93 | -31.48 | | | |
| control: bed-alone `vocals` | -55.42 | -62.34 | | | |
| control: dry-alone rest | -12.43 | -42.06 | | | |

In the dry-alone control, the shot's non-speech audio again goes to the rest:
`drums` has mean -47.16 and peak -12.68, and `other` has mean -44.22. The `bass`
stem of the mix is near-silent and warns.

| Transcript of | Text |
| --- | --- |
| dry shot | "So we have no idea if they work. We have hope." then a "Thank you very much." loop (2728 chars) |
| `vocals` stem | "So we have no idea if they work. We have hope." |
| bed alone | "Thank you so much for watching!" (hallucination) |

## A non-dialogue negative: s04

`acorn-wars` `H3ShotsBatch-shot@s04-the-call-r4.3` was the first pick. It turned
out to hold no speech (run `DialogueStems/20261031-172300-7bc2933b`):
- the dry-shot transcript was "Thank you very much for watching.", which is
  Whisper's hallucination;
- the `vocals` stem averaged -53.65 dBFS, and its transcript was a string of
  garbage glyphs.

It was replaced by `tag`. It is recorded here because it shows what the recipe
does on a speechless shot. The `vocals` stem comes back near-silent, and Whisper
then invents a line, so an empty transcript can't be relied on to say "no
dialogue". Check the stem's level instead.

## Leak notes (from measurements; listening is Don's)

| Mix | `vocals` stem: score leak | rest: dialogue leak | rest: what else is in it |
| --- | --- | --- | --- |
| deflect | Bed-alone `vocals` is about 30 dB under the bed (rms -62.8 vs. -32.8). Expect it to be inaudible under speech. | Dry-alone rest is at rms -60.4. Expect no audible speech in the rest. | Score only. Rest ≈ bed (-34.2 vs. -32.8 rms). |
| button | About 36 dB under the bed (-63.7 vs. -27.5). | Dry-alone rest is at rms -46.5, peak -13. That is the shot's SFX/foley, routed to `drums`, not speech. | Score plus the shot's own SFX. |
| tag | About 31 dB under the bed (-62.3 vs. -31.5). | Dry-alone rest is at rms -42.1, peak -12.4, the shot's SFX in `drums`/`other`. | Score plus the shot's own SFX. |

What to listen for in the kept stems:
- whether music bleeds into `vocals` between words (the controls put it at
  -55 to -61 dBFS peak);
- whether consonants (s, t) bleed into the rest;
- whether the SFX that went to the rest belongs with the dialogue for the
  scene's purpose.

## Findings

1. **The `vocals` stem is the dry dialogue.** On all three shots it matches
   the dry shot within 0.1 dB rms and 0.2 dB peak, and within 0.8 dB in every
   band (the largest gap is tag's high band).
2. **The score barely reaches `vocals`.** Separating the bed alone puts
   -62 to -64 dBFS rms into `vocals`, about 30–36 dB under the bed's level in
   the mix.
3. **The rest is the score plus the shot's own non-speech audio.** The rest's
   rms tracks the bed's within 1.4 dB. htdemucs sends an H3 shot's SFX and
   foley to `drums`/`other`, not `vocals`, so the rest isn't a clean score: it
   carries the shot's room and foley too. When the goal is to re-mix or replace
   the score, keep the original score file. The rest is for when the score file
   is lost or the score is to be ducked as a unit with the effects.
4. **Transcribing `vocals` beats transcribing the dry shot.** All three
   `vocals` transcripts gave the shot's line. On two of three shots the dry
   audio sent whisper-base into a hallucination loop that the stem didn't. The
   one extra word ("Yeah!" on deflect) is a vocalization in the shot.
5. **No `stacked_vocals` on single-speaker dialogue.** The check didn't fire
   on any shot.
6. **A speechless shot gives a near-silent `vocals` stem and an invented
   transcript** (s04). Check the stem's level, not the transcript.

## Kept stems

Each stem is kept in workspace `dialogue-stems-1096` as an `asset:` and is still
readable as an `output:`.

| Mix | `output:` | `asset:` |
| --- | --- | --- |
| deflect | `DialogueStems/20261031-172140-2a115543/final/stems-0.0-vocals.wav` | `dialogue-stems-deflect-vocals.wav` |
| deflect | `DialogueStems/20261031-172140-2a115543/final/stems-0.0-drums.wav` | `dialogue-stems-deflect-drums.wav` |
| deflect | `DialogueStems/20261031-172140-2a115543/final/stems-0.0-bass.wav` | `dialogue-stems-deflect-bass.wav` |
| deflect | `DialogueStems/20261031-172140-2a115543/final/stems-0.0-other.wav` | `dialogue-stems-deflect-other.wav` |
| deflect | `DialogueStems/20261031-172140-2a115543/final/rest-0.0.wav` | `dialogue-stems-deflect-rest.wav` |
| button | `DialogueStems/20261031-172221-2a0e0902/final/stems-0.0-vocals.wav` | `dialogue-stems-button-vocals.wav` |
| button | `DialogueStems/20261031-172221-2a0e0902/final/stems-0.0-drums.wav` | `dialogue-stems-button-drums.wav` |
| button | `DialogueStems/20261031-172221-2a0e0902/final/stems-0.0-bass.wav` | `dialogue-stems-button-bass.wav` |
| button | `DialogueStems/20261031-172221-2a0e0902/final/stems-0.0-other.wav` | `dialogue-stems-button-other.wav` |
| button | `DialogueStems/20261031-172221-2a0e0902/final/rest-0.0.wav` | `dialogue-stems-button-rest.wav` |
| tag | `DialogueStems/20261031-172507-2722c9f3/final/stems-0.0-vocals.wav` | `dialogue-stems-tag-vocals.wav` |
| tag | `DialogueStems/20261031-172507-2722c9f3/final/stems-0.0-drums.wav` | `dialogue-stems-tag-drums.wav` |
| tag | `DialogueStems/20261031-172507-2722c9f3/final/stems-0.0-bass.wav` | `dialogue-stems-tag-bass.wav` |
| tag | `DialogueStems/20261031-172507-2722c9f3/final/stems-0.0-other.wav` | `dialogue-stems-tag-other.wav` |
| tag | `DialogueStems/20261031-172507-2722c9f3/final/rest-0.0.wav` | `dialogue-stems-tag-rest.wav` |

The mixes the stems came from are each run's `intermediate/mix-0.0.wav`.

## Verdict

Pending Don's listening. A `series-episodes` paragraph is written only if the
verdict is "good enough" (plan v2, stage A).
