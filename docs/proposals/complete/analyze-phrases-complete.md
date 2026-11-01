# Phrase timing without Whisper: `analyze_phrases` and `plan_cuts` (#1095)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-01). Plan v1 on #1095 was approved on 2026-10-31 under Don's
delegation with every default (Q1-Q3). v2 folded those answers in with no
change of scope. v3 re-planned after a failed field verify, and Don ruled
Q4 = (a), loosen, on 2026-11-01. It was built as two stages: #1104
(stage A) and #1105 (stage B).

## The report

`plan_cuts` (#600) gets all of its line timing from a `transcribe_audio`
transcript. When Whisper can't hear the words (a non-English song, a
hallucination, wordless vocals or vocalise), there is no fallback. The
field case was `russian-anthem` job `0838f676f51f` (2026-10-12). Whisper
turned a Russian verse into the subtitle-credit hallucination
"Россия Редактор субтитров А.Семкин", and the session gave up on the
planner and cut the song by hand. The same class as #822.

The concept (centre-channel, voice-band energy onsets) comes from
Longshot_Studio's `audio_sync.detect_phrases`. That repo has no license,
so no code was copied: only the idea was taken.

## Verdict

**Build smaller.** One new JSON task, `analyze_phrases`, and `plan_cuts`
accepting its result where it takes a transcript. No new template, no
`analyze_beats` mode, and no MCP tool, REST route or syntax change. The
stereo-centre weighting stays in as a default-on detail (`centre=true`,
Q1), because it costs about 15 lines and the template feeds `plan_cuts` a
vocal stem anyway (#822).

## What the plan found against the issue

1. **The result is a dict, not a list.** `measure_sync` returns a dict,
   and a list result would be saved as one artifact per item.
2. **Empty-text chunks don't "just work" in `plan_cuts`.**
   `_transcript_lines` drops them, so a phrase list would plan as one
   instrumental shot. `plan_cuts` recognises the phrase result by its
   `phrases` key and handles it explicitly.
3. **It is its own task, not an `analyze_beats` mode.** That task
   downmixes on load, and its result shape is for beats.

## What was built

### Stage A (#1104): the `analyze_phrases` task
- `dw/tasks/phrases.py`, registered with `returns="json"`. It answers
  `{phrases: [{start, end}], duration_seconds, method: "centre"|"band",
  warnings}`, in seconds rounded to 0.01.
- Stereo: max(0, P_mid - P_side) in 250-3500 Hz over 20 ms frames,
  smoothed. Mono, `centre: false` or identical channels: band power of the
  downmix (`method: "band"`, plus a warning on mono input).
- Voiced floor: `voice_attribution.voiced_floor_dbfs` (p95 - 35 dB, never
  below -60 dBFS), unless `floor_db` overrides it. Gaps under
  `min_gap_seconds` (0.35) are bridged, and runs under
  `min_phrase_seconds` (0.3) are dropped.
- Refusals go through the registration's domains and
  `task_problems.phrases_problems`. The task warns on no phrases and on
  one phrase covering more than 90% of the track.
- Docs: a `docs/TASKS.md` entry and a `docs/ARCHITECTURE.md` row.

### Stage B (#1105): `plan_cuts` takes phrases
- A phrase branch in `dw/tasks/cuts.py` (`_is_phrases`, `_phrases`,
  `_phrase_lines`). `transcript_problem` accepts the shape.
- Without `lyrics`: one line per phrase, `lyric: null`. `segment_by`
  `line`, `stanza` and `beat` all work.
- With `lyrics` and matching counts: one to one, in order (Q2). A **disproportion
  warning** (the lead's call at the v3 decompose) names any line whose
  phrase is under 1/4 or over 4x the median phrase length. The pairing
  still stands.
- With mismatched counts: a warning names both counts and the knob, then
  the lines are spread evenly over the first phrase's start to the last
  phrase's end. With more phrases than lines, the warning says to raise
  `min_gap_seconds`. With fewer, it says to lower it, or (v3) to raise
  `floor_db` for fused legato or reverberant lines, and to check the phrases
  by ear before trusting a pairing.
- v3: stage A's 90% warning names both remedies (feed `stems.vocals` on a
  mix, raise `floor_db` on a stem already).
- Docs: `docs/TASKS.md` `plan_cuts`, the `music-video-cuts` template
  description and `workflows/templates/minimax/README.md`, and
  `plugins/dw/skills/minimax-h3/references/cuts.md`. Its stale chain now
  includes `separate_stems`, and it documents the swap: `transcribe` →
  `analyze_phrases` on `previous_result:stems.vocals`, plus `lyrics`.

## What the field case showed (the v3 re-plan)

On lem, `verse_s7.wav` (a legato, reverberant choir, 4 lyric lines) gave
one phrase over 1.44-33.00 s at the defaults. `min_gap_seconds` 0.15 still
gave 1 and 0.05 gave 2. Only `floor_db` -30 gave 4, and those were
2.66-5.12, 7.26-15.58, 16.02-16.36 and 18.40-33.00 s, which don't look like
sung lines. Between legato lines the energy never drops below the
track-relative floor, so the plan's Value claim ("this would have fixed
that run") didn't hold for pairing. It does hold for the timeline: the span
is the vocal's true span, not a hallucinated one.

Don ruled **Q4 = (a), loosen**. B finishes with the advice fix, and the
field case is judged on the true span, the warnings and the even spread,
not on pairing. He also added the pairing caution (`floor_db` can make the
counts match on bad boundaries, so check by ear).

## Deferred, and what brings it back

- **Stage C, a local-level (rolling 1-2 s) voicing floor.** Dropped by the
  Q4 ruling. It would come back with a probe on `verse_s7.wav` that finds a
  setting giving 4 phrases at audible line starts, by ear.
- **Duration-aligned in-order mapping** of lyric lines onto phrases when
  the counts differ. It stays a follow-up idea and was not filed. The field
  case is the first data point that count mismatches are common. A second
  field report would justify its own design.
- **Telling lead vocals from backing vocals, and word-level timing without
  Whisper.** Non-goals.

## Verification and bounces

| Stage | Bounces | Verified |
|---|---|---|
| A (#1104) | 0 | mini-ai (mps), develop @ 65f26281: C-F576-C-F582 |
| B (#1105) | 2: a failed field verify (C-F589 on lem, which led to the v3 re-plan), and one architecture-review bounce (a hand-rolled median replaced by `statistics.median`) | C-F583-C-F588 and C-F624-C-F627 on mini-ai @ 0e35506a / 46d1ec23. All of them, plus field case C-F628, on lem @ d71bc1f5 |

C-F589 (its pairing clause) is superseded by C-F628. Its retirement is
open as dkackman/harnest#168.

Cost: the stage comments name no `usage:` figures, so none are recorded
here. The plan estimated $6-9 for both stages, plus about $1 for v3.
