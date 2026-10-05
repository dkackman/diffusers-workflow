# Checking the lip-sync target from shipped parts (#488)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-05). Plan v2 approved by Don as "build smaller" on 2026-10-05;
one stage, #617 (A), verified.

## The report

A field report from an Opus 5.5 session on lem (2026-09-25/26): user review
of a 24-shot H3 musical found three faults that the frame checks and
`assess_output` had passed:

1. **Anatomy vs props:** a crowd told to clap while holding candles rendered
   with three hands.
2. **Direction of travel:** "march up to the house" rendered walking toward
   camera, away from the house.
3. **Lip-sync target:** with two people in frame, the wrong person's mouth
   moved to the vocal (6 shots); in another the singer's mouth barely moved.

The ask was an opt-in `assess_output` rule running a VLM (the `judge`
machinery from #119) over sampled frames.

## Decisions

- **Plan v1: defer.** The in-engine VLM probe, anatomy and travel-direction
  checks were parked (v1 Q1/Q2). #487 (reference-leak probe, closed not
  planned and folded in here) added evidence that cheap structural
  heuristics can't catch this fault class: its `opening_jump_ratio` scored
  the known leak at 1.06 against a clean range of 0.37-2.01.
- **Rescope (Don, 2026-10-03):** v1's third trigger fired when #483
  (timestamped transcription) and #485 (`attribute_voices`) closed. The
  lip-sync target, 7 of the report's 9 faults, is reopened alone.
- **Plan v2: build smaller**, one stage (~$3-4). The proposed stage B
  (`vocal_windows` in `analyze_shots`) is cut: voiced spans from the mixed
  track carry no speaker identity, and `attribute_voices` already names who
  sings each line and shot. The loop could already run apart from one bug
  (a null `end` from Whisper made `attribute_voices` refuse the call) and
  the lack of a template and a written loop.
- **Q1:** transcribe **the song** and map it onto cut time with the two
  timeline rules; the final cut's track is the fallback.
- **Q2:** the null-`end` fix goes in the producer, `transcribe_audio`;
  `attribute_voices._span` stays strict.
- **Q3:** one pointer line in `script-to-video`, not in `minimax-h3` or
  `minimax-music3` (both near the 12,288 B skill cap).
- **Non-goals held:** no verdict or auto-regeneration, no ML model inside
  `assess_output`, no change to `judge`, no VLM probe, no new MCP surface,
  no automatic per-shot `singer` wiring.

## What was built

**Stage A, #617** (server + plugin), `develop` @ `ea458b26`:

- **`transcribe_audio`: `end` is never null with `timestamps` set**
  (`_numeric_chunks` in `dw/tasks/audio_transcription.py`). A None `end`
  gets the clip's duration, measured from the 16 kHz mono array Whisper was
  given; a None `start` takes the previous chunk's end, or 0. The result
  shape is unchanged.
- **Template `workflows/templates/attribute-lines.json`**: variables
  `audio`, `voices`, `timestamps` (default `"segment"`), `model_name`; steps
  `transcribe` -> `attribute` (`lines: "previous_result:transcribe"`), both
  saving `application/json`. Shape `utility`, trait `needs-input-media`.
  This closes #485's deferred "`lines: previous_result:transcribe`" item.
- **`attribute_voices`, from bounce 1** (`dw/tasks/voice_attribution.py`):
  - a zero-length line (`start == end`, which Whisper's word mode emits)
    comes back `voice: null`, `uncertain: true` with the too-short reason,
    instead of refusing the call. Voice spans and windows stay strict;
  - a line longer than `piece_seconds` (2 s) is also scored in 2 s pieces;
    if a second voice holds ≥ `mixed_line_share` (0.25) of the confidently
    attributed pieces' voiced time, the line is `uncertain`, and the
    `reason` names each voice's seconds. Both thresholds are in the result's
    `thresholds` and the `docs/TASKS.md` table. This made the plan's
    Risks claim ("a line spanning two singers comes back `uncertain`") true.
- **`docs/TASKS.md`**: *Checking the lip-sync target* under *Voice
  attribution* (the loop, the 32-moment limit, the `music-video` and
  `join_into_song` timeline rules, the fallback, where it misleads), and a
  note in *Speech Transcription* that `end` is always a number.
- **Plugin**: one line in `plugins/dw/skills/script-to-video/SKILL.md`
  pointing a sung multi-shot piece at that subsection.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #617 | 1 | C-F185: word mode emitted zero-length chunks the template's own `attribute` step refused; segment mode gave one hallucinated 0-20.5 s chunk a confident single verdict. Fixed in `attribute_voices` (zero-length lines, mixed-line rule). |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
The plan estimated ~$3-4.

## Verification notes

- C-F185's segment arm: Whisper-base returned one chunk over the whole duet;
  it came back `uncertain` by the mixed-line rule, as shipped. The case's
  "at least one line names each voice" can't be met by a one-chunk
  transcript, so the tester filed an amendment on dkackman/harnest and left
  C-F185 `pending: #617` until it is ruled on.
- C-F188 named the section "Transcription"; the heading is "Speech
  Transcription". Amendment filed on harnest.
- C-F189 passed on the chain: tiles cut to the crop with ~1 s wavs per
  moment; 32 moments accepted, 33 refused before decoding. A consumer agent
  can't listen to the clips, so whose voice is in each was not judged by ear.
- M-F073 (a real two-singer `minimax/music-video` round) was not run: no
  such render exists to reuse. Its `pending: #617` line stays.

## Deferred

- **The in-engine VLM probe, anatomy and travel-direction checks.** Parked
  under v1's Q1/Q2; #484's notes cover prevention for those, and #487's
  evidence argues for a content judge if they come back.
- **Automatic per-shot `singer` wiring** (#485's deferral stands).
- **`plan.downloads_required` still omits the demucs and Whisper weights**
  the template pulls (#485's follow-up in `dw/plan.py`).
- **Whisper hallucination over accompaniment** in segment mode. The guide
  says to re-run with `timestamps: "word"` or transcribe the `vocals` stem
  of `separate_stems`; no template does that for the caller.
- **M-F073**, until a two-singer `music-video` render exists (#598 stage 4's
  hold-vs-reference A/B is expected to use this loop and can supply one).
