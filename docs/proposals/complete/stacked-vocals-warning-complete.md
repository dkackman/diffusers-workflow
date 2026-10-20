# A `stacked_vocals` warning from `separate_stems` (#947)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-19). Plan v1 was approved by Don on 2026-10-20 (UTC). Stage A,
#955, shipped at `develop` @ `29158446` and was verified on lem.

## The idea

A field report from `two-minds` (*Always On*): MiniMax Music 3 can't sing
two distinct same-gender voices, and its choruses are stacked stereo
doubles. `separate_stems` returns them inside the one `vocals` stem.
Processing the 1:19 chorus as one voice is what distorted it. The issue
asked for two things:

1. a voice/pitch conversion task (WORLD / `pyworld`) to re-voice a window
   of a vocal stem;
2. a flag from `separate_stems` / `attribute_voices` saying a window holds
   more than one voice.

## Verdict and scope (Don, plan v1)

**Build smaller:** ask (2) only, as a warning on `separate_stems`.

- **Q1, warning only.** No `layering` result key; the four-stem dict is
  unchanged (a plain `warnings` key would have broken the wav save and three
  test pins).
- **Q2, voice conversion deferred.** The agent already re-voiced the stem
  off-server with `pyworld` and brought it back, so the gap cost one local
  script. It would add a C-extension core dependency, a new task surface and
  a skill line the music3 SKILL.md has no room for.
- **Q3, `attribute_voices` untouched.** Its separation path collapses to
  mono 16 kHz before a stereo check could run.
- **Cut:** the issue's "more than one pitch track" signal. Multi-pitch on a
  sung stem is unreliable, and two singers in one register read as one
  track anyway.

## What was built (stage A, #955)

- `stacked_spans()` and the `STACK_*` constants in
  `dw/tasks/voice_attribution.py`, run in `separate_stems` on the 44.1 kHz
  stereo vocals stem. Pure numpy. In 1 s windows that are voiced (at least
  half their 20 ms frames pass `voiced_mask` on the mid signal) and within
  14 dB of the stem's level, a window is flagged when the uncentered L/R
  correlation is below 0.45 **and** the side/mid energy ratio is above 0.4.
  Flagged windows merge across a one-window gap; spans under 2 s drop.
- One `emit_warning(kind="stacked_vocals", command="separate_stems")`:
  `the vocals stem holds stacked (layered stereo) voices over 1:19–1:34,
  1:59–2:09; treat those spans as more than one voice before processing them
  as one`. Up to 8 spans in m:ss, then "and N more".
- `docs/TASKS.md` *Stem separation* names the warning and what to do with
  it.
- Unit tests in `tests/test_separate_stems.py`: a known decorrelated span,
  identical L/R, silence, mono, a short run dropped, a merge across a gap,
  truncation, the m:ss text, and two runs of the real `separate_stems` path
  with only demucs faked.

### Calibration (the plan's gate: passed)

On the *Always On* seed-7 vocals stem (`AlwaysOnStems` job `494d1a7469b7`):

| section | windows | L/R correlation | side/mid |
|---|---|---|---|
| verse 1 (0:16–0:49) | 25 | 0.656 – 0.969 | 0.016 – 0.208 |
| verse 2 (0:49–1:19) | 24 | 0.663 – 0.964 | 0.018 – 0.208 |
| chorus 1 (1:19–1:34) | 15 | −0.276 – 0.251 | 0.609 – 1.682 |
| final chorus (1:59–2:09) | 10 | 0.060 – 0.206 | 0.667 – 0.891 |

### Deviations from the plan

1. **A level gate, `STACK_LEVEL_BELOW_DB = 14`.** htdemucs leaves the
   intro's "electrical hum" pad in the vocals stem (0:08–0:12), and it
   measures as decorrelated as the chorus. It sits 16.5 dB or more below the
   stem's level, so the gate drops it. The margin is thin (2.5 dB on that
   side, 4 dB to the quietest chorus window): retune it first if another
   song disagrees.
2. **"Voiced window" means at least half its frames voiced**
   (`STACK_VOICED_SHARE = 0.5`); the plan left the share open.
3. **The bridge (about 1:42–1:47) doesn't flag**: the level gate drops it.

## Verification

Cases C-F506 (the seed-7 known answer, plus the chained
`transcribe_audio`), C-F507 (mono, an instrumental re-separation and a
1.5 s slice don't warn; a 20 s slice warns in its own time) and C-F508 (the
guide names the warning; `separate_stems` keeps its three arguments) all
passed on lem against `29158446`. The fixture is the shared asset
`qa-cast/always-on-seed7.mp3`.

The verify also found a stale example in the same TASKS.md section
(`file_name` for `file_base_name`), fixed separately in #958.

## Cost and bounces

| stage | bounces | cost |
|---|---|---|
| A (#955) | 0 | not recorded on the stage |

## Deferred, and what brings it back

- **The voice/pitch conversion task (ask 1).** Comes back on a second field
  report needing a re-voice, from a piece other than *Always On*. The plan's
  sketch: `shift_voice(audio, semitones, formant_shift, start_seconds,
  duration_seconds, crossfade_seconds)`, WORLD analysis/resynthesis over a
  window, crossfaded back, summed with the existing `mix_audio`; a
  `pyworld` dependency after a wheel check on linux-cuda and macOS-arm64; it
  would refuse a window this check flags as stacked. A learned
  voice-conversion model stays out either way.
- **Structured spans (Q1).** A `layering` `JsonRecord` key saved as
  `<name>-layering.json` (the `crop_face_track` precedent), if a caller
  needs the spans machine-readable.
- **The check in `attribute_voices` (Q3).** Needs the stereo stem kept
  through `separate_vocals`. Worth it only if attribution on a stacked
  chorus misleads a session.
- **A pitch-track signal.** Only if the stereo check proves blind to a real
  stacked span.
