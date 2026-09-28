# `join_into_song`: a spoken scene breaking into a song (#486)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-09-27). Designed on #486 from a field report (*La Vela del Pueblo*: 6
dialogue shots breaking into 18 lip-synced song shots, assembled locally in
~130 lines of ffmpeg). Two design sessions each posted a plan v1; Don chose
the 23:54 one ("B"), approved as **build smaller** with its defaults to Q1-Q4.
Built as stages #513 (A) and #514 (B).

## The problem

A spoken scene - shots that carry their own dialogue audio - has to break
into a song whose shots were lip-synced against frame-exact `slice_audio`
slices of one track:
- each dialogue shot loudness-matched;
- the song entering under the last spoken line, so that song time
  `cue_seconds` lands exactly on the first song shot's frame 0;
- the dialogue ducked after a delay;
- the song shots playing over the **unbroken** song, their own audio dropped.

`assemble-and-score` lays one score under everything, and composing the join
from existing tasks was possible but not viable: `mix_audio` has no per-track
offset, `gain_audio` has no ramp, and the song offset (`dialogue length -
cue`) would be a hand-computed literal that goes stale the moment a dialogue
shot is regenerated. The second half of the report was provenance: the film
built outside dw was not in the gallery, not reachable by `export_job`, and
had no record of its inputs.

## Verdict and what was decided

**Build smaller**: one bespoke task that owns the timing arithmetic.
- Q1: the name is `join_into_song`.
- Q2: a bespoke task rather than generic primitives (`mix_audio.offsets` +
  a `gain_audio` ramp + a silence task), because only a task can measure the
  dialogue length at run time.
- Q3: the task does not normalize its output; `normalize_audio` downstream
  decides the level, as in `music-video`.
- Q4: the sweep findings are filed separately (see *Deferred*).
- Cut: a "register external output" path. A task's output already gets the
  manifest, the realized `workflow.json` (which records every input
  reference) and `export_job`, so doing the join in dw closes the provenance
  half too.
- Cut: a template. The inputs are finished shots from earlier runs; the join
  is one inline step.

## What was built

### Stage A (#513): the task

`join_into_song(dialogue, song_shots, song, cue_seconds=0,
dialogue_target_lufs=None, duck_delay_ms=0, duck_db=-12, duck_ramp_ms=250,
fps=None)` in `dw/tasks/join_into_song.py`, registered in `dw/tasks/task.py`.
- **Picture:** the dialogue frames, then the song-shot frames. A mismatched
  frame size or fps is refused, naming each input and its rate.
- **Placement:** each dialogue shot's track is fitted to its own frame span
  (trimmed or padded; a fit of a frame or more warns
  `dialogue_fitted_to_frames`), so `D` is the sample the first song-shot
  frame sits at. The song goes in at `D - round(cue_seconds * sr)` and runs
  to the end of the picture. A cue longer than the dialogue is a directed
  error naming both. A song too short is padded with silence and warns
  `song_short`.
- **Loudness:** `dialogue_target_lufs` applies one static gain per shot,
  measured then applied (the report's ffmpeg `loudnorm` trap avoided). A
  shot under 400 ms or unmeasurable keeps its level and warns
  `dialogue_unmatched`. A silent dialogue shot becomes silence of its own
  length, so nothing after it shifts.
- **Duck:** from `song_entry + duck_delay_ms`, a linear ramp of
  `duck_ramp_ms` down to `duck_db`.
- **Channels:** the output takes the song's channel count; mono dialogue is
  spread, wider dialogue cut down.
- **Shots:** one record per input, named `shot@<key>` from `dialogue` then
  `song_shots` references (`dw.shots.shot_references`, which `dw/workflow.py`
  now uses instead of reading `videos` alone), samples measured off the built
  waveform, `hard_cut` on every shot after the first. `EXPECTED_SITES` entry
  "populates".
- **Domains** (`dw/task_domains.py`): `cue_seconds`, `duck_delay_ms`,
  `duck_ramp_ms` non-negative, `duck_db` non-positive, `fps` positive. Empty
  lists are refused at run time.
- **Tests:** `tests/test_join_into_song.py` on synthetic clips. Docs: a
  `docs/TASKS.md` section.

Deviations from the plan, all recorded on #513:
1. **An optional `fps`.** Pipeline frames carry no rate, and the song is
   placed in time, so a join of pipeline results needs one. It is refused
   when it contradicts a rate the videos carry.
2. **`D` measured per shot, fitted to frames**, not off the joined dialogue
   track: otherwise a dialogue track overrunning its frames would push the
   song off the song shots' picture.
3. **Shot naming** generalized through `shot_references`.
4. **Channel handling**, which the plan left open.

The bounce fix reached the engine: `Result.save` (`dw/result.py`,
`save_audio_video`) now writes a *declared* `result.fps` back onto the
in-memory artifact. Before, a fresh run handed the next step the pre-save
rate while a cache hit or `output:` reload read the written one, so
`concat_videos`/`dissolve_videos` could silently re-time a re-rated shot.
They now refuse the mismatch, as this task does.

### Stage B (#514): the recipe

- `plugins/dw/skills/minimax-h3/SKILL.md`: a "Which shape" bullet,
  **Dialogue into a song** - `slice_audio` (first slice at `cue_seconds`,
  each next where the last ended) -> `join_into_song` (same `cue_seconds`)
  -> `normalize_audio` at -3 dBFS -> `pair_audio`.
- `docs/WORKFLOW_GUIDE.md`: `### A spoken scene breaking into a song`, with
  the slice arithmetic (`start_seconds = cue_seconds + earlier sung shots'
  num_frames / fps`) and a JSON tail that `tests/test_plugin_skills.py`
  validates.
- Deviation: the full recipe lives in the guide, the skill carries it in
  brief. The H3 skill is at its 12 KB limit (12280/12288 bytes); room was made
  by moving and rewording, with no rule dropped.

## Measured on the fixture

Recipe followed verbatim (C-F158, job `115b1c8941fd`) with the C-F152
fixtures and a -60 dB marker hole at song time 3.0 s, `cue_seconds: 3.0`:
- the hole lands on the seam (frames [249,259) at -79.2 dBFS);
- the song is audible under the last line (15 dB above the dialogue's own
  level there);
- 496 frames, 4 shots at [0,124) [124,248) [248,372) [372,496);
- the film peaks at -3.07 dBFS (true peak -3.05) with no `audio_clipped` or
  `audio_no_headroom`;
- `analyze_sync_drift`: no findings, max offset -0.66 ms.

Stage A's C-F153 measured a -12.03 dB duck and a monotone ramp; C-F154
landed two shots 11 LU apart at -22.98 and -22.99 LUFS.

## Deferred, and why

- **The plan sweep's two latent bugs** (Q4): `concat_videos` skips a silent
  input's audio rather than filling it with silence, so every later shot's
  audio lands early; and `pair_audio` loads a string `audio` but not a string
  `video`. Filed as one bug for the implementer at close-out (#553). The third
  finding (`docs/TASKS.md` missing `normalize_audio`'s `target_lufs`) was
  fixed by #474.
- **Naming a re-rated input in the fps refusal.** The error calls a
  `previous_result:` video `video 3` where the shot map calls it after its
  step (tester's nit on #513). Not filed.
- **Song into dialogue, more than one song, a dissolve at the seam, a
  limiter**: non-goals. A limiter now exists (`normalize_audio(limit=true)`,
  #474) and is the downstream step's to use.
- **A template and register-external-output**: cut (above). A second musical
  field report asking for the shape, or a second deliverable assembled
  outside dw, would bring either back.

## Cost and bounces

The stage comments record no usage figures, so no cost is given here. The
plan estimated about $5 for stage A and $2 for stage B.

| stage | bounces | why |
|---|---|---|
| #513 | 1 | C-F156 arm 5: a 12 fps `previous_result:` shot was joined as 24 fps. The cause was the engine: `Result.save` didn't write a declared `result.fps` back onto the artifact. Fixed there; an `fps` contradicting the carried rate is refused too. |
| #514 | 0 | |

Cases: C-F151-C-F157 (stage A), C-F158 (stage B). C-F151 has a suite
amendment pending on harnest for the added `fps` parameter.
