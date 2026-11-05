# Long audio: narration, music, beds and the audio pass

Part of the `long-form` skill. The templates' own descriptions are the
detail (`get_workflow(name)`). This file says which one fits which request.

## Narration longer than one take

Audio8-TTS holds about 95 s of reference, text and speech together (see
`templates/voice-clone`). A longer narration is cut into takes at sentence
boundaries and joined.

- **`templates/narration-long`** works from plain text in one call. It runs
  `split_takes` (60 s takes at 2.5 words a second by default), then speaks
  each take under one `reference_audio`, `reference_text` and `seed`, and
  joins them. `gap_seconds` (0.3) of silence follows each take, and
  `crossfade_ms` (30) overlaps each join. Abbreviations and initials are not
  sentence ends.
- **`templates/split-takes`, then `templates/narration-from-takes`**: use
  this pair to see or edit the takes before speaking them. `split-takes`
  returns `{takes, count, words}`. Pass the edited list as `takes`. The list
  is a variable because the engine expands `for_each` before a run starts.
- **Listen at the joins.** The same seed does not make takes sound alike. If
  the voice drifts between takes, lower `max_seconds`. A join is also where a
  level step shows, so read `get_gallery_metadata`'s levels before relying on
  the file.
- **Consent.** Clone only a voice the user has the right to use. There is no
  default voice.
- The 0.3 s gap is a first guess, not a measured pause. Tune it by ear.
  The S6 bench round (over 3 minutes) is *pending measurement*.

## Music longer than one Music 3 track

Music 3 is practical to about 300 s. **`templates/minimax/music-long`**
generates the piece as `sections`, each `{name, caption, lyrics, seconds,
audio_duration}`. Each section is trimmed, faded and crossfaded into the next
(`crossfade_ms` 1500). The joined piece is peak-normalized to -3 dBFS.

- Set `audio_duration` to 1.5 x `seconds`. The ceiling is not a target, and
  the engine does no arithmetic, so write both.
- **Continuity between sections is not guaranteed.** Key, tempo,
  instrumentation and mix can change at a seam. Repeat the same style tags in
  every caption, describe the same band in each, and listen at the joins. A
  crossfade hides a change of sound, not a change of tempo.
- **Joins are not on the beat.** Music 3 takes no audio to continue from,
  and no task trims a section at a beat (`analyze_beats` only reports
  them), so each section is cut at its `seconds`. End each section's lyrics
  on a phrase end and let the crossfade cover the rest.
- The S6 bench round (over 5 minutes) is *pending measurement*.
- The caption format belongs to the `minimax-music3` skill.

## A bed under cut scenes

- **`templates/assemble-and-score`** cuts the shots and lays one score under
  them. It mixes each shot's own sound beneath the score, and `match_levels`
  evens the shots out before the join. `series-episodes` has the full
  procedure.
- **Room tone.** `find_loop_bed` picks the window of a recording worth
  looping, measured as it will sound looped. `loop_audio` then lays it under
  dialogue gaps.

## Voice across chain seams

A guide chain carries the previous segment's audio into the next
(`carry_audio`, on by default), so a line or a held voice runs through the
seam. Use `shot: dialogue` for a speaking subject. `last_frame` continuity
drops the voice carry, which is why the bench ran no `last_frame` arms on
the dialogue walk (ruling R28). Check the lines with
`templates/check-script`.

## Put a track under a picture: `pair_audio`

`pair_audio` muxes a video and an audio track into one file. Use it to lay a
narration or a `music-long` piece under a cut, or to put a soundtrack back
after a frames-only step (an upscale or an interpolation). Pass
`fit: "video"` to cut or pad the track to the picture's length. Without it, a
length mismatch is only a warning. When the video carries recorded shots,
their sample fields are measured again against the new track
(`docs/TASKS.md`, *pair_audio*).

## New sound for a finished picture: the H3 audio pass

An H3 step given `source_video`, with no mask and no `hold_audio`, generates
only the soundtrack (the soundscape, the music and the `<d>` lines) while
attending to the real picture. No video row is re-denoised. The step returns
the source's own frames at the source's own frame rate, under the new track.

- Run it without the turbo LoRA, on the dialogue schedule:
  `num_inference_steps` 12, `video_shift` 12, `audio_shift` 3. The turbo LoRA
  is what makes H3 voices sound digital (#984).
- `listen_scale` (1, 0.5 or 0.25) encodes the source smaller to make each
  step cheaper. The audio length and the returned picture stay the source's.
- Validation refuses refine arguments with it, a chained step, and
  `latents`.
- There is no catalog template yet; the draft waits on Don's listen. Compose
  the step from `get_guide("WORKFLOW_GUIDE", section="Audio over a finished
  picture")`, starting from `templates/minimax/video-with-audio`.

This is the lever for replacing or adding a take's sound without redrawing
its picture: a silent LTX or chained take, or a take whose voice came out
wrong.
