# Sitcom multicam: a master, per-speaker singles, cuts on the words

Dialogue scenes cut the way a multicam sitcom does: one wide camera that never
leaves, and a close camera per speaker you cut to when that person talks. The
cut-on-word-boundaries idea is One-shot Multicam's (MIT licence); the pieces
below are this repo's.

## The grammar

- **Master**: one wide take of the whole scene, held start to finish. Its shot
  fields are `shot_size: wide`. It carries the room, the blocking and the full
  dialogue audio, and is the fallback for anything below.
- **Singles**: one `templates/minimax/dialogue-short` clip per speaker line or
  run of lines, `shot_size: medium-close` or `close-up`, the speaker facing
  camera. Cast the speaker from the same reference as the master so the face
  matches.
- **Cuts**: on line boundaries, never mid-word. Cut to a speaker's single as
  they start, back to the master for a reaction or when the line ends.
- **Fallback**: a single whose face drifts from the cast reference is dropped
  and the master covers that span.

Field names and values for each shot come from
`cinematography/references/vocabulary.md`; persist the scene as a shot document
per `script-to-video/references/shot-document.md` (master is shot 1, singles
follow) rather than keeping the list in your head.

## Procedure

1. Render the master, then the dialogue audio it carries (the scene's speech).
2. Run `transcribe_audio` on that audio with `timestamps` set to `"segment"`.
   Without `lyrics`, `plan_cuts` takes each transcript chunk as one line, so
   `"word"` would make one shot per word; a segment is a spoken line. Cut
   times come from the chunk starts, which is what keeps a cut off the middle
   of a word. Do not guess them from the script.
3. Run `plan_cuts` with that `transcript`, `segment_by: "line"`, `fps` of the
   family (H3: `24`), `include_instrumental_gaps: false` (a pause stays with
   the line before it; no music shot in a dialogue scene), `duration_s` the
   audio's length, and the family's frame grid so each single lands on it
   (H3: `modulus: 17`, `remainder: 5`, `min_frames: 124`, `max_frames: 345`;
   past 345 frames split the line). Read each planned shot's `start_frame`
   and `cut_frames` (where the cut sits on the scene timeline and how long it
   holds), `num_frames` (what the single renders; more than `cut_frames`
   when rounded up to the grid, so `trim_video` it to `cut_frames`) and
   `lyric` (the line's text). The shots tile the audio, so every line
   boundary is a cut.
4. Map each planned line to its speaker. Run `templates/attribute-lines` (its
   `attribute_voices` step scores each transcript line against each voice by
   timbre) on the dialogue audio with `voices`: two or more names, each mapped
   to reference spans into that audio, taken from lines you know only that
   person speaks (read off the script and the transcript's times), or to a
   separate clip of the voice. Match its lines to the shots by time. A line
   it marks uncertain, or that disagrees with the script, is resolved from
   the script; the script is the check, not the source. Consecutive shots by
   one speaker join into one single: the first's `start_frame`, the summed
   `cut_frames`.
5. Generate each single with `templates/minimax/dialogue-short`, sized to its
   planned length.
6. Run `templates/face-drift-check` (task `analyze_face_drift`) on each single
   against the cast reference. A `face_drift` or `no_face` finding replaces
   that single with the master's span for the same times. The check is a flag
   to look at, not a verdict.
7. Assemble the cut list in order, master spans and singles interleaved, with
   `templates/assemble-and-score` (its `concat_videos` joins them). The
   dialogue audio runs under the whole cut unbroken; the singles' own audio is
   not used.

Review the assembled scene with `assess_output` before keeping it, the same as
any episode shot.
