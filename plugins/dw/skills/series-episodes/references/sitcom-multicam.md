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
2. Run `transcribe_audio` on that audio with `timestamps` set to `"word"`. The
   word times are the only source of cut points; do not guess them from the
   script.
3. Run `plan_cuts` with that `transcript` as its `transcript` argument and
   `segment_by: "line"`. It returns shots that tile the audio exactly, each a
   start and a length, so every line boundary is a cut. Map each planned line
   to its speaker from the script; consecutive lines by one speaker join into
   one single.
4. Generate each single with `templates/minimax/dialogue-short`, sized to its
   planned length.
5. Run `templates/face-drift-check` (task `analyze_face_drift`) on each single
   against the cast reference. A `face_drift` or `no_face` finding replaces
   that single with the master's span for the same times. The check is a flag
   to look at, not a verdict.
6. Assemble the cut list in order, master spans and singles interleaved, with
   `templates/assemble-and-score` (its `concat_videos` joins them). The
   dialogue audio runs under the whole cut unbroken; the singles' own audio is
   not used.

Review the assembled scene with `assess_output` before keeping it, the same as
any episode shot.
