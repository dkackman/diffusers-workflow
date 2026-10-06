# MiniMax H3: a piece with cuts

Part of the `minimax-h3` skill; *Which shape is the request* says when to
read this.

## The cuts templates

`shots` is one list: a dialogue entry is `name`, `prompt`, `references`
and `num_frames`; a music-video entry is `name`, `prompt`, `start_frame`.
The listing's `lists`
block says what an entry carries; its `cost` carries `per_entry` when one
shot was measured - scale from that, else quote the default list's total.
A cut erases drift and each shot makes its audio: score with
`templates/minimax/music` and `templates/assemble-and-score`'s `pair_audio`
mix under the world sound. Keep one voice across shots with a repeated
audio reference, not a repeated description. Duck a score burying
voice-over rather than raising the world track - a `gain_audio` step per
shot, negative `gain_db`.

## Dialogue into a song

Not a concat: sing each shot to a `slice_audio`
slice, the first at `cue_seconds` (song time on the first sung frame; it
enters that long before the cut), each next where the last ended; then
`join_into_song` (dialogue, `song_shots`, unbroken `song`, same
`cue_seconds`), `normalize_audio` to -3 dBFS, `pair_audio`. Recipe:
`workflows` guide, "A spoken scene breaking into a song".

## Planning a music video's cuts

Plan, read, prompt, render. Run `templates/minimax/music-video-cuts` on
the song with its `lyrics` (it chains `transcribe_audio`, `analyze_beats`
and the `plan_cuts` task); read the plan's `shots` (`lyric`, `kind`,
`start_frame`, `num_frames`); write one prompt per shot, a vocal shot to its
`lyric`, an instrumental one to the mood; render with `music-video`, one
`{name, start_frame, prompt}` entry per shot. `music-video` slices every
shot to one shared `num_frames` (default 124), not the plan's per-shot
length: set it to the shortest shot's, or the cuts drift from the plan.
