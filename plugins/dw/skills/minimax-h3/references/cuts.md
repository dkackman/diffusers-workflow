# MiniMax H3: a piece with cuts

Part of the `minimax-h3` skill; *Which shape is the request* says when to
read this.

## The cuts templates

`shots` is one list: a dialogue entry is `name`, `prompt`, `references`
and `num_frames`; a music-video entry is `name`, `prompt`, `start_frame`, `num_frames`,
`lead_frames` and `cut_frames`, all required.
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

Plan, read, prompt, render. Run `templates/minimax/music-video-cuts`
on the song with its `lyrics` (it chains `transcribe_audio`, `analyze_beats`
and the `plan_cuts` task, on H3's grid with a `lead_s` run-up, default 0.5 s);
read the plan's `shots` (`lyric`, `kind`, `start_frame`, `num_frames`,
`lead_frames`, `cut_frames`) and quote its `render_frames` for the cost: it is
what renders, more than the song's length. Write one prompt per shot, a vocal
shot to its `lyric`, an instrumental one to the mood. Render with
`music-video`: each entry is the plan's shot with `prompt` added and `lyric`
and `kind` dropped (unread, they only warn). Each shot renders `num_frames`
from the song sliced at `start_frame - lead_frames`, then `trim_video` keeps
`cut_frames` after the lead, so the video is the plan's length and the song
fits it. `music-video` does not snap `num_frames`: it must be on `17n + 5`
from 124 to 345, and an off-grid entry (130) is refused. The render rule
sizes it: an entry's `num_frames` is the smallest `17n + 5` at or above
`lead_frames + cut_frames`, and at least 124, so
`num_frames = max(124, next 17n+5 ≥ lead + cut)`. A 48-frame cut with a
12-frame lead renders 124; a 130-frame span (lead + cut) renders 141. A span
needing more than 345 must be split into two shots (`plan_cuts` splits it,
preferring a beat, and warns). The slack past the cut is a tail handle that
`trim_video` drops. A hand-written entry with no run-up has `lead_frames` 0
and `cut_frames` equal to `num_frames`, or shorter with `num_frames` sized by
the same rule.
`music-video`'s `song` variable defaults to the song it writes; pass
`asset:...` to cut to an existing one and skip writing.
