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
on the song with its `lyrics` (it chains `separate_stems`, `transcribe_audio` of the
vocal stem, `analyze_beats` of the full mix and the `plan_cuts` task, on H3's grid with a `lead_s` run-up, default 0.5 s);
read the plan's `shots` (`lyric`, `kind`, `start_frame`, `num_frames`,
`lead_frames`, `cut_frames`) and quote its `render_frames` for the cost: it is
what renders, more than the song's length. When Whisper can't time the words (a non-English verse, a hallucinated
transcript, wordless vocals), replace the `transcribe` step with an
`analyze_phrases` step on `previous_result:stems.vocals` (sung phrase timing
from voiced energy, no Whisper), keep `plan_cuts`' `transcript` on that step
and pass `lyrics`, one line per sung phrase; the default chain is unchanged.
`plan_cuts` recognises the result by its `phrases` key. Without `lyrics` each
phrase is a line with `lyric: null` (kind vocal), and `segment_by` `line`,
`stanza` (a silence of at least `min_gap_seconds` between phrases) and `beat`
all work. With as many lyric lines as phrases they map one to one, in order.
With a different count it warns with both counts and spreads the lines evenly
from the first phrase's start to the last's end, never pairing silently: raise
`analyze_phrases`' `min_gap_seconds` when there are more phrases than lines,
lower it when fewer. A legato or reverberant verse (a choir) can fuse its lines
into one phrase no `min_gap_seconds` splits: `floor_db` is the knob there
(less negative, e.g. `-30`), but listen to the phrases before trusting a
one-to-one pairing - a raised floor can match the counts on wrong boundaries,
and `plan_cuts` warns when a paired phrase is under 1/4 or over 4x the median
length. Suspect filtering and word alignment don't apply; a long
phrase split into pieces gives every piece its whole lyric. The song length
falls back to the phrase result's `duration_seconds`.

Write one prompt per shot, a vocal
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

Mouths and the pasted-over song (#795): `pair_audio` lays the original song over shots whose mouths follow H3's own audio. With the slice as a reference alone, H3 was seen re-singing it shifted by whole beats (up to 1.8 s), so the mouth lands off the song. The `hold_audio` variable holds the slice as well, keeping the reference: it defaults to `previous_result:slice` on `music-video` (null for the reference alone) and to null on the `match_audio` chains, where an audio file sets it. Measured on lem (#808, 4 shots, two seeds), the reference alone put shots up to 1.37 s late (0.18 s early on another), while reference plus hold measured 0.00 s on all eight, correlation 0.95 to 1.00. That is the audio against its slice, not the mouth: #619 measured hold with image-only references at 3 of 6 sung onsets, against 6 of 6 reference-only, and mouth onsets with hold plus reference were not counted. Check a cut with `measure_sync`.

## Probed, not a rule: odd shots, then bridges

Tested on our server (lem, draft tier: 1344x768, 124 frames, 5 steps, int4,
sparse, seed 12345; October 2026; one take, so one sample; #1094, record on
the issue). The claim (from an outside repo): for a long continuous take,
render the odd shots independently from the same references, then fill each
even shot with a `templates/minimax/first-and-last-frame` bridge pinned to
its neighbours (`get_last_frame` of the shot before as `image`,
`get_first_frame` of the shot after as `last_image`), and the take holds its
look better than a chain. A take needs an odd shot count for this: the last
shot of an even count has one neighbour.

Five shots, one astronaut at a marker post, no dialogue. Arm A was a
`last_segment`-style Ref2VA chain; arm B rendered shots 3 and 5 as separate
Ref2VA jobs and shots 2 and 4 as 124-frame bridges. Both shared shot 1 and
one reference image. Both were joined with `concat_videos` `trim_frames: 1`
and measured with `analyze_seams` given `shots=` (concat's `hard_cut` record
would skip `seam_frame_jump`).

- **Refuted on this take, so not a rule: it doesn't win, on identity.** Arm
  B's independent shot 3 rendered two astronauts for its whole length, and
  both bridges interpolated faithfully into and out of the duplicate. Arm A
  kept one subject (its prop doubled and its framing wandered).
- Everything else went to arm B:
  - take drift, shot 1 head vs shot 5 tail (4 frames each): lightness
    -14.09 (A) vs +0.16 (B), saturation +5.09 vs +1.05, contrast -9.86 vs
    -7.51;
  - worst seam: `frame_delta` 19.57 / `jump_ratio` 7.29 (A) vs 1.85 / 0.81
    (B); every arm B seam was under the clip's own motion (`jump_ratio`
    below 1), every arm A seam a visible jump (3.08-7.29);
  - no bridge tail snap: the largest per-frame step over a bridge's last 12
    frames was 0.92 and 1.03, against a body typical of 2.33 and 6.72. So
    WanGP's "discard ~5 tail frames" was not reproduced for a bridge (#985).
- Audio: each bridge makes its own, a different soundscape from the Ref2VA
  shots around it: level steps of 3.60-6.72 dB at all four arm B seams
  (0.35-2.39 dB in the chain), and bridge 2 near-silent at -46.1 dBFS.
  Never for dialogue; an ambient take would need `match_levels` or one bed
  under the whole take.
- So an independent odd shot is a fresh render with no carry: one wrong
  subject count breaks two bridges with it. A cut still erases drift
  (above); a take that must stay continuous stays a chain.
- Deviations: the reference was a shared HF URL, not an `asset:` upload;
  arm A was hand-built as 48-frame carries of the segment before (what
  `last_segment` injects), not one `chain` job; drift used 4 frames a side;
  shots 3 and 5 ran one after the other on one card (host RAM).
- Jobs: shot 1 `71a3c4ee9eba`; arm A `e1d37983f04a`; arm B shot 3
  `d4be72601012`, shot 5 `1ae6a04cb793`, bridge 2 `932c2add5238`, bridge 4
  `305f70b310b9`; joins and measurements `8a8adfbf3c30`.

Drift above is lightness, saturation and contrast. For texture (grain, blotching,
smoothing) over a static region, `compare_outputs(texture_bands=true)` reports
energy by band; see the workflows guide, *Comparing outputs*, *Texture bands*.
