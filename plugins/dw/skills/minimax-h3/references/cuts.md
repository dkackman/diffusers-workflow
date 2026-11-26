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
mix under the world sound. Every shot laid under a separate score writes
`non_diegetic_music: N/A`, or its own music restarts at each cut and fights the
score; when the shot's prompt comes from `enhance_prompt`, say "No music." in its
brief and the enhancer writes N/A. Keep one voice across shots with a repeated
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

Before writing prompts, hand-check the plan's last lyric lines and any held
note against the song: the outside band-video run in
`references/staging.md` § Reported from outside, not probed here reports
that the aligner squeezed long final words, timed on the vocal stem as this
chain already does. Not probed here.

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

## Probed, not a rule: a silent player under a held sung slice

A player who doesn't sing, alone in a shot holding a sung slice of the song,
didn't mouth the vocal, so a shot like that needs no closed-lips sentence
and no vocal-free slice. Refuted on our server (#1561, the record is
`docs/proposals/audits/2026-11-vrgdg-band-probes.md` P4): a drummer-only
shot, Ref2VA turbo 960x544 on `music-video`'s path (the song sliced, then
held with `hold_audio`), two seeds. With no lips wording his mouth was
closed at all six frames read inside the sung phrase, on both seeds.
Stating "His lips stay closed the whole shot; he does not sing." added
nothing, and on one seed the lips parted on a late frame anyway. Read from
stills of one slice of one song with one player face-on; motion wasn't
watched, and a singer and a silent player in one shot is the wrong-mouth
rule in `references/staging.md`.

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
  under the whole take (`assemble-and-score`'s `bed_start_seconds`,
  `bed_seconds` and `bed_gain`, from a `find_loop_bed` pick).
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

## Visual memory at a cut

After a hard cut, H3 sees only the new prompt and its references, so the
room, props and wardrobe of the shots before can change. Try a place card
first (*A place card holds the set*, below), then the window this section
describes. An earlier shot
passed as a `MiniMaxH3VideoReference` can carry them across. That makes the
shot ref2va, which refuses guides, anchors and latent carry, so the
narrated-take shape (fl2va with guides) can't take one. This works for the
ref2va cut shapes: `dialogue-short`, and script-to-video or series-episodes
shots rendered through it.

**Probed (lem, turbo, int4, sparse, seed 1398; November 2026; one scene, one
seed, so one sample; #1398, record `bench/records/visual-memory-cut.md`;
markers as Don viewed them, 2026-11-17).**
E was an establishing wide and A a medium on the character. Each B was the
reverse angle, with a prompt naming the lamp, shelf, window and coat but not
their colours.

| Arm | Reference besides the portrait | Lamp | Shelf | Window | Coat | Framing |
|---|---|---|---|---|---|---|
| B0 (1344x768) | none | lost (white) | dim | lost (orange sky) | lost (purple) | the reverse |
| B2 (1344x768) | 22 frames from E's middle (51-72) | held (red) | held | held (blue night) | held (green) | the reverse, no replay |
| B0s (960x544) | none | lost | dim | lost | lost | the reverse |
| B1s (960x544) | A's whole 124-frame clip | held | held | held | **lost** (purple dress) | pulled back toward A's angle, not a clear reverse |

- **A short window held all four; the whole clip held three and pulled the
  framing back to A.** The whole clip also scored the lowest identity across
  the cut (SFace 0.584, against 0.695 for its control).
- **The whole clip lost the wardrobe colour in both arms that used one.** B1s
  and S1s (the speech arm, also A′'s whole clip at 960x544) each put the
  character in a purple dress, not the green coat; the 22-frame window (B2)
  kept the coat. That is the case for the window over the whole clip.
- **Cost, per denoise pass against the same size's control:** a 22-frame
  window 1.35x (wall 1.36x); the whole clip 4.4x (wall 2.3x to 3.2x). The
  engine prices a video reference by its length (`video_reference_exponent`,
  #1429: about 4x and 1.4x on these two), the figures `SKILL.md` quotes.
- **The whole clip does not fit a 24 GB card at 1344x768** with this
  offload: B1 ran out of CUDA memory (validate now flags it, #1429). It ran
  at 960x544.
- **A window cut from the frames just before the cut replays them.** S2,
  given A′'s last 22 frames, opened on those frames for about 1.4 s, then cut
  inside the clip to its own reverse (all four markers held). A window from
  the middle of a shot (B2) did not replay. Trim a replayed opening off before
  the join.

**The recipe: a short window, not the whole clip.**
- Take the window from the shot that shows the place best, usually the
  establishing wide, from its middle. Not from the frames next to the cut,
  unless you trim the replay.
- **22 frames.** dw refuses a video reference under 13 frames ("it must run
  at least 13 frames at 24 fps"), and `validate_workflow` refuses it too, at
  the reference's path. 22 is the next length on H3's `17n + 5` grid. Start
  on a multiple of 17.
- Drop the window's soundtrack (`audio: null`), so the reference carries the
  picture and not the earlier line.
- It costs one of the 3 video slots (12 files in all).

In one workflow, a `trim_video` step feeds the reference by
`from_previous_result`:

```json
{
    "name": "window",
    "task": {
        "command": "trim_video",
        "arguments": {
            "video": "output:<establishing shot file>",
            "start_frame": 51,
            "num_frames": 22
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24, "subfolder": "intermediate" }
}
```

and the shot's `references` list gains, beside its portraits and voices:

```json
{
    "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3VideoReference",
    "from_previous_result": "window",
    "audio": null
}
```

For a template that takes references as caller data (`dialogue-short`'s
per-shot `references`), run the `trim_video` step as its own one-step run
first, then pass its file as `"from_file": "output:<window file>"` in that
entry, with the same `reference_type` and `audio: null`.

**Speech at a cut: never pass the whole earlier clip.** When either side of
the cut speaks, pass only a short window (the frames before the cut, its
replayed opening trimmed), or no video reference at all. This is the source's
claim (KadoBOT/ComfyUI-H3-Continuity-Studio): a whole clip with speech in it
makes the new shot copy its mouth motion. The probe did not show a copy
(S1s, the whole of A′ at 960x544, spoke its own line on its own timing, as
viewed), but it was one sample, and the whole clip also cost 4.4x a
pass and pulled the angle back, so the rule holds on those grounds too.

## A place card holds the set

A *place card* is one still per recurring set: a wide of the set empty of
the cast, with no text. People who belong to the set (an audience, a crowd,
a jury) stay in it; only the cast is left out. Make it on FLUX 3 Klein (`templates/flux3/text-to-image`),
naming in its prompt the set's markers (the lamp, the window and its light,
the shelf), and keep it beside the cast portraits:
`keep_output(..., asset_name="places/<set>", shared=True)`. Pass it as a
`MiniMaxH3ImageReference` **after** the people in a ref2va shot's
`references`, so the cast's `<Picture N>` numbers don't move, and name it in
the prompt as the room of `<Picture K>`. It is cheap (a still, about 30 s on
lem) and needs no earlier clip, so it is the first thing to try at a cut; the
window of *Visual memory at a cut*, above, comes next.

**Probed, PRELIMINARY (lem, turbo, int4, sparse, 1344x768, 124 frames, seed
1398; November 2026; one seed per arm; #1581, record
`bench/records/place-card.md`; marker reads are the lead's until Don has
viewed the clips).** The same scene as #1398: B0 and B2 are its arms, above.
The card holds the lamp, shelf and window; the coat is not in it.

| Arm | References | Lamp | Shelf | Window | Coat | SFace across A's cut | Framing |
|---|---|---|---|---|---|---|---|
| B0 | portrait | lost | held | lost | lost | 0.842 | the reverse |
| B2 | portrait, 22-frame window of E | held | held | held | held | 0.695 | the reverse |
| R1 | portrait, card | held | held | held | **lost** | 0.673 | a real reverse, not the card's framing |
| R2 | portrait, card (a close-up) | held | held | held | lost | 0.804 | stayed a close-up |
| R4 | two portraits, card (175 frames) | held | held | held | lost | 0.378, unusable (two faces) | the two side by side to camera |
| N1 | card only (no people) | held | held | held | - | - | the empty set; a slight push-in |
| N0 | none (t2va, N1's prompt) | lost | held | lost | - | - | the empty set |

- **The card holds what it shows, and nothing else.** R1 held all three of
  its markers on a reverse the card does not frame; the coat, which only the
  window's frames show, stayed lost. Hold wardrobe or a prop with the window,
  or by putting it in the portrait.
- **Its face cost is about the window's.** R1 sits 0.022 below B2 across the
  cut, within one seed's spread; both sit well below no reference at all.
- **A close-up carries it.** R2 was not pulled wide: the lamp at the frame
  edge and the window behind her read the set. Its face score held (0.804),
  but the close-up's focus is soft, so look at the face.
- **Two people and a card fit a 24 GB card at 175 frames** (it ran; validate
  projects 23.53 GB, under the 3090's 23.6). The two stood side by side to camera rather than across
  the desk, and a prop from one portrait (an accordion) showed in the first
  frames and went. Stage the blocking in the prompt, and keep props out of the
  portraits you don't want in the shot.
- **A shot with no people can take the card alone.** N1 (ref2va, the card as
  its only reference; in `dialogue-short`, an entry whose `references` lists
  only the card) held all three markers; N0, the same prompt on t2va,
  lost the lamp and the window. N1 pushed in slightly despite "Static camera",
  and its audio was near-silent (-71.9 dBFS; the server warns), so lay its
  sound on in the mix.
- **A new look is a new card.** A night version from the same Klein seed and
  prompt, its light changed, kept the layout. Make one card per look, and
  keep each under its own name (`places/<set>-night`).
- **Don't ask Klein's edit for a reverse card.** `templates/flux3/image-edit`
  asked to turn the card around returned the card's own view. It wasn't
  needed: one front card carried the reverse (R1).

**Which to try at a cut:** the card first. Use the 22-frame window when the
card loses what the shot must hold (wardrobe, a prop, a light state the card
lacks), and when an earlier clip that shows the place exists. A card and a
window together were not probed. The card takes one of the 9 image slots
(12 files in all), the window one of the 3 video slots.

A `dialogue-short` shot with one person, the card and her voice:

```json
{
    "name": "reverse",
    "num_frames": 124,
    "references": [
        {"reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference",
         "from_file": "asset:cast/<name>.jpg"},
        {"reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference",
         "from_file": "asset:places/<set>.jpg"},
        {"reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3AudioReference",
         "from_file": "asset:cast/<name>-voice.wav"}
    ],
    "prompt": "Reverse angle. <Picture 1> sits at the desk in the room of <Picture 2>, facing the window ..."
}
```
