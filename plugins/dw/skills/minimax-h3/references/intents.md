# H3 intents: start from a result

Part of the `minimax-h3` skill. Each row is one result someone wants, stated
as the claim that was tested, with where its pattern is written. Status is
the verdict of the measured record behind the row: `confirmed` (it held),
`refuted` (it didn't) or `untested` (no measured record). The numbers and
prompt words live in the pattern, never in this table, so follow the pattern
cell.
An empty template cell means the pattern is prompt-only and fits any H3
template.

The record behind each row stays in the dw repo, in
`tests/intents_evidence.json`.

| intent | template | pattern | status |
| --- | --- | --- | --- |
| Swap a clip's environment or weather and keep its motion, through a video reference | `templates/minimax/composable-references` | `SKILL.md` § Which shape is the request | confirmed |
| A full-length guide plus a style prompt restyles a clip |  | `SKILL.md` § Which shape is the request | refuted |
| Hold a set's furniture and clothes across a cut with the earlier shot as a video reference | `templates/minimax/composable-references` | `references/cuts.md` § Visual memory at a cut | confirmed |
| Hold a set across a cut with a still of the set without the cast (a place card) after the people | `templates/minimax/dialogue-short` | `references/cuts.md` § A place card holds the set | confirmed |
| A shot with no people in a carded set: ref2va with the place card as its only reference | `templates/minimax/dialogue-short` | `references/cuts.md` § A place card holds the set | confirmed |
| A locked first frame lip-syncs to a held song and keeps its pixels (FL2VA with `hold_audio`) | `templates/minimax/image-to-video` | `SKILL.md` § Which shape is the request | confirmed |
| Lip sync to a held track needs the words left out of `<d>` |  | `references/dialogue.md` § Probed and not a rule | refuted |
| Splitting a short line into two `<d>` blocks removes its lead-in silence |  | `references/dialogue.md` § How long a line can be | confirmed |
| A body action written between two `<d>` blocks paces the line |  | `references/dialogue.md` § Probed and not a rule | refuted |
| A timbre-only audio reference replays its source words | `templates/minimax/voice-timbre-reference` | `references/dialogue.md` § Probed and not a rule | refuted |
| What a `<d>` line talks about is acted out on screen |  | `references/dialogue.md` § Probed and not a rule | refuted |
| A reference picture the prompt never names is drawn as another person |  | `references/per-character.md` § Probed and not a rule | refuted |
| A face-once reference sheet holds identity better than a portrait |  | `references/per-character.md` § Probed and not a rule | refuted |
| A per-outfit portrait, edited from the base portrait with the face kept, holds a costume change across cuts |  | `references/per-character.md` § A costume change | untested |
| Leaving the portrait out of a continued segment stops the person being drawn twice |  | `references/chains.md` § Probed and not a rule | refuted |
| A long take built as odd shots, then bridges between them, beats one chain | `templates/minimax/first-and-last-frame` | `references/cuts.md` § Probed, not a rule: odd shots, then bridges | refuted |
| Fix one region of a good take, keeping the rest of the picture and the soundtrack | `templates/minimax/regional-edit` | `references/regional-edit.md` § refine_strength | confirmed |
| Replace a performer from a portrait and keep the set, the other cuts and the sound | `templates/minimax/replace-performer` | `references/regional-edit.md` § Replace a performer | confirmed |
| Narrowing `mask_grow` below the template's fixes a replaced performer's seam | `templates/minimax/replace-performer` | `references/regional-edit.md` § Replace a performer | refuted |
| A long static take holds its look with the keystone slot last (`shot: static_long`) | `templates/minimax/chained-segments` | `long-form/references/levers.md` § The keystone on a long static take | confirmed |
| A narrated monologue holds as one long take | `templates/minimax/narrated-take` | `long-form/references/levers.md` § Narrated take levers | confirmed |
| A long moving walk-and-talk holds as a single take | `templates/minimax/chained-segments` | `long-form/SKILL.md` § 3. Per chain: continuity and the shot preset | refuted |
| Handedness needs "on screen right" beside the anatomical side |  | `references/staging.md` § Probed and not a rule | refuted |
| Dropping the turbo LoRA keeps a dialogue voice from sounding digital |  | `SKILL.md` § Hard rules | untested |
| A medium named with its physical traits renders truer than the medium named alone |  | `references/style.md` § Without a pack: name the medium, then what it looks like | untested |
| The per-segment check catches a segment that cuts to another scene partway through | `templates/minimax/chained-segments` | `long-form/references/run-and-repair.md` § The per-segment check | confirmed |
| Re-rolling a segment whose break the prompt or the carry holds gives a clean attempt | `templates/minimax/chained-segments` | `long-form/references/quality-and-speed.md` § What does not buy quality on a long take | refuted |
| Repair one broken segment of a chain in place, its seams as good as the chain's own | `templates/minimax/repair-segment` | `long-form/references/run-and-repair.md` § Repair one segment in place | confirmed |
| A late capstone frame holds a moving-camera chain | `templates/minimax/chained-segments` | `long-form/references/levers.md` § Structural levers | refuted |
| A cheaper draft tier keeps the standard take's staging at the same seed | `templates/minimax/chained-segments` | `long-form/references/quality-and-speed.md` § The tier dial | confirmed |
| A short static chain holds its look with the segment-1 anchor (`shot: static`) | `templates/minimax/chained-segments` | `long-form/references/levers.md` § Defaults (built into `shot`) | confirmed |
| Adding guide counters holds a moving-camera chain's seams | `templates/minimax/chained-segments` | `long-form/references/quality-and-speed.md` § What does not buy quality on a long take | refuted |
| Narrate past one take and join the takes word-complete | `templates/narration-long` | `long-form/references/audio.md` § Narration longer than one take | confirmed |
| New sound for a finished picture, new audio over the source's own frames | `templates/minimax/add-sound` | `long-form/references/audio.md` § New sound for a finished picture: the H3 audio pass | confirmed |
| Stochastic sampling makes a turbo chain burn less |  | `references/quality-and-speed.md` § What does not help | refuted |
| SelfLift or TST speeds or cleans a chain at no quality cost |  | `references/quality-and-speed.md` § Speed levers | refuted |
| A single reference's backdrop and framing bleed into a reference-to-video clip, so crop or blur it first | `templates/minimax/reference-to-video` | `references/per-character.md` § Probed and not a rule | refuted |
| The words check tells the dialogue takes worth keeping from the ones to re-render | `templates/minimax/chained-segments` | `long-form/references/run-and-repair.md` § The per-segment check | refuted |
| A group member described but given no reference is drawn as a stranger or a clone | `templates/minimax/reference-to-video` | `references/staging.md` § Probed and not a rule | refuted |
| An exit written as a shot's last action doesn't happen |  | `references/staging.md` § Probed and not a rule | refuted |
| A silent player in a shot holding a sung slice mouths the vocal unless closed lips are stated | `templates/minimax/music-video` | `references/cuts.md` § Probed, not a rule: a silent player under a held sung slice | refuted |
| "Exactly one person", "the only drummer" and head and prop counts stop clones, beside placement |  | `references/staging.md` § Reported from outside, not probed here | untested |
| A position map repeated in every wide shot, on a flat floor, keeps a group on its marks |  | `references/staging.md` § Reported from outside, not probed here | untested |
| Calm eye wording avoids a bug-eyed face |  | `references/staging.md` § Reported from outside, not probed here | untested |
| A small prop named plainly keeps its size |  | `references/staging.md` § Reported from outside, not probed here | untested |
| A prop action written as separate visible steps keeps the prop's shape and lets it leave the hand |  | `references/staging.md` § Reported from outside, not probed here | untested |
| Fixed furniture kept out of a walking action stays put |  | `references/staging.md` § Reported from outside, not probed here | untested |
| A hand on another person's face renders larger than written |  | `references/staging.md` § Reported from outside, not probed here | untested |
| A crane-up with its framing stated keeps the subject off the bottom of frame |  | `references/staging.md` § Reported from outside, not probed here | untested |
| A final hit written as played through keeps the players going to the note's end |  | `references/staging.md` § Reported from outside, not probed here | untested |
| Opening wide and moving in at once beats a long static wide |  | `references/staging.md` § Reported from outside, not probed here | untested |
| Hand-checking a music video's last lyric lines and held notes catches squeezed final words | `templates/minimax/music-video-cuts` | `references/cuts.md` § Planning a music video's cuts | untested |
| Hold a walker's screen direction across a cut by writing the travel as frame sides in every shot | `templates/minimax/video-with-audio-768p` | `cinematography/references/continuity.md` § Screen direction (MiniMax-H3) | confirmed |
