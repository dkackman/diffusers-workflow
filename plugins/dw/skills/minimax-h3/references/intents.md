# H3 intents: start from a result

Part of the `minimax-h3` skill. Each row is one result someone wants, stated
as the claim that was tested, with where its pattern is written and the
record that settled it. Status is the record's verdict on that claim:
`confirmed` (it held), `refuted` (it didn't) or `untested` (no record here;
the issue or source is named). The numbers and prompt words live in the
pattern and the record, never in this table, so follow the pattern cell.
An empty template cell means the pattern is prompt-only and fits any H3
template.

`tests/test_plugin_intents.py` pins the format: five columns, a status from
the three, a confirmed or refuted row citing a record that exists, and a
pattern heading that exists.

| intent | template | pattern | evidence | status |
|---|---|---|---|---|
| Swap a clip's environment or weather and keep its motion, through a video reference | `templates/minimax/composable-references` | `SKILL.md` § Which shape is the request | `bench/records/env-swap-ref2va.md` | confirmed |
| A full-length guide plus a style prompt restyles a clip | | `SKILL.md` § Which shape is the request | `docs/proposals/complete/h3-guides-complete.md` | refuted |
| Hold a set's furniture and clothes across a cut with the earlier shot as a video reference | `templates/minimax/composable-references` | `references/cuts.md` § Visual memory at a cut | `bench/records/visual-memory-cut.md` | confirmed |
| A locked first frame lip-syncs to a held song and keeps its pixels (FL2VA with `hold_audio`) | `templates/minimax/image-to-video` | `SKILL.md` § Which shape is the request | `docs/proposals/audits/2026-10-h3-wangp-probes.md` | confirmed |
| Lip sync to a held track needs the words left out of `<d>` | | `references/dialogue.md` § Probed and not a rule | `docs/proposals/audits/2026-10-h3-wangp-probes.md` | refuted |
| Splitting a short line into two `<d>` blocks removes its lead-in silence | | `references/dialogue.md` § How long a line can be | `docs/proposals/audits/2026-11-h3-dialogue-pacing-probes.md` | confirmed |
| A body action written between two `<d>` blocks paces the line | | `references/dialogue.md` § Probed and not a rule | `docs/proposals/audits/2026-11-h3-dialogue-pacing-probes.md` | refuted |
| A timbre-only audio reference replays its source words | `templates/minimax/voice-timbre-reference` | `references/dialogue.md` § Probed and not a rule | `docs/proposals/audits/2026-10-h3-aiskills-probes.md` | refuted |
| What a `<d>` line talks about is acted out on screen | | `references/dialogue.md` § Probed and not a rule | `docs/proposals/audits/2026-11-longvideos-claims.md` | refuted |
| A reference picture the prompt never names is drawn as another person | | `references/per-character.md` § Probed and not a rule | `docs/proposals/audits/2026-11-longvideos-claims.md` | refuted |
| A face-once reference sheet holds identity better than a portrait | | `references/per-character.md` § Probed and not a rule | `docs/proposals/audits/2026-11-face-once-sheet.md` | refuted |
| Leaving the portrait out of a continued segment stops the person being drawn twice | | `references/chains.md` § Probed and not a rule | `docs/proposals/audits/2026-11-longvideos-claims.md` | refuted |
| A long take built as odd shots, then bridges between them, beats one chain | `templates/minimax/first-and-last-frame` | `references/cuts.md` § Probed, not a rule: odd shots, then bridges | `docs/proposals/complete/h3-odd-then-bridge-complete.md` | refuted |
| Fix one region of a good take, keeping the rest of the picture and the soundtrack | `templates/minimax/regional-edit` | `references/regional-edit.md` § refine_strength | `docs/proposals/complete/h3-regional-edit-complete.md` | confirmed |
| Replace a performer from a portrait and keep the set, the other cuts and the sound | `templates/minimax/replace-performer` | `references/regional-edit.md` § Replace a performer | `bench/records/performer-replacement.md` | confirmed |
| Narrowing `mask_grow` below the template's fixes a replaced performer's seam | `templates/minimax/replace-performer` | `references/regional-edit.md` § Replace a performer | `bench/records/performer-replacement.md` | refuted |
| A long static take holds its look with the keystone slot last (`shot: static_long`) | `templates/minimax/chained-segments` | `long-form/references/levers.md` § The keystone on a long static take | `bench/records/structural.md` | confirmed |
| A narrated monologue holds as one long take | `templates/minimax/narrated-take` | `long-form/references/levers.md` § Narrated take levers | `bench/records/narrated.md` | confirmed |
| A long moving walk-and-talk holds as a single take | `templates/minimax/chained-segments` | `long-form/SKILL.md` § 3. Per chain: continuity and the shot preset | `bench/records/drill.md` | refuted |
| Handedness needs "on screen right" beside the anatomical side | | `references/staging.md` § Probed and not a rule | `docs/proposals/audits/2026-10-h3-aiskills-probes.md` | refuted |
| Dropping the turbo LoRA keeps a dialogue voice from sounding digital | | `SKILL.md` § Hard rules | issue only: #984 | untested |
| A medium named with its physical traits renders truer than the medium named alone | | `references/style.md` § Without a pack: name the medium, then what it looks like | MiniMax's packs' practice, unprobed on dw | untested |
| The per-segment check catches a segment that cuts to another scene partway through | `templates/minimax/chained-segments` | `long-form/references/run-and-repair.md` § The per-segment check | `bench/records/check.md` | confirmed |
| Re-rolling a segment whose break the prompt or the carry holds gives a clean attempt | `templates/minimax/chained-segments` | `long-form/references/quality-and-speed.md` § What does not buy quality on a long take | `bench/records/check.md` | refuted |
| Repair one broken segment of a chain in place, its seams as good as the chain's own | `templates/minimax/repair-segment` | `long-form/references/run-and-repair.md` § Repair one segment in place | `bench/records/repair.md` | confirmed |
| A late capstone frame holds a moving-camera chain | `templates/minimax/chained-segments` | `long-form/references/levers.md` § Structural levers | `bench/records/repair.md` | refuted |
| A cheaper draft tier keeps the standard take's staging at the same seed | `templates/minimax/chained-segments` | `long-form/references/quality-and-speed.md` § The tier dial | `bench/records/tiers.md` | confirmed |
| A short static chain holds its look with the segment-1 anchor (`shot: static`) | `templates/minimax/chained-segments` | `long-form/references/levers.md` § Defaults (built into `shot`) | `bench/records/counters.md` | confirmed |
| Adding guide counters holds a moving-camera chain's seams | `templates/minimax/chained-segments` | `long-form/references/quality-and-speed.md` § What does not buy quality on a long take | `bench/records/counters.md` | refuted |
| Narrate past one take and join the takes word-complete | `templates/narration-long` | `long-form/references/audio.md` § Narration longer than one take | `bench/records/audio.md` | confirmed |
| New sound for a finished picture, new audio over the source's own frames | `templates/minimax/add-sound` | `long-form/references/audio.md` § New sound for a finished picture: the H3 audio pass | `bench/records/audio.md` | confirmed |
| Stochastic sampling makes a turbo chain burn less | | `references/quality-and-speed.md` § What does not help | `bench/records/h3-stochastic-sampling.md` | refuted |
| SelfLift or TST speeds or cleans a chain at no quality cost | | `references/quality-and-speed.md` § Speed levers | `bench/records/speed.md` | refuted |
| A single reference's backdrop and framing bleed into a reference-to-video clip, so crop or blur it first | `templates/minimax/reference-to-video` | `references/per-character.md` § Probed and not a rule | `bench/records/ref-prep-blur.md` | refuted |
| The words check tells the dialogue takes worth keeping from the ones to re-render | `templates/minimax/chained-segments` | `long-form/references/run-and-repair.md` § The per-segment check | `bench/records/words.md` | refuted |
