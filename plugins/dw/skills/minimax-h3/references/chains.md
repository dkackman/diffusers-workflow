# H3 chains

Part of the `minimax-h3` skill: what a chain carries across a seam, and how to
keep it from drifting.

## A guide seam

`templates/minimax/chained-segments` defaults to `continuity: "guide"` (best
for every shot type; `last_frame` is the rare case) and holds the previous
segment's last `guide_frames` and its voice as a guide.

- The guide prefix is trimmed off the deliverable, so `trim_frames` and the
  crossfade do not apply.
- `carry_audio` (default true, a chain-block setting) holds the previous
  tail's audio.
- Guide continuity exists only on this template. The reference chains
  (`chain-video-continuity`, `chain-matched-to-audio`,
  `chain-matched-and-aligned`) are ref2va, and guide chains stay off ref2va (ref2va takes only one whole-render guide).

## Drift

Drift compounds per seam. On the reference chains, reference the subject
picture in each segment, prefer `last_segment`, and use the longest segments
memory allows.

Sound drifts too where H3 invents the voice. `assess_output`'s
`analyze_shots` reports each segment's `high_share` and `noise_floor_dbfs`:
the two rising together down a chain's segments is the dulling-voice
signature, and neither alone is. Judge it by ear with `get_output_audio`.

## Probed and not a rule

Tested on our server (Ref2VA turbo, 960x544, two seeds, November 2026; the
record is `docs/proposals/audits/2026-11-longvideos-claims.md`, #1193,
claim 4), and not a rule:

- Refuted: leaving the portrait out of a continued segment, once the
  carried tail shows the person, doesn't stop her being drawn twice. With
  the tail alone and with portrait plus tail, segment 2 grew a look-alike
  follower in the same coat on both seeds, and without the portrait her
  face drifted further from it (similarity 0.18-0.23 against 0.30-0.32).
  Keep the picture in every segment, as above. The follower was already in
  segment 1 of a walk toward a backward-tracking camera: check a walking
  shot's first segment for a second figure before chaining it.

Everything else about drift on a long take is the `long-form` skill's: the
`shot` preset per shot type (`static_long` past 30 s), the per-segment check,
how to measure drift, and the community reports with the bench's verdict on
each (`long-form/references/levers.md`).

Leave `cache` null on a guide chain. The held guide rows count toward the skip test (`first_block`) or its calibration (`mag`), and this is unmeasured.
