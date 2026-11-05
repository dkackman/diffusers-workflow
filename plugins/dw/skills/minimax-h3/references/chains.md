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
  `chain-matched-and-aligned`) are ref2va, and guides stay off ref2va.

## Drift

Drift compounds per seam. On the reference chains, reference the subject
picture in each segment, prefer `last_segment`, and use the longest segments
memory allows.

Everything else about drift on a long take is the `long-form` skill's: the
`shot` preset per shot type, the per-segment check, the 30 s static cut rule,
how to measure drift, and the community reports with the bench's verdict on
each (`long-form/references/levers.md`).
