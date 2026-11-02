# H3 chains

Part of the `minimax-h3` skill: what a chain carries across a seam, and how to
keep it from drifting.

## A guide seam

`templates/minimax/chained-segments` with `continuity: "guide"` holds the
previous segment's last `guide_frames` and its voice as a guide.

- The guide prefix is trimmed off the deliverable, so `trim_frames` and the
  crossfade do not apply.
- `carry_audio` (default true, a chain-block setting) holds the previous
  tail's audio.
- Guide continuity exists only on this template. The reference chains
  (`chain-video-continuity`, `chain-matched-to-audio`,
  `chain-matched-and-aligned`) are ref2va, and guides stay off ref2va.

## Drift

Drift compounds per seam. Reference the subject picture in each segment,
prefer `last_segment`, and use the longest segments memory allows.
