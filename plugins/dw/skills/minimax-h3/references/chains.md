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

### Community reports on drift

These are reports from other users (roycho87/NerdNotes,
`20261102/h3_latents_and_degradation.md`, collecting Banodoco and
r/StableDiffusion threads), paraphrased. They are not dw measurements.

- **Static and half-static shots are reported to drift worst.** On a moving
  shot the background keeps being replaced. On a locked-off shot the drift is
  mostly darkening and colour rather than texture.
- **A hard cut or a new angle is reported to reset drift.** People describe
  about 30 s of clean chain with good setups, with drift usually showing around
  segments 3-6. Where the story allows, cut or change angle roughly that often;
  `references/cuts.md` has the cut itself.
- **Speed LoRAs are reported to add crunch over a long chain.** See the speed
  tradeoff in `references/quality-and-speed.md`; the 4-step default is
  unchanged.
- **Measuring drift:** compare a patch the subject never enters against
  segment 1 at the same frames. Whole-frame averages let changes cancel out.
  `compare_outputs(texture_bands=true)` does the texture side (see
  `references/cuts.md`), and `assess_output(probe="analyze_seams")` reports a
  `jump_ratio` per seam.
- **Colour matching** is reported to fix colour but not texture, and matching
  to a frame with different content makes it worse. Match on a masked area or
  a similar frame. #1033 tracks an engine version of this; it is not promised.
