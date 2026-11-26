# Continuity across a cut

Part of the `cinematography` skill. What holds a person's side of frame, their
eyeline, a walker's direction and the light's side from one card or shot of a
scene to the next, when it is written into each prompt. The record is
`bench/records/continuity-grammar.md` (#1704, **PRELIMINARY**: the marker reads
are the lead's, and stand until Don has viewed the outputs). Each rule was
probed with and without its words on two seeds (1685, 1686), and is a rule
here only where both seeds show the effect.

These words go in the shot's existing `beat` and `light` text. There is no
side, eyeline or light-side key or axis value.

## Sides and eyelines (FLUX 3 Klein)

**The 180-degree rule and matched eyelines (R1, R2): confirmed on Klein.** In
every card of the scene, the two-shot and each single, write each person's
side of frame and the way they face:

- Two-shot: "June at frame left, facing right; Marcus at frame right, facing left."
- June's single: "June at frame left, facing right; Marcus, off screen, at frame right, facing left."
- Marcus's single: "Marcus at frame right, facing left; June, off screen, at frame left, facing right."

With these words the singles kept both sides and looked toward the partner on
both seeds. With names only, the two-shots were right (the person named first
went left), but the singles crossed the line or looked away on both seeds.

Two things came with it, on both seeds:

- Naming the off-screen partner's side turned each single into an
  **over-the-shoulder**, with the partner's shoulder or head at the frame edge.
  That is usable coverage, but not a clean single.
- **Text holds the partner's side, not their identity.** In Marcus's single,
  the shoulder at left was a stranger, not June. A card that shows the partner
  needs the partner's reference (`storyboard/references/coverage-set.md`). A
  clean single without the partner was not probed.

The side is the person's side of the frame and the direction they face, never a
side of their body: "on screen right" for an anatomical side was refuted on H3
(P5, #1059). H3's "face-on at medium close-up" staging rule is unchanged by
this; these singles were Klein stills.

## Screen direction (MiniMax-H3)

**Screen direction (R3): confirmed on H3.** Write the travel as frame sides,
the same in every shot: "walks from frame left to frame right". With it the
walker crossed left to right in both shots (a static wide, then a static
medium) on both seeds, so the direction held across the cut. With "walks along
the street" instead, H3 walked toward the camera in all four shots: no
direction is set, so there is none for a cut to hold. The walker faced three
quarters toward the camera while crossing, not a pure profile.

## Families not probed

**LTX-2.5 and Kandinsky 6: untested** for every rule here. Each rule is also
untested on the family it wasn't probed on: R1 and R2 on H3 and
MiniMax-H3-Image, and R3 on the stills families. Write the words if you like,
but check the frames and don't promise the effect.

## Probed and not a rule

- **R5, the light side held within a scene (Klein): unsettled, so not a rule.**
  "Key light from frame left, warm practical" put the visible lamp at frame
  left in the medium and the close-up on both seeds. But on seed 1686 the
  control already had it there, so the effect showed on one seed of two. The
  face's key stayed a soft gradient at this light. Use the words as a hint
  that places the practical, not as a rule that holds the key. Don's look at
  the close-ups is pending (#1704, Q1).

## Cut without a probe

Rules from the same source (#1685) that were not probed, and why:

- **R4, cut on action:** an editing choice, not a prompt rule. A tail
  reference is an option at a join (`script-to-video/references/joins.md`).
- **R6, "and then" means split:** already stated, as one state per shot
  (`minimax-h3/references/prompt-lint.md`).
- **R7, a hold at the start and end of a shot:** measured against. H3's
  opening silence is a defect we split away (#1184), and a "static" shot
  pushed in (#1581). Our models fight a held frame.
- **R8, no music in shot prompts:** mostly stated in `minimax-music3`; the
  templates that break it are #1706.
- **R9, a voice-over input holds only the words:** measured for H3, in
  `minimax-h3/references/dialogue.md`.
- **R10, design for animation:** a taste rule, and nobody asked for it.
- **R11, one style line, verbatim:** already held in substance by a shared
  style phrase.
- **R12, an expression sheet as an edit of the turnaround:** a generated
  reference sheet gave no gain
  (`docs/proposals/audits/2026-11-face-once-sheet.md`), and Klein's edit fails
  at a reverse card (#1581).
- **R13, "no text" in sheet prompts:** measured against. "No lettering" got
  pseudo-lettering on both seeds on H3 (`minimax-h3/references/dialogue.md`),
  and Klein takes no negative prompt.
