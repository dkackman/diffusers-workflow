# MiniMax H3: dialogue, what's in frame, and text on screen

Part of the `minimax-h3` skill; its *Prompts* section says when to read this.

Everything here was measured on our server, not taken from a prompt guide:
T2VA turbo at 960x544, two seeds per arm, October 2026. The record, with job
ids, is `docs/proposals/audits/2026-10-h3-dialogue-probes.md` in the
diffusers-workflow repo; "row N" below is its verdict table. A rule is listed
only where both seeds showed the effect when the rule was broken and neither
did when it was kept.

## The rule

- **Don't name a thing the shot shouldn't show.** Naming something as off
  screen puts it on screen: a dog written as asleep "out of frame behind the
  camera" was rendered asleep in the visible room or hallway on both seeds,
  and saying where it was didn't keep it out. If a thing isn't in the shot,
  leave it out of the prompt (row 10).

## How long a line can be

Measured, not a limit the model enforces (row 1):

- **No word was lost** at any rate tried: 1.55 to 4.06 written words per
  second of clip at 124 frames, and 2.2 and 4.0 at 345 frames. The last word
  always finished. Above about 3 words/s H3 speeds the delivery up (4.1–4.5
  words/s) rather than dropping words.
- **A short line is not stretched to fill the clip.** H3 holds silence first
  (2–3.4 s of lead-in for 8–10 words in 5.17 s), then speaks fast, at 5–6
  words/s. In one of those clips the speaker's mouth moved during the silent
  lead-in. Size the line to the clip, or shorten the clip, when the timing
  of speech matters.
- **A line at about 2.5–4 written words/s fills the clip**: delivered at
  3.2–4.5 words/s, with phrase pauses of 0.5–2 s nearer the low end. At
  124 frames (5.17 s) that is about 13–21 words; at 345 frames (14.4 s),
  about 36–57.
- **The tail** after the last word ran 0.09–0.88 s, median about 0.37 s.
  Leaving a second of slack is a safety margin, not something H3 needs.

Prompts of any length reach the model whole: the H3 text encoder doesn't
truncate (row 7).

## Text on screen

Text quoted verbatim in the prompt, as the built-in enhancer writes it,
rendered legible on both seeds: a painted sign reading "MARLOWE'S BAKERY".
On one seed, text the prompt didn't ask for (a window, a stray "P2") came
out garbled. And a surface asked to be blank ("plain cream paint with no
lettering") got invented pseudo-lettering on both seeds, so a shot can't
count on "no lettering" (row 12).

## Probed and not a rule

These were tested and either didn't hold or weren't settled, so none is a
rule and none is worth prompt words yet:

- Refuted on both seeds, so not rules:
  - two men and a woman in one scene each kept their own voice (row 2);
  - "soft, gentle whisper, like a lullaby" on a voice already described as
    a deep male one kept it male (row 3), though a male line with no voice
    description wasn't tested;
  - a silent listener in shot kept her mouth closed without a "lips stay
    pressed together" sentence (row 4), and so did a man holding a phone
    whose caller spoke (row 5);
  - `<pause>`, `<softer>`, `<breath>` and a bare `…` inside `<d>` were
    neither spoken nor turned into invented words (rows 6, 6b);
  - a negated noun ("There is no dog in the room") didn't render the
    thing (row 10b).
- Not settled:
  - whether `<breath>` adds a beat: it may lengthen a pause H3 already takes
    at a sentence break, but it doesn't create one (row 6b);
  - whether six in-focus extras clone faces: one seed of two did (row 9);
  - whether voice words in a silent scene's sound line ("a low murmur")
    produce speech: H3 didn't render that sound line at all (row 11);
  - starting a chain's continuation from rest: not run (row 8).
