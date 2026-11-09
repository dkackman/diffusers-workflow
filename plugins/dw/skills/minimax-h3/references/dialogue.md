# MiniMax H3: dialogue, what's in frame, and text on screen

Part of the `minimax-h3` skill; its *Prompts* section says when to read this.

Everything here was measured on our server, not taken from a prompt guide:
T2VA turbo at 960x544, two seeds per arm, October 2026. The record, with job
ids, is `docs/proposals/audits/2026-10-h3-dialogue-probes.md` in the
diffusers-workflow repo; "row N" below is its verdict table. A rule is listed
only where both seeds showed the effect when the rule was broken and neither
did when it was kept.

Lines that cite "pacing row N" come from a second record,
`docs/proposals/audits/2026-11-h3-dialogue-pacing-probes.md` (#1184, November
2026). It ran the skill's non-turbo dialogue settings (no LoRA, 12 steps,
shifts 12/3, int4) at 960x544, 124 frames, one speaker, seeds 1001 and 2002.

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
  lead-in. Size the line to the clip, shorten the clip, or split the line
  (below) when the timing of speech matters.
- **A line at about 2.5–4 written words/s fills the clip**: delivered at
  3.2–4.5 words/s, with phrase pauses of 0.5–2 s nearer the low end. At
  124 frames (5.17 s) that is about 13–21 words; at 345 frames (14.4 s),
  about 36–57.
- **The tail** after the last word ran 0.09–0.88 s, median about 0.37 s.
  Leaving a second of slack is a safety margin, not something H3 needs.
- **Splitting a short line into two `<d>` blocks removes the lead-in**
  (pacing row 1). As one block, `I left the light on for you, Daniel.` opened
  on 2.14 and 2.72 s of silence and was delivered at 4.71 and 5.41 words/s.
  Split at its sentence break, `<d>[English] I left the light on.</d>
  <d>[English] For you, Daniel.</d>`, it started at 0.70 and 0.64 s and was
  delivered at 2.23 and 2.26 words/s. The split doesn't slow the words: the
  first block was still spoken at 6.25 and 6.6 words/s. The pacing is a
  silent gap of about 1.9 s that H3 puts between the blocks. Measured only
  in the non-turbo dialogue settings, at 124 frames, with one speaker.
- **Prose between `<d>` blocks isn't spoken** (0 of 6 clips, pacing row 3),
  so it doesn't count toward the line's length.

Prompts of any length reach the model whole: the H3 text encoder doesn't
truncate (row 7).

## A line per segment in a chain

`templates/minimax/chained-segments` uses `prompt` for every segment, so a
`<d>` line in it is spoken again in each one. For dialogue across a chain:

- **Give each segment its own line.** Pass `prompts`, one per segment, each
  with that segment's `<d>` line and no other. Size each line to the
  segment's clip, as above.
- **Or pass `lines`** when the prompts don't carry the words in `<d>`, or
  when an enhanced prompt's wording may drift from the script. It holds one
  string per segment, exactly as many as there are segments. Unlike
  `prompts`, the last is not repeated. A chain that holds its audio
  (`match_audio`, `hold_audio`) is never words-checked.
- **Cue a carried segment's line past the guide span** (#1308). In a
  `guide` chain every segment after the first opens with `guide_frames`
  (22, about 0.92 s at 24 fps) of the previous segment, and the chain trims
  them off the delivered video. H3 starts a line at frame 0 when the prompt
  gives no time, so the line opens inside the trim and its first words are
  lost ("In order to form a more perfect union." delivered as "It is to form
  a more perfect union."). In the `prompts` entries for segments 2 and on,
  write the line after a timing cue past the guide span, in the form
  the shot prompts use: `From 1.5 seconds into the target video, a calm man
  says: <d>[English] ...</d>`, and leave the first 1.5 s as the speaker's
  closed-mouth beat. Size the line for the shortened remainder of the clip.
  The chain's `line_clipped_at_head` finding (a failed words check) says
  when this is needed. A segment that still opens past its first words
  re-rolls once; for a long monologue use a narrated take. Segment 1 has no
  guide, so its line needs no cue.
- **Check the words.** `check: {"words_similarity_min": 0.8}` with
  `retries: 1` transcribes each segment's kept audio and re-rolls a segment
  that speaks a wrong, missing or garbled line. It reads `lines[k]`, else
  the `<d>` of segment k's prompt. It is off by default, and a segment with
  no line skips it (`words: null`). The threshold, what it misses, and
  what to do when every attempt fails are in the `long-form` skill's
  `references/run-and-repair.md`, from `bench/records/words.md`.

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
    thing (row 10b);
  - what a `<d>` line talks about isn't acted out. A woman in a T-shirt
    saying her brother borrowed her red raincoat drew no raincoat, no
    brother and no extra arms, with or without the sentence "Spoken words
    are only heard: the shot shows what it describes, not what anyone talks
    about" (T2VA turbo, one speaker, static shot;
    `2026-11-longvideos-claims.md` claim 1, #1193). Row 10's rule is
    about the prose, not the line;
  - leaving the words out of `<d>` when a track is held for lip sync
    (WanGP's claim) is not needed: one person in frame, Ref2VA with the
    track held, sung and spoken, moved the mouth at the same instants with
    the words in `<d>` or not (`2026-10-h3-wangp-probes.md`, #985).
  - a timbre-only audio reference didn't replay its source words: with
    `templates/minimax/voice-timbre-reference` as shipped, two Bark source
    clips and two seeds each, `check-script` heard only the new `<d>` line
    and no source word (`2026-10-h3-aiskills-probes.md` row P2, #1059).
    `templates/voice-clone` (Audio8-TTS, #1121) is another source for that
    reference: it speaks a new line in a real speaker's cloned voice, with consent.
- Refuted, though the action landed on one seed of two:
  - a body action written between two `<d>` blocks (`He sniffs and looks
    down at the table.`) doesn't pace a line beyond what the split alone
    gives, and H3 doesn't stage it in the gap. The man sat still through the
    break on both seeds. The look-down came as the second block began, at
    the edge of the break on one seed and after it on the other (pacing
    row 2).
- Not settled:
  - whether `<breath>` adds a beat: it may lengthen a pause H3 already takes
    at a sentence break, but it doesn't create one (row 6b);
  - whether six in-focus extras clone faces: one seed of two did (row 9);
  - whether voice words in a silent scene's sound line ("a low murmur")
    produce speech: H3 didn't render that sound line at all (row 11);
  - starting a chain's continuation from rest: not run (row 8).
