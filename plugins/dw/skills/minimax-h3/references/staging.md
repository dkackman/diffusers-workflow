# MiniMax H3: staging crowds, two people, and reference shots

Part of the `minimax-h3` skill; its *Hard rules* say when to read this. Each
rule came from a render that went wrong on our server (#484).

- Fit a crowd's action to what it holds: candle-holders asked to clap
  rendered three-handed people.
- A directed crowd move needs its landmark in frame: "marches up to the
  house" put it behind camera. Track from the side, landmark ahead.
- Two people in frame can lip-sync the wrong one, seemingly by whichever
  voice enters first: credit a shout to the crowd, spell the sung line in
  `<d>` tags, frame the singer face-on at medium close-up.
- State a crowd's age range; unstated skews young.
- A three-reference shot can open on the portraits' backdrop
  for ~1.5s (about 1 in 25): check a shot's first frames and say in
  `subject_definitions` that the portrait's backdrop, pose and framing are
  not reused.
- Place every subject in a group shot by position ("Howie at frame left,
  Pat at the right end of the desk"), or H3 invents an extra person
  (`references/quality-and-speed.md` § Prompting the base model). The
  outside-reported head-count and position-map wording below goes beside
  that placement, never in place of it: a head-count sentence alone failed
  three times.

## Probed and not a rule

Tested on our server (T2VA turbo, 960x544, two seeds per arm, October 2026,
#1059), and not a rule:

- Refuted: adding "on screen right" to an anatomical side isn't needed for
  handedness. A woman facing the camera asked to raise a mug "in her left
  hand" used her left hand, on screen right, on both seeds, as did the arm
  that added "on screen right" (row P5). That is one easy case: one person
  facing the camera, one prop, a static shot. A person in profile, two
  people and a moving camera weren't tested.

Tested on our server (Ref2VA turbo, 960x544, 124 frames, two seeds per arm,
November 2026, #1561), claims from the outside band-video run below:

- Refuted (P1): a group member with no reference isn't drawn as a stranger
  or a clone. Three band members placed left, centre and right, with the
  third's reference dropped and him described only (look and clothes in
  `subject_definitions`, "newly generated from the description above" in
  `retention_analysis`): both seeds drew exactly three people, the two
  referenced ones matching their pictures and the third a new man drawn from
  the description. So a described member can stay unreferenced. All three
  were placed by position; an unplaced member wasn't tested.
- Refuted, with a timing note (P3): an exit written as the shot's last
  action does happen, but it takes the rest of the shot. He turned at about
  frame 96 and was still at the right edge on the last frame, on both seeds,
  so the empty room was never seen. Written into the first half, the exit
  was done by about frame 62 and the room stayed empty to the end. When a
  shot must end on an empty frame, put the exit in its first half.
- Not settled (P2), waiting on Don's playback: whether wet or rain-soaked
  words near a face put streaks on the skin. Native-size face crops showed a
  mild sheen and a few droplets, no streak, on both seeds; a running streak
  is motion, which stills can't rule out. Until he rules, neither use nor
  avoid the wording on its account.

A silent player under a held sung slice is in `references/cuts.md`
§ Probed, not a rule: a silent player under a held sung slice.

## Reported from outside, not probed here

Source: an outside agent skill written after two unattended 30-scene H3 band
music videos (vrgamegirl19/comfyui-vrgamedevgirl, its band-music-video
skill, read at 12adc26 for #1528). It doesn't say whether those ran on base
H3 or a finetune. None of these is probed here: each is that run's report,
restated, and one of our own renders can promote or remove a line.

- Clones: they report a one-member shot needs "exactly one person" in frame,
  a role named as the only one ("the only drummer"), and a wide shot the
  number of heads and of large props (one drum kit). Use it beside placement.
- Position: they report a fixed position map for the group, repeated in
  every wide shot, kept members on their marks, and a flat floor held where
  a raised stage kept putting the group beside the platform.
- Eyes: they report wide, flashing eye words gave a bug-eyed face, and calm
  eye words didn't.
- Small props: they report describing a small prop in detail (a microphone's
  grille) made it render oversized. Name it plainly, the same way each shot.
- Prop actions: they report writing a prop's use as separate visible steps
  (press, lift away, the mark left behind), or it changed shape or stayed in
  the hand.
- Furniture: they report a walk and a drawer pull in one action made the
  drawer travel with the walker. Keep fixed furniture out of a moving action.
- Touch: they report a hand on another person's face rendered larger than
  written.
- Crane: they report a crane-up without its framing stated left the subject
  at the bottom of the frame.
- Last note: they report a final hit must be written as played through, or
  the players stop before the note ends.
- Wides: they report a long static wide failed every time; open wide and
  move in at once.
