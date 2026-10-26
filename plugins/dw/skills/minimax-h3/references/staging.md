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
