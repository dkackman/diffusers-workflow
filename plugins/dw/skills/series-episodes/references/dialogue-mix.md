# A voice-over the score buries

Part of the `series-episodes` skill. A shot whose voice-over the score buries
is not fixed by `match_levels` or `world_gain` (which lifts the whole world
track, action sound included). Do one of these two things instead.

## Duck the score

Use `minimax-h3`'s ducking recipe: `gain_audio` regions on the score, one per
voice-over shot, applied before the score is passed in.

## Give dialogue its own gain

In a saved copy of `assemble-and-score`, add a `separate_stems` step on
`world_faded`. `previous_result:<step>.vocals` is the dialogue, and a
`mix_audio` of its `drums`, `bass` and `other` is the rest (room tone, foley,
action sound). Then have `mixed` layer
`[soundtrack_resampled, dialogue, rest]`, one gain each, and keep `balanced`
after it: a dialogue gain above 1 on a stem peaking near full scale sums past
0 dBFS, which `balanced` pulls back to -3.

htdemucs keeps H3 speech intact this way (#1091). A shot with little foley
leaves a near-silent rest, and a quiet unpitched bed (room tone, ambience)
lands mostly in `vocals`, not the rest.

Separating an already-scored mix works too, but its rest is the score plus
the shots' foley, not a clean score: keep the original score file to re-mix
or replace it.
