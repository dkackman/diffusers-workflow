# Compiling the vocabulary for LTX-2.5

Part of the `cinematography` skill, for the `ltx-2.5` skill's captions. Read
that skill's *Prompts* section and its `caption-spec.md` reference first;
this table only fills the slot the spec already has.

## The rule this follows

LTX-2.5's trained caption spec requires, for every shot and woven into the
prose, never as tags:

- a shot type, exactly one of extreme wide shot / wide shot / medium shot /
  medium close-up / close-up / extreme close-up;
- a camera motion, always stated: if none, say the camera remains static;
- a camera viewpoint relative to the subject: front-facing, back-facing,
  side view, over-the-shoulder, top-down, low-angle or high-angle.

So one camera sentence carries all three. The vocabulary has no viewpoint
axis: pick the one the request implies, else front-facing, and always write
it. Add one compile rule from the LTX-2 camera-control LoRA cards, which
advise naming what should come into view during the move: for every move
but `static`, say what comes into view as the camera goes (a reveal gives
the model somewhere to move to). The caption is present progressive, one
paragraph, observable words only.

`<subject>` and `<reveal>` are the shot's own nouns; `<viewpoint>` is one of
the spec's seven (for example "a low angle" or "a front-facing angle").

## `shot_size`

| id | Shot type to write |
| --- | --- |
| `extreme-wide` | "an extreme wide shot takes in <the setting>, <subject> small within it" |
| `wide` | "a wide shot frames <subject> and <the setting around them>" |
| `full` | "a wide shot frames <subject> head to toe" (the spec has no full shot; this is the nearest of its six) |
| `medium` | "a medium shot frames <subject> from the waist up" |
| `medium-close` | "a medium close-up frames <subject> from the chest up" |
| `close-up` | "a close-up fills the frame with <subject's face or the object>" |
| `extreme-close-up` | "an extreme close-up isolates <the detail>" |

## `move`

| id | Camera motion to write (every row carries the viewpoint, as the spec requires) |
| --- | --- |
| `static` | "captured from <viewpoint>, the camera remains static" |
| `dolly-in` | "seen from <viewpoint>, as the camera dollies in toward <subject>, <reveal> comes into view" |
| `dolly-out` | "seen from <viewpoint>, as the camera dollies back from <subject>, <reveal> comes into view around them" |
| `truck-left` | "seen from <viewpoint>, as the camera trucks left alongside <subject>, <reveal> slides into view from the left" |
| `truck-right` | "seen from <viewpoint>, as the camera trucks right alongside <subject>, <reveal> slides into view from the right" |
| `crane-up` | "seen from <viewpoint>, as the camera cranes upward above <subject>, <reveal> comes into view below" |
| `crane-down` | "seen from <viewpoint>, as the camera cranes down to <subject>'s level, <reveal> comes into view behind them" |
| `orbit` | "seen from <viewpoint>, as the camera arcs around <subject>, <reveal> comes into view behind them" |
| `tracking` | "seen from <viewpoint>, the camera tracks alongside <subject> at their pace, <reveal> passing through the frame" |
| `whip-pan` | "seen from <viewpoint>, the camera whip-pans <left or right> in a blur, landing on <reveal>" |
| `push-in` | "seen from <viewpoint>, the camera pushes in slowly toward <subject's face>, until <reveal> fills the frame" |
| `handheld` | "seen from <viewpoint>, the handheld camera sways slightly as it follows <subject>, <reveal> drifting into view" |
| `crash-zoom` | "seen from <viewpoint>, the camera crash-zooms in on <subject>, until <reveal> fills the frame" |

Example: "A medium shot frames a woman in a grey wool coat from the waist up,
captured from a low angle as the camera dollies in toward her, the lit
window behind her coming into view over her shoulder."

## The other axes

These go into the visual detail, where the spec asks for lighting, colour
and texture, and in the spec's aesthetic-quality descriptors:

- **lens**: say what it does to the picture - "a wide-angle view that
  stretches the edges", "compressed depth and a shallow focus", "a
  widescreen frame with oval bokeh and horizontal flares", "soft, glowing
  vintage glass". Say the effect, not a lens name.
- **light**: one phrase per value, in observable words - `high-key` "bright,
  even light with almost no shadows", `low-key` "a single hard lamp leaves
  most of the room in deep shadow", `golden-hour` "low, warm sunlight casts
  long shadows and a glowing rim around <subject>", `practical` "lit only by
  the lamps, screens and signs in the scene", `noir` "hard-contrast night
  light, venetian-blind shadows striping the wall, a haze of smoke".
- **era**: "grainy 8mm home-movie footage", "16mm documentary film grain",
  "35mm film with fine grain", "soft VHS camcorder footage", "clean digital
  video".
- **tempo**: speed words on the move - `still` "barely moving", `measured`
  "slowly", `brisk` "steadily", `frantic` "quickly and jerkily". The clip
  length paces the shot; tempo does not shorten it.

Multi-shot captions repeat the triple at each cut, as the `ltx-2.5` skill's
*Several shots* paragraph says.
