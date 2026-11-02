# Compiling the vocabulary for MiniMax H3

Part of the `cinematography` skill, for the `minimax-h3` skill's Context-IR
prompts. Read that skill's *Prompts* section first. If MiniMax's
`h3-prompt-writing` skill is installed, hand it the chosen values in words
and let it write; this table is for writing Context-IR by hand.

## The rule this follows

From MiniMax's base prompt-writing guide (section 4.3, read 2026-11-01):
camera language lives inside `integrated_multimodal_description`, in the
shot it belongs to, written as a natural action ("The camera pushes in with
small amplitude at slow speed toward the folded letter in her hands"), never
stacked as labels at the end of a sentence. A complete move has three
parts: the motion type, an amplitude ("with small amplitude", "with large
amplitude") and a speed ("at slow speed", "at fast speed"); add amplitude
and speed only when they matter. MiniMax's motion types: Push In / Pull
Out, Zoom In / Zoom Out, Pan, Tilt, Truck Left / Right, Pedestal Up / Down,
Arc Shot, Tracking Shot, Static Shot, Shake Slightly / Strongly, POV, Roll.

The opening of `[Shot 1]` states the overall style and the initial
composition, which is where `shot_size`, `lens`, `light` and `era` go. A cut
is "the camera cuts to", and the guide prefers a camera move when only the
distance changes. The stored `minimax/fox_dawn_context_ir` prompt
(`get_prompt`) shows both: "a medium tracking shot moves alongside a red
fox" and "tilts up with small amplitude at slow speed".

## `shot_size`

The guide sets no shot-size words; its own examples use the plain ones.

| id | Composition at the opening of the shot |
| --- | --- |
| `extreme-wide` | "an extreme wide shot frames <the setting>, <subject> small within it" |
| `wide` | "a wide shot frames <subject> in <the setting>" |
| `full` | "a full shot frames <subject> head to toe" |
| `medium` | "a medium shot frames <subject> from the waist up" |
| `medium-close` | "a medium close-up frames <subject> from the chest up" |
| `close-up` | "a close-up frames <subject's face or the object>" |
| `extreme-close-up` | "an extreme close-up frames <the detail>" |

## `move`

| id | The move, in MiniMax's motion types |
| --- | --- |
| `static` | "The camera holds a static shot as <action>." |
| `dolly-in` | "The camera pushes in with large amplitude toward <subject>." |
| `dolly-out` | "The camera pulls out with large amplitude, revealing <the setting>." |
| `truck-left` | "The camera trucks left alongside <subject>." |
| `truck-right` | "The camera trucks right alongside <subject>." |
| `crane-up` | "The camera pedestals up with large amplitude, rising above <subject>." |
| `crane-down` | "The camera pedestals down to <subject>'s eye level." |
| `orbit` | "The camera moves in an arc shot around <subject>." |
| `tracking` | "The camera holds a tracking shot, matching <subject>'s pace." |
| `whip-pan` | "The camera pans right with large amplitude at fast speed, revealing <the new view>." |
| `push-in` | "The camera pushes in with small amplitude at slow speed toward <subject's face>." |
| `handheld` | "The camera shakes slightly as it follows <subject>." |
| `crash-zoom` | "The camera zooms in with large amplitude at fast speed on <subject>." |

## The other axes

In the opening of the shot, with the style:

- **lens**: the effect - "wide-angle", "long-lens compression with a soft
  background", "anamorphic widescreen, horizontal flares", "soft vintage
  glass".
- **light**: the `vocabulary.md` meaning, in the visual description.
- **era**: "shot on grainy 8mm home-movie film", "16mm documentary film",
  "35mm film", "VHS camcorder footage", "clean digital video". For a medium
  (clay, paper) the `minimax-h3` skill's `style.md` reference applies too.
- **tempo**: the speed part of the move - `still` and `measured` "at slow
  speed", `brisk` none (normal speed is omitted), `frantic` "at fast speed"
  plus "shakes strongly" where it fits. In a piece with cuts, tempo is also
  how long each `[Shot n]` holds.

A video reference (`composable-references`) carries camera from a real clip
and is stronger than words: when the move must land, that is the route.
