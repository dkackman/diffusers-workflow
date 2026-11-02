# Compiling the vocabulary for FLUX 3 Klein

Part of the `cinematography` skill, for the `flux3-klein` skill's captions.
Read that skill's *Prompts* section first.

## The rule this follows

Klein draws a still. A still has no camera move, so a `move` compiles to
framing and composition instead: the frame the move would land on, plus
the visual cue a photograph of that move would show (motion blur for a
fast one). Say so to the user when they ask for a move on Klein: the
picture implies the move; it does not perform it.

Two uses make this worth doing. A storyboard frame for a video shot: draw
the frame the shot opens on (or, for `templates/ltx2/keyframes` or
`templates/minimax/first-and-last-frame`, both ends of the move as two
stills). And a still that should read as cinematic.

Klein was trained on long captions from BFL's upsampler. Through
`templates/flux3/upsample-then-generate`, put the values in the short
request in plain words and the upsampler writes them in; in a caption
written by hand (`templates/flux3/text-to-image`), they go in the prose
like any other detail.

## `shot_size`

| id | Framing in the caption |
| --- | --- |
| `extreme-wide` | "an extreme wide view of <the setting>, <subject> small in the frame" |
| `wide` | "a wide view of <subject> within <the setting>" |
| `full` | "a full-length view of <subject>, head to toe" |
| `medium` | "a waist-up view of <subject>" |
| `medium-close` | "a chest-up portrait of <subject>" |
| `close-up` | "a close-up of <subject's face or the object>, filling the frame" |
| `extreme-close-up` | "an extreme close-up of <the detail>" |

## `move`

| id | The still it becomes |
| --- | --- |
| `static` | "a steady, balanced composition, everything sharp" |
| `dolly-in` | the closer, end frame: one shot size tighter than asked, the subject centred |
| `dolly-out` | the wider, end frame: one shot size looser, more of the setting around the subject |
| `truck-left` | the subject off-centre to the right, open space on the left where the camera is heading |
| `truck-right` | the subject off-centre to the left, open space on the right |
| `crane-up` | "a high-angle view looking down on <subject>" |
| `crane-down` | "an eye-level view of <subject>" |
| `orbit` | "a three-quarter view of <subject>", the angle partway round |
| `tracking` | "<subject> in motion, sharp, the background streaked with motion blur" |
| `whip-pan` | "a horizontal motion blur smearing the frame, <the new view> just resolving" |
| `push-in` | a slightly tighter framing on <subject's face>, the background softly out of focus |
| `handheld` | "a slightly tilted, candid framing" |
| `crash-zoom` | "radial zoom blur rushing toward <subject>" |

## The other axes

In the caption, as visual detail:

- **lens**: "wide-angle perspective with stretched edges", "telephoto
  compression with creamy background blur", "anamorphic widescreen with
  oval bokeh and horizontal flares", "soft vintage-lens glow". Name the
  effect, not a lens product.
- **light**: the `vocabulary.md` meaning in observable words.
- **era**: "a grainy 8mm film frame", "a 16mm film still", "a 35mm film
  still", "a VHS video frame", "a clean digital photograph".
- **tempo**: only through motion blur - `still` and `measured` none,
  `brisk` a little on moving parts, `frantic` heavy.
