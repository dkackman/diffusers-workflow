# Compiling the vocabulary for MiniMax-H3-Image

Part of the `cinematography` skill, for the `minimax-h3-image` skill's
prompts. Read that skill's *Prompts* section first.

## The rule this follows

H3-Image draws a still, so as on Klein a `move` compiles to framing: the
frame the move would land on, plus the cue a photograph of it would show.
The picture implies the move but doesn't perform it. Say so to the user.

The main use is a keyframe for an H3 clip. Draw the frame the shot opens on
for `templates/minimax/image-to-video`, or both ends of the move as two
stills for `templates/minimax/first-and-last-frame`. Then compile the same
`move` again with `references/compile-h3.md` for the clip's own prompt,
which says what the camera does next.

The prompt is used verbatim, with no enhancer, so the words go into the
hand-written description in its order: the task, then the layout (where
`shot_size` and the framing a `move` becomes go), then the details and
style (where `lens`, `light` and `era` go). In a multi-reference edit, say
the framing after what each `<Picture N>` keeps.

## `shot_size`

| id | Framing in the description |
| --- | --- |
| `extreme-wide` | "an extreme wide view of <the setting>, <subject> small in the frame" |
| `wide` | "a wide view of <subject> within <the setting>, the whole figure and its surroundings" |
| `full` | "a full-length view of <subject>, head to toe, with a little space above and below" |
| `medium` | "a waist-up view of <subject>" |
| `medium-close` | "a chest-up view of <subject>, the face clearly visible" |
| `close-up` | "a close-up of <subject's face or the object>, filling the frame" |
| `extreme-close-up` | "an extreme close-up of <the detail>, its texture sharp" |

## `move`

| id | The still it becomes |
| --- | --- |
| `static` | "a steady, balanced composition at eye level, everything sharp" |
| `dolly-in` | the closer end frame: one shot size tighter than asked, the subject centred |
| `dolly-out` | the wider end frame: one shot size looser, more of the setting around the subject |
| `truck-left` | the subject off-centre to the right, open space on the left where the camera is heading |
| `truck-right` | the subject off-centre to the left, open space on the right |
| `crane-up` | "a high-angle view looking down on <subject>" |
| `crane-down` | "an eye-level view of <subject>", the camera at the subject's height |
| `orbit` | "a three-quarter view of <subject>", the angle partway round |
| `tracking` | "<subject> in motion, sharp, the background streaked with motion blur" |
| `whip-pan` | "a horizontal motion blur smearing the frame, <the new view> just resolving" |
| `push-in` | a slightly tighter framing on <subject's face>, the background softly out of focus |
| `handheld` | "a slightly tilted, candid framing, as if from a hand-held camera" |
| `crash-zoom` | "radial zoom blur rushing toward <subject>" |

For an H3 opening keyframe, prefer the move's start frame: a `dolly-in`
opens on the looser framing and the clip moves in. Draw the end frame above
only for the last image of `first-and-last-frame`, or when the still is
itself the deliverable.

## The other axes

In the description's details and style, as visual detail:

- **lens**: "wide-angle perspective with stretched edges", "telephoto
  compression with a creamy background blur", "anamorphic widescreen with
  oval bokeh and horizontal flares", "soft vintage-lens glow". Name the
  effect, not a lens product.
- **light**: `high-key` "bright, even high-key light with almost no
  shadows", `low-key` "a single hard light source, most of the frame in
  deep shadow", `golden-hour` "low warm golden-hour sun, long shadows, a
  glowing rim light on <subject>", `practical` "lit only by the lamps,
  screens and neon signs in the scene", `noir` "film noir lighting, hard
  contrast, venetian-blind shadows across <subject>". In an edit that
  keeps a reference's light, leave `light` unset and say "keep the light
  of <Picture N>".
- **era**: "a grainy 8mm film frame", "a 16mm film still", "a 35mm film
  still", "a VHS video frame", "a clean digital photograph". For a
  keyframe, use the same era words the H3 clip's prompt will use.
- **tempo**: only through motion blur. `still` and `measured` get none,
  `brisk` a little on moving parts, and `frantic` heavy.

None of this is measured on H3-Image. Judge the still with
`get_output_image` against the framing asked for.
