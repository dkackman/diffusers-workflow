# Compiling the vocabulary for Kandinsky 6.0

Part of the `cinematography` skill, for the `kandinsky-6` skill's prompts.
Read that skill's *Prompts* section first; this table only fills the slots
its format already has.

## The rule this follows

Kandinsky Lab's beautifier format is one paragraph: the shot (what is in
the frame and the shot size), then exactly one camera sentence, then
appearance, the five seconds of action in order, and light and style last.
The camera sentence starts with the camera and is never left out: with no
move asked for it is "The camera is static." So:

- `shot_size` goes into the opening shot clause.
- `move` is the one camera sentence. Never write a second sentence about the
  camera, and never fold the move into the action sentences.
- `lens`, `light` and `era` go at the end, with light and style.

A clip is five seconds: one move, once. `<subject>` is the shot's own noun.

## `shot_size`

| id | Opening shot clause |
| --- | --- |
| `extreme-wide` | "An extreme wide shot of <the setting>, with <subject> small in it." |
| `wide` | "A wide shot of <subject> in <the setting>." |
| `full` | "A full shot of <subject>, head to toe." |
| `medium` | "A medium shot of <subject> from the waist up." |
| `medium-close` | "A medium close-up of <subject> from the chest up." |
| `close-up` | "A close-up of <subject's face or the object>." |
| `extreme-close-up` | "An extreme close-up of <the detail>." |

## `move`

| id | The one camera sentence |
| --- | --- |
| `static` | "The camera is static." |
| `dolly-in` | "The camera slowly dollies in toward <subject>." |
| `dolly-out` | "The camera slowly dollies back, widening to show <the setting>." |
| `truck-left` | "The camera trucks left alongside <subject>." |
| `truck-right` | "The camera trucks right alongside <subject>." |
| `crane-up` | "The camera cranes up and looks down over <the setting>." |
| `crane-down` | "The camera cranes down to <subject>'s eye level." |
| `orbit` | "The camera circles slowly around <subject>." |
| `tracking` | "The camera tracks alongside <subject>, keeping pace." |
| `whip-pan` | "The camera whip-pans quickly to <the new view>." |
| `push-in` | "The camera pushes in slightly toward <subject's face>." |
| `handheld` | "The camera is handheld and sways gently with the action." |
| `crash-zoom` | "The camera crash-zooms in on <subject>." |

Example opening: "A medium close-up of an old fisherman on a wet pier,
from the chest up. The camera slowly dollies in toward his face. ..."

## The other axes

At the end of the paragraph, with light and style:

- **lens**: the effect in plain words - "wide-angle perspective", "long-lens
  compression with a soft background", "anamorphic widescreen with
  horizontal flares", "soft vintage glass".
- **light**: "bright, even high-key light", "low-key light from a single
  lamp", "warm golden-hour sun", "lit only by the shop's neon signs", "film
  noir lighting, hard shadows".
- **era**: "grainy 8mm home-movie look", "16mm documentary film", "35mm
  film look", "VHS camcorder look", "clean digital video". The format says
  to state a requested style in prose, and these are styles.
- **tempo**: an adverb on the camera sentence - "slowly", "steadily",
  "quickly" - and the pace of the action sentences. `still` pairs with
  `static`.
