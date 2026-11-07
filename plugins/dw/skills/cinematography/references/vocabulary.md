# The cinematography vocabulary

Part of the `cinematography` skill; its *The axes* section says when to read
this. Six axes, each a closed set of ids. A shot picks at most one value per
axis and leaves the rest unset; an unset axis is the model's own choice, not
a default. The ids are what a shot list records and what later skills cite,
so they are spelled exactly as here. The meaning column is the plain-language
idea; the per-family `compile-*.md` tables turn it into the family's own
caption words.

## `shot_size`

How much of the subject the frame holds.

| id | Means |
| --- | --- |
| `extreme-wide` | The place dominates; figures are small in it. An establishing shot. |
| `wide` | The whole setting around the subject, the subject clearly readable. |
| `full` | One figure head to toe, a little room above and below. |
| `medium` | Waist up. |
| `medium-close` | Chest up. The usual dialogue framing. |
| `close-up` | The face, or one object, fills most of the frame. |
| `extreme-close-up` | A detail: the eyes, a hand, a keyhole. |

## `move`

What the camera does during the shot. One move per shot.

| id | class | Means | Note |
| --- | --- | --- | --- |
| `static` | `static` | The camera does not move. | |
| `dolly-in` | `moving` | The camera travels toward the subject, a full move. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `dolly-out` | `moving` | The camera travels away, revealing more of the setting. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `truck-left` | `moving` | The camera slides sideways to the left, parallel to the subject. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `truck-right` | `moving` | The camera slides sideways to the right. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `crane-up` | `moving` | The camera rises, often to look down over the scene. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `crane-down` | `moving` | The camera descends, often settling to the subject's level. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `orbit` | `moving` | The camera arcs around the subject, keeping it centred. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `tracking` | `moving` | The camera moves with a moving subject, keeping pace. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `whip-pan` | `moving` | A very fast turn sideways that blurs to a new view. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `push-in` | `moving` | A slow, small move toward the subject, for emphasis. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `handheld` | `moving` | Unsteady, operator-held movement. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |
| `crash-zoom` | `moving` | A very fast zoom in on one thing. | H3: never `last_frame` with `look_hold`/`carry_pick` (measured on the arc only). |

`dolly-in` and `push-in` differ by size and speed: a dolly travels, a push-in
creeps. `crash-zoom` is a zoom, not a travel: the view narrows while the
camera stays put.

**The `class` column** is a definition, not a measurement:

- `static`: the camera's position and orientation don't change across the
  take. Only the `static` id is in this class.
- `moving`: every other id, including `push-in` and `handheld`, whose motion
  is small but real. `crash-zoom` is here too: its view changes across the
  take even though the camera stays put.

The two errors don't cost the same, so a borderline move goes to `moving`. A
moving shot treated as static has its pose pinned, which froze the camera on
the bench's arc; a static-ish shot treated as moving only loses length (the
`long-form` skill cuts it at about 30 s). The `long-form` skill maps this
column to the H3 `shot` preset. The Note column's H3 rule comes from
`long-form`'s *Two rules that bite*: it was measured on the bench's moving
arc (`bench/records/counters.md`, *Eye notes (S2)*) only; the other moves
inherit it by class, not by a measurement of their own.

## `lens`

The look of the glass, not a product.

| id | Means |
| --- | --- |
| `wide` | Wide-angle: deep space, stretched edges, close objects loom. |
| `normal` | Natural perspective, roughly what the eye sees. |
| `long` | Telephoto: compressed depth, a shallow focus that isolates the subject. |
| `anamorphic` | Widescreen, oval bokeh, horizontal streak flares. |
| `vintage` | Soft edges, lower contrast, warm glow and gentle flare. |

## `light`

| id | Means |
| --- | --- |
| `high-key` | Bright, even, few shadows. |
| `low-key` | Mostly dark, one hard source, deep shadows. |
| `golden-hour` | Low, warm sun, long shadows, glowing rim light. |
| `practical` | Light from sources inside the scene: lamps, screens, candles, neon. |
| `noir` | Hard contrast, slatted or venetian-blind shadows, smoke, night. |

## `era`

The capture medium the picture should feel shot on. Era is said in the
prompt; a grade (`references/grades.md` in this skill) adds colour and grain
afterwards, and the two pair well but are separate choices.

| id | Means |
| --- | --- |
| `8mm` | Home-movie film: heavy grain, soft focus, light flicker, faded colour. |
| `16mm` | Documentary film: visible grain, natural colour, slightly soft. |
| `35mm` | Feature film: fine grain, rich colour, shallow depth of field. |
| `vhs-dv` | Consumer tape or early digital video: smeared colour, soft detail, video sharpening. |
| `digital` | Clean modern digital video: sharp, no grain. |

## `tempo`

How fast the shot feels. A tempo sets how fast the move goes and, in a piece
with cuts, how long each shot holds.

| id | Means |
| --- | --- |
| `still` | Almost nothing moves; long holds. |
| `measured` | Slow, deliberate movement. |
| `brisk` | Normal-to-quick movement; shorter holds. |
| `frantic` | Fast, jittery movement; quick cuts. |

Genre is not an axis: it is the subject's business and stays in the user's
own words. A genre usually implies a few values (a thriller leans
`low-key`, `handheld`, `brisk`); state those as values rather than
naming the genre and hoping.
