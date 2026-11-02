---
name: storyboard
description: Use when a dw MCP server is connected and the user wants a storyboard - a set of boards (stills) for a scene drawn on FLUX 3 Klein from a character reference, a coverage set of one moment from six camera positions, boards used as the first frames of H3, LTX-2.5 or Kandinsky 6 shots, or a timed animatic of the boards with a scratch voice and music bed before any video is paid for.
---

# Storyboards on a dw server

A board is a still that says what a shot will look like before a video model
is paid for. This skill draws the boards on FLUX 3 Klein, builds a coverage
set of one moment, hands boards to a video family as start frames, and cuts
them into a timed animatic. It does not pick a video family's template or
size: that family's skill does, and its rules win. Camera and look words come
from `cinematography`; the shot list's fields from `script-to-video`.

## Before anything

1. `get_server_info`: Klein needs a CUDA card (`flux3-klein`'s rules apply:
   16 steps, no guidance, sides in multiples of 16, gated license).
2. Boards must be on the server: `keep_output(name, asset_name="boards/<scene>-NN")`
   after each draw. Leave the extension off - the engine adds the file's own
   (Klein writes `.jpg`) - and cite `asset:boards/<scene>-NN.jpg` later.
3. All boards of one animatic share one size: keep the same first reference,
   whose aspect ratio sets the board's, for every board of the scene.

## 1. Draw the boards (Popcorn shape)

Higgsfield's Popcorn (the concept only; no code is borrowed) draws a whole
scene's boards from reference images in one pass. Here that is
`templates/flux3/multi-reference`: `images` is the character and location
references in order, and the `instruction` names each one by position, "image
one" the character, "image two" the location. It draws up to eight boards
per scene. Call the template once per board and keep the same references and
`seed` so the cast and place hold; the instruction changes the camera and the
action only, in `cinematography`'s `shot_size`, `move` and `light` words.

- One idea per board: the single state a frame can show, as in the shot list.
- Past eight boards, split the scene. A scene that needs more has several
  beats and wants a second reference set.
- A board is a draw, not a promise. Look at it (`get_output_image`) against
  the reference before it goes anywhere else.
- To change one board and hold the rest, `templates/flux3/image-edit` with
  an instruction naming only the change.

## 2. Coverage of one moment (Shot Board shape)

Shot Board (MIT) names six cuts that cover any moment: establishing, medium,
face close-up, over-the-shoulder, insert and low angle. Pick the hero board
first - the one that best shows the moment - then redraw the other five from
it with `templates/flux3/multi-reference` or `image-edit`, the hero as
"image one". `references/coverage-set.md` has the six, what each frames, and
the instruction pattern. The eight-board cap holds across the set.

## 3. A board as a start frame

A board that passes review becomes a shot's first frame. The video family's
skill picks the template and size; the board must match that size, so resize
it first rather than letting the pipeline crop a face off.

| Family | Template | Board goes in |
| --- | --- | --- |
| H3 | `templates/minimax/image-to-video` | `image` |
| LTX-2.5 | `templates/ltx2/image-to-video` | `image` |
| Kandinsky 6 | `templates/kandinsky6/image-to-video` | `image` |

Write that shot's prompt to say what happens next, not what the board
already shows. The board fixes the first frame only; later frames may drift
from it, so check the last frame (`get_output_frames`) against the board.
After the shots exist, `templates/face-drift-check` measures whether the
character's face held.

## 4. Animatic

Boards, a scratch voice and a music bed cut into one timed clip, cheaply, to
judge pacing and order before video. `references/animatic.md` has the recipe:
`templates/generate-speech` for the voice, `templates/minimax/music` for the
bed, then `templates/animatic` with the boards, one duration per board and
the soundtrack. Change a duration and re-run: it costs seconds, not GPU
hours.

## Run and judge

`validate_workflow`, quote `plan.estimate`, `run_workflow`, `wait_for_job`,
`get_job`. A template's `subfolder` sorts what it writes: `final` for the
deliverable (the animatic), `intermediate` for scratch. Boards you keep as
assets are the deliverable of the drawing pass; set `"role": "intermediate"`
on a board that is only a step toward a hero. Read a project's shot list with
`get_prompt("<project>/shots")` (`list_prompts` finds the project). Judge boards
with `get_output_image` and the animatic with `get_output_frames`.

## Not in scope

- Video generation, the cut and the score of the final film: the family
  skills, `script-to-video` and `series-episodes`.
- Layout by box from a short request (`flux3-klein`'s `layout-boxes` is by
  hand).
- Sound timed to the picture: the animatic holds boards, it does not align
  them to audio.

## Sources

`dw:flux3-klein`, `dw:cinematography`, `dw:script-to-video`; Higgsfield
Popcorn (concept only) and Shot Board (MIT) for the two shapes;
`workflows/templates/animatic.json`; `docs/TASKS.md` (`stills_to_video`).
