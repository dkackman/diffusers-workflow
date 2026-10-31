# Fixing one region of a good take

A take that is right except for one thing - the cup, the jacket, a sign - is fixed
in place with `templates/minimax/regional-edit`: the mask's region is regenerated
from the prompt, the rest of the picture is the take's, and the whole soundtrack
is the take's (`hold_audio` is the `source_video`). Engine rules, refusals and the
measurements behind the bands: `get_guide("workflows", section="H3: regional edit
with source_video and a mask")`.

## The take

`source_video` is the take: an `output:` reference to the earlier run's file, or
an `asset:` after `upload_asset`. The step renders at its `width`, `height` and
`num_frames` (960x544, 124 frames by default), and the take and the mask must
match them: read the take's with `get_gallery_metadata` and pass all three when
it differs. Validation names both sizes on a mismatch. The take must carry audio.

## The mask

One `mask_image`, the take's size, used on every frame. White regenerates, black
is kept, binarized at luma 128; an all-black mask is refused. Either:

- the user's own mask, sent with `upload_asset`;
- or `segment` with a text prompt ("the cup") on one frame of the take: a small
  workflow of `get_frame` (`video` the take's `output:` or `asset:`,
  `frame_index` where the object sits) then `segment` (`image` that frame), as
  in `templates/segment`; `keep_output` the mask and pass it as `asset:`.
  The object must stay inside the mask for the whole clip - a moving one needs a
  mask covering everywhere it goes, or a `mask_video` (one mask per frame, see
  the guide) on the step in place of `mask_image`.

Grain: the mask is max-pooled to the patch grid, 16 px times the patch size per
cell, so any white pixel marks its whole cell, and `mask_grow` (default 1) widens
it by that many cells. In time a cell is a latent frame, about 3.4 pixel frames.
Leave a margin; a tight mask shows its seam.

## refine_strength

The template defaults to 0.9, at its 5 steps. Measured on one 960x544 take at 5 turbo steps,
`video_shift` 6, seed 1062:

| `refine_strength` | Inside the mask |
|---|---|
| 0.85 or less | a light rework: re-detailed, objects and colours stay |
| 0.9 | replaces an object; same person, pose, clothing |
| 0.95 | reworks clothing too; face, hair, pose mostly hold |
| 1.0 | a fresh generation: the prompt lands, identity is lost |

Outside the mask held at every strength. A large recolour (a dark top to a
bright red coat) did not land below 1.0, and 1.0 is a fresh generation: that is
not a regional edit's job - regenerate the take instead. Another `video_shift`
or step count moves the bands (lower shift, lower strength for the same change).

Write the prompt for the whole shot as it should look after the edit, the same
Context-IR as the take's with the changed thing described, not an instruction.

## Judge

`compare_outputs` the edit against the take (what changed, what held),
`get_output_frames(count=12)` for the region across the clip, `get_output_audio`
to confirm the soundtrack is the take's. Not with `refine_tiles` or on a chain
step: validation refuses both.
