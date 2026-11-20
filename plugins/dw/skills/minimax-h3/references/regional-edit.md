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

## Replace a performer

The same engine, looped over a take's cuts: `templates/minimax/replace-performer`
regenerates the person inside a per-frame mask from a portrait, and keeps the rest
of each cut and the take's whole soundtrack. Measured on one cut
(`bench/records/performer-replacement.md`): the new face matches the portrait
better than a fresh ref2va render of it (face_drift median 0.73-0.76 against 0.68,
threshold 0.363); outside the mask the source holds (37.5 dB past 96 px).

1. **Prep.** Run `templates/minimax/replace-performer-cuts` on the take. It
   writes `cuts` (`name`, `start_frame`, `num_frames`, `render_frames`) from
   `detect_cuts`, and a person mask video (`video_frames` then `segment`, prompt
   "person", threshold 0.3). For a two-person shot "person" masks both: set
   `track` to the number of people and the `people` step (`track_people`) writes
   one mask per person (`people-0.0-person-N.mp4`, numbered by size per cut, largest
   first) and `people-0.0-contact_sheet.png`. Read the sheet to choose the number
   for the one to replace, and pass that mask as `mask_video`; check each cut,
   as numbering restarts at every cut. `track` 0 (default) tracks nobody.
2. **Choose cuts.** Merge fragments from the returned list, and pass only the
   cuts to re-render, in the take's order. A cut with nobody to replace in it is
   left out of `cuts`: the template puts each rendered cut back over the take's
   own frames and returns the whole take, so a cut left out stays as the take has
   it, is not rendered and is not priced, and the picture stays under its own
   sound. `render_frames` is what H3 renders for the cut: on the 17n+5
   grid, 124 to 345, and at least its `num_frames`. The cut is padded to it and
   cut back to `num_frames`.

   **Limits**, each refused by `validate_workflow` before anything is queued:
   - 1 to 32 cuts (`for_each`'s limit), in the take's order, no two
     overlapping;
   - each cut's `render_frames` on 17n+5 inside 124-345 and ≥ its `num_frames`;
   - `references` within the ref2va class's limits: at most 9 images and
     12 files in all;
   - a cut with no `seed` of its own renders with the workflow `seed`.
3. **The swap line.** In the prompt's description say it exactly: "Replace the man
   in the grey flannel shirt in `<Video 1>` with `<Picture 1>`, keep the
   background, lighting, camera and performance". Describe the portrait's man in
   `subject_definitions` and say its backdrop is not used.

   **A line for every person in the shot.** Write one line per person, the kept
   ones too, left to right as they stand in the frame. A person the prompt leaves
   out is re-dressed: the model fills them in from the rest of the caption. Write
   each change in the "becomes" form, one line per person: "The man on the left
   in the grey flannel shirt becomes `<Picture 1>`: a woman in her sixties with
   short white hair"; "The woman on the right in the green coat stays as in
   `<Video 1>`". Don't give two people colliding colour words (two "red" jackets,
   a "blue" shirt beside a "navy" one): the colour lands on the wrong person or
   on both.

   **Big changes land, subtle ones don't.** Age, gender, hair and build change
   reliably. A small change - an eyebrow, a slightly different nose, a shade of
   the same hair colour - falls back to the clip's person, so check the frames
   for it before calling the swap done. (These caption rules are after
   h3-recast's, concepts only.)

   **Describe the cut's performance.** At `refine_strength` 1.0 nothing of the
   person survives inside the mask: the pose, action and framing come from the
   prompt and the kept context. A prompt that says "sits on the couch" redraws a
   standing man seated, square to the camera, in the portrait's pose. Say where
   the person is, how they stand or sit, what they do and how the cut is framed,
   and name every other masked person as kept (`fully_preserved`, "as in
   `<Video 1>`"): "person" masks everyone in the shot. One prompt rarely fits
   every cut. A cut carrying `prompt` renders with it, else the workflow `prompt`,
   so give each cut whose framing or cast differs its own (measured on ep62,
   #1374: with a two-shot prompt on a single-person push-in, the close-up came
   back as a wide two-shot).
4. **Run.** `replace-performer` takes `source_video`, `mask_video` (same size as
   the source, both 24 fps), `cuts` and the portrait. Each cut is trimmed from
   both, edited at `refine_strength` 1.0 with the cut's own audio held, cut back
   and put back over the take's frames (`replace_cuts`, a hard cut each side).
   The result is the whole take, full length, under its own soundtrack; for one
   stretch alone, `trim_video` the result.
5. **Re-roll one cut.** A cut's own `seed` overrides the workflow `seed`.
   `rerun_job` with `arguments` giving `cuts` where only the bad cut carries a new
   `seed`; the others keep the workflow seed and render as before. Keep the
   other cuts in the list: one left out goes back to the take's own frames.

The seam: the edit regenerates a band of about 64-80 px around the person (the
patch cell plus `mask_grow`), and that band can redraw nearby set geometry so it
no longer lines up with the kept source. On the bench the brick wall matched, but
the couch back beside the shoulder was redrawn lower and its top edge stepped
against the kept couch; the magazine inside the band was redrawn with garbled
text. `mask_grow` 0 does not fix it (#1374, two seeds): it confines the change to
about 32 px instead of 80, but the edge is harder (`seam_bands.py` step 9.8-11.3
against 4.4-4.5 at 1) and the couch back is still redrawn at the shoulder. The
template keeps `mask_grow` 1. Judge a still at the silhouette, not only the
motion.
