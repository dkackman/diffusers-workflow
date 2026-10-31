# Script to video: cutaway and lead-in joins

Part of the `script-to-video` skill; step 2's per-shot `join` says when to
read this. Both joins are composed from separate runs of existing templates,
joined in the cut by `concat_videos`. Nothing in a template reads `join`: it
tells you which shot is a shot's **source**, and what to render first.

The recipes are written for `dw:minimax-h3`. An LTX-2.5 re-entry works the
same way, through `templates/ltx2/image-to-video` from the same still.

## Taking a frame off a finished shot

Both joins start from a still off a shot that already rendered. That is a
one-step run of the `get_last_frame` (or `get_first_frame`) task, reading the
shot's file by reference:

```json
{
    "id": "shot-frame",
    "steps": [
        {
            "name": "frame",
            "task": {
                "command": "get_last_frame",
                "arguments": { "video": "output:<shot file>" }
            },
            "result": { "content_type": "image/png", "role": "intermediate" }
        }
    ]
}
```

Pass the still on as `{"location": "output:<frame file>"}` (or `keep_output`
it and use `asset:`). It cannot be a step inside the shot's own run: there is
no step between two entries of one `for_each` list (`workflows` guide,
`for_each`).

## Cutaway: A, B, C

A is the scene, B the insert (B-roll, a reaction, a detail), C the return.
C's source is **A**, not B.

1. Render A.
2. Render B as its own shot: a `continue` off A (A's last frame into
   `templates/minimax/image-to-video`) or a fresh `cut`, whichever the script
   asks. Real B-roll is usually a cut.
3. Render C from A: `get_last_frame` of `output:<A>`.
   - A silent or action C: that still as `image` into
     `templates/minimax/image-to-video`.
   - A talking C: `templates/minimax/dialogue-short` with the same cast
     references (portraits and voices) as A, so the voice carries; one
     frame alone does not carry a voice.
   - C's prompt opens on A's `ending_state`, never on B's.
   - For a C that has to pick up motion mid-action, A's tail can go in as a
     video reference instead (`templates/minimax/chain-video-continuity`'s
     pattern): it keeps motion and audio context, at the cost of a
     reference slot and the VRAM that clip takes. It is an option, not the
     recipe.
4. `concat_videos` A, B, C in **story order**. Both seams are real picture
   cuts, so their `hard_cut` record is honest.
5. Check the re-entry by comparing A's last frame with C's first
   (`get_output_frames` on each), not seam 2: seam 2 is B into C, a cut that
   is meant to jump.

## Lead-in: B into C

B's last frame is C's first frame. **Render the target first**: C before B.

1. Render C.
2. `get_first_frame` of `output:<C>`.
3. Render B with `templates/minimax/first-and-last-frame`: `image` is B's
   own opening (the source shot's last frame, or a still), `last_image` is
   C's first frame. When B has no fixed opening, use
   `templates/minimax/last-frame-only` (`last_image` alone).
4. `concat_videos` B, C in story order. The seam is pinned by construction,
   so its `hard_cut` skipping `seam_frame_jump` costs nothing; look at B's
   last frame and C's first with `get_output_frames` anyway.

Keep a lead-in short (the 124-frame floor or near it): the interpolation
between two pinned frames drifts mid-shot over a long B.

## Render order and re-rolls

- Render order follows the sources, not the story: a lead-in's target goes
  before the lead-in, and a cutaway's return waits only on A.
- **Re-rolling C invalidates B** when B leads into C: B is re-rendered off
  the new C's first frame, and the quote names both.
- Re-rolling A invalidates any C that continues from it and any cutaway
  taken off it; quote them with A.
- A forced re-render counts as an attempt against the dependent shot's
  three-attempt budget.
- These are separate shots, so `concat_videos` is the assembly, not a splice:
  "never splice a re-rolled segment into a chain" still holds for chains.
