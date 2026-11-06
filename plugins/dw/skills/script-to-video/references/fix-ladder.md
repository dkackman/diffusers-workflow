# The fix ladder and its caps

Part of the `script-to-video` skill; step 5's *Run and judge* says when to
read this. A shot that reads wrong is fixed in a fixed order, cheapest
change to the least of the take first, and each rung has a cap. The ladder
and its caps are adapted from open-higgsfield-skills (MIT).

## The rungs

1. **Edit the region, or repair the face.** When the take is good and one
   part is wrong, change only that part: `templates/minimax/regional-edit`
   for a region of an H3 take (the rest and the audio stay), and
   `templates/minimax/face-repair` for a soft small face. Read
   `dw:minimax-h3` first. Other families have no such template: go to rung 2.
2. **Re-prompt.** Change the words, keep the seed, so a pinned voice and a
   composition that worked survive. Edit the shot object in
   `<project>/shots`, recompile, `save_prompt` the shot (see
   `references/shot-document.md`). For H3, run
   `minimax-h3/references/prompt-lint.md` over the new prompt first.
3. **Reseed.** `rerun_job(new_seed=True)`. A new seed redraws an unpinned
   voice as well; use it only when the voice is pinned by a reference or does
   not matter.
4. **Ask.** Stop and put the shot, the three attempts and what each changed
   in front of the user. Do not widen a cap to keep going.

## Caps

- **Three failed attempts per shot**, counted across all rungs. A seam
  re-roll counts against the shot's budget too.
- **A GPU-minute cap for the project**, stated with the cost quote before
  the batch runs (`plan.estimate`), and spent down by every re-run. When the
  next fix would pass it, ask first; `acknowledged_cost` is the user's, not
  the loop's.
- **QC before the next segment starts.** In a chain, judge a segment's
  frames before the next one is queued: a bad one is cheaper to fix alone
  than after the rest were drawn from its last frame. (The idea is Body
  Double's, lucataco/open-source-higgsfield.)

A fix is judged by looking: `get_output_frames` on the new take, then the
beat review in `references/beat-review.md`.

## Face drift

`templates/face-drift-check` takes as its `reference` the character's cast
portrait from casting, the one the shots were conditioned on. A `face_drift`
finding is a flag to look at with `get_output_frames`, not a verdict. On a
confirmed drift, re-roll the shot on this ladder, conditioned on the cast
reference; the re-roll counts against the shot's budget.
