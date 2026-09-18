---
name: series-episodes
description: Use when a dw MCP server is connected and the user wants a series - several episodes sharing a cast, cut and scored one at a time rather than a single generation. States the five-beat recut-bed-match_levels-normalize-pair procedure that turns generated shots into a scored episode, and the one step that must not be skipped or the cast drifts between episodes.
---

# Series episodes on a dw server

An episode is not one generation. Each episode's shots come from their own
`minimax-h3` template runs (`templates/minimax/dialogue-short` or
`templates/minimax/music-video`), and the pieces are then cut and scored
together the way `templates/assemble-and-score` does for a single film. A
series is that same procedure run once per episode, against one cast that
must look and sound the same in episode 5 as it did in episode 1 - which
does not happen by default, because each `dialogue-short`/`music-video` run
draws its own portraits unless told not to. This skill states the procedure
and the one step (0 below) that keeps the cast fixed; everything past it is
`minimax-h3`'s and `assemble-and-score`'s own rules, not repeated here.

## Before anything

1. `get_server_info` for the device and workspace.
2. `list_workflows(shape="shot")` for the H3 templates, `list_workflows(shape="sequence")`
   for `assemble-and-score`. Trust the listing over the names quoted here.
3. Read the `minimax-h3` skill before writing any shot - the `shots` list
   shape, the prompt format, and the hard rules (frame count, canvas,
   reference limits) all come from there and are not restated here.

## 0. Draw the cast once, before episode 1

Draw each character's portrait (and, for a voice that must hold, its
reference clip) exactly once, in whichever workflow is convenient - a bare
`templates/minimax/dialogue-short` run works, or any Z-Image step. Then
`keep_output(name, asset_name="cast/<character>.png", shared=True)` for
each portrait, and `upload_asset(..., shared=True)` for a voice clip that
did not come from a run. `shared=True` is the whole point: it puts the
portrait in the asset library every workspace shares
(`docs/WORKSPACES.md`), so an episode generated in a different workspace,
weeks later, still resolves `asset:cast/<character>.png` to the same file.

Every later episode's `shots` entries reference that cast with `from_file`,
never `from_previous_result`:

```json
"references": [
  {"reference_type": "image", "from_file": "asset:cast/priya.png"},
  {"reference_type": "audio", "from_file": "asset:cast/priya-voice.wav"}
]
```

This is the step that goes missing: a `from_previous_result` reference (or
an unreferenced portrait prompt, redrawn from a text description that reads
close enough to pass review) draws a new face for the character in every
episode that uses it, and nothing in validation catches it - the shot
generates cleanly, and the drift only shows up when two episodes are
watched back to back. There is no engine check for "the same character
looks the same" across separate runs; the fixed-asset reference is the only
guard, so it has to be a deliberate, first step rather than an implicit one.

## 1-5. Per episode: generate, then recut, bed, match_levels, normalize, pair

Generate the episode's shots with `templates/minimax/dialogue-short` (or
`music-video`), `shots` referencing the fixed cast from step 0. Each shot
is its own H3 generation, so cast drift *within* one episode does not
happen - only *across* episodes does, which is what step 0 closes.

Once every shot for the episode exists (freshly generated, or promoted from
a run with `keep_output`), cut and score them with
`templates/assemble-and-score` - it is the five-beat pipeline this skill
composes into, not something to re-derive:

- **recut**: `shots` (the episode's clips, in order) goes straight into
  `concat_videos` as hard cuts - nothing is stabilized or rescaled, so the
  clips must already share one size and frame rate.
- **bed**: if the episode's score is shorter than the cut, stretch it first
  with the `loop_audio` task (`target_frames` = the cut's `total_frames`,
  `fps` matching) rather than letting `assemble-and-score` pad the tail
  with silence.
- **match_levels**: shots generated independently drift in loudness -
  `assemble-and-score`'s `match_levels` (`"rms"` or `"peak"`) evens them
  before the cut; leaving it null only warns on a wide spread instead of
  fixing it.
- **normalize**: the mixed world sound and score are normalized together
  (`assemble-and-score`'s `balanced` step, -3 dBFS) so one episode is not
  louder than the next.
- **pair**: the normalized track is muxed onto the cut - the episode's
  deliverable.

One `assemble-and-score` run per episode; each episode's `total_frames` and
`shots` list are its own.

## Run and judge

`output:` a shot straight from its generation run
(`output:dialogue-short/<episode run id>/final/<shot>.mp4`) rather than
downloading and re-uploading it as an asset - only the cast portraits from
step 0 need to be assets, since they are the one thing that must outlive a
single episode's run directory. The `final` segment there is the step's
`subfolder`: the deliverable shot each H3 template writes, as opposed to
`intermediate` scratch that an `output:` reference has no reason to name.

Beyond `minimax-h3`'s own judging checklist (a character that changes
between shots, a portrait imposing its framing, a voice without affect):
watch two episodes back to back and check the cast reads as the same
people - the failure mode step 0 exists to prevent. `get_gallery_metadata`
on each episode's final file for duration and loudness, so a level
mismatch between episodes shows up before a viewer notices it.

## Sources

`workflows/templates/assemble-and-score.json`, `dw/tasks/audio_utils.py`
(`loop_audio`, `match_levels`, `normalize_audio`), `dw/tasks/pair_audio.py`,
`docs/WORKSPACES.md` (the shared asset library), the `minimax-h3` skill
this composes into. Written from issue #217 (dkackman/diffusers-workflow),
which named the drift as a recurring, undocumented failure across two
hand-built episodes.
