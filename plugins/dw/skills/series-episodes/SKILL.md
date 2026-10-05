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
2. `list_workflows(shape="sequence")` for the H3 templates and for
   `assemble-and-score` - both `dialogue-short`/`music-video` and
   `assemble-and-score` are shape `sequence`. Trust the listing over the
   names quoted here.
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

Save each portrait prompt with `save_prompt` under one folder for the series
and reference it as `prompt:<series>/<character>` from every episode.
`list_prompts(tag="<series>")` is then the cast list, and a character
described the same way in episode 6 as in episode 1 is a reference rather
than a paragraph retyped - which is the drift this skill exists to stop.

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
  with silence. A dialogue episode also needs room tone under its line
  gaps: `find_loop_bed` on the recut (`output:` it, so the run's shots keep
  every candidate inside one shot) names the stretch to `slice_audio` and
  the `gain` to `mix_audio` it at.
- **match_levels**: shots generated independently drift in loudness -
  `assemble-and-score`'s `match_levels` (`"rms"` or `"peak"`) evens them
  before the cut; leaving it null only warns on a wide spread instead of
  fixing it. A shot whose voice-over the score buries is a different
  problem, not fixed by `match_levels` or `world_gain` (which lifts the
  whole world track, action sound included) - see `minimax-h3`'s ducking
  recipe: `gain_audio` regions on the score, one per voice-over shot,
  applied before the score is passed in.
- **normalize**: the mixed world sound and score are normalized together
  (`assemble-and-score`'s `balanced` step, -3 dBFS peak ceiling, always
  enforced) so one episode is not louder than the next. A shared peak
  ceiling does not mean a shared loudness - a sparse, dialogue-only episode
  and a dense, score-heavy one can both sit at -3 dBFS peak and still read
  as very different volumes. `assemble-and-score`'s `target_lufs` variable
  (passed through to `normalize_audio`) gains the mix toward a measured
  loudness before the ceiling is applied - but gain down succeeds only
  while the mix's own peak is below the ceiling; a single loud peak (a
  laugh track, a sting) caps gain up the same way, and can pull a gain-down
  episode lower still. Match episodes **downward** anyway: pick one series `target_lufs`
  that every episode can reach - in practice, about the quietest episode
  the ceiling holds back - and pass that same value on every episode's
  `assemble-and-score` run. Don't pick the loudest episode's loudness and
  ask the others to climb to it. A `target_lufs_capped` warning on an
  episode means that episode's own peak sets the series' ceiling; lower the
  series `target_lufs` rather than accepting the mismatch. Only when the
  series should sit louder than its most dynamic episode allows (e.g. -16
  for streaming) pass `limit: true` beside the same `target_lufs` on every
  episode: -3 dBFS becomes a true-peak ceiling a look-ahead limiter holds,
  so the laugh is limited rather than setting the gain. A `limiter_heavy`
  warning on an episode means the series target is squashing it audibly -
  lower the series `target_lufs` rather than living with it.
- **pair**: the normalized track is muxed onto the cut - the episode's
  deliverable.

One `assemble-and-score` run per episode; each episode's `total_frames` and
`shots` list are its own.

## Run and judge

Judge a finished episode with `assess_output(name)` before listening end
to end: it measures every seam and the shots' levels, and says where to look.
A `shot_dead_air` finding is an H3 dialogue gap (0.5-2s of near-silence
between lines) inside one shot, not a seam problem - see `minimax-h3`'s
room-tone bed recipe (`find_loop_bed`, then `slice_audio` -> `loop_audio` ->
`mix_audio`) rather
than treating it as a cut to fix.

`output:` a shot straight from its generation run rather than downloading
and re-uploading it as an asset - only the cast portraits from step 0 need
to be assets, since they are the one thing that must outlive a single
episode's run directory. For `dialogue-short`/`music-video` the per-shot
clips are the `shot` step's own files, and that step's `subfolder` is
`intermediate` - `final` there holds only the already-concatenated episode,
which is not a usable `shots` entry. So a recut references `output:` + the workflow's full catalog path (as
`list_workflows` names it, prefix and all) + `/<episode run
id>/intermediate/<workflow id>-shot@<name>.<index>-0.0.mp4` - not just the
template's own name - read off the run's manifest (`get_job`) rather than
guessed. Or, if the shot should outlive its run directory the way the cast
does, `keep_output` it and reference the resulting `asset:` instead.

Beyond `minimax-h3`'s own judging checklist (a character that changes
between shots, a portrait imposing its framing, a voice without affect):
watch two episodes back to back and check the cast reads as the same
people - the failure mode step 0 exists to prevent. `get_gallery_metadata`
on each episode's final file for duration and loudness, so a level
mismatch between episodes shows up before a viewer notices it - its
`peak_dbfs` is a single sample and does not say how loud the episode reads
as a whole; `integrated_lufs` (BS.1770, whole-track) is the field that
answers that, and is what to compare across episodes. A peak-normalized
episode with one loud outlier (a studio-audience laugh, a sting) can sit
1-2 LU quieter than its neighbors even at the same -3 dBFS peak ceiling -
that gap is `target_lufs`'s to close, matched downward to a series-wide
value every episode's ceiling allows (see the **normalize** bullet above);
`target_lufs` alone cannot raise a capped episode to meet a louder one -
`limit: true` can, for a series that must sit louder (same bullet). To confirm a
line actually rendered rather than judging it by ear, `get_output_audio`
returns sound, not text: `validate_workflow(name="templates/transcribe-audio",
arguments={"input_audio": "output:<name>"})` first (free; it takes the
episode's muxed soundtrack directly), then `run_workflow(...,
acknowledged_cost=<that plan's {fingerprint, minutes, downloads}>,
wait_seconds=60)`, then `get_output_text` on the result, and
`delete_output(job_id=...)` the scratch run afterward. This workflow's
plan is `basis: "unknown"` with `minutes: null` - quote seconds, not
minutes; it runs in a few seconds.

To take the project home, call `export_job` once per job of the project (each
episode, and the cast run), then fetch each zip's `open_url` and unpack it into
`exports/` under the working directory (per its `next`).

## Sources

`workflows/templates/assemble-and-score.json`, `dw/tasks/audio_utils.py`
(`loop_audio`), `dw/tasks/joins.py` (`match_levels`),
`dw/tasks/audio_dynamics.py` (`normalize_audio`), `dw/tasks/pair_audio.py`,
`docs/WORKSPACES.md` (the shared asset library), the `minimax-h3` skill
this composes into.
