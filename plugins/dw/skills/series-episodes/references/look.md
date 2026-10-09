# A series look: one palette on every shot

Optional. It runs per episode, after the shots exist and before
`templates/assemble-and-score` cuts them. It is not one of the five beats,
and an episode cut without it is still a finished episode. Do it for every
episode or for none, because a series where only some episodes are graded
reads as two series.

## The palette is the look

`apply_lut` with `palette` builds a colour lookup from 2 to 16 `#rrggbb`
colours, ordered dark to light. Each pixel keeps its own brightness and
takes its hue from the palette at that brightness, so shadows lean toward
the first colour and highlights toward the last. Nothing is written to disk,
and the same palette gives a byte-identical result on the same input. That
makes the palette the series' look in the same way step 0's portraits are
its cast: choose it once, before episode 1, and pass that exact list to
every episode. Keep it where the series' other fixed choices live, with the
cast and the series `target_lufs`.

`strength` (0..1, default 1) blends the graded result with the original.
Around 0.4-0.7 is a look; 1 is a full re-colour. Fix it with the palette.

A `.cube` from a grading app works in place of a palette: `upload_asset` it
once with `shared=True`, then pass `"lut": "asset:looks/<series>.cube"`
instead of `palette`. Give exactly one of `lut` or `palette`. The parser is
strict: a 3D LUT only, domain 0..1, values in [0,1]. A file outside that is
refused, with the line named.

## Grain, optionally

`film_grain` after the grade gives the cut one texture across shots that
were generated separately. `amount` is 0..1 (keep it low, 0.04-0.1, for
video), `size` is 1 or above, in pixels, and `chroma` is 0..1, where 0 keeps
hue unchanged. Every frame gets different grain. The run's seed is recorded,
so `rerun_job` reproduces the grain exactly.

## The workflow, one run per episode

`shots` takes the episode's clips, as `output:` references off each
generation run's manifest (see the skill's *Run and judge*) or `asset:`
references. `look` is the series palette.

```json
{
    "id": "series-look",
    "variables": {
        "shots": ["asset:shot_1.mp4", "asset:shot_2.mp4"],
        "look": ["#1b2433", "#6f6a5e", "#e6c79a"]
    },
    "steps": [
        {
            "name": "graded",
            "for_each": "variable:shots",
            "task": {
                "command": "apply_lut",
                "arguments": {"media": "item:", "palette": "variable:look", "strength": 0.6}
            },
            "result": {"content_type": "video/mp4", "subfolder": "intermediate"}
        },
        {
            "name": "grain",
            "for_each": "variable:shots",
            "task": {
                "command": "film_grain",
                "arguments": {"media": "previous_result:graded", "amount": 0.06, "size": 1.5}
            },
            "result": {"content_type": "video/mp4", "subfolder": "final"}
        }
    ]
}
```

- `"media": "item:"` is the shot itself, because each entry is a plain
  string. `previous_result:graded` inside `grain` resolves to the graded
  member of the same shot: two `for_each` steps over one list are paired
  entry by entry.
- To skip the grain, drop the `grain` step and set `graded`'s `subfolder` to
  `final`.
- Both tasks keep each clip's size, frame rate, frame count and audio, so
  the graded clips still pass `concat_videos`' same-size, same-rate check.
- `validate_workflow` it first. It is free, and it refuses a bad palette
  entry by name. The run is CPU-only, with no GPU and no downloads.

Pass the graded files, in order, as `assemble-and-score`'s `shots`: the
`final` files of the look run, read off its manifest with `get_job`, as
`output:` references. Or `keep_output` them if they must outlive the run.
Grading after the cut instead works too, with one `apply_lut` on the
finished episode, but grading per shot lets a shot be regenerated and
regraded without regrading the rest.
