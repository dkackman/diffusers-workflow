# Checking an episode's lines

Part of the `series-episodes` skill: confirming every line rendered with
`templates/check-script` rather than judging it by ear.

1. `validate_workflow(name="templates/check-script",
   arguments={"input_audio": "output:<name>", "lines": [...]})` first. It is
   free, and it takes the episode's muxed soundtrack directly. `lines` is the
   script in order, `{text, shot}` to name each line's shot.
2. `run_workflow(..., acknowledged_cost=<that plan's {fingerprint, minutes,
   downloads}>, wait_seconds=60)`. This workflow's plan is
   `basis: "unknown"` with `minutes: null`, so quote seconds, not minutes.
3. `get_output_text` for its `findings`: a dropped or changed line, speech in
   a silent shot, a clipped last word.
4. `delete_output(job_id=...)` the scratch run afterward.
