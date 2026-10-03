# MiniMax Music 3: keeping a run

Part of the `minimax-music3` skill; step 6 of *Run and judge* says when to
read this.

After an inline run worth keeping, `get_job_workflow` and `save_workflow` it,
so the next run is by name rather than pasting JSON; `export_job` bundles
the run — workflow, manifest, job row and media — for git, on the server.
If `auth_required` is false, fetch `open_url` and unpack it into
`exports/` under the session's working directory, never a temp directory
(the archive already unpacks into a job-id folder, don't make one first).
If true, this agent can't attach the token - hand `open_url` to
the person, and keep working via
`get_output_image`/`get_output_audio`/`get_output_frames`.
