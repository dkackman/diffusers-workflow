# MiniMax Music 3: keeping a run

Part of the `minimax-music3` skill; step 6 of *Run and judge* says when to
read this.

After an inline run worth keeping, `get_job_workflow` and `save_workflow` it,
so the next run is by name rather than pasting JSON; `export_job` bundles
the run — workflow, manifest, job row and media — for git, on the server.
`auth_required` is false (the zip needs no token): fetch `open_url`
(prefix a relative one with the server address) and unpack it into
`exports/` under the session's working directory, never a temp directory
(the archive already unpacks into a job-id folder, don't make one first).
Only without HTTP, or if `auth_required` is true (a gated zip), hand
`open_url` to the person, and keep working via
`get_output_image`/`get_output_audio`/`get_output_frames`.
