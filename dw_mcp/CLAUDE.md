# dw_mcp

Guidance for the `dw_mcp/` stdio MCP server package.

`dw_mcp/` is a stdio MCP server (`dw-mcp`, `python -m dw_mcp`) that wraps the
`dw.serve` REST API in a structured tool surface: workflow catalog and
introspection, validate/save/delete, the stored prompt library and its
enhancer, queue a run, poll its events, and view or read a generated file.
It covers the REST surface except the SSE event stream (whose polling twin
`/event-log` is what `get_job_events` uses), the two bulk zips (gallery and
assets), and the SPA's static mount. `POST /api/uploads` *is* covered, by `upload_asset`: an
agent that can only name assets already on the box authors workflows it
cannot supply inputs for, so the tool reads a file on this machine and pushes
its bytes, returning the `asset:` reference rather than a path (`assets.py`).
`upload_asset` also takes `content` (base64) instead of `file_path`, for an
agent with no filesystem in common with a remote `dw.serve --mcp` endpoint -
the bytes travel inline in the call, capped at 4MB rather than `file_path`'s
200MB since they compete with the caller's own context budget (#203).
`get_output_audio` is `get_output_image`'s sibling for sound (`media.py`,
#204, #193): it reads `GET /api/gallery/{name}/audio`, which extracts a
video's muxed track as WAV and cuts an excerpt on `start`/`duration`; a
whole clip over the same 4MB budget is still refused rather than cut short,
and the refusal says to ask for an excerpt. Video has no MCP content type,
so `get_output_frames` returns frames of one as `ImageContent` - specific
moments, a contact sheet, or the frame pair either side of each seam. A session
works in one of the server's workspaces: `--workspace` /
`DW_MCP_WORKSPACE` (a *name* on the server, not a directory - `DW_WORKSPACE`
means something else to the engine), `use_workspace` to switch, and
`DwClient._scoped` adds the selector to every request's query string so no
handler carries a workspace parameter. The default sends nothing, so a
session that never chooses looks exactly like one from before workspaces.
`list_jobs` is bounded (newest 20) rather than complete: the unbounded
listing spilled 176 entries past a client's tool-result limit and could not
be called at all, so the tool asks the API for `limit`/`status`, reverses the
oldest-first list the web UI polls, and reports `total` so a cut answer says
it was cut, and it takes a `workspace` of its own to narrow by.
`list_prompts` had the same disease and the same cure: the library's 44
prompts are 87 KB of prompt bodies, which no client will accept, so the
listing asks for `include_text=false` and carries each prompt's
`description`, `intended_model`, `tags` and `text_chars` instead - the
routing table, with `get_prompt` for the one body that was chosen. It also
forwards `tag` and `intended_model`, because the library is where the
trained caption format for a family is already written out and the reason to
read it is to find that one.
`run_workflow` and `validate_workflow` are the other two handlers that carry
one: each takes an
optional per-call `workspace` that `_scoped`'s `setdefault` lets win over the
session's, so one call can be pinned to a workspace other than the session's
without switching it.
`get_server_info`
(`/api/server`) is the capability call: the device a run will use, the dw
version, the workspace and the workflow/output/prompt/asset directories, which is what tells an
agent authoring remotely whether a CUDA-only choice is even available. It is an HTTP client of a *running* `dw.serve` — it owns no
job state and no GPU worker. `guides.py` proxies `GET /api/guides`: the guides an agent reads are the guides
for the engine it is about to drive, so nothing is read from this package's
own install (the `GUIDES` table lives in `dw/server/guides.py`). They exist
because a request names a subject and the catalog is written in shapes, and an
agent with nowhere to look up the shape authors a fresh workflow instead of
composing one. Only `dw_mcp/server.py` imports the MCP SDK; the
handlers in `catalog.py`, `authoring.py`, `prompts.py`, `diagnose.py`,
`media.py`, `assets.py`, `models.py` and `workspaces.py` are plain `(client, **kwargs)` functions, which is what makes
them testable without an MCP session. It is a top-level package rather than
`dw.mcp` on purpose: importing any `dw.*` submodule runs `dw/__init__.py`
and pulls in torch, which a pure HTTP client has no use for — a test guards
that boundary. Seven tools require `acknowledged_cost=True`
(`run_workflow`, `rerun_job`, `enhance_prompt`, `download_model`,
`delete_model`, `update_diffusers`, `delete_workspace`); `run_workflow` and
`rerun_job` also take the acknowledgement *bound* to the plan
`validate_workflow` answered with - `{fingerprint, minutes, downloads}`,
forwarded verbatim, which the server refuses with 409 when the run's shape
changed since (`_acknowledgement_body` in `diagnose.py`; the 409 is rendered
with the new estimate by `DwClient._format_detail`). The three job-queuing tools return as
soon as the job is queued, since a generation outlasts any client's tool-call
timeout; `run_workflow(wait_seconds=N)` then folds the first `wait_for_job`
into the same call (same `MAX_WAIT_SECONDS` clamp, same budget fields), because
measured over ~1,400 agent-driven cases almost every run was followed by a
wait turn of its own. `delete_output(job_id=...)` is the same economy for
cleanup: the job record's `run_dir` is the `<workflow>/<run id>` the
run-directory delete already accepts, so a whole run goes in one call without
a gallery listing to find its name. Authoring has two halves: `get_schema` describes a workflow and
`get_prompt_schema` a stored prompt, which a workflow reaches by
`"prompt:name"`. See docs/MCP.md.

Claude Code shows an MCP server's `instructions` and each tool description
only up to 2,048 characters, then appends "[truncated]": nothing past that
reaches the agent. The instructions ran to 4,056 and cut off before the run
loop, `acknowledged_cost` and the reference prefixes; `validate_workflow`'s
`plan` paragraph sat past the cut. `CLIENT_TEXT_LIMIT` in
`tests/test_mcp_server.py` pins every one under it, alongside
`SURFACE_BUDGET`'s total. Put what an agent must act on first, and point at a
guide section rather than restating it - the instructions and `list_workflows`
sit within a few dozen characters of the limit.
