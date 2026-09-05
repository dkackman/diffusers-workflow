# dw_mcp

Guidance for the `dw_mcp/` stdio MCP server package.

`dw_mcp/` is a stdio MCP server (`dw-mcp`, `python -m dw_mcp`) that wraps the
`dw.serve` REST API in a structured tool surface: workflow catalog and
introspection, validate/save/delete, the stored prompt library and its
enhancer, queue a run, poll its events, and view or read a generated file.
It covers the REST surface except the SSE event stream (whose polling twin
`/event-log` is what `get_job_events` uses), `POST /api/uploads` (a browser
file picker's route onto the box - an MCP client has no bytes to push there),
the gallery's bulk zip, and the SPA's static mount. `get_server_info`
(`/api/server`) is the capability call: the device a run will use, the dw
version and the workflow/output/prompt directories, which is what tells an
agent authoring remotely whether a CUDA-only choice is even available. It is an HTTP client of a *running* `dw.serve` — it owns no
job state and no GPU worker. Only `dw_mcp/server.py` imports the MCP SDK; the
handlers in `catalog.py`, `authoring.py`, `prompts.py`, `diagnose.py`,
`media.py` and `models.py` are plain `(client, **kwargs)` functions, which is what makes
them testable without an MCP session. It is a top-level package rather than
`dw.mcp` on purpose: importing any `dw.*` submodule runs `dw/__init__.py`
and pulls in torch, which a pure HTTP client has no use for — a test guards
that boundary. Six tools require `acknowledged_cost=True`
(`run_workflow`, `rerun_job`, `enhance_prompt`, `download_model`,
`delete_model`, `update_diffusers`); the three job-queuing tools return as
soon as the job is queued, since a generation outlasts any client's tool-call
timeout. Authoring has two halves: `get_schema` describes a workflow and
`get_prompt_schema` a stored prompt, which a workflow reaches by
`"prompt:name"`. See docs/MCP.md.
