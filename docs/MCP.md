# MCP Server

A fourth way to drive the engine, alongside `dw.run`, `dw.repl`, and
`dw.serve`: a stdio [MCP](https://modelcontextprotocol.io) server that lets
an MCP client — Claude Code first — author, validate, save, run and diagnose
workflows without shell access or a repo checkout.

`dw/mcp/` is an HTTP client of a **running** `dw.serve`. It owns no job
state and no GPU worker of its own; every tool call is a REST request
against the server described in [Server & Web UI](SERVER.md). If `dw.serve`
is not running, every tool fails with a message telling you to start it.

## Install and run

```bash
pip install -e ".[server,mcp]"

# terminal 1
dw-serve

# terminal 2 (or your MCP client's configured command)
dw-mcp
```

`dw-mcp` (equivalently `python -m dw.mcp`) speaks MCP over stdio. Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `$DW_MCP_URL`, else `http://127.0.0.1:8765` | Base URL of the running `dw.serve` |
| `--timeout` | `30` | Seconds to wait on any one API request |

The `DW_MCP_URL` environment variable sets the same default the `--url` flag
overrides.

## Client configuration

Both Claude Code and Claude Desktop take an `mcpServers` block naming the
launch command. Point it at `dw-mcp` (or `python -m dw.mcp` if you have not
installed the console script).

**Claude Code** — `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "diffusers-workflow": {
      "command": "dw-mcp",
      "args": ["--url", "http://127.0.0.1:8765"]
    }
  }
}
```

**Claude Desktop** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "diffusers-workflow": {
      "command": "dw-mcp",
      "args": []
    }
  }
}
```

If `dw-mcp` is not on the launching process's `PATH` (for example, it is
installed in a virtualenv), use the venv's absolute path as `command`, e.g.
`/path/to/venv/bin/dw-mcp`. `DW_MCP_URL` can be set instead of `--url` via
an `"env"` object alongside `"command"`/`"args"` in either config, if your
client and `dw.serve` are not both on the default port.

## Tool reference

25 tools in four groups. Names and arguments below are generated from
`dw/mcp/server.py` — nothing here is renamed or reshaped for the docs.

### Catalog (read-only)

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_workflows()` | — | List stored workflows with descriptions, output kinds and variables |
| `get_workflow(name)` | `name` | Get one stored workflow's full JSON definition |
| `get_schema()` | — | Get the JSON schema every workflow definition must satisfy |
| `list_pipelines()` | — | List every diffusers pipeline class this installation provides |
| `get_pipeline_signature(name)` | `name` | Get a pipeline's real call arguments |
| `list_classes(kind)` | `kind` | List class names of one kind: pipelines, models, schedulers, or quantization |
| `get_class(name, target="init")` | `name`, `target` (`init`\|`call`\|`load`) | Get a class's argument schema |
| `list_tasks()` | — | List every task command a workflow's task step can name |
| `get_task(command)` | `command` | Get a task command's argument schema |
| `list_models()` | — | List what the Hugging Face model cache holds, largest first |
| `get_memory()` | — | Get the worker's VRAM and RAM statistics |
| `get_health()` | — | Check that the server is alive |
| `list_jobs()` | — | List queued, running and recent jobs |
| `list_gallery(limit=50)` | `limit` | List generated output files, newest first |
| `get_gallery_metadata(name)` | `name` | Get the metadata embedded in a generated file: the exact workflow and arguments that produced it |

### Media (read-only)

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `get_output_image(name, max_dimension=768)` | `name`, `max_dimension` | Look at a generated image, downscaled to `max_dimension` on its longest side |

### Authoring

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `validate_workflow(workflow=None, name=None)` | exactly one of `workflow` (inline definition) or `name` (stored workflow) | Check a workflow against the schema and against real pipeline signatures. Free and instant |
| `save_workflow(name, workflow)` | `name`, `workflow` | Save a workflow to the server, overwriting any existing workflow of that name |
| `delete_workflow(name)` | `name` | Permanently delete a stored workflow |

### Diagnose

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `run_workflow(workflow_path=None, inline_workflow=None, arguments=None, acknowledged_cost=False)` | exactly one of `workflow_path` or `inline_workflow`, optional `arguments`, `acknowledged_cost` | Queue a workflow for generation. Returns as soon as the job is queued |
| `get_job(job_id)` | `job_id` | Get a job's status, warnings, output manifest, error and traceback |
| `get_job_events(job_id, after=-1, limit=200)` | `job_id`, `after`, `limit` | Get a page of a job's progress events |
| `cancel_job(job_id)` | `job_id` | Ask a queued or running job to stop |
| `rerun_job(job_id)` | `job_id` | Queue a fresh job from a previous job's stored specification |
| `move_job(job_id, direction)` | `job_id`, `direction` (`up`\|`down`\|`front`\|`back`) | Reorder a queued job |

## The run gate

`run_workflow` refuses unless `acknowledged_cost=true` is passed — a run
occupies the machine for minutes and the engine runs one job at a time, so
the tool's instructions tell the model to describe what is about to run and
get the user's go-ahead first. Passing the flag does not make the tool
wait: it returns as soon as the job is queued, the same way queuing a job
from the web UI does not block the browser tab.

The intended loop:

1. `validate_workflow` — free, checks schema and pipeline signatures, no GPU
   time spent
2. `run_workflow` with `acknowledged_cost=true` — queues the job and returns
   immediately with a `job_id`
3. `get_job_events(job_id)` repeatedly, passing back the previous call's
   `last_seq` as `after`, until the status is terminal
4. `get_job(job_id)` for the finished manifest (or the error and traceback,
   if it failed)
5. `get_output_image(name)` to look at a result image

## Security

The MCP server adds no authentication of its own — it inherits the REST
API's posture exactly, described in full in [Security](SECURITY.md):
localhost binding, no auth, `Origin` header checks, and path confinement in
`dw/security.py` for every workflow, gallery and prompt path a tool touches.
Nothing under `dw/mcp/` re-implements or loosens that confinement; it is
purely a client of the same validated endpoints the web UI uses.

Because it adds no authentication, `dw-mcp` must not be pointed at a
`dw.serve` reachable beyond localhost. Treat `--url`/`DW_MCP_URL` the same
way you would treat opening the web UI to the network: don't.

## Known limits

- **Event history is bounded.** `get_job_events` serves at most the last 200
  events of a finished job (`MAX_PERSISTED_EVENTS` in the job history store).
  A job that ran before this feature existed returns an empty event list
  with a `note` explaining why.
- **Images only.** `get_output_image` decodes and returns images; it refuses
  video and audio outputs. Use `get_gallery_metadata` to inspect other media
  kinds.
- **Not exposed yet:** model download/deletion, dependency/diffusers-version
  management, and the prompt library (`prompt:` references still work
  inside a workflow's own JSON — there is just no tool to browse or edit
  stored prompts).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| "Cannot reach diffusers-workflow at …" | `dw.serve` is not running. Start it with `dw-serve` (or `python -m dw.serve`) and try again |
| A tool call times out | Usually a model loading into VRAM/RAM for the first time; retry, or raise `--timeout` |
| `run_workflow` refuses with a cost message | Not an error — it is the `acknowledged_cost` gate. Confirm with the user and call again with `acknowledged_cost=true` |
