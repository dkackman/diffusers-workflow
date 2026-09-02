# MCP Server

A fourth way to drive the engine, alongside `dw.run`, `dw.repl`, and
`dw.serve`: a stdio [MCP](https://modelcontextprotocol.io) server that lets
an MCP client — Claude Code first — author, validate, save, run and diagnose
workflows without shell access or a repo checkout.

`dw_mcp/` is an HTTP client of a **running** `dw.serve`. It owns no job
state and no GPU worker of its own; every tool call is a REST request
against the server described in [Server & Web UI](SERVER.md). If `dw.serve`
is not running, every tool fails with a message telling you to start it.

## Install and run

```bash
pip install -e ".[server,mcp]"
```

The MCP server needs a running `dw.serve`, so two processes are involved:

```bash
# terminal 1 - the engine. Leave it running.
dw-serve
```

You do not start `dw-mcp` yourself. Your MCP client launches it on demand,
which is why the client needs a command it can actually find (see below).

`dw-mcp` (equivalently `python -m dw_mcp`) speaks MCP over stdio. Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `$DW_MCP_URL`, else `http://127.0.0.1:8765` | Base URL of the running `dw.serve` |
| `--timeout` | `30` | Seconds to wait on any one API request |

The `DW_MCP_URL` environment variable sets the same default the `--url` flag
overrides.

### Use the absolute path to `dw-mcp`

**This is the one setup detail that reliably goes wrong.** If you installed
the way `install.sh` does, `dw-mcp` lives in the project's virtualenv and is
only on `PATH` while that venv is activated. Your MCP client is launched by
your shell, your desktop app, or your editor — usually *without* the venv
activated — so a bare `dw-mcp` fails to spawn:

```
Failed to reconnect to dw: ENOENT
```

Registering it once from an activated terminal hides this: that session
works, and the next one, started somewhere else, does not.

Always register the venv's absolute path. Console scripts have the
interpreter baked into their shebang, so they run correctly with no venv
activated — which is exactly why the absolute path is more robust than
telling people to activate first:

```bash
echo "$(pwd)/venv/bin/dw-mcp"    # the value to register
```

## Client configuration

### Claude Code

The CLI is the shortest path. From the project directory:

```bash
claude mcp add dw -- "$(pwd)/venv/bin/dw-mcp"
```

`--` separates Claude Code's own flags from the command it will spawn. Add
subprocess flags after it:

```bash
claude mcp add dw -- "$(pwd)/venv/bin/dw-mcp" --url http://127.0.0.1:8791
```

Pick the scope deliberately with `-s`:

| Scope | Stored in | Use when |
| --- | --- | --- |
| `local` (default) | `~/.claude.json`, keyed to this project | Just you, just this checkout |
| `user` | `~/.claude.json`, global | You want it in every project. The absolute path makes this work |
| `project` | `.mcp.json`, **committed to the repo** | You intend every clone to get it. Note an absolute path is machine-specific and will not port |

Equivalent hand-written `.mcp.json`, if you prefer a file:

```json
{
  "mcpServers": {
    "dw": {
      "command": "/absolute/path/to/venv/bin/dw-mcp",
      "args": ["--url", "http://127.0.0.1:8765"]
    }
  }
}
```

**A running session does not pick up a registration change** - start a new
one after adding or editing the server.

### Claude Desktop

`claude_desktop_config.json`, same shape - and the same absolute-path rule,
which matters more here because a desktop app never inherits a shell's
`PATH`:

```json
{
  "mcpServers": {
    "dw": {
      "command": "/absolute/path/to/venv/bin/dw-mcp",
      "args": []
    }
  }
}
```

`DW_MCP_URL` can be set instead of `--url` via an `"env"` object alongside
`"command"`/`"args"` in either config.

## Verify the setup

Three checks, in order. Each isolates a different failure, so run them in
sequence rather than jumping to the last one.

**1. The command launches without a venv.** This reproduces the environment
your client actually spawns it in, and is the check that catches ENOENT:

```bash
env -i PATH=/usr/bin:/bin HOME="$HOME" /path/to/venv/bin/dw-mcp --help
```

Prints usage and exits 0. If it does not, the path is wrong or the package
is not installed into that venv.

**2. The client sees the server.** Start a new Claude Code session and run
`/mcp`; `dw` should be listed and connected. From the shell,
`claude mcp list` and `claude mcp get dw` show the same thing.

Note that a "connected" status only means the process launched - it says
nothing about whether `dw.serve` is reachable.

**3. The tools reach the engine.** With `dw-serve` running, ask the client
something free, such as "list my diffusers workflows" (`list_workflows`) or
"check the diffusers-workflow server health" (`get_health`). A real answer
means the whole chain works. "Cannot reach diffusers-workflow at ..." means
step 3 failed while steps 1 and 2 passed - the client is fine and the engine
is not running.

Nothing in this sequence costs GPU time.

## Tool reference

41 tools in six groups. Names and arguments below are transcribed from
`dw_mcp/server.py` — nothing here is renamed or reshaped for the docs.

### Catalog (read-only)

The catalog is large, so the server's instructions point a client at
`list_workflows` first: its listing carries enough about each workflow -
description, output kinds, variable names - to pick one and know what to
pass it, without fetching every candidate's definition. Reusing a stored
workflow is a preference, not a rule; `run_workflow` still takes an
`inline_workflow` for a request nothing on disk covers.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_workflows()` | — | List stored workflows, each with its description, output kinds, step count, variable names and the stored prompts it references. The first call to make for a request an existing workflow might cover |
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

### Media

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `get_output_image(name, max_dimension=768)` | `name`, `max_dimension` | Look at a generated image, downscaled to `max_dimension` on its longest side. Returns the image plus a text part reporting `original_size`, `returned_size` and `bytes`, so a downscale is never silent |
| `get_output_text(name, max_characters=20000)` | `name`, `max_characters` | Read a text output — a prompt enhancement, or any step whose result is `text/plain` or JSON. Reports the file's real length and whether it was truncated |
| `delete_output(name)` | `name` | Permanently remove one generated file from the output directory |

### Authoring

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `validate_workflow(workflow=None, name=None)` | exactly one of `workflow` (inline definition) or `name` (a stored workflow, as `list_workflows` reports it) | Check a workflow against the schema and against real pipeline signatures. Free and instant. Validating by name uses the workflow file's own directory as the base directory, so it sees what a run would |
| `save_workflow(name, workflow)` | `name`, `workflow` | Save a workflow to the server, overwriting any existing workflow of that name |
| `delete_workflow(name)` | `name` | Permanently delete a stored workflow |

### Prompts

The stored prompt library is the other half of authoring: a workflow
argument written as `"prompt:name"` or `"prompt:folder/name"` resolves
against it at load time, so a workflow can be authored and the text it
references written in the same session.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_prompts()` | — | List the stored prompts with their text and descriptions |
| `get_prompt(name)` | `name` | Get one stored prompt's full definition |
| `get_prompt_schema()` | — | Get the JSON schema every stored prompt must satisfy. Its own route rather than a name under `/api/prompts`, so a prompt called `schema` cannot shadow it |
| `save_prompt(name, prompt)` | `name`, `prompt` | Save a prompt, overwriting any prompt of that name. The server validates first, and refuses a `text` that itself begins with `variable:`, `previous_result:`, `constant:` or `prompt:` |
| `delete_prompt(name)` | `name` | Permanently delete a stored prompt. A workflow still referencing it will fail to load |
| `list_enhancers()` | — | List the enhancer presets `enhance_prompt` accepts |
| `enhance_prompt(idea, preset="h3", model_name=None, device=None, acknowledged_cost=False)` | `idea`, `preset`, optional `model_name` and `device`, `acknowledged_cost` | Expand a short idea into a full prompt with a language model. Queued as an ordinary job, so it passes the gate; the enhanced text is the text file in the finished manifest, readable with `get_output_text` |

### Diagnose

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `run_workflow(workflow_path=None, inline_workflow=None, arguments=None, acknowledged_cost=False)` | exactly one of `workflow_path` (a catalog name from `list_workflows`, with or without `.json`, or a path to a workflow file on the server) or `inline_workflow`, optional `arguments`, `acknowledged_cost` | Queue a workflow for generation. Returns as soon as the job is queued |
| `get_job(job_id)` | `job_id` | Get a job's status, warnings, output manifest, error and traceback |
| `get_job_events(job_id, after=-1, limit=200)` | `job_id`, `after`, `limit` | Get a page of a job's progress events |
| `wait_for_job(job_id, timeout_seconds=20)` | `job_id`, `timeout_seconds` | Block until a job reaches a terminal status, or `timeout_seconds` elapses (capped well under a generation's real runtime). Use instead of hand-polling `get_job`/`get_job_events` in a loop; if it returns `still_running: true`, call it again |
| `cancel_job(job_id)` | `job_id` | Ask a queued or running job to stop |
| `rerun_job(job_id, acknowledged_cost=False)` | `job_id`, `acknowledged_cost` | Queue a fresh job from a previous job's stored specification. Costs GPU time, so it passes the same gate as `run_workflow` |
| `move_job(job_id, direction)` | `job_id`, `direction` (`up`\|`down`\|`front`\|`back`) | Reorder a queued job |

### Models

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_models()` | — | (Catalog) List what the Hugging Face cache holds, largest first |
| `download_model(repo_id, acknowledged_cost=False)` | `repo_id`, `acknowledged_cost` | Fetch a model repo into the cache. Costs disk and bandwidth, so it passes the gate. Returns as soon as the download starts |
| `list_downloads()` | — | List downloads the server is running or recently ran |
| `cancel_download(download_id)` | `download_id` | Stop a running download. Partial files stay cached and resume on a retry |
| `delete_model(repo, acknowledged_cost=False)` | `repo`, `acknowledged_cost` | Delete every cached revision of one repo. Not recoverable locally |
| `get_diffusers_state()` | — | Installed diffusers version, its git commit, and any update in flight |
| `update_diffusers(acknowledged_cost=False)` | `acknowledged_cost` | Upgrade diffusers to GitHub HEAD in the background |

The server refuses `delete_model` and `update_diffusers` while a job is
running or queued, and `delete_model` while a download is active — pulling
files or package contents out from under a loaded pipeline is the same
hazard twice. That refusal arrives as the server's own explanation.

## The cost gate

Six tools refuse unless `acknowledged_cost=true` is passed. Each commits
the machine to something the user would want to have been asked about first,
and each says so in its own words — a single shared refusal would be wrong
for each of them in a different way, and a gate the user learns to wave
through is not a gate.

| Tool | What it commits |
| --- | --- |
| `run_workflow` | Minutes of GPU time; the engine runs one job at a time |
| `rerun_job` | The same run, from a stored spec |
| `download_model` | Tens of gigabytes of network and disk |
| `delete_model` | Cached weights, unrecoverably — getting them back means downloading again |
| `update_diffusers` | Replacing the installed library with an untagged development build |
| `enhance_prompt` | A real job on the one-at-a-time engine, delaying any generation behind it |

`rerun_job` is gated for the same reason as `run_workflow`: it queues the
identical work, so leaving it open would make the gate worth nothing — any
job id from `list_jobs` would buy a way around it. `cancel_job` and
`cancel_download` are deliberately *not* gated: they end a cost rather than
starting one, and gating them would make the safe direction the harder one.

Passing the flag does not make a tool wait. Each returns as soon as the work
is queued or started, the same way queuing a job from the web UI does not
block the browser tab.

The intended loop:

1. `validate_workflow` — free, checks schema and pipeline signatures, no GPU
   time spent
2. `run_workflow` with `acknowledged_cost=true` — pass a name straight from
   `list_workflows` as `workflow_path`; queues the job and returns
   immediately with a `job_id`
3. `wait_for_job(job_id)` to block for a bounded interval instead of
   hand-polling — call it again if it comes back `still_running: true` — or
   `get_job_events(job_id)` repeatedly, passing back the previous call's
   `last_seq` as `after`, for incremental progress instead of just a
   terminal/not-terminal status
4. `get_job(job_id)` for the finished manifest (or the error and traceback,
   if it failed)
5. `get_output_image(name)` to look at a result image

## Security

The MCP server adds no authentication of its own — it inherits the REST
API's posture exactly, described in full in [Security](SECURITY.md):
localhost binding, no auth, `Origin` header checks, and path confinement in
`dw/security.py` for every workflow, gallery and prompt path a tool touches.
Nothing under `dw_mcp/` re-implements or loosens that confinement; it is
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
- **Not exposed yet:** the prompt library (`prompt:` references still work
  inside a workflow's own JSON — there is just no tool to browse or edit
  stored prompts).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Failed to reconnect to dw: ENOENT` (or the client cannot start the server) | The client cannot find the command. Register the venv's **absolute** path to `dw-mcp`, not the bare name - see [Use the absolute path](#use-the-absolute-path-to-dw-mcp). A bare name works only when the client was launched from an activated venv, so this often appears in a second terminal after the first one worked |
| The server shows connected, but every tool fails | "Connected" means the `dw-mcp` process launched, not that the engine is reachable. Check `dw.serve` is running |
| "Cannot reach diffusers-workflow at …" | `dw.serve` is not running. Start it with `dw-serve` (or `python -m dw.serve`) and try again |
| A config change seems to have no effect | A running session holds the old config. Start a new session |
| It worked, then broke after rebuilding the venv | Re-run `pip install -e ".[server,mcp]"`. If the repo moved or was renamed, re-register the server with the new absolute path |
| A tool call times out | Usually a model loading into VRAM/RAM for the first time; retry, or raise `--timeout` |
| `run_workflow` or `rerun_job` refuses with a cost message | Not an error — it is the `acknowledged_cost` gate. Confirm with the user and call again with `acknowledged_cost=true` |
