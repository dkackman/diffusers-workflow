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
| `--token` | `$DW_API_TOKEN`, else none | Bearer token, when `dw.serve` was started with `--token` / `DW_API_TOKEN` - the same variable, so one export configures both ends |
| `--workspace` | `$DW_MCP_WORKSPACE`, else the server's default | Which of the server's workspaces the session works in. A *name* on the server, not a directory here - `DW_WORKSPACE` means something else to the engine. `use_workspace` switches it mid-session |
| `--timeout` | `30` | Seconds to wait on any one API request |
| `--no-probe` | off | Skip the startup `GET /api/health` that confirms the server is reachable and the token is accepted |

The `DW_MCP_URL` environment variable sets the same default the `--url` flag
overrides.

A non-loopback `--url` requires a token: `dw-mcp` exits 2 rather than start
without one. At startup it makes one `GET /api/health`, so a wrong URL or
token is reported once with a message instead of as a 401 on every tool
call; that probe is fatal for a remote URL and only a warning for a
loopback one (where it usually means `dw.serve` is not up yet).
[REMOTE.md](REMOTE.md) covers the remote setup end to end.

Claude Code users can add the composition skills as well:
`/plugin marketplace add dkackman/diffusers-workflow` then
`/plugin install dw@diffusers-workflow`. The plugin ships one skill per model
family (MiniMax H3, MiniMax Music 3, LTX-2.5) that picks a template for a
request's shape and states the family's rules - see
[plugins/dw/README.md](../plugins/dw/README.md), which also gives the
optional `npx skills add` lines for MiniMax's own prompt skills. It is
optional; every tool below works without it.

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

### Remote server, no local install

If `dw.serve` runs on another machine with `--mcp` (see
[REMOTE.md](REMOTE.md)), Claude Code connects to it directly:

    claude mcp add --transport http dw http://<box>:8765/mcp \
      --header "Authorization: Bearer <token>"

The same token fetches generated files: see step 7 of `The loop` in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#the-loop).

Nothing from this repository is installed on the client. The stdio setup
below is for a machine that has its own `dw` install, and also works
against a remote `--url` with `--token`.

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

52 tools in six groups. Names and arguments below are transcribed from
`dw_mcp/server.py` — nothing here is renamed or reshaped for the docs.

### Catalog (read-only)

The catalog is large, so the server's instructions point a client at
`list_workflows` first: its listing carries enough about each workflow - a
one-line `summary`, its `shape` and `traits`, its measured `cost`, output
kinds and variable names - to pick one and know what to pass it, without
fetching every candidate's definition. Reusing a stored workflow is a
preference, not a rule; `run_workflow` still takes an `inline_workflow` for
a request nothing on disk covers.

A request usually names a *subject* ("a lego movie trailer set in the marvel
universe") while the catalog is written in *shapes* - a single image, an image
set, one shot, a multi-shot cut sequence, video with speech. Nothing in a
catalog entry will match the subject, so the shape is what has to be decided
first and matched against. `list_guides` indexes the engine's documentation by
section for exactly that, and `list_tasks` is what a shape gets composed from
when no single workflow covers it.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_guides()` | — | List the documentation the engine serves: each guide's name, what it covers, and its section headings. The index is the routing table - match a request's shape against a heading rather than guessing |
| `get_guide(name, section=None)` | `name`, `section` | Get one guide whole, or one section of it. Prefer a section: a guide runs to thousands of lines. Section names match loosely, so a heading copied approximately still resolves |
| `list_workflows(shape=None, traits=None, configures=None, include_models=False)` | `shape`, `traits`, `configures`, `include_models` | List stored workflows. Always the server's compact view: each entry carries `summary`, `shape`, `traits`, `cost`, `kinds`, `variable_names`, and `configures` only when set - `get_workflow` has the full description and definition. `shape` keeps one of `image`, `image-set`, `image-edit`, `shot`, `sequence`, `audio`, `text`, `utility`; `traits` is comma-separated and every one listed must match (`has-audio`, `chained`, `image-conditioned`, `identity-referenced`, `needs-input-media`, `composes-workflows`); an unknown value in either is a 400 listing the vocabulary. Templates only by default - `configures=<template>` lists the checkpoint configs tuned for one, `include_models=true` lists them all. The first call to make for a request an existing workflow might cover |
| `get_workflow(name, variables_only=False)` | `name` | Get one stored workflow's full JSON definition. `variables_only=true` answers with just its variables and their defaults (long strings cut to 200 characters, the cut ones named in `truncated`) — the cheap way to confirm what a variable defaults to |
| `get_schema()` | — | Get the JSON schema every workflow definition must satisfy |
| `list_pipelines()` | — | List every diffusers pipeline class this installation provides |
| `get_pipeline_signature(name)` | `name` | Get a pipeline's real call arguments |
| `list_classes(kind)` | `kind` | List class names of one kind: pipelines, models, schedulers, or quantization |
| `get_class(name, target="init")` | `name`, `target` (`init`\|`call`\|`load`) | Get a class's argument schema from the entry point a workflow reaches it by: `init` the constructor (quantization configs, schedulers), `call` a pipeline's `__call__`, `load` `from_pretrained` plus the curated loading knobs |
| `list_tasks()` | — | List every task command a workflow's task step can name |
| `get_task(command)` | `command` | Get a task command's argument schema |
| `list_models()` | — | List what the Hugging Face model cache holds, largest first |
| `get_memory()` | — | Get the worker's VRAM and RAM statistics |
| `get_health()` | — | Check that the server is alive, and which machine answered: `version`, `device`, whether the worker process is up, the job running now and the queue depth |
| `get_server_info()` | — | What this installation can do and where it keeps things: `device` (the accelerator a run will use), `version`, the `workspace` this session is working in and the workflow/asset/output/prompt `directories` of *that* workspace, the bind address and port, whether a token is required, and whether MCP is mounted. Check the device before authoring - a CUDA-only choice (bitsandbytes, `torch.compile`, flash attention) is not available on an `mps` or `cpu` server |
| `list_jobs(limit=20, status=None, workspace=None)` | optional `limit` (newest N), `status` (one state or a comma-separated set of `queued`, `running`, `succeeded`, `failed`, `cancelled`), `workspace` | List queued, running and recent jobs, **newest first**. Bounded by default: the unbounded listing was over a client's tool-result limit on a server with a few months of history, which made it a tool that could not be called at all. `total` says how many matched and `truncated`/`next` say so when the answer was cut - raise `limit` or narrow with `status`. Without `workspace`, a named workspace lists its own jobs and the default one lists every job the server holds |
| `list_gallery(limit=50)` | `limit` | List generated output files, newest first. A name is `<workflow>/<run id>/<file>`; each entry also carries a ready-made `url`, already scoped to the workspace that made it - a hand-built `/outputs/<name>` URL 404s for anything but the default workspace |
| `get_gallery_metadata(name, envelope=False)` | `name` | Get the metadata embedded in a generated file: the exact workflow and arguments that produced it, and, for audio/video, a `media` block (duration, rate, channels, fps, size, peak/mean dBFS). `envelope=true` adds `media.envelope` — `rms_dbfs` and `peak_dbfs` one entry per second — which is what locates something in a track rather than measuring the whole of it |

### Media

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `get_output_image(name, max_dimension=768)` | `name`, `max_dimension` | Look at a generated image, downscaled to `max_dimension` on its longest side. Returns the image plus a text part reporting `original_size`, `returned_size` and `bytes`, so a downscale is never silent |
| `get_output_text(name, max_characters=20000)` | `name`, `max_characters` | Read a text output — a prompt enhancement, or any step whose result is `text/plain` or JSON. Reports the file's real length and whether it was truncated |
| `download_output(name, destination=None, overwrite=False)` | `name`, `destination`, `overwrite` | Save one output file to local disk, of any content type. `destination` may be a full path, a directory, or omitted to save under the output's own name in the current working directory; `~` expands and missing parent directories are created. `overwrite=True` is required to replace a file already at the resolved path. Returns nothing to the conversation but where the file landed — unlike the other media tools, the point is a file on disk, not a payload in context. Writes on the machine running the MCP server - over `dw.serve --mcp` that is the GPU box. A write that fails there (a path that exists only on the client, for instance) comes back as an error naming the server-side write and the client-side alternatives, not as an anonymous tool failure |
| `delete_output(name)` | `name` | Permanently remove one generated file from the output directory |

### Authoring, assets and workspaces

Authoring happens inside one workspace. A server can hold several - each with
its own `workflows/`, `assets/` and `outputs/`, all sharing one prompt library
- and `use_workspace` picks the one this session reads and writes for the rest
of its life. That is how two agents work against one GPU without saving over
each other; see [Workspaces](WORKSPACES.md#several-workspaces-on-one-server).
The session starts in `default` and stays there unless it is told otherwise.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `validate_workflow(workflow=None, name=None, workspace=None, arguments=None)` | exactly one of `workflow` (inline definition) or `name` (a stored workflow, as `list_workflows` reports it), optional `workspace`, optional `arguments` | Check a workflow against the schema and against real pipeline signatures. Free and instant. Validating by name uses the workflow file's own directory as the base directory, so it sees what a run would. Returns every schema violation in `errors`, each with the JSON path it sits at, so a draft is fixed in one pass, and a `previous_result:` that names no earlier step is one of them. `workspace` names the workspace for this one call without switching the session to it - use it to pin a job whose `output:` or `asset:` references live in a workspace other than the session's. Pass the same `arguments` you will pass to `run_workflow` and they are checked too - an undeclared or renamed variable name, a value that will not coerce to the declared type, and an `asset:`, `prompt:` or `output:` reference that names nothing this workspace can reach, each reported at `arguments.<name>`. `checked_arguments` lists what was covered, so a `valid: true` about the stored defaults cannot be mistaken for one about your values. `run_workflow` makes the same check and refuses a bad argument rather than queuing a job that fails on its first step |
| `list_workspaces()` | — | The server's workspaces and which one this session is using. Each has its own workflows, assets and outputs; the prompt library is shared by all of them |
| `use_workspace(name)` | `name` | Work in that workspace for the rest of the session - every later call reads and writes there. This is how to keep your work out of another agent's namespace rather than sharing the default one. Checked against the server, so a typo fails here rather than scoping every later call to nothing |
| `create_workspace(name, use=False)` | `name`, `use` | Create a workspace. Pass use=true to switch this session to it as well; otherwise the session stays where it was and the result says so |
| `delete_workspace(name, acknowledged_cost=False)` | `name`, `acknowledged_cost` | Permanently delete a workspace and everything in it. Refuses without the acknowledgement, reporting what it would remove |
| `list_assets()` | — | The input media on the server, each with the `asset:` reference a workflow argument carries. Look here before asking for a file - what a workflow needs may already be there |
| `keep_output(name, asset_name=None, overwrite=False, shared=False)` | `name`, optional `asset_name`, `overwrite`, `shared` | Keep a generated file as an input asset under a stable `asset:` name, so a later workflow can rely on it. The copy happens on the server: nothing is downloaded or re-uploaded. `asset_name` may name a folder and takes the kept file's extension when it has none; `shared=true` keeps it in the library every workspace shares, which is where a recurring cast belongs |
| `upload_asset(file_path, asset_name=None, shared=False)` | `file_path` | Push a local image, video or audio file into the server's asset library and get back its `asset:` reference. The file is read from the machine the MCP server runs on, so this is how an input reaches a dw.serve running somewhere else. `asset_name` stores it under a readable name (`cast/priya-voice.wav`) instead of a random one; `shared=true` puts it in the library every workspace shares |
| `delete_asset(name)` | `name` | Permanently remove one file from the asset library, by the name `list_assets` reports. Deletes from whichever library holds it - this workspace's own before the shared one; one from a read-only examples library is refused. Any workflow still carrying that `asset:` reference stops loading |
| `save_workflow(name, workflow)` | `name`, `workflow` | Save a workflow into the server's writable workflow directory, overwriting any existing workflow of that name there. A name that currently resolves to a read-only source (an examples directory) is not overwritten - the copy lands in the writable directory and shadows it |
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
| `save_prompt(name, prompt)` | `name`, `prompt` | Save a prompt, overwriting any prompt of that name. The server validates first, and refuses a `text` that itself begins with a reference prefix (`variable:`, `previous_result:`, `constant:`, `asset:`, `output:`, `prompt:`) |
| `delete_prompt(name)` | `name` | Permanently delete a stored prompt. A workflow still referencing it will fail to load |
| `list_enhancers()` | — | List the enhancer presets `enhance_prompt` accepts |
| `enhance_prompt(idea, preset="h3", model_name=None, device=None, acknowledged_cost=False)` | `idea`, `preset`, optional `model_name` and `device`, `acknowledged_cost` | Expand a short idea into a full prompt with a language model. Queued as an ordinary job, so it passes the gate; the enhanced text is the text file in the finished manifest, readable with `get_output_text` |

### Diagnose

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `run_workflow(workflow_path=None, inline_workflow=None, arguments=None, acknowledged_cost=False, workspace=None)` | exactly one of `workflow_path` (a catalog name from `list_workflows`, with or without `.json`, or a path to a workflow file on the server) or `inline_workflow`, optional `arguments`, `acknowledged_cost`, `workspace` | Queue a workflow for generation. Returns as soon as the job is queued. `workspace` names the workspace for this one call without switching the session to it - use it to pin a job whose `output:` or `asset:` references live in a workspace other than the session's |
| `get_job(job_id)` | `job_id` | Get a job's status, warnings, output manifest, error and traceback. A running job also carries `progress` (below) |
| `get_job_workflow(job_id)` | `job_id` | The workflow the job actually ran. `realized: true` means every mutable input is pinned (arguments, seed, prompts, `output:latest`); `false` means the job predates run tracking and this is the definition as submitted. Pass it to `save_workflow` to keep it under a name |
| `export_job(job_id, overwrite=False)` | `job_id`, `overwrite` | Gather one finished job into `<workspace>/exports/<job id>/` on the server: the realized workflow, the run's manifest, the job row, a README, and copies of the assets, earlier-run inputs and outputs. Returns the directory, a zip URL, the file list with sizes and the total. The three JSON files are in the zip, not repeated here - get_job_workflow and get_job serve them individually. **The directory is on the machine running the server**, like `download_output`'s destination - fetch the zip URL and unpack it into `exports/` under the session's working directory (a deliverable, not a temp file); the archive already unpacks into one folder named after the job id |
| `get_job_events(job_id, after=-1, limit=200)` | `job_id`, `after`, `limit` | Get a page of a job's progress events |
| `wait_for_job(job_id, timeout_seconds=20)` | `job_id`, `timeout_seconds` | Block until a job reaches a terminal status, or `timeout_seconds` elapses. **One call blocks for at most 55 seconds** — a larger `timeout_seconds` is clamped, not honoured, because no MCP client holds a tool call open for a generation's real runtime, so budget one call per ~55s of the job. Every reply carries `waited_seconds`, `timeout_requested_seconds`, `timeout_applied_seconds` and `timeout_capped`, so a capped return is distinguishable from an elapsed one. Use instead of hand-polling `get_job`/`get_job_events` in a loop; if it returns `still_running: true`, call it again. Returns a slim job - status, warnings, error, and the manifest once finished - without the arguments; `get_job` has those. A running job also carries `progress` (below) |
| `cancel_job(job_id)` | `job_id` | Ask a queued or running job to stop |
| `rerun_job(job_id, acknowledged_cost=False, new_seed=False)` | `job_id`, `acknowledged_cost`, `new_seed` | Queue a fresh job from a previous job's stored specification. Costs GPU time, so it passes the same gate as `run_workflow`. `new_seed=true` draws a fresh seed into the workflow's seed variable — without it a seeded workflow's rerun repeats its arguments exactly and the step cache serves the whole run from the earlier one's files (`reused: true`), generating nothing. `get_job_workflow`'s `seed_variable` says whether there is one |
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

Seven tools refuse unless `acknowledged_cost=true` is passed. Each commits
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
| `delete_workspace` | Every workflow, asset and generated file in a workspace, unrecoverably |

`rerun_job` is gated for the same reason as `run_workflow`: it queues the
identical work, so leaving it open would make the gate worth nothing — any
job id from `list_jobs` would buy a way around it. `cancel_job` and
`cancel_download` are deliberately *not* gated: they end a cost rather than
starting one, and gating them would make the safe direction the harder one.

Passing the flag does not make a tool wait. The five that start work return
as soon as it is queued or started, the same way queuing a job from the web UI
does not block the browser tab; `delete_model` and `delete_workspace` are
deletions rather than queued work and complete before they answer.

The intended loop:

1. `validate_workflow` — free, checks schema and pipeline signatures, no GPU
   time spent. Pass the `arguments` you intend to run with: without them the
   verdict covers the stored definition and its stock defaults, not the
   values you wrote
2. `run_workflow` with `acknowledged_cost=true` — pass a name straight from
   `list_workflows` as `workflow_path`; queues the job and returns
   immediately with a `job_id`
3. `wait_for_job(job_id)` to block for a bounded interval instead of
   hand-polling — one call covers at most 55 seconds however large
   `timeout_seconds` is, so a minutes-long render takes several; call it
   again if it comes back `still_running: true` — or
   `get_job_events(job_id)` repeatedly, passing back the previous call's
   `last_seq` as `after`, for incremental progress instead of just a
   terminal/not-terminal status. Each event carries `at`, seconds since the
   job started, so where a step's time went is a subtraction between two
   events - `step_start` to `generating` is the lead-in a reused pipeline
   still pays, `generating` to the first `pipeline_step` the encoding
4. `get_job(job_id)` for the finished manifest (or the error and traceback,
   if it failed)
5. `get_output_image(name)` to look at a result image

While a job runs, `get_job` and `wait_for_job` carry a `progress` block -
the step being run, the phase (`loading`, `generating`, `decoding`,
`saving`) with the model in `phase_detail`, `seconds_in_phase`,
`seconds_since_event`, and `denoise_step`/`denoise_total_steps`, which are
null until the denoise loop starts. A single-step generation is minutes of
one phase, so two polls otherwise come back identical: read `denoise_step`
moving (slow but healthy) against a `denoise_step` that is a number and
stays put while `seconds_since_event` climbs (nothing is happening). A null
`denoise_step` under `generating` is neither - it is the lead-in the
pipeline runs before the loop, encoding the prompt and any reference image
or audio, ~90 s on MiniMax H3 with nothing emitted, so silence there is
expected. `cancel_job` stops at the next denoise or
step boundary, which `denoise_step` is also the measure of.

## Security

The MCP server adds no authentication of its own — it inherits the REST
API's posture exactly, described in full in [Security](SECURITY.md):
localhost binding, no auth, `Origin` header checks, and path confinement in
`dw/security.py` for every workflow, gallery and prompt path a tool touches.
Nothing under `dw_mcp/` re-implements or loosens that confinement; it is
purely a client of the same validated endpoints the web UI uses - except for
`download_output`, the one tool that writes a local file for the MCP client
rather than only reading through the API. It may write anywhere the
client's own filesystem lets it (a full path, a directory, or the current
working directory by default, `~` expanded), the way a shell redirect
would for the same user; a `..` path segment in `destination` is refused,
and an existing file is left alone unless the caller passes
`overwrite=True`.

`dw-mcp` may be pointed at a `dw.serve` on another machine only when that
server was started with a token, and the same token is passed here
(`--token` / `DW_API_TOKEN`); it refuses to start otherwise. The token is
the only authentication, and the connection is plaintext HTTP - use it on
a network you control, or through Tailscale or a TLS proxy beyond that.
[REMOTE.md](REMOTE.md) has the full setup. The same applies to
`dw.serve --mcp`, which serves this tool surface itself at `/mcp` behind
the same token; in that setup `download_output` writes on the server
machine, not the client's.

`save_workflow` and `run_workflow` together let an MCP client write and
then execute a workflow it authored - and a workflow JSON file can execute
arbitrary Python (see [Trust model](SECURITY.md#trust-model)). What
protects a `dw.serve` an MCP client talks to is the server's own
`--trust-workflows` flag, off by default: run `dw-serve` without it (the
default) for any server an MCP client can reach.

## Known limits

- **Event history is bounded.** `get_job_events` serves at most the last 200
  events of a finished job (`MAX_PERSISTED_EVENTS` in the job history store).
  A job that ran before this feature existed returns an empty event list
  with a `note` explaining why.
- **Images only.** `get_output_image` decodes and returns images; it refuses
  video and audio outputs. Use `get_gallery_metadata` to inspect other media
  kinds.
- **Uploads read the MCP server's disk.** `upload_asset(file_path)` pushes a
  local file into the asset library, but "local" means the machine `dw-mcp`
  runs on. Over `dw.serve --mcp` that is the GPU box, so a file sitting on
  the client's laptop is not reachable that way - put it on the server, or
  give the workflow a URL (the arguments that take a path take a URL too).
  `download_output` has the same asymmetry in the other direction: on a
  `--mcp` endpoint it writes on the GPU box, not the client's machine.
- **Prompts are not per-workspace.** Switching workspaces changes which
  workflows, assets and outputs the session sees; the prompt library is one
  library shared by all of them, because `prompt:` is shared by reference.

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
