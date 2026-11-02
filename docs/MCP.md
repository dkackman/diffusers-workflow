# MCP Server

A stdio [MCP](https://modelcontextprotocol.io) server that lets an MCP client
(Claude Code first) author, validate, save, run and diagnose workflows without
shell access or a repo checkout. It is the third way to drive the engine,
beside `dw.run` and `dw.serve`.

`dw_mcp/` is an HTTP client of a **running** `dw.serve`. It holds no job state
and no GPU worker; every tool call is a REST request against the server
described in [Server & Web UI](SERVER.md). If `dw.serve` is not running, every
tool fails with a message telling you to start it.

## Install and run

```bash
pip install -e ".[server,mcp]"
dw-serve          # the engine; leave it running
```

You do not start `dw-mcp` yourself: the MCP client launches it on demand, so
the client needs a command it can find (see below). `dw-mcp` (or
`python -m dw_mcp`) speaks MCP over stdio. Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `$DW_MCP_URL`, else `http://127.0.0.1:8765` | Base URL of the running `dw.serve` |
| `--token` | `$DW_API_TOKEN`, else none | Bearer token, when `dw.serve` was started with `--token` / `DW_API_TOKEN`. The same variable configures both ends |
| `--workspace` | `$DW_MCP_WORKSPACE`, else the server's default | Which of the server's workspaces the session works in. A *name* on the server, not a local directory (`DW_WORKSPACE` means something else to the engine). `use_workspace` switches it mid-session |
| `--timeout` | `30` | Seconds to wait on any one API request |
| `--no-probe` | off | Skip the startup `GET /api/health` that confirms the server is reachable and the token is accepted |

A non-loopback `--url` with no token makes `dw-mcp` exit 2. The startup probe
reports a wrong URL or token once, instead of as a 401 on every tool call; it
is fatal for a remote URL and only a warning for a loopback one, where it
usually means `dw.serve` is not up yet. [REMOTE.md](REMOTE.md) covers the
remote setup end to end.

Claude Code users can add the composition skills with
`/plugin marketplace add dkackman/diffusers-workflow` then
`/plugin install dw@diffusers-workflow`: one skill per model family (MiniMax
H3, MiniMax Music 3, LTX-2.5, Kandinsky 6, FLUX 3 Klein) plus cross-cutting
skills for scripts, series, cinematography and storyboards. See
[plugins/dw/README.md](../plugins/dw/README.md), which also gives the optional
`npx skills add` lines for MiniMax's own prompt skills. Every tool works
without the plugin.

### Use the absolute path to `dw-mcp`

**This is the setup detail that most often goes wrong.** Installed the way
`install.sh` does it, `dw-mcp` lives in the project's virtualenv and is on
`PATH` only while that venv is active. An MCP client is usually launched
without the venv, so a bare `dw-mcp` fails to spawn:

```
Failed to reconnect to dw: ENOENT
```

Registering from an activated terminal hides the problem until a session
starts somewhere else. Register the venv's absolute path instead; a console
script has its interpreter in its shebang, so it runs with no venv active:

```bash
echo "$(pwd)/venv/bin/dw-mcp"    # the value to register
```

## Client configuration

### Remote server, no local install

If `dw.serve` runs on another machine with `--mcp` (see [REMOTE.md](REMOTE.md)),
Claude Code connects to it directly and nothing from this repository is
installed on the client:

    claude mcp add --transport http dw http://<box>:8765/mcp \
      --header "Authorization: Bearer <token>"

Generated files under `/outputs`, `/inputs` and `/exports` need no token
(#592); step 7 of [The loop](WORKFLOW_GUIDE.md#the-loop) fetches them. The
stdio setup below is for a machine with its own `dw` install, and also works
against a remote `--url` with `--token`.

### Claude Code

From the project directory:

```bash
claude mcp add dw -- "$(pwd)/venv/bin/dw-mcp"
claude mcp add dw -- "$(pwd)/venv/bin/dw-mcp" --url http://127.0.0.1:8791   # subprocess flags go after --
```

Pick the scope with `-s`:

| Scope | Stored in | Use when |
| --- | --- | --- |
| `local` (default) | `~/.claude.json`, keyed to this project | Just you, just this checkout |
| `user` | `~/.claude.json`, global | Every project. The absolute path makes this work |
| `project` | `.mcp.json`, **committed to the repo** | Every clone gets it. An absolute path is machine-specific and will not port |

The same registration as a hand-written `.mcp.json`:

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

**A running session does not pick up a registration change**; start a new one.

### Claude Desktop

`claude_desktop_config.json` takes the same shape. The absolute path matters
more here, because a desktop app never inherits a shell's `PATH`:

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

In either config, an `"env"` object beside `"command"`/`"args"` can set
`DW_MCP_URL` instead of `--url`.

## Verify the setup

Run the three checks in order; each isolates a different failure. None costs
GPU time.

**1. The command launches without a venv.** This reproduces the environment
the client spawns it in, and catches ENOENT:

```bash
env -i PATH=/usr/bin:/bin HOME="$HOME" /path/to/venv/bin/dw-mcp --help
```

It prints usage and exits 0. If not, the path is wrong or the package is not
installed into that venv.

**2. The client sees the server.** In a new Claude Code session, `/mcp` lists
`dw` as connected (`claude mcp list` and `claude mcp get dw` from the shell).
"Connected" means only that the process launched, not that `dw.serve` is
reachable.

**3. The tools reach the engine.** With `dw-serve` running, ask for something
free, such as "list my diffusers workflows" (`list_workflows`) or "check the
diffusers-workflow server health" (`get_health`). "Cannot reach
diffusers-workflow at ..." means the engine is not running.

## Tool reference

65 tools in six groups. Names, arguments and defaults are those registered in
`dw_mcp/tools_*.py` (wired in `dw_mcp/server.py`); each tool's own description
is the full contract. Every `workspace` argument pins that one call to a
workspace without moving the session's pin (see
[Authoring, assets and workspaces](#authoring-assets-and-workspaces)).

### Catalog (read-only)

The server's instructions send a client to `list_workflows` first: its
listing carries enough per workflow to choose one and know what to pass it.
A request usually names a *subject* while the catalog is written in *shapes*
(one image, an image set, one shot, a cut sequence), so decide the shape
first. `list_guides` indexes the documentation by section, and `list_tasks`
is what a shape is composed from when no stored workflow covers it;
`run_workflow` still takes an `inline_workflow` for that case.

The engine runs **one job per GPU**: one worker per `dw.serve --devices` card,
each taking jobs from one shared queue. A job record's `device` (`get_job`,
`list_jobs`) names the card it ran on, and a job started by `rerun_job`
carries `rerun_of`.

| Tool | Purpose |
| --- | --- |
| `list_guides()` | The documentation the engine serves: each guide's name, what it covers, and its section headings. Match a request's shape against a heading |
| `get_guide(name, section=None)` | One section of a guide, matched loosely (a `###` subsection resolves too). Without `section` the answer is the guide's index: its opening, its first section, and `sections`/`withheld` naming the rest. A whole guide can cost more than the tool surface does to connect |
| `list_workflows(shape=None, traits=None, configures=None, include_models=False)` | Stored workflows. With no filter each entry is cut to `summary` and `shape` (`view: "summary"`, with a `note`). Filtered, each entry carries `summary`, `shape`, `traits`, `cost`, `observed_minutes`/`observed_runs` (this box's own finished runs), output kinds, `variable_names`, and when set `lists` (per list variable: entry fields, steps over it, default length; `cost[].per_entry` prices one entry), `constraints` and `configures`. `shape` is one of `image`, `image-set`, `image-edit`, `shot`, `sequence`, `audio`, `text`, `utility`; `traits` is comma-separated and all must match (`has-audio`, `chained`, `image-conditioned`, `identity-referenced`, `needs-input-media`, `composes-workflows`); an unknown value is a 400 listing the vocabulary. Templates only by default; `configures=<template>` lists the checkpoint configs tuned for one, `include_models=true` lists them all. `libraries` names the roots searched (`origin`, `root`, `writable`), `shadowed` the names a nearer root hides |
| `get_workflow(name, variables_only=False)` | One stored workflow's full JSON. `variables_only=true` returns just its variables and defaults, long strings cut to 200 characters and named in `truncated` (`shots[0].prompt` for one inside a list) |
| `get_schema(section=None)` | The workflow JSON schema (~8.6k tokens whole). `section` is one of `steps`, `pipelines`, `tasks`, `result`, `variables`, `configuration` and answers `{section, sections, elsewhere, schema}`; `elsewhere` says which section holds each `$ref` the fragment still has. An unknown section is a 404 naming the valid ones |
| `list_pipelines()` | Every diffusers pipeline class this installation provides |
| `get_pipeline_signature(name)` | A pipeline's real call arguments; for a modular pipeline class (`MiniMaxH3ModularPipeline`), its block graph's inputs |
| `list_classes(kind)` | Class names of one kind: pipelines, models, schedulers or quantization |
| `get_class(name, target="init")` | A class's argument schema at the entry point a workflow reaches it by: `init` the constructor (quantization configs, schedulers), `call` a pipeline's `__call__`, `load` `from_pretrained` plus the curated loading knobs |
| `list_tasks()` | Every task command a task step can name |
| `get_task(command)` | A task command's argument schema |
| `list_models()` | What the Hugging Face cache holds, largest first |
| `get_memory(device=None)` | VRAM and RAM per card in `workers` (the top level repeats the first card; `device`, e.g. `cuda:1`, asks for one). `gpu_*` is the card. `host_memory_rss_mb` / `host_memory_peak_rss_mb` are the worker's resident and peak host memory, beside the machine's `host_memory_total_mb` / `host_memory_available_mb`; on Linux `host_memory_rss_anon_mb` / `host_memory_rss_file_mb` split resident memory into owned pages and mapped checkpoint pages the kernel can drop (#709). `host_pinned_reserved_mb` / `host_pinned_allocated_mb` are torch's pinned-host cache, part of RSS and invisible to `gpu_*` (#98). A host field the platform cannot measure is absent. `live: true` is a reading taken now; `live: false` with `info: null` means nothing is resident; `live: false` with `info` is a cached reading with `reason` (`job_running`, `worker_stopped`, `worker_busy`, `worker_unreachable`) and `age_seconds`. Compare only live readings |
| `get_health()` | Liveness and which machine answered: `version`, `device`, `worker_alive`, queue depth, and `workers` (per card: `device`, `name`, `vram_gb`, `alive`, `current_job`, `host_memory_rss_mb`). The top-level `current_job` is the longest-running job. `worker_alive: false` is the normal idle state before the first job or after a memory clear (#206) |
| `get_server_info(cache_entries=False)` | `device`, `version`, this session's `workspace` and its `directories`, the bind address and port, `auth_required`, `trust_workflows`, the MCP mount, and `runtime` (#222: Python, torch, CUDA build, NVIDIA driver, and the diffusers, transformers, accelerate, bitsandbytes, peft, safetensors and sentencepiece versions, `null` when not installed). Check `device` before authoring: CUDA-only choices (bitsandbytes, `torch.compile`, flash attention) fail on `mps` or `cpu`. `caches` sizes dw's disk caches (`inductor`, `triton`, `quantized`); `cache_entries=true` lists the quantized cache's entries. No tool clears a cache; that is `DELETE /api/caches/{name}` ([SERVER.md](SERVER.md)) |
| `list_jobs(limit=20, status=None, workspace=None)` | Queued, running and recent jobs, **newest first**, with queue position, `device`, `acknowledged` and (when held for host RAM) `waiting_for`. `status` is one state or a comma-separated set of `queued`, `running`, `succeeded`, `failed`, `cancelled`. `total`, `truncated` and `next` show a cut answer. Without `workspace`, a named workspace lists its own jobs and the default lists every job |
| `list_gallery(limit=50, subfolder=None, only_orphans=False, workspace=None, folder=None, version=None, media=False, role=None)` | Generated files, newest first, named `<workflow>/<run id>/<file>` (`<file>` may sit in a subfolder: `final/episode.mp4`). Each entry carries `folder`, `subfolder` (`final`, `intermediate`, `''` or any path the workflow wrote), `role` (`final`, `intermediate`, or null; see `dw/roles.py`), `starred` when kept in the web UI, `run_id`, `version` and a workspace-scoped `url` (a hand-built `/outputs/<name>` 404s outside the default workspace). `version` is the run's ordinal (`v5` in the UI): assigned when the run opens, never renumbered, `null` in the flat layout; `output:<folder>/v5/<file>` names a file by it. `folder` with `version` lists one run. `role=["final"]` lists only deliverables. `only_orphans=True` returns run directories with no media (`runs`, each `{name, mtime}`), each a name `delete_output` accepts (#170) |
| `get_gallery_metadata(name, envelope=False, workspace=None, interval_seconds=None, include_workflow=True)` | What produced a file and what it measures. `name` may be an `asset:` reference, for an input's duration, frame count, fps and sample rate before a run (`source` says which; `job` is null). Only an image embeds `metadata`; for audio and video it is null and `next` points at `get_job_workflow(job_id)`. `include_workflow=false` omits the embedded workflow. `media` holds duration, rates, channels, fps, size and peak/mean dBFS; `findings` lists each level problem (`full_scale`, `near_silent`, thresholds in `dw/audio_qc.py`) with its fix. `envelope=true` adds `media.envelope` (`rms_dbfs`, `peak_dbfs`) per second or per `interval_seconds` (0.05 to 3600; over 2400 bins is refused). `media.pcm_sha256` hashes the decoded samples; two hashes compare only when `sample_format`, `frames`, `sample_rate` and `channels` agree. `media.shots` places a joined output's shots (`{name, start_frame, num_frames, start_sample, num_samples}`) |

### Media

| Tool | Purpose |
| --- | --- |
| `get_output_image(name, max_dimension=768, workspace=None, crop=None)` | A generated image, downscaled to `max_dimension` on its longest side, with a text part reporting `original_size`, `returned_size` and `bytes`. `crop` is `[x, y, width, height]` in the original's pixels, cut before the downscale and reported back clamped |
| `get_output_audio(name, start=None, duration=None, workspace=None)` | A soundtrack (an audio output, or a video's muxed track) as base64: its own encoding when whole, WAV when excerpted. A whole clip over the 4MB budget is refused, not cut (#204); ask for `start`/`duration` in seconds, and the text part names the excerpt (`excerpt: 2.0s from 10.0s of 240.0s`) |
| `get_output_frames(name, at=None, seams=None, count=None, boundaries=None, names=None, max_dimension=512, hear=None, workspace=None, crop=None)` | A video as frames (#193). One selector per call: `count` (an evenly spaced contact sheet), `at` (seconds or `"frame:N"`), or `seams` (true, or seam numbers from 1: the frame pair across each join, with `difference`, the mean pixel change 0-255). `boundaries` and `names` modify `seams` only; passing either with `at` or `count` is refused. On an output joined from shots, `seams` reads the recorded `media.shots`; for any other file pass `boundaries` (each later shot's first frame) and `names`. `crop` is `[x, y, width, height]` in source pixels, cut before any downscale. Over the 4MB budget, tiles shrink together. `hear=N` adds N seconds of soundtrack centred on each `at`; a mute clip says `no soundtrack` |
| `get_output_text(name, max_characters=20000, workspace=None)` | A text output (a prompt enhancement, any `text/plain` or JSON result), with its real length and whether it was truncated |
| `assess_output(name, probe=None, detail=False, workspace=None)` | Measures a finished cut in the server process without queueing: `analyze_shots`, `analyze_seams` and `analyze_sync_drift` over one decode, using the shot boundaries its run recorded (the manifest, or the `keep_output` sidecar for an asset). Returns merged `findings` (`{rule, severity, at, value, threshold, says}`) with `rules_applied`, `rules_skipped` and `not_applicable`; `detail=true` adds each probe's body under `probes`; `probe=` returns one probe's body, and an unknown name is refused first. Findings are places to look, not verdicts. See [Assessing a run's output](WORKFLOW_GUIDE.md#assessing-a-runs-output) |
| `compare_outputs(names, at=None, crop=None, baseline=None, max_dimension=384, texture_bands=False, workspace=None)` | 2-8 outputs (gallery names or `asset:`) as one grid, a row per output and a column per moment, with PSNR (`psnr_db`), SSIM (luma, 11x11 Gaussian, sigma 1.5) and `mean_abs_diff` (0-255) against `baseline` (default the first name), per column and as mean/min/max. `at` is 1-8 moments (seconds or `"frame:N"`; negative is a 400), resolved to one frame index against the baseline and read at that frame in every output; default 4 evenly spaced frames of the shortest video. A frame past an output's end is a blank cell (`past end (N frames)`); `fps_differs` flags mismatched rates. One shared `crop` `[x,y,w,h]` is shown and measured; a list of one box per name is display only. A crop must lie wholly inside every frame (400). A size mismatch gives null metrics, never a resize. The grid is held to 4 MiB (never below 64 px a cell). `texture_bands=true` adds RMS texture energy per band (`fine` 2-8 px, `mid` 8-32, `coarse` 32-128) for `luma`, `chroma_rg`, `chroma_bg`, plus `texture_ratio` against the baseline; it needs one shared crop of at least 16 px a side, or none, and allows a single name. Errors: 400 bad shape, crop out of bounds, every moment past every end, or not an image or video; 404 unknown name; 413 over the decode pixel limit; 422 undecodable |
| `download_output(name, destination=None, overwrite=False, workspace=None)` | Saves one output, any type, to disk on the machine running the MCP server, and returns only where it landed. `destination` is a path or a directory (`~` expands, parents are created); a `..` segment is refused; an existing file needs `overwrite=True`. Over stdio, no `destination` saves under the output's name in the working directory. Over `dw.serve --mcp` the file lands on the server, confined to the workspace, and `destination` is required (#353); use the entry's `url`, the inline tools, or `keep_output` instead |
| `delete_output(name=None, workspace=None, job_id=None)` | Permanently removes one generated file, or a whole run: `name` as `<workflow>/<run id>`, or `job_id` (`DELETE /api/jobs/{id}/run`, which answers `job_id`, `run_dir`, `deleted`, `run_swept`). Exactly one of the two. A job with no run directory, an unknown one, or one still queued or running is an error. A `job_id` delete goes to the workspace the job ran in |

### Authoring, assets and workspaces

A server can hold several workspaces, each with its own `workflows/`,
`assets/` and `outputs/`, all sharing one prompt library and one shared asset
library. `use_workspace` sets the one this session reads and writes; a session
starts in `default`. See
[Workspaces](WORKSPACES.md#several-workspaces-on-one-server).

The pin is state on `client.py`'s `DwClient`. A stdio `dw-mcp` has one per
session, but `dw.serve --mcp` builds one client for every agent it serves, so
on that surface the pin is **shared by every connected caller** (#298): another
client's `use_workspace` or `create_workspace(use=true)` moves your pin
without your session calling either. Both tools add a `warning` to their
result on a mounted server when they overwrite a pin that pointed somewhere
other than `default`; the other session gets no warning. A caller that needs
isolation on a mounted server passes `workspace=` on every call that takes it.

| Tool | Purpose |
| --- | --- |
| `validate_workflow(workflow=None, name=None, inline_workflow=None, workflow_path=None, workspace=None, arguments=None)` | Free and instant: schema, real pipeline signatures, references and arguments. Exactly one of `workflow` (inline, an object or a JSON-encoded string) or `name` (a stored workflow); `run_workflow`'s spellings `inline_workflow` and `workflow_path` are accepted too. By name, the workflow file's own directory is the base directory, as a run sees it. `errors` lists every violation with its JSON path (a `previous_result:` naming no earlier step among them); `warnings` covers what runs but is probably wrong (a signature mismatch, a list entry key no step reads, a value the workflow rounds up). Pass the `arguments` you will run with: an undeclared name, a value that will not coerce, a value outside a declared bound, and an `asset:`/`prompt:`/`output:` reference that reaches nothing are each reported at `arguments.<name>`; `checked_arguments` says what was covered. Also checked: a reference set the model would refuse (too many images, videos or audio clips; for MiniMax-H3, audio as the only reference), an unwritable `result.subfolder` or `file_base_name`, and each sub-workflow step. A valid answer carries `plan`: `fingerprint`, step count, list lengths, `downloads_required`, `estimate` (with `basis`), `host_memory` (`{projected_mb, basis, runs}`) and `elided_steps` (steps nothing reads and that save no file, which do not run). `plan` is null when it could not be built |
| `list_workspaces(detail=False)` | The workspaces and which one this session uses. Entries carry `name`, `default`, `usage` and any `note` (clipped, `note_truncated`); `detail=true` adds each entry's folder paths |
| `use_workspace(name)` | Work in that workspace for the rest of the session. Checked against the server, so a typo fails here. On a mounted server the pin is shared (above) |
| `create_workspace(name, use=False)` | Create a workspace; `use=true` also switches to it. The result names the workspace (`name`, `default`, `current`, `next`), not its folders |
| `copy_to_workspace(destination, runs=None, assets=None, source=None)` | Copy runs (`<workflow>/<run id>`, or `<workflow>` for all its runs) and assets (a path as after `asset:`, or a folder `proj/`, keeping path, note and kept shots) into another workspace. `source` defaults to the session's workspace. No `acknowledged_cost`. Media is hard-linked (a real copy across devices), JSON sidecars are copied. All or nothing: what the destination already has is skipped as `already_present`, a different file at an asset's path refuses the whole request, a `vN` in use is renumbered (`renumbered`). Warnings: `unresolved_output_refs`, `absolute_paths`, `no_jobs`. A copied run has no job record. A run whose job is queued or running is refused. To move, copy then `delete_output` / `delete_asset`. See [Workspaces](WORKSPACES.md#copying-between-workspaces) |
| `delete_workspace(name, acknowledged_cost=False)` | Permanently delete a workspace and everything in it. Without the acknowledgement the server refuses (409), reporting what it would remove |
| `list_assets(detail=False, workspace=None, limit=50, prefix=None)` | Input media on the server, each with its `asset:` reference, newest first. Entries carry `name`, `reference`, `kind`, `size`, `origin` and any `note`; `detail=true` adds `folder`, `mtime`, `url`, `writable`, and `absolute_url`/`used_by` when set. `prefix` keeps names starting with it (`cast/`); `total` counts all matches. `libraries` and `shadowed` are as in `list_workflows` |
| `keep_output(name, asset_name=None, overwrite=False, shared=False, workspace=None, note=None)` | Keep a generated file as an input asset under a stable `asset:` name, copied on the server. `asset_name` may name a folder and takes the file's extension when it has none; `shared=true` keeps it in the shared library, where a recurring cast belongs. `note` defaults to the source file's note |
| `set_note(target, note=None, workspace=None)` | Set the note (plain text, up to 4000 characters) on a run (`<workflow>/<run id>`), an output file, `asset:<name>` or `workspace:<name>`; null or `""` clears it. Returns `{target, note, note_source}`. Listings clip notes to 200 characters (`note_truncated`); `get_job` and `get_gallery_metadata` return them whole, and `get_job` shows the current note |
| `upload_asset(file_path=None, content=None, asset_name=None, shared=False, workspace=None, note=None)` | Put an image, video, audio file or `.cube` LUT into the asset library and get its `asset:` reference. Exactly one of `file_path` (read on the machine running the MCP server, up to 200MB) or `content` (base64 inline, up to 4MB, #203; `asset_name` required). The file is stored under the library's `uploads/`, named `asset_name` when given (`cast/priya-voice.wav`, folders allowed, the extension matching the file's kind) and randomly otherwise; `shared=true` uses the shared library |
| `delete_asset(name, workspace=None)` | Permanently remove one asset, by the name `list_assets` reports, from whichever library holds it (the workspace's own before the shared one). One from a read-only examples library is refused. A workflow still carrying the reference stops loading |
| `save_workflow(name, workflow=None, patch=None, workspace=None)` | Save into the writable workflow directory, overwriting a workflow of that name there; `name` may include folders. Exactly one of `workflow` (the full document) or `patch` (a JSON Merge Patch, RFC 7396, onto the stored definition: `null` deletes a key, a list is replaced whole). Either may be a JSON-encoded string. A name that resolves to a read-only source is not overwritten: the copy lands in the writable directory and shadows it |
| `delete_workflow(name, workspace=None)` | Permanently delete a stored workflow. One from a read-only examples directory (`writable: false`) is refused |

### Prompts

A workflow argument written `"prompt:name"` or `"prompt:folder/name"` resolves
against the stored prompt library at load time, so a workflow and the text it
references can be written in the same session. This group also holds the
enhancers and the LoRA catalog.

| Tool | Purpose |
| --- | --- |
| `list_prompts(tag=None, intended_model=None, include_text=False)` | Stored prompts: `description`, `intended_model`, `tags`, `text_chars`, `origin`, `writable`; `libraries` and `shadowed` as in `list_workflows`. Bodies only with `include_text=true`, which for the whole library is too large; filter first |
| `get_prompt(name)` | One stored prompt's full definition |
| `get_prompt_schema()` | The JSON schema a stored prompt must satisfy, on its own route so a prompt named `schema` cannot shadow it |
| `save_prompt(name, prompt)` | Save a prompt, overwriting one of that name. The server validates it, and refuses a `text` that begins with a reference prefix (`variable:`, `previous_result:`, `constant:`, `asset:`, `output:`, `prompt:`) |
| `delete_prompt(name)` | Permanently delete a stored prompt. A workflow referencing it fails to load |
| `list_enhancers()` | The presets `enhance_prompt` accepts |
| `enhance_prompt(idea, preset="h3", model_name=None, device=None, acknowledged_cost=False)` | Expand a short idea into a full prompt with a language model. Queued as an ordinary job behind the cost gate; the text is the text file in the finished manifest (`get_output_text`) |
| `list_loras(model=None, workflow=None, status=None, tag=None)` | LoRAs tried on a base model (proven, trial or rejected) with `use_when`, `trigger` and `scale`; exact match on the base. Guide: `loras` |
| `save_lora(name, entry)` | Save a catalog entry, overwriting one of that name; promote a trial to `proven` with its job in `evidence` |
| `recommend_loras(model, query, limit=8)` | Catalog LoRAs ranked against a style request, then Hugging Face Hub adapters of that exact base. The one tool that queries the Hub (`open_world_hint`); no download, no GPU. A failed or busy Hub returns the catalog rows plus `hub_error` |

### Diagnose

| Tool | Purpose |
| --- | --- |
| `run_workflow(workflow_path=None, inline_workflow=None, workflow=None, name=None, arguments=None, acknowledged_cost=False, workspace=None, wait_seconds=0, note=None)` | Queue a workflow. Exactly one of `workflow_path` (a catalog name from `list_workflows`, with or without `.json`, or a path on the server) or `inline_workflow` (an object or JSON-encoded string); `validate_workflow`'s spellings `name` and `workflow` are accepted too. `arguments` overrides variables by name and is checked as `validate_workflow` checks it; a bad one is refused before queueing. `acknowledged_cost` is `true` or the bound `{fingerprint, minutes, downloads}` from the validate plan; a 409 means the plan changed and carries the new estimate. Returns once queued, unless `wait_seconds` > 0: then it waits as `wait_for_job(job_id, timeout_seconds=wait_seconds)` would, under the same cap, and adds that wait's fields. `note` says what the run tests (see `set_note`) |
| `get_job(job_id)` | Status, warnings, output manifest, error and traceback. Each manifest entry's `subfolder` is the one the step declared. A running job carries `progress`; a job held for host RAM carries `waiting_for` (#869) |
| `get_job_workflow(job_id)` | The workflow the job ran (`GET /api/jobs/{id}/workflow`, [SERVER.md](SERVER.md#jobs-api)). `realized: true` pins every mutable input (arguments, seed, prompts, `output:latest`); `false` is the submitted definition of a job from before run tracking. `seed_variable` names the variable `rerun_job(new_seed=true)` draws into. Pass it to `save_workflow` to keep it |
| `export_job(job_id, overwrite=False)` | Gather a finished job into `<workspace>/exports/<job id>/` **on the server machine**: the realized workflow, the manifest, the job row, a README, and copies of its assets, earlier-run inputs and outputs (`total_bytes` is what was copied). Returns the directory, a zip URL, the file list and the total. The zip `open_url` needs no token (#592): fetch it (prefixing a relative URL with the server address) and unpack it into `exports/` in the session's working directory; it unpacks into one folder named after the job id. `auth_required: true` would mean a gated zip, to hand to the person. Refuses a running job, and an existing export without `overwrite=true` |
| `get_job_events(job_id, after=-1, limit=200, kinds=None)` | A page of progress events. `after` is exclusive: pass back `last_seq`. Each event's `at` is seconds since the job started. `kinds` (e.g. `["log", "warning"]`) filters the page, which `memory` events otherwise dominate. A `phase_stall` entry is a watchdog notice every ~30 s of a quiet phase, not evidence of a hang |
| `wait_for_job(job_id, timeout_seconds=20)` | Block until the job is terminal or `timeout_seconds` elapses. One call blocks for at most the server's cap (`DW_MCP_MAX_WAIT_SECONDS`; the tool's description states the live value); a larger timeout is clamped, and every reply carries `waited_seconds`, `timeout_requested_seconds`, `timeout_applied_seconds` and `timeout_capped`. On `still_running: true`, call again. A deployment raises the cap only as far as its clients hold a silent HTTP request open (Claude Code: `MCP_TOOL_TIMEOUT`). Returns a slim job: status, warnings, error, `run_id`, `run_version`, the manifest once finished, `progress` while running, `waiting_for` when held |
| `cancel_job(job_id)` | Ask a queued or running job to stop. A running job stops at the next denoise or step boundary |
| `clear_memory(device=None)` | Drop every loaded pipeline and the step cache on each idle card (`device`'s alone when named), so a seeded workflow regenerates on its next run. A busy card is skipped (`workers` reports `job_running`); a 409 when no card asked about is idle (#221, #677) |
| `rerun_job(job_id, acknowledged_cost=False, new_seed=False)` | Queue a fresh job from a previous job's stored specification, behind the same gate as `run_workflow`. Without `new_seed=true` a seeded rerun repeats its arguments exactly and the step cache serves it from the earlier files (`reused: true`) |
| `move_job(job_id, direction)` | Reorder a queued job: `direction` is `up`, `down`, `front` or `back` |

### Models

`list_models` (above, under Catalog) reports the cache.

| Tool | Purpose |
| --- | --- |
| `download_model(repo_id, acknowledged_cost=False)` | Fetch a whole model repo at `main` into the cache; returns once the download starts. It does not pre-fetch a single pinned file a task names; the task fetches that on its first run |
| `list_downloads()` | Downloads running or recently run |
| `cancel_download(download_id)` | Stop a running download; the record shows `cancel_requested: true` until `cancelled`. Partial files stay cached and resume on a retry |
| `delete_model(repo, acknowledged_cost=False)` | Delete every cached revision of one repo. Not recoverable locally |
| `get_diffusers_state()` | Installed diffusers version, its git commit, and any update in flight |
| `update_diffusers(acknowledged_cost=False)` | Upgrade diffusers to GitHub HEAD in the background |

The server refuses `delete_model` and `update_diffusers` while a job is
running or queued, and `delete_model` while a download is active, and says
why.

## The cost gate

Seven tools refuse unless `acknowledged_cost` is set. Each refuses in its own
words.

| Tool | What it commits |
| --- | --- |
| `run_workflow` | Minutes of GPU time on a card the queue could give another job |
| `rerun_job` | The same run, from a stored spec; ungated, any job id would bypass the gate |
| `download_model` | Tens of gigabytes of network and disk |
| `delete_model` | Cached weights, unrecoverably |
| `update_diffusers` | Replacing the installed library with an untagged development build |
| `enhance_prompt` | A real job, delaying any generation behind it |
| `delete_workspace` | Every workflow, asset and generated file in a workspace, unrecoverably |

`cancel_job` and `cancel_download` are not gated: they end a cost.

The acknowledgement can be bound to the quote. Pass
`acknowledged_cost={"fingerprint": plan.fingerprint, "minutes":
plan.estimate.minutes, "downloads": [...]}` from `validate_workflow`'s `plan`,
and the server refuses with 409, naming the new plan, if the run's shape
changed since (a longer list, an edited stored prompt, weights that now need
downloading). The `plan` passed unedited binds the same way: with no
`downloads` key, its `downloads_required` repos and `estimate.minutes` stand
in. Bare `true` is for a `plan` that came back null. The job records the form
it got (`acknowledged: none | boolean | bound`).

The flag does not make a tool wait. The five that start work return once it is
queued or started; `delete_model` and `delete_workspace` complete before they
answer.

The loop ([The loop](WORKFLOW_GUIDE.md#the-loop) is the full version):

1. `validate_workflow` with the `arguments` you will run with. Quote its
   `plan`: `estimate.minutes` with its `basis`, and each `downloads_required`
   entry (task repos and single files included) as its own line.
2. `run_workflow` with `acknowledged_cost` bound to the plan (or `true` when
   there was none); it returns a `job_id`.
3. `wait_for_job(job_id, timeout_seconds=...)`, asking for the estimate plus a
   margin and calling again on `still_running: true`; or `get_job_events`,
   passing back `last_seq` as `after`, for incremental progress.
4. `get_job(job_id)` for the manifest, or the error and traceback. A pipeline
   step's entry carries `timing` (load, encode, denoise, decode and save
   seconds; [WORKSPACES.md](WORKSPACES.md#step-timing)).
5. `get_output_image(name)`, `get_output_frames(name, count=12)` and
   `get_output_audio(name, start, duration)` to judge the result.

While a job runs, `get_job` and `wait_for_job` carry `progress`: the step, the
phase (`loading`, `generating`, `decoding`, `saving`) with the model in
`phase_detail`, `seconds_in_phase`, `seconds_since_event`, and
`denoise_step`/`denoise_total_steps` (null until the denoise loop starts). A
run is healthy while `denoise_step` moves between polls minutes apart, and
stuck when it is a number that stays put while `seconds_since_event` climbs.
Two silent stretches are normal: the lead-in under `generating` with a null
`denoise_step` (prompt and reference encoding plus the first denoise pass;
several minutes on MiniMax H3 with a video reference), and `saving`, where the
counter sits at its total. `get_job_events` carries a `log` line per top-level
modular block (`MiniMaxAI/MiniMax-H3: vae_encoder`) and per file written
(`writing shot.mp4 (121 frames)`, `wrote shot.mp4 in 1.3s (1.4 MB)`, #97).
Gaps between denoise steps are uneven under a transformer block cache.
Measured lead-in figures and every field are in
[Progress on a running job](SERVER.md#progress-on-a-running-job) and
[Progress events](SERVER.md#progress-events).

## Security

`dw_mcp` adds no authentication of its own and inherits the REST API's
posture, described in [Security](SECURITY.md): an opt-in bearer token
(`--token` / `DW_API_TOKEN`) on `/api/*` and `/mcp`, `Origin` and `Host`
header checks, and `dw/security.py` path confinement for every workflow,
gallery, asset and prompt path a tool sends. Nothing under `dw_mcp/` loosens
that confinement.

Two tools touch a file outside the API, and whose machine "local" is depends
on the transport:

- **`download_output`** writes a file. Over stdio `dw-mcp` it writes anywhere
  the user's own process may, like a shell redirect. Over `dw.serve --mcp` it
  writes on the server, confined to the workspace: an absolute or `~` path
  outside it is refused, a relative one is joined onto the workspace, and
  `destination` is required (#353, #113). Containment is checked on the
  resolved real path. On both transports a `..` segment is refused and an
  existing file needs `overwrite=True`.
- **`upload_asset(file_path=...)`** reads a file. Over stdio it reads what the
  user may. Over `dw.serve --mcp` `file_path` is confined to the server's
  workspace, workflows, assets, outputs and prompts directories, and the
  refusal comes before the file is looked for, so it is not a path-existence
  oracle (#138). `content` names no path and is not confined.

`dw-mcp` reaches a `dw.serve` on another machine only when that server has a
token and the same token is passed here; it refuses to start otherwise. The
token is the only authentication and the connection is plaintext HTTP: use a
network you control, or Tailscale or a TLS proxy beyond that
([REMOTE.md](REMOTE.md)). `dw.serve --mcp` serves this tool surface at `/mcp`
behind the same token, and refuses to start on a non-loopback host without one.

`save_workflow` and `run_workflow` let an MCP client write and then run a
workflow, and a workflow can execute arbitrary Python
([Trust model](SECURITY.md#trust-model)). What protects the server is its own
`--trust-workflows` flag, off by default: leave it off on any server an MCP
client can reach.

## Known limits

- **Event history is bounded.** A finished job keeps its last 200 events
  (`MAX_PERSISTED_EVENTS`). A job from before event persistence returns an
  empty list with a `note`.
- **No video content type.** A video is seen as frames (`get_output_frames`)
  and heard as its track (`get_output_audio`); a whole clip over the 4MB
  budget is refused, so ask for an excerpt.
- **Files cross machines only by transport.** Over `dw.serve --mcp`,
  `upload_asset(file_path)` and `download_output` act on the GPU box, not the
  client. Send a client file with `upload_asset(content=..., asset_name=...)`
  (4MB cap) or give the workflow a URL; fetch results by their `url`.
- **Prompts are not per-workspace.** The prompt library is one library shared
  by every workspace, because `prompt:` is shared by reference.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Failed to reconnect to dw: ENOENT` (or the client cannot start the server) | The client cannot find the command. Register the venv's **absolute** path to `dw-mcp` ([Use the absolute path](#use-the-absolute-path-to-dw-mcp)). A bare name works only when the client was launched from an activated venv |
| The server shows connected, but every tool fails | "Connected" means `dw-mcp` launched, not that the engine is reachable. Check `dw.serve` is running |
| "Cannot reach diffusers-workflow at …" | `dw.serve` is not running. Start it with `dw-serve` (or `python -m dw.serve`) |
| `dw-mcp` exits 2 at startup | A non-loopback `--url` with no token, or the startup probe failed against a remote URL (unreachable, or the token was rejected with a 401) |
| A config change has no effect | A running session holds the old config. Start a new session |
| It broke after rebuilding the venv | Re-run `pip install -e ".[server,mcp]"`. If the repo moved, re-register the new absolute path |
| A tool call times out | One API request exceeded `--timeout` (30 s); raise it. For a long job use `wait_for_job`, which returns at its cap rather than timing out |
| `run_workflow` or `rerun_job` refuses with a cost message | The `acknowledged_cost` gate. Confirm with the user and call again with `acknowledged_cost` bound to the plan (or `true`); a 409 means the run changed since the quote: re-validate, re-quote, pass the new plan |
