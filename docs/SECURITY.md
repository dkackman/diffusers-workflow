# Security

## Overview

diffusers-workflow validates all file paths, user inputs, and URLs to protect against path traversal, injection, and resource exhaustion. A command-argument sanitizer (`sanitize_command_args()`) is available for any future subprocess use, though `dw/` currently invokes no subprocess/shell commands.

## Security Module (`dw/security.py`)

### Path Validation

- `validate_path()` — Blocks `../` (anywhere in the path), `~/` (or `~\`), and paths rooted at `/dev/`, `/proc/`, `/sys/`. Rejects null bytes and overlong paths (> 4096 chars). Resolves to an absolute, realpath'd path. If `base_dir` is given, raises `PathTraversalError` when the resolved path falls outside it.
- `validate_workflow_path()` — `validate_path()` plus a required `.json` extension (via `validate_file_extension()`)
- `validate_output_path()` — `validate_path()` with `allow_create=True`, for directories/files that don't need to exist yet
- `validate_file_extension()` — Checks a path's extension against an allowed set (used internally by `validate_workflow_path()` and by `arguments.py` for media files)

### Input Validation

- `validate_variable_name()` — Alphanumeric, underscore, hyphen only (pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$`), max 100 chars
- `validate_string_input()` — Max length (default 1000 chars), no null bytes, no control characters other than tab/newline/CR
- `validate_json_size()` — Limits JSON files to 50MB
- `validate_url()` — Scheme must be `http` or `https`; must have a non-empty domain (`netloc`)
- `validate_constant_name()` — Guards `constant:` references before import: dotted-name pattern only, module must already be importable, and anything callable is refused
- `safe_join_path()` — Joins path components after rejecting any that contain `..`, `/`, or `\\`. Defined in `security.py` but not currently called elsewhere in `dw/`.

### Command Sanitization

- `sanitize_command_args()` — Rejects arguments containing shell metacharacters ( `` ` `` `$` `|` `&` `;` `>` `<` and newline/CR). It does **not** call `shlex.quote()` — with `shell=False`, argument list separation is handled safely by Python/the OS, so this function is a defense-in-depth check, not an escaping step.
- As of this writing, `dw/` does not invoke `subprocess` anywhere — the REPL's worker process (`dw/repl_worker.py`, `dw/worker.py`) is a `multiprocessing.Process` communicating over `multiprocessing.Queue`, not a shelled-out command. `sanitize_command_args()` is exercised by `tests/test_security.py` but is otherwise unused; it exists for any future code path that shells out.

## Trust model

**A workflow JSON file can execute arbitrary Python.** This is a deliberate
design choice, in the same spirit as ComfyUI custom nodes - the engine's
dynamic-import machinery is what lets a workflow name any diffusers
pipeline, scheduler, or quantization backend without dw shipping a bespoke
adapter for each one. But the same machinery means loading a workflow file
is not a passive data-load. Three things in a workflow JSON run
`importlib.import_module()` on a name the file supplies, which executes
that module's top-level code:

- **`pre_load_modules`** (`dw/pipeline_processors/pipeline.py`) - a list of
  module names imported before the pipeline loads, for their import-time
  registration side effects (`sdnq` registering its quantization method
  with diffusers, for instance)
- **A dotted `*_type`/`*_dtype`/`dtype`/`config_type` value**
  (`dw/type_helpers.py`, reached from `dw/arguments.py` and
  `dw/pipeline_processors/config_objects.py`) - `"sdnq.SDNQConfig"` imports
  `sdnq` and reads `SDNQConfig` off it; nothing stops the module part from
  naming something with no legitimate reason to appear in a workflow
- **A `constant:`-prefixed reference** (`dw/type_helpers.py`'s
  `load_constant_from_name`, reached from `dw/arguments.py`) - imports the
  module the constant is declared in the same way, before reading the
  attribute off it. `fetch_constant` refuses anything callable it finds,
  but the import itself has already run by that point

**Treat an untrusted workflow file exactly like an untrusted Python
script.** Don't run one from a source you would not run a `.py` file from -
a random download, a link in an issue, an LLM-authored file you have not
read.

### `--trust-workflows`

`dw-run`, `dw-serve`, and `dw-repl` all take a `--trust-workflows` flag,
**off by default**. Untrusted (the default), `pre_load_modules` and any
dotted `*_type`/`*_dtype`/`dtype`/`config_type` value are refused unless
they resolve under a top-level package the tool already depends on for
exactly this purpose - the framework packages (`diffusers`, `torch`,
`torchvision`, `transformers`, `accelerate`, `peft`) and the quantization
backends `pyproject.toml` declares for `config_objects.py`'s dynamic
loading (`sdnq`, `torchao`, `optimum` for optimum-quanto, `gguf`,
`bitsandbytes`), plus `dw` itself (a workflow's `component_type` can name a
pipeline under `dw.community_pipelines`, which ships in this repo, not a
third party one). The refusal names exactly what triggered it and points
back at `--trust-workflows`. The bundled examples under `workflows/` (not
`workflows/archive/`) all stay inside this set and load untrusted; a
workflow that needs to reach outside it - a community pipeline module from
somewhere else, a custom scheduler package - needs `--trust-workflows`.

A dotted `constant:` reference is gated the same way
(`dw/type_helpers.load_constant_from_name`): the module it names is
imported before `fetch_constant` gets to refuse a callable, so the import
itself is what the gate has to stop. A bare name (`constant:SOME_NAME`)
reads from `diffusers` and is always allowed.

Two `from_pretrained_arguments` keys are refused untrusted as well, for
every component and pipeline: `trust_remote_code` (runs the model repo's
own modeling code) and `custom_pipeline` (fetches and imports a pipeline
module from the Hub or a local path). Neither goes through an
`importlib` call of ours, so the dotted-name gate alone would leave
diffusers' remote-code paths open. A workflow that needs them
(`workflows/Krea2Edit.json`, say) needs `--trust-workflows`.

`--trust-workflows` is a blanket, process-wide choice - it is not scoped
per-workflow or per-request. A `dw-serve` instance that accepts jobs from
anything other than yourself (including an MCP client - see below) should
be run without it.

### MCP-authored workflows (M3)

The MCP server's `save_workflow` and `run_workflow` tools let an LLM write
and then execute a workflow through `dw.serve` - `save_workflow` writes a
JSON file into the workflow directory, `run_workflow` queues it (or an
inline definition) as a job. Nothing in `dw_mcp/` inspects what a workflow
it saves or runs actually contains. What protects a server used this way
is exactly the mechanism above: `dw-serve`'s own `--trust-workflows`
default is untrusted, so an MCP-authored or MCP-submitted workflow gets
the same code-execution gate a workflow from any other untrusted source
does, with no code change needed in `dw_mcp` itself. Running `dw-serve
--trust-workflows` removes that gate for every job the server accepts,
MCP-submitted or not - see the blanket-choice note just above.

## Integration Points

| Entry Point | What's Validated |
|-------------|-----------------|
| `workflow.py` | Workflow file paths, JSON size, output directories, sub-workflow paths |
| `run.py`, `validate.py` | CLI arguments, variable names and values |
| `repl.py`, `repl_commands.py` | Interactive command arguments — paths, workflow paths, output paths, variable names/values |
| `arguments.py` | Image/video/audio URLs, file paths, file extensions (`validate_media_location`, `fetch_image`, `fetch_video`) |
| `tasks/gather.py` | URLs passed to the `gather` task |
| `result.py` | Output directories and filenames |
| `server/app.py`, `server/jobs.py` | Every HTTP-supplied path — workflow files confined to the workflow directory, gallery files to the output directory, inline-workflow `base_dir`, `Origin`-header guard on every request |

## MCP Server

`dw_mcp/` introduces no new file access and no authentication of its own. It
is an HTTP client of a running `dw.serve`: every path a tool touches (a
workflow name, a gallery file, a job id) is sent to the REST API as-is and
validated there, exactly as it would be for a browser request from the web
UI. Localhost only — see [MCP Server](MCP.md#security).

The one exception is `download_output`, which writes a local file for the
MCP client rather than only reading through the API. It may write anywhere
the client's own filesystem permissions allow — a full path, a directory, or
the current working directory by default, `~` expanded — since it acts for
the local user the same way a shell redirect would; a `..` path segment in
`destination` is refused regardless, and an existing file at the resolved
path is left alone unless the caller passes `overwrite=True`.

## Exception Hierarchy

```text
SecurityError
  PathTraversalError — path traversal attempt
  InvalidInputError  — input validation failure
```

## Rules

- Always validate paths before file operations
- Use `validate_url()` before loading remote resources
- If a subprocess is ever introduced, use `shell=False` and pass args through `sanitize_command_args()`
- Never use dynamic code execution (`eval`/`exec`) or shell interpretation

## Protected Against

- **Path traversal** — Cannot access files outside allowed directories
- **Command injection** — No shell interpretation is used anywhere in `dw/`; `sanitize_command_args()` is available as a guard should a subprocess call be added
- **Resource exhaustion** — File size limits prevent memory exhaustion
- **Malicious URLs** — Only http/https schemes allowed

## Testing

```bash
pytest tests/test_security.py -v
```
