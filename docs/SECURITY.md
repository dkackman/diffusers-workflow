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
- `validate_string_input()` — Max length, no null bytes, no control characters other than tab/newline/CR. Every caller that checks a caller-supplied variable value (`dw/variables.py`, `dw/run.py`, the REPL) passes `MAX_VARIABLE_VALUE_LENGTH`, 20,000 characters; file names and paths pass their own, shorter caps, so the function's bare default of 1000 is not the limit anything is held to. A variable's *default*, written in the definition, is not capped separately: the author controls the file, and the whole file is capped at 50MB
- `validate_json_size()` — Limits JSON files to 50MB
- `validate_url()` — Scheme must be `http` or `https`; must have a non-empty domain (`netloc`); may not contain a backslash. `urllib.parse` and the HTTP client disagree on which host `http://169.254.169.254\@example.com/` names, so the host the check approved need not be the one dialed; a `\` that belongs in a path is written `%5C`
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
back at `--trust-workflows`. The bundled examples under `workflows/` all
stay inside this set and load untrusted; a
workflow that needs to reach outside it - a community pipeline module from
somewhere else, a custom scheduler package - needs `--trust-workflows`.

The package is not the whole check, because an allowed package holds
things other than classes and re-exports modules outside itself. Untrusted,
two more rules apply (`dw/type_helpers.py`):

- **A type reference must resolve to a class.** A `*_type`/`config_type`
  value is constructed with the workflow's own arguments, so
  `"torch.hub.load"` - in `torch`, and a function that fetches and runs a
  GitHub repo's code - is refused as "not a class", as is a module or any
  other object. Under a `dtype` or `*_dtype` key a `torch.dtype`
  (`"torch.bfloat16"`) is accepted too, since that is data rather than
  something called. A bare name (`"FluxPipeline"`) resolves against
  `diffusers` and is held to the same rule. `validate_workflow` reports the
  refusal at the key's path, for every key the run loads as a type
  (`from_pretrained_arguments.torch_dtype` as much as `config_type`).
- **A `constant:` walk stays inside the package.** A dotted `constant:`
  reference is gated like a type (the module it names is imported before
  `fetch_constant` gets to refuse a callable, so the import itself is what
  the gate has to stop), and then every step of the walk is checked: no
  segment may start with `_`, checked before anything imports, and no module
  the walk passes through may sit outside the allowed packages -
  `constant:torch.os.environ` starts in `torch` and ends in the server's
  environment, and is refused at `torch.os`. Reading a field off a value
  declared in an allowed module still works
  (`...ltx2.utils.GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens`). A bare
  name (`constant:SOME_NAME`) reads from `diffusers`. `validate_workflow`
  resolves every literal `constant:` in a step the way the run does and
  reports a refusal at its path (a variable's default at `variables.<name>`),
  so nothing is queued to find out.

Every `*_type` and `constant:` in the bundled catalog satisfies both rules,
pinned by `tests/test_workflow_trust.py`'s catalog sweep. `--trust-workflows`
lifts both, as it lifts the package check.

Two `from_pretrained_arguments` keys are refused untrusted as well, for
every component and pipeline: `trust_remote_code` (runs the model repo's
own modeling code) and `custom_pipeline` (fetches and imports a pipeline
module from the Hub or a local path). Neither goes through an
`importlib` call of ours, so the dotted-name gate alone would leave
diffusers' remote-code paths open. No bundled catalog entry sets either
(`tests/test_catalog_structure.py` refuses one that does); a workflow that
needs them needs `--trust-workflows`.

### Where a workflow may read and reach (`dw/locations.py`)

Code execution is not the only thing a workflow file chooses. Its arguments
choose *locations* - which image to open, which URL to fetch, which
directory a glob expands over - and until 2026-09-13 each loader trusted the
one it was handed. A workflow could name `/usr/share/pixmaps/debian-logo.png`
as its `image` and get it decoded, glob `/usr/share/pixmaps/*.png` and get
every match republished verbatim as an output, or point an `image` at
`http://127.0.0.1:8765` and have the server fetch its own loopback.
`remote_text_encoder.url` was the worst of them: the request carries this
machine's HuggingFace token.

One policy now answers all of it, untrusted:

- **A path** must resolve inside a root this installation already works in -
  the workflow file's own directory, the asset libraries on the search path,
  the output root. `validate_path` already refuses `..`, so in practice this
  closes the absolute path that pointed somewhere else entirely; a relative
  one that climbs out is refused on its `..` segments, at validation time as
  well as at the loader, so both spellings are answered at the same moment
  rather than one of them three seconds into a queued job (#124). The remedy
  for a file outside is to put it in the asset library and use an `asset:`
  reference. Containment is checked **before** existence, so the refusal
  cannot be used as a file-existence oracle.
- **A glob** is contained the same way, on the fixed directory its pattern
  starts from, and every match is re-checked on its real path so a symlink
  cannot carry the expansion out.
- **An `http(s)` URL** must not resolve to an address inside the deployment -
  loopback, link-local (`169.254.0.0/16`, the cloud metadata address),
  private ranges. Checked after DNS resolution, not on the literal string.
- **`remote_text_encoder.url`** is https-only, and the HuggingFace token is
  attached only for `huggingface.co`, `huggingface.cloud` and `hf.space`. An
  endpoint elsewhere is still reachable; it just does not get the credential.
- **`model_name`** must be a Hub repo id or a path inside a root - the same
  shape check `download_model` has always applied to `repo_id`. A URL is
  neither, and is refused as such rather than resolving into the workflow's
  own directory as a path-shaped name (#117).

Enforced twice: `location_errors` runs inside `validation_errors`, so
`validate_workflow` refuses before a model load is spent on the run, and the
loaders call the same functions for a location that arrives through a
variable or a previous result. All of it yields to `--trust-workflows`.

`GET /api/server` reports `trust_workflows`, so a client can read the posture
it is running against rather than infer it.

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
UI. A remote `dw.serve` is allowed only with a token — see
[MCP Server](MCP.md#security) and [REMOTE.md](REMOTE.md).

The two exceptions are `download_output`, which writes a local file for the
MCP client, and `upload_asset(file_path=...)`, which reads one — neither goes
through the API for that half of its work. Both turn on whose machine "local"
is. Over a **stdio `dw-mcp`** it is the user's own, so both act for the local
user the way a shell redirect would: `download_output` writes anywhere the
process may (a full path, a directory, or the working directory by default,
`~` expanded) and `upload_asset` reads anything it may. Over **`dw.serve
--mcp`** it is the operator's box, which the caller never chose, so both are
confined there: `download_output`'s `destination` to the workspace (#113) and
`upload_asset`'s `file_path` to the directories the server works in — its
workspace, workflows, assets, outputs and prompts (#138). `upload_asset`'s
refusal is ordered ahead of the existence and extension checks so it cannot
be used as a path-existence oracle. A `..` path segment in `destination` is
refused regardless, and an existing file at the resolved path is left alone
unless the caller passes `overwrite=True`.

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
- **Malicious URLs** — Only http/https schemes allowed, and an untrusted
  workflow may not name a host inside the deployment (SSRF)
- **Arbitrary file read through a media argument** — a location a workflow
  supplies is confined to the roots it may read (`dw/locations.py`)
- **Script on the UI's origin through an output** — a step's
  `result.content_type` may not be `text/html` or `text/xml` (compared
  without case or parameters): validation refuses it at
  `steps[i].result.content_type` and the writer refuses it again
  (`dw/content_types.py`). A file of an active type that reaches `/outputs`
  or `/inputs` anyway is served with `Content-Security-Policy: sandbox`

## Testing

```bash
pytest tests/test_security.py tests/test_locations.py tests/test_workflow_trust.py -v
```
