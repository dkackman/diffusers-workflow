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

`dw/mcp/` introduces no new file access and no authentication of its own. It
is an HTTP client of a running `dw.serve`: every path a tool touches (a
workflow name, a gallery file, a job id) is sent to the REST API as-is and
validated there, exactly as it would be for a browser request from the web
UI. Localhost only — see [MCP Server](MCP.md#security).

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
