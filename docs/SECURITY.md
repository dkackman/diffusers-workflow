# Security

## Overview

diffusers-workflow validates all file paths, user inputs, URLs, and subprocess arguments to protect against path traversal, command injection, and resource exhaustion.

## Security Module (`dw/security.py`)

### Path Validation

- `validate_path()` — Blocks `../`, `~/`, `/dev/`, `/proc/`, `/sys/`. Resolves to absolute paths. Enforces base directory restrictions.
- `validate_workflow_path()` — Validates workflow files (`.json` extension required)
- `validate_output_path()` — Validates output directories

### Input Validation

- `validate_variable_name()` — Alphanumeric, underscore, hyphen only (pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$`)
- `validate_string_input()` — Max length, no null bytes, no control characters
- `validate_json_size()` — Limits JSON files to 50MB
- `validate_url()` — http/https only

### Command Sanitization

- `sanitize_command_args()` — Detects shell metacharacters (`;`, `|`, `&`, `$`, `` ` ``), uses `shlex.quote()`
- All subprocess calls use `shell=False`

## Integration Points

| Entry Point | What's Validated |
|-------------|-----------------|
| `workflow.py` | Workflow file paths, JSON size, output directories, sub-workflow paths |
| `run.py`, `validate.py` | CLI arguments, variable names and values |
| `repl.py` | All user input, subprocess commands |
| `arguments.py` | Image/video URLs, file paths, file extensions |
| `result.py` | Output directories and filenames |

## Exception Hierarchy

```text
SecurityError
  PathTraversalError — path traversal attempt
  InvalidInputError  — input validation failure
```

## Rules

- Always validate paths before file operations
- Use `validate_url()` before loading remote resources
- Use `shell=False` in subprocess calls
- Never use dynamic code execution or shell interpretation

## Protected Against

- **Path traversal** — Cannot access files outside allowed directories
- **Command injection** — Shell metacharacters detected and blocked
- **Resource exhaustion** — File size limits prevent memory exhaustion
- **Malicious URLs** — Only http/https schemes allowed

## Testing

```bash
pytest tests/test_security.py -v
```
