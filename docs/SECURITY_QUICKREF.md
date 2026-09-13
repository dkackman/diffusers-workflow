# Security Quick Reference

## MCP Server

`dw_mcp/` adds no new file access and no authentication — it is an HTTP
client of a running `dw.serve`, and every path a tool sends goes through the
same validation the REST API already applies. A remote `dw.serve` is
allowed only with a token, and `dw.serve --mcp` serves the same tool
surface at `/mcp` behind that token. See [MCP Server](MCP.md#security)
and [REMOTE.md](REMOTE.md).

## Imports

```python
from dw.security import (
    validate_path, validate_workflow_path, validate_output_path,
    validate_url, validate_variable_name, validate_string_input,
    sanitize_command_args, SecurityError, PathTraversalError, InvalidInputError
)
```

## Locations a workflow supplies

A media path, glob or URL that comes out of a workflow's arguments is not
just a path - it is untrusted input choosing where the server reads. Use
`dw/locations.py`, never `validate_path`/`validate_url` directly, for
anything a workflow names:

```python
from dw.locations import (
    validate_media_path,    # confined to the workflow dir / assets / outputs
    validate_media_glob,    # the same, on a pattern's fixed prefix
    contained_matches,      # each match re-checked on its real path
    validate_media_url,     # no loopback / link-local / private host
    validate_model_name,    # a Hub repo id, or a contained path
)
```

Containment is checked before existence, so a refusal never discloses
whether the file is there. All of it yields to `--trust-workflows`.

## Common Patterns

```python
# File paths
safe_path = validate_path(user_path, allow_create=False)
safe_path = validate_path(user_path, base_dir="/allowed/dir")
workflow_path = validate_workflow_path("workflow.json")
output_path = validate_output_path(user_path, base_output_dir)

# User input
var_name = validate_variable_name("prompt")          # OK
var_name = validate_variable_name("bad;name")        # raises InvalidInputError
value = validate_string_input(user_input, max_length=1000)
url = validate_url(user_url)                         # http/https only

# Subprocess (dw/ doesn't currently shell out anywhere - pattern for if/when it does)
import subprocess
cmd = sanitize_command_args(["python", "-m", "dw.run", validated_path])
subprocess.Popen(cmd, shell=False)
```

## Error Handling

```python
try:
    path = validate_path(user_path)
except PathTraversalError as e:
    logger.error(f"Path traversal detected: {e}")
except InvalidInputError as e:
    logger.error(f"Invalid input: {e}")
```

## Input Constraints

**Variable names:** Letters, numbers, underscores, hyphens. Must start with letter or underscore.

```text
OK:      prompt, num_images, my-variable, _private
Invalid: my.var, var;name, $var, var name
```

**File paths:** No `../` traversal. No `~/`, `/dev/`, `/proc/`, `/sys/`.

```text
OK:      ./subdir/workflow.json, /full/path/workflow.json, builtin:h3_context_ir.json
Invalid: ../../../etc/passwd, ~/secret.json
```

**URLs:** http and https only.

```text
OK:      https://example.com/image.jpg
Invalid: file:///etc/passwd, ftp://server/file
```

## Common Errors

| Error | Cause | Fix |
| ----- | ----- | --- |
| `Path contains dangerous pattern matching \.\.` | Path traversal | Use absolute or `./` relative paths |
| `Invalid variable name: my.var` | Special characters | Use `my_var` instead |
| `Invalid URL: URL scheme not allowed: file` | Non-http scheme | Use https or local file path |
| `Argument contains dangerous characters: ...` | Shell metacharacters | Remove `` ` `` `$` `\|` `&` `;` `>` `<` |
