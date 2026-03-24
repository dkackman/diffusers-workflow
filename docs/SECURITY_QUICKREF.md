# Security Quick Reference

## Imports

```python
from dw.security import (
    validate_path, validate_workflow_path, validate_output_path,
    validate_url, validate_variable_name, validate_string_input,
    sanitize_command_args, SecurityError, PathTraversalError, InvalidInputError
)
```

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

# Subprocess
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
OK:      ./subdir/workflow.json, /full/path/workflow.json, builtin:augment_prompt.json
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
| `Path contains dangerous pattern: ../` | Path traversal | Use absolute or `./` relative paths |
| `Invalid variable name: my.var` | Special characters | Use `my_var` instead |
| `URL scheme not allowed: file` | Non-http scheme | Use https or local file path |
| `Argument contains dangerous characters` | Shell metacharacters | Remove `;` `\|` `&` `$` `` ` `` |
