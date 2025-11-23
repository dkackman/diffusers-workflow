# Security Quick Reference

## Import Security Functions

```python
from dw.security import (
    validate_path, validate_workflow_path, validate_output_path,
    validate_url, validate_variable_name, validate_string_input,
    sanitize_command_args, SecurityError, PathTraversalError, InvalidInputError
)
```

## Common Security Patterns

### Validating File Paths

```python
# General file path
safe_path = validate_path(user_path, allow_create=False)

# Workflow file (.json only)
workflow_path = validate_workflow_path(user_path)

# Output directory/file
output_path = validate_output_path(user_path, base_output_dir)

# With base directory restriction
safe_path = validate_path(user_path, base_dir="/allowed/dir")
```

### Validating User Input

```python
# Variable names (alphanumeric + underscore/hyphen)
var_name = validate_variable_name(user_input)

# String values with length limits
value = validate_string_input(user_input, max_length=1000)

# Allow empty strings
value = validate_string_input(user_input, max_length=1000, allow_empty=True)

# URLs (http/https only)
url = validate_url(user_url)
```

### Safe Subprocess Execution

```python
import subprocess

# Build command list
cmd = ["python", "-m", "dw.run", workflow_path, f"{var_name}={value}"]

# Sanitize arguments
safe_cmd = sanitize_command_args(cmd)

# Execute without shell
process = subprocess.Popen(safe_cmd, shell=False)
```

### Error Handling

```python
try:
    path = validate_path(user_path)
    workflow = workflow_from_file(path, output_dir)
except PathTraversalError as e:
    logger.error(f"Path traversal detected: {e}")
    raise
except InvalidInputError as e:
    logger.error(f"Invalid input: {e}")
    raise
except SecurityError as e:
    logger.error(f"Security error: {e}")
    raise
```

## Security Checklist

When adding new code that handles user input:

- [ ] Validate all file paths with appropriate validator
- [ ] Check variable names with `validate_variable_name()`
- [ ] Validate string inputs with `validate_string_input()`
- [ ] Validate URLs with `validate_url()`
- [ ] Sanitize subprocess arguments with `sanitize_command_args()`
- [ ] Use `shell=False` in subprocess calls
- [ ] Catch and handle `SecurityError` exceptions
- [ ] Log security validation failures
- [ ] Never use `eval()` or `exec()` on user data

## Dangerous Patterns to Avoid

### ❌ DON'T
```python
# Don't use shell=True
subprocess.run(f"python -m dw.run {user_input}", shell=True)

# Don't skip path validation
with open(user_path, 'r') as f:
    data = f.read()

# Don't trust user input
workflow_path = f"../{user_dir}/workflow.json"

# Don't allow any URL scheme
image = load_image(f"file://{user_path}")

# Don't use eval/exec
result = eval(user_expression)
```

### ✅ DO
```python
# Use shell=False with validated args
cmd = sanitize_command_args(["python", "-m", "dw.run", validated_path])
subprocess.run(cmd, shell=False)

# Validate paths before use
validated = validate_path(user_path, allow_create=False)
with open(validated, 'r') as f:
    data = f.read()

# Use safe path joining
workflow_path = validate_path(os.path.join(base_dir, user_dir, "workflow.json"))

# Validate URL scheme
url = validate_url(user_url)
image = load_image(url)

# Don't use eval/exec at all - use safe alternatives
```

## Security Constants

```python
MAX_PATH_LENGTH = 4096
MAX_FILENAME_LENGTH = 255
MAX_JSON_SIZE = 50 * 1024 * 1024  # 50MB

ALLOWED_JSON_EXTENSIONS = {'.json'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
```

## Testing Security

```python
# Test path traversal protection
with pytest.raises(PathTraversalError):
    validate_path("../../../etc/passwd")

# Test invalid variable names
with pytest.raises(InvalidInputError):
    validate_variable_name("invalid;name")

# Test URL scheme validation
with pytest.raises(InvalidInputError):
    validate_url("file:///etc/passwd")

# Test command injection protection
with pytest.raises(InvalidInputError):
    sanitize_command_args(["rm", "-rf", "; malicious"])
```

## Quick Fixes

### Path traversal detected
**Error**: `PathTraversalError: Path contains dangerous pattern: ../`
**Fix**: Use absolute paths or safe relative paths without `../`

### Invalid variable name
**Error**: `InvalidInputError: Invalid variable name: my.var`
**Fix**: Use only letters, numbers, underscores, hyphens: `my_var`

### URL scheme not allowed
**Error**: `InvalidInputError: URL scheme not allowed: file`
**Fix**: Use http/https URLs or local file paths

### Dangerous characters
**Error**: `InvalidInputError: Argument contains dangerous characters`
**Fix**: Remove shell metacharacters: `;`, `|`, `&`, `$`, `` ` ``