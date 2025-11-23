# Security Implementation Guide

## Overview

This document describes the security measures implemented in diffusers-workflow to protect against common vulnerabilities including path traversal attacks, command injection, and input validation failures.

## Security Module (`dw/security.py`)

### Core Security Functions

#### Path Validation
- **`validate_path()`** - Prevents path traversal attacks by:
  - Checking for dangerous patterns (`../`, `~/`, `/dev/`, `/proc/`, `/sys/`)
  - Resolving paths to absolute canonical forms
  - Enforcing base directory restrictions
  - Validating path and filename lengths

#### Input Validation
- **`validate_variable_name()`** - Ensures variable names contain only safe characters (alphanumeric, underscore, hyphen)
- **`validate_string_input()`** - Validates string inputs for:
  - Maximum length constraints
  - Null byte detection
  - Control character filtering
- **`validate_json_size()`** - Prevents DoS attacks by limiting JSON file sizes to 50MB

#### URL Validation
- **`validate_url()`** - Restricts URL schemes to http/https only
- Prevents file:// and other dangerous protocol handlers

#### Command Sanitization
- **`sanitize_command_args()`** - Protects against command injection by:
  - Detecting shell metacharacters (`;`, `|`, `&`, `$`, `` ` ``)
  - Using `shlex.quote()` for additional escaping
  - Enforcing no shell execution (shell=False in subprocess calls)

### Security Constants

```python
MAX_PATH_LENGTH = 4096
MAX_FILENAME_LENGTH = 255
MAX_JSON_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_JSON_EXTENSIONS = {'.json'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
```

## Security Integration Points

### 1. Workflow Loading (`dw/workflow.py`)
- Validates workflow file paths before loading
- Checks JSON file sizes to prevent memory exhaustion
- Validates output directory paths
- Secures sub-workflow path resolution

```python
# Example: Loading workflow with validation
validated_path = validate_workflow_path(file_spec)
validate_json_size(validated_path)
validated_output = validate_output_path(output_dir, None)
```

### 2. Command Line Interface (`dw/run.py`, `dw/validate.py`)
- Validates all command line arguments
- Sanitizes variable names and values
- Validates file paths before processing

### 3. REPL Interface (`dw/repl.py`)
- Validates user input for all commands
- Sanitizes subprocess command construction
- Validates paths for load/run operations
- Uses `shell=False` to prevent shell injection

### 4. Resource Loading (`dw/arguments.py`)
- Validates image/video URLs
- Restricts file paths to allowed directories
- Validates file extensions against whitelist

### 5. Output Operations (`dw/result.py`)
- Validates output directory paths
- Sanitizes output filenames
- Prevents directory traversal in output paths

## Exception Hierarchy

```
SecurityError (base)
├── PathTraversalError - Path traversal attempt detected
└── InvalidInputError - Input validation failure
```

## Usage Examples

### Safe Path Validation
```python
from dw.security import validate_path, validate_workflow_path

# Validate any path
safe_path = validate_path("/path/to/file", allow_create=False)

# Validate workflow file with extension check
workflow_path = validate_workflow_path("workflow.json")

# Validate with base directory restriction
safe_path = validate_path("subdir/file.json", base_dir="/allowed/dir")
```

### Safe Variable Handling
```python
from dw.security import validate_variable_name, validate_string_input

# Validate variable name
name = validate_variable_name("prompt")  # OK
name = validate_variable_name("bad;name")  # Raises InvalidInputError

# Validate string value
value = validate_string_input(user_input, max_length=1000)
```

### Safe Command Execution
```python
from dw.security import sanitize_command_args
import subprocess

# Build command with validation
cmd = ["python", "-m", "dw.run", "workflow.json"]
sanitized = sanitize_command_args(cmd)

# Execute safely without shell
process = subprocess.Popen(sanitized, shell=False)
```

## Security Best Practices

### DO:
✓ Always validate paths before file operations
✓ Use `validate_url()` before loading remote resources
✓ Sanitize command arguments before subprocess calls
✓ Use `shell=False` in subprocess.Popen()
✓ Validate all user inputs (CLI, REPL, JSON)
✓ Check file extensions against whitelists
✓ Enforce size limits on loaded files

### DON'T:
✗ Use `shell=True` in subprocess calls
✗ Trust user input without validation
✗ Allow arbitrary file path access
✗ Use eval() or exec() on user data
✗ Allow unrestricted URL schemes
✗ Skip validation for "internal" operations

## Testing

Run security tests:
```bash
pytest tests/test_security.py -v
```

## Threat Model

### Protected Against:
1. **Path Traversal** - Cannot access files outside allowed directories
2. **Command Injection** - Shell metacharacters are detected and blocked
3. **Resource Exhaustion** - File size limits prevent memory exhaustion
4. **Malicious URLs** - Only http/https schemes allowed
5. **Invalid Filenames** - Length and character restrictions enforced

### Considerations:
- Assumes trusted workflow JSON schemas (schema validation required)
- Model downloads from HuggingFace Hub use their security measures
- GPU/memory limits should be enforced at system level
- Network requests to model repos are controlled by diffusers library

## Future Enhancements

1. **Content Security**
   - Add virus scanning for uploaded files
   - Implement content-type verification for images/videos

2. **Rate Limiting**
   - Add rate limits for API/URL requests
   - Implement timeout controls for long operations

3. **Audit Logging**
   - Log all security validation failures
   - Track file access patterns

4. **Sandboxing**
   - Consider containerization for workflow execution
   - Implement resource quotas (CPU, memory, disk)

## Security Contact

For security issues, please review the project's security policy or contact the maintainers directly.