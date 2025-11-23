# Changelog - Security Improvements

## [0.38.0] - 2025-11-23

### Security

#### Added
- **Comprehensive security module** (`dw/security.py`) with validation functions:
  - Path traversal protection with `validate_path()`
  - URL scheme validation (http/https only)
  - Variable name sanitization (alphanumeric + underscore/hyphen)
  - String input validation (length limits, null byte detection)
  - Command argument sanitization for subprocess safety
  - File extension whitelisting for images/videos
  - JSON file size limits (50MB) to prevent DoS

#### Changed
- **All file path operations** now validate paths before use
  - `workflow.py`: Validates workflow and sub-workflow paths
  - `run.py`: Validates command line file paths
  - `validate.py`: Validates workflow file paths
  - `arguments.py`: Validates image/video file paths and URLs
  - `result.py`: Validates output directory and file paths

- **All user inputs** are now validated:
  - `run.py`: Variable names and values from command line
  - `repl.py`: All REPL command inputs and arguments
  - `variables.py`: Variable names and string values
  
- **Subprocess execution** hardened in `repl.py`:
  - Command arguments sanitized with `sanitize_command_args()`
  - Shell execution disabled (`shell=False`)
  - Dangerous characters blocked (`;`, `|`, `&`, `$`, `` ` ``)

#### Security Exceptions
- New exception hierarchy for security errors:
  - `SecurityError` (base exception)
  - `PathTraversalError` (path traversal attempts)
  - `InvalidInputError` (input validation failures)

#### Documentation
- Added `SECURITY.md` - Comprehensive security implementation guide
- Added `SECURITY_MIGRATION.md` - Migration guide for existing workflows
- Updated `tests/test_security.py` - Security validation tests
- Updated `.github/copilot-instructions.md` - Security guidelines for AI agents

### Breaking Changes

#### Variable Names
Variable names must now match pattern `^[a-zA-Z_][a-zA-Z0-9_-]*$`:
```bash
# ✗ No longer allowed
python -m dw.run workflow.json "my.var"="value"
python -m dw.run workflow.json "var;name"="value"

# ✓ Use instead
python -m dw.run workflow.json my_var="value"
python -m dw.run workflow.json var_name="value"
```

#### File Paths
Parent directory traversal (`../`) is now blocked:
```json
// ✗ No longer allowed
{"workflow": {"path": "../../../other/workflow.json"}}

// ✓ Use instead
{"workflow": {"path": "./subdir/workflow.json"}}
{"workflow": {"path": "/absolute/path/workflow.json"}}
{"workflow": {"path": "builtin:workflow.json"}}
```

#### URL Schemes
Only http and https URLs are allowed:
```json
// ✗ No longer allowed
{"image": "file:///etc/passwd"}
{"image": "ftp://server/file.jpg"}

// ✓ Use instead
{"image": "https://example.com/image.jpg"}
{"image": "/local/path/image.jpg"}
```

### Migration

To migrate existing workflows:

1. **Check variable names**: Ensure they contain only letters, numbers, underscores, hyphens
2. **Review file paths**: Remove `../` references, use absolute or safe relative paths
3. **Verify URLs**: Ensure all URLs use http or https schemes
4. **Test workflows**: Run `python -m dw.validate workflow.json` on all workflows

See [SECURITY_MIGRATION.md](SECURITY_MIGRATION.md) for detailed migration instructions.

### Security Benefits

These changes protect against:
- **Path Traversal Attacks** - Cannot access files outside allowed directories (e.g., `/etc/passwd`)
- **Command Injection** - Shell metacharacters are detected and blocked
- **Resource Exhaustion** - File size limits prevent memory exhaustion attacks
- **Malicious URLs** - Only safe URL schemes (http/https) are allowed
- **Invalid Input** - All user input is validated before processing

### Testing

New security tests validate:
- Path traversal protection
- URL scheme validation
- Variable name validation
- String input validation
- Command argument sanitization

Run security tests:
```bash
pytest tests/test_security.py -v
```

### Backward Compatibility

Most existing workflows will continue to work if they:
- Use standard variable names (alphanumeric with underscores)
- Use safe file paths (no parent directory traversal)
- Use http/https URLs or local file paths

Workflows using special characters in variable names, path traversal, or non-http(s) URLs will need updates.

### Developer Notes

When adding new features:
- Always validate paths with `validate_path()` or specific validators
- Use `validate_variable_name()` for user-provided names
- Sanitize URLs with `validate_url()` before loading
- Use `sanitize_command_args()` before subprocess calls
- Import from `dw.security` and catch `SecurityError` exceptions
- Never use `eval()`, `exec()`, or `shell=True` for security

### References

- Security implementation: `dw/security.py`
- Security documentation: `SECURITY.md`
- Migration guide: `SECURITY_MIGRATION.md`
- Security tests: `tests/test_security.py`