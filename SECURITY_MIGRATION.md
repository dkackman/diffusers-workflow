# Security Migration Guide

## Overview

Version 0.38.0+ includes comprehensive security improvements to protect against path traversal, command injection, and input validation vulnerabilities. This guide helps you understand what changed and how to update your workflows if needed.

## What Changed

### 1. Path Validation
All file paths are now validated before use. This affects:
- Workflow file loading
- Sub-workflow references
- Image/video file loading
- Output directory paths

### 2. Input Validation
All user inputs are validated:
- Variable names (alphanumeric, underscore, hyphen only)
- Variable values (max length, no control characters)
- URL schemes (http/https only)
- File extensions (whitelist-based)

### 3. Subprocess Security
The REPL now uses secure subprocess execution:
- Command arguments are sanitized
- `shell=False` enforced (no shell interpretation)
- Dangerous characters in arguments are rejected

## Breaking Changes

### Variable Names
Variable names with special characters are no longer allowed:

**Before (allowed):**
```bash
python -m dw.run workflow.json "my.var"="value"
python -m dw.run workflow.json "var;name"="value"
```

**After (not allowed):**
```bash
# These will be rejected:
python -m dw.run workflow.json "my.var"="value"      # Contains '.'
python -m dw.run workflow.json "var;name"="value"    # Contains ';'

# Use these instead:
python -m dw.run workflow.json my_var="value"
python -m dw.run workflow.json var_name="value"
```

**Valid variable names:**
- Must start with letter or underscore
- Can contain letters, numbers, underscores, hyphens
- Examples: `prompt`, `num_images`, `test_var_123`, `my-variable`

### File Paths
Relative paths with parent directory traversal are blocked:

**Before (allowed):**
```json
{
  "workflow": {
    "path": "../../../unsafe/path.json"
  }
}
```

**After (not allowed):**
```json
{
  "workflow": {
    "path": "../../../unsafe/path.json"  // REJECTED: Path traversal
  }
}
```

**Use instead:**
- Absolute paths: `"/full/path/to/workflow.json"`
- Safe relative paths: `"./subdir/workflow.json"`
- Built-in workflows: `"builtin:augment_prompt.json"`

### URL Schemes
Only http and https URLs are allowed:

**Before (allowed):**
```json
{
  "image": "file:///etc/passwd"
}
```

**After (not allowed):**
```json
{
  "image": "file:///etc/passwd"  // REJECTED: Dangerous scheme
}
```

**Use instead:**
```json
{
  "image": "https://example.com/image.jpg",  // OK
  "image": "./local/image.jpg"                // OK
}
```

## Migration Steps

### Step 1: Review Variable Names
Check all your workflows for variable names with special characters:

```bash
# Search for potentially problematic variable names
grep -r '"[^"]*[^a-zA-Z0-9_-]' examples/*.json
```

Update any variable names that contain characters other than letters, numbers, underscores, or hyphens.

### Step 2: Check File Paths
Review all file path references in your workflows:

```bash
# Look for parent directory references
grep -r '\.\.\/' examples/*.json
```

Replace relative paths with:
- Absolute paths for stable locations
- Safe relative paths (no `../`)
- Built-in workflow references where applicable

### Step 3: Verify URLs
Check all URL references in your workflows:

```bash
# Look for non-http(s) URLs
grep -r '".*://' examples/*.json | grep -v 'https\?://'
```

Replace any file://, ftp://, or other scheme URLs with http/https or local file paths.

### Step 4: Test Your Workflows
Run validation on all your workflows:

```bash
for workflow in examples/*.json; do
  python -m dw.validate "$workflow" || echo "Failed: $workflow"
done
```

### Step 5: Update Documentation
If you have custom workflows or scripts, update documentation to reflect:
- Variable naming requirements
- Path restrictions
- URL scheme limitations

## Error Messages

### Common Errors and Solutions

#### "Invalid variable name: X"
**Cause:** Variable name contains special characters
**Solution:** Use only letters, numbers, underscores, hyphens

```bash
# Wrong
python -m dw.run workflow.json "my.var"="value"

# Right
python -m dw.run workflow.json my_var="value"
```

#### "Path contains dangerous pattern: ../"
**Cause:** Path traversal attempt detected
**Solution:** Use absolute or safe relative paths

```json
// Wrong
{"workflow": {"path": "../../other/workflow.json"}}

// Right
{"workflow": {"path": "./subdir/workflow.json"}}
{"workflow": {"path": "/full/path/workflow.json"}}
```

#### "URL scheme not allowed: file"
**Cause:** Dangerous URL scheme used
**Solution:** Use http/https or local file paths

```json
// Wrong
{"image": "file:///path/to/image.jpg"}

// Right
{"image": "/path/to/image.jpg"}
{"image": "https://example.com/image.jpg"}
```

#### "Path outside allowed directory"
**Cause:** Path resolves outside allowed base directory
**Solution:** Keep paths within workflow or output directories

#### "Argument contains dangerous characters"
**Cause:** Shell metacharacters in REPL command
**Solution:** Avoid using `;`, `|`, `&`, `$`, `` ` `` in arguments

## Compatibility

### Backward Compatibility
Most existing workflows will continue to work without changes if they:
- Use standard variable names (alphanumeric with underscores)
- Use safe file paths (no parent directory traversal)
- Use http/https URLs or local file paths

### Testing for Compatibility
To check if your workflows are compatible:

1. Run validation:
   ```bash
   python -m dw.validate your_workflow.json
   ```

2. Check for security errors in logs:
   ```bash
   python -m dw.run your_workflow.json -l DEBUG 2>&1 | grep -i security
   ```

3. Run the workflow with test inputs:
   ```bash
   python -m dw.run your_workflow.json test_var=test_value
   ```

## Getting Help

If you encounter issues migrating your workflows:

1. Check the error message for specific guidance
2. Review [SECURITY.md](SECURITY.md) for detailed security documentation
3. Open an issue with:
   - The error message
   - Your workflow JSON (sanitized)
   - What you're trying to achieve

## Security Benefits

These changes protect against:
- **Path Traversal Attacks** - Prevents access to sensitive system files
- **Command Injection** - Blocks shell command execution via user input
- **Resource Exhaustion** - Limits file sizes and input lengths
- **Malicious URLs** - Restricts URL schemes to safe protocols

While these restrictions may require some workflow updates, they significantly improve the security of the diffusers-workflow system.