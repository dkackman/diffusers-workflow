# Security Quick Reference

The one-page digest of [SECURITY.md](SECURITY.md), which has the detail and
the reasons.

## Imports

```python
from dw.security import (
    validate_path,           # no .., ~/, /dev/, /proc/, /sys/; confined only with a base_dir
    validate_workflow_path,  # validate_path + must exist + .json
    validate_output_path,    # validate_path, may not exist yet
    validate_url,            # http/https, a host, no backslash
    validate_variable_name,  # ^[a-zA-Z_][a-zA-Z0-9_-]*$, max 100
    validate_string_input,   # length, no null or control characters
    sanitize_command_args,   # no ` $ | & ; > < newline CR
    SecurityError,           # base of the three below
    PathTraversalError,
    InvalidInputError,
    UntrustedWorkflowError,
)
```

Names joined onto a directory have their own validators
(`validate_asset_reference`, `validate_output_reference`,
`validate_prompt_reference`, `validate_subfolder`, `validate_workspace_name`,
...): see SECURITY.md's *Name Validation* table.

## Locations a workflow supplies

A path, glob or URL from a workflow's arguments is untrusted input choosing
where the server reads. Use `dw/locations.py`, never
`validate_path`/`validate_url` alone:

```python
from dw.locations import (
    validate_media_path,  # confined to the workflow dir / asset libraries / output root
    validate_media_glob,  # the same, on the pattern's fixed prefix
    contained_matches,    # each match re-checked on its real path
    validate_media_url,   # no loopback / link-local / private / non-global host
    validate_model_name,  # a Hub repo id, or a contained path
)
```

Containment is checked before existence, so a refusal never discloses whether
the file is there. Redirects are re-checked hop by hop (at most 5). A scheme
other than http(s) is always refused; the rest yields to `--trust-workflows`.

## Common Patterns

```python
# File paths - pass a base directory, or nothing is confined
safe_path = validate_path(user_path, base_dir="/allowed/dir")
workflow_path = validate_workflow_path("workflow.json", workflow_dir)
output_path = validate_output_path(user_path, base_output_dir)

# User input
var_name = validate_variable_name("prompt")    # OK
var_name = validate_variable_name("bad;name")  # raises InvalidInputError
value = validate_string_input(user_input, max_length=1000)
url = validate_url(user_url)                   # http/https only

# Subprocess - shell=False, a fixed argument list
import subprocess

cmd = sanitize_command_args(["python", "-m", "pip", "install", validated_spec])
subprocess.run(cmd, shell=False)
```

## Error Handling

```python
try:
    path = validate_path(user_path, base_dir=root)
except PathTraversalError as e:
    logger.error(f"Path traversal detected: {e}")
except InvalidInputError as e:
    logger.error(f"Invalid input: {e}")
```

## Input Constraints

**Variable names:** letters, digits, underscores, hyphens; must start with a
letter or underscore.

```text
OK:      prompt, num_images, my-variable, _private
Invalid: my.var, var;name, $var, var name
```

**File paths:** no `..`, no `~/`, nothing under `/dev/`, `/proc/`, `/sys/`.

```text
OK:      ./subdir/workflow.json, /full/path/workflow.json, builtin:h3_context_ir.json
Invalid: ../../../etc/passwd, ~/secret.json
```

**URLs:** http and https only, with a host and no backslash.

```text
OK:      https://example.com/image.jpg
Invalid: file:///etc/passwd, ftp://server/file
```

## Trust gate (`dw/trust.py`)

Off by default; `dw-serve --trust-workflows` (and `dw-validate`) lifts it for
the whole process. Untrusted, a workflow may not:

- import outside `diffusers`, `torch`, `torchvision`, `transformers`,
  `accelerate`, `peft`, `sdnq`, `torchao`, `optimum`, `gguf`, `bitsandbytes`,
  `dw` (`pre_load_modules`, dotted `*_type`/`*_dtype`/`dtype`/`config_type`,
  `constant:`);
- name a type that is not a constructible class (a model, pipeline,
  scheduler, tokenizer, processor, quantization config, auto factory,
  attention processor or reference dataclass), or a `dtype` key that is not
  a `torch.dtype`;
- walk a `constant:` through `_` segments or out of those packages, or reach
  a callable;
- set `trust_remote_code`, `custom_pipeline` or `custom_generate` anywhere in
  `from_pretrained_arguments`, `generate_kwargs`, `forward_params` or a task's
  arguments.

`validate_workflow` reports each refusal at its JSON path.

## HTTP (`dw/server/http_security.py`)

- `--token` / `DW_API_TOKEN` gates `/api/*` and `/mcp` (401 without it).
  `?token=` works only on GET routes marked `query_token_ok`.
- `/outputs`, `/inputs`, `/exports/*.zip` and the UI are **not** gated.
- A foreign `Origin` is a 403; an unknown `Host` is a 400 unless bound to a
  wildcard address.
- A non-loopback bind without a token warns; `--mcp` there refuses to start.
- `dw_mcp/` adds no auth; a remote `--url` needs a token. Over `dw.serve
  --mcp`, `download_output` and `upload_asset(file_path)` are confined to the
  server's workspace directories. See [MCP.md](MCP.md#security) and
  [REMOTE.md](REMOTE.md).

## Common Errors

| Error | Cause | Fix |
| ----- | ----- | --- |
| `Path contains dangerous pattern matching \.\.` | Path traversal | Use an absolute or `./` relative path |
| `Path outside allowed directory: ...` | Resolved path left `base_dir` | Put the file inside the root; for media use an `asset:` reference |
| `Invalid variable name: my.var` | Special characters | Use `my_var` |
| `Invalid URL: URL scheme not allowed: file` | Non-http scheme | Use https, or a local path |
| `Refusing a redirect to '<target>' (from '<url>'): ...` | A media URL redirected to an internal host or a non-http scheme | Host the media where it is served directly; a refused hop is never fetched |
| `Refusing to load ...`, ending with a pointer to `--trust-workflows` | The trust gate | Stay inside the allowed packages, or trust the workflow's source |
| `Argument contains dangerous characters: ...` | Shell metacharacters | Remove `` ` `` `$` `\|` `&` `;` `>` `<` |
