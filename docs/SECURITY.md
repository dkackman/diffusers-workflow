# Security

## Overview

diffusers-workflow validates file paths, names, user input and URLs against
path traversal, injection, server-side request forgery and resource
exhaustion. The validators are in `dw/security.py`, the trust gate for what a
workflow may import or execute is `dw/trust.py`, the policy for where a
workflow may read and fetch is `dw/locations.py`, and the HTTP layer is
`dw/server/http_security.py`. [SECURITY_QUICKREF.md](SECURITY_QUICKREF.md) is
the one-page digest.

## Security Module (`dw/security.py`)

Every validator raises `PathTraversalError` or `InvalidInputError` (both
`SecurityError`) on refusal.

### Path Validation

- `validate_path(path, base_dir=None, allow_create=True)` - Refuses an empty
  path, a null byte, a path over 4096 characters or a file name over 255,
  `..` anywhere, `~/` (or `~\`), and a path rooted at `/dev/`, `/proc/` or
  `/sys/`. Returns the absolute, realpath'd path. With `base_dir`, the
  resolved path must equal it or lie under it (`PathTraversalError`
  otherwise); symlinks are resolved on both sides first. Without `base_dir` it
  confines nothing, and CodeQL still reports the path. `allow_create=False`
  also requires the path to exist.
- `validate_workflow_path(path, workflow_dir=None)`,
  `validate_prompt_path(path, prompt_dir)`, `validate_lora_path(path, lora_dir)`
  - `validate_path` with `allow_create=False` plus a required `.json`
  extension.
- `validate_output_path(path, output_dir)` - `validate_path` with
  `allow_create=True`, for files and directories that need not exist yet.
- `validate_file_extension(path, allowed)` - The extension against an allowed
  set (`ALLOWED_*_EXTENSIONS`).
- `contained(path, root)` - Whether `path`, symlinks resolved, lies inside
  `root`, for directory walks that must not follow a link out.
- `validate_weight_name(name, ...)` - A file name inside a Hub repo: no
  leading `/` or `~`, no backslash or drive, no empty, `.` or `..` segment.

### Name Validation

Each name is checked before it is joined onto a directory; containment is
then checked by the `validate_path` call that joins it. Every segment starts
with a letter, digit or underscore, which excludes `..`, hidden names and
absolute paths. Patterns end in `\Z`, so a trailing newline is refused.

| Validator | Shape | Max |
| --- | --- | --- |
| `validate_variable_name` | `^[a-zA-Z_][a-zA-Z0-9_-]*$` | 100 |
| `validate_prompt_reference`, `validate_lora_name` | a name, at most one folder deep | 200 |
| `validate_asset_reference` | a file name with its extension, at most four folders deep; `@` allowed | 400 |
| `validate_output_reference` | `<workflow>/<run>/<file>`, 2 to 7 segments; `@` allowed | 500 |
| `validate_output_prefix` | `<workflow>/<run id>` or `<workflow>` | 500 |
| `validate_subfolder` | relative segments under the run directory; no backslash | 200 |
| `validate_workspace_name` | one segment, not one of the workspace root's own folder names | 100 |
| `validate_constant_name` | a dotted Python name, identifiers and dots only | 200 |
| `validate_commit_hash` | 7 to 40 hex characters (the diffusers updater's install pin) | 40 |
| `validate_file_base_name` | no `/` or `\` | - |
| `validate_content_type` | a MIME type with exactly one `/` | - |

### Input Validation

- `validate_string_input(value, max_length=1000)` - Refuses an over-long
  value, a null byte, and any control character other than tab, newline and
  CR. A caller-supplied variable value (`dw/variables.py`, `dw/run.py`) is held
  to `MAX_VARIABLE_VALUE_LENGTH`, 20,000 characters. A variable's *default* is
  not capped separately; the whole workflow file is.
- `validate_json_size(path)` - A JSON file over 50MB is refused before it is
  read.
- `validate_url(url)` - Scheme `http` or `https`, a non-empty host, and no
  backslash anywhere: `urllib.parse` and the HTTP client disagree on which
  host `http://169.254.169.254\@example.com/` names, so a `\` that belongs in
  a path is written `%5C`. It checks shape only; the host policy is
  `dw/locations.py`'s.
- `safe_join_path(*parts)` - Joins components after refusing any containing
  `..`, `/` or `\`. Not called anywhere in `dw/`.

### Command Sanitization

- `sanitize_command_args(args)` - Refuses an argument containing `` ` ``,
  `$`, `|`, `&`, `;`, `>`, `<`, newline or CR. It does not quote: with
  `shell=False` the OS separates arguments, so this is defense in depth.
- `dw/` runs three subprocesses, all `shell=False` with fixed argument lists:
  `pip` in the diffusers updater (`dw/server/updater.py`, through
  `sanitize_command_args`, its commit pinned by `validate_commit_hash`),
  `sysctl` for the Apple chip name (`dw/__init__.py`) and `nvidia-smi` for the
  driver version (`dw/server/sysinfo.py`). The worker is a
  `multiprocessing.Process`, not a shelled-out command.

## Trust model

**A workflow JSON file can execute arbitrary Python.** The dynamic-import
machinery that lets a workflow name any diffusers pipeline, scheduler or
quantization backend also makes loading a workflow more than a data-load.
Three things in a workflow run `importlib.import_module()` on a name the file
supplies, which executes that module's top-level code:

- **`pre_load_modules`** (`dw/pipeline_processors/pipeline.py`) - modules
  imported before the pipeline loads, for their registration side effects
  (`sdnq` registering its quantization method).
- **A dotted `*_type`/`*_dtype`/`dtype`/`config_type` value**
  (`dw/type_helpers.py`, reached from `dw/arguments.py` and
  `dw/pipeline_processors/config_objects.py`) - `"sdnq.SDNQConfig"` imports
  `sdnq` and reads `SDNQConfig` off it.
- **A `constant:` reference** (`dw/type_helpers.py`'s
  `load_constant_from_name`) - imports the module the constant is declared in
  before reading the attribute.

**Treat an untrusted workflow file exactly like an untrusted Python script.**
Do not run one from a source you would not run a `.py` file from: a random
download, a link in an issue, an LLM-authored file you have not read.

### `--trust-workflows`

`dw-serve` and `dw-validate` take `--trust-workflows`, **off by default**.
`dw-run` is a client of `dw-serve`, so its jobs run under the server's
setting. The choice is process-wide, reaches the spawned worker through
`DW_TRUST_WORKFLOWS`, and `GET /api/server` reports it as `trust_workflows`.
It is not scoped per workflow or per request: a server that accepts jobs from
anything other than yourself, an MCP client included, should run without it.

Untrusted (the default), these are refused, each with an error naming what
triggered it and pointing at `--trust-workflows`:

- **A `pre_load_modules` entry or dotted type outside the allowed packages**
  (`require_trusted_dotted_name`). Allowed: `diffusers`, `torch`,
  `torchvision`, `transformers`, `accelerate`, `peft`, the quantization
  backends `sdnq`, `torchao`, `optimum` (optimum-quanto), `gguf`,
  `bitsandbytes`, and `dw` itself (for `dw.community_pipelines`). This is
  checked before anything is imported.
- **A type reference that is not a constructible class**
  (`require_loadable_type`, `require_constructible_class`). A `*_type` value
  is constructed with the workflow's own arguments, so it must be a class
  defined inside an allowed package (not merely re-exported by one) and one
  of: a subclass of a diffusers or transformers model, pipeline, modular
  pipeline, scheduler, tokenizer, processor, feature extractor, image
  processor or quantization config, or of torchao's `AOBaseConfig`; a class in
  an auto-pipeline/auto-model factory module; a dataclass in
  `diffusers.pipelines` or `diffusers.modular_pipelines` with no unreviewed
  `__post_init__`; or a diffusers attention processor. `"torch.hub.load"` is in
  `torch` and is refused as not a class. Under a `dtype` or `*_dtype` key a
  `torch.dtype` (`"torch.bfloat16"`) is accepted too. A bare name
  (`"FluxPipeline"`) resolves against `diffusers` under the same rules.
- **A `constant:` walk that leaves the allowed packages.** The module is gated
  like a type before it imports; no segment may start with `_` (checked before
  any import); no module the walk passes through may sit outside the allowed
  packages (`constant:torch.os.environ` is refused at `torch.os`); and
  `fetch_constant` refuses anything callable. Reading a field off a value
  declared in an allowed module works. A bare `constant:SOME_NAME` reads from
  `diffusers`.
- **Remote code in `from_pretrained_arguments`**: `trust_remote_code` and
  `custom_pipeline`, for every component and pipeline
  (`require_trusted_from_pretrained_arguments`). Neither goes through an
  import of ours, so the dotted-name gate alone would leave them open.
- **Remote code in a model's `generate()`**: `custom_generate` (a Hub repo
  whose `custom_generate/generate.py` transformers downloads and runs) and
  `trust_remote_code`, in `text_generation`'s `generate_kwargs` and in
  `speech_generation`'s `generate_kwargs` and `forward_params`
  (`require_trusted_generate_arguments`), before the model loads.
  `text_generation`'s `generate_kwargs` and `chat_template_kwargs` hold only
  scalars or lists of them, and none may be one of the pipeline call's own
  arguments (`tokenizer_encode_kwargs`, `images`, a template).
- **Remote-code keys anywhere in a task's arguments**: `trust_remote_code`,
  `custom_pipeline` or `custom_generate`, top level or nested, is a validation
  error naming the key and its path, and `Task.run` repeats the check for
  every command before its handler runs (`require_trusted_task_arguments`,
  GHSA-hhfj-p9xj-xjmr). Trusted, the same keys given as a step argument
  outside `generate_kwargs`/`forward_params` are still refused, as arguments
  the task does not take.

`validate_workflow` reports each refusal at the key's path (a variable's
default at `variables.<name>`), so nothing is queued to find out. Every
bundled workflow loads untrusted: `tests/test_workflow_trust.py` sweeps the
catalog, and `tests/test_catalog_structure.py` refuses a catalog entry that
sets `trust_remote_code` or `custom_pipeline`. `--trust-workflows` lifts all
of the above except the trusted-mode argument check.

Model code that dw vendors runs no code from its repo, trusted or not:

- **Audio8-TTS** (`speech_generation`, `dw/vendored/audio8_tts_model.py`):
  five files fetched by `hf_hub_download` at one pinned 40-hex revision;
  safetensors weights; the `codec.pth` pickle read with
  `torch.load(weights_only=True)`. Its `generate_kwargs` / `forward_params`
  pass the trust gate, then an allowlist (`temperature`, `top_p`, `top_k`,
  `do_sample`, `max_new_tokens`); any other key is refused before the model
  loads.
- **MMAudio** (`dw/vendored/mmaudio_model.py`,
  `dw/vendored/mmaudio_sync_model.py`, `dw/tasks/mmaudio_decoder.py`): the
  network and Synchformer encoder vendored at upstream commit `974010a0`. The
  two pickle checkpoints (`weights/mmaudio_large_44k_v2.pth`,
  `ext_weights/synchformer_state_dict.pth`) are pinned to one 40-hex revision
  and read with `torch.load(weights_only=True)` and a strict load. The VAE and
  vocoder are Kandinsky 6 safetensors at a pinned sha, built from
  `config.json` read as JSON. The modules take local paths and never fetch.

`text_generation`'s `system_prompt` may be a Hub file (`{"from_hub": {...}}`),
pinned to a 40-hex commit. Before every use the Hub is asked, without
credentials (`token=False`), whether it serves the file publicly, and the
download carries no token, so the server's login never reaches a gated or
private file on a workflow's say-so. It fails closed: any error from that
ask, being offline, or `HF_HUB_OFFLINE` refuses the run. It is capped at 1
MiB, checked against the Hub's reported size before the download (no
reported size is a refusal) and against the file after it.

### Where a workflow may read and reach (`dw/locations.py`)

A workflow's arguments also choose *locations*: which file to open, which URL
to fetch, which directory a glob expands over. Untrusted:

- **A path** must resolve inside a root this installation works in: the
  workflow file's own directory, the asset libraries on the search path, or
  the output root. A relative path that climbs out is refused on its `..`
  segments, at validation as well as at the loader (#124). Containment is
  checked **before** existence, so a refusal is not a file-existence oracle.
  The remedy for a file outside is an `asset:` reference. Validation finds a
  path by its argument's name (`image`, `*_video`, `location`, ...) or by a
  task's declaration (`@register_command(media_arguments=...)`, read through
  `TASK_MEDIA_ARGUMENTS`, #692). `apply_lut`'s `lut`, `paste_face_track`'s
  `track` and `restore_to_source`'s `fit` are read only from a server file,
  so a URL there is refused (`LOCAL_ONLY_TASK_ARGUMENTS`).
- **A URL with any other scheme than http(s)** (`file://`, `s3://`, ...) is
  refused at validation and at the loader, trusted or not (#618).
- **A glob** is contained on the fixed directory its pattern starts from, and
  every match is re-checked on its real path, so a symlink cannot carry the
  expansion out.
- **An http(s) URL** must not resolve to an address inside the deployment:
  loopback, link-local (`169.254.0.0/16`, cloud metadata), private ranges, and
  anything else not globally routable (`is_global`), checked after DNS
  resolution. That includes `100.64.0.0/10`, **Tailscale's tailnet range**: a
  workflow that fetches media from another tailnet machine needs
  `--trust-workflows`.
- **Every redirect is re-checked.** A media fetch (`safe_get` in
  `dw/outbound.py`) follows at most 5 hops (`MAX_MEDIA_REDIRECTS`), each
  `Location` passing the same scheme and host policy before it is dialed. A
  refused hop is never fetched, and its error begins `Refusing a redirect to
  '<target>' (from '<url>'): ` followed by the rule; an upstream error status
  keeps the HTTP client's own message (#1055). Each hop's host is resolved
  once, by the policy, and dialed at the checked address, so DNS rebinding
  reaches only what was checked; a host that does not resolve or is
  percent-encoded is refused. An untrusted fetch ignores proxy and `~/.netrc`
  settings. The body is capped at `MAX_MEDIA_BYTES` (1 GiB) and the fetch at
  `MEDIA_TOTAL_TIMEOUT` (600 s), closing the connection at the deadline.
  `remote_text_encoder` POSTs through the same path. Two limits: a trusted
  fetch through a SOCKS proxy is not deadline-bounded, and the TLS handshake
  is bounded only by the per-operation timeout.
- **`remote_text_encoder.url`** is https-only, and the Hugging Face token is
  attached only for `huggingface.co`, `huggingface.cloud` and `hf.space`. A
  redirect that leaves the host, or drops to http, is followed without the
  token.
- **`model_name`** must be a Hub repo id or a contained path; a URL is refused
  (#117).

`location_errors` runs inside `validation_errors`, so `validate_workflow`
refuses before a model load is spent, and the loaders call the same functions
for a location that arrives through a variable or a previous result. All of
it yields to `--trust-workflows`, except the scheme rule.

## HTTP authentication and headers (`dw/server/http_security.py`)

- **Bearer token, opt-in.** With `--token` / `DW_API_TOKEN` set, every
  `/api/*` and `/mcp` request needs `Authorization: Bearer <token>`, compared
  in constant time; anything else is a 401. Without a token there is no
  authentication.
- **Query token.** Only a GET or HEAD route marked `query_token_ok`
  (EventSource streams, thumbnails, download links, which a browser loads
  without headers) accepts `?token=`. The marker is matched on the real route,
  not a path suffix.
- **Not gated by the token:** the web UI's static files, `/outputs`, `/inputs`
  and `/exports/{id}.zip` (#592). Anyone who can reach the port can read
  generated files and input assets by name.
- **Binding.** The default bind is `127.0.0.1`. A non-loopback bind without a
  token logs a warning and starts; `--mcp` on a non-loopback host without a
  token exits 2. The connection is plaintext HTTP; use a network you control,
  Tailscale, or a TLS proxy ([REMOTE.md](REMOTE.md)).
- **Origin check.** A request carrying an `Origin` is refused (403) unless
  its host is a loopback name, the bind host, or the host the request was
  addressed to. This stops a drive-by web page queuing jobs, and DNS
  rebinding, where Origin and Host differ. A request without `Origin` passes.
- **Host check.** Unless bound to a wildcard address, a `Host` that is not a
  loopback name or the bind host is a 400.
- **Response headers.** Every response carries `X-Content-Type-Options:
  nosniff` and `X-Frame-Options: DENY` (#407).

## Integration Points

| Entry point | What's validated |
| --- | --- |
| `workflow.py` | Workflow file paths, JSON size, output directories, sub-workflow paths |
| `run.py`, `validate.py` | CLI arguments, variable names and values |
| `arguments.py`, `argument_media.py` | Media locations, file paths and extensions (`validate_media_location`; `fetch_image`, `fetch_video`) |
| `locations.py`, `outbound.py` | Every location a workflow supplies, and every outbound media fetch |
| `tasks/gather.py` | URLs passed to the `gather` task |
| `result.py`, `writers.py`, `content_types.py` | Output directories, file names (`output_file_path`), result content types |
| `server/http_security.py` | Token, Origin and Host on every request |
| `server/jobs.py`, `server/outputs.py`, `server/routes/*.py` | Every HTTP-supplied path: workflows confined to the workflow directories, gallery files to the output directory, assets to their library, an inline workflow's `base_dir` |

## MCP Server

`dw_mcp/` adds no authentication and no file access of its own, with two
exceptions below. It is an HTTP client of `dw.serve`: every path a tool sends
(a workflow name, a gallery file, a job id) is validated by the REST API
exactly as a browser request from the web UI is. A remote `dw.serve` is
allowed only with a token ([MCP Server](MCP.md#security),
[REMOTE.md](REMOTE.md)).

`download_output` writes a local file and `upload_asset(file_path=...)` reads
one, outside the API. Over a **stdio `dw-mcp`**, "local" is the user's own
machine, and both act as the user would with a shell redirect. Over
**`dw.serve --mcp`** it is the operator's box, so both are confined:
`download_output`'s `destination` to the workspace (#113), and
`upload_asset`'s `file_path` to the server's workspace, workflows, assets,
outputs and prompts directories (#138), with the refusal ordered before the
existence and extension checks so it is not a path-existence oracle. On both
transports a `..` segment in `destination` is refused and an existing file is
kept unless `overwrite=True`.

Nothing in `dw_mcp/` inspects what a workflow it saves (`save_workflow`) or
runs (`run_workflow`) contains. An MCP-authored or MCP-submitted workflow is
protected by `dw-serve`'s untrusted default, the same gate as any other
untrusted workflow. `dw-serve --trust-workflows` removes that gate for every
job the server accepts.

## Exception Hierarchy

```text
SecurityError
  PathTraversalError     - path traversal attempt
  InvalidInputError      - input validation failure
  UntrustedWorkflowError - an untrusted workflow reached the code-execution surface
```

### Disk caches

`DELETE /api/caches/{name}` (`dw/disk_caches.py`) empties a directory whose
path comes from an environment variable, so it clears only a root carrying a
`.dw-cache` marker. `dw.serve` writes the marker at start into a root that is
missing or empty (a missing `triton` root is never created), or into a
populated root equal to the cache's default exactly (`~/.cache/dw/torchinductor`
or its `$XDG_CACHE_HOME` form; `~/.triton/cache` only, not a populated
`$TRITON_HOME/.triton/cache`). Any other populated root is listed with
`clearable: false` and its clear is a 400 until someone creates the marker by
hand; deleting the marker stops clearing. `quantized`
(`DW_QUANTIZED_CACHE_DIR`) follows the same rule, and
`DELETE /api/caches/{name}/entries/{entry}` deletes one entry only if the id
matches `^[0-9a-f]{64}$`, checked before any path is built. A clear refuses
while a job is running or queued (for `quantized`, also while a model download
is active), and renames the root's children into `.clearing-*` before deleting
them. Like model deletion, it needs only the bearer token.

## Rules

- Validate a path before any file operation, with a base directory.
- Validate a URL with `validate_url()`, and a workflow-supplied location with
  `dw/locations.py`, before loading a remote resource.
- Run a subprocess with `shell=False` and its arguments through
  `sanitize_command_args()`.
- Never use `eval`, `exec` or `shell=True`.

## Protected Against

- **Path traversal**: no file outside the allowed directories.
- **Command injection**: no shell interpretation anywhere in `dw/`.
- **Arbitrary code from a workflow**: the trust gate, unless
  `--trust-workflows`.
- **Arbitrary file read through a media argument**: a workflow-supplied
  location is confined to the roots it may read (`dw/locations.py`).
- **Malicious URLs and SSRF**: http/https only, and an untrusted workflow may
  not reach a host inside the deployment, including through a redirect.
- **Resource exhaustion**: 50MB workflow JSON, 1 GiB media fetch with a 600 s
  deadline.
- **Decompression bombs**: an image a caller names is decoded at no more than
  `MAX_DECODE_PIXELS` (50M; an 8K frame is 33M), checked after `Image.open`
  and before any decode. `get_output_image` (crops included) refuses one, the
  gallery thumbnail answers 413, `compare_outputs` refuses a video whose frame
  exceeds it (413), and embedded metadata is read from PNG header chunks
  without decoding. Other video decodes and all audio decodes are not limited.
- **Script on the UI's origin through an output**: a result `content_type`
  may not be an active document type (`text/html`, `application/xhtml+xml`,
  `text/xml`, `application/xml`, `image/svg+xml`, compared without case or
  parameters). Validation refuses it at `steps[i].result.content_type`, the
  writer refuses it again (`dw/content_types.py`), and a file of an active type
  that reaches `/outputs` or `/inputs` anyway is served with
  `Content-Security-Policy: sandbox`.

Not protected: generated files and input assets are readable without the
token by anyone who can reach the port, and traffic is plaintext.

## Testing

```bash
pytest tests/test_security.py tests/test_security_auth.py tests/test_security_trust_gate.py \
  tests/test_security_symlinks.py tests/test_locations.py tests/test_outbound.py \
  tests/test_workflow_trust.py -v
```
