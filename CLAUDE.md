# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`diffusers-workflow` is a declarative workflow engine for HuggingFace Diffusers. Users define AI image/video generation pipelines in JSON — variable substitution, multi-step composition, cross-step data flow, and utility tasks — without writing Python. Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

## Common Commands

```bash
# Install
bash ./install.sh && source ./activate

# Run a workflow - sd15.json uses a small, ungated model and a literal
# prompt, so it needs no Hugging Face login and downloads only a few GB
python -m dw.run workflows/sd15.json
python -m dw.run workflows/sd15.json prompt="a cat" num_images_per_prompt=4

# A gated model (e.g. workflows/flux/FluxDev.json) needs Hugging Face auth
# first: huggingface-cli login

# Validate a workflow against schema
python -m dw.validate workflows/ZImage.json

# Basic system test (torch, diffusers import check)
python -m dw.test

# Interactive REPL
python -m dw.repl

# HTTP server + web UI (http://127.0.0.1:8765, API docs at /docs)
python -m dw.serve
```

## Architecture

### Server & Web UI

`dw/serve.py` runs a FastAPI app over the same persistent worker the REPL uses,
queueing jobs FIFO and persisting history to `~/.diffusers_helper/jobs.sqlite`.
See docs/SERVER.md, `dw/server/CLAUDE.md` and `ui/CLAUDE.md`.

### MCP Server

The stdio MCP server lives in `dw_mcp/` — see `dw_mcp/CLAUDE.md` and docs/MCP.md.

### REPL Architecture

The REPL (`dw/repl.py`) uses a **persistent worker subprocess** (`dw/worker.py`) to keep GPU models cached between runs. Communication is via `multiprocessing.Queue`. Worker management is in `dw/repl_worker.py`, command handlers in `dw/repl_commands.py`.

**Critical**: Uses `multiprocessing.set_start_method("spawn")` for CUDA/MPS compatibility.

### Workflow sources

`dw/workflow_sources.py` is the server's workflow search path: the writable
directory first (the workspace's `workflows/`), then any `--examples-dir`, each
read-only. Reads (`listing`, `find_workflow`) span every root front-to-back so an
earlier name shadows a later one; `PUT /api/workflows` always resolves through
`writable_source`, so saving something opened from a read-only root writes a copy
rather than overwriting it, and `DELETE` on a read-only root answers 403. A job
carries the root it is confined to (`JobManager.submit(workflow_dir=...)`), so an
examples workflow runs confined to the examples directory rather than to the
writable one. Packaged `dw/workflows/` is off the path - it is what `builtin:`
sub-workflow steps name, resolved in `dw/workflow.py`.

### Workspaces

`dw/workspace.py` resolves the one directory a run's content belongs to -
`workflows/`, `prompts/`, `assets/`, `outputs/`. Order: `--workspace` >
`DW_WORKSPACE` > the `workspace` setting > the working directory when it holds
any of `workflows/`, `prompts/` or `outputs/` > `~/diffusers-workspace`. A
checkout satisfies rule four, so every default lands where it did before
workspaces existed. Resolution creates nothing; an entry point about to write
calls `ensure()` (or creates the one folder it needs). `set_workspace` pins the
root *and* how it was chosen into the environment, so a spawned worker does not
read an inferred workspace back as one the user named - `get_prompt_dir` yields
to its older discovery (`./prompts`, then the walk up from the workflow file)
for an inferred workspace but not for an explicit one. `--workflow-dir`,
`--output-dir` and `--prompt-dir` each still override one folder. See
docs/WORKSPACES.md, and docs/proposals/workspaces.md for the later stages
(workflow search path, run directories, `asset:`/`output:` references).

### Type System

`arguments.py` + `type_helpers.py` handle dynamic type conversion during workflow loading:
- Keys ending in `_type` or `_dtype`, or named `dtype`, are auto-converted: `"FluxPipeline"` → loaded from `diffusers`, `"torch.bfloat16"` → `torch.bfloat16`
- Values wrapped in `{}` are escaped (stay as strings): `"{nf4}"` → `"nf4"`
- Dotted names use full module path: `"sdnq.SDNQConfig"` → `importlib.import_module("sdnq").SDNQConfig`
- Values prefixed with `constant:` read a value declared in python rather than copying it
  into JSON: `"constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES"`. Resolved
  in `realize_args`, validated by `validate_constant_name()`; anything callable is refused
- Values prefixed with `asset:` resolve to the path of a file in the asset library:
  `"asset:iris.png"` or `"asset:gyre/frames/web.mp4"`. Resolved in `realize_args` before
  every other convention (`dw/assets.py`), rooted at the library rather than the workflow
  file, confined to it, and then loaded by whatever would have loaded a path written
  there. The library is `DW_ASSET_DIR` / `--asset-dir`, else the workspace's `assets/`
  when a workspace was named, else `./assets` if it exists, else found by walking up
  from the workflow file's directory
- Values prefixed with `prompt:` load a stored prompt's `text` from the prompt library:
  `"prompt:name"` or `"prompt:folder/name"`. Resolved in `realize_args` (`dw/prompts.py`),
  rooted at the library rather than the workflow file. The library is `DW_PROMPT_DIR` /
  `--prompt-dir`, else `./prompts` if it exists, else found by walking up from the
  workflow file's directory

### Quantization Support

Quantization configs are defined per-component in workflow JSON and instantiated in `config_objects.py`. Supported frameworks: BitsAndBytes, TorchAO, GGUF, SDNQ, optimum-quanto. The `config_type` field is a free-form string — new quantization backends work automatically via dynamic import.

SDNQ pre-quantized models use a different pattern: `pre_load_modules` imports sdnq (registers with diffusers), then the entire pipeline loads from the pre-quantized repo. Optional `sdnq_optimize` applies quantized matmul post-load (CUDA/XPU only).

### Cross-Platform Device Support

`dw/__init__.py` handles device detection (CUDA > MPS > CPU) and platform-specific optimizations:
- **CUDA**: TF32 matmul, cuDNN benchmark, deterministic mode (configurable via settings)
- **MPS**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (use all unified memory), autocast warnings suppressed, attention slicing enabled by default
- **CPU**: Warning displayed

Detection is overridden by the `DW_DEVICE` environment variable (single run) or the `device` setting (standing), either of which can name a specific accelerator such as `cuda:1`. Device placement is explicit throughout — no default torch device is set, since that would build models directly in VRAM and defeat offloading. Compare backends with `get_device_type()` rather than `== "cuda"`, which a device like `cuda:1` would fail.

A step can override the device it runs on: `device` in a pipeline `configuration` (also the default for that pipeline's components), in a component `configuration`, or in a task's `arguments`.

Every device a workflow names passes through `resolve_device()`, which translates a backend this machine does not have into the one it does and warns — a `cuda` workflow runs on a Mac and an `mps` one runs on a CUDA box. Only the backend is translated: an index survives when the backend matches (`cuda:1` on a single-GPU CUDA box stays a genuine error) and is dropped when it does not. `cpu` is never rewritten, since pinning a step to the CPU is how a GPU-specific problem gets ruled out. Translation happens before anything reads the backend, so the MPS accommodations (the sequential-offload downgrade, attention slicing, the compile skip) fire for a translated device too.

A `components` entry can additionally set `residency: "on_demand"`, which rests the component on the CPU and wraps its `forward`/`encode`/`decode` to move it to the device around each call (`apply_on_demand_placement` in `pipeline.py`). The wrappers use `functools.wraps` because callers introspect the signature — MiniMax H3's denoiser picks its arguments from `signature(transformer.forward)`. It is mutually exclusive with `group_offload` on the same component, and like `group_offload` it suppresses the wholesale `pipeline.to(device)` at load.

Settings in `~/.diffusers_helper/settings.json`: `device`, `enable_tf32`, `cudnn_benchmark`, `cudnn_deterministic`, `log_level`, `log_filename`.

## Security Rules

All entry points use `dw/security.py`. When adding features:
- Validate paths with `validate_path()` / `validate_workflow_path()` / `validate_output_path()`
- Validate variable names with `validate_variable_name()` (pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$`)
- Validate URLs with `validate_url()` (http/https only)
- Sanitize subprocess args with `sanitize_command_args()`
- **Never** use `eval()`, `exec()`, or `shell=True`
- Path traversal (`../`) is blocked

## Critical Gotchas

- **Schema validation runs before variable substitution** — variable defaults must match expected JSON types (use `25` not `"25"` for numbers)
- **Cartesian product explosion** — multiple `previous_result` references multiply: 4 images × 3 masks = 12 iterations
- **Component sharing requires exact key matching** between `shared_components` and `reused_components`
- **Built-in workflows** need explicit argument mapping: `"prompt": "variable:prompt"`
- **MPS differences from CUDA**: no autocast, no bitsandbytes, no flash_attn, no triton, no torch.compile. Model offloading has less benefit on unified memory, and `"offload": "sequential"` is downgraded to `"model"` with a warning there (`place_component`) — per-submodule streaming hands back no residency when the CPU and the accelerator share one pool. `exclude_from_cpu_offload` is sequential-only and does not survive the downgrade.
- **`{}`-escaped strings** in JSON arguments: `"{nf4}"` stays as string `"nf4"`, without braces it would try to load as a type
- **A stored prompt's `text` may not begin with a reference prefix** (`variable:`, `previous_result:`, `constant:`, `asset:`, `prompt:`) — the engine rejects it to prevent double resolution or iteration expansion
- **Audio+video muxing**: pipelines that generate audio alongside video (LTX-2) have the two muxed into one `video/mp4` file with PyAV in `result.py`
- **Run directories**: each execution writes `<output_dir>/<workflow identity>/<run id>/`
  with a `manifest.json` beside its files (`dw/runs.py`, `Workflow.effective_output_dir`).
  Identity is the workflow's path under a `workflows/` tree, else its file name, else its
  `id`; the run id is `<UTC timestamp>-<8 hex of the spec>`, with a `-N` counter if taken.
  A sub-workflow inherits the parent's run directory and writes no manifest of its own.
  `--output-layout flat` / `DW_OUTPUT_LAYOUT` / the `output_layout` setting restores the
  old layout. The gallery groups a workflow's runs under one folder by stripping the run
  id (`strip_run_id`)
- **Step cache**: a process-wide singleton (`dw/step_cache.py`) consulted by every `Workflow.run`, including server jobs; entries are keyed by `(workflow id, step name)` and validated against the output
  *root*, never the per-run directory - a run directory is new every execution and would
  defeat the cache; disabled entirely when the workflow sets no `seed`; a hit reports the earlier run's files with `reused: true` and writes nothing new; `memory clear` drops it

## JSON Workflow Structure

The workflow schema is at `dw/workflow_schema.json` — read it for the full structure.

File paths in workflows are relative to the workflow file. Built-in workflows use `"builtin:filename.json"` (resolves to the packaged `dw/workflows/` — distinct from the top-level `workflows/` folder of runnable examples).
