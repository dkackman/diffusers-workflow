# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`diffusers-workflow` is a declarative workflow engine for HuggingFace Diffusers. Users define AI image/video generation pipelines in JSON — variable substitution, multi-step composition, cross-step data flow, and utility tasks — without writing Python. Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

## Common Commands

```bash
# Install
bash ./install.sh && source ./activate

# Run a workflow
python -m dw.run examples/ZImage.json
python -m dw.run examples/ZImage.json prompt="a cat" num_images_per_prompt=4

# Validate a workflow against schema
python -m dw.validate examples/ZImage.json

# Basic system test (torch, diffusers import check)
python -m dw.test

# Interactive REPL
python -m dw.repl

# HTTP server + web UI (http://127.0.0.1:8765, API docs at /docs)
python -m dw.serve

# Run all tests (580+ tests)
pytest -v

# Run a single test file or test
pytest tests/test_security.py -v
pytest -k "test_variable_substitution" -v

# Coverage
pytest --cov=dw --cov-report=html

# Format
black dw/ tests/
```

## Architecture

### Server & Web UI

`dw/serve.py` runs a FastAPI app (`dw/server/app.py`) over the same persistent worker
the REPL uses: `JobManager` (`dw/server/jobs.py`) queues jobs FIFO, streams progress
over SSE, and persists history to `~/.diffusers_helper/jobs.sqlite`. The SPA lives in
`ui/` (Svelte 5 + Vite; `npm run build` outputs `ui/dist`, which the server serves).
Introspection endpoints (`dw/introspection.py`) describe pipeline/class signatures for
the editor's forms. `dw/hub_cache.py` inventories and deletes from the HF hub cache
(the UI's Models page). Front-end checks from `ui/`: `npm run check`, `npm run lint`,
`npm test`, `npx playwright test` (e2e, starts its own server). See docs/SERVER.md.

Packaging: `pyproject.toml` (console scripts dw-run/dw-validate/dw-repl/dw-serve/dw-test);
`scripts/build_dist.sh` builds the SPA into the wheel.

### REPL Architecture

The REPL (`dw/repl.py`) uses a **persistent worker subprocess** (`dw/worker.py`) to keep GPU models cached between runs. Communication is via `multiprocessing.Queue`. Worker management is in `dw/repl_worker.py`, command handlers in `dw/repl_commands.py`.

**Critical**: Uses `multiprocessing.set_start_method("spawn")` for CUDA/MPS compatibility.

**Hierarchical commands**: `workflow load/run/reload/status/restart`, `arg set/show/clear`, `memory show/clear`, `config set/show`.

### Type System

`arguments.py` + `type_helpers.py` handle dynamic type conversion during workflow loading:
- Keys ending in `_type` or `_dtype`, or named `dtype`, are auto-converted: `"FluxPipeline"` → loaded from `diffusers`, `"torch.bfloat16"` → `torch.bfloat16`
- Values wrapped in `{}` are escaped (stay as strings): `"{nf4}"` → `"nf4"`
- Dotted names use full module path: `"sdnq.SDNQConfig"` → `importlib.import_module("sdnq").SDNQConfig`
- Values prefixed with `constant:` read a value declared in python rather than copying it
  into JSON: `"constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES"`. Resolved
  in `realize_args`, validated by `validate_constant_name()`; anything callable is refused

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
- **MPS differences from CUDA**: no autocast, no bitsandbytes, no flash_attn, no triton, no torch.compile. Model offloading has less benefit on unified memory.
- **`{}`-escaped strings** in JSON arguments: `"{nf4}"` stays as string `"nf4"`, without braces it would try to load as a type
- **Audio+video muxing**: pipelines that generate audio alongside video (LTX-2) have the two muxed into one `video/mp4` file with PyAV in `result.py`

## JSON Workflow Structure

The workflow schema is at `dw/workflow_schema.json`. Key structure:

```json
{
  "id": "workflow_name",
  "variables": { "prompt": "default value", "steps": 25 },
  "steps": [
    {
      "name": "step_name",
      "pipeline": {
        "configuration": { "component_type": "ZImagePipeline", "offload": "model" },
        "from_pretrained_arguments": { "model_name": "...", "torch_dtype": "torch.bfloat16" },
        "transformer": { "configuration": {}, "quantization_config": {}, "from_pretrained_arguments": {} },
        "loras": [{ "model_name": "...", "adapter_name": "...", "scale": 1.0 }],
        "scheduler": { "configuration": { "scheduler_type": "..." }, "from_config_args": {} },
        "arguments": { "prompt": "variable:prompt", "image": "previous_result:prev_step" }
      },
      "result": { "content_type": "image/jpeg" }
    }
  ]
}
```

Steps can also have `"task"` (with `command` + `arguments`) or `"workflow"` (with `path` + `arguments`) instead of `"pipeline"`.

Pipeline configuration options include: `pre_load_modules` (import modules before loading), `sdnq_optimize` (SDNQ quantized matmul), `enable_attention_slicing`, `disable_attention_slicing`, `attention_backend`, `group_offload`, `enable_layerwise_casting`.

File paths in workflows are relative to the workflow file. Built-in workflows use `"builtin:filename.json"` (resolves to `dw/workflows/`).
