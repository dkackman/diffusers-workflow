# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`diffusers-workflow` is a declarative workflow engine for HuggingFace Diffusers. Users define AI image/video generation pipelines in JSON — variable substitution, multi-step composition, cross-step data flow, and utility tasks — without writing Python. Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

**Version:** 0.37.0 | **Python:** 3.10-3.14 | **PyTorch:** 2.0+

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

# Run all tests (260+ tests)
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

### Core Data Flow

```
JSON workflow → schema validation → variable substitution → sequential step execution → results
```

1. `workflow.py` loads JSON, validates against `workflow_schema.json`, substitutes `variable:name` references
2. `step.py` executes each step — generating argument combinations via `previous_results.py` (cartesian product of `previous_result:step_name` references)
3. Each step dispatches to one of: **Pipeline** (HuggingFace inference), **Task** (utility operation), or **Sub-Workflow** (recursive)
4. `result.py` saves outputs as `{output_dir}/{workflow_id}-{step_name}.{index}.{ext}` — supports image, video, audio, text, and JSON content types. Optional `embed_metadata` stores generation parameters in PNG info chunks or JPEG/WebP EXIF.

### Key Modules

| Module | Role |
|--------|------|
| `dw/workflow.py` | Orchestrator: load, validate, variable substitution, step sequencing |
| `dw/step.py` | Step executor: generates iterations, dispatches to pipeline/task/workflow |
| `dw/pipeline_processors/pipeline.py` | Pipeline loading, components, quantization, LoRA, schedulers, offloading |
| `dw/pipeline_processors/config_objects.py` | Quantization and group offload config creation |
| `dw/tasks/task.py` | Task dispatcher (image processing, QR codes, gathering, video, segmentation, captioning, text generation, diffusion upscaling, frame interpolation) |
| `dw/tasks/segment.py` | GroundingDINO + SAM2 text-prompted object segmentation |
| `dw/tasks/image_to_text.py` | Image captioning via transformers image-to-text pipeline (BLIP, BLIP-2, etc.) |
| `dw/tasks/text_generation.py` | Text generation / prompt expansion via transformers text-generation pipeline |
| `dw/tasks/diffusion_upscale.py` | Diffusion-based image upscaling via SD upscale pipelines (x2/x4) |
| `dw/tasks/interpolate_frames.py` | RIFE frame interpolation (2x/4x/8x) with vendored IFNet v4.6 |
| `dw/tasks/rife_model.py` | Vendored RIFE IFNet v4.6 architecture (MIT License, Megvii Inc.) |
| `dw/previous_results.py` | Cross-step data flow via cartesian products |
| `dw/arguments.py` | Argument processing, resource loading, dynamic type conversion |
| `dw/type_helpers.py` | Dynamic type loading: `"FluxPipeline"` → class, `"torch.bfloat16"` → dtype |
| `dw/security.py` | Path validation, input sanitization, URL validation, command safety |
| `dw/variables.py` | Variable substitution system |
| `dw/schema.py` | JSON schema validation |
| `dw/settings.py` | User settings from `~/.diffusers_helper/settings.json` |

### REPL Architecture

The REPL (`dw/repl.py`) uses a **persistent worker subprocess** (`dw/worker.py`) to keep GPU models cached between runs. Communication is via `multiprocessing.Queue`. Worker management is in `dw/repl_worker.py`, command handlers in `dw/repl_commands.py`.

**Critical**: Uses `multiprocessing.set_start_method("spawn")` for CUDA/MPS compatibility.

**Hierarchical commands**: `workflow load/run/reload/status/restart`, `arg set/show/clear`, `memory show/clear`, `config set/show`.

### Type System

`arguments.py` + `type_helpers.py` handle dynamic type conversion during workflow loading:
- Keys ending in `_type` or `_dtype` are auto-converted: `"FluxPipeline"` → loaded from `diffusers`, `"torch.bfloat16"` → `torch.bfloat16`
- Values wrapped in `{}` are escaped (stay as strings): `"{nf4}"` → `"nf4"`
- Dotted names use full module path: `"sdnq.SDNQConfig"` → `importlib.import_module("sdnq").SDNQConfig`

### Quantization Support

Quantization configs are defined per-component in workflow JSON and instantiated in `config_objects.py`. Supported frameworks: BitsAndBytes, TorchAO, GGUF, SDNQ, optimum-quanto. The `config_type` field is a free-form string — new quantization backends work automatically via dynamic import.

SDNQ pre-quantized models use a different pattern: `pre_load_modules` imports sdnq (registers with diffusers), then the entire pipeline loads from the pre-quantized repo. Optional `sdnq_optimize` applies quantized matmul post-load (CUDA/XPU only).

### Cross-Platform Device Support

`dw/__init__.py` handles device detection (CUDA > MPS > CPU) and platform-specific optimizations:
- **CUDA**: TF32 matmul, cuDNN benchmark, deterministic mode (configurable via settings)
- **MPS**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (use all unified memory), autocast warnings suppressed, attention slicing enabled by default
- **CPU**: Warning displayed

Settings in `~/.diffusers_helper/settings.json`: `enable_tf32`, `cudnn_benchmark`, `cudnn_deterministic`, `log_level`, `log_filename`.

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
- **MPS differences from CUDA**: no autocast, no xformers, no bitsandbytes, no flash_attn, no triton. Model offloading has less benefit on unified memory.
- **`{}`-escaped strings** in JSON arguments: `"{nf4}"` stays as string `"nf4"`, without braces it would try to load as a type

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
