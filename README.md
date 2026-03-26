[![CodeQL](https://github.com/dkackman/diffusers-workflow/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/dkackman/diffusers-workflow/actions/workflows/github-code-scanning/codeql)

# diffusers-workflow

A declarative workflow engine for the [Hugging Face Diffusers library](https://github.com/huggingface/diffusers). Define and execute image/video generation pipelines using JSON configuration files — no Python coding required.

**Python 3.10-3.14 | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU**

## Features

- **Declarative JSON workflows** with variable substitution and cross-step data flow
- **Multi-step pipelines** — chain text-to-image, image-to-video, inpainting, ControlNet
- **Quantization** — BitsAndBytes, TorchAO, GGUF, SDNQ, optimum-quanto
- **Inference acceleration** — TeaCache, FirstBlockCache, FasterCache, MagCache, TaylorSeerCache
- **Prompt weighting** — A1111-style `(word:1.5)` syntax with long prompt support
- **LoRA and IP-Adapter** support
- **Composable workflows** from multiple JSON files with `builtin:` references
- **Utility tasks** — upscaling, face restoration, segmentation, captioning, frame interpolation, QR codes, and more
- **Metadata embedding** — store generation parameters in PNG/JPEG/WebP for reproducibility
- **Interactive REPL** with persistent GPU model caching (2-4x faster iteration)
- **Cross-platform** — CUDA, MPS (Apple Silicon), and CPU

## Installation

### Linux / macOS

```bash
bash ./install.sh
source ./activate
python -m dw.test
```

### Windows

```powershell
.\install.ps1
.\venv\scripts\activate
python -m dw.test
```

The install scripts detect your Python version, create a virtual environment, and install all dependencies including platform-specific packages (bitsandbytes/kernels on CUDA, fp4-fp8-for-torch-mps on macOS).

## Usage

### Run a Workflow

```bash
python -m dw.run examples/FluxDev.json
python -m dw.run examples/FluxDev.json prompt="a cat" num_images_per_prompt=4
```

### Validate a Workflow

```bash
python -m dw.validate examples/FluxDev.json
```

### Interactive REPL

```bash
python -m dw.repl
```

```text
dw> workflow load FluxDev
dw> arg set prompt="a beautiful sunset"
dw> model run
[... models load once ...]

dw> arg set prompt="a starry night"
dw> model run
Reusing loaded models from cache
[... 2-4x faster ...]

dw> memory show
dw> ?               # show all command groups
```

See [REPL Commands](docs/REPL_COMMANDS.md) and [Worker Guide](docs/REPL_WORKER_GUIDE.md).

## Workflow Examples

### Simple Image Generation

```json
{
    "id": "flux_example",
    "variables": {
        "prompt": "an apple",
        "num_images_per_prompt": 1
    },
    "steps": [
        {
            "name": "main",
            "pipeline": {
                "configuration": {
                    "component_type": "FluxPipeline",
                    "offload": "sequential"
                },
                "from_pretrained_arguments": {
                    "model_name": "black-forest-labs/FLUX.1-dev",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "prompt": "variable:prompt",
                    "num_inference_steps": 25,
                    "num_images_per_prompt": "variable:num_images_per_prompt",
                    "guidance_scale": 3.5
                }
            },
            "result": {
                "content_type": "image/jpeg"
            }
        }
    ]
}
```

Override variables from the command line:

```bash
python -m dw.run flux_example.json prompt="an orange" num_images_per_prompt=4
```

### Multi-Step Workflow (Image to Video)

Chain steps using `previous_result:step_name` to pass outputs between steps:

```json
{
    "id": "img2vid",
    "steps": [
        {
            "name": "image_generation",
            "pipeline": {
                "configuration": {
                    "component_type": "StableDiffusion3Pipeline",
                    "offload": "model"
                },
                "from_pretrained_arguments": {
                    "model_name": "stabilityai/stable-diffusion-3.5-large",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "prompt": "a luminous owl in a neon forest",
                    "num_inference_steps": 25,
                    "guidance_scale": 4.5
                }
            },
            "result": { "content_type": "image/png" }
        },
        {
            "name": "video",
            "pipeline": {
                "configuration": {
                    "component_type": "CogVideoXImageToVideoPipeline",
                    "offload": "sequential",
                    "vae": { "configuration": { "enable_slicing": true, "enable_tiling": true } }
                },
                "from_pretrained_arguments": {
                    "model_name": "THUDM/CogVideoX-5b-I2V",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "image": "previous_result:image_generation",
                    "prompt": "The owl blinks slowly",
                    "num_inference_steps": 50,
                    "num_frames": 49,
                    "guidance_scale": 6
                }
            },
            "result": { "content_type": "video/mp4" }
        }
    ]
}
```

### Inference Acceleration

Speed up generation with built-in diffusers caching or TeaCache:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "cache": { "type": "first_block", "threshold": 0.05 }
}
```

```json
"configuration": {
    "component_type": "FluxPipeline",
    "teacache": { "rel_l1_thresh": 0.6 }
}
```

### Prompt Weighting

Use A1111-style syntax for per-token weighting:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "prompt_weighting": true
}
```

```text
a (photorealistic:1.4) portrait with (bright red hair:1.3) and [freckles]
```

## JSON Schema

**Interactive schema browser:** [View Schema](https://json-schema.app/view/%23?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdkackman%2Fdiffusers-workflow%2Frefs%2Fheads%2Fmaster%2Fdw%2Fworkflow_schema.json)

See [examples/](examples/) for more workflow files.

## Documentation

### Guides

- [Workflow Guide](docs/WORKFLOW_GUIDE.md) — JSON structure, variables, steps, data flow
- [Quantization](docs/QUANTIZATION.md) — BitsAndBytes, TorchAO, GGUF, SDNQ
- [Inference Acceleration](docs/ACCELERATION.md) — FirstBlockCache, MagCache, TaylorSeer, TeaCache
- [LoRA](docs/LORAS.md) — Loading and stacking LoRA adapters
- [IP-Adapter](docs/IP_ADAPTER.md) — Image-prompt conditioning
- [Prompt Weighting](docs/PROMPT_WEIGHTING.md) — A1111-style syntax
- [Tasks](docs/TASKS.md) — Image processing, ControlNet preprocessors, utilities

### Reference

- [REPL Commands](docs/REPL_COMMANDS.md) — Interactive REPL command reference
- [Worker Guide](docs/REPL_WORKER_GUIDE.md) — GPU persistence and troubleshooting
- [Dependencies](docs/DEPENDENCIES.md) — Installation details
- [Security](docs/SECURITY.md) — Security model
- [Testing](docs/TESTING.md) — Running the test suite
