[![CodeQL](https://github.com/dkackman/diffusers-helper/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/dkackman/diffusers-helper/actions/workflows/github-code-scanning/codeql)

# diffusers-workflow

## Introduction

A declarative workflow engine for the [Hugging Face Diffusers library](https://github.com/huggingface/diffusers). Define and execute complex image/video generation workflows using JSON configuration files—no Python coding required.

*For detailed documentation, see the [wiki](https://github.com/dkackman/diffusers-workflow/wiki).*

## Features

- **Command-line execution** with variable substitution
- **Text/image to image/video** generation workflows
- **LLM-powered** prompt augmentation and image description
- **ControlNet** image processing tasks
- **Composable workflows** from multiple JSON files
- **Utility tasks**: background removal, upscaling, cropping, QR codes

## Installation

**Tested on:** Ubuntu 22.04+ and Python 3.10+. Windows support is experimental.

### bash

```bash
bash ./install.sh
. ./activate
python -m dw.test
```

### powershell

```powershell
.\install.ps1
.\venv\scripts\activate 
python -m dw.test
```

### Run Tests

```bash
. ./activate # or .\venv\scripts\activate on windows
pip install pytest
pytest -v
```

## Usage

### Run a Workflow

```bash
python -m dw.run --help
usage: run.py [-h] [-o OUTPUT_DIR] file_name [variables ...]

Run a workflow from a file.

positional arguments:
  file_name             The filespec to of the workflow to run
  variables             Optional parameters in name=value format

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        The folder to write the outputs to
```

### Validate a Workflow Definition

```bash
python -m dw.validate --help
usage: validate.py [-h] file_name

Validate a project file.

positional arguments:
  file_name   The filespec to of the project to validate

options:
  -h, --help  show this help message and exit
```

## Interactive REPL

Interactive environment for rapid workflow iteration with **persistent GPU model caching** (2-4x faster repeated execution):

```bash
python -m dw.repl

dw> workflow load FluxDev
Loaded workflow: FluxDev

dw> arg set prompt="a beautiful sunset"
Set argument prompt=a beautiful sunset

dw> model run
Starting worker process...
[... models load once ...]
Workflow completed successfully

dw> arg set prompt="a starry night"
dw> model run
Reusing loaded models from cache
[... runs 2-4x faster! ...]
Workflow completed successfully

dw> memory show
GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 8234.5 MB
  ...
```

### Commands

Use `?` to explore hierarchical command groups:

```bash
dw> ?             # Show all groups
dw> workflow ?    # Workflow management
dw> arg ?         # Argument configuration
dw> model ?       # Model execution
dw> memory ?      # Memory management
```

**Documentation:** [Command Reference](docs/REPL_COMMANDS.md) | [Worker Architecture](docs/REPL_WORKER_GUIDE.md)

## JSON Workflow Format

**Schema:** [View JSON Schema](https://json-schema.app/view/%23?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdkackman%2Fdiffusers-workflow%2Frefs%2Fheads%2Fmaster%2Fdw%2Fworkflow_schema.json)

### Examples

#### Simple Image Generation

Variables declared in the workflow can be overridden via command-line arguments:

```bash
python -m dw.run test_workflow.json prompt="an orange" num_images_per_prompt=4
```

```json
{
    "id": "test_workflow",
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
                    "guidance_scale": 3.5,
                    "max_sequence_length": 512
                }
            },
            "result": {
                "content_type": "image/jpeg"
            }
        }
    ]
}
```

#### Multi-Step Workflow (Image → Video)

Demonstrates chaining steps: text-to-image with 4-bit quantization, then image-to-video using the generated image:

```json
{
    "id": "sd35_txt2img2vid",
    "steps": [
        {
            "name": "image_generation",
            "pipeline": {
                "transformer": {
                    "configuration": {
                        "component_type": "SD3Transformer2DModel",
                        "quantization_config": {
                            "configuration": {
                                "config_type": "BitsAndBytesConfig"
                            },
                            "arguments": {
                                "load_in_4bit": true,
                                "bnb_4bit_quant_type": "{nf4}",
                                "bnb_4bit_compute_dtype": "torch.bfloat16"
                            }
                        },
                    },
                    "from_pretrained_arguments": {
                        "model_name": "stabilityai/stable-diffusion-3.5-large",
                        "subfolder": "transformer",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "configuration": {
                    "component_type": "StableDiffusion3Pipeline",
                    "offload": "model"
                },
                "from_pretrained_arguments": {
                    "model_name": "stabilityai/stable-diffusion-3.5-large",
                    "torch_dtype": "torch.bfloat16"
                }
                "arguments": {
                    "prompt": "portrait | wide angle shot of eyes off to one side of frame, lucid dream-like 3d model of owl, game asset, blender, looking off in distance ::8 style | glowing ::8 background | forest, vivid neon wonderland, particles, blue, green, orange ::7 parameters | rule of thirds, golden ratio, asymmetric composition, hyper- maximalist, octane render, photorealism, cinematic realism, unreal engine, 8k ::7 --ar 16:9 --s 1000",
                    "num_inference_steps": 25,
                    "guidance_scale": 4.5,
                    "max_sequence_length": 512
                }
            },
            "result": {
                "content_type": "image/png"
            },
        },
        {
            "name": "image_to_video",
            "pipeline": {
                "configuration": {
                    "offload": "sequential",
                    "component_type": "CogVideoXImageToVideoPipeline",
                    "vae": {
                        "configuration": {
                            "enable_slicing": true,
                            "enable_tiling": true
                        }
                    }
                },
                "from_pretrained_arguments": {
                    "model_name": "THUDM/CogVideoX-5b-I2V",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "image": "previous_result:image_generation",
                    "prompt": "The owl stares intently and blinks",
                    "num_videos_per_prompt": 1,
                    "num_inference_steps": 50,
                    "num_frames": 49,
                    "guidance_scale": 6
                }
            },
            "result": {
                "content_type": "video/mp4",
                "file_base_name": "owl"
            }
        }
    ]
}
```

#### Prompt Augmentation with LLM

Uses a local LLM to enhance prompts before image generation. Demonstrates composable workflows via `builtin:` references.

**Note:** Requires `flash_attn` and the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

```json
{
    "variables": {
        "prompt": "A marmot, wearing a tophat in a woodland setting. Somewhat magical."
    },
    "id": "sd35",
    "steps": [
        {
            "name": "augment_prompt",
            "workflow": {
                "path": "builtin:augment_prompt.json",
                "arguments": {
                    "prompt": "variable:prompt"
                }
            },
            "result": {
                "content_type": "text/plain"
            }
        },
        {
            "name": "image_generation",
            "pipeline": {
                "configuration": {
                    "component_type": "StableDiffusion3Pipeline",
                    "offload": "sequential"
                },
                "from_pretrained_arguments": {
                    "model_name": "stabilityai/stable-diffusion-3.5-large",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "prompt": "previous_result:augment_prompt",
                    "num_inference_steps": 25,
                    "guidance_scale": 4.5,
                    "max_sequence_length": 512,
                    "num_images_per_prompt": 1
                }
            },
            "result": {
                "content_type": "image/jpeg"
            }
        }
    ]
}
```
