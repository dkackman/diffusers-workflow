[![CodeQL](https://github.com/dkackman/diffusers-helper/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/dkackman/diffusers-helper/actions/workflows/github-code-scanning/codeql)

# diffusers-workflow

## Introduction

This is a simple, declaritive workflow engine for the [Huggingface Diffuser project](https://github.com/huggingface/diffusers). This command-line tool simplifies working with the Hugging Face Diffusers library by providing a flexible, JSON-based interface for running generative AI models. Users can define and execute complex image generation tasks without writing custom Python code.

*Please [refer to the wiki](https://github.com/dkackman/diffusers-workflow/wiki) for more detailed instuctions.*

## Features

- Make any workflow command line executable with variable substitution
- Suppport for text to image & video and image to image & video workflows
- Image describing and prompt augmentation using locally installed LLMs
- Image processing tasks for controlnet workflows
- Composable workflows from multiple files
- Other helpful tasks like background removal, upscaling, cropping and qr code generation

## Installation

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

### Install Diffusers from Source

The install script will install [the diffusers library from PyPi](https://pypi.org/project/diffusers/). If you want to install from source and use not yet released diffusers, you can do so with the following commands:

```bash
. ./activate # or .\venv\scripts\activate on windows
pip install git+https://github.com/huggingface/diffusers
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

## JSON Input Format

### Schema

[Json schema](https://json-schema.app/view/%23?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdkackman%2Fdiffusers-workflow%2Frefs%2Fheads%2Fmaster%2Fdw%2Fworkflow_schema.json)

### Examples

#### Simple Image Generation with an Input Variable

This example declares a variable for the `prompt` which can then be set on the command line. The `prompt` variable is then used in the `prompt` argument of the model.

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

#### Multiple Step Workflow

This example demonstrates a multiple step workflow including an image generation step followed by video generation. It includes the use of a transformer model for the image generation, a quantization example and an image to video model.

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

#### Prompt Augmentation

This example uses an instruct LLM to augment the prompt before passing it to the model. This also demonstrates composable child workflows.

Note that this particular LLM requries `flash_attn` which in turn requires the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

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
