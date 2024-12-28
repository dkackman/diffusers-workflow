[![CodeQL](https://github.com/dkackman/diffusers-helper/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/dkackman/diffusers-helper/actions/workflows/github-code-scanning/codeql)

# diffusers-helper

## Introduction

This is a helper for the [Huggingface Diffuser project](https://github.com/huggingface/diffusers). This command-line tool simplifies working with the Hugging Face Diffusers library by providing a flexible, JSON-based interface for running generative AI models. Users can define and execute complex image generation tasks without writing custom Python code.

## Features

- Make any workflow command line executable with variable substitution
- Suppport for text to image & video and image to image & video workflows
- Prompt augmentation using locally installed LLMs
- Image processing tasks for controlnet workflows
- Other helpful tasks like background removal, upscaling, cropping and qr code generation

## Installation

### bash

```bash
bash ./install.sh
. ./activate
python -m dh.test
```

### powershell

```powershell
.\install.ps1
.\venv\scripts\activate 
python -m dh.test
```

### diffusers from source

The install script will install [the diffusers library from PyPi](https://pypi.org/project/diffusers/). If you want to install from source and use not yet released diffusers, you can do so with the following commands:

```bash
. ./activate # or .\venv\scripts\activate on windows
pip install git+https://github.com/huggingface/diffusers
```

## Usage

### Run a job

```bash
python -m dh.run --help
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

### Validate a job definition 

```bash
python -m dh.validate --help
usage: validate.py [-h] file_name

Validate a project file.

positional arguments:
  file_name   The filespec to of the project to validate

options:
  -h, --help  show this help message and exit
```

## JSON Input Format

### Schema

[Json schema](https://json-schema.app/view/%23?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdkackman%2Fdiffusers-helper%2Frefs%2Fheads%2Fmaster%2Fdh%2Fjob_schema.json)

### Examples

#### Simple Image Generation with an Input Variable

This example declares a variable for the `prompt` which can then be set on the command line. The `prompt` variable is then used in the `prompt` argument of the model.

```bash
python -m dh.run test_job.json prompt="an orange" num_images_per_prompt=4
```

```json
{
    "id": "test_job",
    "variables": {
        "prompt": "an apple",
        "num_images_per_prompt": 1
    },
    "steps": [
        {
            "name": "main",
            "pipeline": {
                "configuration": {
                    "pipeline_type": "FluxPipeline",
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
            "pipeline": 
            {
                "transformer": {
                    "configuration": {
                        "pipeline_type": "SD3Transformer2DModel",
                        "bits_and_bytes_configuration": {
                            "load_in_4bit": true,
                            "bnb_4bit_quant_type": "{nf4}",
                            "bnb_4bit_compute_dtype": "torch.bfloat16"
                        }
                    },
                    "from_pretrained_arguments": {
                        "model_name": "stabilityai/stable-diffusion-3.5-large",
                        "subfolder": "transformer",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "configuration": {
                    "pipeline_type": "StableDiffusion3Pipeline",
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
                    "pipeline_type": "CogVideoXImageToVideoPipeline",
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

This example uses an instruct LLM to augment the prompt before passing it to the model.

Note that this particular LLM requries `flash_attn` which in turn requires the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

```json
{
    "variables": {
        "prompt": "A marmot, wearing a tophat in a woodland setting. Somewhat magical."
    },
    "id": "sd35",
    "steps": [
        {
            "name": "messages",
            "task": {
                "command": "format_chat_message",
                "arguments": {
                    "system_prompt": "You are a helpful AI assistant that creates prompts for text to image generative AI. When supplied input generate only the prompt.",
                    "user_message": "variable:prompt"
                }
            }
        },
        {
            "name": "augment_prompt",
            "pipeline": {
                "configuration": {
                    "pipeline_type": "transformers.pipeline",
                    "no_generator": true
                },
                "from_pretrained_arguments": {
                    "task": "text-generation"
                },
                "model": {
                    "configuration": {
                        "pipeline_type": "transformers.AutoModelForCausalLM"
                    },
                    "from_pretrained_arguments": {
                        "model_name": "microsoft/Phi-3.5-mini-instruct",
                        "device_map": "cuda",
                        "torch_dtype": "{auto}",
                        "trust_remote_code": true
                    }
                },
                "tokenizer": {
                    "configuration": {
                        "pipeline_type": "transformers.AutoTokenizer"
                    },
                    "from_pretrained_arguments": {
                        "model_name": "microsoft/Phi-3.5-mini-instruct"
                    }
                },
                "arguments": {
                    "text_inputs": "previous_result:messages",
                    "max_new_tokens": 500,
                    "return_full_text": false,
                    "do_sample": false
                }
            }
        },
        {
            "name": "main",
            "pipeline": {
                "configuration": {
                    "pipeline_type": "StableDiffusion3Pipeline",
                    "offload": "sequential"
                },
                "from_pretrained_arguments": {
                    "model_name": "stabilityai/stable-diffusion-3.5-large",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "prompt": "previous_result:augment_prompt.generated_text",
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