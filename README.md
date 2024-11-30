# diffusers-helper

## Introduction

This is a helper for the [Huggingface Diffuser project](https://github.com/huggingface/diffusers). It provides a command line interface and json input format for driving the diffuser library supporting the most common diffuser use cases. It allows you to run new models without any code changes by referencing the pipeline types, module names, and arbitrary paramters as data.

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

## Usage

### Run a job

```bash
python -m dh.run --help
usage: run.py [-h] file_name [output_dir]
```

- `file_name` is the name of the json file containing the diffuser job configuration
- `output_dir` is an optional directory to write the diffuser job output to. defaults to `./output`


### Validate a job defintino

```bash
python -m dh.validate --help
usage: validate.py [-h] file_name [job_id] [output_dir]
```

- `file_name` is the name of the json file containing the diffuser job configuration
- `job_id` is an optional job id to use for the diffuser job if the file contains multiple jobs

## JSON Input Format

### Schema

[Json schema](https://json-schema.app/view/%23?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdkackman%2Fdiffusers-helper%2Frefs%2Fheads%2Fmaster%2Fdh%2Fproject_schema.json)

### Example

This example demonstrates a multiple step workflow including an image generation step followed by a video generation step. It includes the use of a transformer model for the image generation step and a video generation model for the video generation step.

```json
{
    "id": "sd35 image to video",
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
                    "offload": "full"
                },
                "from_pretrained_arguments": {
                    "model_name": "stabilityai/stable-diffusion-3.5-large",
                    "torch_dtype": "torch.bfloat16"
                },
                "result": {
                    "content_type": "image/png"
                },
                "arguments": {
                    "prompt": "portrait | wide angle shot of eyes off to one side of frame, lucid dream-like 3d model of owl, game asset, blender, looking off in distance ::8 style | glowing ::8 background | forest, vivid neon wonderland, particles, blue, green, orange ::7 parameters | rule of thirds, golden ratio, asymmetric composition, hyper- maximalist, octane render, photorealism, cinematic realism, unreal engine, 8k ::7 --ar 16:9 --s 1000",
                    "num_inference_steps": 25,
                    "guidance_scale": 4.5,
                    "max_sequence_length": 512
                },
                "capture_preprocessor_results": {
                    "base_image": "image"
                }
            }
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
                    "torch_dtype": "torch.bfloat16",
                    "use_safe_tensors": true
                },
                "result": {
                    "content_type": "video/mp4",
                    "file_base_name": "owl"
                },
                "preprocessor_results": {
                    "image": "{base_image}"
                },
                "arguments": {
                    "prompt": "The owl stares intently and blinks",
                    "num_videos_per_prompt": 1,
                    "num_inference_steps": 50,
                    "num_frames": 49,
                    "guidance_scale": 6
                }
            }
        }
    ]
}
```
