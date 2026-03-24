# Workflow Guide

## Structure

Every workflow is a JSON file with an `id`, optional `variables`, and a list of `steps`:

```json
{
    "id": "my_workflow",
    "variables": {
        "prompt": "default prompt text",
        "steps": 25
    },
    "steps": [ ... ]
}
```

**Variables** define defaults that can be overridden from the command line:

```bash
python -m dw.run my_workflow.json prompt="a cat" steps=50
```

Variable names must be alphanumeric with underscores or hyphens.

## Step Types

Each step has a `name` and exactly one of three types:

### Pipeline Steps

Run a HuggingFace Diffusers model:

```json
{
    "name": "generate",
    "pipeline": {
        "configuration": { "component_type": "FluxPipeline" },
        "from_pretrained_arguments": {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "torch_dtype": "torch.bfloat16"
        },
        "arguments": {
            "prompt": "variable:prompt",
            "num_inference_steps": 25
        }
    },
    "result": { "content_type": "image/jpeg" }
}
```

### Task Steps

Run utility operations (image processing, QR codes, data gathering):

```json
{
    "name": "preprocess",
    "task": {
        "command": "canny",
        "arguments": {
            "image": { "location": "https://example.com/photo.jpg" }
        }
    },
    "result": { "content_type": "image/jpeg" }
}
```

### Workflow Steps

Invoke another workflow file:

```json
{
    "name": "augment",
    "workflow": {
        "path": "builtin:augment_prompt.json",
        "arguments": { "prompt": "variable:prompt" }
    },
    "result": { "content_type": "text/plain" }
}
```

Paths can be relative to the current file or use `builtin:` to reference built-in workflows in `dw/workflows/`.

## Cross-Step Data Flow

### Variable References

Reference workflow variables with `variable:name`:

```json
"prompt": "variable:prompt"
```

### Previous Result References

Pass output from one step to another with `previous_result:step_name`:

```json
{
    "steps": [
        {
            "name": "preprocess",
            "task": { "command": "canny", "arguments": { "image": { "location": "photo.jpg" } } }
        },
        {
            "name": "generate",
            "pipeline": {
                "arguments": {
                    "control_image": "previous_result:preprocess",
                    "prompt": "a painting"
                }
            }
        }
    ]
}
```

Multiple `previous_result` references create a **cartesian product**: if step A produces 4 images and step B produces 3 masks, a step referencing both will run 12 times.

## Result Configuration

```json
"result": {
    "content_type": "image/jpeg",
    "save": true,
    "file_base_name": "custom_prefix"
}
```

Supported content types: `image/jpeg`, `image/png`, `image/webp`, `video/mp4`, `audio/wav`, `application/json`, `text/plain`.

For video, add `"fps": 8`. For audio, add `"sample_rate": 44100`.

Output files are saved as `{output_dir}/{workflow_id}-{step_name}.{index}.{ext}`.

## Pipeline Configuration

### Memory Offloading

Control how models use memory:

```json
"configuration": {
    "component_type": "FluxPipeline",
    "offload": "model"
}
```

- `"model"` — Moves entire models between CPU and GPU. Good balance of speed and memory.
- `"sequential"` — Moves individual layers. Slowest but uses least GPU memory.
- Omit for no offloading (fastest, requires enough VRAM).

### VAE Options

```json
"configuration": {
    "vae": {
        "enable_slicing": true,
        "enable_tiling": true
    }
}
```

- `enable_slicing` — Process VAE in slices to reduce memory
- `enable_tiling` — Tile large images through the VAE

### Device and Dtype

Device is auto-detected (CUDA > MPS > CPU). Dtype is set per-component:

```json
"from_pretrained_arguments": {
    "model_name": "black-forest-labs/FLUX.1-dev",
    "torch_dtype": "torch.bfloat16"
}
```

## Schedulers

Override the default scheduler:

```json
"scheduler": {
    "configuration": {
        "scheduler_type": "DPMSolverMultistepScheduler"
    },
    "from_config_args": {
        "use_karras_sigmas": true
    }
}
```

## Seeds

Set a seed for reproducibility at workflow or step level:

```json
{
    "id": "my_workflow",
    "seed": 42,
    "steps": [
        { "name": "step1", "seed": 123, ... }
    ]
}
```

## Type System

Dynamic type conversion applies to certain values:

- Keys ending in `_type` or `_dtype`: `"torch.bfloat16"` becomes `torch.bfloat16`
- Dotted names: `"sdnq.SDNQConfig"` loads the class via importlib
- Escape with braces to keep as string: `"{nf4}"` stays as `"nf4"`
