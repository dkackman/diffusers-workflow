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

Supported content types: `image/jpeg`, `image/png`, `image/webp`, `video/mp4`, `audio/wav`, `audio/flac`, `audio/mpeg` (mp3), `audio/ogg`, `audio/opus`, `application/json`, `text/plain`.

For video, add `"fps": 8`. For audio, add `"sample_rate": 44100`.

### Audio Encoding

Audio is written through soundfile, so both lossless and compressed containers work:

```json
"result": {
    "content_type": "audio/mpeg",
    "sample_rate": 44100,
    "compression_level": 0.3
}
```

- `subtype` — encoding subtype, such as `"PCM_24"` for wav and flac. Defaults to the
  container's own default, which is `"PCM_16"` for wav and flac.
- `compression_level` — 0.0 to 1.0 for flac, mp3 and ogg. Higher means smaller files.
- `bitrate_mode` — `"CONSTANT"`, `"AVERAGE"` or `"VARIABLE"` for compressed formats.

`audio/opus` writes an Opus stream in an ogg container, and only encodes at sample rates
of 8000, 12000, 16000, 24000 or 48000.

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

### Modular Pipelines

Modular pipelines (`ModularPipeline` and its subclasses) load their configuration and
their component weights separately, so `from_pretrained_arguments` only names the model
and `load_components` pulls the weights:

```json
"configuration": {
    "component_type": "MiniMaxMusic3ModularPipeline",
    "load_components": { "dtype": "torch.bfloat16" },
    "components_manager": { "enable_auto_cpu_offload": true }
}
```

- `load_components` — arguments for `load_components()`. Use `dtype` for the component
  dtype and `names` to load only some of the components.
- `components_manager` — attaches a `ComponentsManager`, which tracks the pipeline's
  components. With `enable_auto_cpu_offload` it keeps only the running components on the
  device and moves the rest to system memory, reserving `memory_reserve_margin`
  (default `"3GB"`) of free device memory. It requires a device that reports free memory
  (CUDA) and replaces `offload`, which modular pipelines do not support.

See [examples/MiniMaxMusic.json](../examples/MiniMaxMusic.json) for a full example.

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

- Keys ending in `_type` or `_dtype`, or named `dtype`: `"torch.bfloat16"` becomes `torch.bfloat16`
- Dotted names: `"sdnq.SDNQConfig"` loads the class via importlib
- Escape with braces to keep as string: `"{nf4}"` stays as `"nf4"`
