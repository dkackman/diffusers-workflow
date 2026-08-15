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

For components the pipeline loads itself — which is all of a modular pipeline's — use
`components`, applied once the pipeline is loaded:

```json
"configuration": {
    "component_type": "ModularPipeline",
    "do_not_send_to_device": true,
    "components": {
        "transformer": {
            "group_offload": {
                "offload_type": "block_level",
                "num_blocks_per_group": 1,
                "use_stream": true
            }
        },
        "text_encoder.model": {
            "group_offload": { "offload_type": "leaf_level", "use_stream": true }
        },
        "audio_vae": { "device": "cuda" }
    }
}
```

- `group_offload` — streams the component between system memory and the accelerator a
  block or a leaf module at a time, which is what fits a component larger than the
  device. `onload_device` defaults to the pipeline's device.
- `device` — moves a component that is small enough to stay resident.
- A dotted key reaches a module inside a component, for a component that holds the model
  rather than being one.
- `do_not_send_to_device` belongs with this: the components are placed individually, so
  the pipeline itself must not be moved to the device afterwards.

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
  dtype and `names` to load only some of the components. `quantization_config` is keyed
  by component name, since a modular pipeline loads each component itself:

  ```json
  "load_components": {
      "dtype": "torch.bfloat16",
      "quantization_config": {
          "transformer": {
              "configuration": { "config_type": "TorchAoConfig" },
              "arguments": {
                  "quant_type": "torchao.quantization.Int8WeightOnlyConfig",
                  "modules_to_not_convert": ["proj_in", "proj_out"]
              }
          },
          "text_encoder": {
              "configuration": { "config_type": "transformers.TorchAoConfig" },
              "arguments": { "quant_type": "torchao.quantization.Int8WeightOnlyConfig" }
          }
      }
  }
  ```

  A component the map does not name loads unquantized. Note which `TorchAoConfig` each
  component takes: the diffusers one for its own models, the transformers one for a
  transformers model such as a conditioner.
- `components_manager` — attaches a `ComponentsManager`, which tracks the pipeline's
  components. With `enable_auto_cpu_offload` it keeps only the running components on the
  device and moves the rest to system memory, reserving `memory_reserve_margin`
  (default `"3GB"`) of free device memory. It requires a device that reports free memory
  (CUDA) and replaces `offload`, which modular pipelines do not support.

A modular pipeline returns whatever its `output` argument asks for — one output by name,
or several of them together:

```json
"arguments": {
    "prompt": "variable:prompt",
    "output": ["videos", "audio", "sampling_rate"]
}
```

Asked for several, the outputs come back keyed by name. Video generated with its own
soundtrack is muxed into a single `video/mp4` file, the same way a video pipeline's own
output is, and a later step can still reference any of the outputs by name.

Some repositories hold more than one task's weights. `workflow` names the task, which
prunes the pipeline to the blocks that task runs, so only the components it needs are
downloaded and loaded:

```json
"from_pretrained_arguments": {
    "model_name": "MiniMaxAI/MiniMax-H3",
    "workflow": "t2va"
}
```

See [examples/MiniMaxMusic.json](../examples/MiniMaxMusic.json) and
[examples/MiniMaxH3.json](../examples/MiniMaxH3.json) for full examples.

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

### Objects Built From a File

Some pipelines take arguments that are objects rather than plain media. An argument that
names a type and a `from_file` is constructed by that type's own `from_file()`:

```json
"references": [
    {
        "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference",
        "from_file": "subject.png"
    },
    {
        "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3AudioReference",
        "from_file": "voice.wav"
    }
]
```

Loading the media this way rather than as a plain `image` or `video` argument is what
brings its frame rate or sample rate along with it, which MiniMax-H3 resamples a
reference from. Any other keys in the object are passed to `from_file()` as arguments —
`"fps": 30.0` to correct a container whose metadata is wrong, for instance. The file may
be a path or a URL, and is validated like any other media the workflow names.

See [examples/MiniMaxH3Ref2VA.json](../examples/MiniMaxH3Ref2VA.json) for a full example.
