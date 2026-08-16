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

Each step has a `name` and exactly one of four types:

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

### Pipeline Reference Steps

Re-run an already-loaded pipeline from an earlier step with a fresh set of arguments,
instead of loading the model again. This is how a two-pass technique like RF-Inversion
works: an `invert` step loads the pipeline, and a `main` step reuses it with the
inverted latents:

```json
{
    "name": "main",
    "pipeline_reference": {
        "reference_name": "invert",
        "arguments": {
            "prompt": "variable:prompt",
            "inverted_latents": "previous_result:invert.inverted_latents",
            "image_latents": "previous_result:invert.image_latents"
        }
    },
    "result": { "content_type": "image/jpeg" }
}
```

`reference_name` must name a step earlier in the same workflow that has a `pipeline`.
See [examples/FluxRFInversion.json](../examples/FluxRFInversion.json) for a full example.

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

A task can take `inputs` (a plain array) instead of `arguments`. Each array item becomes
its own iteration, the same way multiple `previous_result` values do:

```json
{
    "name": "prompts",
    "task": {
        "command": "gather_inputs",
        "inputs": ["a marmot on a bicycle", "a bug driving a cycle"]
    }
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

A step whose result is a dict (a task returning several named outputs, or a pipeline
step that returns something like `inverted_latents`) can be referenced property by
property with `previous_result:step_name.property_name`:

```json
"inverted_latents": "previous_result:invert.inverted_latents"
```

### Media Arguments

Images and videos load automatically for arguments named `image`/`*_image` and
`video`/`*_video`. Any other argument - `mask`, `depth_map`, a controlnet's second
conditioning image - can load the same way with an explicit form that says what the
media is instead of relying on its argument name:

```json
"mask": { "media_type": "image", "location": "mask.png" }
```

`media_type` is `"image"` or `"video"`. `location` is a path relative to the workflow
file, or a URL, exactly like the plain `image`/`video` forms.

## Result Configuration

```json
"result": {
    "content_type": "image/jpeg",
    "save": true,
    "file_base_name": "custom_prefix"
}
```

Supported content types: `image/jpeg`, `image/png`, `image/webp`, `image/gif`, `video/mp4`, `audio/wav`, `audio/flac`, `audio/mpeg` (mp3), `audio/ogg`, `audio/opus`, `application/json`, `text/plain`.

For video, add `"fps": 8`. For audio, add `"sample_rate": 44100`. Setting `embed_metadata: true`
on an image result embeds the step's model name and arguments as generation metadata -
PNG info chunks for `image/png`, EXIF `UserComment` (via `piexif`) for `image/jpeg` and
`image/webp`.

A pipeline that generates a video with its own audio track (LTX-2, or a modular pipeline
whose `output` asks for both `videos` and `audio`) is muxed into one `video/mp4` file
with PyAV. `audio_sample_rate` overrides the rate the pipeline itself reports, for the
rare case it needs correcting.

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

Output files are saved as `{output_dir}/{file_base_name}{workflow_id}-{step_name}.{step_index}-{result_index}.{artifact_index}.{ext}`,
where `step_index` is the step's position in the workflow, `result_index` counts the
argument-combination iterations the step ran (see cartesian product, above), and
`artifact_index` counts multiple artifacts within one result (`num_images_per_prompt > 1`,
or a dict result saved key by key). `file_base_name`, when set, is prepended to the
default name rather than replacing it.

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
- A `components` block that group offloads anything already keeps the pipeline itself off
  the device - the components are placed individually, so moving the whole pipeline would
  load it in full before the offload hooks exist. Nothing extra is needed for that.

`do_not_send_to_device` covers the case that is left: a component loaded already placed,
which must not be moved afterwards. A `device_map` load or a quantization that pins its
tensors to one device is the usual reason.

```json
"transformer": {
    "configuration": {
        "component_type": "FluxTransformer2DModel",
        "do_not_send_to_device": true
    },
    "from_pretrained_arguments": {
        "model_name": "black-forest-labs/FLUX.1-dev",
        "subfolder": "transformer",
        "device_map": "cuda"
    }
}
```

#### Releasing a pipeline mid-workflow

Pipelines stay loaded for the whole run (and across REPL runs) so repeated steps reuse
them. When a workflow chains two large models that cannot both fit - generate with one,
upscale with another - release the first once its step completes instead of configuring
offload on everything:

```json
{
    "name": "generate",
    "release_pipeline": true,
    "pipeline": { ... }
}
```

The step-level `release_pipeline` flag unloads the step's pipeline after its results are
saved. A later `pipeline_reference` to a released step is an error, and the REPL's
cross-run cache will not retain it.

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

### LoRAs

Attach one or more LoRAs to a pipeline with `loras`, a sibling of `configuration`:

```json
"loras": [
    { "model_name": "XLabs-AI/flux-RealismLora", "adapter_name": "realism", "scale": 0.8 },
    { "model_name": "user/other-lora", "weight_name": "lora.safetensors", "subfolder": "loras" }
]
```

- `model_name` — the LoRA's hub repo, required.
- `weight_name` / `subfolder` — pick a specific weights file within the repo.
- `adapter_name` — name passed to `set_adapters()`. Defaults to the LoRA's index in the list.
- `scale` — the adapter's weight, passed to `set_adapters()`. Defaults to `1.0`.

See [examples/FluxLora.json](../examples/FluxLora.json) for a full example.

### IP-Adapter

```json
"ip_adapter": {
    "model_name": "h94/IP-Adapter",
    "weight_name": "ip-adapter_sdxl.bin",
    "scale": 0.6
}
```

`model_name` is required; `weight_name`, `subfolder` and `scale` are optional. The
adapter image itself is passed as a normal `ip_adapter_image` pipeline argument. See
[examples/ip-adapter.json](../examples/ip-adapter.json).

### Sharing Components Across Steps

Two pipeline steps that load the same underlying component (a shared text encoder, for
instance) can avoid loading it twice:

```json
"configuration": { "component_type": "FluxPipeline", "shared_components": ["text_encoder"] }
```

```json
"configuration": { "component_type": "FluxPipeline", "reused_components": ["text_encoder"] }
```

The step naming `shared_components` stores those components after it loads; a later step
naming the same names in `reused_components` gets them passed into its own
`from_pretrained_arguments` instead of loading its own copy. The names must match exactly
between the two steps.

### Attention and Performance

```json
"configuration": {
    "component_type": "FluxPipeline",
    "attention_backend": "flash_hub",
    "enable_attention_slicing": true,
    "no_generator": false
}
```

- `enable_attention_slicing` — process attention in slices to reduce memory. Enabled
  automatically on MPS unless `disable_attention_slicing` is set.
- `attention_backend` — selects a diffusers attention backend (e.g. `"flash_hub"`) for
  the duration of each pipeline call.
- `prompt_weighting` — enables A1111-style prompt weighting (`(word:1.5)`, `[word]`,
  `((word))`) and prompts over 77 tokens. Currently supports Flux pipelines. Mutually
  exclusive with `remote_text_encoder`.
- `no_generator` — set `true` to skip creating a `torch.Generator` for pipelines that
  don't accept one.

### Cache Acceleration

Two mutually exclusive ways to speed up inference by skipping redundant computation:

```json
"configuration": {
    "cache": { "type": "first_block", "threshold": 0.05 }
}
```

`cache` wraps diffusers' own cache hooks - `type` is one of `first_block`, `faster`,
`mag`, `taylorseer` or `text_kv`, each with its own tuning fields (`threshold`,
`num_inference_steps`, `max_skip_steps`, `retention_ratio`, `cache_interval`,
`max_order` — see [dw/workflow_schema.json](../dw/workflow_schema.json) for which
fields apply to which type). See
[examples/FluxDevFirstBlockCache.json](../examples/FluxDevFirstBlockCache.json).

```json
"configuration": {
    "teacache": { "rel_l1_thresh": 0.4 }
}
```

`teacache` enables TeaCache, currently for Flux transformers, and requires
`num_inference_steps` among the pipeline's arguments.

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

Set a seed for reproducibility at workflow, step, or pipeline level - most specific wins:
a pipeline's own `seed` overrides its step's, which overrides the workflow's:

```json
{
    "id": "my_workflow",
    "seed": 42,
    "steps": [
        { "name": "step1", "seed": 123, "pipeline": { "seed": 7, ... } }
    ]
}
```

Omit `seed` entirely to let the workflow draw a random one at run time.

## Type System

Dynamic type conversion applies to certain values:

- Keys ending in `_type` or `_dtype`, or named `dtype`: `"torch.bfloat16"` becomes `torch.bfloat16`
- Dotted names: `"sdnq.SDNQConfig"` loads the class via importlib
- Escape with braces to keep as string: `"{nf4}"` stays as `"nf4"`
- `content_type` and `offload_type` are exempt even though they end in `_type` - they
  name a category, not a Python type, so their value always stays a plain string (the
  `{}` escape is accepted but not required for these two keys)

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
be a path — relative to the workflow file, like all media a workflow names — or a URL,
and is validated like any other media. `variable:` references work as the file location;
`previous_result:` references do not — the object is built when the workflow loads,
before any step has run, so reference a saved file's path or a URL instead. A dict that
merely contains a `from_file` key without a `*_type` key is not an object description
and is passed through untouched.

See [examples/MiniMaxH3Ref2VA.json](../examples/MiniMaxH3Ref2VA.json) for a full example.
