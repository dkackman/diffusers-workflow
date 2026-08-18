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
See [examples/flux/FluxRFInversion.json](../examples/flux/FluxRFInversion.json) for a full example.

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

A variable's declared value is both its default and its type — a value passed in is
converted to the type of the default, so declaring `25` and `"25"` are different things
(see the schema note under Variables). Declaring `null` opts out of that: the variable
becomes optional and untyped, taking whatever it is given and staying `null` when it is
given nothing.

```json
"variables": { "image": null }
```

This is how a workflow exposes an argument a caller *may* pass without inventing a
sentinel for its absence — a sub-workflow that behaves differently when handed an image,
say. A caller can only set variables the workflow declares, so an optional argument still
has to be declared to be passable.

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

A reference is resolved wherever it appears in the arguments, not only at the top of
them - an argument holding a list or a nested object can reference a step too, which is
what lets a constructed object be
[built from an earlier step](#objects-built-from-an-earlier-step).

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

A step's `configuration` is dw's own vocabulary rather than the model's — each key drives
a different call — so it is a closed set: a name the schema does not declare fails
validation instead of being ignored. That matters most for the keys it would otherwise
be quietest about. A misspelled `offload` used to validate, load, and run with no
offloading at all, surfacing as an out-of-memory error with nothing pointing at the
spelling; it now fails before the first model loads. Model-side values that are not part
of this vocabulary have blocks of their own: `from_pretrained_arguments` for the
constructor, `arguments` for the call, and `configs` for a modular pipeline's block
configs.

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
        "vae": { "device": "cuda", "residency": "on_demand" },
        "audio_vae": { "device": "cuda" }
    }
}
```

- `group_offload` — streams the component between system memory and the accelerator a
  block or a leaf module at a time, which is what fits a component larger than the
  device. `onload_device` defaults to the pipeline's device.
- `device` — moves a component that is small enough to stay resident.
- `residency` — `"resident"` (the default) leaves the component on its device for the
  whole run; `"on_demand"` rests it in system memory and moves it to the device only
  while one of its own calls runs. See [On-demand components](#on-demand-components).
- A dotted key reaches a module inside a component, for a component that holds the model
  rather than being one.
- A `components` block that group offloads anything, or marks anything `on_demand`,
  already keeps the pipeline itself off the device - the components are placed
  individually, so moving the whole pipeline would load it in full before the hooks and
  wrappers exist. Nothing extra is needed for that.

`preserve_device_placement` covers the case that is left: a component loaded already
placed, which must not be moved afterwards. A `device_map` load or a quantization that
pins its tensors to one device is the usual reason.

```json
"transformer": {
    "configuration": {
        "component_type": "FluxTransformer2DModel",
        "preserve_device_placement": true
    },
    "from_pretrained_arguments": {
        "model_name": "black-forest-labs/FLUX.1-dev",
        "subfolder": "transformer",
        "device_map": "cuda"
    }
}
```

> **Renamed:** this setting was `do_not_send_to_device`. The old name is no longer
> recognized - a workflow still using it will load the component and then move it to the
> device anyway, since an unknown key is ignored rather than rejected. Rename the key.

#### On-demand components

`"residency": "on_demand"` sits between the two placements above. A `device` component
holds VRAM for the whole run, wasted on a component used twice; group offloading
streams per submodule forward, so it restreams the model once per call of every leaf -
ruinous for a VAE, whose tiled decode calls its blocks once per tile. On-demand moves the
model as a whole around each call, so a tiling loop sits inside a single pair of
transfers.

```json
"components": {
    "vae": { "device": "cuda", "residency": "on_demand" },
    "audio_vae": { "device": "cuda", "residency": "on_demand" }
}
```

The component rests on the CPU and is moved to `device` around whichever of `forward`,
`encode` and `decode` it defines, then moved back and the freed VRAM released to the
driver. Nested calls are counted, so a `decode` that calls `forward` internally is moved
once, not twice.

- **Use it for a component that is large but called a handful of times** - a VAE that
  encodes references at the start and decodes the result at the end. Freeing it for the
  denoise loop is the whole point.
- **Not for a component called every step.** A denoising transformer would pay per-call
  transfers 20-50 times; group offloading is the tool for those.
- **Cannot be combined with `group_offload`** on the same component - a group offloaded
  module holds one group at a time and ignores whole-model moves, so the two cannot both
  own its placement. Configuring both is rejected at load.
- **Ignored when the component's device is the CPU**, where there is nothing to move it
  off of.

On a 24GB card, `MiniMaxH3Ref2VA.json` peaks at 18.9GiB of reserved VRAM with on-demand
VAEs against 23.2GiB resident, and the tighter resident fit costs 40 allocator retries -
cache flushes forced by a failed allocation - where the on-demand run has none. The
headroom is also what lets the chained variant run: its later segments carry an extra
reference and need ~1.9GiB more than the first.

The same holds for the frame-conditioned workflows. Generating 124 frames at 960x544
from a keyframe, with everything else held equal:

| VAE placement | peak reserved | allocator retries |
| ------------- | ------------- | ----------------- |
| resident      | 22.71GiB      | 22                |
| on-demand     | 18.03GiB      | 0                 |

The resident run also logs a `memory mapping failed with OOM` warning per retry, with as
little as 3MB free while it tries to map 20MB. It completes - the allocator flushes its
cache and succeeds on the retry - but each one is a synchronising stall, and a run that
close to the limit fails outright on any workload that needs slightly more. Every
MiniMax H3 example uses on-demand VAEs for this reason.

**Example:** [MiniMaxH3Ref2VA.json](../examples/MiniMaxH3Ref2VA.json),
[MiniMaxH3I2V.json](../examples/MiniMaxH3I2V.json)

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

#### Releasing task models mid-workflow

Task models - the checkpoints behind `text_generation`, `segment`, `depth_estimator` and
the rest - are cached separately from pipelines, so that a step running its task once per
result does not reload the same weights on every iteration. Nothing evicts that cache
during a run, which matters when a task loads a large model on the device ahead of a
generation step: a prompt-expanding language model would hold its weights for the whole
run. `release_models` clears it once the step completes:

```json
{
    "name": "expand_prompt",
    "release_models": true,
    "workflow": { "path": "builtin:h3_context_ir.json", "arguments": { ... } }
}
```

The flag applies to any step type, and on a `workflow` step it fires once the whole
sub-workflow has finished. It clears every cached task model, not only this step's, and a
later step needing one of them reloads it.

**Example:** [MiniMaxH3EnhancePrompt.json](../examples/MiniMaxH3EnhancePrompt.json)

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

See [examples/flux/FluxLora.json](../examples/flux/FluxLora.json) for a full example.

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
[examples/archive/ip-adapter.json](../examples/archive/ip-adapter.json).

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
naming the same names in `reused_components` gets them instead of loading its own copy.
The names must match exactly between the two steps. Either list can sit in the step's
`configuration` or beside it on the pipeline itself.

How the component reaches the second pipeline depends on what kind it is. A standard
pipeline takes it as a `from_pretrained` argument. A modular pipeline cannot — it is
built from the component specs in its own index — so it is registered with
`update_components()` before `load_components()` runs, which is also what keeps
`load_components()` from pulling a second copy: it only loads what is not already there.
That is what lets two MiniMax-H3 steps of different tasks (`t2va` and `ref2va` load
different transformer partitions) share the 14GB text encoder and the VAEs between them.

A reused component keeps the device placement the step that shared it gave it. Any
`components` entry naming one is skipped with a log line rather than applied a second
time — offloading hooks do not survive being installed twice, and the step that loaded
the component is the one that decided how it is placed.

Sharing outlives the pipeline that did it: a step can share a component and still set
`release_pipeline`, which frees everything else it loaded while the shared component
stays alive for the steps that reuse it.

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
[examples/flux/FluxDevFirstBlockCache.json](../examples/flux/FluxDevFirstBlockCache.json).

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
- `configs` — values the pipeline's blocks declare and read while they run. They are
  neither components nor call arguments, which is why they have a block of their own:

  ```json
  "configs": {
      "canvas_short_edge": 768,
      "reference_image_short_edge": 1024
  }
  ```

  The names are whatever the pipeline itself declares, so they differ per model rather
  than being a fixed list here — MiniMax-H3 declares `canvas_short_edge` (768),
  `canvas_max_pixels` (1032192) and `reference_image_short_edge` (2048), the last being
  the resolution its image references are encoded at. A name the pipeline does not
  declare raises rather than passing quietly, since a dropped config reads as a setting
  that did nothing.
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

A task is chosen by the arguments the step passes, so one `workflow` name can cover more
than one of them: MiniMax-H3's `fl2va` takes an `image`, a `last_image`, or both. Given
only a `last_image` it generates *up to* that frame, inventing everything that leads to
it — see [examples/MiniMaxH3L2V.json](../examples/MiniMaxH3L2V.json) beside
[examples/MiniMaxH3FL2VA.json](../examples/MiniMaxH3FL2VA.json).

See [examples/MiniMaxMusic.json](../examples/MiniMaxMusic.json) and
[examples/MiniMaxH3.json](../examples/MiniMaxH3.json) for full examples.

### Chained Video Generation

Video pipelines generate short clips - a `chain` block on a pipeline step runs the
pipeline once per segment and stitches the segments into one long video. The model
loads once; each segment's last frame is carried into the next segment as its
keyframe, the duplicated boundary frames are trimmed, and frames and audio are
joined into a single file:

```json
"pipeline": {
    "configuration": { "component_type": "LTX2ImageToVideoPipeline" },
    "from_pretrained_arguments": { "model_name": "Lightricks/LTX-2.5-Diffusers" },
    "chain": {
        "segments": 3,
        "trim_frames": 2,
        "crossfade_ms": 80
    },
    "arguments": { "prompt": "variable:prompt", "image": "variable:image" }
}
```

- `segments` — how many times the pipeline runs. Total length is roughly
  `segments * num_frames`, minus `trim_frames` per seam.
- `match_audio` — instead of a count, derive the length from the audio reference in
  the step's arguments. The audio is sliced into frame-aligned per-segment chunks,
  each segment is generated against its slice, and the final video is muxed with the
  **original, unsliced track** - so the soundtrack has no seams at all. Requires
  `num_frames` (the per-segment length) and a frame rate. Exactly one of `segments`
  or `match_audio` must be given.
- `continuity` — how visual continuity carries across segments. `last_frame` (the
  default and currently only mode) extracts each segment's last frame and passes it
  to the next segment.
- `segment_argument` — where the carried frame lands: `image` (default) for
  image-to-video pipelines, or `references` for reference-conditioned modular
  pipelines, where it is appended as an image reference alongside the workflow's own.
- `trim_frames` — image-to-video pipelines reproduce their keyframe as frame 0, so
  this many frames are dropped from the head of every segment after the first
  (default 1). The matching audio is used as crossfade material, so video and audio
  stay exactly in sync. It also bounds the crossfade window: `trim_frames / fps`
  seconds (at 24 fps, `trim_frames: 2` allows the full default 75 ms fade).
- `crossfade_ms` — equal-power crossfade applied to *generated* audio at each seam
  (default 75). Not used with `match_audio`, which keeps the original track.
- `fps` — frame rate for the chain's audio math. Defaults to the pipeline's
  `frame_rate` argument; pipelines with a fixed rate need it set (MiniMax H3: 24).
- `frame_snap` — the constraint the pipeline puts on `num_frames`, used to snap the
  final `match_audio` segment to a valid length. MiniMax H3 accepts `17n+5` frames
  between 124 and 345: `{ "modulus": 17, "remainder": 5, "min_frames": 124,
  "max_frames": 345 }`.
- `prompts` — optional per-segment prompt list for narrative progression; segment
  `i` uses `prompts[min(i, len - 1)]`.
- `save_segments` — write each completed segment to the output directory as a
  playable mp4 and free its frames, bounding memory to roughly one segment
  regardless of chain length. The final video is streamed from the segment files
  at save time, and they are removed once it is written (`keep_segments: true`
  retains them). A crashed chain leaves the finished segments behind - stitch
  them by hand with `gather_videos` + `concat_videos` (`trim_frames: 0`, the
  trim was already applied). Requires PyAV and a frame rate. The trade-off is
  one extra encode/decode cycle through h264 for the segment files.

The chain runs inside one iteration of the step, so it composes with
`previous_result` fan-out (three keyframes in, three chained videos out), and a
`pipeline_reference` step can carry its own `chain`. Seeds behave like a normal run:
the step's generator advances across segments, so one seed reproduces the whole
chain. Expect some visual drift across many segments with `last_frame` continuity -
it is single-frame conditioning; richer continuity modes are the extension point.

See [examples/LTX2I2VChained.json](../examples/LTX2I2VChained.json),
[examples/MiniMaxH3I2VChained.json](../examples/MiniMaxH3I2VChained.json), and
[examples/MiniMaxH3Ref2VAChained.json](../examples/MiniMaxH3Ref2VAChained.json)
(audio-matched lip-sync of arbitrary length).

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

The seed also reaches sub-workflows: a delegated `workflow` step runs the child under
the parent's seed unless the child names its own. Without that a child draws its own
random seed, and a workflow whose real generation happens inside a sub-workflow would
not reproduce from the seed it was given.

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
reference from. The file may be a path — relative to the workflow file, like all media a
workflow names — or a URL, and is validated like any other media. `variable:` references
work as the file location; `previous_result:` does not, since the object is built when
the workflow loads — use
[`from_previous_result`](#objects-built-from-an-earlier-step) for that. A dict that
merely contains a `from_file` key without a `*_type` key is not an object description
and is passed through untouched.

Any other key goes wherever the type can take it: to `from_file()` where its signature
names it, and onto the object it returns where it does not. That is what corrects a
decoded file, which is the only thing that knows what the container claimed:

```json
{
    "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3VideoReference",
    "from_file": "motion.mp4",
    "fps": 30.0,
    "audio": null
}
```

`fps` overrides a rate the container got wrong — MiniMax-H3 resamples a reference onto
its own 24 fps, so a wrong rate is a request conditioned at the wrong speed — and
`audio: null` drops the decoded soundtrack, leaving a reference that conditions on
motion and camera alone. A name that is neither an argument of `from_file()` nor a field
of the object raises, with the fields it does have.

See [examples/MiniMaxH3Ref2VA.json](../examples/MiniMaxH3Ref2VA.json) for a full example.

### Objects Built From an Earlier Step

The same object can be built from what an earlier step generated, by naming the step
instead of a file:

```json
"references": [
    {
        "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference",
        "from_previous_result": "draw_subject"
    }
]
```

`from_file` cannot do this — it names a file, and the object is built when the workflow
loads, before any step has run. `from_previous_result` waits: the description is checked
at load time and constructed once the step it names has produced its media, which is
what lets one workflow generate a subject and then condition on it without writing it
out and reading it back.

The media never touches the disk, so it arrives as the step produced it. Which field it
lands in comes from the type's own `kind`:

| `kind`  | Built from                                                                |
| ------- | ------------------------------------------------------------------------- |
| `image` | The generated image                                                       |
| `video` | The generated frames, and the soundtrack generated with them if there was one |
| `audio` | The generated soundtrack                                                  |

Any other key is a field of the object and wins over what the media carried —
`"fps": 30.0` where the producing pipeline generated at a rate the consuming one does
not share, for instance. A step that produced several artifacts fans out the same way
every `previous_result` reference does: four images in, four videos out.

See [examples/MiniMaxH3Ref2VAGeneratedSubject.json](../examples/MiniMaxH3Ref2VAGeneratedSubject.json)
for a full example.
