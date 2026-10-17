# Quantization

Quantization reduces model memory usage by storing weights at lower precision. diffusers-workflow ships examples for BitsAndBytes, TorchAO, GGUF, and SDNQ, applied per-component in the pipeline - and any other backend with a config class (optimum-quanto, for example) works through the same dynamic `config_type` import.

## Per-Component Quantization

Quantize individual components (transformer, text encoder, etc.) independently:

```json
{
    "pipeline": {
        "transformer": {
            "configuration": { "component_type": "FluxTransformer2DModel" },
            "quantization_config": {
                "configuration": { "config_type": "..." },
                "arguments": { ... }
            },
            "from_pretrained_arguments": {
                "model_name": "...",
                "subfolder": "transformer",
                "torch_dtype": "torch.bfloat16"
            }
        },
        "configuration": { "component_type": "FluxPipeline" },
        "from_pretrained_arguments": {
            "model_name": "...",
            "torch_dtype": "torch.bfloat16"
        }
    }
}
```

The component is loaded separately with quantization, then the rest of the pipeline loads around it.

## BitsAndBytes (CUDA only)

4-bit and 8-bit quantization via bitsandbytes:

```json
"quantization_config": {
    "configuration": { "config_type": "BitsAndBytesConfig" },
    "arguments": {
        "load_in_4bit": true,
        "bnb_4bit_quant_type": "{nf4}",
        "bnb_4bit_compute_dtype": "torch.bfloat16"
    }
}
```

Note: `"{nf4}"` uses braces to keep the string literal. Without braces, the type system would try to load `nf4` as a Python class.

For 8-bit:

```json
"arguments": { "load_in_8bit": true }
```

**Example:** [flux2-dev.json](../workflows/models/flux2-dev.json) (a pre-quantized 4-bit checkpoint)

## TorchAO

Quantization via TorchAO. `quant_type` must be an `AOBaseConfig` class (diffusers no
longer accepts string shorthands like `"int4wo"`). Name the class as a dotted
`quant_type` and it is instantiated automatically with no arguments before being passed
to `TorchAoConfig`:

```json
"quantization_config": {
    "configuration": { "config_type": "TorchAoConfig" },
    "arguments": {
        "quant_type": "torchao.quantization.Int8WeightOnlyConfig",
        "modules_to_not_convert": ["proj_in", "proj_out"]
    }
}
```

Common choices: `Int8WeightOnlyConfig` (any CUDA card), `Int4WeightOnlyConfig` (smallest),
`Float8DynamicActivationFloat8WeightConfig` (fastest, requires compute capability 8.9+ -
RTX 40-series/Ada or newer).

**Example:** [flux-torchao.json](../workflows/models/flux-torchao.json)

**Pair TorchAO with `torch.compile`.** Int8 weight-only and float8 dynamic-activation
quant types get their fused-kernel speedups only under compilation - uncompiled they are
a memory win but often a speed *loss*. Compiled is not a guarantee either: int8 weight-only
on an RTX 3090 measured over a minute per denoising step for Flux dev, compiled, against
two seconds for bf16 (see [RECIPES_24GB.md](RECIPES_24GB.md#flux-dev-12b)). Measure on the
card before recording a TorchAO recipe as fast. Add a `compile` block to the quantized component
(see [ACCELERATION.md](ACCELERATION.md#torchcompile)):

```json
"configuration": {
    "components": {
        "transformer": {
            "compile": { "repeated_blocks": true, "fullgraph": true }
        }
    }
}
```

## GGUF

Load GGUF-format checkpoint files:

```json
"quantization_config": {
    "configuration": { "config_type": "GGUFQuantizationConfig" },
    "arguments": {
        "compute_dtype": "torch.bfloat16"
    }
}
```

GGUF models load from single files using `from_single_file`:

```json
"from_pretrained_arguments": {
    "from_single_file": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf",
    "torch_dtype": "torch.bfloat16"
}
```

**Example:** [flux-gguf.json](../workflows/models/flux-gguf.json)

## SDNQ (SD.Next Quantization)

SDNQ works two ways: quantize a component on the fly at load time, or load a
pre-quantized model as a complete pipeline.

### On-the-fly (`sdnq.SDNQConfig`)

Quantizes the component while it loads - the pattern the LTX-2 and MiniMax H3
examples use for their large transformers and text encoders:

```json
"quantization_config": {
    "configuration": { "config_type": "sdnq.SDNQConfig" },
    "arguments": {
        "weights_dtype": "{uint4}",
        "quantization_device": "cuda",
        "return_device": "cuda",
        "use_quantized_matmul": true,
        "dequantize_fp32": false
    }
}
```

- `weights_dtype` — the storage dtype (`uint4`, `int8`, ...; brace-escaped so it stays a string)
- `quantization_device` / `return_device` — where the quantization pass runs and where the finished component lands; quantizing on `cuda` is much faster than on CPU
- `use_quantized_matmul` — quantized matmul kernels (CUDA/XPU only)

**Examples:** [text-to-video.json](../workflows/templates/ltx2/text-to-video.json), [video-with-audio.json](../workflows/templates/minimax/video-with-audio.json)

The H3 templates also default to `attention_backend: "sparse"`; the quantization is independent of it ([Attention Backends](ACCELERATION.md#sparse-minimax-h3)).

### Cached on-the-fly (`DW_QUANTIZED_CACHE`)

An on-the-fly load reads the bf16 weights and quantizes them on every cold load. With `DW_QUANTIZED_CACHE=1` the server keeps the result: the first cold load saves each quantized component to disk with sdnq's `save_sdnq_model` (safetensors plus JSON, no pickle), and a later load of the same key reads it back with `load_sdnq_model`, skipping the bf16 read and the quantize. Any other value leaves it off, with no events. It covers both ways a component is quantized with a `sdnq.SDNQConfig` `quantization_config`: a modular pipeline's `load_components`, one config per component (MiniMax H3), and a component loaded on its own through its class's `from_pretrained` (LTX-2.5's transformer and Gemma text encoder). A hit is placed on the config's `return_device` (LTX's transformer goes straight to `cuda`) and loads with no download.

The variables are read by the server and inherited by its workers, so set them in the server's environment (e.g. `.env`) and restart.

- **Where**: `DW_QUANTIZED_CACHE_DIR`, default `<HF_HOME>/dw-quantized`. `dw.serve` resolves it to an absolute path at start and pins it for the workers. Each entry is `<root>/<sha256 key>/` with an `entry.json` (the key's inputs, size, created and last-hit times), written to a `.tmp-*` sibling and renamed into place
- **The key**: repo, subfolder, snapshot commit, quantizer class and config (minus the devices, the quantized-matmul switches and `non_blocking`, which don't change the stored weights), `torch_dtype`, and the sdnq, diffusers, transformers and torch versions. A key that can't pin the weights isn't cached
- **Not cached** (a `skipped` event with the reason):

  | Reason | When |
  | ------ | ---- |
  | `local_path` | the model is a local path |
  | `unsupported` | a `variant`, no type hint, a single-file checkpoint, or a per-component load with a `from_pretrained` argument the key doesn't cover |
  | `unpinned_snapshot` | the snapshot isn't complete on disk, or the recorded commit disagrees with the one the hub refs name - a save is only ever made under the commit an identical later lookup reads |
  | `low_disk` | the save would leave free space below `min(50 GiB, 10% of the volume)` |
  | `save_failed: <ENOSPC or exception type>` | the save failed |

- **Why a `miss`** (its `reason`): `no_entry` (nothing saved under the key yet), `incomplete_snapshot` (the snapshot isn't all on disk, so no lookup was made), or `load_failed: <exception type>: <first line of its message>` (the entry didn't load and was deleted; the save that follows replaces it)
- **Concurrent workers**: two workers saving the same key leave one entry; the other reports `saved` ("already saved by another worker"). A corrupt entry is deleted and quantized fresh
- **What a hit holds**: the whole component as quantized. The save comes before pruning (`truncate_layers`, `remove_modules`), `sdnq_optimize`, LoRAs and offload, which all still apply after a hit
- **At server start** (only when on): a sweep deletes leftover `*.tmp-*` directories and entries built against other package versions, keyed under another `KEY_VERSION` (how the key's inputs are derived; one made before the field records none), or with an unreadable `entry.json`, and logs the count and bytes freed. It then logs the root, bytes in use and free space, warning when free space is below the floor plus the largest entry already cached
- **Clearing**: there is no other eviction. Use the server page's Caches panel (the whole cache, or one entry from its expandable table), or `DELETE /api/caches/quantized` / `DELETE /api/caches/quantized/entries/{id}`. `GET /api/caches` lists the entries (model, subfolder, size, created, last hit) and `low_disk` (free space under the floor). Clearing is refused while a job is running or queued, and while a model download is active; an off cache still lists and clears what is on disk
- **Event**: a `quantized_cache` job event, `{component, result: hit|miss|saved|skipped, reason?, seconds, bytes}`. A `miss`'s `seconds` is the time of the whole `load_components` batch, shared by the components it loaded; on the per-component path it is that one component's load

The cache never fails a job; one that can't help steps aside.

### Pre-quantized models

Pre-quantized SDNQ repos load as complete pipelines. The `sdnq` module must be imported before loading so it can register with diffusers:

```json
{
    "pipeline": {
        "configuration": {
            "component_type": "ZImagePipeline",
            "pre_load_modules": ["sdnq"],
            "sdnq_optimize": ["transformer", "text_encoder"]
        },
        "from_pretrained_arguments": {
            "model_name": "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            "torch_dtype": "torch.bfloat16"
        }
    }
}
```

- `pre_load_modules` — Imports sdnq before pipeline loading (registers quantization method)
- `sdnq_optimize` — Applies quantized matmul to listed components (CUDA/XPU only, skipped on MPS/CPU)

**Example:** [z-image-sdnq.json](../workflows/models/z-image-sdnq.json)

## Modular Pipelines

A modular pipeline pulls its component weights itself via `load_components()` rather than
through `from_pretrained_arguments`, so quantization is keyed by component name under
`load_components.quantization_config` instead of living on a separate component block:

```json
"configuration": {
    "component_type": "MiniMaxMusic3ModularPipeline",
    "load_components": {
        "dtype": "torch.bfloat16",
        "quantization_config": {
            "transformer": {
                "configuration": { "config_type": "TorchAoConfig" },
                "arguments": { "quant_type": "torchao.quantization.Int8WeightOnlyConfig" }
            },
            "language_model": {
                "configuration": { "config_type": "transformers.TorchAoConfig" },
                "arguments": { "quant_type": "torchao.quantization.Int8WeightOnlyConfig" }
            }
        }
    }
}
```

A component the map does not name loads unquantized. Note that a transformers-based
component (a text encoder, for example) takes the `transformers.TorchAoConfig` class,
not the diffusers one - the `config_type` still resolves either via the dynamic import
described below.

## Custom Quantization

Any quantization backend that provides a config class works via the `config_type` field with a dotted module path:

```json
"quantization_config": {
    "configuration": { "config_type": "some_package.SomeQuantConfig" },
    "arguments": { ... }
}
```

The class is loaded dynamically via importlib.

## Platform Notes

| Framework | CUDA | MPS | CPU |
| --------- | ---- | --- | --- |
| BitsAndBytes | Yes | No | No |
| TorchAO | Yes | Partial | No |
| GGUF | Yes | Yes | Yes |
| SDNQ (load) | Yes | Yes | Yes |
| SDNQ (optimize) | Yes | No | No |
