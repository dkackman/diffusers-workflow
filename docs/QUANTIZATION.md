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

**Example:** [bnb_quant.json](../examples/archive/bnb_quant.json)

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

**Example:** [FluxTorchAO.json](../examples/flux/FluxTorchAO.json)

**Pair TorchAO with `torch.compile`.** Int8 weight-only and float8 dynamic-activation
quant types get their fused-kernel speedups only under compilation - uncompiled they are
a memory win but often a speed *loss*. Add a `compile` block to the quantized component
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

**Example:** [FluxGGUF.json](../examples/flux/FluxGGUF.json)

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

**Examples:** [LTX2.json](../examples/LTX2.json), [MiniMaxH3.json](../examples/minimax/MiniMaxH3.json)

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

**Example:** [ZImageSDNQ.json](../examples/ZImageSDNQ.json)

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
            "text_encoder": {
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
