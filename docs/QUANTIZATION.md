# Quantization

Quantization reduces model memory usage by storing weights at lower precision. diffusers-workflow supports five quantization frameworks, applied per-component in the pipeline.

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

**Example:** [bnb_quant.json](../examples/bnb_quant.json)

## TorchAO

Weight-only quantization via TorchAO:

```json
"quantization_config": {
    "configuration": { "config_type": "TorchAoConfig" },
    "arguments": {
        "quant_type": "{int4wo}"
    }
}
```

**Example:** [FluxTorchAO.json](../examples/FluxTorchAO.json)

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

**Example:** [FluxGGUF.json](../examples/FluxGGUF.json)

## SDNQ (SD.Next Quantization)

SDNQ uses pre-quantized models that load as complete pipelines. The `sdnq` module must be imported before loading so it can register with diffusers:

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
