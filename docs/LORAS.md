# LoRA Support

LoRA (Low-Rank Adaptation) models apply lightweight style or subject modifications to a base model. Add one or more LoRAs to any pipeline step.

## Basic Usage

```json
{
    "pipeline": {
        "configuration": { "component_type": "FluxPipeline" },
        "from_pretrained_arguments": {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "torch_dtype": "torch.bfloat16"
        },
        "loras": [
            {
                "model_name": "XLabs-AI/flux-RealismLora"
            }
        ],
        "arguments": {
            "prompt": "a photorealistic landscape"
        }
    }
}
```

## LoRA Properties

```json
"loras": [
    {
        "model_name": "user/lora-repo",
        "weight_name": "specific_weights.safetensors",
        "subfolder": "lora_subfolder",
        "adapter_name": "my_adapter",
        "scale": 0.8
    }
]
```

| Property | Required | Description |
| -------- | -------- | ----------- |
| `model_name` | Yes | HuggingFace Hub repo ID |
| `weight_name` | No | Specific weight file in the repo |
| `subfolder` | No | Subfolder within the repo |
| `adapter_name` | No | Named identifier for the adapter |
| `scale` | No | Blend strength (default: 1.0). Lower = less effect |

## Multiple LoRAs

Stack multiple LoRAs. They are blended via weighted adapter composition:

```json
"loras": [
    {
        "model_name": "XLabs-AI/flux-RealismLora",
        "adapter_name": "realism",
        "scale": 0.7
    },
    {
        "model_name": "user/style-lora",
        "adapter_name": "style",
        "scale": 0.5
    }
]
```

## LoRA with Quantization

LoRAs work with quantized models:

```json
{
    "pipeline": {
        "transformer": {
            "configuration": { "component_type": "SD3Transformer2DModel" },
            "quantization_config": {
                "configuration": { "config_type": "BitsAndBytesConfig" },
                "arguments": { "load_in_4bit": true, "bnb_4bit_quant_type": "{nf4}" }
            },
            "from_pretrained_arguments": {
                "model_name": "stabilityai/stable-diffusion-3.5-large",
                "subfolder": "transformer",
                "torch_dtype": "torch.bfloat16"
            }
        },
        "configuration": { "component_type": "StableDiffusion3Pipeline" },
        "from_pretrained_arguments": {
            "model_name": "stabilityai/stable-diffusion-3.5-large",
            "torch_dtype": "torch.bfloat16"
        },
        "loras": [
            {
                "model_name": "crystalwizard/cubic-abstract-1",
                "weight_name": "cubic-abstract-lora.safetensors"
            }
        ],
        "arguments": { "prompt": "cubart a leaf" }
    }
}
```

## Variable LoRA

Make the LoRA configurable via workflow variables:

```json
{
    "variables": {
        "lora": "XLabs-AI/flux-RealismLora"
    },
    "steps": [{
        "pipeline": {
            "loras": [{ "model_name": "variable:lora" }],
            "arguments": { "prompt": "variable:prompt" }
        }
    }]
}
```

```bash
python -m dw.run workflow.json lora="other-user/other-lora"
```

## Examples

- [FluxLora.json](../examples/FluxLora.json) — Flux with realism LoRA and variables
- [lora.json](../examples/lora.json) — SD 3.5 with yarn art style LoRA
- [bnb_quant.json](../examples/bnb_quant.json) — Quantized model with LoRA
