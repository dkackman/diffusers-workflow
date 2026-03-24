# IP-Adapter

IP-Adapter enables image-prompt conditioning — use a reference image to influence the style or content of generated images alongside a text prompt.

## Usage

Add an `ip_adapter` block to the pipeline definition and pass `ip_adapter_image` in arguments:

```json
{
    "pipeline": {
        "configuration": {
            "component_type": "FluxPipeline",
            "offload": "sequential"
        },
        "from_pretrained_arguments": {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "torch_dtype": "torch.bfloat16"
        },
        "ip_adapter": {
            "model_name": "XLabs-AI/flux-ip-adapter",
            "weight_name": "ip_adapter.safetensors"
        },
        "arguments": {
            "prompt": "A marmot sits at a counter drinking a milkshake",
            "ip_adapter_image": {
                "location": "https://example.com/reference_style.jpg"
            },
            "num_inference_steps": 25,
            "guidance_scale": 3.5
        }
    }
}
```

## Properties

| Property | Required | Description |
| -------- | -------- | ----------- |
| `model_name` | Yes | HuggingFace Hub repo ID for the IP-Adapter weights |
| `weight_name` | No | Specific weight file in the repo |
| `subfolder` | No | Subfolder within the repo |
| `scale` | No | Adapter strength (default: 1.0). Lower = less influence from reference image |

## Image Argument

The `ip_adapter_image` uses the standard image loading format:

```json
"ip_adapter_image": {
    "location": "https://example.com/image.jpg"
}
```

```json
"ip_adapter_image": {
    "location": "./local/reference.png",
    "width": 512,
    "height": 512
}
```

Can also reference a previous step's output:

```json
"ip_adapter_image": "previous_result:preprocessing_step"
```

## Example

See [FluxIP.json](../examples/FluxIP.json) — Flux with IP-Adapter for style transfer.
