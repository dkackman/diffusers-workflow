# IP-Adapter

IP-Adapter enables image-prompt conditioning — use a reference image to influence the style or content of generated images alongside a text prompt. It works with any pipeline whose `load_ip_adapter()` diffusers supports — Flux, Stable Diffusion 1.5, SDXL, SD 3.5, and others.

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

Any other property (e.g. `revision`) is forwarded as-is to the underlying `load_ip_adapter()` call.

## Models Without a Built-in Image Encoder

Some base models (e.g. Stable Diffusion 3.5) don't ship a default image encoder, so diffusers needs one configured explicitly. Add `image_encoder` and `feature_extractor` components alongside `ip_adapter`, the same way any other pipeline component is configured:

```json
{
    "pipeline": {
        "configuration": { "component_type": "StableDiffusion3Pipeline" },
        "feature_extractor": {
            "configuration": { "component_type": "transformers.SiglipImageProcessor" },
            "from_pretrained_arguments": { "model_name": "google/siglip-so400m-patch14-384" }
        },
        "image_encoder": {
            "configuration": { "component_type": "transformers.SiglipVisionModel" },
            "from_pretrained_arguments": { "model_name": "google/siglip-so400m-patch14-384" }
        },
        "ip_adapter": {
            "model_name": "InstantX/SD3.5-Large-IP-Adapter",
            "weight_name": "ip-adapter.bin",
            "scale": 0.6
        },
        "from_pretrained_arguments": {
            "model_name": "stabilityai/stable-diffusion-3.5-large",
            "torch_dtype": "torch.bfloat16"
        },
        "arguments": {
            "prompt": "a marmot drinks a milkshake",
            "ip_adapter_image": { "location": "https://example.com/reference.jpg" }
        }
    }
}
```

See [sd35ip.json](../examples/sd35ip.json) for the full workflow.

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

## Examples

- [FluxIP.json](../examples/FluxIP.json) — Flux with IP-Adapter for style transfer
- [sd35ip.json](../examples/sd35ip.json) — SD 3.5 (quantized) with an explicit `image_encoder`/`feature_extractor` pair
- [ip-adapter.json](../examples/ip-adapter.json) — SD 1.5 and SDXL, each with an `AutoPipelineForText2Image` step
