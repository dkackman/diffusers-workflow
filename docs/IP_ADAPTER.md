# IP-Adapter

IP-Adapter conditions a generation on a reference image alongside the text prompt, to carry its style or content. It works on any pipeline that has diffusers' `load_ip_adapter()`: Flux, Stable Diffusion 1.5, SDXL, SD 3.5 and others.

## Usage

Add an `ip_adapter` block to the pipeline, a sibling of `configuration`, and pass the image as the `ip_adapter_image` argument:

```json
{
    "pipeline": {
        "configuration": { "component_type": "FluxPipeline" },
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

FLUX.1 dev in bf16 needs a placement to fit a 24 GB card; [ip-adapter.json](../workflows/templates/ip-adapter.json) uses the streamed group offload from [RECIPES_24GB.md](RECIPES_24GB.md#flux-dev-12b).

## Properties

| Property | Required | Description |
| -------- | -------- | ----------- |
| `model_name` | Yes | Hugging Face Hub repo id of the IP-Adapter weights |
| `weight_name` | No | Specific weight file in the repo |
| `subfolder` | No | Subfolder within the repo |
| `scale` | No | Adapter strength, set with `set_ip_adapter_scale()` after the load. Unset keeps diffusers' default of 1.0. Lower = less influence from the reference image |

Any other property (e.g. `revision`) is passed to `load_ip_adapter()` unchanged.

## Models Without a Built-in Image Encoder

Some base models (Stable Diffusion 3.5, for one) ship no image encoder, so diffusers needs one configured. Add `image_encoder` and `feature_extractor` component blocks beside `ip_adapter`, configured like any other pipeline component:

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

## Image Argument

`ip_adapter_image` loads like any `*_image` argument ([Media Arguments](WORKFLOW_GUIDE.md#media-arguments), [References](WORKFLOW_GUIDE.md#references)): a URL, a path relative to the workflow file, an `asset:` name on the server, or a `{"location": ...}` dict of any of these. Only `location` is read from the dict; the image is not resized.

```json
"ip_adapter_image": "asset:reference.png"
```

It can also take an earlier step's output:

```json
"ip_adapter_image": "previous_result:preprocessing_step"
```

## Examples

- [ip-adapter.json](../workflows/templates/ip-adapter.json): FLUX.1 dev with `XLabs-AI/flux-ip-adapter`. Its description gives the face-model variant, which keeps a person's face across a new scene: `h94/IP-Adapter`, subfolder `models`, weight `ip-adapter-full-face_sd15.bin`, scale 0.5, on a `StableDiffusionPipeline`.
