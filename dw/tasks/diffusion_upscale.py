"""
Diffusion-based image upscaling via Stable Diffusion upscale pipelines.

Provides text-guided upscaling with better detail recovery than
traditional super-resolution models, especially for faces and textures.

Supports two modes:
  - "x4" (default): StableDiffusionUpscalePipeline (4x, stabilityai/stable-diffusion-x4-upscaler)
  - "x2": StableDiffusionLatentUpscalePipeline (2x, stabilityai/sd-x2-latent-upscaler)
"""

import logging
import torch
import diffusers

logger = logging.getLogger("dw")

_MODELS = {
    "x4": {
        "pipeline_class": "StableDiffusionUpscalePipeline",
        "model_name": "stabilityai/stable-diffusion-x4-upscaler",
    },
    "x2": {
        "pipeline_class": "StableDiffusionLatentUpscalePipeline",
        "model_name": "stabilityai/sd-x2-latent-upscaler",
    },
}


def diffusion_upscale(image, device="cpu", **kwargs):
    """Upscale an image using a Stable Diffusion upscale pipeline.

    Args:
        image: PIL Image to upscale.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            prompt: Text guidance for upscaling (default: "").
            negative_prompt: Negative text guidance (default: None).
            mode: "x4" or "x2" (default: "x4").
            model_name: Override the default model for the selected mode.
            num_inference_steps: Denoising steps (default: 25).
            guidance_scale: Classifier-free guidance scale (default: 9.0).
            noise_level: Noise level for x4 mode (default: 20, ignored for x2).

    Returns:
        PIL Image (upscaled).
    """
    mode = kwargs.get("mode", "x4")
    if mode not in _MODELS:
        raise ValueError(f"mode must be one of {sorted(_MODELS.keys())}, got '{mode}'")

    config = _MODELS[mode]
    model_name = kwargs.get("model_name", config["model_name"])
    prompt = kwargs.get("prompt", "")
    negative_prompt = kwargs.get("negative_prompt", None)
    num_inference_steps = int(kwargs.get("num_inference_steps", 25))
    guidance_scale = float(kwargs.get("guidance_scale", 9.0))
    noise_level = int(kwargs.get("noise_level", 20))

    pipeline_class = getattr(diffusers, config["pipeline_class"])

    logger.info(
        f"Loading {config['pipeline_class']} from {model_name} to {device}"
    )

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = pipeline_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    pipe.to(device)

    call_kwargs = {
        "prompt": prompt,
        "image": image,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    if negative_prompt is not None:
        call_kwargs["negative_prompt"] = negative_prompt

    if mode == "x4":
        call_kwargs["noise_level"] = noise_level

    logger.info(
        f"Upscaling {image.width}x{image.height} with {mode} mode, "
        f"{num_inference_steps} steps"
    )

    with torch.inference_mode():
        result = pipe(**call_kwargs)

    output = result.images[0]
    logger.info(f"Upscaled to {output.width}x{output.height}")
    return output
