"""
Image-to-text captioning via HuggingFace transformers pipeline.

Supports BLIP, BLIP-2, ViT-GPT2, GIT, and other models compatible
with the transformers image-to-text pipeline.
"""

import logging
import torch
from transformers import pipeline as hf_pipeline

logger = logging.getLogger("dw")

_DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


def image_to_text(image, device="cpu", **kwargs):
    """Generate a text caption for an image.

    Args:
        image: PIL Image to caption.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            model_name: HuggingFace model ID (default: Salesforce/blip-image-captioning-base).
            prompt: Optional text prompt for conditional captioning (BLIP-2, etc.).
            max_new_tokens: Max tokens to generate (default: 50).

    Returns:
        Caption string.
    """
    model_name = kwargs.get("model_name", _DEFAULT_MODEL)
    prompt = kwargs.get("prompt", None)
    max_new_tokens = int(kwargs.get("max_new_tokens", 50))

    logger.info(f"Captioning image with {model_name} on {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    # Use device_map instead of device to avoid caching_allocator_warmup
    # buffer pre-allocation failures on MPS and with larger models.
    pipe = hf_pipeline(
        "image-to-text",
        model=model_name,
        device_map=device,
        torch_dtype=dtype,
    )

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    if prompt is not None:
        generate_kwargs["prompt"] = prompt

    results = pipe(image, generate_kwargs=generate_kwargs)

    caption = results[0]["generated_text"].strip()
    logger.info(f"Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    return caption
