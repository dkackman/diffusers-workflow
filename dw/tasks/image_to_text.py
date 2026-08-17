"""
Image captioning via HuggingFace transformers.

Captioning is text generation that happens to be given a picture, so this
delegates to the text_generation task rather than keeping a second copy of
the pipeline handling. Transformers 5 removed the dedicated "image-to-text"
task this used to build, and with it the BLIP-style captioning models; the
work is done by vision-language models now.
"""

import logging
from .text_generation import generate_text, _DEFAULT_VISION_MODEL

logger = logging.getLogger("dw")

_DEFAULT_MODEL = _DEFAULT_VISION_MODEL
_DEFAULT_PROMPT = "Describe this image."


def image_to_text(image, device="cpu", **kwargs):
    """Generate a text caption for an image.

    Args:
        image: PIL Image (or URL/path) to caption.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            model_name: HuggingFace model ID of a vision-language model
                (default: HuggingFaceTB/SmolVLM-256M-Instruct).
            prompt: What to ask about the image (default: "Describe this
                image."). Ask a narrower question for a narrower caption.
            system_prompt: Optional system instruction for the model.
            max_new_tokens: Max tokens to generate (default: 50).

    Returns:
        Caption string.
    """
    prompt = kwargs.pop("prompt", _DEFAULT_PROMPT)
    kwargs.setdefault("model_name", _DEFAULT_MODEL)
    kwargs.setdefault("max_new_tokens", 50)

    caption = generate_text(prompt, device=device, image=image, **kwargs)
    logger.info(f"Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    return caption
