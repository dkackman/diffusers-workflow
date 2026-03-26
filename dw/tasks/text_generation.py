"""
Text generation via HuggingFace transformers text-generation pipeline.

Takes a prompt (and optional system prompt) and generates text using a
local language model. Useful for prompt expansion, rewriting, and
other text-to-text tasks.
"""

import logging
import torch
from transformers import pipeline as hf_pipeline

logger = logging.getLogger("dw")

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def generate_text(prompt, device="cpu", **kwargs):
    """Generate text from a prompt using a local language model.

    Args:
        prompt: The user message / prompt to expand or transform.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            model_name: HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct).
            system_prompt: Optional system instruction for the model.
            max_new_tokens: Max tokens to generate (default: 500).

    Returns:
        Generated text string.
    """
    model_name = kwargs.get("model_name", _DEFAULT_MODEL)
    system_prompt = kwargs.get("system_prompt", None)
    max_new_tokens = int(kwargs.get("max_new_tokens", 500))

    logger.info(f"Generating text with {model_name} on {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = hf_pipeline(
        "text-generation",
        model=model_name,
        device_map=device,
        torch_dtype=dtype,
    )

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    results = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        do_sample=False,
    )

    text = results[0]["generated_text"].strip()
    logger.info(f"Generated: {text[:100]}{'...' if len(text) > 100 else ''}")
    return text
