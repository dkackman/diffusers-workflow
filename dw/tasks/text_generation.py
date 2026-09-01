"""
Text generation via HuggingFace transformers.

Takes a prompt (and optional system prompt) and generates text using a
local language model. Useful for prompt expansion, rewriting, and
other text-to-text tasks.

Supplying an image switches to a vision-language model, so the generated
text can describe what is actually in the picture rather than what the
prompt guesses is there. That is the difference between an image-conditioned
workflow whose prompt agrees with its keyframe and one whose prompt fights it.
"""

import logging
from transformers import pipeline as hf_pipeline
from .. import get_device_type, preferred_task_dtype
from .model_cache import cached_model

logger = logging.getLogger("dw")

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Small enough to stand in for the old captioning default; override for
# anything needing real detail
_DEFAULT_VISION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Greedy decoding against a long, rigid format specification makes the vision
# models loop - finishing the answer, then repeating its closing sections until
# they run out of tokens. A penalty stops that without giving up reproducible
# output, which sampling would. Measured on Qwen3-VL against the H3 prompt
# spec: 1.05 still looped and filled the whole budget, 1.15 ended on its own at
# a length matching the format's own guidance. The text models do not need it
_VISION_REPETITION_PENALTY = 1.15


def _image_part(image):
    """The chat content entry for an image.

    A PIL image goes in as an object; a string is a URL or path the pipeline
    loads itself, and it has to be declared as such - passing it under "image"
    would hand the processor a bare string where it expects pixels.
    """
    return {"type": "image", "url" if isinstance(image, str) else "image": image}


def _build_messages(prompt, system_prompt, image):
    """Chat messages in the shape the chosen pipeline expects.

    Text-generation models take plain string content. Vision models take a
    list of typed parts, because the image is a part of the message rather
    than something alongside it.
    """
    messages = []
    if image is None:
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
    else:
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [_image_part(image), {"type": "text", "text": prompt}],
            }
        )
    return messages


def generate_text(prompt, device="cpu", **kwargs):
    """Generate text from a prompt using a local language model.

    Args:
        prompt: The user message / prompt to expand or transform.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            model_name: HuggingFace model ID. Defaults to
                Qwen/Qwen2.5-1.5B-Instruct, or a small vision-language model
                when an image is supplied. An image needs a model that can
                accept one - a text-only model will fail to load as one.
            system_prompt: Optional system instruction for the model.
            max_new_tokens: Max tokens to generate (default: 500).
            image: Optional PIL image, URL or path. Its presence is what
                selects the vision pipeline.
            repetition_penalty: Vision pipeline only (default: 1.15). Raise it
                if a model still repeats itself, or set 1.0 to disable.
            generate_kwargs: Anything else to hand the model's generate() -
                no_repeat_ngram_size, top_p, min_new_tokens and so on. Merged
                over what this function sets, so it can override those too.

    Returns:
        Generated text string.
    """
    # A workflow declaring an optional image passes it through as null when the
    # caller supplies none, and an empty string is the same statement
    image = kwargs.get("image", None) or None
    system_prompt = kwargs.get("system_prompt", None)
    max_new_tokens = int(kwargs.get("max_new_tokens", 500))

    if image is None:
        pipeline_task = "text-generation"
        model_name = kwargs.get("model_name", _DEFAULT_MODEL)
    else:
        pipeline_task = "image-text-to-text"
        model_name = kwargs.get("model_name", _DEFAULT_VISION_MODEL)

    dtype = preferred_task_dtype(device)

    def load_pipe():
        logger.info(f"Generating text with {model_name} on {device}")
        # transformers materializes the weights across a thread pool, and a
        # device_map makes each of those threads cast its shard straight onto
        # the device. On MPS that races inside torch's Metal shader cache and
        # takes the process down mid-load - the same load crashed in
        # MetalShaderLibrary::exec_unary_kernel twice, once on a garbage
        # pointer and once on NULL. Passing `device` instead leaves the load on
        # the CPU and lets the pipeline move the finished model in one call on
        # this thread
        placement = (
            {"device": device}
            if get_device_type(device) == "mps"
            else {"device_map": device}
        )
        return hf_pipeline(
            pipeline_task,
            model=model_name,
            torch_dtype=dtype,
            **placement,
        )

    # The task is part of the identity - the same model name can be loaded
    # under either pipeline, and they are not interchangeable
    pipe = cached_model(
        ("text_generation", pipeline_task, model_name, str(device), str(dtype)),
        load_pipe,
    )

    messages = _build_messages(prompt, system_prompt, image)

    # Decoding is greedy, so the same input returns the same text every run -
    # which is what a workflow wants, and why there is nothing here to seed
    generation = {"do_sample": False}
    if image is not None:
        generation["repetition_penalty"] = float(
            kwargs.get("repetition_penalty", _VISION_REPETITION_PENALTY)
        )
    # Last, so a workflow can override anything decided above
    generation.update(kwargs.get("generate_kwargs") or {})

    return _generate(pipe, messages, image, max_new_tokens, generation)


def _generate(pipe, messages, image, max_new_tokens, generation):
    """Run the pipeline, which each path calls differently."""
    if image is None:
        # This pipeline collects anything it does not name into the arguments it
        # forwards to generate(), so settings go in as plain keywords
        results = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **generation,
        )
    else:
        # Images live inside the messages, so the chat goes in as `text` - the
        # pipeline rejects a chat and an `images` argument together. Generation
        # settings have to go through generate_kwargs here: anything else this
        # pipeline does not name explicitly is forwarded to the processor and
        # dropped, so a bare do_sample=False would leave sampling on. Passing
        # max_new_tokens both ways is an error, so it stays a direct argument
        results = pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            generate_kwargs=generation,
        )

    text = results[0]["generated_text"].strip()
    logger.info(f"Generated: {text[:100]}{'...' if len(text) > 100 else ''}")
    return text
