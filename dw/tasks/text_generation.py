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
from .. import preferred_task_dtype
from .model_cache import cached_model, hf_pipeline_placement

logger = logging.getLogger("dw")

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Small enough to stand in for the old captioning default; override for
# anything needing real detail
DEFAULT_VISION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

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


def _images(image):
    """The images a call carries, as a list in order, or None for none.

    A template's optional reference list arrives as [] or [None] when the
    caller supplied nothing, and that is the same statement as no image.
    """
    if image is None:
        return None
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    images = [i for i in images if i is not None and i != ""]
    return images or None


def _build_messages(prompt, system_prompt, images):
    """Chat messages in the shape the chosen pipeline expects.

    Text-generation models take plain string content. Vision models take a
    list of typed parts, because the images are part of the message rather
    than something alongside it - every image first, in order, then the text.
    """
    messages = []
    if images is None:
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
                "content": [_image_part(i) for i in images]
                + [{"type": "text", "text": prompt}],
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
            image: Optional PIL image, URL or path, or a list of them - every
                one goes to the model, in order. Its presence is what selects
                the vision pipeline; an empty list is no image.
            chat_template_kwargs: Settings for the model's chat template,
                such as {"enable_thinking": false} for a Qwen3 model that
                must not reason before answering.
            revision: The model repo's commit, branch or tag to load.
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
    images = _images(kwargs.get("image", None) or None)
    system_prompt = kwargs.get("system_prompt", None)
    max_new_tokens = int(kwargs.get("max_new_tokens", 500))
    revision = kwargs.get("revision", None)
    chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError(
            "text_generation: chat_template_kwargs must be an object of settings "
            f"for the chat template, got {type(chat_template_kwargs).__name__}"
        )

    if images is None:
        pipeline_task = "text-generation"
        model_name = kwargs.get("model_name", _DEFAULT_MODEL)
    else:
        pipeline_task = "image-text-to-text"
        model_name = kwargs.get("model_name", DEFAULT_VISION_MODEL)

    dtype = preferred_task_dtype(device)

    def load_pipe():
        logger.info(f"Generating text with {model_name} on {device}")
        placement = hf_pipeline_placement(device)
        if revision is not None:
            placement = {**placement, "revision": revision}
        return hf_pipeline(
            pipeline_task,
            model=model_name,
            torch_dtype=dtype,
            **placement,
        )

    # The task is part of the identity - the same model name can be loaded
    # under either pipeline, and they are not interchangeable
    pipe = cached_model(
        (
            "text_generation",
            pipeline_task,
            model_name,
            str(revision),
            str(device),
            str(dtype),
        ),
        load_pipe,
    )

    messages = _build_messages(prompt, system_prompt, images)

    # Decoding is greedy, so the same input returns the same text every run -
    # which is what a workflow wants, and why there is nothing here to seed
    generation = {"do_sample": False}
    if images is not None:
        generation["repetition_penalty"] = float(
            kwargs.get("repetition_penalty", _VISION_REPETITION_PENALTY)
        )
    # Last, so a workflow can override anything decided above
    generation.update(kwargs.get("generate_kwargs") or {})

    return _generate(
        pipe, messages, images, max_new_tokens, generation, chat_template_kwargs
    )


def _generate(pipe, messages, images, max_new_tokens, generation, chat_template_kwargs):
    """Run the pipeline, which each path calls differently."""
    if images is None:
        # This pipeline collects anything it does not name into the arguments it
        # forwards to generate(), so settings go in as plain keywords. Chat
        # template settings have their own argument, which it hands to
        # apply_chat_template
        extra = (
            {"tokenizer_encode_kwargs": chat_template_kwargs}
            if chat_template_kwargs
            else {}
        )
        results = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **generation,
            **extra,
        )
    else:
        # Images live inside the messages, so the chat goes in as `text` - the
        # pipeline rejects a chat and an `images` argument together. Generation
        # settings have to go through generate_kwargs here: anything else this
        # pipeline does not name explicitly is forwarded to the processor and
        # dropped, so a bare do_sample=False would leave sampling on. Passing
        # max_new_tokens both ways is an error, so it stays a direct argument.
        # Keywords this pipeline does not name go to the processor's
        # apply_chat_template, which is where the chat template settings belong
        results = pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            generate_kwargs=generation,
            **chat_template_kwargs,
        )

    text = results[0]["generated_text"].strip()
    logger.info(f"Generated: {text[:100]}{'...' if len(text) > 100 else ''}")
    return text
