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

import json
import logging
import os
import re

from huggingface_hub import hf_hub_download
from transformers import pipeline as hf_pipeline
from .. import preferred_task_dtype
from ..locations import validate_hub_repo_id, validate_weight_name
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


# A pinned prompt has to stay pinned: a short hash can become ambiguous and a
# branch moves, so only a full commit is accepted
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FROM_HUB_FIELDS = {"repo", "file", "revision", "key"}


# A system prompt is a page of instructions; anything bigger is not one
_MAX_PROMPT_FILE_BYTES = 1024 * 1024


def resolve_system_prompt(value):
    """The system prompt text, reading it from a Hub repo when one is named.

    Some vendors ship the instructions their model was trained against as a
    file in its repo, and change them. Naming the file and a full commit runs
    exactly that text without copying it into a workflow, where it would
    drift from the source.

    Args:
        value: None, the text itself, or {"from_hub": {"repo", "file",
            "revision", "key"}}. `revision` is a full 40-character commit.
            A .json file takes `key` to pick one string out of a top-level
            object; any other file is read whole as text

    Raises:
        ValueError: On any other shape, a revision that is not a full commit,
            a file path that leaves the repo, a missing key, or a value that
            is not text
    """
    if value is None or isinstance(value, str):
        return value
    if (
        not isinstance(value, dict)
        or set(value) != {"from_hub"}
        or not isinstance(value["from_hub"], dict)
    ):
        raise ValueError(
            'system_prompt must be text or {"from_hub": {"repo", "file", "revision", "key"}}'
        )
    spec = value["from_hub"]
    unknown = set(spec) - _FROM_HUB_FIELDS
    if unknown:
        raise ValueError(f"system_prompt.from_hub does not take {sorted(unknown)}")
    repo = spec.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ValueError(
            f"system_prompt.from_hub.repo must be a Hub repo id string, got {repo!r}"
        )
    repo = validate_hub_repo_id(repo, what="system_prompt.from_hub.repo")
    file = validate_weight_name(spec.get("file"), what="system_prompt.from_hub.file")
    revision = spec.get("revision")
    if not isinstance(revision, str) or not _FULL_COMMIT.match(revision):
        raise ValueError(
            "system_prompt.from_hub.revision must be a full 40-character lowercase "
            f"commit hash, so the text cannot change under the workflow; got {revision!r}"
        )
    key = spec.get("key")
    if key is not None and not isinstance(key, str):
        raise ValueError(f"system_prompt.from_hub.key must be a string, got {key!r}")

    # token=False: the server's own Hub login must not reach a private or gated
    # file on a workflow's say-so, since the text is echoed into outputs
    path = hf_hub_download(repo, file, revision=revision, token=False)
    if os.path.getsize(path) > _MAX_PROMPT_FILE_BYTES:
        raise ValueError(
            f"system_prompt.from_hub: {file} is larger than "
            f"{_MAX_PROMPT_FILE_BYTES} bytes, too big for a system prompt"
        )
    with open(path, encoding="utf-8") as f:
        if file.endswith(".json"):
            data = json.load(f)
        else:
            data = f.read()
    if key is not None:
        if not isinstance(data, dict) or key not in data:
            keys = sorted(data) if isinstance(data, dict) else []
            raise ValueError(
                f"system_prompt.from_hub: {file} at {revision[:8]} has no key {key!r}; "
                f"it has {keys}"
            )
        data = data[key]
    if not isinstance(data, str) or not data.strip():
        raise ValueError(
            f"system_prompt.from_hub: {file}{'[' + repr(key) + ']' if key else ''} "
            "is not text"
        )
    return data


# The pipelines' and apply_chat_template's own structural arguments. Chat
# template settings are splatted into those calls, so one of these in a
# workflow (which is untrusted) would not set a template variable - it would
# hand transformers an image or video to fetch that dw never validated, or
# change what the template call does (tokenize, tools, the template itself)
_RESERVED_CHAT_TEMPLATE_KEYS = frozenset(
    {
        "images",
        "videos",
        "audio",
        "audios",
        "text",
        "timeout",
        "processor_kwargs",
        "generate_kwargs",
        "max_new_tokens",
        "return_full_text",
        "return_tensors",
        "tokenize",
        "return_dict",
        "chat_template",
        "add_generation_prompt",
        "continue_final_message",
        "tools",
        "documents",
        "tokenizer_encode_kwargs",
    }
)


def _validated_chat_template_kwargs(value):
    """Chat template settings: an object of JSON scalars, none of them one of
    the call's own structural arguments."""
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "text_generation: chat_template_kwargs must be an object of settings "
            f"for the chat template, got {type(value).__name__}"
        )
    for key, setting in value.items():
        if key in _RESERVED_CHAT_TEMPLATE_KEYS:
            raise ValueError(
                f"text_generation: chat_template_kwargs key '{key}' is not a "
                "chat-template setting - it is an argument of the generation "
                "call itself"
            )
        if setting is not None and not isinstance(setting, (bool, int, float, str)):
            raise ValueError(
                f"text_generation: chat_template_kwargs value for '{key}' must be "
                f"a boolean, number, string or null, got {type(setting).__name__}"
            )
    return value


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
    # Resolved before anything loads, so a bad reference costs nothing
    system_prompt = resolve_system_prompt(kwargs.get("system_prompt", None))
    max_new_tokens = int(kwargs.get("max_new_tokens", 500))
    revision = kwargs.get("revision", None)
    chat_template_kwargs = _validated_chat_template_kwargs(
        kwargs.get("chat_template_kwargs")
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
