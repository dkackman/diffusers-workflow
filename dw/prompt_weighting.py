"""
Prompt weighting and long prompt support for diffusers pipelines.

Parses A1111-style prompt syntax: (word:1.5) for emphasis, [word] for de-emphasis,
((word)) for nested weighting. Supports prompts longer than the 77-token CLIP limit.

Produces prompt_embeds tensors that replace the prompt string argument in pipeline calls.

Based on sd_embed by Andrew Zhu (https://github.com/xhinker/sd_embed)
License: Apache 2.0
"""

import re
import gc
import logging
from typing import Tuple

import torch
from transformers import CLIPTokenizer, T5Tokenizer

logger = logging.getLogger("dw")


# ---------------------------------------------------------------------------
# Prompt parser — A1111-style (word:weight) syntax
# ---------------------------------------------------------------------------

_re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

_re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """Parse a prompt string with attention weights.

    Syntax:
        (abc)       — weight 1.1
        (abc:1.5)   — weight 1.5
        ((abc))     — weight 1.21 (1.1 * 1.1)
        [abc]       — weight 1/1.1 ≈ 0.91
        \\( \\)     — literal parens

    Returns list of [text, weight] pairs.
    """
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in _re_attention.finditer(text):
        text_match = m.group(0)
        weight = m.group(1)

        if text_match.startswith("\\"):
            res.append([text_match[1:], 1.0])
        elif text_match == "(":
            round_brackets.append(len(res))
        elif text_match == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text_match == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_match == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(_re_break, text_match)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _tokenize_clip_with_weights(clip_tokenizer: CLIPTokenizer, prompt: str):
    """Tokenize with CLIP and return (token_ids, weights)."""
    if not prompt:
        prompt = "empty"

    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []
    for word, weight in texts_and_weights:
        token = clip_tokenizer(word, truncation=False).input_ids[1:-1]
        text_tokens.extend(token)
        text_weights.extend([weight] * len(token))
    return text_tokens, text_weights


def _tokenize_t5_with_weights(t5_tokenizer: T5Tokenizer, prompt: str):
    """Tokenize with T5 and return (token_ids, weights)."""
    if not prompt:
        prompt = "empty"

    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []
    for word, weight in texts_and_weights:
        token = t5_tokenizer(word, truncation=False, add_special_tokens=True).input_ids
        text_tokens.extend(token)
        text_weights.extend([weight] * len(token))
    return text_tokens, text_weights


def _group_tokens_and_weights(token_ids, weights, pad_last_block=True):
    """Group tokens into 77-token chunks with BOS/EOS padding."""
    bos, eos = 49406, 49407

    new_token_ids = []
    new_weights = []

    # work on copies to avoid mutating originals
    token_ids = list(token_ids)
    weights = list(weights)

    while len(token_ids) >= 75:
        head_tokens = [token_ids.pop(0) for _ in range(75)]
        head_weights = [weights.pop(0) for _ in range(75)]
        new_token_ids.append([bos] + head_tokens + [eos])
        new_weights.append([1.0] + head_weights + [1.0])

    if len(token_ids) > 0:
        padding_len = 75 - len(token_ids) if pad_last_block else 0
        new_token_ids.append([bos] + token_ids + [eos] * padding_len + [eos])
        new_weights.append([1.0] + weights + [1.0] * padding_len + [1.0])

    return new_token_ids, new_weights


# ---------------------------------------------------------------------------
# Pipeline-specific weighted embedding functions
# ---------------------------------------------------------------------------


def _get_device(pipeline):
    """Get the appropriate compute device for the pipeline."""
    device = pipeline.device
    if device is not None and device.type != "cpu":
        return device

    # pipeline is on CPU (e.g., model offloading) — pick the best accelerator
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_weighted_text_embeddings_flux(
    pipe,
    prompt: str = "",
    prompt2: str = None,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate weighted text embeddings for Flux pipelines.

    Supports long prompts (beyond 77 tokens) and A1111-style weighting syntax.

    Args:
        pipe: A loaded FluxPipeline with tokenizer, tokenizer_2, text_encoder, text_encoder_2
        prompt: Primary prompt with optional weighting syntax
        prompt2: Optional second prompt for T5 encoder (defaults to prompt)
        device: Target device override

    Returns:
        (prompt_embeds, pooled_prompt_embeds) — pass directly to pipe() as kwargs
    """
    prompt2 = prompt if prompt2 is None else prompt2

    target_device = device if device is not None else _get_device(pipe)

    # Move text encoders to device if pipeline is using CPU offloading
    encoders_moved = False
    if pipe.device.type == "cpu":
        pipe.text_encoder.to(target_device)
        pipe.text_encoder_2.to(target_device)
        encoders_moved = True

    # Tokenize with CLIP (tokenizer 1) for pooled embeddings
    prompt_tokens, prompt_weights = _tokenize_clip_with_weights(pipe.tokenizer, prompt)
    prompt_token_groups, _ = _group_tokens_and_weights(prompt_tokens, prompt_weights)

    # Generate pooled CLIP embeddings (mean across token groups)
    pool_embeds_list = []
    for token_group in prompt_token_groups:
        token_tensor = torch.tensor(
            [token_group], dtype=torch.long, device=target_device
        )
        with torch.no_grad():
            embeds = pipe.text_encoder(token_tensor, output_hidden_states=False)
        pool_embeds_list.append(embeds.pooler_output.squeeze(0))

    pooled_prompt_embeds = torch.stack(pool_embeds_list, dim=0)
    pooled_prompt_embeds = pooled_prompt_embeds.mean(dim=0, keepdim=True)
    pooled_prompt_embeds = pooled_prompt_embeds.to(
        dtype=pipe.text_encoder.dtype, device=target_device
    )

    # Tokenize with T5 (tokenizer 2) for main prompt embeddings
    prompt_tokens_2, prompt_weights_2 = _tokenize_t5_with_weights(
        pipe.tokenizer_2, prompt2
    )

    token_tensor_2 = torch.tensor([prompt_tokens_2], dtype=torch.long)
    with torch.no_grad():
        t5_embeds = pipe.text_encoder_2(token_tensor_2.to(target_device))[0].squeeze(0)
    t5_embeds = t5_embeds.to(device=target_device)

    # Apply per-token weights to T5 embeddings
    for i in range(len(prompt_weights_2)):
        if prompt_weights_2[i] != 1.0:
            t5_embeds[i] = t5_embeds[i] * prompt_weights_2[i]

    prompt_embeds = t5_embeds.unsqueeze(0).to(
        dtype=pipe.text_encoder_2.dtype, device=target_device
    )

    # Release encoders back to CPU if we moved them
    if encoders_moved:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Dispatcher — selects the right function based on pipeline type
# ---------------------------------------------------------------------------

# Map pipeline class names to their embedding functions
_PIPELINE_FUNCTIONS = {
    "FluxPipeline": get_weighted_text_embeddings_flux,
    "FluxImg2ImgPipeline": get_weighted_text_embeddings_flux,
    "FluxInpaintPipeline": get_weighted_text_embeddings_flux,
    "FluxControlNetPipeline": get_weighted_text_embeddings_flux,
}


def apply_prompt_weighting(pipeline, arguments):
    """Apply prompt weighting to pipeline arguments if the prompt contains weight syntax.

    Checks if the prompt uses weighting syntax. If so, generates weighted embeddings
    and replaces the prompt string with embedding tensors in the arguments dict.

    Args:
        pipeline: The loaded diffusers pipeline
        arguments: Mutable dict of pipeline call arguments

    Returns:
        True if weighting was applied, False if prompt was left as-is.
    """
    prompt = arguments.get("prompt", None)
    if prompt is None or not isinstance(prompt, str):
        return False

    # Quick check: does the prompt contain any weighting syntax?
    if "(" not in prompt and "[" not in prompt:
        return False

    class_name = pipeline.__class__.__name__
    embed_fn = _PIPELINE_FUNCTIONS.get(class_name)
    if embed_fn is None:
        logger.warning(
            f"Prompt weighting not supported for {class_name}. "
            f"Supported: {', '.join(_PIPELINE_FUNCTIONS.keys())}. "
            f"Passing prompt as plain text."
        )
        return False

    logger.info(f"Applying prompt weighting for {class_name}")
    prompt2 = arguments.pop("prompt_2", None)
    prompt_str = arguments.pop("prompt")

    prompt_embeds, pooled_prompt_embeds = embed_fn(
        pipeline, prompt=prompt_str, prompt2=prompt2
    )

    arguments["prompt_embeds"] = prompt_embeds
    arguments["pooled_prompt_embeds"] = pooled_prompt_embeds

    # Remove negative_prompt if present — can't mix string and embeds
    if "negative_prompt" in arguments:
        logger.debug("Removing negative_prompt (incompatible with prompt_embeds)")
        arguments.pop("negative_prompt")

    return True
