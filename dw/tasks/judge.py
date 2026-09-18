"""
Judge - score a candidate image against a rubric using a vision-language
model.

The first scorer for the select reducer (docs/proposals/score-and-select.md):
`select` needs a number per candidate, and this is the cheapest way to get
one without a purpose-built model - prompt the VLM `image_to_text` already
loads with a rubric and a scale, and parse its reply to one number. Delegates
to text_generation.generate_text, the same vision path image_to_text uses,
rather than keeping a second copy of the pipeline handling.

generate_text decodes greedily (do_sample=False - see its module docstring),
so the same rubric and image already return the same reply every run; there
is no sampling here for a step seed to thread through.
"""

import logging
import re

from .text_generation import generate_text, _DEFAULT_VISION_MODEL

logger = logging.getLogger("dw")

_DEFAULT_MODEL = _DEFAULT_VISION_MODEL

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _judge_prompt(rubric, scale):
    low, high = scale
    return (
        f"{rubric}\n\n"
        f"Respond with a single number from {low} to {high} and nothing else."
    )


def judge(image, rubric, scale, device="cpu", **kwargs):
    """Score an image against a rubric, returning one number.

    Args:
        image: PIL Image (or URL/path) to score.
        rubric: The question or criterion to score the image against.
        scale: [low, high] the score is expected to fall within.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            model_name: HuggingFace model ID of a vision-language model
                (default: image_to_text's default vision model).
            max_new_tokens: Max tokens to generate (default: 20).

    Returns:
        The parsed score as a float.

    Raises:
        ValueError: the model's reply did not contain a number.
    """
    prompt = _judge_prompt(rubric, scale)
    kwargs.setdefault("model_name", _DEFAULT_MODEL)
    kwargs.setdefault("max_new_tokens", 20)

    reply = generate_text(prompt, device=device, image=image, **kwargs)

    match = _NUMBER_PATTERN.search(reply)
    if match is None:
        raise ValueError(f"judge: could not parse a score from the reply: {reply!r}")

    score = float(match.group())
    logger.info(f"Judged: {score}")
    return score
