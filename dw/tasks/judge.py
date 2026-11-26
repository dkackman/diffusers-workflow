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
is no sampling here for a step seed to thread through. It also means a reply
with no number in it would come back unchanged however often the same prompt
were asked, so the second ask is worded differently. A small VLM answers some
rubrics with a critique ("Malformed limbs.") and no digit, and one such reply
must not fail a best-of-n run (#1755).
"""

import logging
import re

from ..events import emit_warning
from .text_generation import generate_text, DEFAULT_VISION_MODEL
from .text_generation import vision_required_weights as required_weights  # noqa: F401

logger = logging.getLogger("dw")

_DEFAULT_MODEL = DEFAULT_VISION_MODEL

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _judge_prompt(rubric, scale):
    low, high = scale
    return (
        f"{rubric}\n\n"
        f"Respond with a single number from {low} to {high} and nothing else."
    )


def _digits_prompt(rubric, scale):
    """The second ask, for a reply to the first that held no number."""
    low, high = scale
    return (
        f"{rubric}\n\n"
        f"Score it as a whole number from {low} to {high}. Answer with digits only."
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
        The parsed score as a float. A reply with no number is asked for
        again in other words; if that one has none either, the score is the
        low end of `scale` and the run gets a warning.
    """
    kwargs.setdefault("model_name", _DEFAULT_MODEL)
    kwargs.setdefault("max_new_tokens", 20)

    replies = []
    for ask in (_judge_prompt, _digits_prompt):
        reply = generate_text(ask(rubric, scale), device=device, image=image, **kwargs)
        match = _NUMBER_PATTERN.search(reply)
        if match is not None:
            score = float(match.group())
            logger.info(f"Judged: {score}")
            return score
        replies.append(reply)

    score = float(scale[0])
    emit_warning(
        f"judge: no score in the reply {replies[0]!r}, nor in the reply "
        f"{replies[1]!r} when asked again; scored {score:g}, the low end of "
        "scale. A select by argmin or first_below reads that as the best score",
        command="judge",
        replies=replies,
        score=score,
    )
    return score
