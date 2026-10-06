"""Reduce a list of candidates to one by a deterministic rule.

The README's headline shape is N candidates -> choose one -> spend an
expensive stage on the winner. Today that choice lives at the agent
boundary (keep_output + asset:); this task is the version of it that needs
no judgment - argmax, a threshold, or a fixed index - so the recipe stays
inside the workflow and replays. See docs/proposals/score-and-select.md.
"""

import logging
import numbers

from ..media_types import Selected
from ..task_domains import SELECT_THRESHOLD_RULES, as_number, select_rule_problems

logger = logging.getLogger("dw")


# How each threshold rule tests a score. Which rules take a threshold, and
# the sentences that refuse a rule missing what it needs, live in
# dw/task_domains.py beside validate's copy of the same check
_THRESHOLD_TESTS = {
    "first_above": lambda score, threshold: score >= threshold,
    "first_below": lambda score, threshold: score <= threshold,
}


def _parse_score(index, score):
    if isinstance(score, bool) or not isinstance(score, (str, int, float)):
        raise ValueError(f"select: score {index} is not a number: {score!r}")
    try:
        return float(score)
    except (TypeError, ValueError):
        raise ValueError(f"select: score {index} is not a number: {score!r}")


def select(candidates, scores, rule, threshold=None, index=None):
    """Task command: pick one candidate out of many by a fixed rule.

    Args:
        candidates: The values to choose among, in order.
        scores: One number (or numeric string) per candidate, same length.
        rule: "argmax" | "argmin" | "first_above" | "first_below" | "index"
        threshold: Required by first_above/first_below.
        index: Required by rule "index" - a position into candidates.

    Returns:
        A Selected wrapping the winning value, its position and its score.

    Raises:
        ValueError: length mismatch, a non-numeric score, an unknown rule,
            a missing threshold/index, an out-of-range index, or (for
            first_above/first_below) no candidate meeting the threshold.
    """
    if len(candidates) != len(scores):
        raise ValueError(
            f"select got {len(candidates)} candidates and {len(scores)} scores "
            "- candidates and scores must be the same length"
        )

    parsed_scores = [_parse_score(i, s) for i, s in enumerate(scores)]

    problems = select_rule_problems(rule, threshold, index)
    if problems:
        raise ValueError(problems[0])

    if rule in ("argmax", "argmin"):
        positions = range(len(parsed_scores))
        picker = max if rule == "argmax" else min
        position = picker(positions, key=lambda i: parsed_scores[i])
    elif rule in SELECT_THRESHOLD_RULES:
        passes = _THRESHOLD_TESTS[rule]
        position = next(
            (i for i, s in enumerate(parsed_scores) if passes(s, threshold)), None
        )
        if position is None:
            raise ValueError(
                f"select: no candidate passes rule '{rule}' at threshold {threshold}"
            )
    else:  # "index" - select_rule_problems has refused a non-whole index
        # The range is checked here rather than by the domain alone: only the
        # run knows the candidate count, so one sentence names both ends
        position = (
            int(index) if isinstance(index, numbers.Integral) else int(as_number(index))
        )
        if position < 0 or position >= len(candidates):
            raise ValueError(
                f"select: index {index} is out of range for {len(candidates)} "
                f"candidates (0 to {len(candidates) - 1})"
            )

    logger.debug(f"select: rule={rule} chose position {position}")
    return Selected(
        value=candidates[position], position=position, score=parsed_scores[position]
    )
