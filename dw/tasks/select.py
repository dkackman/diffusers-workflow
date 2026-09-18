"""Reduce a list of candidates to one by a deterministic rule.

The README's headline shape is N candidates -> choose one -> spend an
expensive stage on the winner. Today that choice lives at the agent
boundary (keep_output + asset:); this task is the version of it that needs
no judgment - argmax, a threshold, or a fixed index - so the recipe stays
inside the workflow and replays. See docs/proposals/score-and-select.md.
"""

import logging

logger = logging.getLogger("dw")


class Selected:
    """The winning candidate, plus the metadata that makes the choice
    replayable (position, score). Compares equal to its own value so a
    caller that only wants the winner can treat it as one."""

    def __init__(self, value, position, score):
        self.value = value
        self.position = position
        self.score = score

    def __eq__(self, other):
        if isinstance(other, Selected):
            return (
                self.value == other.value
                and self.position == other.position
                and self.score == other.score
            )
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"Selected(value={self.value!r}, position={self.position}, score={self.score!r})"


_THRESHOLD_RULES = {
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

    if rule in ("argmax", "argmin"):
        positions = range(len(parsed_scores))
        picker = max if rule == "argmax" else min
        position = picker(positions, key=lambda i: parsed_scores[i])
    elif rule in _THRESHOLD_RULES:
        if threshold is None:
            raise ValueError(f"select rule '{rule}' requires a threshold")
        passes = _THRESHOLD_RULES[rule]
        position = next(
            (i for i, s in enumerate(parsed_scores) if passes(s, threshold)), None
        )
        if position is None:
            raise ValueError(
                f"select: no candidate passes rule '{rule}' at threshold {threshold}"
            )
    elif rule == "index":
        if index is None:
            raise ValueError("select rule 'index' requires an index")
        if index < 0 or index >= len(candidates):
            raise ValueError(
                f"select: index {index} is out of range for {len(candidates)} candidates"
            )
        position = index
    else:
        raise ValueError(f"select: unknown rule: {rule!r}")

    logger.debug(f"select: rule={rule} chose position {position}")
    return Selected(
        value=candidates[position], position=position, score=parsed_scores[position]
    )
