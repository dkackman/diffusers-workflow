"""Assemble one block of text out of parts written once.

A multi-shot workflow says the same things about its characters in every
shot: who they are, what they are wearing, what their voice sounds like. The
engine has no interpolation - a reference is the whole value of an argument,
never a fragment spliced into one - which is deliberate (docs/WORKFLOW_GUIDE.md,
'no interpolation'): a `{{name}}` inside a prompt would make every prompt a
template language nobody declared, and resolution order for a half-substituted
string is a bottomless pit.

The way out is not interpolation but composition. A part is a whole value -
a variable, an earlier step's output, a stored prompt - and this task joins
parts in the order they are given. A character bible is then written once,
as a variable, and named by every shot that needs it; changing the voice
changes it everywhere, and nothing has to be hand-copied to stay in step.

Positional, not named: the parts are a list, joined in order. A named form
('{bible} says {line}') would be the interpolation the engine does not have,
one layer down.
"""

import logging

logger = logging.getLogger("dw")


def compose_text(parts, separator="\n\n", skip_empty=True):
    """Task command: join parts into one block of text.

    Args:
        parts: The parts to join, in order. Each is a whole value - usually
            a "variable:", "prompt:" or "previous_result:" reference the
            engine has already resolved. Numbers are written out; None is
            dropped, so an optional part can be a variable left null
        separator: What goes between the parts. Defaults to a blank line,
            the paragraph break the prompt formats are written in
        skip_empty: Drop parts that are None or empty. With it off, an
            empty part still contributes its separator

    Returns:
        The joined text

    Raises:
        ValueError: If parts is not a list, or holds something that is not
            text or a number - a dict or an image is a sign a reference
            resolved to something other than what was meant
    """
    if not isinstance(parts, list):
        raise ValueError(
            f"compose_text needs a list of parts to join, got {type(parts).__name__}"
        )

    pieces = []
    for index, part in enumerate(parts):
        if part is None:
            if skip_empty:
                continue
            part = ""
        if isinstance(part, bool) or not isinstance(part, (str, int, float)):
            raise ValueError(
                f"compose_text part {index} is a {type(part).__name__} - a part "
                f"is text (or a number), and anything else means the reference "
                f"in that position resolved to something other than text"
            )
        text = part if isinstance(part, str) else str(part)
        if skip_empty and not text.strip():
            continue
        pieces.append(text)

    if not pieces:
        raise ValueError("compose_text was given nothing to join")

    logger.debug(f"compose_text: joined {len(pieces)} of {len(parts)} parts")
    return separator.join(pieces)
