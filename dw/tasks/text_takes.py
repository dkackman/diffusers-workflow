"""split_takes: cut a long text into takes a speech model can say in one go.

A TTS model has a context, not a stamina: Audio8-TTS holds about 95 s of
reference, text and speech together (`templates/voice-clone`), and a line past
that is refused. A narration longer than one take is therefore spoken take by
take under one voice and seed and joined (`templates/narration-long`).

Where a text is cut decides whether the join can be heard. A cut between two
sentences lands on a pause the speaker would have made anyway; a cut inside one
puts a breath where there is none. So takes end at sentence boundaries, and only
a sentence that alone is over the limit is cut inside - at its last comma that
fits, else at a word.

The limit is a duration turned into words with `words_per_second`, an estimate
(written dialogue measured 2.5-4 words/s as filling a clip) - the first take of
a run is the check on it, not this function.
"""

import logging
import math
import re

logger = logging.getLogger("dw")

# A sentence ends at . ! ? or an ellipsis, optionally closed by quotes or
# brackets, and is followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")


def _words_allowed(max_seconds, words_per_second):
    for name, value in (
        ("max_seconds", max_seconds),
        ("words_per_second", words_per_second),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"split_takes: {name} must be above zero, got {value!r}")
    return max(1, int(max_seconds * words_per_second))


def _sentences(text):
    """The sentences of the text, each with its whitespace collapsed."""
    parts, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        # Keep the closing quote with the sentence it closes
        parts.append(text[start : match.start() + len(match.group().rstrip())])
        start = match.end()
    parts.append(text[start:])
    return [" ".join(p.split()) for p in parts if p.strip()]


def _cut_long(sentence, limit):
    """A sentence over the limit as pieces within it: cut after the last comma
    that fits, else at the limit."""
    pieces, words = [], sentence.split()
    while len(words) > limit:
        cut = limit
        for index in range(limit - 1, 0, -1):
            if words[index].endswith((",", ";", ":", "—")):
                cut = index + 1
                break
        pieces.append(" ".join(words[:cut]))
        words = words[cut:]
    pieces.append(" ".join(words))
    return pieces


def split_takes(text, max_seconds=60.0, words_per_second=2.5):
    """Task command: split a text into takes of at most `max_seconds` of speech.

    Args:
        text: The text to speak
        max_seconds: The longest a take may run. For `voice-clone` this is the
            ~95 s shared context less the reference clip, so 60 leaves room
        words_per_second: The speaking rate used to turn the limit into words

    Returns:
        A list of takes, in order. Joined with a space they read as the text,
        less its original whitespace

    Raises:
        ValueError: If the text is empty or a limit is not above zero
    """
    limit = _words_allowed(max_seconds, words_per_second)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("split_takes needs some text to split")

    units = []
    for sentence in _sentences(text):
        if len(sentence.split()) > limit:
            logger.warning(
                f"split_takes: a sentence of {len(sentence.split())} words is over "
                f"the {limit}-word limit and is cut inside"
            )
            units.extend(_cut_long(sentence, limit))
        else:
            units.append(sentence)

    takes, current, used = [], [], 0
    for unit in units:
        size = len(unit.split())
        if current and used + size > limit:
            takes.append(" ".join(current))
            current, used = [], 0
        current.append(unit)
        used += size
    if current:
        takes.append(" ".join(current))
    return takes


def split_takes_task(text, max_seconds=60.0, words_per_second=2.5):
    """Task command: split a text into takes, answered as a JSON document.

    The takes are a `for_each` list (`templates/narration-long`): the engine
    expands `for_each` before a run, so run this first - `templates/split-takes`
    - and pass its `takes` in.

    Args:
        text: The text to speak
        max_seconds: The longest a take may run
        words_per_second: The speaking rate used to turn the limit into words

    Returns:
        {takes, count, words}: the takes in order, how many, and each one's
        word count
    """
    takes = split_takes(text, max_seconds, words_per_second)
    return {
        "takes": takes,
        "count": len(takes),
        "words": [len(t.split()) for t in takes],
    }


def _register():
    from .registry import register_command
    from ..task_domains import POSITIVE

    @register_command(
        "split_takes",
        implementation="dw.tasks.text_takes.split_takes_task",
        returns="json",
        domains={"max_seconds": POSITIVE, "words_per_second": POSITIVE},
    )
    def _handle_split_takes(task, arguments, previous_pipelines):
        """Split a long text into takes a speech model can say in one go"""
        logger.debug("Splitting takes")
        return split_takes_task(**arguments)


_register()
