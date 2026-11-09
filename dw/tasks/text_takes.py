"""split_takes: cut a long text into takes a speech model can say in one go.

Audio8-TTS holds about 95 s of reference, text and speech together
(`templates/voice-clone`), and a line past that is refused - but the limit that
bites is shorter. Measured on the S6 bench round (`bench/records/audio.md`), a
take of 125-144 words (41-48 s of speech) looped, skipped or stopped early,
while takes of up to 109 words were word-complete. A narration longer than one
take is therefore spoken take by take under one voice and seed and joined
(`templates/narration-long`).

Where a text is cut decides whether the join can be heard. A cut between two
sentences lands on a pause the speaker would have made anyway; a cut inside one
puts a breath where there is none. So takes end at sentence boundaries, and only
a sentence that alone is over the limit is cut inside - at its last comma
(or semicolon, colon or dash) before the limit, else, with no such mark, at the
word boundary nearest the limit that does not pass it. Takes are packed
greedily, except that a take already PARAGRAPH_CUT_FILL of the way to the
limit ends at a paragraph break (a blank line) rather than running into the
next paragraph: the join's pause then falls where the text has one.

The limit is a duration turned into words with `words_per_second`, an estimate
(the S6 voice spoke 2.93-3.10 words/s; written dialogue measured 2.5-4 words/s
as filling a clip) - the first take of a run is the check on it, not this
function. The defaults, 37 s at 3.0 words/s, are 111 words: the longest take
the bench measured word-complete.
"""

import logging
import math
import re

logger = logging.getLogger("dw")

# A sentence ends at . ! ? or an ellipsis, optionally closed by quotes or
# brackets, and is followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")

# A paragraph break: a line break, any blank lines, and another line break
_PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*(?:\n[^\S\n]*)+")

# How full a take must be (as a share of the word limit) before a paragraph
# break ends it early. Below this, cutting at the paragraph would leave a
# short take and one more join; above it, the join lands on the pause the
# text already has
PARAGRAPH_CUT_FILL = 0.6

DEFAULT_MAX_SECONDS = 37
DEFAULT_WORDS_PER_SECOND = 3.0


# A period after one of these, or after a single capital letter (an initial),
# is not the end of a sentence
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof st sr jr vs etc e.g i.e mt no fig inc ltd co corp "
    "jan feb mar apr jun jul aug sep sept oct nov dec u.s u.k".split()
)


def _is_abbreviation(before):
    """Whether the text before a boundary ends in an abbreviation or initial."""
    tokens = before.rstrip().rsplit(None, 1)
    token = tokens[-1] if tokens else ""
    if not token.endswith("."):
        return False
    word = token[:-1].lstrip("\"'“‘([").lower()
    return word in _ABBREVIATIONS or (
        len(word) == 1 and word.isalpha() and token[-2].isupper()
    )


def _words_allowed(max_seconds, words_per_second):
    from ..task_domains import real_number

    checked = []
    for name, value in (
        ("max_seconds", max_seconds),
        ("words_per_second", words_per_second),
    ):
        # A command-line value for a variable declared null arrives as text
        if isinstance(value, str):
            value = real_number(value, name, "split_takes")
        checked.append(value)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"split_takes: {name} must be above zero, got {value!r}")
    max_seconds, words_per_second = checked
    return max(1, int(max_seconds * words_per_second))


def _sentences(text):
    """The sentences of the text, each with its whitespace collapsed."""
    parts, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        if _is_abbreviation(text[start : match.start()]):
            continue
        # Keep the closing quote with the sentence it closes
        parts.append(text[start : match.start() + len(match.group().rstrip())])
        start = match.end()
    parts.append(text[start:])
    return [" ".join(p.split()) for p in parts if p.strip()]


def _cut_long(sentence, limit):
    """A sentence over the limit as pieces within it: cut after the last comma
    (; : or dash) before the limit, else - no such mark - at the word boundary
    nearest the limit that does not pass it, so a piece is exactly `limit`
    words."""
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


def split_takes(
    text, max_seconds=DEFAULT_MAX_SECONDS, words_per_second=DEFAULT_WORDS_PER_SECOND
):
    """Task command: split a text into takes of at most `max_seconds` of speech.

    Args:
        text: The text to speak
        max_seconds: The longest a take may run. For `voice-clone` a take
            over about 40 s of speech loses words, well inside its ~95 s
            context; 37 at 3.0 words a second is the longest take measured
            word-complete
        words_per_second: The speaking rate used to turn the limit into words

    Takes end at sentence boundaries. A sentence that alone is over the
    limit is cut at its last comma before the limit; with no such comma, at
    the word boundary nearest the limit (the limit's own word count), and the
    run warns. A take PARAGRAPH_CUT_FILL of the way to the limit ends at a
    paragraph break.

    Returns:
        A list of takes, in order. Joined with a space they read as the text,
        less its original whitespace

    Raises:
        ValueError: If the text is empty or a limit is not above zero
    """
    limit = _words_allowed(max_seconds, words_per_second)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("split_takes needs some text to split")

    # (unit, whether it opens a paragraph)
    units = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        opens = True
        for sentence in _sentences(paragraph):
            if len(sentence.split()) > limit:
                logger.warning(
                    f"split_takes: a sentence of {len(sentence.split())} words is "
                    f"over the {limit}-word limit and is cut inside"
                )
                pieces = _cut_long(sentence, limit)
            else:
                pieces = [sentence]
            for piece in pieces:
                units.append((piece, opens))
                opens = False

    takes, current, used = [], [], 0
    for unit, opens_paragraph in units:
        size = len(unit.split())
        full_at_a_break = opens_paragraph and used >= PARAGRAPH_CUT_FILL * limit
        if current and (used + size > limit or full_at_a_break):
            takes.append(" ".join(current))
            current, used = [], 0
        current.append(unit)
        used += size
    if current:
        takes.append(" ".join(current))
    return takes


def split_takes_task(
    text, max_seconds=DEFAULT_MAX_SECONDS, words_per_second=DEFAULT_WORDS_PER_SECOND
):
    """Task command: split a text into takes, answered as a JSON document.

    The takes are a `for_each` list (`templates/narration-from-takes`): a
    `for_each` list cannot come from a step of the same workflow, so
    `templates/narration-long` runs this and hands the takes to a child
    workflow, which expands its own `for_each`.

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
