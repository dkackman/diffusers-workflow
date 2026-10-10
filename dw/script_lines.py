"""The check_script `lines` and `shots` arguments, parsed (#609).

Kept apart from `dw/tasks/script_check.py` so that `dw/task_problems.py`,
which refuses a literal `lines` or `shots` at validation, can call the
same parser the task runs without importing the task - that import closed
a cycle through `tasks.assess`, `tasks.audio_utils` and `tasks.joins` back
into `task_domains` (where the check lived until #790). The task imports these names back, so a caller that
reads them off `script_check` still finds them.
"""

import re

from .shots import duplicate_shot_names

COMMAND = "check_script"


# Markup an expected line may carry. A tag in angle brackets (<d>, </d>,
# <scenetrans>, <cutoff>, and whatever a later delivery tag turns out to be),
# a bracketed span ([English], [unclear]) and an H3 speaker ID ((S1),
# (S1,S2)) - a bare parenthetical is dialogue and stays
_MARKUP = re.compile(
    r"<\s*/?\s*([^<>]*?)\s*>|\[([^\[\]]*)\]|\(\s*(S\d+(?:\s*,\s*S\d+)*)\s*\)"
)
_NON_WORD = re.compile(r"[^\w']+")


def normalize_words(text):
    """`text` as a list of words: lowercase, punctuation stripped, apostrophes
    kept (a curly one read as straight) but not at a word's edges, where they
    are quotation marks. Punctuation splits words, so 'well-known' is the two
    words 'well known' on both sides of the comparison."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    words = []
    for word in _NON_WORD.sub(" ", text).split():
        word = word.strip("'_")
        if word:
            words.append(word)
    return words


def strip_markup(line):
    """A line with its H3 markup removed, and the words that markup held.

    Returns (text, tokens): the dialogue left once every tag, bracketed span
    and speaker ID is cut out, and the normalized words of what was cut -
    'cutoff', 'unclear', 'english', 's1' - which must not be heard.
    """
    tokens = []

    def cut(match):
        inner = next(group for group in match.groups() if group is not None)
        tokens.extend(normalize_words(inner))
        return " "

    text = _MARKUP.sub(cut, line)
    return " ".join(text.split()), tokens


def parse_lines(lines):
    """The expected lines as [{text, tokens, shot}], or ValueError naming
    `lines`.

    Each entry is a string or {text, shot}; markup is stripped
    (strip_markup) and `shot`, the name of the shot the line is spoken in,
    is None when not given. `[]` means no speech is expected.
    """
    if isinstance(lines, str) or not isinstance(lines, (list, tuple)):
        raise ValueError(
            f"{COMMAND} 'lines' must be a list of strings or {{text, shot}}"
            f" objects ([] for no speech), got {type(lines).__name__}"
        )
    parsed = []
    for index, entry in enumerate(lines):
        shot = None
        if isinstance(entry, dict):
            extra = sorted(set(entry) - {"text", "shot"})
            if extra:
                raise ValueError(
                    f"{COMMAND} 'lines'[{index}] has unknown keys {extra};"
                    " a line object takes only 'text' and 'shot'"
                )
            text = entry.get("text")
            shot = entry.get("shot")
            if shot is not None and (not isinstance(shot, str) or not shot.strip()):
                raise ValueError(
                    f"{COMMAND} 'lines'[{index}] 'shot' must be a shot's name"
                    f" (a non-empty string), got {shot!r}"
                )
        else:
            text = entry
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{COMMAND} 'lines'[{index}] needs non-empty text - a string or"
                " {text, shot}"
            )
        stripped, tokens = strip_markup(text)
        parsed.append({"text": stripped, "tokens": tokens, "shot": shot})
    return parsed


def parse_shots(shots):
    """A `shots` argument checked: None or [] (none given), or a list of shot
    records each with a name. ValueError naming `shots` otherwise."""
    if shots is None:
        return None
    if not isinstance(shots, (list, tuple)):
        raise ValueError(
            f"{COMMAND} 'shots' must be a list of shot records, got"
            f" {type(shots).__name__}"
        )
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise ValueError(
                f"{COMMAND} 'shots'[{index}] must be a shot record (an object),"
                f" got {type(shot).__name__}"
            )
        name = shot.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{COMMAND} 'shots'[{index}] needs a 'name'")
    return list(shots) or None


def shot_names_error(parsed_lines, records, source="argument"):
    """The refusal for a line naming a shot `records` does not hold once -
    unknown, or a name `shots.duplicate_shot_names` finds repeated - or
    None. The message lists the shots it does hold."""
    names = [record.get("name") for record in records]
    known = ", ".join(repr(name) for name in names)
    duplicated = {
        name for dups in (duplicate_shot_names(records) or {}).values() for name in dups
    }
    for index, line in enumerate(parsed_lines):
        shot = line["shot"]
        if shot is None:
            continue
        if shot not in names:
            return (
                f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which the take"
                f" does not have - its shots ({source}) are {known}"
            )
        if shot in duplicated:
            return (
                f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which the"
                f" take's shot map ({source}) holds more than once - a line can"
                f" only name a shot the map holds once: {known}"
            )
    return None
