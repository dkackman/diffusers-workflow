"""The check_script `lines` and `shots` arguments, parsed (#609).

Kept apart from `dw/tasks/script_check.py` so that `dw/task_problems.py`,
which refuses a literal `lines` or `shots` at validation, can call the
same parser the task runs without importing the task - that import closed
a cycle through `tasks.assess`, `tasks.audio_utils` and `tasks.joins` back
into `task_domains` (where the check lived until #790). The task imports these names back, so a caller that
reads them off `script_check` still finds them.

It also holds what the tasks that read a transcript's words share without
importing each other (#822): the repetition-loop detector check_script
discards with and transcribe_audio marks `suspect: "repetition"` with, and
the run grouping plan_cuts warns about suspect chunks by.
"""

import re

from .references import member_keys
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


# A run of heard words that repeats with a period of at most
# REPEAT_MAX_PERIOD words, REPEAT_RUN_MIN or more times over, is a decoding
# loop - Whisper's hallucination over music or room tone ('Pre-pre-pre-...',
# 'thank you thank you ...'), loud enough to pass the energy guard. Speech
# repeats a word a few times; it does not loop. check_script discards such
# words and transcribe_audio marks them `suspect: "repetition"` (#822)
REPEAT_RUN_MIN = 6
REPEAT_MAX_PERIOD = 4


def repetition_runs(keys):
    """[(first, end, period)] for each stretch of `keys` (heard words) that
    repeats with a period of 1 to REPEAT_MAX_PERIOD and spans at least
    REPEAT_RUN_MIN periods; `keys[first:first + period]` is its first
    occurrence."""
    runs = []
    for period in range(1, REPEAT_MAX_PERIOD + 1):
        k = period
        while k < len(keys):
            if keys[k] != keys[k - period]:
                k += 1
                continue
            first = k - period
            while k < len(keys) and keys[k] == keys[k - period]:
                k += 1
            if k - first >= REPEAT_RUN_MIN * period:
                runs.append((first, k, period))
    return runs


def repetition_loops(keys):
    """The indices of `keys` (heard words) that sit in a repetition loop."""
    looped = set()
    for first, end, _ in repetition_runs(keys):
        looped.update(range(first, end))
    return looped


def looped_texts(texts):
    """The indices of `texts` (transcript chunks' text, in order) holding a word
    of a repetition loop. Over words, not chunks: one chunk can hold a whole
    loop ('Pre-pre-pre'), and a chunk with no word is never in one."""
    return looped_texts_firsts(texts)[0]


def looped_texts_firsts(texts):
    """(looped, firsts): `looped` as looped_texts; `firsts` the chunk indices
    that are the first occurrence of a loop, every word of the chunk in it. A
    period-1 loop of 'pre' has one first word; where runs of several periods
    overlap, the shortest period's first occurrence is the one."""
    sequence, owner = [], []
    for index, text in enumerate(texts):
        words = normalize_words(text)
        owner.extend([index] * len(words))
        sequence.extend(words)
    runs = sorted(repetition_runs(sequence), key=lambda run: run[2])
    looped, first_words = set(), set()
    for first, end, period in runs:
        if not set(range(first, end)) <= looped:
            first_words.update(range(first, first + period))
        looped.update(range(first, end))
    chunks = {owner[k] for k in looped}
    firsts = {
        index
        for index in chunks
        if all(k in first_words for k in range(len(owner)) if owner[k] == index)
    }
    return chunks, firsts


# transcribe_audio's `suspect` reasons whose chunk plan_cuts leaves out of the
# sung lines: no word, a repetition loop, no voice under it. A "long" word is
# real lyric, only mistimed, so it stays
SUSPECT_LEFT_OUT = ("no_words", "repetition", "silent")


def suspect_runs(chunks):
    """[(reason, [chunk, ...])]: each run of consecutive chunks carrying the
    same `suspect` reason, so a long hallucination is one run, not hundreds."""
    runs, previous = [], None
    for chunk in chunks:
        reason = chunk.get("suspect")
        if reason and reason == previous:
            runs[-1][1].append(chunk)
        elif reason:
            runs.append((reason, [chunk]))
        previous = reason
    return runs


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

    Each entry is a string or {text, shot, start_frame}; markup is stripped
    (strip_markup) and `shot`, the name of the shot the line is spoken in,
    is None when not given. `start_frame` picks one occurrence of a shot
    name the take's map holds more than once (#919); None when not given.
    `[]` means no speech is expected.
    """
    if isinstance(lines, str) or not isinstance(lines, (list, tuple)):
        raise ValueError(
            f"{COMMAND} 'lines' must be a list of strings or {{text, shot}}"
            f" objects ([] for no speech), got {type(lines).__name__}"
        )
    parsed = []
    for index, entry in enumerate(lines):
        shot = start_frame = None
        if isinstance(entry, dict):
            extra = sorted(set(entry) - {"text", "shot", "start_frame"})
            if extra:
                raise ValueError(
                    f"{COMMAND} 'lines'[{index}] has unknown keys {extra};"
                    " a line object takes only 'text', 'shot' and 'start_frame'"
                )
            text = entry.get("text")
            shot = entry.get("shot")
            start_frame = entry.get("start_frame")
            if start_frame is not None and (
                isinstance(start_frame, bool) or not isinstance(start_frame, int)
            ):
                raise ValueError(
                    f"{COMMAND} 'lines'[{index}] 'start_frame' must be an"
                    f" integer frame number, got {start_frame!r}"
                )
            if start_frame is not None and shot is None:
                raise ValueError(
                    f"{COMMAND} 'lines'[{index}] gives 'start_frame' without"
                    " the 'shot' it picks an occurrence of"
                )
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
        parsed.append(
            {
                "text": stripped,
                "tokens": tokens,
                "shot": shot,
                "start_frame": start_frame,
            }
        )
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


def matching_shot_names(shot, names):
    """The names in `names` a line's `shot` addresses: `shot` itself when
    the map holds it, else every name carrying it as a `@<member>` key
    (`cold_open` for `shot@cold_open` or `...-shot@cold_open.0-0.0.mp4`)."""
    if shot in names:
        return [shot]
    return list(dict.fromkeys(name for name in names if shot in member_keys(name)))


def _repeated(records):
    """The (file, name) pairs `records` holds more than once."""
    return {
        (file, name)
        for file, dups in (duplicate_shot_names(records) or {}).items()
        for name in dups
    }


def occurrence_name(record):
    """A repeated shot's name made unique by where it sits: `shot@bus#1248`."""
    return f"{record['name']}#{record.get('start_frame')}"


def disambiguated(records):
    """`records` with each name the map holds more than once (within a file)
    renamed by `occurrence_name`, so a span or finding can tell the
    occurrences apart (#919). The same list when nothing repeats."""
    repeated = _repeated(records)
    if not repeated:
        return records
    return [
        dict(record, name=occurrence_name(record))
        if (record.get("file"), record.get("name")) in repeated
        else record
        for record in records
    ]


def shot_names_error(parsed_lines, records, source="argument"):
    """The refusal for a line naming a shot `records` does not hold - unknown,
    a short `@<member>` name more than one shot carries, or a name the map
    holds more than once and the line gives no `start_frame` for - or None.
    The message lists the shots it does hold."""
    names = [record.get("name") for record in records]
    repeated = {name for _, name in _repeated(records)}
    known = ", ".join(repr(name) for name in dict.fromkeys(names))
    nameable = ", ".join(
        repr(name) for name in dict.fromkeys(names) if name not in repeated
    )
    for index, line in enumerate(parsed_lines):
        shot = line["shot"]
        if shot is None:
            continue
        matches = matching_shot_names(shot, names)
        if not matches:
            return (
                f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which the take"
                f" does not have - its shots ({source}) are {known}"
            )
        if len(matches) > 1:
            return (
                f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which more"
                f" than one of the take's shots ({source}) carries - name one"
                f" in full: {', '.join(repr(name) for name in matches)}"
            )
        if matches[0] in repeated:
            frames = [
                record.get("start_frame")
                for record in records
                if record.get("name") == matches[0]
            ]
            start_frame = line.get("start_frame")
            if start_frame is None:
                return (
                    f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which the"
                    f" take's shot map ({source}) holds more than once - add"
                    f" 'start_frame' to say which occurrence: one of {frames}."
                    f" Shots named without it: {nameable or 'none'}"
                )
            if start_frame not in frames:
                return (
                    f"{COMMAND} 'lines'[{index}] 'start_frame' {start_frame}"
                    f" matches no occurrence of shot {shot!r} ({source}) - its"
                    f" start_frames are {frames}"
                )
        elif line.get("start_frame") is not None:
            frame = next(
                record.get("start_frame")
                for record in records
                if record.get("name") == matches[0]
            )
            if frame != line["start_frame"]:
                return (
                    f"{COMMAND} 'lines'[{index}] 'start_frame'"
                    f" {line['start_frame']} does not match shot {shot!r}"
                    f" ({source}), which starts at frame {frame}"
                )
    return None


def with_full_shot_names(parsed_lines, records):
    """`parsed_lines` with each line's short shot name replaced by the name
    `disambiguated(records)` holds: the full name, or for a repeated shot
    `occurrence_name`. Call after `shot_names_error` found nothing."""
    names = [record.get("name") for record in records]
    repeated = {name for _, name in _repeated(records)}
    resolved = []
    for line in parsed_lines:
        shot = line["shot"]
        matches = matching_shot_names(shot, names) if shot is not None else []
        if not matches:
            resolved.append(line)
            continue
        full = matches[0]
        if full in repeated:
            full = f"{full}#{line.get('start_frame')}"
        resolved.append(dict(line, shot=full))
    return resolved
