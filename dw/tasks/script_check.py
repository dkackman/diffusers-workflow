"""Whether a take speaks its script (#609).

Confirming that a dialogue take says its lines used to be a procedure run by
hand: transcribe the take, read the transcript, compare it to the script by
eye. The eye missed what Whisper's early stop hid (#559) and had no way to
see a markup word spoken aloud or a last word clipped by the end of the
file. This task does the comparison and reports where to look:

1. the take is transcribed with `transcribe_audio(timestamps="word")` - the
   same code, the same model cache, no second ASR path;
2. every heard word is measured, and one whose span sits at or below the
   dead-air floor the assessment probes already use (`DEAD_AIR_FLOOR_DBFS`,
   `shot_dead_air`'s level, #465) is discarded as unheard - Whisper invents
   words over silence, and the HF pipeline returns no `no_speech_prob` to
   catch them by. Discarded words are reported under `discarded`, never
   raised as a finding;
3. both sides are normalized - lowercase, punctuation stripped, apostrophes
   kept - and H3 markup (`<d>[English] ...</d>`, `<scenetrans>`, `<cutoff>`,
   `[unclear]`, `(S1)`, any `<tag>` or `[tag]`) is stripped from each
   expected line. The stripped tokens' words become the words that must not
   be heard: `tag_spoken`;
4. the heard words are aligned to the expected words in order with
   `difflib.SequenceMatcher`, and each line scores the similarity of its own
   expected and heard words.

It decides nothing. Like the probes, its findings are places to look and are
built with `assessment_rules.finding()`; unlike them it is not
`assessment=True`, and its rules are not in `RULES`: every rule there has to
fire on a synthetic file that holds no script (`tests/test_assessment_rules.py`).
Every threshold is a constant below, named in docs/TASKS.md.
"""

import difflib
import logging
import re

from .. import dsp
from ..assessment_rules import DEAD_AIR_FLOOR_DBFS, finding
from ..for_each import MEMBER_SEPARATOR, render_path
from ..references import author_index
from ..task_domains import check_arguments
from .audio_utils import waveform_and_rate

logger = logging.getLogger("dw")

COMMAND = "check_script"

# Default least similarity between a line's expected and heard words before
# it is a `line_mismatch`. The template's own default is set from stage A's
# two-model measurement (#609 Q3); this one is the task's
DEFAULT_SIMILARITY = 0.85
# The ASR model the task defaults to - transcribe_audio's own default
DEFAULT_MODEL = "openai/whisper-base"
# A heard word is guarded (discarded as unheard) when its loudest
# GUARD_WINDOW_SECONDS window sits at or below GUARD_FLOOR_DBFS. The floor is
# the assessment probes' dead-air floor, read, not copied; the window is the
# one `shot_dead_air` measures in. The loudest window rather than the whole
# span, because a word timestamp is approximate (+-0.1-0.3 s) and a span
# that overhangs the pause after a word must not average a real word away
GUARD_FLOOR_DBFS = DEAD_AIR_FLOOR_DBFS
GUARD_WINDOW_SECONDS = 0.05
# A line is clipped at the end when its last heard word ends inside the
# file's final CLIP_TAIL_SECONDS and that tail measures above
# GUARD_FLOOR_DBFS - still voiced when the file stops, not a word that
# finished into a quiet tail
CLIP_TAIL_SECONDS = 0.25

THRESHOLDS = {
    "similarity": DEFAULT_SIMILARITY,
    "guard_floor_dbfs": GUARD_FLOOR_DBFS,
    "guard_window_seconds": GUARD_WINDOW_SECONDS,
    "clip_tail_seconds": CLIP_TAIL_SECONDS,
}

LINE_RULES = ("line_mismatch", "tag_spoken", "line_clipped_at_end")
SILENT_RULES = ("speech_where_silent",)


def _rule(name, threshold, says):
    return {"name": name, "severity": "warn", "threshold": threshold, "says": says}


def mismatch_rule(similarity):
    return _rule(
        "line_mismatch",
        similarity,
        "the take's words for this line are this similar to the script - listen"
        " before re-rolling: Whisper mishears names and numbers ('2' for 'two')"
        " on a correct take",
    )


TAG_SPOKEN = _rule(
    "tag_spoken",
    None,
    "a markup word from this line was heard spoken aloud - the model read the"
    " tag as dialogue",
)
SPEECH_WHERE_SILENT = _rule(
    "speech_where_silent",
    0,
    "no line was expected, and this many words were heard above the dead-air floor",
)
CLIPPED_AT_END = _rule(
    "line_clipped_at_end",
    GUARD_FLOOR_DBFS,
    "the line's last word ends inside the file's final"
    f" {CLIP_TAIL_SECONDS} s and the tail is still voiced at this level - the"
    " file stops mid-word",
)

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
    """The expected lines as [{text, tokens}], or ValueError naming `lines`.

    Each entry is a string or {text}; markup is stripped (strip_markup).
    `[]` means no speech is expected.
    """
    if isinstance(lines, str) or not isinstance(lines, (list, tuple)):
        raise ValueError(
            f"{COMMAND} 'lines' must be a list of strings or {{text}} objects"
            f" ([] for no speech), got {type(lines).__name__}"
        )
    parsed = []
    for index, entry in enumerate(lines):
        if isinstance(entry, dict):
            extra = sorted(set(entry) - {"text"})
            if extra:
                raise ValueError(
                    f"{COMMAND} 'lines'[{index}] has unknown keys {extra};"
                    " a line object takes only 'text'"
                )
            text = entry.get("text")
        else:
            text = entry
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{COMMAND} 'lines'[{index}] needs non-empty text - a string or"
                " {text}"
            )
        stripped, tokens = strip_markup(text)
        parsed.append({"text": stripped, "tokens": tokens})
    return parsed


def lines_errors(workflow_definition, source_indices=None):
    """Every check_script step whose literal `lines` parse_lines would refuse,
    as [{path, message}]. A `lines` still spelled as a reference string is
    left to the run, which refuses a string that resolves to no list."""
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []
    errors = []
    for index, step in enumerate(steps):
        task = step.get("task") if isinstance(step, dict) else None
        if not isinstance(task, dict) or task.get("command") != COMMAND:
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict) or "lines" not in arguments:
            continue
        lines = arguments["lines"]
        if isinstance(lines, str) and ":" in lines:
            continue
        try:
            parse_lines(lines)
        except ValueError as error:
            source = author_index(source_indices, index)
            name = step.get("name")
            where = (
                f" in member '{name}'"
                if isinstance(name, str) and MEMBER_SEPARATOR in name
                else ""
            )
            path = ("steps", source, "task", "arguments", "lines")
            errors.append({"path": render_path(path), "message": f"{error}{where}"})
    return errors


def word_level(mono, sample_rate, start, end):
    """The loudest GUARD_WINDOW_SECONDS window's rms over [start, end], in
    dBFS, or None for silence. A span shorter than one window (a zero-length
    word chunk) is measured as one window centred on it."""
    window = max(1, int(round(GUARD_WINDOW_SECONDS * sample_rate)))
    first = int(round(start * sample_rate))
    last = int(round(end * sample_rate))
    if last - first < window:
        middle = (first + last) // 2
        first, last = middle - window // 2, middle - window // 2 + window
    first, last = max(0, first), min(len(mono), last)
    loudest = None
    for offset in range(first, last, window):
        level = dsp.rms(mono[offset : min(offset + window, last)])
        if level is not None and (loudest is None or level > loudest):
            loudest = level
    return dsp.dbfs(loudest)


def _round(value, places=3):
    return None if value is None else round(float(value), places)


def guard_words(chunks, mono, sample_rate):
    """Split transcribe_audio's word chunks into (heard, discarded).

    `heard` is [{word, text, start, end}], one per normalized word - a chunk
    that normalizes to nothing ('♪', '...') is neither; `discarded` is
    [{text, start, end, level_dbfs}] for a chunk at or below the floor.
    """
    heard, discarded = [], []
    for chunk in chunks:
        words = normalize_words(chunk.get("text", ""))
        if not words:
            continue
        start, end = chunk["start"], chunk["end"]
        level = word_level(mono, sample_rate, start, end)
        if level is None or level <= GUARD_FLOOR_DBFS:
            discarded.append(
                {
                    "text": chunk.get("text", "").strip(),
                    "start": _round(start),
                    "end": _round(end),
                    "level_dbfs": _round(level, 2),
                }
            )
            continue
        for word in words:
            heard.append(
                {
                    "word": word,
                    "text": chunk.get("text", "").strip(),
                    "start": start,
                    "end": end,
                }
            )
    return heard, discarded


def align(expected_lines, heard):
    """Assign each heard word to the expected line it was spoken for.

    `expected_lines` is [[word, ...], ...]; `heard` is the heard words. The
    words of every line are concatenated and aligned to the heard words in
    order with SequenceMatcher. A matched or replaced heard word belongs to
    the line of the expected word it lines up with; a heard word inserted
    inside a line belongs to that line. One inserted between lines, or
    before the first or after the last, belongs to none - an added line
    lowers no line's similarity - and comes back with the lines either side
    of it, which is where a spoken tag at a line's edge is looked for.

    Returns (assigned, unassigned): assigned[i] is the list of heard indices
    for line i, in order; unassigned is [(heard index, neighbouring line
    indices)].
    """
    owner = [i for i, words in enumerate(expected_lines) for _ in words]
    flat = [word for words in expected_lines for word in words]
    assigned = [[] for _ in expected_lines]
    unassigned = []
    matcher = difflib.SequenceMatcher(
        None, flat, [word["word"] for word in heard], autojunk=False
    )
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "delete":
            continue
        if op == "insert":
            before = owner[i1 - 1] if i1 > 0 else None
            after = owner[i1] if i1 < len(owner) else None
            if before is not None and before == after:
                assigned[before].extend(range(j1, j2))
            else:
                neighbours = tuple(line for line in (before, after) if line is not None)
                unassigned.extend((j, neighbours) for j in range(j1, j2))
            continue
        # equal or replace: spread the heard words over the expected words'
        # lines in proportion (equal is one-to-one)
        span = i2 - i1
        for k, j in enumerate(range(j1, j2)):
            i = i1 + min(span - 1, (k * span) // (j2 - j1))
            assigned[owner[i]].append(j)
    return assigned, unassigned


def line_similarity(expected, heard):
    """How alike two word lists are, 0..1. Two empty lists are identical."""
    if not expected and not heard:
        return 1.0
    return difflib.SequenceMatcher(None, expected, heard, autojunk=False).ratio()


def tail_level(mono, sample_rate):
    """The rms of the file's final CLIP_TAIL_SECONDS, in dBFS (None: silent)."""
    count = max(1, int(round(CLIP_TAIL_SECONDS * sample_rate)))
    return dsp.dbfs(dsp.rms(mono[-count:]))


def _at(line, heard_word=None, seconds=None):
    at = {"line": line}
    if heard_word is not None:
        at["seconds"] = _round(heard_word["start"])
        at["word"] = heard_word["text"]
    else:
        at["seconds"] = _round(seconds)
    return at


def check(parsed_lines, chunks, mono, sample_rate, similarity_threshold):
    """The comparison, on a transcript already taken - everything the task
    does after the ASR model, so it is testable on synthetic word lists and
    waveforms. `chunks` is transcribe_audio's word chunks; `mono` a 1-D
    float waveform."""
    heard, discarded = guard_words(chunks, mono, sample_rate)
    duration = len(mono) / sample_rate if sample_rate else 0.0
    findings = []

    if not parsed_lines:
        if heard:
            findings.append(
                finding(SPEECH_WHERE_SILENT, len(heard), _at(None, heard[0]))
            )
        return {
            "findings": findings,
            "lines": [],
            "discarded": discarded,
            "unmatched": [_heard_entry(word) for word in heard],
            "rules_applied": list(SILENT_RULES),
            "rules_skipped": [
                {"rule": rule, "reason": "lines is [] - no speech was expected"}
                for rule in LINE_RULES
            ],
        }

    expected = [normalize_words(line["text"]) for line in parsed_lines]
    assigned, unassigned = align(expected, heard)
    mismatch = mismatch_rule(similarity_threshold)
    tail_dbfs = tail_level(mono, sample_rate)

    lines = []
    previous_end = 0.0
    for index, (line, words, indices) in enumerate(
        zip(parsed_lines, expected, assigned)
    ):
        own = [heard[j] for j in indices]
        score = line_similarity(words, [word["word"] for word in own])
        start = own[0]["start"] if own else None
        end = own[-1]["end"] if own else None
        lines.append(
            {
                "expected": line["text"],
                "heard": _joined(own),
                "similarity": round(score, 4),
                "start": _round(start),
                "end": _round(end),
                "shot": None,
            }
        )
        if score < similarity_threshold:
            at = _at(index, own[0]) if own else _at(index, seconds=previous_end)
            findings.append(finding(mismatch, round(score, 4), at))

        tokens = set(line["tokens"]) - set(words)
        for word in own:
            if word["word"] in tokens:
                findings.append(finding(TAG_SPOKEN, word["word"], _at(index, word)))

        if (
            own
            and end >= duration - CLIP_TAIL_SECONDS
            and tail_dbfs is not None
            and tail_dbfs > GUARD_FLOOR_DBFS
        ):
            at = {"line": index, "seconds": _round(end), "word": own[-1]["text"]}
            findings.append(finding(CLIPPED_AT_END, _round(tail_dbfs, 2), at))
        if own:
            previous_end = end

    for j, neighbours in unassigned:
        word = heard[j]
        for index in neighbours:
            if word["word"] in set(parsed_lines[index]["tokens"]) - set(
                expected[index]
            ):
                findings.append(finding(TAG_SPOKEN, word["word"], _at(index, word)))
                break

    findings.sort(key=lambda found: found["at"].get("seconds") or 0.0)
    return {
        "findings": findings,
        "lines": lines,
        "discarded": discarded,
        "unmatched": [_heard_entry(heard[j]) for j, _ in unassigned],
        "rules_applied": list(LINE_RULES),
        "rules_skipped": [
            {"rule": rule, "reason": "lines were given - speech is expected"}
            for rule in SILENT_RULES
        ],
    }


def _heard_entry(word):
    return {
        "text": word["text"],
        "start": _round(word["start"]),
        "end": _round(word["end"]),
    }


def _joined(words):
    """The heard words as the transcript spelled them, one chunk once."""
    texts = []
    last = None
    for word in words:
        key = (word["start"], word["end"], word["text"])
        if key != last:
            texts.append(word["text"])
            last = key
    return " ".join(texts)


def check_script(
    audio,
    lines,
    similarity=DEFAULT_SIMILARITY,
    model_name=DEFAULT_MODEL,
    sample_rate=None,
    device="cpu",
):
    """Check that a take speaks its script, line by line.

    Transcribes the take with word timestamps (transcribe_audio), discards
    heard words whose span sits at or below the dead-air floor (Whisper
    invents words over silence), strips H3 markup from each expected line,
    and aligns the heard words to the expected ones in order. Decides
    nothing: findings are places to look.

    Args:
        audio: The take - a path, 'asset:'/'output:' reference, or an
            earlier step's audio or video (its soundtrack is taken).
        lines: The expected lines in order - a list of strings or {text}.
            H3 markup (<d>[English] ...</d>, <scenetrans>, <cutoff>,
            [unclear], (S1)) is stripped, and a markup word heard spoken is
            'tag_spoken'. [] means no speech is expected.
        similarity: Least similarity (0..1) of a line's heard words to its
            expected words before it is a 'line_mismatch'. Default 0.85.
        model_name: HuggingFace ID of the Whisper-class ASR model (default:
            openai/whisper-base).
        sample_rate: Sample rate of a waveform passed directly.
        device: Where to run the ASR model.

    Returns:
        A JSON document: 'findings' (line_mismatch, tag_spoken,
        line_clipped_at_end, speech_where_silent), 'lines' (expected, heard,
        similarity, start, end, shot), 'discarded' (guarded words with their
        level), 'unmatched' (heard words aligned to no line), 'transcript',
        'model_name', 'rules_applied', 'rules_skipped' and the 'thresholds'
        used.
    """
    check_arguments(COMMAND, similarity=similarity)
    parsed_lines = parse_lines(lines)
    waveform, rate = waveform_and_rate(audio, sample_rate, COMMAND)

    from .audio_transcription import transcribe_audio

    transcript = transcribe_audio(
        waveform,
        device=device,
        sample_rate=rate,
        model_name=model_name,
        timestamps="word",
    )
    mono = waveform[0] if waveform.shape[0] == 1 else waveform.mean(axis=0)
    answer = check(parsed_lines, transcript["chunks"], mono, rate, float(similarity))
    answer["transcript"] = transcript["text"]
    answer["model_name"] = model_name
    answer["thresholds"] = dict(THRESHOLDS, similarity=float(similarity))
    return answer
