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
   raised as a finding. So is every word of a repetition loop - the same
   words over and over, which Whisper invents over music or room tone loud
   enough to pass the floor;
3. both sides are normalized - lowercase, punctuation stripped, apostrophes
   kept - and H3 markup (`<d>[English] ...</d>`, `<scenetrans>`, `<cutoff>`,
   `[unclear]`, `(S1)`, any `<tag>` or `[tag]`) is stripped from each
   expected line. The stripped tokens' words become the words that must not
   be heard: `tag_spoken`;
4. the heard words are aligned to the expected words in order with
   `difflib.SequenceMatcher`, and each line scores the similarity of its own
   expected and heard words;
5. when the take's shots are known - the `shots` argument, then the
   artifact's own, then the run manifest beside the file, exactly as the
   probes resolve them (`assess.resolve_shots`) - each line is placed in a
   shot (the one it names, else the one its heard words overlap most), a
   shot no line names is meant to be silent (`speech_in_silent_shot`), and a
   line's last word is checked against the end of its shot as well as the
   end of the file (`line_clipped_at_end`). With no shots known those are
   listed under `rules_skipped` with the reason, never reported clean.

It decides nothing. Like the probes, its findings are places to look and are
built with `assessment_rules.finding()`; unlike them it is not
`assessment=True`, and its rules are not in `RULES`: every rule there has to
fire on a synthetic file that holds no script (`tests/test_assessment_rules.py`).
Every threshold is a constant below, named in docs/TASKS.md.
"""

import difflib
import logging
import re
from types import SimpleNamespace

from .. import dsp
from ..assessment_rules import DEAD_AIR_FLOOR_DBFS, finding
from ..task_domains import check_arguments
from .assess import DEAD_AIR_WINDOW, resolve_shots
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
# the assessment probes' dead-air floor and the window is the one
# `shot_dead_air` measures in (`DEAD_AIR_WINDOW`), both read, not copied. The loudest window rather than the whole
# span, because a word timestamp is approximate (+-0.1-0.3 s) and a span
# that overhangs the pause after a word must not average a real word away
GUARD_FLOOR_DBFS = DEAD_AIR_FLOOR_DBFS
GUARD_WINDOW_SECONDS = DEAD_AIR_WINDOW
# A line is clipped at the end when its last heard word ends inside the
# file's final CLIP_TAIL_SECONDS and that tail measures above
# GUARD_FLOOR_DBFS - still voiced when the file stops, not a word that
# finished into a quiet tail
CLIP_TAIL_SECONDS = 0.25
# A run of heard words that repeats with a period of at most
# REPEAT_MAX_PERIOD words, REPEAT_RUN_MIN or more times over, is discarded
# as a decoding loop - Whisper's hallucination over music or room tone
# ('Pre-pre-pre-...', 'thank you thank you ...'), loud enough to pass the
# energy guard. Speech repeats a word a few times; it does not loop
REPEAT_RUN_MIN = 6
REPEAT_MAX_PERIOD = 4

THRESHOLDS = {
    "similarity": DEFAULT_SIMILARITY,
    "guard_floor_dbfs": GUARD_FLOOR_DBFS,
    "guard_window_seconds": GUARD_WINDOW_SECONDS,
    "clip_tail_seconds": CLIP_TAIL_SECONDS,
    "repeat_run_min": REPEAT_RUN_MIN,
    "repeat_max_period": REPEAT_MAX_PERIOD,
}

LINE_RULES = ("line_mismatch", "tag_spoken", "line_clipped_at_end")
SILENT_RULES = ("speech_where_silent",)
SHOT_RULES = ("speech_in_silent_shot",)


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
    "no line was expected, and this many words were heard above the dead-air floor"
    " and outside a repetition loop",
)
CLIPPED_AT_END = _rule(
    "line_clipped_at_end",
    GUARD_FLOOR_DBFS,
    "the line's last word ends inside the final"
    f" {CLIP_TAIL_SECONDS} s of its shot (or of the file) and that tail is"
    " still voiced at this level - the shot stops mid-word",
)
SPEECH_IN_SILENT_SHOT = _rule(
    "speech_in_silent_shot",
    0,
    "no line names this shot, and this many words were heard in it above the"
    " dead-air floor and outside a repetition loop",
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
    unknown, or a name the shot map repeats - or None. The message lists
    the shots it does hold."""
    names = [record.get("name") for record in records]
    known = ", ".join(repr(name) for name in names)
    for index, line in enumerate(parsed_lines):
        shot = line["shot"]
        if shot is None:
            continue
        count = names.count(shot)
        if count == 0:
            return (
                f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which the take"
                f" does not have - its shots ({source}) are {known}"
            )
        if count > 1:
            return (
                f"{COMMAND} 'lines'[{index}] names shot {shot!r}, which the"
                f" take's shot map ({source}) holds {count} times - a line can"
                f" only name a shot the map holds once: {known}"
            )
    return None


def shot_spans(records, sample_rate, fps, duration):
    """Each shot record as {name, start, end} in seconds on the take, or None
    when one cannot be placed. A recorded sample span is used as is; else
    the frame span by `fps`. Clipped to the take's `duration`."""
    spans = []
    for record in records:
        start, count = record.get("start_sample"), record.get("num_samples")
        if start is not None and count is not None and sample_rate:
            start, end = start / sample_rate, (start + count) / sample_rate
        elif fps and record.get("start_frame") is not None:
            start = record["start_frame"] / fps
            end = start + (record.get("num_frames") or 0) / fps
        else:
            return None
        start = min(max(0.0, float(start)), duration)
        spans.append(
            {
                "name": record["name"],
                "start": start,
                "end": max(start, min(float(end), duration)),
            }
        )
    return spans


def overlapping_shot(spans, start, end):
    """The name of the shot [start, end] overlaps most, or None. A span of no
    length (a one-word line's zero-length chunk) is placed by where it sits."""
    best, best_overlap = None, 0.0
    for span in spans:
        overlap = min(span["end"], end) - max(span["start"], start)
        if overlap > best_overlap or (
            best is None and overlap == 0 and span["start"] <= start < span["end"]
        ):
            best, best_overlap = span["name"], overlap
    return best


def shot_at(spans, seconds):
    """The name of the shot holding `seconds`, or None."""
    for span in spans:
        if span["start"] <= seconds < span["end"]:
            return span["name"]
    return None


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


def repetition_loops(keys):
    """The indices of `keys` (heard words) that sit in a repetition loop: a stretch that
    repeats with a period of 1 to REPEAT_MAX_PERIOD and spans at least
    REPEAT_RUN_MIN periods."""
    looped = set()
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
                looped.update(range(first, k))
    return looped


def guard_words(chunks, mono, sample_rate):
    """Split transcribe_audio's word chunks into (heard, discarded).

    `heard` is [{word, text, start, end}], one per normalized word - a chunk
    that normalizes to nothing ('♪', '...') is neither; `discarded` is
    [{text, start, end, level_dbfs, reason}] for a chunk in a repetition loop
    (`reason` "repetition") or at or below the floor ("below_floor").
    """
    spoken, sequence, owner = [], [], []
    for chunk in chunks:
        words = normalize_words(chunk.get("text", ""))
        if words:
            owner.extend([len(spoken)] * len(words))
            sequence.extend(words)
            spoken.append((chunk, words))
    # Over words, not chunks: one chunk can hold a whole loop ('Pre-pre-pre')
    looped = {owner[k] for k in repetition_loops(sequence)}

    heard, discarded = [], []
    for index, (chunk, words) in enumerate(spoken):
        start, end = chunk["start"], chunk["end"]
        level = word_level(mono, sample_rate, start, end)
        if index in looped:
            reason = "repetition"
        elif level is None or level <= GUARD_FLOOR_DBFS:
            reason = "below_floor"
        else:
            reason = None
        if reason:
            discarded.append(
                {
                    "text": chunk.get("text", "").strip(),
                    "start": _round(start),
                    "end": _round(end),
                    "level_dbfs": _round(level, 2),
                    "reason": reason,
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


def tail_level(mono, sample_rate, end=None):
    """The rms of the CLIP_TAIL_SECONDS before `end` seconds - the file's
    final ones when None - in dBFS (None: silent)."""
    count = max(1, int(round(CLIP_TAIL_SECONDS * sample_rate)))
    last = len(mono) if end is None else min(len(mono), int(round(end * sample_rate)))
    return dsp.dbfs(dsp.rms(mono[max(0, last - count) : last]))


def _at(line, heard_word=None, seconds=None, shot=None):
    at = {"line": line}
    if heard_word is not None:
        at["seconds"] = _round(heard_word["start"])
        at["word"] = heard_word["text"]
    else:
        at["seconds"] = _round(seconds)
    if shot is not None:
        at["shot"] = shot
    return at


def _clipped_tail(mono, sample_rate, end, tails):
    """The first of `tails` - (shot name or None, the second it ends at) -
    that a line ending at `end` is clipped by: the word ends inside its
    final CLIP_TAIL_SECONDS (within that much past it too, a word timestamp
    being approximate) and the tail is voiced above GUARD_FLOOR_DBFS.
    Returns (shot, level) or None."""
    for shot, tail_end in tails:
        if not tail_end - CLIP_TAIL_SECONDS <= end <= tail_end + CLIP_TAIL_SECONDS:
            continue
        level = tail_level(mono, sample_rate, tail_end)
        if level is not None and level > GUARD_FLOOR_DBFS:
            return shot, level
    return None


def _no_shot_reason(shots_source, spans, parsed_lines):
    """Why speech_in_silent_shot cannot run on this call, or None when it
    can."""
    named = list(dict.fromkeys(line["shot"] for line in parsed_lines if line["shot"]))
    if shots_source == "none":
        reason = (
            "no shots are known - none given as 'shots', none carried by the"
            " take, none recorded in a manifest beside the file - so no shot"
            " can be checked for speech, and line_clipped_at_end looks at the"
            " file's end only"
        )
        if named:
            reason += (
                f"; the lines name shots {', '.join(map(repr, named))}, which"
                " nothing places in the take"
            )
        return reason
    if spans is None:
        return (
            f"the shots ({shots_source}) carry neither a sample span nor a"
            " frame span the take's frame rate can place, so no shot can be"
            " checked for speech"
        )
    if not named:
        return (
            "no line names its shot, so no shot is known to be meant silent -"
            " give each line {text, shot}"
        )
    return None


def _shots_entry(spans):
    if spans is None:
        return None
    return [
        {
            "name": span["name"],
            "start": _round(span["start"]),
            "end": _round(span["end"]),
        }
        for span in spans
    ]


def check(
    parsed_lines,
    chunks,
    mono,
    sample_rate,
    similarity_threshold,
    spans=None,
    shots_source="none",
):
    """The comparison, on a transcript already taken - everything the task
    does after the ASR model, so it is testable on synthetic word lists and
    waveforms. `chunks` is transcribe_audio's word chunks; `mono` a 1-D
    float waveform; `spans` the take's shots as shot_spans() places them, or
    None when none are known, and `shots_source` where they came from
    (resolve_shots: argument, artifact, manifest or none)."""
    heard, discarded = guard_words(chunks, mono, sample_rate)
    duration = len(mono) / sample_rate if sample_rate else 0.0
    findings = []

    def heard_shot(word):
        if spans is None:
            return None
        return shot_at(spans, (word["start"] + word["end"]) / 2)

    if not parsed_lines:
        if heard:
            findings.append(
                finding(
                    SPEECH_WHERE_SILENT,
                    len(heard),
                    _at(None, heard[0], shot=heard_shot(heard[0])),
                )
            )
        return {
            "findings": findings,
            "lines": [],
            "discarded": discarded,
            "unmatched": [_heard_entry(word) for word in heard],
            "shots": _shots_entry(spans),
            "shots_source": shots_source,
            "rules_applied": list(SILENT_RULES),
            "rules_skipped": [
                {"rule": rule, "reason": "lines is [] - no speech was expected"}
                for rule in LINE_RULES + SHOT_RULES
            ],
        }

    expected = [normalize_words(line["text"]) for line in parsed_lines]
    assigned, unassigned = align(expected, heard)
    mismatch = mismatch_rule(similarity_threshold)
    by_name = {span["name"]: span for span in spans or ()}

    lines = []
    previous_end = 0.0
    for index, (line, words, indices) in enumerate(
        zip(parsed_lines, expected, assigned)
    ):
        own = [heard[j] for j in indices]
        # A markup word spoken aloud is tag_spoken's to report, not a
        # mismatch too: it is left out of the line's score
        tokens = set(line["tokens"]) - set(words)
        score = line_similarity(
            words, [word["word"] for word in own if word["word"] not in tokens]
        )
        start = own[0]["start"] if own else None
        end = own[-1]["end"] if own else None
        # The shot the line names, else the one its heard words overlap most
        shot = line["shot"]
        if shot is None and spans is not None and own:
            shot = overlapping_shot(spans, start, end)
        span = by_name.get(shot)
        lines.append(
            {
                "expected": line["text"],
                "heard": _joined(own),
                "similarity": round(score, 4),
                "start": _round(start),
                "end": _round(end),
                "shot": shot,
            }
        )
        if score < similarity_threshold:
            if own:
                at = _at(index, own[0], shot=shot)
            else:
                # A dropped line: where its shot starts, else after the last
                # line heard
                at = _at(
                    index,
                    seconds=span["start"] if span else previous_end,
                    shot=shot,
                )
            findings.append(finding(mismatch, round(score, 4), at))

        for word in own:
            if word["word"] in tokens:
                findings.append(
                    finding(TAG_SPOKEN, word["word"], _at(index, word, shot=shot))
                )

        if own:
            tails = ([(shot, span["end"])] if span else []) + [(None, duration)]
            clipped = _clipped_tail(mono, sample_rate, end, tails)
            if clipped is not None:
                at = {"line": index, "seconds": _round(end), "word": own[-1]["text"]}
                if clipped[0] is not None or shot is not None:
                    at["shot"] = clipped[0] or shot
                findings.append(finding(CLIPPED_AT_END, _round(clipped[1], 2), at))
            previous_end = end

    for j, neighbours in unassigned:
        word = heard[j]
        for index in neighbours:
            if word["word"] in set(parsed_lines[index]["tokens"]) - set(
                expected[index]
            ):
                findings.append(
                    finding(
                        TAG_SPOKEN,
                        word["word"],
                        _at(index, word, shot=heard_shot(word)),
                    )
                )
                break

    rules_applied = list(LINE_RULES)
    rules_skipped = [
        {"rule": rule, "reason": "lines were given - speech is expected"}
        for rule in SILENT_RULES
    ]
    no_shot = _no_shot_reason(shots_source, spans, parsed_lines)
    if no_shot is None:
        rules_applied.extend(SHOT_RULES)
        named = {line["shot"] for line in parsed_lines if line["shot"]}
        for span in spans:
            if span["name"] in named:
                continue
            inside = [word for word in heard if heard_shot(word) == span["name"]]
            if inside:
                findings.append(
                    finding(
                        SPEECH_IN_SILENT_SHOT,
                        len(inside),
                        _at(None, inside[0], shot=span["name"]),
                    )
                )
    else:
        rules_skipped.extend({"rule": rule, "reason": no_shot} for rule in SHOT_RULES)

    findings.sort(key=lambda found: found["at"].get("seconds") or 0.0)
    return {
        "findings": findings,
        "lines": lines,
        "discarded": discarded,
        "unmatched": [_heard_entry(heard[j]) for j, _ in unassigned],
        "shots": _shots_entry(spans),
        "shots_source": shots_source,
        "rules_applied": rules_applied,
        "rules_skipped": rules_skipped,
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
    shots=None,
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
        lines: The expected lines in order - a list of strings or
            {text, shot}, 'shot' naming the take's shot the line belongs in.
            H3 markup (<d>[English] ...</d>, <scenetrans>, <cutoff>,
            [unclear], (S1)) is stripped, and a markup word heard spoken is
            'tag_spoken'. [] means no speech is expected.
        similarity: Least similarity (0..1) of a line's heard words to its
            expected words before it is a 'line_mismatch'. Default 0.85.
        model_name: HuggingFace ID of the Whisper-class ASR model (default:
            openai/whisper-base).
        sample_rate: Sample rate of a waveform passed directly.
        device: Where to run the ASR model.
        shots: The take's shot map ([{name, start_frame, num_frames,
            start_sample, num_samples}]) - default: the take's own (an earlier
            step's video), else the run manifest or keep_output sidecar
            beside its file. A line naming a shot the map lacks is refused.

    Returns:
        A JSON document: 'findings' (line_mismatch, tag_spoken,
        line_clipped_at_end, speech_where_silent, speech_in_silent_shot),
        'lines' (expected, heard, similarity, start, end, shot), 'shots'
        (name, start, end in seconds, or null), 'shots_source', 'discarded' (guarded words with their
        level and reason), 'unmatched' (heard words aligned to no line), 'transcript',
        'model_name', 'rules_applied', 'rules_skipped' and the 'thresholds'
        used.
    """
    check_arguments(COMMAND, similarity=similarity)
    parsed_lines = parse_lines(lines)
    parsed_shots = parse_shots(shots)
    path = (
        audio
        if isinstance(audio, str) and not audio.startswith(("http://", "https://"))
        else None
    )
    records, shots_source = resolve_shots(
        path, SimpleNamespace(shots=getattr(audio, "shots", None)), parsed_shots
    )
    if records:
        refusal = shot_names_error(parsed_lines, records, shots_source)
        if refusal:
            raise ValueError(refusal)
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
    spans = None
    if records:
        fps = getattr(audio, "fps", None)
        if (
            fps is None
            and path is not None
            and any(
                record.get("start_sample") is None or record.get("num_samples") is None
                for record in records
            )
        ):
            from ..media import probe_metadata

            fps = (probe_metadata(path) or {}).get("fps")
        spans = shot_spans(records, rate, fps, len(mono) / rate if rate else 0.0)
    answer = check(
        parsed_lines,
        transcript["chunks"],
        mono,
        rate,
        float(similarity),
        spans=spans,
        shots_source=shots_source,
    )
    answer["transcript"] = transcript["text"]
    answer["model_name"] = model_name
    answer["thresholds"] = dict(THRESHOLDS, similarity=float(similarity))
    return answer
