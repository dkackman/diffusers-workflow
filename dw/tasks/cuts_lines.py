"""plan_cuts' lines: when each sung line starts and ends, from the three
timing sources the task takes - a transcript's chunks, the song's own lyrics
aligned word by word onto those chunks, or analyze_phrases' phrases (#1095),
which pair with lyric lines one to one when the counts match and are spread
evenly when they do not.

Split from dw/tasks/cuts.py, which imports from here (never the other way):
cuts.py turns these lines into scenes, frames and a render grid.
"""

import difflib
import logging
import re
import statistics

from ..task_domains import real_number
from ..task_problems import transcript_problem

logger = logging.getLogger("dw")


COMMAND = "plan_cuts"

# A lyric line that is only a section tag ("[Chorus]") is structure, not a
# sung line: it breaks a stanza, as a blank line does
SECTION_TAG = re.compile(r"^\s*\[[^\]]*\]\s*$")

# Two words are the same word sung, misheard by at most this much
WORD_SIMILARITY = 0.75


def as_number(value, name, required=False):
    """A float from a number or numeric string; None stays None unless
    required."""
    number = real_number(value, name, COMMAND, required=required)
    return None if number is None else float(number)


def lines_of_lyrics(lyrics):
    """[(text, stanza index)] for the sung lines of the lyrics. A blank line
    or a section tag on its own line starts a new stanza; neither is a line."""
    if isinstance(lyrics, str):
        rows = lyrics.splitlines()
    elif isinstance(lyrics, (list, tuple)):
        rows = []
        for row in lyrics:
            rows.extend(str(row).splitlines() or [""])
    else:
        raise ValueError(
            f"{COMMAND}'s 'lyrics' is the song's text, one line per line, or a "
            f"list of lines - not {type(lyrics).__name__}"
        )
    lines, stanza, broke = [], 0, False
    for row in rows:
        text = row.strip()
        if not text or SECTION_TAG.match(text):
            broke = bool(lines)
            continue
        if broke:
            stanza += 1
            broke = False
        lines.append((text, stanza))
    if not lines:
        raise ValueError(f"{COMMAND}'s 'lyrics' has no sung lines")
    return lines


def _words(text):
    """The text's words, lower-cased with punctuation dropped."""
    return [word for word in re.findall(r"[\w']+", text.lower().replace("'", ""))]


def _timed_words(chunks):
    """[(word, start, end)] across the transcript: a chunk's span shared out
    over its words by length, since a segment chunk carries no word times."""
    timed = []
    for chunk in chunks:
        words = _words(chunk["text"])
        if not words:
            continue
        span = chunk["end"] - chunk["start"]
        total = sum(len(word) for word in words)
        at = chunk["start"]
        for word in words:
            length = span * len(word) / total
            timed.append((word, at, at + length))
            at += length
    return timed


def _similar(a, b):
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= WORD_SIMILARITY


def _align_words(lyric_words, heard_words):
    """{lyric word index: heard word index} - the longest in-order run of
    words the two share, a word matching one misheard by a little."""
    n, m = len(lyric_words), len(heard_words)
    # Longest common subsequence by similarity, filled from the end
    table = [[0] * (m + 1) for _ in range(n + 1)]
    same = {}
    for i in range(n - 1, -1, -1):
        row, below = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            if _similar(lyric_words[i], heard_words[j]):
                same[i, j] = True
                row[j] = below[j + 1] + 1
            else:
                row[j] = max(below[j], row[j + 1])
    pairs, i, j = {}, 0, 0
    while i < n and j < m:
        if same.get((i, j)) and table[i][j] == table[i + 1][j + 1] + 1:
            pairs[i] = j
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _heard_spans(lyric_lines, chunks):
    """[[start, end] or None] per lyric line - the span of its words the
    transcript heard, aligned word by word, or None when it heard none of
    them, and [[start or None] per word of the line] (#805) - the time each
    lyric word was heard at. A line's span runs no further than the next
    heard line's start. The word-level alignment stays here, apart from
    placing the unheard lines."""
    heard = _timed_words(chunks)
    lyric_words, owner = [], []
    for index, (text, _) in enumerate(lyric_lines):
        for word in _words(text):
            lyric_words.append(word)
            owner.append(index)
    pairs = _align_words(lyric_words, [word for word, _, _ in heard])

    timed = [None] * len(lyric_lines)
    word_times = [[None] * len(_words(text)) for text, _ in lyric_lines]
    first_word = {}
    for lyric_index, line in enumerate(owner):
        first_word.setdefault(line, lyric_index)
    for lyric_index, heard_index in pairs.items():
        line = owner[lyric_index]
        _, start, end = heard[heard_index]
        word_times[line][lyric_index - first_word[line]] = start
        if timed[line] is None:
            timed[line] = [start, end]
        else:
            timed[line][0] = min(timed[line][0], start)
            timed[line][1] = max(timed[line][1], end)

    known = [index for index, span in enumerate(timed) if span is not None]
    for a, b in zip(known, known[1:]):
        timed[a][1] = min(timed[a][1], timed[b][0])
    return timed, word_times


def _line_tokens(text, times, span):
    """[(token, time)] - the line's whitespace-separated tokens, each with
    the time its first word was heard at (#805). A line none of whose words
    were heard is spread evenly over its span; a token that is no word, or
    whose words were not heard, has time None."""
    pieces = text.split()
    counts = [len(_words(piece)) for piece in pieces]
    if not any(time is not None for time in times):
        total = sum(1 for count in counts if count)
        tokens, seen = [], 0
        for piece, count in zip(pieces, counts):
            if count and span is not None:
                at = span[0] + (span[1] - span[0]) * seen / total
                seen += 1
            else:
                at = None
            tokens.append((piece, at))
        return tokens
    tokens, used = [], 0
    for piece, count in zip(pieces, counts):
        heard = [t for t in times[used : used + count] if t is not None]
        used += count
        tokens.append((piece, heard[0] if heard else None))
    return tokens


def align_lyrics(lyric_lines, chunks, duration, warnings, min_line_s):
    """[{start, end, text, stanza}] - the lyrics' lines in their order, timed
    from the transcript. A line none of whose words were heard is placed
    between its neighbours, and warned about; when they touch it takes
    min_line_s from them, so it is still a shot of its own."""
    timed, word_times = _heard_spans(lyric_lines, chunks)
    heard_times = [list(times) for times in word_times]
    known = [index for index, span in enumerate(timed) if span is not None]
    if not known:
        warnings.append(
            f"{COMMAND}: none of the lyrics' words were heard in the transcript "
            "- the lines are spread evenly over the sung part of the song; "
            "check the transcript is of this song"
        )
    vocal_start = chunks[0]["start"] if chunks else 0.0
    vocal_end = chunks[-1]["end"] if chunks else duration
    index = 0
    while index < len(timed):
        if timed[index] is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(timed) and timed[run_end] is None:
            run_end += 1
        following = timed[run_end][0] if run_end < len(timed) else None
        if index > 0:
            low = timed[index - 1][1]
        else:
            low = vocal_start if following is None else min(vocal_start, following)
        high = following if following is not None else max(vocal_end, low)
        count = run_end - index
        # The neighbours took the unheard line's time with its misheard words:
        # give it back from each side, never more than half a neighbour
        short = count * min_line_s - (high - low)
        if short > 1e-9:
            before = timed[index - 1] if index > 0 else None
            after = timed[run_end] if run_end < len(timed) else None
            spare_before = (before[1] - before[0]) / 2 if before else 0.0
            spare_after = (after[1] - after[0]) / 2 if after else 0.0
            take_before = min(spare_before, max(short / 2, short - spare_after))
            take_after = min(spare_after, short - take_before)
            low -= take_before
            high += take_after
            if before:
                before[1] = low
            if after:
                after[0] = high
        step = (high - low) / count
        for offset in range(count):
            timed[index + offset] = [low + offset * step, low + (offset + 1) * step]
            if known:
                warnings.append(
                    f"{COMMAND}: lyric line {index + offset + 1} "
                    f"({lyric_lines[index + offset][0]!r}) was not heard in the "
                    f"transcript - placed between its neighbours, at "
                    f"{timed[index + offset][0]:.2f}-{timed[index + offset][1]:.2f} s"
                )
        index = run_end
    return [
        {
            "start": span[0],
            "end": span[1],
            "text": text,
            "stanza": stanza,
            "tokens": _line_tokens(text, times, span) if known else None,
        }
        for span, times, (text, stanza) in zip(timed, heard_times, lyric_lines)
    ]


def lines_of_transcript(chunks, stanza_gap):
    """[{start, end, text, stanza}] - a line per chunk; a silence of at least
    stanza_gap starts a new stanza."""
    lines, stanza = [], 0
    for index, chunk in enumerate(chunks):
        if not chunk["text"]:
            continue
        if lines and chunk["start"] - lines[-1]["end"] >= stanza_gap:
            stanza += 1
        lines.append(dict(chunk, stanza=stanza))
    return lines


def is_phrases(transcript):
    """Whether this is analyze_phrases' result (#1095): timing with no text,
    told from a transcript by its 'phrases' key."""
    return (
        isinstance(transcript, dict)
        and "phrases" in transcript
        and "chunks" not in transcript
    )


def phrase_list(transcript):
    """[{start, end, text}] in seconds, ascending - an analyze_phrases
    result's phrases, each with no text. A phrase is never suspect."""
    problem = transcript_problem(transcript)
    if problem:
        raise ValueError(problem)
    parsed = []
    for index, phrase in enumerate(transcript["phrases"]):
        if not isinstance(phrase, dict):
            raise ValueError(
                f"{COMMAND}: phrase {index} is not a {{start, end}} object"
            )
        start = as_number(phrase.get("start"), f"phrases[{index}].start", required=True)
        end = as_number(phrase.get("end"), f"phrases[{index}].end", required=True)
        parsed.append({"start": start, "end": max(end, start), "text": ""})
    parsed.sort(key=lambda phrase: phrase["start"])
    return parsed


def lines_of_phrases(phrases, lyrics, duration, stanza_gap, warnings):
    """[{start, end, text, stanza, tokens}] from analyze_phrases' phrases.
    Without lyrics a line per phrase, with no text, and a silence of at least
    stanza_gap starts a stanza. With lyrics whose lines are as many as the
    phrases, line n is phrase n; otherwise the counts are warned about, with
    the analyze_phrases knob to turn, and the lines spread evenly from the
    first phrase to the last - never paired. A phrase has no word times, so
    a split piece of one keeps its whole lyric."""
    if lyrics is None:
        lines, stanza = [], 0
        for phrase in phrases:
            if lines and phrase["start"] - lines[-1]["end"] >= stanza_gap:
                stanza += 1
            lines.append(dict(phrase, stanza=stanza, tokens=None))
        return lines
    if len(lyrics) == len(phrases):
        paired = [
            dict(phrase, text=text, stanza=stanza, tokens=None)
            for phrase, (text, stanza) in zip(phrases, lyrics)
        ]
        _disproportion_warning(paired, warnings)
        return paired
    if len(phrases) > len(lyrics):
        knob = "raise analyze_phrases' 'min_gap_seconds' to bridge more pauses, so the counts match"
    else:
        knob = (
            "lower analyze_phrases' 'min_gap_seconds' to split at shorter "
            "pauses, or, for legato or reverberant singing whose lines fuse, "
            "raise its 'floor_db' (less negative, e.g. -30), so the counts match - then "
            "check the phrases by ear before trusting a one-to-one pairing, since a "
            "raised floor can match the counts on wrong boundaries"
        )
    low = phrases[0]["start"] if phrases else 0.0
    high = phrases[-1]["end"] if phrases else duration
    warnings.append(
        f"{COMMAND}: the lyrics have {len(lyrics)} line(s) and analyze_phrases "
        f"found {len(phrases)} phrase(s) - the lines are spread evenly over "
        f"{low:.2f}-{high:.2f} s, not paired with the phrases; {knob}"
    )
    step = (high - low) / len(lyrics)
    return [
        {
            "start": low + index * step,
            "end": low + (index + 1) * step,
            "text": text,
            "stanza": stanza,
            "tokens": None,
        }
        for index, (text, stanza) in enumerate(lyrics)
    ]


# A paired phrase this far from the median phrase length is likely a bad
# boundary (#1095): on a fused legato verse, a raised floor_db can split a
# breath off as a "phrase" and leave another line spanning two.
DISPROPORTION = 4.0


def _disproportion_warning(paired, warnings):
    """Warn when a lyric line is paired one to one with a phrase under a
    quarter, or over four times, the median phrase duration. The pairing
    stands; the warning is advice to check the boundaries by ear."""
    lengths = [line["end"] - line["start"] for line in paired]
    if len(lengths) < 2:
        return
    median = statistics.median(lengths)
    if median <= 0:
        return
    odd = [
        f"line {index + 1} ({line['text']!r}, {line['end'] - line['start']:.2f} s)"
        for index, line in enumerate(paired)
        if not median / DISPROPORTION
        <= line["end"] - line["start"]
        <= median * DISPROPORTION
    ]
    if odd:
        warnings.append(
            f"{COMMAND}: the lyric lines are paired one to one with "
            f"analyze_phrases' phrases, but {', '.join(odd)} "
            f"{'sits' if len(odd) == 1 else 'sit'} on a phrase "
            f"under 1/{DISPROPORTION:g} or over {DISPROPORTION:g}x the median "
            f"phrase ({median:.2f} s) - the phrase boundaries may be wrong; "
            "check them by ear"
        )
