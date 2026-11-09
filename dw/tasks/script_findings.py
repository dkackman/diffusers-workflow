"""check_script's findings: the rules a take is judged by and the functions
that build each finding from the aligned lines, the heard words and the take's
levels - a clipped tail, a loop, an unheard line, speech where the script has
none, a tag spoken aloud, speech in a shot meant to be silent.

Every threshold is a constant here, named in docs/TASKS.md; findings are built
with `assessment_rules.finding()`. Split from dw/tasks/script_check.py, which
imports from here (never the other way) and does the transcription, the
alignment and the shot placement that these read.
"""

import difflib
import logging

from .. import dsp
from ..assessment_rules import finding
from ..script_lines import (
    REPEAT_MAX_PERIOD,
    REPEAT_RUN_MIN,
    looped_texts_firsts,
    normalize_words,
)
from .script_take import (
    GUARD_FLOOR_DBFS,
    GUARD_WINDOW_SECONDS,
    audible_runs,
    heard_shot,
    overlapping_shot,
    round_seconds,
    shots_entry,
    word_level,
)

logger = logging.getLogger("dw")


def _rule(name, threshold, says):
    return {"name": name, "severity": "warn", "threshold": threshold, "says": says}


def mismatch_rule(similarity):
    return _rule(
        "line_mismatch",
        similarity,
        "the take's words for this line are this similar to the script - listen"
        " before re-rolling: Whisper mishears names and homophones on a correct"
        " take (number words and digits already read as the same)",
    )


# Default least similarity between a line's expected and heard words before
# it is a `line_mismatch`. The template's own default is set from stage A's
# two-model measurement (#609 Q3); this one is the task's
DEFAULT_SIMILARITY = 0.85

# GUARD_FLOOR_DBFS and GUARD_WINDOW_SECONDS, the guard over heard words,
# live in dw/tasks/script_take.py beside `guard_words`
# A line is clipped at the end when its last heard word ends inside the
# file's final CLIP_TAIL_SECONDS and that tail measures above
# GUARD_FLOOR_DBFS - still voiced when the file stops, not a word that
# finished into a quiet tail
CLIP_TAIL_SECONDS = 0.25

# ... and within CLIP_TAIL_BELOW_WORD_DB of the last word's own loudest window:
# a word cut mid-sound leaves a tail near its level, a score bed or room tone
# under a word that finished sits far below it (#953)
CLIP_TAIL_BELOW_WORD_DB = 15.0

# Audible speech the transcript has no words for (#923): Whisper can drop a
# passage from its text on a longer input (the ad-lib heard in a short file
# is gone), and the word alignment hands that time to a neighbouring word,
# which stretches. A gap between a line's words,
# or a word, lasting UNHEARD_MIN_SECONDS or more, whose interior - trimmed by
# UNHEARD_MARGIN_SECONDS each side - holds a window above UNHEARD_FLOOR_DBFS,
# is speech the words do not account for. The floor is far above the
# dead-air floor so room tone and a pause's tail do not trip it
UNHEARD_MIN_SECONDS = 1.0

UNHEARD_MARGIN_SECONDS = 0.25

UNHEARD_FLOOR_DBFS = -40.0

# A block whose samples repeat themselves - one stretch of it matching the
# same stretch BED_MIN_PERIOD_SECONDS or more later at BED_CORRELATION or
# better, over at least BED_MIN_OVERLAP_SECONDS - is a looped bed (loop_audio
# of a room tone), not speech: the same sound does not recur to the sample
# in a voice (#1118)
BED_MIN_PERIOD_SECONDS = 0.3

BED_MIN_OVERLAP_SECONDS = 0.5

BED_CORRELATION = 0.98

THRESHOLDS = {
    "similarity": DEFAULT_SIMILARITY,
    "guard_floor_dbfs": GUARD_FLOOR_DBFS,
    "guard_window_seconds": GUARD_WINDOW_SECONDS,
    "clip_tail_seconds": CLIP_TAIL_SECONDS,
    "clip_tail_below_word_db": CLIP_TAIL_BELOW_WORD_DB,
    "unheard_min_seconds": UNHEARD_MIN_SECONDS,
    "unheard_margin_seconds": UNHEARD_MARGIN_SECONDS,
    "unheard_floor_dbfs": UNHEARD_FLOOR_DBFS,
    "bed_correlation": BED_CORRELATION,
    "repeat_run_min": REPEAT_RUN_MIN,
    "repeat_max_period": REPEAT_MAX_PERIOD,
}

LINE_RULES = (
    "line_mismatch",
    "transcript_loop",
    "tag_spoken",
    "line_clipped_at_end",
    "speech_not_transcribed",
)

SILENT_RULES = ("speech_where_silent",)

SHOT_RULES = ("speech_in_silent_shot", "unscripted_speech")

TAG_SPOKEN = _rule(
    "tag_spoken",
    None,
    "a markup word from this line was heard spoken aloud - the model read the"
    " tag as dialogue",
)

SPEECH_NOT_TRANSCRIBED = _rule(
    "speech_not_transcribed",
    UNHEARD_FLOOR_DBFS,
    "audible sound lasting this many seconds sits between the line's words, between"
    " two lines, or inside one stretched word, with no word for it - the transcript left speech"
    " out and the line's timing absorbed it; listen",
)

TRANSCRIPT_LOOP = _rule(
    "transcript_loop",
    REPEAT_RUN_MIN,
    "the transcript repeated words of the script in a loop for this many chunks"
    " (seen at 'seconds' to 'end') - Whisper stuck, so speech after it may be"
    " missing from 'heard' and lines can read 0.0 without being wrong; listen,"
    " or check each shot on its own, before re-rolling",
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
    f" still voiced at this level, within {CLIP_TAIL_BELOW_WORD_DB} dB of the"
    " word's own level (a quieter tail is a bed or room tone) - the shot"
    " stops mid-word",
)

SPEECH_IN_SILENT_SHOT = _rule(
    "speech_in_silent_shot",
    0,
    "no line names this shot, and this many words were heard in it above the"
    " dead-air floor and outside a repetition loop",
)

UNSCRIPTED_SPEECH = _rule(
    "unscripted_speech",
    0,
    "this many words were heard inside a shot that has a scripted line, matched to"
    " no line - a line spoken twice (see 'repeats_line'), or an ad-lib; listen",
)


def line_similarity(expected, heard):
    """How alike two word lists are, 0..1. Two empty lists are identical."""
    if not expected and not heard:
        return 1.0
    plain = difflib.SequenceMatcher(None, expected, heard, autojunk=False).ratio()
    merged_e = _merge_compounds(expected, heard)
    merged_h = _merge_compounds(heard, expected)
    if merged_e == expected and merged_h == heard:
        return plain
    merged = difflib.SequenceMatcher(None, merged_e, merged_h, autojunk=False).ratio()
    return max(plain, merged)


def _merge_compounds(words, other):
    """`words` with adjacent words joined where the join is a word of `other`:
    a compound written open on one side and closed on the other ('candle
    crumbs' / 'candlecrumbs') is a spelling choice, not a different line.

    Digits join only as a year said in halves - two pieces of two digits
    each ('17 87' is '1787', the fold of 'seventeen eighty seven'). Other
    digits written apart are separate numbers: '2 1' (said 'two one') is
    not '21' (#1297)."""
    targets = set(other)
    out = []
    i = 0
    while i < len(words):
        for span in (3, 2):
            if i + span > len(words):
                continue
            pieces = words[i : i + span]
            if all(piece.isdigit() for piece in pieces) and not _year_halves(pieces):
                continue
            joined = "".join(pieces)
            if joined in targets:
                out.append(joined)
                i += span
                break
        else:
            out.append(words[i])
            i += 1
    return out


def _year_halves(pieces):
    """Whether digit `pieces` are a year said in halves: two of two digits."""
    return len(pieces) == 2 and all(len(piece) == 2 for piece in pieces)


def tail_level(mono, sample_rate, end=None):
    """The rms of the CLIP_TAIL_SECONDS before `end` seconds - the file's
    final ones when None - in dBFS (None: silent)."""
    count = max(1, int(round(CLIP_TAIL_SECONDS * sample_rate)))
    last = len(mono) if end is None else min(len(mono), int(round(end * sample_rate)))
    return dsp.dbfs(dsp.rms(mono[max(0, last - count) : last]))


def entry_at(line, heard_word=None, seconds=None, shot=None):
    at = {"line": line}
    if heard_word is not None:
        at["seconds"] = round_seconds(heard_word["start"])
        at["word"] = heard_word["text"]
    else:
        at["seconds"] = round_seconds(seconds)
    if shot is not None:
        at["shot"] = shot
    return at


def clipped_tail(mono, sample_rate, end, tails, word=None):
    """The first of `tails` - (shot name or None, the second it ends at) -
    that a line ending at `end` is clipped by: the word ends inside its
    final CLIP_TAIL_SECONDS (within that much past it too, a word timestamp
    being approximate) and the tail is voiced above GUARD_FLOOR_DBFS and
    within CLIP_TAIL_BELOW_WORD_DB of the (start, end) span's loudest window,
    when given: a tail far under the word is a bed, not the word. Returns (shot, level) or None."""
    word_db = word_level(mono, sample_rate, *word) if word else None
    floor = GUARD_FLOOR_DBFS
    if word_db is not None:
        floor = max(floor, word_db - CLIP_TAIL_BELOW_WORD_DB)
    for shot, tail_end in tails:
        if not tail_end - CLIP_TAIL_SECONDS <= end <= tail_end + CLIP_TAIL_SECONDS:
            continue
        level = tail_level(mono, sample_rate, tail_end)
        if level is not None and level > floor:
            return shot, level
    return None


def uncovered_speech_findings(spans, heard, duration):
    """A `shot_coverage` skip entry (#1129) naming each stretch of the take
    outside every supplied shot where words were heard: no shot rule looks
    there, so a clean result must not read as checked."""
    if not spans:
        return []
    gaps, cursor = [], 0.0
    for span in sorted(spans, key=lambda s: s["start"]):
        if span["start"] > cursor:
            gaps.append((cursor, span["start"]))
        cursor = max(cursor, span["end"])
    if duration > cursor:
        gaps.append((cursor, duration))
    named = []
    for start, end in gaps:
        if any(start <= (w["start"] + w["end"]) / 2 < end for w in heard):
            named.append(f"{round_seconds(start)}-{round_seconds(end)} s")
    if not named:
        return []
    return [
        {
            "rule": "shot_coverage",
            "reason": "speech was heard outside the supplied shot map and "
            "not checked against any shot: " + ", ".join(named),
        }
    ]


def loop_findings(chunks, script_words):
    """A `transcript_loop` finding for each repetition loop over the script's
    own words: Whisper stuck on a line it heard, so what follows the loop was
    never transcribed and its lines read as mismatches (#961)."""
    spoken = [c for c in chunks if normalize_words(c.get("text", ""))]
    looped, firsts = looped_texts_firsts([c.get("text", "") for c in spoken])
    found, run = [], []
    for index in sorted(looped) + [None]:
        if run and (index is None or index != run[-1] + 1):
            if any(i in firsts for i in run):
                members = [spoken[i] for i in run]
                if all(
                    set(normalize_words(m["text"])) <= script_words for m in members
                ):
                    at = entry_at(None, seconds=members[0]["start"])
                    at["end"] = round_seconds(members[-1]["end"])
                    found.append(finding(TRANSCRIPT_LOOP, len(members), at))
            run = []
        if index is not None:
            run.append(index)
    return found


def silent_take_findings(heard, discarded, spans, shots_source):
    """check's result for `lines: []`: any speech at all is the finding, and
    the line and shot rules have nothing to compare against."""
    findings = []
    if heard:
        findings.append(
            finding(
                SPEECH_WHERE_SILENT,
                len(heard),
                entry_at(None, heard[0], shot=heard_shot(spans, heard[0])),
            )
        )
    return {
        "findings": findings,
        "lines": [],
        "discarded": discarded,
        "unmatched": [heard_entry(word) for word in heard],
        "shots": shots_entry(spans),
        "shots_source": shots_source,
        "rules_applied": list(SILENT_RULES),
        "rules_skipped": [
            {"rule": rule, "reason": "lines is [] - no speech was expected"}
            for rule in LINE_RULES + SHOT_RULES
        ],
    }


def untranscribed_findings(
    mono, sample_rate, index, own, shot, spans=None, parsed_lines=()
):
    """speech_not_transcribed: audible sound in a gap the line's words leave
    - between two of them - or inside one word stretched over
    UNHEARD_MIN_SECONDS (#923). The gap before a line's first word is
    between_lines_findings'."""
    intervals = [(a["end"], b["start"]) for a, b in zip(own, own[1:])]
    findings = []
    for start, end in intervals:
        if end - start >= UNHEARD_MIN_SECONDS:
            findings.extend(
                _audible_findings(
                    mono, sample_rate, index, start, end, shot, spans, parsed_lines
                )
            )
    for word in own:
        # A word is sound itself: only its interior past the margins is a gap
        if (
            word["end"] - word["start"]
            >= UNHEARD_MIN_SECONDS + 2 * UNHEARD_MARGIN_SECONDS
        ):
            findings.extend(
                _audible_findings(
                    mono,
                    sample_rate,
                    index,
                    word["start"],
                    word["end"],
                    shot,
                    spans,
                    parsed_lines,
                )
            )
    return findings


def between_lines_findings(
    mono, sample_rate, index, start, end, shot, strays, spans=None, parsed_lines=()
):
    """speech_not_transcribed for the stretch between one line's last word and
    the next line's first (#960): audible sound with no word, which the
    in-line check cannot see when the timing leaves it outside both lines.
    Heard-but-unmatched words in the gap are unscripted_speech's, but only the
    stretches they cover: each stretch left between them is judged on its own
    (#964). Each stretch is attributed to the shot holding most of it, and
    to the line naming that shot (null when none does), not to the next
    line's shot by default."""
    if end - start < UNHEARD_MIN_SECONDS:
        return []
    inside = sorted(
        (w for w in strays if w["start"] < end and w["end"] > start),
        key=lambda w: w["start"],
    )
    stretches, cursor = [], start
    for word in inside:
        stretches.append((cursor, word["start"]))
        cursor = max(cursor, word["end"])
    stretches.append((cursor, end))
    found = []
    for lo, hi in stretches:
        if hi - lo < UNHEARD_MIN_SECONDS:
            continue
        found.extend(
            _audible_findings(
                mono, sample_rate, index, lo, hi, shot, spans, parsed_lines
            )
        )
    return [f for f in found if f is not None]


def _audible_findings(
    mono, sample_rate, index, start, end, shot, spans=None, parsed_lines=()
):
    """speech_not_transcribed for each audible block of [start, end], each
    attributed to the shot holding most of it - and to the line naming that
    shot, null when none does - not to the stretch's own line by default."""
    found = []
    pieces = []
    for lo, hi in audible_runs(
        mono,
        sample_rate,
        start,
        end,
        UNHEARD_FLOOR_DBFS,
        UNHEARD_MIN_SECONDS,
        UNHEARD_MARGIN_SECONDS,
    ):
        # A block crossing a shot seam is one finding per shot (#1115), so the
        # shot that holds the untranscribed speech is named even if the other
        # holds more of the block.
        cuts = [lo]
        if spans is not None:
            cuts += sorted(
                {s["start"] for s in spans if lo < s["start"] < hi}
                | {s["end"] for s in spans if lo < s["end"] < hi}
            )
        cuts.append(hi)
        parts = _folded(list(zip(cuts, cuts[1:])))
        # Split only when each side is substantial; else one finding, placed
        # by majority as before (#964)
        pieces.extend(parts if len(parts) > 1 else [(lo, hi)])
    for lo, hi in pieces:
        where_line, where_shot = index, shot
        holder = None if spans is None else overlapping_shot(spans, lo, hi)
        if holder is not None and holder != shot:
            where_shot = holder
            where_line = next(
                (i for i, line in enumerate(parsed_lines) if line["shot"] == holder),
                None,
            )
        found.append(_untranscribed(mono, sample_rate, where_line, lo, hi, where_shot))
    return [f for f in found if f is not None]


def _folded(parts):
    """`parts`, consecutive (start, end) slices of one block, with every slice
    too short to stand as its own finding (under UNHEARD_MIN_SECONDS plus both
    margins) folded into the slice before it - the first into the one after -
    so a short shot a block covers whole is never dropped from the report, and
    the block's whole span stays accounted for."""
    least = UNHEARD_MIN_SECONDS + 2 * UNHEARD_MARGIN_SECONDS
    folded = []
    for a, b in parts:
        if folded and (b - a < least or folded[-1][1] - folded[-1][0] < least):
            folded[-1] = (folded[-1][0], b)
        else:
            folded.append((a, b))
    return folded


def is_looped_bed(mono, sample_rate, start, end):
    """True when [start, end] repeats itself sample for sample (#1118): a
    looped room-tone or score bed, which a voice never does. The measurement
    is `dsp.repeat_correlation`; this is the decision."""
    first, last = int(round(start * sample_rate)), int(round(end * sample_rate))
    block = mono[max(0, first) : last]
    return (
        dsp.repeat_correlation(
            block, sample_rate, BED_MIN_PERIOD_SECONDS, BED_MIN_OVERLAP_SECONDS
        )
        >= BED_CORRELATION
    )


def _untranscribed(mono, sample_rate, index, start, end, shot):
    lo, hi = start + UNHEARD_MARGIN_SECONDS, end - UNHEARD_MARGIN_SECONDS
    level = word_level(mono, sample_rate, lo, hi)
    if level is None or level <= UNHEARD_FLOOR_DBFS:
        return None
    if is_looped_bed(mono, sample_rate, start, end):
        return None
    at = entry_at(index, seconds=start, shot=shot)
    at["end"] = round_seconds(end)
    return finding(SPEECH_NOT_TRANSCRIBED, round_seconds(end - start, 2), at)


def clipped_findings(mono, sample_rate, tails, index, own, shot):
    """line_clipped_at_end for one heard line, as a list of zero or one
    finding: its last word ends in a voiced tail of its shot or the file. The
    tail is judged against the whole line's loudest window, not the last
    word's: under a score the ASR stretches the last word over the bed, and a
    bed measured against itself passes the guard (#959)."""
    end = own[-1]["end"]
    clipped = clipped_tail(
        mono, sample_rate, end, tails, (own[0]["start"], own[-1]["end"])
    )
    if clipped is None:
        return []
    at = {"line": index, "seconds": round_seconds(end), "word": own[-1]["text"]}
    if clipped[0] is not None or shot is not None:
        at["shot"] = clipped[0] or shot
    return [finding(CLIPPED_AT_END, round_seconds(clipped[1], 2), at)]


def unassigned_tags_findings(heard, unassigned, parsed_lines, expected, spans):
    """tag_spoken for the heard words align left unassigned: a markup token
    of a neighbouring line, spoken aloud between lines."""
    findings = []
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
                        entry_at(index, word, shot=heard_shot(spans, word)),
                    )
                )
                break
    return findings


def silent_shot_findings(spans, parsed_lines, heard, claimed):
    """speech_in_silent_shot: words heard inside a shot no line names, other
    than those a line in another shot claimed."""
    findings = []
    named = {line["shot"] for line in parsed_lines if line["shot"]}
    for span in spans:
        if span["name"] in named:
            continue
        inside = [
            word
            for j, word in enumerate(heard)
            if heard_shot(spans, word) == span["name"]
            and claimed.get(j) in (None, span["name"])
        ]
        if inside:
            at = entry_at(None, inside[0], shot=span["name"])
            # A word straddling the cut into this shot is placed at the cut
            at["seconds"] = max(at["seconds"], round_seconds(span["start"]))
            findings.append(finding(SPEECH_IN_SILENT_SHOT, len(inside), at))
    return findings


def unscripted_findings(
    spans, parsed_lines, expected, heard, unassigned, similarity_threshold
):
    """unscripted_speech: heard words align left to no line that sit inside a
    shot a line names. `repeats_line` is set when the shot's words, in order,
    contain one of its lines' words again, or score at least the mismatch
    threshold against one of them (a repeat misheard as the line was)."""
    findings = []
    for span in spans:
        lines = [
            expected[i]
            for i, line in enumerate(parsed_lines)
            if line["shot"] == span["name"]
        ]
        if not lines:
            continue
        inside = [
            heard[j]
            for j, _ in unassigned
            if heard_shot(spans, heard[j]) == span["name"]
        ]
        if not inside:
            continue
        at = entry_at(None, inside[0], shot=span["name"])
        words = [word["word"] for word in inside]
        at["end"] = round_seconds(inside[-1]["end"])
        at["repeats_line"] = any(
            line and contains_words(words, line) for line in lines
        ) or any(line and contains_words(line, words) for line in lines)
        if not at["repeats_line"]:
            at["repeats_line"] = any(
                line and line_similarity(line, words) >= similarity_threshold
                for line in lines
            )
        findings.append(finding(UNSCRIPTED_SPEECH, len(inside), at))
    return findings


def contains_words(words, part):
    """Whether `part` occurs in `words` as a contiguous run."""
    size = len(part)
    return any(words[i : i + size] == part for i in range(len(words) - size + 1))


def heard_entry(word):
    return {
        "text": word["text"],
        "start": round_seconds(word["start"]),
        "end": round_seconds(word["end"]),
    }


def joined_words(words):
    """The heard words as the transcript spelled them, one chunk once."""
    texts = []
    last = None
    for word in words:
        key = (word["start"], word["end"], word["text"])
        if key != last:
            texts.append(word["text"])
            last = key
    return " ".join(texts)
