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
import math
from types import SimpleNamespace

from .. import dsp
from ..assessment_rules import DEAD_AIR_FLOOR_DBFS, finding
from ..task_domains import check_arguments
from ..events import emit_log
from ..locations import is_http_url
from ..script_lines import (
    COMMAND,
    REPEAT_MAX_PERIOD,
    REPEAT_RUN_MIN,
    disambiguated,
    looped_texts_firsts,
    normalize_words,
    parse_lines,
    parse_shots,
    shot_names_error,
    with_full_shot_names,
)
from .assess import DEAD_AIR_WINDOW, resolve_shots, sample_span
from .audio_transcription import WHISPER_DEFAULT_MODEL
from .audio_utils import waveform_and_rate

logger = logging.getLogger("dw")

# Default least similarity between a line's expected and heard words before
# it is a `line_mismatch`. The template's own default is set from stage A's
# two-model measurement (#609 Q3); this one is the task's
DEFAULT_SIMILARITY = 0.85
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
# REPEAT_RUN_MIN and REPEAT_MAX_PERIOD, the repetition-loop guard, live in
# dw/script_lines.py beside the detector, which transcribe_audio shares

THRESHOLDS = {
    "similarity": DEFAULT_SIMILARITY,
    "guard_floor_dbfs": GUARD_FLOOR_DBFS,
    "guard_window_seconds": GUARD_WINDOW_SECONDS,
    "clip_tail_seconds": CLIP_TAIL_SECONDS,
    "clip_tail_below_word_db": CLIP_TAIL_BELOW_WORD_DB,
    "unheard_min_seconds": UNHEARD_MIN_SECONDS,
    "unheard_margin_seconds": UNHEARD_MARGIN_SECONDS,
    "unheard_floor_dbfs": UNHEARD_FLOOR_DBFS,
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


# Whisper drops soft speech from a quiet input, so the same shot got different
# verdicts by level (#925). The transcriber hears the take scaled so its loud
# end sits near full scale - a take only 2 dB under that still lost a soft
# "Uh," that +6 dB brought back; the level checks keep reading the original
# waveform.
TRANSCRIBE_PEAK = 0.9
TRANSCRIBE_MAX_GAIN = 100.0  # 40 dB: a near-silent take is not boosted to noise


def level_for_transcription(waveform):
    """(waveform, gain) with the 99.9th-percentile magnitude scaled to
    TRANSCRIBE_PEAK (spikes past full scale clipped); gain is 1.0 for silence."""
    import numpy

    loud = float(numpy.percentile(numpy.abs(waveform), 99.9)) if waveform.size else 0.0
    if loud <= 0.0:
        return waveform, 1.0
    gain = min(TRANSCRIBE_PEAK / loud, TRANSCRIBE_MAX_GAIN)
    leveled = numpy.clip(waveform * gain, -1.0, 1.0).astype(numpy.float32)
    return leveled, gain


def shot_spans(records, sample_rate, fps, duration):
    """Each shot record as {name, start, end} in seconds on the take, or None
    when one cannot be placed. Placed by `assess.sample_span`, the rule the
    probes use (recorded sample span, else frames by `fps`). Clipped to the
    take's `duration`."""
    if not sample_rate:
        return None
    media = SimpleNamespace(fps=fps, sample_rate=sample_rate)
    spans = []
    for record in records:
        start, count, _source = sample_span(record, media)
        if start is None:
            return None
        start, end = start / sample_rate, (start + count) / sample_rate
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
    dBFS, or None for silence (`dsp.loudest_window_dbfs`)."""
    return dsp.loudest_window_dbfs(mono, sample_rate, start, end, GUARD_WINDOW_SECONDS)


def _round(value, places=3):
    return None if value is None else round(float(value), places)


def guard_words(chunks, mono, sample_rate, script_words=None):
    """Split transcribe_audio's word chunks into (heard, discarded).

    `heard` is [{word, text, start, end}], one per normalized word - a chunk
    that normalizes to nothing ('♪', '...') is neither; `discarded` is
    [{text, start, end, level_dbfs, reason, count}] for a chunk at or below
    the floor ("below_floor") or a repetition loop ("repetition": one entry
    per run of consecutive looped chunks, `count` chunks, `text` the first's,
    `start`/`end` the run's span).

    A loop's first occurrence is kept as heard when `script_words` (the
    words the script holds) contains all of its words and it has real
    duration and speech level: Whisper loops on a line it did hear (#961).
    """
    spoken = []
    for chunk in chunks:
        words = normalize_words(chunk.get("text", ""))
        if words:
            spoken.append((chunk, words))
    # The loop detector is script_lines' (shared with transcribe_audio's
    # `suspect: "repetition"`); a chunk with no word is in no loop
    looped, firsts = looped_texts_firsts([chunk.get("text", "") for chunk, _ in spoken])

    heard, discarded = [], []
    run_end = None  # spoken index of the last chunk folded into discarded[-1]
    for index, (chunk, words) in enumerate(spoken):
        start, end = chunk["start"], chunk["end"]
        level = word_level(mono, sample_rate, start, end)
        first_of_line = (
            index in firsts
            and script_words is not None
            and set(words) <= script_words
            and end > start
        )
        if index in looped and not first_of_line:
            reason = "repetition"
        elif level is None or level <= GUARD_FLOOR_DBFS:
            reason = "below_floor"
        else:
            reason = None
        if reason == "repetition" and run_end == index - 1:
            # A loop is one entry: its span, its chunk count, its loudest level
            last = discarded[-1]
            last["end"] = _round(max(last["end"], end))
            last["count"] += 1
            if level is not None and (
                last["level_dbfs"] is None or level > last["level_dbfs"]
            ):
                last["level_dbfs"] = _round(level, 2)
            run_end = index
            continue
        if reason:
            discarded.append(
                {
                    "text": chunk.get("text", "").strip(),
                    "start": _round(start),
                    "end": _round(end),
                    "level_dbfs": _round(level, 2),
                    "reason": reason,
                    "count": 1,
                }
            )
            run_end = index if reason == "repetition" else None
            continue
        run_end = None
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


def _clipped_tail(mono, sample_rate, end, tails, word=None):
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
    script_words = {
        word for line in parsed_lines for word in normalize_words(line["text"])
    }
    heard, discarded = guard_words(chunks, mono, sample_rate, script_words)
    duration = len(mono) / sample_rate if sample_rate else 0.0
    findings = _loop_findings(chunks, script_words) if parsed_lines else []
    loop_starts = [found["at"]["seconds"] for found in findings]

    if not parsed_lines:
        return _silent_take(heard, discarded, spans, shots_source)

    expected = [normalize_words(line["text"]) for line in parsed_lines]
    assigned, unassigned = align(expected, heard)
    mismatch = mismatch_rule(similarity_threshold)
    by_name = {span["name"]: span for span in spans or ()}

    lines = []
    # The shot of the line each heard word was matched to: a word a line in
    # another shot claimed (one straddling the cut) is that line's speech,
    # not speech in a silent shot
    claimed = {}
    strays = [heard[j] for j, _ in unassigned]
    previous_end = 0.0
    heard_end = None  # end of the last word any earlier line was heard to say
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
        claimed.update((j, shot) for j in indices)
        # A line nothing was heard for, at or after a transcript_loop, was
        # never transcribed: Whisper was stuck, so it is not a 0.0 mismatch
        position = span["start"] if span else previous_end
        looped = not own and any(at <= position for at in loop_starts)
        lines.append(
            {
                "expected": line["text"],
                "heard": None if looped else _joined(own),
                "similarity": None if looped else round(score, 4),
                "start": _round(start),
                "end": _round(end),
                "shot": shot,
            }
        )
        if looped:
            lines[-1]["not_transcribed"] = "transcript_loop"
        elif score < similarity_threshold:
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
            if heard_end is not None:
                findings.extend(
                    _between_lines_findings(
                        mono, sample_rate, index, heard_end, start, shot, strays
                    )
                )
            heard_end = max(heard_end or 0.0, end)
            tails = ([(shot, span["end"])] if span else []) + [(None, duration)]
            findings.extend(
                _clipped_findings(mono, sample_rate, tails, index, own, shot)
            )
            findings.extend(
                _untranscribed_findings(mono, sample_rate, index, own, shot)
            )
            previous_end = end

    findings.extend(_unassigned_tags(heard, unassigned, parsed_lines, expected, spans))

    rules_applied = list(LINE_RULES)
    rules_skipped = [
        {"rule": rule, "reason": "lines were given - speech is expected"}
        for rule in SILENT_RULES
    ]
    no_shot = _no_shot_reason(shots_source, spans, parsed_lines)
    if no_shot is None:
        rules_applied.extend(SHOT_RULES)
        findings.extend(_silent_shot_findings(spans, parsed_lines, heard, claimed))
        findings.extend(
            _unscripted_findings(spans, parsed_lines, expected, heard, unassigned)
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


def _loop_findings(chunks, script_words):
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
                    at = _at(None, seconds=members[0]["start"])
                    at["end"] = _round(members[-1]["end"])
                    found.append(finding(TRANSCRIPT_LOOP, len(members), at))
            run = []
        if index is not None:
            run.append(index)
    return found


def _heard_shot(spans, word):
    """The shot a heard word's midpoint falls in, None when no shots are known."""
    if spans is None:
        return None
    return shot_at(spans, (word["start"] + word["end"]) / 2)


def _silent_take(heard, discarded, spans, shots_source):
    """check's result for `lines: []`: any speech at all is the finding, and
    the line and shot rules have nothing to compare against."""
    findings = []
    if heard:
        findings.append(
            finding(
                SPEECH_WHERE_SILENT,
                len(heard),
                _at(None, heard[0], shot=_heard_shot(spans, heard[0])),
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


def _untranscribed_findings(mono, sample_rate, index, own, shot):
    """speech_not_transcribed: audible sound in a gap the line's words leave
    - between two of them - or inside one word stretched over
    UNHEARD_MIN_SECONDS (#923). The gap before a line's first word is
    _between_lines_findings'."""
    intervals = [(a["end"], b["start"]) for a, b in zip(own, own[1:])]
    findings = []
    for start, end in intervals:
        if end - start >= UNHEARD_MIN_SECONDS:
            findings.append(_untranscribed(mono, sample_rate, index, start, end, shot))
    for word in own:
        # A word is sound itself: only its interior past the margins is a gap
        if (
            word["end"] - word["start"]
            >= UNHEARD_MIN_SECONDS + 2 * UNHEARD_MARGIN_SECONDS
        ):
            findings.append(
                _untranscribed(
                    mono, sample_rate, index, word["start"], word["end"], shot
                )
            )
    return [found for found in findings if found is not None]


def _between_lines_findings(mono, sample_rate, index, start, end, shot, strays):
    """speech_not_transcribed for the stretch between one line's last word and
    the next line's first (#960): audible sound with no word, which the
    in-line check cannot see when the timing leaves it outside both lines.
    Heard-but-unmatched words in the gap are unscripted_speech's, but only the
    stretches they cover: each stretch left between them is judged on its own
    (#964)."""
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
    found = [
        _untranscribed(mono, sample_rate, index, lo, hi, shot)
        for lo, hi in stretches
        if hi - lo >= UNHEARD_MIN_SECONDS
    ]
    return [f for f in found if f is not None]


def _untranscribed(mono, sample_rate, index, start, end, shot):
    lo, hi = start + UNHEARD_MARGIN_SECONDS, end - UNHEARD_MARGIN_SECONDS
    level = word_level(mono, sample_rate, lo, hi)
    if level is None or level <= UNHEARD_FLOOR_DBFS:
        return None
    at = _at(index, seconds=start, shot=shot)
    at["end"] = _round(end)
    return finding(SPEECH_NOT_TRANSCRIBED, _round(end - start, 2), at)


def _clipped_findings(mono, sample_rate, tails, index, own, shot):
    """line_clipped_at_end for one heard line, as a list of zero or one
    finding: its last word ends in a voiced tail of its shot or the file. The
    tail is judged against the whole line's loudest window, not the last
    word's: under a score the ASR stretches the last word over the bed, and a
    bed measured against itself passes the guard (#959)."""
    end = own[-1]["end"]
    clipped = _clipped_tail(
        mono, sample_rate, end, tails, (own[0]["start"], own[-1]["end"])
    )
    if clipped is None:
        return []
    at = {"line": index, "seconds": _round(end), "word": own[-1]["text"]}
    if clipped[0] is not None or shot is not None:
        at["shot"] = clipped[0] or shot
    return [finding(CLIPPED_AT_END, _round(clipped[1], 2), at)]


def _unassigned_tags(heard, unassigned, parsed_lines, expected, spans):
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
                        _at(index, word, shot=_heard_shot(spans, word)),
                    )
                )
                break
    return findings


def _silent_shot_findings(spans, parsed_lines, heard, claimed):
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
            if _heard_shot(spans, word) == span["name"]
            and claimed.get(j) in (None, span["name"])
        ]
        if inside:
            at = _at(None, inside[0], shot=span["name"])
            # A word straddling the cut into this shot is placed at the cut
            at["seconds"] = max(at["seconds"], _round(span["start"]))
            findings.append(finding(SPEECH_IN_SILENT_SHOT, len(inside), at))
    return findings


def _unscripted_findings(spans, parsed_lines, expected, heard, unassigned):
    """unscripted_speech: heard words align left to no line that sit inside a
    shot a line names. `repeats_line` is set when the shot's words, in order,
    contain one of its lines' words again."""
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
            if _heard_shot(spans, heard[j]) == span["name"]
        ]
        if not inside:
            continue
        at = _at(None, inside[0], shot=span["name"])
        words = [word["word"] for word in inside]
        at["end"] = _round(inside[-1]["end"])
        at["repeats_line"] = any(
            line and _contains(words, line) for line in lines
        ) or any(line and _contains(line, words) for line in lines)
        findings.append(finding(UNSCRIPTED_SPEECH, len(inside), at))
    return findings


def _contains(words, part):
    """Whether `part` occurs in `words` as a contiguous run."""
    size = len(part)
    return any(words[i : i + size] == part for i in range(len(words) - size + 1))


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


def per_shot_chunks(transcribe, waveform, rate, spans):
    """Word chunks from transcribing each shot's span on its own, timestamps
    moved back onto the take's clock. `transcribe(leveled, rate)` is
    transcribe_audio's chunks for one slice; each is levelled alone."""
    chunks = []
    for span in spans:
        lo, hi = int(round(span["start"] * rate)), int(round(span["end"] * rate))
        if hi <= lo:
            continue
        piece, _gain = level_for_transcription(waveform[..., lo:hi])
        for chunk in transcribe(piece, rate):
            chunks.append(
                dict(
                    chunk,
                    start=chunk["start"] + span["start"],
                    end=chunk["end"] + span["start"],
                )
            )
    return chunks


def check_script(
    audio,
    lines,
    similarity=DEFAULT_SIMILARITY,
    model_name=WHISPER_DEFAULT_MODEL,
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
            {text, shot}, 'shot' naming the take's shot the line belongs in -
            its full recorded name, or the short '@name' of exactly one shot
            ('cold_open' for 'shot@cold_open'). A name the shot map holds
            more than once (a cut joined from episodes sharing shot names)
            needs 'start_frame' on the line to pick the occurrence:
            {text, shot, start_frame}; findings and 'shots' then call it
            'name#start_frame'.
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
        line_clipped_at_end, speech_not_transcribed, speech_where_silent, speech_in_silent_shot,
        unscripted_speech),
        'lines' (expected, heard, similarity, start, end, shot), 'shots'
        (name, start, end in seconds, or null), 'shots_source', 'discarded' (guarded words with their
        level and reason), 'unmatched' (heard words aligned to no line), 'transcript',
        'model_name', 'rules_applied', 'rules_skipped' and the 'thresholds'
        used.
    """
    check_arguments(COMMAND, similarity=similarity)
    parsed_lines = parse_lines(lines)
    parsed_shots = parse_shots(shots)
    path = audio if isinstance(audio, str) and not is_http_url(audio) else None
    records, shots_source = resolve_shots(
        path, SimpleNamespace(shots=getattr(audio, "shots", None)), parsed_shots
    )
    if records:
        refusal = shot_names_error(parsed_lines, records, shots_source)
        if refusal:
            raise ValueError(refusal)
        parsed_lines = with_full_shot_names(parsed_lines, records)
        records = disambiguated(records)
    waveform, rate = waveform_and_rate(audio, sample_rate, COMMAND)

    from .audio_transcription import transcribe_audio

    leveled, gain = level_for_transcription(waveform)
    emit_log(
        f"check_script: transcribing at {20 * math.log10(gain):+.1f} dB gain",
        transcription_gain_db=round(20 * math.log10(gain), 2),
    )
    transcript = transcribe_audio(
        leveled,
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
    chunks, whole = transcript["chunks"], True
    script_words = {w for line in parsed_lines for w in normalize_words(line["text"])}
    if spans and parsed_lines and _loop_findings(chunks, script_words):
        # Whisper looped on the whole cut and never transcribed what follows
        # (#961): the shots are known, so transcribe each on its own (#963)
        emit_log("check_script: transcript_loop on the cut; transcribing per shot")
        chunks, whole = (
            per_shot_chunks(
                lambda piece, sr: transcribe_audio(
                    piece,
                    device=device,
                    sample_rate=sr,
                    model_name=model_name,
                    timestamps="word",
                )["chunks"],
                waveform,
                rate,
                spans,
            ),
            False,
        )
    answer = check(
        parsed_lines,
        chunks,
        mono,
        rate,
        float(similarity),
        spans=spans,
        shots_source=shots_source,
    )
    answer["transcription_pass"] = "whole_cut" if whole else "per_shot"
    if not whole:
        answer["transcription_note"] = (
            "the whole-cut transcript looped, so each shot was transcribed on"
            " its own and 'lines' and 'findings' come from that pass"
        )
        for line in answer["lines"]:
            line["pass"] = "per_shot"
    answer["transcript"] = transcript["text"]
    answer["model_name"] = model_name
    answer["thresholds"] = dict(THRESHOLDS, similarity=float(similarity))
    return answer
