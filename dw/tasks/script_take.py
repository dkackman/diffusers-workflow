"""The take side of `check_script` (`dw/tasks/script_check.py`): what the
transcriber hears, which heard words are kept, and where each sits.

`script_check` reads this module; nothing here reads `script_check`, so the
rule tables and findings stay on the other side of the seam. It holds:

- the levelling the transcriber hears the take at (`level_for_transcription`,
  `per_shot_chunks`) - the level checks keep reading the original waveform;
- the shot placement the probes' rule gives (`shot_spans` via
  `assess.sample_span`, `overlapping_shot`, `shot_at`);
- the guard (`guard_words`): a heard word at or below the dead-air floor
  (`GUARD_FLOOR_DBFS`, the probes' `DEAD_AIR_FLOOR_DBFS`), one repeating the
  previous chunk's exact span, or one inside a repetition loop
  (`script_lines.looped_texts_firsts`) is discarded, never raised as a
  finding. `word_level` is the measure the guard and the level checks share
  (`dsp.loudest_window_dbfs` over `GUARD_WINDOW_SECONDS`, the window
  `shot_dead_air` measures in).
"""

from types import SimpleNamespace

from .. import dsp
from ..assessment_rules import DEAD_AIR_FLOOR_DBFS
from ..script_lines import looped_texts_firsts, normalize_words
from .assess import DEAD_AIR_WINDOW, sample_span

# A heard word is guarded (discarded as unheard) when its loudest
# GUARD_WINDOW_SECONDS window sits at or below GUARD_FLOOR_DBFS. The floor is
# the assessment probes' dead-air floor and the window is the one
# `shot_dead_air` measures in (`DEAD_AIR_WINDOW`), both read, not copied. The loudest window rather than the whole
# span, because a word timestamp is approximate (+-0.1-0.3 s) and a span
# that overhangs the pause after a word must not average a real word away
GUARD_FLOOR_DBFS = DEAD_AIR_FLOOR_DBFS
GUARD_WINDOW_SECONDS = DEAD_AIR_WINDOW

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


def round_seconds(value, places=3):
    return None if value is None else round(float(value), places)


def guard_words(chunks, mono, sample_rate, script_words=None):
    """Split transcribe_audio's word chunks into (heard, discarded).

    `heard` is [{word, text, start, end}], one per normalized word - a chunk
    that normalizes to nothing ('♪', '...') is neither; `discarded` is
    [{text, start, end, level_dbfs, reason, count}] for a chunk at or below
    the floor ("below_floor"), a chunk repeating the previous one's exact span
    ("repeated_span": words cannot share a span, #971) or a repetition loop
    ("repetition": one entry
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
        before = spoken[index - 1] if index else None
        previous_span = (
            None
            if before is None
            else (before[0]["start"], before[0]["end"], before[1])
        )
        first_of_line = (
            index in firsts
            and script_words is not None
            and set(words) <= script_words
            and end > start
        )
        if (start, end, words) == previous_span:
            reason = "repeated_span"
        elif index in looped and not first_of_line:
            reason = "repetition"
        elif level is None or level <= GUARD_FLOOR_DBFS:
            reason = "below_floor"
        else:
            reason = None
        if (
            reason in ("repetition", "repeated_span")
            and run_end == index - 1
            and discarded[-1]["reason"] == reason
        ):
            # A loop is one entry: its span, its chunk count, its loudest level
            last = discarded[-1]
            last["end"] = round_seconds(max(last["end"], end))
            last["count"] += 1
            if level is not None and (
                last["level_dbfs"] is None or level > last["level_dbfs"]
            ):
                last["level_dbfs"] = round_seconds(level, 2)
            run_end = index
            continue
        if reason:
            discarded.append(
                {
                    "text": chunk.get("text", "").strip(),
                    "start": round_seconds(start),
                    "end": round_seconds(end),
                    "level_dbfs": round_seconds(level, 2),
                    "reason": reason,
                    "count": 1,
                }
            )
            run_end = index if reason in ("repetition", "repeated_span") else None
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
