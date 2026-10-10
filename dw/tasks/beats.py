"""analyze_beats: where a song's beats fall, for cutting picture to it (#600).

A music video cuts on the beat, and a generated song (Music 3) carries no
tempo map. This reads one off the audio: an onset envelope, a tempo from its
autocorrelation, and the beats by dynamic programming (`dw/dsp.py`). A track
with no attacks to find - a pad, a swell, silence - falls back to the peaks
of its loudness, and says so.

The caller can correct it with anchors, marks where a beat is known to be:
- two or more pull the detected beats through them piecewise-linearly, so a
  detection that starts late or drifts lands on the marks;
- one alone shifts every beat by the same amount;
- one with `tempo_bpm` skips detection and lays an exact grid through it.

It answers JSON and builds nothing: `beats` are the seconds a cut lands on.
"""

import logging
import math
import types

from .registry import register_command
from ..task_domains import POSITIVE
from ..task_problems import beats_errors

logger = logging.getLogger("dw")

COMMAND = "analyze_beats"

# A track whose loudest 50 ms is under this has nothing to track
SILENT_DBFS = -60.0
# ...and one under this is near-silence - room tone, hiss - whose onset
# envelope, log-compressed, can still look like a pulse
QUIET_DBFS = -40.0
# Seconds and BPM as reported
TIME_PLACES = 4
BPM_PLACES = 2


def _parse_anchors(anchors, duration):
    """[(beat index or None, seconds)] for the anchors, refused when one is
    past the song's end."""
    parsed = []
    for index, anchor in enumerate(anchors or ()):
        if isinstance(anchor, dict):
            beat, seconds = int(float(anchor["beat_index"])), float(anchor["seconds"])
        else:
            beat, seconds = None, float(anchor)
        if seconds > duration:
            raise ValueError(
                f"{COMMAND}: anchors[{index}] ({seconds:g} s) is past the song's "
                f"end ({duration:.3f} s)"
            )
        parsed.append((beat, seconds))
    return parsed


def _coerce_arguments(sample_rate, tempo_bpm, min_bpm, max_bpm, anchors):
    from ..task_domains import (
        check_arguments,
        real_number,
        whole_number,
    )
    from ..task_problems import beats_problems

    values = {
        "sample_rate": whole_number(sample_rate, "sample_rate", COMMAND),
        **{
            name: real_number(value, name, COMMAND)
            for name, value in (
                ("tempo_bpm", tempo_bpm),
                ("min_bpm", min_bpm),
                ("max_bpm", max_bpm),
            )
        },
    }
    check_arguments(COMMAND, **values)
    problems = beats_problems(values["min_bpm"], values["max_bpm"], anchors)
    if problems:
        raise ValueError("; ".join(message for _, message in problems))
    return types.SimpleNamespace(**values)


def _grid(anchor, tempo_bpm, duration):
    """An exact grid at tempo_bpm through one anchor, within the song.

    A bare time is a beat the grid passes through, so it runs both ways from
    it; a {beat_index, seconds} mark places beat 0 at seconds - index periods,
    and the grid starts there.
    """
    beat, seconds = anchor
    period = 60.0 / tempo_bpm
    if beat is None:
        start = seconds - math.floor(seconds / period) * period
    else:
        start = seconds - beat * period
        if start < -1e-9:
            raise ValueError(
                f"{COMMAND}: beat {beat} at {seconds:g} s puts beat 0 at "
                f"{start:.3f} s, before the song starts, at {tempo_bpm:g} BPM"
            )
        start = max(start, 0.0)
    count = int(math.floor((duration - start) / period - 1e-9)) + 1
    return [start + k * period for k in range(max(count, 0))]


def _calibrate(beats, anchors, warnings):
    """(warped beats, calibration) pulling the detected beats onto anchors."""
    import numpy

    from .. import dsp

    calibration = {"offset_s": 0.0, "drift": 0.0, "anchors_used": 0}
    if not anchors:
        return beats, calibration
    if not len(beats):
        warnings.append(
            f"{COMMAND}: no beats were found, so the {len(anchors)} anchor(s) "
            "were not used"
        )
        return beats, calibration
    knots_from, knots_to, used = [], [], set()
    for beat, seconds in anchors:
        if beat is None:
            beat = int(numpy.argmin(numpy.abs(beats - seconds)))
        elif beat >= len(beats):
            raise ValueError(
                f"{COMMAND}: an anchor names beat {beat}, and only {len(beats)} "
                "were found (indexes count from 0)"
            )
        if beat in used:
            warnings.append(
                f"{COMMAND}: the anchor at {seconds:g} s is nearest the same "
                f"beat ({beats[beat]:.3f} s) as an earlier anchor and was not used"
            )
            continue
        used.add(beat)
        knots_from.append(beats[beat])
        knots_to.append(seconds)
    warped = dsp.warp_times(beats, knots_from, knots_to)
    calibration = {
        "offset_s": round(float(knots_to[0] - knots_from[0]), TIME_PLACES),
        "drift": round(
            float((knots_to[-1] - knots_from[-1]) - (knots_to[0] - knots_from[0])),
            TIME_PLACES,
        ),
        "anchors_used": len(knots_from),
    }
    return warped, calibration


def _detect(mono, rate, args, warnings):
    """(beat seconds, bpm, downbeat phase, method) detected from the audio."""
    import numpy

    from .. import dsp

    peaks, loudest = dsp.rms_peaks(mono, rate, args.max_bpm)
    # The onset envelope is log-compressed, so it finds the same pulse in a
    # song at -120 dB as at full scale: the level is checked first
    if loudest < SILENT_DBFS:
        warnings.append(
            f"{COMMAND}: the track is silent (loudest {loudest:.1f} dBFS) - "
            "no beats to find"
        )
        return numpy.zeros(0), None, None, "rms_peaks"
    if loudest < QUIET_DBFS:
        warnings.append(
            f"{COMMAND}: the track is near-silent (loudest {loudest:.1f} dBFS, "
            f"under {QUIET_DBFS:g}) - room tone or noise, no pulse to track; "
            "raise its level first (gain_audio) if it is a real song"
        )
        return numpy.zeros(0), None, None, "rms_peaks"

    envelope, frame_rate = dsp.onset_envelope(mono, rate)
    bpm, periodicity, missed = dsp.estimate_tempo(
        envelope, frame_rate, args.min_bpm, args.max_bpm, hint_bpm=args.tempo_bpm
    )
    if bpm is not None and dsp.is_trackable(envelope, periodicity):
        frames = dsp.track_beats(envelope, frame_rate, bpm)
        if frames.shape[0] >= 2:
            if not dsp.is_confident(envelope, frames, periodicity):
                # Faint, not absent: a song with a long quiet build keeps its
                # beats low against its own loud end, and a noise bed's
                # bumps can repeat about this well - the caller is told, and
                # keeps the beats (a wrong grid is checked against the song;
                # a missing one cannot be)
                warnings.append(
                    f"{COMMAND}: the pulse is faint (periodicity "
                    f"{periodicity:.2f}) - these beats may follow the track's "
                    "texture rather than its beat; check a few against the "
                    "song, and pass anchors or tempo_bpm if they are off"
                )
            if missed is not None:
                warnings.append(
                    f"{COMMAND}: the track's pulse is {missed:.1f} BPM, and no "
                    f"octave of it is within {args.min_bpm:g}-{args.max_bpm:g} "
                    f"BPM - these beats keep to {bpm:.1f} BPM, not the pulse"
                )
            return (
                frames / frame_rate,
                bpm,
                dsp.downbeat_phase(envelope, frames),
                "onset",
            )
    if peaks.shape[0] < 2:
        warnings.append(
            f"{COMMAND}: the track has no clear onsets and too few loudness "
            "peaks to place a beat"
        )
        return numpy.zeros(0), None, None, "rms_peaks"
    warnings.append(
        f"{COMMAND}: no reliable beat was found - the track has no clear, "
        "regular onsets to track; these beats are its loudness peaks, which "
        "follow swells rather than a pulse"
    )
    rate_bpm = 60.0 / float(numpy.median(numpy.diff(peaks)))
    folded = dsp.fold_bpm(rate_bpm, args.min_bpm, args.max_bpm)
    if folded is None:
        warnings.append(
            f"{COMMAND}: the loudness peaks come at {rate_bpm:.1f} BPM, and no "
            f"octave of that is within {args.min_bpm:g}-{args.max_bpm:g} BPM, "
            "so no bpm is reported"
        )
    return peaks, folded, None, "rms_peaks"


def analyze_beats(
    audio, sample_rate=None, tempo_bpm=None, anchors=None, min_bpm=60, max_bpm=200
):
    """Find a song's tempo and the seconds its beats fall on.

    Args:
        audio: The song - a file, a generated audio result, or a video
            carrying a soundtrack
        sample_rate: The rate of a raw waveform; a file or result carries
            its own
        tempo_bpm: The tempo, when known. With exactly one anchor it lays an
            exact grid and nothing is detected; otherwise it steers the
            detection toward that tempo
        anchors: Where beats are known to fall - a list of times in seconds,
            or of {beat_index, seconds} marks placing a detected beat (from
            0) at a time. Two or more warp the detected beats through them;
            one shifts them all; one with tempo_bpm lays a grid
        min_bpm: The slowest tempo searched for
        max_bpm: The fastest tempo searched for, above min_bpm

    Returns:
        {bpm, beats, downbeat_phase, method, calibration, duration_seconds,
        warnings}: `beats` ascend, in seconds; `method` is "onset",
        "rms_peaks" (no clear onsets - loudness peaks instead) or "grid";
        `downbeat_phase` is which of the first four beats starts a bar, or
        None when no beat stands out; `calibration` is the anchors' offset
        and drift in seconds and how many were used
    """
    import numpy

    from .. import dsp
    from ..events import emit_warning
    from .audio_utils import waveform_and_rate

    args = _coerce_arguments(sample_rate, tempo_bpm, min_bpm, max_bpm, anchors)
    waveform, rate = waveform_and_rate(audio, args.sample_rate, COMMAND)
    mono = dsp.mono_float64(waveform)
    duration = mono.shape[0] / rate
    parsed = _parse_anchors(anchors, duration)
    warnings = []

    if len(parsed) == 1 and args.tempo_bpm is not None:
        beats = numpy.array(_grid(parsed[0], args.tempo_bpm, duration))
        bpm, phase, method = args.tempo_bpm, None, "grid"
        calibration = {"offset_s": 0.0, "drift": 0.0, "anchors_used": 1}
    else:
        beats, bpm, phase, method = _detect(mono, rate, args, warnings)
        beats, calibration = _calibrate(beats, parsed, warnings)
        if calibration["anchors_used"]:
            # A warp can carry an end beat past the song's edges; the phase
            # counts from the first beat kept
            dropped = int(numpy.count_nonzero(beats < 0))
            beats = beats[(beats >= 0) & (beats <= duration)]
            if phase is not None:
                phase = (phase - dropped) % 4
        if method == "onset" and len(beats) >= 2:
            bpm = 60.0 / float(numpy.median(numpy.diff(beats)))
            if not calibration["anchors_used"]:
                # Frame rounding can carry the median a hair past the range
                bpm = min(max(bpm, args.min_bpm), args.max_bpm)

    for message in warnings:
        emit_warning(message, kind="analyze_beats", command=COMMAND)
    return {
        "bpm": None if bpm is None else round(float(bpm), BPM_PLACES),
        "beats": [round(float(time), TIME_PLACES) for time in beats],
        "downbeat_phase": phase,
        "method": method,
        "calibration": calibration,
        "duration_seconds": round(duration, TIME_PLACES),
        "warnings": warnings,
    }


@register_command(
    COMMAND,
    implementation="dw.tasks.beats.analyze_beats",
    returns="json",
    domains={
        "sample_rate": POSITIVE,
        "tempo_bpm": POSITIVE,
        "min_bpm": POSITIVE,
        "max_bpm": POSITIVE,
    },
    whole_numbers=("sample_rate",),
    static_check=beats_errors,
)
def _handle_analyze_beats(task, arguments, previous_pipelines):
    """Find a song's tempo and the seconds its beats fall on"""
    logger.debug("Analyzing beats")
    return analyze_beats(**arguments)
