"""analyze_phrases: when a song's sung lines start and stop, without Whisper (#1095).

`plan_cuts` takes all of its line timing from a transcript, and when Whisper
cannot hear the words - a non-English verse, a hallucination, wordless
vocals - there is nothing else to time the lines by. This finds the phrases
from voiced energy alone: no model, no download, seconds of runtime.

On a stereo file it weights the centre, where a lead vocal is mixed: mid =
(L+R)/2 and side = (L-R)/2, both bandpassed to the voice band, and a frame's
power is max(0, P_mid - P_side), so what is panned away from the centre
cancels. A mono file, a stereo file whose channels are the same, or
`centre=false` uses the band energy of the downmix instead (`method:
"band"`). A frame is voiced above a floor relative to the track - the rule
`voice_attribution.voiced_floor_dbfs` applies to a stem - and the voiced runs
become phrases once short gaps are bridged and short runs dropped.

On a dense mix the centred bass and kick sit partly in the band too, and the
phrases fuse; the task warns, and the documented path feeds it
`separate_stems`' `vocals` stem. The result is a dict, so a step saves it as
one JSON artifact.
"""

import logging

import numpy

from .registry import register_command
from ..task_domains import NON_NEGATIVE, NON_POSITIVE, POSITIVE
from ..task_problems import phrases_errors, phrases_problems

logger = logging.getLogger("dw")

COMMAND = "analyze_phrases"

# The voice band both channels are filtered to, in Hz
BAND_LOW_HZ = 250.0
BAND_HIGH_HZ = 3500.0
# A Butterworth biquad's Q, run twice at each edge for a steeper skirt
BAND_Q = 0.7071
# The analysis frame, and the centred moving average that smooths its power
FRAME_SECONDS = 0.02
SMOOTH_FRAMES = 3
# A side channel this far under the mid (in power) is no side at all: the
# channels are the same, and the centre weighting has nothing to cancel
ALL_CENTRE_DB = -60.0
# A track whose loudest frame is under this, in the band, is near-silence -
# room tone or hiss that a relative floor would still call voiced. A given
# `floor_db` is the caller's own floor, and skips this
NEAR_SILENT_DBFS = -50.0
# One phrase covering more than this share of the track is a dense mix, fused
ONE_PHRASE_SHARE = 0.9
# Seconds as reported
TIME_PLACES = 2


def _band(channel, rate):
    """The channel bandpassed to BAND_LOW_HZ..BAND_HIGH_HZ: a highpass and a
    lowpass biquad, each run twice. The lowpass edge is held under Nyquist."""
    from .. import dsp

    high_b, high_a = dsp.biquad_coefficients("highpass", BAND_LOW_HZ, BAND_Q, rate)
    low_edge = min(BAND_HIGH_HZ, 0.45 * rate)
    low_b, low_a = dsp.biquad_coefficients("lowpass", low_edge, BAND_Q, rate)
    for b, a in ((high_b, high_a), (high_b, high_a), (low_b, low_a), (low_b, low_a)):
        channel = dsp.apply_biquad(channel, b, a)
    return channel


def _frame_power(channel, frame):
    """The mean power of each whole `frame`-sample frame of a channel."""
    count = len(channel) // frame
    frames = channel[: count * frame].reshape(count, frame)
    return numpy.mean(frames * frames, axis=1)


def _smooth(power):
    """A centred SMOOTH_FRAMES moving average, the same length as `power`."""
    if len(power) < SMOOTH_FRAMES:
        return power
    kernel = numpy.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    return numpy.convolve(power, kernel, mode="same")


def voiced_power(wave, rate, centre=True):
    """(per-frame power, frame length in samples, method, side_free) for a
    (channels, samples) waveform: the centre-weighted power of a stereo
    file, or the band power of the downmix when the file is mono, when
    `centre` is off, or when its side channel is silent. `side_free` says the
    centre weighting was asked for on stereo but had no side to cancel."""
    from .. import dsp

    frame = max(1, int(round(FRAME_SECONDS * rate)))
    wave = numpy.asarray(wave, dtype=numpy.float64)
    if centre and wave.shape[0] >= 2:
        left, right = wave[0], wave[1]
        mid = _frame_power(_band((left + right) / 2.0, rate), frame)
        side = _frame_power(_band((left - right) / 2.0, rate), frame)
        mid_total, side_total = float(mid.sum()), float(side.sum())
        if mid_total > 0 and side_total > mid_total * 10 ** (ALL_CENTRE_DB / 10):
            power = numpy.maximum(0.0, mid - side)
            return _smooth(power), frame, "centre", False
        # the channels are the same: the mid is the band energy already
        return _smooth(mid), frame, "band", True
    power = _frame_power(_band(dsp.mono_float64(wave), rate), frame)
    return _smooth(power), frame, "band", False


def voiced_runs(voiced, min_gap_frames, min_phrase_frames):
    """[(first frame, end frame)] of the voiced runs in a boolean array, with
    each gap shorter than `min_gap_frames` bridged and each run then shorter
    than `min_phrase_frames` dropped. The end frame is exclusive."""
    runs = []
    index, count = 0, len(voiced)
    while index < count:
        if not voiced[index]:
            index += 1
            continue
        start = index
        while index < count and voiced[index]:
            index += 1
        if runs and start - runs[-1][1] < min_gap_frames:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((start, index))
    return [(start, end) for start, end in runs if end - start >= min_phrase_frames]


def analyze_phrases(
    audio,
    min_gap_seconds=0.35,
    min_phrase_seconds=0.3,
    floor_db=None,
    centre=True,
    sample_rate=None,
):
    """Find when a song's sung phrases start and stop, from voiced energy.

    Args:
        audio: The song or its vocal stem - a file, a generated audio result
            (`previous_result:<separate_stems step>.vocals`), or a video
            carrying a soundtrack. A vocal stem gives the cleanest phrases;
            a dense mix can fuse into one
        min_gap_seconds: A silence shorter than this between two voiced runs
            is bridged, so they are one phrase. Raise it to merge phrases
            split by breaths, lower it to split lines sung close together
        min_phrase_seconds: A phrase shorter than this, after bridging, is
            dropped as a click or breath
        floor_db: The level, in dBFS of a 20 ms frame, a frame must reach to
            count as voiced. Unset, it is the track's 95th-percentile frame
            less 35 dB, never below -60 dBFS
        centre: Weight a stereo file to its centre, where a lead vocal is
            mixed, cancelling what is panned to the sides. False uses the
            band energy of the downmix
        sample_rate: The rate of a raw waveform; a file or result carries
            its own

    Returns:
        {phrases: [{start, end}], duration_seconds, method, warnings}: the
        phrases in time order, in seconds rounded to 0.01. `method` is
        "centre" for the centre-weighted power of a stereo file, "band" for
        the band energy of a mono one, of `centre=false`, or of stereo
        channels that are the same
    """
    from .. import dsp
    from ..events import emit_warning
    from ..task_domains import check_arguments, real_number
    from .audio_utils import waveform_and_rate
    from .voice_attribution import voiced_floor_dbfs

    if audio is None:
        raise ValueError(f"{COMMAND} needs 'audio' - the song or its vocal stem")
    problems = phrases_problems(centre=centre)
    if problems:
        raise ValueError(problems[0][1])
    values = {
        name: real_number(value, name, COMMAND, required=required)
        for name, value, required in (
            ("min_gap_seconds", min_gap_seconds, True),
            ("min_phrase_seconds", min_phrase_seconds, True),
            ("floor_db", floor_db, False),
            ("sample_rate", sample_rate, False),
        )
    }
    # the registration's domains, again for a direct call
    check_arguments(COMMAND, **values)

    wave, rate = waveform_and_rate(audio, values["sample_rate"], COMMAND)
    wave = numpy.atleast_2d(numpy.asarray(wave))
    duration = wave.shape[-1] / rate
    warnings = []
    if wave.shape[0] == 1:
        warnings.append(
            f"{COMMAND}: the audio is mono, so there is no centre to weight - "
            'the phrases come from its band energy (method "band")'
        )

    power, frame, method, _ = voiced_power(wave, rate, centre=bool(centre))
    level = numpy.sqrt(power)
    loudest = dsp.dbfs(float(level.max()), floor=-numpy.inf) if len(level) else None
    near_silent = values["floor_db"] is None and (
        loudest is None or loudest < NEAR_SILENT_DBFS
    )
    phrases = []
    if not near_silent:
        floor = values["floor_db"]
        if floor is None:
            floor = voiced_floor_dbfs(level)
        voiced = level >= 10 ** (floor / 20)
        frame_seconds = frame / rate
        runs = voiced_runs(
            voiced,
            int(numpy.ceil(values["min_gap_seconds"] / frame_seconds - 1e-9)),
            int(numpy.ceil(values["min_phrase_seconds"] / frame_seconds - 1e-9)),
        )
        phrases = [
            {
                "start": round(start * frame_seconds, TIME_PLACES),
                "end": round(min(end * frame_seconds, duration), TIME_PLACES),
            }
            for start, end in runs
        ]

    if not phrases:
        reason = (
            f"the audio is near-silent (its loudest frame is under "
            f"{NEAR_SILENT_DBFS:g} dBFS in the voice band)"
            if near_silent
            else "no frame rose above the voiced floor for long enough"
        )
        if method == "centre":
            reason += (
                ", or all of it is panned away from the centre - "
                "'centre': false listens to the whole downmix"
            )
        warnings.append(
            f"{COMMAND}: no phrases found - {reason}. Check the audio, or set "
            "'floor_db' lower"
        )
    elif (
        len(phrases) == 1
        and duration > 0
        and phrases[0]["end"] - phrases[0]["start"] > ONE_PHRASE_SHARE * duration
    ):
        warnings.append(
            f"{COMMAND}: one phrase covers more than {ONE_PHRASE_SHARE:.0%} of "
            "the track - probably a dense mix whose instruments fill the voice "
            "band. Feed it the vocal stem (separate_stems, then "
            "previous_result:<step>.vocals)"
        )

    for message in warnings:
        emit_warning(message, kind=COMMAND, command=COMMAND)
    return {
        "phrases": phrases,
        "duration_seconds": round(duration, TIME_PLACES),
        "method": method,
        "warnings": warnings,
    }


@register_command(
    COMMAND,
    implementation="dw.tasks.phrases.analyze_phrases",
    returns="json",
    domains={
        "min_gap_seconds": NON_NEGATIVE,
        "min_phrase_seconds": POSITIVE,
        "floor_db": NON_POSITIVE,
        "sample_rate": POSITIVE,
    },
    whole_numbers=("sample_rate",),
    static_check=phrases_errors,
)
def _handle_analyze_phrases(task, arguments, previous_pipelines):
    """Find when a song's sung phrases start and stop, from voiced energy"""
    logger.debug("Analyzing phrases")
    return analyze_phrases(**arguments)
