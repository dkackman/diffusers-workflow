"""measure_sync: how far a shot's audio sits from its source slice (#802).

`analyze_sync_drift` counts samples against frames, so it cannot see audio
whose samples line up but whose content is shifted - a lip-sync model that
re-sings the reference late. This cross-correlates the onset envelopes of the
audio and of the slice it should match, and answers the lag and how well the
two agree. One call measures one shot; it answers JSON and builds nothing.

The lag is positive when the audio is LATE: its events fall that many seconds
after the same events in the reference. A negative lag is early.
"""

import logging

from .beats import SILENT_DBFS
from .registry import register_command
from ..assessment_rules import finding
from ..task_domains import NON_NEGATIVE, POSITIVE, UNIT

logger = logging.getLogger("dw")

COMMAND = "measure_sync"

# A lag past this many seconds is a finding
LAG_THRESHOLD_SECONDS = 0.1
# A correlation under this is a match too weak to trust the lag
MIN_CONFIDENCE = 0.3
# Seconds and correlation as reported
TIME_PLACES = 4
CONFIDENCE_PLACES = 3


def _rule(name, threshold, says):
    """A rule for `assessment_rules.finding()`; not in `RULES`, as this is not a probe"""
    return {"name": name, "severity": "warn", "threshold": threshold, "says": says}


def measure_sync(
    audio,
    reference,
    max_lag_seconds=2.0,
    threshold_seconds=LAG_THRESHOLD_SECONDS,
    min_confidence=MIN_CONFIDENCE,
    sample_rate=None,
):
    """Measure how late (or early) a shot's audio is against its source slice.

    Args:
        audio: The shot's audio - a file, a generated audio result, or a
            video carrying a soundtrack
        reference: The slice of the source it should match, in the same forms
        max_lag_seconds: The largest lag searched, either way. A lag found at
            this edge may be larger still
        threshold_seconds: A lag longer than this, either way, is a finding
        min_confidence: A correlation under this is reported as unreliable
        sample_rate: The rate of a raw waveform; a file or result carries its
            own. Applies to both inputs

    Returns:
        {lag_seconds, confidence, max_lag_seconds, audio_seconds,
        reference_seconds, findings, warnings}: `lag_seconds` is positive when
        the audio is late against the reference, negative when early, and
        None when either track is silent or flat (nothing to line up).
        `confidence` is the correlation at that lag, 0 to 1. `findings` are
        {rule, severity, at, value, threshold, says}
    """
    from .. import dsp
    from ..events import emit_warning
    from ..task_domains import real_number
    from .audio_utils import waveform_and_rate

    if audio is None:
        raise ValueError(f"{COMMAND} needs 'audio' - the shot's audio")
    if reference is None:
        raise ValueError(
            f"{COMMAND} needs 'reference' - the source slice to compare against; "
            "a shot with no slice has nothing to measure"
        )
    values = {
        name: real_number(value, name, COMMAND)
        for name, value in (
            ("max_lag_seconds", max_lag_seconds),
            ("threshold_seconds", threshold_seconds),
            ("min_confidence", min_confidence),
            ("sample_rate", sample_rate),
        )
    }
    if values["max_lag_seconds"] is None or values["max_lag_seconds"] <= 0:
        raise ValueError(f"{COMMAND}: max_lag_seconds must be above 0")

    wave, rate = waveform_and_rate(audio, values["sample_rate"], COMMAND)
    ref_wave, ref_rate = waveform_and_rate(reference, values["sample_rate"], COMMAND)
    if ref_rate != rate:
        ref_wave = dsp.resample_waveform(ref_wave, ref_rate, rate)
    mono, ref_mono = dsp.mono_float64(wave), dsp.mono_float64(ref_wave)

    warnings, findings = [], []
    answer = {
        "lag_seconds": None,
        "confidence": 0.0,
        "max_lag_seconds": values["max_lag_seconds"],
        "audio_seconds": round(mono.shape[0] / rate, TIME_PLACES),
        "reference_seconds": round(ref_mono.shape[0] / rate, TIME_PLACES),
    }

    silent = [
        name
        for name, track in (("audio", wave), ("reference", ref_wave))
        if (dsp.level_dbfs(track) or float("-inf")) < SILENT_DBFS
    ]
    if silent:
        message = (
            f"{COMMAND}: the {' and '.join(silent)} is silent (under "
            f"{SILENT_DBFS:g} dBFS) - there is nothing to line up"
        )
        warnings.append(message)
        findings.append(
            finding(
                _rule("sync_unmeasurable", SILENT_DBFS, message),
                None,
                {"silent": silent},
            )
        )
    else:
        envelope, frame_rate = dsp.onset_envelope(mono, rate)
        ref_envelope, _ = dsp.onset_envelope(ref_mono, rate)
        lag, confidence = dsp.envelope_lag(
            envelope, ref_envelope, frame_rate, values["max_lag_seconds"]
        )
        if lag is None:
            message = (
                f"{COMMAND}: an envelope is flat or too short to line up - no "
                "lag was measured"
            )
            warnings.append(message)
            findings.append(
                finding(_rule("sync_unmeasurable", None, message), None, {})
            )
        else:
            answer["lag_seconds"] = round(float(lag), TIME_PLACES)
            answer["confidence"] = round(float(confidence), CONFIDENCE_PLACES)
            if confidence < values["min_confidence"]:
                findings.append(
                    finding(
                        _rule(
                            "sync_low_confidence",
                            values["min_confidence"],
                            "the audio and reference envelopes barely "
                            "correlate, so the lag is not reliable - they may "
                            "not be the same material",
                        ),
                        answer["confidence"],
                        {},
                    )
                )
            elif abs(lag) > values["threshold_seconds"]:
                findings.append(
                    finding(
                        _rule(
                            "audio_out_of_sync",
                            values["threshold_seconds"],
                            f"the audio is {abs(lag):.3f} s "
                            f"{'late' if lag > 0 else 'early'} against its "
                            "source slice",
                        ),
                        answer["lag_seconds"],
                        {},
                    )
                )
            if abs(lag) >= values["max_lag_seconds"] - 1.0 / frame_rate:
                message = (
                    f"{COMMAND}: the lag sits at the edge of the search "
                    f"(±{values['max_lag_seconds']:g} s) - the true lag may be "
                    "larger; raise max_lag_seconds"
                )
                warnings.append(message)

    for message in warnings:
        emit_warning(message, kind="measure_sync", command=COMMAND)
    return {**answer, "findings": findings, "warnings": warnings}


@register_command(
    COMMAND,
    implementation="dw.tasks.measure_sync.measure_sync",
    returns="json",
    domains={
        "max_lag_seconds": POSITIVE,
        "threshold_seconds": NON_NEGATIVE,
        "min_confidence": UNIT,
        "sample_rate": POSITIVE,
    },
    media_arguments=("reference",),
)
def _handle_measure_sync(task, arguments, previous_pipelines):
    """Measure how far a shot's audio sits from its source slice"""
    logger.debug("Measuring sync")
    return measure_sync(**arguments)
