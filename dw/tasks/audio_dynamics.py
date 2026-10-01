"""Audio dynamics and measurement commands: normalize, compress, filter, analyze.

Each reads a track, changes or measures its level or spectrum, and says what
it did through events. The pure signal processing lives in dw/dsp.py; the
track plumbing (loading, rate handling, the AudioTrack return) is
audio_utils's.
"""

import logging

import numpy

from ..dsp import (
    LIMITER_HEAVY_DB,
    LIMITER_MAX_REDUCTION_DB,
    LIMITER_TOLERANCE_LU,
    UNITY_TOLERANCE,
    apply_biquad,
    biquad_coefficients,
    follow_envelope,
    integrated_lufs,
    level_dbfs,
    limit_at,
    search_gain,
    spectral_balance,
    true_peak_envelope,
)
from ..events import emit_log, emit_warning
from ..task_domains import check_arguments
from .audio_utils import as_track, waveform_and_rate

logger = logging.getLogger("dw")


def normalize_audio(
    audio, peak_dbfs=-1.0, target_lufs=None, limit=False, sample_rate=None
):
    """Task command: scale a track so its loudest sample sits at a level.

    Generated music comes out at whatever level the model happened to land
    on - quiet takes need lifting before they sit under a picture, and a
    hot one needs headroom before the encoder. Peak normalization changes
    nothing but the gain, so the dynamics survive.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        peak_dbfs: The level the loudest sample is moved to, in dB below full
            scale. 0 is full scale; -1 leaves a little headroom. Still
            applies as a ceiling when target_lufs is also given
        target_lufs: Integrated loudness (BS.1770) to gain the track to, in
            LUFS. Peak alone says nothing about how loud a track sounds - a
            sparse voice-over and a dense score can share a peak and still
            sit tens of dB apart to the ear (#361). When given, the gain
            targets this loudness first; peak_dbfs still holds as a ceiling,
            and if reaching target_lufs would cross it the gain stops at the
            ceiling and a warning names the shortfall in LU. None (the
            default) leaves behavior exactly as peak-only
        limit: Hold peak_dbfs with a look-ahead limiter instead of capping the
            gain, so target_lufs can be reached past a transient that sets
            the peak. peak_dbfs becomes a true-peak (4x oversampled, BS.1770)
            ceiling; the limiter itself never adds gain, but the static gain
            applied before it is searched for - not read off target_lufs
            directly - since limiting takes back some of the loudness a
            plain gain would have reached, more on dense material, so the
            reported gain_db can run past target_lufs's own gain. Limiting
            stops at 12 dB of reduction, past which target_lufs is warned as
            capped. False (the default) leaves behavior exactly as without it
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the scaled waveform and its rate; a silent
        track is returned unchanged
    """
    check_arguments("normalize_audio", sample_rate=sample_rate, target_lufs=target_lufs)
    if not isinstance(limit, (bool, numpy.bool_)):
        raise ValueError(
            f"normalize_audio 'limit' must be true or false, got {limit!r}"
        )
    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "normalize_audio")
    if peak_dbfs > 0:
        raise ValueError("normalize_audio 'peak_dbfs' cannot be above full scale (0)")
    peak = float(numpy.abs(waveform).max()) if waveform.size else 0.0
    if peak == 0.0:
        logger.warning("normalize_audio: the track is silent - left unchanged")
        return as_track(waveform, sample_rate, "normalize_audio")
    if limit:
        return as_track(
            _normalize_limited(waveform, sample_rate, peak_dbfs, target_lufs),
            sample_rate,
            "normalize_audio",
        )

    peak_db = 20 * numpy.log10(peak)
    measured_lufs = None
    if target_lufs is None:
        constraint = "peak_dbfs"
        gain_db = peak_dbfs - peak_db
    else:
        ceiling_gain_db = peak_dbfs - peak_db
        current_lufs = integrated_lufs(waveform.T, sample_rate)
        measured_lufs = current_lufs
        if current_lufs is None:
            emit_warning(
                f"normalize_audio: target_lufs={target_lufs} was given, but the "
                "track's loudness could not be measured (shorter than the 400 ms "
                "gating block, or silent throughout) - falling back to peak_dbfs "
                "alone.",
                kind="target_lufs_unmeasurable",
                command="normalize_audio",
                target_lufs=target_lufs,
            )
            gain_db = ceiling_gain_db
            constraint = "peak_ceiling"
        else:
            target_gain_db = target_lufs - current_lufs
            gain_db = min(target_gain_db, ceiling_gain_db)
            constraint = "peak_ceiling" if gain_db < target_gain_db else "target_lufs"
            if gain_db < target_gain_db:
                emit_warning(
                    f"normalize_audio: target_lufs={target_lufs} would need "
                    f"{target_gain_db:+.1f} dB of gain, but peak_dbfs={peak_dbfs} "
                    f"caps the gain at {gain_db:+.1f} dB - lands "
                    f"{target_gain_db - gain_db:.1f} LU below the target.",
                    kind="target_lufs_capped",
                    command="normalize_audio",
                    target_lufs=target_lufs,
                    peak_dbfs=peak_dbfs,
                    shortfall_lu=target_gain_db - gain_db,
                )
    gain = 10 ** (gain_db / 20)
    emit_log(
        f"normalize_audio: measured {peak_db:.1f} dBFS peak"
        + ("" if measured_lufs is None else f", {measured_lufs:.1f} LUFS")
        + f" -> gain {gain_db:+.1f} dB, set by {constraint}",
        command="normalize_audio",
        measured_peak_dbfs=round(peak_db, 1),
        measured_lufs=round(measured_lufs, 1) if measured_lufs is not None else None,
        gain_db=round(gain_db, 1),
        constraint=constraint,
    )
    return as_track(
        (waveform * gain).astype(numpy.float32), sample_rate, "normalize_audio"
    )


def _normalize_limited(waveform, sample_rate, peak_dbfs, target_lufs):
    """normalize_audio(limit=True): a static gain, applied uncapped, with a
    true-peak limiter holding peak_dbfs. The limiter is never a gain stage -
    it only tames what the static gain sends it - but that static gain is
    searched for so the *limited* output lands on target_lufs, since
    limiting takes back some of the loudness a plain target_lufs gain would
    have reached; the search stops at LIMITER_MAX_REDUCTION_DB, past which
    the gain is what stops instead."""
    ceiling = 10 ** (peak_dbfs / 20)
    envelope = true_peak_envelope(waveform)
    input_peak_db = 20 * numpy.log10(float(envelope.max()))
    peak_gain_db = peak_dbfs - input_peak_db
    most_gain_db = peak_gain_db + LIMITER_MAX_REDUCTION_DB

    measured_lufs = None
    target_gain_db = None
    if target_lufs is not None:
        measured_lufs = integrated_lufs(waveform.T, sample_rate)
        if measured_lufs is None:
            emit_warning(
                f"normalize_audio: target_lufs={target_lufs} was given, but the "
                "track's loudness could not be measured (shorter than the 400 ms "
                "gating block, or silent throughout) - falling back to peak_dbfs "
                "alone.",
                kind="target_lufs_unmeasurable",
                command="normalize_audio",
                target_lufs=target_lufs,
            )
        else:
            target_gain_db = target_lufs - measured_lufs

    if target_gain_db is None:
        gain_db = peak_gain_db
        output, curve, trim, output_peak = limit_at(
            waveform, envelope, gain_db, ceiling, sample_rate
        )
        output_lufs = integrated_lufs(output.T, sample_rate)
    else:
        gain_db, (output, curve, trim, output_peak), output_lufs = search_gain(
            waveform,
            envelope,
            ceiling,
            sample_rate,
            target_lufs,
            min(target_gain_db, most_gain_db),
            most_gain_db,
        )

    lowest = (1.0 if curve is None else float(curve.min())) * trim
    max_reduction_db = max(0.0, -20 * numpy.log10(lowest))
    if curve is None:
        limited_fraction = 1.0 if trim < 1.0 else 0.0
    else:
        limited_fraction = float(
            numpy.count_nonzero(curve * trim < 1.0 - UNITY_TOLERANCE) / curve.size
        )
    output_peak_db = 20 * numpy.log10(output_peak)

    shortfall = None
    if target_gain_db is not None:
        if output_lufs is not None:
            shortfall = target_lufs - output_lufs
        elif gain_db < target_gain_db:
            shortfall = target_gain_db - gain_db
    if shortfall is not None and shortfall > LIMITER_TOLERANCE_LU:
        reason = (
            f"would need more than {LIMITER_MAX_REDUCTION_DB:.0f} dB of limiting "
            f"under peak_dbfs={peak_dbfs}, so the gain stops at {gain_db:+.1f} dB"
            if gain_db >= most_gain_db - 1e-9
            else f"was not reached under peak_dbfs={peak_dbfs} at "
            f"{gain_db:+.1f} dB of gain"
        )
        emit_warning(
            f"normalize_audio: target_lufs={target_lufs} {reason} - lands "
            f"{shortfall:.1f} LU below the target.",
            kind="target_lufs_capped",
            command="normalize_audio",
            target_lufs=target_lufs,
            peak_dbfs=peak_dbfs,
            shortfall_lu=round(shortfall, 2),
            limited=True,
        )
    if max_reduction_db > LIMITER_HEAVY_DB:
        emit_warning(
            f"normalize_audio: the limiter reduced the loudest moments by "
            f"{max_reduction_db:.1f} dB to hold peak_dbfs={peak_dbfs} - past "
            f"{LIMITER_HEAVY_DB:.0f} dB, pumping can be audible. A lower "
            "target_lufs asks less of it.",
            kind="limiter_heavy",
            command="normalize_audio",
            max_gain_reduction_db=round(max_reduction_db, 1),
            peak_dbfs=peak_dbfs,
            target_lufs=target_lufs,
        )
    emit_log(
        f"normalize_audio: measured {input_peak_db:.1f} dBTP"
        + ("" if measured_lufs is None else f", {measured_lufs:.1f} LUFS")
        + f" -> gain {gain_db:+.1f} dB, limiter up to {max_reduction_db:.1f} dB "
        f"on {limited_fraction:.1%} of samples -> {output_peak_db:.1f} dBTP"
        + ("" if output_lufs is None else f", {output_lufs:.1f} LUFS"),
        command="normalize_audio",
        measured_true_peak_dbfs=round(input_peak_db, 1),
        measured_lufs=round(measured_lufs, 1) if measured_lufs is not None else None,
        gain_db=round(gain_db, 1),
        constraint="limiter",
        max_gain_reduction_db=round(max_reduction_db, 2),
        limited_fraction=round(limited_fraction, 4),
        output_true_peak_dbfs=round(output_peak_db, 2),
        output_lufs=round(output_lufs, 1) if output_lufs is not None else None,
    )
    return output


COMPRESS_MODES = ("compress", "limit", "gate")

# A floor below which an envelope is treated as digital silence, so its dBFS
# reading is a large negative number rather than -inf
_ENVELOPE_FLOOR_DBFS = -120.0

_ENVELOPE_FLOOR_LINEAR = 10.0 ** (_ENVELOPE_FLOOR_DBFS / 20.0)


def compress_audio(
    audio,
    threshold_dbfs,
    ratio=4.0,
    attack_ms=10.0,
    release_ms=100.0,
    mode="compress",
    sample_rate=None,
):
    """Task command: shape a track's dynamics with an envelope-follower.

    A compressor, a limiter and a gate are the same envelope-follower
    algorithm with different knob settings: a limiter is a ratio pushed
    toward infinity with a fast attack, and a gate is downward expansion
    below the threshold rather than compression above it - so one command
    covers all three through 'mode' rather than three near-duplicate ones.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        threshold_dbfs: The level, in dB below full scale, above which
            'compress'/'limit' reduce gain, or below which 'gate' does
        ratio: How strongly gain is reduced past the threshold. Unused by
            'limit', which reduces enough to hold the signal at the
            threshold regardless
        attack_ms: How fast the envelope follows a rise in level
        release_ms: How fast the envelope follows a fall in level
        mode: 'compress' (downward compression above threshold), 'limit'
            (holds the signal at threshold), or 'gate' (downward expansion
            below threshold)
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the processed waveform and its rate
    """
    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "compress_audio")
    check_arguments(
        "compress_audio",
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        sample_rate=sample_rate,
    )
    if mode not in COMPRESS_MODES:
        raise ValueError(
            f"compress_audio mode must be one of {COMPRESS_MODES}, got {mode!r}"
        )
    if threshold_dbfs > 0:
        raise ValueError(
            "compress_audio 'threshold_dbfs' cannot be above full scale (0)"
        )
    if waveform.size == 0:
        return as_track(waveform, sample_rate, "compress_audio")

    envelope = follow_envelope(waveform, sample_rate, attack_ms, release_ms)
    envelope_dbfs = 20.0 * numpy.log10(numpy.maximum(envelope, _ENVELOPE_FLOOR_LINEAR))

    if mode == "gate":
        past_threshold = numpy.maximum(0.0, threshold_dbfs - envelope_dbfs)
    else:
        past_threshold = numpy.maximum(0.0, envelope_dbfs - threshold_dbfs)

    if mode == "limit":
        reduction_db = past_threshold
    else:
        reduction_db = past_threshold * (1.0 - 1.0 / ratio)

    gain = (10.0 ** (-reduction_db / 20.0)).astype(numpy.float32)
    processed = (waveform * gain[numpy.newaxis, :]).astype(numpy.float32)
    return as_track(processed, sample_rate, "compress_audio")


FILTER_KINDS = ("lowpass", "highpass", "bandpass", "notch")


def filter_audio(audio, cutoff_hz, kind="lowpass", q=0.707, sample_rate=None):
    """Task command: run a track through a single biquad filter stage.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        cutoff_hz: The filter's corner (lowpass/highpass) or center
            (bandpass/notch) frequency
        kind: 'lowpass', 'highpass', 'bandpass', or 'notch'
        q: Resonance/bandwidth of the filter. Higher narrows a bandpass or
            notch, and peaks the corner of a lowpass or highpass
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the filtered waveform and its rate
    """
    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "filter_audio")
    check_arguments("filter_audio", cutoff_hz=cutoff_hz, sample_rate=sample_rate)
    if kind not in FILTER_KINDS:
        raise ValueError(
            f"filter_audio kind must be one of {FILTER_KINDS}, got {kind!r}"
        )
    if q <= 0:
        raise ValueError("filter_audio 'q' must be above zero")
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"filter_audio 'cutoff_hz' ({cutoff_hz}) must be below the "
            f"Nyquist frequency ({nyquist}) for sample_rate {sample_rate}"
        )
    if waveform.size == 0:
        return as_track(waveform, sample_rate, "filter_audio")

    b, a = biquad_coefficients(kind, cutoff_hz, q, sample_rate)
    filtered = numpy.stack(
        [apply_biquad(channel, b, a) for channel in waveform]
    ).astype(numpy.float32)
    return as_track(filtered, sample_rate, "filter_audio")


def analyze_audio(audio, sample_rate=None):
    """Task command: measure a track without changing it.

    Read-only: the waveform passes through unmodified, and what comes back
    is diagnostics rather than an AudioTrack, since there is no processed
    track to hand a later step. Meant to feed a decision earlier in a
    workflow (whether 'compress_audio' or 'filter_audio' is needed, and
    with what settings) rather than to sit in the middle of a chain.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        A dict: peak_dbfs, rms_dbfs, crest_factor_db (peak minus rms), and
        a rough low_dbfs/mid_dbfs/high_dbfs spectral-balance reading whose
        three bands are shares of the same power that gives rms_dbfs, so
        they sit on that scale rather than tens of dB under it. Any value
        is None where a silent track leaves it undefined.
    """
    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "analyze_audio")
    check_arguments("analyze_audio", sample_rate=sample_rate)

    peak_dbfs = level_dbfs(waveform, measure="peak")
    rms_dbfs = level_dbfs(waveform, measure="rms")
    crest_factor_db = (
        peak_dbfs - rms_dbfs if peak_dbfs is not None and rms_dbfs is not None else None
    )
    bands = spectral_balance(waveform, sample_rate)
    return {
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "crest_factor_db": crest_factor_db,
        **bands,
    }
