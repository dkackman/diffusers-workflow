"""BS.1770 loudness measurement, shared by the media probe and the audio
tasks so both read a level the same way.

Peak and loudness are different quantities - a sparse voice-over and a dense
score can share a peak and still sit tens of dB apart in how loud they sound,
because a peak is one sample and loudness is measured over the whole track
(#361). `integrated_lufs` is the BS.1770 integrated measure pyloudnorm
implements; `true_peak_dbfs` is the inter-sample peak BS.1770 defines
alongside it - a sample-peak reading can miss a peak that only appears
between samples, which is what an encoder's reconstruction filter can ring
up past 0 dBFS even when every decoded sample was under it.
"""

import logging
import math

import numpy
import pyloudnorm
import scipy.signal

logger = logging.getLogger("dw")

# The floor a level is reported at rather than -inf, which JSON cannot carry
SILENCE_DBFS = -120.0

# BS.1770's gating block is 400 ms; pyloudnorm refuses anything shorter
MIN_LUFS_SECONDS = 0.4

# Minimum oversampling BS.1770 defines for a true-peak measurement
TRUE_PEAK_OVERSAMPLE = 4


def _dbfs(value):
    if value <= 0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(float(value)))


def integrated_lufs(samples, rate):
    """Integrated loudness in LUFS, or None when it cannot be measured.

    `samples` is (frames, channels) or (frames,). None for a clip shorter
    than the 400 ms gating block, an all-silent clip (pyloudnorm's own
    -inf, which JSON cannot carry either), or anything pyloudnorm refuses -
    a measurement that failed reads as "unknown" rather than as a crashed
    task or probe.
    """
    if samples is None or samples.size == 0:
        return None
    frames = samples.shape[0]
    if frames < int(round(MIN_LUFS_SECONDS * rate)):
        return None
    try:
        meter = pyloudnorm.Meter(rate)
        value = float(meter.integrated_loudness(samples))
    except Exception as e:
        logger.debug(f"integrated_lufs: could not measure: {e}")
        return None
    if not math.isfinite(value):
        return None
    return value


def true_peak_dbfs(samples, oversample=TRUE_PEAK_OVERSAMPLE):
    """The inter-sample (true) peak of a track, in dBFS.

    `samples` is (frames, channels) or (frames,). Oversamples with a
    polyphase FIR (BS.1770's own reconstruction) and reads the peak of the
    interpolated signal, which is what a lossy encoder's own reconstruction
    filter can ring up past a sample-peak reading that stayed under 0 dBFS.
    SILENCE_DBFS for an empty track, never None - unlike integrated
    loudness, a true peak is defined for any track, including a silent one.
    """
    if samples is None or samples.size == 0:
        return SILENCE_DBFS
    try:
        oversampled = scipy.signal.resample_poly(samples, oversample, 1, axis=0)
    except Exception as e:
        logger.debug(f"true_peak_dbfs: could not oversample: {e}")
        oversampled = samples
    peak = float(numpy.abs(oversampled).max(initial=0.0))
    return _dbfs(peak)
