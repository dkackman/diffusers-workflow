"""Pure signal processing: numpy, scipy and pyloudnorm, nothing from dw.

Everything the audio tasks, the media probe and the chain's seam joins
compute from a waveform lives here - level conversion, BS.1770 loudness and
true peak, the limiter, the filters and envelope followers, the spectral
readings and the resampler. It emits no events and checks no arguments: a
task decides what a measurement means and what to say about it, and this
module only measures.

Waveforms are (channels, samples) float32 numpy arrays unless a function
says otherwise (`integrated_lufs` and `true_peak_dbfs` take the
(frames, channels) layout pyloudnorm and soundfile use).

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
from fractions import Fraction

import numpy
import pyloudnorm
import scipy.ndimage
import scipy.signal

logger = logging.getLogger("dw")

# The floor a level is reported at rather than -inf, which JSON cannot carry
SILENCE_DBFS = -120.0

# BS.1770's gating block is 400 ms; pyloudnorm refuses anything shorter
MIN_LUFS_SECONDS = 0.4

# Minimum oversampling BS.1770 defines for a true-peak measurement
TRUE_PEAK_OVERSAMPLE = 4

# The measures `level_dbfs` reads a waveform by
LEVEL_MEASURES = ("peak", "rms")

# The limiter normalize_audio(limit=True) runs. Fixed rather than exposed
# (#474): long enough not to pump on a laugh, short enough not to duck the
# line after it. The look-ahead ramps the gain down before a transient; the
# hold keeps it down across the transient's own cycles; the release then
# recovers linearly in dB
LIMITER_LOOKAHEAD_MS = 5.0
LIMITER_HOLD_MS = 20.0
LIMITER_RELEASE_MS = 150.0
LIMITER_RELEASE_DB = 6.0
# Past this much reduction a target is squashing the track rather than
# levelling it, so the gain stops and target_lufs_capped says so
LIMITER_MAX_REDUCTION_DB = 12.0
# Past this much reduction pumping becomes audible - limiter_heavy
LIMITER_HEAVY_DB = 6.0
# Limiting lowers loudness by an amount that depends on how dense the
# material is, so the gain is searched for rather than corrected once (one
# pass left a dense track 2 LU short, #496). The search stops within this much
# of the target, after at most this many limiting passes; a track still
# further short than the tolerance is warned target_lufs_capped
LIMITER_TOLERANCE_LU = 0.1
LIMITER_SEARCH_PASSES = 8
# Oversampling runs in blocks so a long track never holds 4x of itself; the
# polyphase filter reaches about ten input samples each side, so this much
# overlap makes each block's interior exact
_TRUE_PEAK_BLOCK = 1 << 18
_TRUE_PEAK_OVERLAP = 32
# A gain this close to unity is floating-point noise, not reduction
UNITY_TOLERANCE = 1e-6

# Typical fundamental range for a human voice or a pitched instrument note;
# the periodicity search only looks at lags in this range so a slow room-tone
# swell or hum near DC cannot register as a pitch
PERIODICITY_MIN_HZ = 60.0
PERIODICITY_MAX_HZ = 500.0

SPECTRAL_BANDS = {
    "low_dbfs": (20.0, 250.0),
    "mid_dbfs": (250.0, 4000.0),
    "high_dbfs": (4000.0, 20000.0),
}


def dbfs(amplitude, floor=None):
    """A linear amplitude in dBFS.

    None or an amplitude of zero or below has no level: `floor` is returned
    (None when there is none). With a floor, a quieter reading is clamped to
    it too. Each caller picks the silence it needs - the gallery's
    `peak_dbfs` reports SILENCE_DBFS because JSON cannot carry -inf, a probe
    that serializes reports None, and voice attribution compares numerically
    and passes -inf.
    """
    if amplitude is None or amplitude <= 0:
        return floor
    db = 20.0 * math.log10(float(amplitude))
    return db if floor is None else max(floor, db)


def rms(window):
    """The root-mean-square of a window, or None when it has no samples."""
    if window is None or window.size == 0:
        return None
    return math.sqrt(float(numpy.mean(numpy.square(window, dtype=numpy.float64))))


def peak(window):
    """The largest absolute sample of a window, or None when it has none."""
    if window is None or window.size == 0:
        return None
    return float(numpy.max(numpy.abs(window)))


def level_dbfs(waveform, measure="peak"):
    """A waveform's level in dBFS, measured as `peak` or `rms`.

    `rms` is the same measurement `get_gallery_metadata` reports as
    `mean_dbfs`, so a matched join can be checked against what the gallery
    said about the shots going into it. A silent track has no level: None.
    """
    if measure not in LEVEL_MEASURES:
        raise ValueError(
            f"level measure must be one of {LEVEL_MEASURES}, got '{measure}'"
        )
    if waveform is None or waveform.size == 0:
        return None
    if measure == "peak":
        value = float(numpy.abs(waveform).max())
    else:
        value = float(
            numpy.sqrt(numpy.mean(numpy.square(waveform, dtype=numpy.float64)))
        )
    return dbfs(value)


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


def true_peak_dbfs(samples):
    """The inter-sample (true) peak of a track, in dBFS.

    `samples` is (frames, channels) or (frames,). Oversamples with a
    polyphase FIR (BS.1770's own reconstruction) and reads the peak of the
    interpolated signal, which is what a lossy encoder's own reconstruction
    filter can ring up past a sample-peak reading that stayed under 0 dBFS.
    The oversampling is `true_peak_envelope`'s, in blocks, so a long track
    never holds a 4x copy of itself. SILENCE_DBFS for an empty track, never
    None - unlike integrated loudness, a true peak is defined for any track,
    including a silent one.
    """
    if samples is None or samples.size == 0:
        return SILENCE_DBFS
    waveform = samples.reshape(1, -1) if samples.ndim == 1 else samples.T
    return dbfs(float(true_peak_envelope(waveform).max()), floor=SILENCE_DBFS)


def apply_biquad(channel, b, a):
    """One second-order section, run over a channel: scipy's lfilter, which
    is the direct-form recursion in C, from zero initial conditions."""
    b0, b1, b2 = b
    a1, a2 = a
    return scipy.signal.lfilter(
        numpy.array([b0, b1, b2], dtype=numpy.float64),
        numpy.array([1.0, a1, a2], dtype=numpy.float64),
        numpy.asarray(channel, dtype=numpy.float64),
    )


def layout_name(channels, layout=None):
    """The PyAV layout name for a channel count: mono, stereo, else the
    stream's own layout (`layout.name`) when one is given, else `<n>c`.
    The one spelling `dw.media` and `resample_waveform` share."""
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return layout.name if layout is not None else f"{channels}c"


def _is_positive_rate(value):
    """Whether a value is a rate: something `float()` accepts that is above
    zero (NaN is not), and not a boolean."""
    try:
        return not isinstance(value, bool) and float(value) > 0
    except (TypeError, ValueError):
        return False


def resample_waveform(waveform, sample_rate, target_sample_rate):
    """A waveform at a different rate, as a plain (channels, samples) array.

    The conversion resample_audio performs, without the task's argument
    handling or its AudioTrack return, so a task that has waveforms in hand
    already can reach the rate conversion directly.
    """
    # PyAV's resampler accepts a zero rate and answers with the samples
    # unchanged, which is indistinguishable from a conversion that happened
    # (#140) - so neither rate is allowed to be one that cannot be a rate
    for name, rate in (("sample_rate", sample_rate), ("target", target_sample_rate)):
        if not _is_positive_rate(rate):
            raise ValueError(
                f"resample_waveform needs a {name} above zero, got {rate!r}"
            )
    if sample_rate == target_sample_rate:
        return waveform

    import av
    from av.audio.resampler import AudioResampler

    channels = waveform.shape[0]
    layout = layout_name(channels)
    frame = av.AudioFrame.from_ndarray(
        numpy.ascontiguousarray(waveform, dtype=numpy.float32),
        format="fltp",
        layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0
    frame.time_base = Fraction(1, sample_rate)

    resampler = AudioResampler(format="fltp", layout=layout, rate=target_sample_rate)
    converted = [f.to_ndarray() for f in resampler.resample(frame)]
    converted += [f.to_ndarray() for f in resampler.resample(None)]
    logger.debug(
        f"Resampled {waveform.shape[1]} samples at {sample_rate}Hz "
        f"to {target_sample_rate}Hz"
    )
    return numpy.concatenate(converted, axis=1).astype(numpy.float32)


def as_channels_samples(audio):
    """Normalize a waveform to a (channels, samples) float32 numpy array.

    Accepts torch tensors or numpy arrays shaped (samples,), (channels, samples),
    (samples, channels), or a one-item batch (1, channels, samples). Channel
    position is decided the way normalize_audio in writers.py decides it: there
    are always more samples than channels.
    """
    if hasattr(audio, "detach"):  # a torch tensor, without importing torch
        audio = audio.detach().cpu().float().numpy()
    audio = numpy.asarray(audio, dtype=numpy.float32)

    if audio.ndim == 1:
        return audio[numpy.newaxis, :]

    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"Cannot normalize a waveform batch of {audio.shape[0]}")
        audio = audio[0]

    if audio.ndim != 2:
        raise ValueError(f"A waveform must have 1-3 dimensions, got {audio.ndim}")

    if audio.shape[0] > audio.shape[1]:  # (samples, channels) -> transpose
        audio = audio.T

    return numpy.ascontiguousarray(audio)


def slice_samples(waveform, start, length):
    """Cut length samples out of a (channels, samples) waveform from start.

    A slice reaching past the end of the waveform is zero-padded to the
    requested length, so frame-aligned slicing near the end of a track always
    yields full-size chunks.
    """
    channels, total = waveform.shape
    piece = waveform[:, start : start + length]
    if piece.shape[1] < length:
        padding = numpy.zeros((channels, length - piece.shape[1]), dtype=waveform.dtype)
        piece = numpy.concatenate([piece, padding], axis=1)
    return piece


def matched_channels(*waveforms):
    """Tile mono up so every waveform has the same channel count."""
    channels = max(waveform.shape[0] for waveform in waveforms)
    return tuple(
        (
            numpy.tile(waveform, (channels, 1))
            if waveform.shape[0] == 1 and channels > 1
            else waveform
        )
        for waveform in waveforms
    )


def equal_power_ramps(window):
    """Cosine/sine fade curves that sum to constant power across the window."""
    theta = numpy.linspace(0.0, numpy.pi / 2.0, window, endpoint=False)
    return numpy.cos(theta, dtype=numpy.float32), numpy.sin(theta, dtype=numpy.float32)


def fade_curve(window):
    """A cosine fall from full level to exact silence, both ends included -
    unlike the seam ramps, which stop short of the endpoint so two of them
    tile a crossfade without a doubled sample."""
    theta = numpy.linspace(0.0, numpy.pi / 2.0, window, endpoint=True)
    return numpy.cos(theta, dtype=numpy.float32)


def spectral_flatness(waveform, sample_rate=None, native_sample_rate=None):
    """Geometric-mean-over-arithmetic-mean of the magnitude spectrum, averaged
    across channels - near 0 for tonal/speech material, near 1 for noise-like
    material (see TONAL_FLATNESS_THRESHOLD).

    When the material was upsampled, band-limited interpolation leaves near
    zero energy above the original Nyquist - a large near-silent band that
    depresses the geometric mean relative to the arithmetic one regardless of
    what the material actually is, reading as spuriously tonal (#198). Given
    both rates, the spectrum is limited to bins below the native Nyquist so an
    upsampled tail is measured the same as it would be at its own rate.
    """
    spectrum = numpy.abs(numpy.fft.rfft(waveform, axis=1))
    if sample_rate and native_sample_rate and native_sample_rate < sample_rate:
        native_bins = max(
            2,
            int(spectrum.shape[1] * native_sample_rate / sample_rate),
        )
        spectrum = spectrum[:, :native_bins]
    spectrum = numpy.maximum(spectrum, 1e-10)
    geometric_mean = numpy.exp(numpy.mean(numpy.log(spectrum), axis=1))
    arithmetic_mean = numpy.mean(spectrum, axis=1)
    return float(numpy.mean(geometric_mean / arithmetic_mean))


def harmonicity(waveform, sample_rate):
    """Normalized autocorrelation peak within a plausible pitch range,
    averaged across channels - near 1 for a strongly periodic signal (voiced
    speech, a pitched note), near 0 for noise (see HARMONICITY_THRESHOLD).

    Spectral flatness alone missed real speech (#198): a vowel's formants
    spread its energy broadly enough across the band that flatness reads
    similar to noise, even though the waveform itself repeats every pitch
    period. Autocorrelation measures that repetition directly and is
    insensitive to how the spectrum happens to be shaped, so it catches what
    flatness cannot.
    """
    min_lag = max(int(sample_rate / PERIODICITY_MAX_HZ), 1)
    max_lag = min(int(sample_rate / PERIODICITY_MIN_HZ), waveform.shape[1] - 1)
    if max_lag <= min_lag:
        return 0.0

    scores = []
    for channel in waveform:
        centered = channel - channel.mean()
        energy = float(numpy.dot(centered, centered))
        if energy <= 1e-12:
            continue
        # The autocorrelation through an FFT rather than numpy.correlate,
        # which is O(n^2): on a 2 s window at 48 kHz that is about 10^10
        # operations, and find_loop_bed measures many windows (#218).
        # Padding to at least 2n - 1 keeps the circular correlation from
        # wrapping, so every lag equals the direct sum
        size = 1 << int(2 * centered.shape[0] - 1).bit_length()
        spectrum = numpy.fft.rfft(centered, size)
        correlation = numpy.fft.irfft(spectrum * numpy.conj(spectrum), size)
        window = correlation[min_lag : max_lag + 1]
        if window.size == 0:
            continue
        scores.append(float(numpy.max(window) / energy))
    return max(scores) if scores else 0.0


def spectral_balance(waveform, sample_rate):
    """A rough low/mid/high energy reading in dBFS, from one FFT of the
    channel-averaged track - not a spectrogram, just enough to say whether
    a track leans bright or boomy.

    Each band's power is a share of the same Parseval sum that gives
    rms_dbfs (mean(x**2)): a one-sided rfft bin's power is doubled to
    account for its mirrored negative-frequency twin, except the DC and
    (for even n) Nyquist bins, which have no twin. Summed over the full
    spectrum this equals mean(x**2) exactly, so a *_dbfs band sits on the
    same scale as rms_dbfs rather than ~40 dB under it (#211)."""
    if waveform.size == 0:
        return {name: None for name in SPECTRAL_BANDS}
    mono = waveform.mean(axis=0)
    n = mono.shape[0]
    spectrum = numpy.fft.rfft(mono)
    power = numpy.square(numpy.abs(spectrum), dtype=numpy.float64) / (n * n)
    if n % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    freqs = numpy.fft.rfftfreq(n, d=1.0 / sample_rate)
    result = {}
    for name, (low, high) in SPECTRAL_BANDS.items():
        band = power[(freqs >= low) & (freqs < min(high, sample_rate / 2.0))]
        if band.size == 0:
            result[name] = None
            continue
        energy = float(numpy.sum(band))
        result[name] = 10.0 * numpy.log10(energy) if energy > 0.0 else None
    return result


def true_peak_envelope(waveform):
    """The per-sample true peak of a (channels, samples) waveform, linked
    across channels so the loudest one drives them all and the stereo image
    does not shift. Each sample carries the largest oversampled value on
    either side of it, so an inter-sample peak is owed by both neighbours."""
    total = waveform.shape[1]
    envelope = numpy.zeros(total, dtype=numpy.float32)
    for start in range(0, total, _TRUE_PEAK_BLOCK):
        stop = min(total, start + _TRUE_PEAK_BLOCK)
        low = max(0, start - _TRUE_PEAK_OVERLAP)
        high = min(total, stop + _TRUE_PEAK_OVERLAP)
        block = scipy.signal.resample_poly(
            waveform[:, low:high].astype(numpy.float32),
            TRUE_PEAK_OVERSAMPLE,
            1,
            axis=1,
        )
        linked = numpy.abs(block).max(axis=0)
        per_sample = linked.reshape(high - low, TRUE_PEAK_OVERSAMPLE).max(axis=1)
        envelope[start:stop] = per_sample[start - low : stop - low]
    envelope[1:] = numpy.maximum(envelope[1:], envelope[:-1])
    return envelope


def limiter_curve(envelope, ceiling, sample_rate):
    """The per-sample gain (<= 1) that holds `envelope` under `ceiling`, or
    None when nothing crosses it.

    Vectorised end to end: the required gain goes through a forward sliding
    minimum over the look-ahead, a backward one over the hold, a linear-in-dB
    release (a cumulative minimum), and a boxcar as long as the look-ahead.
    Every value the boxcar averages is the minimum of a window that contains
    the sample it lands on, so the curve never exceeds the required gain -
    it ramps down before a transient rather than delaying the signal.
    """
    required = numpy.minimum(
        1.0, ceiling / numpy.maximum(envelope.astype(numpy.float64), 1e-12)
    )
    if required.min() >= 1.0 - UNITY_TOLERANCE:
        return None
    total = required.size
    half = max(1, int(round(LIMITER_LOOKAHEAD_MS / 2000.0 * sample_rate)))
    lookahead = 2 * half
    centred = scipy.ndimage.minimum_filter1d(required, lookahead + 1, mode="nearest")
    ahead = numpy.concatenate([centred[half:], numpy.full(half, centred[-1])])[:total]

    hold = max(1, int(round(LIMITER_HOLD_MS / 1000.0 * sample_rate)))
    centred = scipy.ndimage.minimum_filter1d(ahead, 2 * hold + 1, mode="nearest")
    held = numpy.empty_like(ahead)
    head = min(hold, total)
    held[:head] = numpy.minimum.accumulate(ahead[:head])
    held[head:] = centred[: total - head]

    rate = LIMITER_RELEASE_DB / (LIMITER_RELEASE_MS / 1000.0 * sample_rate)
    ramp = rate * numpy.arange(total, dtype=numpy.float64)
    held_db = 20.0 * numpy.log10(held)
    released = 10.0 ** ((numpy.minimum.accumulate(held_db - ramp) + ramp) / 20.0)

    padded = numpy.concatenate([numpy.full(lookahead, released[0]), released])
    sums = numpy.concatenate([[0.0], numpy.cumsum(padded)])
    curve = (sums[lookahead + 1 :] - sums[: -lookahead - 1]) / (lookahead + 1)
    return numpy.minimum(curve, required)


def limit_at(waveform, envelope, gain_db, ceiling, sample_rate):
    """One limiting pass at a static gain: the output, the limiter's curve
    (None when it touched nothing), the static trim a reconstruction
    overshoot needed, and the output's true peak (linear)."""
    gain = 10 ** (gain_db / 20)
    curve = limiter_curve(envelope * gain, ceiling, sample_rate)
    if curve is None:
        output = (waveform * gain).astype(numpy.float32)
    else:
        output = (waveform * (gain * curve)[numpy.newaxis, :]).astype(numpy.float32)
    output_peak = float(true_peak_envelope(output).max())
    trim = 1.0
    if output_peak > ceiling * (1.0 + 1e-4):
        trim = ceiling / output_peak
        output = (output * trim).astype(numpy.float32)
        output_peak *= trim
    return output, curve, trim, output_peak


def search_gain(waveform, envelope, ceiling, sample_rate, target_lufs, low_db, high_db):
    """The static gain in [low_db, high_db] whose limited output lands within
    LIMITER_TOLERANCE_LU of target_lufs: (gain_db, limit_at's result, output
    LUFS). Loudness rises with the gain, but by less than the gain once the
    limiter acts, so this starts at the target's own gain (exact while the
    limiter is idle), tries the cap when that falls short, and closes the
    bracket between the two by regula falsi. Where even the cap falls short
    the cap is the answer, and the caller warns the shortfall."""

    def attempt(gain_db):
        limited = limit_at(waveform, envelope, gain_db, ceiling, sample_rate)
        return gain_db, limited, integrated_lufs(limited[0].T, sample_rate)

    def miss(result):
        return None if result[2] is None else target_lufs - result[2]

    low = attempt(low_db)
    if miss(low) is None or miss(low) <= LIMITER_TOLERANCE_LU or low_db >= high_db:
        return low
    high = attempt(high_db)
    if miss(high) is None or miss(high) >= -LIMITER_TOLERANCE_LU:
        return high
    best = min((low, high), key=lambda result: abs(miss(result)))
    for _ in range(LIMITER_SEARCH_PASSES - 2):
        low_miss, high_miss = miss(low), miss(high)
        span = high[0] - low[0]
        gain_db = low[0] + span * low_miss / (low_miss - high_miss)
        # Stay off the bracket's ends, so a curved response cannot stall
        # regula falsi against one side of it
        gain_db = min(max(gain_db, low[0] + 0.1 * span), high[0] - 0.1 * span)
        probe = attempt(gain_db)
        if miss(probe) is None:
            break
        if abs(miss(probe)) < abs(miss(best)):
            best = probe
        if abs(miss(probe)) <= LIMITER_TOLERANCE_LU:
            break
        if miss(probe) > 0:
            low = probe
        else:
            high = probe
    return best


def follow_envelope(waveform, sample_rate, attack_ms, release_ms):
    """A linked (all-channels) peak envelope, smoothed by separate attack and
    release time constants - the same detector a hardware compressor uses,
    tracking the loudest channel so a stereo image does not shift."""
    rectified = numpy.abs(waveform).max(axis=0)
    attack_coef = time_constant_coef(attack_ms, sample_rate)
    release_coef = time_constant_coef(release_ms, sample_rate)
    # The branch on the running level is what makes this a loop rather than a
    # filter, but the per-sample numpy indexing was the expensive half of it:
    # a 3-minute track is ~8M samples, and this runs on the single FIFO
    # worker. tolist() hands the loop plain Python floats, which is the same
    # arithmetic on the same values, several times faster
    samples = rectified.tolist()
    envelope = []
    level = 0.0
    for sample in samples:
        coef = attack_coef if sample > level else release_coef
        level = coef * level + (1.0 - coef) * sample
        envelope.append(level)
    return numpy.asarray(envelope, dtype=rectified.dtype)


def time_constant_coef(time_ms, sample_rate):
    """The per-sample smoothing coefficient for an exponential time constant.
    0 ms means the envelope follows instantly, with no smoothing at all."""
    if time_ms <= 0:
        return 0.0
    return float(numpy.exp(-1.0 / (time_ms / 1000.0 * sample_rate)))


def biquad_coefficients(kind, cutoff_hz, q, sample_rate):
    """RBJ Audio EQ Cookbook coefficients for a single biquad stage,
    normalized so a0 is 1."""
    w0 = 2.0 * numpy.pi * cutoff_hz / sample_rate
    cos_w0 = numpy.cos(w0)
    sin_w0 = numpy.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    if kind == "lowpass":
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
    elif kind == "highpass":
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
    elif kind == "bandpass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
    else:  # notch
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (
        numpy.array([b0, b1, b2], dtype=numpy.float64) / a0,
        numpy.array([a1, a2], dtype=numpy.float64) / a0,
    )
