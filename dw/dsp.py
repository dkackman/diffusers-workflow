"""Pure signal processing: numpy, scipy and pyloudnorm, nothing from dw.

Everything the audio tasks, the media probe and the chain's seam joins
compute from a waveform lives here - level conversion, BS.1770 loudness and
true peak, the limiter, the filters and envelope followers, the spectral
readings, the resampler and the beat tracker (onset envelope, tempo
and beats). It emits no events and checks no arguments: a
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
    out = numpy.concatenate(converted, axis=1).astype(numpy.float32)
    # The resampler's flush rounds the length up, so a track of n samples came
    # back one sample longer than n * target / rate, and a downstream
    # 'fit' then reported a 1-sample trim on a clean run (#716)
    expected = int(round(waveform.shape[1] * target_sample_rate / sample_rate))
    if out.shape[1] > expected:
        out = out[:, :expected]
    return out


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


# Beat tracking (#600). The onset envelope is spectral flux - the positive
# frame-to-frame rise of the log-compressed magnitude spectrum - on a 10 ms
# hop over a ~23 ms window; tempo is the autocorrelation peak of that
# envelope under a log-normal prior; the beats are Ellis's dynamic programme
# (2007), which trades onset strength against keeping each interval near the
# tempo's period
ONSET_HOP_SECONDS = 0.01
ONSET_WINDOW_SECONDS = 0.023
ONSET_LOG_GAIN = 1000.0
# The tempo prior: centred on 120 BPM an octave wide, or on a caller's hint
# half an octave wide
TEMPO_PRIOR_BPM = 120.0
TEMPO_PRIOR_OCTAVES = 1.0
TEMPO_HINT_OCTAVES = 0.5
# When the caller's range holds no clear pulse, the pulse is looked for over
# this range and folded into the caller's by octaves - an 86 BPM song asked
# for at 140-200 is tracked at 172
TEMPO_SEARCH_BPM = (30.0, 300.0)
# The envelope's drift is taken out over this window before its
# autocorrelation: a level change slower than this is not a pulse
TEMPO_DETREND_SECONDS = 1.0
# How strongly the programme holds an interval to the period: the penalty is
# BEAT_TIGHTNESS * log(interval / period)^2
BEAT_TIGHTNESS = 100.0
# An envelope is too flat to track when its autocorrelation at the tempo lag
# is under this fraction of its energy (noise, a held tone) or its peak is
# under this many times its mean (swells with no attack)
TRACKABLE_MIN_PERIODICITY = 0.2
TRACKABLE_MIN_CREST = 3.0
# A tracked pulse is believed when its periodicity is at least this, or when
# its beats stand clear of the envelope's noise floor: their median onset at
# least this many robust deviations (1.4826 MAD) above the envelope's median.
# A noise bed, however loud, gives the programme only its own bumps to land
# on - about 2 deviations - and a weak periodicity; a song has one or the other
CONFIDENT_PERIODICITY = 0.3
CONFIDENT_SALIENCE = 3.0
# A beat at the head or tail is trimmed while its onset is under this
# fraction of the RMS onset at the beats: the programme keeps stepping
# through an intro's silence at the period, onto nothing
BEAT_TRIM_FRACTION = 0.5
# An end beat further than this fraction of the period (or two frames) off
# the period from its neighbour is dropped, and the ends re-extended one
# period at a time onto an onset within that distance: the programme's ends
# have a neighbour on one side only, and settle on a tone's release or a
# late transient instead of the beat
BEAT_EDGE_TOLERANCE = 0.06
# ...onto an onset standing at least this fraction of the weaker beats'
# prominence (the 10th percentile's) above the envelope around it: a soft
# end beat a period from its neighbour counts, a bump in a noise bed
# after the music stops does not
BEAT_EXTEND_FRACTION = 0.75
# The RMS-peak fallback: 50 ms windows on the onset hop, peaks at least this
# prominent relative to the envelope's range
RMS_PEAK_WINDOW_SECONDS = 0.05
RMS_PEAK_PROMINENCE = 0.1


def mono_float64(waveform):
    """A (channels, samples) or 1-D waveform as one float64 channel."""
    mono = numpy.asarray(waveform, dtype=numpy.float64)
    return mono.mean(axis=0) if mono.ndim == 2 else mono


def onset_envelope(mono, sample_rate):
    """(envelope, frames per second): spectral flux on a 10 ms hop.

    Frame k is centred on k / rate seconds, so a frame index divided by the
    rate is a time in the track.
    """
    hop = max(1, int(round(sample_rate * ONSET_HOP_SECONDS)))
    window = 1 << int(math.ceil(math.log2(ONSET_WINDOW_SECONDS * sample_rate)))
    window = max(window, hop + 1)
    if mono.shape[0] < window:
        return numpy.zeros(0), sample_rate / hop
    _, _, spectrum = scipy.signal.stft(
        mono, sample_rate, nperseg=window, noverlap=window - hop, padded=True
    )
    magnitude = numpy.log1p(ONSET_LOG_GAIN * numpy.abs(spectrum))
    # Frame 0 rises from silence, so a hit at 0 s is an onset
    flux = numpy.maximum(0.0, numpy.diff(magnitude, axis=1, prepend=0.0))
    return flux.mean(axis=0), sample_rate / hop


def fold_bpm(bpm, min_bpm, max_bpm, centre=TEMPO_PRIOR_BPM):
    """bpm moved by octaves into [min_bpm, max_bpm], the octave nearest
    `centre` when more than one fits; None when none does."""
    if bpm is None or bpm <= 0:
        return None
    if min_bpm <= bpm <= max_bpm:
        return bpm
    fits = [
        bpm * 2.0**octave
        for octave in range(-4, 5)
        if min_bpm <= bpm * 2.0**octave <= max_bpm
    ]
    return min(fits, key=lambda fit: abs(math.log2(fit / centre)), default=None)


def detrended(envelope, rate):
    """The envelope less its moving mean over TEMPO_DETREND_SECONDS.

    A level that drifts - a fade, a swell, a loud first second of room tone -
    correlates with itself at every lag, which reads as a pulse at whatever
    lag the prior favours. A moving mean is a linear filter, so a pulse
    stays periodic at its own period and only the drift goes.
    """
    if envelope.size < 2:
        return envelope - envelope.mean() if envelope.size else envelope
    # Frame 0 is the rise from silence, not part of the drift
    envelope = envelope.copy()
    envelope[0] = numpy.median(envelope[1:])
    width = max(1, min(envelope.size, int(round(TEMPO_DETREND_SECONDS * rate))))
    return envelope - scipy.ndimage.uniform_filter1d(envelope, width, mode="reflect")


def estimate_tempo(envelope, rate, min_bpm, max_bpm, hint_bpm=None):
    """(bpm, periodicity, missed_pulse_bpm) from the envelope's autocorrelation.

    The tempo is searched for in the range, and the pulse over
    TEMPO_SEARCH_BPM; when the pulse is the more periodic of the two it is
    folded into the range by octaves and wins. When no octave of it fits, the
    range's best tempo is kept and `missed_pulse_bpm` is the pulse it misses;
    otherwise that is None. `periodicity` is the normalized
    autocorrelation at the pulse, 0 to 1. (None, 0.0, None) when the envelope
    holds no energy or the range no lag; a bpm is always in the range.
    """
    centred = detrended(envelope, rate)
    count = centred.shape[0]
    if count < 2:
        return None, 0.0, None
    correlation = scipy.signal.correlate(centred, centred, mode="full", method="fft")
    correlation = correlation[count - 1 :]
    if correlation[0] <= 0:
        return None, 0.0, None
    correlation = correlation / correlation[0]
    centre = TEMPO_PRIOR_BPM if hint_bpm is None else hint_bpm
    width = TEMPO_PRIOR_OCTAVES if hint_bpm is None else TEMPO_HINT_OCTAVES

    def best(low_bpm, high_bpm):
        low = max(1, int(math.floor(rate * 60.0 / high_bpm)))
        high = min(count - 2, int(math.ceil(rate * 60.0 / low_bpm)))
        if high < low:
            return None, 0.0
        lags = numpy.arange(low, high + 1)
        prior = numpy.exp(-0.5 * (numpy.log2(60.0 * rate / lags / centre) / width) ** 2)
        lag = lags[int(numpy.argmax(correlation[lags] * prior))]
        # Parabolic interpolation between the lags either side
        before, at, after = correlation[lag - 1], correlation[lag], correlation[lag + 1]
        curvature = before - 2.0 * at + after
        shift = 0.5 * (before - after) / curvature if curvature < 0 else 0.0
        bpm = min(max(60.0 * rate / (lag + shift), low_bpm), high_bpm)
        return bpm, float(max(at, 0.0))

    bpm, periodicity = best(min_bpm, max_bpm)
    if bpm is None:
        return bpm, periodicity, None
    # The range's best is held against the pulse found over the wide range:
    # a half-time song's strongest period is under 60 BPM, and its octaves
    # in the range share the field with the song's other repeats - a
    # subdivision, a riff - which the prior alone can pick instead
    pulse, pulse_periodicity = best(*TEMPO_SEARCH_BPM)
    if pulse is None or pulse_periodicity <= periodicity:
        return bpm, periodicity, None
    folded = fold_bpm(pulse, min_bpm, max_bpm, centre)
    if folded is None:
        return bpm, pulse_periodicity, pulse
    return folded, pulse_periodicity, None


def is_trackable(envelope, periodicity):
    """Whether an onset envelope has the attacks and the periodicity to track."""
    if not envelope.size or periodicity < TRACKABLE_MIN_PERIODICITY:
        return False
    mean = envelope.mean()
    return mean > 0 and envelope.max() / mean >= TRACKABLE_MIN_CREST


def beat_salience(envelope, beat_frames):
    """How far the beats' median onset stands above the envelope's median, in
    robust deviations (1.4826 MAD); frame 0, the rise from silence, is left
    out of the floor. Infinite when the envelope off the beats is flat."""
    if not beat_frames.shape[0] or envelope.shape[0] < 2:
        return 0.0
    rest = envelope[1:]
    median = float(numpy.median(rest))
    spread = 1.4826 * float(numpy.median(numpy.abs(rest - median)))
    lift = float(numpy.median(envelope[beat_frames])) - median
    if spread <= 0:
        return math.inf if lift > 0 else 0.0
    return lift / spread


def is_confident(envelope, beat_frames, periodicity):
    """Whether tracked beats are a pulse rather than a noise bed's bumps."""
    return (
        periodicity >= CONFIDENT_PERIODICITY
        or beat_salience(envelope, beat_frames) >= CONFIDENT_SALIENCE
    )


def track_beats(envelope, rate, bpm, tightness=BEAT_TIGHTNESS):
    """Beat frame indices by dynamic programming, leading and trailing beats
    that land on next to nothing trimmed and the ends settled on the period.

    Frame 0 is left out of the programme: it holds the track's rise from
    silence whether or not a beat falls there, so only settling the ends
    can put a beat on it, and only when it is a period from the next.
    """
    count = envelope.shape[0]
    period = rate * 60.0 / bpm
    if count < 2 or envelope[1:].std() <= 0:
        return numpy.zeros(0, dtype=int)
    local = envelope / envelope[1:].std()
    local[0] = 0.0
    score = local.copy()
    backlink = numpy.full(count, -1, dtype=int)
    steps = numpy.arange(max(1, int(round(period / 2))), int(round(2 * period)) + 1)
    penalty = -tightness * numpy.log(steps / period) ** 2
    for frame in range(int(steps[0]), count):
        reachable = steps <= frame
        previous = frame - steps[reachable]
        candidates = score[previous] + penalty[reachable]
        best = int(numpy.argmax(candidates))
        if candidates[best] > 0:
            score[frame] = local[frame] + candidates[best]
            backlink[frame] = previous[best]
    peaks = scipy.signal.argrelmax(score)[0]
    if not peaks.size:
        return numpy.zeros(0, dtype=int)
    last = peaks[score[peaks] >= 0.5 * numpy.median(score[peaks])][-1]
    beats = [int(last)]
    while backlink[beats[-1]] >= 0:
        beats.append(int(backlink[beats[-1]]))
    beats = numpy.array(beats[::-1], dtype=int)
    strength = envelope[beats]
    floor = BEAT_TRIM_FRACTION * math.sqrt(float(numpy.mean(strength**2)))
    kept = numpy.nonzero(strength >= floor)[0]
    if not kept.size:
        return numpy.zeros(0, dtype=int)
    kept = beats[kept[0] : kept[-1] + 1]
    floor = BEAT_EXTEND_FRACTION * float(
        numpy.percentile([_prominence(envelope, frame, period) for frame in kept], 10)
    )
    return _settle_ends(list(kept), envelope, period, floor)


def _prominence(envelope, frame, period):
    """How far the envelope at frame stands above its median over the half
    period either side."""
    half = max(1, int(round(period / 2)))
    around = envelope[max(0, frame - half) : frame + half + 1]
    return float(envelope[frame] - numpy.median(around))


def _settle_ends(beats, envelope, period, floor):
    """The beat frames with off-period end beats dropped, then the ends
    extended a period at a time while an onset of at least `floor`
    prominence is there."""
    tolerance = max(2.0, BEAT_EDGE_TOLERANCE * period)
    while len(beats) >= 3 and abs(beats[-1] - beats[-2] - period) > tolerance:
        beats.pop()
    while len(beats) >= 3 and abs(beats[1] - beats[0] - period) > tolerance:
        beats.pop(0)
    reach = int(round(tolerance))

    def onset_near(centre):
        low = max(0, int(round(centre)) - reach)
        high = min(envelope.shape[0], int(round(centre)) + reach + 1)
        if low >= high:
            return None
        frame = low + int(numpy.argmax(envelope[low:high]))
        return frame if _prominence(envelope, frame, period) >= floor else None

    while (frame := onset_near(beats[-1] + period)) is not None and frame > beats[-1]:
        beats.append(frame)
    while (frame := onset_near(beats[0] - period)) is not None and frame < beats[0]:
        beats.insert(0, frame)
    return numpy.array(beats, dtype=int)


def rms_peaks(mono, sample_rate, max_bpm):
    """(peak times in seconds, peak level in dBFS) of a 50 ms RMS envelope,
    peaks at least one max_bpm period apart - the fallback for a track with
    no attacks to find onsets in."""
    hop = max(1, int(round(sample_rate * ONSET_HOP_SECONDS)))
    window = max(hop, int(round(sample_rate * RMS_PEAK_WINDOW_SECONDS)))
    if mono.shape[0] < window:
        return numpy.zeros(0), SILENCE_DBFS
    frames = numpy.lib.stride_tricks.sliding_window_view(mono, window)[::hop]
    envelope = numpy.sqrt(numpy.mean(frames**2, axis=1))
    loudest = float(envelope.max())
    span = loudest - float(envelope.min())
    if span <= 0:
        return numpy.zeros(0), dbfs(loudest, floor=SILENCE_DBFS)
    peaks, _ = scipy.signal.find_peaks(
        envelope,
        distance=max(1, int(round(sample_rate / hop * 60.0 / max_bpm))),
        prominence=RMS_PEAK_PROMINENCE * span,
    )
    return (peaks * hop + window / 2.0) / sample_rate, dbfs(loudest, floor=SILENCE_DBFS)


def downbeat_phase(envelope, beat_frames, meter=4, margin=1.1):
    """Which of `meter` beat positions (0 to meter-1) the bars start on: the
    phase whose beats carry the most onset, when it beats the next by
    `margin`; None when no phase stands out or there are under two bars."""
    if beat_frames.shape[0] < 2 * meter:
        return None
    strength = envelope[beat_frames]
    means = sorted(
        ((float(strength[phase::meter].mean()), phase) for phase in range(meter)),
        reverse=True,
    )
    if means[0][0] <= 0 or means[0][0] < margin * means[1][0]:
        return None
    return means[0][1]


def warp_times(times, knots_from, knots_to):
    """Map times piecewise-linearly through (knots_from -> knots_to), the end
    segments extended past the first and last knot; a single knot is a shift.
    """
    times = numpy.asarray(times, dtype=numpy.float64)
    knots_from = numpy.asarray(knots_from, dtype=numpy.float64)
    knots_to = numpy.asarray(knots_to, dtype=numpy.float64)
    if knots_from.shape[0] == 1:
        return times + (knots_to[0] - knots_from[0])
    warped = numpy.interp(times, knots_from, knots_to)
    for end, (a, b) in (
        (times < knots_from[0], (0, 1)),
        (times > knots_from[-1], (-2, -1)),
    ):
        slope = (knots_to[b] - knots_to[a]) / (knots_from[b] - knots_from[a])
        anchor = 0 if a == 0 else -1
        warped[end] = knots_to[anchor] + (times[end] - knots_from[anchor]) * slope
    return warped
