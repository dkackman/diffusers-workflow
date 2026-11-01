"""dw.dsp: the one true-peak oversampler.

`true_peak_dbfs` used to oversample the whole array while the limiter's
`true_peak_envelope` oversampled in blocks; the two had to agree, and this
pins that they still do on the shapes that could tell them apart.
"""

import math

import numpy
import pytest
import scipy.signal

from dw.dsp import (
    SILENCE_DBFS,
    TRUE_PEAK_OVERSAMPLE,
    spectral_centroid,
    true_peak_dbfs,
    true_peak_envelope,
)

RATE = 16000
TOLERANCE_DB = 0.01


def _tracks():
    rng = numpy.random.default_rng(7)
    short = rng.uniform(-0.5, 0.5, size=(3000, 2)).astype(numpy.float32)
    long_track = rng.uniform(-0.4, 0.4, size=((1 << 18) + 5000, 2)).astype(
        numpy.float32
    )
    time = numpy.arange(RATE) / RATE
    # A tone whose crests fall between samples
    sine = (0.8 * numpy.sin(2 * math.pi * 7333.0 * time + 0.3))[:, None].astype(
        numpy.float32
    )
    impulse = numpy.zeros((4000, 1), dtype=numpy.float32)
    impulse[2000, 0] = 0.9
    return {"short": short, "long": long_track, "sine": sine, "impulse": impulse}


def _whole_array_db(samples):
    oversampled = scipy.signal.resample_poly(samples, TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    return 20.0 * math.log10(float(numpy.abs(oversampled).max()))


@pytest.mark.parametrize("name", ["short", "long", "sine", "impulse"])
def test_true_peak_agrees_with_a_whole_array_oversample(name):
    samples = _tracks()[name]
    reference = _whole_array_db(samples)
    envelope = true_peak_envelope(samples.T)
    assert 20.0 * math.log10(float(envelope.max())) == pytest.approx(
        reference, abs=TOLERANCE_DB
    )
    assert true_peak_dbfs(samples) == pytest.approx(reference, abs=TOLERANCE_DB)


def test_true_peak_of_nothing_or_silence_is_the_floor():
    assert true_peak_dbfs(numpy.zeros((0, 2))) == SILENCE_DBFS
    assert true_peak_dbfs(None) == SILENCE_DBFS
    assert true_peak_dbfs(numpy.zeros((500, 1))) == SILENCE_DBFS


def test_resample_waveform_length_is_exact_716():
    """PyAV's flush rounds up; the track must be round(n * target / rate)."""
    import numpy

    from dw.dsp import resample_waveform

    for n, src, dst in [(330667, 32000, 44100), (123457, 24000, 44100)]:
        out = resample_waveform(numpy.zeros((2, n), dtype="float32"), src, dst)
        assert out.shape[1] == int(round(n * dst / src))


def test_spectral_centroid_of_a_sine_is_its_frequency():
    time = numpy.arange(44100) / 44100
    sine = numpy.sin(2 * math.pi * 1000.0 * time)[None, :]

    assert spectral_centroid(sine, 44100) == pytest.approx(1000.0, rel=0.01)


def test_spectral_centroid_of_white_noise_is_a_quarter_of_the_rate():
    noise = numpy.random.default_rng(3).standard_normal((1, 44100))

    assert spectral_centroid(noise, 44100) == pytest.approx(11025.0, rel=0.03)


def test_spectral_centroid_averages_channels():
    time = numpy.arange(44100) / 44100
    both = numpy.stack([numpy.sin(2 * math.pi * f * time) for f in (1000.0, 3000.0)])

    assert spectral_centroid(both, 44100) == pytest.approx(2000.0, rel=0.01)


def test_spectral_centroid_of_silence_is_zero(recwarn):
    assert spectral_centroid(numpy.zeros((2, 4410)), 44100) == 0.0
    assert not recwarn.list
