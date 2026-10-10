"""measure_sync: the lag's sign, confidence, and the unmeasurable paths."""

import numpy
import pytest

from dw.task_domains import task_argument_errors
from dw.tasks.measure_sync import measure_sync

SR = 16000


def song(seconds=12.0, seed=1):
    """Noise bursts at irregular times: a signal with a unique alignment."""
    rng = numpy.random.default_rng(seed)
    signal = numpy.zeros(int(seconds * SR))
    for time in numpy.cumsum(rng.uniform(0.25, 0.7, 40)):
        if time + 0.1 >= seconds:
            break
        start = int(time * SR)
        burst = rng.standard_normal(int(0.08 * SR)) * numpy.hanning(int(0.08 * SR))
        signal[start : start + len(burst)] += burst
    return (signal / numpy.abs(signal).max() * 0.5).astype(numpy.float32)[None, :]


def shifted(wave, seconds):
    """wave with its content moved `seconds` later (earlier when negative)."""
    n = int(round(seconds * SR))
    out = numpy.zeros_like(wave)
    if n >= 0:
        out[:, n:] = wave[:, : wave.shape[1] - n]
    else:
        out[:, :n] = wave[:, -n:]
    return out


def run(audio, reference, **kwargs):
    return measure_sync(audio, reference, sample_rate=SR, **kwargs)


class TestLag:
    def test_aligned_audio_has_no_lag_and_no_finding(self):
        wave = song()
        result = run(wave, wave)
        assert result["lag_seconds"] == pytest.approx(0.0, abs=0.01)
        assert result["confidence"] > 0.9
        assert result["findings"] == []

    def test_late_audio_is_a_positive_lag(self):
        wave = song()
        result = run(shifted(wave, 0.8), wave)
        assert result["lag_seconds"] == pytest.approx(0.8, abs=0.02)
        (found,) = result["findings"]
        assert found["rule"] == "audio_out_of_sync"
        assert "late" in found["says"]

    def test_early_audio_is_a_negative_lag(self):
        wave = song()
        result = run(shifted(wave, -0.5), wave)
        assert result["lag_seconds"] == pytest.approx(-0.5, abs=0.02)
        assert "early" in result["findings"][0]["says"]

    def test_a_lag_under_the_threshold_is_not_a_finding(self):
        wave = song()
        result = run(shifted(wave, 0.05), wave)
        assert result["lag_seconds"] == pytest.approx(0.05, abs=0.02)
        assert result["findings"] == []

    def test_a_lag_past_the_search_warns_it_may_be_larger(self):
        wave = song()
        result = run(shifted(wave, 0.52), wave, max_lag_seconds=0.5)
        assert any("edge of the search" in message for message in result["warnings"])

    def test_unrelated_audio_is_low_confidence(self):
        result = run(song(seed=1), song(seed=2), min_confidence=0.9)
        assert any(f["rule"] == "sync_low_confidence" for f in result["findings"])


class TestUnmeasurable:
    def test_a_silent_reference_has_no_lag(self):
        wave = song()
        result = run(wave, numpy.zeros_like(wave))
        assert result["lag_seconds"] is None
        assert result["findings"][0]["rule"] == "sync_unmeasurable"
        assert result["findings"][0]["at"] == {"silent": ["reference"]}

    def test_silent_audio_has_no_lag(self):
        wave = song()
        result = run(numpy.zeros_like(wave), wave)
        assert result["findings"][0]["at"] == {"silent": ["audio"]}

    def test_no_slice_is_refused(self):
        with pytest.raises(ValueError, match="reference"):
            run(song(), None)

    def test_a_non_positive_search_is_refused(self):
        with pytest.raises(ValueError, match="max_lag_seconds"):
            run(song(), song(), max_lag_seconds=0)


def test_registered_with_its_domains():
    from dw.tasks.registry import _COMMAND_INFO

    assert _COMMAND_INFO["measure_sync"]["implementation"].endswith("measure_sync")
    workflow = {
        "steps": [
            {
                "name": "sync",
                "task": {
                    "command": "measure_sync",
                    "arguments": {
                        "audio": "a.wav",
                        "reference": "b.wav",
                        "min_confidence": 2,
                    },
                },
            }
        ]
    }
    errors = task_argument_errors(workflow)
    assert any("min_confidence" in error["path"] for error in errors)
