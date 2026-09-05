"""Tests for dw.tasks.audio_utils.mix_audio - layering tracks rather than joining them."""

import numpy
import pytest

from dw.tasks.audio_utils import mix_audio


def _tone(samples, level, channels=2):
    return numpy.full((channels, samples), level, dtype=numpy.float32)


class TestMixAudio:
    def test_tracks_are_summed(self):
        mixed = mix_audio([_tone(100, 0.2), _tone(100, 0.3)], sample_rate=44100)
        assert mixed.shape == (100, 2)
        assert mixed[0, 0] == pytest.approx(0.5)

    def test_gains_are_plain_multipliers(self):
        mixed = mix_audio(
            [_tone(100, 0.4), _tone(100, 0.4)], gains=[1.0, 0.5], sample_rate=44100
        )
        assert mixed[0, 0] == pytest.approx(0.6)

    def test_the_shorter_track_is_padded_with_silence(self):
        mixed = mix_audio([_tone(100, 0.5), _tone(40, 0.25)], sample_rate=44100)
        assert mixed.shape == (100, 2)
        assert mixed[0, 0] == pytest.approx(0.75)
        assert mixed[-1, 0] == pytest.approx(0.5)

    def test_mono_is_tiled_up_to_the_widest_track(self):
        mixed = mix_audio(
            [_tone(50, 0.5, channels=1), _tone(50, 0.25, channels=2)], sample_rate=44100
        )
        assert mixed.shape == (50, 2)
        assert mixed[0, 0] == pytest.approx(0.75)
        assert mixed[0, 1] == pytest.approx(0.75)

    def test_the_sum_is_not_rescaled(self):
        mixed = mix_audio([_tone(10, 0.8), _tone(10, 0.8)], sample_rate=44100)
        assert mixed.max() == pytest.approx(1.6)

    def test_an_empty_list_is_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            mix_audio([], sample_rate=44100)

    def test_a_gain_per_track_is_required(self):
        with pytest.raises(ValueError, match="one gain per track"):
            mix_audio([_tone(10, 0.1), _tone(10, 0.1)], gains=[1.0], sample_rate=44100)

    def test_a_raw_waveform_needs_a_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            mix_audio([_tone(10, 0.1)])
