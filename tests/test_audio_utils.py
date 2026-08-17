"""
Unit tests for waveform utilities - shape normalization, frame-aligned
slicing, and seam joining.
"""

import numpy
import pytest
import torch

from dw.tasks.audio_utils import (
    as_channels_samples,
    crossfade_concat,
    equal_power_crossfade_join,
    frames_to_samples,
    load_audio,
    slice_samples,
)


class TestAsChannelsSamples:
    def test_a_mono_vector_gains_a_channel_axis(self):
        assert as_channels_samples(numpy.zeros(100)).shape == (1, 100)

    def test_channels_first_passes_through(self):
        assert as_channels_samples(numpy.zeros((2, 100))).shape == (2, 100)

    def test_samples_first_is_transposed(self):
        assert as_channels_samples(numpy.zeros((100, 2))).shape == (2, 100)

    def test_a_one_item_batch_unwraps(self):
        # the shape MiniMax H3 generates: (1, 2, samples)
        assert as_channels_samples(torch.zeros(1, 2, 100)).shape == (2, 100)

    def test_a_torch_tensor_converts_to_float32_numpy(self):
        result = as_channels_samples(torch.zeros(2, 50, dtype=torch.bfloat16))

        assert isinstance(result, numpy.ndarray)
        assert result.dtype == numpy.float32

    def test_a_multi_item_batch_raises(self):
        with pytest.raises(ValueError, match="batch"):
            as_channels_samples(numpy.zeros((3, 2, 100)))

    def test_too_many_dimensions_raise(self):
        with pytest.raises(ValueError, match="dimensions"):
            as_channels_samples(numpy.zeros((1, 1, 2, 100)))


class TestFramesToSamples:
    def test_whole_seconds(self):
        assert frames_to_samples(24, 24, 44100) == 44100

    def test_rounds_to_the_nearest_sample(self):
        assert frames_to_samples(1, 24, 44100) == 1838  # 1837.5 rounds up

    def test_zero_frames_is_zero_samples(self):
        assert frames_to_samples(0, 24, 44100) == 0


class TestSliceSamples:
    def test_an_interior_slice(self):
        waveform = numpy.arange(20, dtype=numpy.float32).reshape(1, 20)

        piece = slice_samples(waveform, 5, 10)

        assert piece.shape == (1, 10)
        assert piece[0, 0] == 5

    def test_a_slice_past_the_end_zero_pads(self):
        waveform = numpy.ones((2, 100), dtype=numpy.float32)

        piece = slice_samples(waveform, 80, 50)

        assert piece.shape == (2, 50)
        assert piece[:, :20].min() == 1
        assert piece[:, 20:].max() == 0

    def test_a_slice_entirely_past_the_end_is_silence(self):
        waveform = numpy.ones((1, 100), dtype=numpy.float32)

        piece = slice_samples(waveform, 200, 30)

        assert piece.shape == (1, 30)
        assert piece.max() == 0


class TestEqualPowerCrossfadeJoin:
    def test_the_total_duration_is_preserved(self):
        previous = numpy.ones((2, 1000), dtype=numpy.float32)
        head = numpy.ones((2, 200), dtype=numpy.float32)
        following = numpy.ones((2, 800), dtype=numpy.float32)

        joined = equal_power_crossfade_join(previous, head, following, 16000, 10)

        assert joined.shape == (2, 1800)

    def test_the_window_is_clamped_to_the_head_material(self):
        previous = numpy.ones((1, 1000), dtype=numpy.float32)
        head = numpy.full((1, 50), 2.0, dtype=numpy.float32)
        following = numpy.full((1, 800), 2.0, dtype=numpy.float32)

        # 1 second requested, only 50 samples of head available
        joined = equal_power_crossfade_join(previous, head, following, 16000, 1000)

        assert joined.shape == (1, 1800)
        # untouched right up to the clamped window
        assert joined[0, 949] == pytest.approx(1.0)
        assert joined[0, 950] == pytest.approx(1.0)  # cos(0) blend of 1.0

    def test_constant_power_across_the_fade_of_uncorrelated_signals(self):
        rng = numpy.random.default_rng(0)
        previous = rng.standard_normal((1, 4000), dtype=numpy.float32)
        head = rng.standard_normal((1, 1000), dtype=numpy.float32)
        following = rng.standard_normal((1, 100), dtype=numpy.float32)

        joined = equal_power_crossfade_join(previous, head, following, 16000, 50)

        # 800-sample fade window: power stays near unity, no dip or spike
        window = joined[0, 3200:4000]
        assert 0.8 < float(numpy.mean(window**2)) < 1.2

    def test_no_head_material_declicks_the_seam(self):
        previous = numpy.ones((1, 1000), dtype=numpy.float32)
        following = numpy.ones((1, 1000), dtype=numpy.float32)
        empty_head = numpy.zeros((1, 0), dtype=numpy.float32)

        joined = equal_power_crossfade_join(
            previous,
            following=following,
            head=empty_head,
            sample_rate=16000,
            crossfade_ms=75,
        )

        assert joined.shape == (1, 2000)
        # the seam samples were ramped down and up
        assert joined[0, 999] < 0.1
        assert joined[0, 1000] < 0.1
        assert joined[0, 0] == 1.0
        assert joined[0, -1] == 1.0

    def test_mono_is_tiled_up_to_match_stereo(self):
        previous = numpy.ones((2, 500), dtype=numpy.float32)
        head = numpy.ones((1, 100), dtype=numpy.float32)
        following = numpy.ones((1, 400), dtype=numpy.float32)

        joined = equal_power_crossfade_join(previous, head, following, 16000, 5)

        assert joined.shape == (2, 900)


class TestCrossfadeConcat:
    def test_each_seam_overlaps_by_the_fade_window(self):
        first = numpy.ones((1, 1000), dtype=numpy.float32)
        second = numpy.ones((1, 1000), dtype=numpy.float32)

        # 10ms at 16kHz = 160 samples of overlap
        result = crossfade_concat([first, second], 16000, 10)

        assert result.shape == (1, 2000 - 160)

    def test_a_single_waveform_passes_through(self):
        waveform = numpy.ones((2, 100), dtype=numpy.float32)

        assert crossfade_concat([waveform], 16000, 10).shape == (2, 100)

    def test_input_shapes_are_normalized(self):
        result = crossfade_concat([torch.zeros(1, 2, 200), numpy.zeros(100)], 16000, 0)

        assert result.shape == (2, 300)

    def test_no_waveforms_raise(self):
        with pytest.raises(ValueError, match="No waveforms"):
            crossfade_concat([], 16000, 10)


class TestLoadAudio:
    def test_loads_a_local_wav(self, tmp_path):
        import soundfile

        path = tmp_path / "tone.wav"
        soundfile.write(path, numpy.zeros((100, 2), dtype=numpy.float32), 16000)

        waveform, sample_rate = load_audio(str(path))

        assert waveform.shape == (2, 100)
        assert sample_rate == 16000

    def test_a_disallowed_extension_raises(self, tmp_path):
        path = tmp_path / "audio.exe"
        path.write_bytes(b"not audio")

        with pytest.raises(Exception):
            load_audio(str(path))
