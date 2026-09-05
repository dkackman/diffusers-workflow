"""
Unit tests for waveform utilities - shape normalization, frame-aligned
slicing, and seam joining.
"""

import numpy
import pytest
import torch

from dw.tasks.audio_utils import (
    as_channels_samples,
    bleed_join,
    crossfade_concat,
    equal_power_crossfade_join,
    frames_to_samples,
    load_audio,
    resample_audio,
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


class TestBleedJoin:
    def test_the_tail_rings_on_over_a_silent_head(self):
        previous = numpy.full((1, 100), 0.5, dtype=numpy.float32)
        following = numpy.zeros((1, 100), dtype=numpy.float32)

        joined = bleed_join(previous, following, 100, 500)  # 50 samples

        assert joined.shape == (1, 200)
        assert joined[0, 100] == pytest.approx(0.5, abs=1e-3)  # full at the seam
        assert joined[0, 149] == pytest.approx(0.0, abs=0.05)  # faded by the end
        assert (joined[0, 150:] == 0.0).all()  # nothing beyond the window

    def test_the_seam_is_continuous(self):
        rng = numpy.random.default_rng(0)
        previous = rng.normal(0, 0.2, (1, 400)).astype(numpy.float32)
        following = numpy.zeros((1, 400), dtype=numpy.float32)

        joined = bleed_join(previous, following, 1000, 100)

        # the reversed tail starts on previous' own last sample, so the step
        # across the seam is no larger than the steps inside the material
        seam_step = abs(joined[0, 400] - joined[0, 399])
        assert seam_step <= numpy.abs(numpy.diff(previous[0])).max()

    def test_it_adds_to_what_the_head_already_carries(self):
        previous = numpy.full((1, 100), 0.5, dtype=numpy.float32)
        following = numpy.full((1, 100), 0.1, dtype=numpy.float32)

        joined = bleed_join(previous, following, 100, 500)

        assert joined[0, 100] == pytest.approx(0.6, abs=1e-3)
        assert joined[0, 199] == pytest.approx(0.1, abs=1e-6)

    def test_neither_side_is_shortened(self):
        previous = numpy.full((2, 300), 0.4, dtype=numpy.float32)
        following = numpy.full((2, 700), 0.0, dtype=numpy.float32)

        assert bleed_join(previous, following, 100, 250).shape == (2, 1000)

    def test_the_window_is_clamped_to_the_material(self):
        previous = numpy.full((1, 10), 0.5, dtype=numpy.float32)
        following = numpy.zeros((1, 400), dtype=numpy.float32)

        joined = bleed_join(previous, following, 100, 5000)  # asks for 500

        assert joined.shape == (1, 410)
        assert (joined[0, 20:] == 0.0).all()

    def test_no_window_falls_back_to_a_declick_join(self):
        previous = numpy.full((1, 100), 0.5, dtype=numpy.float32)
        following = numpy.full((1, 100), 0.5, dtype=numpy.float32)

        assert bleed_join(previous, following, 100, 0).shape == (1, 200)

    def test_mono_is_tiled_up_to_match_stereo(self):
        previous = numpy.full((1, 100), 0.5, dtype=numpy.float32)
        following = numpy.zeros((2, 100), dtype=numpy.float32)

        assert bleed_join(previous, following, 100, 500).shape == (2, 200)


class TestResampleAudio:
    def test_it_scales_the_length_to_the_new_rate(self):
        waveform = numpy.zeros((2, 44100), dtype=numpy.float32)

        result = resample_audio(waveform, 32000, sample_rate=44100)

        assert result.shape == (32000, 2)

    def test_it_returns_samples_by_channels(self):
        waveform = numpy.zeros((1, 44100), dtype=numpy.float32)

        assert resample_audio(waveform, 22050, sample_rate=44100).shape == (22050, 1)

    def test_matching_rates_pass_through_untouched(self):
        waveform = numpy.linspace(-1, 1, 1000, dtype=numpy.float32)[None, :]

        result = resample_audio(waveform, 8000, sample_rate=8000)

        assert result.shape == (1000, 1)
        assert numpy.allclose(result[:, 0], waveform[0])

    def test_a_tone_keeps_its_level_and_duration(self):
        rate, seconds = 44100, 0.5
        t = numpy.arange(int(rate * seconds)) / rate
        tone = numpy.sin(2 * numpy.pi * 440 * t).astype(numpy.float32)[None, :]

        result = resample_audio(tone, 32000, sample_rate=44100)

        assert result.shape[0] == pytest.approx(32000 * seconds, rel=0.01)
        level = float(numpy.sqrt((result[:, 0] ** 2).mean()))
        assert level == pytest.approx(
            float(numpy.sqrt((tone[0] ** 2).mean())), rel=0.05
        )

    def test_a_raw_waveform_needs_its_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            resample_audio(numpy.zeros((1, 100), dtype=numpy.float32), 32000)

    def test_it_accepts_a_torch_waveform(self):
        waveform = torch.zeros(2, 44100)

        assert resample_audio(waveform, 32000, sample_rate=44100).shape == (32000, 2)


class TestFadeAudio:
    def test_fades_end_on_silence_and_leave_the_middle_alone(self):
        from dw.tasks.audio_utils import fade_audio

        track = numpy.ones((2, 1000), dtype=numpy.float32)

        faded = fade_audio(track, fade_in_ms=100, fade_out_ms=200, sample_rate=1000)

        assert faded.shape == (1000, 2)
        assert faded[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert faded[-1, 0] == pytest.approx(0.0, abs=1e-6)
        assert faded[99, 0] == pytest.approx(1.0)
        assert faded[800, 0] == pytest.approx(1.0)
        assert numpy.all(faded[100:800] == 1.0)
        assert numpy.all(numpy.diff(faded[:100, 0]) > 0)
        assert numpy.all(numpy.diff(faded[800:, 0]) < 0)

    def test_the_input_is_not_modified(self):
        from dw.tasks.audio_utils import fade_audio

        track = numpy.ones((1, 100), dtype=numpy.float32)
        fade_audio(track, fade_out_ms=50, sample_rate=1000)

        assert numpy.all(track == 1.0)

    def test_a_fade_longer_than_the_track_is_clamped(self):
        from dw.tasks.audio_utils import fade_audio

        faded = fade_audio(numpy.ones((1, 10)), fade_in_ms=5000, sample_rate=100)

        assert faded.shape == (10, 1)

    def test_a_waveform_needs_a_sample_rate(self):
        from dw.tasks.audio_utils import fade_audio

        with pytest.raises(ValueError, match="sample_rate"):
            fade_audio(numpy.ones((1, 10)))

    def test_negative_fades_are_refused(self):
        from dw.tasks.audio_utils import fade_audio

        with pytest.raises(ValueError, match="negative"):
            fade_audio(numpy.ones((1, 10)), fade_in_ms=-1, sample_rate=100)


class TestNormalizeAudio:
    def test_the_peak_lands_on_the_target(self):
        from dw.tasks.audio_utils import normalize_audio

        track = numpy.array([[0.1, -0.25, 0.05]], dtype=numpy.float32)

        scaled = normalize_audio(track, peak_dbfs=-6.0, sample_rate=100)

        assert scaled.shape == (3, 1)
        assert numpy.abs(scaled).max() == pytest.approx(10 ** (-6 / 20), abs=1e-6)
        # Only the gain changed
        assert scaled[:, 0] / track[0] == pytest.approx(
            numpy.full(3, scaled[1, 0] / track[0, 1])
        )

    def test_silence_is_left_alone(self):
        from dw.tasks.audio_utils import normalize_audio

        assert numpy.all(normalize_audio(numpy.zeros((1, 10)), sample_rate=100) == 0)

    def test_a_target_above_full_scale_is_refused(self):
        from dw.tasks.audio_utils import normalize_audio

        with pytest.raises(ValueError, match="full scale"):
            normalize_audio(numpy.ones((1, 10)), peak_dbfs=1.0, sample_rate=100)


class TestAudioTasksTakeAnAudioVideo:
    """Every audio task accepts the video an earlier step generated with its
    soundtrack, and takes the sample rate that video carries."""

    def video(self, level=1.0, rate=100, samples=400):
        from dw.result import AudioVideo

        return AudioVideo(
            [], numpy.full((2, samples), level, dtype=numpy.float32), rate
        )

    def test_slice_audio_takes_the_rate_from_the_video(self):
        from dw.tasks.audio_utils import slice_audio

        sliced = slice_audio(self.video(), start_seconds=1.0, duration_seconds=2.0)

        assert sliced.shape == (200, 2)

    def test_a_given_rate_overrides_the_video_rate(self):
        from dw.tasks.audio_utils import slice_audio

        sliced = slice_audio(
            self.video(), start_seconds=0.0, duration_seconds=1.0, sample_rate=50
        )

        assert sliced.shape == (50, 2)

    def test_resample_audio_takes_the_rate_from_the_video(self):
        from dw.tasks.audio_utils import resample_audio

        assert resample_audio(self.video(), target_sample_rate=100).shape == (400, 2)

    def test_fade_and_normalize_take_a_video(self):
        from dw.tasks.audio_utils import fade_audio, normalize_audio

        faded = fade_audio(self.video(), fade_out_ms=1000)
        assert faded.shape == (400, 2) and faded[-1, 0] == pytest.approx(0.0, abs=1e-6)

        scaled = normalize_audio(self.video(level=0.5), peak_dbfs=0.0)
        assert numpy.abs(scaled).max() == pytest.approx(1.0)

    def test_crossfade_audio_joins_videos_at_their_own_rate(self):
        from dw.tasks.audio_utils import crossfade_audio

        joined = crossfade_audio([self.video(), self.video()], crossfade_ms=1000)

        # 4 s + 4 s - 1 s overlap = 7 s at 100 Hz
        assert joined.shape == (700, 2)

    def test_crossfade_audio_refuses_mixed_rates(self):
        from dw.tasks.audio_utils import crossfade_audio

        with pytest.raises(ValueError, match="one sample rate"):
            crossfade_audio([self.video(rate=100), self.video(rate=200)])

    def test_crossfade_audio_still_needs_a_rate_for_a_bare_waveform(self):
        from dw.tasks.audio_utils import crossfade_audio

        with pytest.raises(ValueError, match="sample_rate"):
            crossfade_audio([self.video(), numpy.ones((2, 100))])

    def test_a_silent_video_is_refused(self):
        from dw.result import AudioVideo
        from dw.tasks.audio_utils import fade_audio

        with pytest.raises(ValueError, match="carries none"):
            fade_audio(AudioVideo([], None, None), fade_in_ms=10)
