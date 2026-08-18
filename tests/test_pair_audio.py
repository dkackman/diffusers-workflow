#!/usr/bin/env python3
"""Tests for pairing a video with a soundtrack generated beside it.

A step that works on frames alone - a latent upsampler, an interpolator - drops
the audio a video-with-audio pipeline generated. pair_audio carries it across.
"""

import os
import sys

import numpy
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.result import AudioVideo
from dw.tasks.pair_audio import pair_audio


def _waveform(samples=100, channels=2):
    return numpy.zeros((channels, samples), dtype=numpy.float32)


def test_pairs_frames_with_a_bare_waveform():
    paired = pair_audio(["frame1", "frame2"], _waveform(), sample_rate=24000)

    assert isinstance(paired, AudioVideo)
    assert paired.frames == ["frame1", "frame2"]
    assert paired.sample_rate == 24000
    assert paired.audio.shape == (2, 100)


def test_takes_the_track_and_its_rate_from_another_result():
    """The common case: 'audio': 'previous_result:base_step'."""
    generated = AudioVideo(["low_res"], _waveform(), 16000)

    paired = pair_audio(["upscaled"], generated)

    assert paired.frames == ["upscaled"]
    assert paired.sample_rate == 16000


def test_explicit_rate_wins_over_the_carried_one():
    generated = AudioVideo(["low_res"], _waveform(), 16000)

    paired = pair_audio(["upscaled"], generated, sample_rate=24000)

    assert paired.sample_rate == 24000


def test_replaces_the_soundtrack_of_a_video_that_has_one():
    original = AudioVideo(["frame"], _waveform(samples=10), 24000)
    replacement = AudioVideo(["other"], _waveform(samples=50), 24000)

    paired = pair_audio(original, replacement)

    assert paired.frames == ["frame"]
    assert paired.audio.shape == (2, 50)


def test_normalizes_a_torch_waveform():
    paired = pair_audio(["frame"], torch.zeros(2, 100), sample_rate=24000)

    assert isinstance(paired.audio, numpy.ndarray)
    assert paired.audio.shape == (2, 100)


def test_frame_arrays_are_carried_through_untouched():
    """A long video must not be copied into PIL images just to be paired."""
    frames = numpy.zeros((8, 16, 16, 3), dtype=numpy.uint8)

    paired = pair_audio(frames, _waveform(), sample_rate=24000)

    assert paired.frames is frames


def test_raises_when_the_named_result_carries_no_audio():
    silent = AudioVideo(["frame"], None, None)

    with pytest.raises(ValueError, match="carries none"):
        pair_audio(["frame"], silent)


def test_raises_when_no_sample_rate_can_be_established():
    with pytest.raises(ValueError, match="sample_rate"):
        pair_audio(["frame"], _waveform())


def test_registered_as_a_task_command():
    from dw.tasks.task import _COMMAND_REGISTRY

    assert "pair_audio" in _COMMAND_REGISTRY
