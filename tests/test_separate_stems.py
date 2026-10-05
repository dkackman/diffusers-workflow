"""Tests for #604: separate_stems - htdemucs' four stems, each its own audio
result. demucs itself is faked (no weights, no network); the separator
plumbing around it - resample, normalize, un-normalize, one track per stem,
the MPS fallback - is the real code."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy
import torch

from dw.introspection import describe_task, list_tasks
from dw.media_types import AudioTrack
from dw.tasks import model_cache
from dw.tasks.voice_attribution import separate_stems

SOURCES = ["drums", "bass", "other", "vocals"]
RATE = 44100


class FakeModel:
    sources = SOURCES
    samplerate = RATE
    audio_channels = 2

    def eval(self):
        return self

    def to(self, **kwargs):
        return self


def fake_demucs(apply_model):
    pretrained = types.SimpleNamespace(get_model=lambda name: FakeModel())
    apply = types.SimpleNamespace(apply_model=apply_model)
    return {
        "demucs": types.SimpleNamespace(pretrained=pretrained, apply=apply),
        "demucs.pretrained": pretrained,
        "demucs.apply": apply,
    }


def quarter_split(model, mix, device=None, split=True):
    """Each stem is a distinct share of the (normalized) mix; they sum to it."""
    shares = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 4, 1, 1)
    return mix[:, None] * shares


def mix(seconds=1.0, rate=RATE):
    t = numpy.arange(int(seconds * rate)) / rate
    wave = (0.5 * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32)
    return numpy.stack([wave, wave * 0.5])


class TestSeparateStems(unittest.TestCase):
    def setUp(self):
        model_cache.clear_model_cache()

    def test_returns_each_stem_as_its_own_audio_track(self):
        track = AudioTrack(mix(), RATE)
        with patch.dict(sys.modules, fake_demucs(quarter_split)):
            stems = separate_stems(track, device="cpu")
        self.assertEqual(sorted(stems), sorted(SOURCES))
        for stem in stems.values():
            self.assertIsInstance(stem, AudioTrack)
            self.assertEqual(stem.sample_rate, RATE)
            self.assertEqual(stem.audio.shape, mix().shape)

    def test_stems_sum_back_to_the_mix(self):
        source = mix()
        with patch.dict(sys.modules, fake_demucs(quarter_split)):
            stems = separate_stems(AudioTrack(source, RATE), device="cpu")
        total = sum(stem.audio for stem in stems.values())
        numpy.testing.assert_allclose(total, source, atol=1e-4)
        numpy.testing.assert_allclose(stems["vocals"].audio, source * 0.4, atol=1e-4)

    def test_mps_failure_retries_on_cpu_with_a_warning(self):
        devices = []

        def flaky(model, mixture, device=None, split=True):
            devices.append(device)
            if len(devices) == 1:
                raise RuntimeError("boom")
            return quarter_split(model, mixture)

        with (
            patch.dict(sys.modules, fake_demucs(flaky)),
            patch("dw.get_device_type", return_value="mps"),
            patch("dw.tasks.voice_attribution.emit_warning") as warn,
        ):
            stems = separate_stems(AudioTrack(mix(), RATE), device="mps")
        self.assertEqual(devices, ["mps", "cpu"])
        self.assertEqual(len(stems), 4)
        self.assertEqual(warn.call_args.kwargs["command"], "separate_stems")

    def test_task_is_registered_and_described(self):
        self.assertIn("separate_stems", list_tasks()["commands"])
        names = [p["name"] for p in describe_task("separate_stems")["parameters"]]
        self.assertIn("audio", names)


if __name__ == "__main__":
    unittest.main()
