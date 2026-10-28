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
from dw.tasks.voice_attribution import (
    STACK_SPANS_LISTED,
    separate_stems,
    stacked_spans,
    stacked_vocals_message,
)

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


def vocals_only(model, mix, device=None, split=True):
    """The whole (normalized) mix comes back as the vocal stem."""
    shares = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1)
    return mix[:, None] * shares


def sung(seconds, stacked=(), rate=8000, quiet=()):
    """A stereo vocal stem, one second per window: the same voice on both
    channels, except in `stacked` windows, where each channel sings its own
    (uncorrelated) line. A `quiet` window sits 30 dB down."""
    t = numpy.arange(rate) / rate
    lead = 0.5 * numpy.sin(2 * numpy.pi * 220 * t)
    double = 0.5 * numpy.sin(2 * numpy.pi * 331 * t)
    windows = []
    for index in range(seconds):
        right = double if index in stacked else lead
        gain = 10 ** (-30 / 20) if index in quiet else 1.0
        windows.append(numpy.stack([lead, right]) * gain)
    return numpy.concatenate(windows, axis=1).astype(numpy.float32), rate


class TestStackedVocals(unittest.TestCase):
    """#947: the vocal stem's stacked (layered stereo) spans."""

    def test_decorrelated_channels_over_a_known_span_are_that_span(self):
        self.assertEqual(stacked_spans(*sung(12, stacked=range(4, 8))), [(4.0, 8.0)])

    def test_identical_channels_never_flag(self):
        self.assertEqual(stacked_spans(*sung(12)), [])

    def test_silence_never_flags(self):
        self.assertEqual(stacked_spans(numpy.zeros((2, 8000 * 10)), 8000), [])

    def test_a_mono_stem_never_flags(self):
        stem, rate = sung(12, stacked=range(4, 8))
        self.assertEqual(stacked_spans(stem[:1], rate), [])

    def test_a_run_shorter_than_the_minimum_span_is_dropped(self):
        self.assertEqual(stacked_spans(*sung(12, stacked={5})), [])
        self.assertEqual(stacked_spans(*sung(12, stacked={5, 6})), [(5.0, 7.0)])

    def test_windows_a_one_window_gap_apart_merge(self):
        stem = sung(12, stacked={2, 3, 5, 6})
        self.assertEqual(stacked_spans(*stem), [(2.0, 7.0)])
        stem = sung(12, stacked={1, 2, 5, 6})
        self.assertEqual(stacked_spans(*stem), [(1.0, 3.0), (5.0, 7.0)])

    def test_a_faint_decorrelated_stretch_is_not_a_second_voice(self):
        stem = sung(12, stacked=range(2, 6), quiet=range(2, 6))
        self.assertEqual(stacked_spans(*stem), [])

    def test_more_than_the_listed_spans_say_how_many_more(self):
        count = STACK_SPANS_LISTED + 2
        stacked = [
            start + offset for start in range(0, count * 4, 4) for offset in (0, 1)
        ]
        spans = stacked_spans(*sung(count * 4, stacked=set(stacked)))
        self.assertEqual(len(spans), count)
        message = stacked_vocals_message(spans)
        self.assertIn("0:00–0:02, 0:04–0:06", message)
        self.assertIn("0:28–0:30 and 2 more;", message)
        self.assertNotIn("0:32", message)

    def test_spans_read_as_minutes_and_seconds(self):
        message = stacked_vocals_message([(79.0, 94.0), (119.0, 129.0)])
        self.assertEqual(
            message,
            "separate_stems: the vocals stem holds stacked (layered stereo) "
            "voices over 1:19–1:34, 1:59–2:09; treat those spans as more than "
            "one voice before processing them as one",
        )


class TestSeparateStemsWarnsStackedVocals(unittest.TestCase):
    """The check runs on the real separate_stems path; only demucs is faked."""

    def setUp(self):
        model_cache.clear_model_cache()

    def separate(self, source):
        with (
            patch.dict(sys.modules, fake_demucs(vocals_only)),
            patch("dw.tasks.voice_attribution.emit_warning") as warn,
        ):
            stems = separate_stems(AudioTrack(source, RATE), device="cpu")
        return stems, warn

    def test_a_stacked_vocal_stem_warns_once_with_its_span(self):
        source, _ = sung(10, stacked=range(2, 5), rate=RATE)
        stems, warn = self.separate(source)
        self.assertEqual(sorted(stems), sorted(SOURCES))
        warn.assert_called_once()
        self.assertEqual(warn.call_args.kwargs["kind"], "stacked_vocals")
        self.assertEqual(warn.call_args.kwargs["command"], "separate_stems")
        self.assertIn("over 0:02–0:05;", warn.call_args.args[0])

    def test_a_single_voice_does_not_warn(self):
        stems, warn = self.separate(sung(10, rate=RATE)[0])
        self.assertEqual(len(stems), 4)
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
