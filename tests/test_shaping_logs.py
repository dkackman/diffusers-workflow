"""#564: shaping arguments leave a log line saying what they did."""

import numpy

from dw.tasks import audio_utils, joins
from dw.tasks.audio_utils import fade_audio, resample_audio, slice_audio
from dw.tasks.joins import _declick_join


def _capture(monkeypatch):
    lines = []
    # the shaping commands log from audio_utils, the declick from joins
    for module in (audio_utils, joins):
        monkeypatch.setattr(
            module, "emit_log", lambda msg, **kw: lines.append((msg, kw))
        )
    return lines


def test_explicit_seam_fade_logs(monkeypatch):
    lines = _capture(monkeypatch)
    a = numpy.ones((1, 4800), dtype=numpy.float32)
    _declick_join(a, a, 48000, 60, seam=2)
    assert "seam 2" in lines[0][0] and lines[0][1]["fade_ms"] == 60.0


def test_default_declick_is_silent(monkeypatch):
    lines = _capture(monkeypatch)
    a = numpy.ones((1, 4800), dtype=numpy.float32)
    _declick_join(a, a, 48000)
    assert lines == []


def test_fade_out_logs(monkeypatch):
    lines = _capture(monkeypatch)
    fade_audio(
        numpy.ones((1, 48000), dtype=numpy.float32), fade_out_ms=300, sample_rate=48000
    )
    assert lines[0][1]["fade_out_ms"] == 300.0


def test_slice_offset_logs(monkeypatch):
    lines = _capture(monkeypatch)
    slice_audio(
        numpy.ones((1, 48000), dtype=numpy.float32),
        start_frame=6,
        num_frames=12,
        fps=24,
        sample_rate=48000,
    )
    assert lines[0][1]["start_seconds"] == 0.25


def test_noop_resample_says_so(monkeypatch):
    lines = _capture(monkeypatch)
    resample_audio(numpy.ones((1, 4800), dtype=numpy.float32), 44100, sample_rate=44100)
    assert "unchanged" in lines[0][0]
