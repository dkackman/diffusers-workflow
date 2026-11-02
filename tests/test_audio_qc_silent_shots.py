"""#1163: a cut whose shots are all silent_by_design gets no near-silent warning."""

from types import SimpleNamespace

import pytest

from dw import audio_qc


@pytest.mark.parametrize(
    "shots, expected",
    [
        (None, False),
        ([], False),
        ([{"silent_by_design": True}, {"silent_by_design": True}], True),
        ([{"silent_by_design": True}, {"name": "x"}], False),
        ([{"silent_by_design": True}, {"silent_by_design": False}], False),
    ],
)
def test_all_shots_silent_by_design(shots, expected):
    assert audio_qc.all_shots_silent_by_design(shots) is expected


def _run(shots, monkeypatch):
    monkeypatch.setattr(
        audio_qc,
        "_probe_written_media",
        lambda path: {"mean_dbfs": -120.0, "peak_dbfs": -120.0},
    )
    monkeypatch.setattr(audio_qc, "remeasure_shots_after_mux", lambda *a, **k: None)
    monkeypatch.setattr(
        audio_qc, "warn_if_written_above_full_scale", lambda *a, **k: None
    )
    seen = []
    monkeypatch.setattr(audio_qc, "emit_warning", lambda msg, **kw: seen.append(msg))
    audio_qc.check_written_media(
        "/tmp/cut.mp4",
        SimpleNamespace(shots=shots),
        "video/mp4",
        video_fps=lambda: 24,
        consumed_by_normalizer=False,
        headroom_warned=False,
        predicted_peak_dbfs=None,
    )
    return seen


def test_all_silent_shots_emit_no_near_silent_warning(monkeypatch):
    shots = [{"silent_by_design": True}, {"silent_by_design": True}]
    assert _run(shots, monkeypatch) == []


def test_a_mixed_cut_still_warns(monkeypatch):
    shots = [{"silent_by_design": True}, {"name": "b"}]
    assert any("near-silent" in m for m in _run(shots, monkeypatch))
