"""bench/encode_cadence.py (#1752): the synthetic line-art source, the metric
functions and one real encode-and-measure pass run on a tiny clip, so the
script keeps importing and its numbers keep meaning what the record says."""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import encode_cadence as ec  # noqa: E402


def test_synthetic_source_is_a_deterministic_pan():
    a = ec.synthetic_line_art(frames=4, width=64, height=48, step=3)
    b = ec.synthetic_line_art(frames=4, width=64, height=48, step=3)
    assert a.shape == (4, 48, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    # Frame 1 is frame 0 moved 3 px left: the window pans right over the canvas
    assert np.array_equal(a[1][:, :-3], a[0][:, 3:])
    # Hard black strokes and flat fills both present
    assert (a < 40).all(axis=-1).any()
    assert len(np.unique(a.reshape(-1, 3), axis=0)) > 4


def test_psnr_and_ssim():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert ec.psnr(ref, ref) == float("inf")
    assert ec.ssim(ref, ref) == pytest.approx(1.0)
    noisy = np.clip(ref.astype(int) + rng.integers(-8, 9, ref.shape), 0, 255).astype(
        np.uint8
    )
    assert 25 < ec.psnr(ref, noisy) < 40
    assert ec.ssim(ref, noisy) < 1.0


def test_period_component_finds_a_planted_dip():
    # A 3-frame cadence: every third frame 1.5 dB down, over a slow drift
    t = np.arange(60)
    series = 40 + 0.02 * t - 1.5 * (t % 3 == 2)
    found = ec.period_component(series)
    assert found["period"] == pytest.approx(3.0)
    assert found["depth"] == pytest.approx(1.5, abs=0.15)
    flat = ec.period_component(40 + 0.02 * t)
    assert flat["depth"] < 0.05


def test_verdict_reads_only_dw_paths():
    def row(gap, depth):
        return {"b_gap_db": gap, "period": {"depth": depth}}

    assert (
        ec.verdict({"a_audio_path": row(0.2, 0.1), "c_bf0": row(None, 3.0)})[
            "confirmed"
        ]
        is False
    )
    assert ec.verdict({"b_silent_path": row(1.2, 0.1)})["confirmed"] is True
    assert ec.verdict({"a_audio_path": row(None, 1.0)})["confirmed"] is True


def test_run_source_encodes_four_ways(tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("imageio_ffmpeg")
    frames = ec.synthetic_line_art(frames=12, width=96, height=64)
    record = ec.run_source("tiny", frames, 24, tmp_path)
    assert set(record["encodes"]) == set(ec.ENCODES)
    for name, r in record["encodes"].items():
        assert r["frames"] == 12, name
        assert r["psnr_mean"] > 20, name
        assert r["bytes"] > 0
    # bf=0 really reaches libx264: no B-frames in that file, some in the default
    assert record["encodes"]["c_bf0"]["count"]["B"] == 0
    assert record["encodes"]["a_audio_path"]["count"]["B"] > 0
    # crf 18 really reaches it: a larger file than the default crf 23
    assert (
        record["encodes"]["d_crf18"]["bytes"]
        > record["encodes"]["a_audio_path"]["bytes"]
    )
    assert "confirmed" in record["verdict"]
    assert "tiny" in ec.table({"sources": [record]})
