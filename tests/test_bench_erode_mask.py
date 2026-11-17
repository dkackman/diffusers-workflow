"""bench/erode_mask.py: the erosion on synthetic masks, and a real video round trip."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import erode_mask  # noqa: E402


def square_mask(t=2, h=96, w=128, top=20, left=30, side=40):
    mask = np.zeros((t, h, w), bool)
    mask[:, top : top + side, left : left + side] = True
    return mask


@pytest.mark.parametrize("px", [1, 5, 16])
def test_a_square_shrinks_by_n_per_side(px):
    out = erode_mask.erode(square_mask(), px)
    for frame in out:
        rows, cols = np.nonzero(frame)
        assert rows.min() == 20 + px and rows.max() == 20 + 40 - 1 - px
        assert cols.min() == 30 + px and cols.max() == 30 + 40 - 1 - px


def test_n_zero_is_the_identity():
    mask = square_mask()
    assert np.array_equal(erode_mask.erode(mask, 0), mask)


def test_the_frame_border_does_not_erode():
    mask = np.zeros((1, 64, 64), bool)
    mask[0, 32:, :] = True  # a person cut off by the bottom and both sides
    out = erode_mask.erode(mask, 8)[0]
    assert out[40:, :].all() and not out[:40, :].any()


def test_binarize_thresholds_at_128_like_h3_mask_cells():
    from dw.pipeline_processors.h3_mask_cells import binarized

    frames = np.zeros((1, 4, 4, 3), np.uint8)
    frames[0, 0, :] = np.array([127, 128, 129, 255], np.uint8)[:, None]
    assert erode_mask.binarize(frames)[0, 0].tolist() == [False, True, True, True]
    assert np.array_equal(erode_mask.binarize(frames), binarized(frames))


def test_removed_fraction():
    mask = square_mask(t=1)
    out = erode_mask.erode(mask, 10)  # 40x40 -> 20x20
    assert erode_mask.removed_fraction(mask, out) == pytest.approx(0.75)
    assert erode_mask.removed_fraction(mask[:, :0], out[:, :0]) == 0.0


def test_the_script_keeps_size_frame_count_and_fps(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    src = tmp_path / "mask.mp4"
    out = tmp_path / "eroded.mp4"
    mask = square_mask(t=9)
    erode_mask.write_mask(mask, str(src), fps=24)

    run = subprocess.run(
        [sys.executable, str(BENCH / "erode_mask.py"), str(src), str(out), "--px", "4"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "removed" in run.stdout

    frames, fps = erode_mask.read_mask(str(out))
    assert frames.shape == (9, 96, 128, 3)
    assert fps == pytest.approx(24)
    assert np.array_equal(erode_mask.binarize(frames), erode_mask.erode(mask, 4))
