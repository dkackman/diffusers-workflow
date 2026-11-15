"""bench/seam_bands.py: the band math on synthetic frames."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import seam_bands  # noqa: E402


def test_bands_and_hard_edge_step():
    t, h, w = 3, 8, 64
    mask = np.zeros((t, h, w), np.uint8)
    mask[1, :, :16] = 255  # union is the left 16 columns, set on one frame only
    source = np.zeros((t, h, w, 3), np.uint8)
    edit = source.copy()
    edit[:, :, :16] = 100  # the edit changes only inside the mask
    rows, step = seam_bands.measure(source, edit, mask, band=16)
    by = {r["band"]: r for r in rows}
    assert by[0]["diff"] == 100 and by[1]["diff"] == 0
    assert by[1]["hi"] == 16 and by[2]["lo"] == 17
    assert step == 0  # band 1 vs 2 both unchanged: no seam outside


def test_step_is_a_number_for_a_hard_seam():
    t, h, w = 2, 4, 64
    mask = np.zeros((t, h, w), np.uint8)
    mask[:, :, :16] = 255
    source = np.zeros((t, h, w, 3), np.uint8)
    edit = source.copy()
    edit[:, :, :32] = 50  # change spills through band 1, stops at its edge
    _, step = seam_bands.measure(source, edit, mask, band=16)
    assert step == 50
