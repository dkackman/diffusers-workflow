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


def _square_case():
    t, h, w = 2, 128, 128
    mask = np.zeros((t, h, w), np.uint8)
    mask[:, 32:96, 32:96] = 255  # a 64 px square: 32 px deep at its centre
    source = np.zeros((t, h, w, 3), np.uint8)
    rng = np.random.default_rng(0)
    edit = rng.integers(0, 255, (t, h, w, 3), dtype=np.uint8)
    return source, edit, mask


def test_inner_bands_partition_the_inside_by_distance():
    from scipy import ndimage

    source, edit, mask = _square_case()
    union = seam_bands.union_mask(mask)
    idx = seam_bands.band_index(union, band=8, inner=2)
    depth = ndimage.distance_transform_edt(union)
    # every inside pixel is in exactly one of 0, -1, -2, by its depth
    assert set(np.unique(idx[union])) == {0, -1, -2}
    assert ((depth[idx == -1] > 0) & (depth[idx == -1] <= 8)).all()
    assert ((depth[idx == -2] > 8) & (depth[idx == -2] <= 16)).all()
    assert (depth[idx == 0][union[idx == 0]] > 16).all()
    # outside is untouched
    assert np.array_equal(idx[~union], seam_bands.band_index(union, band=8)[~union])

    rows, _ = seam_bands.measure(source, edit, mask, band=8, inner=2)
    by = {r["band"]: r for r in rows}
    assert [r["band"] for r in rows][:3] == [0, -2, -1]
    assert by[0]["pixels"] + by[-1]["pixels"] + by[-2]["pixels"] == int(union.sum())
    assert (by[-1]["lo"], by[-1]["hi"]) == (1, 8)
    assert (by[-2]["lo"], by[-2]["hi"]) == (9, 16)


def test_inner_jump_is_the_drop_across_the_edge():
    t, h, w = 2, 4, 64
    mask = np.zeros((t, h, w), np.uint8)
    mask[:, :, :32] = 255
    source = np.zeros((t, h, w, 3), np.uint8)
    edit = source.copy()
    edit[:, :, :16] = 80  # change stops 16 px inside the mask: an inner edge
    rows, step = seam_bands.measure(source, edit, mask, band=16, inner=1)
    by = {r["band"]: r for r in rows}
    assert by[-1]["diff"] == 0 and by[0]["diff"] == 80
    assert seam_bands.inner_jump(rows) == 0 and step == 0
    edit[:, :, :32] = 80
    rows, _ = seam_bands.measure(source, edit, mask, band=16, inner=1)
    assert seam_bands.inner_jump(rows) == 80


def test_no_inner_leaves_measure_and_the_table_unchanged():
    source, edit, mask = _square_case()
    plain = seam_bands.measure(source, edit, mask, band=16)
    assert seam_bands.measure(source, edit, mask, band=16, inner=0) == plain
    rows, step = plain
    assert [r["band"] for r in rows] == list(range(len(rows)))
    assert "in " not in seam_bands.table(rows, step)
    assert "inner jump" not in seam_bands.table(rows, step)
    # the outside bands and step read the same with --inner on
    inner_rows, inner_step = seam_bands.measure(source, edit, mask, band=16, inner=2)
    assert inner_step == step
    assert [r for r in inner_rows if r["band"] > 0] == [
        r for r in rows if r["band"] > 0
    ]
