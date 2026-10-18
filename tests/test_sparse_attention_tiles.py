"""The H3 sparse-attention tile map (#931): packed rows -> 4x4x4 video tiles."""

import pytest
import torch

from dw.sparse_attention import SparseLayoutError, TileMap, build_tile_map
from tests.sparse_layouts import _layout


def test_t2va_grid_and_tiles():
    tags, pos = _layout(text=356, ref_rows=0, audio=414, t=37, h=17, w=30)
    tm = build_tile_map(pos, tags, "cpu")
    assert isinstance(tm, TileMap)
    assert tm.video_grid == (37, 17, 30)
    assert tm.block_len.sum().item() == tm.rows == len(tags)
    assert tm.num_prefix_tiles == 13  # 356 -> 6 tiles, 414 -> 7
    assert len(tm.block_len) - tm.num_prefix_tiles == 10 * 5 * 8
    assert torch.unique(tm.dst).numel() == tm.rows
    assert (tm.block_len <= 64).all()


def test_ref2va_reference_rows_in_prefix():
    tags, pos = _layout(text=356, ref_rows=5506, audio=414, t=37, h=24, w=42)
    tm = build_tile_map(pos, tags, "cpu")
    assert tm.video_grid == (37, 24, 42)
    ref = torch.arange(356, 356 + 5506)
    assert (tm.dst[ref] < tm.num_prefix_tiles * 64).all()


def test_prefix_tiles_are_segment_pure():
    tags, pos = _layout(text=70, ref_rows=0, audio=70, t=4, h=4, w=4)
    tm = build_tile_map(pos, tags, "cpu")
    # 70 text -> 64 + 6, 70 audio -> 64 + 6: four prefix tiles, never one mixed
    assert tm.block_len[: tm.num_prefix_tiles].tolist() == [64, 6, 64, 6]


def test_rows_within_a_video_tile_are_spatially_local():
    tags, pos = _layout(text=8, ref_rows=0, audio=8, t=8, h=8, w=8)
    tm = build_tile_map(pos, tags, "cpu")
    video = pos[-8 * 8 * 8 :]
    ranked = torch.stack(
        [torch.unique(video[:, a], return_inverse=True)[1] for a in range(3)], dim=1
    )
    tile_of = tm.dst[-8 * 8 * 8 :] // 64
    assert tile_of.min().item() == tm.num_prefix_tiles
    for tile in tile_of.unique().tolist():
        rows = ranked[tile_of == tile]
        # every full tile spans at most 4 distinct values per axis
        assert rows.shape[0] == 64
        for axis in range(3):
            assert rows[:, axis].unique().numel() <= 4


def test_not_a_full_grid_raises():
    tags, pos = _layout(text=8, ref_rows=0, audio=8, t=4, h=4, w=4)
    with pytest.raises(SparseLayoutError):
        build_tile_map(pos[:-1], tags[:-1], "cpu")


def test_no_target_video_raises():
    tags = torch.ones(128, dtype=torch.long)
    with pytest.raises(SparseLayoutError):
        build_tile_map(torch.zeros(128, 3, dtype=torch.float64), tags, "cpu")
