"""Sparse attention for MiniMax H3's generated-video rows (#931).

Sparse here means comfy-kitchen's Sol-Attn: the generated video's rows are
grouped into 4x4x4 (t, h, w) tiles of 64 rows, and each query tile attends only
to the key tiles that matter, instead of every row. Everything that is not the
generated video (text, references, target audio) stays dense.

The layout rule: H3 packs text, then the references in request order, then the
target audio, then the target video last. A reference image carries the same
video tag as the target, so the target video is the trailing run of video rows,
not every video-tagged row. Rows before it go first, in segment-pure tiles
(contiguous runs of one tag, at most 64 rows each) that are always attended in
full; the video tiles follow.

The decided-once rule: whether a pipeline call runs sparse is decided once,
before the first forward, and is all or nothing. A layout check that fails
after that decision raises; it never drops a single forward back to dense,
because a run that mixes the two is worse than either.
"""

from dataclasses import dataclass

import torch

# Rows per tile: 4 x 4 x 4 in (t, h, w)
TILE = 64
_SIDE = 4


class SparseLayoutError(ValueError):
    """The packed rows are not a layout the tile map can describe."""


@dataclass(frozen=True)
class TileMap:
    """Where each packed row lives in the padded, tiled sequence.

    `dst` sends packed row -> padded row; `block_len` is the live row count of
    each tile; the first `num_prefix_tiles` tiles are the dense prefix and the
    rest are video tiles of the `video_grid` (t, h, w) latent grid.
    """

    dst: torch.Tensor
    block_len: torch.Tensor
    num_prefix_tiles: int
    padded_rows: int
    rows: int
    video_grid: tuple[int, int, int]


def build_tile_map(
    position_ids: torch.Tensor, token_tags: torch.Tensor, device
) -> TileMap:
    """Map each packed row to a padded tiled row.

    Every row that is not target video goes first, in segment-pure tiles; the
    target video follows in 4x4x4 tiles of its (t, h, w) position ids.
    """
    tags = token_tags.cpu()
    pos = position_ids.cpu()
    s = tags.numel()
    if pos.shape != (s, 3):
        raise SparseLayoutError(
            f"position_ids {tuple(pos.shape)} do not match {s} token tags"
        )

    # Target video = the trailing run of video rows
    non_video = (tags != 0).nonzero()
    target_start = int(non_video[-1]) + 1 if len(non_video) else 0
    if target_start >= s:
        raise SparseLayoutError("no target video rows at the end of the sequence")

    dst = torch.empty(s, dtype=torch.long)
    live: list[int] = []
    tile = 0

    # Prefix segments are contiguous runs of one tag
    prefix_tags = tags[:target_start]
    breaks = torch.ones(target_start, dtype=torch.bool)
    if target_start > 1:
        breaks[1:] = prefix_tags[1:] != prefix_tags[:-1]
    starts = breaks.nonzero().squeeze(1).tolist() + [target_start]
    for a, b in zip(starts[:-1], starts[1:]):
        for c in range(a, b, TILE):
            n = min(TILE, b - c)
            dst[c : c + n] = tile * TILE + torch.arange(n)
            live.append(n)
            tile += 1
    num_prefix_tiles = tile

    # H3's rotary coordinates are fractional (aspect-normalized space, uneven
    # time), so rank each axis's distinct values to get the integer grid index
    vrows = torch.arange(target_start, s)
    vp = torch.stack(
        [torch.unique(pos[vrows, a], return_inverse=True)[1] for a in range(3)], dim=1
    )
    gt, gh, gw = (vp.max(0).values + 1).tolist()
    if gt * gh * gw != len(vrows):
        raise SparseLayoutError(
            f"target video is not a full grid: {gt}x{gh}x{gw} != {len(vrows)} rows"
        )
    nh, nw = (gh + _SIDE - 1) // _SIDE, (gw + _SIDE - 1) // _SIDE
    tid = ((vp[:, 0] // _SIDE) * nh + vp[:, 1] // _SIDE) * nw + vp[:, 2] // _SIDE
    within = ((vp[:, 0] % _SIDE) * _SIDE + vp[:, 1] % _SIDE) * _SIDE + vp[:, 2] % _SIDE
    order = torch.argsort(tid * TILE + within, stable=True)
    _, counts = torch.unique_consecutive(tid[order], return_counts=True)
    offs = 0
    for n in counts.tolist():
        dst[vrows[order[offs : offs + n]]] = tile * TILE + torch.arange(n)
        live.append(n)
        tile += 1
        offs += n

    if sum(live) != s or torch.unique(dst).numel() != s:
        raise SparseLayoutError("tile map does not cover every row exactly once")
    return TileMap(
        dst=dst.to(device),
        block_len=torch.tensor(live, dtype=torch.int32, device=device),
        num_prefix_tiles=num_prefix_tiles,
        padded_rows=tile * TILE,
        rows=s,
        video_grid=(gt, gh, gw),
    )
