"""MiniMax-H3 regional edit: the mask-to-rows arithmetic (#1062).

A `mask_image` or `mask_video` - white regenerates - is binarized at
`MASK_THRESHOLD`, max-pooled onto the transformer's patch-row grid (a cell any
white pixel touches regenerates) and dilated by `mask_grow` cells; the kept
rows are the complement, flat in packed order. numpy only at import: Pillow's
luma conversion, torch's adaptive max pool and scipy's dilation are imported
where they run, so validation can load this without them.

Split from dw/pipeline_processors/h3_mask.py so that the blocks module
(dw/pipeline_processors/h3_mask_steps.py, which `h3_mask.blocks()` imports on
first use) reads `keep_rows` from here and never imports its own importer -
the `import_cycles` ratchet in scripts/arch_metrics.py.
"""

import numpy as np

from .h3_rules import MASK_THRESHOLD


def binarized(mask):
    """`(n, h, w)` bool - True to regenerate - from a `mask_image` (a PIL image,
    n = 1) or `(n, h, w, 3|4)` / `(n, h, w)` uint8 `mask_video` frames. Every
    frame goes through Pillow's L conversion, the one remove_object's
    `binarize_mask` uses, and is white at or above `MASK_THRESHOLD`, so both
    masks binarize by one rule."""
    from PIL import Image

    if isinstance(mask, Image.Image):
        images = [mask]
    else:
        images = [Image.fromarray(frame) for frame in np.asarray(mask)]
    luma = np.stack([np.asarray(image.convert("L")) for image in images])
    return luma >= MASK_THRESHOLD


def _pool(mask, cells):
    """`mask` max-pooled onto `cells` (t, h, w) cells by torch's adaptive max
    pool, cell i of an axis covering pixels floor(i * n / cells) up to
    ceil((i + 1) * n / cells) - so a pixel on a boundary counts toward both
    cells it touches. torch is imported here, on first use, so that loading
    this module (at pipeline load, and by validation) does not."""
    import torch
    import torch.nn.functional as F

    grid = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
    pooled = F.adaptive_max_pool3d(grid[None, None], cells)[0, 0]
    return pooled.numpy() > 0


def _dilate(grid, grow):
    """`grid` widened by `grow` cells along every axis, diagonals included (a
    cube structuring element), clipped at the edges. scipy reads 0 iterations
    as "until nothing changes", so `grow == 0` returns the grid as it is."""
    if grow == 0:
        return grid
    from scipy.ndimage import binary_dilation

    structure = np.ones((3,) * grid.ndim, dtype=bool)
    return binary_dilation(grid, structure=structure, iterations=grow)


def regenerate_cells(
    mask, latent_frames, latent_height, latent_width, patch_size, grow
):
    """`(t, h, w)` bool over the patch-row grid - True where a row regenerates -
    from an `(n, height, width)` bool pixel mask (n = 1 for a `mask_image`,
    which covers every frame).

    A latent frame covers the pixel frames proportional to it, a patch cell
    `16 * patch` pixels; a cell regenerates if any pixel it covers is white,
    then the grid is dilated by `grow` cells. The grid flattens frame-major then
    row-major, the order `patchify_video_latents` packs rows in.
    """
    patch_t, patch_h, patch_w = patch_size
    cells = (
        latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
    )
    grid = np.asarray(mask, dtype=bool)
    if grid.shape[0] == 1:
        grid = np.repeat(grid, cells[0], axis=0)
    return _dilate(_pool(grid, cells), grow)


def keep_rows(mask, latent_frames, latent_height, latent_width, patch_size, grow):
    """The generated video rows a regional edit keeps, as a flat bool array in
    packed order - the complement of `regenerate_cells`."""
    return ~regenerate_cells(
        mask, latent_frames, latent_height, latent_width, patch_size, grow
    ).reshape(-1)
