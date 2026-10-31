"""MiniMax-H3 regional edit: re-denoising a `source_video` inside a mask while
everything outside it stays the source (#1062).

The refine schedule (dw/pipeline_processors/h3_hold.py) re-denoises a whole
canvas from `refine_strength`. A regional edit is that refine with most rows
held: a `mask_video` or `mask_image` - white regenerates - is binarized at
`MASK_THRESHOLD`, max-pooled onto the transformer's patch-row grid (a cell any
white pixel touches regenerates) and dilated by `mask_grow` cells. Every other
generated row is a kept row, written back after each denoise step to the source
noised to that step's sigma, with the noise the refine block re-noised the
canvas with. At the last step the sigma is 0 and a kept row is the source
latent exactly; the decode still round-trips it through the VAE.

Two blocks go into each of H3's core-denoise sequences, always, as no-ops when
`source_video` is not passed:

- `DwH3SourceLatentsStep`, before `set_timesteps` (after the hold block) - VAE-
  encodes `source_video` over the generated video rows of `latents`, and keeps
  the clean rows (`SOURCE_ROWS`) and which ones the mask keeps (`KEEP_ROWS`);
- `DwH3MaskBlendStep`, a sub-block of the `denoise` loop after `update` -
  writes the kept rows back after every step. Conditioning rows are never
  touched.

The refine block (unchanged otherwise) hands its re-noise on as
`REFINE_ROW_NOISE`. This module puts the blocks in and holds the mask-to-rows
arithmetic (numpy, with Pillow's luma conversion, torch's adaptive max pool and
scipy's dilation imported where they run); the blocks are dw/pipeline_processors/h3_mask_steps.py,
imported in `blocks()` only, since it imports diffusers when it loads (#790).
The argument rules shared with validation are
dw/pipeline_processors/h3_rules.py (`mask_problems`, `media_problems`).
"""

import logging

import numpy as np

from .h3_hold import core_denoise_sequences
from .h3_rules import (
    BLEND_AFTER,
    BLEND_BLOCK,
    MASK_THRESHOLD,
    REFINE_BEFORE,
    SOURCE_BEFORE,
    SOURCE_BLOCK,
)

logger = logging.getLogger("dw")


def blocks():
    """(DwH3SourceLatentsStep, DwH3MaskBlendStep) - the classes in
    dw/pipeline_processors/h3_mask_steps.py, imported here on first use because
    that module imports diffusers when it loads (#790)."""
    from . import h3_mask_steps

    return h3_mask_steps.DwH3SourceLatentsStep, h3_mask_steps.DwH3MaskBlendStep


def _loop(sequence, prefix):
    """The sequence's denoise loop, or None - the step `refine` anchors to."""
    loop = sequence.sub_blocks.get(prefix + REFINE_BEFORE)
    return loop if hasattr(loop, "sub_blocks") else None


def insert_mask_blocks(pipeline):
    """Put the source and blend blocks into every H3 core-denoise sequence.

    Idempotent - a block already in place is left alone. Returns whether the
    pipeline now takes a regional edit. The two go in as a pair: a sequence
    missing either anchor (a diffusers that renamed `set_timesteps`, the
    `denoise` loop or its `update`) gets neither, with a warning, and a
    pipeline left with none refuses `source_video` at the call.
    """
    source_block, blend_block = blocks()
    inserted = False
    for prefix, sequence in core_denoise_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        source_name, source_before = prefix + SOURCE_BLOCK, prefix + SOURCE_BEFORE
        loop = _loop(sequence, prefix)
        loop_names = list(loop.sub_blocks) if loop is not None else []
        if source_name in sub_blocks and BLEND_BLOCK in loop_names:
            inserted = True
            continue
        if source_before not in sub_blocks or BLEND_AFTER not in loop_names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{source_before}' step "
                f"or '{prefix}{REFINE_BEFORE}' loop with an '{BLEND_AFTER}' to "
                f"anchor the regional edit to, so it cannot take a source_video"
            )
            continue
        if source_name not in sub_blocks:
            names = list(sub_blocks)
            sub_blocks.insert(source_name, source_block(), names.index(source_before))
        if BLEND_BLOCK not in loop_names:
            loop.sub_blocks.insert(
                BLEND_BLOCK, blend_block(), loop_names.index(BLEND_AFTER) + 1
            )
        inserted = True
    return inserted


def edits_regions(pipeline):
    """Whether a pipeline's block graph carries the regional edit."""
    for prefix, sequence in core_denoise_sequences(pipeline):
        loop = _loop(sequence, prefix)
        if (
            prefix + SOURCE_BLOCK in sequence.sub_blocks
            and loop is not None
            and BLEND_BLOCK in loop.sub_blocks
        ):
            return True
    return False


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
