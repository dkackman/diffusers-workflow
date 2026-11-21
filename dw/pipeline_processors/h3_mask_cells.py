"""MiniMax-H3 regional edit: the mask-to-rows arithmetic (#1062).

A `mask_image` or `mask_video` - white regenerates - is binarized at
`MASK_THRESHOLD`, max-pooled onto the transformer's patch-row grid (a cell any
white pixel touches regenerates) and dilated by `mask_grow` cells; the kept
rows are the complement, flat in packed order. numpy only at import: Pillow's
luma conversion, torch's adaptive max pool and scipy's dilation are imported
where they run, so validation can load this without them.

Beside it, what a source or mask must measure to fit the step, asked by the
static check (dw/hold_audio.py) and before the call alike: `media_problems`
(its size and frame count against the canvas), and an audio pass's canvas
and frame retiming (`listen_canvas`, `retimed_indices`).

Split from dw/pipeline_processors/h3_mask.py so that the blocks module
(dw/pipeline_processors/h3_mask_steps.py, which `h3_mask.blocks()` imports on
first use) reads `keep_rows` from here and never imports its own importer -
the `import_cycles` ratchet in scripts/arch_metrics.py.
"""

import numpy as np

from ..variable_constraints import aligned, aligned_down
from .h3_rules import (
    H3_FPS,
    MASK_IMAGE_INPUT,
    MASK_THRESHOLD,
    RENDER_GRID,
    SOURCE_VIDEO_INPUT,
    TILE_GRID,
)


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


def listen_canvas(width, height, scale):
    """The (width, height) an audio pass at `scale` runs its kept rows on: the
    shorter side scaled and snapped to the nearest TILE_GRID multiple, ties
    up, the longer derived from the source's aspect and snapped the same way,
    neither below one cell of the grid. Snapping each side on its own would
    squeeze 960x544 at 0.25 to 256x128, 13% off its aspect."""

    def snapped(side):
        return max(TILE_GRID, int(side / TILE_GRID + 0.5) * TILE_GRID)

    short = snapped(min(width, height) * scale)
    long = snapped(short * max(width, height) / min(width, height))
    return (long, short) if width >= height else (short, long)


def retimed_indices(frame_count, fps):
    """The source frame each conditioning frame of an audio pass shows: the
    source's `frame_count` frames at `fps` resampled in time, nearest frame,
    to H3_FPS over the same duration, the count snapped up onto RENDER_GRID
    as `num_frames` is (the frames past the end repeat the last). The video
    rows then encode a 24 fps picture of the same moments, and the audio
    generated against them covers the source's real duration (R25)."""
    count = aligned(max(1, round(frame_count / fps * H3_FPS)), RENDER_GRID)
    return [min(frame_count - 1, int(i * fps / H3_FPS + 0.5)) for i in range(count)]


def media_problems(
    name, shape, width, height, num_frames=None, fps=None, frames_constraint=None
):
    """Why a `source_video` or mask of `shape` - (frames, height, width) -
    does not fit a `width` x `height` canvas, as a list. `num_frames` is the
    step's frame count, when it is known; a `mask_image` (one frame) fits any.
    `fps` is an audio pass's source rate: at any rate but H3_FPS the source is
    retimed onto the grid (`retimed_indices`), so its own count need not be.
    `frames_constraint` is the workflow's declared `num_frames` constraint
    (`variable_constraints`, the owner of the ceiling), when the caller has
    it: a `source_video` past its `max_frames` is then named too long."""
    ceiling = (frames_constraint or {}).get("max_frames")
    problems = []
    if (width, height) != (shape[2], shape[1]):
        problems.append(
            f"{name} is {shape[2]}x{shape[1]}, and the step renders "
            f"{width}x{height} - they must match"
        )
    # An audio pass at another rate conditions on the source retimed to H3_FPS,
    # so it is that count, not the source's own, the step must render
    expected = shape[0]
    retimed = name == SOURCE_VIDEO_INPUT and fps not in (None, H3_FPS)
    if retimed:
        expected = len(retimed_indices(shape[0], fps))
    if num_frames is not None and name != MASK_IMAGE_INPUT and expected != num_frames:
        if retimed:
            have = (
                f"{name} has {shape[0]} frames at {fps:g} fps ({expected} once "
                f"retimed to {H3_FPS} fps)"
            )
        else:
            have = f"{name} has {shape[0]} frames"
        problem = (
            f"{have}, and the step renders num_frames={num_frames} - they must match"
        )
        if (
            name == SOURCE_VIDEO_INPUT
            and isinstance(ceiling, int)
            and expected > ceiling
        ):
            fits = aligned_down(ceiling, frames_constraint)
            problem += (
                f". This workflow renders at most {ceiling} frames, so this clip "
                f"is too long - trim it to {fits if fits is not None else ceiling} "
                f"frames or fewer"
            )
        problems.append(problem)
    elif (
        num_frames is not None and name == SOURCE_VIDEO_INPUT and fps in (None, H3_FPS)
    ):
        # Off the 17n + 5 grid the pipeline snaps num_frames up and the source
        # is padded to fit, so the mask drifts in time and the output is longer
        # than the source: refused here, where the count can still be changed
        on_grid = aligned(num_frames, RENDER_GRID)
        if on_grid != num_frames:
            below = aligned_down(num_frames, RENDER_GRID)
            problems.append(
                f"{name} has {shape[0]} frames, off the 17n + 5 grid the step "
                f"renders on: the pipeline would render {on_grid} and pad the "
                f"source, drifting the mask - trim the clip to {below} frames or "
                f"pad it to {on_grid}"
            )
    return problems
