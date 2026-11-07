"""The SelfLift latent arithmetic (#1195; the method is
dw/pipeline_processors/h3_selflift.py's docstring, which re-exports these
names): packing latents into rows, the half canvas, the lift itself, and the
full canvas's packed layout shrunk onto the small one. The loop block
(`h3_selflift_steps.py`) reads them from here, so the block never imports its
importer - the layering `h3_anchor_stats.py` and `h3_mask_cells.py` set.
Torch only: no diffusers.
"""

import torch
import torch.nn.functional as F

from .h3_rules import SELFLIFT_INPUT

# 0.5 only: a quarter of the rows, and the paper's and Directors-Cut's scale
START_SCALES = (0.5,)


def scale_factor(settings):
    """The whole number of full-canvas cells per half-canvas cell, per axis."""
    return round(1 / settings.get("start_scale", START_SCALES[0]))


# The latents - packing, the half canvas and the lift


def patchify(latents, patch):
    """(1, C, F, H, W) latents as packed rows, frame-major then row-major:
    diffusers' `patchify_video_latents`, for a batch of one."""
    pt, ph, pw = patch
    _, channels, frames, height, width = latents.shape
    rows = latents.reshape(
        1, channels, frames // pt, pt, height // ph, ph, width // pw, pw
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return rows.reshape(-1, channels * pt * ph * pw).contiguous()


def unpatchify(rows, frames, height, width, patch):
    """`patchify`'s inverse: packed rows as (1, C, frames, height, width)."""
    pt, ph, pw = patch
    channels = rows.shape[1] // (pt * ph * pw)
    latents = rows.reshape(
        frames // pt, height // ph, width // pw, channels, pt, ph, pw
    )
    latents = latents.permute(3, 0, 4, 1, 5, 2, 6)
    return latents.reshape(1, channels, frames, height, width)


def _per_frame(latents, operation):
    """Apply a 2-d `operation` to every frame of (1, C, F, H, W)."""
    _, channels, frames, height, width = latents.shape
    flat = latents.permute(0, 2, 1, 3, 4).reshape(frames, channels, height, width)
    out = operation(flat)
    return out.reshape(1, frames, channels, *out.shape[-2:]).permute(0, 2, 1, 3, 4)


def shrink_clean(latents, factor):
    """Clean latents on the small canvas: the mean of each f x f block."""
    return _per_frame(latents, lambda x: F.avg_pool2d(x, factor))


def shrink_noise(noise, factor):
    """Unit noise on the small canvas: f x the mean of each f x f block, so
    i.i.d. unit-variance noise stays i.i.d. unit-variance."""
    return shrink_clean(noise, factor) * factor


def _up_nearest(latents, factor):
    return _per_frame(
        latents, lambda x: F.interpolate(x, scale_factor=factor, mode="nearest")
    )


def detail_noise(noise, factor):
    """The full-canvas noise less its block means: the high band the small
    canvas never saw, variance 1 - 1/f^2, summing to zero on every block."""
    return noise - _up_nearest(shrink_clean(noise, factor), factor)


def upsample(latents, factor, mode="bilinear"):
    """Small-canvas latents at the full canvas, per frame."""
    if mode == "nearest":
        return _up_nearest(latents, factor)
    return _per_frame(
        latents,
        lambda x: F.interpolate(
            x, scale_factor=factor, mode="bilinear", align_corners=False
        ),
    )


def lift(latents, velocity, sigma, noise, factor, mode="bilinear"):
    """The small canvas's latents at `sigma`, lifted to the full canvas at
    the same sigma: `(1 - sigma) up(x0) + sigma E`. `velocity` is the last
    (data-ward) prediction, `noise` the call's full-canvas draw. All
    (1, C, F, h, w) small or (1, C, F, H, W) full; float32.

    With `mode` bilinear, pool(up(x0)) is not x0 (the interpolation mixes
    neighbouring blocks), so the low band shifts slightly; nearest is exact.
    E is unit i.i.d. only as far as the model's implied eps is unit noise:
    an approximation, close at sigma about 0.86."""
    latents, velocity = latents.float(), velocity.float()
    denoised = latents + sigma * velocity
    trajectory_noise = denoised - velocity
    lifted_noise = _up_nearest(trajectory_noise, factor) / factor + detail_noise(
        noise.float(), factor
    )
    return (1.0 - sigma) * upsample(denoised, factor, mode) + sigma * lifted_noise


# The layout - the full canvas's packed sequence on the small canvas


def shrink_layout(fields, plan, rows_per_frame, full_grid, half_grid):
    """The packed layout of the small canvas, derived from the full one.

    `fields` holds `position_ids` (S, 3), `token_tags` and the three index
    tensors; `plan` is the per-step `(timesteps, timestep_indices)` list;
    `full_grid` and `half_grid` are one frame's (h, w) rotary coordinates on
    each canvas, in row order. Every run of `rows_per_frame` video rows -
    which must sit together in the sequence on the full grid - becomes one
    half-grid frame at the same time, tag and timestep; text rows are kept;
    audio rows keep their time and move a width pin on the full grid's edge to
    the half grid's. Returns `(fields, plan)`, other `fields` entries passed
    through.

    Raises:
        ValueError: If the video rows are not whole frames of the full grid
    """
    positions = fields["position_ids"]
    tags = fields["token_tags"]
    video = fields["video_indices"]
    audio = fields["audio_indices"]
    text = fields["text_indices"]
    device = positions.device
    full_grid = full_grid.to(device, positions.dtype)
    half_grid = half_grid.to(device, positions.dtype)
    half_rows = half_grid.shape[0]
    if video.numel() % rows_per_frame:
        raise ValueError(
            f"{SELFLIFT_INPUT}: {video.numel()} video rows are not whole frames "
            f"of {rows_per_frame} rows"
        )
    frames = video.reshape(-1, rows_per_frame)
    if not bool(
        (frames == frames[:, :1] + torch.arange(rows_per_frame, device=device)).all()
    ):
        raise ValueError(
            f"{SELFLIFT_INPUT}: a frame's video rows do not sit together in the "
            f"sequence"
        )
    if not torch.equal(
        positions[frames][:, :, 1:], full_grid.expand(len(frames), -1, -1)
    ):
        raise ValueError(
            f"{SELFLIFT_INPUT}: a video frame is not on the call's canvas grid "
            f"(a reference or a guide at another size) - it cannot be shrunk"
        )
    length = positions.shape[0]
    counts = torch.ones(length, dtype=torch.long, device=device)
    counts[video] = 0
    counts[frames[:, 0]] = half_rows
    starts = counts.cumsum(0) - counts
    new_length = int(counts.sum())
    singles = torch.cat([text, audio])

    new_positions = positions.new_zeros((new_length, 3))
    new_positions[starts[singles]] = positions[singles]
    width = positions[audio, 2]
    moved = new_positions[starts[audio]]
    moved[width == full_grid[:, 1].min(), 2] = half_grid[:, 1].min()
    moved[width == full_grid[:, 1].max(), 2] = half_grid[:, 1].max()
    new_positions[starts[audio]] = moved
    new_video = (
        starts[frames[:, 0]][:, None] + torch.arange(half_rows, device=device)
    ).reshape(-1)
    new_positions[new_video, 0] = positions[frames[:, 0], 0].repeat_interleave(
        half_rows
    )
    new_positions[new_video, 1:] = half_grid.repeat(len(frames), 1)

    def per_row(values):
        out = values.new_empty(new_length)
        out[starts[singles]] = values[singles]
        out[new_video] = values[frames[:, 0]].repeat_interleave(half_rows)
        return out

    new_fields = dict(fields)
    new_fields.update(
        position_ids=new_positions,
        token_tags=per_row(tags),
        video_indices=new_video,
        audio_indices=starts[audio],
        text_indices=starts[text],
    )
    new_plan = []
    for timesteps, indices in plan:
        if not bool((indices[frames] == indices[frames[:, :1]]).all()):
            raise ValueError(
                f"{SELFLIFT_INPUT}: a video frame's rows run at different timesteps"
            )
        new_plan.append((timesteps, per_row(indices)))
    return new_fields, new_plan
