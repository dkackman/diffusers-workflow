"""The latent statistics arithmetic behind MiniMax-H3's `latent_anchor`
(dw/pipeline_processors/h3_anchor.py, which re-exports these names): a (B, C, T,
H, W) latent's per-channel mean and std, and the rescale of one latent onto
another's statistics. The block (`h3_anchor_steps.py`) reads them from here, so
the block never imports its importer - the layering `h3_mask_cells.py` set.
Torch only: no diffusers.
"""

import torch

# The floor a generated channel's std is held to, so a flat channel is moved
# onto the anchor's mean rather than divided by zero
STD_FLOOR = 1e-6
# The dims a (batch, channel, frames, height, width) latent is reduced over
CHANNEL_DIMS = (0, 2, 3, 4)


def channel_statistics(latents):
    """(mean, std) per channel of a (B, C, T, H, W) latent, each shaped
    (1, C, 1, 1, 1), in float32 - over every batch item and latent position."""
    values = latents.to(torch.float32)
    mean = values.mean(dim=CHANNEL_DIMS, keepdim=True)
    std = values.std(dim=CHANNEL_DIMS, keepdim=True, unbiased=False)
    return mean, std


def match_channel_statistics(latents, anchor_mean, anchor_std, strength=1.0):
    """`latents` (B, C, T, H, W) rescaled per channel to `anchor_mean` and
    `anchor_std`: `(x - mean_x) / std_x * std_a + mean_a`, computed in float32
    and returned in the latent's own dtype and device.

    `strength` in [0, 1] blends the match with the input, `x + strength *
    (matched - x)`: 1 is the full match, 0 returns `latents` itself."""
    if strength == 0:
        return latents
    if latents.dim() != 5:
        raise ValueError(
            f"latent_anchor works on a (batch, channel, frames, height, width) "
            f"video latent, got shape {tuple(latents.shape)}"
        )
    values = latents.to(torch.float32)
    mean, std = channel_statistics(values)
    anchor_mean = anchor_mean.to(values.device, torch.float32).reshape(1, -1, 1, 1, 1)
    anchor_std = anchor_std.to(values.device, torch.float32).reshape(1, -1, 1, 1, 1)
    if anchor_mean.shape[1] != values.shape[1]:
        raise ValueError(
            f"latent_anchor has {anchor_mean.shape[1]} latent channels and the "
            f"render {values.shape[1]}"
        )
    matched = (values - mean) / std.clamp_min(STD_FLOOR) * anchor_std + anchor_mean
    if strength != 1.0:
        matched = values + strength * (matched - values)
    return matched.to(latents.dtype)
