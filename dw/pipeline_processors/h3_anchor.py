"""MiniMax-H3's latent statistics anchor (#1166): `latent_anchor` rescales the
generated video latent, channel by channel, to the mean and standard deviation of
an anchor clip's latent.

A long guide chain drifts in colour and contrast from segment to segment: each
segment's latent statistics wander a little from the last one's, and the guide
carries the drift on. With `latent_anchor` set, the anchor clip is VAE-encoded on
the render's canvas exactly as a guide is (`fit_guide_frames` and diffusers'
`encode_vae_condition`), its per-channel mean and std are taken over every latent
position, and the generated latent is moved onto them:

    x' = (x - mean_x) / std_x * std_a + mean_a      (per VAE latent channel)

The step (`DwH3LatentAnchorStep`, dw/pipeline_processors/h3_anchor_steps.py) runs
right after `after_denoise` has unpacked the video rows and before decode, so it
reads the normalized video latent the decoder denormalizes; audio latents are
never touched. A full match: there is no strength knob.

The block is inserted at load into every t2va/fl2va core-denoise sequence
(`insert_latent_anchor`) and passes the latent through untouched when no
`latent_anchor` is given, so a run without one is bit-identical to stock. The
block narrator (progress.reported_blocks) skips it then - neither named in the job
log nor run - and a call that sets one logs a line naming the anchor's source
(Pipeline._with_latent_anchor, #1181). ref2va gets no block, and `anchor_refusal`
names why.

The concept follows H3MD Director's segment colour anchoring (concept only, no
code; docs/ATTRIBUTIONS.md). The block module imports diffusers when it loads, so
it is imported in `anchor_block()` only, as h3_guides does its blocks (#790).
"""

import logging

import torch

from .h3_hold import core_denoise_sequences
from .h3_guides import guide_frames_array
from .h3_rules import (
    ANCHOR_AFTER,
    ANCHOR_BLOCK,
    LATENT_ANCHOR_INPUT,
    snap_guide_length,
)

logger = logging.getLogger("dw")

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


def match_channel_statistics(latents, anchor_mean, anchor_std):
    """`latents` (B, C, T, H, W) rescaled per channel to `anchor_mean` and
    `anchor_std`: `(x - mean_x) / std_x * std_a + mean_a`, computed in float32
    and returned in the latent's own dtype and device."""
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
    return matched.to(latents.dtype)


def anchor_block():
    """The anchor step class (imports diffusers on first use)."""
    from .h3_anchor_steps import DwH3LatentAnchorStep

    return DwH3LatentAnchorStep


def _anchor_sequences(pipeline):
    """[(prefix, sequence)] for the core-denoise sequences that generate fresh
    video - t2va and fl2va; ref2va, whose layout is its own, is left out."""
    found = []
    for prefix, sequence in core_denoise_sequences(pipeline):
        names = list(sequence.sub_blocks)
        if prefix + ANCHOR_AFTER not in names:
            continue
        if prefix + "prepare_latents_ref2va" in names:
            continue
        found.append((prefix, sequence))
    return found


def insert_latent_anchor(pipeline):
    """Put the anchor step right after `after_denoise` in every t2va/fl2va
    core-denoise sequence. Idempotent; a no-op on any other pipeline. Returns
    whether the pipeline now takes `latent_anchor`."""
    inserted = False
    block = None
    for prefix, sequence in _anchor_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        names = list(sub_blocks)
        if prefix + ANCHOR_BLOCK not in names:
            block = block or anchor_block()
            sub_blocks.insert(
                prefix + ANCHOR_BLOCK, block(), names.index(prefix + ANCHOR_AFTER) + 1
            )
        inserted = True
    return inserted


def takes_latent_anchor(pipeline):
    """Whether a loaded pipeline has the anchor step in a core sequence."""
    return any(
        prefix + ANCHOR_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def anchor_refusal(pipeline):
    """Why this pipeline cannot take `latent_anchor`, or None."""
    if takes_latent_anchor(pipeline):
        return None
    return (
        f"{LATENT_ANCHOR_INPUT} is a MiniMax-H3 t2va/fl2va argument, and "
        f"{type(pipeline).__name__} has no '{ANCHOR_AFTER}' step it follows"
    )


def prepared_anchor(where, value, base_dir):
    """A `latent_anchor` value as `(n, h, w, 3)` uint8 frames cut to a
    whole-latent length - from a video path or reference, or frames (a step's
    result, a list of images, an array or tensor).

    Raises:
        ValueError: If the value is not frames of a video
    """
    if isinstance(value, str):
        # argument_media imports diffusers, which h3_guides waits for (#790)
        from ..argument_media import fetch_video

        value = fetch_video(value, base_dir)
    try:
        frames = guide_frames_array(value)
    except ValueError as error:
        raise ValueError(f"{where}: {error}".replace("a guide's", "the")) from error
    return frames[: snap_guide_length(frames.shape[0])]


__all__ = [
    "anchor_block",
    "anchor_refusal",
    "channel_statistics",
    "insert_latent_anchor",
    "match_channel_statistics",
    "prepared_anchor",
    "takes_latent_anchor",
]
