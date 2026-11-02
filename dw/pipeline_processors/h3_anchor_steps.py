"""The MiniMax-H3 latent statistics anchor step dw puts after `after_denoise`.

What the anchor is, and where the step goes, is dw/pipeline_processors/h3_anchor.py,
which inserts it. This module imports diffusers when it loads, so nothing imports
it at module level - `h3_anchor.anchor_block()` imports it on first use.
"""

import logging

import numpy as np
import torch
from diffusers.models import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3 import encoders
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from .h3_anchor import channel_statistics, match_channel_statistics
from .h3_guide_steps import fit_guide_frames
from .h3_rules import LATENT_ANCHOR_INPUT, snap_guide_length

logger = logging.getLogger("dw")


def anchor_statistics(components, frames, height, width):
    """(mean, std) per latent channel of `frames` ((n, h, w, 3) uint8), encoded
    on the (height, width) canvas the way a guide is: covered and centre-cropped
    onto it, cut to a whole-latent length, and VAE-encoded into the normalized
    latent space the generated latents live in."""
    frames = frames[: snap_guide_length(frames.shape[0])]
    pixels = fit_guide_frames(frames, height, width)
    pixels = torch.from_numpy(np.ascontiguousarray(pixels))
    device = components._execution_device
    # Through the module, so the encode is the one installed when it runs
    encoded = encoders.encode_vae_condition(
        components.vae,
        pixels.to(device).permute(3, 0, 1, 2)[None],
        components.pixel_mean,
        components.pixel_std,
        components.keyframe_encode_seed,
    )
    return channel_statistics(encoded)


class DwH3LatentAnchorStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw: rescales each channel of the unpacked video latents to the mean and std of `latent_anchor`'s "
            "latent - the anchor clip VAE-encoded on the canvas - before decode. Audio is untouched. Does "
            "nothing without `latent_anchor`."
        )

    @property
    def expected_components(self):
        return [ComponentSpec("vae", AutoencoderKLMiniMaxH3)]

    @property
    def inputs(self):
        return [
            InputParam(name="latents", type_hint=torch.Tensor, required=True),
            InputParam(
                name=LATENT_ANCHOR_INPUT,
                default=None,
                description=(
                    "Video frames, (n, h, w, 3) uint8, whose per-channel latent mean and std the generated "
                    "video latent is rescaled to before decode."
                ),
            ),
            InputParam(name="height", type_hint=int, required=True),
            InputParam(name="width", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        frames = getattr(block_state, LATENT_ANCHOR_INPUT)
        if frames is None:
            return components, state
        mean, std = anchor_statistics(
            components, frames, block_state.height, block_state.width
        )
        block_state.latents = match_channel_statistics(block_state.latents, mean, std)
        self.set_block_state(state, block_state)
        return components, state


__all__ = ["DwH3LatentAnchorStep", "anchor_statistics"]
