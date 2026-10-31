"""The MiniMax-H3 regional-edit blocks (#1062) - what
dw/pipeline_processors/h3_mask.py inserts and describes.

This module imports diffusers when it loads, so nothing imports it at module
level - `h3_mask.blocks()` imports it on first use (#790).
"""

import logging

import numpy as np
import torch
from diffusers.models import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3 import encoders
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    patchify_video_latents,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from .h3_mask import keep_rows
from .h3_rules import (
    KEEP_ROWS,
    MASK_GROW_DEFAULT,
    MASK_GROW_INPUT,
    REFINE_ROW_NOISE,
    REFINE_STRENGTH_INPUT,
    REGENERATE_MASK,
    SOURCE_ROWS,
    SOURCE_VIDEO_INPUT,
    media_problems,
)

logger = logging.getLogger("dw")


def fit_source_frames(frames, num_frames):
    """`frames` padded to the `num_frames` H3 aligned the render to, by
    repeating the last frame - the frames past the source's own are never
    regenerated from anything but it."""
    if frames.shape[0] >= num_frames:
        return frames[:num_frames]
    pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
    return np.concatenate([frames, pad])


class DwH3SourceLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw: with `source_video`, VAE-encodes it over the generated video rows of `latents` for the refine "
            "schedule to re-denoise, and keeps the clean rows and which of them the mask leaves alone for the "
            "loop's blend. Does nothing without `source_video`."
        )

    @property
    def expected_components(self):
        return [ComponentSpec("vae", AutoencoderKLMiniMaxH3)]

    @property
    def inputs(self):
        return [
            InputParam(
                name=SOURCE_VIDEO_INPUT,
                default=None,
                description=(
                    "The video to edit, as (frames, height, width, 3) uint8 at the render's size and frame "
                    "count. Needs `refine_strength`."
                ),
            ),
            InputParam(
                name=REGENERATE_MASK,
                default=None,
                description=(
                    "Internal: the mask as (frames, height, width) bool, True to regenerate; one frame covers "
                    "every frame. Without it the whole source is re-denoised."
                ),
            ),
            InputParam(
                name=MASK_GROW_INPUT,
                type_hint=int,
                default=MASK_GROW_DEFAULT,
                description="Patch cells the mask is widened by.",
            ),
            InputParam(name="latents", type_hint=torch.Tensor, required=True),
            InputParam(name="num_condition_video_rows", type_hint=int, default=0),
            InputParam(name="height", type_hint=int, required=True),
            InputParam(name="width", type_hint=int, required=True),
            InputParam(name="num_frames", type_hint=int, required=True),
            InputParam(name="num_latent_frames", type_hint=int, required=True),
            InputParam(name="latent_height", type_hint=int, required=True),
            InputParam(name="latent_width", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam(SOURCE_ROWS, type_hint=torch.Tensor),
            OutputParam(KEEP_ROWS, type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        frames = getattr(block_state, SOURCE_VIDEO_INPUT)
        if frames is None:
            return components, state

        height, width = block_state.height, block_state.width
        # The canvas H3 resolved - the step's own width and height were
        # checked against the source before the call
        problems = media_problems(SOURCE_VIDEO_INPUT, frames.shape[:3], width, height)
        if problems:
            raise ValueError(problems[0])
        pixels = fit_source_frames(frames, block_state.num_frames)
        device = components._execution_device
        encoded = encoders.encode_vae_condition(
            components.vae,
            torch.from_numpy(np.ascontiguousarray(pixels))
            .to(device)
            .permute(3, 0, 1, 2)[None],
            components.pixel_mean,
            components.pixel_std,
            components.keyframe_encode_seed,
        )
        latents = block_state.latents
        start = block_state.num_condition_video_rows or 0
        rows = patchify_video_latents(
            encoded.to(latents.device, torch.float32), components.patch_size
        ).to(latents.dtype)
        if rows.shape != latents[start:].shape:
            raise ValueError(
                f"source_video encodes to {tuple(rows.shape)} rows, and the step "
                f"generates {tuple(latents[start:].shape)}"
            )
        block_state.latents = torch.cat([latents[:start], rows])
        setattr(block_state, SOURCE_ROWS, rows.clone())

        mask = getattr(block_state, REGENERATE_MASK)
        keep = None
        if mask is not None:
            grow = getattr(block_state, MASK_GROW_INPUT)
            keep = keep_rows(
                mask,
                block_state.num_latent_frames,
                block_state.latent_height,
                block_state.latent_width,
                components.patch_size,
                MASK_GROW_DEFAULT if grow is None else grow,
            )
            if keep.shape[0] != rows.shape[0]:
                raise ValueError(
                    f"the mask maps to {keep.shape[0]} rows, and the step "
                    f"generates {rows.shape[0]}"
                )
            logger.info(
                f"Regional edit: regenerating {int((~keep).sum())} of "
                f"{keep.shape[0]} video rows"
            )
            keep = torch.from_numpy(keep).to(latents.device)
        setattr(block_state, KEEP_ROWS, keep)
        self.set_block_state(state, block_state)
        return components, state


class DwH3MaskBlendStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw: after each denoise step, writes the rows a regional edit keeps back to the source noised to "
            "the step's sigma, with the refine schedule's own noise. Does nothing without a mask."
        )

    @property
    def inputs(self):
        return [
            InputParam(name="latents", type_hint=torch.Tensor, required=True),
            InputParam(name="num_condition_video_rows", type_hint=int, default=0),
            InputParam(name=SOURCE_ROWS, type_hint=torch.Tensor, default=None),
            InputParam(name=KEEP_ROWS, type_hint=torch.Tensor, default=None),
            InputParam(name=REFINE_ROW_NOISE, type_hint=torch.Tensor, default=None),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        keep = getattr(block_state, KEEP_ROWS, None)
        if keep is None:
            return components, block_state
        noise = getattr(block_state, REFINE_ROW_NOISE, None)
        if noise is None:
            raise ValueError(
                f"a regional edit re-denoises from {REFINE_STRENGTH_INPUT}, and "
                f"the refine schedule left no noise to hold the kept rows with"
            )
        scheduler = components.scheduler
        # After update stepped to the next point of the schedule
        index = scheduler.step_index if scheduler.step_index is not None else i + 1
        sigma = scheduler.sigmas[index]
        source = getattr(block_state, SOURCE_ROWS)
        rows = block_state.latents[block_state.num_condition_video_rows or 0 :]
        # In H3's t = 1 - σ convention, as the refine block re-noised them
        rows[keep] = scheduler.scale_noise(source[keep], 1 - sigma, noise[keep]).to(
            rows.dtype
        )
        return components, block_state
