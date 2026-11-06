"""The SelfLift loop sub-block (#1195; the method and the feasibility note are
dw/pipeline_processors/h3_selflift.py's docstring): EXPERIMENTAL.

`DwH3SelfLiftStep` sits in front of `denoiser` in a t2va/fl2va loop and does
nothing unless the call passes `experimental_selflift`. Then, before the
first forward, it swaps the call's full-canvas video rows, layout and
row-timestep plan for the small canvas's, and before forward `lift_step` it
swaps the full ones back with the lifted latents. The audio rows are never
touched.

Imports diffusers when it loads: h3_selflift.insert_selflift imports it on
first use.
"""

import logging

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    _frame_position_grid,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

from .h3_rules import GATED_GUIDE_ROWS, SELFLIFT_INPUT
from .h3_selflift_latents import (
    lift,
    patchify,
    scale_factor,
    shrink_clean,
    shrink_layout,
    shrink_noise,
    unpatchify,
)

logger = logging.getLogger("dw")

# The block state key the full canvas's state waits under until the lift
_STASH = "_dw_selflift_full"


class DwH3SelfLiftStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw (experimental): with `experimental_selflift`, runs the loop's first `lift_step` forwards on a "
            "smaller canvas and lifts the latents to the call's canvas, at the same sigma, before the rest. "
            "Nothing otherwise."
        )

    @property
    def inputs(self):
        return [
            InputParam(
                name=SELFLIFT_INPUT,
                type_hint=dict,
                default=None,
                description=(
                    "Experimental: {lift_step, start_scale 0.5, upsample bilinear|nearest} - the first lift_step "
                    "evaluations on a canvas start_scale the size, then the rest at full size."
                ),
            ),
            InputParam(name="num_latent_frames", type_hint=int, default=None),
            InputParam(name="latent_height", type_hint=int, default=None),
            InputParam(name="latent_width", type_hint=int, default=None),
            InputParam(name="num_condition_video_rows", type_hint=int, default=0),
            InputParam(name=GATED_GUIDE_ROWS, type_hint=list, default=None),
        ]

    @property
    def intermediate_outputs(self):
        return []

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        settings = getattr(block_state, SELFLIFT_INPUT, None)
        if settings is None:
            return components, block_state
        if i == 0:
            self._shrink(components, block_state, settings)
        elif i == settings["lift_step"]:
            self._lift(components, block_state, i, settings)
        return components, block_state

    def _geometry(self, components, block_state, settings):
        factor = scale_factor(settings)
        patch = tuple(components.patch_size)
        height, width = block_state.latent_height, block_state.latent_width
        if patch[0] != 1 or height % (factor * patch[1]) or width % (factor * patch[2]):
            raise ValueError(
                f"{SELFLIFT_INPUT}: the {height}x{width} latent canvas does not "
                f"divide into {factor}x{factor} blocks of {patch[1]}x{patch[2]} "
                f"patches - use a width and height that are multiples of "
                f"{32 * factor}"
            )
        return factor, patch, height, width

    def _shrink(self, components, block_state, settings):
        steps = len(block_state.row_timestep_plan)
        lift_step = settings["lift_step"]
        if lift_step >= steps:
            raise ValueError(
                f"{SELFLIFT_INPUT} lift_step {lift_step} leaves no evaluation on "
                f"the full canvas: the schedule runs {steps}"
            )
        if getattr(block_state, GATED_GUIDE_ROWS, None):
            raise ValueError(
                f"{SELFLIFT_INPUT} does not combine with a capstone's gated rows"
            )
        factor, patch, height, width = self._geometry(components, block_state, settings)
        small = (height // factor, width // factor)
        full_grid = _frame_position_grid(height, width, patch[1], patch[2])[0]
        half_grid = _frame_position_grid(*small, patch[1], patch[2])[0]
        rows_per_frame = full_grid.shape[0]
        latents = block_state.latents
        conditions = block_state.num_condition_video_rows or 0
        if conditions % rows_per_frame:
            raise ValueError(
                f"{SELFLIFT_INPUT}: {conditions} condition rows are not whole "
                f"frames of the call's canvas"
            )
        frames = block_state.num_latent_frames
        noise = unpatchify(latents[conditions:], frames, height, width, patch)
        held = latents[:conditions]
        small_rows = [patchify(shrink_noise(noise, factor), patch)]
        if conditions:
            condition = unpatchify(
                held, conditions // rows_per_frame, height, width, patch
            )
            small_rows.insert(0, patchify(shrink_clean(condition, factor), patch))
        fields, plan = shrink_layout(
            block_state.denoiser_input_fields,
            block_state.row_timestep_plan[:lift_step],
            rows_per_frame,
            full_grid,
            half_grid,
        )
        setattr(
            block_state,
            _STASH,
            {
                "noise": noise.clone(),
                "condition_rows": held.clone(),
                "conditions": conditions,
                "fields": block_state.denoiser_input_fields,
                "plan": block_state.row_timestep_plan,
            },
        )
        block_state.latents = torch.cat(small_rows).to(latents.dtype)
        block_state.num_condition_video_rows = conditions // (factor * factor)
        block_state.denoiser_input_fields = fields
        block_state.row_timestep_plan = plan + block_state.row_timestep_plan[lift_step:]
        logger.info(
            f"SelfLift: the first {lift_step} of {len(block_state.row_timestep_plan)} "
            f"steps at {small[1] * 16}x{small[0] * 16}, then {width * 16}x{height * 16}"
        )

    def _lift(self, components, block_state, i, settings):
        full = getattr(block_state, _STASH)
        factor, patch, height, width = self._geometry(components, block_state, settings)
        small = (height // factor, width // factor)
        frames = block_state.num_latent_frames
        conditions = block_state.num_condition_video_rows or 0
        latents = block_state.latents
        sigma = float(components.scheduler.sigmas[i])
        lifted = lift(
            unpatchify(latents[conditions:], frames, *small, patch),
            unpatchify(
                block_state.noise_pred[0, conditions:].float(), frames, *small, patch
            ),
            sigma,
            full["noise"].to(latents.device),
            factor,
            settings.get("upsample", "bilinear"),
        )
        block_state.latents = torch.cat(
            [full["condition_rows"], patchify(lifted, patch).to(latents.dtype)]
        )
        block_state.num_condition_video_rows = full["conditions"]
        block_state.denoiser_input_fields = full["fields"]
        block_state.row_timestep_plan = full["plan"]
        delattr(block_state, _STASH)
        logger.info(
            f"SelfLift: lifted to the full canvas at step {i} (sigma {sigma:.3f})"
        )
