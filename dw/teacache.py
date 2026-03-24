"""
TeaCache - Training-free inference acceleration for diffusion transformers.

Caches intermediate transformer computations and skips redundant steps
when the input hasn't changed significantly between timesteps.

Based on: https://github.com/ali-vilab/TeaCache
Adapted from: https://github.com/Teriks/dgenerate (Apache 2.0)

Implemented: Flux (FluxTransformer2DModel)
Registry includes: Mochi, LTX-Video, CogVideoX, Lumina2, HunyuanVideo, Wan2.1
(these require custom forward functions to be added)

Each model requires a custom forward function because transformer architectures
differ. The core caching algorithm is the same: extract a signal from the first
block's normalization, compare via polynomial rescaling, skip if below threshold.
"""

import json
import typing
import logging
from pathlib import Path
from contextlib import contextmanager

import torch
import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.getLogger("dw")


# ---------------------------------------------------------------------------
# Model registry loaded from JSON
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(__file__).parent / "teacache_models.json"


def _load_registry():
    """Load the model registry from the JSON file."""
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


def _get_model_info(transformer, variant=None):
    """Look up model info from registry.

    Args:
        transformer: The transformer model instance.
        variant: Optional explicit variant name (e.g., "wan2.1_t2v_1.3b").
            If None, uses class_defaults mapping.

    Returns:
        dict with coefficients, default_threshold, threshold_guide.
    """
    registry = _load_registry()
    class_name = transformer.__class__.__name__

    if variant is not None:
        info = registry["models"].get(variant)
        if info is None:
            available = ", ".join(registry["models"].keys())
            raise ValueError(
                f"TeaCache variant '{variant}' not found. Available: {available}"
            )
        return info

    # Look up default variant for this class
    default_variant = registry["class_defaults"].get(class_name)
    if default_variant is None:
        supported_classes = ", ".join(registry["class_defaults"].keys())
        raise ValueError(
            f"TeaCache does not support {class_name}. Supported: {supported_classes}"
        )

    return registry["models"][default_variant]


# ---------------------------------------------------------------------------
# Forward function factories, one per supported transformer architecture.
# ---------------------------------------------------------------------------


def _create_flux_teacache_forward(num_inference_steps, rel_l1_thresh, coefficients):
    """Create TeaCache forward for FluxTransformer2DModel."""
    cnt = 0
    accumulated_rel_l1_distance = 0
    previous_modulated_input = None
    previous_residual = None
    rescale_func = np.poly1d(coefficients)

    def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> typing.Union[torch.FloatTensor, Transformer2DModelOutput]:
        nonlocal cnt, accumulated_rel_l1_distance, previous_modulated_input, previous_residual

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # TeaCache: extract cache signal from first block's normalization
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.transformer_blocks[0].norm1(inp, emb=temb_)
        )

        # Decide whether to compute or reuse cached result
        if cnt == 0 or cnt == num_inference_steps - 1:
            should_calc = True
            accumulated_rel_l1_distance = 0
        else:
            relative_diff = (
                (
                    (modulated_inp - previous_modulated_input).abs().mean()
                    / previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            accumulated_rel_l1_distance += rescale_func(relative_diff)

            if accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                accumulated_rel_l1_distance = 0

        previous_modulated_input = modulated_inp
        cnt += 1
        if cnt == num_inference_steps:
            cnt = 0

        if not should_calc:
            hidden_states += previous_residual
        else:
            ori_hidden_states = hidden_states.clone()

            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: typing.Dict[str, typing.Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    encoder_hidden_states, hidden_states = (
                        torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(
                        controlnet_block_samples
                    )
                    interval_control = int(np.ceil(interval_control))
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states
                            + controlnet_block_samples[
                                index_block % len(controlnet_block_samples)
                            ]
                        )
                    else:
                        hidden_states = (
                            hidden_states
                            + controlnet_block_samples[index_block // interval_control]
                        )

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: typing.Dict[str, typing.Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    encoder_hidden_states, hidden_states = (
                        torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(
                        controlnet_single_block_samples
                    )
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[
                            index_block // interval_control
                        ]
                    )

            previous_residual = hidden_states - ori_hidden_states

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    return teacache_forward


# Map transformer class names to their forward factory functions.
# Models in the JSON registry without a factory here will get an informative error.
_FORWARD_FACTORIES = {
    "FluxTransformer2DModel": _create_flux_teacache_forward,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def teacache_context(
    pipeline, num_inference_steps, rel_l1_thresh=None, coefficients=None, variant=None
):
    """Context manager that enables TeaCache on a pipeline's transformer.

    Auto-detects the transformer type and applies the appropriate
    TeaCache forward function. Restores original forward on exit.

    Args:
        pipeline: A DiffusionPipeline with a .transformer attribute
        num_inference_steps: Number of inference steps (must match pipeline call)
        rel_l1_thresh: Cache threshold override. If None, uses model default.
            Higher = more speedup, more quality loss.
        coefficients: Polynomial coefficients override. If None, uses model default.
            List of 5 floats for np.poly1d rescaling function.
        variant: Explicit model variant name (e.g., "wan2.1_t2v_1.3b").
            Required when a transformer class has multiple variants (CogVideoX, Wan).
            If None, uses class_defaults from the registry.
    """
    transformer = pipeline.transformer
    class_name = transformer.__class__.__name__

    # Look up model info from registry
    model_info = _get_model_info(transformer, variant)

    # Check we have a forward implementation for this class
    factory = _FORWARD_FACTORIES.get(class_name)
    if factory is None:
        supported = ", ".join(_FORWARD_FACTORIES.keys())
        raise ValueError(
            f"No TeaCache forward implementation for {class_name}. "
            f"Implemented: {supported}. "
            f"The model is in the registry but needs a custom forward function."
        )

    # Use overrides or defaults
    if rel_l1_thresh is None:
        rel_l1_thresh = model_info["default_threshold"]
    if coefficients is None:
        coefficients = model_info["coefficients"]

    original_forward = transformer.forward

    teacache_forward_fn = factory(num_inference_steps, rel_l1_thresh, coefficients)
    transformer.forward = teacache_forward_fn.__get__(
        transformer, transformer.__class__
    )

    logger.info(
        f"TeaCache enabled for {class_name}: "
        f"steps={num_inference_steps}, threshold={rel_l1_thresh}"
    )

    try:
        yield pipeline
    finally:
        transformer.forward = original_forward
        logger.debug("TeaCache disabled, original forward restored")
