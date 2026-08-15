#!/usr/bin/env python3
"""
Tests for TeaCache's multi-forward-per-step guard.

TeaCache's Flux forward wrapper (dw/teacache.py) assumes exactly one
transformer forward call per denoising step. Pipelines that run true
classifier-free guidance (e.g. FluxPipeline with negative_prompt and
true_cfg_scale > 1) call the transformer twice per step -- once for the
conditional pass and once for the unconditional pass -- using the identical
timestep both times. Without a guard, the second call would silently reuse
and corrupt the cached previous_modulated_input/previous_residual state,
producing corrupted images with no error.

These tests exercise the guard directly against the wrapped forward function
returned by ``_create_flux_teacache_forward``, using a minimal fake
transformer so no real model weights are required.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.teacache import _create_flux_teacache_forward


class _FakeBlock:
    """Stand-in for a FluxTransformerBlock: passes tensors through unchanged."""

    def norm1(self, hidden_states, emb):
        return hidden_states, None, None, None, None

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        joint_attention_kwargs,
    ):
        return encoder_hidden_states, hidden_states


class _FakeFluxTransformer:
    """Minimal stand-in for FluxTransformer2DModel.

    Implements just enough surface area (attributes/callables referenced by
    teacache_forward) for the wrapped forward to run end to end without
    real model weights.
    """

    def __init__(self):
        self.gradient_checkpointing = False
        self.x_embedder = lambda x: x
        self.context_embedder = lambda x: x
        self.time_text_embed = lambda timestep, pooled: torch.zeros(1, 8)
        self.pos_embed = lambda ids: None
        self.norm_out = lambda hidden_states, temb: hidden_states
        self.proj_out = lambda x: x
        self.transformer_blocks = [_FakeBlock()]
        self.single_transformer_blocks = []


def _make_bound_forward(num_inference_steps=4, rel_l1_thresh=0.6):
    """Build a fake transformer and bind the TeaCache Flux forward to it."""
    transformer = _FakeFluxTransformer()
    forward_fn = _create_flux_teacache_forward(
        num_inference_steps, rel_l1_thresh, coefficients=[0, 0, 0, 0, 0]
    )
    bound_forward = forward_fn.__get__(transformer, _FakeFluxTransformer)
    return transformer, bound_forward


def _call(bound_forward, timestep_value):
    """Invoke the wrapped forward once with a given scalar timestep value."""
    return bound_forward(
        hidden_states=torch.randn(1, 4, 8),
        encoder_hidden_states=torch.randn(1, 3, 8),
        pooled_projections=torch.zeros(1, 8),
        timestep=torch.tensor([timestep_value]),
        img_ids=torch.zeros(2, 3),
        txt_ids=torch.zeros(2, 3),
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
    )


def test_duplicate_timestep_raises_runtime_error():
    """Two forward calls with the identical timestep (true CFG) must raise."""
    _, bound_forward = _make_bound_forward()

    _call(bound_forward, 0.9)  # first call: no prior timestep, always allowed

    with pytest.raises(RuntimeError, match="true classifier-free guidance"):
        _call(bound_forward, 0.9)  # duplicate timestep: simulates the uncond pass


def test_duplicate_timestep_error_names_both_features():
    """The guard's error message must name both TeaCache and true CFG."""
    _, bound_forward = _make_bound_forward()
    _call(bound_forward, 0.9)

    with pytest.raises(RuntimeError) as excinfo:
        _call(bound_forward, 0.9)

    message = str(excinfo.value)
    assert "TeaCache" in message
    assert "true classifier-free guidance" in message
    assert "negative_prompt" in message
    assert "true_cfg_scale" in message


def test_distinct_timesteps_do_not_raise():
    """A normal single-forward-per-step denoising loop must not trip the guard."""
    _, bound_forward = _make_bound_forward(num_inference_steps=4)

    # Distinct, monotonically decreasing timesteps -- one forward per step.
    for t in (1.0, 0.75, 0.5, 0.25):
        result = _call(bound_forward, t)
        assert result.sample is not None


def test_distinct_timesteps_across_counter_wrap_do_not_raise():
    """The guard must keep working after the internal step counter wraps."""
    _, bound_forward = _make_bound_forward(num_inference_steps=4)

    # Two full passes worth of distinct timesteps (8 calls); the internal
    # `cnt` counter wraps back to 0 after every 4 calls.
    timesteps = [1.0, 0.75, 0.5, 0.25] * 2
    for t in timesteps:
        _call(bound_forward, t)  # should never raise


def test_non_adjacent_repeated_timestep_does_not_raise():
    """Only an *immediately repeated* timestep should trip the guard.

    A timestep value recurring later (not back-to-back) is not evidence of a
    second forward within the same step, so it must not raise.
    """
    _, bound_forward = _make_bound_forward(num_inference_steps=4)

    for t in (1.0, 0.75, 1.0):  # 1.0 repeats, but not back-to-back
        _call(bound_forward, t)  # should never raise
