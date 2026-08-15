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

import functools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.teacache import _create_flux_teacache_forward, teacache_context


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

    def forward(self, *args, **kwargs):
        """Stand-in for the real (undecorated) transformer forward.

        Never expected to run directly in these tests: teacache_context always
        replaces this -- or, when an accelerate hook is installed,
        ``_old_forward`` -- before any calls are made.
        """
        raise NotImplementedError(
            "fake transformer's real forward should never be invoked directly"
        )


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


# ---------------------------------------------------------------------------
# teacache_context + accelerate offload hook composition
#
# accelerate's enable_model_cpu_offload/enable_sequential_cpu_offload installs
# an AlignDevicesHook via add_hook_to_module (accelerate/hooks.py): it saves
# the module's real (bound) forward as module._old_forward, then replaces
# module.forward with functools.partial(new_forward, module), where
# new_forward calls module._hf_hook.pre_forward(...) (this is what moves the
# module's weights onto the execution device) before invoking
# module._old_forward(*args, **kwargs). teacache_context must not clobber
# that wrapper: doing so silently drops the pre_forward device placement and
# produces a device-mismatch RuntimeError when the module is offloaded to
# CPU. Instead, when a hook is present, teacache_context must wrap
# _old_forward itself and leave .forward (the hook wrapper) alone.
#
# The fixtures below mirror accelerate/hooks.py's add_hook_to_module/
# new_forward mechanics closely enough to exercise that composition without
# depending on accelerate or real model weights.
# ---------------------------------------------------------------------------


class _FakeAlignDevicesHook:
    """Stand-in for accelerate's AlignDevicesHook.

    Records every pre_forward call so tests can assert it still fires (i.e.
    device placement would still happen) once teacache_context is active.
    """

    def __init__(self, pre_forward_calls):
        self._pre_forward_calls = pre_forward_calls

    def pre_forward(self, module, *args, **kwargs):
        self._pre_forward_calls.append(True)
        return args, kwargs

    def post_forward(self, module, output):
        return output


def _install_fake_accelerate_hook(transformer):
    """Attach a hook the way accelerate.hooks.add_hook_to_module does.

    Mirrors accelerate/hooks.py: stashes the current (bound) forward as
    ``_old_forward``, sets ``_hf_hook``, and replaces ``.forward`` with a
    wrapper that calls ``hook.pre_forward`` -> ``_old_forward`` ->
    ``hook.post_forward``.
    """
    pre_forward_calls = []
    old_forward = transformer.forward
    transformer._old_forward = old_forward
    transformer._hf_hook = _FakeAlignDevicesHook(pre_forward_calls)

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        output = module._old_forward(*args, **kwargs)
        return module._hf_hook.post_forward(module, output)

    transformer.forward = functools.update_wrapper(
        functools.partial(new_forward, transformer), old_forward
    )
    return pre_forward_calls


class FluxTransformer2DModel(_FakeFluxTransformer):
    """Fake transformer named to match the TeaCache registry/factory lookup.

    teacache_context dispatches on ``transformer.__class__.__name__``, so the
    class needs this exact name even though it shares the real
    FluxTransformer2DModel's name only nominally (no import collision: this
    is the only class of that name in scope).
    """


class _FakePipeline:
    """Minimal stand-in for a DiffusionPipeline: just needs .transformer."""

    def __init__(self, transformer):
        self.transformer = transformer


def _forward_call_kwargs(timestep_value=0.9):
    return dict(
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


def test_teacache_context_composes_with_accelerate_hook():
    """With an accelerate-style hook installed, entering teacache_context
    must leave .forward as the hook wrapper (untouched) and instead replace
    _old_forward, so calling transformer.forward(...) runs both the hook's
    pre_forward (device placement) and the TeaCache forward."""
    transformer = FluxTransformer2DModel()
    pre_forward_calls = _install_fake_accelerate_hook(transformer)
    hook_wrapper = transformer.forward
    original_old_forward = transformer._old_forward
    pipeline = _FakePipeline(transformer)

    with teacache_context(pipeline, num_inference_steps=4, variant="flux"):
        assert transformer.forward is hook_wrapper
        assert transformer._old_forward is not original_old_forward

        result = transformer.forward(**_forward_call_kwargs())

    assert len(pre_forward_calls) == 1  # hook's pre_forward ran
    assert result.sample is not None  # TeaCache forward produced output


def test_teacache_context_restores_old_forward_on_exit():
    """Exiting the context must restore the pre-context _old_forward exactly,
    while the hook wrapper installed on .forward is never touched."""
    transformer = FluxTransformer2DModel()
    _install_fake_accelerate_hook(transformer)
    hook_wrapper = transformer.forward
    original_old_forward = transformer._old_forward

    pipeline = _FakePipeline(transformer)

    with teacache_context(pipeline, num_inference_steps=4, variant="flux"):
        pass

    assert transformer.forward is hook_wrapper
    assert transformer._old_forward is original_old_forward


def test_teacache_context_no_hook_replaces_and_restores_forward():
    """Without an accelerate hook (no offload configured), teacache_context
    must keep replacing .forward directly, exactly as before this fix."""
    transformer = FluxTransformer2DModel()
    pipeline = _FakePipeline(transformer)
    # Bound methods are recreated on every attribute access, so identity
    # can't be compared directly; compare the underlying function instead.
    original_forward_func = transformer.forward.__func__

    with teacache_context(pipeline, num_inference_steps=4, variant="flux"):
        assert transformer.forward.__func__ is not original_forward_func
        assert not hasattr(transformer, "_old_forward")

        result = transformer.forward(**_forward_call_kwargs())
        assert result.sample is not None

    assert transformer.forward.__func__ is original_forward_func
