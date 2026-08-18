#!/usr/bin/env python3
"""Tests for registering transformer block metadata with diffusers' cache hooks.

first_block, mag and layer_skip resolve a model's block class through
diffusers.hooks._helpers.TransformerBlockRegistry and raise when it is absent.
dw.cache_blocks fills in the blocks diffusers has not registered upstream.
"""

import inspect
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw import cache_blocks
from dw.cache_blocks import (
    ARGUMENT_REMAP_KEY,
    build_metadata,
    register_cache_blocks,
    _load_registry,
)
from dw.type_helpers import load_type_from_name


@pytest.fixture(autouse=True)
def reset_registration_flag():
    """Let each test drive registration itself rather than inherit another's."""
    original = cache_blocks._registered
    cache_blocks._registered = False
    yield
    cache_blocks._registered = original


def test_registry_file_is_well_formed():
    """Every entry names the fields diffusers' TransformerBlockMetadata takes."""
    from diffusers.hooks._helpers import TransformerBlockMetadata

    registry = _load_registry()
    assert registry, "registry should not be empty"

    allowed = (
        set(TransformerBlockMetadata.__dataclass_fields__)
        - {"_cls", "_cached_parameter_indices"}
    ) | {ARGUMENT_REMAP_KEY}
    for class_name, metadata in registry.items():
        assert "." in class_name, f"'{class_name}' should be a full dotted path"
        assert set(metadata).issubset(allowed), f"'{class_name}' has unknown fields"


def test_second_stream_is_read_from_the_argument_the_entry_names():
    """A remapped entry must resolve the block's own second stream, not the text one.

    FBCBlockHook reads the second stream back out of the forward arguments under
    the fixed name 'encoder_hidden_states' when the cache skips a block. LTX-2's
    blocks take an 'encoder_hidden_states' of their own - the text conditioning -
    while returning audio as their second stream, so without the remap the hook
    hands the text embeddings back in the audio slot.
    """
    from diffusers.hooks._helpers import TransformerBlockRegistry

    register_cache_blocks()

    remapped = {
        name: entry
        for name, entry in _load_registry().items()
        if ARGUMENT_REMAP_KEY in entry
    }
    assert remapped, "expected at least one entry naming its second stream"

    for class_name, entry in remapped.items():
        metadata = TransformerBlockRegistry.get(load_type_from_name(class_name))
        second_stream, text = object(), object()
        arguments = {
            entry[ARGUMENT_REMAP_KEY]: second_stream,
            "encoder_hidden_states": text,
        }

        assert (
            metadata._get_parameter_from_args_kwargs(
                "encoder_hidden_states", (), arguments
            )
            is second_stream
        )
        # Positional resolution reads the forward signature, so this also asserts
        # the block takes a parameter by the name the entry gives - and that the
        # remap picks it out rather than whatever sits at the text stream's index
        parameters = list(
            inspect.signature(load_type_from_name(class_name).forward).parameters
        )[1:]
        positional = tuple(object() for _ in parameters)
        assert (
            metadata._get_parameter_from_args_kwargs(
                "encoder_hidden_states", positional, {}
            )
            is positional[parameters.index(entry[ARGUMENT_REMAP_KEY])]
        )


def test_metadata_without_a_remap_keeps_diffusers_behaviour():
    """An entry that names no second-stream argument builds plain metadata."""
    from diffusers.hooks._helpers import TransformerBlockMetadata

    metadata = build_metadata(
        TransformerBlockMetadata,
        {"return_hidden_states_index": 0, "return_encoder_hidden_states_index": None},
    )

    assert type(metadata) is TransformerBlockMetadata


def test_registers_missing_blocks():
    """A block diffusers does not know becomes resolvable after registration."""
    from diffusers.hooks._helpers import TransformerBlockRegistry

    register_cache_blocks()

    for class_name, expected in _load_registry().items():
        block_class = load_type_from_name(class_name)
        metadata = TransformerBlockRegistry.get(block_class)
        for field, value in expected.items():
            assert getattr(metadata, field) == value


def test_hidden_states_resolves_positionally_and_by_keyword():
    """The metadata must find hidden_states however the hook's caller passed it.

    FBCHeadBlockHook reads hidden_states out of the forward's args/kwargs before
    calling it. Metadata naming a parameter the block does not take would raise
    there, mid-forward, rather than at registration.
    """
    from diffusers.hooks._helpers import TransformerBlockRegistry

    register_cache_blocks()

    for class_name in _load_registry():
        block_class = load_type_from_name(class_name)
        metadata = TransformerBlockRegistry.get(block_class)
        name = metadata.hidden_states_argument_name
        sentinel = object()

        assert (
            metadata._get_parameter_from_args_kwargs(name, (), {name: sentinel})
            is sentinel
        )
        # Positional resolution reads the forward signature, so this also asserts
        # the block actually takes a parameter by that name
        assert (
            metadata._get_parameter_from_args_kwargs(name, (sentinel,), {}) is sentinel
        )


def test_registration_is_idempotent():
    """Called per pipeline load, so repeat calls must not accumulate or raise."""
    from diffusers.hooks._helpers import TransformerBlockRegistry

    register_cache_blocks()
    cache_blocks._registered = False
    register_cache_blocks()
    register_cache_blocks()

    for class_name in _load_registry():
        TransformerBlockRegistry.get(load_type_from_name(class_name))


def test_does_not_override_upstream_registration(monkeypatch):
    """An upstream registration wins - dw only fills gaps."""
    from diffusers.hooks._helpers import (
        TransformerBlockMetadata,
        TransformerBlockRegistry,
    )

    class_name = next(iter(_load_registry()))
    block_class = load_type_from_name(class_name)

    upstream = TransformerBlockMetadata(return_hidden_states_index=7)
    monkeypatch.setitem(TransformerBlockRegistry._registry, block_class, upstream)

    register_cache_blocks()

    assert TransformerBlockRegistry.get(block_class) is upstream


def test_unknown_class_is_skipped_not_raised(monkeypatch):
    """A diffusers predating a listed model should skip it, not break loading."""
    monkeypatch.setattr(
        cache_blocks,
        "_load_registry",
        lambda: {"diffusers.models.transformers.transformer_nonexistent.NopeBlock": {}},
    )

    register_cache_blocks()


class _FakePipeline:
    """Stands in for a ModularPipeline - only .transformer is consulted."""

    def __init__(self, transformer=None):
        self.transformer = transformer


def _cached_tiny_model():
    from diffusers import FirstBlockCacheConfig
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    register_cache_blocks()
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        text_dim=16,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        freq_dim=16,
        rope_freq_dim=4,
    )
    model.enable_cache(FirstBlockCacheConfig(threshold=0.05))
    return model


def _state_manager(model):
    """The StateManager the model's first-block cache hook reads through."""
    from diffusers.hooks import HookRegistry

    registry = HookRegistry.check_if_exists_or_initialize(model.transformer_blocks[0])
    return registry.hooks[registry._hook_order[0]].state_manager


def test_cache_state_needs_a_context():
    """Without the context, the hook raises - the failure the wrapper prevents."""
    from dw.pipeline_processors.pipeline import stateful_cache_context

    manager = _state_manager(_cached_tiny_model())

    with pytest.raises(ValueError, match="No context is set"):
        manager.get_state()


def test_cache_context_supplies_state():
    from dw.pipeline_processors.pipeline import stateful_cache_context

    model = _cached_tiny_model()
    manager = _state_manager(model)

    with stateful_cache_context(_FakePipeline(model)):
        assert manager.get_state() is not None


def test_cache_state_does_not_leak_between_runs():
    """A pipeline this process keeps loaded must not reuse the last run's residuals."""
    from dw.pipeline_processors.pipeline import stateful_cache_context

    model = _cached_tiny_model()
    manager = _state_manager(model)
    pipeline = _FakePipeline(model)

    with stateful_cache_context(pipeline):
        first = manager.get_state()

    with stateful_cache_context(pipeline):
        assert manager.get_state() is not first


def test_cache_context_clears_after_an_error():
    """An errored run must not strand a context that the next run inherits."""
    from dw.pipeline_processors.pipeline import stateful_cache_context

    model = _cached_tiny_model()
    manager = _state_manager(model)

    with pytest.raises(RuntimeError):
        with stateful_cache_context(_FakePipeline(model)):
            raise RuntimeError("boom")

    assert manager._current_context is None


@pytest.mark.parametrize("transformer", [None, object()])
def test_cache_context_is_a_noop_without_caching(transformer):
    """Pipelines with no transformer, or an uncached one, pass straight through."""
    from dw.pipeline_processors.pipeline import stateful_cache_context

    with stateful_cache_context(_FakePipeline(transformer)):
        pass


def test_enable_cache_succeeds_on_registered_model():
    """The end-to-end path that failed before registration."""
    from diffusers import FirstBlockCacheConfig
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    register_cache_blocks()

    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        text_dim=16,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        freq_dim=16,
        rope_freq_dim=4,
    )

    model.enable_cache(FirstBlockCacheConfig(threshold=0.05))
    assert model.is_cache_enabled
    model.disable_cache()


class _DualStreamBlock(torch.nn.Module):
    """The shape LTX-2's block has: two streams in and out, text alongside.

    The two streams are advanced by different constants so a cached step that
    returned the wrong one is visible in the values, and the text conditioning
    is a different length so it is visible in the shapes too.
    """

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states=None):
        return hidden_states + 1.0, audio_hidden_states + 10.0


class _DualStreamModel(torch.nn.Module):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            _DualStreamBlock() for _ in range(num_blocks)
        )

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states):
        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states, audio_hidden_states


def _register_dual_stream_block(remap):
    from diffusers.hooks._helpers import (
        TransformerBlockMetadata,
        TransformerBlockRegistry,
    )

    entry = {"return_hidden_states_index": 0, "return_encoder_hidden_states_index": 1}
    if remap is not None:
        entry[ARGUMENT_REMAP_KEY] = remap
    TransformerBlockRegistry.register(
        model_class=_DualStreamBlock,
        metadata=build_metadata(TransformerBlockMetadata, entry),
    )


def _run_two_steps(model):
    """Drive the model twice - the second step is the one the cache skips.

    The context is set the way CacheMixin.cache_context sets it; the stub model
    is a bare Module rather than a diffusers model, so it has no such method of
    its own.
    """
    from diffusers.hooks import HookRegistry

    hidden_states = torch.zeros(1, 6, 4)
    audio_hidden_states = torch.zeros(1, 5, 4)
    encoder_hidden_states = torch.zeros(1, 3, 4)

    registry = HookRegistry.check_if_exists_or_initialize(model)
    registry._set_context("test")
    try:
        for _ in range(2):
            hidden_states, audio_hidden_states = model(
                hidden_states, audio_hidden_states, encoder_hidden_states
            )
    finally:
        registry._set_context(None)
    return hidden_states, audio_hidden_states


def test_cached_step_carries_the_second_stream_not_the_text(monkeypatch):
    """The end the remap exists for: a skipped block keeps the audio stream.

    Without it the tail block hands `encoder_hidden_states` back in the audio
    slot, which on LTX-2 is the text conditioning - a different sequence length,
    so the run fails downstream rather than merely drifting.
    """
    from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
    from diffusers.hooks._helpers import TransformerBlockRegistry

    monkeypatch.setitem(TransformerBlockRegistry._registry, _DualStreamBlock, None)
    _register_dual_stream_block("audio_hidden_states")

    model = _DualStreamModel()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))

    hidden_states, audio_hidden_states = _run_two_steps(model)

    assert audio_hidden_states.shape == (1, 5, 4)
    assert hidden_states.shape == (1, 6, 4)
    # Two steps of two blocks: the first computes both, the second is cached from
    # the head block's residuals rather than skipping the streams' advance
    assert torch.all(audio_hidden_states > 0), "the audio stream stopped advancing"


def test_without_the_remap_the_text_stream_leaks_into_the_audio_slot(monkeypatch):
    """Pins why the remap is needed, so a later simplification cannot drop it."""
    from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
    from diffusers.hooks._helpers import TransformerBlockRegistry

    monkeypatch.setitem(TransformerBlockRegistry._registry, _DualStreamBlock, None)
    _register_dual_stream_block(None)

    model = _DualStreamModel()
    apply_first_block_cache(model, FirstBlockCacheConfig(threshold=0.05))

    _, audio_hidden_states = _run_two_steps(model)

    # The text conditioning's sequence length, carried into the audio stream
    assert audio_hidden_states.shape == (1, 3, 4)
