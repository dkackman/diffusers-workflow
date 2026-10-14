#!/usr/bin/env python3
"""Tests for a step cache's skip counts (the step-cache section of dw/pipeline_processors/timing.py).

The counts read diffusers' private leader-hook state, so the pin tests below
fail first when an upgrade moves it. The pipeline tests run a tiny real
FluxPipeline with true CFG, so each step is a cond and an uncond forward, each
in a cache context diffusers opens inside dw's.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.pipeline_processors import components
from dw.pipeline_processors import timing as cache_stats
from dw.pipeline_processors.timing import StepTimer, timed_step

STEPS = 6


class _Recorder:
    """Stands in for the event context and emit_warning."""

    def __init__(self):
        self.events = []
        self.warnings = []

    def emit(self, event_type, **data):
        self.events.append((event_type, data))

    def warn(self, message, **data):
        self.warnings.append((message, data))


@pytest.fixture
def recorder(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(cache_stats, "get_context", lambda: recorder)
    monkeypatch.setattr(cache_stats, "emit_warning", recorder.warn)
    return recorder


def _flux_pipeline(cache_config):
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        FluxPipeline,
        FluxTransformer2DModel,
    )

    torch.manual_seed(0)
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        axes_dims_rope=[2, 2, 4],
    )
    pipeline = FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )
    if cache_config is not None:
        components.enable_cache_on_transformer(pipeline, cache_config)
    return pipeline


def _call(pipeline, timer, segment=None):
    """One CFG pipeline call the way Pipeline._call_pipeline wraps it."""
    with timed_step(timer):
        with components.stateful_cache_context(pipeline, segment=segment):
            pipeline(
                prompt_embeds=torch.randn(1, 8, 16),
                pooled_prompt_embeds=torch.randn(1, 8),
                negative_prompt_embeds=torch.randn(1, 8, 16),
                negative_pooled_prompt_embeds=torch.randn(1, 8),
                true_cfg_scale=2.0,
                num_inference_steps=STEPS,
                height=16,
                width=16,
                output_type="latent",
            )


def _first_block(threshold):
    from diffusers import FirstBlockCacheConfig

    return FirstBlockCacheConfig(threshold=threshold)


# ------------------------------------------------------------ pins


def test_the_private_diffusers_names_are_where_the_counter_reads_them():
    """An upgrade that moves any of these nulls the counts in production;
    this fails CI first."""
    from diffusers.hooks import first_block_cache, mag_cache
    from diffusers.hooks.hooks import HookRegistry, StateManager

    assert (
        first_block_cache._FBC_LEADER_BLOCK_HOOK
        == cache_stats.LEADER_HOOKS["first_block"]
    )
    assert mag_cache._MAG_CACHE_LEADER_BLOCK_HOOK == cache_stats.LEADER_HOOKS["mag"]
    assert hasattr(first_block_cache.FBCSharedBlockState(), "should_compute")
    assert hasattr(mag_cache.MagCacheState(), "should_compute")
    assert callable(StateManager.get_state)
    assert callable(HookRegistry.get_hook)


def test_the_config_classes_map_to_workflow_types():
    import diffusers

    for class_name in cache_stats._CACHE_TYPES:
        assert hasattr(diffusers, class_name), class_name


# ------------------------------------------------------------ real pipeline


def test_cfg_counts_two_forwards_per_step_and_they_add_up(recorder):
    pipeline = _flux_pipeline(_first_block(0.5))
    timer = StepTimer(step="render")
    _call(pipeline, timer)

    cache = timer.timing()["cache"]
    assert cache["type"] == "first_block"
    assert cache["threshold"] == 0.5
    assert cache["forwards"] == 2 * STEPS
    assert cache["computed"] + cache["skipped"] == cache["forwards"]
    assert cache["skipped"] > 0, "a loose threshold on a tiny model skips"
    assert cache["hit_rate"] == round(cache["skipped"] / cache["forwards"], 3)
    assert recorder.warnings == []


def test_a_tight_threshold_skips_nothing(recorder):
    pipeline = _flux_pipeline(_first_block(1e-6))
    timer = StepTimer(step="render")
    _call(pipeline, timer)

    cache = timer.timing()["cache"]
    assert (cache["forwards"], cache["computed"], cache["skipped"]) == (
        2 * STEPS,
        2 * STEPS,
        0,
    )
    assert cache["hit_rate"] == 0.0


def test_the_event_names_the_step_and_segment_and_goes_out_before_the_reset(
    recorder, monkeypatch
):
    pipeline = _flux_pipeline(_first_block(0.5))
    order = []
    reset = pipeline.transformer._reset_stateful_cache

    def recording_reset(*args, **kwargs):
        order.append("reset")
        return reset(*args, **kwargs)

    monkeypatch.setattr(pipeline.transformer, "_reset_stateful_cache", recording_reset)
    real_emit = recorder.emit
    monkeypatch.setattr(
        recorder,
        "emit",
        lambda event_type, **data: (
            order.append(event_type),
            real_emit(event_type, **data),
        ),
    )

    _call(pipeline, StepTimer(step="render"), segment="segment 1/2")

    # A DiffusionPipeline resets its own state as its call ends; the counts
    # were read per forward, so only dw's reset - the last - has to follow
    assert order[-2:] == ["cache_stats", "reset"]
    assert order.count("cache_stats") == 1
    event_type, data = recorder.events[0]
    assert data["step"] == "render" and data["segment"] == "segment 1/2"
    assert data["forwards"] == 2 * STEPS


def test_counts_restart_each_call_and_a_chain_sums_them(recorder):
    pipeline = _flux_pipeline(_first_block(0.5))
    timer = StepTimer(step="chain")
    _call(pipeline, timer, segment="segment 1/2")
    _call(pipeline, timer, segment="segment 2/2")

    per_call = [
        data for event_type, data in recorder.events if event_type == "cache_stats"
    ]
    assert [data["forwards"] for data in per_call] == [2 * STEPS, 2 * STEPS]
    cache = timer.timing()["cache"]
    assert cache["forwards"] == 4 * STEPS
    assert cache["skipped"] == sum(data["skipped"] for data in per_call)
    assert cache["hit_rate"] == round(cache["skipped"] / cache["forwards"], 3)


def test_the_forward_hook_is_removed_after_the_call(recorder):
    pipeline = _flux_pipeline(_first_block(0.5))
    _call(pipeline, StepTimer(step="render"))
    assert not pipeline.transformer._forward_hooks


def test_no_cache_no_event_no_timing_key(recorder):
    pipeline = _flux_pipeline(None)
    timer = StepTimer(step="render")
    _call(pipeline, timer)

    assert recorder.events == [] and recorder.warnings == []
    assert "cache" not in timer.timing()


def test_moved_hook_state_nulls_the_counts_warns_once_and_the_run_finishes(
    recorder, monkeypatch
):
    monkeypatch.setitem(cache_stats.LEADER_HOOKS, "first_block", "no_such_hook")
    pipeline = _flux_pipeline(_first_block(0.5))
    timer = StepTimer(step="render")
    _call(pipeline, timer)

    cache = timer.timing()["cache"]
    assert cache["type"] == "first_block"
    assert all(
        cache[key] is None for key in ("forwards", "computed", "skipped", "hit_rate")
    )
    assert len(recorder.warnings) == 1
    assert recorder.warnings[0][1] == {
        "kind": "cache_stats_unavailable",
        "cache_type": "first_block",
    }
    assert [event_type for event_type, _ in recorder.events] == ["cache_stats"]


# ------------------------------------------------------------ the counter alone


class _State:
    def __init__(self, decisions):
        self._decisions = iter(decisions)

    def get_state(self):
        return SimpleNamespace(should_compute=next(self._decisions))


def _counter(config, state_manager=None, hook_name=None):
    """A CacheCounter over a fake transformer whose leader hook holds `state_manager`."""

    class Registry:
        def get_hook(self, name):
            if state_manager is not None and name == hook_name:
                return SimpleNamespace(state_manager=state_manager)
            return None

    module = SimpleNamespace(_diffusers_hook=Registry())
    transformer = SimpleNamespace(_cache_config=config, modules=lambda: [module])
    return cache_stats.CacheCounter(transformer)


def test_each_forward_is_read_once():
    counter = _counter(
        _first_block(0.1), _State([True, False, False, True]), "fbc_leader_block_hook"
    )
    for _ in range(4):
        counter.after_forward(None, (), None)
    stats = counter.stats()
    assert (stats["forwards"], stats["computed"], stats["skipped"]) == (4, 2, 2)
    assert stats["hit_rate"] == 0.5


def test_a_state_without_should_compute_is_unavailable_not_an_error():
    class NoDecision:
        def get_state(self):
            return SimpleNamespace()

    counter = _counter(_first_block(0.1), NoDecision(), "fbc_leader_block_hook")
    counter.after_forward(None, (), None)
    assert counter.unavailable and "AttributeError" in counter.unavailable
    assert counter.stats()["forwards"] is None


def test_no_context_is_unavailable_not_an_error():
    class NoContext:
        def get_state(self):
            raise ValueError("No context is set")

    counter = _counter(_first_block(0.1), NoContext(), "fbc_leader_block_hook")
    counter.after_forward(None, (), None)
    assert counter.unavailable and counter.stats()["computed"] is None


def test_mag_reports_its_config_keys():
    from diffusers import MagCacheConfig

    counter = _counter(
        MagCacheConfig(threshold=0.1, num_inference_steps=8, mag_ratios=[1.0] * 8),
        _State([True, False]),
        "mag_cache_leader_block_hook",
    )
    counter.after_forward(None, (), None)
    counter.after_forward(None, (), None)
    stats = counter.stats()
    assert stats["type"] == "mag" and stats["threshold"] == 0.1
    assert stats["num_inference_steps"] == 8
    assert {"max_skip_steps", "retention_ratio", "calibrate"} <= set(stats)
    assert (stats["forwards"], stats["skipped"]) == (2, 1)


@pytest.mark.parametrize(
    "class_name,cache_type",
    [
        ("TaylorSeerCacheConfig", "taylorseer"),
        ("FasterCacheConfig", "faster"),
        ("TextKVCacheConfig", "text_kv"),
    ],
)
def test_types_without_a_decision_report_type_only_and_no_warning(
    recorder, class_name, cache_type
):
    config = type(class_name, (), {})()
    counter = _counter(config)
    assert not counter.unavailable
    stats = cache_stats.report_cache_stats(counter)
    assert stats["type"] == cache_type
    assert all(
        stats[key] is None
        for key in ("threshold", "forwards", "computed", "skipped", "hit_rate")
    )
    assert recorder.warnings == []


# ------------------------------------------------------------ timer


def test_the_timer_sums_counts_and_nulls_on_any_null():
    timer = StepTimer()
    timer.cache_counted(
        {
            "type": "mag",
            "threshold": 0.1,
            "forwards": 4,
            "computed": 3,
            "skipped": 1,
            "hit_rate": 0.25,
        }
    )
    timer.cache_counted(
        {
            "type": "mag",
            "threshold": 0.1,
            "forwards": 4,
            "computed": 1,
            "skipped": 3,
            "hit_rate": 0.75,
        }
    )
    assert timer.timing()["cache"]["forwards"] == 8
    assert timer.timing()["cache"]["hit_rate"] == 0.5

    timer.cache_counted(
        {
            "type": "mag",
            "threshold": 0.1,
            "forwards": None,
            "computed": None,
            "skipped": None,
            "hit_rate": None,
        }
    )
    cache = timer.timing()["cache"]
    assert cache["forwards"] is None and cache["hit_rate"] is None


def test_a_timer_without_a_cache_has_no_cache_key():
    assert "cache" not in StepTimer().timing()
