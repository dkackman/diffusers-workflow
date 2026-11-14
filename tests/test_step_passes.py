"""A fixed `denoising_step_list` counts len - 1 passes in every step-count
consumer, and a pipeline without one counts as it did (#1316, #1342)."""

import types

import pytest

from dw import events
from dw.step_passes import (
    STEP_LIST_KEY,
    STEP_LIST_UNSUPPORTED,
    pipeline_step_passes,
    step_list,
    step_list_passes,
)

SEVEN = [1.0, 0.9375, 0.8333, 0.6667, 0.5, 0.25, 0.0]


class StepListPipeline:
    """A loaded pipeline whose config fixes a 7-point schedule."""

    def __init__(self, points=SEVEN):
        self.config = {STEP_LIST_KEY: points}

    def __call__(self, prompt=None, callback_on_step_end=None):
        return None


class PlainPipeline:
    def __init__(self):
        self.config = {"num_train_timesteps": 1000}

    def __call__(self, prompt=None, callback_on_step_end=None):
        return None


# The helper


def test_seven_points_are_six_passes():
    assert step_list_passes(StepListPipeline()) == 6
    assert pipeline_step_passes(StepListPipeline(), {}) == 6


def test_the_step_list_wins_over_num_inference_steps():
    assert pipeline_step_passes(StepListPipeline(), {"num_inference_steps": 30}) == 6


def test_an_argument_step_list_wins_over_the_config():
    assert step_list_passes(StepListPipeline(), {STEP_LIST_KEY: [1.0, 0.5, 0.0]}) == 2


def test_a_modular_pipeline_exposes_it_as_an_attribute():
    pipeline = types.SimpleNamespace(denoising_step_list=SEVEN)
    assert step_list_passes(pipeline) == 6


def test_a_workflow_pipeline_block_is_read_at_validate_time():
    block = {"from_pretrained_arguments": {STEP_LIST_KEY: SEVEN}, "arguments": {}}
    assert step_list_passes(block) == 6
    assert step_list_passes({"arguments": {STEP_LIST_KEY: SEVEN}}) == 6


def test_without_a_step_list_the_n_minus_one_rule_holds():
    assert step_list(PlainPipeline()) is None
    assert step_list_passes(PlainPipeline(), {"num_inference_steps": 30}) is None
    assert pipeline_step_passes(PlainPipeline(), {"num_inference_steps": 30}) == 29
    assert pipeline_step_passes(None, {}, default=50) == 49
    assert pipeline_step_passes(None, {}) is None


@pytest.mark.parametrize(
    "value", [[1.0], [], "1,0", [1.0, "x"], [1.0, True], [1.0, float("nan")], None]
)
def test_a_value_that_is_not_a_step_list_is_ignored(value):
    assert step_list_passes(None, {STEP_LIST_KEY: value}) is None


def test_the_engine_names_no_model():
    import dw.step_passes as module

    with open(module.__file__) as f:
        assert "pika" not in f.read().lower()


# TST


def test_tst_steps_without_a_step_list_are_unchanged():
    from dw.pipeline_processors.h3_tst import DEFAULT_STEPS, _evaluations

    assert _evaluations(30) == 29
    assert _evaluations(None) == DEFAULT_STEPS - 1


def test_tst_is_refused_on_a_step_list():
    from dw.pipeline_processors.h3_tst import step_list_refusal

    reason = step_list_refusal(StepListPipeline(), {})
    assert reason is not None and STEP_LIST_UNSUPPORTED in reason
    assert step_list_refusal(PlainPipeline(), {"num_inference_steps": 30}) is None


# MagCache calibration


class _MagConfig:
    calibrate = True
    num_inference_steps = 28


class _MagPipeline(StepListPipeline):
    def __init__(self, points=SEVEN):
        super().__init__(points)
        self.transformer = types.SimpleNamespace(_cache_config=_MagConfig())


def test_mag_calibration_is_skipped_on_a_step_list(monkeypatch):
    from dw.pipeline_processors import mag_calibration

    warnings = []
    monkeypatch.setattr(mag_calibration, "emit_warning", warnings.append)
    pipeline = _MagPipeline()
    with mag_calibration.captured_mag_calibration(pipeline, {}):
        # No handler is installed and the declared count is not touched
        assert pipeline.transformer._cache_config.num_inference_steps == 28
        assert not mag_calibration.logging.getLogger(
            mag_calibration._MAG_LOGGER
        ).handlers
    assert len(warnings) == 1
    assert "MagCache calibration" in warnings[0]
    assert STEP_LIST_UNSUPPORTED in warnings[0]


# Progress


def _step_totals(pipeline, arguments):
    from dw.pipeline_processors.pipeline import Pipeline

    owner = types.SimpleNamespace(pipeline=pipeline, _takes_step_callback=lambda: True)
    seen = []
    context = events.RunContext(on_event=seen.append)
    token = events.activate_context(context)
    try:
        call = Pipeline._with_step_callback(owner, arguments)
        call["callback_on_step_end"](pipeline, 0, None, {})
    finally:
        events.deactivate_context(token)
    return [
        e for e in seen if isinstance(e, dict) and e.get("event") == "pipeline_step"
    ]


def _total(steps):
    assert steps, "no pipeline_step event"
    event = steps[0]
    data = event.get("data", event)
    return data.get("total_steps")


def test_progress_counts_six_passes_on_a_step_list():
    assert _total(_step_totals(StepListPipeline(), {"prompt": "x"})) == 6


def test_progress_without_a_step_list_is_unchanged():
    totals = _step_totals(PlainPipeline(), {"prompt": "x", "num_inference_steps": 30})
    assert _total(totals) == 30


# Validate-time warnings


def _accel_pipeline(steps=None, step_list=None, cache=None, tst=None):
    arguments = {"prompt": "a cat"}
    if steps is not None:
        arguments["num_inference_steps"] = steps
    if step_list is not None:
        arguments[STEP_LIST_KEY] = step_list
    if tst is not None:
        arguments["experimental_tst"] = tst
    configuration = {}
    if cache is not None:
        configuration["cache"] = cache
    return {"configuration": configuration, "arguments": arguments}


def test_the_short_schedule_check_counts_six_passes():
    from dw.acceleration_warnings import _cache_warnings, _steps

    pipeline = _accel_pipeline(steps=30, step_list=SEVEN)
    assert _steps(pipeline) == 6
    warnings = _cache_warnings({"type": "first_block"}, pipeline, "p", "", None)
    assert any("cache_short_schedule" in w and "6 steps" in w for w in warnings)


def test_the_short_schedule_check_without_a_step_list_is_unchanged():
    from dw.acceleration_warnings import _cache_warnings, _steps

    pipeline = _accel_pipeline(steps=30)
    assert _steps(pipeline) == 30
    assert _cache_warnings({"type": "first_block"}, pipeline, "p", "", None) == []


def test_validate_warns_mag_calibration_on_a_step_list():
    from dw.acceleration_warnings import STEP_LIST_FEATURE, _cache_warnings

    pipeline = _accel_pipeline(step_list=SEVEN)
    warnings = _cache_warnings(
        {"type": "mag", "calibrate": True}, pipeline, "p", "", None
    )
    found = [w for w in warnings if STEP_LIST_FEATURE in w]
    assert len(found) == 1 and STEP_LIST_UNSUPPORTED in found[0]
    plain = _cache_warnings(
        {"type": "mag", "calibrate": True}, _accel_pipeline(steps=30), "p", "", None
    )
    assert not [w for w in plain if STEP_LIST_FEATURE in w]


def _expanded_context(pipeline):
    return types.SimpleNamespace(
        expanded={"steps": [{"name": "gen", "pipeline": pipeline}]},
        source_indices=[0],
        device_type="cuda",
        capacity_gb=None,
    )


def test_validate_warns_tst_on_a_step_list():
    from dw.acceleration_warnings import STEP_LIST_FEATURE, acceleration_warnings

    tst = {"tau": 0.2}
    warnings = acceleration_warnings(
        _expanded_context(_accel_pipeline(step_list=SEVEN, tst=tst))
    )
    found = [w for w in warnings if STEP_LIST_FEATURE in w]
    assert len(found) == 1 and "TST" in found[0]
    plain = acceleration_warnings(_expanded_context(_accel_pipeline(steps=30, tst=tst)))
    assert not [w for w in plain if STEP_LIST_FEATURE in w]


# Cost scaling


def _definition(default_points):
    return {
        "variables": {STEP_LIST_KEY: default_points, "num_inference_steps": 30},
        "cost_drivers": [STEP_LIST_KEY, "num_inference_steps"],
        "cost": [{"device": "cuda", "vram_gb": 24, "minutes": 6.0}],
    }


def test_cost_scales_by_the_passes_ratio():
    from dw.cost_scaling import _driver_scaling

    definition = _definition([1.0, 0.5, 0.25, 0.0])  # 3 passes measured
    scaling = _driver_scaling(
        definition,
        {"variables": {STEP_LIST_KEY: SEVEN, "num_inference_steps": 30}},
        {},
    )
    factor, axes, unscaled = scaling
    assert factor == pytest.approx(2.0)
    assert "passes 6 vs 3" in axes
    assert unscaled == []


def test_cost_at_the_default_step_list_is_unscaled():
    from dw.cost_scaling import _driver_scaling

    definition = _definition(SEVEN)
    factor, axes, unscaled = _driver_scaling(
        definition,
        {"variables": {STEP_LIST_KEY: list(SEVEN), "num_inference_steps": 30}},
        {},
    )
    assert factor == pytest.approx(1.0) and axes == [] and unscaled == []


def test_cost_steps_scaling_without_a_step_list_is_unchanged():
    from dw.cost_scaling import _driver_scaling

    definition = {
        "variables": {"num_inference_steps": 30},
        "cost_drivers": ["num_inference_steps"],
        "cost": [{"device": "cuda", "vram_gb": 24, "minutes": 6.0}],
    }
    factor, axes, _ = _driver_scaling(
        definition, {"variables": {"num_inference_steps": 60}}, {}
    )
    assert factor == pytest.approx(2.0)
    assert axes == ["steps 60 vs 30"]
