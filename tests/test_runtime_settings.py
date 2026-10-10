"""A cache hit re-applies LoRA scale/alpha and scheduler shift in place."""

from unittest.mock import MagicMock, patch

from dw.pipeline_processors.adapters import adapter_settings, apply_adapter_settings
from dw.pipeline_processors.components import apply_scheduler_shift


def test_adapter_settings_name_by_original_index_and_skip_nulled():
    loras = [
        {"model_name": None, "scale": 0.3},
        {"model_name": "a/b", "scale": 0.5, "alpha": 8},
        {"model_name": "c/d", "adapter_name": "turbo"},
    ]
    names, weights, alphas = adapter_settings(loras)
    assert names == ["1", "turbo"]
    assert weights == [0.5, 1.0]
    assert alphas == {"1": 8}
    assert loras[1]["scale"] == 0.5  # read, not popped


def test_apply_adapter_settings_sets_alpha_then_scales():
    pipeline = MagicMock()
    calls = []
    with patch(
        "dw.pipeline_processors.adapters.set_adapter_alpha",
        side_effect=lambda p, n, a: calls.append(("alpha", n, a)),
    ):
        pipeline.set_adapters.side_effect = lambda n, w: calls.append(("scales", n, w))
        apply_adapter_settings(
            [{"model_name": "a/b", "scale": 0.5, "alpha": 8}], pipeline
        )
    assert calls == [("alpha", "0", 8), ("scales", ["0"], [0.5])]


def test_apply_adapter_settings_is_a_no_op_without_active_loras():
    pipeline = MagicMock()
    apply_adapter_settings([{"model_name": None}], pipeline)
    apply_adapter_settings(None, pipeline)
    pipeline.set_adapters.assert_not_called()


def test_apply_scheduler_shift_sets_the_shift():
    pipeline = MagicMock()
    apply_scheduler_shift({"shift": 12}, pipeline, "scheduler")
    pipeline.scheduler.set_shift.assert_called_once_with(12.0)


def test_apply_scheduler_shift_without_a_shift_touches_nothing():
    pipeline = MagicMock()
    apply_scheduler_shift(None, pipeline, "scheduler")
    apply_scheduler_shift({"scheduler_type": "X"}, pipeline, "scheduler")
    pipeline.scheduler.set_shift.assert_not_called()


def test_wrap_resident_reapplies_runtime_settings():
    from dw.pipeline_ownership import wrap_resident

    model = MagicMock()
    cached = MagicMock()
    cached.pipeline = model
    step_definition = {
        "name": "s",
        "pipeline": {
            "configuration": {"component_type": "X", "no_generator": True},
            "loras": [{"model_name": "a/b", "scale": 0.25}],
            "scheduler": {"shift": 9},
        },
    }
    with patch("dw.pipeline_ownership.emit_phase"):
        wrapper = wrap_resident(cached, step_definition, 1, "cpu", "/out", "p")
    assert wrapper.pipeline is model
    model.scheduler.set_shift.assert_called_once_with(9.0)
    model.set_adapters.assert_called_once_with(["0"], [0.25])


def test_wrap_resident_with_nothing_mutable_calls_nothing():
    from dw.pipeline_ownership import wrap_resident

    model = MagicMock()
    cached = MagicMock()
    cached.pipeline = model
    step_definition = {
        "name": "s",
        "pipeline": {"configuration": {"component_type": "X", "no_generator": True}},
    }
    with patch("dw.pipeline_ownership.emit_phase"):
        wrap_resident(cached, step_definition, 1, "cpu", "/out", "p")
    model.set_adapters.assert_not_called()
    model.scheduler.set_shift.assert_not_called()
