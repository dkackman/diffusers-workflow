"""A cache hit re-applies LoRA scale/alpha and scheduler shift in place."""

from unittest.mock import MagicMock, patch

import pytest

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
        with patch(
            "dw.pipeline_processors.adapters.set_adapter_scales",
            side_effect=lambda p, n, w: calls.append(("scales", n, w)),
        ):
            apply_adapter_settings(
                [{"model_name": "a/b", "scale": 0.5, "alpha": 8}], pipeline
            )
    assert calls == [("alpha", "0", 8), ("scales", ["0"], [0.5])]
    # The warm path never re-activates: diffusers' set_adapters goes through
    # peft's set_adapter, which sets requires_grad on tensors that are
    # inference tensors after a run under offload
    pipeline.set_adapters.assert_not_called()


def test_the_cold_path_activates_through_set_adapters():
    pipeline = MagicMock()
    apply_adapter_settings(
        [{"model_name": "a/b", "scale": 0.5}], pipeline, activate=True
    )
    pipeline.set_adapters.assert_called_once_with(["0"], [0.5])


class _FakeLoraLayer:
    """A peft-shaped layer: the scaling dict set_scale() recomputes."""

    def __init__(self, adapters, alpha=8.0, rank=16):
        self.lora_alpha = {name: alpha for name in adapters}
        self.r = {name: rank for name in adapters}
        self.scaling = {name: alpha / rank for name in adapters}
        self.activated = 0

    def set_scale(self, adapter, scale):
        if adapter in self.scaling:
            self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def set_adapter(self, names):
        self.activated += 1


def test_the_warm_path_sets_scales_without_touching_activation(monkeypatch):
    from dw.pipeline_processors import adapters as module

    layers = [_FakeLoraLayer(["turbo"]), _FakeLoraLayer(["turbo", "other"])]
    monkeypatch.setattr(module, "_lora_layers", lambda pipeline: iter(layers))
    pipeline = MagicMock()

    apply_adapter_settings(
        [{"model_name": "a/b", "adapter_name": "turbo", "scale": 0.25}], pipeline
    )

    assert [layer.scaling["turbo"] for layer in layers] == [0.125, 0.125]
    assert layers[1].scaling["other"] == 0.5
    assert all(layer.activated == 0 for layer in layers)
    pipeline.set_adapters.assert_not_called()


def test_the_warm_path_refuses_an_adapter_no_layer_carries(monkeypatch):
    from dw.pipeline_processors import adapters as module

    monkeypatch.setattr(
        module, "_lora_layers", lambda pipeline: iter([_FakeLoraLayer(["other"])])
    )
    with pytest.raises(ValueError, match="turbo"):
        apply_adapter_settings(
            [{"model_name": "a/b", "adapter_name": "turbo"}], MagicMock()
        )


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
    scales = MagicMock()
    with (
        patch("dw.pipeline_ownership.emit_phase"),
        patch("dw.pipeline_processors.adapters.set_adapter_scales", scales),
    ):
        wrapper = wrap_resident(cached, step_definition, 1, "cpu", "/out", "p")
    assert wrapper.pipeline is model
    model.scheduler.set_shift.assert_called_once_with(9.0)
    scales.assert_called_once_with(model, ["0"], [0.25])
    model.set_adapters.assert_not_called()


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
