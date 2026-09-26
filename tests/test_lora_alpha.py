"""A lora's network alpha is declared by the workflow, not taken from the file.

peft scales an adapter by `scale * alpha / rank`. The alpha normally comes
from the checkpoint, and that is the right figure whenever the file records
the one it was trained at; the override is for a file that does not (#147).
The 128 below is only a value distinct from the rank - no catalog template
states one, see tests/test_h3_schedule.py.
"""

import pytest
import torch

from dw.pipeline_processors.pipeline import load_loras, set_adapter_alpha


class FakeLoraLayer(torch.nn.Module):
    """Stands in for a peft layer: the two dicts set_scale() divides."""

    def __init__(self, adapters, rank=128):
        super().__init__()
        self.lora_alpha = {name: 8.0 for name in adapters}
        self.r = {name: rank for name in adapters}
        self.scaling = {name: 8.0 / rank for name in adapters}

    def set_scale(self, adapter, scale):
        if adapter in self.scaling:
            self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]


class FakeTransformer(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class FakePipeline:
    _lora_loadable_modules = ["transformer"]

    def __init__(self, layers):
        self.transformer = FakeTransformer(layers)
        self.loaded = []

    def load_lora_weights(self, model_name, adapter_name=None, **kwargs):
        self.loaded.append((model_name, adapter_name, kwargs))

    def set_adapters(self, names, weights):
        for layer in self.transformer.layers:
            for name, weight in zip(names, weights):
                layer.set_scale(name, weight)


@pytest.fixture(autouse=True)
def _peft_layer(monkeypatch):
    """_lora_layers asks peft what a lora layer is; the stub answers for it."""
    import peft.tuners.tuners_utils as tuners_utils

    monkeypatch.setattr(tuners_utils, "BaseTunerLayer", FakeLoraLayer, raising=False)


def test_alpha_overrides_what_the_checkpoint_declared():
    layers = [FakeLoraLayer(["turbo"]) for _ in range(3)]
    pipeline = FakePipeline(layers)

    load_loras(
        [
            {
                "model_name": "lightx2v/Minimax-h3-Turbo",
                "weight_name": "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
                "adapter_name": "turbo",
                "scale": 1.0,
                "alpha": 128,
            }
        ],
        pipeline,
    )

    # upstream's effective_scale = lora_scale * alpha / rank
    for layer in layers:
        assert layer.lora_alpha["turbo"] == 128.0
        assert layer.scaling["turbo"] == pytest.approx(1.0)


def test_without_an_alpha_the_checkpoints_own_figure_stands():
    layers = [FakeLoraLayer(["turbo"])]
    pipeline = FakePipeline(layers)

    load_loras(
        [{"model_name": "m", "adapter_name": "turbo", "scale": 1.0}],
        pipeline,
    )

    assert layers[0].lora_alpha["turbo"] == 8.0
    assert layers[0].scaling["turbo"] == pytest.approx(8.0 / 128)


def test_the_alpha_never_reaches_load_lora_weights():
    """It is applied to the layers afterwards; diffusers takes no such kwarg."""
    pipeline = FakePipeline([FakeLoraLayer(["turbo"])])
    load_loras(
        [{"model_name": "m", "adapter_name": "turbo", "alpha": 128}],
        pipeline,
    )
    _, _, kwargs = pipeline.loaded[0]
    assert "alpha" not in kwargs


def test_the_scale_still_multiplies_the_alpha():
    layers = [FakeLoraLayer(["turbo"])]
    pipeline = FakePipeline(layers)
    load_loras(
        [{"model_name": "m", "adapter_name": "turbo", "scale": 0.5, "alpha": 128}],
        pipeline,
    )
    assert layers[0].scaling["turbo"] == pytest.approx(0.5)


def test_an_alpha_that_would_apply_to_nothing_is_refused():
    """Silently scaling nothing is the failure this exists to prevent."""
    pipeline = FakePipeline([FakeLoraLayer(["other"])])
    with pytest.raises(ValueError, match="no loaded layer carries it"):
        set_adapter_alpha(pipeline, "turbo", 128)


@pytest.mark.parametrize("alpha", [0, -1])
def test_a_non_positive_alpha_is_refused(alpha):
    pipeline = FakePipeline([FakeLoraLayer(["turbo"])])
    with pytest.raises(ValueError, match="must be positive"):
        set_adapter_alpha(pipeline, "turbo", alpha)
