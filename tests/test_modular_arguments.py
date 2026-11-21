"""diffusers 0.43 readiness, stage A of #1315: the modular None rule and the
attention-backend guard."""

import pytest
from diffusers.modular_pipelines import ModularPipeline

from dw.pipeline_processors.modular_arguments import call_arguments, without_none
from dw.pipeline_processors.pipeline import Pipeline


class _Modular(ModularPipeline):
    """A ModularPipeline that records what it is called with."""

    def __init__(self):
        self.seen = None

    def __call__(self, **arguments):
        self.seen = arguments
        return "output"


class _Classic:
    def __init__(self):
        self.seen = None

    def __call__(self, **arguments):
        self.seen = arguments
        return "output"


def _wrapper(fake):
    return Pipeline(
        {
            "configuration": {},
            "from_pretrained_arguments": {"model_name": "x"},
            "arguments": {},
        },
        1,
        "cpu",
        pipeline=fake,
    )


ARGUMENTS = {"a": None, "b": 0, "c": [None], "d": False, "e": ""}


def test_a_modular_call_drops_its_none_arguments():
    """0.43 keeps an explicit None over a block's default; dw passes none."""
    fake = _Modular()
    assert _wrapper(fake)._execute_pipeline(dict(ARGUMENTS)) == "output"
    assert "a" not in fake.seen
    # Falsy values and a nested None are the caller's, and stay
    assert fake.seen == {"b": 0, "c": [None], "d": False, "e": ""}


def test_a_classic_call_keeps_its_none_arguments():
    fake = _Classic()
    assert _wrapper(fake)._execute_pipeline(dict(ARGUMENTS)) == "output"
    assert fake.seen == ARGUMENTS


def test_without_none_is_top_level_only():
    assert without_none({"a": None, "b": {"c": None}}) == {"b": {"c": None}}


def test_call_arguments_leaves_a_classic_pipelines_arguments_alone():
    arguments = {"a": None}
    assert call_arguments(_Classic(), arguments) is arguments
    assert call_arguments(_Modular(), arguments) == {}


class _Component:
    def __init__(self):
        self.backends = []

    def set_attention_backend(self, backend):
        self.backends.append(backend)


def test_backend_pin_without_the_registry_attribute_just_pins(monkeypatch):
    """#1315: the save/restore reads a private attribute; a diffusers without
    it still gets the pin, and nothing raises."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    from dw.pipeline_processors.components import set_component_attention_backend

    monkeypatch.delattr(_AttentionBackendRegistry, "_active_backend")

    def no_restore(*args, **kwargs):
        pytest.fail("the restore ran without the attribute it restores")

    monkeypatch.setattr(_AttentionBackendRegistry, "set_active_backend", no_restore)
    component = _Component()
    set_component_attention_backend(component, "sage_hub")
    assert component.backends == ["sage_hub"]


def test_backend_pin_with_the_registry_attribute_restores_it(monkeypatch):
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    from dw.pipeline_processors.components import set_component_attention_backend

    monkeypatch.setattr(_AttentionBackendRegistry, "_active_backend", "native")
    restored = []
    monkeypatch.setattr(
        _AttentionBackendRegistry, "set_active_backend", restored.append
    )
    component = _Component()
    set_component_attention_backend(component, "sage_hub")
    assert component.backends == ["sage_hub"] and restored == ["native"]
