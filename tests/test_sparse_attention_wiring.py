"""attention_backend 'sparse' wired into components, the call, the cache and
the schema (#931).

'sparse' is not a diffusers backend: it installs dw's H3 processor instead of
reaching set_attention_backend or the per-call attention_backend() context,
and its tau is a runtime setting the pipeline cache re-applies on a hit.
"""

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch

from dw.pipeline_processors import components
from dw.pipeline_processors import pipeline as pipeline_module
from dw.pipeline_processors.pipeline import Pipeline, apply_runtime_settings
from dw.schema import load_schema, validate_data
from dw.sparse_attention import (
    DEFAULT_TAU,
    install_sparse,
    is_sparse_installed,
)
from dw.step_cache import pipeline_cache_key, weights_identity
from tests.sparse_layouts import CONFIG


@pytest.fixture
def tiny_h3():
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(**CONFIG).eval()


@pytest.fixture
def component_calls(monkeypatch):
    """Record what configure_components does with a component's backend."""
    calls = []

    def install(component, tau):
        calls.append(("install", tau))
        return type(component).__name__ == "MiniMaxH3Transformer3DModel"

    monkeypatch.setattr(components, "install_sparse", install)
    monkeypatch.setattr(
        components, "uninstall_sparse", lambda component: calls.append(("uninstall",))
    )
    monkeypatch.setattr(
        components,
        "set_component_attention_backend",
        lambda component, backend: calls.append(("set_backend", backend)),
    )
    monkeypatch.setattr(components, "emit_log", lambda *a, **kw: None)
    return calls


@pytest.fixture
def component_warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(
        components, "emit_warning", lambda message, **data: seen.append(message)
    )
    return seen


class MiniMaxH3Transformer3DModel:
    """Only the class name matters to the recorded install stand-in."""


def _configure(component, entry):
    pipeline = SimpleNamespace(transformer=component)
    components.configure_components(
        pipeline, {"components": {"transformer": entry}}, "cpu"
    )


# --- component level -------------------------------------------------------


def test_component_sparse_installs_and_never_reaches_diffusers(component_calls):
    _configure(
        MiniMaxH3Transformer3DModel(),
        {"attention_backend": "sparse", "sparse_tau": 0.7},
    )
    assert component_calls == [("install", 0.7)]


def test_component_sparse_default_tau(component_calls):
    _configure(MiniMaxH3Transformer3DModel(), {"attention_backend": "sparse"})
    assert component_calls == [("install", DEFAULT_TAU)]
    assert DEFAULT_TAU == 0.5


def test_component_sparse_on_non_h3_warns_and_runs_dense(
    component_calls, component_warnings
):
    _configure(SimpleNamespace(), {"attention_backend": "sparse"})
    assert component_warnings == [
        "transformer: attention_backend 'sparse' needs a MiniMax-H3 transformer; "
        "runs dense"
    ]
    assert ("set_backend", "sparse") not in component_calls


def test_non_sparse_backend_uninstalls_sparse(component_calls):
    _configure(MiniMaxH3Transformer3DModel(), {"attention_backend": "sage_hub"})
    assert component_calls == [("uninstall",), ("set_backend", "sage_hub")]


def test_no_backend_uninstalls_sparse(component_calls):
    _configure(MiniMaxH3Transformer3DModel(), {"device": None})
    assert component_calls == [("uninstall",)]


def test_a_real_reconfigure_removes_the_processors(tiny_h3, monkeypatch):
    monkeypatch.setattr(components, "emit_log", lambda *a, **kw: None)
    _configure(tiny_h3, {"attention_backend": "sparse", "sparse_tau": 0.7})
    assert is_sparse_installed(tiny_h3)
    assert tiny_h3._dw_sparse_state.tau == 0.7
    _configure(tiny_h3, {})
    assert not is_sparse_installed(tiny_h3)


@pytest.mark.parametrize("tau", [-0.5, "0.5", True])
def test_component_bad_tau_raises(tiny_h3, monkeypatch, tau):
    """A 'variable:' reference can resolve to anything: a bad tau is the
    workflow's error, never a reason to run dense (Ruling 13)."""
    monkeypatch.setattr(components, "emit_log", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="sparse_tau must be a number >= 0"):
        _configure(tiny_h3, {"attention_backend": "sparse", "sparse_tau": tau})
    assert not is_sparse_installed(tiny_h3)


# --- pipeline level --------------------------------------------------------


class FakePipeline:
    """A pipeline whose call records which components ran with sparse installed."""

    def __init__(self, **parts):
        self.__dict__.update(parts)
        self.seen = None

    def __call__(self, **arguments):
        self.seen = {
            name: is_sparse_installed(value)
            for name, value in vars(self).items()
            if name != "seen"
        }
        return "output"


@pytest.fixture
def no_diffusers_backend(monkeypatch):
    """Fail if the call enters diffusers' per-call attention_backend()."""

    def refuse(backend):
        raise AssertionError(f"attention_backend({backend!r}) entered")

    monkeypatch.setattr(pipeline_module, "attention_backend", refuse)


@pytest.fixture
def call_warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pipeline_module, "emit_warning", lambda message, **data: seen.append(message)
    )
    return seen


def _wrapper(fake, configuration):
    definition = {
        "configuration": configuration,
        "from_pretrained_arguments": {"model_name": "h3-test"},
        "arguments": {},
    }
    return Pipeline(definition, 1, "cpu", pipeline=fake)


def test_pipeline_level_sparse_installs_on_h3_components_for_the_call(
    tiny_h3, no_diffusers_backend, call_warnings
):
    other = torch.nn.Linear(2, 2)
    fake = FakePipeline(transformer=tiny_h3, vae=other)
    wrapper = _wrapper(fake, {"attention_backend": "sparse", "sparse_tau": 0.8})

    assert wrapper._execute_pipeline({}) == "output"

    assert fake.seen["transformer"] is True and fake.seen["vae"] is False
    assert not is_sparse_installed(tiny_h3)
    assert not call_warnings


def test_pipeline_level_sparse_uses_its_tau(tiny_h3, no_diffusers_backend, monkeypatch):
    taus = []
    real = pipeline_module.install_sparse

    def recording(component, tau):
        installed = real(component, tau)
        if installed:
            taus.append(tau)
        return installed

    monkeypatch.setattr(pipeline_module, "install_sparse", recording)
    _wrapper(
        FakePipeline(transformer=tiny_h3), {"attention_backend": "sparse"}
    )._execute_pipeline({})
    _wrapper(
        FakePipeline(transformer=tiny_h3),
        {"attention_backend": "sparse", "sparse_tau": 1.0},
    )._execute_pipeline({})
    assert taus == [DEFAULT_TAU, 1.0]


def test_pipeline_level_sparse_reads_a_components_mapping(
    tiny_h3, no_diffusers_backend
):
    seen = {}

    class Modular:
        components = {"transformer": tiny_h3}

        def __call__(self, **arguments):
            seen["installed"] = is_sparse_installed(tiny_h3)
            return "output"

    _wrapper(Modular(), {"attention_backend": "sparse"})._execute_pipeline({})
    assert seen == {"installed": True}
    assert not is_sparse_installed(tiny_h3)


def test_pipeline_level_sparse_on_non_h3_runs_dense(
    no_diffusers_backend, call_warnings
):
    fake = FakePipeline(transformer=torch.nn.Linear(2, 2))
    wrapper = _wrapper(fake, {"attention_backend": "sparse"})

    assert wrapper._execute_pipeline({}) == "output"
    assert call_warnings == [
        "h3-test: attention_backend 'sparse' found no MiniMax-H3 transformer; "
        "runs dense"
    ]


@pytest.mark.parametrize("tau", [-1, "fast", False])
def test_pipeline_level_bad_tau_raises(tiny_h3, no_diffusers_backend, tau):
    fake = FakePipeline(transformer=tiny_h3)
    wrapper = _wrapper(fake, {"attention_backend": "sparse", "sparse_tau": tau})
    with pytest.raises(ValueError, match="sparse_tau must be a number >= 0"):
        wrapper._execute_pipeline({})
    assert fake.seen is None  # the pipeline was never called
    assert not is_sparse_installed(tiny_h3)


def test_pipeline_level_install_comes_off_when_the_call_raises(
    tiny_h3, no_diffusers_backend
):
    seen = {}

    class Failing(FakePipeline):
        def __call__(self, **arguments):
            seen["installed"] = is_sparse_installed(tiny_h3)
            raise RuntimeError("out of memory")

    wrapper = _wrapper(Failing(transformer=tiny_h3), {"attention_backend": "sparse"})
    with pytest.raises(RuntimeError, match="out of memory"):
        wrapper._execute_pipeline({})
    assert seen == {"installed": True}
    assert not is_sparse_installed(tiny_h3)


def test_component_install_survives_a_pipeline_level_call(
    tiny_h3, no_diffusers_backend, call_warnings
):
    install_sparse(tiny_h3, 0.7)
    fake = FakePipeline(transformer=tiny_h3)
    _wrapper(
        fake, {"attention_backend": "sparse", "sparse_tau": 1.0}
    )._execute_pipeline({})

    assert fake.seen["transformer"] is True
    assert is_sparse_installed(tiny_h3)
    # Not this call's install, so not this call's tau either
    assert tiny_h3._dw_sparse_state.tau == 0.7
    assert not call_warnings


def test_other_backends_still_reach_diffusers(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def recording(backend):
        entered.append(backend)
        yield

    monkeypatch.setattr(pipeline_module, "attention_backend", recording)
    _wrapper(FakePipeline(), {"attention_backend": "sage_hub"})._execute_pipeline({})
    assert entered == ["sage_hub"]


def test_every_call_enters_sparse_call(monkeypatch):
    labels = []

    @contextlib.contextmanager
    def recording(pipeline, label):
        labels.append(label)
        yield SimpleNamespace(ran_sparse=False, reason=None)

    monkeypatch.setattr(pipeline_module, "sparse_call", recording)
    wrapper = _wrapper(FakePipeline(), {})
    wrapper._execute_pipeline({})
    wrapper.segment_label = "segment 2"
    wrapper._execute_pipeline({})
    assert labels == ["h3-test", "segment 2"]


# --- the pipeline cache ----------------------------------------------------


def test_sparse_tau_is_runtime_not_identity():
    a = {
        "configuration": {
            "components": {
                "transformer": {"attention_backend": "sparse", "sparse_tau": 0.5}
            }
        }
    }
    b = copy.deepcopy(a)
    b["configuration"]["components"]["transformer"]["sparse_tau"] = 1.0
    assert pipeline_cache_key(a) == pipeline_cache_key(b)

    c = copy.deepcopy(a)
    del c["configuration"]["components"]["transformer"]["sparse_tau"]
    assert pipeline_cache_key(a) == pipeline_cache_key(c)

    d = copy.deepcopy(a)
    d["configuration"]["components"]["transformer"]["attention_backend"] = "sage_hub"
    assert pipeline_cache_key(a) != pipeline_cache_key(d)


def test_pipeline_level_sparse_tau_is_runtime_not_identity():
    a = {"configuration": {"attention_backend": "sparse", "sparse_tau": 0.5}}
    b = copy.deepcopy(a)
    b["configuration"]["sparse_tau"] = 1.0
    assert pipeline_cache_key(a) == pipeline_cache_key(b)


def test_weights_identity_leaves_the_definition_alone():
    definition = {
        "configuration": {
            "sparse_tau": 0.5,
            "components": {"transformer": {"sparse_tau": 0.5}},
        }
    }
    before = copy.deepcopy(definition)
    weights_identity(definition)
    assert definition == before


def test_cache_hit_reapplies_tau(tiny_h3):
    install_sparse(tiny_h3, 0.5)
    model = SimpleNamespace(transformer=tiny_h3, vae=None)
    definition = {
        "configuration": {
            "components": {
                "transformer": {"attention_backend": "sparse", "sparse_tau": 1.0}
            }
        }
    }
    apply_runtime_settings(definition, model)
    assert tiny_h3._dw_sparse_state.tau == 1.0

    del definition["configuration"]["components"]["transformer"]["sparse_tau"]
    apply_runtime_settings(definition, model)
    assert tiny_h3._dw_sparse_state.tau == DEFAULT_TAU


def test_cache_hit_bad_tau_raises(tiny_h3):
    install_sparse(tiny_h3, 0.5)
    model = SimpleNamespace(transformer=tiny_h3)
    definition = {
        "configuration": {
            "components": {
                "transformer": {"attention_backend": "sparse", "sparse_tau": -1}
            }
        }
    }
    with pytest.raises(ValueError, match="sparse_tau must be a number >= 0"):
        apply_runtime_settings(definition, model)
    assert tiny_h3._dw_sparse_state.tau == 0.5


def test_cache_hit_skips_a_component_without_sparse(tiny_h3):
    model = SimpleNamespace(transformer=tiny_h3)
    definition = {
        "configuration": {
            "components": {
                "transformer": {"attention_backend": "sparse", "sparse_tau": 1.0}
            }
        }
    }
    apply_runtime_settings(definition, model)
    assert not is_sparse_installed(tiny_h3)


# --- the schema ------------------------------------------------------------


def test_schema_accepts_sparse_tau():
    workflow = {
        "id": "sparse_test",
        "steps": [
            {
                "name": "main",
                "pipeline": {
                    "configuration": {
                        "component_type": "MiniMaxH3ModularPipeline",
                        "attention_backend": "sparse",
                        "sparse_tau": 0.5,
                        "components": {
                            "transformer": {
                                "attention_backend": "sparse",
                                "sparse_tau": 1.0,
                            }
                        },
                    },
                    "from_pretrained_arguments": {"model_name": "test"},
                    "arguments": {"prompt": "test"},
                },
            }
        ],
    }
    schema = load_schema("workflow")
    status, message = validate_data(workflow, schema)
    assert status is True, message

    workflow["steps"][0]["pipeline"]["configuration"]["sparse_tau"] = -1
    status, _ = validate_data(workflow, schema)
    assert status is False


def test_schema_describes_sparse():
    schema = load_schema("workflow")
    configuration = schema["$defs"]["pipeline_configuration"]["properties"]
    entry = configuration["components"]["additionalProperties"]["properties"]
    for block in (configuration, entry):
        assert "'sparse'" in block["attention_backend"]["description"]
        # A string is only a 'variable:' reference: every H3 template wires one.
        assert block["sparse_tau"]["type"] == ["number", "string"]
        assert block["sparse_tau"]["pattern"] == "^variable:"
        assert "0.5" in block["sparse_tau"]["description"]
        assert "#931" in block["sparse_tau"]["description"]
