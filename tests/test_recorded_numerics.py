"""#1643 stage 2: the torch numerics a run records, and defaults read from
their source - the enhancer's model from its builtin, H3's step count from
the diffusers block."""

import json
import sys
import types

import pytest
import torch

from dw import torch_numerics
from dw.pipeline_processors import h3_lead_in, h3_rules, h3_tst
from dw.workflow_run import RunRecord, write_run_manifest

from .test_run_manifest_roles import _stand_in
from .test_runs import _workflow_definition, fake_pipeline  # noqa: F401

# --- H: the manifest's torch block ------------------------------------------


@pytest.fixture
def flipped_flags():
    """The three flags set away from whatever they were, so a recorded value
    can only have come from a live read."""
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    torch.backends.cuda.matmul.allow_tf32 = not saved[0]
    torch.backends.cudnn.benchmark = not saved[1]
    torch.backends.cudnn.deterministic = not saved[2]
    yield tuple(not value for value in saved)
    (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    ) = saved


def test_torch_numerics_reads_the_live_flags(flipped_flags):
    import diffusers

    tf32, benchmark, deterministic = flipped_flags
    assert torch_numerics() == {
        "version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "enable_tf32": tf32,
        "cudnn_benchmark": benchmark,
        "cudnn_deterministic": deterministic,
    }


def test_the_manifest_records_a_torch_block(tmp_path, flipped_flags):
    import diffusers

    tf32, benchmark, deterministic = flipped_flags
    write_run_manifest(
        _stand_in(tmp_path, [{"step": "a", "files": []}]),
        RunRecord(status="completed"),
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # A dict, not torch_flags_report()'s string
    assert manifest["torch"] == {
        "version": str(torch.__version__),
        "diffusers_version": diffusers.__version__,
        "enable_tf32": tf32,
        "cudnn_benchmark": benchmark,
        "cudnn_deterministic": deterministic,
    }
    # Additive: the keys that were there still are
    assert {"dw_version", "device", "steps", "status"} <= set(manifest)


def test_a_real_run_writes_the_block(tmp_path, fake_pipeline):  # noqa: F811
    from dw.workflow import Workflow

    Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Num.json").run({})

    run_dir = next((tmp_path / "Num").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["torch"] == torch_numerics()


# --- F: the enhancer's model ------------------------------------------------


def _h3_arguments(**kwargs):
    from dw.server.enhancers import build_enhance_workflow

    step = build_enhance_workflow("h3", "an idea", **kwargs)["steps"][0]
    return step["workflow"]["arguments"]


def test_the_h3_enhancer_leaves_the_model_to_its_builtin():
    assert "model_name" not in _h3_arguments()


def test_a_named_model_still_reaches_the_h3_builtin():
    assert _h3_arguments(model_name="Qwen/Qwen2.5-7B-Instruct")["model_name"] == (
        "Qwen/Qwen2.5-7B-Instruct"
    )


def test_t2i_keeps_its_default_model():
    from dw.server.enhancers import PRESETS, build_enhance_workflow

    step = build_enhance_workflow("t2i", "an idea")["steps"][0]
    assert step["task"]["arguments"]["model_name"] == PRESETS["t2i"]["default_model"]
    step = build_enhance_workflow("t2i", "an idea", model_name="m/x")["steps"][0]
    assert step["task"]["arguments"]["model_name"] == "m/x"


def test_the_advertised_h3_model_is_the_one_the_builtin_runs():
    """`default_model` is what list_enhancers and the UI show; the builtin's
    `model_name` variable is what runs. They can't drift apart."""
    from importlib import resources

    from dw.server.enhancers import PRESETS, preset_descriptions

    builtin = json.loads(
        resources.files("dw").joinpath("workflows", "h3_context_ir.json").read_text()
    )
    declared = builtin["variables"]["model_name"]
    assert PRESETS["h3"]["default_model"] == declared
    advertised = {d["key"]: d["default_model"] for d in preset_descriptions()}
    assert advertised["h3"] == declared


# --- C: H3's step count off the block ---------------------------------------


@pytest.fixture
def fresh_default():
    h3_rules.default_num_inference_steps.cache_clear()
    yield
    h3_rules.default_num_inference_steps.cache_clear()


def test_the_steps_are_the_blocks_declared_default(fresh_default):
    from diffusers.modular_pipelines.minimax_h3 import before_denoise

    declared = next(
        p.default
        for p in before_denoise.MiniMaxH3SetTimestepsStep().inputs
        if p.name == "num_inference_steps"
    )
    assert h3_tst.DEFAULT_STEPS == declared
    assert h3_lead_in.DEFAULT_STEPS == declared
    # Unchanged by the move
    assert declared == 50


def _stub_before_denoise(monkeypatch, step_class):
    module = types.ModuleType("diffusers.modular_pipelines.minimax_h3.before_denoise")
    if step_class is not None:
        module.MiniMaxH3SetTimestepsStep = step_class
    package = types.ModuleType("diffusers.modular_pipelines.minimax_h3")
    package.before_denoise = module
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", package)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.modular_pipelines.minimax_h3.before_denoise",
        module,
    )


def _step_declaring(default):
    class Step:
        inputs = [types.SimpleNamespace(name="num_inference_steps", default=default)]

    return Step


@pytest.mark.parametrize(
    "step_class",
    [
        None,  # diffusers has no such step
        _step_declaring(None),  # it declares no default
        _step_declaring("50"),  # not a whole number
    ],
)
def test_the_steps_fall_back_when_diffusers_declares_none(
    monkeypatch, fresh_default, step_class
):
    _stub_before_denoise(monkeypatch, step_class)
    assert h3_rules.default_num_inference_steps() == (
        h3_rules.FALLBACK_NUM_INFERENCE_STEPS
    )
    assert h3_tst.DEFAULT_STEPS == h3_rules.FALLBACK_NUM_INFERENCE_STEPS == 50


def test_the_steps_follow_a_different_declared_default(monkeypatch, fresh_default):
    _stub_before_denoise(monkeypatch, _step_declaring(30))
    assert h3_tst.DEFAULT_STEPS == 30
    assert h3_tst._evaluations(None) == 29
    assert h3_lead_in.evaluations(None, {}) == 29


def test_the_steps_fall_back_without_the_h3_package(monkeypatch, fresh_default):
    # None in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", None)
    assert h3_rules.default_num_inference_steps() == 50


def test_an_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        h3_tst.NOT_A_THING  # noqa: B018
