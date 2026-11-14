"""`turbo_lead_in` (#1326, #1352): the first N transformer evaluations of an
H3 call run the `turbo` LoRA adapter at scale 0, then its configured scale.

Real peft LoRA layers carrying two adapters under a fake H3 transformer, the
real `lead_in_call` hook, the validate and run rules, and the step wrapper."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.lora import LoraConfig

from dw.pipeline_processors import chain
from dw.pipeline_processors.h3_lead_in import (
    LEAD_IN_INPUT,
    lead_in_call,
    lead_in_count,
    lead_in_errors,
)
from dw.pipeline_processors.pipeline_h3 import H3Arguments

TURBO_SCALE = 1.5  # lora_alpha 3 / r 2
STYLE_SCALE = 4.0  # lora_alpha 8 / r 2
# Noise levels of the denoise loop, one forward per evaluation
TIMESTEPS = [1.0, 0.75, 0.5, 0.25]


def lora_layer():
    config = LoraConfig(r=2, lora_alpha=3)
    layer = LoraLinear(torch.nn.Linear(4, 4), "turbo", config=config, r=2, lora_alpha=3)
    layer.update_layer("style", r=2, lora_alpha=8, config=config)
    return layer


class FakeH3(torch.nn.Module):
    """Stands in for the H3 transformer: records each layer's scalings."""

    def __init__(self, layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(lora_layer() for _ in range(layers))
        self.seen = []

    def forward(self, hidden_states=None, timestep=None, **kwargs):
        self.seen.append(
            [(m.scaling["turbo"], m.scaling["style"]) for m in self.layers]
        )
        return hidden_states


FakeH3.__name__ = "MiniMaxH3Transformer3DModel"


def timestep(level):
    """Per-token timesteps: clean conditioning rows at 0, the target at `level`."""
    return torch.tensor([[0.0, 0.0, level, level]])


def pipeline_of(*transformers):
    names = ["transformer", "transformer_2"]
    return SimpleNamespace(components=dict(zip(names, transformers)))


def run_loop(transformer, levels=TIMESTEPS):
    for level in levels:
        transformer(hidden_states=torch.zeros(1), timestep=timestep(level))


def turbo_seen(transformer):
    """Each evaluation's turbo scaling (the same on every layer)."""
    values = []
    for per_layer in transformer.seen:
        assert len({turbo for turbo, _ in per_layer}) == 1
        values.append(per_layer[0][0])
    return values


@pytest.fixture
def h3():
    return FakeH3()


def test_fixture_layers_carry_two_adapters(h3):
    assert h3.layers[0].scaling == {"turbo": TURBO_SCALE, "style": STYLE_SCALE}


# 1. The switch, per evaluation


@pytest.mark.parametrize("lead_in", [1, 2, 3])
def test_turbo_is_zero_for_the_first_n_evaluations_only(h3, lead_in):
    pipeline = pipeline_of(h3)
    with lead_in_call(pipeline, lead_in, 4, "clip"):
        run_loop(h3)
    expected = [0.0] * lead_in + [TURBO_SCALE] * (4 - lead_in)
    assert turbo_seen(h3) == expected
    assert {style for per in h3.seen for _, style in per} == {STYLE_SCALE}


def test_scaling_is_restored_exactly_and_no_hook_remains(h3):
    with lead_in_call(pipeline_of(h3), 2, 4, "clip"):
        assert h3.layers[0].scaling["turbo"] == 0.0
        run_loop(h3)
    assert all(
        m.scaling == {"turbo": TURBO_SCALE, "style": STYLE_SCALE} for m in h3.layers
    )
    assert not h3._forward_pre_hooks


# 2. Distinct timesteps are the count


def test_two_partitions_at_one_timestep_are_one_evaluation():
    first, second = FakeH3(), FakeH3()
    with lead_in_call(pipeline_of(first, second), 1, 4, "clip"):
        for level in TIMESTEPS:
            for part in (first, second):
                part(timestep=timestep(level))
    assert turbo_seen(first) == [0.0] + [TURBO_SCALE] * 3
    # The second partition's forward at evaluation 1 sees the lead-in too
    assert turbo_seen(second) == [0.0] + [TURBO_SCALE] * 3
    assert not first._forward_pre_hooks and not second._forward_pre_hooks


# 3. An error mid-loop


def test_restored_when_the_loop_raises(h3):
    with pytest.raises(RuntimeError, match="boom"):
        with lead_in_call(pipeline_of(h3), 2, 4, "clip"):
            h3(timestep=timestep(1.0))
            assert h3.layers[0].scaling["turbo"] == 0.0
            raise RuntimeError("boom")
    assert all(m.scaling["turbo"] == TURBO_SCALE for m in h3.layers)
    assert not h3._forward_pre_hooks


# 4. No-ops and refusals of the context manager


@pytest.mark.parametrize("lead_in", [0, None])
def test_zero_or_none_touches_nothing(h3, lead_in):
    with lead_in_call(pipeline_of(h3), lead_in, 4, "clip") as state:
        assert state is None
        assert not h3._forward_pre_hooks
        run_loop(h3)
    assert turbo_seen(h3) == [TURBO_SCALE] * 4


def test_no_turbo_layer_lists_the_adapters_it_has():
    transformer = FakeH3(layers=1)
    layer = transformer.layers[0]
    layer.delete_adapter("turbo")
    with pytest.raises(ValueError, match="style"):
        with lead_in_call(pipeline_of(transformer), 1, 4, "clip"):
            pass
    assert not transformer._forward_pre_hooks


def test_no_h3_transformer_raises():
    pipeline = SimpleNamespace(components={"unet": torch.nn.Linear(2, 2)})
    with pytest.raises(ValueError, match="no MiniMax-H3 transformer"):
        with lead_in_call(pipeline, 1, 4, "clip"):
            pass


# 5. The event


def test_one_event_per_call(h3):
    with patch("dw.pipeline_processors.h3_lead_in.emit_log") as emit:
        with lead_in_call(pipeline_of(h3), 2, 4, "clip"):
            run_loop(h3)
    emit.assert_called_once()
    assert (
        "turbo lead-in: evaluations 1..2 with adapter 'turbo' at 0 of 4"
        in (emit.call_args.args[0])
    )


# 6. The validate rule


def h3_workflow(arguments=None, configuration=None, loras=None, workflow="fl2va"):
    return {
        "steps": [
            {
                "name": "clip",
                "pipeline": {
                    "configuration": {
                        "component_type": "ModularPipeline",
                        **(configuration or {}),
                    },
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": workflow,
                    },
                    "loras": loras
                    if loras is not None
                    else [{"model_name": "x", "adapter_name": "turbo"}],
                    "arguments": {"num_inference_steps": 5, **(arguments or {})},
                },
            }
        ]
    }


def errors_for(value, **kwargs):
    arguments = {LEAD_IN_INPUT: value, **kwargs.pop("arguments", {})}
    return lead_in_errors(h3_workflow(arguments, **kwargs))


PATH = f"steps[0].pipeline.arguments.{LEAD_IN_INPUT}"


@pytest.mark.parametrize("value", [1, 3])
def test_a_valid_lead_in_passes(value):
    assert errors_for(value) == []


@pytest.mark.parametrize("value", [0, None, "variable:n"])
def test_zero_null_and_unresolved_are_left_alone(value):
    assert errors_for(value) == []


@pytest.mark.parametrize("value", [1.5, "2", True, -1])
def test_a_bad_value_is_refused(value):
    errors = errors_for(value)
    assert [e["path"] for e in errors] == [PATH]
    assert "whole number" in errors[0]["message"]


def test_n_at_or_past_the_evaluations_is_refused():
    errors = errors_for(4)  # 5 steps run 4 evaluations
    assert "must be less than 4" in errors[0]["message"]
    assert errors_for(3) == []


def test_missing_steps_default_to_fifty():
    arguments = {"num_inference_steps": None}
    assert errors_for(48, arguments=arguments) == []
    assert "must be less than 49" in errors_for(49, arguments=arguments)[0]["message"]


def test_no_turbo_adapter_lists_the_names_it_has():
    loras = [
        {"model_name": "a", "adapter_name": "style"},
        {"model_name": "b"},
    ]
    message = errors_for(1, loras=loras)[0]["message"]
    assert "'style'" in message and "'1'" in message
    errors = errors_for(1, loras=[{"model_name": "a"}])
    assert "'0'" in errors[0]["message"]
    assert errors_for(1, loras=[]) and "none" in errors_for(1, loras=[])[0]["message"]


def test_first_block_cache_is_refused_and_mag_is_not():
    bad = errors_for(1, configuration={"cache": {"type": "first_block"}})
    assert "first_block" in bad[0]["message"]
    assert errors_for(1, configuration={"cache": {"type": "mag"}}) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"configuration": {"component_type": "FluxPipeline"}},
        {"workflow": "not_an_h3_workflow"},
    ],
)
def test_a_non_h3_step_is_refused(kwargs):
    errors = errors_for(1, **kwargs)
    assert "MiniMax-H3 argument" in errors[0]["message"]


# 7. The run rule


PIPELINE = SimpleNamespace(components={})


@pytest.mark.parametrize("value", [None, 0])
def test_run_zero_or_none_is_no_lead_in(value):
    assert lead_in_count(value, PIPELINE, {"num_inference_steps": 5}, {}) is None


def test_run_accepts_a_valid_count():
    assert lead_in_count(3, PIPELINE, {"num_inference_steps": 5}, {}) == 3


@pytest.mark.parametrize("value", [1.5, "2", True, -1])
def test_run_refuses_a_bad_value(value):
    with pytest.raises(ValueError, match="whole number"):
        lead_in_count(value, PIPELINE, {}, {})


def test_run_refuses_n_past_the_evaluations():
    with pytest.raises(ValueError, match="must be less than 4"):
        lead_in_count(4, PIPELINE, {"num_inference_steps": 5}, {})


def test_run_refuses_a_first_block_cache():
    with pytest.raises(ValueError, match="first_block"):
        lead_in_count(1, PIPELINE, {}, {"cache": {"type": "first_block"}})
    assert lead_in_count(1, PIPELINE, {}, {"cache": {"type": "mag"}}) == 1


# 8. The step wrapper


def step_for(pipeline):
    class Step(H3Arguments):
        name = "clip"
        configuration = {}

    Step.pipeline = pipeline
    return Step()


def test_take_lead_in_strips_the_argument(h3):
    step = step_for(pipeline_of(h3))
    original = {"num_inference_steps": 5, LEAD_IN_INPUT: 2}
    arguments, lead_in = step._take_lead_in(original)
    assert LEAD_IN_INPUT not in arguments and lead_in == 2
    assert LEAD_IN_INPUT in original
    assert step._take_lead_in({"a": 1}) == ({"a": 1}, None)
    assert step._take_lead_in({LEAD_IN_INPUT: None}) == ({}, None)


def test_take_lead_in_refuses_a_pipeline_without_h3():
    step = step_for(SimpleNamespace(components={"unet": torch.nn.Linear(2, 2)}))
    with pytest.raises(ValueError, match="no MiniMax-H3 transformer"):
        step._take_lead_in({LEAD_IN_INPUT: 1})


def test_take_lead_in_wraps_a_rule_refusal_with_the_step_name(h3):
    step = step_for(pipeline_of(h3))
    with pytest.raises(ValueError, match="Step 'clip'.*must be less than 4"):
        step._take_lead_in({"num_inference_steps": 5, LEAD_IN_INPUT: 4})


# 9. Chains: each segment is its own call, so the count starts again


def test_each_segment_restarts_the_count(h3):
    pipeline = pipeline_of(h3)
    for _ in range(2):
        h3.seen.clear()
        with lead_in_call(pipeline, 2, 4, "clip"):
            run_loop(h3)
        assert turbo_seen(h3) == [0.0, 0.0, TURBO_SCALE, TURBO_SCALE]
        assert h3.layers[0].scaling["turbo"] == TURBO_SCALE
        assert not h3._forward_pre_hooks


def test_segment_arguments_carry_the_lead_in():
    config = SimpleNamespace(prompts=[], total_frames=None, source_audio=None)
    segment = SimpleNamespace(index=0)
    result = chain._segment_arguments(
        {LEAD_IN_INPUT: 2, "num_inference_steps": 5}, config, segment, None, None
    )
    assert result[LEAD_IN_INPUT] == 2
