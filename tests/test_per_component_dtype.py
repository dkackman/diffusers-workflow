"""#1247: from_pretrained_arguments.torch_dtype accepts diffusers' per-component
dict form - each value loads as a dtype, an escaped or unknown one is refused
naming its component, and the readers that assumed one dtype handle the dict."""

import json
import logging
import os

import jsonschema
import pytest
import torch

from dw.arguments import realize_args, realize_per_component_dtype
from dw.pipeline_processors.components import _warn_float16_on_mps
from dw.quantized_cache import _Candidate, component_dtype
from dw.step_cache import pipeline_cache_key
from dw.type_references import component_type_errors, unbraced_type_warnings


def _arguments(torch_dtype):
    return {"model_name": "org/model", "torch_dtype": torch_dtype}


# --- realization ---------------------------------------------------------


def test_a_dict_realizes_each_value_to_a_dtype():
    arguments = _arguments({"default": "torch.bfloat16", "vocoder": "torch.float32"})
    realize_args(arguments)
    assert arguments["torch_dtype"] == {
        "default": torch.bfloat16,
        "vocoder": torch.float32,
    }


def test_component_names_are_not_special_cased():
    arguments = _arguments({"default": "torch.bfloat16", "anything_at_all": "torch.float16"})
    realize_args(arguments)
    assert arguments["torch_dtype"]["anything_at_all"] is torch.float16


def test_a_second_realize_pass_keeps_the_loaded_dtypes():
    # Arguments are realized for the variables and again for the steps
    arguments = _arguments({"default": "torch.bfloat16", "vocoder": "torch.float32"})
    realize_args(arguments)
    realize_args(arguments)
    assert arguments["torch_dtype"]["vocoder"] is torch.float32


def test_the_dict_form_survives_a_variables_pass_then_realizes():
    variables = {"dtypes": {"default": "torch.bfloat16", "vocoder": "torch.float32"}}
    realize_args(variables, apply_key_conventions=False)
    arguments = _arguments(variables["dtypes"])
    realize_args(arguments)
    assert arguments["torch_dtype"]["vocoder"] is torch.float32


def test_an_escaped_value_is_refused_naming_the_component():
    with pytest.raises(ValueError, match="'vocoder'.*escaped"):
        realize_args(_arguments({"default": "torch.bfloat16", "vocoder": "{float32}"}))


def test_an_unknown_name_is_refused_naming_the_component():
    with pytest.raises(ValueError, match="'audio_vae'.*unknown dtype"):
        realize_args(
            _arguments({"default": "torch.bfloat16", "audio_vae": "torch.float33"})
        )


def test_a_non_string_value_is_refused_naming_the_component():
    with pytest.raises(ValueError, match="'vocoder'"):
        realize_per_component_dtype({"vocoder": 32})


def test_a_string_torch_dtype_is_unchanged():
    arguments = _arguments("torch.bfloat16")
    realize_args(arguments)
    assert arguments["torch_dtype"] is torch.bfloat16


def test_another_dtype_key_with_a_dict_is_left_alone():
    arguments = {"weights_dtype": {"a": "{int8}"}}
    realize_args(arguments)
    assert arguments["weights_dtype"] == {"a": "{int8}"}


# --- validation -----------------------------------------------------------


def _definition(torch_dtype):
    return {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "configuration": {"component_type": "LTX2Pipeline"},
                    "from_pretrained_arguments": _arguments(torch_dtype),
                    "arguments": {},
                },
            }
        ]
    }


def test_validation_passes_a_good_dict():
    definition = _definition({"default": "torch.bfloat16", "vocoder": "torch.float32"})
    assert component_type_errors(definition) == []
    assert unbraced_type_warnings(definition) == []


def test_validation_refuses_an_escaped_value_at_the_components_path():
    errors = component_type_errors(
        _definition({"default": "torch.bfloat16", "vocoder": "{float32}"})
    )
    assert len(errors) == 1
    assert errors[0]["path"].endswith("torch_dtype.vocoder")
    assert "'vocoder'" in errors[0]["message"]


def test_validation_refuses_an_unknown_name_at_the_components_path():
    errors = component_type_errors(
        _definition({"default": "torch.bfloat16", "audio_vae": "torch.float33"})
    )
    assert len(errors) == 1
    assert errors[0]["path"].endswith("torch_dtype.audio_vae")
    assert "'audio_vae'" in errors[0]["message"]


def test_validation_warns_when_default_is_missing():
    warnings = unbraced_type_warnings(_definition({"vocoder": "torch.float32"}))
    assert len(warnings) == 1
    assert "float32" in warnings[0] and "default" in warnings[0]


def test_validation_leaves_a_string_torch_dtype_as_before():
    assert component_type_errors(_definition("torch.bfloat16")) == []


def _from_pretrained_schema():
    path = os.path.join(os.path.dirname(__file__), "..", "dw", "workflow_schema.json")
    with open(path) as f:
        schema = json.load(f)
    return {"$defs": schema["$defs"], "$ref": "#/$defs/from_pretrained_arguments"}


def test_the_schema_allows_the_object_form_and_the_string_form():
    schema = _from_pretrained_schema()
    jsonschema.validate(
        _arguments({"default": "torch.bfloat16", "vocoder": "torch.float32"}), schema
    )
    jsonschema.validate(_arguments("torch.bfloat16"), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_arguments({"vocoder": 32}), schema)


# --- the readers that assumed one dtype ------------------------------------


def test_mps_float16_warning_names_each_float16_component(caplog):
    with caplog.at_level(logging.WARNING, logger="dw"):
        _warn_float16_on_mps(
            "pipeline",
            {"torch_dtype": {"default": torch.bfloat16, "vocoder": torch.float16}},
            "mps",
        )
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "pipeline.vocoder loads in float16" in messages[0]


def test_mps_float16_warning_is_quiet_for_a_dict_without_float16(caplog):
    with caplog.at_level(logging.WARNING, logger="dw"):
        _warn_float16_on_mps(
            "pipeline",
            {"torch_dtype": {"default": torch.bfloat16, "vocoder": torch.float32}},
            "mps",
        )
    assert caplog.records == []


def test_mps_float16_warning_still_fires_for_a_single_dtype(caplog):
    with caplog.at_level(logging.WARNING, logger="dw"):
        _warn_float16_on_mps("unet", {"torch_dtype": torch.float16}, "mps")
    assert any("unet loads in float16" in r.getMessage() for r in caplog.records)


def test_quantized_cache_resolves_a_dict_per_component():
    dtypes = {"default": torch.bfloat16, "vocoder": torch.float32}
    assert component_dtype(dtypes, "vocoder") is torch.float32
    assert component_dtype(dtypes, "transformer") is torch.bfloat16
    assert component_dtype({"vocoder": torch.float32}, "transformer") is torch.float32
    assert component_dtype(torch.bfloat16, "transformer") is torch.bfloat16


def test_quantized_cache_keys_on_the_components_own_dtype():
    from types import SimpleNamespace

    from sdnq import SDNQConfig

    spec = SimpleNamespace(
        pretrained_model_name_or_path="org/model", type_hint=object, load_id=None
    )
    config = SDNQConfig(weights_dtype="int8")
    candidate = _Candidate(
        "transformer",
        spec,
        {
            "quantization_config": config,
            "torch_dtype": {"default": torch.bfloat16, "vocoder": torch.float32},
        },
    )
    assert candidate.torch_dtype is torch.bfloat16


def _pipeline(torch_dtype):
    return {
        "configuration": {"component_type": "LTX2Pipeline"},
        "from_pretrained_arguments": _arguments(torch_dtype),
    }


def test_warm_pipeline_cache_keys_the_dtype_variants_apart():
    bf16 = pipeline_cache_key(_pipeline("torch.bfloat16"))
    vocoder = pipeline_cache_key(
        _pipeline({"default": "torch.bfloat16", "vocoder": "torch.float32"})
    )
    both = pipeline_cache_key(
        _pipeline(
            {
                "default": "torch.bfloat16",
                "vocoder": "torch.float32",
                "audio_vae": "torch.float32",
            }
        )
    )
    assert len({bf16, vocoder, both}) == 3
