"""The introspection contracts a UI and the submission warnings rely on."""

from dw.introspection import (
    describe_pipeline,
    unknown_call_arguments,
    workflow_argument_warnings,
    load_pipeline_class,
)

import pytest


def test_describe_merges_signature_and_docstring():
    description = describe_pipeline("ZImagePipeline")
    parameters = {p["name"]: p for p in description["parameters"]}
    assert parameters["num_inference_steps"]["default"] == 50
    assert parameters["num_inference_steps"]["required"] is False
    # docstring detail rides along when present
    assert "description" in parameters["prompt"]
    # self and *args/**kwargs never appear as parameters
    assert "self" not in parameters


def test_load_pipeline_class_rejects_non_bare_names():
    for bad in ("os.path", "../etc", "", "diffusers.ZImagePipeline", "no_such_thing"):
        with pytest.raises(ValueError):
            load_pipeline_class(bad)


def test_warnings_are_never_wrong():
    """The pre-load check only flags what is provably unaccepted."""
    # a class that cannot be resolved yields no warnings, not an error
    assert unknown_call_arguments("NoSuchPipeline", ["whatever"]) == []

    # escaped and dotted component types are left alone
    workflow = {
        "id": "w",
        "steps": [
            {
                "name": "escaped",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "arguments": {"bogus": 1},
                },
            },
            {
                "name": "dotted",
                "pipeline": {
                    "configuration": {"component_type": "some.module.Pipeline"},
                    "arguments": {"bogus": 1},
                },
            },
            {
                "name": "real",
                "pipeline": {
                    "configuration": {"component_type": "ZImagePipeline"},
                    "arguments": {"prompt": "p", "guidance_scael": 3},
                },
            },
        ],
    }
    warnings = workflow_argument_warnings(workflow)
    assert len(warnings) == 1
    assert "guidance_scael" in warnings[0] and "real" in warnings[0]


def test_describe_class_init_target_reads_constructors():
    from dw.introspection import describe_class

    description = describe_class("BitsAndBytesConfig", target="init")
    names = [p["name"] for p in description["parameters"]]
    assert "load_in_4bit" in names and "bnb_4bit_quant_type" in names


def test_describe_class_load_target_merges_curated_knobs():
    from dw.introspection import describe_class

    description = describe_class("AutoencoderKL", target="load")
    names = [p["name"] for p in description["parameters"]]
    for knob in ("torch_dtype", "variant", "subfolder", "revision"):
        assert knob in names
    # the model path is the editor's own field, never a discovered argument
    assert "pretrained_model_name_or_path" not in names
    # **kwargs loading means nothing is provably wrong - no warnings possible
    assert description["accepts_kwargs"] is True


def test_class_enumeration_by_kind():
    from dw.introspection import list_classes

    assert "AutoencoderKL" in list_classes("models")
    assert "FlowMatchEulerDiscreteScheduler" in list_classes("schedulers")
    assert "BitsAndBytesConfig" in list_classes("quantization")
    with pytest.raises(ValueError):
        list_classes("nonsense")


def test_allowlist_admits_sdnq_and_nothing_else():
    from dw.introspection import load_allowed_class

    pytest.importorskip("sdnq")
    assert load_allowed_class("sdnq.SDNQConfig").__name__ == "SDNQConfig"
    for blocked in ("os.path", "subprocess.Popen", "dw.security.validate_path"):
        with pytest.raises(ValueError):
            load_allowed_class(blocked)


def test_scheduler_compatibles_reported_when_present():
    from dw.introspection import describe_class

    description = describe_class("EulerDiscreteScheduler", target="init")
    assert "DDIMScheduler" in description.get("compatibles", [])
