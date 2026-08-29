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
