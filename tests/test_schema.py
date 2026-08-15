import pytest
from dw.schema import validate_data, load_schema


def test_load_schema():
    # Test that we can load the workflow schema
    schema = load_schema("workflow")
    assert schema is not None
    assert "$schema" in schema
    assert "properties" in schema


def test_validate_data_valid(valid_workflow_json):
    # Test validation with valid workflow
    schema = load_schema("workflow")
    status, message = validate_data(valid_workflow_json, schema)
    assert status is True
    assert message == "Validation successful"


def test_validate_data_invalid(invalid_workflow_json):
    # Test validation with invalid workflow
    schema = load_schema("workflow")
    status, message = validate_data(invalid_workflow_json, schema)
    assert status is False
    assert "Validation error" in message


def test_validate_data_malformed_json():
    # Test validation with malformed JSON
    schema = load_schema("workflow")
    malformed_data = "{not valid json}"
    status, message = validate_data(malformed_data, schema)
    assert status is False
    assert "error" in message.lower()


def test_validate_required_fields():
    # Test validation of required fields
    schema = load_schema("workflow")
    incomplete_data = {
        "id": "test_workflow"
        # Missing required 'steps' field
    }
    status, message = validate_data(incomplete_data, schema)
    assert status is False
    assert "steps" in message  # Error should mention missing 'steps' field


def _pipeline_step(configuration, **step_extra):
    return {
        "id": "test_workflow",
        "steps": [
            {
                "name": "main",
                **step_extra,
                "pipeline": {
                    "configuration": {"component_type": "FluxPipeline"} | configuration,
                    "from_pretrained_arguments": {"model_name": "test"},
                    "arguments": {"prompt": "test"},
                },
            }
        ],
    }


def test_component_compile_configuration_validates():
    schema = load_schema("workflow")
    workflow = _pipeline_step(
        {
            "components": {
                "transformer": {
                    "compile": {
                        "repeated_blocks": True,
                        "fullgraph": True,
                        "mode": "max-autotune",
                        "dynamic": False,
                    },
                    "attention_backend": "flash_hub",
                }
            }
        }
    )
    status, message = validate_data(workflow, schema)
    assert status is True, message


def test_compile_options_are_typed():
    schema = load_schema("workflow")
    workflow = _pipeline_step(
        {"components": {"transformer": {"compile": {"fullgraph": "yes"}}}}
    )
    status, _ = validate_data(workflow, schema)
    assert status is False


def test_release_pipeline_step_flag_validates():
    schema = load_schema("workflow")
    workflow = _pipeline_step({}, release_pipeline=True)
    status, message = validate_data(workflow, schema)
    assert status is True, message
