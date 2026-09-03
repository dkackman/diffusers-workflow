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


def test_residency_priority_configuration_validates():
    schema = load_schema("workflow")
    workflow = _pipeline_step(
        {
            "components": {
                "text_encoder": {
                    "residency": "on_demand",
                    "residency_priority": 5,
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


def test_null_variable_declares_an_optional_argument():
    """A workflow exposing an argument a caller may omit declares it null."""
    schema = load_schema("workflow")
    workflow = _pipeline_step({})
    workflow["variables"] = {"image": None, "prompt": "a cat"}
    status, message = validate_data(workflow, schema)
    assert status is True, message


def test_release_models_step_flag_validates():
    schema = load_schema("workflow")
    workflow = _pipeline_step({}, release_models=True)
    status, message = validate_data(workflow, schema)
    assert status is True, message


def test_release_models_step_flag_is_typed():
    schema = load_schema("workflow")
    workflow = _pipeline_step({}, release_models="yes")
    status, _ = validate_data(workflow, schema)
    assert status is False


def _chained_workflow(chain):
    workflow = _pipeline_step({})
    workflow["steps"][0]["pipeline"]["chain"] = chain
    return workflow


class TestSchedulerSchema:
    def _with_scheduler(self, **blocks):
        workflow = _pipeline_step({})
        workflow["steps"][0]["pipeline"].update(blocks)
        return workflow

    def test_a_shift_only_scheduler_validates(self):
        schema = load_schema("workflow")
        status, message = validate_data(
            self._with_scheduler(scheduler={"shift": 6}), schema
        )
        assert status is True, message

    def test_both_schedulers_validate(self):
        schema = load_schema("workflow")
        status, message = validate_data(
            self._with_scheduler(scheduler={"shift": 6}, audio_scheduler={"shift": 3}),
            schema,
        )
        assert status is True, message

    def test_a_variable_reference_shift_validates(self):
        # Schema validation runs before variable substitution
        schema = load_schema("workflow")
        status, message = validate_data(
            self._with_scheduler(scheduler={"shift": "variable:flow_shift"}), schema
        )
        assert status is True, message

    def test_a_scheduler_type_replacement_still_validates(self):
        schema = load_schema("workflow")
        status, message = validate_data(
            self._with_scheduler(
                scheduler={"configuration": {"scheduler_type": "DDIMScheduler"}}
            ),
            schema,
        )
        assert status is True, message

    def test_an_empty_scheduler_block_is_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(self._with_scheduler(scheduler={}), schema)
        assert status is False

    def test_a_negative_shift_is_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(self._with_scheduler(scheduler={"shift": -1}), schema)
        assert status is False


class TestChainSchema:
    def test_a_segment_count_chain_validates(self):
        schema = load_schema("workflow")
        status, message = validate_data(
            _chained_workflow({"segments": 4, "trim_frames": 2}), schema
        )
        assert status is True, message

    def test_a_match_audio_chain_with_frame_snap_validates(self):
        schema = load_schema("workflow")
        chain = {
            "match_audio": True,
            "segment_argument": "references",
            "frame_snap": {
                "modulus": 17,
                "remainder": 5,
                "min_frames": 124,
                "max_frames": 345,
            },
        }
        status, message = validate_data(_chained_workflow(chain), schema)
        assert status is True, message

    def test_a_last_segment_chain_validates(self):
        schema = load_schema("workflow")
        chain = {
            "segments": 3,
            "continuity": "last_segment",
            "segment_argument": "references",
            "carry_frames": 48,
            "carry_audio": True,
        }
        status, message = validate_data(_chained_workflow(chain), schema)
        assert status is True, message

    def test_a_zero_carry_frames_is_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(
            _chained_workflow({"segments": 2, "carry_frames": 0}), schema
        )
        assert status is False

    def test_segments_and_match_audio_together_are_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(
            _chained_workflow({"segments": 4, "match_audio": True}), schema
        )
        assert status is False

    def test_a_chain_without_a_length_is_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(
            _chained_workflow({"continuity": "last_frame"}), schema
        )
        assert status is False

    def test_an_unknown_continuity_mode_is_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(
            _chained_workflow({"segments": 2, "continuity": "teleport"}), schema
        )
        assert status is False

    def test_a_variable_reference_segment_count_validates(self):
        # Schema validation runs before variable substitution
        schema = load_schema("workflow")
        status, message = validate_data(
            _chained_workflow({"segments": "variable:segments"}), schema
        )
        assert status is True, message

    def test_an_unknown_chain_property_is_rejected(self):
        schema = load_schema("workflow")
        status, _ = validate_data(
            _chained_workflow({"segments": 2, "overlap": 3}), schema
        )
        assert status is False

    def test_per_segment_prompts_validate(self):
        schema = load_schema("workflow")
        status, message = validate_data(
            _chained_workflow({"segments": 2, "prompts": ["first", "second"]}), schema
        )
        assert status is True, message

    def test_a_pipeline_reference_accepts_a_chain(self):
        schema = load_schema("workflow")
        workflow = _pipeline_step({})
        workflow["steps"].append(
            {
                "name": "chained",
                "pipeline_reference": {
                    "reference_name": "main",
                    "chain": {"segments": 3},
                    "arguments": {"prompt": "test"},
                },
            }
        )
        status, message = validate_data(workflow, schema)
        assert status is True, message
