import pytest
from dw.schema import (
    MAX_VALIDATION_ERRORS,
    SCHEMA_SECTIONS,
    SchemaSectionError,
    schema_section,
    format_validation_errors,
    load_schema,
    validate_data,
    validate_data_all,
)


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


class TestEveryError:
    """An agent iterating on a draft should get every schema violation in one
    round trip, each with the JSON path it sits at."""

    def _three_violations(self):
        workflow = _pipeline_step({}, seed="not-a-number")
        workflow["steps"][0]["release_models"] = "yes"
        workflow["variables"] = "not-an-object"
        return workflow

    def test_independent_violations_are_all_reported_with_paths(self):
        errors = validate_data_all(self._three_violations(), load_schema("workflow"))

        paths = [e["path"] for e in errors]
        assert "steps[0].seed" in paths
        assert "steps[0].release_models" in paths
        assert "variables" in paths
        assert all(e["message"] for e in errors)

    def test_errors_are_sorted_by_path(self):
        errors = validate_data_all(self._three_violations(), load_schema("workflow"))

        assert [e["path"] for e in errors] == sorted(e["path"] for e in errors)

    def test_a_single_error_is_reported_exactly_as_validate_data_does(self):
        # The one-error case is the common one and must not change shape
        # or wording: the CLI, the REPL and the editor all show it
        workflow = _pipeline_step({}, seed="not-a-number")
        schema = load_schema("workflow")

        errors = validate_data_all(workflow, schema)
        _status, message = validate_data(workflow, schema)

        assert len(errors) == 1
        assert (
            message
            == f"Validation error at {errors[0]['path']}: {errors[0]['message']}"
        )

    def test_a_valid_definition_yields_no_errors(self):
        assert validate_data_all(_pipeline_step({}), load_schema("workflow")) == []

    def test_the_list_is_capped(self):
        # anyOf branches produce dozens of near-identical entries; 25 is
        # more than an agent fixes in one pass
        workflow = _pipeline_step({})
        workflow["steps"] = [
            {"name": f"s{i}", "seed": "x", "pipeline": "nope"} for i in range(40)
        ]

        errors = validate_data_all(workflow, load_schema("workflow"))

        assert len(errors) == MAX_VALIDATION_ERRORS

    def test_duplicates_on_path_and_message_collapse(self):
        errors = validate_data_all(self._three_violations(), load_schema("workflow"))

        assert len(errors) == len({(e["path"], e["message"]) for e in errors})

    def test_a_root_error_has_no_path(self):
        errors = validate_data_all({"id": "x"}, load_schema("workflow"))

        assert errors[0]["path"] is None
        assert "steps" in errors[0]["message"]


class TestFormatting:
    def test_one_error_is_the_familiar_line(self):
        text = format_validation_errors(
            [{"path": "steps[0].seed", "message": "'x' is not of type 'integer'"}]
        )

        assert text == "Validation error at steps[0].seed: 'x' is not of type 'integer'"

    def test_one_root_error_has_no_location(self):
        text = format_validation_errors(
            [{"path": None, "message": "'steps' is a required property"}]
        )

        assert text == "Validation error: 'steps' is a required property"

    def test_several_errors_are_one_per_line_under_one_heading(self):
        text = format_validation_errors(
            [
                {"path": None, "message": "'steps' is a required property"},
                {"path": "variables", "message": "'x' is not of type 'object'"},
            ]
        )

        assert text == (
            "Validation errors (2):\n"
            "  at root: 'steps' is a required property\n"
            "  at variables: 'x' is not of type 'object'"
        )
        # The CLI and the REPL count this prefix once per failure
        assert text.count("Validation error") == 1

    def test_a_capped_list_says_so(self):
        errors = [
            {"path": f"steps[{i}]", "message": "bad"}
            for i in range(MAX_VALIDATION_ERRORS)
        ]

        text = format_validation_errors(errors)

        assert text.startswith(f"Validation errors (first {MAX_VALIDATION_ERRORS}):\n")


def test_for_each_is_a_list_or_a_variable_reference():
    schema = load_schema("workflow")
    base = {
        "id": "t",
        "steps": [
            {
                "name": "shot",
                "task": {"command": "x", "arguments": {}},
                "for_each": None,
            }
        ],
    }
    for good in (["a", "b"], [{"name": "a"}], "variable:shots"):
        base["steps"][0]["for_each"] = good
        assert validate_data_all(base, schema) == []
    for bad in (4, "shots", {"a": 1}):
        base["steps"][0]["for_each"] = bad
        assert validate_data_all(base, schema) != []


def test_an_empty_for_each_list_fails_schema_validation():
    schema = load_schema("workflow")
    base = {
        "id": "t",
        "steps": [
            {
                "name": "shot",
                "task": {"command": "x", "arguments": {}},
                "for_each": [],
            }
        ],
    }
    errors = validate_data_all(base, schema)
    assert errors != []
    assert any("for_each" in e["path"] for e in errors)


class TestResultSubfolder:
    def test_subfolder_is_a_described_string_with_no_pattern(self):
        result = load_schema("workflow")["$defs"]["result"]["properties"]
        assert result["subfolder"]["type"] == "string"
        assert "final" in result["subfolder"]["description"]
        # Schema validation runs before substitution, so a pattern would
        # reject 'variable:dest' and 'item:subfolder'
        assert "pattern" not in result["subfolder"]

    def test_a_workflow_with_a_subfolder_validates(self):
        definition = {
            "id": "s",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {
                        "content_type": "image/png",
                        "subfolder": "variable:dest",
                    },
                }
            ],
        }
        assert validate_data_all(definition, load_schema("workflow")) == []


class TestSections:
    """One part of the schema at a time - the whole thing is ~8.6k tokens
    in a single call, spent by an agent that wanted the shape of a result
    block (#101)."""

    @pytest.fixture
    def schema(self):
        return load_schema("workflow")

    def test_every_definition_belongs_to_a_section(self, schema):
        """An orphan definition would be unreachable by section - findable
        only in the whole schema, which is the call sections exist to
        avoid."""
        owned = {name for part in SCHEMA_SECTIONS.values() for name in part["defs"]}
        assert set(schema["$defs"]) - owned == set()

    def test_a_section_is_a_fraction_of_the_whole(self, schema):
        import json

        whole = len(json.dumps(schema))
        answer = schema_section(schema, "result")
        assert len(json.dumps(answer["schema"])) < whole / 5

    def test_a_section_carries_its_own_definitions(self, schema):
        answer = schema_section(schema, "tasks")
        assert list(answer["schema"]["$defs"]) == ["task"]

    def test_a_reference_it_does_not_hold_says_where_it_lives(self, schema):
        answer = schema_section(schema, "steps")
        assert answer["elsewhere"]["result"] == "result"
        assert answer["elsewhere"]["task"] == "tasks"

    def test_variables_carries_the_top_level_properties(self, schema):
        answer = schema_section(schema, "variables")
        assert "variables" in answer["schema"]["properties"]
        assert "seed" in answer["schema"]["properties"]
        assert "steps" not in answer["schema"]["properties"]

    def test_every_section_answers(self, schema):
        for name in SCHEMA_SECTIONS:
            assert schema_section(schema, name)["section"] == name

    def test_an_unknown_section_names_the_ones_there_are(self, schema):
        with pytest.raises(SchemaSectionError, match="steps"):
            schema_section(schema, "pipline")


class TestClosedObjects:
    """#118: a step is the one object an agent could invent control flow on
    and be told it validates. `when`, `retry`, a typo'd `relase_pipeline` -
    the engine reads none of them, so the expensive work ran with the input
    silently having had no effect. Closed, with a message that says what the
    object does take, because "Additional properties are not allowed ('when'
    was unexpected)" does not.
    """

    def workflow(self, **step_keys):
        return {
            "id": "probe",
            "steps": [
                {
                    "name": "video",
                    "task": {"command": "gather_inputs", "arguments": {}},
                    **step_keys,
                }
            ],
        }

    @pytest.mark.parametrize(
        "stray",
        [
            {"when": "previous_result:judge.pass"},
            {"retry": 3},
            {"select": "argmax"},
            {"relase_pipeline": True},
        ],
    )
    def test_an_unknown_step_property_is_an_error(self, stray):
        schema = load_schema("workflow")
        errors = validate_data_all(self.workflow(**stray), schema)
        assert [error["path"] for error in errors] == ["steps[0]"]
        message = errors[0]["message"]
        assert list(stray)[0] in message
        # names the step and what a step actually takes
        assert '"video"' in message
        assert "release_pipeline" in message and "for_each" in message

    def test_several_stray_keys_are_reported_together(self):
        schema = load_schema("workflow")
        errors = validate_data_all(self.workflow(when="x", retry=3), schema)
        assert len(errors) == 1
        assert "unknown properties" in errors[0]["message"]
        assert '"retry"' in errors[0]["message"]
        assert '"when"' in errors[0]["message"]

    def test_a_well_formed_step_still_validates(self):
        schema = load_schema("workflow")
        assert validate_data_all(self.workflow(), schema) == []

    @pytest.mark.parametrize(
        "definition",
        [
            {"task": {"command": "gather_inputs", "arguments": {}, "inupts": {}}},
            {"workflow": {"path": "builtin:x.json", "argumnets": {}}},
            {"pipeline_reference": {"reference_name": "draw", "chian": []}},
        ],
    )
    def test_the_other_swept_objects_are_closed_too(self, definition):
        """task, workflow_reference and pipeline_reference carried no stray
        key anywhere in the catalog, so closing them breaks nothing. The
        pipeline object is deliberately left open - component names are its
        keys (latent_upsampler, prompt_enhancer, processor all appear in
        shipped templates), so it cannot be swept the same way."""
        schema = load_schema("workflow")
        workflow = {"id": "probe", "steps": [{"name": "s", **definition}]}
        errors = validate_data_all(workflow, schema)
        assert errors, f"{definition} should not have validated"

    def test_the_pipeline_object_is_open_only_to_a_component(self):
        """#123: `pipeline` cannot be closed outright - a component's name is
        one of its keys, and `latent_upsampler`, `prompt_enhancer` and
        `processor` all appear that way in shipped templates. So it takes the
        rule `declared_component_names` applies: any other key whose value is
        a component definition, and nothing else."""
        schema = load_schema("workflow")
        extra = schema["$defs"]["pipeline"]["additionalProperties"]
        assert extra["required"] == ["from_pretrained_arguments"]
        assert extra["type"] == "object"
        # pipeline_component itself stays open - its own keys are the
        # from_pretrained kwargs a component takes
        assert (
            schema["$defs"]["pipeline_component"].get("additionalProperties")
            is not False
        )

    @pytest.mark.parametrize(
        "stray",
        [
            {"pipeline_type": "diffusers.Whatever"},
            {"model_name": "org/typo"},
            {"trasformer": {}},
        ],
    )
    def test_a_key_that_is_not_a_component_is_an_error(self, stray):
        """#123: `pipeline_type` and `model_name` are real names from the
        wrong level and `trasformer` is a typo; all three used to validate
        and be ignored, and the `model_name` one ran against whatever
        `from_pretrained_arguments` said."""
        schema = load_schema("workflow")
        workflow = {
            "id": "probe",
            "steps": [
                {
                    "name": "draw",
                    "pipeline": {
                        "configuration": {"component_type": "StableDiffusionPipeline"},
                        "from_pretrained_arguments": {"model_name": "org/model"},
                        "arguments": {"prompt": "x"},
                        **stray,
                    },
                    "result": {"content_type": "image/jpeg"},
                }
            ],
        }

        errors = validate_data_all(workflow, schema)

        assert [error["path"] for error in errors] == [
            f"steps[0].pipeline.{list(stray)[0]}"
        ]
        # one message per stray key, and it is the one that says what is
        # wrong rather than a complaint about the component it is not
        assert errors[0]["message"].startswith(f'unknown property "{list(stray)[0]}"')
        assert "from_pretrained_arguments" in errors[0]["message"]

    def test_a_component_named_by_a_key_still_validates(self):
        schema = load_schema("workflow")
        workflow = {
            "id": "probe",
            "steps": [
                {
                    "name": "draw",
                    "pipeline": {
                        "configuration": {"component_type": "LTXPipeline"},
                        "from_pretrained_arguments": {"model_name": "org/model"},
                        "arguments": {"prompt": "x"},
                        "latent_upsampler": {
                            "configuration": {"component_type": "LatentUpsampler"},
                            "from_pretrained_arguments": {"model_name": "org/up"},
                        },
                    },
                    "result": {"content_type": "video/mp4"},
                }
            ],
        }

        assert validate_data_all(workflow, schema) == []

    @pytest.mark.parametrize(
        "stray",
        [{"varaibles": {"x": 1}}, {"sedd": 42}, {"when": "always"}],
    )
    def test_an_unknown_top_level_property_is_an_error(self, stray):
        """#123: `sedd` was the sharp one - validation answered that the
        workflow set no seed and advised setting one, with the misspelling
        of it in front of it."""
        schema = load_schema("workflow")
        workflow = {
            "id": "probe",
            "steps": [
                {"name": "s", "task": {"command": "gather_inputs", "arguments": {}}}
            ],
            **stray,
        }

        errors = validate_data_all(workflow, schema)

        assert [error["path"] for error in errors] == [None]
        assert list(stray)[0] in errors[0]["message"]
        assert "steps" in errors[0]["message"]

    @pytest.mark.parametrize(
        "stray", [{"subfoldr": "final"}, {"fille_base_name": "x"}, {"fsp": 24}]
    )
    def test_an_unknown_result_property_is_an_error(self, stray):
        """#123: a mistyped `subfoldr` means the deliverable quietly lands at
        the run root rather than in the `final/` the convention promises."""
        schema = load_schema("workflow")
        workflow = {
            "id": "probe",
            "steps": [
                {
                    "name": "s",
                    "task": {"command": "gather_inputs", "arguments": {}},
                    "result": {"content_type": "text/plain", **stray},
                }
            ],
        }

        errors = validate_data_all(workflow, schema)

        assert [error["path"] for error in errors] == ["steps[0].result"]
        assert list(stray)[0] in errors[0]["message"]
        assert "subfolder" in errors[0]["message"]

    def test_the_engine_injected_key_is_not_advertised(self):
        """'argument_template' is written onto a sub-workflow by the engine,
        so it is legal - but listing it among the properties on offer would
        invite an author to write it by hand (#123)."""
        schema = load_schema("workflow")
        errors = validate_data_all(
            {
                "id": "probe",
                "sedd": 1,
                "steps": [{"name": "s", "task": {"command": "x", "arguments": {}}}],
            },
            schema,
        )
        assert "argument_template" not in errors[0]["message"]
        assert "seed" in errors[0]["message"]

    def test_the_later_swept_objects_are_closed_in_the_schema(self):
        """#123: the three the #118 sweep left open."""
        schema = load_schema("workflow")
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["result"]["additionalProperties"] is False

    def test_the_swept_objects_are_closed_in_the_schema(self):
        schema = load_schema("workflow")
        for name in ("step", "task", "workflow_reference", "pipeline_reference"):
            assert schema["$defs"][name]["additionalProperties"] is False
