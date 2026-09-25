"""A pipeline step configuring a component its component_type does not
register.

`validate_workflow` used to answer `valid: true` for this - `configuration.
components` is only walked against the loaded pipeline instance, deep inside
`configure_components`, so a name the class never registers (LTX2InContext-
Pipeline has no `duration_head`) validated clean, queued, spent a couple of
minutes loading a checkpoint and LoRA weights the plan had already quoted
for downloading, and only then died on diffusers' own ValueError (#442).
This mirrors #345's component_type_errors: checked against the same
resolver the run itself uses, so the rule can never refuse a name that
would in fact have worked.
"""

from dw.introspection import component_name_errors, unknown_pipeline_components


def pipeline_step(component_type, components, name="a", reused=None):
    configuration = {"component_type": component_type, "components": components}
    if reused is not None:
        configuration["reused_components"] = reused
    return {
        "name": name,
        "pipeline": {"configuration": configuration},
        "result": {"content_type": "video/mp4"},
    }


def errors_for(component_type, components, reused=None):
    return component_name_errors(
        {
            "id": "cn",
            "steps": [pipeline_step(component_type, components, reused=reused)],
        }
    )


class TestUnknownPipelineComponents:
    def test_a_component_the_class_does_not_register_is_unknown(self):
        assert unknown_pipeline_components(
            "LTX2InContextPipeline", ["transformer", "duration_head"]
        ) == ["duration_head"]

    def test_every_real_component_is_known(self):
        assert (
            unknown_pipeline_components(
                "LTX2InContextPipeline",
                ["transformer", "vae", "audio_vae", "vocoder", "connectors"],
            )
            == []
        )

    def test_a_dotted_path_is_checked_by_its_first_segment(self):
        assert (
            unknown_pipeline_components("LTX2InContextPipeline", ["text_encoder.model"])
            == []
        )

    def test_an_unresolvable_class_reports_nothing(self):
        assert unknown_pipeline_components("NoSuchPipeline", ["anything"]) == []


class TestComponentNameErrors:
    def test_the_442_regression_is_refused(self):
        errors = errors_for(
            "LTX2InContextPipeline",
            {"transformer": {"device": "cuda"}, "duration_head": {"device": "cuda"}},
        )
        assert len(errors) == 1
        assert (
            errors[0]["path"]
            == "steps[0].pipeline.configuration.components.duration_head"
        )
        assert (
            "LTX2InContextPipeline has no component 'duration_head'"
            in (errors[0]["message"])
        )

    def test_a_real_pipeline_with_real_components_is_accepted(self):
        assert (
            errors_for(
                "LTX2InContextPipeline",
                {"transformer": {"device": "cuda"}, "vae": {"device": "cuda"}},
            )
            == []
        )

    def test_a_component_shared_from_an_earlier_step_is_not_flagged(self):
        """duration_head isn't real anywhere, but this documents that a name
        the step's own configuration.reused_components lists is skipped
        rather than checked - it is the sharing step's problem, not this
        one's, if it were ever wrong."""
        assert (
            errors_for(
                "LTX2InContextPipeline",
                {"vae": {"device": "cuda"}},
                reused=["vae"],
            )
            == []
        )

    def test_a_non_dict_components_value_is_left_to_schema_validation(self):
        definition = {
            "id": "cn",
            "steps": [
                {
                    "name": "a",
                    "pipeline": {
                        "configuration": {
                            "component_type": "LTX2InContextPipeline",
                            "components": "not-a-dict",
                        }
                    },
                    "result": {"content_type": "video/mp4"},
                }
            ],
        }
        assert component_name_errors(definition) == []

    def test_an_escaped_or_dotted_component_type_is_left_alone(self):
        assert errors_for("{LTX2InContextPipeline}", {"duration_head": {}}) == []
        assert (
            errors_for(
                "diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora.LTX2InContextPipeline",
                {"duration_head": {}},
            )
            == []
        )

    def test_a_step_with_no_pipeline_is_skipped(self):
        definition = {
            "id": "cn",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "concat_videos", "arguments": {}},
                    "result": {"content_type": "video/mp4"},
                }
            ],
        }
        assert component_name_errors(definition) == []
