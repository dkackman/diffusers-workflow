import pytest

from dw.security import InvalidInputError
from dw.content_types import content_type_errors, content_type_fault


def _step(name, result=None, **extra):
    step = {"name": name, "task": {"command": "noop", "arguments": {}}}
    if result is not None:
        step["result"] = result
    step.update(extra)
    return step


class TestContentTypeFault:
    @pytest.mark.parametrize(
        "value",
        [
            "audio/mp3",
            "audio/wav",
            "image/jpeg",
            "image/png",
            "text/plain",
            "video/mp4",
        ],
    )
    def test_real_catalog_values_are_clean(self, value):
        assert content_type_fault(value) is None

    def test_a_bare_word_is_faulted(self):
        fault = content_type_fault("video")
        assert fault is not None
        assert "video" in fault
        assert "MIME" in fault

    def test_a_non_string_is_faulted(self):
        assert content_type_fault(3) is not None

    def test_an_unsupported_audio_subtype_is_faulted(self):
        fault = content_type_fault("audio/mp4")
        assert fault is not None
        assert "audio/mp3" in fault or "audio/wav" in fault

    def test_an_unsupported_video_container_is_faulted(self):
        fault = content_type_fault("video/webm")
        assert fault is not None
        assert "video/mp4" in fault

    def test_image_and_text_beyond_the_catalog_are_permissive(self):
        assert content_type_fault("image/webp") is None
        assert content_type_fault("text/markdown") is None
        assert content_type_fault("application/json") is None


class TestContentTypeErrors:
    def test_a_clean_definition_has_none(self):
        definition = {
            "steps": [
                _step("a", {"content_type": "image/png"}),
                _step("b", {"content_type": "video/mp4"}),
            ]
        }
        assert content_type_errors(definition) == []

    def test_the_reported_bug_is_caught_at_its_path(self):
        definition = {"steps": [_step("a", {"content_type": "video"})]}
        errors = content_type_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.content_type"
        assert "video" in errors[0]["message"]

    def test_an_expanded_member_reports_the_source_step_and_names_the_member(self):
        definition = {
            "steps": [
                _step("intro", {"content_type": "image/png"}),
                _step("shot@a", {"content_type": "image/png"}),
                _step("shot@b", {"content_type": "video"}),
            ]
        }
        errors = content_type_errors(definition, source_indices=[0, 1, 1])
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[1].result.content_type"
        assert "shot@b" in errors[0]["message"]

    def test_an_unsubstituted_reference_is_left_alone(self):
        definition = {"steps": [_step("a", {"content_type": "variable:kind"})]}
        assert content_type_errors(definition) == []

    def test_no_steps_is_fine(self):
        assert content_type_errors({}) == []
        assert content_type_errors({"steps": "nope"}) == []

    def test_a_step_without_content_type_is_left_alone(self):
        definition = {"steps": [_step("a", {"subfolder": "final"})]}
        assert content_type_errors(definition) == []


class TestValidationErrorsIntegration:
    def _workflow(self, definition, tmp_path):
        from dw.workflow import Workflow

        return Workflow(definition, str(tmp_path), "/w/workflows/CT.json")

    def test_validation_errors_reports_the_bug_case(self, tmp_path):
        definition = {
            "id": "ct_test",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "video"},
                }
            ],
        }
        errors = self._workflow(definition, tmp_path).validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].result.content_type"]

    def test_a_variable_driven_content_type_is_checked_by_its_value(self, tmp_path):
        definition = {
            "id": "ct_test",
            "variables": {"kind": "image/png"},
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "variable:kind"},
                }
            ],
        }
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors() == []
        errors = workflow.validation_errors(arguments={"kind": "video"})
        assert [e["path"] for e in errors] == ["steps[0].result.content_type"]
