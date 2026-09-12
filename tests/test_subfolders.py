import pytest

from dw.security import InvalidInputError
from dw.subfolders import step_subfolder, subfolder_errors


def _step(name, result=None, **extra):
    step = {"name": name, "task": {"command": "noop", "arguments": {}}}
    if result is not None:
        step["result"] = result
    step.update(extra)
    return step


class TestStepSubfolder:
    def test_absent_is_the_empty_string(self):
        assert step_subfolder(_step("a")) == ""
        assert step_subfolder(_step("a", {"content_type": "image/png"})) == ""

    def test_present_is_returned_validated(self):
        assert step_subfolder(_step("a", {"subfolder": "final"})) == "final"

    def test_a_bad_one_raises(self):
        with pytest.raises(InvalidInputError):
            step_subfolder(_step("a", {"subfolder": "../x"}))

    def test_a_non_string_raises(self):
        with pytest.raises(InvalidInputError):
            step_subfolder(_step("a", {"subfolder": 3}))


class TestSubfolderErrors:
    def test_a_clean_definition_has_none(self):
        definition = {
            "steps": [
                _step("a", {"content_type": "image/png", "subfolder": "final"}),
                _step("b", {"content_type": "image/png"}),
            ]
        }
        assert subfolder_errors(definition) == []

    def test_a_bad_subfolder_is_reported_at_its_path(self):
        definition = {"steps": [_step("a", {"subfolder": "../escape"})]}
        errors = subfolder_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.subfolder"
        assert "Subfolder" in errors[0]["message"] or "subfolder" in errors[0]["message"]

    def test_a_separator_in_file_base_name_is_reported_at_its_path(self):
        definition = {"steps": [_step("a", {"file_base_name": "final/"})]}
        errors = subfolder_errors(definition)
        assert [e["path"] for e in errors] == ["steps[0].result.file_base_name"]
        assert "subfolder" in errors[0]["message"]

    def test_an_expanded_member_reports_the_source_step_and_names_the_member(self):
        # expand_for_each turned steps[1] into two members; the author wrote
        # one step, so the path is the source's and the member is named
        definition = {
            "steps": [
                _step("intro", {"subfolder": "final"}),
                _step("shot@a", {"subfolder": "ok"}),
                _step("shot@b", {"subfolder": "bad/../x"}),
            ]
        }
        errors = subfolder_errors(definition, source_indices=[0, 1, 1])
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[1].result.subfolder"
        assert "shot@b" in errors[0]["message"]

    def test_an_unsubstituted_reference_is_left_alone(self):
        # A 'variable:' still spelled out is one nothing resolved; that is
        # the undeclared-variable pass's complaint, not this one's
        definition = {"steps": [_step("a", {"subfolder": "variable:dest"})]}
        assert subfolder_errors(definition) == []

    def test_no_steps_is_fine(self):
        assert subfolder_errors({}) == []
        assert subfolder_errors({"steps": "nope"}) == []


class TestValidationErrorsIntegration:
    def _workflow(self, definition, tmp_path):
        from dw.workflow import Workflow

        return Workflow(definition, str(tmp_path), "/w/workflows/Sub.json")

    def test_validation_errors_reports_a_bad_subfolder(self, tmp_path):
        definition = {
            "id": "sub_test",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "../x"},
                }
            ],
        }
        errors = self._workflow(definition, tmp_path).validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].result.subfolder"]

    def test_a_variable_driven_subfolder_is_checked_by_its_value(self, tmp_path):
        definition = {
            "id": "sub_test",
            "variables": {"dest": "final"},
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "variable:dest"},
                }
            ],
        }
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors() == []
        errors = workflow.validation_errors(arguments={"dest": "../x"})
        assert [e["path"] for e in errors] == ["steps[0].result.subfolder"]

    def test_an_item_driven_subfolder_is_checked_per_member(self, tmp_path):
        definition = {
            "id": "sub_test",
            "steps": [
                {
                    "name": "shot",
                    "for_each": [
                        {"name": "a", "dest": "shots/a"},
                        {"name": "b", "dest": "../x"},
                    ],
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "item:dest"},
                }
            ],
        }
        errors = self._workflow(definition, tmp_path).validation_errors()
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.subfolder"
        assert "shot@b" in errors[0]["message"]
