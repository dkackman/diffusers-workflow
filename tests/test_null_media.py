"""A for_each item's null media reference, caught before the checkpoint loads.

`realize_args` raises when an object description sitting directly under a key
(not inside a list) names a type but resolves no media to build it from -
but only at run time, after the checkpoint is already loaded. validate_workflow
is supposed to catch what realize_args would refuse; this mirrors that dict-
context raise so a caller finds out for free (#478). List-context entries are
left alone: a template relies on realize_args silently dropping a null one
there to make a reference optional (`dialogue-short.json`'s shots), so that
shape must never be flagged.
"""

import json

import pytest

from dw.step_value_checks import null_media_errors
from dw.workflow import Workflow

H3 = "diffusers.modular_pipelines.minimax_h3"
IMAGE = f"{H3}.MiniMaxH3ImageReference"


def reference(reference_type=IMAGE, from_file="x.jpg"):
    return {"reference_type": reference_type, "from_file": from_file}


def workflow_with(arguments):
    return {
        "id": "refs",
        "steps": [
            {
                "name": "shot",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "ref2va",
                    },
                    "arguments": arguments,
                },
                "result": {"content_type": "video/mp4"},
            }
        ],
    }


def messages(arguments):
    return [e["message"] for e in null_media_errors(workflow_with(arguments))]


class TestDictContextIsRefused:
    def test_a_bare_null_reference_is_an_error(self):
        found = messages({"ref_a": reference(), "ref_b": reference(from_file=None)})
        assert len(found) == 1
        assert "'ref_b' names an object to build" in found[0]
        assert "media it would be built from is null" in found[0]

    def test_the_path_points_at_the_null_from_key(self):
        errors = null_media_errors(workflow_with({"ref_b": reference(from_file=None)}))
        assert len(errors) == 1
        assert errors[0]["path"].endswith("pipeline.arguments.ref_b.from_file")

    def test_a_fully_populated_reference_is_fine(self):
        assert messages({"ref_a": reference()}) == []


class TestListContextIsLeftAlone:
    """This is the shape realize_args silently drops - the documented way a
    template makes a reference optional. Flagging it would be a regression."""

    def test_a_null_entry_in_a_list_is_not_an_error(self):
        assert messages({"references": [reference(), reference(from_file=None)]}) == []

    def test_a_list_of_only_null_entries_is_not_an_error(self):
        assert messages({"references": [reference(from_file=None)]}) == []


class TestUnresolvedIsLeftToAnotherPass:
    def test_a_still_variable_reference_is_not_flagged(self):
        assert messages({"ref_a": {"reference_type": "variable:kind"}}) == []
        assert messages({"ref_a": "variable:references"}) == []


class TestThroughTheTemplate:
    """The issue's own shape: a for_each member's item: reference resolves to
    a bare (non-list) null-media dict once expanded."""

    def _workflow(self, arguments):
        return Workflow(workflow_with(arguments), "outputs", "refs.json")

    def test_validation_errors_reports_the_dict_context_null(self):
        errors = self._workflow(
            {"ref_a": reference(), "ref_b": reference(from_file=None)}
        ).validation_errors()
        messages_found = [e["message"] for e in errors]
        assert any("'ref_b' names an object to build" in m for m in messages_found)

    def test_dialogue_short_s_optional_list_references_still_validate(self):
        path = "workflows/templates/minimax/dialogue-short.json"
        with open(path) as handle:
            workflow = Workflow(json.load(handle), "outputs", path)
        assert workflow.validation_errors() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
