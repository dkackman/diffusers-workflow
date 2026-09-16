"""The name of an `asset:`/`prompt:`/`output:` reference, refused for free.

#162: a reference whose name no workspace could ever resolve failed the job
at run time, after the queue, rather than in the free pre-flight.
"""

import tempfile

from dw.reference_names import reference_fault, reference_name_errors
from dw.workflow import workflow_from_definition


def workflow_referencing(value):
    return {
        "id": "referencing",
        "steps": [
            {
                "name": "a",
                "task": {"command": "no_op", "arguments": {"video": value}},
            }
        ],
    }


class TestTheFault:
    def test_a_for_each_members_file_name_is_fine(self):
        assert (
            reference_fault("output:t/20260914-171601-adeee23c/i/X-shot@open.5-0.0.mp4")
            is None
        )

    def test_a_malformed_output_name_is_named(self):
        fault = reference_fault("output:t/ru n/x.mp4")

        assert "' '" in fault

    def test_a_malformed_asset_name_is_named(self):
        assert reference_fault("asset:qa cast/priya.jpg") is not None

    def test_a_well_formed_reference_is_silent(self):
        assert reference_fault("asset:qa-cast/priya.jpg") is None
        assert reference_fault("prompt:ltx2/hummingbird_garden") is None

    def test_a_deferred_reference_is_not_this_passs_complaint(self):
        """`output:variable:x` is a value substitution has not reached; the
        undeclared-variable pass owns that."""
        assert reference_fault("output:variable:x") is None

    def test_a_plain_string_is_not_a_reference(self):
        assert reference_fault("a normal prompt about a cat") is None


class TestTheValidationPass:
    def test_a_malformed_reference_is_refused_before_the_queue(self):
        workflow = workflow_from_definition(
            workflow_referencing("output:t/ru n/x.mp4"), tempfile.mkdtemp()
        )

        problems = workflow.validation_errors()

        assert any(
            problem["path"] == "steps[0].task.arguments.video" for problem in problems
        )

    def test_a_member_file_name_validates(self):
        workflow = workflow_from_definition(
            workflow_referencing(
                "output:t/20260914-171601-adeee23c/i/X-shot@open.5-0.0.mp4"
            ),
            tempfile.mkdtemp(),
        )

        assert workflow.validation_errors() == []

    def test_an_error_inside_a_member_names_the_member(self):
        definition = {
            "id": "listed",
            "variables": {"shots": [{"name": "one"}, {"name": "two"}]},
            "steps": [
                {
                    "name": "shot",
                    "for_each": "variable:shots",
                    "task": {
                        "command": "no_op",
                        "arguments": {"video": "output:t/ru n/x.mp4"},
                    },
                }
            ],
        }

        problems = workflow_from_definition(
            definition, tempfile.mkdtemp()
        ).validation_errors()

        assert problems
        # The path is the step the author wrote, not the expanded member
        assert all(
            problem["path"] == "steps[0].task.arguments.video" for problem in problems
        )
        assert any("shot@one" in problem["message"] for problem in problems)

    def test_nothing_is_reported_for_a_definition_with_no_references(self):
        assert reference_name_errors({"steps": [{"name": "a", "task": {}}]}) == []
