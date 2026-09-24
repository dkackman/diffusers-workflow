"""Tests for #212: `validate_workflow` refusing a `result` block on a step
whose command returns a scalar rather than an artifact.

`judge` returns a bare float; a `result` block on such a step validated
clean and then died at run time inside `save_artifact` with
`write() argument must be str, not float` after the fan-out ahead of it had
already generated (#212). These are the free pre-flight versions.

The same check covers the `returns="json"` assessment probes (#387): a
"result" on one of those may only be `application/json`, since any other
content type would explode the answer dict key by key into files or die on
a number.
"""

import unittest

from dw.scalar_result_validation import scalar_result_errors
from dw.workflow import Workflow


def _step(name, command, result=None, arguments=None):
    step = {"name": name, "task": {"command": command, "arguments": arguments or {}}}
    if result is not None:
        step["result"] = result
    return step


class TestScalarResultErrors(unittest.TestCase):
    def test_result_on_judge_is_an_error(self):
        definition = {
            "steps": [
                _step(
                    "score",
                    "judge",
                    result={"content_type": "text/plain", "subfolder": "intermediate"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["path"], "steps[0].result")
        self.assertIn("judge", errors[0]["message"])
        self.assertIn("not an artifact", errors[0]["message"])

    def test_no_result_on_judge_is_fine(self):
        definition = {"steps": [_step("score", "judge")]}

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_result_on_artifact_returning_command_is_fine(self):
        definition = {
            "steps": [
                _step(
                    "shot",
                    "gather_images",
                    result={"content_type": "image/png"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_unknown_command_is_ignored(self):
        definition = {
            "steps": [
                _step(
                    "mystery",
                    "not_a_real_command",
                    result={"content_type": "image/png"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_member_step_names_the_member_in_the_message(self):
        definition = {
            "steps": [
                _step(
                    "score@shot-1",
                    "judge",
                    result={"content_type": "text/plain"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("member 'score@shot-1'", errors[0]["message"])

    def test_non_dict_step_is_skipped(self):
        definition = {"steps": ["not-a-step"]}

        errors = scalar_result_errors(definition)

        self.assertEqual(errors, [])


class TestJsonReturningCommands(unittest.TestCase):
    """A `returns="json"` command (the assessment probes, #387) may only be
    saved as `application/json` - any other content type would explode the
    answer dict key by key into files, or die on a number."""

    def test_application_json_is_fine(self):
        definition = {
            "steps": [
                _step(
                    "seams",
                    "analyze_seams",
                    result={"content_type": "application/json"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_text_plain_is_an_error_naming_application_json(self):
        definition = {
            "steps": [
                _step(
                    "seams",
                    "analyze_seams",
                    result={"content_type": "text/plain"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["path"], "steps[0].result")
        self.assertIn("application/json", errors[0]["message"])
        self.assertIn("analyze_seams", errors[0]["message"])

    def test_image_png_is_an_error_naming_application_json(self):
        definition = {
            "steps": [
                _step(
                    "seams",
                    "analyze_seams",
                    result={"content_type": "image/png"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["path"], "steps[0].result")
        self.assertIn("application/json", errors[0]["message"])

    def test_scalar_kind_behaviour_is_unchanged(self):
        # judge still refuses a result block outright, regardless of
        # content_type - the json check is additional, not a replacement
        definition = {
            "steps": [
                _step(
                    "score",
                    "judge",
                    result={"content_type": "application/json"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("not an artifact", errors[0]["message"])

    def test_artifact_kind_behaviour_is_unchanged(self):
        definition = {
            "steps": [
                _step(
                    "shot",
                    "gather_images",
                    result={"content_type": "application/json"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])


class TestThroughValidationErrors(unittest.TestCase):
    """The same check, exercised through the public `validation_errors` on a
    full workflow definition rather than the pre-substituted form directly."""

    def _step(self, content_type):
        return {
            "name": "seams",
            "task": {
                "command": "analyze_seams",
                "arguments": {"video": "asset:cut.mp4"},
            },
            "result": {"content_type": content_type},
        }

    def _workflow(self, content_type):
        return Workflow(
            {"id": "assess", "steps": [self._step(content_type)]},
            "outputs",
            None,
        )

    def test_application_json_validates_clean(self):
        self.assertEqual(self._workflow("application/json").validation_errors(), [])

    def test_text_plain_is_refused(self):
        errors = self._workflow("text/plain").validation_errors()

        messages = [e["message"] for e in errors if e["path"] == "steps[0].result"]
        self.assertTrue(messages)
        self.assertTrue(any("application/json" in m for m in messages))

    def test_image_png_is_refused(self):
        errors = self._workflow("image/png").validation_errors()

        messages = [e["message"] for e in errors if e["path"] == "steps[0].result"]
        self.assertTrue(messages)
        self.assertTrue(any("application/json" in m for m in messages))

    def test_scalar_command_through_validation_errors_is_unchanged(self):
        workflow = Workflow(
            {
                "id": "assess",
                "steps": [
                    {
                        "name": "score",
                        "task": {
                            "command": "judge",
                            "arguments": {
                                "image": "asset:cut.png",
                                "prompt": "a cat",
                            },
                        },
                        "result": {"content_type": "text/plain"},
                    }
                ],
            },
            "outputs",
            None,
        )

        errors = workflow.validation_errors()

        messages = [e["message"] for e in errors if e["path"] == "steps[0].result"]
        self.assertTrue(messages)
        self.assertTrue(any("not an artifact" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
