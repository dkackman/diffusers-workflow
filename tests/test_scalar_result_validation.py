"""Tests for #212: `validate_workflow` refusing a `result` block on a step
whose command returns a scalar rather than an artifact.

`judge` returns a bare float; a `result` block on such a step validated
clean and then died at run time inside `save_artifact` with
`write() argument must be str, not float` after the fan-out ahead of it had
already generated (#212). These are the free pre-flight versions.
"""

import unittest

from dw.scalar_result_validation import scalar_result_errors


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


if __name__ == "__main__":
    unittest.main()
