"""Tests for static validation of select's arguments (docs/proposals/score-and-select.md #4).

select's own run-time errors are covered by tests/test_select.py; these are
the free pre-flight versions - a literal bad enough to refuse before the
queue rather than a run started only to fail on its first reducer step.
"""

import unittest

from dw.select_validation import select_errors


def _step(name="pick", **arguments):
    return {
        "name": name,
        "task": {"command": "select", "arguments": arguments},
    }


class TestSelectErrors(unittest.TestCase):
    def test_unknown_rule_is_an_error(self):
        definition = {"steps": [_step(rule="top_k", candidates=["a"], scores=[1])]}

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("rule", errors[0]["message"])
        self.assertEqual(errors[0]["path"], "steps[0].task.arguments.rule")

    def test_known_rule_is_fine(self):
        definition = {
            "steps": [_step(rule="argmax", candidates=["a", "b"], scores=[1, 2])]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_threshold_rule_without_threshold_is_an_error(self):
        definition = {
            "steps": [_step(rule="first_above", candidates=["a"], scores=[1])]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("threshold", errors[0]["message"])

    def test_threshold_rule_with_threshold_is_fine(self):
        definition = {
            "steps": [
                _step(
                    rule="first_above",
                    candidates=["a"],
                    scores=[1],
                    threshold=0.5,
                )
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_non_threshold_rule_with_threshold_is_an_error(self):
        definition = {
            "steps": [
                _step(rule="argmax", candidates=["a"], scores=[1], threshold=0.5)
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("threshold", errors[0]["message"])

    def test_index_rule_without_index_is_an_error(self):
        definition = {"steps": [_step(rule="index", candidates=["a"], scores=[1])]}

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("index", errors[0]["message"])

    def test_index_rule_with_index_is_fine(self):
        definition = {
            "steps": [_step(rule="index", candidates=["a"], scores=[1], index=0)]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_non_index_rule_with_index_is_an_error(self):
        definition = {
            "steps": [_step(rule="argmax", candidates=["a"], scores=[1], index=0)]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("index", errors[0]["message"])

    def test_candidates_gather_scores_plain_list_is_an_error(self):
        definition = {
            "steps": [
                _step(rule="argmax", candidates="gather:still", scores=[1, 2])
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("candidates", errors[0]["message"])
        self.assertIn("scores", errors[0]["message"])

    def test_candidates_and_scores_both_gather_is_fine(self):
        definition = {
            "steps": [
                _step(rule="argmax", candidates="gather:still", scores="gather:judge")
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_candidates_and_scores_both_plain_lists_is_fine(self):
        definition = {
            "steps": [_step(rule="argmax", candidates=["a", "b"], scores=[1, 2])]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_expanded_gather_size_mismatch_is_an_error(self):
        # After for_each expansion a gather: reference has already been
        # replaced by the list of previous_result: references it drew from -
        # this is the shape select_errors actually sees in validation_errors.
        definition = {
            "steps": [
                _step(
                    rule="argmax",
                    candidates=["previous_result:still@a", "previous_result:still@b"],
                    scores=["previous_result:judge@x"],
                )
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(len(errors), 1)
        self.assertIn("scores", errors[0]["path"])
        self.assertIn("2 entries", errors[0]["message"])
        self.assertIn("1", errors[0]["message"])

    def test_expanded_gather_same_size_is_fine(self):
        definition = {
            "steps": [
                _step(
                    rule="argmax",
                    candidates=["previous_result:still@a", "previous_result:still@b"],
                    scores=["previous_result:judge@a", "previous_result:judge@b"],
                )
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])

    def test_non_select_steps_are_ignored(self):
        definition = {
            "steps": [
                {
                    "name": "still",
                    "task": {"command": "generate_image", "arguments": {}},
                }
            ]
        }

        errors = select_errors(definition, source_indices=[0])

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
