"""
Unit tests for step module
Tests step execution, iteration handling, and error management
"""

import pytest
from unittest.mock import Mock, MagicMock
from dw.step import Step
from dw.result import Result


class TestStep:
    """Test Step class functionality"""

    def test_step_initialization(self):
        step_def = {"name": "test_step", "seed": 42}
        step = Step(step_def, default_seed=123)

        assert step.name == "test_step"
        assert step.default_seed == 42  # Uses step-specific seed

    def test_step_uses_default_seed(self):
        step_def = {"name": "test_step"}
        step = Step(step_def, default_seed=999)

        assert step.default_seed == 999

    def test_step_run_single_iteration(self):
        step_def = {"name": "test_step", "result": {}}
        step = Step(step_def, default_seed=42)

        # Mock action
        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"prompt": "test"}
        mock_action.run = Mock(return_value="result_value")

        # Mock previous results (no dependencies)
        previous_results = {}
        previous_pipelines = {}

        result = step.run(previous_results, previous_pipelines, mock_action)

        assert isinstance(result, Result)
        assert mock_action.run.called
        assert result.result_list == ["result_value"]

    def test_step_run_multiple_iterations(self):
        step_def = {"name": "test_step", "result": {}}
        step = Step(step_def, default_seed=42)

        # Mock action that returns different values
        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"image": "previous_result:images"}

        call_count = 0

        def mock_run(args, pipelines):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        mock_action.run = Mock(side_effect=mock_run)

        # Mock previous results with multiple items
        prev_result = Result({})
        prev_result.add_result(["img1", "img2", "img3"])
        previous_results = {"images": prev_result}
        previous_pipelines = {}

        result = step.run(previous_results, previous_pipelines, mock_action)

        assert mock_action.run.call_count == 3
        assert result.result_list == ["result_1", "result_2", "result_3"]

    def test_step_run_with_no_iterations(self):
        step_def = {"name": "test_step", "result": {}}
        step = Step(step_def, default_seed=42)

        # Mock action with empty argument template
        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = []  # List template with no items

        previous_results = {}
        previous_pipelines = {}

        result = step.run(previous_results, previous_pipelines, mock_action)

        # Should return empty result
        assert isinstance(result, Result)
        assert result.result_list == []

    def test_step_run_error_propagation(self):
        step_def = {"name": "test_step", "result": {}}
        step = Step(step_def, default_seed=42)

        # Mock action that raises an error
        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"prompt": "test"}
        mock_action.run = Mock(side_effect=ValueError("Test error"))

        previous_results = {}
        previous_pipelines = {}

        with pytest.raises(ValueError) as exc_info:
            step.run(previous_results, previous_pipelines, mock_action)

        assert "Test error" in str(exc_info.value)

    def test_step_run_cartesian_product(self):
        """Test that multiple result references create cartesian product"""
        step_def = {"name": "test_step", "result": {}}
        step = Step(step_def, default_seed=42)

        # Mock action
        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {
            "image": "previous_result:images",
            "prompt": "previous_result:prompts",
        }

        results = []

        def capture_args(args, pipelines):
            results.append(args.copy())
            return f"result_{len(results)}"

        mock_action.run = Mock(side_effect=capture_args)

        # Create previous results
        images_result = Result({})
        images_result.add_result(["img1.jpg", "img2.jpg"])

        prompts_result = Result({})
        prompts_result.add_result(["prompt A", "prompt B"])

        previous_results = {"images": images_result, "prompts": prompts_result}
        previous_pipelines = {}

        result = step.run(previous_results, previous_pipelines, mock_action)

        # Should create 2x2 = 4 combinations
        assert mock_action.run.call_count == 4
        assert len(results) == 4

        # Verify all combinations were created
        combinations = [(r["image"], r["prompt"]) for r in results]
        assert ("img1.jpg", "prompt A") in combinations
        assert ("img1.jpg", "prompt B") in combinations
        assert ("img2.jpg", "prompt A") in combinations
        assert ("img2.jpg", "prompt B") in combinations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
