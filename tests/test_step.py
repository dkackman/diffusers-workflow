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

    # --- embed_metadata tests ---

    def test_embed_metadata_disabled_by_default(self):
        """embed_metadata not set → Result.metadata stays None"""
        step_def = {"name": "test_step", "result": {}}
        step = Step(step_def, default_seed=42)

        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"prompt": "test"}
        mock_action.run = Mock(return_value="result_value")

        result = step.run({}, {}, mock_action)

        assert result.metadata is None

    def test_embed_metadata_false_leaves_metadata_none(self):
        """embed_metadata explicitly False → Result.metadata stays None"""
        step_def = {"name": "test_step", "result": {"embed_metadata": False}}
        step = Step(step_def, default_seed=42)

        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"prompt": "test"}
        mock_action.run = Mock(return_value="result_value")

        result = step.run({}, {}, mock_action)

        assert result.metadata is None

    def test_embed_metadata_true_pipeline_step(self):
        """embed_metadata=True for a pipeline step → Result carries expected metadata"""
        step_def = {
            "name": "gen_step",
            "pipeline": {
                "from_pretrained_arguments": {"model_name": "my-org/my-model"},
                "arguments": {"prompt": "a cat", "num_inference_steps": 25},
            },
            "result": {"embed_metadata": True},
        }
        step = Step(step_def, default_seed=42)

        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"prompt": "a cat"}
        mock_action.run = Mock(return_value="img.png")

        result = step.run({}, {}, mock_action)

        assert result.metadata is not None
        assert result.metadata["step_name"] == "gen_step"
        assert result.metadata["model_name"] == "my-org/my-model"
        assert result.metadata["arguments"] == {"prompt": "a cat", "num_inference_steps": 25}

    def test_embed_metadata_true_task_step(self):
        """embed_metadata=True for a task step → Result carries task metadata"""
        step_def = {
            "name": "proc_step",
            "task": {
                "command": "process_image",
                "arguments": {"operation": "resize", "width": 512},
            },
            "result": {"embed_metadata": True},
        }
        step = Step(step_def, default_seed=42)

        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"operation": "resize"}
        mock_action.run = Mock(return_value="out.png")

        result = step.run({}, {}, mock_action)

        assert result.metadata is not None
        assert result.metadata["step_name"] == "proc_step"
        assert result.metadata["task_command"] == "process_image"
        assert result.metadata["arguments"] == {"operation": "resize", "width": 512}

    def test_embed_metadata_pipeline_without_model_name(self):
        """embed_metadata=True for pipeline step with no model_name → no model_name key"""
        step_def = {
            "name": "anon_step",
            "pipeline": {
                "from_pretrained_arguments": {},
                "arguments": {"prompt": "test"},
            },
            "result": {"embed_metadata": True},
        }
        step = Step(step_def, default_seed=42)

        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"prompt": "test"}
        mock_action.run = Mock(return_value="out.png")

        result = step.run({}, {}, mock_action)

        assert result.metadata is not None
        assert "model_name" not in result.metadata
        assert result.metadata["step_name"] == "anon_step"

    def test_embed_metadata_shared_across_all_iterations(self):
        """With embed_metadata=True and multiple iterations the single Result carries metadata"""
        step_def = {
            "name": "multi_step",
            "pipeline": {
                "from_pretrained_arguments": {"model_name": "org/model"},
                "arguments": {"image": "previous_result:images"},
            },
            "result": {"embed_metadata": True},
        }
        step = Step(step_def, default_seed=42)

        mock_action = Mock()
        mock_action.name = "mock_action"
        mock_action.argument_template = {"image": "previous_result:images"}

        call_count = 0

        def mock_run(args, pipelines):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        mock_action.run = Mock(side_effect=mock_run)

        images_result = Result({})
        images_result.add_result(["img1.jpg", "img2.jpg", "img3.jpg"])
        previous_results = {"images": images_result}

        result = step.run(previous_results, {}, mock_action)

        # Three iterations should have all run
        assert mock_action.run.call_count == 3
        assert len(result.result_list) == 3

        # Metadata is set once on the Result (not per-iteration)
        assert result.metadata is not None
        assert result.metadata["step_name"] == "multi_step"
        assert result.metadata["model_name"] == "org/model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
