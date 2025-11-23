"""
Integration tests for workflow execution
Tests end-to-end workflow scenarios
"""

import pytest
import os
import json
import tempfile
from dw.workflow import Workflow, workflow_from_file
from dw.result import Result


@pytest.fixture
def temp_workflow_dir():
    """Create temporary directory for test workflows"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def simple_qr_workflow():
    """Create a simple QR code generation workflow"""
    return {
        "id": "qr_test_workflow",
        "variables": {"content": "Hello World"},
        "steps": [
            {
                "name": "generate_qr",
                "task": {
                    "command": "qr_code",
                    "arguments": {"qr_code_contents": "variable:content"},
                },
                "result": {"content_type": "image/jpeg", "save": True},
            }
        ],
    }


@pytest.fixture
def multi_step_workflow():
    """Create workflow with multiple steps and dependencies"""
    return {
        "id": "multi_step_test",
        "variables": {"text1": "First", "text2": "Second"},
        "steps": [
            {
                "name": "gather_inputs",
                "task": {
                    "command": "gather_inputs",
                    "inputs": ["variable:text1", "variable:text2"],
                },
                "result": {"content_type": "application/json", "save": False},
            },
            {
                "name": "format_message",
                "task": {
                    "command": "format_chat_message",
                    "arguments": {
                        "system_prompt": "System",
                        "user_message": "variable:text1",
                    },
                },
                "result": {"content_type": "application/json", "save": False},
            },
        ],
    }


class TestWorkflowExecution:
    """Test complete workflow execution"""

    def test_simple_workflow_execution(self, simple_qr_workflow, temp_workflow_dir):
        """Test executing a simple single-step workflow"""
        workflow = Workflow(simple_qr_workflow, temp_workflow_dir, "")
        workflow.validate()

        result = workflow.run({})

        assert result is not None
        assert len(result) > 0

    def test_workflow_with_variable_override(
        self, simple_qr_workflow, temp_workflow_dir
    ):
        """Test executing workflow with variable override"""
        workflow = Workflow(simple_qr_workflow, temp_workflow_dir, "")
        workflow.validate()

        # Override the content variable
        result = workflow.run({"content": "Overridden Content"})

        assert result is not None

    def test_multi_step_workflow(self, multi_step_workflow, temp_workflow_dir):
        """Test workflow with multiple steps"""
        workflow = Workflow(multi_step_workflow, temp_workflow_dir, "")
        workflow.validate()

        result = workflow.run({})

        # Should return the last step's results
        assert result is not None

    def test_workflow_from_file_execution(self, simple_qr_workflow, temp_workflow_dir):
        """Test loading and executing workflow from file"""
        # Write workflow to file
        workflow_path = os.path.join(temp_workflow_dir, "test_workflow.json")
        with open(workflow_path, "w") as f:
            json.dump(simple_qr_workflow, f)

        # Load and execute
        workflow = workflow_from_file(workflow_path, temp_workflow_dir)
        workflow.validate()
        result = workflow.run({})

        assert result is not None

    def test_workflow_result_saving(self, simple_qr_workflow, temp_workflow_dir):
        """Test that workflow results are saved to output directory"""
        workflow = Workflow(simple_qr_workflow, temp_workflow_dir, "")
        workflow.validate()
        workflow.run({})

        # Check that output files were created
        output_files = os.listdir(temp_workflow_dir)
        # Should have at least one output file
        assert len(output_files) > 0
        # Should have files matching the pattern workflow_id-step_name
        assert any("qr_test_workflow" in f for f in output_files)


class TestWorkflowErrorHandling:
    """Test error handling in workflow execution"""

    def test_invalid_workflow_fails_validation(self):
        """Test that invalid workflows fail validation"""
        invalid_workflow = {
            "id": "invalid",
            # Missing required 'steps' field
        }

        workflow = Workflow(invalid_workflow, "./output", "")

        with pytest.raises(Exception) as exc_info:
            workflow.validate()

        assert "Validation error" in str(exc_info.value)

    def test_missing_variable_reference(self, temp_workflow_dir):
        """Test error when referencing undefined variable"""
        workflow_data = {
            "id": "missing_var_test",
            "variables": {},
            "steps": [
                {
                    "name": "test_step",
                    "task": {
                        "command": "qr_code",
                        "arguments": {"qr_code_contents": "variable:undefined_var"},
                    },
                    "result": {"content_type": "image/jpeg", "save": False},
                }
            ],
        }

        workflow = Workflow(workflow_data, temp_workflow_dir, "")
        workflow.validate()

        with pytest.raises(Exception) as exc_info:
            workflow.run({})

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_task_command(self, temp_workflow_dir):
        """Test error when using invalid task command"""
        workflow_data = {
            "id": "invalid_task_test",
            "steps": [
                {
                    "name": "test_step",
                    "task": {"command": "nonexistent_command", "arguments": {}},
                    "result": {"content_type": "application/json", "save": False},
                }
            ],
        }

        workflow = Workflow(workflow_data, temp_workflow_dir, "")
        workflow.validate()

        with pytest.raises(ValueError) as exc_info:
            workflow.run({})

        assert "Unknown task" in str(exc_info.value)


class TestWorkflowStepDependencies:
    """Test workflows with step dependencies using previous_result"""

    def test_simple_dependency(self, temp_workflow_dir):
        """Test workflow where one step depends on previous step"""
        workflow_data = {
            "id": "dependency_test",
            "steps": [
                {
                    "name": "step1",
                    "task": {
                        "command": "gather_inputs",
                        "inputs": ["value1", "value2"],
                    },
                    "result": {"content_type": "application/json", "save": False},
                },
                {
                    "name": "step2",
                    "task": {
                        "command": "gather_inputs",
                        "inputs": ["previous_result:step1"],
                    },
                    "result": {"content_type": "application/json", "save": False},
                },
            ],
        }

        workflow = Workflow(workflow_data, temp_workflow_dir, "")
        workflow.validate()
        result = workflow.run({})

        # step2 should receive the results from step1
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
