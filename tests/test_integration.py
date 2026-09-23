"""
Integration tests for workflow execution
Tests end-to-end workflow scenarios
"""

import pytest
import os
import json
import tempfile
from PIL import Image
from dw.workflow import Workflow, workflow_from_file


def decode_qr(image):
    """The text a QR code image carries, read back with OpenCV.

    OpenCV's default detector misses some codes at 768px that it reads at
    another size (it cannot read "Overridden Content" at 768), so this tries
    the ArUco-based detector and a second size before giving up. Decoding is
    deterministic, so the fallbacks add no flakiness."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    gray = image.convert("L")
    attempts = (
        (cv2.QRCodeDetectorAruco, gray),
        (cv2.QRCodeDetector, gray),
        (cv2.QRCodeDetector, gray.resize((256, 256))),
    )
    for detector, candidate in attempts:
        text, _, _ = detector().detectAndDecode(np.array(candidate))
        if text:
            return text
    return ""


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
                        "user_message": "previous_result:gather_inputs",
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

        assert len(result) == 1
        assert decode_qr(result[0]) == "Overridden Content"

    def test_multi_step_workflow(self, multi_step_workflow, temp_workflow_dir):
        """Test workflow with multiple steps"""
        workflow = Workflow(multi_step_workflow, temp_workflow_dir, "")
        workflow.validate()

        result = workflow.run({})

        # The last step's results: one chat message per value the first step
        # gathered, each carrying that step's substituted variable
        assert result == [
            {
                "text_inputs": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "First"},
                ]
            },
            {
                "text_inputs": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Second"},
                ]
            },
        ]

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

        assert decode_qr(result[0]) == "Hello World"
        # The file's name is the run's identity, and the saved image is the
        # one the step returned
        [entry] = workflow.manifest
        [saved] = entry["files"]
        relative = os.path.relpath(saved, os.path.realpath(temp_workflow_dir))
        assert relative.split(os.sep)[0] == "test_workflow"
        with Image.open(saved) as image:
            assert decode_qr(image) == "Hello World"

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
        # Fatal at run time, so validation reports it rather than letting the
        # run reach the step that spells it
        with pytest.raises(Exception) as validation_error:
            workflow.validate()
        assert "undefined_var" in str(validation_error.value)

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

        # Refused at validation rather than nine steps into a run (#285)
        with pytest.raises(Exception) as exc_info:
            workflow.validate()

        assert "steps[0].task.command" in str(exc_info.value)
        assert "not a registered task command" in str(exc_info.value)


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
                        "arguments": {"value": "previous_result:step1"},
                    },
                    "result": {"content_type": "application/json", "save": False},
                },
            ],
        }

        workflow = Workflow(workflow_data, temp_workflow_dir, "")
        workflow.validate()
        result = workflow.run({})

        # step2 runs once per result step1 produced, receiving each one
        assert result == [{"value": "value1"}, {"value": "value2"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
