import pytest
from dh.workflow import Workflow, workflow_from_file
import os

def test_workflow_validation_valid(valid_workflow_json):
    workflow = Workflow(valid_workflow_json, "./output")
    workflow.validate()  # Should not raise exception

def test_workflow_validation_invalid(invalid_workflow_json):
    workflow = Workflow(invalid_workflow_json, "./output")
    with pytest.raises(Exception) as exc_info:
        workflow.validate()
    assert "Validation error" in str(exc_info.value)

def test_workflow_name(valid_workflow_json):
    workflow = Workflow(valid_workflow_json, "./output")
    assert workflow.name == "test_workflow"

def test_workflow_from_file(test_data_dir):
    workflow_path = os.path.join(test_data_dir, "workflows", "valid_workflow.json")
    workflow = workflow_from_file(workflow_path, "./output")
    assert isinstance(workflow, Workflow)
