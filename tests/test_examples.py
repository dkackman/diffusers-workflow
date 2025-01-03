import os
import pytest
from dw.workflow import workflow_from_file


def get_example_files():
    """Get all JSON files from the examples directory"""
    examples_dir = "./examples"
    return [
        os.path.join(examples_dir, f)
        for f in os.listdir(examples_dir)
        if f.endswith(".json")
    ]


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_workflow(example_file):
    """Test that each example workflow file can be loaded and validates"""
    try:
        workflow = workflow_from_file(example_file, ".")
        workflow.validate()
    except Exception as e:
        pytest.fail(f"Example {example_file} failed validation: {str(e)}")
