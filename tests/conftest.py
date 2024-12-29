import pytest
import os
import json


@pytest.fixture
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def valid_workflow_json():
    return {
        "id": "test_workflow",
        "variables": {"prompt": "test prompt", "num_images": 1},
        "steps": [
            {
                "name": "test_step",
                "task": {
                    "command": "qr_code",
                    "arguments": {"qr_code_contents": "variable:prompt"},
                },
                "result": {"content_type": "image/jpeg"},
            }
        ],
    }


@pytest.fixture
def invalid_workflow_json():
    return {
        "id": "test_workflow",
        # Missing required 'steps' field
        "variables": {"prompt": "test prompt"},
    }
