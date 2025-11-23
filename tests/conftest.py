import pytest
import os
import json
import tempfile
from PIL import Image


@pytest.fixture
def test_data_dir():
    """Get path to test data directory"""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_image():
    """Create a temporary test image"""
    img = Image.new("RGB", (100, 100), color="red")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def valid_workflow_json():
    """Valid workflow JSON for testing"""
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
    """Invalid workflow JSON for testing"""
    return {
        "id": "test_workflow",
        # Missing required 'steps' field
        "variables": {"prompt": "test prompt"},
    }


@pytest.fixture
def minimal_workflow_json():
    """Minimal valid workflow for testing"""
    return {"id": "minimal_workflow", "steps": []}


@pytest.fixture
def mock_pipeline():
    """Mock pipeline for testing"""

    class MockPipeline:
        def __init__(self):
            self.called = False

        def __call__(self, **kwargs):
            self.called = True
            return type("MockOutput", (), {"images": ["mock_image"]})()

    return MockPipeline()
