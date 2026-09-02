import pytest
import os
import json
import tempfile
import warnings
from PIL import Image

# Suppress FutureWarnings from dependencies (e.g., timm library deprecated imports)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")


@pytest.fixture(autouse=True)
def _trust_workflows_by_default(monkeypatch):
    """Most of this suite exercises engine mechanics, not the trust gate
    itself - default every test to trusted so a dotted type/pre_load_modules
    fixture used to test something else does not also need to be an
    in-ecosystem name. tests/test_workflow_trust.py explicitly sets this
    False (or unsets it) to exercise the untrusted-by-default behavior.
    """
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")


@pytest.fixture(autouse=True)
def _clear_task_model_cache():
    """Ensure dw.tasks.model_cache is empty at the start of every test.

    Task modules (segment, image_to_text, etc.) route model loading through
    a process-wide cache keyed on (task, model name, device, ...). Without
    resetting it between tests, a cache hit in one test can starve a later
    test's mocked loader of the call it expects.
    """
    from dw.tasks.model_cache import clear_model_cache

    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def all_backends_available(monkeypatch):
    """Let a device named in a test reach the code under test unchanged.

    resolve_device translates a device whose backend this machine does not have,
    so a test that hardcodes 'cuda' to exercise placement or offload plumbing would
    otherwise be testing the translation instead. Portability itself is covered in
    tests/test_device_portability.py.
    """
    import dw

    monkeypatch.setattr(dw, "backend_available", lambda backend: True)


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
