# The UI's response contract runs strict under test: an undeclared key a
# route emits is a 500 here, not a silent pass (dw/server/api_models.py)
import os as _os

_os.environ.setdefault("DW_STRICT_RESPONSES", "1")

import gc
import pytest
import os
import tempfile
import warnings
from PIL import Image

# Suppress FutureWarnings from dependencies (e.g., timm library deprecated imports)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")


def pytest_xdist_auto_num_workers(config):
    # Every worker pays a torch import, which costs a run of one or two files
    # more than it saves; `-n auto` parallelizes a directory or the whole suite
    if any(os.path.isfile(arg.split("::")[0]) for arg in config.args):
        return 0
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # The engine calls gc.collect() between steps; over the whole suite's heap
    # each one costs ~0.2s, a third of the run. Freezing what earlier tests
    # left behind keeps a collection to the objects this test creates.
    gc.freeze()


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
def _isolate_settings_dir(tmp_path, monkeypatch):
    """Point dw's settings directory at the test's own tmp_path.

    Every run takes a lock under it (dw.runs.run_lock_path), so without
    this the suite writes into the real ~/.diffusers_helper - and a test
    would read whatever settings.json the machine running it happens to
    hold. A test about the settings directory itself sets its own.
    """
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path / "helper"))


@pytest.fixture(autouse=True)
def _isolate_device_capacity(monkeypatch):
    """Check a vram_estimate against the catalog's measured cards, never the
    accelerator the suite happens to run on.

    The check uses the serving device's own capacity when no 'cost' entry
    describes its backend (dw/vram_estimate.py `_entries_for`), so on a
    24 GB Mac - a 17.8 GB Metal working set - a template's own defaults were
    refused and its tests failed there and nowhere else. With no capacity
    read, every machine gets what CI gets. A test about the device ceiling
    patches its own capacity, which wins over this one.
    """
    import dw.validation
    import dw.workflow_run

    for module in (dw.validation, dw.workflow_run):
        monkeypatch.setattr(module, "device_capacity_gb", lambda device=None: None)


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


@pytest.fixture(autouse=True)
def _clear_step_cache_after_test():
    """Ensure dw.step_cache.step_cache is empty after every test.

    It is a process-global singleton keyed on (workflow id, step name), so a
    populated entry would let any later test that runs a seeded workflow
    twice - or runs one this suite already ran - be served a cached result
    instead of executing its step.
    """
    from dw.step_cache import step_cache

    step_cache.clear()
    yield
    step_cache.clear()


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
