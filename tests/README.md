# Test Suite Documentation

## Overview

Comprehensive test suite for the diffusers-workflow project covering core functionality, security, and integration scenarios.

## Test Organization

37 files, ~580 tests as of this writing (`pytest tests/ --collect-only -q` for the current count).

### Unit Tests
- `test_security.py` - Security validation and input sanitization
- `test_variables.py` - Variable substitution and type conversion
- `test_workflow.py` - Workflow loading, validation, result eviction, seed resolution
- `test_task.py` - Task execution
- `test_schema.py` - JSON schema validation
- `test_previous_results.py` - Result reference handling and cartesian products
- `test_result.py` - Result storage and file saving
- `test_arguments.py` - Argument realization and resource loading
- `test_step.py` - Step execution and iteration handling
- `test_type_helpers.py` - Dynamic type loading
- `test_device.py` / `test_device_helpers.py` - Device selection and shared device/dtype helpers
- `test_gather.py` - Resource gathering from files and URLs
- `test_worker.py` - REPL worker subprocess (spawns a real `multiprocessing.Process`)
- `test_repl_commands.py` / `test_repl_hierarchical.py` / `test_repl_reorganization.py` - REPL command structure
- `test_pipeline_caching.py` / `test_pipeline_components.py` / `test_modular_pipeline.py` - Pipeline caching, component discovery, `load_components`
- `test_model_cache.py` - Shared task model cache (`dw.tasks.model_cache`)
- `test_image_utils.py` / `test_resize_bucket.py` / `test_strip_exif_and_watermark.py` / `test_tensor_image.py` / `test_list_images.py` - Image processing task commands
- `test_diffusion_upscale.py` / `test_interpolate_frames.py` / `test_depth_estimator.py` / `test_segment.py` - Diffusion upscale, RIFE interpolation, depth hints, segmentation
- `test_image_to_text.py` / `test_text_generation.py` - Captioning and text-generation tasks
- `test_prompt_weighting.py` / `test_teacache.py` - Prompt weighting device handling, TeaCache forward guard
- `test_argument_updates.py` - Cached pipelines pick up fresh arguments across runs

### Integration Tests
- `test_integration.py` - End-to-end workflow execution scenarios

### Test Examples
- `test_examples.py` - Validates every workflow in `examples/` against the schema (one parametrized test per file)

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_security.py -v
```

### Run Specific Test Class
```bash
python -m pytest tests/test_task.py::TestTaskDevice -v
```

### Run Specific Test
```bash
python -m pytest tests/test_task.py::TestTaskDevice::test_defaults_to_the_workflow_device -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=dw --cov-report=html --cov-report=term
```

### Run Test Runner Script
```bash
python -m tests.run_tests
```

## Test Coverage

The test suite covers:

### Core Functionality (✓)
- Workflow loading and validation
- Step execution with dependencies
- Variable substitution and type conversion
- Result cartesian products
- File I/O operations

### Security (✓)
- Path traversal prevention
- URL validation
- Variable name validation
- String input sanitization
- Command argument sanitization

### Error Handling (✓)
- Invalid workflow schemas
- Missing variables
- Invalid task commands
- File system errors
- Type conversion failures

### Edge Cases (✓)
- Empty workflows
- Null/None values
- Nested data structures
- Mutable default arguments
- Boolean string conversions

## Fixtures

Available in `conftest.py`:
- `_clear_task_model_cache` - **Autouse.** Clears `dw.tasks.model_cache` before and after every test, so a cache hit in one test can't starve a later test's mocked loader of the call it expects. Applies automatically; no need to request it.
- `test_data_dir` - Path to test data directory
- `temp_output_dir` - Temporary directory for test outputs
- `temp_image` - Temporary test image file
- `valid_workflow_json` - Valid workflow for testing
- `invalid_workflow_json` - Invalid workflow for testing
- `minimal_workflow_json` - Minimal valid workflow
- `mock_pipeline` - Mock pipeline object

## Test Data

Test data files in `tests/test_data/`:
- `workflows/valid_workflow.json` - Valid test workflow
- `workflows/invalid_workflow.json` - Invalid test workflow
- Sample images for image processing tests

## Best Practices

1. **Isolation** - Each test should be independent
2. **Cleanup** - Use fixtures and context managers for resource cleanup
3. **Mocking** - Mock external dependencies (APIs, heavy models)
4. **Naming** - Use descriptive test names (test_should_do_something_when_condition)
5. **Coverage** - Aim for >80% code coverage on core modules

## Adding New Tests

1. Create test file: `tests/test_<module>.py`
2. Import module: `from dw.<module> import ...`
3. Write test class: `class TestFeatureName:`
4. Write test methods: `def test_specific_behavior(self):`
5. Use fixtures: Add parameters for needed fixtures
6. Assert behavior: Use pytest assertions

Example:
```python
import pytest
from dw.my_module import my_function

class TestMyFunction:
    def test_normal_case(self):
        result = my_function("input")
        assert result == "expected"
    
    def test_error_case(self):
        with pytest.raises(ValueError):
            my_function("invalid")
```

## Continuous Integration

Tests should be run:
- Before committing code
- In CI/CD pipeline
- Before releases

## Dependencies

Required:
- `pytest >= 7.0`

Optional:
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-timeout` - Test timeout handling
