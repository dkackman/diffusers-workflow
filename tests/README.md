# Test Suite Documentation

## Overview

Comprehensive test suite for the diffusers-workflow project covering core functionality, security, and integration scenarios.

## Test Organization

### Unit Tests
- `test_security.py` - Security validation and input sanitization
- `test_variables.py` - Variable substitution and type conversion
- `test_workflow.py` - Workflow loading and validation
- `test_task.py` - Task execution
- `test_schema.py` - JSON schema validation
- `test_previous_results.py` - Result reference handling and cartesian products
- `test_result.py` - Result storage and file saving
- `test_arguments.py` - Argument realization and resource loading
- `test_step.py` - Step execution and iteration handling
- `test_type_helpers.py` - Dynamic type loading

### Integration Tests
- `test_integration.py` - End-to-end workflow execution scenarios

### Test Examples
- `test_examples.py` - Example workflow validation

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
python -m pytest tests/test_security.py::TestPathValidation -v
```

### Run Specific Test
```bash
python -m pytest tests/test_security.py::TestPathValidation::test_path_traversal -v
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
