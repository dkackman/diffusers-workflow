# Testing Guide - Quick Start

## Installation

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or run the setup script
./setup_tests.sh
```

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Specific Module
```bash
python -m pytest tests/test_security.py -v
```

### With Coverage
```bash
python -m pytest tests/ --cov=dw --cov-report=html
# Open htmlcov/index.html to view report
```

### Using Test Runner
```bash
python -m tests.run_tests
```

## Test Files

| File | Purpose | Tests |
|------|---------|-------|
| test_security.py | Security validation | 10 |
| test_variables.py | Variable handling | 12 |
| test_workflow.py | Workflow execution | 8 |
| test_previous_results.py | Result references | 15 |
| test_result.py | Result storage | 15 |
| test_arguments.py | Argument processing | 15 |
| test_step.py | Step execution | 7 |
| test_gather.py | Resource gathering | 15 |
| test_type_helpers.py | Type loading | 12 |
| test_integration.py | End-to-end tests | 12 |
| test_task.py | Task execution | 7 |
| test_schema.py | Schema validation | 5 |

**Total: 134+ tests across 13 files**

## Key Features Tested

✓ Workflow loading and validation  
✓ Variable substitution  
✓ Step dependencies (cartesian products)  
✓ Security validation (paths, URLs, inputs)  
✓ File I/O operations  
✓ Error handling  
✓ Type conversion  
✓ Resource gathering  

## Documentation

- `tests/README.md` - Detailed test documentation
- `TEST_IMPLEMENTATION_SUMMARY.md` - Implementation details
- This file - Quick reference

## Common Commands

```bash
# Run tests matching pattern
pytest tests/ -k "security"

# Run with verbose output
pytest tests/ -vv

# Stop after first failure
pytest tests/ -x

# Show local variables in failures
pytest tests/ -l

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Generate coverage report
pytest tests/ --cov=dw --cov-report=term-missing
```

## CI/CD Integration

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest tests/ --cov=dw --cov-report=xml
```

## Need Help?

See `tests/README.md` for comprehensive documentation.
