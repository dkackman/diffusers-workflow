# Testing

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_security.py -v

# Match pattern
pytest tests/ -k "variable_substitution" -v

# Stop on first failure
pytest tests/ -x

# With coverage
pytest tests/ --cov=dw --cov-report=html
# Open htmlcov/index.html to view report
```

## Test Files

| File | Area |
| ---- | ---- |
| test_security.py | Path, URL, input validation |
| test_variables.py | Variable substitution |
| test_workflow.py | Workflow loading and validation |
| test_previous_results.py | Cross-step result references |
| test_result.py | Output file handling |
| test_arguments.py | Argument processing and type conversion |
| test_step.py | Step execution |
| test_gather.py | Resource gathering tasks |
| test_type_helpers.py | Dynamic type loading |
| test_integration.py | End-to-end workflow tests |
| test_task.py | Task dispatch |
| test_schema.py | JSON schema validation |
| test_worker.py | REPL worker subprocess |

## Quick Validation

```bash
# Verify torch and diffusers are working
python -m dw.test

# Validate a workflow against schema
python -m dw.validate examples/FluxDev.json
```
