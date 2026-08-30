# Testing

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_security.py -v

# Match pattern
pytest tests/ -k "variables" -v

# Stop on first failure
pytest tests/ -x

# With coverage
pytest tests/ --cov=dw --cov-report=html
# Open htmlcov/index.html to view report
```

## Test Files

Around 57 files and 1,500+ tests as of this writing (`pytest tests/ --collect-only -q` for the current count). The table below maps the core areas; newer features carry their own `test_<feature>.py` alongside them (server: `test_server.py`, events/cancellation: `test_events.py`, `test_worker_execute.py`, introspection: `test_introspection.py`, hub cache: `test_hub_cache.py`, chaining: `test_chain.py`).

| File | Area |
| ---- | ---- |
| test_security.py | Path, URL, input validation |
| test_variables.py | Variable substitution |
| test_workflow.py | Workflow loading, validation, result eviction, seed resolution |
| test_previous_results.py | Cross-step result references / cartesian products |
| test_result.py | Output file handling |
| test_arguments.py | Argument processing and type conversion |
| test_step.py | Step execution |
| test_gather.py | Resource gathering tasks |
| test_type_helpers.py | Dynamic type loading |
| test_integration.py | End-to-end workflow tests |
| test_task.py | Task dispatch |
| test_schema.py | JSON schema validation |
| test_worker.py | REPL worker subprocess |
| test_repl_commands.py, test_repl_hierarchical.py, test_repl_reorganization.py | REPL command structure |
| test_pipeline_caching.py, test_pipeline_components.py, test_modular_pipeline.py | Pipeline caching, component discovery, `load_components` |
| test_device.py, test_device_helpers.py | Device selection, shared device/dtype helpers |
| test_image_utils.py, test_resize_bucket.py, test_strip_exif_and_watermark.py, test_tensor_image.py, test_list_images.py | Image processing task commands |
| test_diffusion_upscale.py, test_interpolate_frames.py, test_depth_estimator.py, test_segment.py | Diffusion upscale, RIFE interpolation, depth hints, segmentation |
| test_image_to_text.py, test_text_generation.py | Captioning and text generation tasks |
| test_model_cache.py | Shared task model cache |
| test_prompt_weighting.py, test_teacache.py | Prompt weighting device handling, TeaCache forward guard |
| test_argument_updates.py | Cached pipelines pick up fresh arguments across runs |
| test_examples.py | Validates every workflow in `workflows/` against the schema |

## conftest.py

An autouse `_clear_task_model_cache` fixture clears `dw.tasks.model_cache` before and after every test, so a cache hit in one test can't starve a later test's mocked loader of the call it expects. Test-specific fixtures (`test_data_dir`, `temp_output_dir`, `temp_image`, `valid_workflow_json`, etc.) are also defined here — see `tests/README.md` for the list.

## Quick Validation

```bash
# Verify torch and diffusers are working
python -m dw.test

# Validate a workflow against schema
python -m dw.validate workflows/flux/FluxDev.json
```
