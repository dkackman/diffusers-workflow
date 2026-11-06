# Testing

## Running Tests

```bash
# All tests - parallel across every core (pytest-xdist, `-n auto` in pytest.ini)
pytest tests/ -v

# Single file - naming test files runs them in one process, since a worker's
# torch import costs more than it saves there; -n N forces workers anyway
pytest tests/test_security.py -v

# Debugging with -s or --pdb needs one process
pytest tests/ -n0 -s

# The real-model tests: the release gate, excluded by default (-m "not integration"
# in pytest.ini). They need CUDA or MPS and download SD 1.5 and facebook/mms-tts-eng
# on first run; one process, since each test holds a model on the device
pytest tests/ -m integration -n0

# Match pattern
pytest tests/ -k "variables" -v

# Stop on first failure
pytest tests/ -x

# With coverage; open htmlcov/index.html
pytest tests/ --cov=dw --cov-report=html
```

`scripts/preflight.sh` runs every local pre-merge check and lists the ones that failed: `ruff format` and `ruff check --fix` (both rewrite files in place), pytest, the integration tests, the architecture ratchet (`scripts/arch_metrics.py --check docs/stabilization/baseline.json`) and the UI's `npm run preflight`.

## Test Files

Over 300 files (`pytest tests/ --collect-only -q` for the current count). The table maps the core areas; a feature carries its own `test_<feature>.py`. `tests/README.md` lists more.

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
| test_real_models.py | Real-model release gate (`-m integration`) |
| test_task.py | Task dispatch |
| test_schema.py | JSON schema validation |
| test_worker.py, test_worker_execute.py | Worker subprocess |
| test_worker_manager.py | Worker process lifecycle management |
| test_server.py, test_events.py | HTTP server, job events and cancellation |
| test_pipeline_caching.py, test_pipeline_components.py, test_modular_pipeline.py | Pipeline caching, component discovery, `load_components` |
| test_device.py, test_device_helpers.py | Device selection, shared device/dtype helpers |
| test_image_utils.py, test_resize_bucket.py, test_strip_exif_and_watermark.py, test_tensor_image.py, test_list_images.py | Image processing task commands |
| test_diffusion_upscale.py, test_interpolate_frames.py, test_depth_estimator.py, test_segment.py | Diffusion upscale, RIFE interpolation, depth hints, segmentation |
| test_image_to_text.py, test_text_generation.py | Captioning and text generation tasks |
| test_model_cache.py | Shared task model cache |
| test_prompt_weighting.py | Prompt weighting device handling |
| test_argument_updates.py | Cached pipelines pick up fresh arguments across runs |
| test_examples.py | Validates every workflow in `workflows/` against the schema |
| test_introspection.py, test_hub_cache.py, test_chain.py | Introspection, Hub cache, chaining |
| test_mcp_client.py, test_mcp_catalog.py, test_mcp_authoring.py, test_mcp_diagnose.py, test_mcp_media.py, test_mcp_server.py | MCP tool surface |
| test_architecture_map.py | Every path, name and test `docs/ARCHITECTURE.md` names exists |

The MCP tests fake the REST API with `httpx.MockTransport`; they never start `dw.serve` or touch a GPU. `test_mcp_server.py` builds a real `MCPServer` and skips (`pytest.importorskip("mcp", ...)`) when the `mcp` extra is not installed.

## conftest.py

`tests/conftest.py` sets `DW_STRICT_RESPONSES=1`, so a key a server route emits that its response model does not declare is a 500. Its autouse fixtures make each test independent of the machine and of earlier tests:

- `_trust_workflows_by_default` sets `DW_TRUST_WORKFLOWS=1`; `tests/test_workflow_trust.py` unsets it to test the trust gate.
- `_isolate_settings_dir` points `DIFFUSERS_HELPER_ROOT` at the test's `tmp_path`.
- `_isolate_device_capacity` checks a `vram_estimate` against the catalog's measured cards, never the local accelerator.
- `_clear_task_model_cache` and `_clear_step_cache_after_test` empty the process-wide task model cache and step cache, so a cache hit in one test cannot starve a later test's mocked loader.

Shared fixtures (`test_data_dir`, `temp_output_dir`, `temp_image`, `valid_workflow_json`, and others) are defined there too.

## UI Tests

From `ui/`:

```bash
npm run check          # svelte-check + tsc
npm run lint           # eslint
npm test               # vitest unit tests
npx playwright test    # e2e (npm run e2e)
npm run preflight      # check, lint, format, ratchet, build, test, e2e
```

`ui/playwright.config.ts` starts its own server (`ui/e2e/serve_fixture.py --port 8971`, against a scratch copy of the example workflows and prompts), so do not run `dw.serve` on that port. It runs on `DW_E2E_PYTHON` when set, else `venv/bin/python` or `.venv/bin/python`, else `python3`. It uses one worker because the fixture server's first import of a diffusers pipeline module is not concurrency-safe. The e2e specs serve the built UI, so run `npm run build` first.

## CI

`.github/workflows/ci.yml` runs on pushes to `master` and `develop`, on `v<semver>` tags and on pull requests:

- **backend** (Python 3.12, CPU torch): the architecture ratchet, `ruff format --check`, `ruff check`, and `pytest -q`. Each step runs when the install succeeded, so one failure does not hide another.
- **ui**: the response contract (`npm run gen:api` must leave `src/lib/generated` unchanged), prettier, lint, the UI ratchet, type check, unit tests and build.
- **e2e**: Playwright against a real `dw.serve`, on pull requests into `develop` or `master` (except the `develop` to `master` release PR) and on pushes to `develop`.

The `wheel`, `release`, `pypi` and `docker` jobs run on tags; see [RELEASING.md](RELEASING.md).

## Quick Validation

```bash
# End-to-end check: downloads SD 1.5 and generates one image
python -m dw.test

# Validate a workflow against schema
python -m dw.validate workflows/models/flux-dev.json
```
