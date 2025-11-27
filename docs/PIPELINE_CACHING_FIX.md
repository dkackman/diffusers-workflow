# Pipeline Caching Fix - GPU Model Persistence

## Problem

The REPL worker was designed to keep models loaded in GPU memory across multiple workflow runs, but **models were being reloaded from disk on every execution** even though the worker maintained a pipeline cache.

### Root Cause

The workflow's `create_step_action()` method was **always creating new `Pipeline` objects** and calling `Pipeline.load()`, which reloaded models from disk/HuggingFace cache every time. The pipeline cache (`self.loaded_pipelines` in the worker) was being passed to the workflow but **never checked** before creating new pipelines.

**Before the fix:**
```python
# workflow.py - create_step_action()
if "pipeline" in step_definition:
    pipeline = Pipeline(...)
    pipeline.load(shared_components)  # ❌ ALWAYS LOADS
    previous_pipelines[step_definition["name"]] = pipeline
    return pipeline
```

## Solution

Modified `workflow.py` to **check the cache first** before creating/loading pipelines:

```python
# workflow.py - create_step_action()
if "pipeline" in step_definition:
    step_name = step_definition['name']
    
    # Check if pipeline already loaded in cache (GPU persistence)
    if step_name in previous_pipelines:
        logger.debug(f"Reusing cached pipeline for step: {step_name}")
        return previous_pipelines[step_name]  # ✅ REUSE
    
    # Not in cache - load fresh
    logger.debug(f"Creating pipeline for step: {step_name}")
    pipeline = Pipeline(...)
    pipeline.load(shared_components)
    previous_pipelines[step_name] = pipeline
    return pipeline
```

## How It Works

### First Run (Cold Start)
1. Worker receives workflow execution command
2. Workflow hash computed - detects new/changed workflow
3. Worker clears cache: `self._cleanup_all()`
4. Workflow loaded fresh
5. Pipeline created and models loaded into GPU: `pipeline.load()`
6. Pipeline stored in cache: `self.loaded_pipelines[step_name] = pipeline`
7. Workflow executes successfully

### Subsequent Runs (Hot Cache)
1. Worker receives workflow execution command
2. Workflow hash matches - **no change detected**
3. Worker message: "Reusing loaded models from cache"
4. Workflow calls `create_step_action()`
5. **Cache hit**: `step_name in previous_pipelines` returns `True`
6. Cached pipeline returned - **no model loading**
7. Workflow executes using cached GPU models

### Workflow Change Detection
1. Worker computes SHA256 hash of workflow file
2. If hash differs from cached hash:
   - Full cleanup: `self._cleanup_all()` clears all cached pipelines
   - New workflow loaded
   - Models loaded fresh on next execution
3. If hash matches:
   - Models stay loaded in GPU
   - Significant performance improvement

## Performance Impact

### Before Fix
- **Every run**: Load models from disk → GPU (30-60 seconds for large models)
- GPU memory allocated fresh each time
- High disk I/O and memory bandwidth usage

### After Fix
- **First run**: Load models from disk → GPU (30-60 seconds)
- **Subsequent runs**: Reuse GPU models (instant - no loading!)
- Minimal overhead between runs (only garbage collection)
- Dramatic speedup for iterative workflows

## Testing

The fix includes comprehensive tests in `test_pipeline_caching.py`:

1. **Single step caching**: Verifies same pipeline reused across runs
2. **Multi-step caching**: Verifies different steps have separate cache entries
3. **Cache isolation**: Confirms pipelines aren't mixed between steps

### Test Results
```
✅ First run loaded exactly once
✅ Second run reused cached pipeline (no reload)
✅ Both runs returned the same Pipeline object
✅ Multi-step test: 2 loads for 2 steps, then reuse
```

## Architecture Components

### Worker (`dw/worker.py`)
- Maintains `self.loaded_pipelines` dict (pipeline cache)
- Detects workflow changes via SHA256 hash
- Passes cache to workflow: `workflow.run(arguments, self.loaded_pipelines)`
- Cleans cache only on workflow change or explicit clear

### Workflow (`dw/workflow.py`)
- Receives `previous_pipelines` parameter (the cache)
- **NEW**: Checks cache before creating pipelines
- Stores pipelines in cache for reuse
- Performs lightweight cleanup between runs (gc + CUDA cache clear)

### Pipeline (`dw/pipeline_processors/pipeline.py`)
- Wraps HuggingFace Diffusers pipelines
- `load()` method loads models from disk/HF cache
- Once loaded, pipeline object persists in worker cache

## Memory Management

### Between Runs (Same Workflow)
- **Keep**: Loaded models in GPU
- **Clear**: Python garbage, CUDA tensor cache
- **Monitor**: Memory growth warnings (>500MB increase)

### On Workflow Change
- **Clear**: All cached pipelines
- **Clear**: Shared components
- **Reset**: Worker state
- **Aggressive CUDA cleanup**: `torch.cuda.empty_cache()` + `synchronize()`

## Usage in REPL

```bash
dw> load examples/FluxDev.json
Loading workflow: FluxDev

dw> run prompt="a cat"
Loading workflow from examples/FluxDev.json
Loading pipeline: black-forest-labs/FLUX.1-dev  # ← First run: loads model
Workflow completed (30-60 seconds)

dw> run prompt="a dog"  
Reusing loaded models from cache  # ← Subsequent run: instant!
Workflow completed (5-10 seconds)

dw> clear
Full cleanup performed
GPU memory cleared

dw> run prompt="a bird"
Loading pipeline: black-forest-labs/FLUX.1-dev  # ← Loads fresh after clear
Workflow completed (30-60 seconds)
```

## Log Messages

### Cache Hit (Model Reuse)
```
DEBUG - Reusing cached pipeline for step: generate
INFO  - Reusing loaded models from cache
```

### Cache Miss (Fresh Load)
```
DEBUG - Creating pipeline for step: generate
INFO  - Loading pipeline from model: black-forest-labs/FLUX.1-dev
INFO  - Loading pipeline_generate from model: black-forest-labs/FLUX.1-dev
```

### Workflow Change
```
INFO  - Workflow file changed - reloading models...
INFO  - Performing full cleanup
INFO  - Loading workflow from examples/FluxDev.json
```

## Benefits

1. **Massive Performance Improvement**: Subsequent runs are 5-10x faster
2. **Better GPU Utilization**: Models stay warm in GPU memory
3. **Reduced Disk I/O**: No repeated model file reads
4. **Lower Memory Fragmentation**: Fewer allocation/deallocation cycles
5. **Improved Developer Experience**: Rapid iteration on prompts and parameters

## Technical Details

- Cache key: Step name from workflow JSON (`step_definition['name']`)
- Cache scope: Per-worker process (isolated per REPL session)
- Cache invalidation: SHA256 hash comparison of workflow file
- Cache lifetime: Until workflow changes or explicit clear command
- Thread safety: Single-threaded worker (no race conditions)

## Related Files

- `dw/worker.py`: Worker process with pipeline cache
- `dw/workflow.py`: Workflow execution with cache checking
- `dw/pipeline_processors/pipeline.py`: Pipeline loading logic
- `test_pipeline_caching.py`: Comprehensive test suite
- `docs/REPL_WORKER_GUIDE.md`: REPL architecture documentation

## Migration Notes

This is a **backward-compatible enhancement**. No changes required to:
- Workflow JSON files
- Command-line usage
- API calls
- Existing workflows

The caching is transparent and automatic.
