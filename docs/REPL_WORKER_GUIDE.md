# REPL Worker Implementation Guide

## Overview

The REPL now uses a **persistent worker subprocess** to execute workflows. This keeps models loaded in GPU memory across multiple runs, dramatically improving iteration speed.

**IMPORTANT:** The worker uses the `spawn` multiprocessing start method for CUDA compatibility. This is automatically configured when you import the REPL module.

## Key Features

### 🚀 GPU Persistence
- Models loaded once and reused across multiple runs
- No model reload cost on subsequent executions
- Automatic detection of workflow file changes

### 🧹 Aggressive Memory Management
- Garbage collection after every workflow run
- CUDA cache clearing between runs
- Full cleanup when workflow changes
- Memory growth monitoring and warnings

### 💪 Robustness
- Worker crashes don't affect REPL
- Automatic worker restart when needed
- Graceful shutdown on exit
- 5-minute timeout per execution

### 📊 Memory Monitoring
- Real-time GPU memory reporting
- Track memory usage across runs
- Warning when memory grows unexpectedly

## Usage

### Starting the REPL

```bash
python -m dw.repl
```

Or with custom log level:

```bash
python -m dw.repl -l DEBUG
```

### Basic Workflow

```
dw> load FluxDev
Loaded workflow: FluxDev
Workflow validated successfully

dw> arg prompt="a cat wearing a hat"
Set argument prompt=a cat wearing a hat

dw> run
Starting worker process...
Worker process started
Running workflow: FluxDev
Workflow file changed - reloading models...
Loading workflow from examples/FluxDev.json
Models loaded for workflow: FluxDev
Executing workflow: FluxDev
[... workflow execution output ...]

GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 8234.5 MB
  Reserved: 8456.2 MB
  Free: 15234.8 MB
  Total: 24000.0 MB
  Runs in this session: 1

Workflow completed successfully
(Workflow has been executed 1 time(s) in this session)

dw> run
Running workflow: FluxDev
Reusing loaded models from cache
Executing workflow: FluxDev
[... faster execution - no model loading! ...]

GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 8240.1 MB
  Reserved: 8456.2 MB
  Free: 15228.2 MB
  Total: 24000.0 MB
  Runs in this session: 2

Workflow completed successfully
(Workflow has been executed 2 time(s) in this session)
```

### Available Commands

#### Workflow Management
- `load [workflow]` - Load a workflow file
- `reload` - Reload current workflow
- `status` - Show current workflow status

#### Execution
- `run` - Execute the loaded workflow (starts worker on first run)
- `arg [name=value]` - Set workflow arguments
- `clear_args` - Clear all arguments

#### Memory Management
- `clear` - Clear GPU memory and restart worker
- `memory` - Show current GPU memory usage

#### Configuration
- `set [name=value]` - Set global options (output_dir, log_level, workflow_dir)

#### General
- `help` - Show all commands
- `exit` - Exit REPL (shuts down worker)

## How It Works

### Architecture

```
┌─────────────────────────────────────────────┐
│            REPL Process                     │
│  - User interface                           │
│  - Command parsing                          │
│  - Security validation                      │
│                                             │
│  ┌─────────────┐       ┌─────────────┐    │
│  │ Command     │◄─────►│ Result      │    │
│  │ Queue       │       │ Queue       │    │
│  └──────┬──────┘       └──────▲──────┘    │
└─────────┼───────────────────┼──────────────┘
          │                   │
          │  IPC              │  IPC
          │                   │
┌─────────▼───────────────────┴──────────────┐
│         Worker Process                      │
│  - Workflow execution                       │
│  - Model caching                            │
│  - Memory management                        │
│                                             │
│  ┌──────────────────────────────┐          │
│  │   GPU Memory                 │          │
│  │   - Models stay loaded       │          │
│  │   - Reused across runs       │          │
│  │   - Cleared on workflow      │          │
│  │     change or 'clear' cmd    │          │
│  └──────────────────────────────┘          │
└─────────────────────────────────────────────┘
```

### Worker Lifecycle

1. **First `run` command**: Worker process starts
2. **Subsequent `run` commands**: Worker reuses loaded models
3. **Workflow file change**: Worker detects hash change, reloads everything
4. **`load` different workflow**: Worker shuts down, restarts on next run
5. **`clear` command**: Worker clears GPU and reloads on next run
6. **`exit` command**: Worker shuts down gracefully

### Memory Management Strategy

The worker implements **three levels** of memory cleanup:

#### 1. Between Steps (Minimal)
```python
# After each workflow step
gc.collect()
```

#### 2. Between Runs (Aggressive)
```python
# After each workflow execution
gc.collect()  # Python garbage collection
torch.cuda.empty_cache()  # Clear CUDA cache
# Monitor for memory growth >500MB
```

#### 3. Full Cleanup (Complete)
```python
# When workflow changes or on 'clear' command
loaded_pipelines.clear()
shared_components.clear()
gc.collect() × 3  # Multiple passes
torch.cuda.empty_cache()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
```

### File Change Detection

The worker computes a SHA256 hash of the workflow JSON file:
- Hash stored after first load
- Compared on every `run` command
- If hash differs → full cleanup and reload
- If hash same → reuse cached models

### Error Handling

#### Worker Crashes
- REPL detects crash via queue timeout or exit
- Error message shown to user
- Worker marked as inactive
- Next `run` starts fresh worker

#### Execution Errors
- Error message and traceback sent to REPL
- Worker stays alive (models still cached)
- Can fix and re-run immediately

#### Timeout
- 5-minute timeout per execution
- Prevents indefinite hangs
- Worker shutdown on timeout

## Performance

### Typical Speedup

**First run** (cold start):
```
Model loading: 30-60 seconds
Inference: 10-30 seconds
Total: 40-90 seconds
```

**Subsequent runs** (warm cache):
```
Model loading: 0 seconds ✨
Inference: 10-30 seconds
Total: 10-30 seconds
```

**Speedup: 2-4x faster iteration!**

### Memory Overhead

- Worker process: ~200MB
- Queue overhead: <10MB
- Total overhead: <250MB (negligible compared to models)

## Troubleshooting

### "Worker did not shutdown gracefully"
- Worker is busy or hung
- REPL terminates it forcefully
- Safe - next run starts fresh worker

### "Error: Security validation failed"
- Invalid path or input detected
- Check file paths and arguments
- Security system prevents malicious inputs

### Memory keeps growing
- Check for memory leaks in workflow
- Use `clear` command to reset
- Monitor with `memory` command
- Worker warns if growth >500MB

### GPU out of memory
1. Use `clear` command to free memory
2. Reduce batch size or model size
3. Check for competing processes using GPU
4. Restart worker with `exit` + restart REPL

### Worker crashes repeatedly
- Check workflow JSON is valid
- Look for errors in output
- Try different workflow to isolate issue
- Check CUDA/PyTorch installation

### "Cannot re-initialize CUDA in forked subprocess"
- This error means multiprocessing isn't using `spawn` method
- Should be automatically fixed by importing `dw.repl`
- If you see this, ensure you're using `python -m dw.repl`
- Don't import torch before setting spawn method

## Advanced Usage

### Running Multiple Iterations

```
dw> load FluxDev
dw> arg prompt="a cat"
dw> run
# ... first run, models loaded ...

dw> arg prompt="a dog"
dw> run
# ... instant start, models cached! ...

dw> arg prompt="a bird"
dw> run
# ... instant start again! ...
```

### Checking Memory Growth

```
dw> memory
GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 8234.5 MB
  Reserved: 8456.2 MB
  Free: 15234.8 MB
  Total: 24000.0 MB
  Runs in this session: 5

dw> run
# ... execute workflow ...

dw> memory
GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 8240.1 MB  # Slight growth is normal
  Reserved: 8456.2 MB
  Free: 15228.2 MB
  Total: 24000.0 MB
  Runs in this session: 6
```

### Forcing Clean State

```
dw> clear
Clearing GPU memory...
GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 0.0 MB  # All cleared!
  Reserved: 0.0 MB
  Free: 24000.0 MB
  Total: 24000.0 MB
  Runs in this session: 0

GPU memory cleared successfully

dw> run
# ... models reload from scratch ...
```

## Testing

Run the test suite:

```bash
python test_worker.py
```

This tests:
- Worker lifecycle (start/stop)
- Memory status reporting
- Memory clearing
- Simple workflow execution
- Model caching across runs

## Implementation Details

### Key Files

- `dw/worker.py` - Worker process implementation
- `dw/repl.py` - REPL with worker management (sets `spawn` method)
- `dw/workflow.py` - Pipeline caching support
- `test_worker.py` - Test suite (also sets `spawn` method)

### Security

All security validation happens in REPL **before** sending to worker:
- Path validation (no traversal attacks)
- Variable name validation (alphanumeric + _-)
- Input sanitization (length limits, control chars)
- No `eval()`, `exec()`, or `shell=True`

Worker only receives pre-validated inputs.

### Multiprocessing

Uses `multiprocessing.Queue` for IPC with **`spawn` start method** (required for CUDA):
- **Command Queue**: REPL → Worker
- **Result Queue**: Worker → REPL

The `spawn` method is critical because:
- Default `fork()` method on Linux doesn't work with CUDA
- CUDA contexts cannot be inherited across fork
- `spawn` creates a fresh Python interpreter in the subprocess
- Automatically set when importing `dw.repl` module

Commands serialized as dictionaries:
```python
{'type': 'execute', 'workflow_path': '...', 'arguments': {...}}
{'type': 'shutdown'}
{'type': 'clear_memory'}
{'type': 'memory_status'}
```

Results include type, message, and metadata.

### Logging

Worker has independent logging:
- Configured via `log_level` parameter
- Logs to same handlers as main process
- DEBUG shows detailed worker operations

## Future Enhancements

Possible improvements:
- [ ] LRU cache for multiple workflows
- [ ] Configurable memory thresholds
- [ ] Auto-cleanup on memory pressure
- [ ] Worker pool for parallel execution
- [ ] Persistent cache across REPL sessions
- [ ] Metrics dashboard
- [ ] Memory profiling mode
