# CUDA Multiprocessing Fix

## Problem

When the worker subprocess tried to initialize CUDA, it failed with:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## Root Cause

- Python's default multiprocessing on Linux uses `fork()` 
- `fork()` copies the parent process memory, including CUDA context
- CUDA cannot be re-initialized in a forked process
- This is a known limitation of CUDA with forked processes

## Solution

Set multiprocessing to use `spawn` method instead of `fork`:

```python
import multiprocessing

# Set spawn method before creating any processes
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
```

## Changes Made

### `dw/repl.py`
- Added `import multiprocessing` at top
- Set spawn method immediately after imports
- Changed `Queue()` to `multiprocessing.Queue()`
- Changed `Process()` to `multiprocessing.Process()`

### `test_worker.py`
- Added spawn method configuration at top of file
- Updated all Queue/Process calls to use `multiprocessing.` prefix

### `dw/worker.py`
- Added spawn method configuration in `__main__` block for standalone testing

## How Spawn Works

**Fork (default on Linux):**
```
Parent Process
├─ CUDA initialized
└─> Fork → Child Process (copies parent memory)
             └─ CUDA context copied but invalid ❌
```

**Spawn (required for CUDA):**
```
Parent Process
├─ CUDA initialized
└─> Spawn → Fresh Child Process (new Python interpreter)
              └─ CUDA initialized from scratch ✅
```

## Trade-offs

**Pros:**
- ✅ Works with CUDA
- ✅ Clean process separation
- ✅ No shared memory issues

**Cons:**
- ⚠️  Slower process startup (new interpreter)
- ⚠️  Cannot share memory directly
- ⚠️  Must pickle data for IPC

The spawn overhead is negligible compared to model loading time, so this is the right choice.

## Verification

All tests pass after fix:
```bash
python test_worker.py
# Results: 4 passed, 0 failed ✓
```

## References

- [PyTorch Multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [Python Multiprocessing Contexts](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
- [CUDA Runtime Error](https://github.com/pytorch/pytorch/issues/3492)
