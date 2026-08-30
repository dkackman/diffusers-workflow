# REPL Worker Guide

## Overview

The REPL uses a persistent worker subprocess to keep models loaded in GPU memory across runs. After the first run loads a model, subsequent runs skip the loading step entirely.

The worker uses the `spawn` multiprocessing start method, required for CUDA and MPS compatibility. This is configured automatically.

## Usage

```bash
python -m dw.repl
```

```text
dw> workflow load flux/FluxDev
dw> arg set prompt="a cat wearing a hat"
dw> workflow run         # first run — loads model (~30-60s)
dw> arg set prompt="a dog in a park"
dw> workflow run         # instant start — model cached
dw> memory show          # check GPU memory
dw> memory clear         # free GPU memory
```

## How It Works

The REPL process handles user input and validation. The worker process handles model loading, caching, and inference. They communicate via multiprocessing queues.

- **First `workflow run`**: Worker starts and loads the model
- **Subsequent runs**: Worker reuses cached models
- **Workflow file edited**: Worker detects the change (SHA256 hash) and reloads
- **`workflow load` (different file)**: The worker switches in place - it frees the old workflow's models before loading the new one
- **`workflow restart`**: Worker shuts down immediately; a fresh one starts on the next run
- **`memory clear`**: Frees GPU memory, models reload on next run
- **`exit`**: Worker shuts down gracefully

## Memory Management

The worker cleans up automatically between runs (garbage collection + GPU cache clearing). If memory grows unexpectedly, use `memory show` to check and `memory clear` to reset.

## Troubleshooting

**Worker crashes**: The REPL detects it and starts a fresh worker on the next `workflow run`. Error messages are shown in the REPL.

**Execution errors**: The worker stays alive (models cached) so you can fix the issue and re-run immediately.

**Long runs**: There is no execution timeout - a run waits as long as the worker is alive (liveness is polled every second, so a crashed worker is noticed immediately). Ctrl+C cancels the run in place, keeping models cached; a second Ctrl+C stops the worker.

**GPU out of memory**: Use `memory clear`, reduce model size, or check for other processes using the GPU.

**"Cannot re-initialize CUDA in forked subprocess"**: Use `python -m dw.repl` to start the REPL — don't import torch before the REPL sets the spawn method.
