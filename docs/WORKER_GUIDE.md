# Worker Guide

## Overview

`dw.serve` runs a persistent worker subprocess (`dw/worker.py`, managed by
`dw/worker_manager.py`) to keep models loaded in GPU memory across jobs.
After the first job loads a model, later jobs against the same workflow skip
the loading step entirely.

The worker uses the `spawn` multiprocessing start method, required for CUDA
and MPS compatibility. This is configured automatically.

## How It Works

`JobManager` sends commands to the worker and reads results over
`multiprocessing.Queue`s.

- **First run of a workflow**: Worker starts and loads the model
- **Subsequent runs**: Worker reuses cached models
- **Workflow file edited**: A job runs the definition the server checked when it was submitted, so an edit reaches only jobs submitted after it. Pipelines are cached by what they load, so only a pipeline whose definition changed reloads
- **A different workflow queued**: The worker switches in place - it frees the old workflow's models before loading the new one
- **`POST /api/memory/clear`**: Frees GPU memory, models reload on next run
- **Server shutdown**: Worker shuts down gracefully

## Memory Management

The worker cleans up automatically between runs (garbage collection + GPU
cache clearing). If memory grows unexpectedly, `GET /api/memory` reports the
current reading and `POST /api/memory/clear` resets it.

Switching to a different workflow releases only the models the next workflow
does not load: the worker prepares the incoming definition, and a pipeline
both workflows load (same weights identity) stays warm. The step cache and
the task model cache are still emptied, and if the incoming definition cannot
be prepared everything is released. `clear_memory` (`POST /api/memory/clear`)
remains the way to empty the card by hand.

## Troubleshooting

**Worker crashes**: `JobManager` detects it and starts a fresh worker on the
next job. `WorkerManager.crash_details()` reports why, when the OS can say
(a signal such as SIGKILL) - see the job's error message.

**Execution errors**: The worker stays alive (models cached) so the next job
can run immediately.

**Long runs**: There is no execution timeout - a run waits as long as the
worker is alive (liveness is polled every second, so a crashed worker is
noticed immediately). `POST /api/jobs/{id}/cancel` cancels a running job in
place, keeping models cached.

**GPU out of memory**: Use `POST /api/memory/clear`, reduce model size, or
check for other processes using the GPU.
