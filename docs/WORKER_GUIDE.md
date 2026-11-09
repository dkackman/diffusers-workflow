# Worker Guide

## Overview

`dw.serve` runs each job in a persistent worker subprocess (`dw/worker.py`,
managed by `dw/worker_manager.py`), one per `--devices` card, so a model
loaded by one job stays in memory for the next. Workers use the `spawn`
start method (required for CUDA and MPS), start on the first job rather
than with the server, and inherit the server's environment. A change to
engine code needs a server restart before a job sees it.

## How It Works

`JobManager` (`dw/server/jobs.py`) sends commands to a worker and reads its
replies over a pair of `multiprocessing.Queue`s.

- **First job on a card**: the worker starts and loads the model.
- **Later jobs**: cached pipelines are reused.
- **Workflow file edited**: a job runs the definition the server checked at
  submit, so an edit reaches only jobs submitted after it. Pipelines are
  cached by what they load, so only a pipeline whose definition changed
  reloads.
- **A different workflow**: the worker prepares the incoming definition and
  frees only the models it does not load; a pipeline both workflows load
  (same weights identity) stays warm. The step cache and the task model
  cache are emptied, and if the incoming definition cannot be prepared
  everything is released.
- **`POST /api/memory/clear`** (MCP `clear_memory`): drops every pipeline
  and the step cache on each idle card; models reload on the next job. The
  worker process stays alive (same pid), so process-level state is not
  renewed; a fresh process needs the worker to exit.
- **diffusers updated from the Models page**: an idle worker is shut down,
  so the next job imports the new version.
- **PyTorch settings**: the worker applies dw's TF32 and cuDNN settings
  (`configure_torch`, `dw/__init__.py`) when it starts and again at each
  job's start, since the parent server runs no pipeline. The defaults are
  TF32 off, the cuDNN autotuner off and deterministic cuDNN on, so the same
  seed gives the same output; a fresh worker's first job logs what it ran
  with (`torch: cudnn benchmark=false deterministic=true, tf32=false`). See
  [ACCELERATION.md](ACCELERATION.md#tf32-and-cudnn).
- **Server shutdown**: workers shut down gracefully (10 s, then terminate).

How jobs are assigned to cards, the host-RAM hold and the OOM ranking are in
[SERVER.md](SERVER.md#jobs-api).

## Memory Management

The worker collects garbage and clears the device cache between runs.
`GET /api/memory` reports a card's reading and `POST /api/memory/clear`
empties it; both take an optional `?device=`.

## Cold Loads

With `DW_QUANTIZED_CACHE=1`, the first cold load of an SDNQ
on-the-fly-quantized component pays the quantize and a save to disk; later
cold loads (a new worker, after a restart or a memory clear) read the cached
entry. See [QUANTIZATION.md](QUANTIZATION.md#cached-on-the-fly-dw_quantized_cache).

## Troubleshooting

**Worker crashes**: only its own job fails, and the next job on that card
starts a fresh worker with a cold cache. The job's error carries
`WorkerManager.crash_details()` - the signal (such as SIGKILL) when the OS
reports one.

**Execution errors**: the worker stays alive with its models cached, except
after a CUDA error that poisons the process's context
(`poisons_cuda_context`, `dw/worker_protocol.py`); then the worker restarts.

**Long runs**: there is no execution timeout. A run waits as long as its
worker is alive; liveness is polled every second, so a dead worker is
noticed within a second. `POST /api/jobs/{id}/cancel` cancels in place and
keeps models cached; during a model load or a task step it takes effect
once that finishes (a `cancel_pending` event says so).

**GPU out of memory**: `POST /api/memory/clear`, a smaller or quantized
model, offloading ([ACCELERATION.md](ACCELERATION.md)), or check for other
processes using the GPU.
