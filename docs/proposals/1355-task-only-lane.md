# #1355 - task-only jobs wait FIFO behind GPU jobs

Problem: `Dispatcher._pick` (`dw/server/dispatch.py`) hands a job a free card
slot in queue order. A task-only workflow (no pipeline step; ffmpeg-class
tasks, 9-14 s) holds a slot like any other, so it waits behind every H3-class
job, minutes to tens of minutes. A consumer that assembles between
generations loses that wait each time.

## Options

A. **Task lane (the issue's first ask).** A job whose definition has no
   pipeline step (`dw/plan.py::_unpriced_reason` already detects it) skips
   `_pending`'s slot gate and runs on its own thread via a separate
   CPU-side worker (not a card's persistent worker, which is busy and
   holds the model cache). Needs: a lane worker (spawned subprocess, one at
   a time, or N), host-RAM accounting for it, `get_health.workers`
   reporting it, `device: null` semantics, cancel and event plumbing. Risk:
   tasks that use the GPU (upscale, resample on cuda) would contend with a
   running job; the lane must be limited to tasks known to be CPU-only, or
   it needs a per-task flag.

B. **Wait hint only.** Keep FIFO; when a job is queued and no card is free,
   emit a `queue_wait` event (reason `gpu_busy`) and fill `waiting_for` with
   "queued behind N jobs on <cards>; running jobs started T ago", plus the
   observed-cost estimate of the running jobs where known
   (`dw/server/observed_cost.py`). Consumers can tell waiting from stuck. Small,
   reversible, no engine change.

Recommendation: B now (it is cheap and solves "stuck vs waiting"), A as a
separate approved piece of work limited to CPU-only tasks.
