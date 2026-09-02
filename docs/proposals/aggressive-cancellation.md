# Proposal: interrupting model loads and task steps mid-operation

Status: **not committed** - scoping notes for a future decision, written
alongside the small `cancel_pending` UX fix (see PR/commit that added this
file). Nothing here should be implemented without a fresh look at whether it
is still worth the cost.

## The gap this does not close

`RunContext.check_cancelled()` is a cooperative flag: something in the run
has to call it. Today that happens between workflow steps (`step.py`), inside
the per-diffusion-step callback (`pipeline.py`'s `callback_on_step_end`), and
once before a task starts (`task.py`). Two phases have no such checkpoint:

- **Loading** (`Pipeline.load` / `from_pretrained`, LoRA loads, component
  loads) - a single call into `diffusers`/`transformers`/`safetensors` that
  does not return control until every shard is on device. A 30GB model load
  can run for minutes with nothing to interrupt it.
- **Task steps** (`Task.run`'s command handlers) - one Python call per task;
  whatever it does (subprocess, CPU-bound decode, an HTTP call) runs to
  completion once started.

The fix landed alongside this doc (`RunContext.note_phase` /
`NON_INTERRUPTIBLE_PHASES` in `dw/events.py`) only makes that wait visible -
it emits a `cancel_pending` event so the UI can say "cancelling after current
phase..." instead of going silent. It does not make the wait shorter. This
doc is about actually shortening it.

## Why "just check a flag more often" does not work here

Cooperative cancellation needs a call site the running code passes through
periodically. `from_pretrained` and most task commands are opaque calls into
library code `dw` does not control and that offers no callback or polling
hook comparable to `callback_on_step_end`. There is nothing in `dw`'s own
code to add a `check_cancelled()` call *to* - the interpreter is inside
someone else's C-extension-backed I/O loop (safetensors deserialization,
shard downloads, tensor `.to(device)` copies) for the duration.

So real interruption is not a matter of finding one more place to call
`check_cancelled()`. It requires either (a) a hook the loader actually
exposes, or (b) reaching in from outside the call - a different thread or
process that can act on it without the loader's cooperation.

## Options, roughly in order of how much they'd actually buy

### 1. Poll for per-shard/per-file cancellation hooks in diffusers/HF loaders

`from_pretrained` loads a checkpoint as a sequence of shard files (for a
sharded `safetensors` model) or component-by-component (each pipeline
sub-model is its own `from_pretrained` call already, which is why `loading`
already reports "which model" via `detail=`). In principle a large load could
be decomposed further: load component by component ourselves and check
`check_cancelled()` between them, rather than one call per component.

**What this buys**: coarser interruption than none - a multi-component
pipeline (transformer + text encoder + VAE, say) could stop between
components instead of only before the whole pipeline starts. It does nothing
for a load that is dominated by one huge component (the transformer, usually)
- cancelling mid-transformer-load is still not possible without diffusers
exposing a hook inside `from_pretrained` itself, and it does not.

**Risk/effort**: Medium-low. `dw`'s `Pipeline.load` already loads components
somewhat independently (`load_optional_component`, LoRA loads are already
separate calls per `emit_phase("loading", detail=f"LoRA: {model_name}")`).
Adding a `check_cancelled()` between those calls is a small, safe change in
the same spirit as the `cancel_pending` fix - worth doing as a low-risk
follow-up on its own, independent of anything below. It caps how much of a
multi-component load is uninterruptible; it does not remove the cap.

### 2. Run the load (or the task) in a killable unit and restart around it

Move the non-interruptible operation off the worker's main thread and give it
somewhere it can be killed from outside:

- **A separate thread**, cancelled via... nothing safe. Python has no
  supported way to kill a thread from outside; `ctypes`-based tricks to raise
  an exception in a target thread are unreliable against C-extension calls
  (which is exactly where the time is being spent) and can corrupt CPython's
  internal state. Not viable.
- **A separate process**, killed via `SIGKILL`/`Process.terminate()`. This
  *is* viable - a process can always be killed regardless of what it's
  doing - but changes what "cancel" costs: the worker process is exactly the
  thing that exists to keep models warm across runs (see `worker.py`'s
  module docstring and `WorkerManager` in `repl_worker.py`, which already
  knows how to `terminate()`/`kill()` a hung worker on *shutdown*). Killing
  the process mid-load to honor a cancel throws away every model already
  cached in that process - the next run, cancelled or not, pays a full cold
  load again. That is the real trade: instant cancel vs. losing the warm
  cache, and it is not obviously worth it for the common case (a user who
  cancels a slow load usually wants a *different* run next, not the same
  model reloaded from scratch either way - so the cache loss may be moot in
  practice, but it is not free in the case where they queue the same
  workflow again).

  A cheaper variant: keep the persistent worker as-is for everything else,
  but run *just* the load in a short-lived child process, stream the loaded
  state back (impossible for live GPU tensors without re-loading in the
  parent - so this collapses to "load happens twice" and is worse than doing
  nothing). Not viable as stated; mentioned only to rule it out.

- **A subprocess for task steps specifically** (not model loads) is more
  promising in isolation: a task step already goes through
  `sanitize_command_args()`-style handling for real subprocess tasks, and
  killing a subprocess loses no GPU state - there is nothing warm to lose.
  For task handlers that already shell out, propagating cancel as a process
  kill is comparatively low-risk. For task handlers that run in-process
  (pure Python, no subprocess), the same "no safe thread kill" problem
  applies and there is no cheap answer.

**Risk/effort**: High for model loads specifically, because the entire
value proposition of the persistent worker (`CLAUDE.md`: "keep GPU models
cached between runs") is what a kill-to-cancel would sacrifice. Medium for
subprocess-shaped task steps alone.

### 3. Accept the current boundary, invest in making the wait short and honest

The wait for a non-interruptible phase to finish is bounded - it is not
"forever," it is "however long this one `from_pretrained` call takes,"
usually seconds to low minutes. The `cancel_pending` fix already shipped
covers the honesty half (say what's happening). The other lever is making
these phases *shorter* on average rather than interruptible - e.g. surfacing
load time in the UI so users learn which models are slow to cancel out of,
or defaulting to smaller/quantized variants where load time dominates. This
is not "aggressive cancellation" at all; it is listed here because it may be
the better use of the same effort budget for the actual pain point (a user
stuck watching a spinner), without touching process lifecycle or the warm
cache at all.

## Recommendation for future scoping (not a decision)

- Option 1 (checkpoint between components/LoRAs during a load) is the only
  one that is both low-risk and strictly additive to what already exists
  (`Pipeline.load`'s existing per-component `emit_phase("loading", ...)`
  calls). If this gets picked up, it should land as its own small,
  independently-tested change - not bundled with anything below.
- Option 2's process-kill approach is the only way to get *true* mid-load
  interruption, and it is a real architecture change (worker lifecycle,
  cache-loss UX, what "cancel" promises callers) that deserves its own design
  doc and explicit sign-off on losing the warm cache on a hard cancel - not
  something to fold into a "small fix."
- Task steps that already shell out to a subprocess are the cheapest real win
  under option 2 and could be scoped separately from model loading entirely.
- Option 3 is worth keeping in mind as the "do less, cover more of the pain"
  alternative before committing to either of the above.

None of this is scheduled. Revisit if users report cancel latency as an
actual problem in practice, not just as a theoretical gap.
