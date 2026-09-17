"""Progress reporting and cooperative cancellation for workflow runs.

A RunContext travels ambiently (contextvars) through a run: Workflow.run
activates it, and Step/Pipeline reach it with get_context() rather than
threading a parameter through every action signature. Callers that never
pass a context get a no-op one, so the CLI path pays nothing.
"""

import logging
import threading
import time
import contextvars

logger = logging.getLogger("dw")

# How long a phase may go without any event before the watchdog speaks up,
# and (since emitting a stall event itself counts as an event - see
# RunContext._watchdog_loop) the cadence it repeats on while the stall
# continues. 30s is well past the 7-10s lead-in a healthy step-callback
# pipeline shows before its first `pipeline_step` (see
# docs/proposals/step-callback-lead-in-instrumentation.md) but short enough
# that a genuine stall is visible long before a human would give up on it.
# The report goes to the event log only - the job's persisted `warnings`
# list filters `kind == "phase_stall"` (dw/server/jobs.py), because a stall
# that resolved is not a warning about the result.
PHASE_STALL_THRESHOLD_SECONDS = 30.0
# How often the watchdog thread wakes to check - independent of the
# threshold above, just fine-grained enough that the reported
# seconds_since_phase_start doesn't overshoot by much.
PHASE_STALL_CHECK_INTERVAL_SECONDS = 5.0


class WorkflowCancelled(Exception):
    """Raised inside a run when its RunContext has been cancelled."""


class RunContext:
    """Carries the event sink and the cancellation flag for one workflow run.

    cancel() may be called from any thread (the worker's command watcher);
    everything else runs on the thread executing the workflow. The
    phase-stall watchdog (_watchdog_loop) is the same story - it reads
    _current_phase/_last_event_at from its own background thread with no
    lock, which is fine for simple attribute reads/writes under the GIL.
    """

    def __init__(self, on_event=None):
        self._on_event = on_event
        self._cancel = threading.Event()
        # The most recent phase this run reported (see emit_phase/PHASES) -
        # cancel() reads it to tell a client whether the cancel takes effect
        # right away or has to wait out an in-flight, non-interruptible phase
        self._current_phase = None
        # Pipeline cache keys this run resolved - the worker evicts entries a
        # run no longer touches, so an edited workflow drops stale models
        self.touched_pipelines = set()
        # Watchdog state. monotonic, not wall clock, since a stall is
        # measured in elapsed time, not affected by clock adjustments.
        self._last_event_at = time.monotonic()
        self._phase_started_at = time.monotonic()
        # Reference-counted: a sub-workflow runs inside its parent's
        # RunContext (Workflow.run reuses the ambient one), so the watchdog
        # starts on the outermost run() and stops on the outermost's exit,
        # not on every nested one
        self._run_depth = 0
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()

    def emit(self, event_type, **data):
        self._last_event_at = time.monotonic()
        if self._on_event is None:
            return
        try:
            self._on_event({"event": event_type, **data})
        except Exception as e:
            # A broken sink must not kill a run that is otherwise fine
            logger.warning(f"Progress event sink failed on '{event_type}': {e}")

    def note_phase(self, phase):
        """Record the run's latest phase - called by emit_phase, read by
        cancel() to judge whether the current phase can be interrupted, and
        by the watchdog to time how long the current phase has run."""
        self._current_phase = phase
        self._phase_started_at = time.monotonic()

    def enter_run(self):
        """Called around Workflow.run - starts the watchdog on the
        outermost call, a no-op on a nested (sub-workflow) one."""
        self._run_depth += 1
        if self._run_depth == 1:
            self._watchdog_stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="dw-phase-stall"
            )
            self._watchdog_thread.start()

    def exit_run(self):
        """Matching exit for enter_run - stops the watchdog once the
        outermost run() has left, whether it succeeded or raised."""
        self._run_depth -= 1
        if self._run_depth == 0 and self._watchdog_thread is not None:
            self._watchdog_stop.set()
            self._watchdog_thread.join(timeout=PHASE_STALL_CHECK_INTERVAL_SECONDS * 2)
            self._watchdog_thread = None

    def _watchdog_loop(self):
        """Generic, pipeline-agnostic: this only ever looks at 'how long
        since any event' and 'what phase are we in', never at what a
        particular pipeline does inside a phase (see
        docs/proposals/step-callback-lead-in-instrumentation.md, Option A).
        A stall report is itself an event, so it naturally repeats on
        PHASE_STALL_THRESHOLD_SECONDS while the silence continues and stops
        the moment a real progress event arrives.
        """
        while not self._watchdog_stop.wait(PHASE_STALL_CHECK_INTERVAL_SECONDS):
            phase = self._current_phase
            if phase is None:
                continue
            now = time.monotonic()
            if now - self._last_event_at < PHASE_STALL_THRESHOLD_SECONDS:
                continue
            seconds_since_phase_start = round(now - self._phase_started_at, 1)
            message = (
                f"still in phase '{phase}', {seconds_since_phase_start:.1f}s "
                "since it started with no progress event"
            )
            logger.warning(message)
            self.emit(
                "warning",
                message=message,
                kind="phase_stall",
                phase=phase,
                seconds_since_phase_start=seconds_since_phase_start,
            )

    def cancel(self):
        self._cancel.set()
        if self._current_phase in NON_INTERRUPTIBLE_PHASES:
            # A model load or a task step has no checkpoint to catch this
            # flag until it finishes - say so, or the client sees the cancel
            # request go silent for however long that takes
            self.emit("cancel_pending", phase=self._current_phase)

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def check_cancelled(self):
        if self._cancel.is_set():
            raise WorkflowCancelled("Workflow run was cancelled")

    def touch_pipeline(self, cache_key):
        self.touched_pipelines.add(cache_key)


_active_context = contextvars.ContextVar("dw_run_context", default=None)


def get_context():
    """The active run's context, or a no-op one outside any run."""
    context = _active_context.get()
    return context if context is not None else RunContext()


def current_context():
    """The active run's context, or None outside any run."""
    return _active_context.get()


def activate_context(context):
    """Make a context the active one; returns a token for deactivate_context."""
    return _active_context.set(context)


def deactivate_context(token):
    _active_context.reset(token)


# The coarse states a run passes through. Small on purpose: a phase says what
# the run is waiting on, not what any one library is doing internally
PHASES = ("loading", "cached", "generating", "decoding", "saving", "task")

# Phases with no checkpoint of their own: nothing inside a from_pretrained()
# call or a task handler consults the cancel flag, so a cancel requested
# during one of these can only take effect once the phase finishes on its own
NON_INTERRUPTIBLE_PHASES = ("loading", "task")


def emit_warning(message, **data):
    """Report something the run's result carries but its status will not.

    A warning a step discovers at run time - shots being cut together 10 dB
    apart, a video about to be written at a frame rate nothing chose - is
    only useful where whoever asked for the run can read it. The server's
    own log is not that place: a consumer over the API or MCP sees the event
    stream and the job's `warnings` list and nothing else, so a diagnostic
    that only reaches the log does not exist out there (#82).

    Logged as well as emitted, because the CLI and the REPL have no event
    sink and the log is the whole of their surface.
    """
    logger.warning(message)
    get_context().emit("warning", message=message, **data)


def emit_phase(phase, detail=None):
    """Report a coarse phase change on the active run.

    A step spends most of its wall clock outside the denoise loop - pulling
    weights, decoding latents, encoding video - and a step counter says
    nothing about any of that. These are the rest of the story. They are rare
    enough (a handful per step) to carry a free-text detail alongside, which
    is what makes 'loading' readable as 'which model'.
    """
    context = get_context()
    context.note_phase(phase)
    context.emit("phase", phase=phase, detail=detail)


def emit_log(message, **data):
    """Narrate one step of a long, otherwise silent stretch of a run.

    A `log` event rather than a phase: `PHASES` is a closed set a consumer
    switches on, and "which file is being written" is a detail inside one of
    them, not a new state. The modular block lead-in (#95) is the same shape
    at the other end of a step.

    Logged as well as emitted, because the CLI and the REPL have no event
    sink and the log is the whole of their surface.
    """
    logger.info(message)
    get_context().emit("log", message=message, **data)
