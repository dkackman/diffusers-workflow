"""Progress reporting and cooperative cancellation for workflow runs.

A RunContext travels ambiently (contextvars) through a run: Workflow.run
activates it, and Step/Pipeline reach it with get_context() rather than
threading a parameter through every action signature. Callers that never
pass a context get a no-op one, so the CLI path pays nothing.
"""

import logging
import threading
import contextvars

logger = logging.getLogger("dw")


class WorkflowCancelled(Exception):
    """Raised inside a run when its RunContext has been cancelled."""


class RunContext:
    """Carries the event sink and the cancellation flag for one workflow run.

    cancel() may be called from any thread (the worker's command watcher);
    everything else runs on the thread executing the workflow.
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

    def emit(self, event_type, **data):
        if self._on_event is None:
            return
        try:
            self._on_event({"event": event_type, **data})
        except Exception as e:
            # A broken sink must not kill a run that is otherwise fine
            logger.warning(f"Progress event sink failed on '{event_type}': {e}")

    def note_phase(self, phase):
        """Record the run's latest phase - called by emit_phase, read by
        cancel() to judge whether the current phase can be interrupted."""
        self._current_phase = phase

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
