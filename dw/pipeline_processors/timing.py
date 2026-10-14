"""Where a pipeline step's time went, measured in the worker (#867).

A `StepTimer` is opened around one step by the step loop
(`dw/workflow_run.py`) and fed marks from wherever they happen: the load
around `create_step_action`, the pipeline call in `Pipeline._call_pipeline`,
the first denoiser forward (a self-removing pre-forward hook) or the opening
of a modular pipeline's denoise bar (`progress.py`), each step callback or bar
advance, and the result save. Its `timing()` is the dict a step's manifest
entry carries as `timing`.

The clock is the worker's monotonic one, not the server's event timestamps:
the job history keeps only the last 200 events, so load and encode marks are
gone from a long job by the time anyone reads it. A field whose mark never
happened is null, never guessed.

The timer travels ambiently (a contextvar) like the RunContext does, so the
marks need no parameter threaded through Step and Pipeline. A composed child's
steps open their own timer and put the parent's back when they finish. With no
timer open every mark is a no-op.
"""

import contextlib
import contextvars
import statistics
import time

_active_timer = contextvars.ContextVar("dw_step_timer", default=None)


def _seconds(value):
    return None if value is None else round(value, 1)


def _sum(values):
    """The sum of a field over segments, or None when any segment lacks it."""
    if not values or any(value is None for value in values):
        return None
    return sum(values)


class _Segment:
    """The marks of one pipeline call: a chain segment, or one iteration."""

    def __init__(self, started):
        self.started = started
        self.denoise_started = None
        self.pass_ends = []
        self.ended = None

    def spans(self):
        """(encode, denoise, first_pass, decode, pass intervals) in seconds."""
        passes = self.pass_ends
        denoise_started = self.denoise_started
        encode = denoise = first_pass = decode = None
        if denoise_started is not None:
            encode = denoise_started - self.started
            if passes:
                denoise = passes[-1] - denoise_started
                first_pass = passes[0] - denoise_started
        elif not passes and self.ended is not None:
            # Neither a step callback nor a bar: the call is all there is to
            # time, and none of it can be split into encode or decode
            denoise = self.ended - self.started
        if passes and self.ended is not None:
            decode = self.ended - passes[-1]
        intervals = [later - earlier for earlier, later in zip(passes, passes[1:])]
        return encode, denoise, first_pass, decode, intervals


class StepTimer:
    """The marks of one pipeline step, from load start to save end."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._load_started = None
        self._load_ended = None
        self._load_cached = None
        self._save_started = None
        self._save_ended = None
        self._segments = []
        self._compiled = None

    def load_started(self):
        self._load_started = self._clock()

    def load_ended(self):
        self._load_ended = self._clock()

    def load_cached(self, cached):
        """A fresh load anywhere in the step's load makes it uncached; a
        pipeline served from the worker's cache marks it cached otherwise."""
        if self._load_cached is not False:
            self._load_cached = cached

    def call_started(self, compiled=None):
        self._segments.append(_Segment(self._clock()))
        if compiled is not None:
            self._compiled = bool(self._compiled) or compiled

    def denoise_started(self):
        """The first denoiser forward, or the denoise bar opening. Only the
        first one in a call counts: a second bar is a second loop of the same
        denoise, not a new start."""
        segment = self._current_segment()
        if segment is not None and segment.denoise_started is None:
            segment.denoise_started = self._clock()

    def pass_ended(self):
        segment = self._current_segment()
        if segment is not None:
            segment.pass_ends.append(self._clock())

    def call_ended(self):
        segment = self._current_segment()
        if segment is not None:
            segment.ended = self._clock()

    def save_started(self):
        self._save_started = self._clock()

    def save_ended(self):
        self._save_ended = self._clock()

    def _current_segment(self):
        if not self._segments or self._segments[-1].ended is not None:
            return None
        return self._segments[-1]

    def timing(self):
        """The step's `timing`: every `_s` in seconds to 0.1, a null for any
        mark that never happened. A chain (or any step that calls its
        pipeline more than once) sums its calls into one, with `pass_s` the
        median over every pass after each call's first."""
        spans = [segment.spans() for segment in self._segments]
        intervals = [
            interval for segment_spans in spans for interval in segment_spans[4]
        ]
        pass_counts = [len(segment.pass_ends) or None for segment in self._segments]
        return {
            "load_s": _seconds(self._span(self._load_started, self._load_ended)),
            "load_cached": self._load_cached,
            "encode_s": _seconds(_sum([s[0] for s in spans])),
            "denoise_s": _seconds(_sum([s[1] for s in spans])),
            "passes": _sum(pass_counts),
            "first_pass_s": _seconds(_sum([s[2] for s in spans])),
            "pass_s": _seconds(statistics.median(intervals)) if intervals else None,
            "compiled": self._compiled,
            "decode_s": _seconds(_sum([s[3] for s in spans])),
            "save_s": _seconds(self._span(self._save_started, self._save_ended)),
            "segments": len(self._segments),
        }

    @staticmethod
    def _span(started, ended):
        if started is None or ended is None:
            return None
        return ended - started


@contextlib.contextmanager
def timed_step(timer):
    """Make `timer` the one marks land on for the duration of a step; None
    opens nothing, so a step that is not timed hides an outer step's timer
    rather than feeding it."""
    token = _active_timer.set(timer)
    try:
        yield timer
    finally:
        _active_timer.reset(token)


@contextlib.contextmanager
def step_timer(step_data):
    """Open a StepTimer for a pipeline step, None for any other. Opening None
    also keeps a composed child's pipeline marks off an outer step's timer."""
    runs_a_pipeline = "pipeline" in step_data or "pipeline_reference" in step_data
    with timed_step(StepTimer() if runs_a_pipeline else None) as timer:
        yield timer


def current_timer():
    """The open step's timer, or None."""
    return _active_timer.get()


def mark(name, *args, **kwargs):
    """Call one of the open timer's marks; a no-op with no timer open."""
    timer = _active_timer.get()
    if timer is not None:
        getattr(timer, name)(*args, **kwargs)


def is_compiled(module):
    """Whether a denoiser runs through torch.compile, in whole or by its
    repeated blocks (`apply_compile` compiles in place, which leaves
    `_compiled_call_impl` set on each compiled module)."""
    if module is None or not hasattr(module, "modules"):
        return None
    return any(
        getattr(child, "_compiled_call_impl", None) is not None
        for child in module.modules()
    )


@contextlib.contextmanager
def denoise_start_hook(module):
    """Mark denoise start at the first forward of `module`, then remove the
    hook: one clock read, so a compiled denoiser sees it once and never
    again. Removed on the way out too, for a call that never reached it."""
    if module is None or not hasattr(module, "register_forward_pre_hook"):
        yield
        return
    handle = None

    def first_forward(_module, _inputs):
        mark("denoise_started")
        handle.remove()

    handle = module.register_forward_pre_hook(first_forward)
    try:
        yield
    finally:
        handle.remove()
