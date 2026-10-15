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

A step cache's skip counts (`timing.cache`) are counted here too, in the
section at the end.
"""

import contextlib
import contextvars
import logging
import statistics
import time

import torch

from .. import get_device_type
from ..events import emit_warning, get_context
from .config_objects import _MAG_CACHE_KEYS, _TAYLORSEER_CACHE_KEYS

logger = logging.getLogger("dw")

_active_timer = contextvars.ContextVar("dw_step_timer", default=None)

# A first pass this much slower than a steady one, on a loaded pipeline, is
# worth a warning (#924)
_SLOW_FIRST_PASS_EXTRA_S = 30.0


def _synced_clock():
    """The monotonic clock, read after the device has finished its queued work.

    CUDA and MPS run kernels asynchronously: without the wait, a pass or the
    call's end is stamped when the host queued the work, and the denoise time
    surfaces at the first host read, in decode (#896)."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    return time.monotonic()


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

    def __init__(self, clock=_synced_clock, step=None):
        self._clock = clock
        self.step = step
        self._cache = None
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
            self._warn_on_slow_first_pass(segment)

    def _warn_on_slow_first_pass(self, segment):
        """Say so once when a cached pipeline's first pass dwarfs the steady one.

        The pipeline is loaded, so the minutes are not a load. A later entry
        that runs the cached pipeline at a larger shape (more frames) pays a
        one-off warm-up on its first pass - re-allocation, offload buffers,
        kernels meeting a new shape - and without a word the consumer sees only
        a `phase_stall` (#924). Judged at the second pass interval, when a
        steady pass is known."""
        passes = segment.pass_ends
        if (
            len(passes) != 3
            or self._load_cached is not True
            or segment.denoise_started is None
        ):
            return
        first = passes[0] - segment.denoise_started
        steady = passes[2] - passes[1]
        if first - steady < _SLOW_FIRST_PASS_EXTRA_S or first < steady * 3:
            return
        emit_warning(
            f"First denoise pass took {first:.0f}s against {steady:.0f}s for "
            "the passes after it, with the pipeline already loaded: a one-off "
            "warm-up (typically a cached pipeline meeting a larger shape than "
            "it last ran, such as more frames). Later passes run at the "
            "steady rate.",
            kind="first_pass_slow",
            first_pass_s=round(first, 1),
            pass_s=round(steady, 1),
            step=self.step,
        )

    def call_ended(self):
        segment = self._current_segment()
        if segment is not None:
            segment.ended = self._clock()

    def cache_counted(self, stats):
        """One call's step-cache counts (report_cache_stats, below). A chain's calls sum
        into one, null if any call's counts are; the rest is the first's."""
        if self._cache is None:
            self._cache = dict(stats)
            return
        for key in ("forwards", "computed", "skipped"):
            self._cache[key] = _sum([self._cache[key], stats.get(key)])
        forwards = self._cache["forwards"]
        self._cache["hit_rate"] = (
            round(self._cache["skipped"] / forwards, 3) if forwards else None
        )

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
        timing = {
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
        # Only a step that set a `cache` has one
        if self._cache is not None:
            timing["cache"] = self._cache
        return timing

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
    timer = StepTimer(step=step_data.get("name")) if runs_a_pipeline else None
    with timed_step(timer) as timer:
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


# ------------------------------------------------------------ step-cache skip counts
#
# How many denoiser forwards a step cache skipped.
#
# diffusers counts nothing: a first_block or mag cache's leader hook keeps only
# the latest decision (`should_compute` on its shared state) and overwrites it on
# the next forward. So dw counts it itself, once per forward of the cache
# transformer, from a forward hook that `stateful_cache_context` registers for
# the pipeline call and removes when the call ends.
#
# Per forward, not per step: classifier-free guidance runs a "cond" and an
# "uncond" forward each step, and a diffusers pipeline opens a
# `cache_context("cond")`/`("uncond")` of its own around each one, inside dw's.
# Each context keeps its own state, so the decision is read inside the forward
# hook - still in the context the forward ran in - never after it returns.
#
# The state is private diffusers API. A diffusers whose hook names or state
# attributes moved gives null counts and one `cache_stats_unavailable` warning,
# never a failed run; `tests/test_cache_stats.py` pins the installed names so
# an upgrade that moves them fails CI first.
#
# taylorseer, faster and text_kv keep no decision state at all: they report
# their type and config with null counts, and no warning, since nothing is
# missing.


# diffusers' config class -> the workflow's `cache.type`
_CACHE_TYPES = {
    "FirstBlockCacheConfig": "first_block",
    "MagCacheConfig": "mag",
    "TaylorSeerCacheConfig": "taylorseer",
    "FasterCacheConfig": "faster",
    "TextKVCacheConfig": "text_kv",
}

# The cache types whose leader hook keeps a per-forward decision, and the name
# diffusers registers that hook under (`_FBC_LEADER_BLOCK_HOOK`,
# `_MAG_CACHE_LEADER_BLOCK_HOOK`)
LEADER_HOOKS = {
    "first_block": "fbc_leader_block_hook",
    "mag": "mag_cache_leader_block_hook",
}

# Config keys reported alongside the counts, beyond `threshold`
_CONFIG_KEYS = {
    "mag": tuple(key for key in _MAG_CACHE_KEYS if key != "threshold"),
    "taylorseer": _TAYLORSEER_CACHE_KEYS,
}


def _leader_state_manager(transformer, hook_name):
    """The StateManager the leader hook decides through, or None."""
    for module in transformer.modules():
        registry = getattr(module, "_diffusers_hook", None)
        get_hook = getattr(registry, "get_hook", None)
        if get_hook is None:
            continue
        hook = get_hook(hook_name)
        if hook is not None:
            return getattr(hook, "state_manager", None)
    return None


class CacheCounter:
    """Counts one pipeline call's computed and skipped forwards."""

    def __init__(self, transformer):
        config = getattr(transformer, "_cache_config", None)
        self.cache_type = _CACHE_TYPES.get(type(config).__name__, type(config).__name__)
        self.config = config
        self.counted = self.cache_type in LEADER_HOOKS
        self.forwards = self.computed = self.skipped = 0
        self.unavailable = None
        self._state_manager = None
        if self.counted:
            self._state_manager = _leader_state_manager(
                transformer, LEADER_HOOKS[self.cache_type]
            )
            if self._state_manager is None or not hasattr(
                self._state_manager, "get_state"
            ):
                self.unavailable = (
                    f"no '{LEADER_HOOKS[self.cache_type]}' hook with a state manager"
                )

    def after_forward(self, _module, _args, _output):
        """A torch forward hook: read the decision the leader made for the
        forward that just ran, in the cache context it ran in."""
        if not self.counted or self.unavailable:
            return
        try:
            should_compute = self._state_manager.get_state().should_compute
        except (AttributeError, ValueError) as error:
            # ValueError: diffusers' "No cache context is set"
            self.unavailable = f"{type(error).__name__}: {error}"
            return
        if not isinstance(should_compute, bool):
            self.unavailable = f"should_compute is {type(should_compute).__name__}"
            return
        self.forwards += 1
        if should_compute:
            self.computed += 1
        else:
            self.skipped += 1

    def stats(self):
        """`timing.cache`: type, threshold, counts and hit rate, plus the
        type's config keys. Counts are null for a type that keeps no
        decision, or when the decision could not be read."""
        threshold = getattr(self.config, "threshold", None)
        has_counts = self.counted and not self.unavailable
        forwards = self.forwards if has_counts else None
        stats = {
            "type": self.cache_type,
            "threshold": threshold if self.counted else None,
            "forwards": forwards,
            "computed": self.computed if has_counts else None,
            "skipped": self.skipped if has_counts else None,
            "hit_rate": round(self.skipped / forwards, 3) if forwards else None,
        }
        for key in _CONFIG_KEYS.get(self.cache_type, ()):
            if hasattr(self.config, key):
                stats[key] = getattr(self.config, key)
        return stats


def counting_forwards(transformer):
    """Register a CacheCounter's forward hook on `transformer`.

    Returns:
        (counter, remove) - call `remove` once the counts are read
    """
    counter = CacheCounter(transformer)
    if not counter.counted or counter.unavailable:
        return counter, lambda: None
    handle = transformer.register_forward_hook(counter.after_forward)
    return counter, handle.remove


def report_cache_stats(counter, segment=None):
    """Emit the call's `cache_stats` event, and its warning when the counts
    could not be read, and hand the stats to the step's timer. The event
    names the step the open timer is for, and a chain's `segment`."""
    stats = counter.stats()
    timer = current_timer()
    where = {"step": timer.step if timer is not None else None}
    if segment is not None:
        where["segment"] = segment
    if counter.unavailable:
        emit_warning(
            f"Step cache '{counter.cache_type}' ran, but its skip counts are "
            f"unavailable from this diffusers ({counter.unavailable}); "
            "timing.cache carries null counts",
            kind="cache_stats_unavailable",
            cache_type=counter.cache_type,
        )
    logger.info(f"Step cache stats: {stats}")
    get_context().emit("cache_stats", **where, **stats)
    mark("cache_counted", stats)
    return stats
