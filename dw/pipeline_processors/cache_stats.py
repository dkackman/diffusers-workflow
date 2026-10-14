"""How many denoiser forwards a step cache skipped (#867).

diffusers counts nothing: a first_block or mag cache's leader hook keeps only
the latest decision (`should_compute` on its shared state) and overwrites it on
the next forward. So dw counts it itself, once per forward of the cache
transformer, from a forward hook that `stateful_cache_context` registers for
the pipeline call and removes when the call ends.

Per forward, not per step: classifier-free guidance runs a "cond" and an
"uncond" forward each step, and a diffusers pipeline opens a
`cache_context("cond")`/`("uncond")` of its own around each one, inside dw's.
Each context keeps its own state, so the decision is read inside the forward
hook - still in the context the forward ran in - never after it returns.

The state is private diffusers API. A diffusers whose hook names or state
attributes moved gives null counts and one `cache_stats_unavailable` warning,
never a failed run; `tests/test_cache_stats.py` pins the installed names so
an upgrade that moves them fails CI first.

taylorseer, faster and text_kv keep no decision state at all: they report
their type and config with null counts, and no warning, since nothing is
missing.
"""

import logging

from ..events import emit_warning, get_context
from .config_objects import _MAG_CACHE_KEYS, _TAYLORSEER_CACHE_KEYS
from .timing import current_timer, mark

logger = logging.getLogger("dw")

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
