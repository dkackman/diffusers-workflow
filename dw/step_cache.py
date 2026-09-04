"""Skip re-executing a step whose resolved definition, seed, and upstream
results are all unchanged since the last run in this process.

Reusing a loaded pipeline (Workflow.run's previous_pipelines) is only half
of what makes REPL iteration fast - the other half is not re-running a
step's forward pass at all when nothing feeding it changed, the way
Mellon's NodeBase skips a node whose resolved params match its last call
(deep value equality, not just identity - a step's arguments are plain
dicts/lists/scalars after variable substitution, not hashable).

A step is safe to skip only if:
  1. its own resolved definition (step_data) matches last run's, AND
  2. its seed matches last run's - step_data does NOT carry the seed
     (Workflow.run resolves it separately, and draws a fresh random one
     per run when the workflow sets none), so seed must be compared
     explicitly or two differently-seeded runs would wrongly look identical
  3. every previous_result: it reads was ITSELF served from cache this run
     - otherwise a change upstream leaves this step's cached output stale
  4. the effective output directory matches last run's - like seed, this
     is out-of-band (not part of step_data), and a cache hit reuses the
     entry's saved_files/manifest paths verbatim, so a mismatch must force
     a miss or a changed output dir would silently keep pointing at the
     old directory's files
  5. every file the cached result names still exists - a hit republishes
     those paths into the manifest and job history, so a file deleted
     since (gallery delete button, or by hand) must force a re-run

The cache is per-process and bounded (DEFAULT_MAX_ENTRIES, LRU): it holds
realized media, so unbounded growth would work against the OOM avoidance
release_unreferenced_results exists for.
"""

import logging
import os
from collections import OrderedDict

from .workflow import referenced_result_names

logger = logging.getLogger("dw")


def deep_equal(a, b):
    """Value equality across the JSON-ish types a resolved step definition holds."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    try:
        # realize_args runs before the step loop, so a resolved argument can
        # hold a numpy array or a tensor, whose == yields an array rather
        # than a bool (and an exotic object's == can raise outright). A value
        # that cannot answer "are these equal" cleanly is treated as unequal:
        # a cache miss just re-runs the step, where a raised exception would
        # abort the whole run
        return bool(a == b)
    except (ValueError, TypeError, RuntimeError):
        return False


class StepCache:
    """Per-process cache of the last Result each step name produced.

    Bounded: entries hold realized media, so an unbounded cache would work
    directly against release_unreferenced_results' OOM avoidance. The
    least-recently-used entry is evicted once the cap is reached.
    """

    DEFAULT_MAX_ENTRIES = 50

    def __init__(self, max_entries=None):
        # step name -> {"step_data", "step_seed", "result", "output_dir"},
        # ordered least- to most-recently-used
        self._entries = OrderedDict()
        self.max_entries = (
            self.DEFAULT_MAX_ENTRIES if max_entries is None else max_entries
        )

    def clear(self):
        self._entries.clear()

    def get(self, step_data, step_seed, hits_this_run, output_dir=None):
        """Return the cached Result for this step if it's still valid, else None."""
        name = step_data["name"]
        entry = self._entries.get(name)
        if entry is None:
            return None
        if entry["step_seed"] != step_seed:
            return None
        if entry["output_dir"] != output_dir:
            return None
        if not deep_equal(entry["step_data"], step_data):
            return None

        upstream = referenced_result_names([step_data])
        if not all(self._is_hit(ref, hits_this_run) for ref in upstream):
            return None

        # A hit reports the entry's saved_files verbatim into the manifest
        # and job history - if the user deleted one of them (gallery delete
        # button, or by hand), the entry is stale and the step must re-run
        if not self._saved_files_exist(entry["result"]):
            logger.debug(
                f"Cached result for step '{name}' names a file that no longer "
                "exists - treating as a miss"
            )
            self._entries.pop(name, None)
            return None

        self._entries.move_to_end(name)
        return entry["result"]

    def put(self, step_data, step_seed, result, output_dir=None):
        name = step_data["name"]
        self._entries[name] = {
            "step_data": step_data,
            "step_seed": step_seed,
            "result": result,
            "output_dir": output_dir,
        }
        self._entries.move_to_end(name)
        while len(self._entries) > self.max_entries:
            evicted, _ = self._entries.popitem(last=False)
            logger.debug(f"Step cache full - evicting least recently used '{evicted}'")

    @staticmethod
    def _saved_files_exist(result):
        return all(
            os.path.exists(path) for path in getattr(result, "saved_files", None) or []
        )

    @staticmethod
    def _is_hit(ref, hits_this_run):
        # A reference resolves to a result whose name it equals or extends
        # with a property ('step.mask') - same rule Workflow.run's
        # release_unreferenced_results uses for the inverse question.
        return any(ref == n or ref.startswith(n + ".") for n in hits_this_run)


step_cache = StepCache()
