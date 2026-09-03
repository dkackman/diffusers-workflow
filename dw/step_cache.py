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
"""

import logging

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
    return a == b


class StepCache:
    """Per-process cache of the last Result each step name produced."""

    def __init__(self):
        self._entries = {}  # step name -> {"step_data", "step_seed", "result"}

    def clear(self):
        self._entries.clear()

    def get(self, step_data, step_seed, hits_this_run):
        """Return the cached Result for this step if it's still valid, else None."""
        name = step_data["name"]
        entry = self._entries.get(name)
        if entry is None:
            return None
        if entry["step_seed"] != step_seed:
            return None
        if not deep_equal(entry["step_data"], step_data):
            return None

        upstream = referenced_result_names([step_data])
        if not all(self._is_hit(ref, hits_this_run) for ref in upstream):
            return None

        return entry["result"]

    def put(self, step_data, step_seed, result):
        self._entries[step_data["name"]] = {
            "step_data": step_data,
            "step_seed": step_seed,
            "result": result,
        }

    @staticmethod
    def _is_hit(ref, hits_this_run):
        # A reference resolves to a result whose name it equals or extends
        # with a property ('step.mask') - same rule Workflow.run's
        # release_unreferenced_results uses for the inverse question.
        return any(ref == n or ref.startswith(n + ".") for n in hits_this_run)


step_cache = StepCache()
