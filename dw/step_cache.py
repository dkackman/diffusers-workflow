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
     AND the upstream entry now in the cache is the same generation this
     entry was computed from - "hit this run" alone is not enough, because
     a run cancelled between an upstream's put and this step's leaves this
     entry describing an upstream that has since been recomputed
  4. the effective output directory matches last run's - like seed, this
     is out-of-band (not part of step_data), and a cache hit reuses the
     entry's saved_files/manifest paths verbatim, so a mismatch must force
     a miss or a changed output dir would silently keep pointing at the
     old directory's files
  5. every file the cached result names still exists - a hit republishes
     those paths into the manifest and job history, so a file deleted
     since (gallery delete button, or by hand) must force a re-run
  6. the entry retained the step's Result if this run needs one - an entry
     stored for a step nothing downstream read holds only its saved_files

Entries are keyed by (workflow_id, step_name), not by step name alone:
saved files are named "{workflow_id}-{step_name}.{index}", so a bare-name
key would let a different workflow (or the same one after an id rename, or
a sub-workflow sharing a name with its parent) hit and republish the other
workflow's file paths while writing none of its own.

The cache is per-process and bounded (DEFAULT_MAX_ENTRIES, LRU): it holds
realized media, so unbounded growth would work against the OOM avoidance
release_unreferenced_results exists for.
"""

import copy
import itertools
import logging
import os
from collections import OrderedDict

logger = logging.getLogger("dw")

# Monotonic across the process: every put stamps its entry, and a downstream
# entry records the stamp of each upstream it was computed from. Never reset
# (clear() included) - a reused number would make a stale entry look fresh.
_generations = itertools.count(1)


def reference_resolves_to(reference, name):
    """Whether a previous_result reference resolves to the result `name`.

    A reference either names a result outright or extends it with a property
    ('segment.mask'). The step cache, release_unreferenced_results and
    get_previous_results all ask some form of this question.
    """
    return reference == name or reference.startswith(name + ".")


def referenced_result_names(steps):
    """Every previous_result reference the given steps make, as full names.

    Scans nested dicts and lists, so references inside pipeline arguments,
    task arguments and sub-workflow argument maps are all found - including a
    constructed object's 'from_previous_result', which names a step without
    the 'previous_result:' prefix.
    """
    prefix = "previous_result:"
    names = set()

    def scan(value):
        if isinstance(value, str) and value.startswith(prefix):
            names.add(value[len(prefix) :])
        elif isinstance(value, dict):
            reference = value.get("from_previous_result")
            if isinstance(reference, str):
                names.add(reference)
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    for step in steps:
        scan(step)
    return names


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
        # (workflow_id, step_name) -> {"step_data", "step_seed", "result",
        # "output_dir", "generation", "upstream_generations", "retained"},
        # ordered least- to most-recently-used
        self._entries = OrderedDict()
        self.max_entries = (
            self.DEFAULT_MAX_ENTRIES if max_entries is None else max_entries
        )

    def clear(self):
        # The generation counter deliberately survives: it only has to be
        # monotonic, and restarting it could make a stale reference match
        self._entries.clear()

    def get(
        self, workflow_id, step_data, step_seed, hits_this_run, output_dir, needs_result
    ):
        """Return the cached Result for this step if it's still valid, else None.

        `needs_result` says whether this run reads the step's Result (a later
        step references it, or it is the workflow's return value); an entry
        stored without one cannot serve such a run.
        """
        name = step_data["name"]
        key = (workflow_id, name)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry["step_seed"] != step_seed:
            return None
        if entry["output_dir"] != output_dir:
            return None
        if needs_result and not entry["retained"]:
            return None
        if not deep_equal(entry["step_data"], step_data):
            return None

        upstream = referenced_result_names([step_data])
        if not all(self._is_hit(ref, hits_this_run) for ref in upstream):
            return None
        # Hitting this run is not the same as being the run this entry was
        # computed from - compare the upstream generations too
        for upstream_name, generation in entry["upstream_generations"].items():
            current = self._entries.get((workflow_id, upstream_name))
            if current is None or current["generation"] != generation:
                logger.debug(
                    f"Cached result for step '{name}' was computed from an "
                    f"older '{upstream_name}' - treating as a miss"
                )
                return None

        # A hit reports the entry's saved_files verbatim into the manifest
        # and job history - if the user deleted one of them (gallery delete
        # button, or by hand), the entry is stale and the step must re-run
        if not self._saved_files_exist(entry["result"]):
            logger.debug(
                f"Cached result for step '{name}' names a file that no longer "
                "exists - treating as a miss"
            )
            self._entries.pop(key, None)
            return None

        self._entries.move_to_end(key)
        return entry["result"]

    def put(self, workflow_id, step_data, step_seed, result, output_dir, retain_result):
        """Record this step's outcome.

        `retain_result` says whether anything reads the Result itself. When
        False only a stripped copy is kept - saved_files and definition, but
        no result_list - so decoded frames and latent tensors are not pinned
        for the life of the cache, which is exactly what
        release_unreferenced_results drops them to avoid.

        A Result whose result_list holds an artifact save() already spilled
        and cleaned up (a chain step's save_segments - see Result.retainable)
        is downgraded to the same stripped storage even when retain_result is
        True: its files are gone already by the time put() runs (save()
        happens before put() in the step loop), so keeping the result_list
        would let a later cache hit fail opening files that no longer exist.
        """
        retain_result = retain_result and getattr(result, "retainable", True)
        name = step_data["name"]
        key = (workflow_id, name)
        self._entries[key] = {
            "step_data": step_data,
            "step_seed": step_seed,
            "result": result if retain_result else self._strip_result(result),
            "retained": retain_result,
            "output_dir": output_dir,
            "generation": next(_generations),
            "upstream_generations": self._upstream_generations(workflow_id, step_data),
        }
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            (evicted_workflow, evicted_step), _ = self._entries.popitem(last=False)
            logger.debug(
                "Step cache full - evicting least recently used "
                f"'{evicted_workflow}/{evicted_step}'"
            )

    def _upstream_generations(self, workflow_id, step_data):
        """The generation of every cached step this step's references read."""
        generations = {}
        for reference in referenced_result_names([step_data]):
            for (entry_workflow, entry_step), entry in self._entries.items():
                if entry_workflow == workflow_id and reference_resolves_to(
                    reference, entry_step
                ):
                    generations[entry_step] = entry["generation"]
        return generations

    @staticmethod
    def _strip_result(result):
        """A shallow copy of the Result with its realized media dropped."""
        stripped = copy.copy(result)
        stripped.result_list = []
        return stripped

    @staticmethod
    def _saved_files_exist(result):
        # Read straight through: a result-like object with no saved_files is
        # a programming error, not an entry that quietly skips the file check
        return all(os.path.exists(path) for path in result.saved_files)

    @staticmethod
    def _is_hit(ref, hits_this_run):
        return any(reference_resolves_to(ref, n) for n in hits_this_run)


step_cache = StepCache()
