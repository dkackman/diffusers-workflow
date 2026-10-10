"""Skip re-executing a step whose resolved definition, seed, and upstream
results are all unchanged since the last run in this process.

Reusing a loaded pipeline (Workflow.run's previous_pipelines) is only half
of what makes iteration on the server's persistent worker fast - the other half is not re-running a
step's forward pass at all when nothing feeding it changed, the way
Mellon's NodeBase skips a node whose resolved params match its last call
(deep value equality, not just identity - a step's arguments are plain
dicts/lists/scalars after variable substitution, not hashable).

A step is safe to skip only if:
  1. its own resolved definition (step_data) matches last run's, AND - step_data
     is the whole step, `result` block included, so changing a step's
     `subfolder` misses the cache
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

The cache is per-process and bounded two ways, LRU either way: by entry
count (DEFAULT_MAX_ENTRIES), sized so a maximal for_each run (32 entries
over two groups plus fixed steps, ~70 members) never evicts its own earlier
members before it ends - a smaller cap would turn a long list-driven run
into one that thrashes its own cache - and by the approximate byte size of
the retained media itself (DEFAULT_MAX_RETAINED_BYTES). A for_each group
whose members are each a full decoded video (dialogue-short's `shot`, gathered
by `episode`) is retained in full for every member - correctly, since the
final step genuinely reads all of them - but nothing ever released that once
the run finished, so a handful of members held several GB resident
indefinitely (#368). The byte cap does not know which entries a future run
would most want back; it just keeps the newest-used bytes under budget the
same way the entry cap keeps the newest-used count under one, oldest first.
"""

import copy
import dataclasses
import hashlib
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from . import references

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
    names = set()
    for step in steps:
        for _, name, _ in references.iter_previous_result_references(
            step, descend_into_from=True
        ):
            names.add(name)
    return names


def component_names(pipeline_definition, key):
    """The component names one of a pipeline definition's sharing lists holds.

    The lists were only ever read off the pipeline itself, while the schema and
    the guide put them in its configuration - a workflow written to the docs
    shared nothing and said nothing about it. Both places are read now.

    Args:
        pipeline_definition: A pipeline's definition dict (`step["pipeline"]`)
        key: 'shared_components' or 'reused_components'

    Returns:
        List of component names
    """
    configuration = pipeline_definition.get("configuration", {})
    return list(pipeline_definition.get(key, [])) + list(configuration.get(key, []))


def pipeline_cache_key(pipeline_definition):
    """Stable identity for a loaded pipeline.

    Hashes everything that shapes loading - configuration, components,
    quantization, loras - and excludes what varies per call (arguments, seed,
    chain), so a cache hit means "this exact model stack is already loaded".
    Keying the cache by identity instead of step name means two workflows
    whose steps happen to share a name can no longer collide, and a rerun of
    an edited workflow keeps every pipeline whose definition did not change.

    Computed after variable substitution but the excluded keys keep realized
    per-run values (images, generators) out of the hash; realized types and
    dtypes stringify stably via default=str.
    """
    load_definition = {
        k: v
        for k, v in pipeline_definition.items()
        if k not in ("arguments", "seed", "chain")
    }
    serialized = json.dumps(load_definition, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def step_pipeline_keys(steps):
    """Step name -> effective pipeline key for every pipeline step, taken
    before any step runs.

    A step that reuses nothing keys by its pipeline_cache_key exactly. A
    step that reuses components keys by the sorted, de-duplicated set of
    pipeline_cache_keys over its reuse closure: itself and, transitively,
    every step `component_source` resolves its reused names to, and theirs.
    Only a fresh load resolves reused components, so the pipeline a reusing
    step loaded holds its sources' components as they were then; its own
    definition never names the source model, so a key hashed from that
    definition alone stayed the same when a source changed, and the
    pipeline cache handed back a pipeline holding the old component while
    the step cache served a step referencing it a stale hit. With the
    closure, a change to any definition up a borrow chain - the origin's, or
    an intermediate's that passes a component on - changes every key below
    it. It is a set rather than a chain so that identical steps passing one
    component along (for_each members that each reuse and share `vae`)
    close over the same definitions and keep sharing one loaded pipeline.

    Pipeline.load edits the definition it is handed (placement resolves a
    group_offload block in place, a LoRA entry is consumed), so a key hashed
    after a step loaded is not the key the same step hashes to when it never
    loaded - a deferred cache hit, or the cache_hits probe. A key that has to
    agree across those cases is read from this table, not re-hashed.
    """
    keys = {}
    # Step name -> the own pipeline_cache_keys of its reuse closure
    closures = {}
    for index, step_data in enumerate(steps):
        if "pipeline" not in step_data:
            continue
        pipeline_definition = step_data["pipeline"]
        own_key = pipeline_cache_key(pipeline_definition)
        closure = {own_key}
        reused = component_names(pipeline_definition, "reused_components")
        for name in reused:
            source = component_source(steps[:index], name)
            if source is not None:
                closure |= closures[source["name"]]
        closures[step_data["name"]] = closure
        if reused:
            serialized = json.dumps(sorted(closure))
            keys[step_data["name"]] = hashlib.sha256(serialized.encode()).hexdigest()
        else:
            keys[step_data["name"]] = own_key
    return keys


def component_source(earlier_steps, name):
    """The latest of `earlier_steps` whose pipeline shares component `name`
    - the step a `reused_components` entry resolves to - or None."""
    for earlier in reversed(earlier_steps):
        earlier_pipeline = earlier.get("pipeline")
        if earlier_pipeline and name in component_names(
            earlier_pipeline, "shared_components"
        ):
            return earlier
    return None


def borrowed_pipeline_keys(steps, index, pipeline_keys):
    """The earlier steps step `index` borrows its pipeline from, by name, to
    that step's effective pipeline key.

    A step that borrows another step's pipeline (`pipeline_reference`) or
    components (`reused_components`, matched to an earlier step's
    `shared_components`) has an unchanged step_data when only the source
    step's model changes - its own definition never names the model. Folding
    the source steps' keys into the lookup snapshot makes a source model
    change a miss here too, rather than a stale hit that republishes a
    now-wrong pipeline. The keys are effective keys (step_pipeline_keys), so
    a change further up a borrow chain than the direct source is a miss too.

    The keys come from `pipeline_keys` (step_pipeline_keys), computed from
    the definition before any step ran, never from `steps[:index]` as they
    stand now: a load edits its own definition in place, and a source step
    that loaded before this lookup (a cold run) and one that did not (a
    deferred cache hit) would otherwise hash to different keys, and the
    borrowing step would never match its own entry. Which steps are
    borrowed is still resolved statically from `steps[:index]`, the same
    list the run loop iterates.
    """
    step = steps[index]
    earlier_steps = steps[:index]
    borrowed = {}

    pipeline_reference = step.get("pipeline_reference")
    if isinstance(pipeline_reference, dict):
        reference_name = pipeline_reference.get("reference_name")
        if reference_name:
            for earlier in earlier_steps:
                if earlier.get("name") == reference_name and "pipeline" in earlier:
                    borrowed[reference_name] = pipeline_keys[reference_name]
                    break

    pipeline_definition = step.get("pipeline")
    if isinstance(pipeline_definition, dict):
        for name in component_names(pipeline_definition, "reused_components"):
            source = component_source(earlier_steps, name)
            if source is not None:
                borrowed[source["name"]] = pipeline_keys[source["name"]]

    return borrowed


# Tasks that reset a result's level before anything downstream ships it -
# a result only these read is not itself a headroom concern (dw/audio_qc.py,
# warn_without_headroom)
NORMALIZING_COMMANDS = {"normalize_audio", "match_levels"}


def resets_level(task):
    """Whether a task definition resets the level of the audio it reads: a
    normalizing command, or a join whose `match_levels` is set (the join
    levels its inputs and holds the result under full scale - #671)."""
    if task.get("command") in NORMALIZING_COMMANDS:
        return True
    arguments = task.get("arguments")
    return isinstance(arguments, dict) and bool(arguments.get("match_levels"))


def normalized_downstream(steps, name):
    """Whether a later normalize_audio/match_levels step, or a join with
    `match_levels` set, consumes result `name`.

    write_song's raw Music 3 mp3 (templates/minimax/music-video) always lands
    at or over full scale and is always normalized before the deliverable
    mux - the pre-ship level is the template's documented, designed-in input
    condition, not a mistake, so a headroom warning on that intermediate
    save trains the reader to ignore job.warnings (#286). Scoped to the two
    tasks that actually reset level, not to any downstream consumer: a step
    that only reads the raw result (a conditioning slice, say) does not
    change what its own written file will sound like.
    """
    for step in steps:
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict) or not resets_level(task):
            continue
        if any(
            reference_resolves_to(ref, name) for ref in referenced_result_names([step])
        ):
            return True
    return False


def audio_replaced_downstream(steps, name):
    """Whether a later `pair_audio` takes result `name` as its `video`.

    pair_audio lays a new track over the picture and drops the one the video
    carried, so a warning about that track's level (the spread between joined
    shots, #800) describes audio that never ships. Only the `video` argument
    counts: a step that takes `name` as its `audio` keeps it.
    """
    for step in steps:
        task = step.get("task") if isinstance(step, dict) else None
        if not isinstance(task, dict) or task.get("command") != "pair_audio":
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict) or "video" not in arguments:
            continue
        if any(
            reference_resolves_to(ref, name)
            for ref in referenced_result_names([{"video": arguments["video"]}])
        ):
            return True
    return False


def copy_containers(value):
    """A copy of `value` whose dicts, lists and tuples are its own and whose
    leaves are shared.

    For a structure that is about to be edited by replacing entries - a
    key assigned or popped, an item replaced - but never by changing a leaf
    in place: the edits stay in the copy, while a leaf that is realized
    media (an image, a frame list's tensor, a loaded type) is the object the
    original holds rather than a duplicate of it, and a leaf that cannot be
    deep-copied at all is no reason to fail. A tuple is copied as a tuple.

    Only the exact built-in types are containers here: a subclass - an
    OrderedDict, a namedtuple, a torch.Size - is a leaf, since rebuilding it
    as its base type would change what it is.
    A dict or list subclass is shared as a leaf too, so a caller copying a
    definition relies on its containers being plain dicts and lists, which
    parsed JSON is.
    """
    kind = type(value)
    if kind is dict:
        return {key: copy_containers(item) for key, item in value.items()}
    if kind is list:
        return [copy_containers(item) for item in value]
    if kind is tuple:
        return tuple(copy_containers(item) for item in value)
    return value


def deep_equal(a, b):
    """Value equality across the JSON-ish types a resolved step definition holds.

    realize_args runs before the step loop (and before for_each expansion's
    per-member `from_file` construction), so a resolved argument can be a
    diffusers reference dataclass wrapping in-memory media - a
    MiniMaxH3ImageReference, an LTX2ReferenceCondition - built fresh by
    `from_file()` on every run from the same source file. Two such instances
    hold identical media but are never `==`: PIL.Image has no value equality
    (falls back to identity), and comparing torch.Tensor/np.ndarray fields
    with `==` yields an array rather than a bool, which the dataclass's
    generated `__eq__` cannot resolve to True/False and which the fallback
    below (correctly) treats as "cannot tell, so unequal". Left unhandled,
    that made every for_each member carrying an image/audio/video reference
    an unconditional cache miss - identical inputs included (#253). These
    checks give those types real value equality before the generic fallback.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, Image.Image):
        return a.mode == b.mode and a.size == b.size and a.tobytes() == b.tobytes()
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    if isinstance(a, torch.Tensor):
        return (
            a.shape == b.shape
            and a.dtype == b.dtype
            and bool(torch.equal(a.cpu(), b.cpu()))
        )
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return all(
            deep_equal(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
        )
    try:
        # An exotic object's == can still raise, or (for a type not caught
        # above) return something other than a bool. A value that cannot
        # answer "are these equal" cleanly is treated as unequal: a cache
        # miss just re-runs the step, where a raised exception would abort
        # the whole run
        return bool(a == b)
    except (ValueError, TypeError, RuntimeError):
        return False


def _approx_bytes(value, seen):
    """Approximate resident size of a retained result item, in bytes.

    Walks the same shapes a pipeline output actually takes - a dict/dataclass
    of arrays wrapping frames and audio, nested lists of per-frame images -
    rather than every Python object, so an unrecognized type (a bare string,
    a plain number) costs nothing rather than raising. `seen` is shared
    across one Result's whole result_list so an object two artifacts both
    reference (get_artifact_list's fitted-in-place audio, say) is not
    double-counted.
    """
    key = id(value)
    if key in seen:
        return 0
    seen.add(key)
    if isinstance(value, torch.Tensor):
        return value.element_size() * value.nelement()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Image.Image):
        bands = len(value.getbands()) or 1
        return value.width * value.height * bands
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        return sum(_approx_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_approx_bytes(v, seen) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            _approx_bytes(getattr(value, f.name), seen)
            for f in dataclasses.fields(value)
        )
    return 0


def _result_bytes(result):
    seen = set()
    return sum(_approx_bytes(item, seen) for item in result.result_list)


class StepCache:
    """Per-process cache of the last Result produced for each
    (workflow_id, step_name).

    Bounded: entries hold realized media, so an unbounded cache would work
    directly against release_unreferenced_results' OOM avoidance. The
    least-recently-used entry is evicted once the cap is reached. The
    default is sized so a maximal for_each run - 32 entries over two groups
    plus fixed steps, ~70 members - fits without evicting its own earlier
    members.
    """

    DEFAULT_MAX_ENTRIES = 128
    # 4 GiB: enough for several retained shot@ videos at once, small next to
    # the VRAM/RAM a generation step itself needs, and never the only thing
    # standing between a run and OOM - release_unreferenced_results and the
    # entry cap both still apply
    DEFAULT_MAX_RETAINED_BYTES = 4 * 1024**3

    def __init__(self, max_entries=None, max_retained_bytes=None):
        # (workflow_id, step_name) -> {"step_data", "step_seed", "result",
        # "output_dir", "generation", "upstream_generations", "retained",
        # "size"}, ordered least- to most-recently-used
        self._entries = OrderedDict()
        self.max_entries = (
            self.DEFAULT_MAX_ENTRIES if max_entries is None else max_entries
        )
        self.max_retained_bytes = (
            self.DEFAULT_MAX_RETAINED_BYTES
            if max_retained_bytes is None
            else max_retained_bytes
        )
        self._retained_bytes = 0

    def clear(self):
        # The generation counter deliberately survives: it only has to be
        # monotonic, and restarting it could make a stale reference match
        self._entries.clear()
        self._retained_bytes = 0

    def stats(self):
        """A read-only snapshot for a caller that cannot see inside the
        worker process otherwise (#418) - entries/retained_bytes are exactly
        the eviction accounting above, so this is how a stale-drop or an
        LRU-evict actually freeing bytes becomes externally observable."""
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "retained_bytes": self._retained_bytes,
            "max_retained_bytes": self.max_retained_bytes,
        }

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
            logger.debug(f"No cache entry for step '{name}' - treating as a miss")
            return None
        if entry["step_seed"] != step_seed:
            logger.debug(
                f"Cached result for step '{name}' used a different seed - treating as a miss"
            )
            return None
        # The output *root* a run was told to write to. A run directory is
        # new every execution and would defeat the cache; the root changing
        # means the caller asked for output somewhere the cached files are not
        if entry["output_dir"] != output_dir:
            logger.debug(
                f"Cached result for step '{name}' used a different output_dir - treating as a miss"
            )
            return None
        if needs_result and not entry["retained"]:
            logger.debug(
                f"Cached result for step '{name}' did not retain its Result, and this run needs one - treating as a miss"
            )
            return None
        if not deep_equal(entry["step_data"], step_data):
            logger.debug(
                f"Cached result for step '{name}' has different resolved arguments - treating as a miss"
            )
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
            stale = self._entries.pop(key, None)
            if stale is not None:
                self._retained_bytes -= stale["size"]
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
        size = _result_bytes(result) if retain_result else 0
        previous = self._entries.get(key)
        if previous is not None:
            self._retained_bytes -= previous["size"]
        self._entries[key] = {
            "step_data": step_data,
            "step_seed": step_seed,
            "result": result if retain_result else self._strip_result(result),
            "retained": retain_result,
            "output_dir": output_dir,
            "generation": next(_generations),
            "upstream_generations": self._upstream_generations(workflow_id, step_data),
            "size": size,
        }
        self._retained_bytes += size
        self._entries.move_to_end(key)
        while len(self._entries) > 1 and (
            len(self._entries) > self.max_entries
            or self._retained_bytes > self.max_retained_bytes
        ):
            (evicted_workflow, evicted_step), evicted = self._entries.popitem(
                last=False
            )
            self._retained_bytes -= evicted["size"]
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
        # A fresh cache, not the shallow copy's shared one: save() fills
        # _artifact_cache with the artifacts it extracted - decoded frames, a
        # waveform, possibly still on the GPU - so carrying it over would pin
        # exactly what dropping result_list exists to release. The original
        # Result keeps its own cache; only this entry's copy starts empty.
        stripped._artifact_cache = {}
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
