"""Which pipeline each step of a run owns, and the memory a release reclaims.

`PipelineOwnership` holds a run's four step->pipeline tables: the keys the
previous run loaded under (`prior`), this run's effective keys (`running`),
the keys this run's steps actually recorded (`keys_by_step`, which the worker
carries into the next command) and the cache hits whose pipeline was not
resident (`deferred`). The resources themselves - the persistent pipeline
dict and the run's shared components - stay with the caller.

The module functions are every memory-reclaim lookup a run makes: the
release a step asks for, the eviction of a superseded variant, and the
cleanup between steps. They live in one module so `empty_device_cache` and
`release_host_caches` are each looked up in exactly one place. Beside them,
the two ways a pipeline step comes to own a pipeline: `wrap_resident` around
a model already loaded, and `load_fresh`.

This module does not import `dw.workflow`.
"""

import gc
import logging
from dataclasses import dataclass

import torch

from . import device_memory_stats, empty_device_cache
from .events import emit_phase, get_context
from .host_memory import release_host_caches
from .pipeline_processors.pipeline import Pipeline
from .step_cache import pipeline_cache_key, step_pipeline_keys
from .tasks.model_cache import clear_model_cache

logger = logging.getLogger("dw")


@dataclass
class _Deferred:
    """A cache hit whose pipeline was not resident and so was not loaded."""

    step_data: dict
    seed: int
    released: bool = False


class PipelineOwnership:
    """A run's step->pipeline tables.

    A `Workflow` holds one from construction, so a step action created
    outside `run` works, and `run` replaces it with a fresh one: a persistent
    worker reuses a `Workflow` across jobs, and what one run recorded or
    deferred says nothing about what the next has resident.
    """

    def __init__(self, prior_step_keys=None):
        # Last run's step->key map: a redefined step's old model is evicted
        # BEFORE its replacement loads, or the transition holds both at once
        self.prior = prior_step_keys or {}
        # This run's key table, set by begin() before the first step. None
        # means no table: a step driven outside run() hashes its own key
        self.running = None
        # Step name -> cache key for this run, so release_pipeline and
        # pipeline_reference still address pipelines by the step that made
        # them
        self.keys_by_step = {}
        # Step name -> _Deferred for each cache hit whose pipeline was not
        # resident and so was not loaded
        self.deferred = {}

    def begin(self, steps):
        """Take this run's key table from its realized steps."""
        self.running = step_pipeline_keys(steps)

    def load_key(self, step_definition):
        """The pipeline cache key a pipeline step loads under.

        Read from the run's table (step_pipeline_keys, taken by run() before
        anything loaded) - a load edits the definition it is handed, so a
        key re-hashed from it could disagree with the one the table, a
        deferred hit and the cache_hits probe use. The table also holds the
        effective key - a step that reuses components folds in its sources'
        keys - which the step's own definition cannot give. Hashed here only
        for a step driven outside run(), which has no table; a table that
        lacks the step is an internal error, never a silent re-hash.
        """
        if self.running is None:
            return pipeline_cache_key(step_definition["pipeline"])
        step_name = step_definition["name"]
        if step_name not in self.running:
            raise RuntimeError(
                f"Internal error: pipeline step '{step_name}' is not in this "
                "run's pipeline key table"
            )
        return self.running[step_name]

    def record(self, step_name, key):
        """Record which cache key a step's pipeline lives under this run."""
        self.keys_by_step[step_name] = key

    def key_for(self, step_name):
        """The cache key `step_name` recorded this run, or None."""
        return self.keys_by_step.get(step_name)

    def defer(self, step_name, step_data, seed):
        """Remember a hit whose pipeline a later borrowing step loads."""
        self.deferred[step_name] = _Deferred(step_data, seed)

    def mark_released(self, step_name):
        """A deferred step asked for its release.

        Release frees the pipeline but keeps what it published: a later
        reuser still gets the components, and the lazy load drops the
        pipeline again straight after. A step that did not defer is left
        alone.
        """
        if step_name in self.deferred:
            self.deferred[step_name].released = True

    def superseded_key(self, step_name, cache_key, pipelines):
        """The resident key `step_name` loaded under last run, when loading
        `cache_key` replaces it - else None.

        Only this step's own variant: a key another step of THIS run loads
        under NOW is that step's warm model, and releasing it here would
        reload it cold a moment later while holding both stacks. Judged on
        the other steps' current keys (`running`), not their prior ones:
        when every step sharing one model variable changes at once, each
        still has the old key as its prior key, and nobody will load it
        again - holding it would be the two-stack transition #150 fixed.
        `prior` is merged across every job the worker has run, never pruned,
        so a name from an earlier, unrelated workflow does not count either -
        it is not among the running steps. If nothing touches the key this
        run, the end-of-run sweep drops it.
        """
        prior_key = self.prior.get(step_name)
        still_shared = any(
            other != step_name and key == prior_key
            for other, key in (self.running or {}).items()
        )
        if (
            prior_key
            and prior_key != cache_key
            and prior_key in pipelines
            and not still_shared
        ):
            return prior_key
        return None


def allocated_mb():
    """Device memory in use right now, for the pipeline_released event -
    None where the backend cannot say, so a reading is never confused with
    a genuine zero."""
    try:
        stats = device_memory_stats()
    except Exception:  # a progress figure is never worth failing a run over
        return None
    return stats["allocated_mb"] if stats["available"] else None


def _release_host_caches(step_name):
    """Hand the host memory a release freed back to the OS, not at job end.

    `release_host_caches` only touches blocks nothing is using, so anything
    still loaded is undisturbed. A cleanup is never worth failing a run for.
    """
    try:
        released = release_host_caches()
    except Exception as e:
        logger.debug(f"Could not release host caches after {step_name}: {e}")
        return
    if released:
        logger.info(f"Release after {step_name} returned {released:.0f} MB to the OS")


def finish_release(workflow_id, step_name, index, before):
    """Free what a popped pipeline held and announce the release.

    `before` is measured by the caller ahead of dropping its own
    references, which may free the pipeline on the spot.
    """
    gc.collect()
    empty_device_cache()
    # The device cache is not the only one the release fills: the
    # pinned-host staging buffers the pipeline offloaded through and the
    # heap arenas its weights were read into stay in this process's RSS
    # until they are handed back, which otherwise waits for the end of
    # the job - ~10 GB held through every step after the release (#368)
    _release_host_caches(step_name)
    # Say so on the event stream. The release is otherwise invisible to
    # a consumer: it sits inside the sub-second window between a step's
    # generation and its files appearing, which is too narrow to catch
    # by polling get_memory, and it is exactly the ordering this event
    # exists to make readable (it precedes the step's step_end, and on a
    # released card the figures show the drop rather than implying it)
    get_context().emit(
        "pipeline_released",
        workflow=workflow_id,
        step=step_name,
        index=index,
        gpu_memory_allocated_mb=allocated_mb(),
        gpu_memory_allocated_before_mb=before,
    )


def evict_superseded(pipelines, step_name, prior_key):
    """Release `step_name`'s previous variant before its replacement loads,
    so the swap never holds old and new stacks simultaneously."""
    logger.info(
        f"Step '{step_name}' was redefined - releasing its previous "
        "pipeline before loading the new one"
    )
    before = allocated_mb()
    pipelines.pop(prior_key, None)
    gc.collect()
    empty_device_cache()
    _release_host_caches(step_name)
    # Say so on the event stream, for the same reason the explicit
    # release does: without it a reload-on-top-of-a-resident-model
    # is indistinguishable from a cold load, and the difference is
    # whether the next thing that happens is an OOM kill (#150).
    # 'reason' separates it from the release a step asked for
    get_context().emit(
        "pipeline_released",
        step=step_name,
        reason="superseded",
        gpu_memory_allocated_mb=allocated_mb(),
        gpu_memory_allocated_before_mb=before,
    )


def reclaim_after_step(step_data, step_name):
    """The cleanup between steps, once the run has dropped its references
    to the step's action and result."""
    # Task models are cached for the life of the process - the cache
    # exists so a step's cartesian product loads its model once, and
    # nothing else evicts it. A prompt-expanding language model
    # feeding a generation step would otherwise hold its weights on
    # the device for the whole run
    if step_data.get("release_models", False):
        logger.info(f"Releasing task models for step: {step_name}")
        clear_model_cache()
        gc.collect()
        _release_host_caches(step_name)

    # Cleanup between steps (but keep pipelines loaded). Returning
    # cached blocks to the device lets the next step's differently
    # shaped allocations use them
    gc.collect()
    empty_device_cache()


def wrap_resident(
    cached_pipeline,
    step_definition,
    default_seed,
    device,
    output_dir,
    file_prefix,
    base_dir=None,
):
    """A new Pipeline wrapper for a step around a model already resident
    under its key, whose shared components the caller has already
    republished (a later step's reused_components otherwise finds nothing).
    `output_dir` and `file_prefix` are where the step writes and what it
    names its files."""
    # Create new Pipeline wrapper with updated step definition
    # but reuse the loaded model from cache
    new_pipeline_wrapper = Pipeline(
        step_definition["pipeline"],
        default_seed,
        device,
        cached_pipeline.pipeline,  # Reuse the actual loaded model
        output_dir=output_dir,
        file_prefix=file_prefix,
        base_dir=base_dir,
    )
    # Set up generator with potentially new seed. no_generator is a
    # boolean - only an explicit true disables the generator - and the
    # generator lives on the pipeline's own device, which may override
    # the workflow default (the fresh-load path resolves it the same way)
    if not new_pipeline_wrapper.configuration.get("no_generator", False):
        logger.debug("Setting up generator for cached pipeline with new arguments")
        new_pipeline_wrapper.argument_template["generator"] = torch.Generator(
            new_pipeline_wrapper.device
        ).manual_seed(
            new_pipeline_wrapper.pipeline_definition.get("seed", default_seed)
        )

    # A cache hit and a cold load look identical from the outside -
    # same step, same dot - and they differ by minutes
    emit_phase("cached", detail=new_pipeline_wrapper.name)
    return new_pipeline_wrapper


def load_fresh(
    step_definition,
    shared_components,
    default_seed,
    device,
    output_dir,
    file_prefix,
    base_dir=None,
):
    """A step's pipeline loaded from scratch, behind the trust gate."""
    pipeline = Pipeline(
        step_definition["pipeline"],
        default_seed,
        device,
        output_dir=output_dir,
        file_prefix=file_prefix,
        base_dir=base_dir,
    )
    # Before the marker, not after it: a definition refused by the
    # trust gate must not have announced a load it never began, or a
    # consumer reading job events cannot tell 'refused before load'
    # from 'loaded, then refused' (#137)
    pipeline.check_trusted()
    # Loading is the longest silence in a run: weights, quantization,
    # adapters and placement all happen inside this call
    emit_phase("loading", detail=pipeline.name)
    pipeline.load(shared_components)
    return pipeline
