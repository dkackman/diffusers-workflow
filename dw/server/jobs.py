"""Job queue over the persistent worker processes - one per card.

The server runs one worker per `--devices` entry (a WorkerSlot each, its
process managed by a WorkerManager). A dispatcher thread hands each free
card the first queued job that fits it: queue order, except that a job too
big for every free card waits while a smaller one behind it takes the card
(backfill). Each running job has its own thread, so a crash or a cancel
reaches only the worker it names (#462). With one device this is the old
single FIFO worker.

Among the free cards a job fits, it goes to the one it has an affinity for
(`_choose_slot`): a rerun to the card that ran the original, then any job
to the card whose worker last ran the same workflow identity - that worker's
pipelines and step cache are warm, and a worker frees both whenever the
identity changes. A cache probe and a plan's estimate ask about the card
`route` names, the one the job would be dispatched to now. Memory is per
card too: each slot keeps its own last reading.

`JobManager.slots` is the pool. `_current_job_id`, `_worker_lock` and
`worker_manager` are compatibility names for single-worker callers: the
first slot and the oldest running job. New code reads `slots`.

Jobs collect their progress events with sequence numbers so an SSE client
can attach late (or reconnect) and replay from where it left off.
"""

import os
import copy
import json
import queue
import secrets
import time
import uuid
import logging
import threading

from ..worker_protocol import (
    Cancelled,
    ClearMemory,
    Execute,
    Failed,
    MemoryCleared,
    MemoryInfo,
    MemoryStatus,
    MemoryStatusReply,
    Output,
    ProbeCache,
    Progress,
    Succeeded,
    UnknownReply,
    WorkerCrashed,
    WorkflowLoaded,
    parse_reply,
    workflow_identity,
)
from ..devices import card_of, device_ordinal, ordinal_of
from ..host_memory import process_rss_mb
from ..worker_manager import WorkerManager
from ..workflow_run import SEED_BITS
from ..security import (
    SecurityError,
    validate_json_size,
    validate_output_path,
    validate_path,
    validate_workflow_path,
)
from .. import references
from ..runs import REALIZED_FILE_NAME
from ..settings import resolve_path
from ..workspace import DEFAULT_WORKSPACE_NAME
from .job_history import JobHistory
from .job_record import (
    ACK_NONE,
    CANCELLED,
    FAILED,
    QUEUED,
    RERUN_SPEC_KEYS,
    RUNNING,
    SUCCEEDED,
    TERMINAL_STATES,
    Job,
)
from .outputs import output_kinds

logger = logging.getLogger("dw")

# Finished jobs kept in memory for SSE replay grace; older ones live in
# history only, so a long-running server's memory stays bounded
TERMINAL_JOBS_KEPT = 20


class WorkerBusy(RuntimeError):
    """A memory clear found no idle card to clear (a 409 at the route)."""


class WorkerSlot:
    """One card's worker and what the dispatcher knows about it: the job it
    is running, and the lock that serializes talking to it. A job holds its
    slot's lock for its whole run, as the single worker's did; a memory or
    cache request takes it with a bounded wait."""

    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.Lock()
        self.current_job_id = None
        self.started_at = None
        # The workflow identity the worker last ran - JobManager.identity_of
        self.last_identity = None
        # This card's last memory reading and when it was taken
        self.last_memory = None
        self.last_memory_at = None
        # The worker process _rank_for_oom last looked at
        self.ranked_pid = None

    @property
    def device(self):
        return getattr(self.manager, "device", None)

    def capacity_gb(self):
        """The card's VRAM in the catalog's GB, or None where it cannot be
        read - a card of unknown size takes any job, so an unreadable
        ceiling never strands the queue."""
        try:
            return self.manager.capacity_gb()
        except Exception:
            logger.debug("Could not read a worker's card capacity", exc_info=True)
            return None

    def ceiling_gb(self):
        """The most a declared vram_estimate may project on this card, or
        None where it cannot be read (WorkerManager.ceiling_gb)."""
        try:
            return self.manager.ceiling_gb()
        except Exception:
            logger.debug("Could not read a worker's card ceiling", exc_info=True)
            return None

    def label(self):
        """The card as a job record names it, or None."""
        try:
            return self.manager.device_label()
        except Exception:
            logger.debug("Could not read the worker's device", exc_info=True)
            return None

    def ordinal(self):
        """The card as `cuda:1` - what a `device` argument names it by."""
        return device_ordinal(self.device)

    def warm_identity(self):
        """The workflow identity this card's worker holds warm, or None when
        no worker is running - a worker that is not running holds nothing."""
        if not self.manager.worker_active:
            return None
        return self.last_identity

    def alive(self):
        process = getattr(self.manager, "worker_process", None)
        return bool(
            self.manager.worker_active and process is not None and process.is_alive()
        )

    def fits(self, need):
        """Whether a job needing `need` - (GB, hard) or None - may start
        here. A hard (declared vram_estimate) need is held to the card's
        ceiling, as admission holds it; a soft catalog `cost` figure to the
        card's size in the catalog's GB."""
        if need is None:
            return True
        gb, hard = need
        capacity = self.ceiling_gb() if hard else self.capacity_gb()
        return capacity is None or gb <= capacity


class JobManager:
    """Queues jobs and dispatches them onto one worker process per card."""

    def __init__(
        self,
        output_dir,
        log_level="INFO",
        worker_manager=None,
        history_path=None,
        workflow_dir=None,
        devices=None,
        worker_managers=None,
    ):
        """`devices` is the server's `--devices` list: one worker per entry
        when it names more than one. `worker_managers` supplies the workers
        themselves (tests); `worker_manager` is the one-worker form of it.
        With none of the three, one worker on the device dw runs on."""
        self.output_dir = validate_output_path(output_dir, None)
        # Confines workflow_path/base_dir/sub-workflow resolution for every
        # job this manager submits - the server's configured workflow_dir
        self.workflow_dir = workflow_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_level = log_level
        if worker_managers is None:
            if worker_manager is not None:
                worker_managers = [worker_manager]
            elif devices and len(devices) > 1:
                worker_managers = [WorkerManager(device) for device in devices]
            else:
                # One card: the worker follows get_device(), as it always has
                worker_managers = [WorkerManager()]
        self.slots = [WorkerSlot(manager) for manager in worker_managers]
        # The first card's worker - what the single-worker callers address
        self.worker_manager = self.slots[0].manager
        self.history = JobHistory(history_path or resolve_path("jobs.sqlite"))
        self.jobs = {}
        # Reentrant: cancel() finishes a queued job while holding it, and
        # _finish's terminal-job trim needs it again on the same thread
        self._lock = threading.RLock()  # guards job state transitions
        # Pending job ids in run order - a list, not a Queue, so the queue
        # can be reordered while jobs wait
        self._pending = []
        self._wake = threading.Condition(self._lock)
        # The VRAM each job needs to start, by job id (dispatch reads it)
        self._needs = {}
        # The card a rerun prefers - the original's ordinal - by job id
        self._preferred = {}
        self._stop = threading.Event()
        # The thread running each dispatched job, joined by shutdown()
        self._job_threads = set()
        self._runner = threading.Thread(
            target=self._run_loop, daemon=True, name="job-runner"
        )
        self._runner.start()

    @property
    def _worker_lock(self):
        """The first card's lock - the single-worker name for it."""
        return self.slots[0].lock

    @property
    def _current_job_id(self):
        """The oldest running job's id, or None while every card is idle."""
        with self._lock:
            running = [slot for slot in self.slots if slot.current_job_id]
            if not running:
                return None
            return min(running, key=lambda slot: slot.started_at or 0).current_job_id

    @_current_job_id.setter
    def _current_job_id(self, job_id):
        with self._lock:
            self.slots[0].current_job_id = job_id
            self.slots[0].started_at = time.time() if job_id else None

    @property
    def last_memory(self):
        """The first card's last reading - the single-worker name for it."""
        return self.slots[0].last_memory

    @last_memory.setter
    def last_memory(self, info):
        self.slots[0].last_memory = info

    @property
    def last_memory_at(self):
        return self.slots[0].last_memory_at

    @last_memory_at.setter
    def last_memory_at(self, at):
        self.slots[0].last_memory_at = at

    # ------------------------------------------------------------- affinity

    @staticmethod
    def identity_of(source, file_spec, definition):
        """The workflow identity a worker caches by - the worker's own rule,
        `worker_protocol.workflow_identity`."""
        return workflow_identity(source, file_spec, definition)

    def _job_identity(self, job):
        spec = job.spec
        return self.identity_of(
            spec.get("source"), spec.get("file_spec"), spec.get("definition")
        )

    def slot_for(self, device):
        """The slot running `device` ('cuda:1'), or raise ValueError naming
        the cards this server has."""
        wanted = device_ordinal(device)
        for slot in self.slots:
            if slot.ordinal() == wanted:
                return slot
        cards = ", ".join(str(slot.ordinal()) for slot in self.slots)
        raise ValueError(
            f"This server has no worker on {device}: its cards are {cards}"
        )

    def _choose_slot(self, candidates, identity=None, preferred=None):
        """The candidate a job with `identity` goes to: the card a rerun
        prefers, then the card whose worker holds that identity warm, then
        a card whose worker is not running (nothing warm to evict), then
        the first in `--devices` order."""
        if preferred is not None:
            for slot in candidates:
                if slot.ordinal() == preferred:
                    return slot
        if identity is not None:
            for slot in candidates:
                if slot.warm_identity() == identity:
                    return slot
        for slot in candidates:
            if not slot.manager.worker_active:
                return slot
        return candidates[0]

    def route(self, identity=None, need=None):
        """The slot a job with `identity` and VRAM `need` (admission's
        `vram_need`) would be dispatched to now: a free card it fits when
        there is one, else the card it would wait for. What a cache probe
        and a plan's estimate ask about."""
        need = self._dispatch_need(need)
        with self._lock:
            fitting = [slot for slot in self.slots if slot.fits(need)] or self.slots
            free = [slot for slot in fitting if slot.current_job_id is None]
            return self._choose_slot(free or fitting, identity)

    # ------------------------------------------------------------- VRAM fit

    def _capacities(self, hard=False):
        """(slot, GB) for every card whose size can be read - its ceiling
        for a hard need, its catalog size for a soft one (WorkerSlot.fits)."""
        readings = [
            (slot, slot.ceiling_gb() if hard else slot.capacity_gb())
            for slot in self.slots
        ]
        return [(slot, capacity) for slot, capacity in readings if capacity]

    def largest_ceiling_gb(self):
        """The most VRAM any card here can be held to (WorkerSlot.ceiling_gb),
        or None when no card has reported its size yet. Admission checks a
        declared vram_estimate against this rather than the process's own
        device, which under --devices is only the first card."""
        capacities = self._capacities(hard=True)
        return max(capacity for _, capacity in capacities) if capacities else None

    def _unfit_message(self, need):
        """Why no card here can ever run a job needing `need` (admission's
        `vram_need`, (GB, hard)), or None when one can. Only a declared
        vram_estimate's projection is hard: a catalog `cost` figure is what
        a card the model was measured on held, not a floor, so it orders
        dispatch but refuses nothing."""
        if not need or not need[1] or need[0] is None:
            return None
        capacities = self._capacities(hard=True)
        if not capacities:
            return None
        slot, largest = max(capacities, key=lambda reading: reading[1])
        if need[0] <= largest:
            return None
        name = slot.label() or slot.device or "this server's card"
        return (
            f"This job needs {need[0]:.1f} GB of VRAM, more than any card here "
            f"has: the largest is {name} ({largest:g} GB usable)"
        )

    def check_fits(self, admission):
        """Raise ValueError - a 400 at the route - for a request that is
        not admissible, or that needs more VRAM than the largest card here.
        An inadmissible request's message gains the card sentence when that
        applies too, so the caller learns both at once."""
        unfit = self._unfit_message(getattr(admission, "vram_need", None))
        if not admission.ok:
            message = admission.message()
            raise ValueError(f"{message}; {unfit}" if unfit else message)
        if unfit:
            raise ValueError(unfit)

    def _dispatch_need(self, need):
        """What a job must find on a free card before it starts - (GB, hard)
        for WorkerSlot.fits - or None for any card. A soft (`cost`) figure no card here meets is dropped,
        so the job runs on whatever card is free, as it would have on one
        card."""
        if not need or need[0] is None:
            return None
        gb, hard = need
        if hard or any(capacity >= gb for _, capacity in self._capacities()):
            return (gb, hard)
        return None

    # ------------------------------------------------------------- submission

    def submit(
        self,
        *,
        admitted,
        workflow_path=None,
        workflow=None,
        arguments=None,
        base_dir=None,
        workflow_dir=None,
        output_dir=None,
        asset_dir=None,
        workspace=None,
        catalog_name=None,
        acknowledged=ACK_NONE,
        acknowledged_cost=None,
        warnings=None,
        vram_need=None,
        preferred_device=None,
    ):
        """Record a job request and queue it.

        Callers admit first (`dw.server.admission.admit`) and pass the
        admitted Workflow as `admitted`. submit records and queues; it does
        not re-check. The job carries admission's snapshot - its definition,
        file_spec and name - and the worker runs that, so a file edited
        while the job waits runs as it was admitted. Raises ValueError only
        when the request names neither or both of `workflow_path` and
        `workflow`.

        `workflow_dir` overrides this job's confinement root for a workflow
        that lives outside the writable directory - an example or a builtin,
        which the caller has already resolved against the search path. The
        worker confines the job's file_spec to whatever this job records, so
        the override travels with the job rather than widening the manager.

        `output_dir`, `asset_dir` and `workspace` name which workspace this
        job runs in. They travel with the job for the same reason: one
        server holds several workspaces, and the process-wide roots would
        make every job belong to whichever one was configured at startup.

        `catalog_name` is the listing name the caller resolved `workflow_path`
        from, kept for history; None for an inline definition.

        `acknowledged` is the form of cost acknowledgement the caller gave
        (none/boolean/bound) and `acknowledged_cost` the bound object - both
        recorded, neither checked here; the route checks (#85).

        `warnings` are the admission's (`Admission.warnings`) - every warning
        /api/validate reports for this request - recorded on the job.

        `vram_need` is the admission's (`Admission.vram_need`): the job
        starts only on a card with that much VRAM (`check_fits` has already
        refused one no card here holds).

        `preferred_device` is the card ('cuda:1') the job goes to when that
        card is free and fits it - a rerun's original card (`rerun`).

        `workflow_path`, `workflow` and `base_dir` are the request as the
        caller made it, kept for a rerun (which admits afresh) and for
        `definition()`; the snapshot is not persisted.
        """
        arguments = arguments or {}
        if (workflow_path is None) == (workflow is None):
            raise ValueError("Provide exactly one of workflow_path or workflow")

        confinement = workflow_dir or self.workflow_dir
        job_output_dir = (
            validate_output_path(output_dir, None) if output_dir else self.output_dir
        )
        os.makedirs(job_output_dir, exist_ok=True)

        if workflow_path is not None:
            spec = {"workflow_path": workflow_path, "source": "path"}
        else:
            # Without one, the directory admission resolved relative paths
            # against - the synthetic file_spec's own - kept so a rerun
            # admits against the same directory
            spec = {
                "workflow": workflow,
                "base_dir": base_dir or os.path.dirname(admitted.file_spec),
                "source": "inline",
            }
        # The snapshot admission checked: what the worker builds and runs
        spec["definition"] = admitted.workflow_definition
        spec["file_spec"] = admitted.file_spec
        spec["workflow_name"] = admitted.name
        spec["arguments"] = arguments
        spec["workflow_dir"] = confinement

        # Which workspace this job runs in, and the roots that follow from
        # it - recorded on the job so history, the worker command and a
        # rerun all agree without re-deriving them
        spec["workspace"] = workspace
        spec["catalog_name"] = catalog_name
        # The acknowledgement form travels with the job so history can say
        # whether this run was consented to at its actual size (#85)
        spec["acknowledged"] = acknowledged
        if acknowledged_cost is not None:
            spec["acknowledged_cost"] = acknowledged_cost
        spec["output_dir"] = job_output_dir
        if asset_dir:
            spec["asset_dir"] = asset_dir
        spec["warnings"] = list(warnings or [])

        job = Job(spec)
        with self._lock:
            self.jobs[job.id] = job
            self._needs[job.id] = self._dispatch_need(vram_need)
            if preferred_device:
                self._preferred[job.id] = preferred_device
        job.add_event({"event": "job_status", "status": QUEUED})
        with self._wake:
            self._pending.append(job.id)
            self._wake.notify()
        logger.info(f"Queued job {job.id} for workflow {job.workflow_name}")
        return job

    def get(self, job_id):
        """A live Job, or a historical detail dict for a finished past run."""
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        return self.history.get(job_id)

    def definition(self, job_id):
        """The workflow JSON a job names, for a read-only view of it and for
        what a rerun will draw its seed into.

        A job launched from a path answers with that file as it is now, the
        same file a rerun admits - re-read from disk, confined to the root
        the job ran against. When the file can no longer be read (it moved,
        grew past the size limit or stopped parsing), a live job falls back
        to the snapshot admission checked, so the page still shows what ran.
        A restored job has no snapshot (it is not persisted). An inline
        definition comes straight from the spec. None when there is no such
        job, or nothing is left to show - a graph of the run is a nicety,
        never a reason to fail the page.
        """
        job = self.jobs.get(job_id)
        if job is not None:
            spec = job.spec
            snapshot = spec.get("definition")
        else:
            historical = self.history.get(job_id)
            if historical is None:
                return None
            spec = historical.get("spec") or {}
            snapshot = None
        inline = spec.get("workflow")
        if inline is not None:
            return copy.deepcopy(inline)
        path = spec.get("workflow_path")
        if path:
            try:
                validated = validate_workflow_path(
                    path, spec.get("workflow_dir") or self.workflow_dir
                )
                validate_json_size(validated)
                with open(validated, "r") as file:
                    return json.load(file)
            except (SecurityError, OSError, ValueError):
                logger.debug(f"Workflow file for job {job_id} can no longer be read")
        if snapshot is not None:
            return copy.deepcopy(snapshot)
        return None

    def run_location(self, job_id):
        """(output root, run dir) for the run a job wrote, or None when the
        job is unknown or never started a run.

        The job's own root, not the manager's: one server holds several
        workspaces, and a job carries the root it ran against. The root is
        validated; joining `run_dir` onto it stays the caller's, confined to
        that root, so a run_dir read back out of the database cannot name
        anything outside it.
        """
        job = self.jobs.get(job_id)
        if job is not None:
            run_dir = job.run_dir
            output_dir = job.spec.get("output_dir") or self.output_dir
        else:
            historical = self.history.get(job_id)
            if historical is None:
                return None
            run_dir = historical.get("run_dir")
            output_dir = (historical.get("spec") or {}).get(
                "output_dir"
            ) or self.output_dir
        if not run_dir:
            return None
        return validate_output_path(output_dir, None), run_dir

    def realized(self, job_id):
        """The realized workflow a job ran, or None when the job predates
        run tracking or its run directory no longer holds the file."""
        try:
            location = self.run_location(job_id)
            if location is None:
                return None
            root, run_dir = location
            path = validate_path(os.path.join(root, run_dir, REALIZED_FILE_NAME), root)
            validate_json_size(path)
            with open(path, "r") as file:
                return json.load(file)
        except (SecurityError, OSError, ValueError) as e:
            logger.debug(f"No realized workflow for job {job_id}: {e}")
            return None

    def arguments(self, job_id):
        """The arguments a job was submitted with, or {} when none are on
        record."""
        job = self.jobs.get(job_id)
        if job is not None:
            return job.spec.get("arguments") or {}
        historical = self.history.get(job_id)
        return (historical or {}).get("arguments") or {}

    def seed_variable(self, job_id):
        """The variable this job's workflow draws its seed from, or None.

        Read from the workflow as written, never from the realized copy the
        run wrote: realization pins the top-level seed to the integer the run
        used, so a realized workflow always looks like it names a literal.

        None means a new-seed rerun has nowhere to put one - either the seed
        is a literal (an argument cannot override it) or the workflow names
        no seed at all, in which case every run already draws a fresh one and
        the step cache is off.
        """
        definition = self.definition(job_id)
        seed = (definition or {}).get("seed")
        name = references.ref_name(references.VARIABLE, seed)
        if name is None:
            return None
        return name if name in (definition.get("variables") or {}) else None

    def rerun_spec(self, job_id, new_seed=False):
        """The spec and arguments a rerun of `job_id` would submit, as
        (spec, arguments), or None for an unknown job - split from rerun()
        so a route can admit the run before queuing it (#85).

        `new_seed` draws a fresh seed into the workflow's seed variable
        (see rerun). Raises ValueError when it cannot, and when the named
        workspace the job ran in is gone."""
        job = self.jobs.get(job_id)
        if job is not None:
            spec = {key: job.spec[key] for key in RERUN_SPEC_KEYS if key in job.spec}
            arguments = job.spec.get("arguments", {})
        else:
            historical = self.history.get(job_id)
            if historical is None:
                return None
            spec = {
                key: historical["spec"][key]
                for key in RERUN_SPEC_KEYS
                if key in historical["spec"]
            }
            arguments = historical["arguments"]

        if new_seed:
            variable = self.seed_variable(job_id)
            if variable is None:
                raise ValueError(
                    "This workflow does not draw its seed from a variable, so "
                    "a rerun cannot change it. A workflow with no seed at all "
                    "already draws a fresh one every run."
                )
            # Bounded so the number survives its trip through a browser as
            # JSON - see SEED_BITS
            arguments = {**arguments, variable: secrets.randbits(SEED_BITS)}

        workspace = spec.get("workspace")
        if (
            workspace
            and workspace != DEFAULT_WORKSPACE_NAME
            and spec.get("output_dir")
            and not os.path.isdir(spec["output_dir"])
        ):
            raise ValueError(f"Workspace '{workspace}' the job ran in no longer exists")
        return spec, arguments

    def rerun(
        self,
        job_id,
        *,
        admitted,
        new_seed=False,
        acknowledged=ACK_NONE,
        acknowledged_cost=None,
        warnings=None,
        arguments=None,
        vram_need=None,
    ):
        """Queue a fresh job from a previous job's spec.

        Every root the original ran against (workflow_dir/output_dir/
        asset_dir/workspace) rides along, not just the workflow identity -
        otherwise a rerun of a job from a named workspace would fall back to
        the manager's process-wide default and silently run somewhere else.

        `new_seed` draws a fresh seed into the workflow's seed variable. A
        plain rerun of a seeded workflow repeats its arguments exactly, which
        makes every step a step-cache hit: it republishes the earlier run's
        files in a fraction of a second and generates nothing. That is the
        cache doing its job - the same seed and the same inputs would produce
        the same pixels - so the way to actually get another image is to
        change the seed, and this is that.

        Like submit, rerun does not re-check: callers admit first. The route
        admits the arguments `rerun_spec(job_id, new_seed)` answered and
        passes them back as `arguments`, with the admission's `warnings` and
        its Workflow as `admitted`, so what is queued is what was admitted -
        the seed is not drawn twice.

        `acknowledged` and `acknowledged_cost` are this request's own; the
        original's bound object rides along in the spec for the record when
        the request brought none.

        The rerun prefers the card the original ran on: its step cache is
        there, which is what makes a same-seed rerun finish at once. When
        that card is busy, the rerun takes any free card it fits.
        """
        prepared = self.rerun_spec(job_id, new_seed=new_seed and arguments is None)
        if prepared is None:
            return None
        spec, recorded = prepared
        return self.submit(
            workflow_path=spec.get("workflow_path"),
            workflow=spec.get("workflow"),
            admitted=admitted,
            arguments=recorded if arguments is None else arguments,
            base_dir=spec.get("base_dir"),
            workflow_dir=spec.get("workflow_dir"),
            output_dir=spec.get("output_dir"),
            asset_dir=spec.get("asset_dir"),
            workspace=spec.get("workspace"),
            catalog_name=spec.get("catalog_name"),
            acknowledged=acknowledged,
            acknowledged_cost=(
                acknowledged_cost
                if acknowledged_cost is not None
                else spec.get("acknowledged_cost")
            ),
            warnings=warnings,
            vram_need=vram_need,
            preferred_device=self._original_device(job_id),
        )

    def _original_device(self, job_id):
        """The ordinal of the card `job_id` ran on ('cuda:1'), or None."""
        job = self.jobs.get(job_id)
        if job is not None:
            label = job.device
        else:
            label = (self.history.get(job_id) or {}).get("device")
        return ordinal_of(label)

    def queue_position(self, job_id):
        """Index in the waiting queue, or None when the job is not queued."""
        with self._lock:
            return self._pending.index(job_id) if job_id in self._pending else None

    def describe(self, job):
        """A live job's detail plus its queue position while it waits - what
        the per-job endpoints return, so a client holding one job can say
        where it stands without fetching the whole list."""
        detail = job.detail()
        position = self.queue_position(job.id)
        if position is not None:
            detail["queue_position"] = position
        return detail

    def list(self, workspace=None, statuses=None):
        """All jobs, live and historical, sorted by creation. `workspace` filters to one workspace; omitted, the list
        spans every workspace the server holds, unchanged from before
        workspaces existed. `statuses` filters to a set of job states
        ('queued', 'running', 'succeeded', 'failed', 'cancelled'); omitted,
        every state is listed."""
        statuses = set(statuses) if statuses else None
        with self._lock:
            live = sorted(self.jobs.values(), key=lambda j: j.created_at)
            positions = {job_id: i for i, job_id in enumerate(self._pending)}
        live_ids = {job.id for job in live}
        summaries = []
        for job in live:
            summary = job.summary()
            if workspace and summary["workspace"] != workspace:
                continue
            if statuses and summary["status"] not in statuses:
                continue
            if job.id in positions:
                summary["queue_position"] = positions[job.id]
            summaries.append(summary)
        for historical in self.history.recent_summaries(
            workspace=workspace, statuses=statuses
        ):
            if historical["id"] not in live_ids:
                summaries.append(historical)
        summaries.sort(key=lambda summary: summary["created_at"] or 0)
        return summaries

    # ------------------------------------------------------------ cancel/stop

    def cancel(self, job_id):
        """Cancel a queued or running job. A running job's cancel reaches
        only the card running it; the other cards' jobs carry on. Returns
        the job's status after the request, or None for an unknown job."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        with self._lock:
            if job.status in TERMINAL_STATES:
                return job.status
            if job.status == QUEUED:
                if job.id in self._pending:
                    self._pending.remove(job.id)
                self._finish(job, CANCELLED)
                return job.status
            slot = self._slot_running(job.id)
            if job.status == RUNNING and slot is not None:
                try:
                    # The card running this job, and only that one
                    slot.manager.cancel()
                except Exception as e:
                    logger.warning(f"Could not send cancel for job {job_id}: {e}")
        return job.status

    def move(self, job_id, direction):
        """Reorder a queued job: 'up'/'down' swap with a neighbour,
        'front'/'back' go to the ends. The order is the order to start: with
        several cards a job behind may still start first, on a free card the
        one ahead does not fit. Returns the new pending order, or None for a
        job that is not queued (finished, running, unknown)."""
        if direction not in ("up", "down", "front", "back"):
            raise ValueError(f"Unknown queue direction '{direction}'")
        with self._lock:
            if job_id not in self._pending:
                return None
            index = self._pending.index(job_id)
            self._pending.pop(index)
            if direction == "front":
                index = 0
            elif direction == "back":
                index = len(self._pending)
            elif direction == "up":
                index = max(0, index - 1)
            else:
                index = min(len(self._pending), index + 1)
            self._pending.insert(index, job_id)
            return list(self._pending)

    def shutdown(self):
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        # The dispatcher and every job thread share the one wait the single
        # runner had, so a job finishing now still records its outcome
        deadline = time.monotonic() + 5
        self._runner.join(timeout=5)
        with self._lock:
            running = list(self._job_threads)
        for thread in running:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for slot in self.slots:
            slot.manager.shutdown_worker()

    # ---------------------------------------------------------------- runner

    def _run_loop(self):
        """The dispatcher: hand each free card the first queued job that
        fits it, and start that job on its own thread."""
        while True:
            with self._wake:
                while True:
                    if self._stop.is_set():
                        return
                    picked = self._next_dispatch()
                    if picked is not None:
                        break
                    self._wake.wait()
                slot, job = picked
                job.status = RUNNING
                job.started_at = time.time()
                job.device = self._job_device(slot)
                slot.current_job_id = job.id
                slot.started_at = job.started_at
                # The worker switches to this identity for the run
                slot.last_identity = self._job_identity(job)
            job.add_event({"event": "job_status", "status": RUNNING})
            thread = threading.Thread(
                target=self._run_job,
                args=(slot, job),
                daemon=True,
                name=f"job-{job.id}",
            )
            with self._lock:
                self._job_threads.add(thread)
            thread.start()

    def _next_dispatch(self):
        """(slot, job) for the first queued job, in queue order, that fits
        a free card - the one it has an affinity for (`_choose_slot`) - or
        None. A job too big for every free card keeps its place, and the
        scan goes on past it: a smaller job behind it takes the card
        meanwhile (backfill). Called holding _lock."""
        free = [slot for slot in self.slots if slot.current_job_id is None]
        if not free:
            return None
        for job_id in list(self._pending):
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                self._pending.remove(job_id)  # cancelled while waiting
                self._needs.pop(job_id, None)
                self._preferred.pop(job_id, None)
                continue
            need = self._needs.get(job_id)
            fitting = [slot for slot in free if slot.fits(need)]
            if fitting:
                slot = self._choose_slot(
                    fitting, self._job_identity(job), self._preferred.get(job_id)
                )
                self._pending.remove(job_id)
                self._needs.pop(job_id, None)
                self._preferred.pop(job_id, None)
                return slot, job
        return None

    def _slot_running(self, job_id):
        """The slot whose worker is running `job_id`, or None."""
        with self._lock:
            for slot in self.slots:
                if slot.current_job_id == job_id:
                    return slot
        return None

    def _finish(self, job, status, error=None, traceback_text=None):
        job.finish(status, error=error, traceback_text=traceback_text)
        try:
            self.history.record(job)
        except Exception as e:
            logger.warning(f"Could not persist job {job.id}: {e}")
        self._trim_terminal_jobs()

    def _trim_terminal_jobs(self):
        """Drop the oldest finished jobs from memory - history has them, and
        get()/list() fall through to it. Recent ones stay for event replay."""
        with self._lock:
            terminal = [
                job
                for job in sorted(self.jobs.values(), key=lambda j: j.created_at)
                if job.status in TERMINAL_STATES
            ]
            for job in terminal[:-TERMINAL_JOBS_KEPT]:
                del self.jobs[job.id]

    def _run_job(self, slot, job):
        """A dispatched job's thread: run it, then leave the set shutdown()
        joins."""
        try:
            self._run_on(slot, job)
        finally:
            with self._lock:
                self._job_threads.discard(threading.current_thread())

    def _run_on(self, slot, job):
        """Run one dispatched job on its slot's worker. The dispatcher has
        already marked it RUNNING and the slot busy."""
        manager = slot.manager
        with slot.lock:
            try:
                if job.status != RUNNING:
                    # Cancelled between dispatch and here
                    return
                manager.ensure_worker(self.log_level)
                self._rank_for_oom(slot)
                command = Execute(
                    # The snapshot admission checked, which the worker runs
                    # as it is rather than reading the file again
                    definition=job.spec.get("definition"),
                    file_spec=job.spec.get("file_spec"),
                    source=job.spec.get("source"),
                    workflow_dir=job.spec.get("workflow_dir"),
                    # The job's own roots, so a job queued for one workspace
                    # still runs in it after the manager has served another
                    output_dir=job.spec.get("output_dir") or self.output_dir,
                    arguments=job.spec["arguments"],
                    log_level=self.log_level,
                    asset_dir=job.spec.get("asset_dir") or None,
                )
                manager.send_command(command.to_wire())
                outcome = self._consume_results(job, manager, slot)
            except Exception as e:
                logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                outcome = (FAILED, str(e), None)
            finally:
                # Cleared BEFORE the terminal status becomes visible - a
                # client seeing "succeeded" must find the card idle - and
                # the dispatcher woken, since the card is free again
                with self._wake:
                    slot.current_job_id = None
                    slot.started_at = None
                    self._wake.notify_all()
        if job.status not in TERMINAL_STATES:
            status, error, traceback_text = outcome
            self._finish(job, status, error=error, traceback_text=traceback_text)

    def _rank_for_oom(self, slot):
        """A worker this call just started is scored for the kernel's OOM
        killer above every other live worker, so a host-RAM squeeze kills
        the later-started worker - one job - rather than whichever is
        largest. Nothing is written while it is the only worker, so one card
        runs exactly as before."""
        manager = slot.manager
        pid = manager.pid() if hasattr(manager, "pid") else None
        if pid is None or pid == slot.ranked_pid:
            return
        slot.ranked_pid = pid
        others = [
            other.manager for other in self.slots if other is not slot and other.alive()
        ]
        if not others or not hasattr(manager, "set_oom_score_adj"):
            return
        highest = max((other.oom_score_adj or 0) for other in others)
        manager.set_oom_score_adj(min(1000, highest + 100))

    def _job_device(self, slot=None):
        """The card the worker runs on, for the job record. A label that
        cannot be read leaves the job's `device` null rather than failing
        the job - the run itself does not depend on it."""
        return (slot or self.slots[0]).label()

    def _record_manifest(self, job, manifest):
        """What the run wrote, named the way clients address outputs.

        Recorded for a failed or cancelled run as well as a successful one -
        the files the steps before the stop wrote are on disk either way,
        and a manifest that omits them is the difference between "this run
        produced nothing" and "this run produced four of five shots"
        (T015)."""
        job.manifest = self._relative_manifest(
            manifest if manifest is not None else [], job.spec.get("output_dir")
        )

    def _relative_manifest(self, manifest, output_dir=None):
        """A manifest list with every entry's 'files' relativised - the
        rendering `get_job` and `step_end`/`workflow_end` events must all
        agree on (#284)."""
        return [
            (
                {
                    **entry,
                    "files": self._relative_output_names(entry["files"], output_dir),
                }
                if "files" in entry
                else entry
            )
            for entry in manifest
        ]

    def _relative_output_names(self, paths, output_dir=None):
        """The worker reports absolute paths; clients build '/outputs/<name>'
        URLs, and a run writes under '<output_dir>/<identity>/<run id>/'
        (dw/workflow.py's effective_output_dir) - so every file is reported
        by its name relative to the output directory of the job that wrote
        it, with forward slashes. A path outside it (a task step writing
        elsewhere) is left as it came.

        The job's own directory, not the manager's: a job in a named
        workspace writes under that workspace, and naming it relative to the
        default workspace would produce '../<name>/outputs/...' - a path, not
        a name."""
        root = output_dir or self.output_dir
        names = []
        for path in paths:
            relative = os.path.relpath(path, root)
            if relative.startswith(".."):
                names.append(path)
            else:
                names.append(relative.replace(os.sep, "/"))
        return names

    def _consume_results(self, job, manager=None, slot=None):
        """Read `manager`'s worker messages until the run ends; returns the
        terminal (status, error, traceback) for _run_job to apply once the
        card no longer counts the job as current. `slot` is the card's: it
        keeps the run's memory readings."""
        manager = manager or self.worker_manager
        while True:
            try:
                message = manager.get_result()
            except RuntimeError as e:
                # The worker died without managing to send anything - a
                # signal, not an exception, so worker_main's handler never
                # ran and there is no traceback to be had. The exit code is
                # the only diagnosis available, and marking the crash here
                # matters beyond this job: the manager would otherwise go on
                # believing a dead process is active, and every later call
                # that talks to it (memory_status above all) would fail
                # against a queue nobody is reading
                detail = manager.crash_details()
                manager.mark_crashed()
                reason = f"Worker process died: {detail or e}"
                logger.error(f"Job {job.id}: {reason}")
                return (FAILED, reason, None)
            reply = parse_reply(message)

            if isinstance(reply, Progress):
                self._record_progress(job, dict(reply.event))
            elif isinstance(reply, Output):
                job.add_event({"event": "log", "message": reply.message or ""})
            elif isinstance(reply, WorkflowLoaded):
                job.add_event({"event": "log", "message": reply.workflow_name or ""})
            elif isinstance(reply, MemoryInfo):
                self._record_run_memory(job, reply.info, slot)
            elif isinstance(reply, Succeeded):
                self._record_manifest(job, reply.manifest)
                return (SUCCEEDED, None, None)
            elif isinstance(reply, Cancelled):
                self._record_manifest(job, reply.manifest)
                return (CANCELLED, None, None)
            elif isinstance(reply, Failed) and reply.request_id is None:
                # A failed run's steps too: the ones before the failure wrote
                # real files, and a job that reports an empty manifest hides
                # them behind the error that stopped the run
                self._record_manifest(job, reply.manifest)
                return (FAILED, reply.message, reply.traceback)
            elif isinstance(reply, WorkerCrashed):
                manager.mark_crashed()
                return (
                    FAILED,
                    f"Worker crashed: {reply.message}",
                    reply.traceback,
                )
            elif isinstance(reply, UnknownReply):
                logger.warning(f"Unknown worker message type: {reply.type}")
            else:
                # A request's reply (memory_status, memory_cleared,
                # probe_cache, or an error answering one) whose reader gave
                # up before it landed - nobody is waiting for it now
                logger.debug(f"Discarding a stray worker reply: {reply.TYPE}")

    def _record_progress(self, job, event):
        """One run event onto the job, its file names relativised the way
        clients address outputs."""
        if "files" in event:
            event["files"] = self._relative_output_names(
                event["files"], job.spec.get("output_dir")
            )
            # A running job's page renders each output as its step ends,
            # before there is a manifest to classify
            event["output_kinds"] = output_kinds([{"files": event["files"]}])
        if "manifest" in event:
            # workflow_end carries the run's full manifest nested under this
            # key - it must match get_job's rendering of the same list rather
            # than leaking absolute paths (#284)
            event["manifest"] = self._relative_manifest(
                event["manifest"], job.spec.get("output_dir")
            )
        if event.get("event") == "run_start":
            job.run_id = event.get("run_id")
            job.run_dir = event.get("run_dir")
            job.run_version = event.get("version")
        job.add_event(event)

    def _record_run_memory(self, job, info, slot=None):
        """A memory reading the run reported. One per phase boundary now,
        not just once post-run (#273) - each folds into the cached reading
        memory_status() answers from while the job is busy, which is what
        makes that call fresh instead of a refusal for the run's whole
        duration."""
        slot = slot or self.slots[0]
        self._record_memory(info, slot)
        job.add_event({"event": "memory", "info": slot.last_memory})
        # The worker's own high-water mark, latest reading wins (it is
        # monotonic for the process' life) - persisted as a real column
        # rather than only inside the trimmed event tail (#243)
        info = info or {}
        peak = info.get("host_memory_peak_rss_mb")
        if peak is not None:
            job.host_memory_peak_rss_mb = peak
        # This job's own contribution to that process-lifetime peak, computed
        # against its own baseline (#272) - max() is defensive; by
        # construction each reading only grows
        job_peak = info.get("host_memory_job_peak_rss_mb")
        if job_peak is not None:
            job.host_memory_job_peak_rss_mb = max(
                job_peak, job.host_memory_job_peak_rss_mb or 0
            )

    def is_busy(self):
        """True while a job is running on any card or queued - the window in
        which a worker may be reading model files a cache delete would rip
        out."""
        with self._lock:
            if any(slot.current_job_id is not None for slot in self.slots):
                return True
            return any(job.status == QUEUED for job in self.jobs.values())

    def is_worker_busy(self, slot):
        """True while `slot`'s card is running a job."""
        with self._lock:
            return slot.current_job_id is not None

    def restart_worker_if_idle(self):
        """Shut every idle worker down so its next start picks up upgraded
        imports; the next job on that card respawns it via ensure_worker.
        A busy worker is left alone - a run in flight keeps the version it
        started with. Returns whether every worker was restarted."""
        if self.is_busy():
            return False
        restarted = True
        for slot in self.slots:
            if self.is_worker_busy(slot) or not slot.lock.acquire(timeout=2):
                restarted = False
                continue
            try:
                slot.manager.shutdown_worker()
            finally:
                slot.lock.release()
        return restarted

    def workers(self):
        """One entry per card for /api/health: its device, name, VRAM, the
        job it is running, whether its worker process is alive, and that
        process's resident host memory - the pool shares one machine's RAM.
        """
        with self._lock:
            running = [(slot, slot.current_job_id) for slot in self.slots]
        described = []
        for slot, job_id in running:
            pid = slot.manager.pid() if hasattr(slot.manager, "pid") else None
            entry = {
                "device": device_ordinal(slot.device),
                "name": card_of(slot.label()),
                "vram_gb": slot.capacity_gb(),
                "current_job": job_id,
                "alive": slot.alive(),
            }
            rss = process_rss_mb(pid) if pid is not None else None
            if rss is not None:
                entry["host_memory_rss_mb"] = rss
            described.append(entry)
        return described

    # ---------------------------------------------------------------- memory

    def _record_memory(self, info, slot=None):
        """Remember a card's reading and when it was taken, so a later
        cached answer can say how old it is. `slot` defaults to the first
        card."""
        slot = slot or self.slots[0]
        slot.last_memory = info
        slot.last_memory_at = time.time() if info is not None else None

    def _cached_memory(self, reason, slot=None):
        """A card's last reading, labelled with why it is not a live one. A
        caller comparing two readings must compare only `live: true` ones - a
        cached `info` was taken at another moment, and while a job loads a
        model it understates what is resident by however much has loaded
        since."""
        slot = slot or self.slots[0]
        info = slot.last_memory
        age = None
        if info is not None and slot.last_memory_at is not None:
            age = round(time.time() - slot.last_memory_at, 1)
        return {
            "live": False,
            "info": info,
            "stale": info is not None,
            "reason": reason,
            "age_seconds": age,
        }

    def probe_cache(self, command, timeout=5, slot=None):
        """Which steps a worker's step cache would serve for `command` (the
        fields an execute command carries, minus its type), or None when the
        answer cannot be had right now - a job is running on that card, its
        worker is busy, or it did not answer in time. Never blocks a request
        behind a running job, for the same reason memory_status does not.

        The worker asked is `slot`'s, else the one `route` would dispatch
        this workflow to: each card has its own step cache.

        No worker running is a definite answer, not an unknown one: the
        cache lives in the worker process, so a worker that is not running
        holds nothing.
        """
        if slot is None:
            slot = self.route(
                self.identity_of(
                    command.get("source"),
                    command.get("file_spec"),
                    command.get("definition"),
                )
            )
        if self.is_worker_busy(slot):
            return None
        if not slot.manager.worker_active:
            return []
        if not slot.lock.acquire(timeout=2):
            return None
        # A probe that timed out still answers eventually, onto the same
        # queue the next request reads - so each carries an id and request()
        # discards every reply that is not its own, rather than reporting
        # the previous workflow's hit list as this plan's
        try:
            reply = slot.manager.request(
                ProbeCache(request_id=uuid.uuid4().hex, **command), timeout
            )
        except (RuntimeError, queue.Empty) as e:
            logger.debug(f"Worker did not answer the cache probe: {e}")
            return None
        finally:
            slot.lock.release()
        cached = getattr(reply, "cached", None)
        return list(cached) if isinstance(cached, list) else None

    def memory_status(self, timeout=5, device=None):
        """Memory per card. With `device`, that card's reading, naming it;
        without, the first card's reading at the top level (as the
        single-worker server answered) plus `workers`, one entry per card.

        Live stats when a card's worker is idle; the run's last report while
        it is busy. The lock acquire is bounded: a job holds its card's lock
        for its whole duration, and a poll that raced a job start must fall
        back to the cached reading, not block for hours.

        `live` says whether `info` was measured by this call. `stale` and
        `reason` say why it was not, and `age_seconds` how old the cached
        reading is; `info` is null when there has never been a reading, which
        means nothing is resident rather than that the answer is unknown.

        Raises ValueError for a `device` this server has no worker on."""
        if device is not None:
            return self._slot_memory(self.slot_for(device), timeout)
        workers = [self._slot_memory(slot, timeout) for slot in self.slots]
        first = {key: value for key, value in workers[0].items() if key != "device"}
        return {**first, "workers": workers}

    def _slot_memory(self, slot, timeout):
        """One card's memory_status entry, naming the card."""
        return {"device": slot.ordinal(), **self._read_slot_memory(slot, timeout)}

    def _read_slot_memory(self, slot, timeout):
        if self.is_worker_busy(slot):
            return self._cached_memory("job_running", slot)
        if not slot.manager.worker_active:
            return self._cached_memory("worker_stopped", slot)
        if not slot.lock.acquire(timeout=2):
            return self._cached_memory("worker_busy", slot)
        try:
            reply = slot.manager.request(
                MemoryStatus(request_id=uuid.uuid4().hex), timeout
            )
        except (RuntimeError, queue.Empty) as e:
            # A dead worker is exactly when the last reading taken before it
            # died is worth the most, so report that rather than failing the
            # request. Without this the caller gets a 503 at the one moment
            # it most wants a number
            detail = slot.manager.crash_details()
            logger.warning(f"Worker unavailable for memory status: {detail or e}")
            if detail is not None:
                # crash_details only answers for a process the OS has reaped,
                # so a worker that is merely slow to reply keeps its state -
                # a timeout is not evidence of death
                slot.manager.mark_crashed()
            return self._cached_memory("worker_unreachable", slot)
        finally:
            slot.lock.release()
        if isinstance(reply, MemoryStatusReply):
            self._record_memory(reply.info, slot)
            return {
                "live": True,
                "info": slot.last_memory,
                "stale": False,
                "reason": None,
                "age_seconds": 0.0,
            }
        return self._cached_memory("worker_unreachable", slot)

    def clear_memory(self, timeout=30, device=None):
        """Drop every loaded pipeline and the step cache on each idle card -
        `device`'s alone when one is named - and report the readings taken
        right after. A card running a job is left alone: clearing under a
        run would rip out what it is using. The 30s timeout (vs.
        `memory_status`'s 5s) allows for this: actually freeing CUDA memory
        takes longer than reading a counter does.

        Returns `{cleared, info, workers}`, `workers` one entry per card:
        `{device, cleared: true, info}`, or `{device, cleared: false,
        reason: "job_running", job}` for a card running a job. `info` is
        the first cleared card's reading - null when nothing was resident,
        as with no worker running. With `device`, `{cleared, info, device}`
        for that card alone.

        Raises WorkerBusy when no card asked about is idle, ValueError for a
        `device` this server has no worker on, and RuntimeError when a
        worker does not answer with a clear."""
        slots = [self.slot_for(device)] if device is not None else self.slots
        workers = [self._clear_slot(slot, timeout) for slot in slots]
        cleared = [entry for entry in workers if entry["cleared"]]
        if not cleared:
            if device is not None:
                busy = workers[0]
                raise WorkerBusy(f"{busy['device']} is running job {busy['job']}")
            raise WorkerBusy("Every card is running a job")
        if device is not None:
            return {
                "cleared": True,
                "info": cleared[0]["info"],
                "device": cleared[0]["device"],
            }
        return {"cleared": True, "info": cleared[0]["info"], "workers": workers}

    def _clear_slot(self, slot, timeout):
        """Clear one card's worker, or say why it was left alone."""
        device = slot.ordinal()
        with self._lock:
            job_id = slot.current_job_id
        if job_id is not None:
            return {
                "device": device,
                "cleared": False,
                "reason": "job_running",
                "job": job_id,
            }
        if not slot.manager.worker_active:
            # Nothing to clear, and not a fault: the pipelines and the step
            # cache both live in the worker process, so no worker running
            # means both are already gone. An on-demand worker is
            # legitimately absent on an idle server (#206). A null `info` is
            # "no reading was taken", not a failure
            slot.last_identity = None
            return {"device": device, "cleared": True, "info": None}
        if not slot.lock.acquire(timeout=2):
            # A job was dispatched to it since the check above
            with self._lock:
                job_id = slot.current_job_id
            return {
                "device": device,
                "cleared": False,
                "reason": "job_running",
                "job": job_id,
            }
        try:
            reply = slot.manager.request(
                ClearMemory(request_id=uuid.uuid4().hex), timeout
            )
        finally:
            slot.lock.release()
        if not isinstance(reply, MemoryCleared):
            raise RuntimeError(f"unexpected worker reply: {reply.TYPE}")
        # Its pipelines and step cache are gone: nothing is warm there now
        slot.last_identity = None
        self._record_memory(reply.info, slot)
        return {"device": device, "cleared": True, "info": slot.last_memory}
