"""Job queue over the persistent worker processes - one per card.

The server runs one worker per `--devices` entry (a WorkerSlot each, its
process managed by a WorkerManager). A dispatcher thread hands each free
card the first queued job that fits it: queue order, except that a job too
big for every free card waits while a smaller one behind it takes the card
(backfill). Each running job has its own thread, so a crash or a cancel
reaches only the worker it names (#462). With one device this is the old
single FIFO worker.

Among the free cards a job fits, it goes to the one it has an affinity for
(`pool.choose_slot`): a rerun to the card that ran the original, then any
job to the card whose worker last ran the same workflow identity - that
worker's pipelines and step cache are warm, and a worker frees both whenever
the identity changes. A cache probe and a plan's estimate ask about the card
`route` names, the one the job would be dispatched to now. Memory is per
card too: each slot keeps its own last reading.

This module owns the queue, the dispatcher and the job history readers. The
rest lives beside it, and JobManager keeps a thin delegator for every name a
route or test calls:
- `dw/server/pool.py`: WorkerSlot and the lock-free fit and affinity policy;
- `dw/server/job_results.py`: folding a run's worker messages into its Job;
- `dw/server/worker_memory.py`: a card's memory reading, cache probe and
  memory clear.

`JobManager.slots` is the pool. `worker_manager` is the first card's worker,
the single-device default; `running_job_id()` is the oldest running job.

Jobs collect their progress events with sequence numbers so an SSE client
can attach late (or reconnect) and replay from where it left off.
"""

import os
import copy
import json
import secrets
import time
import logging
import threading

from ..worker_protocol import Execute, workflow_identity
from ..devices import device_ordinal
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
from . import job_results, pool, worker_memory
from .job_history import JobHistory
from .job_record import (
    ACK_NONE,
    CANCELLED,
    FAILED,
    QUEUED,
    RERUN_SPEC_KEYS,
    RUNNING,
    TERMINAL_STATES,
    Job,
)
from .pool import WorkerSlot
from .worker_memory import WorkerBusy

__all__ = ["JobManager", "TERMINAL_JOBS_KEPT", "WorkerBusy", "WorkerSlot"]

logger = logging.getLogger("dw")

# Finished jobs kept in memory for SSE replay grace; older ones live in
# history only, so a long-running server's memory stays bounded
TERMINAL_JOBS_KEPT = 20


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

    def running_job_id(self):
        """The oldest running job's id, or None while every card is idle -
        what /api/health reports as `current_job`; `workers()` names each
        card's."""
        with self._lock:
            running = [slot for slot in self.slots if slot.current_job_id]
            if not running:
                return None
            return min(running, key=lambda slot: slot.started_at or 0).current_job_id

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
        the cards this server has (pool.slot_for)."""
        return pool.slot_for(self.slots, device)

    def route(self, identity=None, need=None):
        """The slot a job with `identity` and VRAM `need` (admission's
        `vram_need`) would be dispatched to now: a free card it fits when
        there is one, else the card it would wait for. What a cache probe
        and a plan's estimate ask about."""
        need = pool.dispatch_need(self.slots, need)
        with self._lock:
            fitting = [slot for slot in self.slots if slot.fits(need)] or self.slots
            free = [slot for slot in fitting if slot.current_job_id is None]
            return pool.choose_slot(free or fitting, identity)

    # ------------------------------------------------------------- VRAM fit

    def largest_ceiling(self):
        """(GB, label) of the largest card here (pool.largest_ceiling)."""
        return pool.largest_ceiling(self.slots)

    def largest_ceiling_gb(self):
        """The most VRAM any card here can be held to, or None
        (pool.largest_ceiling_gb)."""
        return pool.largest_ceiling_gb(self.slots)

    def _unfit_message(self, need):
        """Why no card here can ever run a job needing `need`, or None
        (pool.unfit_message)."""
        return pool.unfit_message(self.slots, need)

    def check_fits(self, admission):
        """Raise ValueError - a 400 at the route - for a request that is
        not admissible, or too big for every card here (pool.check_fits)."""
        pool.check_fits(self.slots, admission)

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
            self._needs[job.id] = pool.dispatch_need(self.slots, vram_need)
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
            return job.device_ordinal
        return self.history.device_ordinal(job_id)

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
                self._needs.pop(job.id, None)
                self._preferred.pop(job.id, None)
                self._finish(job, CANCELLED)
                return job.status
            slot = self._slot_running(job.id)
            if job.status == RUNNING and slot is not None:
                # Recorded first: the job's thread may not have sent Execute
                # yet, and a cancel reaching an idle worker is ignored
                job.cancel_requested = True
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
                job.device_ordinal, job.device_card = slot.device_fields()
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
                slot = pool.choose_slot(
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
                if job.status != RUNNING or job.cancel_requested:
                    # Cancelled between dispatch and here: nothing was sent,
                    # so the outcome is the cancel itself
                    outcome = (CANCELLED, None, None)
                else:
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
                    if job.cancel_requested:
                        # Asked while the worker was still idle; now it has
                        # the job, and a second cancel lands behind it
                        manager.cancel()
                    outcome = job_results.consume_results(
                        job,
                        manager,
                        slot,
                        job.spec.get("output_dir") or self.output_dir,
                        worker_memory.record_memory,
                    )
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
                "name": slot.device_fields()[1],
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
        """Keep `info` as a card's reading - the first card's by default
        (worker_memory.record_memory)."""
        worker_memory.record_memory(info, slot or self.slots[0])

    def probe_cache(self, command, timeout=5, slot=None):
        """Which steps a worker's step cache would serve for `command`, or
        None when that cannot be had now (worker_memory.probe_cache).

        The worker asked is `slot`'s, else the one `route` would dispatch
        this workflow to: each card has its own step cache."""
        if slot is None:
            slot = self.route(
                self.identity_of(
                    command.get("source"),
                    command.get("file_spec"),
                    command.get("definition"),
                )
            )
        return worker_memory.probe_cache(slot, self._lock, command, timeout)

    def memory_status(self, timeout=5, device=None):
        """Memory per card, or `device`'s alone (worker_memory.memory_status).
        Raises ValueError for a `device` this server has no worker on."""
        return worker_memory.memory_status(self.slots, self._lock, timeout, device)

    def clear_memory(self, timeout=30, device=None):
        """Clear each idle card's pipelines and step cache, or `device`'s
        alone (worker_memory.clear_memory). Raises WorkerBusy when no card
        asked about is idle."""
        return worker_memory.clear_memory(self.slots, self._lock, timeout, device)
