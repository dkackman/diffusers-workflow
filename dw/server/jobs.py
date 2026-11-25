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
- `dw/server/dispatch.py`: the queue scan picking the next job and card,
  with the host-RAM hold (`Dispatcher`, a mixin of JobManager);
- `dw/server/reruns.py`: a rerun of one job, and of a dead server's restart
  batch (`Reruns`, a mixin of JobManager);
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
import time
import uuid
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager

from ..events import NON_INTERRUPTIBLE_PHASES
from ..worker_protocol import Execute, workflow_identity
from ..worker_manager import WorkerManager
from ..security import (
    SecurityError,
    validate_json_size,
    validate_output_path,
    validate_path,
    validate_workflow_path,
)
from .. import model_use
from ..runs import MANIFEST_FILE_NAME, REALIZED_FILE_NAME
from ..settings import resolve_path
from ..workspace import DEFAULT_WORKSPACE_NAME
from . import dispatch, job_results, pool, worker_memory
from .changes import GALLERY as GALLERY_KIND
from .changes import JOBS as JOBS_KIND
from .changes import REQUESTS as REQUESTS_KIND
from .dispatch import Dispatcher
from .job_history import JobHistory, close_dead_servers
from .job_history import hold_server_lock
from .job_history import release_server_lock
from .outputs import job_run_directory, proof_output, with_run_note, with_run_state
from .job_record import (
    ACK_NONE,
    CANCELLED,
    FAILED,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
    Job,
    row_reason_fields,
)
from .pool import WorkerSlot
from .reruns import Reruns
from .worker_memory import WorkerBusy

__all__ = ["JobManager", "TERMINAL_JOBS_KEPT", "WorkerBusy", "WorkerSlot"]

logger = logging.getLogger("dw")

# Finished jobs kept in memory for SSE replay grace; older ones live in
# history only, so a long-running server's memory stays bounded
TERMINAL_JOBS_KEPT = 20
# Closed jobs whose proof output (`JobManager._proof`) is kept in memory: a
# polled requests list then reads each manifest from history once, not once
# per poll. About 150 bytes each.
PROOFS_KEPT = 4096

# How long a cancel waits for a worker in a phase with no checkpoint (a model
# load, a task) before the worker is terminated (#892)
CANCEL_GRACE_SECONDS = 30.0


def _submitted(arguments):
    """A linked job's arguments as a dict: a history row's stored JSON text
    parsed, a live job's dict as is; {} when there are none or the text
    does not parse to an object."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


class JobsActive(Exception):
    """A job is running or queued."""


class JobManager(Dispatcher, Reruns):
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
        # Stamped on each job's history row; its lock says it is alive (#1220)
        self.instance = uuid.uuid4().hex
        self._server_lock = hold_server_lock(self.history, self.instance)
        # Before the runner starts and before the app takes a request
        close_dead_servers(self.history, self.instance)
        self.jobs = {}
        self._proofs = OrderedDict()
        self._digests = OrderedDict()
        # Reentrant: cancel() finishes a queued job while holding it, and
        # _finish's terminal-job trim needs it again on the same thread
        self._lock = threading.RLock()  # guards job state transitions
        # Held across a restart-batch rerun, admission included, so two
        # concurrent batches cannot both pass its already-rerun check (#1266)
        self._batch_lock = threading.Lock()
        # workspace name -> the folder holding its requests.json, or None;
        # set by the app, which knows the workspaces. Unset, a rerun does
        # not check the link it inherits
        self.ledger_root_for = None
        # The app's workspace change counters (dw/server/changes.py), or
        # None for a manager built without an app
        self.changes = None
        # Pending job ids in run order - a list, not a Queue, so the queue
        # can be reordered while jobs wait
        self._pending = []
        self._wake = threading.Condition(self._lock)
        # The VRAM each job needs to start, by job id (dispatch reads it)
        self._needs = {}
        # The host RAM (MB) each job is projected to peak at, by job id - only
        # jobs with a figure; dispatch holds one the host can't fit (#869)
        self._host_needs = {}
        # The card a rerun prefers - the original's ordinal - by job id
        self._preferred = {}
        # Idle workers a held job asked released for host RAM, by job id,
        # and when each card's was last released (#1407)
        self._release_idle = {}
        self._released_at = {}
        # Callers holding dispatch (hold_dispatch): while any does, no queued
        # job is started, though jobs may still be submitted
        self._dispatch_holds = 0
        self._stop = threading.Event()
        # The thread running each dispatched job, joined by shutdown()
        self._job_threads = set()
        self._runner = threading.Thread(
            target=self._run_loop, daemon=True, name="job-runner"
        )
        self._runner.start()

    def queued_count(self):
        """How many jobs wait for a card - /api/health's `queued`. Read from
        the in-memory queue: `list()` walks the whole history and every run's
        note on disk, which stalled health while a job loaded (#1434)."""
        with self._lock:
            return len(self._pending)

    def running_job_id(self):
        """The oldest running job's id, or None while every card is idle -
        what /api/health reports as `current_job`; `workers()` names each
        card's."""
        with self._lock:
            running = [slot for slot in self.slots if slot.current_job_id]
            if not running:
                return None
            return min(running, key=lambda slot: slot.started_at or 0).current_job_id

    # ------------------------------------------------------- restart record

    def _record_open(self, job):
        """Write a live job's row - queued at submit, running at dispatch and
        at run_start. Logged, never raised: a missing row must not cost the
        job (#1220)."""
        try:
            self.history.record(job)
        except Exception as e:
            logger.warning(f"Could not record job {job.id} in history: {e}")

    def _on_run_start(self, job):
        """The worker's `run_start`: the row goes to running, and every Hub
        repo admission found the job names is stamped as used now
        (dw/model_use.py, #1445) - the run has started, so a job cancelled
        while it queued, or refused at dispatch, never stamps. The stamp is
        best-effort (`model_use.record` swallows its own failures)."""
        self._record_open(job)
        if job.repos_named:
            model_use.record(job.repos_named, time.time())

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
        host_ram_need=None,
        repos_named=None,
        preferred_device=None,
        note=None,
        rerun_of=None,
        overrides=None,
        request=None,
        item=None,
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

        `host_ram_need` is the admission's (`Admission.host_ram_need`): the
        host RAM the run is projected to hold at its peak, kept on the job
        as `job.host_ram_need` (#869). The job waits to start while the
        host cannot hold it beside the running jobs (`_host_ram_wait`).

        `repos_named` is the admission's (`Admission.repos_named`): the Hub
        repos the run names, stamped in dw/model_use.py when its run starts
        (`_run_on`, #1445). A job cancelled while queued stamps none.

        `preferred_device` is the card ('cuda:1') the job goes to when that
        card is free and fits it - a rerun's original card (`rerun`).

        `note` is the run's note, already cleaned (`admitted_note`): kept on
        the job and written into the run directory when the run opens.
        `request` and `item` link the job to a requests-ledger item (#1458).

        `rerun_of` is the job this one reruns, recorded so the list can say so;
        `overrides` the variables the rerun changed, recorded beside it.

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
        spec["warnings"] = list(warnings or [])
        # Each recorded only when given
        optional = dict(asset_dir=asset_dir, note=note, rerun_of=rerun_of)
        optional["overrides"] = overrides
        if request and item:  # admission checked it; a rerun's, `rerun` warns on
            optional.update(request=request, item=item)
        spec.update({key: value for key, value in optional.items() if value})

        job = Job(spec)
        job.host_ram_need = host_ram_need
        job.repos_named = list(repos_named or [])
        job.server_instance = self.instance
        # Its row before anyone can see it: no outcome is written first (#1220)
        self._record_open(job)
        with self._lock:
            self.jobs[job.id] = job
            self._needs[job.id] = pool.dispatch_need(self.slots, vram_need)
            if host_ram_need:
                self._host_needs[job.id] = host_ram_need[0]
            if preferred_device:
                self._preferred[job.id] = preferred_device
        job.add_event({"event": "job_status", "status": QUEUED})
        with self._wake:
            self._pending.append(job.id)
            self._wake.notify()
        logger.info(f"Queued job {job.id} for workflow {job.workflow_name}")
        self._changed(job)
        return job

    def _changed(self, job, *kinds):
        """A job's status moved: bump `jobs` (and `kinds`) in its workspace,
        and `requests` too when it is linked to a request item, since the
        ledger shows each item's jobs. Never raises: a counter is a hint."""
        if self.changes is None:
            return
        try:
            kinds = (JOBS_KIND, *kinds)
            if getattr(job, "request", None):
                kinds = (*kinds, REQUESTS_KIND)
            self.changes.bump(job.spec.get("workspace"), *kinds)
        except Exception as e:
            logger.warning(f"Could not bump change counters for job {job.id}: {e}")

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

    def request_jobs(self, workspace, request_id):
        """Every job linked to `request_id` in `workspace`, live queue and
        history, oldest first, as `{job, item, status, run_id, run_dir,
        run_exists, started_at, finished_at, created_at, note, reason,
        workflow, arguments, output, prompt_digests}` - `reason` a failed or
        cancelled job's first error line, `workflow` the catalog name (None
        for an inline one), `arguments` as submitted, `output` and
        `prompt_digests` callables (no manifest is read for a job nobody
        shows), `run_exists` whether the run directory `delete_job_run`
        removes is still on disk. One indexed history query."""
        found = {}
        for row in self.history.request_jobs(workspace, request_id):
            found[row["id"]] = self._linked(
                row["id"], row, row["run_dir"], row["output_dir"]
            )
        with self._lock:
            live = [
                job
                for job in self.jobs.values()
                if job.request == request_id
                and (job.spec.get("workspace") or DEFAULT_WORKSPACE_NAME) == workspace
            ]
        for job in live:
            found[job.id] = self._linked(
                job.id,
                {
                    "item": job.item,
                    "status": job.status,
                    "run_id": job.run_id,
                    "note": job.note,
                    "error": job.error,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "workflow_name": job.catalog_name,
                    "arguments": job.spec.get("arguments"),
                },
                job.run_dir,
                job.spec.get("output_dir"),
            )
        return sorted(found.values(), key=lambda entry: entry["created_at"] or 0)

    def _linked(self, job_id, row, run_dir, output_dir):
        directory = None
        if run_dir:
            try:
                root = validate_output_path(output_dir or self.output_dir, None)
                directory = job_run_directory(root, run_dir)
            except SecurityError:
                directory = None
        exists = directory is not None
        return {
            "job": job_id,
            "item": row["item"],
            "status": row["status"],
            "run_id": row["run_id"],
            "run_dir": run_dir,
            "run_exists": exists,
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "created_at": row["created_at"],
            "note": row["note"],
            "reason": row_reason_fields(row["status"], row.get("error")).get("reason"),
            # The take's plan as submitted (`ledger_views._current` compares)
            "workflow": row.get("workflow_name"),
            "arguments": _submitted(row.get("arguments")),
            "output": lambda: self._proof(job_id),
            # {prompt name: sha256 of the text the run inlined}, from its
            # run manifest - None for a run that recorded none (one from
            # before #1558, or one whose directory is gone). A callable, so
            # only a take the status rule reaches is read
            "prompt_digests": lambda: self._prompt_digests(job_id, directory),
        }

    def _prompt_digests(self, job_id, directory):
        """A take's recorded prompt digests (`workflow.prompt_digests` in its
        manifest.json), read once per job and kept (`PROOFS_KEPT` of them):
        the status rule only asks a succeeded take, whose manifest is final.
        None when the run recorded none."""
        with self._lock:
            if job_id in self._digests:
                self._digests.move_to_end(job_id)
                return self._digests[job_id]
        digests = None
        if directory is not None:
            try:
                with open(
                    os.path.join(directory, MANIFEST_FILE_NAME), "r", encoding="utf-8"
                ) as file:
                    recorded = (json.load(file).get("workflow") or {}).get(
                        "prompt_digests"
                    )
                if isinstance(recorded, dict):
                    digests = {
                        str(name): value
                        for name, value in recorded.items()
                        if isinstance(value, str)
                    }
            except (OSError, ValueError, AttributeError):
                digests = None
        with self._lock:
            live = self.jobs.get(job_id)
            if live is None or live.status == "succeeded":
                self._digests[job_id] = digests
                while len(self._digests) > PROOFS_KEPT:
                    self._digests.popitem(last=False)
        return digests

    def _proof(self, job_id):
        """A linked job's `proof_output`: a live job's from the manifest it
        holds, a closed one's from its history row - read once and kept
        (`PROOFS_KEPT` of them), since that row never changes."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None and job_id in self._proofs:
                self._proofs.move_to_end(job_id)
                return self._proofs[job_id]
        if job is not None:
            return proof_output(job.manifest)
        proof = proof_output(self.history.manifest_of(job_id))
        with self._lock:
            self._proofs[job_id] = proof
            while len(self._proofs) > PROOFS_KEPT:
                self._proofs.popitem(last=False)
        return proof

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

    def list(self, workspace=None, statuses=None, request=None, item=None):
        """All jobs, live and historical, sorted by creation. `workspace` filters to one workspace; omitted, the list
        spans every workspace the server holds, unchanged from before
        workspaces existed. `statuses` filters to a set of job states
        ('queued', 'running', 'succeeded', 'failed', 'cancelled'); omitted,
        every state is listed."""
        return self.list_page(workspace, statuses, request=request, item=item)[0]

    def list_page(
        self, workspace=None, statuses=None, limit=None, request=None, item=None
    ):
        """`list`'s jobs cut to the newest `limit` (every one when None),
        and how many matched before the cut. Each run's current note is
        read after the cut, so a small page does not read the notes of
        the whole history."""
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
            # `request`/`item` filter in SQL for history, here for the live
            if request and summary.get("request") != request:
                continue
            if item and summary.get("item") != item:
                continue
            if job.id in positions:
                summary["queue_position"] = positions[job.id]
            output_dir = job.spec.get("output_dir")
            summaries.append((summary, output_dir, job.run_dir))
        # only the newest `limit` rows can reach the page; the live ids are
        # dropped below, so fetch that many extra. The total is counted in
        # SQL, not from the capped list (#1466)
        live_count = len(summaries)
        history_limit = None if limit is None else limit + len(live_ids)
        history_total = self.history.count_summaries(
            workspace=workspace,
            statuses=statuses,
            exclude_ids=live_ids,
            request=request,
            item=item,
        )
        for historical in self.history.recent_summaries(
            limit=history_limit if history_limit is not None else -1,
            workspace=workspace,
            statuses=statuses,
            request=request,
            item=item,
        ):
            run_dir = historical.pop("_run_dir", None)
            output_dir = historical.pop("_output_dir", None)
            if historical["id"] not in live_ids:
                summaries.append((historical, output_dir, run_dir))
        summaries.sort(key=lambda entry: entry[0]["created_at"] or 0)
        total = live_count + history_total
        if limit is not None:
            # the newest are the interesting ones and the list is oldest
            # first, so the cut comes off the front; max(0, ...) because a
            # bare negative start would be read from the end instead
            summaries = summaries[max(0, len(summaries) - limit) :] if limit else []
        return [self._current_note(*entry) for entry in summaries], total

    def _current_note(self, summary, output_dir, run_dir):
        """The summary with its run's current note (outputs.with_run_note),
        when the run directory is still there."""
        if not run_dir:
            return summary
        root = validate_output_path(output_dir or self.output_dir, None)
        return with_run_note(summary, job_run_directory(root, run_dir))

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
                self._forget_dispatch(job)
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
                self._escalate_cancel(slot, job)
        return job.status

    def _escalate_cancel(self, slot, job):
        """A model load or a task step never reaches the checkpoint that
        honors a cancel, and a cold download can last hours. Once the grace
        period has passed and the job is still in such a phase, the worker
        is terminated and the job ends cancelled (#892); it respawns for the
        next job."""
        timer = threading.Timer(
            CANCEL_GRACE_SECONDS, self._terminate_if_stuck, args=(slot, job)
        )
        timer.daemon = True
        timer.start()

    def _terminate_if_stuck(self, slot, job):
        with self._lock:
            stuck = (
                job.status == RUNNING
                and slot.current_job_id == job.id
                and job.phase in NON_INTERRUPTIBLE_PHASES
            )
        if not stuck:
            return
        job.add_event(
            {
                "event": "log",
                "message": f"Cancel still pending in phase '{job.phase}' after "
                f"{CANCEL_GRACE_SECONDS:g}s: terminating the worker",
            }
        )
        try:
            slot.manager.terminate_worker()
        except Exception as e:
            logger.warning(f"Could not terminate the worker for job {job.id}: {e}")

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
            # The order decides which job the next pass tries first, a held
            # one included (#869)
            self._wake.notify()
            order = list(self._pending)
        self._changed(self.jobs[job_id])
        return order

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
        self._host_claims().withdraw()
        release_server_lock(self._server_lock)
        self._server_lock = None

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
                    if self._release_idle:
                        self._release_idle_workers()
                        continue
                    # Host RAM frees outside dw too, so a job held for it is
                    # looked at again on a timer, not only on a wake (#869)
                    held = any(self._held_for_host_ram())
                    self._wake.wait(dispatch.HOST_RAM_RECHECK_SECONDS if held else None)
                slot, job = picked
                job.status = RUNNING
                job.started_at = time.time()
                job.device_ordinal, job.device_card = slot.device_fields()
                slot.current_job_id = job.id
                slot.started_at = job.started_at
                # The worker switches to this identity for the run
                slot.last_identity = self._job_identity(job)
            job.add_event({"event": "job_status", "status": RUNNING})
            # Before the job's thread exists, so _finish always writes last
            self._record_open(job)
            self._changed(job)
            thread = threading.Thread(
                target=self._run_job,
                args=(slot, job),
                daemon=True,
                name=f"job-{job.id}",
            )
            with self._lock:
                self._job_threads.add(thread)
            thread.start()

    def _slot_running(self, job_id):
        """The slot whose worker is running `job_id`, or None."""
        with self._lock:
            for slot in self.slots:
                if slot.current_job_id == job_id:
                    return slot
        return None

    def _finish(self, job, status, error=None, traceback_text=None):
        job.finish(status, error=error, traceback_text=traceback_text)
        self._stamp_roles(job)
        try:
            self.history.record(job)
        except Exception as e:
            logger.warning(f"Could not persist job {job.id}: {e}")
        self._trim_terminal_jobs()
        if job.manifest:
            self._changed(job, GALLERY_KIND)
        else:
            self._changed(job)

    def _stamp_roles(self, job):
        """Put each manifest entry's role on the job while its run directory
        is read, so history serves the entries a live job does (#957) - the
        run may be moved before the job is read back. Roles only: no note."""
        try:
            location = self.run_location(job.id)
            if location is not None and isinstance(job.manifest, list):
                job.manifest = with_run_state(
                    {"manifest": job.manifest}, *location, note=False
                )["manifest"]
        except Exception as e:
            logger.warning(f"Could not read roles for job {job.id}: {e}")

    def forget_workspace(self, workspace):
        """Drop a deleted workspace's finished jobs, in memory and in history
        (#913), so a workspace later created under the same name lists none
        of them. Queued or running jobs are the caller's to refuse first."""
        with self._lock:
            for job in [
                j
                for j in self.jobs.values()
                if j.status in TERMINAL_STATES and j.spec.get("workspace") == workspace
            ]:
                del self.jobs[job.id]
        return self.history.forget_workspace(workspace)

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
                    self._start_worker(slot, job)
                    self._rank_for_oom(slot, job)
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
                        on_run_start=lambda: self._on_run_start(job),
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

    def is_busy(self):
        """True while a job is running on any card or queued - the window in
        which a worker may be reading model files a cache delete would rip
        out."""
        with self._lock:
            if any(slot.current_job_id is not None for slot in self.slots):
                return True
            return any(job.status == QUEUED for job in self.jobs.values())

    @contextmanager
    def hold_dispatch(self):
        """Hold the queue still while the caller changes files a job writes
        or reads. Raises JobsActive if a job is running or queued; otherwise
        no job starts until the block exits - one submitted meanwhile waits.
        The check and the hold are one step under the lock, so no job can
        slip in between them."""
        with self._lock:
            if self.is_busy():
                raise JobsActive()
            self._dispatch_holds += 1
        try:
            yield
        finally:
            with self._wake:
                self._dispatch_holds -= 1
                self._wake.notify()

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

        def workspace_of(job_id):
            job = self.jobs.get(job_id)
            return job.spec.get("workspace") if job else None

        return worker_memory.describe_workers(self.slots, self._lock, workspace_of)

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
