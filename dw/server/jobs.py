"""Job queue over the persistent worker process.

One runner thread executes jobs FIFO against the single GPU worker, managed
by WorkerManager. Jobs collect their progress events with
sequence numbers so an SSE client can attach late (or reconnect) and replay
from where it left off.
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
)
from ..worker_manager import WorkerManager
from ..workflow import SEED_BITS
from ..security import (
    SecurityError,
    validate_json_size,
    validate_output_path,
    validate_path,
    validate_workflow_path,
)
from ..realize import VARIABLE_PREFIX
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

logger = logging.getLogger("dw")

# Finished jobs kept in memory for SSE replay grace; older ones live in
# history only, so a long-running server's memory stays bounded
TERMINAL_JOBS_KEPT = 20


class JobManager:
    """Serializes job execution onto the one GPU worker process."""

    def __init__(
        self,
        output_dir,
        log_level="INFO",
        worker_manager=None,
        history_path=None,
        workflow_dir=None,
    ):
        self.output_dir = validate_output_path(output_dir, None)
        # Confines workflow_path/base_dir/sub-workflow resolution for every
        # job this manager submits - the server's configured workflow_dir
        self.workflow_dir = workflow_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_level = log_level
        self.worker_manager = worker_manager or WorkerManager()
        self.history = JobHistory(history_path or resolve_path("jobs.sqlite"))
        self.jobs = {}
        self.last_memory = None
        self.last_memory_at = None
        # Reentrant: cancel() finishes a queued job while holding it, and
        # _finish's terminal-job trim needs it again on the same thread
        self._lock = threading.RLock()  # guards job state transitions
        # Pending job ids in run order - a list, not a Queue, so the queue
        # can be reordered while jobs wait
        self._pending = []
        self._wake = threading.Condition(self._lock)
        self._worker_lock = threading.Lock()  # guards worker communication
        self._current_job_id = None
        self._stop = threading.Event()
        self._runner = threading.Thread(
            target=self._run_loop, daemon=True, name="job-runner"
        )
        self._runner.start()

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
        """The workflow JSON a job ran, for a read-only view of it.

        A live job answers from the snapshot admission checked, so the file
        moving, growing or breaking after submit changes nothing. A restored
        job has no snapshot (it is not persisted): an inline definition
        comes straight from the spec, and one launched from a path is
        re-read from disk, confined to the root the job ran against. None
        when there is no such job, or when the file it named has since
        moved, grown past the size limit or stopped parsing - a
        graph of the run is a nicety, never a reason to fail the page.
        """
        job = self.jobs.get(job_id)
        if job is not None:
            spec = job.spec
            snapshot = spec.get("definition")
            if snapshot is not None:
                return copy.deepcopy(snapshot)
        else:
            historical = self.history.get(job_id)
            if historical is None:
                return None
            spec = historical.get("spec") or {}
        inline = spec.get("workflow")
        if inline is not None:
            return copy.deepcopy(inline)
        path = spec.get("workflow_path")
        if not path:
            return None
        try:
            validated = validate_workflow_path(
                path, spec.get("workflow_dir") or self.workflow_dir
            )
            validate_json_size(validated)
            with open(validated, "r") as file:
                return json.load(file)
        except (SecurityError, OSError, ValueError):
            logger.debug(f"No workflow definition available for job {job_id}")
            return None

    def realized(self, job_id):
        """The realized workflow a job ran, or None when the job predates
        run tracking or its run directory no longer holds the file.

        Read from the job's own output directory, not the manager's: one
        server holds several workspaces, and a job carries the root it ran
        against. The join is confined to that root, so a run_dir read back
        out of the database cannot name anything outside it.
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
        try:
            root = validate_output_path(output_dir, None)
            path = validate_path(os.path.join(root, run_dir, REALIZED_FILE_NAME), root)
            validate_json_size(path)
            with open(path, "r") as file:
                return json.load(file)
        except (SecurityError, OSError, ValueError) as e:
            logger.debug(f"No realized workflow for job {job_id}: {e}")
            return None

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
        if not isinstance(seed, str) or not seed.startswith(VARIABLE_PREFIX):
            return None
        name = seed.removeprefix(VARIABLE_PREFIX)
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
        )

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
        """Cancel a queued or running job. Returns the job's status after the
        request, or None for an unknown job."""
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
            if job.status == RUNNING and self._current_job_id == job.id:
                try:
                    self.worker_manager.cancel()
                except Exception as e:
                    logger.warning(f"Could not send cancel for job {job_id}: {e}")
        return job.status

    def move(self, job_id, direction):
        """Reorder a queued job: 'up'/'down' swap with a neighbour,
        'front'/'back' go to the ends. Returns the new pending order, or
        None for a job that is not queued (finished, running, unknown)."""
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
        self._runner.join(timeout=5)
        self.worker_manager.shutdown_worker()

    # ---------------------------------------------------------------- runner

    def _run_loop(self):
        while not self._stop.is_set():
            with self._wake:
                while not self._pending and not self._stop.is_set():
                    self._wake.wait()
                if self._stop.is_set():
                    return
                job_id = self._pending.pop(0)
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                continue  # cancelled while waiting
            self._run_job(job)

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

    def _run_job(self, job):
        with self._worker_lock:
            with self._lock:
                if job.status != QUEUED:
                    return
                job.status = RUNNING
                job.started_at = time.time()
                self._current_job_id = job.id
            job.add_event({"event": "job_status", "status": RUNNING})
            try:
                self.worker_manager.ensure_worker(self.log_level)
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
                self.worker_manager.send_command(command.to_wire())
                outcome = self._consume_results(job)
            except Exception as e:
                logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                outcome = (FAILED, str(e), None)
            finally:
                # Cleared BEFORE the terminal status becomes visible - a
                # client seeing "succeeded" must find the manager idle
                with self._lock:
                    self._current_job_id = None
            if job.status not in TERMINAL_STATES:
                status, error, traceback_text = outcome
                self._finish(job, status, error=error, traceback_text=traceback_text)

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

    def _consume_results(self, job):
        """Read worker messages until the run ends; returns the terminal
        (status, error, traceback) for _run_job to apply once the manager
        no longer counts the job as current."""
        while True:
            try:
                message = self.worker_manager.get_result()
            except RuntimeError as e:
                # The worker died without managing to send anything - a
                # signal, not an exception, so worker_main's handler never
                # ran and there is no traceback to be had. The exit code is
                # the only diagnosis available, and marking the crash here
                # matters beyond this job: the manager would otherwise go on
                # believing a dead process is active, and every later call
                # that talks to it (memory_status above all) would fail
                # against a queue nobody is reading
                detail = self.worker_manager.crash_details()
                self.worker_manager.mark_crashed()
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
                self._record_run_memory(job, reply.info)
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
                self.worker_manager.mark_crashed()
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

    def _record_run_memory(self, job, info):
        """A memory reading the run reported. One per phase boundary now,
        not just once post-run (#273) - each folds into the cached reading
        memory_status() answers from while the job is busy, which is what
        makes that call fresh instead of a refusal for the run's whole
        duration."""
        self._record_memory(info)
        job.add_event({"event": "memory", "info": self.last_memory})
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
        """True while a job is running or queued - the window in which the
        worker may be reading model files a cache delete would rip out."""
        with self._lock:
            if self._current_job_id is not None:
                return True
            return any(job.status == QUEUED for job in self.jobs.values())

    def restart_worker_if_idle(self):
        """Shut the idle worker down so its next start picks up upgraded
        imports; the next job respawns it via ensure_worker. Returns False
        without touching a busy worker - a run in flight keeps the version
        it started with."""
        if self.is_busy():
            return False
        if not self._worker_lock.acquire(timeout=2):
            return False
        try:
            self.worker_manager.shutdown_worker()
            return True
        finally:
            self._worker_lock.release()

    # ---------------------------------------------------------------- memory

    def _record_memory(self, info):
        """Remember a reading and when it was taken, so a later cached answer
        can say how old it is."""
        self.last_memory = info
        self.last_memory_at = time.time() if info is not None else None

    def _cached_memory(self, reason):
        """The last reading, labelled with why it is not a live one. A caller
        comparing two readings must compare only `live: true` ones - a cached
        `info` was taken at another moment, and while a job loads a model it
        understates what is resident by however much has loaded since."""
        info = self.last_memory
        age = None
        if info is not None and self.last_memory_at is not None:
            age = round(time.time() - self.last_memory_at, 1)
        return {
            "live": False,
            "info": info,
            "stale": info is not None,
            "reason": reason,
            "age_seconds": age,
        }

    def probe_cache(self, command, timeout=5):
        """Which steps the worker's step cache would serve for `command` (the
        fields an execute command carries, minus its type), or None when the
        answer cannot be had right now - a job is running, the worker is
        busy, or it did not answer in time. Never blocks a request behind a
        running job, for the same reason memory_status does not.

        No worker running is a definite answer, not an unknown one: the
        cache lives in the worker process, so a worker that is not running
        holds nothing.
        """
        if self._current_job_id is not None:
            return None
        if not self.worker_manager.worker_active:
            return []
        if not self._worker_lock.acquire(timeout=2):
            return None
        # A probe that timed out still answers eventually, onto the same
        # queue the next request reads - so each carries an id and request()
        # discards every reply that is not its own, rather than reporting
        # the previous workflow's hit list as this plan's
        try:
            reply = self.worker_manager.request(
                ProbeCache(request_id=uuid.uuid4().hex, **command), timeout
            )
        except (RuntimeError, queue.Empty) as e:
            logger.debug(f"Worker did not answer the cache probe: {e}")
            return None
        finally:
            self._worker_lock.release()
        cached = getattr(reply, "cached", None)
        return list(cached) if isinstance(cached, list) else None

    def memory_status(self, timeout=5):
        """Live memory stats when the worker is idle; the run's last report
        while it is busy. The lock acquire is bounded: the runner holds
        _worker_lock for a job's whole duration, and a poll that raced a job
        start must fall back to the cached reading, not block for hours.

        `live` says whether `info` was measured by this call. `stale` and
        `reason` say why it was not, and `age_seconds` how old the cached
        reading is; `info` is null when there has never been a reading, which
        means nothing is resident rather than that the answer is unknown."""
        if self._current_job_id is not None:
            return self._cached_memory("job_running")
        if not self.worker_manager.worker_active:
            return self._cached_memory("worker_stopped")
        if not self._worker_lock.acquire(timeout=2):
            return self._cached_memory("worker_busy")
        try:
            reply = self.worker_manager.request(
                MemoryStatus(request_id=uuid.uuid4().hex), timeout
            )
        except (RuntimeError, queue.Empty) as e:
            # A dead worker is exactly when the last reading taken before it
            # died is worth the most, so report that rather than failing the
            # request. Without this the caller gets a 503 at the one moment
            # it most wants a number
            detail = self.worker_manager.crash_details()
            logger.warning(f"Worker unavailable for memory status: {detail or e}")
            if detail is not None:
                # crash_details only answers for a process the OS has reaped,
                # so a worker that is merely slow to reply keeps its state -
                # a timeout is not evidence of death
                self.worker_manager.mark_crashed()
            return self._cached_memory("worker_unreachable")
        finally:
            self._worker_lock.release()
        if isinstance(reply, MemoryStatusReply):
            self._record_memory(reply.info)
            return {
                "live": True,
                "info": self.last_memory,
                "stale": False,
                "reason": None,
                "age_seconds": 0.0,
            }
        return self._cached_memory("worker_unreachable")

    def clear_memory(self, timeout=30):
        """Drop every loaded pipeline and the step cache, then report the
        memory reading taken right after. Callers must check `is_busy()`
        first - this does not itself refuse a running/queued job, and racing
        one would clear state a queued run still expects resident. The
        30s timeout (vs. `memory_status`'s 5s) allows for this: actually
        freeing CUDA memory takes longer than reading a counter does.

        Returns the reading taken after the clear, or None when there was no
        worker to clear - nothing was resident in that case."""
        if not self.worker_manager.worker_active:
            # Nothing to clear, and not a fault: the pipelines and the step
            # cache both live in the worker process, so no worker running
            # means both are already gone. An on-demand worker is legitimately
            # absent on an idle server (#206), which this used to answer with
            # a 503 saying the worker was unavailable. None is "no reading
            # was taken", not a failure
            return None
        if not self._worker_lock.acquire(timeout=2):
            raise RuntimeError("worker busy")
        try:
            reply = self.worker_manager.request(
                ClearMemory(request_id=uuid.uuid4().hex), timeout
            )
        finally:
            self._worker_lock.release()
        if not isinstance(reply, MemoryCleared):
            raise RuntimeError(f"unexpected worker reply: {reply.TYPE}")
        self._record_memory(reply.info)
        return self.last_memory
