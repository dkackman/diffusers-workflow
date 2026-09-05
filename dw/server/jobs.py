"""Job queue over the persistent worker process.

One runner thread executes jobs FIFO against the single GPU worker - the
same WorkerManager the REPL uses. Jobs collect their progress events with
sequence numbers so an SSE client can attach late (or reconnect) and replay
from where it left off.
"""

import os
import copy
import json
import sqlite3
import time
import uuid
import logging
import threading

from ..repl_worker import WorkerManager
from ..workflow import workflow_from_file, workflow_from_definition
from ..introspection import workflow_argument_warnings
from ..security import validate_output_path
from ..settings import resolve_path

logger = logging.getLogger("dw")

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = (SUCCEEDED, FAILED, CANCELLED)

# The spec fields a rerun needs - shared by persistence and live rerun
RERUN_SPEC_KEYS = ("workflow_path", "workflow", "base_dir")

# Finished jobs kept in memory for SSE replay grace; older ones live in
# history only, so a long-running server's memory stays bounded
TERMINAL_JOBS_KEPT = 20

# A long run emits thousands of progress events; the tail is what explains
# the outcome. Bounded so history stays a summary store, not an event log
MAX_PERSISTED_EVENTS = 200


class JobHistory:
    """Finished jobs, persisted so the Jobs view survives server restarts.

    Records land at terminal state only - a crash mid-run loses that run's
    row, which is the right trade for never blocking the runner on disk.
    The last MAX_PERSISTED_EVENTS progress events ride along, so a job can
    still explain itself after a restart; everything earlier is dropped.
    """

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    workflow TEXT,
                    status TEXT,
                    created_at REAL,
                    started_at REAL,
                    finished_at REAL,
                    arguments TEXT,
                    spec TEXT,
                    manifest TEXT,
                    warnings TEXT,
                    error TEXT,
                    events TEXT
                )""")
            # Databases written before events were persisted are missing the
            # column; ALTER is the whole migration, and rows keep NULL
            columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "events" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN events TEXT")

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=5)

    def record(self, job):
        # The spec's workflow_name/warnings are derived; keep what rerun needs
        rerun_spec = {key: job.spec[key] for key in RERUN_SPEC_KEYS if key in job.spec}
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs (id, workflow, status, created_at,"
                " started_at, finished_at, arguments, spec, manifest, warnings,"
                " error, events) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.id,
                    job.workflow_name,
                    job.status,
                    job.created_at,
                    job.started_at,
                    job.finished_at,
                    json.dumps(job.spec.get("arguments", {}), default=str),
                    json.dumps(rerun_spec, default=str),
                    json.dumps(job.manifest, default=str),
                    json.dumps(job.warnings, default=str),
                    job.error,
                    json.dumps(job.events[-MAX_PERSISTED_EVENTS:], default=str),
                ),
            )

    def recent_summaries(self, limit=200):
        """Summary rows only - the jobs list is polled, and parsing four JSON
        blobs per row just to show six scalars was pure waste."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, workflow, status, created_at, started_at, finished_at"
                " FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "workflow": row[1],
                "status": row[2],
                "created_at": row[3],
                "started_at": row[4],
                "finished_at": row[5],
                "historical": True,
            }
            for row in rows
        ]

    def get(self, job_id):
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, workflow, status, created_at, started_at, finished_at,"
                " arguments, spec, manifest, warnings, error FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._to_detail(row) if row else None

    def events_for(self, job_id):
        """A finished job's persisted event tail. [] for a job recorded
        before events were kept, None for a job history has never seen -
        the caller needs to tell 'no events' from 'no such job'."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT events FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        if not row[0]:
            return []
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return []

    def job_for_file(self, file_name):
        """The most recent job that actually wrote this output file.

        LIKE metacharacters are escaped - generated names routinely contain
        '_', which would otherwise match any character and let a similarly
        named later job claim the file.

        A manifest entry marked 'reused' is a step-cache hit republishing an
        earlier run's files, so it is skipped: attribution belongs to the job
        that wrote the file, not to every later run that reused it.
        """
        escaped = (
            file_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        # Unbounded on purpose: every later fixed-seed rerun republishes the
        # file with 'reused', so a LIMIT would let the writing job fall out of
        # the window after that many reruns and leave the file unattributed.
        # The LIKE filter already restricts the scan to manifests naming it.
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, status, manifest FROM jobs WHERE manifest LIKE ? ESCAPE '\\'"
                " ORDER BY finished_at DESC",
                (f"%{escaped}%",),
            ).fetchall()
        for row in rows:
            if self._manifest_wrote(row[2], file_name):
                return {"id": row[0], "status": row[1]}
        return None

    @staticmethod
    def _manifest_wrote(manifest_text, file_name):
        """Whether this manifest names the file in an entry it wrote itself.

        A manifest that will not parse falls back to the LIKE match that
        found it - a row recorded before entries carried 'reused' cannot
        have been a reuse anyway.
        """
        try:
            manifest = json.loads(manifest_text)
        except (TypeError, ValueError):
            return True
        if not isinstance(manifest, list):
            return True
        # A manifest entry names a file the way the run recorded it - a
        # server-recorded manifest holds names relative to the output
        # directory (_relative_output_names), a directly-run workflow's holds
        # absolute paths. The caller names it relative to the output
        # directory, so match on the tail either way - the same relationship
        # the LIKE substring match relied on
        wanted = file_name.replace(os.sep, "/")

        def names_file(path):
            normalized = path.replace(os.sep, "/")
            return normalized == wanted or normalized.endswith("/" + wanted)

        return any(
            not entry.get("reused")
            and any(names_file(path) for path in entry.get("files") or [])
            for entry in manifest
            if isinstance(entry, dict)
        )

    @staticmethod
    def _to_detail(row):
        def parse(text, fallback):
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return fallback

        return {
            "id": row[0],
            "workflow": row[1],
            "status": row[2],
            "created_at": row[3],
            "started_at": row[4],
            "finished_at": row[5],
            "arguments": parse(row[6], {}),
            "spec": parse(row[7], {}),
            "manifest": parse(row[8], []),
            "warnings": parse(row[9], []),
            "error": row[10],
            "traceback": None,
            "event_count": 0,
            "historical": True,
        }


class Job:
    """One workflow execution request and everything observed about it."""

    def __init__(self, spec):
        self.id = uuid.uuid4().hex[:12]
        self.spec = spec
        self.workflow_name = spec.get("workflow_name", "unknown")
        self.status = QUEUED
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.manifest = []
        self.warnings = spec.get("warnings", [])
        self.error = None
        self.traceback = None
        self.events = []
        self.condition = threading.Condition()

    def add_event(self, event):
        with self.condition:
            self.events.append({"seq": len(self.events), **event})
            self.condition.notify_all()

    def finish(self, status, error=None, traceback_text=None):
        self.status = status
        self.finished_at = time.time()
        self.error = error
        self.traceback = traceback_text
        self.add_event({"event": "job_status", "status": status})

    def events_after(self, after_seq):
        # Clamped: an 'after' below -1 would slice from the END of the log
        # (events[-4:] for after=-5) and silently drop the earlier events a
        # client asking for everything expects
        after_seq = max(after_seq, -1)
        with self.condition:
            return self.events[after_seq + 1 :]

    def wait_for_event(self, after_seq, timeout):
        """Block until an event past after_seq exists or the job ends."""
        with self.condition:
            if len(self.events) > after_seq + 1 or self.status in TERMINAL_STATES:
                return
            self.condition.wait(timeout)

    def summary(self):
        return {
            "id": self.id,
            "workflow": self.workflow_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def detail(self):
        return {
            **self.summary(),
            "arguments": self.spec.get("arguments", {}),
            "warnings": self.warnings,
            "manifest": self.manifest,
            "error": self.error,
            "traceback": self.traceback,
            "event_count": len(self.events),
        }


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
        workflow_path=None,
        workflow=None,
        arguments=None,
        base_dir=None,
        workflow_dir=None,
    ):
        """Validate a job request and queue it. Raises ValueError on a bad
        request so the HTTP layer can answer 400 before anything runs.

        `workflow_dir` overrides this job's confinement root for a workflow
        that lives outside the writable directory - an example or a builtin,
        which the caller has already resolved against the search path. The
        worker re-validates against whatever this job records, so the
        override travels with the job rather than widening the manager.
        """
        arguments = arguments or {}
        if (workflow_path is None) == (workflow is None):
            raise ValueError("Provide exactly one of workflow_path or workflow")

        confinement = workflow_dir or self.workflow_dir

        if workflow_path is not None:
            # Loads and schema-validates now - a bad path or file fails the
            # request, not the queue
            loaded = workflow_from_file(workflow_path, self.output_dir, confinement)
            loaded.validate()
            spec = {
                "workflow_path": workflow_path,
                "workflow_name": loaded.name,
                "arguments": arguments,
                "workflow_dir": confinement,
            }
        else:
            # workflow_from_definition validates base_dir - it is HTTP-supplied
            # path input and goes through the security layer like every path
            loaded = workflow_from_definition(
                copy.deepcopy(workflow), self.output_dir, base_dir, self.workflow_dir
            )
            loaded.validate()
            spec = {
                "workflow": workflow,
                # Must match workflow_from_definition's fallback - the worker
                # re-validates this against workflow_dir
                "base_dir": base_dir
                or (
                    os.path.abspath(self.workflow_dir)
                    if self.workflow_dir
                    else os.getcwd()
                ),
                "workflow_name": loaded.name,
                "arguments": arguments,
                "workflow_dir": self.workflow_dir,
            }

        # Signature-level check of pipeline arguments - the typo that would
        # otherwise be a TypeError after the model loads becomes a warning
        # the client sees at submission
        spec["warnings"] = workflow_argument_warnings(loaded.workflow_definition)

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

    def rerun(self, job_id):
        """Queue a fresh job from a previous job's spec."""
        job = self.jobs.get(job_id)
        if job is not None:
            spec = {
                key: job.spec[key]
                for key in ("workflow_path", "workflow", "base_dir")
                if key in job.spec
            }
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
        return self.submit(
            workflow_path=spec.get("workflow_path"),
            workflow=spec.get("workflow"),
            arguments=arguments,
            base_dir=spec.get("base_dir"),
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

    def list(self):
        with self._lock:
            live = sorted(self.jobs.values(), key=lambda j: j.created_at)
            positions = {job_id: i for i, job_id in enumerate(self._pending)}
        live_ids = {job.id for job in live}
        summaries = []
        for job in live:
            summary = job.summary()
            if job.id in positions:
                summary["queue_position"] = positions[job.id]
            summaries.append(summary)
        for historical in self.history.recent_summaries():
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
                command = {
                    "type": "execute",
                    "arguments": job.spec["arguments"],
                    "output_dir": self.output_dir,
                    "log_level": self.log_level,
                }
                if "workflow_path" in job.spec:
                    command["workflow_path"] = job.spec["workflow_path"]
                else:
                    command["workflow"] = job.spec["workflow"]
                    command["base_dir"] = job.spec["base_dir"]
                command["workflow_dir"] = job.spec.get("workflow_dir")
                self.worker_manager.send_command(command)
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

    def _relative_output_names(self, paths):
        """The worker reports absolute paths; clients build '/outputs/<name>'
        URLs, and a workflow under a subfolder writes under
        '<output_dir>/<sub>/' (dw/workflow.py's effective_output_dir) - so
        every file is reported by its name relative to output_dir, with
        forward slashes. A path outside output_dir (a task step writing
        elsewhere) is left as it came."""
        names = []
        for path in paths:
            relative = os.path.relpath(path, self.output_dir)
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
            message = self.worker_manager.get_result()
            message_type = message.get("type")

            if message_type == "progress":
                event = {k: v for k, v in message.items() if k != "type"}
                if "files" in event:
                    event["files"] = self._relative_output_names(event["files"])
                job.add_event(event)
            elif message_type in ("output", "workflow_loaded"):
                text = message.get("message") or message.get("workflow_name", "")
                job.add_event({"event": "log", "message": text})
            elif message_type == "memory_info":
                self.last_memory = message.get("info")
                job.add_event({"event": "memory", "info": self.last_memory})
            elif message_type == "success":
                job.manifest = [
                    (
                        {**entry, "files": self._relative_output_names(entry["files"])}
                        if "files" in entry
                        else entry
                    )
                    for entry in message.get("manifest", [])
                ]
                return (SUCCEEDED, None, None)
            elif message_type == "cancelled":
                return (CANCELLED, None, None)
            elif message_type == "error":
                return (
                    FAILED,
                    message.get("message"),
                    message.get("traceback"),
                )
            elif message_type == "worker_crashed":
                self.worker_manager.mark_crashed()
                return (
                    FAILED,
                    f"Worker crashed: {message.get('message')}",
                    message.get("traceback"),
                )
            else:
                logger.warning(f"Unknown worker message type: {message_type}")

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

    def memory_status(self, timeout=5):
        """Live memory stats when the worker is idle; the run's last report
        while it is busy. The lock acquire is bounded: the runner holds
        _worker_lock for a job's whole duration, and a poll that raced a job
        start must fall back to the cached reading, not block for hours."""
        if self._current_job_id is not None:
            return {"live": False, "info": self.last_memory}
        if not self.worker_manager.worker_active:
            return {"live": False, "info": self.last_memory}
        if not self._worker_lock.acquire(timeout=2):
            return {"live": False, "info": self.last_memory}
        try:
            self.worker_manager.send_command({"type": "memory_status"})
            result = self.worker_manager.get_result(timeout=timeout)
        finally:
            self._worker_lock.release()
        if result.get("type") == "memory_status":
            self.last_memory = result.get("info")
            return {"live": True, "info": self.last_memory}
        return {"live": False, "info": self.last_memory}
