"""Job queue over the persistent worker process.

One runner thread executes jobs FIFO against the single GPU worker - the
same WorkerManager the REPL uses. Jobs collect their progress events with
sequence numbers so an SSE client can attach late (or reconnect) and replay
from where it left off.
"""

import os
import copy
import queue
import time
import uuid
import logging
import threading

from ..repl_worker import WorkerManager
from ..workflow import workflow_from_file, Workflow
from ..introspection import workflow_argument_warnings
from ..security import validate_output_path

logger = logging.getLogger("dw")

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = (SUCCEEDED, FAILED, CANCELLED)


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

    def __init__(self, output_dir, log_level="INFO", worker_manager=None):
        self.output_dir = validate_output_path(output_dir, None)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_level = log_level
        self.worker_manager = worker_manager or WorkerManager()
        self.jobs = {}
        self.last_memory = None
        self._queue = queue.Queue()
        self._lock = threading.Lock()  # guards job state transitions
        self._worker_lock = threading.Lock()  # guards worker communication
        self._current_job_id = None
        self._stop = threading.Event()
        self._runner = threading.Thread(
            target=self._run_loop, daemon=True, name="job-runner"
        )
        self._runner.start()

    # ------------------------------------------------------------- submission

    def submit(self, workflow_path=None, workflow=None, arguments=None, base_dir=None):
        """Validate a job request and queue it. Raises ValueError on a bad
        request so the HTTP layer can answer 400 before anything runs."""
        arguments = arguments or {}
        if (workflow_path is None) == (workflow is None):
            raise ValueError("Provide exactly one of workflow_path or workflow")

        if workflow_path is not None:
            # Loads and schema-validates now - a bad path or file fails the
            # request, not the queue
            loaded = workflow_from_file(workflow_path, self.output_dir)
            loaded.validate()
            spec = {
                "workflow_path": workflow_path,
                "workflow_name": loaded.name,
                "arguments": arguments,
            }
        else:
            base_dir = base_dir or os.getcwd()
            loaded = Workflow(
                copy.deepcopy(workflow),
                self.output_dir,
                os.path.join(base_dir, "__inline__.json"),
            )
            loaded.validate()
            spec = {
                "workflow": workflow,
                "base_dir": base_dir,
                "workflow_name": loaded.name,
                "arguments": arguments,
            }

        # Signature-level check of pipeline arguments - the typo that would
        # otherwise be a TypeError after the model loads becomes a warning
        # the client sees at submission
        spec["warnings"] = workflow_argument_warnings(loaded.workflow_definition)

        job = Job(spec)
        with self._lock:
            self.jobs[job.id] = job
        job.add_event({"event": "job_status", "status": QUEUED})
        self._queue.put(job.id)
        logger.info(f"Queued job {job.id} for workflow {job.workflow_name}")
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def list(self):
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created_at)
        return [job.summary() for job in jobs]

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
                job.finish(CANCELLED)
                return job.status
            if job.status == RUNNING and self._current_job_id == job.id:
                try:
                    self.worker_manager.cancel()
                except Exception as e:
                    logger.warning(f"Could not send cancel for job {job_id}: {e}")
        return job.status

    def shutdown(self):
        self._stop.set()
        self._queue.put(None)
        self._runner.join(timeout=5)
        self.worker_manager.shutdown_worker()

    # ---------------------------------------------------------------- runner

    def _run_loop(self):
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                continue
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                continue  # cancelled while waiting
            self._run_job(job)

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
                self.worker_manager.send_command(command)
                self._consume_results(job)
            except Exception as e:
                logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                if job.status not in TERMINAL_STATES:
                    job.finish(FAILED, error=str(e))
            finally:
                with self._lock:
                    self._current_job_id = None

    def _consume_results(self, job):
        while True:
            message = self.worker_manager.get_result()
            message_type = message.get("type")

            if message_type == "progress":
                event = {k: v for k, v in message.items() if k != "type"}
                job.add_event(event)
            elif message_type in ("output", "workflow_loaded"):
                text = message.get("message") or message.get("workflow_name", "")
                job.add_event({"event": "log", "message": text})
            elif message_type == "memory_info":
                self.last_memory = message.get("info")
                job.add_event({"event": "memory", "info": self.last_memory})
            elif message_type == "success":
                job.manifest = message.get("manifest", [])
                job.finish(SUCCEEDED)
                return
            elif message_type == "cancelled":
                job.finish(CANCELLED)
                return
            elif message_type == "error":
                job.finish(
                    FAILED,
                    error=message.get("message"),
                    traceback_text=message.get("traceback"),
                )
                return
            elif message_type == "worker_crashed":
                self.worker_manager.worker_active = False
                self.worker_manager.worker_process = None
                job.finish(
                    FAILED,
                    error=f"Worker crashed: {message.get('message')}",
                    traceback_text=message.get("traceback"),
                )
                return
            else:
                logger.warning(f"Unknown worker message type: {message_type}")

    # ---------------------------------------------------------------- memory

    def memory_status(self, timeout=5):
        """Live memory stats when the worker is idle; the run's last report
        while it is busy (the runner owns the queues during a job)."""
        if self._current_job_id is not None:
            return {"live": False, "info": self.last_memory}
        if not self.worker_manager.worker_active:
            return {"live": False, "info": self.last_memory}
        with self._worker_lock:
            self.worker_manager.send_command({"type": "memory_status"})
            result = self.worker_manager.get_result(timeout=timeout)
        if result.get("type") == "memory_status":
            self.last_memory = result.get("info")
            return {"live": True, "info": self.last_memory}
        return {"live": False, "info": self.last_memory}
