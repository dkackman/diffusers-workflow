"""Job queue over the persistent worker process.

One runner thread executes jobs FIFO against the single GPU worker - the
same WorkerManager the REPL uses. Jobs collect their progress events with
sequence numbers so an SSE client can attach late (or reconnect) and replay
from where it left off.
"""

import os
import copy
import json
import queue
import random
import sqlite3
import time
import uuid
import logging
import threading

from ..repl_worker import WorkerManager
from ..workflow import workflow_from_file, workflow_from_definition
from ..introspection import workflow_argument_warnings
from ..variables import argument_errors
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

logger = logging.getLogger("dw")

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = (SUCCEEDED, FAILED, CANCELLED)

# The spec fields a rerun needs - shared by persistence and live rerun
RERUN_SPEC_KEYS = (
    "workflow_path",
    "workflow",
    "base_dir",
    # A rerun belongs in the workspace the original ran in, so the roots
    # that decided that are part of what history keeps
    "workspace",
    "output_dir",
    "asset_dir",
    # so a rerun is attributed to the same catalog entry
    "catalog_name",
    "workflow_dir",
)

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
            # Every row predating workspaces belongs to the default one -
            # history that cannot say which workspace a job ran in stops
            # making sense the moment there are two
            if "workspace" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN workspace TEXT DEFAULT 'default'"
                )
                connection.execute(
                    "UPDATE jobs SET workspace = 'default' WHERE workspace IS NULL"
                )
            # The catalog name the job was run from, beside `workflow` (the
            # definition's id). Ids are not unique across a catalog forever;
            # names are, and a later runtime-by-workflow join wants the exact
            # one. Rows before this column stay NULL: old history is
            # unjoinable, new history is exact
            if "workflow_name" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN workflow_name TEXT")
            # Which run of the workflow this job was - the directory under the
            # output root that holds its manifest and its realized workflow.
            # NULL for every row predating run tracking, and the manager
            # refuses to guess one from file paths
            if "run_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN run_id TEXT")
            if "run_dir" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN run_dir TEXT")

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=5)

    def record(self, job):
        # The spec's workflow_name/warnings are derived; keep what rerun needs
        rerun_spec = {key: job.spec[key] for key in RERUN_SPEC_KEYS if key in job.spec}
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs (id, workflow, status, created_at,"
                " started_at, finished_at, arguments, spec, manifest, warnings,"
                " error, events, workspace, workflow_name, run_id, run_dir) VALUES"
                " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    job.spec.get("workspace") or DEFAULT_WORKSPACE_NAME,
                    job.catalog_name,
                    job.run_id,
                    job.run_dir,
                ),
            )

    def recent_summaries(self, limit=200, workspace=None, statuses=None):
        """Summary rows only - the jobs list is polled, and parsing four JSON
        blobs per row just to show six scalars was pure waste.

        `workspace` filters to one workspace's rows; omitted, history spans
        all of them the way the list already did before workspaces existed.
        `statuses` filters to a set of terminal states - in SQL rather than
        over the returned rows, or the newest-first cap above would be
        spent on rows the filter then drops.
        """
        query = (
            "SELECT id, workflow, status, created_at, started_at, finished_at,"
            " workspace, workflow_name, run_id FROM jobs"
        )
        params = []
        clauses = []
        if workspace:
            clauses.append("workspace = ?")
            params.append(workspace)
        if statuses:
            statuses = list(statuses)
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "id": row[0],
                "workflow": row[1],
                "status": row[2],
                "created_at": row[3],
                "started_at": row[4],
                "finished_at": row[5],
                "workspace": row[6] or DEFAULT_WORKSPACE_NAME,
                "workflow_name": row[7],
                "run_id": row[8],
                "historical": True,
            }
            for row in rows
        ]

    def get(self, job_id):
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, workflow, status, created_at, started_at, finished_at,"
                " arguments, spec, manifest, warnings, error, workspace,"
                " workflow_name, run_id, run_dir FROM jobs WHERE id = ?",
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

    def job_for_file(self, file_name, workspace=None):
        """The most recent job that actually wrote this output file.

        LIKE metacharacters are escaped - generated names routinely contain
        '_', which would otherwise match any character and let a similarly
        named later job claim the file.

        A manifest entry marked 'reused' is a step-cache hit republishing an
        earlier run's files, so it is skipped: attribution belongs to the job
        that wrote the file, not to every later run that reused it.

        `workspace` narrows the scan to one workspace - two workspaces can
        each produce a file with the same relative name, and without this a
        later job in another workspace could wrongly claim the match.
        """
        escaped = (
            file_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        # Unbounded on purpose: every later fixed-seed rerun republishes the
        # file with 'reused', so a LIMIT would let the writing job fall out of
        # the window after that many reruns and leave the file unattributed.
        # The LIKE filter already restricts the scan to manifests naming it.
        query = (
            "SELECT id, status, manifest FROM jobs WHERE manifest LIKE ? ESCAPE '\\'"
        )
        params = [f"%{escaped}%"]
        if workspace:
            query += " AND workspace = ?"
            params.append(workspace)
        query += " ORDER BY finished_at DESC"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
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
            "workspace": row[11] or DEFAULT_WORKSPACE_NAME,
            "workflow_name": row[12],
            "run_id": row[13],
            "run_dir": row[14],
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
        self.catalog_name = spec.get("catalog_name")
        self.status = QUEUED
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.manifest = []
        self.warnings = spec.get("warnings", [])
        self.error = None
        self.traceback = None
        # Which run this job turned out to be - reported by the worker's
        # run_start event, unknown until then and forever for a job that
        # never got that far
        self.run_id = None
        self.run_dir = None
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
            "workflow_name": self.catalog_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # Which workspace this job runs in - a live job's spec may not
            # carry one yet (e.g. a caller that never named a workspace),
            # so it defaults the same way history's column does
            "workspace": self.spec.get("workspace") or DEFAULT_WORKSPACE_NAME,
            "run_id": self.run_id,
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
            "run_dir": self.run_dir,
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
        output_dir=None,
        asset_dir=None,
        workspace=None,
        catalog_name=None,
    ):
        """Validate a job request and queue it. Raises ValueError on a bad
        request so the HTTP layer can answer 400 before anything runs.

        `workflow_dir` overrides this job's confinement root for a workflow
        that lives outside the writable directory - an example or a builtin,
        which the caller has already resolved against the search path. The
        worker re-validates against whatever this job records, so the
        override travels with the job rather than widening the manager.

        `output_dir`, `asset_dir` and `workspace` name which workspace this
        job runs in. They travel with the job for the same reason: one
        server holds several workspaces, and the process-wide roots would
        make every job belong to whichever one was configured at startup.

        `catalog_name` is the listing name the caller resolved `workflow_path`
        from, kept for history; None for an inline definition.
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
            # Loads and schema-validates now - a bad path or file fails the
            # request, not the queue
            loaded = workflow_from_file(workflow_path, job_output_dir, confinement)
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
                copy.deepcopy(workflow), job_output_dir, base_dir, confinement
            )
            loaded.validate()
            spec = {
                "workflow": workflow,
                # Must match workflow_from_definition's fallback - the worker
                # re-validates this against workflow_dir
                "base_dir": base_dir
                or (os.path.abspath(confinement) if confinement else os.getcwd()),
                "workflow_name": loaded.name,
                "arguments": arguments,
                # Must be the same root the worker re-validates base_dir
                # against (workflow_from_definition -> validate_path) - this
                # job's own confinement, not the manager's process-wide
                # default, or a named workspace's inline job fails after a
                # 201 the moment base_dir and workflow_dir disagree
                "workflow_dir": confinement,
            }

        # Which workspace this job runs in, and the roots that follow from
        # it - recorded on the job so history, the worker command and a
        # rerun all agree without re-deriving them
        spec["workspace"] = workspace
        spec["catalog_name"] = catalog_name
        spec["output_dir"] = job_output_dir
        if asset_dir:
            spec["asset_dir"] = asset_dir

        # The caller's own arguments, checked against the variables this
        # workflow declares. set_variables makes the same check at the top of
        # the run, so a bad name failed a job that had already been queued -
        # and a workflow declaring no variables dropped every argument in
        # silence. Refused here instead, while it is still a 400
        problems = argument_errors(loaded.workflow_definition, arguments)
        if problems:
            raise ValueError(
                "; ".join(
                    f"{problem['path']}: {problem['message']}" for problem in problems
                )
            )

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

    def definition(self, job_id):
        """The workflow JSON a job ran, for a read-only view of it.

        An inline definition comes straight from the spec; a job launched
        from a path is re-read from disk, confined to the root the job ran
        against. None when there is no such job, or when the file it named
        has since moved, grown past the size limit or stopped parsing - a
        graph of the run is a nicety, never a reason to fail the page.
        """
        job = self.jobs.get(job_id)
        if job is not None:
            spec = job.spec
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

    def rerun(self, job_id, new_seed=False):
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
        """
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
            # Bounded to 53 bits rather than the 64 torch allows: this number
            # goes out as JSON and comes back through a browser, where every
            # integer is a double, and a seed that changed on the way through
            # would be a seed nobody can reproduce
            arguments = {**arguments, variable: random.getrandbits(53)}

        workspace = spec.get("workspace")
        if (
            workspace
            and workspace != DEFAULT_WORKSPACE_NAME
            and spec.get("output_dir")
            and not os.path.isdir(spec["output_dir"])
        ):
            raise ValueError(f"Workspace '{workspace}' the job ran in no longer exists")

        return self.submit(
            workflow_path=spec.get("workflow_path"),
            workflow=spec.get("workflow"),
            arguments=arguments,
            base_dir=spec.get("base_dir"),
            workflow_dir=spec.get("workflow_dir"),
            output_dir=spec.get("output_dir"),
            asset_dir=spec.get("asset_dir"),
            workspace=workspace,
            catalog_name=spec.get("catalog_name"),
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
                command = {
                    "type": "execute",
                    "arguments": job.spec["arguments"],
                    # The job's own roots, so a job queued for one workspace
                    # still runs in it after the manager has served another
                    "output_dir": job.spec.get("output_dir") or self.output_dir,
                    "log_level": self.log_level,
                }
                if job.spec.get("asset_dir"):
                    command["asset_dir"] = job.spec["asset_dir"]
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

    def _record_manifest(self, job, message):
        """What the run wrote, named the way clients address outputs.

        Recorded for a failed or cancelled run as well as a successful one -
        the files the steps before the stop wrote are on disk either way,
        and a manifest that omits them is the difference between "this run
        produced nothing" and "this run produced four of five shots"
        (T015)."""
        job.manifest = [
            (
                {
                    **entry,
                    "files": self._relative_output_names(
                        entry["files"], job.spec.get("output_dir")
                    ),
                }
                if "files" in entry
                else entry
            )
            for entry in message.get("manifest", [])
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
            message_type = message.get("type")

            if message_type == "progress":
                event = {k: v for k, v in message.items() if k != "type"}
                if "files" in event:
                    event["files"] = self._relative_output_names(
                        event["files"], job.spec.get("output_dir")
                    )
                if event.get("event") == "run_start":
                    job.run_id = event.get("run_id")
                    job.run_dir = event.get("run_dir")
                job.add_event(event)
            elif message_type in ("output", "workflow_loaded"):
                text = message.get("message") or message.get("workflow_name", "")
                job.add_event({"event": "log", "message": text})
            elif message_type == "memory_info":
                self.last_memory = message.get("info")
                job.add_event({"event": "memory", "info": self.last_memory})
            elif message_type == "success":
                self._record_manifest(job, message)
                return (SUCCEEDED, None, None)
            elif message_type == "cancelled":
                self._record_manifest(job, message)
                return (CANCELLED, None, None)
            elif message_type == "error":
                # A failed run's steps too: the ones before the failure wrote
                # real files, and a job that reports an empty manifest hides
                # them behind the error that stopped the run
                self._record_manifest(job, message)
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
            return {"live": False, "info": self.last_memory}
        finally:
            self._worker_lock.release()
        if result.get("type") == "memory_status":
            self.last_memory = result.get("info")
            return {"live": True, "info": self.last_memory}
        return {"live": False, "info": self.last_memory}
