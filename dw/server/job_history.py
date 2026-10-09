"""Finished jobs, persisted so the Jobs view survives server restarts.

The `jobs` table has a `workspace` column; a database that predates it gets
the column added and every existing row backfilled to `default`, since
history that cannot say which workspace a job ran in stops making sense once
there are two.

The card a job ran on is three columns: `device`, the label every reader
returns (`"cuda:1 NVIDIA GeForce RTX 3090"`), and `device_ordinal`/
`device_card`, the same card as the two fields rerun affinity and observed
cost read (#693). A database that predates the two gets them backfilled from
`device` in SQL when it is first opened, so no reader parses the label.
"""

import json
import logging
import os
import sqlite3
import threading

from ..workspace import DEFAULT_WORKSPACE_NAME
from .observed_cost import EVENT_CAP, LOADING_MARKER
from .job_record import (
    ACK_NONE,
    MAX_PERSISTED_EVENTS,
    RERUN_SPEC_KEYS,
    SUCCEEDED,
)

logger = logging.getLogger("dw")


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
            # That run's ordinal among the workflow's runs - the 'v4' the
            # gallery shows. NULL before the column, and for a job that
            # never opened a run
            if "run_version" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN run_version INTEGER")
            # Which form of cost acknowledgement queued the job. Rows before
            # the column are 'none' - nothing recorded is nothing recorded
            if "acknowledged" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN acknowledged TEXT DEFAULT 'none'"
                )
            # The worker's own high-water mark for this run (#243) - NULL for
            # a row predating the column and for any run that never reported
            # one (cancelled/errored before the worker's final memory_info)
            if "host_memory_peak_rss_mb" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN host_memory_peak_rss_mb REAL"
                )
            # This job's own contribution to that process-lifetime figure -
            # growth since the job's first phase-boundary reading, or its
            # current rss when it caused no growth (#272). NULL for a row
            # predating the column and for any run that never got a
            # memory_info message at all
            if "host_memory_job_peak_rss_mb" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN host_memory_job_peak_rss_mb REAL"
                )
            # The card the job ran on - "cuda:1 NVIDIA GeForce RTX 3090"
            # (#462). NULL for a row predating the column: which card that
            # was is not known, and guessing would mislabel a box whose
            # card has been swapped
            if "device" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN device TEXT")
            # The same card as two fields - its ordinal ("cuda:1") and its
            # name ("NVIDIA GeForce RTX 3090") - so rerun affinity and
            # observed cost read them rather than parse `device` (#693).
            # `device` stays the label every reader returns. Rows before the
            # columns are backfilled from it once, in SQL: an ordinal never
            # holds a space, so the label splits at its first one, and a
            # label with none (`cpu`, `mps`) is all ordinal with no card
            if "device_ordinal" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN device_ordinal TEXT")
            if "device_card" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN device_card TEXT")
            if "device_ordinal" not in columns or "device_card" not in columns:
                connection.execute(
                    "UPDATE jobs SET"
                    " device_ordinal = CASE WHEN INSTR(device, ' ') > 0"
                    " THEN SUBSTR(device, 1, INSTR(device, ' ') - 1)"
                    " ELSE device END,"
                    " device_card = CASE WHEN INSTR(device, ' ') > 0"
                    " THEN SUBSTR(device, INSTR(device, ' ') + 1) END"
                    " WHERE device IS NOT NULL"
                )

    def _connect(self):
        # WAL mode lets a reader (the web UI polling job status, an MCP
        # get_job call) proceed without blocking behind whatever write the
        # worker is mid-transaction on, and vice versa - the default
        # rollback-journal mode takes a database-wide lock for the
        # duration of a write. journal_mode is a property of the database
        # file, not the connection, but PRAGMA is cheap and idempotent, so
        # it is set on every connect rather than assumed to have stuck.
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def record(self, job):
        # The spec's workflow_name/warnings are derived; keep what rerun needs
        rerun_spec = {key: job.spec[key] for key in RERUN_SPEC_KEYS if key in job.spec}
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs (id, workflow, status, created_at,"
                " started_at, finished_at, arguments, spec, manifest, warnings,"
                " error, events, workspace, workflow_name, run_id, run_dir,"
                " acknowledged, host_memory_peak_rss_mb,"
                " host_memory_job_peak_rss_mb, run_version, device,"
                " device_ordinal, device_card) VALUES"
                " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    job.acknowledged,
                    # A test double or an older in-memory Job predating this
                    # column reports None here rather than failing record()
                    # (#243) - the same "absent means unknown" the column
                    # itself allows
                    getattr(job, "host_memory_peak_rss_mb", None),
                    getattr(job, "host_memory_job_peak_rss_mb", None),
                    getattr(job, "run_version", None),
                    getattr(job, "device", None),
                    getattr(job, "device_ordinal", None),
                    getattr(job, "device_card", None),
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
            " workspace, workflow_name, run_id, acknowledged, run_version,"
            " device FROM jobs"
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
                "acknowledged": row[9] or ACK_NONE,
                "run_version": row[10],
                "device": row[11],
                "historical": True,
            }
            for row in rows
        ]

    def get(self, job_id):
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, workflow, status, created_at, started_at, finished_at,"
                " arguments, spec, manifest, warnings, error, workspace,"
                " workflow_name, run_id, run_dir, acknowledged, events,"
                " run_version, device FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._to_detail(row) if row else None

    def device_ordinal(self, job_id):
        """The ordinal of the card a finished job ran on ('cuda:1'), or None
        for a job that never ran or that history has never seen."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT device_ordinal FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row[0] if row else None

    def watermark(self):
        """How far the table has got - what a derived figure caches against.

        A job landing changes every observed cost and changes no file, so an
        mtime cache cannot see it (dw/server/observed_cost.py). Counted over
        `workflow_name IS NOT NULL` rather than every row, because
        `orphan_workflow_history` (#274) detaches a deleted workflow's rows by
        clearing that column rather than deleting the row - an ordinary
        `COUNT(*)` would not move, and `ObservedCosts` would keep serving the
        purged figure until an unrelated job happened to land. Counting only
        the joinable rows falls by exactly the amount a purge detaches, the
        same as a prune lowering it.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*), MAX(finished_at) FROM jobs"
                " WHERE workflow_name IS NOT NULL"
            ).fetchone()
        return (row[0], row[1]) if row else (0, None)

    def finished_runs(self):
        """Every successful, named run grouped by (workspace, workflow name),
        as the rows an observed cost is derived from.

        One query for the whole catalog rather than one per workflow. The
        cold/warm split is decided in SQL on the persisted event tail - a
        `loading` phase as `json.dumps` wrote it - so 200 events per row are
        never parsed to answer a yes/no question, and whether that tail hit
        its cap comes back too, because a run whose `loading` phase was
        trimmed away has to count as neither rather than as warm.

        Rows with no `workflow_name` (recorded before the column existed, or
        run from an inline definition, or orphaned by `orphan_workflow_history`)
        are unjoinable and left out. The workspace dimension is always in the
        key here; whether a caller treats two workspaces as one history (a
        shared catalog source, #154) or as separate (a workspace's own
        writable copy, #274) is decided in `ObservedCosts.rows_for`, which is
        the layer that knows which kind of source it was asked about.
        """
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT workflow_name, workspace, started_at, finished_at,"
                " arguments, manifest, INSTR(COALESCE(events, ''), ?) > 0,"
                " COALESCE(json_array_length(COALESCE(events, '[]')), 0) >= ?,"
                " host_memory_peak_rss_mb, host_memory_job_peak_rss_mb, device,"
                " device_ordinal, device_card FROM jobs WHERE status = ? AND workflow_name IS NOT NULL"
                " AND started_at IS NOT NULL AND finished_at IS NOT NULL",
                (LOADING_MARKER, EVENT_CAP, SUCCEEDED),
            ).fetchall()
        grouped = {}
        for (
            name,
            workspace,
            started,
            finished,
            arguments,
            manifest,
            had_load,
            at_cap,
            peak_rss_mb,
            job_peak_rss_mb,
            device,
            ordinal,
            card,
        ) in rows:
            key = (workspace or DEFAULT_WORKSPACE_NAME, name)
            grouped.setdefault(key, []).append(
                {
                    "started_at": started,
                    "finished_at": finished,
                    "duration": finished - started,
                    "arguments": arguments,
                    "manifest": manifest,
                    "had_load": bool(had_load),
                    "events_at_cap": bool(at_cap),
                    "host_memory_peak_rss_mb": peak_rss_mb,
                    "host_memory_job_peak_rss_mb": job_peak_rss_mb,
                    "device": device,
                    "device_ordinal": ordinal,
                    "device_card": card,
                }
            )
        return grouped

    def orphan_workflow_history(self, workspace, workflow_name):
        """Detach this (workspace, workflow_name)'s finished runs from cost
        history (#274).

        Deleting a workflow does not delete the job rows that ran it - those
        stay for `list_jobs`/`get_job` and any other audit trail - but a name
        reused afterwards, in this workspace or a fresh one copied from it,
        must not inherit the old identity's figures. Setting `workflow_name`
        to NULL is enough: `finished_runs()` already excludes rows where it
        is NULL, the same rule that already excludes a run from an inline
        definition.
        """
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET workflow_name = NULL"
                " WHERE workspace = ? AND workflow_name = ?",
                (workspace, workflow_name),
            )

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

        spec = parse(row[7], {})
        # The persisted tail is capped at MAX_PERSISTED_EVENTS, and
        # get_job_events serves that same tail - so counting it, rather than
        # hardcoding 0, keeps event_count truthful about what a caller who
        # pages through get_job_events will actually see (#289)
        events = parse(row[16], [])
        return {
            "id": row[0],
            "workflow": row[1],
            "status": row[2],
            "created_at": row[3],
            "started_at": row[4],
            "finished_at": row[5],
            "arguments": parse(row[6], {}),
            "spec": spec,
            "manifest": parse(row[8], []),
            "warnings": parse(row[9], []),
            "error": row[10],
            "workspace": row[11] or DEFAULT_WORKSPACE_NAME,
            "workflow_name": row[12],
            "run_id": row[13],
            "run_dir": row[14],
            "run_version": row[17],
            "device": row[18],
            "acknowledged": row[15] or ACK_NONE,
            "acknowledged_cost": (spec or {}).get("acknowledged_cost"),
            "traceback": None,
            "event_count": len(events) if isinstance(events, list) else 0,
            "historical": True,
        }
