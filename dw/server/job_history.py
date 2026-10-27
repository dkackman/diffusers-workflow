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

import contextlib
import json
import logging
import os
import sqlite3
import threading

from ..workspace import DEFAULT_WORKSPACE_NAME
from .host_ram_need import memory_key
from .observed_cost import EVENT_CAP, LOADING_MARKER
from .job_record import (
    ACK_NONE,
    MAX_PERSISTED_EVENTS,
    RERUN_SPEC_KEYS,
    SUCCEEDED,
)

logger = logging.getLogger("dw")


def _memory_key_of(job):
    """The memory key of the definition the job was admitted with, or None -
    history is never refused a row over a key it could not compute."""
    try:
        return memory_key(job.spec.get("definition"), job.spec.get("arguments"))
    except Exception:
        logger.debug("No memory key for job %s", job.id, exc_info=True)
        return None


# The columns added since the table was first written, in the order they
# were added: a database missing one gets it by ALTER, and its rows keep the
# column's default (NULL unless the DDL says otherwise)
_ADDED_COLUMNS = (
    # Databases written before events were persisted
    ("events", "TEXT"),
    # The workspace the job ran in; rows before it are backfilled to
    # 'default' (`JobHistory.__init__`)
    ("workspace", "TEXT DEFAULT 'default'"),
    # The catalog name the job was run from, beside `workflow` (the
    # definition's id). Ids are not unique across a catalog forever; names
    # are, and a later runtime-by-workflow join wants the exact one. Rows
    # before this column stay NULL: old history is unjoinable, new history
    # is exact
    ("workflow_name", "TEXT"),
    # Which run of the workflow this job was - the directory under the
    # output root that holds its manifest and its realized workflow. NULL
    # for every row predating run tracking, and the manager refuses to
    # guess one from file paths
    ("run_id", "TEXT"),
    ("run_dir", "TEXT"),
    # That run's ordinal among the workflow's runs - the 'v4' the gallery
    # shows. NULL before the column, and for a job that never opened a run
    ("run_version", "INTEGER"),
    # Which form of cost acknowledgement queued the job. Rows before the
    # column are 'none' - nothing recorded is nothing recorded
    ("acknowledged", "TEXT DEFAULT 'none'"),
    # The worker's own high-water mark for this run (#243) - NULL for a row
    # predating the column and for any run that never reported one
    # (cancelled/errored before the worker's final memory_info)
    ("host_memory_peak_rss_mb", "REAL"),
    # This job's own contribution to that process-lifetime figure - growth
    # since the job's first phase-boundary reading, or its current rss when
    # it caused no growth (#272). NULL for a row predating the column and
    # for any run that never got a memory_info message at all
    ("host_memory_job_peak_rss_mb", "REAL"),
    # The card the job ran on - "cuda:1 NVIDIA GeForce RTX 3090" (#462).
    # NULL for a row predating the column: which card that was is not
    # known, and guessing would mislabel a box whose card has been swapped
    ("device", "TEXT"),
    # The same card as two fields - its ordinal ("cuda:1") and its name
    # ("NVIDIA GeForce RTX 3090") - so rerun affinity and observed cost read
    # them rather than parse `device` (#693). `device` stays the label
    # every reader returns. Rows before the columns are backfilled from it
    # once (`_BACKFILL_DEVICE_FIELDS`)
    ("device_ordinal", "TEXT"),
    ("device_card", "TEXT"),
    # The note the job was submitted with. NULL for a row before the column
    # and for a job submitted without one
    ("note", "TEXT"),
    # What decides the run's host-RAM footprint - its pipelines,
    # quantization and offload, arguments resolved
    # (`host_ram_need.memory_key`, #869). NULL for a row predating the
    # column, which counts as no data rather than a guess, and for a run
    # that loads no pipeline
    ("memory_key", "TEXT"),
    # The job a rerun reran. NULL for a fresh submit and for every row
    # before the column
    ("rerun_of", "TEXT"),
    # A database may hold a column no longer declared here (`kept`, #917,
    # retired once #944 made a workspace delete retire every row rather
    # than keep the ones an asset named): the migration only adds what is
    # missing (`_add_missing_columns`), so a stray physical column is left
    # alone and never read
)


def _json(value):
    return json.dumps(value, default=str)


def _rerun_spec(job):
    """The spec's workflow_name/warnings are derived; keep what rerun needs."""
    return {key: job.spec[key] for key in RERUN_SPEC_KEYS if key in job.spec}


# What `JobHistory.record` writes: one (column, getter) per recorded column, so
# a new job field is one entry here (plus its `_ADDED_COLUMNS` DDL), not a
# placeholder and a positional value. `getattr(..., None)` is for a test double
# or an older in-memory Job predating a column: absent means unknown, which the
# column itself allows (#243)
_RECORDED = (
    ("id", lambda job: job.id),
    ("workflow", lambda job: job.workflow_name),
    ("status", lambda job: job.status),
    ("created_at", lambda job: job.created_at),
    ("started_at", lambda job: job.started_at),
    ("finished_at", lambda job: job.finished_at),
    ("arguments", lambda job: _json(job.spec.get("arguments", {}))),
    ("spec", lambda job: _json(_rerun_spec(job))),
    ("manifest", lambda job: _json(job.manifest)),
    ("warnings", lambda job: _json(job.warnings)),
    ("error", lambda job: job.error),
    ("events", lambda job: _json(job.events[-MAX_PERSISTED_EVENTS:])),
    ("workspace", lambda job: job.spec.get("workspace") or DEFAULT_WORKSPACE_NAME),
    ("workflow_name", lambda job: job.catalog_name),
    ("run_id", lambda job: job.run_id),
    ("run_dir", lambda job: job.run_dir),
    ("acknowledged", lambda job: job.acknowledged),
    (
        "host_memory_peak_rss_mb",
        lambda job: getattr(job, "host_memory_peak_rss_mb", None),
    ),
    (
        "host_memory_job_peak_rss_mb",
        lambda job: getattr(job, "host_memory_job_peak_rss_mb", None),
    ),
    ("run_version", lambda job: getattr(job, "run_version", None)),
    ("device", lambda job: getattr(job, "device", None)),
    ("device_ordinal", lambda job: getattr(job, "device_ordinal", None)),
    ("device_card", lambda job: getattr(job, "device_card", None)),
    ("note", lambda job: getattr(job, "note", None)),
    ("memory_key", _memory_key_of),
    ("rerun_of", lambda job: getattr(job, "rerun_of", None)),
)

# `device_ordinal`/`device_card` from `device`, in SQL: an ordinal never
# holds a space, so the label splits at its first one, and a label with none
# (`cpu`, `mps`) is all ordinal with no card
_BACKFILL_DEVICE_FIELDS = (
    "UPDATE jobs SET"
    " device_ordinal = CASE WHEN INSTR(device, ' ') > 0"
    " THEN SUBSTR(device, 1, INSTR(device, ' ') - 1)"
    " ELSE device END,"
    " device_card = CASE WHEN INSTR(device, ' ') > 0"
    " THEN SUBSTR(device, INSTR(device, ' ') + 1) END"
    " WHERE device IS NOT NULL"
)


def _add_missing_columns(connection):
    """ALTER in each `_ADDED_COLUMNS` column the jobs table lacks; the names
    added. ALTER is the whole migration - the column and its DDL are this
    module's constants, never input."""
    present = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    added = set()
    for column, ddl in _ADDED_COLUMNS:
        if column not in present:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")
            added.add(column)
    return added


# Appended to the workspace of a retired workspace's rows (#944); no workspace
# name can contain a space and parenthesis.
DELETED_SUFFIX = " (deleted)"


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
            added = _add_missing_columns(connection)
            # Every row predating workspaces belongs to the default one -
            # history that cannot say which workspace a job ran in stops
            # making sense the moment there are two
            if "workspace" in added:
                connection.execute(
                    "UPDATE jobs SET workspace = 'default' WHERE workspace IS NULL"
                )
            if "device_ordinal" in added or "device_card" in added:
                connection.execute(_BACKFILL_DEVICE_FIELDS)

    @contextlib.contextmanager
    def _connect(self):
        # WAL mode lets a reader (the web UI polling job status, an MCP
        # get_job call) proceed without blocking behind whatever write the
        # worker is mid-transaction on, and vice versa - the default
        # rollback-journal mode takes a database-wide lock for the
        # duration of a write. journal_mode is a property of the database
        # file, not the connection, but PRAGMA is cheap and idempotent, so
        # it is set on every connect rather than assumed to have stuck.
        #
        # The block commits on success and rolls back on an exception (what
        # sqlite3's own context manager does), then closes the connection:
        # one left to GC checkpoints the WAL whenever it is collected,
        # rewriting jobs.sqlite at an arbitrary later moment.
        with contextlib.closing(sqlite3.connect(self.db_path, timeout=5)) as connection:
            # Readers take columns by name; a positional index is a silent
            # miss the next time a column is added
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection

    def record(self, job):
        columns = ", ".join(column for column, _ in _RECORDED)
        placeholders = ", ".join("?" for _ in _RECORDED)
        values = tuple(getter(job) for _, getter in _RECORDED)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO jobs ({columns}) VALUES ({placeholders})",
                values,
            )

    def peaks_for_key(self, key):
        """This machine's job peak RSS, in MB, for every successful run with
        this memory key - inline and renamed runs too, since the key is not
        the workflow's name (#869). Empty for a key no run has reported."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT host_memory_job_peak_rss_mb AS peak FROM jobs WHERE status = ?"
                " AND memory_key = ? AND host_memory_job_peak_rss_mb IS NOT NULL",
                (SUCCEEDED, key),
            ).fetchall()
        return [row["peak"] for row in rows]

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
            " device, note, run_dir,"
            " CASE WHEN json_valid(spec) THEN json_extract(spec, '$.output_dir') END"
            " AS output_dir,"
            " rerun_of"
            " FROM jobs"
        )
        params = []
        clauses = []
        if workspace:
            clauses.append("workspace = ?")
            params.append(workspace)
        else:
            clauses.append("workspace NOT LIKE ?")
            params.append(f"%{DELETED_SUFFIX}")
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
                "id": row["id"],
                "workflow": row["workflow"],
                "status": row["status"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "workspace": row["workspace"] or DEFAULT_WORKSPACE_NAME,
                "workflow_name": row["workflow_name"],
                "run_id": row["run_id"],
                "acknowledged": row["acknowledged"] or ACK_NONE,
                "run_version": row["run_version"],
                "device": row["device"],
                "historical": True,
                # Only when set: JobSummary.note is absent, never null
                **({"note": row["note"]} if row["note"] else {}),
                # Where the run is, for the list to read its current note
                # from; JobManager.list takes these off before serving
                "_run_dir": row["run_dir"],
                "_output_dir": row["output_dir"],
                **({"rerun_of": row["rerun_of"]} if row["rerun_of"] else {}),
            }
            for row in rows
        ]

    def get(self, job_id):
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, workflow, status, created_at, started_at, finished_at,"
                " arguments, spec, manifest, warnings, error, workspace,"
                " workflow_name, run_id, run_dir, acknowledged, events,"
                " run_version, device, note, rerun_of FROM jobs WHERE id = ?",
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
        return row["device_ordinal"] if row else None

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
                "SELECT COUNT(*) AS runs, MAX(finished_at) AS last_finished FROM jobs"
                " WHERE workflow_name IS NOT NULL"
            ).fetchone()
        return (row["runs"], row["last_finished"]) if row else (0, None)

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

    def forget_workspace(self, workspace):
        """Retire the rows recorded in this workspace (#913). A new workspace
        reusing the name starts with none: every row moves to a workspace
        name no workspace can be called, so no listing of the name finds it.
        The rows stay, so a job id still resolves for `get_job` and
        `get_job_workflow` (#944), which fold the recorded arguments into the
        run's definition without the workspace. Returns the rows retired."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET workspace = ? WHERE workspace = ?",
                (f"{workspace}{DELETED_SUFFIX}", workspace),
            )
            return cursor.rowcount

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
        if not row["events"]:
            return []
        try:
            return json.loads(row["events"])
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
            if self._manifest_wrote(row["manifest"], file_name):
                return {"id": row["id"], "status": row["status"]}
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

        spec = parse(row["spec"], {})
        # The persisted tail is capped at MAX_PERSISTED_EVENTS, and
        # get_job_events serves that same tail - so counting it, rather than
        # hardcoding 0, keeps event_count truthful about what a caller who
        # pages through get_job_events will actually see (#289)
        events = parse(row["events"], [])
        return {
            "id": row["id"],
            "workflow": row["workflow"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "arguments": parse(row["arguments"], {}),
            "spec": spec,
            "manifest": parse(row["manifest"], []),
            "warnings": parse(row["warnings"], []),
            "error": row["error"],
            "workspace": row["workspace"] or DEFAULT_WORKSPACE_NAME,
            "workflow_name": row["workflow_name"],
            "run_id": row["run_id"],
            "run_dir": row["run_dir"],
            "run_version": row["run_version"],
            "device": row["device"],
            "acknowledged": row["acknowledged"] or ACK_NONE,
            "acknowledged_cost": (spec or {}).get("acknowledged_cost"),
            "traceback": None,
            "event_count": len(events) if isinstance(events, list) else 0,
            "historical": True,
            **({"note": row["note"]} if row["note"] else {}),
            **({"rerun_of": row["rerun_of"]} if row["rerun_of"] else {}),
        }
