"""A deleted workspace takes its job history with it (#913)."""

from dw.server.job_history import JobHistory
from dw.server.job_record import Job


def _job(job_id, workspace):
    job = Job(
        {
            "workflow_name": "w",
            "catalog_name": "w",
            "arguments": {},
            "workspace": workspace,
        }
    )
    job.id = job_id
    job.status = "succeeded"
    job.started_at = 10.0
    job.finished_at = 70.0
    return job


def test_forgetting_a_workspace_leaves_a_same_named_one_empty(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))
    history.record(_job("a", "sweep"))
    history.record(_job("b", "sweep"))
    history.record(_job("c", "other"))

    assert history.forget_workspace("sweep") == 2

    assert history.recent_summaries(workspace="sweep") == []
    assert [r["id"] for r in history.recent_summaries(workspace="other")] == ["c"]
    assert history.get("a")["status"] == "succeeded"
    assert history.recent_summaries() and [
        r["id"] for r in history.recent_summaries()
    ] == ["c"]


def test_a_database_with_the_retired_kept_column_still_opens(tmp_path):
    """The `kept` column (#917) is no longer declared: a database that grew
    it before #944 keeps the physical column, and the migration adds only
    what is missing rather than tripping over a column it no longer names."""
    import sqlite3

    from dw.server.job_history import _ADDED_COLUMNS

    assert "kept" not in {column for column, _ in _ADDED_COLUMNS}
    path = str(tmp_path / "jobs.sqlite")
    JobHistory(path).record(_job("a", "sweep"))
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE jobs ADD COLUMN kept INTEGER DEFAULT 0")

    history = JobHistory(path)
    history.record(_job("b", "sweep"))

    assert {r["id"] for r in history.recent_summaries(workspace="sweep")} == {"a", "b"}
    assert history.forget_workspace("sweep") == 2
    assert history.get("a")["status"] == "succeeded"
