"""WAL mode on the jobs database - a real concurrency benefit (readers do
not block behind a writer) for a near-zero-risk one-line change. See
docs/proposals/maintenance-screen.md, "Phase 0"."""

from dw.server.jobs import JobHistory


def test_the_jobs_database_uses_wal_mode(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))

    with history._connect() as connection:
        (mode,) = connection.execute("PRAGMA journal_mode").fetchone()

    assert mode.lower() == "wal"
