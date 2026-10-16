"""WAL mode on the jobs database - a real concurrency benefit (readers do
not block behind a writer) for a near-zero-risk one-line change. See
docs/proposals/maintenance-screen.md, "Phase 0"."""

from dw.server.job_history import JobHistory


def test_the_jobs_database_uses_wal_mode(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))

    with history._connect() as connection:
        (mode,) = connection.execute("PRAGMA journal_mode").fetchone()

    assert mode.lower() == "wal"


def test_every_connection_is_closed_when_its_block_ends(tmp_path, monkeypatch):
    """A connection left to GC checkpoints the WAL whenever it is collected,
    rewriting jobs.sqlite under whatever is watching it; each one closes at
    the end of its `with` block instead."""
    import sqlite3

    import pytest

    import dw.server.job_history as job_history

    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(job_history.sqlite3, "connect", tracking_connect)
    history = JobHistory(str(tmp_path / "jobs.sqlite"))
    history.get("missing")
    history.watermark()

    assert len(opened) == 3
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_a_failed_block_rolls_back_and_still_closes(tmp_path):
    import sqlite3

    import pytest

    history = JobHistory(str(tmp_path / "jobs.sqlite"))
    with pytest.raises(RuntimeError):
        with history._connect() as connection:
            connection.execute("INSERT INTO jobs (id, status) VALUES ('x', 'failed')")
            raise RuntimeError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    assert history.get("x") is None
