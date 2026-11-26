"""GET /api/server's `storage` block (#1457 stage A): what the job history
and the log take on disk. Read-only, and a figure that can't be read is null,
never an error on the route."""

import logging
import os
import sqlite3

import pytest

from dw.log_setup import log_storage_stats, setup_logging
from dw.server.job_history import JobHistory

from tests.test_server import valid_workflow, wait_for_status
from tests.test_server_info import client


@pytest.fixture
def dw_logger_restored():
    """setup_logging replaces the 'dw' logger's handlers; put the originals
    back so a test here does not move another test's log."""
    logger = logging.getLogger("dw")
    saved, level = list(logger.handlers), logger.level
    yield logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if handler not in saved:
            handler.close()
    for handler in saved:
        logger.addHandler(handler)
    logger.setLevel(level)


def _no_file_handlers(logger):
    for handler in list(logger.handlers):
        if getattr(handler, "baseFilename", None):
            logger.removeHandler(handler)


# --- job history


def test_job_history_rows_is_a_real_count_and_rises_with_a_job(tmp_path):
    """`rows` counts the table, and a job that runs through the server adds
    exactly one - the stage's acceptance, on the real submit path."""
    with client(tmp_path) as c:
        before = c.get("/api/server").json()["storage"]["job_history"]
        assert before["rows"] == 0
        assert before["oldest"] is None and before["newest"] is None
        job = c.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(c, job["id"], ["succeeded"])
        after = c.get("/api/server").json()["storage"]["job_history"]
    assert after["rows"] == 1
    assert after["path"] == str(tmp_path / "jobs.sqlite")
    assert after["oldest"] is not None and after["oldest"] == after["newest"]


def test_job_history_rows_is_not_capped_at_the_listing(tmp_path):
    """The job listing's `total` stops at 200 (#1466); this count does not."""
    history = JobHistory(tmp_path / "jobs.sqlite")
    with sqlite3.connect(history.db_path) as connection:
        connection.executemany(
            "INSERT INTO jobs (id, status, created_at, workspace) VALUES (?, ?, ?, ?)",
            [(f"j{n}", "succeeded", 1000.0 + n, "default") for n in range(250)],
        )
    stats = history.storage_stats()
    assert stats["rows"] == 250
    assert stats["oldest"] == 1000.0
    assert stats["newest"] == 1249.0


def test_job_history_size_includes_the_wal_and_shm_sidecars(tmp_path):
    history = JobHistory(tmp_path / "jobs.sqlite")
    db = tmp_path / "jobs.sqlite"
    (tmp_path / "jobs.sqlite-wal").write_bytes(b"w" * 1000)
    (tmp_path / "jobs.sqlite-shm").write_bytes(b"s" * 500)
    # Sized before the count connects, so the stand-in sidecars are what it sees
    db_size = os.path.getsize(db)
    stats = history.storage_stats()
    assert stats["size_bytes"] == db_size + 1500


def test_job_history_unreadable_is_null_not_an_error(tmp_path):
    history = JobHistory(tmp_path / "jobs.sqlite")
    history.db_path = str(tmp_path / "missing-dir" / "jobs.sqlite")
    stats = history.storage_stats()
    assert stats["rows"] is None
    assert stats["size_bytes"] == 0 or stats["size_bytes"] is None
    assert stats["oldest"] is None and stats["newest"] is None


def test_route_survives_a_history_that_raises(tmp_path, monkeypatch):
    def boom(self):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(JobHistory, "storage_stats", boom)
    with client(tmp_path) as c:
        response = c.get("/api/server")
    assert response.status_code == 200
    assert response.json()["storage"]["job_history"] is None


# --- logs


def test_logs_report_the_active_file_and_its_backups(tmp_path, dw_logger_restored):
    log = tmp_path / "dw.log"
    setup_logging(str(log), "INFO")
    dw_logger_restored.info("a line")
    for n in (1, 2):
        (tmp_path / f"dw.log.{n}").write_bytes(b"x" * 100)
    # Past backupCount (7): not one of this handler's backups
    (tmp_path / "dw.log.8").write_bytes(b"x" * 100)
    stats = log_storage_stats()
    assert stats["path"] == str(log)
    assert stats["size_bytes"] == os.path.getsize(log) > 0
    assert stats["backups"] == 2
    assert stats["total_bytes"] == stats["size_bytes"] + 200


def test_logs_backups_never_exceed_seven(tmp_path, dw_logger_restored):
    setup_logging(str(tmp_path / "dw.log"), "INFO")
    for n in range(1, 12):
        (tmp_path / f"dw.log.{n}").write_bytes(b"x")
    assert log_storage_stats()["backups"] == 7


def test_logs_null_when_no_log_file_is_configured(dw_logger_restored):
    _no_file_handlers(dw_logger_restored)
    assert log_storage_stats() is None


def test_route_reports_logs_and_null_without_a_file(tmp_path, dw_logger_restored):
    setup_logging(str(tmp_path / "dw.log"), "INFO")
    dw_logger_restored.info("a line")
    with client(tmp_path) as c:
        logs = c.get("/api/server").json()["storage"]["logs"]
        assert logs["size_bytes"] > 0
        assert 0 <= logs["backups"] <= 7
        _no_file_handlers(dw_logger_restored)
        assert c.get("/api/server").json()["storage"]["logs"] is None
