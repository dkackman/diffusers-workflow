"""The `device` column on the jobs table (#462): the card a job ran on,
added to a database that predates it and round-tripped for a job that has one."""

import sqlite3

from dw.server.job_history import JobHistory
from dw.server.job_record import Job

LABEL = "cuda:1 NVIDIA GeForce RTX 3090"


def _job(job_id, device=None):
    job = Job({"workflow_name": "w", "catalog_name": "w", "arguments": {}})
    job.id = job_id
    job.status = "succeeded"
    job.started_at = 10.0
    job.finished_at = 70.0
    job.device = device
    return job


def _old_shape_database(path):
    """A jobs.sqlite as it was before the column: the current table with
    `device` dropped, holding one finished job."""
    history = JobHistory(path)
    history.record(_job("old"))
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE jobs DROP COLUMN device")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(jobs)")]
    assert "device" not in columns


def _columns(path):
    with sqlite3.connect(path) as connection:
        return [row[1] for row in connection.execute("PRAGMA table_info(jobs)")]


class TestMigration:
    def test_opening_an_old_database_adds_the_column(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        _old_shape_database(path)

        JobHistory(path)

        assert "device" in _columns(path)

    def test_an_old_row_has_no_device_in_every_reader(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        _old_shape_database(path)

        history = JobHistory(path)

        (summary,) = history.recent_summaries()
        assert summary["id"] == "old"
        assert summary["device"] is None
        assert history.get("old")["device"] is None
        (row,) = history.finished_runs()[("default", "w")]
        assert row["device"] is None

    def test_opening_a_migrated_database_again_is_harmless(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        _old_shape_database(path)
        JobHistory(path)

        history = JobHistory(path)

        assert _columns(path).count("device") == 1
        assert history.get("old") is not None


class TestRoundTrip:
    def test_a_recorded_device_comes_back_from_every_reader(self, tmp_path):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(_job("new", device=LABEL))

        (summary,) = history.recent_summaries()
        assert summary["device"] == LABEL
        assert history.get("new")["device"] == LABEL
        (row,) = history.finished_runs()[("default", "w")]
        assert row["device"] == LABEL

    def test_a_job_that_never_ran_records_no_device(self, tmp_path):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(_job("queued"))

        assert history.get("queued")["device"] is None
