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
    if device:
        job.device_ordinal, _, card = device.partition(" ")
        job.device_card = card or None
    return job


def _old_shape_database(path):
    """A jobs.sqlite as it was before the column: the current table with
    `device` (and the later #693 fields) dropped, holding one finished job."""
    history = JobHistory(path)
    history.record(_job("old"))
    with sqlite3.connect(path) as connection:
        for column in ("device", "device_ordinal", "device_card"):
            connection.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
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
        assert history.device_ordinal("new") == "cuda:1"
        assert row["device_ordinal"] == "cuda:1"
        assert row["device_card"] == "NVIDIA GeForce RTX 3090"

    def test_a_job_that_never_ran_records_no_device(self, tmp_path):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(_job("queued"))

        assert history.get("queued")["device"] is None
        assert history.device_ordinal("queued") is None


def _pre_fields_database(path, devices):
    """A jobs.sqlite as a server before #693 left it: `device` holds the
    label, and the ordinal and card columns do not exist yet."""
    history = JobHistory(path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE jobs DROP COLUMN device_ordinal")
        connection.execute("ALTER TABLE jobs DROP COLUMN device_card")
        for index, device in enumerate(devices):
            connection.execute(
                "INSERT INTO jobs (id, workflow, status, started_at, finished_at,"
                " arguments, manifest, events, workflow_name, device)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    f"job-{index}",
                    "w",
                    "succeeded",
                    10.0,
                    70.0,
                    "{}",
                    "[]",
                    "[]",
                    "w",
                    device,
                ),
            )
    return history


class TestFieldBackfill:
    """#693: opening a database from before the ordinal and card columns
    splits every row's `device` label into the two, in SQL."""

    SHAPES = [
        ("cuda:1 NVIDIA GeForce RTX 3090", "cuda:1", "NVIDIA GeForce RTX 3090"),
        ("cpu", "cpu", None),
        ("mps", "mps", None),
        ("mps Apple M5 Pro (MPS)", "mps", "Apple M5 Pro (MPS)"),
        (None, None, None),
    ]

    def test_every_label_shape_splits_into_its_ordinal_and_card(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        _pre_fields_database(path, [label for label, _, _ in self.SHAPES])

        history = JobHistory(path)

        rows = {
            (row["device"], row["device_ordinal"], row["device_card"])
            for row in history.finished_runs()[("default", "w")]
        }
        assert rows == set(self.SHAPES)
        for index, (label, ordinal, _) in enumerate(self.SHAPES):
            assert history.device_ordinal(f"job-{index}") == ordinal
            # What every reader returns is the label as it was
            assert history.get(f"job-{index}")["device"] == label

    def test_the_backfill_runs_once(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        _pre_fields_database(path, ["cuda:1 NVIDIA GeForce RTX 3090"])
        JobHistory(path)
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE jobs SET device_card = 'kept'")

        history = JobHistory(path)

        (row,) = history.finished_runs()[("default", "w")]
        assert row["device_card"] == "kept"
        assert _columns(path).count("device_ordinal") == 1

    def test_a_database_from_before_any_device_column_gets_all_three(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        _old_shape_database(path)

        history = JobHistory(path)

        assert {"device", "device_ordinal", "device_card"} <= set(_columns(path))
        assert history.device_ordinal("old") is None
