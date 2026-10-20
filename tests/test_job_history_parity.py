"""A job taken through `JobHistory.record` reads back the keys a live job
serves: a field added to `Job.summary()`/`detail()` and missed on the history
side fails here rather than going silently absent from `get_job` after a
restart."""

import pytest

from dw.server.job_history import JobHistory
from dw.server.job_record import Job

# The one place the two sides are allowed to differ. Live-only: the poll state
# a finished job no longer has. History-only: the marker, the rerun spec in
# detail, and the run location
# `JobManager.list` takes off before serving.
LIVE_ONLY = {"progress", "cancel_requested", "waiting_for"}
HISTORY_ONLY_SUMMARY = {"historical", "_run_dir", "_output_dir"}
HISTORY_ONLY_DETAIL = {"historical", "spec"}


def _job(**extra):
    job = Job(
        {
            "workflow_name": "w",
            "catalog_name": "w",
            "arguments": {"prompt": "a cat"},
            "workspace": "default",
            **extra,
        }
    )
    job.status = "succeeded"
    job.started_at = 10.0
    job.finished_at = 70.0
    job.run_id = "run-1"
    job.run_dir = "/tmp/run-1"
    job.run_version = 3
    job.device_ordinal = "cuda:1"
    job.device_card = "NVIDIA GeForce RTX 3090"
    return job


@pytest.mark.parametrize(
    "extra",
    [{}, {"note": "a test", "rerun_of": "abc123def456"}],
    ids=["optional-unset", "optional-set"],
)
def test_history_serves_the_keys_a_live_job_does(tmp_path, extra):
    job = _job(**extra)
    history = JobHistory(tmp_path / "jobs.sqlite")
    history.record(job)

    (summary,) = history.recent_summaries()
    detail = history.get(job.id)

    assert set(summary) - HISTORY_ONLY_SUMMARY == set(job.summary()) - LIVE_ONLY
    assert set(detail) - HISTORY_ONLY_DETAIL == set(job.detail()) - LIVE_ONLY
    for key, value in job.summary().items():
        assert summary[key] == value, key
        assert detail[key] == value, key
