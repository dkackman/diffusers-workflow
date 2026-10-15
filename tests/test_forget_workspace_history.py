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
    assert history.get("a") is None
