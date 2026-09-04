"""A step-cache hit republishes an earlier job's files into the reusing
job's manifest, marked `reused`. File attribution must stay with the job
that actually wrote the file."""

from dw.server.jobs import JobHistory, Job


def _record(history, job_id, finished_at, manifest):
    job = Job({"workflow_name": "w", "arguments": {}})
    job.id = job_id
    job.manifest = manifest
    job.status = "succeeded"
    job.finished_at = finished_at
    history.record(job)


def test_job_for_file_skips_reused_manifest_entries(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))
    path = "/out/run-generate.0.png"

    _record(history, "writer", 1.0, [{"step": "generate", "files": [path]}])
    _record(
        history,
        "reuser",
        2.0,
        [{"step": "generate", "files": [path], "reused": True}],
    )

    assert history.job_for_file("run-generate.0.png")["id"] == "writer"


def test_job_for_file_returns_none_when_every_match_is_reused(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))
    path = "/out/run-generate.0.png"

    _record(
        history,
        "reuser",
        2.0,
        [{"step": "generate", "files": [path], "reused": True}],
    )

    assert history.job_for_file("run-generate.0.png") is None


def test_job_for_file_still_finds_a_plain_writing_job(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))

    _record(
        history, "writer", 1.0, [{"step": "s", "files": ["/out/test_image-0.0.png"]}]
    )

    assert history.job_for_file("test_image-0.0.png")["id"] == "writer"
