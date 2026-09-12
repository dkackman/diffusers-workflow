"""What a poll learns about a job that is still running.

A single-step generation - every `shot`-shape template, the most expensive
thing the server does - emits `generating` and then nothing until it is
finished. Polls come back byte-identical for minutes, so "no new events" is
the normal state of a healthy run and cannot be read as trouble. The
progress block is what a caller reads instead: the step, the phase, how long
it has been in it, and the denoise counter once that loop is running.
"""

from dw.server.jobs import Job, RUNNING, QUEUED, SUCCEEDED
from dw_mcp.diagnose import slim_job


def running_job(*events):
    job = Job({"workflow_name": "shot"})
    job.status = RUNNING
    for event in events:
        job.add_event(event)
    return job


def test_a_job_that_has_not_started_reports_no_progress():
    job = Job({"workflow_name": "shot"})
    job.status = QUEUED

    assert job.progress() is None
    assert job.detail()["progress"] is None


def test_a_finished_job_reports_no_progress():
    """Its manifest is a better answer than a phase it has left."""
    job = running_job({"event": "phase", "phase": "generating"})
    job.finish(SUCCEEDED)

    assert job.progress() is None


def test_the_step_and_phase_are_reported():
    job = running_job(
        {"event": "step_start", "step": "shot_a", "index": 0, "total_steps": 2},
        {"event": "phase", "phase": "loading", "detail": "MiniMaxAI/MiniMax-H3"},
    )

    progress = job.progress()
    assert progress["step"] == "shot_a"
    assert progress["step_index"] == 0 and progress["total_steps"] == 2
    assert progress["phase"] == "loading"
    assert progress["phase_detail"] == "MiniMaxAI/MiniMax-H3"
    assert progress["seconds_in_phase"] >= 0
    assert progress["seconds_since_event"] >= 0


def test_the_denoise_counter_appears_once_the_loop_runs():
    job = running_job({"event": "phase", "phase": "generating", "detail": "h3"})

    assert "denoise_step" not in job.progress()

    job.add_event({"event": "pipeline_step", "step": 3, "total_steps": 20})

    progress = job.progress()
    assert progress["denoise_step"] == 3
    assert progress["denoise_total_steps"] == 20


def test_a_new_step_drops_the_previous_step_counter():
    """Otherwise the last step's '20 of 20' reads as this one's progress."""
    job = running_job(
        {"event": "step_start", "step": "a", "index": 0, "total_steps": 2},
        {"event": "pipeline_step", "step": 20, "total_steps": 20},
        {"event": "step_start", "step": "b", "index": 1, "total_steps": 2},
    )

    progress = job.progress()
    assert progress["step"] == "b"
    assert "denoise_step" not in progress


def test_the_phase_clock_restarts_with_the_phase_but_the_event_clock_does_not():
    job = running_job({"event": "phase", "phase": "loading"})
    job.phase_started_at -= 30
    job.last_event_at -= 30

    assert job.progress()["seconds_in_phase"] >= 30

    job.add_event({"event": "phase", "phase": "generating"})

    progress = job.progress()
    assert progress["phase"] == "generating"
    assert progress["seconds_in_phase"] < 1
    assert progress["seconds_since_event"] < 1


def test_progress_survives_the_trim_of_the_event_log():
    """The log keeps its tail; the summary is kept as events arrive, so a
    long render's phase is not something a caller has to page back for."""
    job = running_job({"event": "phase", "phase": "generating", "detail": "h3"})
    for seq in range(500):
        job.add_event({"event": "log", "message": f"line {seq}"})

    assert job.progress()["phase"] == "generating"


def test_the_mcp_poll_carries_it():
    job = running_job(
        {"event": "step_start", "step": "shot_a", "index": 0, "total_steps": 1},
        {"event": "phase", "phase": "generating", "detail": "h3"},
        {"event": "pipeline_step", "step": 7, "total_steps": 20},
    )

    slim = slim_job(job.detail())

    assert slim["progress"]["denoise_step"] == 7
    assert slim["progress"]["phase"] == "generating"
