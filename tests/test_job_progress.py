"""What a poll learns about a job that is still running.

A single-step generation - every `shot`-shape template, the most expensive
thing the server does - emits `generating` and then nothing until it is
finished. Polls come back byte-identical for minutes, so "no new events" is
the normal state of a healthy run and cannot be read as trouble. The
progress block is what a caller reads instead: the step, the phase, how long
it has been in it, and the denoise counter once that loop is running.
"""

import time

from dw.server.jobs import Job, RUNNING, QUEUED, SUCCEEDED
from dw_mcp.diagnose import slim_job


def running_job(*events):
    job = Job({"workflow_name": "shot"})
    job.status = RUNNING
    job.started_at = time.time()
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


def test_the_denoise_counter_is_null_until_the_loop_runs():
    """Null, not absent: the lead-in to `generating` - encoding the prompt
    and any reference image or audio - is a minute or more of silence, and
    an absent key there cannot be told from a loop that has stopped."""
    job = running_job({"event": "phase", "phase": "generating", "detail": "h3"})

    lead_in = job.progress()
    assert lead_in["denoise_step"] is None
    assert lead_in["denoise_total_steps"] is None

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
    assert progress["denoise_step"] is None


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


def test_the_generating_lead_in_is_distinguishable_from_a_stalled_loop():
    """Two polls a minute apart inside H3's ~90 s of pre-loop encoding come
    back identical apart from the clock - which is the signature a caller is
    told means 'stuck'. The counter being present and null is what separates
    the two: null, the loop has not started; a number that stops moving
    while `seconds_since_event` climbs, it has stopped."""
    lead_in = running_job({"event": "phase", "phase": "generating", "detail": "h3"})
    lead_in.phase_started_at -= 90
    lead_in.last_event_at -= 90

    quiet = lead_in.progress()
    assert quiet["seconds_since_event"] >= 90
    assert quiet["denoise_step"] is None

    stalled = running_job(
        {"event": "phase", "phase": "generating", "detail": "h3"},
        {"event": "pipeline_step", "step": 4, "total_steps": 20},
    )
    stalled.last_event_at -= 90

    stuck = stalled.progress()
    assert stuck["seconds_since_event"] >= 90
    assert stuck["denoise_step"] == 4


def test_every_event_is_stamped_with_seconds_since_the_job_started():
    """The gap between `generating` and the first `pipeline_step` is a
    pipeline's pre-loop encoding, and between `step_start` and `generating`
    on a reused pipeline is what 'slow start' actually costs - neither is a
    number without a clock on the events themselves."""
    job = running_job()
    job.started_at -= 100
    job.add_event(
        {"event": "step_start", "step": "shot_a", "index": 0, "total_steps": 1}
    )
    job.add_event({"event": "phase", "phase": "generating", "detail": "h3"})

    step_start, generating = job.events_after(-1)
    assert 100 <= step_start["at"] < 101
    assert generating["at"] >= step_start["at"]
    assert job.events_after(-1)[0]["seq"] == 0


def test_a_queued_event_is_stamped_against_creation_until_the_job_starts():
    job = Job({"workflow_name": "shot"})
    job.add_event({"event": "job_status", "status": "queued"})

    assert job.events_after(-1)[0]["at"] >= 0
