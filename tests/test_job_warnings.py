"""What lands in a finished job's `warnings` list.

`warnings` is where a caller who polled the finished job looks (#82); the
event log keeps the moment something happened. A phase-stall report (#176)
is a moment - "still in phase 'loading', 60.0s ..." on a job that then
succeeded is not a warning about the result - so it stays in the event log
and out of `warnings`, where the regression suites assert `warnings: []`
on a clean run.
"""

from dw.server.jobs import Job


def _job():
    return Job({"workflow_name": "x"})


def test_an_ordinary_warning_is_persisted_with_its_step():
    job = _job()
    job._note_progress({"event": "step_start", "step": "encode"})
    job._note_progress(
        {"event": "warning", "kind": "audio_clipped", "message": "peak above 0 dBFS"}
    )
    assert job.warnings == ["encode: peak above 0 dBFS"]


def test_a_phase_stall_report_stays_out_of_the_persisted_warnings():
    job = _job()
    job._note_progress({"event": "step_start", "step": "load"})
    for seconds in (35.0, 65.0, 95.0):
        job._note_progress(
            {
                "event": "warning",
                "kind": "phase_stall",
                "phase": "loading",
                "seconds_since_phase_start": seconds,
                "message": (
                    f"still in phase 'loading', {seconds:.1f}s since it started "
                    "with no progress event"
                ),
            }
        )
    job._note_progress(
        {"event": "warning", "kind": "audio_clipped", "message": "peak above 0 dBFS"}
    )
    assert job.warnings == ["load: peak above 0 dBFS"]
