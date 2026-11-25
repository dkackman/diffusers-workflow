"""A failed or cancelled take's `reason` (RQ-3): the first line of the job's
error, on an attempt only when the read asked `reasons=true`."""

from dw.server.job_record import REASON_MAX

from . import test_ledger_job_link as link
from .test_ledger_job_link import WS, finish, submit, view
from .test_ledger_page_actions import put

client, root, planned = link.client, link.root, link.planned

URL = f"/api/workspaces/{WS}/requests/cats"


def ended(client, error, status="failed", live=False):
    """One failed job on item `a`, its error and status as given: in the
    history row, or (`live`) on the job still held by the manager."""
    client.script.fail_next = True
    job = finish(client, submit(client))
    if live:
        client.manager.jobs[job["id"]].error = error
        return job
    client.manager.jobs.pop(job["id"])
    with client.manager.history._connect() as connection:
        connection.execute(
            "UPDATE jobs SET error = ?, status = ? WHERE id = ?",
            (error, status, job["id"]),
        )
    return job


def attempts(found):
    return found["items"][0]["attempts"]


def asked(client):
    return attempts(client.get(URL, params={"reasons": "true"}).json())


def test_a_failed_take_carries_its_reason_when_asked(planned):
    ended(planned, "CUDA out of memory.\nTraceback (most recent call last):\n  ...")
    assert asked(planned)[0]["reason"] == "CUDA out of memory."


def test_a_traceback_gives_its_first_line_clipped(planned):
    ended(planned, "x" * 500 + "\nTraceback")
    reason = asked(planned)[0]["reason"]
    assert reason == "x" * REASON_MAX + "…"


def test_a_restart_tombstone_gives_its_reason(planned):
    ended(planned, "server restart: the job was running\nmore", status="cancelled")
    attempt = asked(planned)[0]
    assert attempt["status"] == "cancelled"
    assert attempt["reason"] == "server restart: the job was running"


def test_a_succeeded_take_has_no_reason(planned):
    finish(planned, submit(planned))
    assert "reason" not in asked(planned)[0]


def test_a_read_without_reasons_has_no_reason(planned):
    ended(planned, "CUDA out of memory.")
    found = view(planned)
    assert attempts(found)[0]["status"] == "failed"
    assert all("reason" not in a for a in attempts(found))


def test_a_put_without_reasons_has_no_reason(planned):
    ended(planned, "CUDA out of memory.")
    found = put(planned, view(planned), "no change")
    assert all("reason" not in a for a in attempts(found))


def test_a_put_with_reasons_has_them(planned):
    ended(planned, "CUDA out of memory.")
    found = view(planned)
    body = {key: found[key] for key in ("title", "goal", "items", "revision")} | {
        "why": "no change"
    }
    r = planned.put(URL, json=body, params={"reasons": "true"})
    assert r.status_code == 200, r.text
    assert attempts(r.json())[0]["reason"] == "CUDA out of memory."


def test_a_live_failed_job_gives_its_reason(planned):
    job = ended(planned, "boom\nmore", live=True)
    assert job["id"] in planned.manager.jobs
    assert asked(planned)[0]["reason"] == "boom"
