"""A jobs-list row says why a job ended (#1449): `reason`, the error's first
line clipped to 160 characters, on failed and cancelled rows that have an
error, and `restart_batch` on a restart tombstone - the same helper fills
both for live jobs and history rows."""

import pytest

from dw.server.job_record import (
    CANCELLED,
    FAILED,
    QUEUED,
    REASON_MAX,
    RUNNING,
    SUCCEEDED,
    row_reason_fields,
)

from .test_server import (
    failing_script,
    hanging_script,
    server,  # noqa: F401 - the fixture
    valid_workflow,
    wait_for_status,
)


def test_the_reason_is_the_first_line_only():
    error = "CUDA out of memory\nTraceback (most recent call last):\n  File x"
    assert row_reason_fields(FAILED, error) == {"reason": "CUDA out of memory"}


def test_a_long_first_line_is_clipped_at_160_with_an_ellipsis():
    error = "x" * 400 + "\nsecond"
    reason = row_reason_fields(FAILED, error)["reason"]
    assert reason == "x" * REASON_MAX + "…"
    assert REASON_MAX == 160


def test_a_first_line_of_exactly_160_is_not_clipped():
    error = "y" * 160
    assert row_reason_fields(CANCELLED, error) == {"reason": error}


@pytest.mark.parametrize("status", [SUCCEEDED, QUEUED, RUNNING])
def test_a_row_that_did_not_fail_or_cancel_carries_neither_field(status):
    assert row_reason_fields(status, "server restart: queued", "inst-1") == {}


@pytest.mark.parametrize("error", [None, "", "   \n  "])
def test_a_cancel_with_no_error_carries_neither_field(error):
    assert row_reason_fields(CANCELLED, error, "inst-1") == {}


def test_a_tombstone_carries_its_server_instance_as_restart_batch():
    fields = row_reason_fields(
        CANCELLED, "server restart: queued, never started", "inst-1"
    )
    assert fields == {
        "reason": "server restart: queued, never started",
        "restart_batch": "inst-1",
    }


@pytest.mark.parametrize(
    "status, error, instance",
    [
        # no instance: a row from before #1245
        (CANCELLED, "server restart: queued, never started", None),
        # not the prefix
        (CANCELLED, "cancelled: user asked", "inst-1"),
        # the prefix, but failed rather than cancelled
        (FAILED, "server restart: queued, never started", "inst-1"),
    ],
)
def test_restart_batch_needs_cancelled_the_prefix_and_an_instance(
    status, error, instance
):
    fields = row_reason_fields(status, error, instance)
    assert "restart_batch" not in fields
    assert fields["reason"] == error


def test_a_failed_job_lists_its_reason_and_detail_keeps_the_full_error(server):  # noqa: F811
    with server(failing_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        detail = wait_for_status(client, job["id"], ["failed"])
        (row,) = [
            r for r in client.get("/api/jobs").json()["jobs"] if r["id"] == job["id"]
        ]
        assert row["reason"] == detail["error"].strip().splitlines()[0]
        assert "\n" not in row["reason"]
        assert "restart_batch" not in row
        assert detail["error"]


def test_a_queued_cancel_lists_neither_field(server):  # noqa: F811
    with server(hanging_script) as client:
        running = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, running["id"], ["running"])
        queued = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        client.post(f"/api/jobs/{queued['id']}/cancel")
        rows = {r["id"]: r for r in client.get("/api/jobs").json()["jobs"]}
        for job_id in (queued["id"], running["id"]):
            assert "reason" not in rows[job_id]
            assert "restart_batch" not in rows[job_id]
        client.post(f"/api/jobs/{running['id']}/cancel")
        wait_for_status(client, running["id"], ["cancelled"])


def test_a_running_job_with_a_cancel_request_lists_under_strict_responses(server):  # noqa: F811
    """`cancel_requested` is declared on JobSummary: the suite runs strict
    (tests/conftest.py), so an undeclared key would fail this list."""
    with server(hanging_script) as client:
        running = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, running["id"], ["running"])
        manager = client.app.state.job_manager
        # Held as the window between a cancel request and the job stopping
        manager.jobs[running["id"]].cancel_requested = True
        response = client.get("/api/jobs")
        assert response.status_code == 200, response.text
        (row,) = [r for r in response.json()["jobs"] if r["id"] == running["id"]]
        assert row["cancel_requested"] is True
        manager.jobs[running["id"]].cancel_requested = False
        client.post(f"/api/jobs/{running['id']}/cancel")
        wait_for_status(client, running["id"], ["cancelled"])
