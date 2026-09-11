"""Running and diagnosing jobs. The gate on run_workflow is the point: a
run costs GPU time on an engine that runs one job at a time."""

import time

import httpx
import pytest

from dw_mcp import diagnose
from dw_mcp.client import DwApiError, DwClient

WORKFLOW = {"id": "w", "steps": []}
SUBMITTED = {"id": "job-1", "status": "queued", "queue_position": 2}


def scripted(routes):
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append(
            {"key": key, "body": request.read(), "params": dict(request.url.params)}
        )
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def submitting():
    return scripted({("POST", "/api/jobs"): (201, SUBMITTED)})


def test_run_refuses_without_an_acknowledged_cost():
    """The client-agnostic floor of the confirm gate. Claude Code does not
    implement MCP elicitation today, so this is what actually fires."""
    client, seen = submitting()

    with pytest.raises(DwApiError) as caught:
        diagnose.run_workflow(client, workflow_path="w.json")

    assert "acknowledged_cost" in str(caught.value)
    assert seen == [], "nothing may be queued before the cost is acknowledged"


def test_run_submits_once_the_cost_is_acknowledged():
    client, seen = submitting()

    result = diagnose.run_workflow(
        client, workflow_path="w.json", acknowledged_cost=True
    )

    assert result["job_id"] == "job-1"
    assert result["status"] == "queued"
    assert result["queue_position"] == 2
    assert len(seen) == 1


def test_run_returns_immediately_rather_than_waiting_for_the_job():
    """A generation takes minutes; no MCP client will hold a call open. The
    contract is submit-then-poll, so exactly one request goes out."""
    client, seen = submitting()

    result = diagnose.run_workflow(
        client, workflow_path="w.json", acknowledged_cost=True
    )

    assert [entry["key"] for entry in seen] == [("POST", "/api/jobs")]
    assert "get_job_events" in result["next"]


def test_run_sends_an_inline_workflow_when_given_one():
    client, seen = submitting()

    diagnose.run_workflow(client, inline_workflow=WORKFLOW, acknowledged_cost=True)

    assert b'"workflow"' in seen[0]["body"]


def test_run_never_sends_base_dir():
    """base_dir decides where an inline workflow's relative paths resolve -
    a path-authority parameter the tool surface deliberately withholds."""
    client, seen = submitting()

    diagnose.run_workflow(client, inline_workflow=WORKFLOW, acknowledged_cost=True)

    assert b"base_dir" not in seen[0]["body"]


def test_run_refuses_both_workflow_sources():
    client, seen = submitting()

    with pytest.raises(DwApiError, match="exactly one"):
        diagnose.run_workflow(
            client,
            workflow_path="w.json",
            inline_workflow=WORKFLOW,
            acknowledged_cost=True,
        )
    assert seen == []


def test_run_refuses_neither_workflow_source():
    client, seen = submitting()

    with pytest.raises(DwApiError, match="exactly one"):
        diagnose.run_workflow(client, acknowledged_cost=True)
    assert seen == []


def test_run_passes_variable_overrides():
    client, seen = submitting()

    diagnose.run_workflow(
        client,
        workflow_path="w.json",
        arguments={"prompt": "a cat"},
        acknowledged_cost=True,
    )

    assert b"a cat" in seen[0]["body"]


def test_run_surfaces_a_rejected_workflow():
    client, _seen = scripted(
        {("POST", "/api/jobs"): (400, {"detail": "steps must not be empty"})}
    )

    with pytest.raises(DwApiError, match="steps must not be empty"):
        diagnose.run_workflow(client, inline_workflow=WORKFLOW, acknowledged_cost=True)


def test_get_job_returns_the_detail_payload():
    client, _seen = scripted(
        {
            ("GET", "/api/jobs/job-1"): (
                200,
                {"id": "job-1", "status": "failed", "error": "CUDA out of memory"},
            )
        }
    )

    assert diagnose.get_job(client, "job-1")["error"] == "CUDA out of memory"


def test_get_job_events_pages_from_the_event_log():
    client, seen = scripted(
        {
            ("GET", "/api/jobs/job-1/event-log"): (
                200,
                {
                    "id": "job-1",
                    "status": "running",
                    "events": [{"seq": 3, "event": "phase"}],
                    "last_seq": 3,
                    "truncated": True,
                    "note": None,
                },
            )
        }
    )

    result = diagnose.get_job_events(client, "job-1", after=2, limit=50)

    assert seen[0]["params"] == {"after": "2", "limit": "50"}
    assert result["last_seq"] == 3
    assert result["truncated"] is True


def test_get_job_events_defaults_to_the_whole_log():
    client, seen = scripted(
        {("GET", "/api/jobs/job-1/event-log"): (200, {"events": [], "last_seq": -1})}
    )

    diagnose.get_job_events(client, "job-1")

    assert seen[0]["params"]["after"] == "-1"


def test_cancel_rerun_and_move_call_their_routes():
    client, seen = scripted(
        {
            ("POST", "/api/jobs/job-1/cancel"): (
                200,
                {"id": "job-1", "status": "cancelled"},
            ),
            ("POST", "/api/jobs/job-1/rerun"): (
                201,
                {"id": "job-2", "status": "queued"},
            ),
            ("POST", "/api/jobs/job-1/move"): (200, {"id": "job-1", "queue": []}),
        }
    )

    diagnose.cancel_job(client, "job-1")
    diagnose.rerun_job(client, "job-1", acknowledged_cost=True)
    diagnose.move_job(client, "job-1", "front")

    assert [entry["key"][1] for entry in seen] == [
        "/api/jobs/job-1/cancel",
        "/api/jobs/job-1/rerun",
        "/api/jobs/job-1/move",
    ]
    assert b"front" in seen[2]["body"]


def test_move_surfaces_a_job_that_has_left_the_queue():
    client, _seen = scripted(
        {
            ("POST", "/api/jobs/job-1/move"): (
                409,
                {"detail": "Job is not queued - only queued jobs move"},
            )
        }
    )

    with pytest.raises(DwApiError, match="only queued jobs move"):
        diagnose.move_job(client, "job-1", "up")


def test_rerun_refuses_without_an_acknowledged_cost():
    """A rerun queues the same generation from a stored spec - the same GPU
    minutes on the same one-job-at-a-time engine. The gate on run_workflow
    would be pointless if a job id from list_jobs bought a way around it."""
    client, seen = scripted({("POST", "/api/jobs/job-1/rerun"): (201, {"id": "job-2"})})

    with pytest.raises(DwApiError) as caught:
        diagnose.rerun_job(client, "job-1")

    assert "acknowledged_cost" in str(caught.value)
    assert seen == [], "nothing may be queued before the cost is acknowledged"


def test_rerun_reuses_the_run_refusal_message():
    """One gate, one wording - a second message would drift from the first."""
    client, _seen = scripted({})

    with pytest.raises(DwApiError) as caught:
        diagnose.rerun_job(client, "job-1")

    assert str(caught.value) == diagnose.COST_REFUSAL


def test_rerun_submits_once_the_cost_is_acknowledged():
    client, seen = scripted(
        {("POST", "/api/jobs/job-1/rerun"): (201, {"id": "job-2", "status": "queued"})}
    )

    result = diagnose.rerun_job(client, "job-1", acknowledged_cost=True)

    assert result["id"] == "job-2"
    assert [entry["key"][1] for entry in seen] == ["/api/jobs/job-1/rerun"]


def sequenced(route, bodies):
    """Like `scripted`, but one route replays a list of bodies in order -
    for a poller that expects the job to look different call to call. The
    last body repeats once the list runs out."""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append({"key": key, "params": dict(request.url.params)})
        if key != route:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        index = min(len(seen) - 1, len(bodies) - 1)
        return httpx.Response(200, json=bodies[index])

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_wait_for_job_returns_promptly_once_terminal(monkeypatch):
    """Two polls: running, then succeeded. Should return right after the
    second poll rather than waiting out the timeout."""
    monkeypatch.setattr(diagnose, "WAIT_POLL_SECONDS", 0.01)
    client, seen = sequenced(
        ("GET", "/api/jobs/job-1"),
        [
            {"id": "job-1", "status": "running"},
            {"id": "job-1", "status": "succeeded", "manifest": ["out.png"]},
        ],
    )

    result = diagnose.wait_for_job(client, "job-1", timeout_seconds=5)

    assert result["status"] == "succeeded"
    assert result["still_running"] is False
    assert result["job"]["manifest"] == ["out.png"]
    assert len(seen) == 2, "must not keep polling once the job is terminal"


def test_wait_for_job_reports_still_running_at_timeout(monkeypatch):
    """The job never finishes within the budget: returns still_running
    rather than hanging past timeout_seconds, and never oversleeps it."""
    monkeypatch.setattr(diagnose, "WAIT_POLL_SECONDS", 0.01)
    client, seen = sequenced(
        ("GET", "/api/jobs/job-1"), [{"id": "job-1", "status": "running"}]
    )

    started = time.monotonic()
    result = diagnose.wait_for_job(client, "job-1", timeout_seconds=0.03)
    elapsed = time.monotonic() - started

    assert result["status"] == "running"
    assert result["still_running"] is True
    assert "next" in result
    assert len(seen) >= 2, "should have polled more than once inside the budget"
    assert elapsed < 1, "must return once timeout_seconds elapses, not hang"


def test_wait_for_job_caps_the_timeout_it_is_given():
    """A caller asking for an absurd timeout does not get an absurd wait -
    the value is clamped before it ever reaches the poll loop."""
    client, seen = sequenced(
        ("GET", "/api/jobs/job-1"), [{"id": "job-1", "status": "succeeded"}]
    )

    diagnose.wait_for_job(client, "job-1", timeout_seconds=10_000)

    assert len(seen) == 1, "a terminal status on the first poll returns immediately"


def test_wait_for_job_does_not_require_acknowledged_cost():
    """It reads an already-queued job rather than starting anything, so the
    cost gate other job-queuing tools carry does not apply here."""
    client, _seen = sequenced(
        ("GET", "/api/jobs/job-1"), [{"id": "job-1", "status": "succeeded"}]
    )

    result = diagnose.wait_for_job(client, "job-1")

    assert result["status"] == "succeeded"


FAT_JOB = {
    "id": "job-1",
    "workflow_name": "minimax/dialogue-short",
    "status": "running",
    "created_at": 1.0,
    "started_at": 2.0,
    "finished_at": None,
    "workspace": "default",
    "run_id": None,
    "arguments": {"shot_1_cold_open": "x" * 6000},
    "warnings": [],
    "manifest": None,
    "error": None,
    "traceback": None,
    "event_count": 12,
    "run_dir": None,
}


def test_wait_for_job_does_not_echo_the_arguments_on_every_poll(monkeypatch):
    """An H3 workflow's arguments are 4-6k tokens of prompt text; a
    45-minute render is polled many times. They are get_job's to serve,
    once."""
    monkeypatch.setattr(diagnose, "MAX_WAIT_SECONDS", 0)
    client, _ = scripted({("GET", "/api/jobs/job-1"): (200, FAT_JOB)})

    result = diagnose.wait_for_job(client, "job-1", timeout_seconds=0)

    assert result["still_running"] is True
    assert "arguments" not in result["job"]
    assert "traceback" not in result["job"]
    assert result["job"]["status"] == "running"
    assert result["job"]["event_count"] == 12


def test_wait_for_job_keeps_the_manifest_and_error_once_terminal(monkeypatch):
    done = {**FAT_JOB, "status": "failed", "manifest": {"steps": []}, "error": "boom"}
    client, _ = scripted({("GET", "/api/jobs/job-1"): (200, done)})

    result = diagnose.wait_for_job(client, "job-1")

    assert result["job"]["manifest"] == {"steps": []}
    assert result["job"]["error"] == "boom"
    assert "arguments" not in result["job"]


class TestGetJobWorkflow:
    def test_a_realized_workflow_comes_back_with_the_flag_set(self):
        client, seen = scripted(
            {
                ("GET", "/api/jobs/job-1/workflow"): (
                    200,
                    {"id": "job-1", "definition": WORKFLOW, "realized": True},
                )
            }
        )

        result = diagnose.get_job_workflow(client, "job-1")

        assert result["job_id"] == "job-1"
        assert result["realized"] is True
        assert result["workflow"] == WORKFLOW
        assert len(seen) == 1

    def test_a_pre_tracking_job_reports_the_submitted_definition(self):
        client, _ = scripted(
            {
                ("GET", "/api/jobs/job-1/workflow"): (
                    200,
                    {"id": "job-1", "definition": WORKFLOW, "realized": False},
                )
            }
        )

        result = diagnose.get_job_workflow(client, "job-1")

        assert result["realized"] is False
        assert result["workflow"] == WORKFLOW

    def test_next_names_the_two_tools_that_use_it(self):
        client, _ = scripted(
            {
                ("GET", "/api/jobs/job-1/workflow"): (
                    200,
                    {"id": "job-1", "definition": WORKFLOW, "realized": True},
                )
            }
        )

        result = diagnose.get_job_workflow(client, "job-1")

        assert "save_workflow" in result["next"]
        assert "run_workflow" in result["next"]

    def test_an_unknown_job_raises_the_client_error(self):
        client, _ = scripted({})

        with pytest.raises(DwApiError):
            diagnose.get_job_workflow(client, "nope")
