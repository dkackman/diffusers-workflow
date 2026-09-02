"""Queue a run and work out what happened.

Two rules shape this module. A run costs real GPU time on an engine that
runs one job at a time, so `run_workflow` - and `rerun_job`, which queues
the same work - refuses until the caller has acknowledged that. And a generation takes minutes, longer than any MCP
client will hold a tool call open, so submitting returns immediately and
progress is polled from the event log.
"""

import time

from dw_mcp.client import DwApiError, api_path

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

# How often wait_for_job re-polls /api/jobs/{id} - matches SSE_POLL_SECONDS,
# the interval the SSE stream itself re-checks a job at (dw/server/app.py).
WAIT_POLL_SECONDS = 1.0

# A generation can run for minutes, far longer than an MCP client holds a
# tool call open, so wait_for_job's own budget stays well under that no
# matter what a caller asks for.
MAX_WAIT_SECONDS = 55

COST_REFUSAL = (
    "Running a workflow occupies the GPU for minutes and the engine runs one "
    "job at a time. Tell the user what is about to run, get their go-ahead, "
    "then call again with acknowledged_cost=true. `validate_workflow` is free "
    "and checks the definition first."
)


def run_workflow(
    client,
    workflow_path=None,
    inline_workflow=None,
    arguments=None,
    acknowledged_cost=False,
):
    """Queue a workflow. `workflow_path` is either a catalog name from
    `list_workflows` or a path to a workflow file on the server. Returns as
    soon as it is queued - it does not wait for the job to finish. Poll
    `get_job_events` for progress."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    if (workflow_path is None) == (inline_workflow is None):
        raise DwApiError(
            "Provide exactly one of `workflow_path` (a catalog name or a "
            "path to a workflow on the server) or `inline_workflow` (a "
            "definition to run as-is)."
        )
    payload = {"arguments": arguments or {}}
    if workflow_path is not None:
        payload["workflow_path"] = workflow_path
    else:
        payload["workflow"] = inline_workflow
    # base_dir is deliberately absent: it decides where an inline workflow's
    # relative paths resolve, and the MCP surface does not hand that out
    job = client.post_json("/api/jobs", payload)
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "queue_position": job.get("queue_position"),
        "next": "Poll get_job_events(job_id) for progress, then get_job(job_id) "
        "for the manifest or the error.",
    }


def get_job(client, job_id):
    """A job's status, arguments, warnings, manifest, error and traceback."""
    return client.get_json(api_path("api", "jobs", job_id))


def get_job_events(client, job_id, after=-1, limit=200):
    """One page of a job's progress events. `after` is exclusive - pass back
    the previous call's `last_seq` to continue."""
    return client.get_json(
        api_path("api", "jobs", job_id, "event-log"),
        params={"after": after, "limit": limit},
    )


def wait_for_job(client, job_id, timeout_seconds=20):
    """Block until a job reaches a terminal status, or `timeout_seconds`
    elapses - a bounded alternative to polling `get_job`/`get_job_events` by
    hand. Does not queue anything, so it does not require
    `acknowledged_cost`; it only reads a job someone already queued.

    `timeout_seconds` is clamped to [0, MAX_WAIT_SECONDS]: a generation can
    run for minutes, far longer than an MCP client holds a tool call open,
    so this never blocks past a budget kept well under that. Returns as
    soon as the job's status is succeeded, failed or cancelled. If the
    timeout elapses first, returns the job's last-seen status with
    `still_running: true` instead of hanging - call again to keep waiting."""
    timeout_seconds = max(0.0, min(float(timeout_seconds), MAX_WAIT_SECONDS))
    deadline = time.monotonic() + timeout_seconds
    while True:
        job = client.get_json(api_path("api", "jobs", job_id))
        status = job.get("status")
        if status in TERMINAL_STATUSES:
            return {
                "job_id": job_id,
                "status": status,
                "still_running": False,
                "job": job,
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "job_id": job_id,
                "status": status,
                "still_running": True,
                "job": job,
                "next": "Call wait_for_job again, or get_job_events for "
                "incremental progress.",
            }
        time.sleep(min(WAIT_POLL_SECONDS, remaining))


def cancel_job(client, job_id):
    """Ask a queued or running job to stop."""
    return client.post_json(api_path("api", "jobs", job_id, "cancel"))


def rerun_job(client, job_id, acknowledged_cost=False):
    """Queue a fresh job from a previous job's stored spec. This costs the
    same GPU time as `run_workflow` and passes through the same gate - a
    rerun is a run, and the gate would be worth nothing if a job id bought
    a way around it."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    return client.post_json(api_path("api", "jobs", job_id, "rerun"))


def move_job(client, job_id, direction):
    """Reorder a queued job: up, down, front, or back."""
    return client.post_json(
        api_path("api", "jobs", job_id, "move"), {"direction": direction}
    )
