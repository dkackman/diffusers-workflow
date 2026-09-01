"""Queue a run and work out what happened.

Two rules shape this module. A run costs real GPU time on an engine that
runs one job at a time, so `run_workflow` - and `rerun_job`, which queues
the same work - refuses until the caller has acknowledged that. And a generation takes minutes, longer than any MCP
client will hold a tool call open, so submitting returns immediately and
progress is polled from the event log.
"""

from dw.mcp.client import DwApiError, path_segment

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
    """Queue a workflow. Returns as soon as it is queued - it does not wait
    for the job to finish. Poll `get_job_events` for progress."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    if (workflow_path is None) == (inline_workflow is None):
        raise DwApiError(
            "Provide exactly one of `workflow_path` (a workflow on the "
            "server) or `inline_workflow` (a definition to run as-is)."
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
    return client.get_json(f"/api/jobs/{path_segment(job_id)}")


def get_job_events(client, job_id, after=-1, limit=200):
    """One page of a job's progress events. `after` is exclusive - pass back
    the previous call's `last_seq` to continue."""
    return client.get_json(
        f"/api/jobs/{path_segment(job_id)}/event-log",
        params={"after": after, "limit": limit},
    )


def cancel_job(client, job_id):
    """Ask a queued or running job to stop."""
    return client.post_json(f"/api/jobs/{path_segment(job_id)}/cancel")


def rerun_job(client, job_id, acknowledged_cost=False):
    """Queue a fresh job from a previous job's stored spec. This costs the
    same GPU time as `run_workflow` and passes through the same gate - a
    rerun is a run, and the gate would be worth nothing if a job id bought
    a way around it."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    return client.post_json(f"/api/jobs/{path_segment(job_id)}/rerun")


def move_job(client, job_id, direction):
    """Reorder a queued job: up, down, front, or back."""
    return client.post_json(
        f"/api/jobs/{path_segment(job_id)}/move", {"direction": direction}
    )
