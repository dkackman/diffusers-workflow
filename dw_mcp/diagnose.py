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
    workspace=None,
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
    # A named workspace pins this one job rather than the session: a
    # restarted session forgets use_workspace, and a job that resolves
    # output: references in the wrong root fails after it was queued
    params = {"workspace": workspace} if workspace else None
    job = client.post_json("/api/jobs", payload, params=params)
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "queue_position": job.get("queue_position"),
        "next": "Poll get_job_events(job_id) for progress, then get_job(job_id) "
        "for the manifest or the error.",
    }


def get_job(client, job_id):
    """A job's status, arguments, warnings, manifest, error and traceback.
    A running job also carries `progress` - the step, the phase and how long
    it has been in it, with a denoise counter that is null until that loop
    starts. Null under `generating` is the pipeline's silent lead-in, not a
    hang; see wait_for_job."""
    return client.get_json(api_path("api", "jobs", job_id))


def get_job_workflow(client, job_id):
    """The workflow a job ran. `realized: true` means every mutable input
    is pinned (arguments, seed, prompts, output:latest); false means the
    job predates run tracking and this is the definition as submitted.
    Pass it to save_workflow to rerun it by name, or edit it and pass it
    to run_workflow as inline_workflow."""
    body = client.get_json(api_path("api", "jobs", job_id, "workflow"))
    return {
        "job_id": job_id,
        "realized": bool(body.get("realized")),
        "workflow": body.get("definition"),
        # Which variable rerun_job(new_seed=True) would draw into, null when
        # the workflow has none - see rerun_job on why that matters
        "seed_variable": body.get("seed_variable"),
        "next": "Pass `workflow` to save_workflow to keep it in the catalog "
        "under a name, or edit it and pass it to run_workflow as "
        "inline_workflow.",
    }


def get_job_events(client, job_id, after=-1, limit=200):
    """One page of a job's progress events. `after` is exclusive - pass back
    the previous call's `last_seq` to continue."""
    return client.get_json(
        api_path("api", "jobs", job_id, "event-log"),
        params={"after": after, "limit": limit},
    )


_SLIM_KEYS = (
    "id",
    "workflow_name",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "workspace",
    "run_id",
    "queue_position",
    "warnings",
    "error",
    "event_count",
    # Where a running job has got to: the step, the phase and how long it
    # has been in it, plus the denoise counter when one is running. A
    # single-step generation emits nothing for minutes at a time, so this
    # is what separates a slow job from a hung one on a poll that would
    # otherwise come back byte-identical
    "progress",
)


def slim_job(job):
    """A job row without its arguments and traceback - what a poll needs.

    The arguments of an H3 workflow are thousands of tokens of prompt text,
    repeated on every poll of a long render; get_job serves them once. The
    manifest is kept only once the job is terminal, when it names files.
    """
    slim = {key: job.get(key) for key in _SLIM_KEYS if key in job}
    if job.get("status") in TERMINAL_STATUSES:
        slim["manifest"] = job.get("manifest")
    return slim


def wait_for_job(client, job_id, timeout_seconds=20):
    """Block until a job reaches a terminal status, or `timeout_seconds`
    elapses - a bounded alternative to polling `get_job`/`get_job_events` by
    hand. Does not queue anything, so it does not require
    `acknowledged_cost`; it only reads a job someone already queued.

    `timeout_seconds` is clamped to [0, MAX_WAIT_SECONDS]: a generation can
    run for minutes, far longer than an MCP client holds a tool call open,
    so this never blocks past a budget kept well under that. Every reply
    says what was applied - `timeout_applied_seconds` is the budget the
    call actually ran under, `timeout_requested_seconds` what was asked
    for, and `waited_seconds` how long this call blocked - so a caller
    asking for 600 can tell a capped return from an elapsed one rather
    than inferring it from wall clock. Returns as soon as the job's status
    is succeeded, failed or cancelled. If the timeout elapses first,
    returns the job's last-seen status with `still_running: true` instead
    of hanging - call again to keep waiting. Returns a slim job - status,
    warnings, error, and the manifest once finished - without the
    arguments; get_job has those.

    A running job carries `progress`: the step it is on, the phase
    (`loading`, `generating`, `decoding`, `saving`) with the model or step
    named in `phase_detail`, `seconds_in_phase`, `seconds_since_event`, and
    `denoise_step`/`denoise_total_steps`, which are null until the denoise
    loop starts. Two calls with the same phase and a growing
    `seconds_in_phase` but a moving `denoise_step` is a slow run; one where
    `denoise_step` is a number that does not move while
    `seconds_since_event` climbs is a stuck one.

    `denoise_step: null` under `generating` is neither: it is the lead-in
    the pipeline runs before the loop - encoding the prompt and any
    reference image or audio - which emits nothing and is well over a
    minute on a large video model (~90 s on MiniMax H3). Silence there is
    expected; `seconds_since_event` only says something once
    `denoise_step` is a number, or in any other phase."""
    requested = max(0.0, float(timeout_seconds))
    applied = min(requested, float(MAX_WAIT_SECONDS))
    capped = applied < requested
    started = time.monotonic()
    deadline = started + applied
    while True:
        job = client.get_json(api_path("api", "jobs", job_id))
        status = job.get("status")
        budget = {
            "waited_seconds": round(time.monotonic() - started, 1),
            "timeout_requested_seconds": round(requested, 1),
            "timeout_applied_seconds": round(applied, 1),
            "timeout_capped": capped,
        }
        if status in TERMINAL_STATUSES:
            return {
                "job_id": job_id,
                "status": status,
                "still_running": False,
                **budget,
                "job": slim_job(job),
                "next": "get_job(job_id) for the arguments and traceback, "
                "get_job_workflow(job_id) for the realized workflow.",
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            next_step = (
                "Call wait_for_job again, or get_job_events for incremental "
                "progress."
            )
            if capped:
                next_step = (
                    f"You asked to wait {round(requested, 1)}s but one call "
                    f"blocks for at most {MAX_WAIT_SECONDS}s, so this "
                    "returned early rather than timing out. The job is still "
                    "running: call wait_for_job again (each call covers "
                    f"~{MAX_WAIT_SECONDS}s of it), or get_job_events for "
                    "incremental progress."
                )
            return {
                "job_id": job_id,
                "status": status,
                "still_running": True,
                **budget,
                "job": slim_job(job),
                "next": next_step,
            }
        time.sleep(min(WAIT_POLL_SECONDS, remaining))


def cancel_job(client, job_id):
    """Ask a queued or running job to stop."""
    return client.post_json(api_path("api", "jobs", job_id, "cancel"))


def rerun_job(client, job_id, acknowledged_cost=False, new_seed=False):
    """Queue a fresh job from a previous job's stored spec. This costs the
    same GPU time as `run_workflow` and passes through the same gate - a
    rerun is a run, and the gate would be worth nothing if a job id bought
    a way around it.

    `new_seed` draws a fresh seed into the workflow's seed variable. Without
    it the arguments repeat exactly, and a seeded workflow's rerun is served
    whole from the step cache - the earlier run's files, republished in a
    fraction of a second, with `reused: true`. Ask for a new seed when the
    point is a different image rather than the same one again."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    return client.post_json(
        api_path("api", "jobs", job_id, "rerun"), {"new_seed": new_seed}
    )


def move_job(client, job_id, direction):
    """Reorder a queued job: up, down, front, or back."""
    return client.post_json(
        api_path("api", "jobs", job_id, "move"), {"direction": direction}
    )
