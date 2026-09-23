"""Queue a run and work out what happened.

Two rules shape this module. A run costs real GPU time on an engine that
runs one job at a time, so `run_workflow` - and `rerun_job`, which queues
the same work - refuses until the caller has acknowledged that. And a generation takes minutes, longer than any MCP
client will hold a tool call open, so submitting returns immediately and
progress is polled from the event log.
"""

import os
import time

from dw_mcp.client import DwApiError, api_path, coerce_json_object

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

# How often wait_for_job re-polls /api/jobs/{id} - matches SSE_POLL_SECONDS,
# the interval the SSE stream itself re-checks a job at (dw/server/app.py).
WAIT_POLL_SECONDS = 1.0

# A generation can run for minutes, far longer than an MCP client holds a
# tool call open, so wait_for_job's own budget stays well under that no
# matter what a caller asks for. Some clients hold a tool call open far
# longer than the 55s this was tuned against (#248), so a deployment that
# knows its own harness's tool-call budget can raise the cap with
# DW_MCP_MAX_WAIT_SECONDS - unset, it stays 55.
MAX_WAIT_SECONDS = float(os.environ.get("DW_MCP_MAX_WAIT_SECONDS", 55))

COST_REFUSAL = (
    "Running a workflow occupies the GPU for minutes and the engine runs one "
    "job at a time. Call `validate_workflow` with the arguments you will run "
    "with (free): its `plan` says what will execute - `estimate.minutes` with "
    "its `basis`, and any weights in `downloads_required` this box has to "
    "fetch first. Tell the user that number, get their go-ahead, then call "
    'again with acknowledged_cost bound to the plan: {"fingerprint": '
    'plan.fingerprint, "minutes": plan.estimate.minutes, "downloads": '
    "[each non-null downloads_required repo]} - the server then refuses (409) if the "
    "run's shape changed since. acknowledged_cost=true is for a `plan` that "
    "was null."
)


def _acknowledgement_body(acknowledged_cost):
    """What an acknowledgement adds to a request body: a bound one is the
    dict itself, verbatim, so the server compares what the agent quoted; a
    bare true is sent as true, so the job records `acknowledged: boolean`
    rather than reading as one that never passed a gate at all (#85). A
    dict without a fingerprint is a mistake caught here, before anything is
    queued."""
    if isinstance(acknowledged_cost, dict):
        if not acknowledged_cost.get("fingerprint"):
            raise DwApiError(
                "A bound acknowledged_cost needs `fingerprint` - the "
                "plan.fingerprint the validate answer carried. Validate again "
                "and pass {fingerprint, minutes, downloads} from its plan."
            )
        # A from_single_file URL sits in downloads_required with repo: null;
        # an agent copying the list verbatim should not earn a 422 for it
        downloads = acknowledged_cost.get("downloads")
        if isinstance(downloads, list):
            acknowledged_cost = {
                **acknowledged_cost,
                "downloads": [repo for repo in downloads if repo],
            }
        return {"acknowledged_cost": acknowledged_cost}
    return {"acknowledged_cost": bool(acknowledged_cost)}


def run_workflow(
    client,
    workflow_path=None,
    inline_workflow=None,
    workflow=None,
    name=None,
    arguments=None,
    acknowledged_cost=False,
    workspace=None,
    wait_seconds=0,
):
    """Queue a workflow. `workflow_path` (or `name` - the same thing
    `validate_workflow` calls it) is either a catalog name from
    `list_workflows` or a path to a workflow file on the server;
    `inline_workflow` (or `workflow` - the same thing `validate_workflow`
    calls it) is a full definition. A document just checked with
    `validate_workflow` can be handed straight to this call under either
    spelling. Returns as soon as it is queued - it does not wait for the job
    to finish. Poll `get_job_events` for progress.

    `wait_seconds` folds the first `wait_for_job` into this call: when it
    is above 0 the queued job is waited on exactly as
    `wait_for_job(job_id, timeout_seconds=wait_seconds)` would - same clamp
    to MAX_WAIT_SECONDS, same `waited_seconds` / `timeout_*` /
    `still_running` fields - and the answer carries the queued-job fields
    plus that wait's slim job. Almost every run is followed by a wait, and
    an unattended agent pays a whole tool turn for it; where the cap covers
    the job's runtime this one call is the run and the wait. The gate is
    untouched: queuing is refused before anything is waited on, and a
    refused queue (409 on a stale plan) returns nothing extra.

    `acknowledged_cost` is true or, better, the plan it was quoted from:
    {fingerprint, minutes, downloads} from `validate_workflow` - see
    COST_REFUSAL. A bound one the server checks; a 409 means the run's
    shape changed since the quote and the message carries the new plan."""
    if workflow_path is not None and name is not None:
        raise DwApiError(
            "`workflow_path` and `name` are the same thing - provide only one."
        )
    inline_workflow = coerce_json_object(inline_workflow, "inline_workflow")
    workflow = coerce_json_object(workflow, "workflow")
    if inline_workflow is not None and workflow is not None:
        raise DwApiError(
            "`inline_workflow` and `workflow` are the same thing - provide only one."
        )
    path = workflow_path if workflow_path is not None else name
    inline = inline_workflow if inline_workflow is not None else workflow
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    if (path is None) == (inline is None):
        raise DwApiError(
            "Provide exactly one of `workflow_path`/`name` (a catalog name "
            "or a path to a workflow on the server) or "
            "`inline_workflow`/`workflow` (a definition to run as-is)."
        )
    payload = {"arguments": arguments or {}}
    payload.update(_acknowledgement_body(acknowledged_cost))
    if path is not None:
        payload["workflow_path"] = path
    else:
        payload["workflow"] = inline
    # base_dir is deliberately absent: it decides where an inline workflow's
    # relative paths resolve, and the MCP surface does not hand that out
    # A named workspace pins this one job rather than the session: a
    # restarted session forgets use_workspace, and a job that resolves
    # output: references in the wrong root fails after it was queued
    params = {"workspace": workspace} if workspace else None
    job = client.post_json("/api/jobs", payload, params=params)
    queued = {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "queue_position": job.get("queue_position"),
        "workspace": job.get("workspace"),
        "next": "Poll get_job_events(job_id) for progress, then get_job(job_id) "
        "for the manifest or the error.",
    }
    if not wait_seconds or float(wait_seconds) <= 0 or queued["job_id"] is None:
        return queued
    # The same loop wait_for_job runs, not a second one: its clamp, its
    # budget fields and its `next` are what a caller already paces against.
    # The wait's status and next overwrite the queued ones, since the job
    # has moved on from "queued" by the time either is read
    waited = wait_for_job(client, queued["job_id"], timeout_seconds=wait_seconds)
    return {**queued, **waited}


def get_job(client, job_id):
    """A job's status, arguments, warnings, manifest, error and traceback.
    A running job also carries `progress` - the step, the phase and how long
    it has been in it, with a denoise counter that is null until that loop
    starts. Null under `generating` is the pipeline's silent lead-in, not a
    hang; see wait_for_job. A FAILED job keeps `progress` too, frozen at the
    moment it died - the phase it was in is the fastest way to tell what
    killed it, faster than reading `traceback`."""
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
    # The run's ordinal - the 'v5' the gallery labels its files with - so
    # the caller can name the run it just waited on without another call
    "run_version",
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
    so this never blocks past a budget kept well under that - 55s by
    default, raised for a deployment whose harness tolerates a longer tool
    call via the `DW_MCP_MAX_WAIT_SECONDS` env var, in which case one call
    can cover a whole short job rather than needing several polls. Every
    reply says what was applied - `timeout_applied_seconds` is the budget
    the call actually ran under, `timeout_requested_seconds` what was asked
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
    loop starts. `denoise_total_steps` is the schedule that actually runs,
    which is not always the `num_inference_steps` asked for - MiniMax H3
    runs N-1 evaluations for N (#110). Two calls with the same phase and a growing
    `seconds_in_phase` but a moving `denoise_step` is a slow run; one where
    `denoise_step` is a number that does not move while
    `seconds_since_event` climbs is a stuck one.

    `denoise_step: null` under `generating` is neither: it is the lead-in
    the pipeline runs before the loop - encoding the prompt and every
    reference - which emits nothing and is well over a minute on a large
    video model. Its length follows what it has to encode: ~90 s on
    MiniMax H3 for a prompt with an image or audio reference, ~10 min once
    a *video* reference is among them (measured 629 s for one 5 s 960x544
    clip on an RTX 3090). Silence there is expected, and `get_job_events`
    says which block it is inside while it lasts - one `log` line per
    top-level block of a modular pipeline. `seconds_since_event` only says
    something once `denoise_step` is a number, or in any other phase.

    Even then it is coarse: where a transformer block cache is configured
    the denoise steps are uneven - several cheap ones, then a full one -
    so on H3 a 140 s gap between steps is a healthy run. Read liveness as
    `denoise_step` having moved between polls minutes apart rather than as
    silence under a fixed threshold."""
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
                "Call wait_for_job again, or get_job_events for incremental progress."
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
    point is a different image rather than the same one again.

    `acknowledged_cost` takes the same bound form as `run_workflow`; a
    fresh seed never changes a fingerprint, so the original plan still
    binds a new-seed rerun."""
    if not acknowledged_cost:
        raise DwApiError(COST_REFUSAL)
    return client.post_json(
        api_path("api", "jobs", job_id, "rerun"),
        {"new_seed": new_seed, **_acknowledgement_body(acknowledged_cost)},
    )


def move_job(client, job_id, direction):
    """Reorder a queued job: up, down, front, or back."""
    return client.post_json(
        api_path("api", "jobs", job_id, "move"), {"direction": direction}
    )
