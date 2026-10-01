"""The job routes: queue, list, read, rerun, export, reorder, cancel and
stream a job, and the free pre-flight (`POST /api/validate`) that checks a
request the way a queued one would be checked.

Handlers read the server's state from `request.app.state`; nothing here
closes over an app, so two apps in one process never share a job manager.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Union
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...events import select_kinds
from ...host_memory_projection import CEILING_FRACTION, host_memory_warnings
from ...plan import build_plan, gate_warnings
from ...schema import format_validation_errors
from ...security import SecurityError
from ...library import SubWorkflowNotFound, resolve_sub_workflow
from ...workspace import Workspace
from ..admission import (
    ACKNOWLEDGED_COST_FIELD,
    AcknowledgedCost,
    JobRequest,
    ValidatorFailure,
    acknowledgement_form,
    admit_for,
    bound_plan_for,
    check_bound_acknowledgement,
)
from ..catalog import (
    catalog_name_for,
    catalog_name_from_root,
    resolve_workflow_reference,
)
from ..deps import (
    internal_error,
    observed_for_name,
    selected_workspace,
    sources_for,
    workspace_for,
)
from ..exports import export_job
from ..http_security import query_token_ok
from ..job_record import (
    ACK_BOUND,
    MAX_PERSISTED_EVENTS,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
)
from ..outputs import absolute_served_url, asset_library_for_job, served_url

logger = logging.getLogger("dw")

router = APIRouter()

# How long one SSE poll waits for a new event before checking liveness
SSE_POLL_SECONDS = 1.0


def _historical_log_note(stored):
    """What a restored job's event page has to admit about itself.

    History keeps only the last MAX_PERSISTED_EVENTS of a run, so a page can
    be complete as a page and still be missing the start of the job. The
    first stored event's seq is the direct signal: anything above zero means
    the head was dropped at record time. Length is not the signal - a job
    that emitted exactly MAX_PERSISTED_EVENTS events lost nothing.
    """
    if not stored:
        return "This job kept no event log - events were not retained with job history."
    if stored[0].get("seq", 0) > 0:
        return (
            f"Only the last {MAX_PERSISTED_EVENTS} events of this job were "
            f"retained; everything before seq {stored[0]['seq']} was dropped "
            "when the job was recorded."
        )
    return None


@router.post("/api/jobs", status_code=201)
def submit_job(
    http_request: Request,
    request: JobRequest,
    ws: Workspace = Depends(selected_workspace),
):
    """Queue a workflow. The workspace it runs in comes from the body or,
    for a client that scopes every call the same way, the query string -
    the body wins when both are given."""
    state = http_request.app.state
    manager = http_request.app.state.job_manager
    try:
        workspace = workspace_for(state, request.workspace or ws.name)
        resolved, source = resolve_workflow_reference(
            request.workflow_path, sources_for(state, workspace)
        )
        # The root this run is confined to: the source the workflow came
        # from, so an example runs where it lives while an inline
        # definition stays held to this workspace's own workflows
        workflow_dir = source.root if source else workspace.workflows
        form = acknowledgement_form(request.acknowledged_cost)
        # Everything POST /api/validate checks, so a caller who skipped
        # the free pre-flight still gets no job id for a run that cannot
        # start - and a bound acknowledgement is checked against the plan
        # of the run admitted, before anything is queued (#85)
        admission = admit_for(
            state,
            workspace,
            workflow_path=resolved,
            workflow=request.workflow,
            arguments=request.arguments,
            base_dir=request.base_dir,
            output_dir=workspace.outputs,
            workflow_dir=workflow_dir,
            plan_for=(
                bound_plan_for(request.arguments, workspace)
                if form == ACK_BOUND
                else None
            ),
        )
        if not admission.ok:
            raise ValueError(admission.message())
        if form == ACK_BOUND:
            check_bound_acknowledgement(
                admission.plan, request.acknowledged_cost, workspace
            )
    except HTTPException:
        raise
    except Exception as e:
        # workflow_from_file / validate / the security layer all raise for
        # bad requests - everything up to here is the client's fault
        raise HTTPException(status_code=400, detail=str(e))
    # Admission succeeded: from here a failure is the server's, not the
    # request's, and its message may carry internals - the log keeps it
    try:
        job = manager.submit(
            # The Workflow admission built and checked - what the job
            # runs, whatever happens to the file while it waits
            admitted=admission.workflow,
            workflow_path=resolved,
            workflow=request.workflow,
            arguments=request.arguments,
            base_dir=request.base_dir,
            workflow_dir=workflow_dir,
            # The roots this job runs against, so it stays in its
            # workspace however many others the server serves meanwhile
            output_dir=workspace.outputs,
            asset_dir=workspace.assets,
            workspace=workspace.name,
            # The listing name, when the request came as one - what a
            # later runtime-by-workflow report joins on. Derived from
            # the resolved path rather than echoing what was asked
            # for, so 'Basic', 'Basic.json' and an absolute path
            # inside the source all record the one catalog name
            catalog_name=catalog_name_for(resolved, source),
            acknowledged=form,
            acknowledged_cost=(
                request.acknowledged_cost.model_dump() if form == ACK_BOUND else None
            ),
            # The job carries every warning validate would have answered
            warnings=admission.warnings,
        )
        return manager.describe(job)
    except HTTPException:
        raise
    except Exception:
        raise internal_error("Job submission failed after admission")


@router.get("/api/jobs")
def list_jobs(
    request: Request,
    workspace: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
):
    """All jobs by default - a plain filter, not `selected_workspace`,
    since the jobs list spans every workspace the server holds unless a
    caller asks to narrow it.

    `status` narrows to one state or a comma-separated set of them.
    `limit` keeps the newest N, and `total` always reports how many
    matched before the cut, so a caller can tell a bounded answer from a
    complete one. The default is still every matching job, oldest first -
    what the web UI polls."""
    manager = request.app.state.job_manager
    statuses = [part.strip() for part in status.split(",")] if status else None
    statuses = [part for part in statuses if part] if statuses else None
    if statuses:
        unknown = [
            state
            for state in statuses
            if state not in (QUEUED, RUNNING, *TERMINAL_STATES)
        ]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown job status {', '.join(unknown)} - one of "
                f"{', '.join((QUEUED, RUNNING, *TERMINAL_STATES))}",
            )
    jobs = manager.list(workspace=workspace, statuses=statuses)
    total = len(jobs)
    if limit is not None:
        if limit < 0:
            raise HTTPException(status_code=400, detail="limit must not be negative")
        # the newest are the interesting ones, and the list is oldest
        # first - so the cut comes off the front, not the back. max(0, ...)
        # because a limit above what matched is no cut at all: a bare
        # negative start would be read from the end instead, and answer a
        # limit of 12 against 9 matching jobs with the last 3 of them
        jobs = jobs[max(0, len(jobs) - limit) :] if limit else []
    return {"jobs": jobs, "total": total}


@router.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    manager = request.app.state.job_manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    # a historical job is already a detail dict; a live one renders itself
    return job if isinstance(job, dict) else manager.describe(job)


@router.get("/api/jobs/{job_id}/workflow")
def get_job_workflow(request: Request, job_id: str):
    """The workflow this job ran, for the read-only graph on the job page
    and for `get_job_workflow` over MCP.

    `realized: true` means every mutable input is pinned - the copy the
    run itself wrote. `false` means the job predates run tracking (or its
    run directory is gone) and this is the definition as submitted. 404
    when neither is readable - the job itself still is."""
    manager = request.app.state.job_manager
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    realized = manager.realized(job_id)
    definition = realized if realized is not None else manager.definition(job_id)
    if definition is None:
        raise HTTPException(
            status_code=404, detail="No workflow definition for this job"
        )
    return {
        "id": job_id,
        "definition": definition,
        "realized": realized is not None,
        # Which variable a new-seed rerun would draw into, or null when
        # there is none - read from the workflow as written, since the
        # realized copy above has its seed pinned to the integer it used
        "seed_variable": manager.seed_variable(job_id),
    }


class RerunRequest(BaseModel):
    new_seed: bool = Field(
        default=False,
        description="Draw a fresh seed into the workflow's seed variable. "
        "Without it a rerun repeats the original arguments exactly, which "
        "the step cache serves from the earlier run - the same seed and "
        "inputs would produce the same files.",
    )
    acknowledged_cost: Optional[Union[bool, AcknowledgedCost]] = ACKNOWLEDGED_COST_FIELD


@router.post("/api/jobs/{job_id}/rerun", status_code=201)
def rerun_job(request: Request, job_id: str, body: RerunRequest = RerunRequest()):
    """Queue a fresh job from a previous job's stored spec, admitted as
    a new submission would be - a reference that resolved when the
    original ran may not any more. Takes `acknowledged_cost` as POST
    /api/jobs does; a bound one is checked against the stored spec's
    plan - the fresh seed of `new_seed` does not change a fingerprint."""
    state = request.app.state
    manager = request.app.state.job_manager
    form = acknowledgement_form(body.acknowledged_cost)
    try:
        prepared = manager.rerun_spec(job_id, new_seed=body.new_seed)
        if prepared is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        spec, arguments = prepared
        workspace = workspace_for(state, spec.get("workspace"))
        admission = admit_for(
            state,
            workspace,
            workflow_path=spec.get("workflow_path"),
            workflow=spec.get("workflow"),
            arguments=arguments,
            base_dir=spec.get("base_dir"),
            output_dir=spec.get("output_dir") or manager.output_dir,
            workflow_dir=spec.get("workflow_dir") or manager.workflow_dir,
            plan_for=(
                bound_plan_for(arguments, workspace) if form == ACK_BOUND else None
            ),
        )
        if not admission.ok:
            raise ValueError(admission.message())
        if form == ACK_BOUND:
            check_bound_acknowledgement(
                admission.plan, body.acknowledged_cost, workspace
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Admission succeeded: a failure from here is the server's (see submit_job)
    try:
        job = manager.rerun(
            job_id,
            acknowledged=form,
            acknowledged_cost=(
                body.acknowledged_cost.model_dump() if form == ACK_BOUND else None
            ),
            warnings=admission.warnings,
            # The arguments admitted - the fresh seed already drawn
            arguments=arguments,
            admitted=admission.workflow,
        )
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return manager.describe(job)
    except HTTPException:
        raise
    except Exception:
        raise internal_error("Job rerun failed after admission")


@router.post("/api/jobs/{job_id}/export", status_code=201)
def export_job_route(
    request: Request,
    job_id: str,
    overwrite: bool = False,
    ws: Workspace = Depends(selected_workspace),
):
    """Gather one finished job into '<workspace>/exports/<job id>/': the
    workflow it ran, the run's manifest, the job row, the media it used
    and the media it made, plus a README. 404 for an unknown job, 409 for
    one still running or for an export that already exists without
    `overwrite`.

    The three JSON files come back inline as well as on disk - the
    directory is on the server, and a client on another machine has no
    other way to read them without fetching the zip."""
    state = request.app.state
    manager = request.app.state.job_manager
    try:
        summary = export_job(
            manager,
            job_id,
            ws.root,
            asset_library_for_job(state, job_id, ws),
            overwrite=overwrite,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        message = str(e)
        if message.startswith("Unknown job"):
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=409, detail=message)
    body = summary.as_dict()
    zip_path = f"/exports/{quote(job_id)}.zip"
    body["zip_url"] = served_url(zip_path, ws)
    absolute_zip_url = absolute_served_url(zip_path, ws)
    if absolute_zip_url is not None:
        body["absolute_zip_url"] = absolute_zip_url
    # Same rule get_server_info's field states (#353): whether the zip
    # URL above needs a bearer token an MCP-only agent has no way to
    # attach itself, which is what tells the caller whether to fetch it
    # or hand it to the person.
    body["auth_required"] = bool(state.api_token)
    for key, name in (
        ("workflow", "workflow.json"),
        ("manifest", "manifest.json"),
        ("job", "job.json"),
    ):
        try:
            with open(os.path.join(summary.directory, name), "r") as file:
                body[key] = json.load(file)
        except (OSError, ValueError):
            body[key] = None
    return body


class MoveRequest(BaseModel):
    direction: str = Field(description="up, down, front, or back")


@router.post("/api/jobs/{job_id}/move")
def move_job(request: Request, job_id: str, body: MoveRequest):
    """Reorder a queued job. 409 once it is running or finished -
    only the waiting portion of the queue can be rearranged."""
    manager = request.app.state.job_manager
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    try:
        order = manager.move(job_id, body.direction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if order is None:
        raise HTTPException(
            status_code=409, detail="Job is not queued - only queued jobs move"
        )
    return {"id": job_id, "queue": order}


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str):
    manager = request.app.state.job_manager
    status = manager.cancel(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {"id": job_id, "status": status}


@router.get("/api/jobs/{job_id}/events")
@query_token_ok
async def job_events(request: Request, job_id: str, after: int = -1):
    """Server-sent events: every progress event from `after` (exclusive)
    until the job reaches a terminal state. Reconnect with the last seen
    seq (or let EventSource send Last-Event-ID) to resume without loss."""
    manager = request.app.state.job_manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if isinstance(job, dict):
        # historical jobs carry no event log - an immediately-closed
        # stream lets clients treat them uniformly
        return StreamingResponse(iter(()), media_type="text/event-stream")

    last_event_id = request.headers.get("last-event-id")
    if last_event_id is not None:
        try:
            after = max(after, int(last_event_id))
        except ValueError:
            pass

    async def stream():
        last_seq = after
        while True:
            events = job.events_after(last_seq)
            for event in events:
                last_seq = event["seq"]
                yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"
            if job.status in TERMINAL_STATES and not job.events_after(last_seq):
                return
            await asyncio.to_thread(job.wait_for_event, last_seq, SSE_POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/jobs/{job_id}/event-log")
def job_event_log(
    request: Request,
    job_id: str,
    after: int = -1,
    limit: int = 200,
    kinds: list[str] | None = Query(None),
):
    """Job events as one JSON page rather than a stream, for clients that
    poll instead of holding a connection open (the MCP server). `after` is
    exclusive, matching the SSE route's parameter of the same name.
    `kinds` restricts the page to events whose `event` or `kind` is one
    of the named values (e.g. `log`, `warning`, `phase_stall`) - a consumer confirming what a step applied wants
    those two and not the `memory`/bookkeeping events that otherwise
    dominate the payload."""
    manager = request.app.state.job_manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    limit = max(1, min(limit, 1000))
    if isinstance(job, dict):
        # A job restored from sqlite: history persists a bounded tail of
        # its events, so it can still explain itself after a restart
        status = job.get("status")
        stored = manager.history.events_for(job_id) or []
        pending = [event for event in stored if event.get("seq", -1) > after]
        note = _historical_log_note(stored)
    else:
        status = job.status
        pending = job.events_after(after)
        note = None
    pending = select_kinds(pending, kinds)
    page = pending[:limit]
    return {
        "id": job_id,
        "status": status,
        "events": page,
        "last_seq": page[-1]["seq"] if page else max(after, -1),
        # this page is cut short; `note` covers what record time dropped
        "truncated": len(pending) > len(page),
        "note": note,
    }


def _probe_command_for(candidate, request, workspace, workflow_dir):
    """The execute-shaped command a cache probe of this validate request
    needs - the same fields _run_job sends, built from the candidate
    admission checked, so the worker builds the workflow exactly as a
    job would. Its keys must be ProbeCache's fields (dw/worker_protocol.py):
    JobManager.probe_cache builds ProbeCache(**command), so a key that
    is not one raises TypeError there and the plan comes back null."""
    command = {
        "definition": candidate.workflow_definition,
        "file_spec": candidate.file_spec,
        "source": "path" if request.workflow_path is not None else "inline",
        "arguments": request.arguments,
        "output_dir": workspace.outputs,
        "workflow_dir": workflow_dir,
    }
    if workspace.assets:
        command["asset_dir"] = workspace.assets
    return command


def _validation_plan(state, candidate, request, workspace, source, catalog_name, sizes):
    """The plan a valid /api/validate answer carries: what the run will
    execute for these arguments, fingerprinted so an acknowledgement can
    be bound to it (#85). Best effort - None when it cannot be built,
    since the verdict is the schema's and the planner may not change it.
    """
    source_root = source.root if source else workspace.workflows
    definition = candidate.workflow_definition
    try:
        from ... import get_device, get_device_type

        command = _probe_command_for(candidate, request, workspace, source_root)

        def observed_for_child(path, child_definition, arguments=None):
            """A composed child's own observed figure, keyed by the
            catalog name it resolves to - so a parent with no figure of
            its own can quote what this box's runs of the *child* took
            rather than falling back to unknown (#268).

            `arguments` are the composing step's own overrides - the
            same role `arguments` plays for the top-level `observed`
            callback - so a child whose composing step shifted a
            declared scalar `cost_driver` (#341) is bucketed against
            *that* value rather than always the child's stored
            defaults, which silently answered the default bucket's
            history for every override."""
            base_dir = (
                os.path.dirname(os.path.abspath(candidate.file_spec))
                if candidate.file_spec
                else None
            )
            try:
                child_path, child_library_root = resolve_sub_workflow(
                    path, base_dir or ".", candidate.workflow_dir
                )
            except (SecurityError, OSError, ValueError, SubWorkflowNotFound):
                return None
            child_root = child_library_root.root if child_library_root else None
            child_name = catalog_name_from_root(child_path, child_root)
            if not child_name:
                return None
            # The workspace's own workflows/ is the writable one (#274). The
            # root's `writable` tag is not enough: the run's own confinement
            # is tagged writable whatever directory it is, so equality with
            # the workspace's workflows/ stays the test
            child_workspace = (
                workspace.name if child_root == workspace.workflows else None
            )
            return observed_for_name(
                state,
                child_name,
                child_definition,
                arguments,
                workspace=child_workspace,
            )

        return build_plan(
            candidate,
            request.arguments,
            device=get_device_type(get_device()),
            prompt_dir=workspace.prompts,
            lookup_sizes=sizes,
            cache_probe=lambda arguments: state.job_manager.probe_cache(
                {**command, "arguments": arguments}
            ),
            # What this box's own runs of this shape took, which is what
            # the estimate quotes ahead of a curated figure (#154) - the
            # same aggregate the listing reports, asked with the
            # caller's arguments rather than the defaults
            observed=(
                (
                    lambda arguments: observed_for_name(
                        state,
                        catalog_name,
                        definition,
                        arguments,
                        workspace=workspace.name if source.writable else None,
                    )
                )
                if catalog_name
                else None
            ),
            observed_for_child=observed_for_child,
        )
    except Exception:
        logger.exception("Plan could not be built")
        return None


@router.post("/api/validate")
def validate_workflow(
    http_request: Request,
    request: JobRequest,
    ws: Workspace = Depends(selected_workspace),
    sizes: bool = Query(
        True,
        description="Ask the hub how large each missing model is; false "
        "skips the network for a faster answer",
    ),
):
    """Schema-validate a workflow and check its pipeline arguments
    against real signatures, without queuing anything. Give either an
    inline workflow or a workflow_path - a path on the server or a
    stored workflow name from /api/workflows. The workspace it resolves
    in comes from the body or the query string, body first. A valid
    answer also carries a plan: the fingerprint of the work these
    arguments produce, the step count, the list lengths, the model
    repos not in the cache, and an estimate from the workflow's cost
    block."""
    state = http_request.app.state
    if (request.workflow is None) == (request.workflow_path is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of workflow or workflow_path",
        )
    try:
        workspace = workspace_for(state, request.workspace or ws.name)
        # Built from the file so relative paths inside it resolve against
        # its own directory, exactly as a run would; (None, None) for an
        # inline definition
        resolved, source = resolve_workflow_reference(
            request.workflow_path, sources_for(state, workspace)
        )
        # The listing name the job history is keyed on, so the plan can
        # quote what this box's own runs of it took (#154); an inline
        # definition has none, so no history
        catalog_name = catalog_name_for(resolved, source) if resolved else None
        admission = admit_for(
            state,
            workspace,
            workflow_path=resolved,
            workflow=request.workflow,
            arguments=request.arguments,
            base_dir=request.base_dir,
            output_dir=workspace.outputs,
            # Confined to the source it came from, not to the writable
            # root - an example is read where it lives
            workflow_dir=source.root if source else workspace.workflows,
            # `arguments` defaults to `{}` on the model, so an omitted
            # field and an explicit `{}` are otherwise indistinguishable
            # here - and the two mean different things: omitted is
            # "check the document", explicit is "check a run with these
            # arguments" (#364)
            supplied="arguments" in request.model_fields_set,
            plan_for=lambda candidate: _validation_plan(
                state, candidate, request, workspace, source, catalog_name, sizes
            ),
        )
    except HTTPException:
        raise
    except ValidatorFailure:
        # Not the schema's verdict on the workflow - validation_errors()
        # reports that by returning it. It is the validator itself
        # failing, and its message could carry internals, so the log
        # keeps the detail (admission logged it) and the client is told the
        # category
        detail = "The workflow could not be validated - the server log has the detail"
        return {
            "valid": False,
            "error": detail,
            "errors": [{"path": None, "message": detail}],
            "warnings": [],
        }
    except SecurityError as e:
        # Messages the security layer writes itself - safe to surface
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Anything else could carry internals in its message; the log
        # keeps the detail, the client gets the category. What is left
        # here is resolving the request and loading the workflow (and
        # the catalog it is checked against) - a check that fails after
        # loading is a ValidatorFailure, answered above
        logger.exception("Workflow could not be loaded for validation")
        raise HTTPException(
            status_code=400,
            detail="The workflow could not be loaded - the server log has the detail",
        )
    if admission.schema_errors:
        return {
            "valid": False,
            "error": format_validation_errors(admission.errors),
            "errors": admission.errors,
            "warnings": [],
        }
    if not admission.ok:
        # The arguments a caller is about to run with, checked the way
        # the run would check them. The warnings are the expansion's
        # over the defaults, since the arguments did not fold
        return {
            "valid": False,
            "error": format_validation_errors(admission.errors),
            "errors": admission.errors,
            "warnings": admission.warnings,
            "checked_arguments": sorted(request.arguments or {}),
        }
    answer = {
        "valid": True,
        "error": None,
        "errors": [],
        "warnings": list(admission.warnings),
    }
    if request.arguments:
        # Naming what was checked is the difference between 'the stored
        # definition is valid' and 'the values you are about to pass are'
        answer["checked_arguments"] = sorted(request.arguments)
    answer["plan"] = admission.plan
    if answer["plan"]:
        # cached_steps is 0 both when nothing hit and when the probe ran
        # against the wrong workspace's output root (#184) - echoing
        # what it was actually probed against turns the second case
        # from a silent miss into something a caller can read
        answer["plan"]["workspace"] = workspace.name
        answer["plan"]["output_dir"] = workspace.outputs
        answer["warnings"] += gate_warnings(answer["plan"]["downloads_required"])
        if catalog_name:
            answer["warnings"] += _host_memory_warnings(
                state,
                catalog_name,
                admission.workflow.workflow_definition,
                answer["plan"]["list_entries"],
                workspace=workspace.name if source.writable else None,
            )
    return answer


def _host_memory_warnings(state, name, definition, list_entries, *, workspace=None):
    """Whether this box's own history says the requested list is
    projected to exceed host RAM (#243) - best effort, since a warning
    that 500s the free pre-flight would be worse than skipping it."""
    costs = getattr(state, "observed_costs", None)
    if costs is None:
        return []
    try:
        from ...host_memory import host_memory_stats

        rows = costs.rows_for(name, workspace=workspace)
        ceiling_mb = (host_memory_stats().get("total_mb") or 0) * CEILING_FRACTION
        return host_memory_warnings(definition, list_entries, rows, ceiling_mb)
    except Exception:
        logger.debug("host memory projection failed for %s", name, exc_info=True)
        return []
