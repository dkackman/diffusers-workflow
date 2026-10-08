"""Turning a running job's worker messages into the job's record.

`consume_results` reads one worker's replies until the run ends and folds
each into the Job: progress events, logs, memory readings, and the manifest
the run wrote. Every file name is reported relative to the job's own output
directory - the rendering `get_job` and the `step_end`/`workflow_end`
events must all agree on (#284).

The functions take the output directory, the slot and the memory recorder
as arguments and hold no reference to the JobManager. They run on the job's
thread under `slot.lock`: `JobManager._run_on` is the caller.
"""

import os
import logging

from ..worker_protocol import (
    Cancelled,
    Failed,
    MemoryInfo,
    Output,
    Progress,
    Succeeded,
    UnknownReply,
    WorkerCrashed,
    WorkflowLoaded,
    parse_reply,
)
from .job_record import CANCELLED, FAILED, SUCCEEDED
from .outputs import output_kinds

logger = logging.getLogger("dw")


def relative_output_names(paths, output_dir):
    """The worker reports absolute paths; clients build '/outputs/<name>'
    URLs, and a run writes under '<output_dir>/<identity>/<run id>/'
    (dw/workflow.py's effective_output_dir) - so every file is reported
    by its name relative to the output directory of the job that wrote
    it, with forward slashes. A path outside it (a task step writing
    elsewhere) is left as it came.

    The job's own directory, not the manager's: a job in a named
    workspace writes under that workspace, and naming it relative to the
    default workspace would produce '../<name>/outputs/...' - a path, not
    a name."""
    names = []
    for path in paths:
        relative = os.path.relpath(path, output_dir)
        if relative.startswith(".."):
            names.append(path)
        else:
            names.append(relative.replace(os.sep, "/"))
    return names


def relative_manifest(manifest, output_dir):
    """A manifest list with every entry's 'files' relativised - the
    rendering `get_job` and `step_end`/`workflow_end` events must all
    agree on (#284)."""
    return [
        (
            {**entry, "files": relative_output_names(entry["files"], output_dir)}
            if "files" in entry
            else entry
        )
        for entry in manifest
    ]


def record_manifest(job, manifest, output_dir):
    """What the run wrote, named the way clients address outputs.

    Recorded for a failed or cancelled run as well as a successful one -
    the files the steps before the stop wrote are on disk either way,
    and a manifest that omits them is the difference between "this run
    produced nothing" and "this run produced four of five shots"
    (T015)."""
    job.manifest = relative_manifest(
        manifest if manifest is not None else [], output_dir
    )


def record_progress(job, event, output_dir):
    """One run event onto the job, its file names relativised the way
    clients address outputs."""
    if "files" in event:
        event["files"] = relative_output_names(event["files"], output_dir)
        # A running job's page renders each output as its step ends,
        # before there is a manifest to classify
        event["output_kinds"] = output_kinds([{"files": event["files"]}])
    if "manifest" in event:
        # workflow_end carries the run's full manifest nested under this
        # key - it must match get_job's rendering of the same list rather
        # than leaking absolute paths (#284)
        event["manifest"] = relative_manifest(event["manifest"], output_dir)
    if event.get("event") == "run_start":
        job.run_id = event.get("run_id")
        job.run_dir = event.get("run_dir")
        job.run_version = event.get("version")
    job.add_event(event)


def record_run_memory(job, info, slot, record_memory):
    """A memory reading the run reported. One per phase boundary now,
    not just once post-run (#273) - each folds into the cached reading
    memory_status() answers from while the job is busy, which is what
    makes that call fresh instead of a refusal for the run's whole
    duration. `record_memory(info, slot)` keeps it as the card's reading
    (worker_memory.record_memory)."""
    record_memory(info, slot)
    job.add_event({"event": "memory", "info": slot.last_memory})
    # The worker's own high-water mark, latest reading wins (it is
    # monotonic for the process' life) - persisted as a real column
    # rather than only inside the trimmed event tail (#243)
    info = info or {}
    peak = info.get("host_memory_peak_rss_mb")
    if peak is not None:
        job.host_memory_peak_rss_mb = peak
    # This job's own contribution to that process-lifetime peak, computed
    # against its own baseline (#272) - max() is defensive; by
    # construction each reading only grows
    job_peak = info.get("host_memory_job_peak_rss_mb")
    if job_peak is not None:
        job.host_memory_job_peak_rss_mb = max(
            job_peak, job.host_memory_job_peak_rss_mb or 0
        )


def consume_results(job, manager, slot, output_dir, record_memory):
    """Read `manager`'s worker messages until the run ends; returns the
    terminal (status, error, traceback) for the caller to apply once the
    card no longer counts the job as current. `slot` is the card's: it
    keeps the run's memory readings, through `record_memory`. `output_dir`
    is the job's own, which its file names are reported relative to."""
    while True:
        try:
            message = manager.get_result()
        except RuntimeError as e:
            # The worker died without managing to send anything - a
            # signal, not an exception, so worker_main's handler never
            # ran and there is no traceback to be had. The exit code is
            # the only diagnosis available, and marking the crash here
            # matters beyond this job: the manager would otherwise go on
            # believing a dead process is active, and every later call
            # that talks to it (memory_status above all) would fail
            # against a queue nobody is reading
            detail = manager.crash_details()
            manager.mark_crashed()
            reason = f"Worker process died: {detail or e}"
            logger.error(f"Job {job.id}: {reason}")
            return (FAILED, reason, None)
        reply = parse_reply(message)

        if isinstance(reply, Progress):
            record_progress(job, dict(reply.event), output_dir)
        elif isinstance(reply, Output):
            job.add_event({"event": "log", "message": reply.message or ""})
        elif isinstance(reply, WorkflowLoaded):
            job.add_event({"event": "log", "message": reply.workflow_name or ""})
        elif isinstance(reply, MemoryInfo):
            record_run_memory(job, reply.info, slot, record_memory)
        elif isinstance(reply, Succeeded):
            record_manifest(job, reply.manifest, output_dir)
            return (SUCCEEDED, None, None)
        elif isinstance(reply, Cancelled):
            record_manifest(job, reply.manifest, output_dir)
            return (CANCELLED, None, None)
        elif isinstance(reply, Failed) and reply.request_id is None:
            # A failed run's steps too: the ones before the failure wrote
            # real files, and a job that reports an empty manifest hides
            # them behind the error that stopped the run
            record_manifest(job, reply.manifest, output_dir)
            return (FAILED, reply.message, reply.traceback)
        elif isinstance(reply, WorkerCrashed):
            manager.mark_crashed()
            return (
                FAILED,
                f"Worker crashed: {reply.message}",
                reply.traceback,
            )
        elif isinstance(reply, UnknownReply):
            logger.warning(f"Unknown worker message type: {reply.type}")
        else:
            # A request's reply (memory_status, memory_cleared,
            # probe_cache, or an error answering one) whose reader gave
            # up before it landed - nobody is waiting for it now
            logger.debug(f"Discarding a stray worker reply: {reply.TYPE}")
