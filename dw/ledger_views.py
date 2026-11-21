"""The requests ledger's views: a stored request as read, with what its
linked jobs derive.

Pure functions over the stored request and the job dicts the caller hands
over (`{job, item, status, run_id, run_dir, run_exists, note, created_at,
started_at, finished_at, output}`); nothing here reads the job store or the
file. An item's `status`, `ready`, `attempts`, `result` and
`spent_minutes`, a request's `spent_minutes` and `remaining_minutes`, a list
row's `cover`, and each note reference's `attempt` and `gone` are derived
here at read time and never stored. `dw/ledger.py` owns the file, the
writes and their rules, and answers these views as `ledger.request_view`
and `ledger.summary_view`.
"""

# The list's `cover`: this many result outputs, found among at most
# COVER_SCAN of a request's latest results - the bound on what one row of a
# polled list may read
COVER_SIZE = 3
COVER_SCAN = 8


def _minutes(job):
    """How long a linked job ran, in minutes to two places, or None when it
    has not both started and finished."""
    started, finished = job.get("started_at"), job.get("finished_at")
    if started is None or finished is None:
        return None
    return round(max(finished - started, 0) / 60, 2)


def _item_status(item, jobs):
    """An item's status, derived from the jobs linked to it: dropped; done
    (any succeeded); running; queued; failed (every linked job failed or was
    cancelled); planned (none)."""
    if item.get("dropped"):
        return "dropped"
    statuses = {job.get("status") for job in jobs}
    if "succeeded" in statuses:
        return "done"
    if "running" in statuses:
        return "running"
    if "queued" in statuses:
        return "queued"
    if statuses:
        return "failed"
    return "planned"


def _output(job):
    """A linked job's proof - `{name, kind}` of the one output that stands
    for it, or None - while its run directory exists. The caller derives
    it; one handed over as a callable is read only here, so a view pays
    only for the proofs it shows."""
    output = job.get("output") if job.get("run_exists") else None
    return output() if callable(output) else output


def _result(item, jobs):
    """`(job, accepted_missing)`: the accepted job when it is set and its
    run still exists, else the latest-finished succeeded job whose run
    exists, else None."""
    accepted = item.get("accepted")
    missing = False
    if accepted:
        for job in jobs:
            if job.get("job") == accepted.get("job") and job.get("run_exists"):
                return job, False
        missing = True
    kept = [
        job
        for job in jobs
        if job.get("status") == "succeeded" and job.get("run_exists")
    ]
    if not kept:
        return None, missing
    return max(kept, key=lambda job: job.get("finished_at") or 0), missing


def _ready(request, grouped):
    """The keys of the items a job could be queued for now: planned, with a
    stored workflow and every dependency done, in an open request."""
    if request.get("state") != "open":
        return set()
    items = request.get("items", [])
    status = {
        i.get("key"): _item_status(i, grouped.get(i.get("key"), [])) for i in items
    }
    return {
        i.get("key")
        for i in items
        if status[i.get("key")] == "planned"
        and i.get("workflow")
        and all(status.get(dep) == "done" for dep in i.get("depends_on") or [])
    }


def _cover(request, grouped):
    """Up to COVER_SIZE outputs of the latest results, newest first, reading
    at most COVER_SCAN proofs."""
    results = []
    for item in request.get("items", []):
        job, _ = _result(item, grouped.get(item.get("key"), []))
        if job is not None and not item.get("dropped"):
            results.append(job)
    results.sort(key=lambda job: job.get("finished_at") or 0, reverse=True)
    cover = []
    for job in results[:COVER_SCAN]:
        output = _output(job)
        if output and output not in cover:
            cover.append(output)
        if len(cover) == COVER_SIZE:
            break
    return cover


def _jobs_by_item(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault(job.get("item"), []).append(job)
    for group in grouped.values():
        group.sort(key=lambda job: job.get("created_at") or 0)
    return grouped


def _positions(jobs):
    """Each linked job's `(item, attempt, held)`: its item, its 1-based
    place among `jobs` (the attempts of one item, oldest first), and whether
    what it made still stands - its run exists, or it is queued or running
    and has opened no run yet."""
    return {
        job.get("job"): (job.get("item"), place, _held(job))
        for place, job in enumerate(jobs, 1)
    }


def _held(job):
    if job.get("run_exists"):
        return True
    return not job.get("run_dir") and job.get("status") in ("queued", "running")


def _read_ref(ref, positions, live):
    """One stored reference as read: `attempt` when it names a job (the
    job's 1-based place among its item's attempts, null when the job is no
    longer linked to that item), and `gone: True` when its item is not in
    `live` (dropped or out of the plan), or its job is unlinked, or its run
    is gone (a job that opened a run since removed, or ended with none)."""
    gone = ref.get("item") not in live
    out = dict(ref)
    if "job" in ref:
        item, attempt, held = positions.get(ref["job"], (None, None, False))
        if item != ref.get("item"):
            attempt, held = None, False
        out["attempt"] = attempt
        gone = gone or not held
    if gone:
        out["gone"] = True
    return out


def _noted(notes, positions, live):
    """The notes as read, each reference in a note's `refs` enriched by
    `_read_ref`. `live` is the keys of the items in the plan and not
    dropped. The stored notes are not changed."""
    out = []
    for note in notes or []:
        if note.get("refs"):
            refs = [_read_ref(ref, positions, live) for ref in note["refs"]]
            note = {**note, "refs": refs}
        out.append(note)
    return out


def _live_keys(items):
    return {item.get("key") for item in items if not item.get("dropped")}


def item_view(item, jobs, ready=False):
    """One item as read, with what its linked `jobs` derive: `status`,
    `ready`, `attempts` (every linked job, oldest first, as `{job, status,
    note, minutes, output}` - `note` only when set, `minutes` null until it
    has run, `output` its proof or null), `result` (`{job, run_id,
    output}`), `accepted_missing` (only when true) and `spent_minutes`."""
    attempts = [
        {
            "job": job.get("job"),
            "status": job.get("status"),
            **({"note": job["note"]} if job.get("note") else {}),
            "minutes": _minutes(job),
            "output": _output(job),
        }
        for job in jobs
    ]
    chosen, missing = _result(item, jobs)
    result = chosen and {
        "job": chosen["job"],
        "run_id": chosen.get("run_id"),
        "output": attempts[jobs.index(chosen)]["output"],
    }
    return {
        **item,
        "notes": _noted(item.get("notes"), _positions(jobs), _live_keys([item])),
        "status": _item_status(item, jobs),
        "ready": ready,
        "attempts": attempts,
        "result": result,
        **({"accepted_missing": True} if missing else {}),
        "spent_minutes": _spent(jobs),
    }


def _spent(jobs):
    return round(sum(_minutes(job) or 0 for job in jobs), 2)


def _remaining(request, grouped):
    """What is left to do: the estimates of the items not done and not
    dropped."""
    total = 0
    for item in request.get("items", []):
        if _item_status(item, grouped.get(item.get("key"), [])) in ("done", "dropped"):
            continue
        total += item.get("estimate_minutes") or 0
    return round(total, 2)


def request_view(request, jobs=()):
    """A stored request as read: every item with its derived fields
    (`item_view`) and the request's `spent_minutes` and `remaining_minutes`.
    `jobs` are the linked jobs (`{job, item, status, run_id, run_dir,
    run_exists, note, created_at, started_at, finished_at, output}`)."""
    grouped = _jobs_by_item(jobs)
    ready = _ready(request, grouped)
    positions = {}
    for key, group in grouped.items():
        if key is not None:
            positions.update(_positions(group))
    view = {
        **request,
        "notes": _noted(
            request.get("notes"), positions, _live_keys(request.get("items", []))
        ),
        "agent_wrote_at": request.get("agent_wrote_at"),
    }
    view["items"] = [
        item_view(item, grouped.get(item.get("key"), []), item.get("key") in ready)
        for item in request.get("items", [])
    ]
    view["spent_minutes"] = _spent(jobs)
    view["remaining_minutes"] = _remaining(request, grouped)
    return view


def summary_view(request, jobs=()):
    """One row of the list: what a request is, without its items."""
    grouped = _jobs_by_item(jobs)
    counts = {}
    for item in request.get("items", []):
        status = _item_status(item, grouped.get(item.get("key"), []))
        counts[status] = counts.get(status, 0) + 1
    history = request.get("history") or []
    return {
        "id": request.get("id"),
        "title": request.get("title"),
        "state": request.get("state"),
        "revision": request.get("revision"),
        "items": len(request.get("items", [])),
        "status_counts": counts,
        "ready": len(_ready(request, grouped)),
        "cover": _cover(request, grouped),
        "spent_minutes": _spent(jobs),
        "remaining_minutes": _remaining(request, grouped),
        "updated_at": history[-1].get("at") if history else None,
    }
