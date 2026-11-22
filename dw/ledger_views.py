"""The requests ledger's views: a stored request as read, with what its
linked jobs derive.

Pure functions over the stored request and the job dicts the caller hands
over (`{job, item, status, run_id, run_dir, run_exists, note, created_at,
started_at, finished_at, workflow, arguments, output, prompt_digests}`)
and, for rule 2, a `prompts` callable (`name -> sha256 of that prompt's
text now`, or None when it no longer resolves: `dw.prompts.PromptDigests`);
nothing here reads the job store or the file. An item's `status`, `ready`, `attempts`, `result` and
`spent_minutes` (and a revised item's `revised_because`), a request's `spent_minutes` and `remaining_minutes`, a list
row's `cover`, and each note reference's `attempt` and `gone` are derived
here at read time and never stored. `dw/ledger.py` owns the file, the
writes and their rules, and answers these views as `ledger.request_view`
and `ledger.summary_view`.
"""

from datetime import datetime, timezone

from . import references

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


def _epoch(text):
    """A stored ISO time as epoch seconds, or None when it is not one."""
    moment = _moment(text)
    return moment.timestamp() if moment is not None else None


def _current(item, job):
    """Rule 1: whether succeeded take `job` is current for `item`'s plan:
    the plan has not changed since the take was created (`changed_at` not
    later than its `created_at`), or the item's `workflow` and `arguments`
    equal the take's as submitted (`_same_workflow`)."""
    changed = _epoch(item.get("changed_at"))
    created = job.get("created_at")
    if changed is None or (created is not None and changed <= created):
        return True
    return _same_workflow(item, job) and (item.get("arguments") or {}) == (
        job.get("arguments") or {}
    )


def _same_workflow(item, job):
    """Whether the item names the workflow the take ran from. A take with no
    recorded catalog name (an inline definition, an older row) is not
    counted as a difference: no claim without evidence."""
    return job.get("workflow") is None or item.get("workflow") == job.get("workflow")


def _accepted_since(item):
    """Whether an acceptance was recorded at or after the plan's last
    change (`accepted_basis.at` >= `changed_at`)."""
    basis = item.get("accepted_basis")
    at = _epoch(basis.get("at")) if isinstance(basis, dict) else None
    changed = _epoch(item.get("changed_at"))
    return at is not None and changed is not None and at >= changed


def prompt_names(arguments):
    """The prompt names the `prompt:` references in `arguments` name,
    wherever they sit (a nested dict or list too), first-seen order - each
    as the run manifest records it (`ref_name(...).strip()`, dw/realize.py)."""
    names = []

    def walk(value):
        if isinstance(value, str):
            name = references.ref_name(references.PROMPT, value)
            if name is not None and name.strip() not in names:
                names.append(name.strip())
        elif isinstance(value, dict):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    walk(arguments or {})
    return names


def current_prompts(item, prompts):
    """`{name: digest}` for each prompt the item's current arguments
    reference that still resolves (`prompts(name)` not None). Empty with no
    `prompts` callable, so a caller that hands none gets rule 1 alone."""
    if prompts is None:
        return {}
    found = {}
    for name in prompt_names(item.get("arguments")):
        digest = prompts(name)
        if digest is not None:
            found[name] = digest
    return found


def _recorded_digests(job):
    digests = job.get("prompt_digests")
    if callable(digests):
        digests = digests()
    return digests if isinstance(digests, dict) else {}


def _moved_prompts(item, job, now):
    """Rule 2: the names in `now` (the item's current prompt digests) whose
    text differs from what take `job` ran with. For the accepted take, the
    acceptance basis's digest for a name wins when it holds one (a re-accept
    takes the text as it stood then); every other take is judged by its own
    recorded digests, so a newer take run on the current text is current
    even while an older one is accepted. A name with no recorded digest (a
    take from before digests were recorded) counts as equal, and a prompt
    that no longer resolves is not in `now`: no claim without evidence."""
    if not now:
        return []
    accepted = {}
    chosen = item.get("accepted")
    if isinstance(chosen, dict) and chosen.get("job") == job.get("job"):
        basis = item.get("accepted_basis")
        prompts = basis.get("prompts") if isinstance(basis, dict) else None
        accepted = prompts if isinstance(prompts, dict) else {}
    recorded = _recorded_digests(job)
    moved = []
    for name, digest in now.items():
        was = accepted.get(name, recorded.get(name))
        if isinstance(was, str) and was != digest:
            moved.append(name)
    return moved


def _take_current(item, job, now):
    """Whether take `job` is current: its plan holds (rule 1, or an
    acceptance recorded since the plan last changed) and no prompt it
    references has new text (rule 2)."""
    plan = _accepted_since(item) or _current(item, job)
    return plan and not _moved_prompts(item, job, now)


def is_revised(item, jobs, prompts=None):
    """Whether an item with a succeeded take reads `revised`: no succeeded
    take is current (`_take_current`) - its plan changed after every one and
    no acceptance was recorded since (rule 1), or a prompt its arguments
    reference has text other than the take ran with (rule 2; `prompts` the
    `name -> digest` callable, None for rule 1 alone). An item with no
    `changed_at` (stored before the stamp) is revised by rule 2 only."""
    takes = [job for job in jobs if job.get("status") == "succeeded"]
    if not takes:
        return False
    now = current_prompts(item, prompts)
    return not any(_take_current(item, job, now) for job in takes)


def revised_because(item, jobs, result=None, prompts=None):
    """What moved on a revised item, as a list (`["arguments"]`,
    `["workflow", "arguments"]`, `["prompt:p"]`, `["arguments", "prompt:p"]`):
    the fields that differ from its result take, else from its latest
    succeeded take, then each prompt whose text moved (`_moved_prompts`)."""
    take = result
    if take is None:
        takes = [job for job in jobs if job.get("status") == "succeeded"]
        take = takes[-1] if takes else {}
    plan_held = _accepted_since(item) or _current(item, take)
    because = []
    if not plan_held and not _same_workflow(item, take):
        because.append("workflow")
    if not plan_held and (item.get("arguments") or {}) != (take.get("arguments") or {}):
        because.append("arguments")
    because += [
        f"prompt:{name}"
        for name in _moved_prompts(item, take, current_prompts(item, prompts))
    ]
    # changed_at moved past every take but nothing differs from this one:
    # another take carried the difference; the plan itself is what moved
    return because or ["arguments"]


def _item_status(item, jobs, prompts=None):
    """An item's status, derived from the jobs linked to it: dropped; done
    (any succeeded), or revised when its plan changed after every succeeded
    take (`is_revised`); running; queued; failed (every linked job failed
    or was cancelled); planned (none)."""
    if item.get("dropped"):
        return "dropped"
    statuses = {job.get("status") for job in jobs}
    if "succeeded" in statuses:
        return "revised" if is_revised(item, jobs, prompts) else "done"
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


def _ready(request, grouped, prompts=None):
    """The keys of the items a job could be queued for now: planned or
    revised, with a stored workflow and every dependency done, in an open
    request."""
    if request.get("state") != "open":
        return set()
    items = request.get("items", [])
    status = {
        i.get("key"): _item_status(i, grouped.get(i.get("key"), []), prompts)
        for i in items
    }
    return {
        i.get("key")
        for i in items
        if status[i.get("key")] in ("planned", "revised")
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


def _moment(text):
    """A stored time as an aware datetime, or None when it is not one."""
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return moment if moment.utcoffset() is not None else None


def _review(item, notes, now):
    """The hold on a planned or revised item of an open request, `{until, state,
    seconds_left}`: `redirected` when a note (the item's or the request's,
    in `notes`) is at or after the moment the hold was set (any note, when
    the stamp is missing or unreadable), else `due` once
    `now` reaches the deadline, else `holding`. None when the item has no
    readable hold. `seconds_left` is whole and never negative."""
    until = _moment(item.get("review_until"))
    if until is None:
        return None
    set_at = _moment(item.get("review_set_at"))
    now = now or datetime.now(timezone.utc)
    left = max(int((until - now).total_seconds()), 0)
    if any(
        set_at is None or ((at := _moment(note.get("at"))) is not None and at >= set_at)
        for note in notes
    ):
        state = "redirected"
    else:
        state = "due" if now >= until else "holding"
    return {"until": item["review_until"], "state": state, "seconds_left": left}


def item_view(
    item, jobs, ready=False, request_notes=(), state="open", now=None, prompts=None
):
    """One item as read, with what its linked `jobs` derive: `status`,
    `ready`, `attempts` (every linked job, oldest first, as `{job, status,
    note, minutes, output}` - `note` only when set, `minutes` null until it
    has run, `output` its proof or null), `result` (`{job, run_id,
    output}`), `accepted_missing` (only when true), `spent_minutes` and
    `revised_because` (only on a revised item: `revised_because`) and
    `review` (only on a planned or revised item of an open request with a hold: see
    `_review`; `request_notes` are the request's own notes, `state` its
    state, `now` the aware time to read it at; `prompts` the `name ->
    digest` callable rule 2 reads, None for rule 1 alone)."""
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
    status = _item_status(item, jobs, prompts)
    review = (
        _review(item, [*(item.get("notes") or []), *request_notes], now)
        if status in ("planned", "revised") and state == "open"
        else None
    )
    because = (
        revised_because(item, jobs, chosen if chosen in jobs else None, prompts)
        if status == "revised"
        else None
    )
    return {
        **item,
        **{
            field: item.get(field) if isinstance(item.get(field), str) else None
            for field in ("review_until", "review_set_at")
        },
        "notes": _noted(item.get("notes"), _positions(jobs), _live_keys([item])),
        "changed_at": item.get("changed_at")
        if isinstance(item.get("changed_at"), str)
        else None,
        "accepted_basis": item.get("accepted_basis")
        if isinstance(item.get("accepted_basis"), dict)
        else None,
        "status": status,
        **({"revised_because": because} if because else {}),
        "ready": ready,
        "attempts": attempts,
        "result": result,
        **({"accepted_missing": True} if missing else {}),
        "spent_minutes": _spent(jobs),
        **({"review": review} if review else {}),
    }


def _spent(jobs):
    return round(sum(_minutes(job) or 0 for job in jobs), 2)


def _remaining(request, grouped, prompts=None):
    """What is left to do: the estimates of the items not done and not
    dropped (a revised item counts again)."""
    total = 0
    for item in request.get("items", []):
        status = _item_status(item, grouped.get(item.get("key"), []), prompts)
        if status in ("done", "dropped"):
            continue
        total += item.get("estimate_minutes") or 0
    return round(total, 2)


def request_view(request, jobs=(), now=None, prompts=None):
    """A stored request as read: every item with its derived fields
    (`item_view`) and the request's `spent_minutes` and `remaining_minutes`.
    `jobs` are the linked jobs (`{job, item, status, run_id, run_dir,
    run_exists, note, created_at, started_at, finished_at, output}`). `now`
    is an aware datetime to read the review holds at, the current time when
    omitted. `prompts` is the `name -> digest` callable rule 2 reads (None:
    rule 1 alone); wrap it in one memo per view (`dw.prompts.PromptDigests`)."""
    grouped = _jobs_by_item(jobs)
    ready = _ready(request, grouped, prompts)
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
    now = now or datetime.now(timezone.utc)
    view["items"] = [
        item_view(
            item,
            grouped.get(item.get("key"), []),
            item.get("key") in ready,
            request.get("notes") or [],
            request.get("state"),
            now,
            prompts,
        )
        for item in request.get("items", [])
    ]
    view["spent_minutes"] = _spent(jobs)
    view["remaining_minutes"] = _remaining(request, grouped, prompts)
    return view


def summary_view(request, jobs=(), prompts=None):
    """One row of the list: what a request is, without its items. `prompts`
    as `request_view`'s."""
    grouped = _jobs_by_item(jobs)
    counts = {}
    for item in request.get("items", []):
        status = _item_status(item, grouped.get(item.get("key"), []), prompts)
        counts[status] = counts.get(status, 0) + 1
    history = request.get("history") or []
    return {
        "id": request.get("id"),
        "title": request.get("title"),
        "state": request.get("state"),
        "revision": request.get("revision"),
        "items": len(request.get("items", [])),
        "status_counts": counts,
        "ready": len(_ready(request, grouped, prompts)),
        "cover": _cover(request, grouped),
        "spent_minutes": _spent(jobs),
        "remaining_minutes": _remaining(request, grouped, prompts),
        "updated_at": history[-1].get("at") if history else None,
    }
