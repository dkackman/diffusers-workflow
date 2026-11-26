"""The `prompt` argument jobs were submitted with, for the opt-in `prompts=true`
of the jobs listing (by job id) and the gallery (by run id).

A job's stored arguments never change: they are the ones it was submitted
with, and closing its row writes the same ones again. That is what makes the
memo safe - a historical job's prompt is read once and never invalidated. A
live job is read from its spec, which the manager holds.

The `workspace` filter relies on `job_history.py` having backfilled every row
recorded before workspaces existed to the default one: history that cannot say
which workspace a job ran in stops making sense once there are two.
"""

import json
import threading
from collections import OrderedDict

from .outputs import prompt_text

# (history path, job id) -> its submitted prompt or None
_PROMPT_MEMO, _PROMPT_MEMO_LOCK = OrderedDict(), threading.Lock()
PROMPT_MEMO_MAX = 4096


def submitted_prompts(history, ids, by_run=False, workspace=None):
    """(id, run id, json_type, json_extract) of `$.prompt` for `workspace`'s
    rows (any for None) whose id (run id `by_run`) is in `ids`, oldest first
    so the newest of a shared run id comes last - not the blob."""
    p, rows, col = "arguments, '$.prompt'", [], ("id", "run_id")[by_run]
    sql = f"SELECT id, run_id, json_type({p}), json_extract({p}) FROM jobs"
    for chunk in (ids[at : at + 500] for at in range(0, len(ids), 500)):
        query = f"{sql} WHERE {col} IN ({','.join('?' * len(chunk))})"
        query += " AND IFNULL(?, workspace) IS workspace"
        query += " AND json_valid(arguments) ORDER BY created_at"
        rows += history.select(query, [*chunk, workspace])
    return rows


def _stored_prompt(json_type, value):
    """`prompt_text` of a `$.prompt` as `submitted_prompts` read it out of
    SQL: text as it is, a list from its JSON, anything else None."""
    if json_type == "array":
        value = json.loads(value)
    elif json_type != "text":
        return None
    return prompt_text({"prompt": value})


def job_prompts(manager, job_ids):
    """Job id -> submitted prompt for `job_ids`: a live job's from its spec,
    a historical one's from the history, each read once and memoised
    (bounded at PROMPT_MEMO_MAX); a row that is not valid JSON memoises None."""
    history, found = manager.history, {}
    live = {job_id: manager.jobs.get(job_id) for job_id in job_ids}
    with _PROMPT_MEMO_LOCK:
        unread = [
            job_id
            for job_id, job in live.items()
            if job is None and (history.db_path, job_id) not in _PROMPT_MEMO
        ]
        rows = submitted_prompts(history, unread) if unread else []
        read = dict.fromkeys(unread)
        read.update(
            (job_id, _stored_prompt(kind, value)) for job_id, _r, kind, value in rows
        )
        _PROMPT_MEMO.update(((history.db_path, j), p) for j, p in read.items())
        while len(_PROMPT_MEMO) > PROMPT_MEMO_MAX:
            _PROMPT_MEMO.popitem(last=False)
        memo = {j: _PROMPT_MEMO.get((history.db_path, j)) for j in live}
    for job_id, job in live.items():
        prompt = prompt_text(job.spec.get("arguments")) if job else memo[job_id]
        if prompt:
            found[job_id] = prompt
    return found


def run_prompts(history, workspace, run_ids):
    """Run id -> the prompt its job was submitted with, for `run_ids`; a run
    whose job history no longer holds has no entry. Jobs sharing a run id
    (a rerun) give the newest one's prompt."""
    rows = submitted_prompts(history, list(run_ids), by_run=True, workspace=workspace)
    newest = {run: _stored_prompt(kind, value) for _job, run, kind, value in rows}
    return {run: prompt for run, prompt in newest.items() if prompt}
