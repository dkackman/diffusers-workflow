"""Whether a list-driven or composed run is projected to exceed host RAM.

`dw/host_memory.py` reads what a *running* worker is holding; this module is
the pre-flight half, for the failure mode `device_memory_stats`/VRAM warnings
never covered - a `for_each` that keeps its pipeline resident across a
12-entry list, or a per-clip step whose footprint scales with the list, can
SIGKILL on host RAM with the accelerator nowhere near full (#243).

Scope is v1, by the repo owner's own sign-off on the issue: observed, not
curated (no author-declared memory figure - `host_memory_job_peak_rss_mb` is
a worker-reported field, not a schema key); warn, not refuse (host RAM headroom
is a property of *this machine*, not something a caller chose, so it never
blocks a run); no cross-machine normalization; and a cold start - no history
for this workflow at all - means no check, the same rule `observed_cost.py`
uses for GPU minutes.
"""

import json
import logging

from .for_each import FOR_EACH_KEY

logger = logging.getLogger("dw")

# The fraction of physical RAM a projected peak may reach before this module
# says anything - matches nothing else in the codebase; there is no existing
# "safe headroom" constant to share, because nothing else projects against
# the machine's total RAM rather than a curated figure.
CEILING_FRACTION = 0.9

_RELEASE_FLAGS = ("release_pipeline", "release_models")


def releases_between_iterations(definition):
    """Whether every `for_each` step in this workflow drops its pipeline (or
    its models) between entries, rather than holding it resident for the
    whole list.

    `release_pipeline`/`release_models` survive on a for_each step's *last*
    member only (`dw/for_each.py`), so their presence on the template step is
    what says the author asked for the resident-vs-released shape here - the
    same field, read for a different question. A workflow with no `for_each`
    step at all answers True: there is no list to hold anything resident
    across, so the "keeps everything resident" projection would not describe
    it.
    """
    for_each_steps = [
        step
        for step in definition.get("steps") or []
        if isinstance(step, dict) and FOR_EACH_KEY in step
    ]
    if not for_each_steps:
        return True
    return all(
        any(step.get(flag) for flag in _RELEASE_FLAGS) for step in for_each_steps
    )


def _requested_count(list_entries):
    """The largest list length the caller's own arguments drive, or None
    when this run has no list at all - such a run is never the failure mode
    this module exists for."""
    if not list_entries:
        return None
    counts = [value for value in list_entries.values() if isinstance(value, int)]
    return max(counts) if counts else None


def _row_count(row, list_entries, variables):
    """The list length a historical row ran with, read from its own stored
    arguments rather than the current request's - a row that ran a shorter
    or longer list is still comparable, once divided out, for the resident
    projection's per-entry figure.

    A row's stored `arguments` are only the caller's *overrides*
    (`job.spec["arguments"]`), never merged with the workflow's declared
    defaults - a run of the catalog default (the common case for this
    module's own regression case) is recorded as `{}`. Falling back to
    `variables[name]` the same way `observed_cost._bucket_key` does is what
    lets such a row resolve to its actual list length rather than to no
    length at all, which used to drop every default-arguments row out of the
    per-entry figure and left the resident shape silent regardless of
    history (#264).

    Only the variable names the *current* request's `list_entries` names are
    read back, on the assumption that a workflow's list-driving variables
    are stable across its history - the same assumption `observed_cost.py`
    makes bucketing by declared `cost_drivers`.
    """
    try:
        arguments = json.loads(row.get("arguments") or "{}")
    except (TypeError, ValueError):
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    counts = []
    for name in list_entries:
        value = arguments.get(name, variables.get(name))
        if isinstance(value, list):
            counts.append(len(value))
    return max(counts) if counts else None


def _already_survived(rows, list_entries, requested, projected_mb, variables):
    """Whether a run at least as large as this request already finished on
    this box at or above the projected peak.

    A projection built from this workflow's own history can land above the
    ceiling for a shape that is itself part of that history - the run being
    projected already happened and succeeded (#254). `rows` only ever holds
    successful runs (`JobHistory.finished_runs()`), so a matching row is a
    completed data point, not a guess: warning about a peak already reached
    without incident tells a caller nothing they don't already know from ten
    minutes ago.
    """
    for row in rows:
        peak = row.get("host_memory_job_peak_rss_mb")
        if not isinstance(peak, (int, float)) or peak < projected_mb:
            continue
        count = _row_count(row, list_entries, variables)
        if count is not None and count >= requested:
            return True
    return False


def host_memory_warnings(definition, list_entries, rows, ceiling_mb):
    """Warnings for a projected host-memory peak this machine cannot hold,
    or [] when there is nothing to project from or nothing to warn about.

    `rows` are this workflow's finished runs as `JobHistory.finished_runs()`
    groups them - the same history `observed_cost.py` reads, extended with
    `host_memory_job_peak_rss_mb` per row (#243, repointed from the
    process-lifetime `host_memory_peak_rss_mb` by #272 - a small job run
    right after a heavy one no longer inherits the heavy job's peak).
    `ceiling_mb` is this box's own RAM, scaled by `CEILING_FRACTION`; the
    caller reads that once per request rather than this module importing
    `host_memory` for a per-validate syscall.
    """
    requested = _requested_count(list_entries)
    if requested is None or not rows or not ceiling_mb:
        return []
    variables = definition.get("variables") or {}
    peaks = [row["host_memory_job_peak_rss_mb"] for row in rows]
    peaks = [value for value in peaks if isinstance(value, (int, float))]
    if not peaks:
        # Cold start: history exists for this workflow, but no run of it
        # ever reported a host-memory reading - nothing to project from
        return []
    if releases_between_iterations(definition):
        projected_mb = max(peaks)
        shape = "the largest single iteration observed"
    else:
        per_entry = []
        for row in rows:
            peak = row["host_memory_job_peak_rss_mb"]
            count = _row_count(row, list_entries, variables)
            if isinstance(peak, (int, float)) and count:
                per_entry.append(peak / count)
        if not per_entry:
            return []
        per_entry.sort()
        median_per_entry = per_entry[len(per_entry) // 2]
        projected_mb = median_per_entry * requested
        shape = f"{requested} entries held resident together"
    if projected_mb <= ceiling_mb:
        return []
    if _already_survived(rows, list_entries, requested, projected_mb, variables):
        return []
    return [
        "Projected host memory for this run (~"
        f"{round(projected_mb)} MB, {shape}) exceeds this machine's usable RAM "
        f"(~{round(ceiling_mb)} MB) - based on {len(rows)} run(s) of this "
        "workflow's own history on this machine, not a curated figure. The "
        "run is not blocked, but it may be killed by the OS partway through."
    ]
