"""A `join_windows` list of the wrong length, refused before the run when the
source's frame count is already knowable.

`join_windows` (`dw/tasks/windows.py`) raises once it has decoded its source:
a window list that is not exactly `ceil(source_frames / (num_frames -
overlap))` long fails with the count it needs and the entries to add or drop.
That is correct, but late - a windowed template runs a model step over every
window before the join, so a list one entry short spends every window's GPU
minutes reaching a refusal that was visible from the workflow document and
the source's header alone (#601).

Moved here, into `validation_errors`, in the shape of
`dw/dissolve_frame_errors.py` (#400), for exactly the cases where the count is
knowable without running anything: `source` is an `asset:`/`output:`
reference or a literal path the run may read (`dw/probe_paths.py`), probed
header-only (`probe_metadata`, B9); `num_frames` and `overlap` are literal
after substitution; and `videos` is a list - which is what `gather:<step>`
becomes once `for_each` has expanded, one entry per member. The rule itself
is `window_count_problem` in `dw/task_domains.py`, the one the task calls at
run time, so the two cannot drift.

The check belongs to the task, not to a template: any `join_windows` step
gets it. A `previous_result:` (or any reference `expand_for_each` left
unresolved) or `variable:` source, a non-literal `num_frames` or `overlap`, or
a `videos` that is not yet a list names no count and is left to the run-time
refusal - silence there is correct, as `dissolve_frame_errors` documents.
An `overlap` that is not below `num_frames` is `join_windows_errors`' to
report, so it is skipped here rather than reported twice.
"""

import numbers

from .for_each import MEMBER_SEPARATOR, render_path
from .media import probe_metadata
from .probe_paths import resolve_probe_path
from .references import author_index
from .task_domains import window_count_problem


def _literal_whole(value):
    """`value` as an int when it is a literal whole number, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _frame_count(path, probe):
    """The frame count `join_windows` would see for this file, or None when
    it cannot be probed or carries no video stream."""
    info = probe(path)
    if info is None or info.get("kind") != "video":
        return None
    count = info.get("frame_count")
    return count if isinstance(count, int) and count > 0 else None


def window_count_errors(
    workflow_definition, source_indices=None, base_dir=None, *, probe=probe_metadata
):
    """Every `join_windows` step whose window list is not the length its
    statically-resolvable source needs, as [{path, message}].

    Walks the substituted, expanded definition, the same convention
    `dissolve_frame_errors` follows: `source_indices` maps an expanded step
    back to the one the author wrote, and a step inside a `for_each` member
    names the member. `probe` is the metadata-only `probe_metadata` or the
    validation's memoizing wrapper of it - see `dissolve_frame_errors`.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict) or task.get("command") != "join_windows":
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict):
            continue
        videos = arguments.get("videos")
        if not isinstance(videos, list) or not videos:
            continue
        num_frames = _literal_whole(arguments.get("num_frames"))
        overlap = _literal_whole(arguments.get("overlap"))
        if num_frames is None or overlap is None:
            continue
        if num_frames <= 0 or overlap < 0 or overlap >= num_frames:
            continue
        path = resolve_probe_path(arguments.get("source"), base_dir, "a source video")
        source_frames = None if path is None else _frame_count(path, probe)
        if source_frames is None:
            continue
        problem = window_count_problem(len(videos), source_frames, num_frames, overlap)
        if problem is None:
            continue

        source = author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        errors.append(
            {
                "path": render_path(("steps", source)),
                "message": f"{problem}{where}",
            }
        )
    return errors


__all__ = ["window_count_errors"]
