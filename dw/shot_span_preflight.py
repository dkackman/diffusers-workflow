"""A `shots` argument to an assessment probe (`analyze_shots`,
`analyze_seams`, `analyze_sync_drift`) whose frame span already runs past a
statically-knowable video's real length, warned about before the run (#425).

The probes silently clip an overrunning shot record to the file (`_clip` in
`dw/tasks/assess.py`) and only say so at run time
(`_shot_span_findings`, same module) - a message correct but late once the
run has already spent the decode. Mirrors `slice_preflight.py` (#402): walk
the expanded definition, `resolve_path_references` an `asset:`/`output:`
video into a real path, and `probe_media` it - the same resolution and
decode the run itself would do, just ahead of the queue.

Deliberately narrower than the run-time check, same as #402's: a
`previous_result:` video (nothing written yet), a remote URL, a literal path
outside the directories the run may read, or a source `probe_media` cannot
read, all answer "unknown" rather than guessing - silence here is correct,
not a gap, since the run-time warning still fires once the file exists. Only
the `shots` argument's `start_frame`/`num_frames` are checked; a `shots`
argument sourced from a `variable:`/`previous_result:`/`gather:` reference
names no records yet and is left to the run-time check.
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .media_info import probe_media
from .probe_paths import resolve_probe_path

PROBE_COMMANDS = ("analyze_shots", "analyze_seams", "analyze_sync_drift")


def _frame_count(path):
    """The frame count a probe would see for this file, or None when it
    cannot be probed or carries no video stream."""
    info = probe_media(path)
    if info is None or info.get("kind") != "video":
        return None
    return info.get("frame_count")


def _as_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def shot_span_warnings(workflow_definition, source_indices=None, base_dir=None):
    """Every assessment-probe step whose `shots` argument already reaches
    past a statically-resolvable video's real frame count, as messages.

    Walks the substituted, expanded definition, the same convention
    `slice_past_end_warnings` follows: `source_indices` maps an expanded step
    back to the one the author wrote, and a path inside a `for_each` member
    names the member.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    warnings = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict) or task.get("command") not in PROBE_COMMANDS:
            continue
        task_args = task.get("arguments")
        if not isinstance(task_args, dict):
            continue
        shots = task_args.get("shots")
        if not isinstance(shots, list) or not shots:
            continue

        path = resolve_probe_path(task_args.get("video"), base_dir, "a video argument")
        if path is None:
            continue
        frame_count = _frame_count(path)
        if frame_count is None:
            continue

        problems = []
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            start_frame = _as_number(shot.get("start_frame", 0))
            num_frames = _as_number(shot.get("num_frames"))
            if start_frame is None or num_frames is None:
                continue
            end = start_frame + num_frames
            if end > frame_count:
                problems.append(
                    f"shot {shot.get('name')!r} reaches frame {int(end)}, "
                    f"{int(end - frame_count)} past the file's {frame_count} frames"
                )
        if not problems:
            continue

        source = (
            source_indices[index]
            if source_indices is not None and index < len(source_indices)
            else index
        )
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        path_str = render_path(("steps", source, "task", "arguments", "shots"))
        warnings.append(
            f"{path_str}: {task['command']} will clip {'; '.join(problems)}{where} "
            "- the probe measures a shorter window than the record asks for, "
            "silently, unless the record is corrected"
        )
    return warnings


__all__ = ["shot_span_warnings"]
