"""A `dissolve_videos` overlap too wide for one of its own inputs, refused
before the run when the frame counts are already knowable.

`dissolve_videos` (`dw/tasks/dissolve_videos.py`) raises once it has decoded
every input: a video with fewer frames than its share of `dissolve_frames`
overlaps (`seams * dissolve_frames`) fails with "video N has M frames, too
few for its S dissolve(s) of F frames". That is correct, but late - a chain
that generates each shot before joining them can spend many GPU minutes
reaching a step that was always going to fail, for an arithmetic mistake
visible from the workflow document alone (#400).

Moved here, into `validation_errors`, for exactly the cases where a video's
frame count is knowable without running anything: a literal file path inside
the directories the run may read (`dw/probe_paths.py`), or an
`asset:`/`output:` reference, with a literal `dissolve_frames`.
`resolve_path_references` is what turns either into a real path before the
run reads it; `probe_metadata` reads that file's header - the frame count
alone, no decode (B9) - the way `dw/server/app.py` decodes `probe_media` for
gallery metadata. A `previous_result:` (or any reference
`expand_for_each` left unresolved), a `variable:`/`item:`/`gather:` reference,
or a non-literal `dissolve_frames`, names no frame count yet and is left to
the existing run-time check - silence there is correct, not a gap, since the
length is not known until the step that produces it runs.

`concat_videos`'s `trim_frames` and `crossfade_audio`'s crossfade window were
each considered for the same treatment - the issue that motivated this module
asked whether they "probably have the same gap". They do not: neither raises
when an input is too short. `concat_videos` silently truncates
(`frames.extend(clip[head_trim:])`), and `crossfade_audio` silently clamps
its window to the shortest side (`crossfade_concat`) - a different, and
already silent, shape of problem with no run-time error to move earlier.
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .media_info import probe_metadata
from .probe_paths import resolve_probe_path
from .references import author_index
from .task_domains import dissolve_shortfalls


def _frame_count(path, probe):
    """The frame count `dissolve_videos` would see for this file, or None
    when it cannot be probed or carries no video stream."""
    info = probe(path)
    if info is None or info.get("kind") != "video":
        return None
    return info.get("frame_count")


def dissolve_frame_errors(
    workflow_definition, source_indices=None, base_dir=None, *, probe=probe_metadata
):
    """Every `dissolve_videos` step whose overlap already exceeds a
    statically-resolvable input's real frame count, as [{path, message}].

    Walks the substituted, expanded definition, the same convention
    `video_extension_errors` and `task_argument_errors` follow:
    `source_indices` maps an expanded step back to the one the author wrote,
    and a path inside a `for_each` member names the member.

    `probe` defaults to the metadata-only `probe_metadata` (B9) - validation
    never needs a level or a loudness figure, only a frame count - and takes
    a memoizing wrapper so a file referenced by more than one check is
    probed once per validation rather than once per check.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict) or task.get("command") != "dissolve_videos":
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict):
            continue
        videos = arguments.get("videos")
        if not isinstance(videos, list) or len(videos) < 2:
            continue
        dissolve_frames = arguments.get("dissolve_frames", 12)
        if not isinstance(dissolve_frames, (int, float)) or isinstance(
            dissolve_frames, bool
        ):
            continue
        if dissolve_frames <= 0:
            continue

        frame_counts = []
        for video in videos:
            path = resolve_probe_path(video, base_dir, "a video argument")
            frame_counts.append(None if path is None else _frame_count(path, probe))
        problems = dissolve_shortfalls(frame_counts, dissolve_frames)
        if not problems:
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
                "path": render_path(
                    ("steps", source, "task", "arguments", "dissolve_frames")
                ),
                "message": f"dissolve_videos: {'; '.join(problems)}{where}",
            }
        )
    return errors


__all__ = ["dissolve_frame_errors"]
