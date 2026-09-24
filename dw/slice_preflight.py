"""A `slice_audio` step whose source duration validate can already learn,
sliced past where that source ends, warned about before the run (#402).

`slice_audio` zero-pads a slice that reaches past its source and only says so
at run time (`_warn_on_slice_past_end`, `dw/tasks/audio_utils.py`) - a
correct message, but late when the slice sits downstream of a long render
(#402's repro: `assemble-and-score` scoring a 15.5 s cut with a 4.96 s
`score` asset, `validate_workflow` answering clean). Mirrors
`dissolve_frame_errors.py` (#400): walk the expanded definition,
`resolve_path_references` an `asset:`/`output:` audio into a real path, and
`probe_media` it - the same resolution and decode the run itself would do,
just ahead of the queue.

Deliberately narrower than the run-time check, same as #400's: a
`previous_result:` audio (nothing written yet), a literal path, a remote URL,
or a source `probe_media` cannot read, all answer "unknown" rather than
guessing - silence here is correct, not a gap, since the run-time warning
still fires once the file exists. `variable:` needs no hop of its own: by the
time `validation_errors`/`adapter_warnings` hand this module the *expanded*
definition, `replace_variables` has already substituted every `variable:`
reference (or the run cannot start at all), so what is left unresolved is
only a reference that genuinely cannot resolve yet. The threshold
(`SLICE_PAD_WARN_MS`) and the requested-length arithmetic mirror
`slice_audio`'s own two argument shapes, so the two agree on the same
padding for the same arguments.
"""

import os

from .arguments import resolve_path_references
from .assets import is_asset_reference
from .for_each import MEMBER_SEPARATOR, render_path
from .media_info import probe_media
from .runs import is_output_reference
from .tasks.audio_utils import SLICE_PAD_WARN_MS

# Left to the run-time check: not yet resolved to a real file at the point
# validation walks the expanded definition.
_UNRESOLVED_PREFIXES = ("previous_result:", "variable:", "item:", "gather:")


def _resolve_audio_path(value, base_dir):
    """The local file `value` names, or None when it is not yet resolvable,
    is not a local file, or does not exist - any of which defers the check
    to the run, exactly as `slice_audio` itself would then load it."""
    if not isinstance(value, str):
        return None
    if value.startswith(_UNRESOLVED_PREFIXES):
        return None
    if value.startswith(("http://", "https://")):
        return None
    if is_asset_reference(value) or is_output_reference(value):
        try:
            value = resolve_path_references(value, base_dir)
        except Exception:
            # Existence/traversal problems belong to reference_name_errors
            # and reference resolution at run time, not to this check
            return None
        if not isinstance(value, str):
            return None
    return value if os.path.isfile(value) else None


def _source_seconds(path):
    """The duration `slice_audio` would see for this file, or None when it
    cannot be probed or carries no audio."""
    info = probe_media(path)
    if info is None or info.get("kind") != "audio":
        return None
    return info.get("duration_seconds")


def _as_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _requested_region(task_args):
    """The (start_seconds, length_seconds) `slice_audio` would compute for
    these arguments, or None when the shape given cannot be resolved to a
    length without running anything - mirrors `slice_audio`'s own branch
    order in `dw/tasks/audio_utils.py`."""
    start_seconds = _as_number(task_args.get("start_seconds"))
    duration_seconds = _as_number(task_args.get("duration_seconds"))
    start_frame = _as_number(task_args.get("start_frame"))
    num_frames = _as_number(task_args.get("num_frames"))
    fps = _as_number(task_args.get("fps"))

    if (
        task_args.get("start_seconds") is not None
        or task_args.get("duration_seconds") is not None
    ):
        if duration_seconds is None:
            # Runs to the source's own end - cannot overrun it
            return None
        return start_seconds or 0.0, duration_seconds
    if (
        task_args.get("start_frame") is not None
        or task_args.get("num_frames") is not None
    ):
        if num_frames is None or not fps:
            return None
        return (start_frame or 0.0) / fps, num_frames / fps
    return None


def slice_past_end_warnings(workflow_definition, source_indices=None, base_dir=None):
    """Every `slice_audio` step whose source's real duration is already
    knowable and whose requested slice reaches past it, as messages.

    Walks the substituted, expanded definition, the same convention
    `dissolve_frame_errors` follows: `source_indices` maps an expanded step
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
        if not isinstance(task, dict) or task.get("command") != "slice_audio":
            continue
        task_args = task.get("arguments")
        if not isinstance(task_args, dict):
            continue

        path = _resolve_audio_path(task_args.get("audio"), base_dir)
        if path is None:
            continue
        source_seconds = _source_seconds(path)
        if not source_seconds:
            continue

        region = _requested_region(task_args)
        if region is None:
            continue
        requested_start, requested_length = region

        available = max(0.0, min(source_seconds - requested_start, requested_length))
        padded_seconds = requested_length - available
        if padded_seconds * 1000.0 < SLICE_PAD_WARN_MS:
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
        path_str = render_path(("steps", source, "task", "arguments", "audio"))
        warnings.append(
            f"{path_str}: slice_audio will run {padded_seconds:.2f} s past "
            f"the end of a {source_seconds:.2f} s source ({task_args.get('audio')}), "
            f"so that much of the {requested_start + requested_length:.2f} s "
            f"requested will be digital silence{where}. If you meant to fill "
            f"a cut of this length, make a bed with the 'loop_audio' task "
            f"('target_frames' + 'fps' matches one exactly) and slice that; "
            f"if you meant the tail pad, nothing is wrong."
        )
    return warnings


__all__ = ["slice_past_end_warnings"]
