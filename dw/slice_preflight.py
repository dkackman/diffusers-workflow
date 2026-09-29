"""A `slice_audio` step whose source duration validate can already learn,
sliced past where that source ends, warned about before the run (#402).

`slice_audio` zero-pads a slice that reaches past its source and only says so
at run time (`_warn_on_slice_past_end`, `dw/tasks/audio_utils.py`) - a
correct message, but late when the slice sits downstream of a long render
(#402's repro: `assemble-and-score` scoring a 15.5 s cut with a 4.96 s
`score` asset, `validate_workflow` answering clean). Mirrors
`dissolve_frame_errors.py` (#400): walk the expanded definition,
`resolve_path_references` an `asset:`/`output:` audio into a real path, and
`probe_metadata` it - the same resolution the run itself would do, and its
duration without the decode (B9).

Deliberately narrower than the run-time check, same as #400's: a
`previous_result:` audio (nothing written yet), a remote URL, a literal path
outside the directories the run may read (`dw/probe_paths.py` - a literal
inside them is resolved against the workflow's directory and probed), or a
source `probe_metadata` cannot read, all answer "unknown" rather than
guessing - silence here is correct, not a gap, since the run-time warning
still fires once the file exists. `variable:` needs no hop of its own: by the
time `validation_errors`/`adapter_warnings` hand this module the *expanded*
definition, `replace_variables` has already substituted every `variable:`
reference (or the run cannot start at all), so what is left unresolved is
only a reference that genuinely cannot resolve yet. The padding rule
(`slice_padding`, with its `SLICE_PAD_WARN_MS` threshold) is the one
`slice_audio` itself calls, in dw/task_domains.py, and the region is worked
out in samples with the run's own arithmetic (`frames_to_samples`, #557's
end rounding), so the two agree on the same padding for the same arguments.
"""

from fractions import Fraction

from .for_each import MEMBER_SEPARATOR, render_path
from .media_info import probe_metadata
from .probe_paths import resolve_probe_path
from .references import author_index
from .task_domains import frames_to_samples, slice_padding


def _source(path, probe):
    """The (total samples, sample rate) `slice_audio` would see for this
    file, or None when it cannot be probed, carries no audio, or its header
    gives no duration or no rate. The count is the header's duration at the
    header's rate - exact for a wav, the header's figure for a compressed
    format."""
    info = probe(path)
    if info is None or info.get("kind") != "audio":
        return None
    seconds, rate = info.get("duration_seconds"), info.get("sample_rate")
    if not seconds or not rate:
        # The padding is worked out in samples, as the run works it out -
        # with no rate there is no count to work it out from
        return None
    return int(round(seconds * rate)), rate


def _as_number(value, kind):
    """A literal as `slice_audio` coerces it: a string through `kind` (so an
    fps string is an exact Fraction, as the run makes it), a number as it
    stands, anything else - a bool, a reference - None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            return kind(value)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return value if isinstance(value, (int, float)) else None


def _requested_region(task_args, sample_rate):
    """The (start, length) in samples `slice_audio` would compute for these
    arguments, or None when the shape given cannot be resolved to a length
    without running anything - `slice_audio`'s own branch order and
    arithmetic in `dw/tasks/audio_utils.py`, including #557's rounding of a
    frame-addressed end directly rather than as two rounded halves."""
    if (
        task_args.get("start_seconds") is not None
        or task_args.get("duration_seconds") is not None
    ):
        duration_seconds = _as_number(task_args.get("duration_seconds"), float)
        if duration_seconds is None:
            # Runs to the source's own end - cannot overrun it
            return None
        start_seconds = _as_number(task_args.get("start_seconds"), float)
        start = int(round((start_seconds or 0) * sample_rate))
        return start, int(round(duration_seconds * sample_rate))
    if (
        task_args.get("start_frame") is not None
        or task_args.get("num_frames") is not None
    ):
        num_frames = _as_number(task_args.get("num_frames"), int)
        fps = _as_number(task_args.get("fps"), Fraction)
        if num_frames is None or not fps:
            return None
        start_frame = _as_number(task_args.get("start_frame"), int) or 0
        start = frames_to_samples(start_frame, fps, sample_rate)
        end = frames_to_samples(start_frame + num_frames, fps, sample_rate)
        return start, end - start
    return None


def slice_past_end_warnings(
    workflow_definition, source_indices=None, base_dir=None, *, probe=probe_metadata
):
    """Every `slice_audio` step whose source's real duration is already
    knowable and whose requested slice reaches past it, as messages.

    Walks the substituted, expanded definition, the same convention
    `dissolve_frame_errors` follows: `source_indices` maps an expanded step
    back to the one the author wrote, and a path inside a `for_each` member
    names the member.

    `probe` defaults to the metadata-only `probe_metadata` (B9); see
    `dissolve_frame_errors` for why and for the memoizing-wrapper contract.
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

        path = resolve_probe_path(task_args.get("audio"), base_dir, "an audio argument")
        if path is None:
            continue
        probed = _source(path, probe)
        if probed is None:
            continue
        total, sample_rate = probed
        # A literal sample_rate relabels the file's samples at that rate, as
        # it does in the run (#180) - same samples, different seconds. Only a
        # number: the run hands a string rate on uncoerced and cannot slice
        # at it, so a string is no relabel validation can reason about
        override = task_args.get("sample_rate")
        if isinstance(override, str):
            override = None
        override = _as_number(override, float)
        if override and override > 0:
            sample_rate = override

        region = _requested_region(task_args, sample_rate)
        if region is None:
            continue
        start, length = region
        padded_seconds = slice_padding(total, start, length, sample_rate)
        if padded_seconds is None:
            continue
        source_seconds = total / float(sample_rate)
        requested_end = (start + length) / float(sample_rate)

        source = author_index(source_indices, index)
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
            f"so that much of the {requested_end:.2f} s "
            f"requested will be digital silence{where}. If you meant to fill "
            f"a cut of this length, make a bed with the 'loop_audio' task "
            f"('target_frames' + 'fps' matches one exactly) and slice that; "
            f"if you meant the tail pad, nothing is wrong."
        )
    return warnings


__all__ = ["slice_past_end_warnings"]
