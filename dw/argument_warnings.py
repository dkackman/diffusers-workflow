"""Warnings about arguments a workflow writes that do nothing.

An argument a pipeline's `__call__` does not accept, a variable no
declaration names, and the task arguments that modify a behavior their
neighbours switched off (`crossfade_ms` with nothing trimmed, and the like).
Best effort and advisory: nothing here refuses a workflow.
"""

from . import references
from .introspection import CLASS_NAME_PATTERN, unknown_call_arguments
from .variables import undeclared_variable_references


def _resolved_value(arguments, key, values):
    """`arguments[key]` as a number, resolving a `variable:name` reference
    against `values` (declared defaults merged with the caller's own
    arguments, the way `constraint_warnings` resolves a constrained
    variable). `None` when the key is absent, not a `variable:` reference or
    a literal, or the reference does not resolve to a number - callers tell
    that apart from an actual 0 by checking `key in arguments` themselves
    where it matters."""
    if key not in arguments:
        return None
    value = arguments[key]
    if references.is_ref(references.VARIABLE, value):
        value = values.get(references.ref_name(references.VARIABLE, value))
    return value if isinstance(value, (int, float)) else None


def _inert_crossfade_warnings(step, command, arguments):
    """concat_videos draws its crossfade from the trimmed-off material, so
    with nothing trimmed a `crossfade_ms` the author wrote does nothing. A
    referenced trim is unknown until the run and is left alone."""
    if command != "concat_videos":
        return []
    crossfade = arguments.get("crossfade_ms")
    trim = arguments.get("trim_frames", 0)
    if not isinstance(crossfade, (int, float)) or crossfade <= 0 or trim != 0:
        return []
    return [
        f"Step '{step.get('name')}': 'crossfade_ms' has no effect when "
        f"'trim_frames' is 0 - the crossfade is drawn from the trimmed "
        f"material. At a hard cut, 'audio_bleed_ms' or 'seam_fade_ms' is "
        f"what shapes the seam"
    ]


def _inert_seam_fade_warnings(step, command, arguments, values):
    """concat_videos takes the bleed path, not the fade path, at a hard cut
    with nothing trimmed while audio_bleed_ms is non-zero - so a seam_fade_ms
    the author wrote alongside it does nothing (#288). Both variables are
    ordinary `variable:` references in the templates that pair them, so
    `seam_fade_ms` and `audio_bleed_ms` are resolved against `values`
    (declared defaults merged with the caller's own arguments) rather than
    left alone the way an unresolved `trim_frames` is - it is exactly the
    templated case, with `audio_bleed_ms` left at its non-zero default and
    only `seam_fade_ms` passed as an argument, that this warning exists for.
    `trim_frames` stays a literal-only check, as in `_inert_crossfade_warnings`."""
    if command != "concat_videos":
        return []
    if "seam_fade_ms" not in arguments or "audio_bleed_ms" not in arguments:
        return []
    seam_fade = _resolved_value(arguments, "seam_fade_ms", values)
    bleed = _resolved_value(arguments, "audio_bleed_ms", values)
    trim = arguments.get("trim_frames", 0)
    if seam_fade is None or seam_fade <= 0 or bleed is None or bleed <= 0 or trim != 0:
        return []
    return [
        f"Step '{step.get('name')}': 'seam_fade_ms' has no effect while "
        f"'audio_bleed_ms' is {bleed} - a hard cut takes the bleed path "
        f"instead of the fade path. Pass 'audio_bleed_ms': 0 for "
        f"'seam_fade_ms' to apply."
    ]


def _inert_bleed_gain_warnings(step, command, arguments, values):
    """concat_videos applies audio_bleed_gain_db to the bled tail
    audio_bleed_ms carries across the seam - with no bleed there is nothing
    for the gain to shape, so an audio_bleed_gain_db the author wrote does
    nothing while audio_bleed_ms is 0, whether that 0 is an explicit
    argument or the task's own default left untouched (#290, the same no-op
    class #288 closed for seam_fade_ms). `audio_bleed_ms` is read with the
    task's default of 0 rather than requiring the key, since "forgot the
    bleed" is exactly the case this warning is for; resolved against
    `values` for the same reason _inert_seam_fade_warnings is - a templated
    case pairs both as `variable:` references."""
    if command != "concat_videos":
        return []
    if "audio_bleed_gain_db" not in arguments:
        return []
    gain = _resolved_value(arguments, "audio_bleed_gain_db", values)
    bleed = (
        _resolved_value(arguments, "audio_bleed_ms", values)
        if "audio_bleed_ms" in arguments
        else 0
    )
    if gain is None or gain == 0 or bleed is None or bleed != 0:
        return []
    return [
        f"Step '{step.get('name')}': 'audio_bleed_gain_db' has no effect "
        f"when 'audio_bleed_ms' is 0 - pass a non-zero 'audio_bleed_ms' for "
        f"the gain to apply."
    ]


def _inert_bleed_single_input_warnings(step, command, arguments, values):
    """concat_videos bleeds across the seam between two inputs, so one input
    has no seam for a non-zero audio_bleed_ms to act on - the request is
    dropped silently (#565). `videos` is resolved the way the numbers are:
    a literal list, or a `variable:` reference to one in `values`."""
    if command != "concat_videos" or "audio_bleed_ms" not in arguments:
        return []
    bleed = _resolved_value(arguments, "audio_bleed_ms", values)
    videos = arguments.get("videos")
    if references.is_ref(references.VARIABLE, videos):
        videos = values.get(references.ref_name(references.VARIABLE, videos))
    if bleed is None or bleed <= 0 or not isinstance(videos, list) or len(videos) != 1:
        return []
    return [
        f"Step '{step.get('name')}': 'audio_bleed_ms' ({bleed}) has no effect "
        f"with one input - the bleed acts on seams between inputs, and a "
        f"joined input's own inner seams are not reworked."
    ]


def _inert_match_levels_dbfs_warnings(step, command, arguments):
    """concat_videos and dissolve_videos only call match_levels() - the
    function that reads match_levels_dbfs as its target - when match_levels
    itself is truthy (`if match_levels:`), so a caller who passes only the
    target dBFS and leaves match_levels unset (off by default) has stated an
    intent the engine silently drops: the shots join unmatched with no trace,
    warning or otherwise (#291, the same "modifier without its enabler" class
    #288 and #290 closed for seam_fade_ms and audio_bleed_gain_db). Literal
    check only, like _inert_crossfade_warnings' trim_frames - match_levels is
    "rms"/"peak"/falsy, not a number a variable: reference would need
    resolving to compare against a domain."""
    if command not in ("concat_videos", "dissolve_videos"):
        return []
    if "match_levels_dbfs" not in arguments or arguments.get("match_levels"):
        return []
    dbfs = arguments.get("match_levels_dbfs")
    if not isinstance(dbfs, (int, float)):
        return []
    return [
        f"Step '{step.get('name')}': 'match_levels_dbfs' has no effect when "
        f'\'match_levels\' is unset - pass "rms" or "peak" for the target '
        f"to apply."
    ]


def workflow_argument_warnings(workflow_definition, arguments=None):
    """Best-effort pre-load check of a workflow's arguments.

    For each pipeline step whose component_type is a bare diffusers class
    name, reports argument names that class's __call__ does not accept - the
    typo that today surfaces as a TypeError after the model has loaded.
    Escaped ({...}) and dotted component types are left alone. Task steps
    get the same check against their registered implementation's signature.

    `arguments`, when given, is a caller's own values for this run -
    checks that need a task argument's actual value (an inert `crossfade_ms`
    or `seam_fade_ms`) resolve a `variable:name` reference against the
    caller's arguments merged over the workflow's declared defaults, the
    same values `constraint_warnings` checks a constraint against.
    """
    warnings = []
    values = {**(workflow_definition.get("variables") or {}), **(arguments or {})}
    declared = sorted(workflow_definition.get("variables") or {})
    for path, name in undeclared_variable_references(workflow_definition):
        hint = (
            " - a reference is the whole value, nothing is interpolated around it"
            if any(c in name for c in " ,")
            else ""
        )
        warnings.append(
            f"{path}: '{references.make_ref(references.VARIABLE, name)}' names no declared variable{hint}; "
            f"declared: {', '.join(declared) or '<none>'}"
        )
    for step in workflow_definition.get("steps", []):
        task = step.get("task")
        if task and isinstance(task.get("arguments"), dict):
            command = task.get("command")
            # An unknown or missing task argument is an error rather than a
            # warning now (task_signature_errors, #141) - reported once, by
            # the pass whose verdict it changes
            warnings.extend(_inert_crossfade_warnings(step, command, task["arguments"]))
            warnings.extend(
                _inert_seam_fade_warnings(step, command, task["arguments"], values)
            )
            warnings.extend(
                _inert_bleed_gain_warnings(step, command, task["arguments"], values)
            )
            warnings.extend(
                _inert_match_levels_dbfs_warnings(step, command, task["arguments"])
            )
            warnings.extend(
                _inert_bleed_single_input_warnings(
                    step, command, task["arguments"], values
                )
            )
        pipeline = step.get("pipeline")
        if not pipeline:
            continue
        component_type = pipeline.get("configuration", {}).get("component_type")
        if not isinstance(component_type, str) or not CLASS_NAME_PATTERN.match(
            component_type
        ):
            continue
        argument_names = list(pipeline.get("arguments", {}))
        unknown = unknown_call_arguments(component_type, argument_names)
        for argument_name in unknown:
            warnings.append(
                f"Step '{step.get('name')}': {component_type} does not accept "
                f"argument '{argument_name}'"
            )
    return warnings
