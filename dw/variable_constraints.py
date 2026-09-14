"""What a variable's value is allowed to be, declared by the author.

`validate_workflow` passed `num_frames: 61` on an H3 template and answered
`valid: true`, naming `num_frames` in `checked_arguments` - so the answer
claimed to cover the caller's value. The run then spent 138.7 s loading the
weights and the turbo LoRA, entered the text encoder, and failed on a check
against two integers:

    MiniMax-H3 generates between 5.0 and 15.0 seconds at 24 fps, so
    `num_frames`, rounded up to the next `17 * n + 5` the video VAE can
    encode, must be between 120 and 360, got 61 (rounded up to 73).

Every term in that message is a property of the model. None of it needed a
loaded pipeline - and nothing reachable over the API stated it, so a consumer
reading `num_frames: 124` with no range had no way to know the rule was
`17n + 5` from 124 (#96).

The rule is declared in the workflow, where `frame_snap` already puts the
same numbers for a chain step, and the engine holds no model knowledge of its
own (CLAUDE.md). One shape, not two: a `variable_constraints` entry takes
`frame_snap`'s field names, and a chain step's `frame_snap` may be the string
`"constraint:<variable>"` so a template states `17n + 5` once rather than
twice in one file.

Checked in three places for the reasons the task-argument domains are
(dw/task_domains.py): statically in `validation_errors`, so a stored default
or a caller's argument outside the rule is a free refusal at the JSON path it
sits at; at run time before anything loads, where a value that arrived from
somewhere the static pass cannot see is snapped or refused; and reported
beside the variable's default by the catalog, which is the half that stops
the next consumer picking 61.
"""

import logging
import math
import numbers

logger = logging.getLogger("dw")

CONSTRAINTS_KEY = "variable_constraints"
# What a chain step's `frame_snap` writes instead of repeating the numbers
CONSTRAINT_PREFIX = "constraint:"
# The fields a `frame_snap` block carries, which are the fields a constraint
# is checked on - the rest of a constraint says what to do about a violation
SNAP_FIELDS = ("modulus", "remainder", "min_frames", "max_frames")


def declared_constraints(definition):
    """The workflow's constraint block, or {}."""
    constraints = definition.get(CONSTRAINTS_KEY)
    return constraints if isinstance(constraints, dict) else {}


def snap_block(constraint):
    """The `frame_snap`-shaped part of a constraint - what a chain step needs
    when it names the constraint rather than repeating it."""
    return {
        key: constraint[key]
        for key in SNAP_FIELDS
        if isinstance(constraint, dict) and key in constraint
    }


def _as_integer(value):
    """The value as an int, or None when this is not a number to check.

    A string that is still a reference, a null, a list: not a violation, just
    not something this pass can answer about.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def aligned(value, constraint):
    """The smallest value on the constraint's grid that is not below `value`,
    ignoring the range, or None when there is no grid to align to.

    The range is checked against *this* number rather than against what the
    caller wrote, because that is what the pipeline does: diffusers'
    `align_num_frames` snaps first and the duration bound then holds for the
    aligned count, which is why 346 frames is refused (it becomes 362) and
    108 is accepted (it becomes 124).
    """
    number = _as_integer(value)
    modulus = constraint.get("modulus")
    if number is None or not modulus:
        return None
    remainder = constraint.get("remainder", 0) % modulus
    steps = math.ceil((number - remainder) / modulus)
    target = steps * modulus + remainder
    while target < number:
        target += modulus
    return target


def effective(value, constraint):
    """The number the run will actually use - `value` itself, or what
    `snap: "up"` rounds it to. None when this is not a number to check."""
    number = _as_integer(value)
    if number is None:
        return None
    if constraint.get("snap") != "up":
        return number
    target = aligned(number, constraint)
    return number if target is None else target


def snapped(value, constraint):
    """What `value` becomes, or None when it is left alone.

    Only `snap: "up"` snaps, and only off the grid: rounding is a change to
    what the caller asked for, so it happens where the author said it should
    and nowhere else. 61 frames on an H3 template is not a request for 124 -
    it rounds to 73, which the range refuses; 130 is a request for 141 (#96).
    """
    number = _as_integer(value)
    target = effective(value, constraint)
    if target is None or target == number:
        return None
    return target


def violations(value, constraint):
    """Why `value` breaks `constraint`, as phrases, or [] when it does not.

    Judged on the value the run would use, so a constraint that rounds is
    only ever refused for a bound the rounded number still breaks.
    """
    number = effective(value, constraint)
    if number is None:
        return []
    problems = []
    modulus = constraint.get("modulus")
    remainder = constraint.get("remainder", 0)
    if modulus and (number - remainder) % modulus != 0:
        problems.append(f"must be {modulus} * n + {remainder}")
    minimum = constraint.get("min_frames")
    if minimum is not None and number < minimum:
        problems.append(f"must be at least {minimum}")
    maximum = constraint.get("max_frames")
    if maximum is not None and number > maximum:
        problems.append(f"must be at most {maximum}")
    return problems


def _accepted(constraint):
    """The rule as a phrase a consumer can act on, which is the half of #96
    that stops the next caller picking 61: the range and the step, together."""
    parts = []
    low, high = constraint.get("min_frames"), constraint.get("max_frames")
    if low is not None and high is not None:
        parts.append(f"{low} to {high}")
    elif low is not None:
        parts.append(f"{low} or more")
    elif high is not None:
        parts.append(f"{high} or less")
    modulus = constraint.get("modulus")
    if modulus:
        parts.append(f"{modulus} * n + {constraint.get('remainder', 0)}")
    return ", ".join(parts)


def refusal(name, value, constraint):
    """The one wording every layer uses for a value outside its constraint."""
    problems = violations(value, constraint)
    if not problems:
        return None
    reason = constraint.get("reason")
    because = f" - {reason}" if reason else ""
    target = snapped(value, constraint)
    rounding = f" (rounds up to {target})" if target is not None else ""
    return (
        f"'{name}' is {value}{rounding}, which this workflow does not "
        f"accept: {'; '.join(problems)}. Accepted: {_accepted(constraint)}{because}. "
        f"The rule is declared on the workflow, so it is checked before "
        f"anything loads rather than after the weights are in memory"
    )


def snap_notice(name, value, constraint):
    """The wording for a value that will be rounded, or None.

    A value the range still refuses after rounding is not a notice - it is a
    refusal, and saying both would be two answers to one mistake.
    """
    target = snapped(value, constraint)
    if target is None or violations(value, constraint):
        return None
    reason = constraint.get("reason")
    because = f" - {reason}" if reason else ""
    return (
        f"'{name}' is {value}, which this workflow rounds up to {target}"
        f"{because}. The run generates {target}, not {value}"
    )


def constraint_errors(definition, arguments=None, supplied=()):
    """Every declared constraint a value breaks, as [{path, message}].

    A value the caller supplied is reported at `arguments.<name>`, where they
    wrote it; a stored default at `variables.<name>`. A constraint that
    declares `snap: "up"` and can reach a legal value is a warning rather
    than an error - the run will round it, and saying so is the point.
    """
    constraints = declared_constraints(definition)
    if not constraints:
        return []
    variables = definition.get("variables") or {}
    values = {**variables, **(arguments or {})}
    errors = []
    for name in sorted(constraints):
        constraint = constraints[name]
        if not isinstance(constraint, dict) or name not in values:
            continue
        message = refusal(name, values[name], constraint)
        if message is None:
            # Either legal, or legal once rounded - a value the workflow
            # rounds is a warning (constraint_warnings), never an error
            continue
        where = "arguments" if name in (supplied or ()) else "variables"
        errors.append({"path": f"{where}.{name}", "message": message})
    return errors


def constraint_warnings(definition, arguments=None):
    """Every value a declared constraint will round, as messages.

    The silent half of #96: the engine already rounded `num_frames` up to the
    VAE's grid with a `logger.warning`, which by the #82 rule does not exist
    out there - a caller got a frame count they did not ask for and nothing
    said so.
    """
    constraints = declared_constraints(definition)
    if not constraints:
        return []
    values = {**(definition.get("variables") or {}), **(arguments or {})}
    notices = []
    for name in sorted(constraints):
        constraint = constraints[name]
        if not isinstance(constraint, dict) or name not in values:
            continue
        notice = snap_notice(name, values[name], constraint)
        if notice is not None:
            notices.append(notice)
    return notices


def apply_constraints(definition, variables):
    """Refuse or round the run's variable values, before anything loads.

    The run-time half, and the backstop for everything the static pass cannot
    see - an inline workflow, a value a parent workflow passed down. Rounds in
    place and emits a warning for each rounded value, so a frame count the run
    changed reaches the job's `warnings` rather than only the log; raises
    ValueError for a value no rule can reach.
    """
    from .events import emit_warning

    constraints = declared_constraints(definition)
    if not constraints or not isinstance(variables, dict):
        return
    for name in sorted(constraints):
        constraint = constraints[name]
        if not isinstance(constraint, dict) or name not in variables:
            continue
        value = variables[name]
        notice = snap_notice(name, value, constraint)
        if notice is not None:
            variables[name] = snapped(value, constraint)
            emit_warning(
                notice, kind="value_snapped", variable=name, value=variables[name]
            )
            continue
        message = refusal(name, value, constraint)
        if message is not None:
            raise ValueError(message)


def resolve_constraint_references(definition):
    """Replace every `"frame_snap": "constraint:<name>"` with the declared
    constraint's numbers, in place.

    So a template states `17n + 5` once - on the variable, where the catalog
    reports it and validation checks it - rather than twice, with a chain
    step's copy free to drift from it. A name that is not declared is an
    error rather than a silently absent constraint: a chain that snapped to
    nothing would stitch segments the pipeline refuses.
    """
    constraints = declared_constraints(definition)

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        reference = node.get("frame_snap")
        if isinstance(reference, str) and reference.startswith(CONSTRAINT_PREFIX):
            name = reference[len(CONSTRAINT_PREFIX) :]
            if name not in constraints:
                raise ValueError(
                    f"'frame_snap': '{reference}' names no entry of this "
                    f"workflow's 'variable_constraints'. Declared: "
                    + (", ".join(sorted(constraints)) or "<none>")
                )
            node["frame_snap"] = snap_block(constraints[name])
        for value in node.values():
            walk(value)

    walk(definition)
    return definition


def constraint_reference_errors(definition):
    """`frame_snap: "constraint:x"` naming nothing declared, as
    [{path, message}] - the validation-time form of the error
    resolve_constraint_references raises."""
    constraints = declared_constraints(definition)
    errors = []

    def walk(node, path):
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            where = f"{path}.{key}" if path else key
            if (
                key == "frame_snap"
                and isinstance(value, str)
                and value.startswith(CONSTRAINT_PREFIX)
                and value[len(CONSTRAINT_PREFIX) :] not in constraints
            ):
                errors.append(
                    {
                        "path": where,
                        "message": (
                            f"'{value}' names no entry of this workflow's "
                            f"'variable_constraints'. Declared: "
                            + (", ".join(sorted(constraints)) or "<none>")
                        ),
                    }
                )
            walk(value, where)

    walk(definition, "")
    return errors
