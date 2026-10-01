"""Three step-value checkers: `fps_errors` (a result's `fps`),
`null_media_errors` (a bare object description whose media resolved null) and
`select_errors` (a `select` step's arguments).

Each is a pure function of an expanded definition (and the source step
indices that map an expanded step back to the one the author wrote) returning
`[{path, message}]`. They were folded into dw/validation.py in 2b to pay for
creating it; their bodies sit here so the registry and the runner stay whole
in validation.py. Their only caller is the error registry there.
"""

from . import references
from .arguments import names_no_media
from .for_each import MEMBER_SEPARATOR, render_path
from .references import (
    FROM_ARGUMENTS_KEY,
    FROM_FILE_KEY,
    FROM_PREVIOUS_RESULT_KEY,
    author_index,
)
from .task_domains import (
    SELECT_RULES,
    SELECT_THRESHOLD_RULES,
    select_rule_problems,
)

# --- A step's result 'fps' (was dw/result_fps.py) ---------------------------
#
# Widened the same way `subfolder` was (dw/subfolders.py): the schema lets the
# raw document hold a `variable:` or `item:` reference so a workflow can keep
# one frame-rate variable rather than a literal duplicated between the step
# that generates and the step that writes (#363). This owns the resolved-value
# check: once substitution has run, `result.fps` has to be a real, positive
# frame rate, and it must be a whole number - `encode_video` (dw/result.py,
# the writer used whenever a step's result carries audio) hands the value
# straight to PyAV as `rate=int(fps)`, so a fractional rate like 23.976 would
# be silently floored to 23 rather than written as asked. A frame_rate
# variable declared as a float (24.0) still passes, since it carries no
# fractional part; only a genuine fraction is refused.
#
# Containment doesn't apply here - there's no path to join - so unlike
# subfolder_errors this only ever checks value shape.

FPS_KEY = "fps"

# Reference prefixes substitution resolves before this pass runs. One still
# spelled out here is one nothing resolved, and that is the undeclared-
# variable pass's complaint rather than a shape error
_UNRESOLVED_PREFIXES = references.SUBSTITUTED


def fps_errors(workflow_definition, source_indices=None):
    """Every result 'fps' that cannot be written, as [{path, message}].

    The definition handed here has already been substituted and expanded,
    so every value in it is literal; a 'variable:' or 'item:' still spelled
    out is left alone. `source_indices`, when given, is the source step
    index of each step - a 'for_each' group turns one written step into
    several, and the path an error carries has to be one the author can
    find in the file they wrote; the member is named in the message.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        result = step.get("result")
        if not isinstance(result, dict) or FPS_KEY not in result:
            continue
        value = result[FPS_KEY]
        if isinstance(value, str) and value.startswith(_UNRESOLVED_PREFIXES):
            continue

        source = references.author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        message = None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            message = f"Invalid fps: {value!r} - fps is a number of frames per second"
        elif value <= 0:
            message = f"Invalid fps: {value!r} - fps must be greater than zero"
        elif float(value) != int(value):
            message = (
                f"Invalid fps: {value!r} - fps must be a whole number; the video "
                f"writer truncates a fractional rate rather than honoring it"
            )
        if message is not None:
            errors.append(
                {
                    "path": render_path(("steps", source, "result", FPS_KEY)),
                    "message": f"{message}{where}",
                }
            )
    return errors


# --- An object description whose media resolved null (was dw/null_media.py) --
#
# `realize_args` builds a pipeline argument that names a type and where its
# media comes from - MiniMax-H3's references are the catalog's example. When
# that media source resolves to null (typically a `variable:` left unset), the
# object is `OMITTED`: dropped silently when it sits in a list, since that is
# how a template makes a reference optional (`dw/arguments.py`'s list branch).
# But the *same* dict sitting directly under a key, not in a list, has nothing
# to leave it out of - `realize_args` raises there, at run time, after the
# checkpoint is already loaded (#478).
#
# This mirrors that raise, not the silent drop: only a bare object description
# in "dict context" is an error here, exactly the shapes `realize_args`' own
# dict branch would refuse. One found inside a list is left alone, since a
# template author relies on that being silently dropped rather than reported.

_FROM_KEYS = (FROM_FILE_KEY, FROM_PREVIOUS_RESULT_KEY, FROM_ARGUMENTS_KEY)


def _walk(value, path, in_list, errors):
    if isinstance(value, dict):
        if not in_list and names_no_media(value):
            from_key = next(k for k in _FROM_KEYS if value.get(k, False) is None)
            errors.append(
                {
                    "path": render_path(path + (from_key,)),
                    "message": (
                        f"'{path[-1]}' names an object to build but the media "
                        f"it would be built from is null. An optional one "
                        f"belongs in a list, where it can be left out; on its "
                        f"own there is nothing to leave it out of"
                    ),
                }
            )
            return
        for key, item in value.items():
            _walk(item, path + (key,), False, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk(item, path + (index,), True, errors)


def null_media_errors(workflow_definition, source_indices=None):
    """Every bare object description whose media is null, as [{path, message}].

    The definition handed here has already been substituted and expanded, so a
    `for_each` member's own arguments are checked as they will run;
    `source_indices` maps each expanded step back to the step the author
    wrote, and the member is named in the message - the same convention
    reference_limit_errors uses.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        arguments = pipeline.get("arguments") if isinstance(pipeline, dict) else None
        if not isinstance(arguments, dict):
            continue
        source = author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        step_errors = []
        for key, value in arguments.items():
            _walk(
                value,
                ("steps", source, "pipeline", "arguments", key),
                False,
                step_errors,
            )
        for error in step_errors:
            errors.append(
                {
                    "path": error["path"],
                    "message": f"{error['message']}{where}.",
                }
            )
    return errors


# --- select's arguments (was dw/select_validation.py, docs/proposals/score-and-select.md) --
#
# select's own signature carries no domain - `rule` is any string, and
# `threshold`/`index` are only meaningful for some rules - so a step whose
# rule is misspelled, or whose threshold is missing, or whose index rule has
# none, validates clean today and dies on select's own run-time ValueError
# after the fan-out ahead of it has already generated. Checked here in
# `validation_errors` (free), and unchanged as select's own run-time check
# for a value arriving from a `variable:` or an earlier step, which this
# static pass can never see.


def _select_problem_key(rule):
    """The argument a select_rule_problems sentence is about: the rule when
    it is unknown, else the one argument that rule needs."""
    if rule not in SELECT_RULES:
        return "rule"
    return "threshold" if rule in SELECT_THRESHOLD_RULES else "index"


def _select_where(name):
    return (
        f" in member '{name}'"
        if isinstance(name, str) and MEMBER_SEPARATOR in name
        else ""
    )


def _is_gather(value):
    return references.is_ref(references.GATHER, value)


def _is_expanded_gather(value):
    """A gather: reference after for_each expansion has replaced it with the
    list of previous_result: references it drew from - the only form
    select_errors ever actually sees once validation_errors expands the
    definition before calling it."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(references.is_ref(references.PREVIOUS_RESULT, v) for v in value)
    )


def select_errors(workflow_definition, source_indices=None):
    """Every select step whose arguments cannot be right, as [{path, message}]."""
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict) or task.get("command") != "select":
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict):
            continue

        source = references.author_index(source_indices, index)
        name = step.get("name")
        where = _select_where(name)

        def add(key, message):
            path = render_path(("steps", source, "task", "arguments", key))
            errors.append({"path": path, "message": f"{message}{where}."})

        rule = arguments.get("rule")
        if isinstance(rule, str):
            # What the run itself refuses, in its own sentence - at most one,
            # keyed to the argument it is about
            problems = {
                _select_problem_key(rule): problem
                for problem in select_rule_problems(
                    rule, arguments.get("threshold"), arguments.get("index")
                )
            }
            if "rule" in problems:
                add("rule", problems["rule"])
            else:
                # Beside it, what the run ignores and validation refuses
                # anyway: an argument this rule does not read
                if "threshold" in problems:
                    add("threshold", problems["threshold"])
                elif rule not in SELECT_THRESHOLD_RULES and "threshold" in arguments:
                    add(
                        "threshold",
                        f"select: 'threshold' is only meaningful for rule "
                        f"'first_above'/'first_below', not {rule!r}",
                    )
                if "index" in problems:
                    add("index", problems["index"])
                elif rule != "index" and "index" in arguments:
                    add(
                        "index",
                        f"select: 'index' is only meaningful for rule 'index', "
                        f"not {rule!r}",
                    )

        candidates = arguments.get("candidates")
        scores = arguments.get("scores")
        if "candidates" in arguments and "scores" in arguments:
            if _is_gather(candidates) != _is_gather(scores):
                add(
                    "candidates",
                    "select: 'candidates' and 'scores' must both be "
                    "'gather:' references or both plain lists, got "
                    f"candidates={candidates!r} scores={scores!r}",
                )
            elif _is_expanded_gather(candidates) and _is_expanded_gather(scores):
                if len(candidates) != len(scores):
                    add(
                        "scores",
                        "select: 'candidates' and 'scores' gather from "
                        f"for_each groups of different sizes: candidates has "
                        f"{len(candidates)} entries, scores has {len(scores)}",
                    )

    return errors
