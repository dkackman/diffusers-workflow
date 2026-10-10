"""Four step-value checkers: `fps_errors` (a result's `fps`),
`null_media_errors` (a bare object description whose media resolved null),
`select_errors` (a `select` step's arguments) and `chain_prompts_errors` (a
chain's `prompts` that is not a list).

Each is a pure function of an expanded definition (and the source step
indices that map an expanded step back to the one the author wrote) returning
`[{path, message}]`. They were folded into dw/validation.py in 2b to pay for
creating it; their bodies sit here so the registry and the runner stay whole
in validation.py. Their only caller is the error registry there.
"""

from . import references
from .arguments import names_no_media
from .for_each import render_path
from .references import (
    FROM_ARGUMENTS_KEY,
    FROM_FILE_KEY,
    FROM_PREVIOUS_RESULT_KEY,
    iter_steps,
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


def fps_errors(workflow_definition, source_indices=None):
    """Every result 'fps' that cannot be written, as [{path, message}].

    The definition handed here has already been substituted and expanded,
    so every value in it is literal; a 'variable:' or 'item:' still spelled
    out is left alone. `source_indices`, when given, is the source step
    index of each step - a 'for_each' group turns one written step into
    several, and the path an error carries has to be one the author can
    find in the file they wrote; the member is named in the message.
    """
    errors = []
    for _, _, result, source, where in iter_steps(
        workflow_definition.get("steps"), source_indices, "result"
    ):
        if FPS_KEY not in result:
            continue
        value = result[FPS_KEY]
        # A prefix substitution resolves before this pass: one still spelled
        # out is the undeclared-variable pass's complaint, not a shape error
        if references.is_ref(references.SUBSTITUTED, value):
            continue

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
    errors = []
    for _, _, pipeline, source, where in iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict):
            continue
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


def _written_candidate_count(written_steps, source, candidates):
    """How many candidates the run will have, when validation can know: the
    author wrote `candidates` as a list. A list a variable resolved to is not
    counted - a composed child is validated with its defaults, and the
    parent's list arrives only at run time."""
    if not isinstance(candidates, list):
        return None
    if not isinstance(written_steps, list) or not 0 <= source < len(written_steps):
        return None
    task = (
        written_steps[source].get("task")
        if isinstance(written_steps[source], dict)
        else None
    )
    arguments = task.get("arguments") if isinstance(task, dict) else None
    if not isinstance(arguments, dict) or not isinstance(
        arguments.get("candidates"), list
    ):
        return None
    return len(candidates)


def _select_rule_errors(rule, arguments, candidate_count, add):
    """What select_errors refuses about `rule` and the arguments it reads,
    through `add(key, message)`."""
    # What the run itself refuses, in its own sentence - at most one,
    # keyed to the argument it is about
    problems = {
        _select_problem_key(rule): problem
        for problem in select_rule_problems(
            rule,
            arguments.get("threshold"),
            arguments.get("index"),
            # The range is refused here, in the run's own sentence,
            # when the count is the author's
            candidate_count,
        )
    }
    if "rule" in problems:
        add("rule", problems["rule"])
        return
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
            f"select: 'index' is only meaningful for rule 'index', not {rule!r}",
        )


def select_errors(workflow_definition, source_indices=None, written=None):
    """Every select step whose arguments cannot be right, as [{path, message}].

    `written` is the definition as the author wrote it (by default the one
    handed here), which says whether a `candidates` list is the run's own.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []
    written_steps = (workflow_definition if written is None else written).get("steps")

    errors = []
    for _, _, task, source, where in iter_steps(steps, source_indices, "task"):
        if task.get("command") != "select":
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict):
            continue

        def add(key, message):
            path = render_path(("steps", source, "task", "arguments", key))
            errors.append({"path": path, "message": f"{message}{where}."})

        candidates = arguments.get("candidates")
        rule = arguments.get("rule")
        if isinstance(rule, str):
            _select_rule_errors(
                rule,
                arguments,
                _written_candidate_count(written_steps, source, candidates),
                add,
            )

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


# --- A chain's 'prompts' (#653) ---------------------------------------------
#
# The schema lets `chain.prompts` hold a string so a `variable:` reference
# validates, but the resolved value has to be a list: a bare string would be
# walked character by character (dw/previous_results.py), giving each segment
# one letter of it as its prompt.


def chain_prompts_errors(workflow_definition, source_indices=None):
    """Every chain whose resolved 'prompts' is a string, as [{path, message}]."""
    errors = []
    for _, _, pipeline, source, _ in iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        chain = pipeline.get("chain")
        if not isinstance(chain, dict) or not isinstance(chain.get("prompts"), str):
            continue
        path = render_path(("steps", source, "pipeline", "chain", "prompts"))
        errors.append(
            {
                "path": path,
                "message": (
                    "chain 'prompts' must be a list with one prompt per segment, "
                    f"got the string {chain['prompts']!r} - to give every segment "
                    "one prompt, set 'prompt' instead, or pass a one-element list"
                ),
            }
        )
    return errors
