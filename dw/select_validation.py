"""Static validation for select's arguments (docs/proposals/score-and-select.md).

select's own signature carries no domain - `rule` is any string, and
`threshold`/`index` are only meaningful for some rules - so a step whose
rule is misspelled, or whose threshold is missing, or whose index rule has
none, validates clean today and dies on select's own run-time ValueError
after the fan-out ahead of it has already generated. Checked here in
`validation_errors` (free), and unchanged as select's own run-time check
for a value arriving from a `variable:` or an earlier step, which this
static pass can never see.
"""

from . import references
from .for_each import MEMBER_SEPARATOR, render_path
from .task_domains import SELECT_RULES, SELECT_THRESHOLD_RULES, select_rule_problems


def _problem_key(rule):
    """The argument a select_rule_problems sentence is about: the rule when
    it is unknown, else the one argument that rule needs."""
    if rule not in SELECT_RULES:
        return "rule"
    return "threshold" if rule in SELECT_THRESHOLD_RULES else "index"


def _where(name):
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
        where = _where(name)

        def add(key, message):
            path = render_path(("steps", source, "task", "arguments", key))
            errors.append({"path": path, "message": f"{message}{where}."})

        rule = arguments.get("rule")
        if isinstance(rule, str):
            # What the run itself refuses, in its own sentence - at most one,
            # keyed to the argument it is about
            problems = {
                _problem_key(rule): problem
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
