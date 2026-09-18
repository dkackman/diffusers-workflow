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

from .for_each import MEMBER_SEPARATOR, render_path

_RULES = {"argmax", "argmin", "first_above", "first_below", "index"}
_THRESHOLD_RULES = {"first_above", "first_below"}


def _where(name):
    return (
        f" in member '{name}'"
        if isinstance(name, str) and MEMBER_SEPARATOR in name
        else ""
    )


def _is_gather(value):
    return isinstance(value, str) and value.startswith("gather:")


def _is_expanded_gather(value):
    """A gather: reference after for_each expansion has replaced it with the
    list of previous_result: references it drew from - the only form
    select_errors ever actually sees once validation_errors expands the
    definition before calling it."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, str) and v.startswith("previous_result:") for v in value)
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

        source = (
            source_indices[index]
            if source_indices is not None and index < len(source_indices)
            else index
        )
        name = step.get("name")
        where = _where(name)

        def add(key, message):
            path = render_path(("steps", source, "task", "arguments", key))
            errors.append({"path": path, "message": f"{message}{where}."})

        rule = arguments.get("rule")
        if isinstance(rule, str) and rule not in _RULES:
            add("rule", f"select: unknown rule: {rule!r}")
        elif isinstance(rule, str):
            has_threshold = "threshold" in arguments
            if rule in _THRESHOLD_RULES and not has_threshold:
                add("threshold", f"select rule '{rule}' requires a threshold")
            elif rule not in _THRESHOLD_RULES and has_threshold:
                add(
                    "threshold",
                    f"select: 'threshold' is only meaningful for rule "
                    f"'first_above'/'first_below', not {rule!r}",
                )

            has_index = "index" in arguments
            if rule == "index" and not has_index:
                add("index", "select rule 'index' requires an index")
            elif rule != "index" and has_index:
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
