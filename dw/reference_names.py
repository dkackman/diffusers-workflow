"""The shape of an `asset:` / `prompt:` / `output:` name, checked for free.

An `output:` reference naming a file a `for_each` step wrote - the `@` in
`shot@opening_statement` - validated clean and then failed the job at run
time, after the queue, with a message that described a *valid* name and said
nothing about what it had objected to (#162). Two things were wrong with
that, and only one of them was the `@`: a name that can never resolve, in
any workspace, is not a run-time discovery. Its shape is a property of the
string alone.

So the shape is checked here, in `validation_errors`, which is what
`POST /api/validate`, `validate_workflow` and the pre-queue check all run -
while *existence* stays where it was. Whether a run id is still on disk
depends on the workspace and on what pruning has taken, and the definition's
own references (a template's `prompt:ltx2/hummingbird_garden`) are not the
caller's to answer for; the caller's `arguments` are separately resolved
against the workspace by the validate route.
"""

from .assets import ASSET_PREFIX
from .for_each import MEMBER_SEPARATOR, render_path
from .prompts import PROMPT_PREFIX
from .runs import OUTPUT_PREFIX
from .security import (
    InvalidInputError,
    validate_asset_reference,
    validate_output_reference,
    validate_prompt_reference,
)

# Substitution and expansion run before this pass, so every string reaching
# it is literal. One still spelled with a deferred prefix is nothing this
# pass resolved, and the undeclared-variable pass owns that complaint
_UNRESOLVED_PREFIXES = ("variable:", "item:", "previous_result:", "gather:")


def _output_name(reference):
    """The part of an `output:` reference the name rule applies to.

    `latest` in the run-id position is expanded before the path is joined,
    so it is checked as the ordinary segment it looks like.
    """
    return reference.removeprefix(OUTPUT_PREFIX).strip()


_KINDS = (
    (OUTPUT_PREFIX, validate_output_reference, _output_name),
    (ASSET_PREFIX, validate_asset_reference, None),
    (PROMPT_PREFIX, validate_prompt_reference, None),
)


def reference_name_errors(workflow_definition, source_indices=None):
    """Every reference in the definition whose *name* is malformed, as
    [{path, message}].

    Walks the substituted, expanded definition. `source_indices`, when
    given, maps each expanded step back to the step the author wrote, so an
    error inside a `for_each` member carries a path in their file and names
    the member.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
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
        for path, value in _strings(step, ("steps", source)):
            problem = reference_fault(value)
            if problem is not None:
                errors.append(
                    {"path": render_path(path), "message": f"{problem}{where}"}
                )
    return errors


def reference_fault(value):
    """Why this string's reference name is malformed, or None.

    None for anything that is not a reference, and for one still spelled
    with a deferred prefix behind the reference prefix - `output:` on a
    value substitution has not reached yet is not this pass's complaint.
    """
    if not isinstance(value, str):
        return None
    for prefix, check, extract in _KINDS:
        if not value.startswith(prefix):
            continue
        rest = value[len(prefix) :].strip()
        if not rest or rest.startswith(_UNRESOLVED_PREFIXES):
            return None
        try:
            check(extract(value) if extract else rest)
        except InvalidInputError as e:
            return str(e)
        return None
    return None


def _strings(value, path):
    """Every string inside a step, paired with the path it sits at."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, path + (index,))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, path + (key,))


__all__ = ["reference_fault", "reference_name_errors"]
