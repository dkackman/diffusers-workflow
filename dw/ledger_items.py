"""The requests ledger's item checks: one incoming item of a plan, field by
field, as it is stored.

`_item` answers the stored form of one live item a write lists - its
title, `depends_on`, estimate, workflow, arguments, `accepted` take and
review hold, each checked and normalized by its own function here, with the
server's stamps (`changed_at`, `accepted_basis`) - and `_check_dependencies`
refuses a plan whose `depends_on` names a missing item, itself or a cycle.
`_text` and `_key` are the two checks every other field of a write goes
through too. Each raises LedgerError; none reads or writes the file.
`dw/ledger.py` owns the file, the lock and the write, and calls into these
from `put_request`; this module never imports it.
"""

import math
import re
from datetime import datetime, timezone

from .ledger_views import LedgerError, current_prompts, is_revised
from .security import (
    MAX_VARIABLE_VALUE_LENGTH,
    InvalidInputError,
    validate_ledger_key,
    validate_string_input,
    validate_variable_name,
)


MAX_TITLE_LENGTH = 200
MAX_DEPENDS_ON = 16
MAX_ESTIMATE_MINUTES = 10_000
MAX_ARGUMENTS = 64
MAX_WORKFLOW_NAME_LENGTH = 200
MIN_REVIEW_SECONDS = 60  # a review hold's shortest span
MAX_REVIEW_HOURS = 24  # and its longest

# A stored workflow name: a catalog or library name such as `my-shot`,
# `models/flux-dev` or `builtin:upscale`. Stored only - nothing resolves it
# here - but held to name characters so a later stage can.
_WORKFLOW_NAME_PATTERN = r"^[\w][\w.:-]*(/[\w][\w.:-]*)*\Z"


def _text(value, what, max_length, *, one_line=False, allow_empty=False):
    if not isinstance(value, str):
        raise LedgerError(f"{what} must be text")
    try:
        validate_string_input(value, max_length=max_length, allow_empty=allow_empty)
    except InvalidInputError as e:
        raise LedgerError(f"{what}: {e}") from e
    if one_line and ("\n" in value or "\r" in value):
        raise LedgerError(f"{what} must be one line")
    if not allow_empty and not value.strip():
        raise LedgerError(f"{what} cannot be blank")
    return value


def _key(value, what):
    try:
        return validate_ledger_key(value, what)
    except InvalidInputError as e:
        raise LedgerError(str(e)) from e


def _estimate(value, key):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"item {key!r}: estimate_minutes must be a number or null")
    # The range first: isfinite raises on an integer too large for a float
    if not 0 <= value <= MAX_ESTIMATE_MINUTES or not math.isfinite(value):
        raise LedgerError(
            f"item {key!r}: estimate_minutes must be between 0 and "
            f"{MAX_ESTIMATE_MINUTES}, got {value}"
        )
    return value


def _workflow(value, key):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LedgerError(f"item {key!r}: workflow must be a name or null")
    if len(value) > MAX_WORKFLOW_NAME_LENGTH:
        raise LedgerError(
            f"item {key!r}: workflow name too long: {len(value)} > "
            f"{MAX_WORKFLOW_NAME_LENGTH}"
        )
    if not re.match(_WORKFLOW_NAME_PATTERN, value) or ".." in value.split("/"):
        raise LedgerError(
            f"item {key!r}: invalid workflow name {value!r} - a catalog or "
            "library name such as 'my-shot' or 'models/flux-dev'"
        )
    return value


def _arguments(value, key):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LedgerError(f"item {key!r}: arguments must be an object")
    if len(value) > MAX_ARGUMENTS:
        raise LedgerError(
            f"item {key!r}: at most {MAX_ARGUMENTS} arguments, got {len(value)}"
        )
    out = {}
    for name, argument in value.items():
        try:
            validate_variable_name(name)
        except InvalidInputError as e:
            raise LedgerError(f"item {key!r}: {e}") from e
        if isinstance(argument, str):
            _text(
                argument,
                f"item {key!r} argument {name!r}",
                MAX_VARIABLE_VALUE_LENGTH,
                allow_empty=True,
            )
        elif isinstance(argument, float) and not math.isfinite(argument):
            raise LedgerError(f"item {key!r} argument {name!r} must be finite")
        elif argument is not None and not isinstance(argument, (bool, int, float)):
            raise LedgerError(
                f"item {key!r} argument {name!r}: only a string, number, "
                "boolean or null is stored, not "
                f"{'a list' if isinstance(argument, list) else 'an object'}"
            )
        out[name] = argument
    return out


def _accepted(value, key, existing, linked):
    if value is None:
        return None
    if isinstance(value, str):
        value = {"job": value}
    if not isinstance(value, dict) or set(value) - {"job", "run_id"}:
        raise LedgerError(
            f"item {key!r}: accepted must be {{job}} (or a job id) or null"
        )
    if existing and value.get("job") == existing.get("job"):
        return existing
    job = value.get("job")
    for entry in linked:
        if entry.get("job") == job:
            if entry.get("status") != "succeeded":
                raise LedgerError(
                    f"item {key!r}: job {job!r} has not succeeded, so it "
                    "cannot be accepted"
                )
            return {"job": job, "run_id": entry.get("run_id")}
    raise LedgerError(
        f"item {key!r}: job {job!r} is not linked to this request and item, "
        "so it cannot be accepted"
    )


def _check_dependencies(items):
    keys = {item["key"] for item in items}
    graph = {}
    for item in items:
        deps = item["depends_on"]
        for dep in deps:
            if dep == item["key"]:
                raise LedgerError(f"item {dep!r} depends on itself")
            if dep not in keys:
                raise LedgerError(
                    f"item {item['key']!r} depends on {dep!r}, which is not an "
                    "item of this request"
                )
        graph[item["key"]] = deps
    # Iterative three-colour DFS: a grey node met again closes a cycle
    colour = dict.fromkeys(graph, 0)
    for start in graph:
        if colour[start]:
            continue
        stack = [(start, iter(graph[start]))]
        colour[start] = 1
        path = [start]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                colour[node] = 2
                stack.pop()
                path.pop()
            elif colour[child] == 1:
                cycle = path[path.index(child) :] + [child]
                raise LedgerError("depends_on has a cycle: " + " -> ".join(cycle))
            elif colour[child] == 0:
                colour[child] = 1
                stack.append((child, iter(graph[child])))
                path.append(child)


def _depends_on(raw, key):
    depends_on = raw.get("depends_on") or []
    if not isinstance(depends_on, list):
        raise LedgerError(f"item {key!r}: depends_on must be a list of keys")
    if len(depends_on) > MAX_DEPENDS_ON:
        raise LedgerError(
            f"item {key!r}: depends on at most {MAX_DEPENDS_ON} items, "
            f"got {len(depends_on)}"
        )
    keys = [_key(dep, f"item {key!r} depends_on key") for dep in depends_on]
    return list(dict.fromkeys(keys))


def _review_until(value, key, previous, at):
    """`(review_until, review_set_at)` as stored, in UTC. A value equal to the
    stored one keeps its stamp, so a plan written back unchanged does not
    restart the hold; a new one falls MIN_REVIEW_SECONDS..MAX_REVIEW_HOURS out."""
    if value is None:
        return None, None
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        when = None
    if when is None or when.utcoffset() is None:
        raise _review_error(key, "invalid_review_until", "an ISO time with an offset")
    when = when.astimezone(timezone.utc).replace(microsecond=0)
    if when.isoformat() == previous.get("review_until"):
        return previous["review_until"], previous.get("review_set_at")
    ahead = (when - datetime.fromisoformat(at)).total_seconds()
    if ahead < MIN_REVIEW_SECONDS:
        bound = f"at least {MIN_REVIEW_SECONDS} seconds from now"
        raise _review_error(key, "review_too_soon", bound)
    if ahead > MAX_REVIEW_HOURS * 3600:
        raise _review_error(key, "review_too_far", f"within {MAX_REVIEW_HOURS} hours")
    return when.isoformat(), at


def _review_error(key, code, must_be):
    return LedgerError(
        f"item {key!r} review_until must be {must_be}", status=422, code=code
    )


def _item(raw, key, previous, linked, by, at, prompts=None, warnings=None):
    """One live item as stored, with the server's stamps: `changed_at`, set
    when the item is created or its `workflow` or `arguments` differ from the
    stored ones and kept otherwise (so a GET->PUT round trip moves nothing);
    and `accepted_basis` (`{at, prompts}`), recorded when the write names an
    `accepted` take while the stored item reads revised and the write
    leaves its plan as it was - the item then reads done until the plan
    moves again, or a prompt it references gets text other than the digest
    `prompts` records (`ledger_views.is_revised`; `prompts` the `name ->
    digest` callable). A write that names `accepted` while it also changes
    the plan records no basis, so the take is not approved for a plan it never
    saw: that is said in `warnings` (a list the caller owns), per item."""
    until, set_at = _review_until(raw.get("review_until"), key, previous, at)
    workflow = _workflow(raw.get("workflow"), key)
    arguments = _arguments(raw.get("arguments"), key)
    accepted = _accepted(raw.get("accepted"), key, previous.get("accepted"), linked)
    unchanged = bool(previous) and (
        workflow == previous.get("workflow")
        and arguments == (previous.get("arguments") or {})
    )
    changed_at = previous.get("changed_at") if unchanged else at
    # The basis vouches for the accepted take: with none accepted, none stands
    basis = previous.get("accepted_basis") if accepted is not None else None
    if accepted is not None and unchanged and is_revised(previous, linked, prompts):
        basis = {"at": at}
        held = current_prompts(previous, prompts)
        if held:
            basis["prompts"] = held
    elif accepted is not None and previous and not unchanged and warnings is not None:
        warnings.append(
            f"item {key!r}: accepted not recorded as current - this write also "
            "changed its workflow or arguments, so the item reads revised; "
            "re-accept in a separate later write to keep the old take"
        )
    return {
        "key": key,
        "title": _text(
            raw.get("title"), f"item {key!r} title", MAX_TITLE_LENGTH, one_line=True
        ),
        "depends_on": _depends_on(raw, key),
        "estimate_minutes": _estimate(raw.get("estimate_minutes"), key),
        "workflow": workflow,
        "arguments": arguments,
        "dropped": False,
        "accepted": accepted,
        "notes": list(previous.get("notes") or []),
        "review_until": until,
        "review_set_at": set_at,
        "changed_at": changed_at,
        **({"accepted_basis": basis} if isinstance(basis, dict) else {}),
    }
