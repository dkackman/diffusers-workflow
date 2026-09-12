"""Expand a step's 'for_each' list into one ordinary step per entry.

A template that generates one shot per entry of a list is otherwise written
the long way - 'shot_1', 'shot_2', ... each a near-copy of the one before -
and a six-shot episode is a different file from a five-shot one. This pass
runs on the definition after variable substitution and before the run id is
computed (Workflow.run) and before the reference check (validation_errors),
and produces a definition with no 'for_each' in it: every member is an
ordinary step, so the step loop, the step cache, the manifest and the
reference checker never learn a new reference kind.

Inside a member:
  - 'item:'       is the whole entry; 'item:field' one field of an object
                  entry, spliced in whole whatever its type
  - a reference to another for_each group over the SAME list resolves to
    the member with the same key ('slice' inside 'shot@open' -> 'slice@open')
Outside (or inside, for any other group):
  - 'gather:shot' is the list of every member's result, as explicit
    'previous_result:shot@<key>' strings; inside a list it splices
  - 'previous_result:shot' naming a group is an error that says to gather

Members are named '<group>@<key>' - the entry's own 'name' when it carries
one, else its index - and '@' is reserved in every step name. Names rather
than indexes because the step cache (dw/step_cache.py) keys on the step
name: inserting a shot in the middle of a list must not shift every later
member onto a different entry's cache line.
"""

import copy

from .arguments import FROM_PREVIOUS_RESULT_KEY, PREVIOUS_RESULT_PREFIX
from .security import InvalidInputError, validate_variable_name
from .step_cache import reference_resolves_to

FOR_EACH_KEY = "for_each"
ITEM_PREFIX = "item:"
GATHER_PREFIX = "gather:"
MEMBER_SEPARATOR = "@"
# Each entry is a full generation. Stated against the step cache's bound
# (DEFAULT_MAX_ENTRIES = 50): a run whose expanded steps exceed the cache
# evicts its own earlier members, so this is kept well under it
MAX_FOR_EACH_ENTRIES = 32
# release_pipeline / release_models would drop the model after the first
# member and reload it for the second, so they are carried onto the last one
_LAST_MEMBER_ONLY = ("release_pipeline", "release_models")


class ForEachError(ValueError):
    """A for_each step that cannot be expanded, with the JSON path at fault."""

    def __init__(self, path, message):
        super().__init__(message)
        self.path = path


def member_name(group, key):
    return f"{group}{MEMBER_SEPARATOR}{key}"


def expand_for_each(definition, source_indices=None):
    """The definition with every 'for_each' step replaced by its members.

    Returns a new structure; `definition` is left as it was passed in.
    Raises ForEachError for anything that cannot be expanded.

    `source_indices`, when a list is passed, has the index of the step each
    expanded step was written as appended to it, so a later check can report
    an error at a path in the file the author wrote rather than at an
    expanded index that exists nowhere. A parallel list rather than a key on
    the step: step_data is what the step cache keys on and what the schema
    validates, and neither may learn a new field.
    """
    steps = definition.get("steps") if isinstance(definition, dict) else None
    if not isinstance(steps, list):
        return definition

    # group name -> {"keys": [...], "entries": [...]} for every group
    # expanded so far, in step order, so a reference can only reach an
    # earlier group - the same rule previous_result: has always had
    groups = {}
    expanded = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            expanded.append(copy.deepcopy(step))
            _record(source_indices, index)
            continue
        path = ("steps", index)
        name = step.get("name")
        if isinstance(name, str) and MEMBER_SEPARATOR in name:
            raise ForEachError(
                render_path(path + ("name",)),
                f"Step name '{name}' contains '{MEMBER_SEPARATOR}', which is "
                f"reserved for the members of a for_each step",
            )
        if FOR_EACH_KEY not in step:
            expanded.append(_rewrite(step, path, groups, member=None))
            _record(source_indices, index)
            continue

        entries = step[FOR_EACH_KEY]
        keys = _entry_keys(entries, path + (FOR_EACH_KEY,))
        template = {k: v for k, v in step.items() if k != FOR_EACH_KEY}
        last = len(keys) - 1
        for position, (key, entry) in enumerate(zip(keys, entries)):
            member = {
                "group": name,
                "key": key,
                "entry": entry,
                "entries": entries,
                "index": position,
            }
            expanded_step = _rewrite(template, path, groups, member)
            expanded_step["name"] = member_name(name, key)
            if position != last:
                for flag in _LAST_MEMBER_ONLY:
                    expanded_step.pop(flag, None)
            expanded.append(expanded_step)
            _record(source_indices, index)
        groups[name] = {"keys": keys, "entries": entries}

    result = {k: v for k, v in definition.items() if k != "steps"}
    result["steps"] = expanded
    return result


def _record(source_indices, index):
    if source_indices is not None:
        source_indices.append(index)


def _entry_keys(entries, path):
    """The key of every entry - its 'name' when it is an object carrying
    one, else its index - validated and unique."""
    if not isinstance(entries, list):
        if isinstance(entries, str) and entries.startswith("variable:"):
            hint = f" - '{entries}' was not substituted; is the variable declared?"
        else:
            hint = ""
        raise ForEachError(
            render_path(path),
            f"for_each must be a list, got {type(entries).__name__}{hint}",
        )
    if len(entries) > MAX_FOR_EACH_ENTRIES:
        raise ForEachError(
            render_path(path),
            f"for_each has {len(entries)} entries; the limit is {MAX_FOR_EACH_ENTRIES}",
        )
    keys = []
    for index, entry in enumerate(entries):
        key = str(index)
        if isinstance(entry, dict) and "name" in entry:
            key = entry["name"]
            key_path = render_path(path + (index, "name"))
            if not isinstance(key, str):
                raise ForEachError(key_path, "An entry's name must be a string")
            try:
                validate_variable_name(key)
            except InvalidInputError as e:
                raise ForEachError(key_path, f"Invalid entry name '{key}': {e}") from e
            if key in keys:
                raise ForEachError(key_path, f"Duplicate entry name '{key}'")
        keys.append(key)
    return keys


def _rewrite(value, path, groups, member):
    """Rebuild `value` with item:, gather: and group references resolved.

    `member` is None outside a for_each step; inside one it carries the
    group, key and entry of the member being built.
    """
    if isinstance(value, dict):
        rebuilt = {}
        for key, item in value.items():
            if key == FROM_PREVIOUS_RESULT_KEY and isinstance(item, str):
                rebuilt[key] = _rewrite_reference(item, path + (key,), groups, member)
            else:
                rebuilt[key] = _rewrite(item, path + (key,), groups, member)
        return rebuilt
    if isinstance(value, list):
        rebuilt = []
        for index, item in enumerate(value):
            if isinstance(item, str) and item.startswith(GATHER_PREFIX):
                # A gather inside a list splices into it
                rebuilt.extend(_gather(item, path + (index,), groups))
            else:
                rebuilt.append(_rewrite(item, path + (index,), groups, member))
        return rebuilt
    if isinstance(value, str):
        if value.startswith(GATHER_PREFIX):
            return _gather(value, path, groups)
        if value.startswith(ITEM_PREFIX):
            return _item(value, path, member)
        if value.startswith(PREVIOUS_RESULT_PREFIX):
            reference = value[len(PREVIOUS_RESULT_PREFIX) :]
            return PREVIOUS_RESULT_PREFIX + _rewrite_reference(
                reference, path, groups, member
            )
        return value
    # A leaf is only copied where the copy is needed: inside a member, where
    # the same template value is about to appear in every one of them.
    # Outside, the leaf is handed back as it is - this pass runs on every run
    # of every workflow, after realize_args has turned 'asset:' arguments
    # into loaded images and decoded frame lists, and copying all of that
    # would multiply the media a run holds. The 'input untouched' contract
    # still holds because nothing here ever mutates a leaf
    return _copy_leaf(value) if member is not None else value


def _copy_leaf(value):
    """A copy of a leaf, or the leaf itself when it cannot be copied.

    An open handle or a live model object reaching a member is not a reason
    to fail a run - the step cache makes the same choice for a realized
    argument it cannot deep-copy (dw/workflow.py).
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _item(value, path, member):
    if member is None:
        raise ForEachError(
            render_path(path), f"'{value}' is only meaningful inside a for_each step"
        )
    field = value[len(ITEM_PREFIX) :]
    entry = member["entry"]
    if field == "":
        return _copy_leaf(entry)
    if not isinstance(entry, dict):
        raise ForEachError(
            render_path(path),
            f"'{value}' asks for a field of entry '{member['key']}' of "
            f"for_each step '{member['group']}', which is not an object",
        )
    if field not in entry:
        raise ForEachError(
            render_path(path),
            f"'{value}' names no field of entry '{member['key']}' of for_each "
            f"step '{member['group']}'; it has: {sorted(entry)}",
        )
    return _copy_leaf(entry[field])


def _gather(value, path, groups):
    group = value[len(GATHER_PREFIX) :]
    if group not in groups:
        raise ForEachError(
            render_path(path),
            f"'{value}' names no earlier for_each step. "
            f"for_each steps available here: {sorted(groups)}",
        )
    return [
        PREVIOUS_RESULT_PREFIX + member_name(group, key)
        for key in groups[group]["keys"]
    ]


def _rewrite_reference(reference, path, groups, member):
    """A previous_result reference (without its prefix) as the expanded
    definition spells it: unchanged unless it names a for_each group."""
    if reference.startswith("variable:"):
        return reference
    group = next((g for g in groups if reference_resolves_to(reference, g)), None)
    if group is None:
        if member is not None and reference_resolves_to(reference, member["group"]):
            raise ForEachError(
                render_path(path),
                f"'{reference}' names its own for_each step '{member['group']}'",
            )
        return reference
    if member is not None and member["entries"] == groups[group]["entries"]:
        # Same list: shot@open reads slice@open
        return member_name(group, member["key"]) + reference[len(group) :]
    where = (
        f"from inside for_each step '{member['group']}', which runs over a different list"
        if member is not None
        else "from outside a for_each step"
    )
    raise ForEachError(
        render_path(path),
        f"'{reference}' names the for_each step '{group}' {where}. Use "
        f"'{GATHER_PREFIX}{group}' for every member's result, or a reference "
        f"from a for_each step over the same list for the same-keyed member",
    )


def render_path(path):
    """'steps[3].task.arguments.videos[1]' - the same shape schema errors use."""
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = str(part)
    return rendered
