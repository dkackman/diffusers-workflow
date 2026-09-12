# List-Driven Steps (`for_each`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A step can carry `"for_each": <list>` and expand into one ordinary step per entry before the run starts, so a template takes a `shots` list instead of one hand-written step per shot.

**Architecture:** Expansion is a source transform, not a runtime concept. A new module `dw/for_each.py` holds one pure function, `expand_for_each(definition) -> definition`, that runs immediately after `replace_variables` in `Workflow.run` and, with the caller's arguments folded in, inside `Workflow.validation_errors`. After the pass the definition is an ordinary workflow: `item:` is substituted, `gather:group` is rewritten to an explicit `previous_result:` list, a same-key sibling reference is rewritten to the member with this key, and every member is named `group@key`. Nothing downstream - the step loop, the step cache, the manifest, the reference checker - learns a new reference kind.

**Tech Stack:** Python 3, pytest. No new dependencies.

**Spec:** `docs/proposals/list-driven-steps.md` (revised 2026-09-11). This plan is stage 1 of that proposal plus the agent-facing documentation; the template rewrite (stage 2) and catalog cost/entry-shape reporting (stage 3) are a later plan.

## Global Constraints

- Member separator is `@`; `@` is reserved in every step name, so a hand-written step name containing `@` is an error.
- Entry `name` must match `^[a-zA-Z_][a-zA-Z0-9_-]*$` (the `validate_variable_name` pattern in `dw/security.py`) and be unique within its list; an entry without a `name` (or a non-object entry) is keyed by its index.
- Ceiling: `MAX_FOR_EACH_ENTRIES = 32`; a longer list is refused with an error naming the ceiling.
- `for_each` runs over exactly one list; no index (`item:@index`), no numeric `for_each`, no zip.
- `item:field` yields whatever the field holds - string, number, list, object - spliced in as definition text. Bare `item:` is the whole entry.
- `gather:group` written inside a list splices into it; written as a scalar value it becomes a list. `gather:` naming a step that is not an earlier `for_each` group is an error.
- Inside a member, `previous_result:other` / `from_previous_result: "other"` naming an *earlier* `for_each` group over the *same* list (deep-equal) rewrites to `other@<this key>`. Naming a group any other way (from outside a group, from a group over a different list, or one's own group) is an error that says to use `gather:`.
- `release_pipeline` and `release_models` on a `for_each` step survive on the last member only.
- The realized workflow (`dw/realize.py`) keeps `for_each`; the manifest and events name the expanded members. No change to `dw/realize.py` is needed - it works on `self.workflow_definition`, which the pass never touches.
- Never use `eval()`, `exec()`, or `shell=True`. Errors carry a JSON path in the `steps[3].pipeline.arguments.prompt` shape `_render_path` in `dw/previous_results.py` produces.
- Run the suite with `python -m pytest tests -q -x`; commit after each task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility |
|---|---|
| `dw/for_each.py` (create) | The pass: constants, `ForEachError`, `expand_for_each`, `for_each_errors`. Pure; no I/O, no engine imports beyond `dw.security.validate_variable_name`, `dw.step_cache.reference_resolves_to` and the two constants from `dw.arguments`. |
| `dw/workflow.py` (modify) | Call the pass in `run` after substitution; `validation_errors(arguments=None)` substitutes, expands, then reference-checks the expanded definition. |
| `dw/workflow_schema.json` (modify) | `for_each` on a step. |
| `dw/server/app.py` (modify) | `validate_workflow` passes `request.arguments` into `validation_errors`. |
| `tests/test_for_each.py` (create) | Unit tests for the pass, including the two templates expanded back to what they contain today. |
| `tests/test_workflow.py`, `tests/test_validate_arguments.py` (modify) | Integration: run-time expansion, validation ordering. |
| `docs/WORKFLOW_GUIDE.md`, `CLAUDE.md`, `docs/SERVER.md` (modify) | The agent-facing conventions, the CLAUDE.md mirror, the manifest naming note. |

---

### Task 1: The pass - names, ceiling, `item:` substitution

**Files:**
- Create: `dw/for_each.py`
- Test: `tests/test_for_each.py`

**Interfaces:**
- Consumes: `validate_variable_name` (`dw/security.py:390`, raises `InvalidInputError`), `reference_resolves_to` (`dw/step_cache.py:58`), `FROM_PREVIOUS_RESULT_KEY`, `PREVIOUS_RESULT_PREFIX` (`dw/arguments.py:49,58`).
- Produces: `expand_for_each(definition: dict) -> dict` (new dict, input untouched), `class ForEachError(ValueError)` with `.path: str`, constants `FOR_EACH_KEY = "for_each"`, `ITEM_PREFIX = "item:"`, `GATHER_PREFIX = "gather:"`, `MEMBER_SEPARATOR = "@"`, `MAX_FOR_EACH_ENTRIES = 32`, and `member_name(group: str, key: str) -> str`. Tasks 2 and 3 extend this module; Tasks 4-6 call it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_for_each.py
import copy

import pytest

from dw.for_each import (
    MAX_FOR_EACH_ENTRIES,
    ForEachError,
    expand_for_each,
    member_name,
)


def definition(*steps, **extra):
    return {"id": "test", "steps": list(steps), **extra}


class TestNaming:
    def test_member_name_joins_with_at(self):
        assert member_name("shot", "wide_open") == "shot@wide_open"

    def test_an_object_entry_is_named_by_its_name(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "wide_open"}, {"name": "closeup"}],
                    "task": {"command": "x", "arguments": {}},
                }
            )
        )
        assert [s["name"] for s in expanded["steps"]] == [
            "shot@wide_open",
            "shot@closeup",
        ]

    def test_a_nameless_entry_is_named_by_its_index(self):
        expanded = expand_for_each(
            definition(
                {"name": "shot", "for_each": ["a", "b"], "task": {"arguments": {}}}
            )
        )
        assert [s["name"] for s in expanded["steps"]] == ["shot@0", "shot@1"]

    def test_for_each_is_dropped_from_the_members(self):
        expanded = expand_for_each(
            definition({"name": "shot", "for_each": ["a"], "task": {}})
        )
        assert "for_each" not in expanded["steps"][0]

    def test_the_input_is_not_mutated(self):
        original = definition({"name": "shot", "for_each": ["a"], "task": {}})
        before = copy.deepcopy(original)
        expand_for_each(original)
        assert original == before

    def test_a_workflow_without_for_each_comes_back_equal(self):
        original = definition({"name": "plain", "task": {"arguments": {"a": 1}}})
        assert expand_for_each(original) == original

    def test_a_hand_written_at_sign_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(definition({"name": "shot@0", "task": {}}))
        assert e.value.path == "steps[0].name"
        assert "@" in str(e.value)

    def test_a_duplicate_entry_name_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": [{"name": "closeup"}, {"name": "closeup"}],
                        "task": {},
                    }
                )
            )
        assert e.value.path == "steps[0].for_each[1].name"
        assert "closeup" in str(e.value)

    def test_an_invalid_entry_name_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": [{"name": "wide open"}], "task": {}}
                )
            )
        assert e.value.path == "steps[0].for_each[0].name"

    def test_a_non_list_for_each_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(definition({"name": "shot", "for_each": 4, "task": {}}))
        assert e.value.path == "steps[0].for_each"
        assert "list" in str(e.value)

    def test_an_unsubstituted_variable_is_an_error_that_says_so(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition({"name": "shot", "for_each": "variable:shots", "task": {}})
            )
        assert "variable:shots" in str(e.value)

    def test_an_empty_list_expands_to_no_steps(self):
        expanded = expand_for_each(
            definition(
                {"name": "shot", "for_each": [], "task": {}},
                {"name": "after", "task": {}},
            )
        )
        assert [s["name"] for s in expanded["steps"]] == ["after"]

    def test_the_ceiling_is_enforced(self):
        entries = [{"name": f"s{i}"} for i in range(MAX_FOR_EACH_ENTRIES + 1)]
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition({"name": "shot", "for_each": entries, "task": {}})
            )
        assert str(MAX_FOR_EACH_ENTRIES) in str(e.value)

    def test_a_list_at_the_ceiling_is_fine(self):
        entries = [{"name": f"s{i}"} for i in range(MAX_FOR_EACH_ENTRIES)]
        expanded = expand_for_each(
            definition({"name": "shot", "for_each": entries, "task": {}})
        )
        assert len(expanded["steps"]) == MAX_FOR_EACH_ENTRIES


class TestItemSubstitution:
    def test_a_field_is_substituted(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a", "prompt": "the band walks on"}],
                    "pipeline": {"arguments": {"prompt": "item:prompt"}},
                }
            )
        )
        assert expanded["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "the band walks on"
        )

    def test_a_bare_item_is_the_whole_entry(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": ["first prompt", "second prompt"],
                    "pipeline": {"arguments": {"prompt": "item:"}},
                }
            )
        )
        prompts = [s["pipeline"]["arguments"]["prompt"] for s in expanded["steps"]]
        assert prompts == ["first prompt", "second prompt"]

    def test_a_structured_field_is_spliced_whole(self):
        references = [
            {"reference_type": "T", "from_previous_result": "draw_a"},
            {"reference_type": "T", "from_previous_result": "draw_b"},
        ]
        expanded = expand_for_each(
            definition(
                {"name": "draw_a", "task": {}},
                {"name": "draw_b", "task": {}},
                {
                    "name": "shot",
                    "for_each": [{"name": "open", "references": references}],
                    "pipeline": {"arguments": {"references": "item:references"}},
                },
            )
        )
        assert expanded["steps"][2]["pipeline"]["arguments"]["references"] == references

    def test_a_number_keeps_its_type(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "slice",
                    "for_each": [{"name": "a", "start_frame": 124}],
                    "task": {"arguments": {"start_frame": "item:start_frame"}},
                }
            )
        )
        assert expanded["steps"][0]["task"]["arguments"]["start_frame"] == 124

    def test_item_is_substituted_at_any_depth(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a", "voice": "asset:a.wav"}],
                    "pipeline": {
                        "arguments": {"references": [{"from_file": "item:voice"}]}
                    },
                }
            )
        )
        assert expanded["steps"][0]["pipeline"]["arguments"]["references"] == [
            {"from_file": "asset:a.wav"}
        ]

    def test_a_missing_field_is_an_error_at_its_path(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": [{"name": "a"}],
                        "pipeline": {"arguments": {"prompt": "item:prompt"}},
                    }
                )
            )
        assert e.value.path == "steps[0].pipeline.arguments.prompt"
        assert "prompt" in str(e.value) and "a" in str(e.value)

    def test_a_field_of_a_string_entry_is_an_error(self):
        with pytest.raises(ForEachError):
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": ["just a prompt"],
                        "pipeline": {"arguments": {"prompt": "item:prompt"}},
                    }
                )
            )

    def test_item_outside_a_for_each_step_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition({"name": "plain", "task": {"arguments": {"p": "item:x"}}})
            )
        assert e.value.path == "steps[0].task.arguments.p"

    def test_members_do_not_share_mutable_values(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a"}, {"name": "b"}],
                    "pipeline": {"arguments": {"references": [{"k": 1}]}},
                }
            )
        )
        first, second = expanded["steps"]
        first["pipeline"]["arguments"]["references"][0]["k"] = 2
        assert second["pipeline"]["arguments"]["references"][0]["k"] == 1


class TestRelease:
    def test_release_flags_survive_on_the_last_member_only(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                    "release_pipeline": True,
                    "release_models": True,
                    "pipeline": {},
                }
            )
        )
        flags = [
            (s.get("release_pipeline"), s.get("release_models"))
            for s in expanded["steps"]
        ]
        assert flags == [(None, None), (None, None), (True, True)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_for_each.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dw.for_each'`

- [ ] **Step 3: Write the module**

```python
# dw/for_each.py
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
import re

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


def expand_for_each(definition):
    """The definition with every 'for_each' step replaced by its members.

    Returns a new structure; `definition` is left as it was passed in.
    Raises ForEachError for anything that cannot be expanded.
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
            continue
        path = ("steps", index)
        name = step.get("name")
        if isinstance(name, str) and MEMBER_SEPARATOR in name:
            raise ForEachError(
                _render_path(path + ("name",)),
                f"Step name '{name}' contains '{MEMBER_SEPARATOR}', which is "
                f"reserved for the members of a for_each step",
            )
        if FOR_EACH_KEY not in step:
            expanded.append(_rewrite(step, path, groups, member=None))
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
        groups[name] = {"keys": keys, "entries": entries}

    result = {k: v for k, v in definition.items() if k != "steps"}
    result["steps"] = expanded
    return result


def _entry_keys(entries, path):
    """The key of every entry - its 'name' when it is an object carrying
    one, else its index - validated and unique."""
    if not isinstance(entries, list):
        if isinstance(entries, str) and entries.startswith("variable:"):
            hint = f" - '{entries}' was not substituted; is the variable declared?"
        else:
            hint = ""
        raise ForEachError(
            _render_path(path), f"for_each must be a list, got {type(entries).__name__}{hint}"
        )
    if len(entries) > MAX_FOR_EACH_ENTRIES:
        raise ForEachError(
            _render_path(path),
            f"for_each has {len(entries)} entries; the limit is {MAX_FOR_EACH_ENTRIES}",
        )
    keys = []
    for index, entry in enumerate(entries):
        key = str(index)
        if isinstance(entry, dict) and "name" in entry:
            key = entry["name"]
            key_path = _render_path(path + (index, "name"))
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
    return copy.deepcopy(value)


def _item(value, path, member):
    if member is None:
        raise ForEachError(
            _render_path(path), f"'{value}' is only meaningful inside a for_each step"
        )
    field = value[len(ITEM_PREFIX) :]
    entry = member["entry"]
    if field == "":
        return copy.deepcopy(entry)
    if not isinstance(entry, dict):
        raise ForEachError(
            _render_path(path),
            f"'{value}' asks for a field of entry '{member['key']}' of "
            f"for_each step '{member['group']}', which is not an object",
        )
    if field not in entry:
        raise ForEachError(
            _render_path(path),
            f"'{value}' names no field of entry '{member['key']}' of for_each "
            f"step '{member['group']}'; it has: {sorted(entry)}",
        )
    return copy.deepcopy(entry[field])


def _gather(value, path, groups):
    group = value[len(GATHER_PREFIX) :]
    if group not in groups:
        raise ForEachError(
            _render_path(path),
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
                _render_path(path),
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
        _render_path(path),
        f"'{reference}' names the for_each step '{group}' {where}. Use "
        f"'{GATHER_PREFIX}{group}' for every member's result, or a reference "
        f"from a for_each step over the same list for the same-keyed member",
    )


def _render_path(path):
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
```

Note: `_render_path` duplicates `dw/previous_results.py:_render_path`; move that one here and import it from `previous_results` (`from .for_each import _render_path` would be circular the other way - `for_each` must not import `previous_results`). Do the move: delete the copy in `previous_results.py` and add `from .for_each import _render_path` there, renaming it `render_path` (public) in both places.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_for_each.py tests/test_previous_results.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dw/for_each.py dw/previous_results.py tests/test_for_each.py
git commit -m "feat(for_each): expansion pass - member naming, ceiling, item: substitution

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `gather:` and group references

**Files:**
- Modify: `dw/for_each.py` (already written in Task 1; this task tests the `_gather` / `_rewrite_reference` behaviour and fixes what the tests find)
- Test: `tests/test_for_each.py`

**Interfaces:**
- Consumes: Task 1's module.
- Produces: the same `expand_for_each`; behaviour now pinned for `gather:`, same-key siblings and the directed errors.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_for_each.py`:

```python
class TestGather:
    def group(self, *extra):
        return definition(
            {
                "name": "shot",
                "for_each": [{"name": "open"}, {"name": "close"}],
                "pipeline": {"arguments": {}},
            },
            *extra,
        )

    def test_a_scalar_gather_becomes_the_member_list(self):
        expanded = expand_for_each(
            self.group(
                {"name": "edit", "task": {"arguments": {"videos": "gather:shot"}}}
            )
        )
        assert expanded["steps"][-1]["task"]["arguments"]["videos"] == [
            "previous_result:shot@open",
            "previous_result:shot@close",
        ]

    def test_a_gather_inside_a_list_splices(self):
        expanded = expand_for_each(
            self.group(
                {"name": "intro", "task": {}},
                {
                    "name": "edit",
                    "task": {
                        "arguments": {
                            "videos": ["previous_result:intro", "gather:shot"]
                        }
                    },
                },
            )
        )
        assert expanded["steps"][-1]["task"]["arguments"]["videos"] == [
            "previous_result:intro",
            "previous_result:shot@open",
            "previous_result:shot@close",
        ]

    def test_gather_of_an_empty_group_is_an_empty_list(self):
        expanded = expand_for_each(
            definition(
                {"name": "shot", "for_each": [], "task": {}},
                {"name": "edit", "task": {"arguments": {"videos": "gather:shot"}}},
            )
        )
        assert expanded["steps"][-1]["task"]["arguments"]["videos"] == []

    def test_gather_of_a_plain_step_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "one", "task": {}},
                    {"name": "edit", "task": {"arguments": {"videos": "gather:one"}}},
                )
            )
        assert e.value.path == "steps[1].task.arguments.videos"
        assert "no earlier for_each step" in str(e.value)

    def test_gather_of_a_later_group_is_an_error(self):
        with pytest.raises(ForEachError):
            expand_for_each(
                definition(
                    {"name": "edit", "task": {"arguments": {"videos": "gather:shot"}}},
                    {"name": "shot", "for_each": ["a"], "task": {}},
                )
            )

    def test_gather_in_a_sub_workflow_argument_map(self):
        expanded = expand_for_each(
            self.group(
                {
                    "name": "score",
                    "workflow": {"path": "builtin:x.json", "arguments": {"clips": "gather:shot"}},
                }
            )
        )
        assert expanded["steps"][-1]["workflow"]["arguments"]["clips"] == [
            "previous_result:shot@open",
            "previous_result:shot@close",
        ]


class TestGroupReferences:
    def test_previous_result_naming_a_group_says_to_gather(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": ["a"], "task": {}},
                    {
                        "name": "edit",
                        "task": {"arguments": {"video": "previous_result:shot"}},
                    },
                )
            )
        assert e.value.path == "steps[1].task.arguments.video"
        assert "gather:shot" in str(e.value)

    def test_from_previous_result_naming_a_group_says_to_gather(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": ["a"], "task": {}},
                    {
                        "name": "edit",
                        "task": {
                            "arguments": {"refs": [{"from_previous_result": "shot"}]}
                        },
                    },
                )
            )
        assert e.value.path == "steps[1].task.arguments.refs[0].from_previous_result"

    def test_a_property_reference_to_a_group_is_also_refused(self):
        with pytest.raises(ForEachError):
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": ["a"], "task": {}},
                    {
                        "name": "edit",
                        "task": {"arguments": {"v": "previous_result:shot.frames"}},
                    },
                )
            )

    def test_a_reference_to_an_ordinary_step_is_untouched(self):
        expanded = expand_for_each(
            definition(
                {"name": "draw", "task": {}},
                {
                    "name": "shot",
                    "for_each": ["a"],
                    "pipeline": {
                        "arguments": {
                            "references": [{"from_previous_result": "draw"}],
                            "still": "previous_result:draw.image",
                        }
                    },
                },
            )
        )
        arguments = expanded["steps"][1]["pipeline"]["arguments"]
        assert arguments["references"] == [{"from_previous_result": "draw"}]
        assert arguments["still"] == "previous_result:draw.image"

    def test_a_variable_spelled_from_previous_result_is_untouched(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "edit",
                    "task": {"arguments": {"r": [{"from_previous_result": "variable:x"}]}},
                }
            )
        )
        assert expanded["steps"][0]["task"]["arguments"]["r"] == [
            {"from_previous_result": "variable:x"}
        ]


class TestSameKeySiblings:
    def shots(self):
        return [
            {"name": "open", "start_frame": 0},
            {"name": "close", "start_frame": 124},
        ]

    def test_a_sibling_over_the_same_list_resolves_to_the_same_key(self):
        shots = self.shots()
        expanded = expand_for_each(
            definition(
                {
                    "name": "slice",
                    "for_each": shots,
                    "task": {"arguments": {"start_frame": "item:start_frame"}},
                },
                {
                    "name": "shot",
                    "for_each": shots,
                    "pipeline": {
                        "arguments": {
                            "references": [{"from_previous_result": "slice"}],
                            "audio": "previous_result:slice.audio",
                        }
                    },
                },
            )
        )
        names = [s["name"] for s in expanded["steps"]]
        assert names == ["slice@open", "slice@close", "shot@open", "shot@close"]
        close = expanded["steps"][3]["pipeline"]["arguments"]
        assert close["references"] == [{"from_previous_result": "slice@close"}]
        assert close["audio"] == "previous_result:slice@close.audio"

    def test_the_same_list_means_equal_not_identical(self):
        expanded = expand_for_each(
            definition(
                {"name": "slice", "for_each": self.shots(), "task": {}},
                {
                    "name": "shot",
                    "for_each": self.shots(),
                    "task": {"arguments": {"a": "previous_result:slice"}},
                },
            )
        )
        assert expanded["steps"][2]["task"]["arguments"]["a"] == (
            "previous_result:slice@open"
        )

    def test_a_sibling_over_a_different_list_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "slice", "for_each": ["a", "b"], "task": {}},
                    {
                        "name": "shot",
                        "for_each": ["a"],
                        "task": {"arguments": {"x": "previous_result:slice"}},
                    },
                )
            )
        assert "different list" in str(e.value)

    def test_a_step_referencing_its_own_group_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": ["a", "b"],
                        "task": {"arguments": {"x": "previous_result:shot"}},
                    }
                )
            )
        assert "its own" in str(e.value)

    def test_a_gather_inside_a_member_still_gathers(self):
        expanded = expand_for_each(
            definition(
                {"name": "slice", "for_each": ["a", "b"], "task": {}},
                {
                    "name": "shot",
                    "for_each": ["x"],
                    "task": {"arguments": {"all": "gather:slice"}},
                },
            )
        )
        assert expanded["steps"][2]["task"]["arguments"]["all"] == [
            "previous_result:slice@0",
            "previous_result:slice@1",
        ]
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_for_each.py -q`
Expected: the new classes PASS against the Task 1 module. If any fails, fix `_gather` / `_rewrite_reference` in `dw/for_each.py` until they pass - the tests are the specification, the code is not.

- [ ] **Step 3: Commit**

```bash
git add dw/for_each.py tests/test_for_each.py
git commit -m "test(for_each): gather:, same-key siblings and the directed group errors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The two templates, expanded back to what they contain today

This is the test that proves stage 2 is mechanical. It writes each template's shot section the list-driven way *inside the test* and asserts the expansion equals the hand-written steps that exist now.

**Files:**
- Test: `tests/test_for_each.py`
- Read: `workflows/templates/minimax/music-video.json`, `workflows/templates/minimax/dialogue-short.json`

**Interfaces:**
- Consumes: `expand_for_each`.
- Produces: nothing new; a regression guard for stage 2.

- [ ] **Step 1: Write the test for `music-video`**

Append to `tests/test_for_each.py`:

```python
import json
import os

TEMPLATES = os.path.join(
    os.path.dirname(__file__), "..", "workflows", "templates", "minimax"
)


def load_template(name):
    with open(os.path.join(TEMPLATES, name)) as f:
        return json.load(f)


def steps_by_name(definition):
    return {s["name"]: s for s in definition["steps"]}


def without_pipeline_reference(step, pipeline_step):
    """A hand-written 'pipeline_reference' shot as the full pipeline block
    the expansion produces: the reference's arguments over the referenced
    step's pipeline."""
    rebuilt = {k: v for k, v in step.items() if k != "pipeline_reference"}
    pipeline = copy.deepcopy(pipeline_step["pipeline"])
    pipeline["arguments"] = step["pipeline_reference"]["arguments"]
    rebuilt["pipeline"] = pipeline
    return rebuilt


class TestMusicVideoTemplate:
    """music-video's four slices and four shots, written as two for_each
    groups over one 'shots' list, expand to the steps the template holds
    by hand today."""

    def test_the_hand_written_shots_are_what_the_list_expands_to(self):
        template = load_template("music-video.json")
        today = steps_by_name(template)
        shots = [
            {"name": "wide_open", "prompt": "variable:shot_1_wide_open", "start_frame": 0},
            {"name": "closeup", "prompt": "variable:shot_2_closeup", "start_frame": 124},
            {"name": "room", "prompt": "variable:shot_3_room", "start_frame": 248},
            {"name": "finale", "prompt": "variable:shot_4_finale", "start_frame": 372},
        ]
        slice_template = copy.deepcopy(today["slice_1"])
        slice_template["name"] = "slice"
        slice_template["for_each"] = shots
        slice_template["task"]["arguments"]["start_frame"] = "item:start_frame"

        shot_template = copy.deepcopy(today["shot_1_wide_open"])
        shot_template["name"] = "shot"
        shot_template["for_each"] = shots
        shot_template["pipeline"]["arguments"]["prompt"] = "item:prompt"
        for reference in shot_template["pipeline"]["arguments"]["references"]:
            if reference.get("from_previous_result") == "slice_1":
                reference["from_previous_result"] = "slice"

        edit = copy.deepcopy(today["edit"])
        edit["task"]["arguments"]["videos"] = "gather:shot"

        expanded = expand_for_each(
            definition(
                today["draw_singer"], today["write_song"], slice_template,
                today["soundtrack"], shot_template, edit, today["music_video"],
            )
        )
        got = steps_by_name(expanded)

        # Each expanded slice is today's slice with the new name
        for key, old in zip(["wide_open", "closeup", "room", "finale"], range(1, 5)):
            expected = copy.deepcopy(today[f"slice_{old}"])
            expected["name"] = f"slice@{key}"
            assert got[f"slice@{key}"] == expected

        # Each expanded shot is today's shot (as a full pipeline block) with
        # the new name and its slice renamed
        hand_written = ["shot_1_wide_open", "shot_2_closeup", "shot_3_room", "shot_4_finale"]
        for key, old, index in zip(["wide_open", "closeup", "room", "finale"], hand_written, range(1, 5)):
            step = today[old]
            if "pipeline_reference" in step:
                step = without_pipeline_reference(step, today["shot_1_wide_open"])
            expected = copy.deepcopy(step)
            expected["name"] = f"shot@{key}"
            for reference in expected["pipeline"]["arguments"]["references"]:
                if reference.get("from_previous_result") == f"slice_{index}":
                    reference["from_previous_result"] = f"slice@{key}"
            assert got[f"shot@{key}"] == expected

        assert got["edit"]["task"]["arguments"]["videos"] == [
            f"previous_result:shot@{k}" for k in ["wide_open", "closeup", "room", "finale"]
        ]
```

- [ ] **Step 2: Run it, and adjust the test to the template's real shape**

Run: `python -m pytest tests/test_for_each.py::TestMusicVideoTemplate -q`

The test was written from the template as of 2026-09-11 (`slice_1..4` with `start_frame` 0/124/248/372; shots 2-4 as `pipeline_reference` to `shot_1_wide_open`; `edit.videos` the four shots). If the assertion fails on a field this plan did not anticipate (a `pipeline_reference` carrying more than `arguments`, say), read the diff pytest prints and fix the *test's* reconstruction so it mirrors the template - the expansion is not to be bent to a template. If it fails because the expansion is wrong, fix `dw/for_each.py`.

Expected after adjustment: PASS

- [ ] **Step 3: Write the test for `dialogue-short`**

Append:

```python
class TestDialogueShortTemplate:
    """dialogue-short's five shots, whose reference lists differ in length
    by shot, written as one for_each group whose entries carry the whole
    references list."""

    def test_the_hand_written_shots_are_what_the_list_expands_to(self):
        template = load_template("dialogue-short.json")
        today = steps_by_name(template)
        hand_written = [
            ("cold_open", "shot_1_cold_open"),
            ("deflect", "shot_2_deflect"),
            ("react", "shot_3_react"),
            ("button", "shot_4_button"),
            ("tag", "shot_5_tag"),
        ]
        first = today["shot_1_cold_open"]
        full = {
            old: (step if "pipeline_reference" not in step else without_pipeline_reference(step, first))
            for old, step in today.items()
            if old.startswith("shot_")
        }
        shots = []
        for key, old in hand_written:
            arguments = full[old]["pipeline"]["arguments"]
            entry = {"name": key, "prompt": arguments["prompt"], "references": arguments["references"]}
            # Only the tag shot has its own frame count in the template
            if arguments.get("num_frames") != first["pipeline"]["arguments"]["num_frames"]:
                entry["num_frames"] = arguments["num_frames"]
            shots.append(entry)

        shot_template = copy.deepcopy(full["shot_1_cold_open"])
        shot_template["name"] = "shot"
        shot_template["for_each"] = shots
        shot_template["pipeline"]["arguments"]["prompt"] = "item:prompt"
        shot_template["pipeline"]["arguments"]["references"] = "item:references"

        expanded = expand_for_each(
            definition(today["draw_character_a"], today["draw_character_b"], shot_template)
        )
        got = steps_by_name(expanded)
        for key, old in hand_written:
            expected = copy.deepcopy(full[old])
            expected["name"] = f"shot@{key}"
            if "num_frames" not in shots[[k for k, _ in hand_written].index(key)]:
                # The member inherits the template's frame count, which is
                # what the hand-written shot has too
                pass
            got_step = got[f"shot@{key}"]
            assert got_step["pipeline"]["arguments"]["prompt"] == expected["pipeline"]["arguments"]["prompt"]
            assert got_step["pipeline"]["arguments"]["references"] == expected["pipeline"]["arguments"]["references"]
```

Then read `dialogue-short.json` (`python3 -c "import json; d=json.load(open('workflows/templates/minimax/dialogue-short.json')); [print(s['name'], json.dumps(s)[:600]) for s in d['steps']]"`) and extend the entry to carry every argument that differs between shots (the template's `num_frames` vs `tag_num_frames` is one; there may be others), mapping each to an `item:` in `shot_template`. The final assertion should compare the *whole* step: `assert got[f"shot@{key}"] == expected`, with `expected["name"]` set as above. Replace the two field assertions with that once the entry shape is right.

- [ ] **Step 4: Run both template tests**

Run: `python -m pytest tests/test_for_each.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_for_each.py
git commit -m "test(for_each): music-video and dialogue-short expand back to their hand-written steps

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Schema - `for_each` on a step

**Files:**
- Modify: `dw/workflow_schema.json` (the `step` definition, around line 100-135)
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: a definition with `"for_each"` on a step validates; `for_each` accepts a list or a `variable:` string.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py`:

```python
def test_for_each_is_a_list_or_a_variable_reference():
    schema = load_schema("workflow")
    base = {
        "id": "t",
        "steps": [{"name": "shot", "task": {"command": "x"}, "for_each": None}],
    }
    for good in (["a", "b"], [{"name": "a"}], "variable:shots"):
        base["steps"][0]["for_each"] = good
        assert validate_data_all(base, schema) == []
    for bad in (4, "shots", {"a": 1}):
        base["steps"][0]["for_each"] = bad
        assert validate_data_all(base, schema) != []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_schema.py::test_for_each_is_a_list_or_a_variable_reference -q`
Expected: FAIL (the `4` and `"shots"` cases validate today because the step allows unknown keys - check: if the test passes outright, the step has no `additionalProperties: false`; the test's `bad` loop is then the failing half).

- [ ] **Step 3: Add the property**

In `dw/workflow_schema.json`, inside `"step": {"properties": {...}}`, after `"name"`:

```json
"for_each": {
    "description": "Run this step once per entry of a list. A list, or a 'variable:' reference to one. Inside the step, 'item:' is the entry and 'item:field' one of its fields; a later step reads every member's result with 'gather:<step name>'. Members are named '<step name>@<entry name>' (or '@<index>' for an entry without a name), so '@' is reserved in step names.",
    "type": ["array", "string"],
    "pattern": "^variable:",
    "maxItems": 32
},
```

`"pattern"` only applies to strings and `"maxItems"` only to arrays, which is how `seed` already mixes the two on the same key.

- [ ] **Step 4: Run the schema tests**

Run: `python -m pytest tests/test_schema.py tests/test_configuration_schema.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dw/workflow_schema.json tests/test_schema.py
git commit -m "feat(schema): for_each on a step

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Wire the pass into `Workflow.run` and `validation_errors`

**Files:**
- Modify: `dw/workflow.py:286-296` (`validation_errors`), `dw/workflow.py:370-386` (the substitution block in `run`)
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: `expand_for_each`, `ForEachError` from Task 1; `replace_variables`, `set_variables`, `VariableNotFoundError` from `dw/variables.py`; `argument_errors` from `dw/variables.py`.
- Produces: `Workflow.validation_errors(self, arguments=None)` - the new keyword is what Task 6 passes; `Workflow.expanded_definition(self, arguments=None) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_workflow.py` (it already imports `Workflow`/`workflow_from_definition`-style helpers - follow the file's existing pattern for constructing a workflow from a dict, e.g. the one `test_validate_catches_a_reference_to_a_step_that_does_not_exist` at line 583 uses):

```python
def _for_each_workflow(**overrides):
    definition = {
        "id": "fe",
        "variables": {"shots": [{"name": "a", "text": "A"}, {"name": "b", "text": "B"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"command": "compose_text", "arguments": {"parts": ["item:text"]}},
                "result": {"content_type": "text/plain"},
            },
            {
                "name": "edit",
                "task": {"command": "compose_text", "arguments": {"parts": "gather:shot"}},
                "result": {"content_type": "text/plain"},
            },
        ],
    }
    definition.update(overrides)
    return definition


def test_validation_expands_for_each_before_the_reference_check(tmp_path):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)  # the file's helper
    assert workflow.validation_errors() == []


def test_validation_reports_a_for_each_error_at_its_path(tmp_path):
    definition = _for_each_workflow()
    definition["steps"][1]["task"]["arguments"]["parts"] = "previous_result:shot"
    workflow = _workflow_from(definition, tmp_path)
    errors = workflow.validation_errors()
    assert len(errors) == 1
    assert errors[0]["path"] == "steps[1].task.arguments.parts"
    assert "gather:shot" in errors[0]["message"]


def test_validation_expands_the_callers_list_not_the_default(tmp_path):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)
    # Two entries share a name only in the caller's list
    errors = workflow.validation_errors(
        arguments={"shots": [{"name": "a"}, {"name": "a"}]}
    )
    assert errors and "Duplicate entry name 'a'" in errors[0]["message"]


def test_validation_falls_back_to_the_default_when_the_arguments_are_bad(tmp_path):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)
    # An undeclared argument is argument_errors' finding, not validation's
    assert workflow.validation_errors(arguments={"nope": 1}) == []


def test_validation_of_an_undeclared_variable_still_does_not_raise(tmp_path):
    definition = _for_each_workflow()
    definition["steps"][0]["for_each"] = "variable:missing"
    workflow = _workflow_from(definition, tmp_path)
    errors = workflow.validation_errors()
    assert errors and "variable:missing" in errors[0]["message"]


def test_run_expands_for_each_and_names_the_members(tmp_path):
    workflow = _workflow_from(_for_each_workflow(seed=1), tmp_path)
    workflow.run({})
    names = [entry["step"] for entry in workflow.manifest]
    assert names == ["shot@a", "shot@b", "edit"]


def test_run_substitutes_the_callers_list(tmp_path):
    workflow = _workflow_from(_for_each_workflow(seed=1), tmp_path)
    workflow.run({"shots": [{"name": "only", "text": "X"}]})
    names = [entry["step"] for entry in workflow.manifest]
    assert names == ["shot@only", "edit"]
```

Check the manifest entry's key for the step name (`grep -n "manifest.append" dw/workflow.py`) and use whatever key it carries in place of `entry["step"]`.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_workflow.py -k "for_each" -q`
Expected: FAIL - `validation_errors()` reports `gather:shot`/`item:` nothing but the run fails on `for_each` as an unknown step key, and `validation_errors(arguments=...)` is a TypeError.

- [ ] **Step 3: Implement**

In `dw/workflow.py` add the imports:

```python
from .for_each import expand_for_each, ForEachError
from .variables import (
    argument_errors,
    replace_variables,
    set_variables,
    VariableNotFoundError,
)
```

(merge with the existing `from .variables import replace_variables, set_variables` at line 45.)

Replace `validation_errors`:

```python
    def expanded_definition(self, arguments=None):
        """The definition as the run will see it: variables substituted -
        the caller's `arguments` folded in when they are all good, else the
        declared defaults - and every for_each step expanded.

        Raises ForEachError for a for_each that cannot be expanded. A
        'variable:' that names nothing is left in place rather than raised:
        validate_workflow already reports that as a warning, and the
        reference check is happy to skip a reference it cannot read.
        """
        definition = copy.deepcopy(self.workflow_definition)
        variables = definition.get("variables")
        if isinstance(variables, dict):
            if arguments and not argument_errors(definition, arguments):
                set_variables(arguments, variables)
            try:
                definition = replace_variables(definition, variables)
            except VariableNotFoundError:
                pass
        return expand_for_each(definition)

    def validation_errors(self, arguments=None):
        """Every schema violation in the definition, as [{path, message}];
        empty when it validates. `arguments` are the caller's, so a
        for_each over a list the caller supplies is checked as it will run."""
        errors = validate_data_all(self.workflow_definition, load_schema("workflow"))
        # Only once the shape is known good: the passes below walk the
        # steps array and a definition that fails the schema may have no
        # such array to walk
        if errors:
            return errors
        try:
            expanded = self.expanded_definition(arguments)
        except ForEachError as e:
            return [{"path": e.path, "message": str(e)}]
        return previous_result_reference_errors(expanded)
```

`set_variables` may need `realize_constants` first when a default is `constant:` - it does not for the list case, and `argument_errors` already runs `set_variables` on the raw declared block, so this mirrors it.

In `run`, directly after `workflow_def = replace_variables(workflow_def, variables)` (line 386) and *outside* the `if variables is not None` block:

```python
            # One ordinary step per entry of every for_each list, before the
            # seed, the run id and the realized workflow are computed, so
            # each covers what actually runs. A ForEachError here fails the
            # run before anything loads
            workflow_def = expand_for_each(workflow_def)
```

Note: `steps = workflow_def["steps"]` (or wherever the loop reads them) must be read *after* this line - check with `grep -n "steps = " dw/workflow.py` and move the read below the expansion if it sits above.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_workflow.py tests/test_for_each.py tests/test_previous_results.py -q`
Expected: PASS. The `previous_result_reference_errors` behaviour changed subtly - it now runs on the *substituted* definition, so a `from_previous_result: "variable:x"` whose variable is declared is now checked by its resolved value. If an existing test pins the old skip, read it: the skip was for a value that "is not knowable before substitution", which a declared default now is. Update the test's expectation, not the code.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q -x`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dw/workflow.py tests/test_workflow.py
git commit -m "feat(for_each): expand in Workflow.run and validate the expanded definition

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The server pre-flight validates with the caller's list

**Files:**
- Modify: `dw/server/app.py:1270` (the `errors = candidate.validation_errors()` call in `validate_workflow`)
- Test: `tests/test_validate_arguments.py`

**Interfaces:**
- Consumes: `Workflow.validation_errors(arguments=...)` from Task 5.

- [ ] **Step 1: Write the failing test**

Look at how `tests/test_validate_arguments.py` builds its client and posts to `/api/validate` (`grep -n "def test\|client.post" tests/test_validate_arguments.py | head`), then append in the same style:

```python
def test_validate_expands_for_each_with_the_callers_list(client):
    workflow = {
        "id": "fe",
        "variables": {"shots": [{"name": "a", "text": "A"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"command": "compose_text", "arguments": {"parts": ["item:text"]}},
                "result": {"content_type": "text/plain"},
            },
            {
                "name": "edit",
                "task": {"command": "compose_text", "arguments": {"parts": "gather:shot"}},
                "result": {"content_type": "text/plain"},
            },
        ],
    }
    ok = client.post("/api/validate", json={"workflow": workflow}).json()
    assert ok["valid"] is True

    bad = client.post(
        "/api/validate",
        json={"workflow": workflow, "arguments": {"shots": [{"name": "x"}, {"name": "x"}]}},
    ).json()
    assert bad["valid"] is False
    assert bad["errors"][0]["path"] == "steps[0].for_each[1].name"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_validate_arguments.py -k for_each -q`
Expected: FAIL - the second call answers `valid: True` because the default list was expanded.

- [ ] **Step 3: Pass the arguments through**

In `dw/server/app.py`, change

```python
            errors = candidate.validation_errors()
```

to

```python
            # The caller's list is the one a for_each expands over, so the
            # pre-flight checks the step set that will actually run
            errors = candidate.validation_errors(arguments=request.arguments)
```

- [ ] **Step 4: Run the server tests**

Run: `python -m pytest tests/test_validate_arguments.py tests/test_server_jobs.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_validate_arguments.py
git commit -m "feat(server): POST /api/validate expands for_each over the caller's list

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Documentation - the agent-facing conventions

This is how an agent learns the feature. `get_guide("workflows", section=...)` serves `docs/WORKFLOW_GUIDE.md` by heading, and the MCP instructions send an agent to "Authoring a workflow from an agent" before it writes JSON; CLAUDE.md's Type System block is the mirror the repo keeps in sync by rule.

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` (the `### References` list at line 244, and a new subsection after `### Several \`previous_result\` references multiply`, line 318)
- Modify: `CLAUDE.md` (the Type System bullets, after the `prompt:` bullet at line 152; the Critical Gotchas list)
- Modify: `docs/SERVER.md` (where the manifest is described - `grep -n "manifest" docs/SERVER.md`)
- Test: `tests/test_server_guides.py` / `tests/test_mcp_guides.py` only if they pin section names (`grep -n "Authoring\|sections" tests/test_server_guides.py`); nothing to add otherwise.

- [ ] **Step 1: Add the three prefixes to the References list in `docs/WORKFLOW_GUIDE.md`**

After the `prompt:` bullet (ends "rather than resolving twice."), add:

```markdown
- `item:` — only inside a step that carries `for_each`: `item:` is the
  entry the member was made for, `item:field` one field of an object entry,
  spliced in whole whatever its type — a string, a number, a list of
  references. See "One step per entry" below.
- `gather:` — `gather:shot` is the result of *every* member of the
  `for_each` step `shot`, in list order, as one list. Inside a list it splices
  into it. It is how a step downstream of a fan-out reads the whole group;
  `previous_result:shot` naming a `for_each` step is an error that says so.
```

- [ ] **Step 2: Add the subsection**

After the paragraph ending "the signal to restructure the workflow, not to add another reference." and before `### The loop`, add:

```markdown
### One step per entry: `for_each`

A step that carries `for_each` runs once per entry of a list — a shot per
entry of `shots` — and the list is a variable the caller supplies, so a
six-shot episode is an argument rather than a different file.

```json
{
  "name": "shot",
  "for_each": "variable:shots",
  "pipeline": {
    "arguments": {
      "prompt": "item:prompt",
      "references": "item:references"
    }
  }
}
```

with

```json
"shots": [
  { "name": "wide_open", "prompt": "the band walks on, wide",
    "references": [{ "reference_type": "…", "from_previous_result": "draw_singer" }] },
  { "name": "closeup", "prompt": "closeup on the singer",
    "references": [{ "reference_type": "…", "from_previous_result": "draw_singer" }] }
]
```

and downstream

```json
{ "name": "edit",
  "task": { "command": "concat_videos", "arguments": { "videos": "gather:shot" } } }
```

Before the run starts, the engine replaces the `for_each` step with one
ordinary step per entry, named `shot@wide_open`, `shot@closeup` — the
entry's `name`, or its index for an entry without one. Those are the names
the manifest, the job's events and the gallery show, and `@` is reserved
for them: a hand-written step name may not contain it. An entry's `name`
must be unique in its list and match `^[a-zA-Z_][a-zA-Z0-9_-]*$`. Give
entries names: the step cache keys on the member name, so a shot inserted
in the middle of a named list leaves every other shot cached, while an
indexed list shifts every later shot onto a different entry and regenerates
it.

`item:field` is the whole value of that field, so an entry can carry
anything a step argument can — including a `references` list whose length
differs by shot, with `from_previous_result` and `asset:` strings inside
it. Nothing is interpolated: `"item:prompt"` is the field, `"shot: item:prompt"`
is a literal string.

Two `for_each` steps over the *same* list are paired by name: inside
`shot@closeup`, a reference to another `for_each` step `slice` over the same
`shots` list resolves to `slice@closeup`. That is how a shot reads the audio
slice cut for it when slicing and generating are two steps. It is the one
pairing the engine has; `for_each` runs over exactly one list, and there is
no zip and no loop index.

Limits: a list has at most 32 entries. `release_pipeline` on a `for_each`
step releases after the *last* member. Each entry is a full generation, so
quote `cost × len(list)` before running a list-driven workflow, and
`validate_workflow` with the `arguments` you will run with: it expands your
list, not the template's default, and reports a duplicate name or a missing
field at the entry's path.
```

- [ ] **Step 3: Mirror in `CLAUDE.md`**

After the `prompt:` bullet in Type System (line 152-156), add:

```markdown
- A step carrying `for_each` (a list, or `variable:` naming one) is expanded by
  `expand_for_each` (`dw/for_each.py`) into one ordinary step per entry, named
  `<step>@<entry name or index>`, immediately after `replace_variables` in
  `Workflow.run` and, with the caller's arguments folded, in `validation_errors`.
  Inside a member `item:` / `item:field` is the entry (any type, spliced whole);
  a later step reads the group with `gather:<step>` (a list; splices inside a
  list); two groups over the same list pair by key (`slice` inside `shot@x` is
  `slice@x`). `previous_result:` naming a group is a directed error. `@` is
  reserved in step names; entry names are validated and unique; 32 entries max;
  `release_pipeline`/`release_models` survive on the last member only. The
  realized workflow keeps `for_each`; the manifest names the members
```

And in Critical Gotchas, after the `previous_result:` static-check bullet:

```markdown
- **`for_each` expands before the reference check** — `validation_errors` substitutes
  (the caller's `arguments` when they are all good, else the defaults) and expands
  first, so `gather:` and `item:` errors carry the path of the template step
  (`steps[0].for_each[1].name`) while a bad reference inside a member carries the
  member's. Since substitution now precedes the check, a `from_previous_result`
  spelled by a *declared* variable is checked by its value
```

- [ ] **Step 4: The manifest note in `docs/SERVER.md`**

Find the manifest description (`grep -n "manifest" docs/SERVER.md`) and add one sentence where the manifest's step entries are described:

```markdown
A `for_each` step appears in the manifest as its members (`shot@wide_open`,
`shot@closeup`), because the manifest records what ran; the run's
`workflow.json` keeps the `for_each` form, because it records what was asked.
```

- [ ] **Step 5: Run the guide tests**

Run: `python -m pytest tests/test_server_guides.py tests/test_mcp_guides.py tests/test_plugin_skills.py -q`
Expected: PASS (the guide is served by heading; a new heading is a new section, nothing pins the count).

- [ ] **Step 6: Commit**

```bash
git add docs/WORKFLOW_GUIDE.md CLAUDE.md docs/SERVER.md
git commit -m "docs(for_each): the item:/gather:/for_each conventions for agents, CLAUDE.md mirror, manifest note

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: End-to-end on the server, then mark the proposal

**Files:**
- Modify: `docs/proposals/list-driven-steps.md` (status line)
- Test: `tests/test_server_jobs.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write a job-level test**

In `tests/test_server_jobs.py`, following the file's pattern for submitting an inline workflow and waiting for it (`grep -n "def test.*job\|/api/jobs" tests/test_server_jobs.py | head`), add a test that submits the `_for_each_workflow` from Task 5 (copy the dict - do not import from another test module) with `"arguments": {"shots": [{"name": "one", "text": "1"}, {"name": "two", "text": "2"}, {"name": "three", "text": "3"}]}` and asserts the finished job's manifest step names are `["shot@one", "shot@two", "shot@three", "edit"]` and that `edit`'s text output is `"123"` (or however `compose_text` joins - check `dw/tasks/` for its default separator and assert the actual value).

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_server_jobs.py -k for_each -q`
Expected: PASS. If the worker path fails where the direct `Workflow.run` test passed, the difference is the spawned worker's own validation (`dw/worker.py:185`, `workflow.validate()` with no arguments) - that call validates against the defaults, which is fine, because the run itself then substitutes and expands the real arguments and any ForEachError fails the job with its message.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests -q`
Expected: PASS, no skips added.

- [ ] **Step 4: Update the proposal status**

In `docs/proposals/list-driven-steps.md`, change the status line to:

```markdown
Status: **stage 1 implemented** (expansion pass, validation, schema, docs -
`dw/for_each.py`); stages 2 and 3 (templates, catalog cost and entry shape)
not started. Written for MCP feedback ticket T003.
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_server_jobs.py docs/proposals/list-driven-steps.md
git commit -m "test(for_each): a list-driven job end to end; proposal marks stage 1 done

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:** expansion pass (T1), `item:` structured values (T1), same-key siblings (T2), `gather:` splice + errors (T2), directed errors (T2), `@` reserved + entry-name validation (T1), release flags on last member (T1), ceiling 32 (T1), schema (T4), validation ordering with folded arguments (T5, T6), realized workflow keeps `for_each` (no change needed - `realize_workflow` reads `self.workflow_definition`, which the pass never touches; stated in Global Constraints), manifest naming documented (T7), the `realize_args` check on entry objects (verified during planning: `resolve_path_references` in `dw/arguments.py:221` walks lists but leaves dicts alone, so an `asset:` inside an entry object stays a string until the member step realizes it - no exemption needed), templates expand back (T3), agent docs (T7). Cost/entry-shape in the catalog and the template rewrite are stages 2-3, deliberately out of this plan.

**Placeholders:** Task 3's `dialogue-short` test asks the implementer to read the template and extend the entry - that is deliberate, because the entry shape is exactly what stage 2 has to get right and the template's argument differences are the ground truth; the assertion to converge on (`got[...] == expected`) is stated. Task 5 names the file's workflow-construction helper generically (`_workflow_from`) - use the helper `test_validate_catches_a_reference_to_a_step_that_does_not_exist` uses.

**Type consistency:** `expand_for_each(definition) -> dict`, `ForEachError(path, message)` with `.path`, `member_name(group, key)`, `MAX_FOR_EACH_ENTRIES` are used identically across Tasks 1-6; `validation_errors(arguments=None)` is defined in Task 5 and called in Task 6.
