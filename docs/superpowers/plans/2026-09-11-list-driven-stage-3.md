# List-driven steps stage 3 (cost and entry shape) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tell an agent composing over MCP what an entry of a list-driven template must contain and what a run over its own list costs, and settle the three engine questions stage 2 left open.

**Architecture:** The fields an entry needs are derived from the `item:` references the `for_each` steps make (`list_fields` in `dw/for_each.py`), and surface as a `lists` block in the catalog listing and as an unknown-key warning in validation — one computation used twice. Cost stays measured: a `cost` entry gains an optional `per_entry` block, passed through the listing unchanged; the skill and the guide quote `fixed + per_entry × N`. The variables endpoint previews inside list entries. Rulings: an empty `for_each` list is an error, `expanded_definition` realizes constants, the step cache bound rises to 128. The editor's flow view learns `gather:` and `from_previous_result` edges.

**Tech Stack:** Python 3.10, pytest, JSON schema, FastAPI routes, TypeScript + vitest for `ui/`.

**Spec:** `docs/proposals/list-driven-steps.md` — "Stage 3: cost and entry shape (design)".

## Global Constraints

- Full suite green before every commit that touches `dw/`, `dw_mcp/` or `tests/`: `python -m pytest -q -x tests/` (baseline on `develop` at 9cc4e7c: 3555 passed, 5 skipped). `black --check dw dw_mcp tests` clean (CI runs it).
- Catalog rules from `docs/superpowers/specs/2026-09-06-agent-catalog-legibility-design.md`: cost is measured, never derived; structure is derived from the raw definition, never declared twice. `derive_catalog_metadata` reads raw JSON only — no variable substitution, no type loading.
- The catalog pins for the two templates hold unchanged (`tests/test_catalog_structure.py` `EXPECTED_SHAPES`, `COSTED`). No template gets a `per_entry` block in this plan — the numbers come from a GPU run the user makes later; the code and tests must be correct with `per_entry` absent.
- `plugins/dw/skills/minimax-h3/SKILL.md` stays ≤ 12288 bytes (it is at 12063).
- Security rules in CLAUDE.md. No new filesystem access outside the validators.
- `ui/` changes pass `npm run check`, `npm run lint`, `npm run format` (prettier writes; commit the result) and `npm test` from `ui/`.
- Commit trailer on every commit: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Docs that must move together (CLAUDE.md says so): the Type System bullets in `CLAUDE.md` and the "Authoring a workflow from an agent" section of `docs/WORKFLOW_GUIDE.md`.

---

### Task 1: `list_fields` and the catalog's `lists` block

**Files:**
- Modify: `dw/for_each.py` (add `list_fields`)
- Modify: `dw/server/catalog_shape.py` (`derive_catalog_metadata` returns `lists`; `COMPACT_FIELDS` gains `"lists"`; `project_listing` drops an empty `lists` in the compact view)
- Modify: `dw/server/app.py:194-262` (`workflow_details` carries `lists`; the fallback detail has `"lists": {}`)
- Test: `tests/test_for_each.py`, `tests/test_catalog_shape.py`, `tests/test_server.py` (near `test_workflow_details_name_their_variables`, ~line 2997)

**Interfaces:**
- Produces: `list_fields(definition) -> dict` in `dw/for_each.py`. For every step whose `for_each` is a string `variable:<name>`, keyed by `<name>`: `{"fields": [...] | None, "steps": [step names in order]}`. `fields` is the sorted union of `<field>` over every `item:<field>` string found anywhere in those steps, with `"name"` first; when any of those steps uses bare `item:` (the whole entry) `fields` is `None`. A step whose `for_each` is a literal list is not a list argument and is skipped. Non-dict steps and a missing/malformed `steps` yield `{}`. Reads raw JSON, no substitution.
- Produces: `derive_catalog_metadata(definition)["lists"]` — `list_fields` plus `"entries": len(default)` when `variables[name]` is a list, else `None`. Listing entries carry `lists`; compact view includes it only when non-empty.

- [ ] **Step 1: Write the failing tests for `list_fields`**

Append to `tests/test_for_each.py`:

```python
class TestListFields:
    """What an entry has to carry is read off the item: references the
    for_each steps make - the catalog and the unknown-key warning both
    derive from this."""

    def test_fields_are_the_item_references_with_name_first(self):
        fields = list_fields(
            definition(
                {
                    "name": "slice",
                    "for_each": "variable:shots",
                    "task": {"arguments": {"start_frame": "item:start_frame"}},
                },
                {
                    "name": "shot",
                    "for_each": "variable:shots",
                    "pipeline": {
                        "arguments": {
                            "prompt": "item:prompt",
                            "references": [{"x": "item:references"}],
                        }
                    },
                },
            )
        )
        assert fields == {
            "shots": {
                "fields": ["name", "prompt", "references", "start_frame"],
                "steps": ["slice", "shot"],
            }
        }

    def test_a_bare_item_means_the_entry_is_a_value(self):
        fields = list_fields(
            definition(
                {"name": "say", "for_each": "variable:lines", "task": {"arguments": {"text": "item:"}}}
            )
        )
        assert fields == {"lines": {"fields": None, "steps": ["say"]}}

    def test_a_literal_list_is_not_an_argument(self):
        assert list_fields(definition({"name": "s", "for_each": ["a"], "task": {}})) == {}

    def test_a_step_reading_no_field_still_lists_name(self):
        fields = list_fields(
            definition({"name": "s", "for_each": "variable:xs", "task": {"arguments": {}}})
        )
        assert fields == {"xs": {"fields": ["name"], "steps": ["s"]}}

    def test_malformed_definitions_yield_nothing(self):
        assert list_fields({}) == {}
        assert list_fields({"steps": "nope"}) == {}
        assert list_fields({"steps": ["not a dict"]}) == {}
```

Add `list_fields` to the `from dw.for_each import (...)` at the top of the file.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q tests/test_for_each.py -k ListFields`
Expected: ImportError on `list_fields`.

- [ ] **Step 3: Implement `list_fields`**

In `dw/for_each.py`, after `render_path`:

```python
def list_fields(definition):
    """What an entry of each list-driven variable has to carry, read off
    the definition: for every step whose for_each is 'variable:<name>',
    the fields its 'item:<field>' references name.

    Returns {<variable>: {"fields": [...] or None, "steps": [...]}} -
    fields sorted with 'name' first, or None when a step splices the
    whole entry with a bare 'item:' (the entries are values, not
    objects). A literal for_each list is not an argument and is skipped.
    Reads the raw definition, no substitution, so the catalog and the
    validator derive the same answer from the file as written.
    """
    steps = definition.get("steps") if isinstance(definition, dict) else None
    if not isinstance(steps, list):
        return {}
    found = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        target = step.get(FOR_EACH_KEY)
        if not (isinstance(target, str) and target.startswith("variable:")):
            continue
        variable = target.removeprefix("variable:")
        entry = found.setdefault(variable, {"fields": set(), "steps": []})
        entry["steps"].append(step.get("name"))
        for value in _strings(step):
            if not value.startswith(ITEM_PREFIX):
                continue
            field = value[len(ITEM_PREFIX) :]
            if field == "":
                entry["fields"] = None
            elif entry["fields"] is not None:
                entry["fields"].add(field)
    return {
        variable: {
            "fields": (
                None
                if entry["fields"] is None
                else ["name"] + sorted(entry["fields"] - {"name"})
            ),
            "steps": entry["steps"],
        }
        for variable, entry in found.items()
    }


def _strings(value):
    """Every string anywhere inside a JSON value, except the for_each key
    itself."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != FOR_EACH_KEY:
                yield from _strings(item)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_for_each.py -k ListFields`
Expected: PASS.

- [ ] **Step 5: Write the failing catalog tests**

Append to `tests/test_catalog_shape.py`:

```python
def for_each_step(name, variable, arguments):
    return {
        "name": name,
        "for_each": f"variable:{variable}",
        "pipeline": {
            "configuration": {"component_type": "{Fake}"},
            "arguments": arguments,
        },
        "result": {"content_type": "video/mp4"},
    }


def test_lists_name_the_fields_an_entry_takes_and_the_default_length():
    meta = derive_catalog_metadata(
        definition(
            for_each_step("shot", "shots", {"prompt": "item:prompt", "n": "item:num_frames"}),
            variables={"shots": [{"name": "a", "prompt": "p", "num_frames": 1}] * 3},
        )
    )
    assert meta["lists"] == {
        "shots": {
            "fields": ["name", "num_frames", "prompt"],
            "steps": ["shot"],
            "entries": 3,
        }
    }


def test_lists_is_empty_without_for_each_and_entries_is_none_without_a_list_default():
    assert derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg")))["lists"] == {}
    meta = derive_catalog_metadata(
        definition(for_each_step("shot", "shots", {"prompt": "item:prompt"}))
    )
    assert meta["lists"]["shots"]["entries"] is None


def test_compact_carries_lists_only_when_there_are_any():
    listing = {
        "plain": entry("image"),
        "cut": entry(
            "sequence",
            lists={"shots": {"fields": ["name", "prompt"], "steps": ["shot"], "entries": 2}},
        ),
    }
    compact = project_listing(listing, view="compact")
    assert "lists" not in compact["plain"]
    assert compact["cut"]["lists"]["shots"]["fields"] == ["name", "prompt"]
```

Change the `entry()` helper to include `"lists": {}` in its base dict, and change `test_compact_drops_prose_and_model_configs_and_keeps_user_workflows` to `assert set(compact["templates/tti"]) == set(COMPACT_FIELDS) - {"configures", "lists"}`.

Append to `tests/test_server.py` next to `test_workflow_details_name_their_variables` (read that test first and reuse its fixtures; it writes a workflow into the server's workflow dir and reads `GET /api/workflows`):

```python
def test_workflow_details_describe_their_lists(server):
    """A list-driven workflow's listing says what an entry carries, so an
    agent can write the list without opening the definition."""
    client, workflow_dir = server  # match the fixture shape the neighbour uses
    write_workflow(
        workflow_dir,
        "cut.json",
        {
            "id": "cut",
            "variables": {"shots": [{"name": "a", "prompt": "p"}, {"name": "b", "prompt": "q"}]},
            "steps": [
                {
                    "name": "shot",
                    "for_each": "variable:shots",
                    "task": {"command": "noop", "arguments": {"text": "item:prompt"}},
                }
            ],
        },
    )
    details = client.get("/api/workflows").json()["details"]["cut"]
    assert details["lists"] == {
        "shots": {"fields": ["name", "prompt"], "steps": ["shot"], "entries": 2}
    }
    compact = client.get("/api/workflows", params={"view": "compact"}).json()
    assert compact["details"]["cut"]["lists"]["shots"]["entries"] == 2
```

Adapt the fixture usage and the response shape to what the neighbouring test actually does — read it before writing; the assertions above are the contract, the plumbing is the file's.

- [ ] **Step 6: Run to verify they fail**

Run: `python -m pytest -q tests/test_catalog_shape.py tests/test_server.py -k "lists"`
Expected: KeyError on `lists`.

- [ ] **Step 7: Implement the catalog side**

`dw/server/catalog_shape.py`: import `from ..for_each import list_fields` (it is a pure function; the module docstring's "reads the raw JSON only" still holds). In `derive_catalog_metadata`, before the return:

```python
    variables = definition.get("variables")
    variables = variables if isinstance(variables, dict) else {}
    lists = {
        name: {
            **fields,
            "entries": (
                len(variables[name]) if isinstance(variables.get(name), list) else None
            ),
        }
        for name, fields in list_fields(definition).items()
    }
```

and add `"lists": lists` to the returned dict; update the docstring ("Returns {shape, traits, summary, summary_truncated, lists, declared}" and one sentence on `lists`).

`COMPACT_FIELDS` gains `"lists"` after `"variable_names"`. In `project_listing`, replace the `configures` special case with a set:

```python
# Carried in the compact view only when set: a template has no
# `configures`, and most workflows have no list-driven step
_COMPACT_WHEN_SET = frozenset({"configures", "lists"})
...
            slim = {
                key: detail.get(key)
                for key in COMPACT_FIELDS
                if key not in _COMPACT_WHEN_SET or detail.get(key)
            }
```

`dw/server/app.py` `workflow_details`: add `"lists": metadata["lists"],` to `detail` and `"lists": {},` to the fallback detail. Also update the docstring of `workflow_details` (one clause: "and, for a list-driven workflow, what an entry of each list carries").

- [ ] **Step 8: Run the tests, then the full suite**

Run: `python -m pytest -q tests/test_catalog_shape.py tests/test_server.py tests/test_for_each.py tests/test_mcp_catalog.py` then `python -m pytest -q -x tests/`
Expected: PASS. If a test elsewhere pins the exact key set of a listing entry or of `COMPACT_FIELDS`, update it to include `lists` and say so in the report.

- [ ] **Step 9: Commit**

```bash
git add dw/for_each.py dw/server/catalog_shape.py dw/server/app.py tests/test_for_each.py tests/test_catalog_shape.py tests/test_server.py
git commit -m "feat(catalog): a listing names the fields an entry of each for_each list carries

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: An entry key no step reads is a validation warning

**Files:**
- Modify: `dw/for_each.py` (add `entry_field_warnings`)
- Modify: `dw/server/app.py` (the `validate_workflow` route, where `answer["warnings"]` is built, ~line 1345)
- Test: `tests/test_for_each.py`, `tests/test_validate_arguments.py`

**Interfaces:**
- Consumes: `list_fields` (Task 1), `set_variables`, `argument_errors` (`dw/variables.py`).
- Produces: `entry_field_warnings(definition, arguments=None) -> list[str]`. Folds good `arguments` into a copy of the declared variables (exactly as `expanded_definition` does), then for each list-driven variable whose value is a list of dicts, reports every key of every entry that is not in that list's fields (and not `name`), one warning per entry, in the style of `workflow_argument_warnings`: `"arguments.shots[1]: entry 'deflect' carries 'num_frame', which no step reads; entries of 'shots' take: name, num_frames, prompt, references"` — `arguments.` when the caller supplied that variable, `variables.` otherwise. Lists whose `fields` is `None` (value entries) produce no warnings. Never raises: a malformed value is skipped.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_for_each.py`:

```python
class TestEntryFieldWarnings:
    def workflow(self):
        return definition(
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"arguments": {"text": "item:prompt", "n": "item:num_frames"}},
            },
            variables={"shots": [{"name": "a", "prompt": "p", "num_frames": 1}]},
        )

    def test_a_key_no_step_reads_is_reported_at_the_entry(self):
        w = self.workflow()
        w["variables"]["shots"].append({"name": "b", "prompt": "q", "num_frame": 2})
        assert entry_field_warnings(w) == [
            "variables.shots[1]: entry 'b' carries 'num_frame', which no step reads; "
            "entries of 'shots' take: name, num_frames, prompt"
        ]

    def test_a_caller_s_list_is_reported_under_arguments(self):
        warnings = entry_field_warnings(
            self.workflow(),
            arguments={"shots": [{"name": "a", "prompt": "p", "num_frames": 1, "note": "x"}]},
        )
        assert warnings == [
            "arguments.shots[0]: entry 'a' carries 'note', which no step reads; "
            "entries of 'shots' take: name, num_frames, prompt"
        ]

    def test_bad_arguments_fall_back_to_the_defaults(self):
        assert entry_field_warnings(self.workflow(), arguments={"nope": 1}) == []

    def test_value_entries_and_clean_entries_warn_about_nothing(self):
        assert entry_field_warnings(self.workflow()) == []
        w = definition(
            {"name": "say", "for_each": "variable:lines", "task": {"arguments": {"t": "item:"}}},
            variables={"lines": ["a", "b"]},
        )
        assert entry_field_warnings(w) == []

    def test_an_entry_without_a_name_is_named_by_its_index(self):
        w = self.workflow()
        w["variables"]["shots"] = [{"prompt": "p", "num_frames": 1, "extra": 0}]
        assert entry_field_warnings(w)[0].startswith(
            "variables.shots[0]: entry 0 carries 'extra'"
        )
```

Import `entry_field_warnings` at the top.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q tests/test_for_each.py -k EntryFieldWarnings` — ImportError.

- [ ] **Step 3: Implement**

In `dw/for_each.py` after `list_fields` (import `copy` is already there; add `from .variables import argument_errors, set_variables` — `dw/variables.py` imports nothing from this module, so there is no cycle; verify with `python -c "import dw.for_each"`):

```python
def entry_field_warnings(definition, arguments=None):
    """Every entry key of a list-driven variable that no step reads.

    A caller who writes 'num_frame' for 'num_frames' gets the template's
    value for the field they meant to set, in silence; this names the
    key, at the entry it sits in, with the fields the list takes. A
    warning rather than an error: an entry may carry a note on purpose.
    Good `arguments` are folded in first, and a list the caller supplied
    is reported under 'arguments.', where they wrote it.
    """
    if not isinstance(definition, dict):
        return []
    variables = definition.get("variables")
    if not isinstance(variables, dict):
        return []
    variables = copy.deepcopy(variables)
    supplied = set()
    if arguments and not argument_errors(definition, arguments):
        set_variables(arguments, variables)
        supplied = set(arguments)
    warnings = []
    for variable, spec in list_fields(definition).items():
        fields = spec["fields"]
        entries = variables.get(variable)
        if fields is None or not isinstance(entries, list):
            continue
        where = "arguments" if variable in supplied else "variables"
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            unknown = sorted(set(entry) - set(fields))
            if not unknown:
                continue
            label = repr(entry["name"]) if isinstance(entry.get("name"), str) else index
            warnings.append(
                f"{where}.{variable}[{index}]: entry {label} carries "
                f"{', '.join(repr(k) for k in unknown)}, which no step reads; "
                f"entries of '{variable}' take: {', '.join(fields)}"
            )
    return warnings
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_for_each.py -k EntryFieldWarnings` — PASS.

- [ ] **Step 5: Write the failing server test**

In `tests/test_validate_arguments.py` (read its fixtures: it posts to `/api/validate` with a workflow and arguments), add:

```python
def test_an_entry_key_no_step_reads_is_a_warning_not_an_error(client_and_workspace):
    client, _ = client_and_workspace  # adapt to the file's fixture
    workflow = {
        "id": "cut",
        "variables": {"shots": [{"name": "a", "text": "p"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"command": "noop", "arguments": {"text": "item:text"}},
            }
        ],
    }
    response = client.post(
        "/api/validate",
        json={"workflow": workflow, "arguments": {"shots": [{"name": "a", "text": "p", "txt": "q"}]}},
    )
    body = response.json()
    assert body["valid"] is True
    assert any(w.startswith("arguments.shots[0]: entry 'a' carries 'txt'") for w in body["warnings"])
```

Use whatever `task` command the file's other tests use for a no-op step so the schema and the task registry accept it.

- [ ] **Step 6: Hook it in**

`dw/server/app.py` `validate_workflow`: import `entry_field_warnings` from `..for_each`; where `answer` is built, make it

```python
        answer = {
            "valid": True,
            "error": None,
            "errors": [],
            "warnings": workflow_argument_warnings(definition)
            + entry_field_warnings(definition, request.arguments),
        }
```

- [ ] **Step 7: Run tests, full suite, commit**

Run: `python -m pytest -q tests/test_validate_arguments.py tests/test_for_each.py` then the full suite.

```bash
git add dw/for_each.py dw/server/app.py tests/test_for_each.py tests/test_validate_arguments.py
git commit -m "feat(validate): an entry key no for_each step reads is a warning at the entry's path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `per_entry` cost and the nested variables preview

**Files:**
- Modify: `dw/workflow_schema.json` (the `cost` items schema, ~line 36-50)
- Modify: `dw/server/app.py` (`get_workflow_variables`, ~line 1588-1630)
- Modify: `dw_mcp/catalog.py:31-42` (docstring), `docs/MCP.md` (`get_workflow` and `list_workflows` rows, ~lines 215-216), `docs/SERVER.md` (~line 274)
- Test: `tests/test_catalog_shape.py`, `tests/test_catalog_structure.py`, `tests/test_server.py` (`test_workflow_variables_answer_without_the_whole_definition`, ~line 1139)

**Interfaces:**
- Produces: schema — `cost[].per_entry` optional object `{variable: string, minutes: number ≥ 0, entries: integer ≥ 1}`, all three required, `additionalProperties: false`. Catalog tests — for every bundled workflow with a `per_entry`, `variable` is a key of `list_fields(definition)` and `entries == len(variables[variable])`. Variables endpoint — the 200-character preview applies to strings inside list/dict values too, `truncated` naming them as `shots[0].prompt`.

- [ ] **Step 1: Write the failing tests**

`tests/test_catalog_shape.py`, extend `test_the_schema_declares_the_vocabulary`:

```python
    per_entry = cost_item["properties"]["per_entry"]
    assert set(per_entry["required"]) == {"variable", "minutes", "entries"}
    assert per_entry["additionalProperties"] is False
```

and add:

```python
def test_a_per_entry_cost_validates_and_a_partial_one_does_not():
    schema = load_schema("workflow")
    base = definition(pipeline_step("g", "image/jpeg"))
    cost = {"device": "cuda", "vram_gb": 24, "minutes": 42}
    ok, _ = validate_data(
        {**base, "cost": [{**cost, "per_entry": {"variable": "shots", "minutes": 7.2, "entries": 5}}]},
        schema,
    )
    assert ok
    bad, message = validate_data(
        {**base, "cost": [{**cost, "per_entry": {"variable": "shots", "minutes": 7.2}}]}, schema
    )
    assert not bad and "per_entry" in message
```

`tests/test_catalog_structure.py`, next to `test_a_declared_cost_is_well_formed` (same parametrization over every bundled workflow):

```python
@pytest.mark.parametrize("path", WORKFLOWS)  # whatever name the file uses
def test_a_per_entry_cost_names_a_list_the_steps_read(path):
    """per_entry is measured over the default list, so the variable it
    names must be one a for_each expands and `entries` must be that
    list's length - an edited default that forgot the cost block fails
    here."""
    from dw.for_each import list_fields

    definition = load(path)
    for entry in definition.get("cost") or []:
        per_entry = entry.get("per_entry")
        if per_entry is None:
            continue
        lists = list_fields(definition)
        assert per_entry["variable"] in lists, f"{path}: {per_entry['variable']} is not a for_each list"
        default = (definition.get("variables") or {}).get(per_entry["variable"])
        assert isinstance(default, list) and len(default) == per_entry["entries"], path
```

and one synthetic test in `tests/test_catalog_shape.py` or here that exercises the assertion logic on an in-memory definition with a wrong `entries` (so the test is not vacuous while no template has `per_entry`): factor the checks into a helper `per_entry_problems(definition) -> list[str]` in the test module and assert it reports the mismatch.

`tests/test_server.py`, next to `test_workflow_variables_answer_without_the_whole_definition` (read it; mirror its fixture and how it writes a workflow):

```python
def test_workflow_variables_preview_inside_list_entries(server):
    long = "x" * 500
    # write a workflow whose variables are {"shots": [{"name": "a", "prompt": long}], "n": 1}
    ...
    body = client.get("/api/workflows/cut/variables").json()
    assert body["variables"]["shots"][0]["prompt"] == "x" * 200
    assert body["variables"]["shots"][0]["name"] == "a"
    assert body["truncated"] == ["shots[0].prompt"]
    full = client.get("/api/workflows/cut/variables", params={"full": "true"}).json()
    assert full["variables"]["shots"][0]["prompt"] == long and full["truncated"] == []
```

- [ ] **Step 2: Run to verify they fail** — schema keys missing; `truncated` empty / prompt whole.

- [ ] **Step 3: Implement**

`dw/workflow_schema.json`, inside the `cost` items `properties`:

```json
"per_entry": {
    "type": "object",
    "description": "For a list-driven workflow: the measured cost of one entry of 'variable' (every member it produces), and the length of the default list 'minutes' was measured with, so a run over N entries is (minutes - per_entry.minutes * entries) + per_entry.minutes * N.",
    "required": ["variable", "minutes", "entries"],
    "properties": {
        "variable": {"type": "string"},
        "minutes": {"type": "number", "minimum": 0},
        "entries": {"type": "integer", "minimum": 1}
    },
    "additionalProperties": false
}
```

`dw/server/app.py` `get_workflow_variables`: replace the loop with a recursive preview:

```python
        def preview(value, path):
            if isinstance(value, str) and len(value) > VARIABLE_VALUE_PREVIEW:
                truncated.append(path)
                return value[:VARIABLE_VALUE_PREVIEW]
            if isinstance(value, list):
                return [preview(item, f"{path}[{i}]") for i, item in enumerate(value)]
            if isinstance(value, dict):
                return {key: preview(item, f"{path}.{key}") for key, item in value.items()}
            return value

        variables = definition.get("variables") or {}
        values, truncated = {}, []
        for variable, value in variables.items():
            values[variable] = value if full else preview(value, variable)
```

Update the docstring: "Long strings - a shot's prompt runs to kilobytes, and a list-driven workflow's default list holds several - are cut to their first 200 characters wherever they sit and named in `truncated` (`shots[0].prompt`)".

Docs: `dw_mcp/catalog.py` `get_workflow` docstring adds "…including strings inside a list default, named like `shots[0].prompt`"; `docs/MCP.md` `get_workflow` row says the same; `docs/SERVER.md` line ~274 says the same; `docs/MCP.md` `list_workflows` row adds `lists` to the compact field list with one clause ("`lists`, present for a list-driven workflow, names the fields an entry of each list takes, the steps over it and the default's length") and says `cost` may carry `per_entry`.

- [ ] **Step 4: Run tests, full suite, commit**

```bash
git add dw/workflow_schema.json dw/server/app.py dw_mcp/catalog.py docs/MCP.md docs/SERVER.md tests/test_catalog_shape.py tests/test_catalog_structure.py tests/test_server.py
git commit -m "feat(catalog): per_entry cost on a measured entry; variables preview reaches inside list entries

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Rulings — empty list, constants in validation, cache bound

**Files:**
- Modify: `dw/for_each.py` (`_entry_keys`), `dw/workflow_schema.json` (`for_each` gets `"minItems": 1`), `dw/workflow.py` (`expanded_definition`), `dw/step_cache.py` (`DEFAULT_MAX_ENTRIES`)
- Test: `tests/test_for_each.py`, `tests/test_schema.py`, `tests/test_workflow.py`, `tests/test_step_cache.py`

- [ ] **Step 1: Rewrite the two empty-list tests and add the new ones**

In `tests/test_for_each.py`, replace `test_an_empty_list_expands_to_no_steps` (~line 127) with:

```python
    def test_an_empty_list_is_an_error_at_the_step(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(definition({"name": "shot", "for_each": [], "task": {}}))
        assert e.value.path == "steps[0].for_each"
        assert "would run no steps" in str(e.value)
```

and replace `test_gather_of_an_empty_group_is_an_empty_list` (~line 410) with a test that a `gather:` of a group that failed to expand never gets that far (the error above fires first) — or simply delete it and say so in the report, since the situation cannot arise.

`tests/test_schema.py`: add a test that a step with `"for_each": []` fails schema validation with `for_each` in the message (mirror the file's existing for_each schema tests).

`tests/test_workflow.py`: add

```python
def test_validation_realizes_a_constant_default_list(tmp_path):
    """A list defaulted to a constant: name used to fail validation with the
    string unsubstituted and then run fine, since only the run realized
    constants."""
    definition = _for_each_workflow()
    definition["variables"]["shots"] = "constant:tests.test_workflow.CONSTANT_SHOTS"
    workflow = _workflow_from(definition, tmp_path)
    assert workflow.validation_errors() == []
    assert [s["name"] for s in workflow.expanded_definition()["steps"]][:2] == ["shot@a", "shot@b"]
```

with `CONSTANT_SHOTS = [{"name": "a", "text": "1"}, {"name": "b", "text": "2"}]` at module level (match `_for_each_workflow`'s entry field name). `validate_constant_name` (`dw/security.py`) must accept a `tests.` module path — check its rules; if it refuses, put the constant on a module it accepts and say which.

`tests/test_step_cache.py`: the cap test uses the symbol, so it needs no edit; add one assertion `assert StepCache.DEFAULT_MAX_ENTRIES >= 2 * MAX_FOR_EACH_ENTRIES + 8` with a comment: a maximal for_each run over two groups plus fixed steps must fit, or a run evicts its own earlier members.

- [ ] **Step 2: Run to verify they fail** — empty list expands; schema accepts `[]`; constant default fails validation; cap assertion fails at 50.

- [ ] **Step 3: Implement**

`dw/for_each.py` `_entry_keys`, after the type check:

```python
    if not entries:
        raise ForEachError(
            render_path(path),
            "for_each over an empty list would run no steps - a workflow that "
            "generates nothing is never what was asked for",
        )
```

`dw/workflow_schema.json`: `"minItems": 1` beside `"maxItems": 32`.

`dw/workflow.py` `expanded_definition`: import `realize_constants` from `.arguments` and call `realize_constants(variables)` first thing inside the `if isinstance(variables, dict):` block, with a comment: the run realizes constants before folding arguments, and a list defaulted to a `constant:` name must expand here as it does there — a name lookup, no download. Update the docstring's first sentence accordingly.

`dw/step_cache.py`: `DEFAULT_MAX_ENTRIES = 128`, and extend the class/module comment: sized so a maximal for_each run (32 entries over two groups plus fixed steps, ~70 members) never evicts its own members before it ends.

- [ ] **Step 4: Update the prose that quoted the old numbers**

`docs/proposals/list-driven-steps.md`: in "Limits and cost" (~line 216-224) the sentence about `DEFAULT_MAX_ENTRIES = 50` and "leaves the cache alone" — append "(raised to 128 in stage 3)". In the stage 3 section's "Decisions carried" bullet, change "Raise the bound to 128, or make eviction skip…" to state that it was raised to 128. `CLAUDE.md` gotcha about the step cache needs no number. The stage-2 "Notes" paragraph that says the empty-list question "is a stage-2 decision" — leave; the stage 3 section rules it.

- [ ] **Step 5: Run tests, full suite, commit**

```bash
git add dw/for_each.py dw/workflow_schema.json dw/workflow.py dw/step_cache.py tests/test_for_each.py tests/test_schema.py tests/test_workflow.py tests/test_step_cache.py docs/proposals/list-driven-steps.md
git commit -m "fix(for_each): an empty list is an error; validation realizes constants; step cache holds a maximal run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The skill, the guide, the MCP instructions and CLAUDE.md quote `lists` and `per_entry`

**Files:**
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md` (the cost sentences in "A piece with cuts", ~lines 63-67)
- Modify: `tests/test_plugin_skills.py` (`test_the_cuts_templates_are_described_as_list_driven`)
- Modify: `docs/WORKFLOW_GUIDE.md` ("Limits:" paragraph of `### One step per entry: for_each`, ~line 412-420)
- Modify: `dw_mcp/server.py` (the instructions string, ~line 75-80: "summary, shape, traits, cost and variable names")
- Modify: `CLAUDE.md` (Type System `for_each` bullet, ~line 157-170)
- Modify: `docs/proposals/list-driven-steps.md` (status line, lines 3-6)

- [ ] **Step 1: Extend the skill test**

In `test_the_cuts_templates_are_described_as_list_driven`, replace the last assertion with:

```python
        # cost: the listing's per_entry block when present, the honest
        # fallback when it is not
        assert "`per_entry`" in text
        assert "`lists`" in text
```

- [ ] **Step 2: Run it** — fails on `` `per_entry` ``.

- [ ] **Step 3: Edit the skill**

Replace the sentences from "The listing's `cost`" through "then multiply by the entries you write." with:

```
  The listing's `lists` block says what an entry carries; its `cost`
  carries `per_entry` when one shot was measured: quote
  `minutes - per_entry.minutes × per_entry.entries + per_entry.minutes × N`
  for N entries. Without `per_entry`, quote the total and say it is the
  default list's.
```

Keep the two entry-shape sentences before it. Measure with `wc -c`; stay ≤ 12288 (the replacement is shorter than what it replaces).

- [ ] **Step 4: The guide, the MCP instructions, CLAUDE.md, the proposal**

`docs/WORKFLOW_GUIDE.md` "Limits:" paragraph — replace the sentence from "the listing's `cost` is for the whole workflow" through "then multiply by the entries you write." with: "the listing's `lists` block names the fields an entry takes and the steps over it, and its `cost` carries `per_entry` once one entry has been measured — quote `minutes - per_entry.minutes × per_entry.entries + per_entry.minutes × N` for N entries, and without `per_entry` quote the total as the default list's. An entry key no step reads is a validation warning at the entry's path, so a misspelt field is caught before the run."

`dw_mcp/server.py` instructions: "carries each workflow's summary, shape, traits, cost and variable names" → "carries each workflow's summary, shape, traits, cost (with `per_entry` for a list-driven one), variable names and, for a list-driven workflow, `lists` - what an entry of each list carries". Check `tests/test_mcp_*.py` for a pin on that sentence and update it if one exists.

`CLAUDE.md` Type System `for_each` bullet: append "The catalog derives `lists` (`list_fields`, `dw/for_each.py`): the fields an entry takes are the `item:` references the steps make, `name` first; an entry key no step reads is a validation warning (`entry_field_warnings`). A `cost` entry may carry `per_entry` (`{variable, minutes, entries}`), measured, never derived. An empty `for_each` list is an error; `expanded_definition` realizes constants first."

`docs/proposals/list-driven-steps.md` status: "**stages 1, 2 and 3 implemented** (…; catalog `lists`, `per_entry` cost schema, entry-key warning, rulings, flow-view edges, 2026-09-11). `per_entry` figures for the two templates await a measured run."

- [ ] **Step 5: Tests, suite, commit**

Run: `python -m pytest -q tests/test_plugin_skills.py tests/test_server_guides.py tests/test_mcp_catalog.py` then the full suite.

```bash
git add plugins/dw/skills/minimax-h3/SKILL.md tests/test_plugin_skills.py docs/WORKFLOW_GUIDE.md dw_mcp/server.py CLAUDE.md docs/proposals/list-driven-steps.md
git commit -m "docs(for_each): the listing's lists block and per_entry cost, quoted by the skill and the guide

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The editor's flow view sees `gather:` and `from_previous_result`

**Files:**
- Modify: `ui/src/lib/flow.ts` (`refTarget`, `flowGraph`, `dataFlowGraph`, `danglingReferenceDetails`)
- Test: `ui/src/lib/flow.test.ts`

**Interfaces:**
- Produces: a reference is any of (a) a string `previous_result:<step>[.suffix]`, (b) a string `gather:<step>`, (c) the string value under a `from_previous_result` key. All three make an edge from `<step>` to the referencing step in `flowGraph` and `dataFlowGraph` (the edge's `attribute` label for (c) is the label of the path *above* the `from_previous_result` key — `references` — so it reads as the argument that carries it). `danglingReferenceDetails` reports a `gather:<step>` that names no earlier step, with the message `Step '<name>': gather:<step> - no earlier step has that name`.

- [ ] **Step 1: Write the failing tests**

Append to `ui/src/lib/flow.test.ts`:

```ts
describe('for_each references', () => {
  const cut = {
    steps: [
      step('draw', {}),
      step('slice', { audio: 'previous_result:song' }),
      {
        name: 'shot',
        for_each: 'variable:shots',
        pipeline: {
          configuration: { component_type: 'ModularPipeline' },
          arguments: {
            prompt: 'item:prompt',
            references: [
              { reference_type: 'T', from_previous_result: 'draw' },
              { reference_type: 'T', from_previous_result: 'slice' },
            ],
          },
        },
      },
      step('edit', { videos: 'gather:shot' }),
    ],
  }

  it('from_previous_result inside a reference list is an edge labeled by the list', () => {
    const graph = dataFlowGraph(cut)
    expect(graph.edges).toContainEqual({ from: 'draw', to: 'shot', attribute: 'references' })
    expect(graph.edges).toContainEqual({ from: 'slice', to: 'shot', attribute: 'references' })
    expect(graph.nodes.find((n) => n.name === 'shot')?.isEntryPoint).toBe(false)
  })

  it('gather: is an edge from the for_each step to the step that gathers it', () => {
    const graph = dataFlowGraph(cut)
    expect(graph.edges).toContainEqual({ from: 'shot', to: 'edit', attribute: 'videos' })
    expect(graph.nodes.find((n) => n.name === 'edit')?.isEntryPoint).toBe(false)
    expect(flowGraph(cut)[3].inputs).toEqual(['shot'])
    expect(flowGraph(cut)[2].consumers).toEqual(['edit'])
  })

  it('a gather of a step that does not exist is dangling', () => {
    const wf = { steps: [step('edit', { videos: 'gather:nope' })] }
    expect(danglingReferenceDetails(wf)).toEqual([
      { stepIndex: 0, message: "Step 'edit': gather:nope - no earlier step has that name" },
    ])
  })
})
```

Adjust `danglingReferenceDetails`' expected shape to match what the existing test at line ~96 asserts (read it first).

- [ ] **Step 2: Run** — `cd ui && npm test -- flow` — fails.

- [ ] **Step 3: Implement**

In `flow.ts`, generalise `refTarget` to take the value and the path:

```ts
/** The earlier step a value refers to: `previous_result:<step>[.suffix]`,
 * `gather:<step>` (every member of a for_each step), or the string under
 * a `from_previous_result` key inside a reference object. */
function refTarget(s: string, path: string[] = []): string | null {
  if (s.startsWith('previous_result:')) return s.slice('previous_result:'.length).split('.')[0]
  if (s.startsWith('gather:')) return s.slice('gather:'.length)
  if (path[path.length - 1] === 'from_previous_result') return s
  return null
}
```

`flowGraph` currently uses `scanStrings` (no path); switch it to `scanStringsWithPath` so `from_previous_result` is seen. In `dataFlowGraph`, pass `path` to `refTarget`; for the attribute label, when the path ends in `from_previous_result`, use `attributeLabel(path.slice(0, -1))` so the label is the carrying argument (`references`) — check what `attributeLabel` does with numeric indices and make sure `references.0` collapses to `references` (read lines 84-94). In `danglingReferenceDetails`, add a `gather:` branch mirroring the `previous_result:` one with the message above.

- [ ] **Step 4: Run the UI checks**

Run from `ui/`: `npm run check && npm run lint && npm run format && npm test`
Expected: all pass; commit any prettier rewrites of the files you touched.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/flow.ts ui/src/lib/flow.test.ts
git commit -m "ui(flow): edges for gather: and from_previous_result, so a list-driven workflow's editor is not an orphan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
