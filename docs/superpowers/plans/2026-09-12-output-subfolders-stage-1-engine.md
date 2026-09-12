# Output Subfolders, Stage 1 (Engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A step's `result.subfolder` places that step's files in a subfolder of the run directory, carried through the manifests and the `step_end` event, validated statically and at run time, with `strip_run_id` still grouping a workflow's runs.

**Architecture:** The step's target directory is computed once in `Workflow.step_output_dir` and handed to both `Result.save` and the pipeline wrapper (so chain segment spills follow it). Shape is checked by a segment pattern aligned with `output:` references (`dw/security.py`), containment by `validate_output_path` against the run directory. A new small module `dw/subfolders.py` owns the static pass that `validation_errors` runs after `for_each` expansion, beside `previous_result_reference_errors`. `dw/runs.py` gains `split_run_path`, which finds the run-id segment anywhere in a path; `strip_run_id` becomes a wrapper over it.

**Tech Stack:** Python 3, pytest. Tests run with `python -m pytest tests/<file> -q` from the repo root after `source ./activate`.

**Spec:** `docs/proposals/output-folders.md` — the sections *The field*, *Where the checks live*, *`file_base_name` may no longer contain a separator*, *The engine*, *The two manifests*, *Tests* and stage 1 of *Phasing*. Stages 2-4 (server/MCP, steering, UI) are separate plans.

## Global Constraints

- The field is named `subfolder` everywhere: schema, manifest entries, `step_end` event.
- No default: a step without `subfolder` writes exactly the paths it writes today.
- Shape pattern, verbatim: `^[\w][\w.-]*(/[\w][\w.-]*)*\Z` (one or more segments; each segment starts with a word character; no `\`, no empty segment, no leading `.`/`-`).
- The schema gets a `description` only, never a `pattern` — schema validation runs before `variable:`/`item:` substitution.
- Filesystem access goes through `dw/security.py` validators (`validate_output_path`) — CodeQL models them as sanitizers; a bare `os.path.join` reaching the disk is a real alert.
- `file_base_name` containing `/` or `\` is refused both statically and at run time.
- Never use `eval()`, `exec()`, or `shell=True`.
- Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Work on a branch off `develop` (suggested name `output-subfolders-stage-1`).

---

## File map

| File | Responsibility in this plan |
|---|---|
| `dw/runs.py` | `split_run_path(relative) -> (identity, run_id, subfolder)`; `strip_run_id` wraps it |
| `dw/security.py` | `SUBFOLDER_PATTERN`, `validate_subfolder(name)`, `validate_file_base_name(name)` |
| `dw/subfolders.py` (new) | `SUBFOLDER_KEY`, `step_subfolder(step_definition)`, `subfolder_errors(expanded, source_indices)` |
| `dw/workflow.py` | `validation_errors` runs `subfolder_errors`; `step_output_dir(step_definition)`; feeds `Result.save` and `create_step_action`; `subfolder` on manifest entry and `step_end` |
| `dw/result.py` | `Result.save` refuses a separator in `file_base_name` |
| `dw/workflow_schema.json` | `subfolder` description on `$defs/result`; `file_base_name` description amended |
| `CLAUDE.md` | One gotcha bullet on subfolders under *Critical Gotchas* |
| `tests/test_runs.py`, `tests/test_security.py`, `tests/test_subfolders.py` (new), `tests/test_schema.py` | Tests |

---

### Task 1: `split_run_path` in `dw/runs.py`

**Files:**
- Modify: `dw/runs.py:311-322` (`strip_run_id`)
- Test: `tests/test_runs.py` (class `TestRunIds`, near line 68)

**Interfaces:**
- Produces: `split_run_path(relative_path: str) -> tuple[str, str, str]` returning `(identity, run_id, subfolder)`, each `""` when absent. `strip_run_id(relative_path) -> str` unchanged in signature and behaviour for every path it accepted before.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_runs.py`, importing `split_run_path` alongside `strip_run_id` in the `from dw.runs import (...)` block, and inside the class that holds `test_stripping_a_run_id_gives_the_workflow_folder`:

```python
    def test_splitting_a_run_path_names_its_three_parts(self):
        run_id = new_run_id({"id": "x"})
        assert split_run_path(f"ltx2/Gyre/{run_id}/still.png") == (
            "ltx2/Gyre",
            run_id,
            "",
        )
        assert split_run_path(f"ltx2/Gyre/{run_id}/final/still.png") == (
            "ltx2/Gyre",
            run_id,
            "final",
        )
        assert split_run_path(f"ltx2/Gyre/{run_id}/shots/act-1/x.mp4") == (
            "ltx2/Gyre",
            run_id,
            "shots/act-1",
        )
        # A counter suffix is still a run id
        assert split_run_path(f"Gyre/{run_id}-2/final/x.mp4") == (
            "Gyre",
            f"{run_id}-2",
            "final",
        )

    def test_a_path_with_no_run_id_splits_to_its_directory(self):
        # Flat layout: nothing to anchor on, so the directory is the identity
        assert split_run_path("ltx2/final/still.png") == ("ltx2/final", "", "")
        assert split_run_path("still.png") == ("", "", "")

    def test_stripping_a_run_id_ignores_what_follows_it(self):
        run_id = new_run_id({"id": "x"})
        assert strip_run_id(f"ltx2/Gyre/{run_id}/final/still.png") == "ltx2/Gyre"
        assert strip_run_id(f"{run_id}/final/still.png") == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_runs.py -q -k "split_run_path or ignores_what_follows"`
Expected: FAIL — `ImportError: cannot import name 'split_run_path'`.

- [ ] **Step 3: Implement `split_run_path` and rewrite `strip_run_id` over it**

Replace `strip_run_id` in `dw/runs.py` with:

```python
def split_run_path(relative_path):
    """The three parts of a run-relative path: (identity, run id, subfolder).

    'ltx2/Gyre/20260905-181530-a1b2c3d4/final/still.png' ->
    ('ltx2/Gyre', '20260905-181530-a1b2c3d4', 'final'). The run id is
    found wherever it sits, not only as the last directory - a step's
    'subfolder' puts segments after it. A path with no run id in it (the
    flat layout) has its whole directory as identity and nothing else,
    which is what it was before subfolders existed.

    Only the first segment matching RUN_ID_PATTERN counts. A workflow
    *file* named in that shape would produce a matching identity segment;
    that is unsupported rather than impossible.
    """
    parts = [part for part in (relative_path or "").split("/") if part]
    directory = parts[:-1]
    for position, segment in enumerate(directory):
        if is_run_id(segment):
            return (
                "/".join(directory[:position]),
                segment,
                "/".join(directory[position + 1 :]),
            )
    return "/".join(directory), "", ""


def strip_run_id(relative_path):
    """The workflow identity a run-relative path belongs to.

    'ltx2/Gyre/20260905-181530-a1b2c3d4/still-0.png' -> 'ltx2/Gyre', and
    the same with a subfolder after the run id. A path with no run id in it
    comes back with its own directory unchanged, which is what a
    flat-layout output does.
    """
    return split_run_path(relative_path)[0]
```

- [ ] **Step 4: Run the whole runs test file**

Run: `python -m pytest tests/test_runs.py -q`
Expected: all PASS, including the pre-existing `strip_run_id` tests.

- [ ] **Step 5: Commit**

```bash
git add dw/runs.py tests/test_runs.py
git commit -m "feat(runs): split_run_path finds the run id anywhere; strip_run_id wraps it

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Shape validators in `dw/security.py`

**Files:**
- Modify: `dw/security.py` (after `validate_output_reference`, around line 556)
- Test: `tests/test_security.py`

**Interfaces:**
- Produces: `SUBFOLDER_PATTERN: str`, `MAX_SUBFOLDER_LENGTH: int = 200`, `validate_subfolder(name: str) -> str` (raises `InvalidInputError`), `validate_file_base_name(name: str) -> str` (raises `InvalidInputError` on `/` or `\`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_security.py` (match its existing import style — it imports from `dw.security`):

```python
class TestSubfolderValidation:
    def test_a_segment_path_is_accepted(self):
        from dw.security import validate_subfolder

        assert validate_subfolder("final") == "final"
        assert validate_subfolder("shots/act-1") == "shots/act-1"
        assert validate_subfolder("a.b_c-d/e") == "a.b_c-d/e"

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "../final",
            "final/..",
            "/final",
            "final/",
            "final//x",
            ".hidden",
            "-dash",
            "final\\x",
            "final x",
            "fin:al",
        ],
    )
    def test_the_shape_refuses_what_output_references_refuse(self, bad):
        from dw.security import InvalidInputError, validate_subfolder

        with pytest.raises(InvalidInputError):
            validate_subfolder(bad)

    def test_too_long_is_refused(self):
        from dw.security import InvalidInputError, validate_subfolder

        with pytest.raises(InvalidInputError):
            validate_subfolder("a" * 201)


class TestFileBaseNameValidation:
    def test_a_plain_name_is_accepted(self):
        from dw.security import validate_file_base_name

        assert validate_file_base_name("episode_") == "episode_"

    @pytest.mark.parametrize("bad", ["final/", "a/b", "a\\b"])
    def test_a_separator_is_refused_and_names_subfolder(self, bad):
        from dw.security import InvalidInputError, validate_file_base_name

        with pytest.raises(InvalidInputError, match="subfolder"):
            validate_file_base_name(bad)
```

Add `import pytest` at the top of the file if it is not already imported.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_security.py -q -k "Subfolder or FileBaseName"`
Expected: FAIL — `ImportError: cannot import name 'validate_subfolder'`.

- [ ] **Step 3: Implement the validators**

Insert after `validate_output_reference` in `dw/security.py`:

```python
# A step's result 'subfolder': a relative path under the run directory. The
# segment rule is OUTPUT_REFERENCE_PATTERN's, so every subfolder the engine
# writes is one a later workflow can name with 'output:'. It also refuses a
# backslash, which DANGEROUS_PATTERNS does not - '"final\\x"' would be one
# directory on POSIX and two on Windows
SUBFOLDER_PATTERN = r"^[\w][\w.-]*(/[\w][\w.-]*)*\Z"
MAX_SUBFOLDER_LENGTH = 200


def validate_subfolder(name: str) -> str:
    """
    Validate the shape of a result 'subfolder'.

    Containment is checked separately, by the validate_output_path call
    that joins it onto the run directory.

    Args:
        name: The subfolder as written in the workflow

    Returns:
        The validated name

    Raises:
        InvalidInputError: If the name is not a valid subfolder
    """
    return _validate_name(
        name,
        SUBFOLDER_PATTERN,
        MAX_SUBFOLDER_LENGTH,
        "Subfolder",
        "a subfolder is one or more path segments under the run directory, "
        "each starting with a letter, digit or underscore, like 'final' or "
        "'shots/act-1'",
    )


def validate_file_base_name(name: str) -> str:
    """
    Validate a result 'file_base_name': a name, never a path.

    A separator here used to pass validation and then fail at open() because
    the directory did not exist. Placement is what 'subfolder' is for.

    Raises:
        InvalidInputError: If the name carries a path separator
    """
    if "/" in name or "\\" in name:
        raise InvalidInputError(
            f"Invalid file_base_name: {name} - a file_base_name is a name, not "
            f"a path; to write into a subfolder of the run directory set "
            f"'subfolder' on the result instead"
        )
    return name
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_security.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/security.py tests/test_security.py
git commit -m "feat(security): validate_subfolder and validate_file_base_name

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `dw/subfolders.py` — the static pass, wired into `validation_errors`

**Files:**
- Create: `dw/subfolders.py`
- Modify: `dw/workflow.py:351-382` (`validation_errors`), imports near line 35-75
- Test: `tests/test_subfolders.py` (new)

**Interfaces:**
- Consumes: `validate_subfolder`, `validate_file_base_name` from Task 2; `render_path`, `MEMBER_SEPARATOR` from `dw/for_each.py`.
- Produces: `SUBFOLDER_KEY = "subfolder"`; `step_subfolder(step_definition: dict) -> str` (validated value, or `""` when absent; raises `InvalidInputError` on a bad one); `subfolder_errors(workflow_definition: dict, source_indices: list[int] | None = None) -> list[dict]` with `{path, message}` entries in the same shape `previous_result_reference_errors` returns.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_subfolders.py`:

```python
import pytest

from dw.security import InvalidInputError
from dw.subfolders import step_subfolder, subfolder_errors


def _step(name, result=None, **extra):
    step = {"name": name, "task": {"command": "noop", "arguments": {}}}
    if result is not None:
        step["result"] = result
    step.update(extra)
    return step


class TestStepSubfolder:
    def test_absent_is_the_empty_string(self):
        assert step_subfolder(_step("a")) == ""
        assert step_subfolder(_step("a", {"content_type": "image/png"})) == ""

    def test_present_is_returned_validated(self):
        assert step_subfolder(_step("a", {"subfolder": "final"})) == "final"

    def test_a_bad_one_raises(self):
        with pytest.raises(InvalidInputError):
            step_subfolder(_step("a", {"subfolder": "../x"}))

    def test_a_non_string_raises(self):
        with pytest.raises(InvalidInputError):
            step_subfolder(_step("a", {"subfolder": 3}))


class TestSubfolderErrors:
    def test_a_clean_definition_has_none(self):
        definition = {
            "steps": [
                _step("a", {"content_type": "image/png", "subfolder": "final"}),
                _step("b", {"content_type": "image/png"}),
            ]
        }
        assert subfolder_errors(definition) == []

    def test_a_bad_subfolder_is_reported_at_its_path(self):
        definition = {"steps": [_step("a", {"subfolder": "../escape"})]}
        errors = subfolder_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.subfolder"
        assert "Subfolder" in errors[0]["message"] or "subfolder" in errors[0]["message"]

    def test_a_separator_in_file_base_name_is_reported_at_its_path(self):
        definition = {"steps": [_step("a", {"file_base_name": "final/"})]}
        errors = subfolder_errors(definition)
        assert [e["path"] for e in errors] == ["steps[0].result.file_base_name"]
        assert "subfolder" in errors[0]["message"]

    def test_an_expanded_member_reports_the_source_step_and_names_the_member(self):
        # expand_for_each turned steps[1] into two members; the author wrote
        # one step, so the path is the source's and the member is named
        definition = {
            "steps": [
                _step("intro", {"subfolder": "final"}),
                _step("shot@a", {"subfolder": "ok"}),
                _step("shot@b", {"subfolder": "bad/../x"}),
            ]
        }
        errors = subfolder_errors(definition, source_indices=[0, 1, 1])
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[1].result.subfolder"
        assert "shot@b" in errors[0]["message"]

    def test_an_unsubstituted_reference_is_left_alone(self):
        # A 'variable:' still spelled out is one nothing resolved; that is
        # the undeclared-variable pass's complaint, not this one's
        definition = {"steps": [_step("a", {"subfolder": "variable:dest"})]}
        assert subfolder_errors(definition) == []

    def test_no_steps_is_fine(self):
        assert subfolder_errors({}) == []
        assert subfolder_errors({"steps": "nope"}) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_subfolders.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.subfolders'`.

- [ ] **Step 3: Create the module**

Create `dw/subfolders.py`:

```python
"""A step's result 'subfolder': where under the run directory its files go.

A run writes everything into one directory, so a finished episode sits
beside the twenty scratch files that went into it, distinguished only by
the step name in each file name. 'subfolder' on a step's result block puts
that step's files into a subfolder of the run directory instead - by
convention 'final' or 'intermediate', though the engine treats no name
specially. Nothing else about placement changes: a step without one writes
where it always did.

This module owns the shape check and the static pass over an expanded
definition. Containment - that the joined path really is inside the run
directory - is the engine's, at the moment it joins (see
Workflow.step_output_dir).
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .security import (
    InvalidInputError,
    validate_file_base_name,
    validate_subfolder,
)

SUBFOLDER_KEY = "subfolder"
FILE_BASE_NAME_KEY = "file_base_name"

# Reference prefixes substitution resolves before this pass runs. One still
# spelled out here is one nothing resolved, and that is the undeclared-
# variable pass's complaint rather than a shape error
_UNRESOLVED_PREFIXES = ("variable:", "item:")


def step_subfolder(step_definition):
    """The validated subfolder a step's result names, or '' when it names
    none.

    Raises:
        InvalidInputError: If the value is not a string or not a valid
            subfolder
    """
    result = step_definition.get("result")
    if not isinstance(result, dict) or SUBFOLDER_KEY not in result:
        return ""
    value = result[SUBFOLDER_KEY]
    if not isinstance(value, str):
        raise InvalidInputError(
            f"Invalid subfolder: {value!r} - a subfolder is a string like 'final'"
        )
    return validate_subfolder(value)


def subfolder_errors(workflow_definition, source_indices=None):
    """Every result 'subfolder' or 'file_base_name' that cannot be written,
    as [{path, message}].

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
        if not isinstance(result, dict):
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
        for key, check in (
            (SUBFOLDER_KEY, validate_subfolder),
            (FILE_BASE_NAME_KEY, validate_file_base_name),
        ):
            if key not in result:
                continue
            value = result[key]
            if isinstance(value, str) and value.startswith(_UNRESOLVED_PREFIXES):
                continue
            try:
                if not isinstance(value, str):
                    raise InvalidInputError(
                        f"Invalid {key}: {value!r} - expected a string"
                    )
                check(value)
            except InvalidInputError as e:
                errors.append(
                    {
                        "path": render_path(("steps", source, "result", key)),
                        "message": f"{e}{where}",
                    }
                )
    return errors
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_subfolders.py -q`
Expected: all PASS.

- [ ] **Step 5: Write the failing integration test for `validation_errors`**

Append to `tests/test_subfolders.py`:

```python
class TestValidationErrorsIntegration:
    def _workflow(self, definition, tmp_path):
        from dw.workflow import Workflow

        return Workflow(definition, str(tmp_path), "/w/workflows/Sub.json")

    def test_validation_errors_reports_a_bad_subfolder(self, tmp_path):
        definition = {
            "id": "sub_test",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "../x"},
                }
            ],
        }
        errors = self._workflow(definition, tmp_path).validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].result.subfolder"]

    def test_a_variable_driven_subfolder_is_checked_by_its_value(self, tmp_path):
        definition = {
            "id": "sub_test",
            "variables": {"dest": "final"},
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "variable:dest"},
                }
            ],
        }
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors() == []
        errors = workflow.validation_errors(arguments={"dest": "../x"})
        assert [e["path"] for e in errors] == ["steps[0].result.subfolder"]

    def test_an_item_driven_subfolder_is_checked_per_member(self, tmp_path):
        definition = {
            "id": "sub_test",
            "steps": [
                {
                    "name": "shot",
                    "for_each": [
                        {"name": "a", "dest": "shots/a"},
                        {"name": "b", "dest": "../x"},
                    ],
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "item:dest"},
                }
            ],
        }
        errors = self._workflow(definition, tmp_path).validation_errors()
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.subfolder"
        assert "shot@b" in errors[0]["message"]
```

Run: `python -m pytest tests/test_subfolders.py -q -k Integration`
Expected: FAIL — `validation_errors()` returns `[]` for the bad subfolder (the schema accepts any string and nothing checks it yet). `noop` is fine as a command: the schema types `task.command` as a free-form string, and these tests only validate, never run.

- [ ] **Step 6: Wire `subfolder_errors` into `validation_errors`**

In `dw/workflow.py`, add the import beside the `previous_result_reference_errors` import (find it with `grep -n previous_result_reference_errors dw/workflow.py`):

```python
from .subfolders import step_subfolder, subfolder_errors
```

Change the last line of `validation_errors` from:

```python
        return previous_result_reference_errors(expanded, source_indices)
```

to:

```python
        return previous_result_reference_errors(
            expanded, source_indices
        ) + subfolder_errors(expanded, source_indices)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_subfolders.py tests/test_previous_results.py tests/test_for_each.py -q`
Expected: all PASS (`step_subfolder` is imported now and used in Task 5; an unused import here is fine for one commit).

- [ ] **Step 8: Commit**

```bash
git add dw/subfolders.py dw/workflow.py tests/test_subfolders.py
git commit -m "feat(validate): subfolder_errors checks result.subfolder and file_base_name after expansion

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Schema descriptions

**Files:**
- Modify: `dw/workflow_schema.json` (`$defs/result`, around line 1099-1115)
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `$defs/result.properties.subfolder` (`type: string`, description only).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py` (check its imports; it already imports `load_schema` or `validate_data_all` from `dw.schema` — use whichever it has, or add `from dw.schema import load_schema, validate_data_all`):

```python
class TestResultSubfolder:
    def test_subfolder_is_a_described_string_with_no_pattern(self):
        result = load_schema("workflow")["$defs"]["result"]["properties"]
        assert result["subfolder"]["type"] == "string"
        assert "final" in result["subfolder"]["description"]
        # Schema validation runs before substitution, so a pattern would
        # reject 'variable:dest' and 'item:subfolder'
        assert "pattern" not in result["subfolder"]

    def test_a_workflow_with_a_subfolder_validates(self):
        definition = {
            "id": "s",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "noop", "arguments": {}},
                    "result": {"content_type": "image/png", "subfolder": "variable:dest"},
                }
            ],
        }
        assert validate_data_all(definition, load_schema("workflow")) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_schema.py -q -k Subfolder`
Expected: FAIL — `KeyError: 'subfolder'`.

- [ ] **Step 3: Add the property and amend `file_base_name`**

In `dw/workflow_schema.json`, inside `$defs.result.properties`, change the `file_base_name` description and add `subfolder` after it:

```json
                "file_base_name": {
                    "description": "The base name for saving result and metadata files. A name, not a path - it may not contain a separator; to place a step's files in a subfolder of the run directory set 'subfolder'. The file extension is determined by context and content_type.",
                    "type": "string"
                },
                "subfolder": {
                    "description": "A subfolder of the run directory this step's files are written into, as a relative path like 'final' or 'shots/act-1'. Optional: without it the files land at the root of the run directory as they always have. By convention a step whose output the user will be shown is 'final' and everything else is 'intermediate', which is how a consumer tells the deliverable from the scratch files without knowing the workflow; the engine treats no name specially. May be a 'variable:' or, inside a for_each step, an 'item:' reference. Each segment must start with a letter, digit or underscore; '..' and '\\' are refused.",
                    "type": "string"
                },
```

- [ ] **Step 4: Run the schema tests**

Run: `python -m pytest tests/test_schema.py tests/test_configuration_schema.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow_schema.json tests/test_schema.py
git commit -m "feat(schema): result.subfolder, description only; file_base_name is a name not a path

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `Workflow.step_output_dir` feeds `Result.save` and the pipeline wrapper

**Files:**
- Modify: `dw/workflow.py` — new method beside `effective_output_dir` (line ~285); the `result.save(...)` call at ~743; the three `output_dir=self.effective_output_dir` kwargs in `create_step_action` at ~962, ~1003, ~1032
- Modify: `dw/result.py:305-313` (`file_base_name` handling in `Result.save`)
- Test: `tests/test_runs.py`

**Interfaces:**
- Consumes: `step_subfolder` (Task 3), `validate_output_path` (already imported in `workflow.py`), `validate_file_base_name` (Task 2).
- Produces: `Workflow.step_output_dir(step_definition: dict) -> str` — the absolute directory this step writes into, created; equals `self.effective_output_dir` when the step has no subfolder. Raises `InvalidInputError` / `PathTraversalError` on a bad or escaping subfolder.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_runs.py` a helper beside `_workflow_definition()`:

```python
def _foldered_definition(subfolder="final"):
    definition = _workflow_definition()
    definition["steps"][0]["result"]["subfolder"] = subfolder
    return definition
```

and a new class:

```python
class TestSubfolders:
    def test_a_step_without_a_subfolder_writes_where_it_always_did(
        self, tmp_path, fake_pipeline
    ):
        from dw.workflow import Workflow

        Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        assert (run / "runs_test-gen0.0-0.0.png").is_file()
        assert not any(child.is_dir() for child in run.iterdir())

    def test_a_step_with_a_subfolder_writes_into_it(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        assert (run / "final" / "runs_test-gen0.0-0.0.png").is_file()
        assert not (run / "runs_test-gen0.0-0.0.png").exists()
        # The manifest still sits at the root of the run
        assert (run / "manifest.json").is_file()

    def test_a_nested_subfolder_is_created(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(
            _foldered_definition("shots/act-1"), str(tmp_path), "/w/workflows/Gyre.json"
        ).run({})
        (run,) = (tmp_path / "Gyre").iterdir()
        assert (run / "shots" / "act-1" / "runs_test-gen0.0-0.0.png").is_file()

    def test_step_output_dir_is_the_run_directory_without_a_subfolder(self, tmp_path):
        from dw.workflow import Workflow

        workflow = Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json")
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        step = workflow.workflow_definition["steps"][0]
        assert workflow.step_output_dir(step) == workflow.effective_output_dir

    def test_step_output_dir_refuses_an_escape_at_run_time(self, tmp_path):
        from dw.security import SecurityError
        from dw.workflow import Workflow

        workflow = Workflow(
            _foldered_definition("../other"), str(tmp_path), "/w/workflows/Gyre.json"
        )
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        with pytest.raises(SecurityError):
            workflow.step_output_dir(workflow.workflow_definition["steps"][0])

    def test_the_pipeline_wrapper_is_pointed_at_the_subfolder(self, tmp_path, fake_pipeline):
        # A chain step's save_segments spill writes through the pipeline's
        # output_dir, so it has to be the step's directory, not the run's
        from dw.workflow import Workflow

        workflow = Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json")
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        step = workflow.workflow_definition["steps"][0]
        action = workflow.create_step_action(step, {}, {}, 7, "cpu")
        assert action.output_dir == os.path.join(workflow.effective_output_dir, "final")

    def test_a_separator_in_file_base_name_is_refused_at_save(self, tmp_path):
        from dw.result import Result
        from dw.security import InvalidInputError

        result = Result({"content_type": "image/png", "file_base_name": "final/"})
        with pytest.raises(InvalidInputError, match="subfolder"):
            result.save(str(tmp_path), "w-step.0")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_runs.py -q -k TestSubfolders`
Expected: the "writes where it always did" test PASSES (that is the regression guard); every other test FAILS (`AttributeError: 'Workflow' object has no attribute 'step_output_dir'`, files landing at the root, `InvalidInputError` not raised).

- [ ] **Step 3: Add `step_output_dir`**

In `dw/workflow.py`, directly after the `effective_output_dir` property:

```python
    def step_output_dir(self, step_definition):
        """Where one step writes: the run directory, or the subfolder of it
        the step's result names.

        Computed here, once, rather than inside Result.save, because two
        things write on a step's behalf - Result.save for its results and
        the pipeline wrapper for a chain's save_segments spill - and both
        have to land in the same place. The shape was checked statically by
        validation_errors; it is checked again here for a definition that
        reached the engine without it, and containment (that the joined
        path is really inside the run directory) is checked on the join.
        """
        base = self.effective_output_dir
        subfolder = step_subfolder(step_definition)
        if not subfolder:
            return base
        target = validate_output_path(os.path.join(base, subfolder), base)
        os.makedirs(target, exist_ok=True)
        return target
```

Confirm `os` is imported in `dw/workflow.py` (it is used by `effective_output_dir`).

- [ ] **Step 4: Feed it to `Result.save` and to the pipeline wrappers**

In the step loop (~line 743), change:

```python
                    saved_files = result.save(
                        self.effective_output_dir, f"{workflow_id}-{step.name}.{i}"
                    )
```

to:

```python
                    saved_files = result.save(
                        self.step_output_dir(step_data),
                        f"{workflow_id}-{step.name}.{i}",
                    )
```

In `create_step_action`, replace each of the three `output_dir=self.effective_output_dir,` kwargs (the cached-pipeline reuse branch ~962, the fresh `Pipeline(...)` ~1003, and the `pipeline_reference` branch ~1032) with:

```python
                    output_dir=self.step_output_dir(step_definition),
```

(`step_definition` is the parameter name in `create_step_action`; keep each line's existing indentation.) Do not touch the `workflow` (sub-workflow) branch: a child's steps compute their own directories against the inherited run directory, and the parent's `subfolder` governs only what the parent saves from the child's return value — which the `result.save` change above already covers.

- [ ] **Step 5: Refuse a separator in `file_base_name` at save time**

In `dw/result.py`, change the import line to include the new validator:

```python
from .security import (
    SecurityError,
    validate_file_base_name,
    validate_output_path,
    validate_string_input,
)
```

and in `Result.save`, change:

```python
        if "file_base_name" in self.result_definition:
            custom_base = validate_string_input(
                self.result_definition["file_base_name"],
                max_length=MAX_BASE_NAME_LENGTH,
            )
            file_base_name = custom_base + validated_base_name
```

to:

```python
        if "file_base_name" in self.result_definition:
            custom_base = validate_file_base_name(
                validate_string_input(
                    self.result_definition["file_base_name"],
                    max_length=MAX_BASE_NAME_LENGTH,
                )
            )
            file_base_name = custom_base + validated_base_name
```

Note the check happens before the `save`/`content_type` early return? It does not need to — the early return is above; a `file_base_name` on a step that saves nothing is harmless. Leave the order as it is (early return first, then base name).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_runs.py tests/test_result.py tests/test_result_output_naming.py tests/test_chain.py tests/test_step_cache.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add dw/workflow.py dw/result.py tests/test_runs.py
git commit -m "feat(engine): result.subfolder places a step's files; step_output_dir feeds Result.save and the pipeline wrapper

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `subfolder` on the manifest entry and the `step_end` event

**Files:**
- Modify: `dw/workflow.py` — manifest entry assembly (~line 761) and `step_end_data` (~769)
- Test: `tests/test_runs.py`

**Interfaces:**
- Consumes: `step_subfolder` (Task 3).
- Produces: every manifest step entry carries `"subfolder": str` (`""` when none); every `step_end` event carries `subfolder=str`. Stage 2 and the UI read these.

- [ ] **Step 1: Write the failing tests**

Add to `TestSubfolders` in `tests/test_runs.py`:

```python
    def test_the_manifest_entry_carries_the_subfolder(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        manifest = json.loads((run / "manifest.json").read_text())
        (entry,) = manifest["steps"]
        assert entry["subfolder"] == "final"
        assert entry["files"] == ["final/runs_test-gen0.0-0.0.png"]

    def test_an_unfoldered_entry_carries_the_empty_string(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["steps"][0]["subfolder"] == ""

    def test_a_reused_entry_carries_the_definitions_subfolder(
        self, tmp_path, fake_pipeline
    ):
        from dw.workflow import Workflow

        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        first, second = sorted((tmp_path / "Gyre").iterdir())
        manifest = json.loads((second / "manifest.json").read_text())
        (entry,) = manifest["steps"]
        assert entry["reused"] is True
        assert entry["subfolder"] == "final"
        # The file is the first run's, absolute, inside its 'final'
        assert os.path.dirname(entry["files"][0]) == str(first / "final")

    def test_the_step_end_event_carries_the_subfolder(self, tmp_path, fake_pipeline):
        from dw.events import RunContext
        from dw.workflow import Workflow

        seen = []
        context = RunContext(on_event=seen.append)
        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}, previous_pipelines={}, context=context
        )
        step_ends = [event for event in seen if event.get("event") == "step_end"]
        assert step_ends and step_ends[0]["subfolder"] == "final"
```

(`RunContext(on_event=...)` and `run(..., context=context)` is the same setup `tests/test_events.py::_run` uses.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_runs.py -q -k "carries or step_end_event"`
Expected: FAIL — `KeyError: 'subfolder'`.

- [ ] **Step 3: Add the field in both places**

In the step loop of `dw/workflow.py`, change:

```python
                manifest_entry = {"step": step.name, "files": saved_files}
                if reused:
                    manifest_entry["reused"] = True
```

to:

```python
                subfolder = step_subfolder(step_data)
                manifest_entry = {
                    "step": step.name,
                    "files": saved_files,
                    "subfolder": subfolder,
                }
                if reused:
                    manifest_entry["reused"] = True
```

and:

```python
                step_end_data = {"files": saved_files}
                if reused:
                    step_end_data["reused"] = True
```

to:

```python
                step_end_data = {"files": saved_files, "subfolder": subfolder}
                if reused:
                    step_end_data["reused"] = True
```

- [ ] **Step 4: Run the tests to verify they pass, then the wider set**

Run: `python -m pytest tests/test_runs.py -q`
Expected: all PASS.

Run: `python -m pytest tests/ -q -x --ignore=tests/test_plugin_skills.py`
Expected: all PASS. If a server or MCP test asserts the exact key set of a manifest entry or a `step_end` payload, update that assertion to include `subfolder` — that is the field arriving, not a regression. Note which test needed it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow.py tests/test_runs.py
git commit -m "feat(engine): manifest entries and step_end carry subfolder

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `output:` reaches into a subfolder — the no-change claim, pinned by a test

**Files:**
- Test: `tests/test_runs.py`

**Interfaces:**
- Consumes: `resolve_output_reference` (`dw/runs.py`, unchanged).

- [ ] **Step 1: Write the test**

Add to `TestSubfolders` in `tests/test_runs.py` (import `resolve_output_reference` from `dw.runs` in the file's import block if it is not already there):

```python
    def test_an_output_reference_reaches_into_a_subfolder(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        resolved = resolve_output_reference(
            "output:Gyre/latest/final/runs_test-gen0.0-0.0.png", root=str(tmp_path)
        )
        assert resolved == str(run / "final" / "runs_test-gen0.0-0.0.png")
        assert resolve_output_reference(
            f"output:Gyre/{run.name}/final/runs_test-gen0.0-0.0.png", root=str(tmp_path)
        ) == resolved
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_runs.py -q -k reaches_into`
Expected: PASS on the first run — the resolver already walks every remaining segment. If it fails, stop: the design's no-change claim about `_resolve_segments` is wrong and needs a look before anything else, not a patch here.

- [ ] **Step 3: Commit**

```bash
git add tests/test_runs.py
git commit -m "test(runs): output: references reach a step's subfolder unchanged

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: CLAUDE.md gotcha and the design's status line

**Files:**
- Modify: `CLAUDE.md` (*Critical Gotchas*, after the *Run directories* bullet)
- Modify: `docs/proposals/output-folders.md:3` (status)

- [ ] **Step 1: Add the gotcha**

After the `**Run directories**` bullet in `CLAUDE.md`'s *Critical Gotchas*, add:

```markdown
- **Result subfolders**: a step's `result.subfolder` (`dw/subfolders.py`) puts its files
  in a subfolder of the run directory - `<run>/final/x.mp4` - by convention `final` or
  `intermediate`; the engine treats no name specially and there is no default.
  `Workflow.step_output_dir` computes the directory once and hands it to both
  `Result.save` and the pipeline wrapper, so a chain's `save_segments` spill follows it.
  Shape is `SUBFOLDER_PATTERN` (the `output:` segment rule, so every subfolder is
  `output:`-addressable), checked by `subfolder_errors` in `validation_errors` after
  `for_each` expansion and again at run time; containment is `validate_output_path`
  against the run directory. Manifest entries and `step_end` carry `subfolder`.
  `split_run_path` finds the run id anywhere in a path, so `strip_run_id` still groups a
  workflow's runs. `file_base_name` may not contain a separator - it is a name, not a path
```

- [ ] **Step 2: Update the design's status**

Change line 3 of `docs/proposals/output-folders.md` to:

```markdown
Status: **stage 1 (engine) implemented; stages 2-4 pending**. Written for MCP feedback ticket T016;
```

- [ ] **Step 3: Run the full suite once**

Run: `python -m pytest tests/ -q --ignore=tests/test_plugin_skills.py`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/proposals/output-folders.md
git commit -m "docs: result subfolders gotcha; output-folders design marks stage 1 done

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage (stage 1 items):** schema description → Task 4; `subfolder_errors` in `validation_errors` → Task 3; `step_output_dir` feeding `Result.save` and `create_step_action` → Task 5; `file_base_name` check, static and run time → Tasks 2, 3, 5; `subfolder` on manifest entry and `step_end` → Task 6; `split_run_path`/`strip_run_id` → Task 1; engine tests listed in the spec's *Tests* section — unfoldered identical (5), foldered lands (5), shape refusals (2), symlink/escape containment (5, `refuses_an_escape_at_run_time`), `subfolder_errors` path for literal and `item:` member (3), `file_base_name` static and run time (3, 5), `for_each` member per subfolder (3 covers validation; a run-level `item:` placement test is not included because the `noop` task fixture cannot save — the substitution path is the one `_rewrite` already walks, verified in the design review), chain `keep_segments` follows the subfolder (5, via the wrapper's `output_dir`), reused entry (6), parent's subfolder does not move a child's files (not tested — no code path changed for sub-workflows; the design states the semantics), `output:` resolves (7), `split_run_path` cases (1). `get_job` carrying the field needs no code (the server spreads `**entry`).

**Placeholders:** none; every step has its code. The `noop` task command and the `RunContext(on_event=...)` capture were checked against the schema and `tests/test_events.py` while writing.

**Type consistency:** `step_subfolder(step_definition) -> str`, `subfolder_errors(definition, source_indices) -> list[dict]`, `validate_subfolder(name) -> str`, `validate_file_base_name(name) -> str`, `split_run_path(path) -> (str, str, str)`, `Workflow.step_output_dir(step_definition) -> str` — used with these names and shapes in every task that references them.
