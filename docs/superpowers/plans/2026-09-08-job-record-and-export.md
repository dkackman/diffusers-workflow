# Job Record and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every run leaves a realized copy of the workflow that produced it beside its manifest; a server job knows which run is its own; an MCP client can read that workflow and export one job as a git-ready directory and a zip.

**Architecture:** A new pure module `dw/realize.py` walks a workflow definition and pins every mutable input (arguments folded into variable defaults, the drawn seed, stored prompt text inlined, `output:.../latest/...` rewritten to a concrete run id), returning a schema-clean copy plus annotations for the manifest. `Workflow.run` writes that copy as `workflow.json` in the run directory and emits a `run_start` event carrying the run id; the server's `JobManager` records that on the job and in `jobs.sqlite`, which lets it read the realized file back. On top of that record, `dw/server/exports.py` gathers one finished job into `<workspace>/exports/<job id>/` and a route streams the same tree as a zip; two MCP tools expose the read and the export.

**Tech Stack:** Python 3, FastAPI/Starlette, sqlite3, pytest, httpx `MockTransport` (MCP tests), the MCP Python SDK (`dw_mcp/server.py`).

**Spec:** `docs/superpowers/specs/2026-09-08-job-record-and-export-design.md`
(Context: `docs/proposals/job-record-and-export.md`, `docs/proposals/resume.md`.)

**Running tests:** the suite needs the project venv. Every pytest command in this
plan is written as `source ./activate && python -m pytest ...` and must be run
from the repo root (`/Users/don/src/dkackman/diffusers-workflow`).

## Global Constraints

Copied verbatim from the spec's "Global constraints", and implicitly part of
every task's requirements:

- Model knowledge stays out of engine code; nothing here names a model.
- Every path read or written goes through `dw/security.py` validators:
  `validate_output_path`, `validate_path`, the asset and output resolvers.
- The realized file must validate against `dw/workflow_schema.json`
  unchanged: annotations that the schema would reject live in the manifest.
- Writing the realized file and the manifest is best effort: a run that
  produced its files has succeeded whether or not the record landed
  (`write_manifest`'s rule).
- Nothing bundled requires `--trust-workflows`.
- `exports` is a reserved workspace name, beside `workflows`, `prompts`,
  `assets`, `outputs`.

Plus the repo rules in `CLAUDE.md` that bear on this work: never `eval()`,
`exec()` or `shell=True`; path traversal is blocked; schema validation runs
before variable substitution.

## File Structure

**Created**

- `dw/realize.py` — the realization rules. Pure: no I/O beyond the resolvers it
  calls, never mutates its input, never raises for an unresolvable reference.
- `dw/server/exports.py` — gathering one finished job into a directory.
- `dw_mcp/exports.py` — the MCP handler over the export route.
- `tests/test_realize.py`, `tests/test_server_jobs.py`,
  `tests/test_server_exports.py`, `tests/test_mcp_exports.py`.

**Modified**

- `dw/runs.py` — `REALIZED_FILE_NAME`, `write_realized_workflow`.
- `dw/workflow.py` — call realization at run start, emit `run_start`, carry the
  two new manifest keys.
- `dw/workspace.py` — `EXPORTS_SUBDIR`, reserved name, `_foreign_entries`.
- `dw/server/jobs.py` — `Job.run_id`/`run_dir`, the sqlite columns,
  `JobManager.realized`.
- `dw/server/app.py` — the `realized` flag on the workflow route, the export
  route, the zip route.
- `dw_mcp/diagnose.py`, `dw_mcp/client.py`, `dw_mcp/server.py` — the two tools.
- Docs and plugin skills (Task 7).

---

### Task 1: Realization module (`sonnet`)

*Model rationale: one new file, but the semantics span four resolver modules
(`variables`, `prompts`, `runs`, `security`) and the tree walk must be exactly
right — integration judgement, not typing.*

**Files:**
- Create: `dw/realize.py`
- Test: `tests/test_realize.py`

**Interfaces:**
- Consumes: `dw.variables.set_variables(values, variables)` (mutates
  `variables` in place); `dw.prompts.fetch_prompt(reference, prompt_dir=None,
  base_dir=None) -> str` and `dw.prompts.PROMPT_PREFIX`;
  `dw.runs.resolve_output_reference(reference, root=None) -> str`,
  `dw.runs.output_root() -> str`, `dw.runs.OUTPUT_PREFIX`, `dw.runs.LATEST`,
  `dw.runs.is_output_reference(value)`;
  `dw.security.validate_workflow_path(path, confine_to)`,
  `dw.security.SecurityError`.
- Produces: `realize_workflow(definition, arguments, seed, base_dir=None,
  prompt_dir=None, output_root=None, workflow_dir=None) -> (dict, dict)`.
  The second element is
  `{"prompts": [str, ...], "sub_workflows": {str: str | None}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_realize.py`:

```python
"""Realization: a copy of a workflow with every mutable input pinned, so the
file beside a run's manifest reproduces that run whatever changes later."""

import copy
import hashlib
import json
import os

import pytest

from dw.realize import realize_workflow
from dw.runs import new_run_id
from dw.schema import load_schema, validate_data


def definition():
    return {
        "id": "realize_test",
        "variables": {"prompt": "a default", "steps": 25},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {
                        "prompt": "variable:prompt",
                        "num_inference_steps": "variable:steps",
                    },
                },
            }
        ],
    }


@pytest.fixture
def prompt_library(tmp_path):
    library = tmp_path / "prompts"
    (library / "scenic").mkdir(parents=True)
    (library / "scenic" / "dusk.json").write_text(
        json.dumps({"text": "a harbour at dusk"})
    )
    return str(library)


@pytest.fixture
def output_root(tmp_path):
    """An output root holding one finished run of 'ltx2/Gyre'."""
    root = tmp_path / "outputs"
    run_id = new_run_id({"a": 1})
    run = root / "ltx2" / "Gyre" / run_id
    run.mkdir(parents=True)
    (run / "still.png").write_bytes(b"not really a png")
    return str(root), run_id


class TestVariablesAndSeed:
    def test_arguments_become_the_variable_defaults(self):
        realized, _ = realize_workflow(
            definition(), {"prompt": "a cat", "steps": 4}, 7
        )
        assert realized["variables"] == {"prompt": "a cat", "steps": 4}

    def test_variable_references_are_left_alone(self):
        realized, _ = realize_workflow(definition(), {"prompt": "a cat"}, 7)
        arguments = realized["steps"][0]["pipeline"]["arguments"]
        assert arguments["prompt"] == "variable:prompt"

    def test_the_seed_is_written_even_when_the_definition_had_none(self):
        realized, _ = realize_workflow(definition(), {}, 991)
        assert realized["seed"] == 991

    def test_the_input_definition_is_not_mutated(self):
        original = definition()
        before = copy.deepcopy(original)
        realize_workflow(original, {"prompt": "a cat"}, 7)
        assert original == before


class TestPrompts:
    def test_a_stored_prompt_is_inlined_and_annotated(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"

        realized, annotations = realize_workflow(
            source, {}, 7, prompt_dir=prompt_library
        )

        arguments = realized["steps"][0]["pipeline"]["arguments"]
        assert arguments["prompt"] == "a harbour at dusk"
        assert annotations["prompts"] == ["scenic/dusk"]

    def test_a_name_is_annotated_once_in_first_seen_order(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"
        source["steps"][0]["pipeline"]["arguments"]["negative_prompt"] = (
            "prompt:scenic/dusk"
        )

        _, annotations = realize_workflow(source, {}, 7, prompt_dir=prompt_library)

        assert annotations["prompts"] == ["scenic/dusk"]

    def test_an_unresolvable_prompt_is_left_as_written(self, prompt_library):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:missing"

        realized, annotations = realize_workflow(
            source, {}, 7, prompt_dir=prompt_library
        )

        assert realized["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "prompt:missing"
        )
        assert annotations["prompts"] == []


class TestOutputReferences:
    def test_latest_is_pinned_to_the_run_it_resolved_to(self, output_root):
        root, run_id = output_root
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = (
            "output:ltx2/Gyre/latest/still.png"
        )

        realized, _ = realize_workflow(source, {}, 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == (
            f"output:ltx2/Gyre/{run_id}/still.png"
        )

    def test_an_explicit_run_id_is_kept_as_written(self, output_root):
        root, run_id = output_root
        written = f"output:ltx2/Gyre/{run_id}/still.png"
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = written

        realized, _ = realize_workflow(source, {}, 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == written

    def test_an_unresolvable_output_is_left_as_written(self, output_root):
        root, _ = output_root
        written = "output:ltx2/Nothing/latest/still.png"
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["image"] = written

        realized, _ = realize_workflow(source, {}, 7, output_root=root)

        assert realized["steps"][0]["pipeline"]["arguments"]["image"] == written


class TestReferencesThatAreKept:
    @pytest.mark.parametrize(
        "value",
        [
            "asset:iris.png",
            "constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES",
            "previous_result:gen",
        ],
    )
    def test_kept_verbatim(self, value):
        source = definition()
        source["steps"][0]["pipeline"]["arguments"]["thing"] = value

        realized, _ = realize_workflow(source, {}, 7)

        assert realized["steps"][0]["pipeline"]["arguments"]["thing"] == value


class TestSubWorkflows:
    def test_a_local_path_is_kept_and_digested(self, tmp_path):
        tree = tmp_path / "workflows"
        (tree / "steps").mkdir(parents=True)
        child = tree / "steps" / "upscale.json"
        child.write_text(json.dumps({"id": "child", "steps": []}))

        source = definition()
        source["steps"].append(
            {"name": "up", "workflow": {"path": "steps/upscale.json"}}
        )

        realized, annotations = realize_workflow(
            source, {}, 7, base_dir=str(tree), workflow_dir=str(tree)
        )

        assert realized["steps"][1]["workflow"]["path"] == "steps/upscale.json"
        digest = annotations["sub_workflows"]["steps/upscale.json"]
        assert digest == hashlib.sha256(child.read_bytes()).hexdigest()

    def test_an_unreadable_sub_workflow_digests_to_null(self, tmp_path):
        tree = tmp_path / "workflows"
        tree.mkdir()
        source = definition()
        source["steps"].append({"name": "up", "workflow": {"path": "gone.json"}})

        _, annotations = realize_workflow(
            source, {}, 7, base_dir=str(tree), workflow_dir=str(tree)
        )

        assert annotations["sub_workflows"] == {"gone.json": None}

    def test_a_builtin_is_not_digested(self):
        source = definition()
        source["steps"].append(
            {"name": "up", "workflow": {"path": "builtin:upscale.json"}}
        )

        realized, annotations = realize_workflow(source, {}, 7)

        assert realized["steps"][1]["workflow"]["path"] == "builtin:upscale.json"
        assert annotations["sub_workflows"] == {}


def test_the_realized_file_validates_against_the_schema(prompt_library):
    source = definition()
    source["steps"][0]["pipeline"]["arguments"]["prompt"] = "prompt:scenic/dusk"

    realized, _ = realize_workflow(
        source, {"steps": 4}, 991, prompt_dir=prompt_library
    )

    ok, message = validate_data(realized, load_schema("workflow"))
    assert ok, message
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ./activate && python -m pytest tests/test_realize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.realize'`.

- [ ] **Step 3: Write the module**

Create `dw/realize.py`:

```python
"""Realizing a workflow: a copy of a definition with every mutable input
pinned, so the file left beside a run's manifest reproduces that run however
the library, the catalog or the output tree change afterwards.

What "mutable" means here is precisely the set of things that can differ
between two runs of the same file: the arguments a caller passed, the seed a
seedless workflow drew, the text a stored prompt held at the time, and which
run 'output:.../latest/...' picked. Everything else - 'asset:', 'constant:',
'previous_result:', 'builtin:' and a sub-workflow's path - is a name whose
meaning is pinned by something already recorded (the asset library, the
manifest's dw_version, the file itself), so it is kept as written and, for a
local sub-workflow, digested into the manifest instead.

Two rules hold this module together. It never mutates its input: the caller
hands it the definition the run is about to work from. And it never fails a
run: a reference that will not resolve is left exactly as written, so the
engine raises its own error at the point it would have raised anyway.
"""

import copy
import hashlib
import logging
import os

from .prompts import PROMPT_PREFIX, fetch_prompt
from .runs import (
    LATEST,
    OUTPUT_PREFIX,
    is_output_reference,
    output_root as default_output_root,
    resolve_output_reference,
)
from .security import SecurityError, validate_workflow_path
from .variables import set_variables

logger = logging.getLogger("dw")

BUILTIN_PREFIX = "builtin:"


def realize_workflow(
    definition,
    arguments,
    seed,
    base_dir=None,
    prompt_dir=None,
    output_root=None,
    workflow_dir=None,
):
    """A copy of `definition` with every mutable input pinned.

    Args:
        definition: The workflow as loaded, before Workflow.run's deep copy.
            Never mutated.
        arguments: The run's argument dict, folded into the variable defaults
            exactly as `set_variables` folds them for the run itself.
        seed: The seed the run resolved - an integer, never None, because
            `Workflow.run` draws one when the workflow names none.
        base_dir: The workflow file's directory, anchoring prompt discovery
            and a sub-workflow's relative path.
        prompt_dir: The prompt library, for `prompt:` inlining.
        output_root: The output directory `output:` names resolve against.
        workflow_dir: The root a sub-workflow path is confined to, as
            `Workflow` confines it; None for an unconfined CLI run.

    Returns:
        (realized, annotations) - the pinned copy, and
        {"prompts": [name, ...], "sub_workflows": {path: sha256 or None}}
        for the manifest to carry, since the schema has nowhere to put them.
    """
    annotations = {"prompts": [], "sub_workflows": {}}
    realized = copy.deepcopy(definition)

    variables = realized.get("variables")
    if isinstance(variables, dict):
        # Exactly what the run computed: set_variables coerces each value to
        # the type of the declared default and rejects an undeclared name
        set_variables(arguments or {}, variables)

    realized["seed"] = seed
    realized = _pin(realized, annotations, base_dir, prompt_dir, output_root)
    _record_sub_workflows(
        realized.get("steps"), annotations, base_dir, workflow_dir
    )
    return realized, annotations


def _pin(value, annotations, base_dir, prompt_dir, output_root):
    """Rebuild a value with prompt and output references pinned.

    One recursive function over dicts, lists and strings - the shape
    `referenced_result_names` in dw/step_cache.py walks - so a reference is
    found wherever it sits: a pipeline argument, a task argument, a
    sub-workflow's argument map, an element of a list.
    """
    if isinstance(value, str):
        if value.startswith(PROMPT_PREFIX):
            return _inline_prompt(value, annotations, prompt_dir, base_dir)
        if is_output_reference(value):
            return _pin_output(value, output_root)
        return value
    if isinstance(value, dict):
        return {
            key: _pin(item, annotations, base_dir, prompt_dir, output_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _pin(item, annotations, base_dir, prompt_dir, output_root)
            for item in value
        ]
    return value


def _inline_prompt(reference, annotations, prompt_dir, base_dir):
    """The stored text, and the name recorded for the manifest.

    A stored prompt's text may not itself begin with a reference prefix (an
    engine rule `fetch_prompt` enforces), so inlining cannot introduce a
    second resolution.
    """
    try:
        text = fetch_prompt(reference, prompt_dir, base_dir)
    except (SecurityError, OSError, ValueError) as e:
        logger.warning(f"Realization kept {reference} as written: {e}")
        return reference
    name = reference.removeprefix(PROMPT_PREFIX).strip()
    if name not in annotations["prompts"]:
        annotations["prompts"].append(name)
    return text


def _pin_output(reference, output_root):
    """'output:<identity>/latest/<file>' rewritten to the run it resolved to.

    An explicit run id is already pinned, so it is returned untouched without
    touching the disk - realizing must not fail on a reference the run has
    not reached yet.
    """
    name = reference.removeprefix(OUTPUT_PREFIX).strip()
    if LATEST not in name.split("/"):
        return reference
    root = output_root or default_output_root()
    try:
        resolved = resolve_output_reference(reference, root)
        relative = os.path.relpath(resolved, root).replace(os.sep, "/")
    except (SecurityError, OSError, ValueError) as e:
        logger.warning(f"Realization kept {reference} as written: {e}")
        return reference
    return f"{OUTPUT_PREFIX}{relative}"


def _record_sub_workflows(steps, annotations, base_dir, workflow_dir):
    """Digest every sub-workflow a step names by local path.

    The schema's 'workflow' step takes a path, not a definition, so the
    realized file keeps the path and the manifest records what the file held.
    A builtin is packaged with the engine and pinned by the manifest's
    dw_version, so it is not digested.
    """

    def scan(value):
        if isinstance(value, dict):
            reference = value.get("workflow")
            if isinstance(reference, dict):
                path = reference.get("path")
                if isinstance(path, str) and not path.startswith(BUILTIN_PREFIX):
                    annotations["sub_workflows"][path] = _digest(
                        path, base_dir, workflow_dir
                    )
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    for step in steps or []:
        scan(step)


def _digest(path, base_dir, workflow_dir):
    """The SHA-256 of a sub-workflow file, or None when it cannot be read.

    Resolved the way `Workflow.create_step_action` resolves it - relative to
    the referencing file's directory, then through `validate_workflow_path`
    confined to `workflow_dir` - so a path this run could not have loaded is
    not one realization reads either.
    """
    try:
        candidate = (
            path
            if os.path.isabs(path)
            else os.path.normpath(os.path.join(base_dir or ".", path))
        )
        validated = validate_workflow_path(candidate, workflow_dir)
        with open(validated, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except (SecurityError, OSError, ValueError) as e:
        logger.debug(f"No digest for sub-workflow {path}: {e}")
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ./activate && python -m pytest tests/test_realize.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add dw/realize.py tests/test_realize.py
cat > /tmp/dw-commit-1.txt <<'EOF'
Realization: pin a workflow's mutable inputs into a runnable copy

realize_workflow folds the run's arguments into the variable defaults,
writes the seed the run resolved, inlines stored prompt text and rewrites
'output:.../latest/...' to the run id it picked. asset:, constant:,
previous_result: and builtin: are kept; a local sub-workflow path is kept
and digested into the annotations the manifest will carry.

Never mutates its input and never fails a run: a reference that will not
resolve is left exactly as written, so the engine raises its own error.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-1.txt
```

---

### Task 2: Write the realized file from `Workflow.run` (`sonnet`)

*Model rationale: touches the engine's run loop, `dw/runs.py` and the manifest
shape at once, and the placement inside `run`'s try/finally has to be exactly
right — multi-file integration.*

**Files:**
- Modify: `dw/runs.py` (add `REALIZED_FILE_NAME`, `write_realized_workflow`
  after `write_manifest`)
- Modify: `dw/workflow.py` (imports; `Workflow.run` ~line 405-425;
  `_write_run_manifest` ~line 668)
- Test: `tests/test_runs.py`

**Interfaces:**
- Consumes: `dw.realize.realize_workflow(definition, arguments, seed,
  base_dir=None, prompt_dir=None, output_root=None, workflow_dir=None) ->
  (dict, dict)` from Task 1.
- Produces: `dw.runs.REALIZED_FILE_NAME == "workflow.json"`;
  `dw.runs.write_realized_workflow(run_dir, realized) -> str | None`; a
  manifest whose `workflow` block carries `realized`, `prompts` and
  `sub_workflows`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runs.py` (a new class at the end of the file):

```python
class TestRealizedWorkflow:
    def test_the_run_directory_holds_a_realized_copy(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        definition = _workflow_definition()
        definition["variables"] = {"prompt": "a default"}
        definition["steps"][0]["pipeline"]["arguments"]["prompt"] = "variable:prompt"
        Workflow(definition, str(tmp_path), "/w/workflows/ltx2/Gyre.json").run(
            {"prompt": "a cat"}
        )

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        realized = json.loads((run_dir / "workflow.json").read_text())
        assert realized["variables"] == {"prompt": "a cat"}
        assert realized["seed"] == 7
        # the reference stays, so the file is still runnable with overrides
        assert realized["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "variable:prompt"
        )

    def test_the_manifest_points_at_it(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/ltx2/Gyre.json"
        ).run({})

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["workflow"]["realized"] == "workflow.json"
        assert manifest["workflow"]["prompts"] == []
        assert manifest["workflow"]["sub_workflows"] == {}

    def test_a_seedless_run_pins_the_seed_it_drew(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        definition = _workflow_definition()
        del definition["seed"]
        Workflow(definition, str(tmp_path), "/w/workflows/ltx2/Gyre.json").run({})

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        realized = json.loads((run_dir / "workflow.json").read_text())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert isinstance(realized["seed"], int)
        assert realized["seed"] == manifest["seed"]

    def test_the_flat_layout_writes_none(self, tmp_path, fake_pipeline, monkeypatch):
        from dw.workflow import Workflow

        monkeypatch.setenv(OUTPUT_LAYOUT_ENV_VAR, FLAT_LAYOUT)
        Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/ltx2/Gyre.json"
        ).run({})

        assert not list(tmp_path.rglob("workflow.json"))

    def test_writing_it_is_best_effort(self, tmp_path):
        from dw.runs import write_realized_workflow

        # a run directory that cannot be made - the run still succeeded
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        assert write_realized_workflow(str(blocker / "run"), {"id": "x"}) is None

    def test_writing_it_returns_the_path(self, tmp_path):
        from dw.runs import write_realized_workflow

        path = write_realized_workflow(str(tmp_path / "run"), {"id": "x"})
        assert path == str(tmp_path / "run" / "workflow.json")
        assert json.loads(open(path).read()) == {"id": "x"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ./activate && python -m pytest tests/test_runs.py::TestRealizedWorkflow -v`
Expected: FAIL — `ImportError: cannot import name 'write_realized_workflow'`
and `KeyError: 'realized'`.

- [ ] **Step 3: Add `write_realized_workflow` to `dw/runs.py`**

Add beside `MANIFEST_FILE_NAME` (after the `MANIFEST_FILE_NAME = "manifest.json"`
line):

```python
# The workflow a run actually ran, written beside its manifest. Named
# 'workflow.json' rather than something run-specific because the directory
# already says which run it is, and 'python -m dw.run workflow.json' from
# inside it is the whole reproduction story
REALIZED_FILE_NAME = "workflow.json"
```

Add after `write_manifest`:

```python
def write_realized_workflow(run_dir, realized):
    """Leave the workflow that produced a run beside what it produced.

    The submitted definition says what was asked for; this says what ran -
    arguments folded in, the seed pinned, stored prompts inlined,
    'output:latest' resolved. Best effort, exactly like write_manifest: a run
    that produced its files has succeeded whether or not this lands.
    """
    path = os.path.join(run_dir, REALIZED_FILE_NAME)
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(path, "w") as file:
            json.dump(realized, file, indent=2, default=str)
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"Could not write {path}: {e}")
        return None
    return path
```

- [ ] **Step 4: Call it from `Workflow.run`**

In `dw/workflow.py`, extend the `.runs` import block (currently
`FLAT_LAYOUT, activate_output_root, ...`) with the two new names, keeping
alphabetical order within the block:

```python
from .runs import (
    FLAT_LAYOUT,
    REALIZED_FILE_NAME,
    activate_output_root,
    deactivate_output_root,
    workflow_identity,
    manifest_relative_files,
    new_run_id,
    output_layout,
    run_directory,
    write_manifest,
    write_realized_workflow,
)
from .realize import realize_workflow
```

`REALIZED_FILE_NAME` is imported here because the manifest records the name.

In `Workflow.run`, beside the other pre-`try` locals (next to
`resolved_seed = None`), add:

```python
        # What the run recorded about itself, read by _write_run_manifest in
        # the finally below - initialized here so a failure before the run
        # directory exists still writes a well-formed manifest
        realized_name = None
        annotations = {"prompts": [], "sub_workflows": {}}
```

Then, immediately after the `if not self._run_dir_inherited:` block that sets
`self._run_dir` (just before `# Initialize collections for sharing state
between steps`), insert:

```python
            # The record of what actually ran, written before the first step
            # so a crash or a cancel still leaves it. A sub-workflow inherits
            # the parent's directory and writes none of its own, as with the
            # manifest, and the flat layout has no directory to write into
            if self._run_dir and not self._run_dir_inherited:
                try:
                    realized, annotations = realize_workflow(
                        self.workflow_definition,
                        arguments,
                        default_seed,
                        base_dir=base_dir,
                        output_root=self.output_dir,
                        workflow_dir=self.workflow_dir,
                    )
                    if write_realized_workflow(self._run_dir, realized):
                        realized_name = REALIZED_FILE_NAME
                except Exception as e:
                    # Never fatal: the record is worth less than the run
                    logger.warning(
                        f"Could not realize workflow {workflow_id}: {e}"
                    )
```

Note the arguments: `self.workflow_definition` (the original, before `run`'s
deep copy), the local `base_dir` computed earlier in `run`, `default_seed`
(which at this point is `resolved_seed`), and `self.output_dir` as the output
root. `prompt_dir` is left to `fetch_prompt`'s own discovery, which is what the
run itself uses.

- [ ] **Step 5: Carry the annotations into the manifest**

In `dw/workflow.py`, change the `finally` block's call:

```python
            if self._run_dir and not self._run_dir_inherited:
                self._write_run_manifest(
                    run_id,
                    status,
                    started_at,
                    arguments,
                    resolved_seed,
                    realized_name,
                    annotations,
                )
```

and the method:

```python
    def _write_run_manifest(
        self,
        run_id,
        status,
        started_at,
        arguments,
        seed,
        realized_name=None,
        annotations=None,
    ):
```

with its `workflow` block becoming:

```python
                "workflow": {
                    "id": self.name,
                    "file": self.file_spec,
                    "identity": workflow_identity(self.file_spec, self.name),
                    # The realized copy beside this manifest, or null when
                    # writing it did not land - the manifest is the only
                    # place that difference is visible
                    "realized": realized_name,
                    # Annotations the schema has nowhere to put: which
                    # stored prompts were inlined, and what each local
                    # sub-workflow file held when it ran
                    "prompts": (annotations or {}).get("prompts", []),
                    "sub_workflows": (annotations or {}).get("sub_workflows", {}),
                },
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `source ./activate && python -m pytest tests/test_runs.py tests/test_realize.py tests/test_workflow.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add dw/runs.py dw/workflow.py tests/test_runs.py
cat > /tmp/dw-commit-2.txt <<'EOF'
Every run writes its realized workflow beside its manifest

Workflow.run realizes the definition once the seed and the run directory
have settled and writes it as workflow.json before the first step, so a
crash or a cancel still leaves it. The manifest's workflow block gains
'realized', 'prompts' and 'sub_workflows'.

Best effort throughout: a failed realization warns and the run continues.
A sub-workflow inherits the parent's directory and writes none of its own;
the flat layout has no run directory and so writes none either.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-2.txt
```

---

### Task 3: The job knows its run (`sonnet`)

*Model rationale: one event in the engine, a schema migration and two new
fields in the job store, and a route flag — four files that only make sense
together.*

**Files:**
- Modify: `dw/workflow.py` (emit `run_start` after the realization block)
- Modify: `dw/server/jobs.py` (imports; migration ~line 77-115; `record`,
  `recent_summaries`, `get`, `_to_detail` ~line 118-300; `Job.__init__`,
  `summary`, `detail` ~line 306-375; new `JobManager.realized`; the progress
  branch of `_consume_results` ~line 800-830)
- Modify: `dw/server/app.py` (`GET /api/jobs/{job_id}/workflow` ~line 834)
- Test: `tests/test_server_jobs.py` (new)

**Interfaces:**
- Consumes: `dw.runs.REALIZED_FILE_NAME`, `dw.runs.workflow_identity(file_spec,
  workflow_id)` from Task 2.
- Produces: a `progress` event `{"event": "run_start", "run_id": str,
  "identity": str, "run_dir": str}`; `Job.run_id`, `Job.run_dir`;
  `JobManager.realized(job_id) -> dict | None`; `GET
  /api/jobs/{id}/workflow` answering `{"id", "definition", "realized": bool}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_jobs.py`:

```python
"""The job store's record of which run a job was: the run_start event, the
two columns, and reading the realized workflow back off disk.

The manager tests proper live in tests/test_server.py; this file is the run
record, which is a store concern rather than a routing one.
"""

import json
import time

import pytest

from dw.runs import REALIZED_FILE_NAME, new_run_id
from dw.server.jobs import TERMINAL_STATES, JobHistory, JobManager

from .test_server import ScriptedWorkerManager, valid_workflow


RUN_ID = new_run_id({"workflow": "spec"})
RUN_DIR = f"server_test/{RUN_ID}"


def tracked_script(command):
    """A worker that reports its run before doing anything else - what
    Workflow.run emits once the run directory is chosen."""
    yield {
        "type": "progress",
        "event": "run_start",
        "run_id": RUN_ID,
        "identity": "server_test",
        "run_dir": RUN_DIR,
    }
    yield {"type": "success", "message": "ok", "run_count": 1, "manifest": []}


@pytest.fixture
def manager(tmp_path):
    made = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(tracked_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(tmp_path),
    )
    yield made
    made.shutdown()


def finished_job(manager):
    job = manager.submit(workflow=valid_workflow(), base_dir=None)
    deadline = time.time() + 5
    while job.status not in TERMINAL_STATES and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "succeeded", job.error
    return job


def test_run_start_populates_the_job(manager):
    job = finished_job(manager)
    assert job.run_id == RUN_ID
    assert job.run_dir == RUN_DIR
    assert job.summary()["run_id"] == RUN_ID
    assert job.detail()["run_dir"] == RUN_DIR


def test_both_persist_and_read_back(manager):
    job = finished_job(manager)
    historical = manager.history.get(job.id)
    assert historical["run_id"] == RUN_ID
    assert historical["run_dir"] == RUN_DIR


def test_realized_reads_the_file_the_run_wrote(manager, tmp_path):
    job = finished_job(manager)
    run_dir = tmp_path / "outputs" / "server_test" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / REALIZED_FILE_NAME).write_text(json.dumps({"id": "realized"}))

    assert manager.realized(job.id) == {"id": "realized"}


def test_realized_is_none_without_the_file(manager):
    job = finished_job(manager)
    assert manager.realized(job.id) is None


def test_realized_is_none_for_a_pre_tracking_row(manager):
    """A job recorded before run tracking has no run_dir, and the manager
    does not guess one from file paths."""
    job = finished_job(manager)
    with manager.history._connect() as connection:
        connection.execute(
            "UPDATE jobs SET run_id = NULL, run_dir = NULL WHERE id = ?", (job.id,)
        )
    manager.jobs.pop(job.id)

    assert manager.realized(job.id) is None


def test_realized_refuses_a_run_dir_that_escapes_the_output_root(manager, tmp_path):
    job = finished_job(manager)
    job.run_dir = "../../etc"
    assert manager.realized(job.id) is None


def test_a_database_without_the_columns_is_migrated(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.sqlite")
    # A store written before run tracking: the ALTER is the whole migration
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
            " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
            " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
        )
        connection.execute(
            "INSERT INTO jobs (id, status) VALUES ('old-1', 'succeeded')"
        )

    history = JobHistory(path)
    row = history.get("old-1")
    assert row["run_id"] is None and row["run_dir"] is None
```

Also add to `tests/test_server.py`, beside the other workflow-route tests:

```python
def test_job_workflow_reports_whether_it_is_realized(server):
    """A job with no run record falls back to the submitted definition and
    says so - the flag is what tells a client which it is looking at."""
    with server(success_script) as client:
        submitted = client.post(
            "/api/jobs", json={"workflow": valid_workflow(), "arguments": {}}
        ).json()
        wait_for_status(client, submitted["id"], TERMINAL_STATES)
        body = client.get(f"/api/jobs/{submitted['id']}/workflow").json()

    assert body["realized"] is False
    assert body["definition"]["id"] == "server_test"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ./activate && python -m pytest tests/test_server_jobs.py "tests/test_server.py::test_job_workflow_reports_whether_it_is_realized" -v`
Expected: FAIL — `AttributeError: 'Job' object has no attribute 'run_id'` and
`KeyError: 'realized'`.

- [ ] **Step 3: Emit `run_start` from `Workflow.run`**

In `dw/workflow.py`, immediately after the realization block added in Task 2
(still inside `if self._run_dir and not self._run_dir_inherited:`, at the same
indentation as the `try:` it contains), add:

```python
                # Which run this is, so a server job can find the directory
                # it wrote. Emitted even when the realized file did not land:
                # the manifest is still there, and so are the files
                run_context.emit(
                    "run_start",
                    run_id=run_id,
                    identity=workflow_identity(self.file_spec, workflow_id),
                    run_dir=os.path.relpath(
                        self._run_dir, self.output_dir
                    ).replace(os.sep, "/"),
                )
```

The worker already forwards every emitted event as a `progress` message, so no
worker change is needed.

- [ ] **Step 4: Add the fields, the columns and `realized()` to `dw/server/jobs.py`**

Extend the security import block:

```python
from ..security import (
    SecurityError,
    validate_json_size,
    validate_output_path,
    validate_path,
    validate_workflow_path,
)
from ..runs import REALIZED_FILE_NAME
```

In `JobHistory.__init__`, after the `workflow_name` migration:

```python
            # Which run of the workflow this job was - the directory under the
            # output root that holds its manifest and its realized workflow.
            # NULL for every row predating run tracking, and the manager
            # refuses to guess one from file paths
            if "run_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN run_id TEXT")
            if "run_dir" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN run_dir TEXT")
```

In `record`, extend the column list, the placeholders and the values tuple:

```python
                "INSERT OR REPLACE INTO jobs (id, workflow, status, created_at,"
                " started_at, finished_at, arguments, spec, manifest, warnings,"
                " error, events, workspace, workflow_name, run_id, run_dir) VALUES"
                " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
```

and, after `job.catalog_name` in the tuple:

```python
                    job.run_id,
                    job.run_dir,
```

In `recent_summaries`, add `run_id` to the SELECT and the dict, so a live
summary and a historical one keep the same shape:

```python
        query = (
            "SELECT id, workflow, status, created_at, started_at, finished_at,"
            " workspace, workflow_name, run_id FROM jobs"
        )
```

```python
                "workflow_name": row[7],
                "run_id": row[8],
                "historical": True,
```

In `get`, extend the SELECT:

```python
                "SELECT id, workflow, status, created_at, started_at, finished_at,"
                " arguments, spec, manifest, warnings, error, workspace,"
                " workflow_name, run_id, run_dir FROM jobs WHERE id = ?",
```

and in `_to_detail`, after `"workflow_name": row[12],`:

```python
            "run_id": row[13],
            "run_dir": row[14],
```

In `Job.__init__`, after `self.traceback = None`:

```python
        # Which run this job turned out to be - reported by the worker's
        # run_start event, unknown until then and forever for a job that
        # never got that far
        self.run_id = None
        self.run_dir = None
```

In `Job.summary`, after the `"workspace"` entry:

```python
            "run_id": self.run_id,
```

In `Job.detail`, after `"event_count": len(self.events),`:

```python
            "run_dir": self.run_dir,
```

In `JobManager._consume_results`, inside the `if message_type == "progress":`
branch, between building `event` and `job.add_event(event)`:

```python
                if event.get("event") == "run_start":
                    job.run_id = event.get("run_id")
                    job.run_dir = event.get("run_dir")
```

And add `realized` beside `definition`:

```python
    def realized(self, job_id):
        """The realized workflow a job ran, or None when the job predates
        run tracking or its run directory no longer holds the file.

        Read from the job's own output directory, not the manager's: one
        server holds several workspaces, and a job carries the root it ran
        against. The join is confined to that root, so a run_dir read back
        out of the database cannot name anything outside it.
        """
        job = self.jobs.get(job_id)
        if job is not None:
            run_dir = job.run_dir
            output_dir = job.spec.get("output_dir") or self.output_dir
        else:
            historical = self.history.get(job_id)
            if historical is None:
                return None
            run_dir = historical.get("run_dir")
            output_dir = (
                historical.get("spec") or {}
            ).get("output_dir") or self.output_dir
        if not run_dir:
            return None
        try:
            root = validate_output_path(output_dir, None)
            path = validate_path(
                os.path.join(root, run_dir, REALIZED_FILE_NAME), root
            )
            validate_json_size(path)
            with open(path, "r") as file:
                return json.load(file)
        except (SecurityError, OSError, ValueError) as e:
            logger.debug(f"No realized workflow for job {job_id}: {e}")
            return None
```

- [ ] **Step 5: Add the flag to the route**

In `dw/server/app.py`, replace the body of `get_job_workflow`:

```python
    @app.get("/api/jobs/{job_id}/workflow")
    def get_job_workflow(job_id: str):
        """The workflow this job ran, for the read-only graph on the job page
        and for `get_job_workflow` over MCP.

        `realized: true` means every mutable input is pinned - the copy the
        run itself wrote. `false` means the job predates run tracking (or its
        run directory is gone) and this is the definition as submitted. 404
        when neither is readable - the job itself still is."""
        if manager.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        realized = manager.realized(job_id)
        definition = realized if realized is not None else manager.definition(job_id)
        if definition is None:
            raise HTTPException(
                status_code=404, detail="No workflow definition for this job"
            )
        return {
            "id": job_id,
            "definition": definition,
            "realized": realized is not None,
        }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `source ./activate && python -m pytest tests/test_server_jobs.py tests/test_server.py tests/test_server_workspaces.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add dw/workflow.py dw/server/jobs.py dw/server/app.py tests/test_server_jobs.py tests/test_server.py
cat > /tmp/dw-commit-3.txt <<'EOF'
A job knows which run it was, and can read that run's workflow

Workflow.run emits a run_start event carrying the run id, the workflow
identity and the run directory relative to the output root. The Job records
both, jobs.sqlite gains run_id and run_dir (ALTER when absent, NULL for
older rows), and JobManager.realized reads workflow.json back out of the
job's own output root, confined to it.

GET /api/jobs/{id}/workflow now answers {id, definition, realized}: the
realized copy when there is one, the submitted definition otherwise.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-3.txt
```

---

### Task 4: `get_job_workflow` MCP tool (`haiku`)

*Model rationale: one handler, one registration, one test file — mechanical,
with the complete code below.*

**Files:**
- Modify: `dw_mcp/diagnose.py` (after `get_job`)
- Modify: `dw_mcp/server.py` (the diagnose group, ~line 605-660)
- Test: `tests/test_mcp_diagnose.py`

**Interfaces:**
- Consumes: `GET /api/jobs/{id}/workflow` answering `{"id", "definition",
  "realized": bool}` from Task 3; `dw_mcp.client.api_path(*segments)`;
  `DwClient.get_json(path, params=None)`.
- Produces: `dw_mcp.diagnose.get_job_workflow(client, job_id) -> {"job_id",
  "realized", "workflow", "next"}`, registered as the `get_job_workflow` MCP
  tool with `READ_ONLY` annotations.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_diagnose.py`:

```python
class TestGetJobWorkflow:
    def test_a_realized_workflow_comes_back_with_the_flag_set(self):
        client, seen = scripted(
            {
                ("GET", "/api/jobs/job-1/workflow"): (
                    200,
                    {"id": "job-1", "definition": WORKFLOW, "realized": True},
                )
            }
        )

        result = diagnose.get_job_workflow(client, "job-1")

        assert result["job_id"] == "job-1"
        assert result["realized"] is True
        assert result["workflow"] == WORKFLOW
        assert len(seen) == 1

    def test_a_pre_tracking_job_reports_the_submitted_definition(self):
        client, _ = scripted(
            {
                ("GET", "/api/jobs/job-1/workflow"): (
                    200,
                    {"id": "job-1", "definition": WORKFLOW, "realized": False},
                )
            }
        )

        result = diagnose.get_job_workflow(client, "job-1")

        assert result["realized"] is False
        assert result["workflow"] == WORKFLOW

    def test_next_names_the_two_tools_that_use_it(self):
        client, _ = scripted(
            {
                ("GET", "/api/jobs/job-1/workflow"): (
                    200,
                    {"id": "job-1", "definition": WORKFLOW, "realized": True},
                )
            }
        )

        result = diagnose.get_job_workflow(client, "job-1")

        assert "save_workflow" in result["next"]
        assert "run_workflow" in result["next"]

    def test_an_unknown_job_raises_the_client_error(self):
        client, _ = scripted({})

        with pytest.raises(DwApiError):
            diagnose.get_job_workflow(client, "nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ./activate && python -m pytest tests/test_mcp_diagnose.py::TestGetJobWorkflow -v`
Expected: FAIL — `AttributeError: module 'dw_mcp.diagnose' has no attribute 'get_job_workflow'`.

- [ ] **Step 3: Write the handler**

In `dw_mcp/diagnose.py`, after `get_job`:

```python
def get_job_workflow(client, job_id):
    """The workflow a job ran. `realized: true` means every mutable input
    is pinned (arguments, seed, prompts, output:latest); false means the
    job predates run tracking and this is the definition as submitted.
    Pass it to save_workflow to rerun it by name, or edit it and pass it
    to run_workflow as inline_workflow."""
    body = client.get_json(api_path("api", "jobs", job_id, "workflow"))
    return {
        "job_id": job_id,
        "realized": bool(body.get("realized")),
        "workflow": body.get("definition"),
        "next": "Pass `workflow` to save_workflow to keep it in the catalog "
        "under a name, or edit it and pass it to run_workflow as "
        "inline_workflow.",
    }
```

- [ ] **Step 4: Register the tool**

In `dw_mcp/server.py`, in the diagnose group, after the `get_job` definition:

```python
    def get_job_workflow(job_id: str) -> dict:
        """Get the workflow a job actually ran. When `realized` is true every
        mutable input is pinned - the caller's arguments folded into the
        variables, the seed the run used, stored prompt text inlined, and any
        `output:.../latest/...` rewritten to the run it resolved to - so the
        definition reproduces that run however the library changes. When it is
        false the job predates run tracking and this is the definition as
        submitted. After a long inline run worth keeping, this then
        `save_workflow` is how it gets a name."""
        return diagnose.get_job_workflow(client, job_id)
```

and add it to the read-only registrations:

```python
    tool(get_job, READ_ONLY)
    tool(get_job_workflow, READ_ONLY)
    tool(get_job_events, READ_ONLY)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `source ./activate && python -m pytest tests/test_mcp_diagnose.py tests/test_mcp_server.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dw_mcp/diagnose.py dw_mcp/server.py tests/test_mcp_diagnose.py
cat > /tmp/dw-commit-4.txt <<'EOF'
MCP: get_job_workflow returns the workflow a job ran

A read-only tool over GET /api/jobs/{id}/workflow. 'realized: true' means
every mutable input is pinned; false means the job predates run tracking.
The 'next' sentence points at save_workflow and run_workflow, which is how
an inline run worth keeping gets a name.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-4.txt
```

---

### Task 5: Export a job (`opus`)

*Model rationale: the largest and most judgement-heavy piece — a new module
with a generated README, two routes with distinct failure codes, a reserved
workspace name that ripples through workspace listing, and copy semantics that
have to hold for both a run-directory job and a pre-tracking one.*

**Files:**
- Create: `dw/server/exports.py`
- Modify: `dw/workspace.py` (`EXPORTS_SUBDIR`, `RESERVED_WORKSPACE_NAMES`,
  `_foreign_entries`)
- Modify: `dw/server/app.py` (import `exports`; the export route beside
  `rerun_job` ~line 846; the zip route beside `/outputs` ~line 2252)
- Test: `tests/test_server_exports.py` (new)

**Interfaces:**
- Consumes: `JobManager.get(job_id)` (a live `Job` or a historical detail
  dict), `JobManager.describe(job)`, `JobManager.definition(job_id)`,
  `JobManager.realized(job_id)` from Task 3; `dw.runs.MANIFEST_FILE_NAME`,
  `dw.runs.REALIZED_FILE_NAME`, `dw.runs.OUTPUT_PREFIX`,
  `dw.runs.resolve_output_reference(reference, root)`;
  `dw.assets.ASSET_PREFIX`; `dw.security.validate_asset_reference`,
  `validate_output_path`, `validate_path`; `_asset_roots(ws)` and
  `_served_url(path, ws)` in `dw/server/app.py`.
- Produces: `dw.workspace.EXPORTS_SUBDIR == "exports"`;
  `dw.server.exports.ExportSummary` (dataclass with `job_id`, `directory`,
  `files`, `total_bytes`, `missing`, and `as_dict()`);
  `dw.server.exports.export_directory(workspace_root, job_id) -> str`;
  `dw.server.exports.export_job(manager, job_id, workspace_root, asset_roots,
  overwrite=False) -> ExportSummary`;
  `POST /api/jobs/{job_id}/export` and `GET /exports/{job_id}.zip`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_exports.py`:

```python
"""Exporting one finished job: a directory that stands on its own, and the
same tree as a zip."""

import io
import json
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from dw.runs import REALIZED_FILE_NAME, new_run_id
from dw.server.app import create_app
from dw.server.jobs import JobManager, TERMINAL_STATES
from dw.workspace import Workspace

from .test_server import (
    ScriptedWorkerManager,
    hanging_script,
    valid_workflow,
    wait_for_status,
)

RUN_ID = new_run_id({"workflow": "export"})
RUN_DIR = f"server_test/{RUN_ID}"


def exporting_script(command):
    """A run that reports its directory and writes one file."""
    output_dir = command["output_dir"]
    run_dir = os.path.join(output_dir, "server_test", RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "still.png"), "wb") as file:
        file.write(b"an image")
    with open(os.path.join(run_dir, REALIZED_FILE_NAME), "w") as file:
        json.dump(
            {
                "id": "server_test",
                "seed": 7,
                "steps": [
                    {
                        "name": "gen",
                        "pipeline": {
                            "configuration": {"component_type": "{Fake}"},
                            "from_pretrained_arguments": {"model_name": "m"},
                            "arguments": {"image": "asset:iris.png"},
                        },
                    }
                ],
            },
            file,
        )
    with open(os.path.join(run_dir, "manifest.json"), "w") as file:
        json.dump(
            {
                "run_id": RUN_ID,
                "status": "completed",
                "seed": 7,
                "steps": [{"step": "gen", "files": ["still.png"]}],
            },
            file,
        )
    yield {
        "type": "progress",
        "event": "run_start",
        "run_id": RUN_ID,
        "identity": "server_test",
        "run_dir": RUN_DIR,
    }
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": [
            {"step": "gen", "files": [os.path.join(run_dir, "still.png")]}
        ],
    }


@pytest.fixture
def workspace_root(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.assets, "iris.png"), "wb") as file:
        file.write(b"an iris")
    return root


@pytest.fixture
def server(workspace_root, tmp_path):
    def make(script=exporting_script):
        manager = JobManager(
            workspace_root.outputs,
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=workspace_root.workflows,
        )
        app = create_app(
            workflow_dir=workspace_root.workflows,
            output_dir=workspace_root.outputs,
            job_manager=manager,
            prompt_dir=workspace_root.prompts,
            asset_dir=workspace_root.assets,
            workspace=workspace_root.root,
        )
        return TestClient(app, base_url="http://localhost")

    return make


def finished(client):
    submitted = client.post(
        "/api/jobs", json={"workflow": valid_workflow(), "arguments": {}}
    ).json()
    wait_for_status(client, submitted["id"], TERMINAL_STATES)
    return submitted["id"]


class TestExportDirectory:
    def test_it_gathers_the_whole_run(self, server, workspace_root):
        with server() as client:
            job_id = finished(client)
            response = client.post(f"/api/jobs/{job_id}/export")

        assert response.status_code == 201
        body = response.json()
        directory = body["directory"]
        assert directory == os.path.join(workspace_root.root, "exports", job_id)
        for name in ("README.md", "workflow.json", "manifest.json", "job.json"):
            assert os.path.isfile(os.path.join(directory, name))
        assert os.path.isfile(os.path.join(directory, "assets", "iris.png"))
        assert os.path.isfile(os.path.join(directory, "outputs", "still.png"))
        assert body["total_bytes"] > 0
        assert body["missing"] == []
        assert body["zip_url"] == f"/exports/{job_id}.zip"

    def test_the_workflow_is_the_realized_one(self, server):
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        recorded = json.loads(
            open(os.path.join(body["directory"], "job.json")).read()
        )
        assert recorded["realized"] is True
        assert "traceback" not in recorded and "event_count" not in recorded
        assert body["workflow"]["seed"] == 7

    def test_the_readme_names_the_job_and_says_how_to_run_it(self, server):
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        readme = open(os.path.join(body["directory"], "README.md")).read()
        assert job_id in readme
        assert "python -m dw.run workflow.json" in readme
        assert "Git LFS" in readme

    def test_an_unresolvable_asset_is_reported_missing(self, server, workspace_root):
        os.unlink(os.path.join(workspace_root.assets, "iris.png"))
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()

        assert body["missing"] == ["asset:iris.png"]

    def test_an_unknown_job_is_404(self, server):
        with server() as client:
            assert client.post("/api/jobs/nope/export").status_code == 404

    def test_a_live_job_is_409(self, server):
        with server(hanging_script) as client:
            submitted = client.post(
                "/api/jobs", json={"workflow": valid_workflow(), "arguments": {}}
            ).json()
            wait_for_status(client, submitted["id"], ("running",))
            response = client.post(f"/api/jobs/{submitted['id']}/export")
            client.post(f"/api/jobs/{submitted['id']}/cancel")

        assert response.status_code == 409

    def test_a_second_export_without_overwrite_is_409(self, server):
        with server() as client:
            job_id = finished(client)
            assert client.post(f"/api/jobs/{job_id}/export").status_code == 201
            again = client.post(f"/api/jobs/{job_id}/export")
            assert again.status_code == 409
            forced = client.post(f"/api/jobs/{job_id}/export?overwrite=true")
            assert forced.status_code == 201


class TestExportZip:
    def test_it_lists_the_same_entries_as_the_directory(self, server):
        with server() as client:
            job_id = finished(client)
            body = client.post(f"/api/jobs/{job_id}/export").json()
            response = client.get(f"/exports/{job_id}.zip")

        assert response.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        assert sorted(archive.namelist()) == sorted(
            f"{job_id}/{entry['path']}" for entry in body["files"]
        )

    def test_no_export_is_404(self, server):
        with server() as client:
            job_id = finished(client)
            assert client.get(f"/exports/{job_id}.zip").status_code == 404


class TestReservedName:
    def test_exports_cannot_name_a_workspace(self, server):
        with server() as client:
            response = client.post("/api/workspaces", json={"name": "exports"})
        assert response.status_code == 400
        assert "cannot name a workspace" in response.json()["detail"]

    def test_an_exports_folder_is_not_listed_as_a_workspace(
        self, server, workspace_root
    ):
        with server() as client:
            job_id = finished(client)
            client.post(f"/api/jobs/{job_id}/export")
            names = [w["name"] for w in client.get("/api/workspaces").json()["workspaces"]]
        assert names == ["default"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ./activate && python -m pytest tests/test_server_exports.py -v`
Expected: FAIL — `assert 404 == 201` for the export route (it does not exist).

- [ ] **Step 3: Reserve the name in `dw/workspace.py`**

Beside the other subdir constants (after `SUBDIRS = (...)`):

```python
# Where a job export lands: '<root>/exports/<job id>/'. Not a workspace
# folder - it is beside them, holding gathered copies rather than working
# content - but it is a name a workspace may not take, and workspace_names
# must not mistake it for one
EXPORTS_SUBDIR = "exports"
```

and:

```python
RESERVED_WORKSPACE_NAMES = SUBDIRS + (EXPORTS_SUBDIR,)
```

and in `_foreign_entries`:

```python
def _foreign_entries(path):
    """What a directory holds besides a workspace's own three folders and
    the exports it may have gathered."""
    ignored = NAMED_SUBDIRS + (EXPORTS_SUBDIR,)
    try:
        return sorted(entry for entry in os.listdir(path) if entry not in ignored)
    except OSError:
        return []
```

`workspace_names` already skips `RESERVED_WORKSPACE_NAMES`, so an `exports/`
folder at the root stops being a candidate workspace as soon as the constant
lands.

- [ ] **Step 4: Write `dw/server/exports.py`**

```python
"""Gathering one finished job into a directory that stands on its own.

The record of a run is split across four places on one machine: the job row,
the run's manifest, the workflow that ran, and the media on either side of it.
This module puts them in one tree - text at the top, media in folders, no
absolute paths inside the files - so the run can be committed, moved or handed
to someone else.

    exports/<job id>/
        README.md       what this is, how it was made, how to run it again
        workflow.json   the realized workflow, when the run wrote one
        manifest.json   the run's manifest, or a synthesized stand-in
        job.json        the job row: status, times, arguments, warnings, error
        assets/         every asset: the workflow names, under its own name
        inputs/         every file an output: reference named, by run
        outputs/        every file the manifest lists

Copies are copies, never hard links: exports/ is what a user moves or deletes,
and a hard link would make deleting it look like deleting the original.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field

from ..assets import ASSET_PREFIX
from ..runs import (
    MANIFEST_FILE_NAME,
    OUTPUT_PREFIX,
    is_output_reference,
    resolve_output_reference,
)
from ..security import (
    SecurityError,
    validate_asset_reference,
    validate_output_path,
    validate_path,
)
from ..workspace import EXPORTS_SUBDIR
from .jobs import TERMINAL_STATES

logger = logging.getLogger("dw")

WORKFLOW_FILE_NAME = "workflow.json"
JOB_FILE_NAME = "job.json"
README_FILE_NAME = "README.md"

# The job detail keys an export does not carry: a traceback is a developer's
# artifact of one process, and an event count describes a log this tree does
# not hold
OMITTED_JOB_KEYS = ("traceback", "event_count")

README_TEMPLATE = """# {workflow_name} - job {job_id}

{realized_sentence}

## The run

| | |
| --- | --- |
| Job | `{job_id}` |
| Workflow | `{workflow_name}` |
| Catalog entry | {catalog_name} |
| Status | {status} |
| Started | {started_at} |
| Finished | {finished_at} |
| Device | {device} |
| Engine | dw {dw_version} |
| Seed | `{seed}` |

Arguments as submitted:

```json
{arguments}
```

## Stored prompts

{prompts}

## Sub-workflows

{sub_workflows}

## Inputs

{inputs}

## Running it again

From a checkout of diffusers-workflow, with this directory as the working
directory:

```bash
python -m dw.run workflow.json
```

Over MCP, pass the contents of `workflow.json` to `run_workflow` as
`inline_workflow`. Either way the `asset:` and `output:` names in it resolve
against the server's libraries, not against this directory - the copies under
`assets/` and `inputs/` are the record of what those names meant, not a
substitute for them.

## Missing

{missing}

## A note on committing this

`assets/`, `inputs/` and `outputs/` hold generated and source media, which is
usually large and always binary. If this tree goes into Git, put those three
directories in Git LFS; the four files at the top are text and belong in Git
proper.
"""


@dataclass
class ExportSummary:
    """What an export produced, computed from what actually landed."""

    job_id: str
    directory: str
    files: list = field(default_factory=list)
    total_bytes: int = 0
    missing: list = field(default_factory=list)

    def as_dict(self):
        return {
            "job_id": self.job_id,
            "directory": self.directory,
            "files": self.files,
            "total_bytes": self.total_bytes,
            "missing": self.missing,
        }


def export_directory(workspace_root, job_id):
    """Where one job's export lives, confined to the workspace root.

    The job id is caller input - it arrives in a URL - so it is joined and
    then validated rather than trusted to be the hex string the manager
    generates.
    """
    root = validate_output_path(os.path.join(workspace_root, EXPORTS_SUBDIR), None)
    return validate_path(os.path.join(root, job_id), root)


def export_job(manager, job_id, workspace_root, asset_roots, overwrite=False):
    """Gather one finished job into '<workspace_root>/exports/<job id>/'.

    Args:
        manager: The JobManager holding the job (live or historical).
        job_id: The job to export.
        workspace_root: The workspace the export lands in.
        asset_roots: The workspace's asset search path, in order - the same
            order 'asset:' resolves in, so what is copied is what the run
            loaded.
        overwrite: Replace an existing export rather than refusing.

    Returns:
        An ExportSummary.

    Raises:
        ValueError: Unknown job, or a job that has not finished.
        FileExistsError: The export exists and overwrite is False.
    """
    job = manager.get(job_id)
    if job is None:
        raise ValueError(f"Unknown job {job_id}")
    detail = job if isinstance(job, dict) else manager.describe(job)
    status = detail.get("status")
    if status not in TERMINAL_STATES:
        raise ValueError(
            f"Job {job_id} is {status} - only a finished job can be exported"
        )

    spec = job.get("spec", {}) if isinstance(job, dict) else job.spec
    run_dir = job.get("run_dir") if isinstance(job, dict) else job.run_dir
    output_root = validate_output_path(
        spec.get("output_dir") or manager.output_dir, None
    )

    target = export_directory(workspace_root, job_id)
    if os.path.exists(target):
        if not overwrite:
            raise FileExistsError(
                f"An export of job {job_id} already exists - pass overwrite to "
                f"replace it"
            )
        shutil.rmtree(target)
    os.makedirs(target)

    summary = ExportSummary(job_id=job_id, directory=target)

    realized = manager.realized(job_id)
    workflow = realized if realized is not None else (manager.definition(job_id) or {})
    _write_json(summary, target, WORKFLOW_FILE_NAME, workflow)

    manifest = _run_manifest(output_root, run_dir)
    if manifest is None:
        # No run directory, or it no longer holds a manifest: the job row's
        # own per-step file list is what is left, and it says so
        manifest = {
            "synthesized": True,
            "run_id": None,
            "status": status,
            "steps": detail.get("manifest") or [],
        }
    _write_json(summary, target, MANIFEST_FILE_NAME, manifest)

    record = {
        key: value for key, value in detail.items() if key not in OMITTED_JOB_KEYS
    }
    record["realized"] = realized is not None
    _write_json(summary, target, JOB_FILE_NAME, record)

    _copy_assets(summary, workflow, target, asset_roots)
    _copy_inputs(summary, workflow, target, output_root)
    _copy_outputs(summary, manifest, target, output_root, run_dir)

    readme = _readme(
        job_id, detail, manifest, workflow, summary, realized is not None
    )
    _write_text(summary, target, README_FILE_NAME, readme)
    return summary


# -------------------------------------------------------------- the pieces


def _run_manifest(output_root, run_dir):
    """The manifest the run left, or None when there is no reading it."""
    if not run_dir:
        return None
    try:
        path = validate_path(
            os.path.join(output_root, run_dir, MANIFEST_FILE_NAME), output_root
        )
        with open(path, "r") as file:
            return json.load(file)
    except (SecurityError, OSError, ValueError) as e:
        logger.debug(f"No run manifest to export from {run_dir}: {e}")
        return None


def _references(value, prefix):
    """Every string under `value` that begins with `prefix`, deduplicated.

    The same recursive shape realization walks, so a reference is found
    wherever it sits in the tree.
    """
    found = set()

    def scan(item):
        if isinstance(item, str):
            if item.startswith(prefix):
                found.add(item)
        elif isinstance(item, dict):
            for child in item.values():
                scan(child)
        elif isinstance(item, list):
            for child in item:
                scan(child)

    scan(value)
    return sorted(found)


def _copy_assets(summary, workflow, target, asset_roots):
    """Every 'asset:' the workflow names, under its own name in assets/."""
    for reference in _references(workflow, ASSET_PREFIX):
        try:
            name = validate_asset_reference(
                reference.removeprefix(ASSET_PREFIX).strip()
            )
        except SecurityError:
            summary.missing.append(reference)
            continue
        source = None
        for root in asset_roots:
            try:
                candidate = validate_path(os.path.join(root, name), root)
            except SecurityError:
                continue
            if os.path.isfile(candidate):
                source = candidate
                break
        if source is None:
            summary.missing.append(reference)
            continue
        _copy(summary, source, target, os.path.join("assets", *name.split("/")))


def _copy_inputs(summary, workflow, target, output_root):
    """Every 'output:' the workflow names, kept under the run it came from.

    The reference itself is not rewritten - the realized workflow is the
    immutable record of the run - so the directory name is the reference's
    own name, and the README says where each one came from.
    """
    for reference in _references(workflow, OUTPUT_PREFIX):
        if not is_output_reference(reference):
            continue
        name = reference.removeprefix(OUTPUT_PREFIX).strip()
        try:
            source = resolve_output_reference(reference, output_root)
        except (SecurityError, OSError, ValueError):
            summary.missing.append(reference)
            continue
        _copy(summary, source, target, os.path.join("inputs", *name.split("/")))


def _copy_outputs(summary, manifest, target, output_root, run_dir):
    """Every file the manifest lists, under outputs/.

    A manifest entry names a file relative to the run directory when the run
    wrote it, and absolutely when a step-cache hit republished an earlier
    run's file. Both land here: the first under its own relative name, the
    second under its path relative to the output root, which keeps the
    identity and run id that say where it really came from.
    """
    run_root = os.path.join(output_root, run_dir) if run_dir else output_root
    for entry in manifest.get("steps") or []:
        if not isinstance(entry, dict):
            continue
        for recorded in entry.get("files") or []:
            source = (
                recorded
                if os.path.isabs(recorded)
                else os.path.join(run_root, recorded)
            )
            try:
                source = validate_path(source, output_root)
            except SecurityError:
                summary.missing.append(recorded)
                continue
            if not os.path.isfile(source):
                summary.missing.append(recorded)
                continue
            try:
                relative = os.path.relpath(source, run_root)
            except ValueError:  # different drive on Windows
                relative = os.path.basename(source)
            if relative.startswith(os.pardir):
                relative = os.path.relpath(source, output_root)
            _copy(
                summary,
                source,
                target,
                os.path.join("outputs", *relative.split(os.sep)),
            )


def _readme(job_id, detail, manifest, workflow, summary, realized):
    workflow_block = manifest.get("workflow") or {}
    prompts = workflow_block.get("prompts") or []
    sub_workflows = workflow_block.get("sub_workflows") or {}
    inputs = [
        entry["path"]
        for entry in summary.files
        if entry["path"].startswith("inputs/")
    ]
    return README_TEMPLATE.format(
        job_id=job_id,
        workflow_name=detail.get("workflow") or "unknown",
        catalog_name=f"`{detail['workflow_name']}`"
        if detail.get("workflow_name")
        else "none - an inline definition",
        realized_sentence=(
            "`workflow.json` is the *realized* workflow: every mutable input "
            "is pinned, so it reproduces this run whatever changes afterwards."
            if realized
            else "`workflow.json` is the definition as submitted - this job "
            "predates run tracking, so its arguments and prompts are not "
            "pinned into it."
        ),
        status=detail.get("status"),
        started_at=detail.get("started_at"),
        finished_at=detail.get("finished_at"),
        device=manifest.get("device", "unknown"),
        dw_version=manifest.get("dw_version", "unknown"),
        seed=manifest.get("seed", workflow.get("seed", "not recorded")),
        arguments=json.dumps(detail.get("arguments") or {}, indent=2),
        prompts=_bullets(
            f"`{name}` - inlined into `workflow.json`" for name in prompts
        )
        or "None: this workflow named no stored prompt.",
        sub_workflows=_bullets(
            f"`{path}` - sha256 `{digest}`" if digest else f"`{path}` - unreadable"
            for path, digest in sorted(sub_workflows.items())
        )
        or "None: this workflow composed no other workflow by path.",
        inputs=_bullets(
            f"`{path}` - copied from the run named in its own path"
            for path in inputs
        )
        or "None: this workflow named no file from an earlier run.",
        missing=_bullets(f"`{name}`" for name in summary.missing)
        or "Nothing: every file this run referenced was found and copied.",
    )


def _bullets(lines):
    rendered = "\n".join(f"- {line}" for line in lines)
    return rendered


# ------------------------------------------------------------------- files


def _copy(summary, source, target, relative):
    """Copy one file into the export and record it."""
    destination = os.path.join(target, relative)
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as e:
        logger.warning(f"Could not copy {source} into the export: {e}")
        summary.missing.append(source)
        return
    _record(summary, destination, target)


def _write_json(summary, target, name, payload):
    _write_text(summary, target, name, json.dumps(payload, indent=2, default=str))


def _write_text(summary, target, name, text):
    path = os.path.join(target, name)
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")
        return
    _record(summary, path, target)


def _record(summary, path, target):
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    summary.files.append(
        {"path": os.path.relpath(path, target).replace(os.sep, "/"), "bytes": size}
    )
    summary.total_bytes += size
```

- [ ] **Step 5: Add the routes to `dw/server/app.py`**

Add to the imports, beside `from .enhancers import ...`:

```python
from .exports import export_directory, export_job
```

Add after the `rerun_job` route:

```python
    @app.post("/api/jobs/{job_id}/export", status_code=201)
    def export_job_route(
        job_id: str,
        overwrite: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Gather one finished job into '<workspace>/exports/<job id>/': the
        workflow it ran, the run's manifest, the job row, the media it used
        and the media it made, plus a README. 404 for an unknown job, 409 for
        one still running or for an export that already exists without
        `overwrite`.

        The three JSON files come back inline as well as on disk - the
        directory is on the server, and a client on another machine has no
        other way to read them without fetching the zip."""
        try:
            summary = export_job(
                manager, job_id, ws.root, _asset_roots(ws), overwrite=overwrite
            )
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            message = str(e)
            if message.startswith("Unknown job"):
                raise HTTPException(status_code=404, detail=message)
            raise HTTPException(status_code=409, detail=message)
        body = summary.as_dict()
        body["zip_url"] = _served_url(f"/exports/{quote(job_id)}.zip", ws)
        for key, name in (
            ("workflow", "workflow.json"),
            ("manifest", "manifest.json"),
            ("job", "job.json"),
        ):
            try:
                with open(os.path.join(summary.directory, name), "r") as file:
                    body[key] = json.load(file)
            except (OSError, ValueError):
                body[key] = None
        return body
```

Add beside the `/outputs` route (ungated for the same reason it is: the auth
middleware only gates `/api/`):

```python
    @app.get("/exports/{job_id}.zip")
    def export_zip(job_id: str, ws: Workspace = Depends(selected_workspace)):
        """One job's export as a zip, built on request from the directory
        rather than kept as a second copy. Entries are named
        '<job id>/<relative path>', so unzipping anywhere gives the same tree
        the server holds."""
        try:
            directory = export_directory(ws.root, job_id)
        except SecurityError:
            raise HTTPException(status_code=404, detail="No export for this job")
        if not os.path.isdir(directory):
            raise HTTPException(status_code=404, detail="No export for this job")

        handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        handle.close()
        with zipfile.ZipFile(handle.name, "w", zipfile.ZIP_DEFLATED) as archive:
            for current, _dirs, names in os.walk(directory):
                for name in sorted(names):
                    path = os.path.join(current, name)
                    entry = os.path.relpath(path, directory).replace(os.sep, "/")
                    archive.write(path, f"{job_id}/{entry}")

        def stream():
            with open(handle.name, "rb") as file:
                while True:
                    chunk = file.read(64 * 1024)
                    if not chunk:
                        return
                    yield chunk

        return StreamingResponse(
            stream(),
            media_type="application/zip",
            headers={
                "content-disposition": f'attachment; filename="{job_id}.zip"'
            },
            # The archive is a temp file, not a second permanent copy - it
            # goes as soon as the response has been sent
            background=BackgroundTask(os.unlink, handle.name),
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `source ./activate && python -m pytest tests/test_server_exports.py tests/test_server_workspaces.py tests/test_workspace.py -v`
Expected: PASS.

- [ ] **Step 7: Run the whole server and engine suite for regressions**

Run: `source ./activate && python -m pytest tests/test_server.py tests/test_server_jobs.py tests/test_runs.py tests/test_realize.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add dw/server/exports.py dw/workspace.py dw/server/app.py tests/test_server_exports.py
cat > /tmp/dw-commit-5.txt <<'EOF'
Export one finished job as a git-ready directory and a zip

POST /api/jobs/{id}/export gathers the realized workflow, the run's
manifest, the job row, every asset: and output: file the workflow named and
every file the manifest lists into <workspace>/exports/<job id>/, with a
generated README saying what the run was, where each input came from, how
to run it again, and that the media belongs in Git LFS. 404 for an unknown
job, 409 for a live one or an existing export without overwrite.

GET /exports/{id}.zip streams the same tree, built on request. 'exports'
joins the reserved workspace names, so the folder is never mistaken for a
workspace.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-5.txt
```

---

### Task 6: `export_job` MCP tool (`haiku`)

*Model rationale: one handler module, one client method tweak, one
registration, one test file — mechanical, with complete code below.*

**Files:**
- Create: `dw_mcp/exports.py`
- Modify: `dw_mcp/client.py` (`post_json` gains `params`)
- Modify: `dw_mcp/server.py` (import `exports`; register in the jobs group)
- Test: `tests/test_mcp_exports.py` (new)

**Interfaces:**
- Consumes: `POST /api/jobs/{id}/export?overwrite=` answering the
  `ExportSummary` fields plus `zip_url`, `workflow`, `manifest`, `job` from
  Task 5; `dw_mcp.client.api_path`, `DwApiError`.
- Produces: `dw_mcp.exports.export_job(client, job_id, overwrite=False) ->
  dict` with keys `job_id`, `where`, `directory`, `zip_url`, `files`,
  `total_bytes`, `missing`, `workflow`, `manifest`, `job`, `next`;
  `DwClient.post_json(path, payload=None, params=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_exports.py`:

```python
"""Exporting a job over MCP: the directory is on the server, and the tool
says so - the lesson download_output taught."""

import httpx
import pytest

from dw_mcp import exports
from dw_mcp.client import DwApiError, DwClient

SUMMARY = {
    "job_id": "job-1",
    "directory": "/srv/studio/exports/job-1",
    "files": [
        {"path": "workflow.json", "bytes": 412},
        {"path": "outputs/still.png", "bytes": 90210},
    ],
    "total_bytes": 90622,
    "missing": [],
    "zip_url": "/exports/job-1.zip",
    "workflow": {"id": "w", "steps": []},
    "manifest": {"run_id": "20260908-120000-abcdef01"},
    "job": {"id": "job-1", "status": "succeeded"},
}


def scripted(routes):
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append({"key": key, "params": dict(request.url.params)})
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def exporting(status=201, body=None):
    return scripted(
        {("POST", "/api/jobs/job-1/export"): (status, body or SUMMARY)}
    )


def test_it_returns_the_directory_the_zip_and_the_file_list():
    client, seen = exporting()

    result = exports.export_job(client, "job-1")

    assert result["job_id"] == "job-1"
    assert result["directory"] == "/srv/studio/exports/job-1"
    assert result["zip_url"] == "/exports/job-1.zip"
    assert result["total_bytes"] == 90622
    assert [entry["path"] for entry in result["files"]] == [
        "workflow.json",
        "outputs/still.png",
    ]
    assert len(seen) == 1


def test_the_three_json_files_come_back_inline():
    client, _ = exporting()

    result = exports.export_job(client, "job-1")

    assert result["workflow"] == {"id": "w", "steps": []}
    assert result["manifest"]["run_id"] == "20260908-120000-abcdef01"
    assert result["job"]["status"] == "succeeded"


def test_it_says_where_the_directory_is():
    client, _ = exporting()

    result = exports.export_job(client, "job-1")

    assert result["where"] == (
        "/srv/studio/exports/job-1 on the machine running the MCP server"
    )


def test_overwrite_travels_as_a_query_parameter():
    client, seen = exporting()

    exports.export_job(client, "job-1", overwrite=True)

    assert seen[0]["params"]["overwrite"] == "true"


def test_a_409_reaches_the_model_as_a_readable_refusal():
    client, _ = exporting(status=409, body={"detail": "An export already exists"})

    with pytest.raises(DwApiError) as caught:
        exports.export_job(client, "job-1")

    assert "already exists" in str(caught.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ./activate && python -m pytest tests/test_mcp_exports.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw_mcp.exports'`.

- [ ] **Step 3: Let `post_json` carry query parameters**

In `dw_mcp/client.py`:

```python
    def post_json(self, path, payload=None, params=None):
        """`params` is for a route whose options are query parameters rather
        than a body - the export route, which takes `overwrite` beside the
        workspace selector `_scoped` adds."""
        return self._json(
            self._request("POST", path, json=payload or {}, params=params), path
        )
```

- [ ] **Step 4: Write `dw_mcp/exports.py`**

```python
"""Bundling a finished job so it can leave the server.

The one thing this module has to keep saying: the directory it makes is on
the machine running dw.serve, which over a `dw.serve --mcp` endpoint is the
GPU box and not where the agent is. The zip URL is the way to it from
anywhere else.
"""

from dw_mcp.client import api_path


def export_job(client, job_id, overwrite=False):
    """Gather one finished job into a directory on the machine running
    dw.serve: workflow.json (realized), manifest.json, job.json, README,
    assets/, inputs/, outputs/. Returns the directory, the zip URL, the
    file list with sizes and the total, and the three JSON files inline.
    The directory is on the server machine, not this one - use the zip
    URL to fetch it elsewhere."""
    body = client.post_json(
        api_path("api", "jobs", job_id, "export"),
        params={"overwrite": "true" if overwrite else "false"},
    )
    directory = body.get("directory")
    return {
        "job_id": job_id,
        "where": f"{directory} on the machine running the MCP server",
        "directory": directory,
        "zip_url": body.get("zip_url"),
        "files": body.get("files") or [],
        "total_bytes": body.get("total_bytes"),
        "missing": body.get("missing") or [],
        "workflow": body.get("workflow"),
        "manifest": body.get("manifest"),
        "job": body.get("job"),
        "next": "Report the directory as a path on the server, and hand the "
        "user the zip URL if they want the files locally.",
    }
```

- [ ] **Step 5: Register the tool**

In `dw_mcp/server.py`, add `exports` to the `from dw_mcp import (...)` block
(alphabetically, after `diagnose`), and in the diagnose/jobs group after
`move_job`:

```python
    def export_job(job_id: str, overwrite: bool = False) -> dict:
        """Gather one finished job into a directory on the server: the
        realized workflow, the run's manifest, the job row, a README, and
        copies of every asset it used, every earlier run's file it read and
        every file it made. Returns the directory, a zip URL, the file list
        with sizes and the total, and the three JSON files inline. THE
        DIRECTORY IS ON THE MACHINE RUNNING THE SERVER, not on yours - report
        it as a server path and hand the user the zip URL if they want the
        files locally. Refuses a job that is still running; refuses an
        existing export unless overwrite=true."""
        return exports.export_job(client, job_id, overwrite=overwrite)
```

and register it beside the other writers:

```python
    for fn in (run_workflow, cancel_job, rerun_job, move_job, export_job):
        tool(fn, WRITES)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `source ./activate && python -m pytest tests/test_mcp_exports.py tests/test_mcp_diagnose.py tests/test_mcp_server.py tests/test_mcp_media.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add dw_mcp/exports.py dw_mcp/client.py dw_mcp/server.py tests/test_mcp_exports.py
cat > /tmp/dw-commit-6.txt <<'EOF'
MCP: export_job bundles a finished job for git

A handler over POST /api/jobs/{id}/export returning the directory, the zip
URL, the file list with sizes and the three JSON documents inline. The
'where' sentence says the directory is on the machine running the server -
the lesson download_output taught.

DwClient.post_json gains an optional params argument, since the route's
overwrite flag is a query parameter beside the workspace selector.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-6.txt
```

---

### Task 7: Docs and skills (`sonnet`)

*Model rationale: eight prose files that must stay consistent with each other
and with the code — judgement about wording, but no design decisions left.*

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` (the seed/run-directory prose ~line 995-1000
  and the "Authoring a workflow from an agent" section ~line 237)
- Modify: `CLAUDE.md` (the type-system list and the run-directories gotcha)
- Modify: `docs/MCP.md` (the Diagnose table ~line 277)
- Modify: `docs/SERVER.md` (the Jobs API table ~line 131, and Files ~line 201)
- Modify: `docs/WORKSPACES.md` (the Runs section ~line 167)
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md`,
  `plugins/dw/skills/minimax-music3/SKILL.md`,
  `plugins/dw/skills/ltx-2.5/SKILL.md` ("Run and judge")
- Modify: `docs/proposals/job-record-and-export.md` (status line),
  `docs/proposals/resume.md` (stage 2)

**Interfaces:**
- Consumes: everything Tasks 1-6 produced. No new code.

- [ ] **Step 1: `docs/WORKFLOW_GUIDE.md`**

After the paragraph ending "…can be reproduced after the fact." (the seed
prose), add:

```markdown
Beside that manifest the run also writes `workflow.json` — the *realized*
workflow, meaning the one that actually ran. Every mutable input is pinned into
it: the caller's `arguments` folded into the `variables` defaults, the seed the
run used, each `prompt:` reference replaced by the stored text, and each
`output:<identity>/latest/<file>` rewritten to the run id it resolved to.
`asset:`, `constant:`, `previous_result:` and `builtin:` are kept as written —
each already names something pinned by the asset library or by the manifest's
`dw_version` — and a sub-workflow named by local path is kept with its file's
SHA-256 recorded in the manifest. The manifest also lists which stored prompts
were inlined, since inlining loses the name.

The file is a valid workflow: `python -m dw.run workflow.json` from inside the
run directory reproduces the run, and so does handing it to `run_workflow` as
`inline_workflow`. Writing it is best effort, exactly like the manifest — a run
that produced its files has succeeded either way — and `--output-layout flat`
writes no run directory, so it writes neither file.
```

In the "Authoring a workflow from an agent" section, at the end of the
References list, add:

```markdown
After a long inline run that is worth keeping, `get_job_workflow(job_id)`
returns the realized workflow — the definition with the arguments, seed and
prompts of that run pinned into it — and `save_workflow` gives it a name, so
the next run is by name rather than by pasting JSON again. `export_job(job_id)`
bundles the whole run (workflow, manifest, job row, the media on both sides)
into a directory on the server plus a zip URL, for a run worth committing or
handing to someone else.
```

- [ ] **Step 2: `CLAUDE.md`**

In the type-system bullet list, after the `prompt:` bullet, add:

```markdown
- Every run directory holds `workflow.json` beside its manifest: the *realized*
  workflow, with the run's arguments folded into the variable defaults, the seed
  it used, stored prompt text inlined and `output:.../latest/...` pinned to the
  run it resolved to. Written by `realize_workflow` (`dw/realize.py`) at run
  start, best effort. Over MCP, `get_job_workflow` reads it back and
  `save_workflow` names it; `export_job` bundles the run
```

In the **Run directories** gotcha, after the sentence ending "…and a sub-workflow
inherits the parent's run directory and writes no manifest of its own", add:

```markdown
  The realized workflow is written into the same directory as `workflow.json`
  (`dw/realize.py`, `write_realized_workflow`), and the manifest's `workflow`
  block carries `realized`, `prompts` (the stored prompts inlined) and
  `sub_workflows` (path -> SHA-256). A job records the run it was
  (`run_id`/`run_dir` on `Job` and in `jobs.sqlite`), which is how
  `JobManager.realized` finds the file. `exports` is a reserved workspace name:
  `POST /api/jobs/{id}/export` gathers one finished job into
  `<workspace>/exports/<job id>/` and `GET /exports/<job id>.zip` streams it
```

- [ ] **Step 3: `docs/MCP.md`**

In the Diagnose table, after the `get_job` row:

```markdown
| `get_job_workflow(job_id)` | `job_id` | The workflow the job actually ran. `realized: true` means every mutable input is pinned (arguments, seed, prompts, `output:latest`); `false` means the job predates run tracking and this is the definition as submitted. Pass it to `save_workflow` to keep it under a name |
| `export_job(job_id, overwrite=False)` | `job_id`, `overwrite` | Gather one finished job into `<workspace>/exports/<job id>/` on the server: the realized workflow, the run's manifest, the job row, a README, and copies of the assets, earlier-run inputs and outputs. Returns the directory, a zip URL, the file list with sizes and the three JSON files inline. **The directory is on the machine running the server**, like `download_output`'s destination - report it as a server path and hand the user the zip URL for a local copy |
```

- [ ] **Step 4: `docs/SERVER.md`**

Replace the `GET /api/jobs/{id}/workflow` row and add two more:

```markdown
| `GET /api/jobs/{id}/workflow` | The workflow the job ran: `{id, definition, realized}`. `realized: true` is the copy the run itself wrote (`workflow.json` in its run directory), with arguments, seed, prompts and `output:latest` pinned; `false` falls back to the submitted definition, which is what a job from before run tracking has. 404 means neither is readable - the job itself still is |
| `POST /api/jobs/{id}/export?workspace=&overwrite=` | Gather one finished job into `<workspace>/exports/<job id>/`: `workflow.json`, `manifest.json`, `job.json`, `README.md`, `assets/`, `inputs/`, `outputs/`. 201 with the file list, total bytes, anything it could not find, a `zip_url`, and the three JSON files inline. 404 unknown job, 409 for a job still running or an existing export without `overwrite` |
| `GET /exports/{id}.zip?workspace=` | The same tree as one archive, built on request rather than kept as a second copy. Entries are named `<job id>/<relative path>`. Ungated exactly as `/outputs` is |
```

In the "Files and models" section, add a sentence after the outputs paragraph:

```markdown
`exports/` sits beside the workspace's own folders, holding one directory per
exported job. It is a reserved name: no workspace can be called `exports`, and
the folder is never listed as one.
```

- [ ] **Step 5: `docs/WORKSPACES.md`**

In the Runs section, update the tree and add a sentence:

```markdown
```
outputs/
  ltx2/Gyre/
    20260905-181530-a1b2c3d4/
      Gyre-still.0-0.0.png
      Gyre-video.1-0.0.mp4
      manifest.json
      workflow.json
```
```

and after the paragraph describing the run id:

```markdown
`workflow.json` is the realized workflow — the definition with this run's
arguments, seed and stored prompts pinned into it, so the directory reproduces
itself. `manifest.json` points at it and lists which prompts were inlined.
```

At the end of the "Several workspaces on one server" section, add:

```markdown
A fifth name is reserved beside `workflows`, `prompts`, `assets` and `outputs`:
`exports`. `POST /api/jobs/{id}/export` gathers one finished job into
`<root>/exports/<job id>/`, and that folder is never mistaken for a workspace.
```

- [ ] **Step 6: The three plugin skills**

In each of `plugins/dw/skills/minimax-h3/SKILL.md`,
`plugins/dw/skills/minimax-music3/SKILL.md` and
`plugins/dw/skills/ltx-2.5/SKILL.md`, append one numbered bullet to the end of
the "Run and judge" list (renumbering is not needed — append after the last
item):

```markdown
- After an inline run worth keeping, `get_job_workflow` and `save_workflow` it,
  so the next run is by name rather than by pasting JSON; `export_job` bundles
  the run — workflow, manifest, job row and media — for git.
```

- [ ] **Step 7: The proposals**

In `docs/proposals/job-record-and-export.md`, change the status line to:

```markdown
Status: implemented 2026-09-08 (design:
`docs/superpowers/specs/2026-09-08-job-record-and-export-design.md`, plan:
`docs/superpowers/plans/2026-09-08-job-record-and-export.md`). Supplies the
"manifest carries step identity" stage of [resume.md](resume.md), which stays
the design for resuming a run; this proposal is the record that resume reads,
plus the two ways to get it off the server.
```

In `docs/proposals/resume.md`, in the Staging list, replace stage 2 with:

```markdown
2. **Manifest carries step identity.** *Satisfied by the realized workflow*
   ([job-record-and-export.md](job-record-and-export.md)): every run now writes
   `workflow.json` beside its manifest with every mutable input pinned, so each
   step's definition as it actually ran is on disk to compare against, and the
   manifest's per-step files say what it made. A per-step digest may still be
   worth adding for a cheaper comparison, but the information is no longer
   missing.
```

- [ ] **Step 8: Verify the docs still pass their own test**

Run: `source ./activate && python -m pytest tests/test_docs_links.py tests/test_plugin_skills.py -v`
Expected: PASS. `test_docs_links.py` checks that every `workflows/...` path a
doc names exists — none of the additions above names one, so a failure here
means a typo introduced a path.

- [ ] **Step 9: Run the whole suite**

Run: `source ./activate && python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add docs CLAUDE.md plugins
cat > /tmp/dw-commit-7.txt <<'EOF'
Document the realized workflow, the run record and the export

WORKFLOW_GUIDE gets the realized file beside the manifest and, in the
agent-authoring section, the get_job_workflow -> save_workflow loop after a
long inline run. CLAUDE.md mirrors it in the type-system list and the
run-directories gotcha. MCP.md documents the two tools with the
server-machine caveat, SERVER.md the three routes, WORKSPACES.md the
workflow.json in a run directory and the reserved 'exports' name.

Each family skill's "Run and judge" gains the same one-line habit. The
proposal moves to implemented, and resume.md records that its stage 2 is
satisfied by the realized file.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git commit -F /tmp/dw-commit-7.txt
```

---

## Self-review notes

Recorded here so an executor knows which parts of the spec were interpreted
rather than transcribed:

1. **`realize_workflow`'s signature gains `workflow_dir=None`.** The spec's
   sub-workflow rule says the digest resolves "as `Workflow` resolves it
   (relative to `base_dir`, confined to `workflow_dir`)", but the signature it
   prints has no such parameter. The added keyword defaults to `None` (an
   unconfined CLI run), so the spec's signature remains callable as written.
2. **`base_dir=self.base_dir` in the spec's call site does not exist.**
   `Workflow` has no `base_dir` attribute; `run` computes it as a local. The
   plan passes that local.
3. **Asset resolution in the export.** The spec names
   `resolve_asset_reference`, whose search path is derived from process-wide
   discovery. The plan instead walks the `asset_roots` the caller passed, using
   `validate_asset_reference` + `validate_path` — the same algorithm, restricted
   to exactly the roots `_asset_roots(ws)` built, which is what the spec's
   parenthetical actually asks for and what makes the function testable.
4. **`EXPORTS_SUBDIR` lives in `dw/workspace.py`, not `dw/server/exports.py`.**
   `RESERVED_WORKSPACE_NAMES` and `_foreign_entries` need it, and
   `dw/workspace.py` must not import from `dw.server`. `dw/server/exports.py`
   re-imports it from there, so the spec's name is still available at the
   spec's module.
5. **The export route returns more than `ExportSummary`.** The spec's MCP tool
   promises "the three JSON files it also returns inline", and the MCP client
   cannot read the server's disk — so the route's body is the summary plus
   `zip_url` plus `workflow`, `manifest` and `job`. `ExportSummary` itself keeps
   the fields the spec gives it.
6. **`overwrite` reaches the route as a query parameter**, as the spec writes
   it, which meant giving `DwClient.post_json` an optional `params` argument.
7. **`recent_summaries` gains `run_id` too.** The spec only names `summary()`,
   but `JobManager.list` mixes live summaries and historical rows, and a field
   present in one and absent in the other is a shape a client cannot rely on.
8. **Which manifest the export copies its outputs from.** The spec says "every
   file the manifest lists, copied with the manifest's relative names" without
   saying which manifest. The plan uses the run's own manifest when the run
   directory is known (names relative to the run directory, absolute for a
   step-cache hit republishing an earlier run) and the job row's synthesized
   one otherwise; an absolute path is named by its path relative to the output
   root, which keeps the identity and run id that explain where it came from.
