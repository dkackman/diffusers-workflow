# Acknowledged-cost binding, stage 1: the plan on validate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `POST /api/validate` (and MCP `validate_workflow`) answers with a `plan` - a fingerprint of the work the run will execute for the arguments given, the step count, the list lengths, the model repos it would have to download first, and a cost estimate whose basis is named - so an agent quotes what will run rather than the catalog's defaults-only `cost`.

**Architecture:** A new pure module `dw/plan.py` builds the plan from the `Workflow` the route already constructed: it realizes the definition with the caller's arguments (`realize_workflow`, gaining a `pin_outputs=False` switch so a `latest` landing between validate and run does not change the fingerprint), expands `for_each` through `Workflow.expanded_definition`, scrubs the seed and the documentation keys, and hashes the rest. The estimate reads the workflow's own `cost` block (and each composed child's); downloads are the `model_name`s `scan_models` does not find. The route attaches the plan best-effort and never lets it change the verdict. MCP surfaces it and the skills/docs teach quoting from it.

**Tech Stack:** Python 3.12, FastAPI (`dw/server/app.py`), pytest, `huggingface_hub` (`scan_cache_dir`, `model_info`).

**Spec:** [docs/superpowers/specs/2026-09-12-acknowledged-cost-binding-design.md](../specs/2026-09-12-acknowledged-cost-binding-design.md) - stage 1 sections 1-5. Stage 2 (the bound acknowledgement and the 409) is a separate plan.

## Global Constraints

- Model knowledge stays out of engine code: no repo name, no per-model minute figure anywhere in `dw/`. Every number comes from a workflow's `cost` block.
- `dw/plan.py` imports nothing from `dw.server` or `dw.worker`.
- Plan construction is best effort at the API: a failure is logged and answered as `plan: null`; the validate verdict is the schema's, never the planner's. An invalid answer carries no `plan` key at all.
- Every path the planner reads goes through the resolvers the run uses (`realize_workflow`, `resolve_sub_workflow` + `validate_workflow_path`, `scan_models`). It opens no file by a path it computed itself.
- Fixtures declare their own `cost` blocks and their own `model_name` strings; no test names a real model.
- The hub is never a reason for validate to fail: `model_info` errors, timeouts and missing tokens give `gb: null`, logged at `debug` only.
- `plugins/dw/skills/*/SKILL.md` must each stay under `12 * 1024` bytes (`tests/test_plugin_skills.py::SKILL_SIZE_LIMIT`). `ltx-2.5` is at 12182 and `minimax-h3` at 12239 - the cost-step rewrite must not add net bytes there.
- Work happens in the `cost-plan` worktree (`.claude/worktrees/cost-plan`, branch `cost-plan`, already merged with `develop` at 1af85ac). Run tests from there with `python -m pytest`.
- Commit messages follow the repo's shape: `feat(engine): #85 - ...`, `docs(mcp): #85 - ...`, one sentence in the imperative, ending with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## File structure

| File | Responsibility |
|---|---|
| `dw/realize.py` (modify) | `realize_workflow(pin_outputs=...)`; `read_sub_workflow()` factored out of `_digest` so the planner reads a child by the run's own resolution |
| `dw/plan.py` (new) | `build_plan(candidate, arguments, ...)` and its four parts: `fingerprint`, `estimate`, `downloads_required`, `list_entries` |
| `dw/server/app.py` (modify) | `plan` on the valid answer of `POST /api/validate`; `?sizes=` query parameter |
| `dw_mcp/server.py`, `dw_mcp/diagnose.py` (modify) | docstring and `COST_REFUSAL` teach quoting from the plan |
| `docs/SERVER.md`, `docs/MCP.md`, `docs/WORKFLOW_GUIDE.md`, `CLAUDE.md` (modify) | the `plan` block and the quoting rule |
| `plugins/dw/skills/{ltx-2.5,minimax-h3,minimax-music3}/SKILL.md` (modify) | the "quote cost" step |
| `tests/test_realize.py`, `tests/test_plan.py` (new), `tests/test_server.py`, `tests/test_mcp_server.py` | coverage |

---

### Task 1: `realize_workflow(pin_outputs=False)` and `read_sub_workflow`

**Files:**
- Modify: `dw/realize.py` (`realize_workflow` signature at line 44, `_pin` at ~140, `_pin_output` at ~172, `_digest` at ~220)
- Test: `tests/test_realize.py`

**Interfaces:**
- Produces: `realize_workflow(definition, arguments, seed, base_dir=None, prompt_dir=None, output_root=None, workflow_dir=None, pin_outputs=True) -> (realized, annotations)`. With `pin_outputs=False`, every `output:.../latest/...` reference is returned exactly as written; prompt inlining and the seed pin are unchanged.
- Produces: `read_sub_workflow(path, base_dir, workflow_dir) -> bytes | None` - the file a sub-workflow step's `path` resolves to, read through `resolve_sub_workflow` and `validate_workflow_path` exactly as `_digest` did, `None` when it cannot be read (logged at debug).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_realize.py` (the module already has `definition()`, `prompt_library` and `output_root` fixtures - `output_root` yields `(root, run_id)` with one run of `ltx2/Gyre` holding `still.png`):

```python
class TestUnpinnedOutputs:
    def test_pin_outputs_false_leaves_latest_as_written(self, output_root):
        root, _ = output_root
        spec = definition()
        spec["steps"][0]["pipeline"]["arguments"]["image"] = (
            "output:ltx2/Gyre/latest/still.png"
        )
        realized, _ = realize_workflow(spec, {}, 7, output_root=root, pin_outputs=False)
        assert (
            realized["steps"][0]["pipeline"]["arguments"]["image"]
            == "output:ltx2/Gyre/latest/still.png"
        )

    def test_pin_outputs_false_still_inlines_prompts(self, prompt_library):
        spec = definition()
        spec["variables"]["prompt"] = "prompt:scenic/dusk"
        realized, annotations = realize_workflow(
            spec, {}, 7, prompt_dir=prompt_library, pin_outputs=False
        )
        assert realized["variables"]["prompt"] == "a harbour at dusk"
        assert annotations["prompts"] == ["scenic/dusk"]

    def test_the_default_still_pins(self, output_root):
        root, run_id = output_root
        spec = definition()
        spec["steps"][0]["pipeline"]["arguments"]["image"] = (
            "output:ltx2/Gyre/latest/still.png"
        )
        realized, _ = realize_workflow(spec, {}, 7, output_root=root)
        assert (
            realized["steps"][0]["pipeline"]["arguments"]["image"]
            == f"output:ltx2/Gyre/{run_id}/still.png"
        )


class TestReadSubWorkflow:
    def test_reads_a_child_beside_the_parent(self, tmp_path):
        from dw.realize import read_sub_workflow

        child = {"id": "child", "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        raw = read_sub_workflow("child.json", str(tmp_path), str(tmp_path))
        assert json.loads(raw) == child

    def test_a_missing_child_reads_as_none(self, tmp_path):
        from dw.realize import read_sub_workflow

        assert read_sub_workflow("nope.json", str(tmp_path), str(tmp_path)) is None

    def test_a_child_outside_the_confinement_reads_as_none(self, tmp_path):
        from dw.realize import read_sub_workflow

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "child.json").write_text(json.dumps({"id": "c", "steps": []}))
        confined = tmp_path / "confined"
        confined.mkdir()
        assert (
            read_sub_workflow("../outside/child.json", str(confined), str(confined))
            is None
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_realize.py -k "Unpinned or ReadSub" -v`
Expected: FAIL - `TypeError: ... unexpected keyword argument 'pin_outputs'` and `ImportError: cannot import name 'read_sub_workflow'`.

- [ ] **Step 3: Implement**

In `dw/realize.py`:

Change the signature and docstring of `realize_workflow`:

```python
def realize_workflow(
    definition,
    arguments,
    seed,
    base_dir=None,
    prompt_dir=None,
    output_root=None,
    workflow_dir=None,
    pin_outputs=True,
):
```

Add to the `Args:` block:

```
        pin_outputs: Whether an 'output:.../latest/...' reference is rewritten
            to the run it resolves to. The run leaves this True; the planner
            (dw/plan.py) passes False so a run finishing between a validate
            call and the queue call does not change the fingerprint of
            identical work.
```

Change the `_pin` call to `realized = _pin(realized, annotations, base_dir, prompt_dir, output_root, pin_outputs)` and `_pin` to:

```python
def _pin(value, annotations, base_dir, prompt_dir, output_root, pin_outputs=True):
    """Rebuild a value with prompt references inlined and, when
    `pin_outputs`, output references pinned."""

    def transform(string):
        if string.startswith(PROMPT_PREFIX):
            return _inline_prompt(string, annotations, prompt_dir, base_dir)
        if pin_outputs and is_output_reference(string):
            return _pin_output(string, output_root)
        return string

    return _map_strings(value, transform)
```

Replace `_digest` with a public reader plus the digest over it:

```python
def read_sub_workflow(path, base_dir, workflow_dir):
    """The bytes of the sub-workflow file a step's `path` names, or None
    when it cannot be read.

    Resolved the way `Workflow.create_step_action` resolves it - beside the
    referencing file, then across the workflow search path, then through
    `validate_workflow_path` confined to the root it came from - so a path
    this run could not have loaded is not one realization (or the planner)
    reads either, and a catalog name the run composed is read rather than
    recorded as unreadable (#90).
    """
    try:
        candidate, root = resolve_sub_workflow(path, base_dir or ".", workflow_dir)
        validated = validate_workflow_path(candidate, root)
        with open(validated, "rb") as file:
            return file.read()
    except (SecurityError, OSError, ValueError, SubWorkflowNotFound) as e:
        logger.debug(f"Sub-workflow {path} could not be read: {e}")
        return None


def _digest(path, base_dir, workflow_dir):
    """The SHA-256 of a sub-workflow file, or None when it cannot be read."""
    raw = read_sub_workflow(path, base_dir, workflow_dir)
    return hashlib.sha256(raw).hexdigest() if raw is not None else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_realize.py -v`
Expected: all PASS, the pre-existing digest tests included.

- [ ] **Step 5: Commit**

```bash
git add dw/realize.py tests/test_realize.py
git commit -m "feat(engine): #85 - realize_workflow can leave 'latest' unpinned, and a sub-workflow is read by one resolver

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `dw/plan.py` - realization, expansion, `steps`, `list_entries`, the fingerprint

**Files:**
- Create: `dw/plan.py`
- Test: `tests/test_plan.py` (new)

**Interfaces:**
- Consumes: `realize_workflow(..., pin_outputs=False)` from Task 1; `Workflow.expanded_definition()` (`dw/workflow.py:393`), `Workflow(definition, output_dir, file_spec, workflow_dir)`.
- Produces:

```python
def build_plan(
    candidate,  # a dw.workflow.Workflow, as the route constructed it
    arguments,  # the caller's dict, already past argument_errors
    *,
    device,  # "cuda" | "mps" | "cpu" - the serving backend
    prompt_dir=None,
    cache_dir=None,
    lookup_sizes=True,
    cache_probe=None,  # stage 2; ignored here, cached_steps is always None
):
    """What a run of `candidate` with `arguments` will execute and cost."""
```

returning

```python
{
    "fingerprint": "sha256:<64 hex>",
    "steps": int,
    "list_entries": {variable: int},
    "cached_steps": None,
    "downloads_required": [...],  # Task 4; [] until then
    "estimate": {...},  # Task 3; placeholder until then
}
```

and the helpers Tasks 3-4 fill in: `fingerprint(expanded, definition) -> str`, `list_entries(definition_as_written, realized) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plan.py`:

```python
"""The plan a validate call answers with: what a run will execute for the
arguments given, fingerprinted so an acknowledgement can be bound to it."""

import copy
import json
import os

import pytest

from dw.plan import build_plan
from dw.runs import new_run_id
from dw.workflow import workflow_from_definition


def definition():
    """A list-driven workflow with a stored prompt and an output reference,
    so every mutable input the fingerprint must ignore or honour is here."""
    return {
        "id": "plan_test",
        "description": "docs only",
        "summary": "docs only",
        "seed": "variable:seed",
        "cost": [{"device": "cuda", "name": "card", "vram_gb": 8, "minutes": 10}],
        "variables": {
            "seed": 1,
            "prompt": "prompt:scenic/dusk",
            "frames": 25,
            "shots": [{"name": "a", "prompt": "one"}, {"name": "b", "prompt": "two"}],
        },
        "steps": [
            {
                "name": "still",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "org/still-model"},
                    "arguments": {
                        "prompt": "variable:prompt",
                        "image": "output:ltx2/Gyre/latest/still.png",
                        "num_frames": "variable:frames",
                    },
                },
            },
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"command": "x", "arguments": {"prompt": "item:prompt"}},
            },
        ],
    }


@pytest.fixture
def prompt_library(tmp_path):
    library = tmp_path / "prompts"
    (library / "scenic").mkdir(parents=True)
    (library / "scenic" / "dusk.json").write_text(
        json.dumps({"text": "a harbour at dusk"})
    )
    return library


@pytest.fixture
def output_root(tmp_path):
    root = tmp_path / "outputs"
    run = root / "ltx2" / "Gyre" / new_run_id({"a": 1})
    run.mkdir(parents=True)
    (run / "still.png").write_bytes(b"png")
    return root


@pytest.fixture
def plan(tmp_path, prompt_library, output_root, monkeypatch):
    """build_plan over the fixture, with the hub cache empty and the hub
    unreachable, so downloads never touch the network in these tests."""
    import dw.plan

    monkeypatch.setattr(dw.plan, "scan_models", lambda cache_dir=None: {"repos": []})
    monkeypatch.setattr(
        dw.plan,
        "model_info",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    def make(spec=None, arguments=None, **overrides):
        spec = definition() if spec is None else spec
        candidate = workflow_from_definition(
            copy.deepcopy(spec), str(output_root), str(tmp_path), str(tmp_path)
        )
        kwargs = dict(device="cuda", prompt_dir=str(prompt_library), lookup_sizes=False)
        kwargs.update(overrides)
        return build_plan(candidate, arguments or {}, **kwargs)

    return make


class TestShape:
    def test_the_documented_keys(self, plan):
        answer = plan()
        assert set(answer) == {
            "fingerprint",
            "steps",
            "list_entries",
            "cached_steps",
            "downloads_required",
            "estimate",
        }
        assert answer["fingerprint"].startswith("sha256:")
        assert len(answer["fingerprint"]) == len("sha256:") + 64
        assert answer["cached_steps"] is None

    def test_steps_counts_the_expanded_members(self, plan):
        assert plan()["steps"] == 3  # still, shot@a, shot@b

    def test_list_entries_is_the_callers_list_length(self, plan):
        shots = [{"name": n, "prompt": n} for n in "abcde"]
        answer = plan(arguments={"shots": shots})
        assert answer["list_entries"] == {"shots": 5}
        assert answer["steps"] == 6

    def test_a_literal_for_each_list_is_not_an_entry(self, plan):
        spec = definition()
        spec["steps"][1]["for_each"] = [{"name": "x"}, {"name": "y"}]
        del spec["variables"]["shots"]
        assert plan(spec)["list_entries"] == {}


class TestFingerprintIsStableAcross:
    def test_a_different_top_level_seed(self, plan):
        assert plan()["fingerprint"] == plan(arguments={"seed": 99})["fingerprint"]

    def test_a_different_step_seed(self, plan):
        spec = definition()
        spec["steps"][0]["seed"] = 5
        spec["steps"][0]["pipeline"]["seed"] = 5
        other = copy.deepcopy(spec)
        other["steps"][0]["seed"] = 6
        other["steps"][0]["pipeline"]["seed"] = 6
        assert plan(spec)["fingerprint"] == plan(other)["fingerprint"]

    def test_key_order(self, plan):
        spec = definition()
        reordered = {k: spec[k] for k in reversed(list(spec))}
        assert plan(spec)["fingerprint"] == plan(reordered)["fingerprint"]

    def test_description_summary_and_cost_edits(self, plan):
        spec = definition()
        spec["description"] = "rewritten"
        spec["summary"] = "rewritten"
        spec["cost"][0]["minutes"] = 99
        spec["configures"] = "templates/x"
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_a_new_run_landing_under_latest(self, plan, output_root):
        before = plan()["fingerprint"]
        newer = output_root / "ltx2" / "Gyre" / new_run_id({"b": 2})
        newer.mkdir(parents=True)
        (newer / "still.png").write_bytes(b"png2")
        assert plan()["fingerprint"] == before

    def test_argument_order(self, plan):
        a = plan(arguments={"frames": 9, "seed": 3})["fingerprint"]
        b = plan(arguments={"seed": 3, "frames": 9})["fingerprint"]
        assert a == b


class TestFingerprintChangesWith:
    def test_a_longer_list(self, plan):
        longer = [{"name": n, "prompt": n} for n in "abc"]
        assert plan()["fingerprint"] != plan(arguments={"shots": longer})["fingerprint"]

    def test_a_changed_stored_prompt(self, plan, prompt_library):
        before = plan()["fingerprint"]
        (prompt_library / "scenic" / "dusk.json").write_text(
            json.dumps({"text": "a harbour at dawn"})
        )
        assert plan()["fingerprint"] != before

    def test_a_changed_numeric_argument(self, plan):
        assert plan()["fingerprint"] != plan(arguments={"frames": 121})["fingerprint"]

    def test_a_different_asset_name(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["arguments"]["image"] = "asset:a.png"
        other = copy.deepcopy(spec)
        other["steps"][0]["pipeline"]["arguments"]["image"] = "asset:b.png"
        assert plan(spec)["fingerprint"] != plan(other)["fingerprint"]

    def test_a_step_added_removed_or_renamed(self, plan):
        base = plan()["fingerprint"]
        renamed = definition()
        renamed["steps"][0]["name"] = "frame"
        removed = definition()
        del removed["steps"][1]
        added = definition()
        added["steps"].append(
            {"name": "extra", "task": {"command": "x", "arguments": {}}}
        )
        assert (
            len(
                {
                    base,
                    plan(renamed)["fingerprint"],
                    plan(removed)["fingerprint"],
                    plan(added)["fingerprint"],
                }
            )
            == 4
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_plan.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'dw.plan'`.

- [ ] **Step 3: Implement `dw/plan.py`**

```python
"""The plan a validate call answers with: what a run of a workflow with a
caller's arguments will actually execute, what it will have to download
first, and what the workflow's own cost block says it will take - with a
fingerprint over the work, so an acknowledgement can be bound to it and a
run whose shape changed after consent refused (#85, stage 2).

Everything here is derived from the same resolvers the run uses -
`realize_workflow` folds the arguments and inlines the prompts, and
`Workflow.expanded_definition` substitutes and expands `for_each` - so the
plan describes the run and not an approximation of it. Nothing here knows a
model: every minute comes from a `cost` block and every repo name from a
`from_pretrained_arguments`.
"""

import copy
import hashlib
import json
import logging
import os

from huggingface_hub import model_info

from .hub_cache import scan_models
from .realize import (
    BUILTIN_PREFIX,
    VARIABLE_PREFIX,
    read_sub_workflow,
    realize_workflow,
)
from .security import validate_url
from .workflow import Workflow

logger = logging.getLogger("dw")

FINGERPRINT_PREFIX = "sha256:"
# Top-level keys that document a workflow rather than shape its work
DOCUMENTATION_KEYS = ("cost", "description", "summary", "configures")
FOR_EACH_KEY = "for_each"
SIZE_LOOKUP_TIMEOUT = 5.0
GIB = 1024**3


def build_plan(
    candidate,
    arguments,
    *,
    device,
    prompt_dir=None,
    cache_dir=None,
    lookup_sizes=True,
    cache_probe=None,
):
    """What a run of `candidate` with `arguments` will execute and cost.

    Args:
        candidate: The Workflow the route built - it carries the file spec
            (so base_dir), the output root and the confinement a run has.
        arguments: The caller's arguments, already past `argument_errors`;
            an undeclared name or an uncoercible value raises here.
        device: The backend that is serving - 'cuda', 'mps' or 'cpu'.
        prompt_dir: The prompt library, for inlining.
        cache_dir: The hub cache to check downloads against; None for the
            default.
        lookup_sizes: Whether to ask the hub how large a missing repo is.
        cache_probe: Stage 2's step-cache probe; unused, `cached_steps` is
            always None until then.
    """
    definition = candidate.workflow_definition
    base_dir = (
        os.path.dirname(os.path.abspath(candidate.file_spec))
        if candidate.file_spec
        else None
    )
    realized, _ = realize_workflow(
        definition,
        arguments,
        seed=0,
        base_dir=base_dir,
        prompt_dir=prompt_dir,
        output_root=candidate.output_dir,
        workflow_dir=candidate.workflow_dir,
        pin_outputs=False,
    )
    # Arguments are already folded into the realized variables, so the
    # expansion takes none; it substitutes and expands exactly as the run
    expanded = Workflow(
        realized, candidate.output_dir, candidate.file_spec, candidate.workflow_dir
    ).expanded_definition()
    entries = list_entries(definition, realized)
    return {
        "fingerprint": fingerprint(expanded, definition),
        "steps": len(expanded.get("steps") or []),
        "list_entries": entries,
        "cached_steps": None,
        "downloads_required": [],
        "estimate": None,
    }


def list_entries(definition, realized):
    """{variable: length} for every `for_each` that names a list variable,
    read from the folded variables - a literal list is not an argument and
    is not listed."""
    variables = realized.get("variables") or {}
    entries = {}
    for step in definition.get("steps") or []:
        if not isinstance(step, dict):
            continue
        reference = step.get(FOR_EACH_KEY)
        if isinstance(reference, str) and reference.startswith(VARIABLE_PREFIX):
            name = reference.removeprefix(VARIABLE_PREFIX)
            value = variables.get(name)
            if isinstance(value, list):
                entries[name] = len(value)
    return entries


def fingerprint(expanded, definition):
    """SHA-256 over the expanded definition with everything that is not
    work removed: the seed wherever it sits, and the documentation keys.

    `definition` is the workflow as written, consulted for whether the
    top-level seed named a variable - if it did, that variable's folded
    value is the seed too and is blanked at its source.
    """
    doc = copy.deepcopy(expanded)
    doc.pop("seed", None)
    for key in DOCUMENTATION_KEYS:
        doc.pop(key, None)
    written_seed = definition.get("seed")
    if isinstance(written_seed, str) and written_seed.startswith(VARIABLE_PREFIX):
        name = written_seed.removeprefix(VARIABLE_PREFIX)
        variables = doc.get("variables")
        if isinstance(variables, dict) and name in variables:
            variables[name] = None
    for step in doc.get("steps") or []:
        if isinstance(step, dict):
            step.pop("seed", None)
            pipeline = step.get("pipeline")
            if isinstance(pipeline, dict):
                pipeline.pop("seed", None)
    # default=repr: a realized 'constant:' can be any Python value, and the
    # fingerprint only needs it to be stable, not round-trippable
    serialized = json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr
    )
    return FINGERPRINT_PREFIX + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_plan.py -v`
Expected: PASS. If `test_a_different_step_seed` fails on schema grounds, the fixture's step-level `seed` is not schema-checked here (nothing validates in `build_plan`), so check the scrub order instead.

- [ ] **Step 5: Commit**

```bash
git add dw/plan.py tests/test_plan.py
git commit -m "feat(engine): #85 - a plan fingerprints the work a run will execute for the caller's arguments

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: the estimate

**Files:**
- Modify: `dw/plan.py`
- Test: `tests/test_plan.py`

**Interfaces:**
- Consumes: `read_sub_workflow(path, base_dir, workflow_dir)` from Task 1; `list_entries` from Task 2.
- Produces: `estimate(definition, expanded, list_entries, device, base_dir, workflow_dir) -> dict` with keys `minutes` (float | None), `basis` (`"per_entry" | "catalog" | "other_device" | "unknown"`), `device`, `measured_on` (str | None), `partial` (bool). `build_plan` fills `"estimate"` with it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plan.py`:

```python
def cost(device="cuda", minutes=10, per_entry=None, name="card"):
    entry = {"device": device, "name": name, "vram_gb": 8, "minutes": minutes}
    if per_entry:
        entry["per_entry"] = per_entry
    return entry


class TestEstimate:
    def test_no_cost_block_is_unknown(self, plan):
        spec = definition()
        del spec["cost"]
        assert plan(spec)["estimate"] == {
            "minutes": None,
            "basis": "unknown",
            "device": "cuda",
            "measured_on": None,
            "partial": False,
        }

    def test_an_empty_cost_list_is_unknown(self, plan):
        spec = definition()
        spec["cost"] = []
        assert plan(spec)["estimate"]["basis"] == "unknown"

    def test_the_serving_devices_entry_is_the_catalog_figure(self, plan):
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2"), cost("cuda", 10, name="4090")]
        assert plan(spec)["estimate"] == {
            "minutes": 10.0,
            "basis": "catalog",
            "device": "cuda",
            "measured_on": "4090",
            "partial": False,
        }

    def test_another_devices_entry_is_reported_as_such(self, plan):
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
        assert plan(spec)["estimate"] == {
            "minutes": 40.0,
            "basis": "other_device",
            "device": "cuda",
            "measured_on": "M2",
            "partial": False,
        }

    def test_per_entry_scales_by_the_callers_list(self, plan):
        spec = definition()
        # 10 minutes for the 2-entry default, of which 3 per entry: 4 fixed
        spec["cost"] = [
            cost("cuda", 10, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        shots = [{"name": n, "prompt": n} for n in "abcde"]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert answer["minutes"] == 4 + 3 * 5
        assert answer["basis"] == "per_entry"

    def test_per_entry_floors_at_zero(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("cuda", 1, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        assert plan(spec)["estimate"]["minutes"] == 0.0

    def test_per_entry_naming_no_list_falls_back_to_catalog(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("cuda", 10, {"variable": "other", "minutes": 3, "entries": 2})
        ]
        answer = plan(spec)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_other_device_beats_per_entry(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("mps", 10, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        answer = plan(spec)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "other_device")

    def test_minutes_is_rounded_to_one_decimal(self, plan):
        spec = definition()
        spec["cost"] = [cost("cuda", 10.04)]
        assert plan(spec)["estimate"]["minutes"] == 10.0


def composing(child_path):
    return {
        "id": "parent",
        "cost": [cost("cuda", 2)],
        "steps": [
            {"name": "own", "task": {"command": "x", "arguments": {}}},
            {"name": "child", "workflow": {"path": child_path, "arguments": {}}},
        ],
    }


class TestSubWorkflowEstimate:
    def test_a_childs_catalog_cost_is_added(self, plan, tmp_path):
        child = {"id": "child", "cost": [cost("cuda", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        answer = plan(composing("child.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (7.0, False)

    def test_a_child_without_a_cost_makes_the_estimate_partial(self, plan, tmp_path):
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        answer = plan(composing("child.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)

    def test_an_unreadable_child_makes_the_estimate_partial(self, plan):
        answer = plan(composing("missing.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)

    def test_a_builtin_adds_nothing_and_is_not_partial(self, plan):
        answer = plan(composing("builtin:text-to-image.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, False)

    def test_a_child_measured_on_another_device_is_still_added(self, plan, tmp_path):
        child = {"id": "child", "cost": [cost("mps", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        answer = plan(composing("child.json"))["estimate"]
        assert answer["minutes"] == 7.0
        # the parent's own basis is what is reported
        assert answer["basis"] == "catalog"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_plan.py -k "Estimate" -v`
Expected: FAIL - `estimate` is `None`.

- [ ] **Step 3: Implement**

In `dw/plan.py`, replace `"estimate": None` in `build_plan` with

```python
        "estimate": estimate(
            definition, expanded, entries, device, base_dir, candidate.workflow_dir
        ),
```

and add:

```python
UNKNOWN = "unknown"
CATALOG = "catalog"
PER_ENTRY = "per_entry"
OTHER_DEVICE = "other_device"


def estimate(definition, expanded, list_entries, device, base_dir, workflow_dir):
    """Minutes from the workflow's own cost block, scaled by the caller's
    list when the block was measured per entry, plus each composed child's.

    `basis` names where the figure came from - the honesty is in the field,
    not in a fabricated number: 'unknown' is no cost block at all,
    'other_device' a figure measured on a backend other than the one
    serving (reported so the agent has something to scale, flagged so it
    is not quoted as a measurement), 'catalog' the stored total, and
    'per_entry' that total re-priced for the list actually passed.
    """
    own = _price(definition.get("cost"), device, list_entries)
    minutes = own["minutes"]
    partial = False
    for step in expanded.get("steps") or []:
        reference = step.get("workflow") if isinstance(step, dict) else None
        path = reference.get("path") if isinstance(reference, dict) else None
        if not isinstance(path, str) or path.startswith(BUILTIN_PREFIX):
            continue
        # A builtin is the parent's to price; a local child prices itself
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        child_cost = None
        if raw is not None:
            try:
                child_cost = json.loads(raw).get("cost")
            except (ValueError, AttributeError):
                child_cost = None
        child = _price(child_cost, device, {})
        if child["minutes"] is None:
            partial = True
        elif minutes is not None:
            minutes += child["minutes"]
        else:
            minutes = child["minutes"]
    return {
        "minutes": round(minutes, 1) if minutes is not None else None,
        "basis": own["basis"],
        "device": device,
        "measured_on": own["measured_on"],
        "partial": partial,
    }


def _price(cost, device, list_entries):
    """One cost list priced for `device` and `list_entries`, as
    {minutes, basis, measured_on}."""
    entries = [entry for entry in (cost or []) if isinstance(entry, dict)]
    if not entries:
        return {"minutes": None, "basis": UNKNOWN, "measured_on": None}
    chosen = next((entry for entry in entries if entry.get("device") == device), None)
    basis = CATALOG
    if chosen is None:
        chosen = entries[0]
        basis = OTHER_DEVICE
    minutes = float(chosen.get("minutes", 0))
    per = chosen.get("per_entry")
    if (
        basis == CATALOG
        and isinstance(per, dict)
        and per.get("variable") in list_entries
    ):
        count = list_entries[per["variable"]]
        each = float(per.get("minutes", 0))
        measured_with = int(per.get("entries", 0))
        minutes = max(0.0, (minutes - each * measured_with) + each * count)
        basis = PER_ENTRY
    return {"minutes": minutes, "basis": basis, "measured_on": chosen.get("name")}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_plan.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/plan.py tests/test_plan.py
git commit -m "feat(engine): #85 - the plan prices a run from its cost block, per entry when measured, plus each composed child

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `downloads_required`

**Files:**
- Modify: `dw/plan.py`
- Test: `tests/test_plan.py`

**Interfaces:**
- Consumes: `scan_models(cache_dir)` (`dw/hub_cache.py:37`, returns `{"repos": [{"repo_id": ...}, ...], ...}`), `validate_url` (`dw/security.py:327`, raises on a non-http(s) URL), `huggingface_hub.model_info(repo_id, files_metadata=True, timeout=...)`.
- Produces: `downloads_required(expanded, base_dir, workflow_dir, cache_dir, lookup_sizes) -> list[dict]`; each `{"repo": str, "gb": float | None}` or `{"repo": None, "url": str, "gb": None}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plan.py`:

```python
class TestDownloadsRequired:
    def test_a_repo_not_in_the_cache_is_required(self, plan):
        assert plan()["downloads_required"] == [{"repo": "org/still-model", "gb": None}]

    def test_a_cached_repo_is_not(self, plan, monkeypatch):
        import dw.plan

        monkeypatch.setattr(
            dw.plan,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": "org/still-model"}]},
        )
        assert plan()["downloads_required"] == []

    def test_cache_dir_reaches_scan_models(self, plan, monkeypatch):
        import dw.plan

        seen = []
        monkeypatch.setattr(
            dw.plan,
            "scan_models",
            lambda cache_dir=None: seen.append(cache_dir) or {"repos": []},
        )
        plan(cache_dir="/somewhere")
        assert seen == ["/somewhere"]

    def test_a_local_directory_is_not_a_download(self, plan, tmp_path):
        local = tmp_path / "weights"
        local.mkdir()
        spec = definition()
        spec["steps"][0]["pipeline"]["from_pretrained_arguments"]["model_name"] = str(
            local
        )
        assert plan(spec)["downloads_required"] == []

    def test_a_single_file_url_is_listed_without_a_size(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["from_pretrained_arguments"] = {
            "from_single_file": "https://example.test/x.safetensors"
        }
        assert plan(spec)["downloads_required"] == [
            {"repo": None, "url": "https://example.test/x.safetensors", "gb": None}
        ]

    def test_a_single_file_local_path_is_not_listed(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["from_pretrained_arguments"] = {
            "from_single_file": "checkpoints/x.safetensors"
        }
        assert plan(spec)["downloads_required"] == []

    def test_components_and_children_are_scanned_and_deduplicated(self, plan, tmp_path):
        spec = definition()
        spec["steps"][0]["pipeline"]["components"] = {
            "vae": {"from_pretrained_arguments": {"model_name": "org/vae"}}
        }
        child = {
            "id": "child",
            "steps": [
                {
                    "name": "c",
                    "pipeline": {
                        "configuration": {"component_type": "{Fake}"},
                        "from_pretrained_arguments": {"model_name": "org/still-model"},
                        "arguments": {},
                    },
                }
            ],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        spec["steps"].append(
            {"name": "sub", "workflow": {"path": "child.json", "arguments": {}}}
        )
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
            "org/still-model",
            "org/vae",
        ]

    def test_sizes_come_from_the_hub_in_gib(self, plan, monkeypatch):
        import dw.plan

        class Sibling:
            def __init__(self, size):
                self.size = size

        class Info:
            siblings = [Sibling(2 * 1024**3), Sibling(None), Sibling(512 * 1024**2)]

        calls = []

        def fake_model_info(name, **kwargs):
            calls.append((name, kwargs))
            return Info()

        monkeypatch.setattr(dw.plan, "model_info", fake_model_info)
        answer = plan(lookup_sizes=True)["downloads_required"]
        assert answer == [{"repo": "org/still-model", "gb": 2.5}]
        assert calls[0][0] == "org/still-model"
        assert calls[0][1]["files_metadata"] is True
        assert calls[0][1]["timeout"] == 5.0

    def test_a_hub_failure_is_a_null_size(self, plan, monkeypatch):
        import dw.plan

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(dw.plan, "model_info", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {"repo": "org/still-model", "gb": None}
        ]

    def test_lookup_sizes_false_never_calls_the_hub(self, plan, monkeypatch):
        import dw.plan

        def boom(*a, **k):
            raise AssertionError("must not be called")

        monkeypatch.setattr(dw.plan, "model_info", boom)
        plan(lookup_sizes=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_plan.py -k Downloads -v`
Expected: FAIL - `downloads_required` is `[]`.

- [ ] **Step 3: Implement**

In `build_plan`, replace `"downloads_required": []` with

```python
        "downloads_required": downloads_required(
            expanded, base_dir, candidate.workflow_dir, cache_dir, lookup_sizes
        ),
```

and add:

```python
FROM_PRETRAINED_KEY = "from_pretrained_arguments"
MODEL_NAME_KEY = "model_name"
SINGLE_FILE_KEY = "from_single_file"


def downloads_required(expanded, base_dir, workflow_dir, cache_dir, lookup_sizes):
    """The hub repos and checkpoint URLs the run would fetch before its
    first step: every `model_name` in the expanded definition (and in each
    composed child) that `scan_models` does not find, plus every
    `from_single_file` that is a URL. Sizes come from the hub when asked
    and are None whenever it does not answer - an offline box is a state,
    not an error, so nothing here raises or logs above debug.
    """
    names = []
    urls = []
    _collect_sources(expanded, names, urls)
    for step in expanded.get("steps") or []:
        reference = step.get("workflow") if isinstance(step, dict) else None
        path = reference.get("path") if isinstance(reference, dict) else None
        if not isinstance(path, str) or path.startswith(BUILTIN_PREFIX):
            continue
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        if raw is None:
            continue
        try:
            _collect_sources(json.loads(raw), names, urls)
        except ValueError:
            continue
    present = {repo.get("repo_id") for repo in scan_models(cache_dir).get("repos", [])}
    required = []
    for name in names:
        if name in present or os.path.isdir(name):
            continue
        required.append({"repo": name, "gb": _size_gb(name) if lookup_sizes else None})
    for url in urls:
        required.append({"repo": None, "url": url, "gb": None})
    return required


def _collect_sources(tree, names, urls):
    """Every from_pretrained source in a tree, first-seen order, deduplicated."""
    if isinstance(tree, dict):
        source = tree.get(FROM_PRETRAINED_KEY)
        if isinstance(source, dict):
            name = source.get(MODEL_NAME_KEY)
            if isinstance(name, str) and name not in names:
                names.append(name)
            single = source.get(SINGLE_FILE_KEY)
            if isinstance(single, str) and _is_url(single) and single not in urls:
                urls.append(single)
        for value in tree.values():
            _collect_sources(value, names, urls)
    elif isinstance(tree, list):
        for value in tree:
            _collect_sources(value, names, urls)


def _is_url(value):
    if not value.startswith(("http://", "https://")):
        return False
    try:
        validate_url(value)
        return True
    except Exception:
        return False


def _size_gb(name):
    """A repo's size in GiB to one decimal, or None when the hub does not
    say - unreachable, gated without a token, or a file with no size."""
    try:
        info = model_info(name, files_metadata=True, timeout=SIZE_LOOKUP_TIMEOUT)
        total = sum(s.size for s in (info.siblings or []) if getattr(s, "size", None))
    except Exception as e:
        logger.debug(f"No size for {name}: {e}")
        return None
    return round(total / GIB, 1) if total else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_plan.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/plan.py tests/test_plan.py
git commit -m "feat(engine): #85 - the plan names the weights a run would download first

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `plan` on `POST /api/validate`

**Files:**
- Modify: `dw/server/app.py` (the `validate_workflow` route at ~1256-1366; imports near line 53-66)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `build_plan(candidate, arguments, device=, prompt_dir=, lookup_sizes=)` from Tasks 2-4; `get_device`, `get_device_type` from `dw/__init__.py`.
- Produces: on a valid answer, `answer["plan"]` is the plan or `None`; the route takes `sizes: bool = Query(True)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py` (the module has `server`, `success_script`, `valid_workflow`, `video_workflow(job_id, with_cost=True)`):

```python
class TestValidatePlan:
    def test_a_valid_answer_carries_a_plan(self, server, monkeypatch):
        import dw.plan

        monkeypatch.setattr(
            dw.plan, "scan_models", lambda cache_dir=None: {"repos": []}
        )
        with server(success_script) as client:
            result = client.post(
                "/api/validate?sizes=false",
                json={"workflow": video_workflow("planned", with_cost=True)},
            ).json()
        assert result["valid"] is True
        plan = result["plan"]
        assert set(plan) == {
            "fingerprint",
            "steps",
            "list_entries",
            "cached_steps",
            "downloads_required",
            "estimate",
        }
        assert plan["steps"] == 1
        assert plan["estimate"]["basis"] in {"catalog", "other_device"}
        assert plan["estimate"]["minutes"] == 2.0
        assert plan["downloads_required"] == [{"repo": "m", "gb": None}]

    def test_an_invalid_answer_carries_no_plan(self, server):
        with server(success_script) as client:
            result = client.post(
                "/api/validate", json={"workflow": {"id": "broken", "steps": "no"}}
            ).json()
        assert result["valid"] is False
        assert "plan" not in result

    def test_a_planner_failure_is_a_null_plan_not_a_verdict(self, server, monkeypatch):
        import dw.server.app as app_module

        def boom(*a, **k):
            raise RuntimeError("planner broke")

        monkeypatch.setattr(app_module, "build_plan", boom)
        with server(success_script) as client:
            result = client.post(
                "/api/validate", json={"workflow": valid_workflow("v")}
            ).json()
        assert result["valid"] is True
        assert result["plan"] is None

    def test_sizes_reaches_the_planner(self, server, monkeypatch):
        import dw.server.app as app_module

        seen = []

        def spy(candidate, arguments, **kwargs):
            seen.append(kwargs["lookup_sizes"])
            return {
                "fingerprint": "sha256:0",
                "steps": 0,
                "list_entries": {},
                "cached_steps": None,
                "downloads_required": [],
                "estimate": None,
            }

        monkeypatch.setattr(app_module, "build_plan", spy)
        with server(success_script) as client:
            client.post("/api/validate", json={"workflow": valid_workflow("v")})
            client.post(
                "/api/validate?sizes=false", json={"workflow": valid_workflow("v")}
            )
        assert seen == [True, False]

    def test_the_plan_sees_the_callers_arguments(self, server, monkeypatch):
        import dw.plan

        monkeypatch.setattr(
            dw.plan, "scan_models", lambda cache_dir=None: {"repos": []}
        )
        with server(success_script) as client:
            body = {"workflow": valid_workflow("v")}
            one = client.post("/api/validate?sizes=false", json=body).json()["plan"]
            body["arguments"] = {"prompt": "something else"}
            two = client.post("/api/validate?sizes=false", json=body).json()["plan"]
        assert one["fingerprint"] != two["fingerprint"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server.py -k ValidatePlan -v`
Expected: FAIL - `KeyError: 'plan'` / `AttributeError: module has no attribute 'build_plan'`.

- [ ] **Step 3: Implement**

In `dw/server/app.py`, beside the other `..` imports (line ~53-66), add:

```python
from ..plan import build_plan
```

Change the route signature:

```python
    @app.post("/api/validate")
    def validate_workflow(
        request: JobRequest,
        ws: Workspace = Depends(selected_workspace),
        sizes: bool = Query(
            True,
            description="Ask the hub how large each missing model is; false "
            "skips the network for a faster answer",
        ),
    ):
```

(`Query` is not yet imported: line 24 is `from fastapi import Depends, FastAPI, HTTPException, Request` - add `Query` to it.)

Extend the docstring's last sentence with: `A valid answer also carries a plan: the fingerprint of the work these arguments produce, the step count, the list lengths, the model repos not in the cache, and an estimate from the workflow's cost block.`

After the `answer = {...}` block and the `checked_arguments` conditional, before `return answer`:

```python
        # What the run will execute for these arguments, fingerprinted so
        # an acknowledgement can be bound to it (#85). Best effort: the
        # verdict above is the schema's and the planner may not change it
        try:
            from .. import get_device, get_device_type

            answer["plan"] = build_plan(
                candidate,
                request.arguments,
                device=get_device_type(get_device()),
                prompt_dir=workspace.prompts,
                lookup_sizes=sizes,
            )
        except Exception:
            logger.exception("Plan could not be built")
            answer["plan"] = None
        return answer
```

(`get_device`/`get_device_type` are imported inside functions elsewhere in this file - lines ~2687 and ~2720 - because `dw/__init__` does device detection at import; follow that.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_server.py -k "ValidatePlan or validate" -v`
Expected: PASS, the pre-existing validate tests included.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "feat(server): #85 - a valid pre-flight answers with the run's plan

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: MCP surface and the refusal message

**Files:**
- Modify: `dw_mcp/server.py` (`validate_workflow` docstring at ~648-688), `dw_mcp/diagnose.py` (`COST_REFUSAL` at line 25)
- Test: `tests/test_mcp_server.py`, `tests/test_mcp_diagnose.py`

**Interfaces:**
- Consumes: the `plan` field from Task 5, unchanged through `authoring.validate_workflow` (it returns the server's JSON as-is, so no client change).
- Produces: nothing new in code; the docstring and the refusal are the deliverable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_server.py` (check how the module reaches the tool functions - it builds the server with `create_server`/`build_server` and reads tools by name; mirror the nearest docstring test around line 865):

```python
async def test_validate_workflow_teaches_quoting_from_the_plan():
    tools = await tools_of(server_over(ok({})))
    doc = tools["validate_workflow"].description
    assert "plan" in doc
    assert "downloads_required" in doc
    assert "estimate" in doc
    assert "basis" in doc
```

(`tools_of`, `server_over` and `ok` are the module's existing helpers - line ~144 - and the file's async tests run under the existing pytest-asyncio configuration; copy the decorator, if any, from `test_read_only_tools_are_annotated_read_only`.)

Append to `tests/test_mcp_diagnose.py`:

```python
def test_the_refusal_says_to_quote_the_plan():
    from dw_mcp.diagnose import COST_REFUSAL

    assert "plan" in COST_REFUSAL
    assert "validate_workflow" in COST_REFUSAL
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mcp_server.py tests/test_mcp_diagnose.py -k "plan" -v`
Expected: FAIL on the `plan` / `downloads_required` assertions.

- [ ] **Step 3: Implement**

`dw_mcp/diagnose.py`:

```python
COST_REFUSAL = (
    "Running a workflow occupies the GPU for minutes and the engine runs one "
    "job at a time. Call `validate_workflow` with the arguments you will run "
    "with (free): its `plan` says what will execute - `estimate.minutes` with "
    "its `basis`, and any weights in `downloads_required` this box has to "
    "fetch first. Tell the user that number, get their go-ahead, then call "
    "again with acknowledged_cost=true."
)
```

`dw_mcp/server.py`, append a paragraph to the `validate_workflow` docstring (before the closing `"""`):

```
        A valid answer carries `plan`: what will execute for these
        arguments. Quote `plan.estimate.minutes` with its `basis` -
        `per_entry` or `catalog` is a measured figure re-priced for your
        list, `other_device` a figure from another accelerator (say so),
        `unknown` no figure at all - and name each `downloads_required`
        entry as its own line item ("and 41 GB of weights this box does not
        have"); `gb` is null when the hub could not be asked. `steps` and
        `list_entries` say how many members the list actually produced.
        `plan` is null when it could not be built; the verdict stands.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_mcp_server.py tests/test_mcp_diagnose.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/server.py dw_mcp/diagnose.py tests/test_mcp_server.py tests/test_mcp_diagnose.py
git commit -m "docs(mcp): #85 - the number to say out loud is the plan's, not the listing's

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: docs, skills and the release note

**Files:**
- Modify: `docs/SERVER.md` (~252-269), `docs/MCP.md` (`## The cost gate`, ~312-347, and the `validate_workflow` row of the tool table), `docs/WORKFLOW_GUIDE.md` (~439-445 and step 4 of "The loop" ~481-487), `CLAUDE.md` (the `POST /api/validate` line under Critical Gotchas), `plugins/dw/skills/{ltx-2.5,minimax-h3,minimax-music3}/SKILL.md` (the "Run and judge" step 2), `docs/proposals/acknowledged-cost-binding.md` (status line)
- Test: `tests/test_plugin_skills.py`, `tests/test_docs_links.py` (existing)

**Interfaces:**
- Consumes: the plan shape from Tasks 2-5. No code.

- [ ] **Step 1: Check the skills' byte headroom**

Run: `wc -c plugins/dw/skills/*/SKILL.md`
Expected: `ltx-2.5` 12182, `minimax-h3` 12239, `minimax-music3` 10505; cap 12288. The two tight ones must not grow.

- [ ] **Step 2: Docs**

`docs/SERVER.md`, after the `POST /api/validate` bullet's last sentence (`...about the stored defaults only.`), add a paragraph inside the same bullet:

```
  A valid answer also carries `plan`, what the run will execute for those
  arguments: `fingerprint` (`sha256:…` over the realized, expanded
  definition with the seed and the documentation keys removed and
  `output:…/latest/…` left unpinned - the same work hashes the same, a
  longer list or an edited stored prompt does not); `steps`, the expanded
  member count; `list_entries`, `{variable: length}` for each `for_each`
  over a list variable; `cached_steps`, reserved (`null`);
  `downloads_required`, each `model_name` the hub cache does not hold as
  `{repo, gb}` (`gb` from the hub, `null` when it could not be asked -
  `?sizes=false` skips the hub) and each `from_single_file` URL as
  `{repo: null, url, gb: null}`; and `estimate`, `{minutes, basis,
  device, measured_on, partial}` from the workflow's own `cost` block -
  `basis` is `catalog` (the stored total), `per_entry` (re-priced for the
  list passed, when the entry carries `per_entry`), `other_device` (no
  entry for the serving backend; the first entry's figure, which is a
  warning rather than a quote) or `unknown`; a composed child's cost is
  added and `partial` is true when a child has none. `plan` is `null` when
  it could not be built; an invalid answer carries no `plan` key.
```

`docs/MCP.md`:
- In the tool table, `validate_workflow` row: append to its description ` A valid answer carries `plan` - the fingerprint, step count, list lengths, `downloads_required` and `estimate` (with `basis`) for the arguments given; quote from it`.
- In "The intended loop", step 1: after `not the values you wrote` add `, and its `plan` is the number to say out loud: `estimate.minutes` with its `basis`, plus each `downloads_required` entry as a line item of its own`.

`docs/WORKFLOW_GUIDE.md`:
- Around line 439-445 (`quote the cost before running a list-driven workflow...`): replace `quote \`minutes - per_entry.minutes × per_entry.entries + per_entry.minutes × N\` for N entries, and without \`per_entry\` quote the total as the default list's.` with `\`validate_workflow\` with your \`arguments\` answers with a \`plan\` whose \`estimate\` already does that arithmetic (\`basis: per_entry\`), and without \`per_entry\` reports the default list's total (\`basis: catalog\`) - quote the plan's figure and say which basis it has.`
- Step 4 of "The loop" (`run_workflow with acknowledged_cost=true, after telling the user what it costs...`): after `Without the acknowledgement the call is refused.` insert `The figure to tell them is the \`plan\` on the validate answer - \`estimate.minutes\` with its \`basis\`, and every \`downloads_required\` entry named as its own line item, since weights not on this box are minutes and gigabytes the cost block never counted.` Keep the rest of the step (the `models/` lookup for a workflow with no cost of its own) as the fallback for `basis: unknown`.

`CLAUDE.md`, the Critical Gotchas bullet that begins `**A caller's \`arguments\` are checked before anything is queued**`: append one sentence: `A valid \`POST /api/validate\` answer also carries \`plan\` (\`dw/plan.py\`): the fingerprint of the work, step and list counts, \`downloads_required\` and a cost \`estimate\` with its \`basis\` - the number an agent quotes; \`plan: null\` when it could not be built, never a changed verdict.`

`docs/proposals/acknowledged-cost-binding.md`, first line under the title: change `Status: **design only**` to `Status: **stage 1 implemented** (the plan on validate); stage 2 (binding, the 409) not started. Design: docs/superpowers/specs/2026-09-12-acknowledged-cost-binding-design.md.` and keep the rest of that paragraph.

- [ ] **Step 3: Skills**

Each "Run and judge" step 2 currently opens `Quote the listing's \`cost\``. Rewrite the opening clause only, keeping every model-specific number that follows it (those are pinned by `tests/test_plugin_skills.py`):

`plugins/dw/skills/ltx-2.5/SKILL.md` step 2, replace
```
2. Quote the listing's `cost` - the run's whole wall clock, model loading
   included. Only `templates/ltx2/text-to-video` and `templates/ltx2/two-stage`
   declare one; for the other six say so and give the shape of the spend
```
with
```
2. Quote `plan.estimate` from the validate answer (whole wall clock, loading
   included) and name any `downloads_required`. Only `text-to-video` and
   `two-stage` carry a `cost`; for the other six say so and give the shape
```
(Net change is negative; re-run `wc -c` and confirm under 12288.)

`plugins/dw/skills/minimax-h3/SKILL.md` step 2, replace
```
2. Quote the listing's `cost` (warm minutes on the card it was measured on;
   a first load is longer). When it declares none, say so and give the shape
```
with
```
2. Quote `plan.estimate` from the validate answer (warm minutes; a first
   load, and any `downloads_required`, is longer). When `basis` is
   `unknown`, say so and give the shape
```
Then trim elsewhere in that file if `wc -c` exceeds 12288 - the "Run and judge" step 1 `first - free, and it catches arguments the pipeline does not accept.` can lose `does not accept` → `rejects` (-9 bytes) and similar; keep every number.

`plugins/dw/skills/minimax-music3/SKILL.md` step 2, replace
```
2. Quote the listing's `cost` (warm minutes on the card it was measured on;
   a first load is longer). When the listing declares none, say so and give
```
with
```
2. Quote `plan.estimate` from the validate answer (warm minutes on the card it
   was measured on; a first load is longer). When `basis` is `unknown`, say so and give
```

- [ ] **Step 4: Run the doc and skill tests**

Run: `python -m pytest tests/test_plugin_skills.py tests/test_docs_links.py -v && wc -c plugins/dw/skills/*/SKILL.md`
Expected: PASS; every skill under 12288 bytes.

- [ ] **Step 5: Commit**

```bash
git add docs/SERVER.md docs/MCP.md docs/WORKFLOW_GUIDE.md CLAUDE.md plugins/dw/skills docs/proposals/acknowledged-cost-binding.md
git commit -m "docs: #85 - the plan on validate, and the skills quote from it rather than the listing

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: whole-suite verification

**Files:** none new.

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q -x --ignore=tests/test_integration.py 2>&1 | tail -5`
Expected: all pass (the tree was 3850 passing at ae4bf9b; expect that plus the new tests). `test_integration.py` needs a GPU and is skipped by marker in CI - run it only if it runs on this box today.

- [ ] **Step 2: Run the UI tests if any `.ts` changed** (none should have): skip.

- [ ] **Step 3: Check the branch is clean and fully committed**

Run: `git status --short && git log --oneline develop..HEAD`
Expected: clean; seven commits after the merge commit.

---

## Self-review

**Spec coverage (stage 1, sections 1-5):**
- 1.1 realization/expansion, `steps`, `list_entries` - Task 2. Note the deviation: `build_plan` takes the route's `Workflow` rather than `(definition, base_dir, output_root, workflow_dir)`, because expansion has to go through `Workflow.expanded_definition` (realization does not substitute `variable:` references, and `for_each` needs the substituted list) and the `Workflow` already carries the confinement the run has. The spec's constraint on `dw.server`/`dw.worker` imports holds.
- 1.2 fingerprint and its table - Task 2, every cell a test.
- 1.3 estimate rules 1-5 - Task 3, including sub-workflows, `partial`, builtin.
- 1.4 downloads, sizes, `lookup_sizes` - Task 4.
- 1.5 `cached_steps: null` - Task 2.
- 2 the route, `?sizes=`, `plan: null`, no `plan` on invalid - Task 5.
- 3 MCP docstring, `COST_REFUSAL` - Task 6; `get_guide`'s section is WORKFLOW_GUIDE.md - Task 7.
- 4 docs and skills - Task 7.
- 5 tests - Tasks 1-6.
- Release note items - the proposal's status line in Task 7; the release note itself is owed with the other 2026-09-12 items and is not part of this plan.

**Type consistency:** `build_plan(candidate, arguments, *, device, prompt_dir, cache_dir, lookup_sizes, cache_probe)` is used identically in Tasks 2-5; `estimate(...)` and `downloads_required(...)` signatures match between their definition and their call in `build_plan`; `read_sub_workflow(path, base_dir, workflow_dir)` matches Task 1.
