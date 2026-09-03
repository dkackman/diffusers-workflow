# Cross-Step Memory Manager, Step-Output Cache & Artifact Downloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `on_demand` components a cross-step eviction strategy instead of failing on OOM, skip re-executing REPL steps whose inputs are unchanged, add a forced-download affordance everywhere a delete button already exists, and stop workflow reruns from silently overwriting prior output files.

**Architecture:** Two new process-wide singletons (`dw/pipeline_processors/memory_manager.py`, `dw/step_cache.py`) slot into existing extension points (`apply_on_demand_placement`, `Workflow.run`'s step loop) without changing their public signatures for callers that don't opt in. Output collision avoidance is a one-function change at `dw/result.py`'s single save choke point (`output_file_path`). Downloads are three new FastAPI routes returning `FileResponse`/JSON-with-`Content-Disposition`, paired with three small Svelte additions that reuse the existing delete-button pattern.

**Tech Stack:** Python 3.10+, PyTorch, FastAPI, Svelte 5 + lucide-svelte icons, pytest.

**Spec:** This plan (no separate spec doc — requirements were established via direct discussion and grounded by reading the current source; see "Design rationale" callouts inline).

## Global Constraints

- No new third-party dependencies.
- Every new module gets a `logger = logging.getLogger("dw")` (backend) matching existing convention — never `print`.
- All new FastAPI routes go through the same path-validation helpers the file already uses (`validate_path`, `resolve_workflow_name`, `resolve_prompt_name`, `_output_file`) — never construct a filesystem path from user input without one.
- New frontend code follows existing patterns in `ui/src/lib/api.ts` and the page components it's added to — read the surrounding code before editing, don't restructure it.
- Every task ends green: relevant `pytest` file(s) pass, and for frontend tasks `npm run check` in `ui/` passes.

---

### Task 1: Stop silently overwriting output files on rerun

**Model:** sonnet (contained, single-function change; correctness matters more than architecture)

**Design rationale:** ComfyUI's `SaveImage` node scans the output directory and increments a counter so a run never reuses a filename already on disk — repeat runs are guaranteed collision-free with no cooperation required from the caller. Mellon has no such guarantee: its default filename template embeds a random id, but a fixed template silently overwrites. `dw/result.py`'s `output_file_path()` is the single choke point every artifact write passes through (`dw/result.py:265-267` for JSON results, `dw/result.py:325` for every other content type, including the two places that recurse through it — dict artifacts at `dw/result.py:313-322` and multi-waveform audio at `dw/result.py:349-358`). Adding the check there, once, covers every artifact kind. This plan follows ComfyUI's precedent (never overwrite) rather than Mellon's (overwrite unless the template avoids it), since dw's existing naming (`{workflow_id}-{step_name}.{i}-{j}.{k}.ext`) is exactly the "fixed template" case Mellon warns silently clobbers.

**Files:**
- Modify: `dw/result.py:25-34` (`output_file_path`)
- Modify: `dw/server/app.py:904-917` (stale comment describing the old overwrite behavior)
- Test: `tests/test_result_output_naming.py` (new)

**Interfaces:**
- Produces: `output_file_path(output_dir, file_name) -> str` — same signature as today; now guarantees the returned path does not exist yet at call time.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_result_output_naming.py
import os

from dw.result import output_file_path


def test_output_file_path_no_collision_is_unchanged(tmp_path):
    path = output_file_path(str(tmp_path), "run-step.0-0.0.png")

    assert path == str(tmp_path / "run-step.0-0.0.png")


def test_output_file_path_dedupes_existing_file(tmp_path):
    (tmp_path / "run-step.0-0.0.png").write_bytes(b"first")

    path = output_file_path(str(tmp_path), "run-step.0-0.0.png")

    assert path == str(tmp_path / "run-step.0-0.0-2.png")


def test_output_file_path_dedupes_multiple_collisions(tmp_path):
    (tmp_path / "run-step.0-0.0.png").write_bytes(b"first")
    (tmp_path / "run-step.0-0.0-2.png").write_bytes(b"second")

    path = output_file_path(str(tmp_path), "run-step.0-0.0.png")

    assert path == str(tmp_path / "run-step.0-0.0-3.png")


def test_output_file_path_preserves_extension(tmp_path):
    (tmp_path / "clip.0-0.0.mp4").write_bytes(b"first")

    path = output_file_path(str(tmp_path), "clip.0-0.0.mp4")

    assert path == str(tmp_path / "clip.0-0.0-2.mp4")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_result_output_naming.py -v`
Expected: FAIL — `test_output_file_path_dedupes_existing_file` and the two others asserting a `-2`/`-3` suffix fail because `output_file_path` currently returns the colliding name unchanged; `test_output_file_path_no_collision_is_unchanged` already passes.

- [ ] **Step 3: Implement collision avoidance**

Replace `dw/result.py:25-34`:

```python
def output_file_path(output_dir, file_name):
    """The path a result file is written to, confined to the output directory.

    Every part of the name is workflow-supplied - the workflow id, the step
    name, a result's file_base_name, and the keys of a dict artifact - and
    they are concatenated into a file name. Without this a name carrying a
    path separator would write outside the output directory, so the joined
    path goes through the same validator every other path in the engine does.

    A rerun that would otherwise produce a name already on disk gets a
    '-2', '-3', ... counter instead of silently overwriting it - the same
    guarantee ComfyUI's SaveImage node makes by scanning its output
    directory before every write.
    """
    candidate = validate_output_path(os.path.join(output_dir, file_name), output_dir)
    return _dedupe_existing_path(candidate)


def _dedupe_existing_path(path):
    """Append an incrementing counter before the extension until `path` is free."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        candidate = f"{base}-{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_result_output_naming.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run the full existing result/workflow test suite for regressions**

Run: `pytest tests/ -k "result or workflow" -v`
Expected: PASS — no existing test asserts that a second run overwrites a first (if one does, it was asserting the bug; update it to expect a `-2` suffix instead)

- [ ] **Step 6: Update the now-stale gallery comment**

In `dw/server/app.py`, find the comment block around line 904-917 (inside `_gallery_entries`, on the `url` field) that reads:

```python
                    # Quoted (slashes kept literal): a name carrying '#', '?'
                    # or '%' would otherwise break the src the gallery
                    # renders it into. The mtime rides along because a rerun
                    # overwrites the same name - on an unchanging URL the
                    # browser would keep showing the image it cached from
                    # the previous run
```

Replace the last three lines with:

```python
                    # Quoted (slashes kept literal): a name carrying '#', '?'
                    # or '%' would otherwise break the src the gallery
                    # renders it into. The mtime still rides along for cache
                    # busting when a file's content changes without its name
                    # changing (e.g. a manual overwrite outside the engine) -
                    # normal reruns get a fresh name instead, see
                    # dw/result.py's output_file_path
```

- [ ] **Step 7: Commit**

```bash
git add dw/result.py dw/server/app.py tests/test_result_output_naming.py
git commit -m "fix: never overwrite an existing output file on rerun"
```

---

### Task 2: Cross-step memory manager for on-demand components

**Model:** opus (concurrency/device-placement correctness, OOM-retry control flow, new architectural extension point)

**Design rationale:** `apply_on_demand_placement` (`dw/pipeline_processors/pipeline.py:901-987`) already moves a single component to the accelerator around its own calls and back to `offload_device` when idle — but it has no idea what else is currently resident. Two `residency: on_demand` components used by different steps of the same workflow can both try to be on the accelerator at once and the second one's `component.to(device)` (line 958) just raises `torch.OutOfMemoryError`. There is no cross-step or cross-pipeline registry of loaded models anywhere in the codebase today — confirmed by reading `dw/workflow.py` and `dw/pipeline_processors/pipeline.py` in full: `shared_components`/`pipelines` are workflow-declared sharing within one run, not a device-residency tracker. Mellon's `utils/memory_menager.py` solves exactly this for its node graph: a process-wide cache of loaded models tagged with `priority` + `last_used`, where a failed `.to(device)` walks the cache sorted by `(priority, last_used)`, evicts the lowest-priority/oldest resident, and retries. This task ports that idea to dw's existing `on_demand` mechanism — it does not touch `offload: model/sequential` or `group_offload`, which are diffusers' own hooks that dw doesn't control the individual `.to()` calls for.

**Files:**
- Create: `dw/pipeline_processors/memory_manager.py`
- Modify: `dw/pipeline_processors/pipeline.py:901-987` (`apply_on_demand_placement`)
- Test: `tests/test_memory_manager.py` (new)

**Interfaces:**
- Produces: `memory_manager` (module-level `MemoryManager` singleton) with `.register(component, priority=1)`, `.unregister(component)`, `.load(component, device, offload_device) -> None`, `.mark_idle(component, offload_device) -> None`.
- Consumed by Task 3: `apply_on_demand_placement`'s new `priority` parameter and the `on_demand` wrapper closure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory_manager.py
import torch
import pytest

from dw.pipeline_processors.memory_manager import MemoryManager


class FakeModel:
    """Stands in for an nn.Module: tracks device via .to(), and can be told
    to raise OutOfMemoryError on its next .to() call to a given device."""

    def __init__(self, name):
        self.name = name
        self.device = torch.device("cpu")
        self.oom_on = None

    def to(self, device):
        device = torch.device(device) if not isinstance(device, torch.device) else device
        if self.oom_on is not None and str(device) == str(self.oom_on):
            self.oom_on = None  # only OOM once, so a retry after eviction succeeds
            raise torch.OutOfMemoryError(f"{self.name} cannot fit on {device}")
        self.device = device
        return self


def test_load_moves_component_with_no_contention():
    mm = MemoryManager()
    model = FakeModel("a")
    mm.register(model, priority=1)

    mm.load(model, "cuda:0", "cpu")

    assert str(model.device) == "cuda:0"


def test_load_evicts_lower_priority_resident_on_oom():
    mm = MemoryManager()
    low = FakeModel("low")
    high = FakeModel("high")
    mm.register(low, priority=1)
    mm.register(high, priority=5)

    mm.load(low, "cuda:0", "cpu")
    assert str(low.device) == "cuda:0"

    high.oom_on = "cuda:0"
    mm.load(high, "cuda:0", "cpu")

    assert str(high.device) == "cuda:0"
    assert str(low.device) == "cpu"  # evicted to make room


def test_load_evicts_least_recently_used_when_priority_ties():
    mm = MemoryManager()
    first = FakeModel("first")
    second = FakeModel("second")
    mm.register(first, priority=1)
    mm.register(second, priority=1)

    mm.load(first, "cuda:0", "cpu")
    mm.load(second, "cuda:0", "cpu")  # both would now be "on cuda:0" in real life;
    # here they're just two independent fakes, so simulate contention directly:

    third = FakeModel("third")
    mm.register(third, priority=1)
    third.oom_on = "cuda:0"
    mm.load(third, "cuda:0", "cpu")

    # first was loaded before second, so it's the least-recently-used tie-break
    assert str(first.device) == "cpu"
    assert str(second.device) == "cuda:0"  # untouched - not the eviction candidate


def test_load_reraises_when_nothing_left_to_evict():
    mm = MemoryManager()
    model = FakeModel("only")
    mm.register(model, priority=1)
    model.oom_on = "cuda:0"

    with pytest.raises(torch.OutOfMemoryError):
        mm.load(model, "cuda:0", "cpu")


def test_unregister_removes_component_from_eviction_pool():
    mm = MemoryManager()
    victim = FakeModel("victim")
    mm.register(victim, priority=1)
    mm.load(victim, "cuda:0", "cpu")
    mm.unregister(victim)

    other = FakeModel("other")
    mm.register(other, priority=1)
    other.oom_on = "cuda:0"

    with pytest.raises(torch.OutOfMemoryError):
        mm.load(other, "cuda:0", "cpu")  # victim is gone, nothing left to evict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.pipeline_processors.memory_manager'`

- [ ] **Step 3: Implement `MemoryManager`**

```python
# dw/pipeline_processors/memory_manager.py
"""Cross-step model residency tracking for 'residency: on_demand' components.

Each on_demand component (apply_on_demand_placement, pipeline.py) moves
itself to the accelerator around its own calls and back off when idle, with
no awareness of what else is currently resident. On a card too small to
hold two such components at once, the second one's move to the accelerator
raises torch.OutOfMemoryError instead of making room.

MemoryManager is a process-wide registry: every on_demand component is
registered with a priority, and a move that hits OOM evicts whichever
*other* registered component is lowest-priority and least-recently-used,
then retries - the same trade Mellon's memory manager (utils/memory_menager.py)
makes for its node graph. It is not consulted by 'offload: model/sequential'
or 'group_offload', which install diffusers' own hooks - dw does not own
the individual .to() calls those make, so there is nothing to intercept.
"""
import time
import logging

import torch

logger = logging.getLogger("dw")


class MemoryManager:
    def __init__(self):
        self._entries = {}  # id(component) -> {component, priority, last_used, device}

    def register(self, component, priority=1):
        self._entries[id(component)] = {
            "component": component,
            "priority": priority,
            "last_used": time.time(),
            "device": component.device,
        }

    def unregister(self, component):
        self._entries.pop(id(component), None)

    def load(self, component, device, offload_device):
        """Move `component` onto `device`, evicting lower-priority residents on OOM."""
        entry = self._entries.get(id(component))
        if entry is not None:
            entry["last_used"] = time.time()

        while True:
            try:
                component.to(device)
                if entry is not None:
                    entry["device"] = torch.device(device) if not isinstance(device, torch.device) else device
                return
            except torch.OutOfMemoryError:
                victim_id = self._pick_eviction_candidate(device, exclude_id=id(component))
                if victim_id is None:
                    raise
                self._evict(victim_id, offload_device)

    def mark_idle(self, component, offload_device):
        """Record that `component` has been moved back to `offload_device`."""
        entry = self._entries.get(id(component))
        if entry is not None:
            entry["device"] = torch.device(offload_device) if not isinstance(offload_device, torch.device) else offload_device

    def _pick_eviction_candidate(self, device, exclude_id):
        device = str(device)
        candidates = [
            (entry["priority"], entry["last_used"], comp_id)
            for comp_id, entry in self._entries.items()
            if comp_id != exclude_id and str(entry["device"]) == device
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2]

    def _evict(self, comp_id, offload_device):
        entry = self._entries.get(comp_id)
        if entry is None:
            return
        logger.debug(f"Evicting a lower-priority on-demand component to free {entry['device']}")
        entry["component"].to(offload_device)
        entry["device"] = torch.device(offload_device) if not isinstance(offload_device, torch.device) else offload_device


memory_manager = MemoryManager()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory_manager.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Wire `apply_on_demand_placement` through the memory manager**

In `dw/pipeline_processors/pipeline.py`, add the import near the top with the other `pipeline_processors`-local imports:

```python
from .memory_manager import memory_manager
```

Then change `apply_on_demand_placement` (currently `dw/pipeline_processors/pipeline.py:901-987`) to accept and use a `priority` argument. Replace the signature and body's direct `component.to(...)` calls:

```python
def apply_on_demand_placement(
    component, component_name, device, group_offloaded, offload_device="cpu", priority=1
):
    """Keep a component in system memory and move it to the device only while it runs.
    ...
    (docstring unchanged)

    Args:
        component: The component to place
        component_name: Name of the component, for logging
        device: Device to run the component on
        group_offloaded: Whether group offloading was applied to this component
        offload_device: Where the component rests between calls
        priority: This component's standing in the cross-step eviction order -
            higher survives longer under memory pressure from other on_demand
            components. See MemoryManager.

    Raises:
        ValueError: If the component is also group offloaded
    """
    if group_offloaded:
        raise ValueError(
            f"Component '{component_name}' sets both 'group_offload' and "
            "'residency: on_demand'. A group offloaded module holds one group at a "
            "time and ignores the whole-model moves on-demand placement makes, so "
            "the two cannot both own its placement - pick one"
        )

    if get_device_type(device) == "cpu":
        logger.debug(
            f"Ignoring 'residency: on_demand' for {component_name} - {device} is the "
            "device it would rest on anyway"
        )
        return

    component.to(offload_device)
    memory_manager.register(component, priority=priority)

    state = {"depth": 0}

    def wrap(entry_point):
        original = getattr(component, entry_point, None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def on_demand(*args, **kwargs):
            if state["depth"] == 0:
                memory_manager.load(component, device, offload_device)
            state["depth"] += 1
            try:
                return original(*args, **kwargs)
            finally:
                state["depth"] -= 1
                if state["depth"] == 0:
                    component.to(offload_device)
                    memory_manager.mark_idle(component, offload_device)
                    empty_device_cache()

        setattr(component, entry_point, on_demand)
        return True

    wrapped = [name for name in _ON_DEMAND_ENTRY_POINTS if wrap(name)]
    if not wrapped:
        memory_manager.unregister(component)
        raise ValueError(
            f"Component '{component_name}' sets 'residency: on_demand' but defines "
            f"none of {', '.join(_ON_DEMAND_ENTRY_POINTS)}, so there is no call to "
            "move it around"
        )
    logger.info(
        f"Placing {component_name} on demand: resting on {offload_device}, "
        f"running on {device} around {', '.join(wrapped)} (priority {priority})"
    )
```

- [ ] **Step 6: Update the call site to pass priority (placeholder value — Task 3 wires the real config field)**

In `dw/pipeline_processors/pipeline.py`, at the `configure_components` call site (`dw/pipeline_processors/pipeline.py:712-718`):

```python
        if residency == "on_demand":
            apply_on_demand_placement(
                component,
                component_name,
                device if device is not None else default_device,
                group_offload_configuration is not None,
                priority=component_configuration.get("residency_priority", 1),
            )
```

- [ ] **Step 7: Run the full pipeline test suite for regressions**

Run: `pytest tests/ -k "pipeline or on_demand or residency" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add dw/pipeline_processors/memory_manager.py dw/pipeline_processors/pipeline.py tests/test_memory_manager.py
git commit -m "feat: cross-step eviction for residency:on_demand components under memory pressure"
```

---

### Task 3: `residency_priority` schema field, docs

**Model:** sonnet (schema + docs, low architectural risk but needs to match existing conventions exactly)

**Files:**
- Modify: `dw/workflow_schema.json` (add `residency_priority` to a component's configuration properties)
- Modify: `docs/ACCELERATION.md` (document the new field)
- Modify: `CLAUDE.md` (Critical Gotchas section)
- Test: extend `tests/test_schema.py` or equivalent schema test (find the existing test file with `grep -rl "residency" tests/`)

**Interfaces:**
- Consumes: `MemoryManager` from Task 2 (no code interface — this task is schema/docs only, plus the `residency_priority` key `apply_on_demand_placement`/`configure_components` already read as of Task 2 Step 6).

- [ ] **Step 1: Locate the schema's component configuration block**

Run: `grep -n '"residency"' dw/workflow_schema.json`

Read the surrounding `properties` object for a component's `configuration`.

- [ ] **Step 2: Add `residency_priority` next to `residency` in the schema**

Add a sibling property (matching the exact JSON Schema style already used for `residency` — an enum or string — in that file):

```json
"residency_priority": {
    "type": "integer",
    "default": 1,
    "description": "This component's standing in the cross-step eviction order for 'residency: on_demand' components under memory pressure - higher survives longer. See docs/ACCELERATION.md."
}
```

- [ ] **Step 3: Write/extend a schema validation test**

Find the existing schema test file: `grep -rl "workflow_schema" tests/`. Add a case validating a workflow whose component configuration sets `"residency": "on_demand", "residency_priority": 5` against the schema and asserting it's accepted, following that file's existing test style exactly (read 2-3 neighboring tests first to match fixture/assertion conventions before writing).

- [ ] **Step 4: Run the schema tests**

Run: `pytest tests/ -k schema -v`
Expected: PASS

- [ ] **Step 5: Document the field**

In `docs/ACCELERATION.md`, add a subsection near wherever `on_demand`/`residency` is currently documented (search: `grep -n "on_demand" docs/ACCELERATION.md`):

```markdown
### Cross-step eviction priority

Multiple `residency: on_demand` components across different steps of one
workflow share a single process-wide memory manager. If a second on-demand
component can't fit on the accelerator alongside a resident one, the
manager evicts the lowest-priority, least-recently-used resident and
retries — rather than failing the run. Set `residency_priority` (default
`1`, higher survives longer) on a component's configuration to protect it:

```json
"components": {
    "text_encoder": {
        "configuration": { "residency": "on_demand", "residency_priority": 5 }
    }
}
```

This only applies to `residency: on_demand` — `offload: model`/`sequential`
and `group_offload` install diffusers' own hooks and are unaffected.
```

- [ ] **Step 6: Update CLAUDE.md's Critical Gotchas**

Add one line to the "Critical Gotchas" bullet list in `CLAUDE.md`:

```markdown
- **`residency: on_demand` components share one process-wide eviction pool** (`dw/pipeline_processors/memory_manager.py`) — under memory pressure the lowest `residency_priority` (default `1`), least-recently-used resident is evicted and retried, not the calling component. Only applies to `on_demand`, not `offload`/`group_offload`.
```

- [ ] **Step 7: Commit**

```bash
git add dw/workflow_schema.json docs/ACCELERATION.md CLAUDE.md tests/
git commit -m "docs: document residency_priority and cross-step eviction"
```

---

### Task 4: Step-output cache core

**Model:** opus (equality semantics across tensors/PIL/numpy and the seed-identity gotcha are correctness-critical — a wrong cache hit silently returns a stale image)

**Design rationale:** dw's REPL already reuses loaded pipelines across `workflow run` calls (`previous_pipelines`, `dw/workflow.py:308-315`) for a documented 2-4x speedup — but it still re-executes every step's forward pass even when nothing feeding that step changed. Mellon's `NodeBase.__call__` (`mellon/NodeBase.py`) does the finer-grained version: it skips re-executing a node whose resolved params are unchanged via a type-aware `deep_equal` (handles tensors, PIL images, ndarrays, nested dicts/lists). This task ports that idea to dw's step loop.

**Critical correctness gotcha found while reading `dw/workflow.py`:** a step's *resolved* definition (`step_data`, after `replace_variables`) does **not** include its seed — `step_seed = step_data.get("seed", default_seed)` (`dw/workflow.py:350`) is computed separately and `default_seed` is `torch.Generator().seed()` freshly drawn per run when no seed is set in the workflow (`dw/workflow.py:297-302`). Comparing only `step_data` between runs would treat two runs with different random seeds as identical and incorrectly reuse the first run's image forever. The cache key must include `step_seed` explicitly.

**Second correctness boundary:** a step is only safe to skip if every `previous_result:` reference it makes (`referenced_result_names`, already defined at `dw/workflow.py:131-157` and reused as-is here) was *itself* served from cache in this same run — otherwise a change three steps upstream leaves a stale result three steps downstream. Sub-workflow steps (`"workflow"` key rather than `"pipeline"`/`"task"`) are excluded from caching in this task — their nested manifest rollup (`dw/workflow.py:369-370`) adds bookkeeping this task doesn't need to solve; deferred as future work.

**Files:**
- Create: `dw/step_cache.py`
- Test: `tests/test_step_cache.py` (new)

**Interfaces:**
- Consumes: `referenced_result_names` from `dw/workflow.py:131-157` (already exists, unmodified).
- Produces: `step_cache` (module-level `StepCache` singleton) with `.get(step_data, step_seed, hits_this_run) -> Result | None`, `.put(step_data, step_seed, result) -> None`, `.clear() -> None`. Consumed by Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_step_cache.py
from dw.step_cache import StepCache, deep_equal


class FakeResult:
    def __init__(self, label):
        self.label = label
        self.saved_files = [f"{label}.png"]


def test_deep_equal_matches_identical_nested_dicts():
    a = {"prompt": "a cat", "settings": {"steps": 9, "images": [1, 2, 3]}}
    b = {"prompt": "a cat", "settings": {"steps": 9, "images": [1, 2, 3]}}
    assert deep_equal(a, b)


def test_deep_equal_rejects_changed_nested_value():
    a = {"prompt": "a cat", "settings": {"steps": 9}}
    b = {"prompt": "a cat", "settings": {"steps": 25}}
    assert not deep_equal(a, b)


def test_step_cache_hit_on_unchanged_step_data_and_seed():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    cache.put(step_data, 42, result)

    hit = cache.get(step_data, 42, hits_this_run=set())

    assert hit is result


def test_step_cache_miss_when_step_data_changes():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    changed = {"name": "gen", "pipeline": {"arguments": {"prompt": "a dog"}}}
    hit = cache.get(changed, 42, hits_this_run=set())

    assert hit is None


def test_step_cache_miss_when_seed_changes():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    hit = cache.get(step_data, 99, hits_this_run=set())

    assert hit is None


def test_step_cache_miss_when_referenced_step_did_not_hit_this_run():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"image": "previous_result:image_generation"}},
    }
    cache.put(step_data, 7, FakeResult("first"))

    # image_generation was NOT in hits_this_run - it re-ran and may have changed
    hit = cache.get(step_data, 7, hits_this_run=set())

    assert hit is None


def test_step_cache_hit_when_referenced_step_did_hit_this_run():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"image": "previous_result:image_generation"}},
    }
    result = FakeResult("first")
    cache.put(step_data, 7, result)

    hit = cache.get(step_data, 7, hits_this_run={"image_generation"})

    assert hit is result


def test_step_cache_hit_when_referenced_step_hit_via_property_suffix():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"mask": "previous_result:segment.mask"}},
    }
    result = FakeResult("first")
    cache.put(step_data, 7, result)

    hit = cache.get(step_data, 7, hits_this_run={"segment"})

    assert hit is result


def test_step_cache_miss_on_first_run():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}

    assert cache.get(step_data, 42, hits_this_run=set()) is None


def test_step_cache_clear():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    cache.clear()

    assert cache.get(step_data, 42, hits_this_run=set()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_step_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dw.step_cache'`

- [ ] **Step 3: Implement `StepCache`**

```python
# dw/step_cache.py
"""Skip re-executing a step whose resolved definition, seed, and upstream
results are all unchanged since the last run in this process.

Reusing a loaded pipeline (Workflow.run's previous_pipelines) is only half
of what makes REPL iteration fast - the other half is not re-running a
step's forward pass at all when nothing feeding it changed, the way
Mellon's NodeBase skips a node whose resolved params match its last call
(deep value equality, not just identity - a step's arguments are plain
dicts/lists/scalars after variable substitution, not hashable).

A step is safe to skip only if:
  1. its own resolved definition (step_data) matches last run's, AND
  2. its seed matches last run's - step_data does NOT carry the seed
     (Workflow.run resolves it separately, and draws a fresh random one
     per run when the workflow sets none), so seed must be compared
     explicitly or two differently-seeded runs would wrongly look identical
  3. every previous_result: it reads was ITSELF served from cache this run
     - otherwise a change upstream leaves this step's cached output stale
"""
import logging

from .workflow import referenced_result_names

logger = logging.getLogger("dw")


def deep_equal(a, b):
    """Value equality across the JSON-ish types a resolved step definition holds."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


class StepCache:
    """Per-process cache of the last Result each step name produced."""

    def __init__(self):
        self._entries = {}  # step name -> {"step_data", "step_seed", "result"}

    def clear(self):
        self._entries.clear()

    def get(self, step_data, step_seed, hits_this_run):
        """Return the cached Result for this step if it's still valid, else None."""
        name = step_data["name"]
        entry = self._entries.get(name)
        if entry is None:
            return None
        if entry["step_seed"] != step_seed:
            return None
        if not deep_equal(entry["step_data"], step_data):
            return None

        upstream = referenced_result_names([step_data])
        if not all(self._is_hit(ref, hits_this_run) for ref in upstream):
            return None

        return entry["result"]

    def put(self, step_data, step_seed, result):
        self._entries[step_data["name"]] = {
            "step_data": step_data,
            "step_seed": step_seed,
            "result": result,
        }

    @staticmethod
    def _is_hit(ref, hits_this_run):
        # A reference resolves to a result whose name it equals or extends
        # with a property ('step.mask') - same rule Workflow.run's
        # release_unreferenced_results uses for the inverse question.
        return any(ref == n or ref.startswith(n + ".") for n in hits_this_run)


step_cache = StepCache()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_step_cache.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add dw/step_cache.py tests/test_step_cache.py
git commit -m "feat: add per-process step-output cache core"
```

---

### Task 5: Wire the step cache into `Workflow.run`

**Model:** sonnet (mechanical integration of an already-tested unit into an existing loop, but must preserve exact existing behavior for every non-hit path)

**Files:**
- Modify: `dw/workflow.py:337-385` (the step execution loop inside `Workflow.run`)
- Modify: `docs/REPL_COMMANDS.md` (document cache-clearing)
- Test: `tests/test_workflow_step_cache.py` (new — integration-level, exercises `Workflow.run` twice)

**Interfaces:**
- Consumes: `step_cache` singleton from Task 4 (`dw/step_cache.py`).

- [ ] **Step 1: Write the failing integration test**

This needs a minimal real (or fake) step action. Find how existing workflow tests build a runnable `Workflow` with a cheap/fake pipeline step — run `grep -rl "class Workflow" tests/ | head -3` and `grep -rln "def test_.*workflow.*run\|Workflow(" tests/*.py | head -5` to find the lightest-weight existing fixture pattern (likely a `sd15`-style tiny workflow or a mocked `create_step_action`). Match that pattern exactly rather than inventing a new fixture style. Using that pattern, write:

```python
# tests/test_workflow_step_cache.py
from unittest.mock import patch

from dw.step_cache import step_cache

# NOTE for implementer: replace `build_test_workflow_and_call_count_spy` below
# with whatever this repo's existing lightweight workflow-execution test
# fixture is (see grep in Step 1's instructions) - it must give you a
# Workflow instance plus a way to count how many times a step actually
# executed its pipeline/task body, so these two tests can assert 1 call
# on the first run and still 1 (not 2) on the second, unchanged run.


def test_second_run_with_unchanged_step_reuses_cached_result():
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()

    workflow.run({})
    workflow.run({})

    assert call_count() == 1


def test_second_run_with_changed_variable_recomputes_that_step():
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy()

    workflow.run({"prompt": "a cat"})
    workflow.run({"prompt": "a dog"})

    assert call_count() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_step_cache.py -v`
Expected: FAIL — `call_count() == 1` assertion fails with `2`, since nothing consults `step_cache` yet.

- [ ] **Step 3: Wire the cache into the step loop**

In `dw/workflow.py`, add the import near the top:

```python
from .step_cache import step_cache
```

Then in `Workflow.run`, initialize `hits_this_run = set()` immediately before the `for i, step_data in enumerate(steps):` loop (`dw/workflow.py:337-338`), and replace the loop body from `step = Step(step_data, step_seed, self.workflow_definition)` through the `saved_files = result.save(...)` / `self.manifest.append(...)` block (`dw/workflow.py:352-370`) with:

```python
                step = Step(step_data, step_seed, self.workflow_definition)

                is_cacheable = "workflow" not in step_data
                cached_result = (
                    step_cache.get(step_data, step_seed, hits_this_run)
                    if is_cacheable
                    else None
                )

                if cached_result is not None:
                    logger.info(f"Step '{step.name}' unchanged - reusing cached result")
                    result = cached_result
                    saved_files = result.saved_files
                    hits_this_run.add(step.name)
                    step_action = None
                else:
                    step_action = self.create_step_action(
                        step_data,
                        shared_components,
                        pipelines,
                        step_seed,
                        get_device(),
                    )
                    result = step.run(results, pipelines, step_action)
                    saved_files = result.save(
                        self.effective_output_dir, f"{workflow_id}-{step.name}.{i}"
                    )
                    if is_cacheable:
                        step_cache.put(step_data, step_seed, result)

                last_result = result
                results[step.name] = result
                self.manifest.append({"step": step.name, "files": saved_files})
                # A sub-workflow's saves land in the child's manifest - roll
                # them up so job history and the gallery see every file
                if isinstance(step_action, Workflow):
                    self.manifest.extend(getattr(step_action, "manifest", []))
```

Leave everything below this (the `run_context.emit("step_end", ...)` call and everything after, `dw/workflow.py:371-395`) unchanged — `saved_files` is defined on both branches so it still works.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_step_cache.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run the full workflow test suite for regressions**

Run: `pytest tests/ -k workflow -v`
Expected: PASS

- [ ] **Step 6: Extend the REPL's `memory clear` command to also clear the step cache**

Find the `memory clear` command handler: `grep -n "memory clear\|def.*memory_clear\|\"clear\"" dw/repl_commands.py`. Add `step_cache.clear()` (importing `from .step_cache import step_cache` at the top of that file if not already present) alongside whatever it currently clears, and update the command's help text / `docs/REPL_COMMANDS.md` entry for `memory clear` to mention it now also drops cached step results.

- [ ] **Step 7: Commit**

```bash
git add dw/workflow.py dw/repl_commands.py docs/REPL_COMMANDS.md tests/test_workflow_step_cache.py
git commit -m "feat: skip re-executing workflow steps whose inputs are unchanged"
```

---

### Task 6: Download endpoints for gallery outputs, workflows, and prompts

**Model:** sonnet (routing-order correctness matters — see the gotcha below — but the change itself is small and mechanical)

**Design rationale:** Comparing to ComfyUI/Mellon isn't relevant here (neither exposes a REST download-as-attachment endpoint the way dw's web UI would need — this is purely dw's own gap: `dw/server/app.py` never imports `FileResponse` and never sets `Content-Disposition`, so the only way to get a file out of the UI today is right-click-save on the `/outputs/{name}` static mount link, or the analogous open-file link for workflows/prompts, which just navigates the browser). Model manager entries are excluded from this task — a cached HF repo's weights are gigabytes on disk and "download to your browser" isn't the same operation as the model manager's existing delete; that page keeps only its existing delete button.

**Routing-order gotcha:** FastAPI/Starlette matches routes in registration order, and `@app.get("/api/workflows/{name:path}")` (`dw/server/app.py:732`, `get_workflow`) already greedily matches anything under `/api/workflows/...` including `/api/workflows/foo/download` (captured as `name="foo/download"`). The new `download_workflow`/`download_prompt` routes MUST be registered *before* the existing `get_workflow`/`get_prompt` routes or they will never be reached. Gallery has no competing plain `GET /api/gallery/{name:path}` route (only `/metadata` and `/thumbnail` suffixes exist), so `download_output`'s position relative to those is not order-sensitive, but it's placed next to them for readability.

**Files:**
- Modify: `dw/server/app.py` (import + 3 new routes)
- Test: `tests/test_server_downloads.py` (new)

**Interfaces:**
- Produces: `GET /api/gallery/{name:path}/download`, `GET /api/workflows/{name:path}/download`, `GET /api/prompts/{name:path}/download` — each returns the artifact with `Content-Disposition: attachment`.

- [ ] **Step 1: Write the failing tests**

First find how existing server tests build a test client — run `grep -rn "TestClient\|def client" tests/test_server*.py | head -10` and match that fixture exactly (output_dir/workflow_dir/prompt_dir setup, app construction).

```python
# tests/test_server_downloads.py
# Uses whatever TestClient/tmp-directory fixture tests/test_server*.py already
# establishes for app.state.workflow_dir / app.state.prompt_dir / manager.output_dir -
# match that setup exactly rather than reinventing it.
import json


def test_download_output_sets_content_disposition_attachment(client, output_dir):
    (output_dir / "run-step.0-0.0.png").write_bytes(b"fake-png-bytes")

    response = client.get("/api/gallery/run-step.0-0.0.png/download")

    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"
    assert "attachment" in response.headers["content-disposition"]
    assert "run-step.0-0.0.png" in response.headers["content-disposition"]


def test_download_workflow_sets_content_disposition_attachment(client, workflow_dir):
    (workflow_dir / "demo.json").write_text(json.dumps({"id": "demo", "steps": []}))

    response = client.get("/api/workflows/demo/download")

    assert response.status_code == 200
    assert json.loads(response.content) == {"id": "demo", "steps": []}
    assert "attachment" in response.headers["content-disposition"]

    # the plain (non-download) GET route must still work - regression guard
    # for the routing-order gotcha
    plain = client.get("/api/workflows/demo")
    assert plain.status_code == 200
    assert plain.json() == {"id": "demo", "steps": []}


def test_download_prompt_sets_content_disposition_attachment(client, prompt_dir):
    (prompt_dir / "greeting.json").write_text(json.dumps({"text": "hello"}))

    response = client.get("/api/prompts/greeting/download")

    assert response.status_code == 200
    assert json.loads(response.content) == {"text": "hello"}
    assert "attachment" in response.headers["content-disposition"]

    plain = client.get("/api/prompts/greeting")
    assert plain.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_server_downloads.py -v`
Expected: FAIL — `404 Not Found` for all three (routes don't exist yet)

- [ ] **Step 3: Add `FileResponse` to the imports**

In `dw/server/app.py:22`, change:

```python
from fastapi.responses import StreamingResponse, JSONResponse, Response
```

to:

```python
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
```

- [ ] **Step 4: Add the gallery download route**

In `dw/server/app.py`, immediately before `delete_output` (currently at line 1008), add:

```python
    @app.get("/api/gallery/{name:path}/download")
    def download_output(name: str):
        """Serve one output file as a forced download rather than an inline view."""
        path = _output_file(name)
        return FileResponse(path, filename=os.path.basename(name))
```

- [ ] **Step 5: Add the workflow download route — BEFORE `get_workflow`**

In `dw/server/app.py`, insert immediately after `delete_workflow` (currently ending at line 730) and *before* `@app.get("/api/workflows/{name:path}")` / `def get_workflow` (currently line 732):

```python
    @app.get("/api/workflows/{name:path}/download")
    def download_workflow(name: str):
        """Serve a workflow definition as a forced download."""
        path = resolve_workflow_name(app.state.workflow_dir, name)
        return FileResponse(
            path, filename=os.path.basename(path), media_type="application/json"
        )
```

- [ ] **Step 6: Add the prompt download route — BEFORE `get_prompt`**

In `dw/server/app.py`, insert immediately after `delete_prompt` (currently ending at line 797) and *before* `@app.get("/api/prompts/{name:path}")` / `def get_prompt` (currently line 799):

```python
    @app.get("/api/prompts/{name:path}/download")
    def download_prompt(name: str):
        """Serve a stored prompt as a forced download."""
        path = resolve_prompt_name(app.state.prompt_dir, name)
        return FileResponse(
            path, filename=os.path.basename(path), media_type="application/json"
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_server_downloads.py -v`
Expected: PASS (3 tests, including the two routing-order regression assertions)

- [ ] **Step 8: Run the full server test suite for regressions**

Run: `pytest tests/ -k server -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add dw/server/app.py tests/test_server_downloads.py
git commit -m "feat: add download endpoints for gallery outputs, workflows, and prompts"
```

---

### Task 7: Download buttons in the UI

**Model:** haiku (purely mechanical — repeat an existing icon-button pattern three times against known anchor points)

**Files:**
- Modify: `ui/src/lib/api.ts` (three URL-builder helpers, next to `deleteOutput`/`deleteWorkflow`/`deletePrompt`)
- Modify: `ui/src/lib/pages/GalleryPage.svelte` (button, near line 177-184)
- Modify: `ui/src/lib/pages/WorkflowPage.svelte` (button, near line 105-108)
- Modify: `ui/src/lib/pages/PromptEditorPage.svelte` (button, near line 505-507)

**Interfaces:**
- Consumes: the three download routes from Task 6.
- Produces: `api.outputDownloadUrl(name)`, `api.workflowDownloadUrl(name)`, `api.promptDownloadUrl(name)` — plain string builders (not fetch calls), for direct use in an `<a href>`.

- [ ] **Step 1: Read the exact surrounding code before editing**

Read `ui/src/lib/api.ts` in full once (it's small) to see the `request<T>`/`encodePath` helpers and the exact style of `deleteOutput`/`deleteWorkflow`/`deletePrompt`. Read `GalleryPage.svelte` lines 1-70 and 160-190, `WorkflowPage.svelte` lines 90-115, and `PromptEditorPage.svelte` lines 495-515, to see the exact existing delete-button markup and its icon import.

- [ ] **Step 2: Add URL-builder helpers to `api.ts`**

Immediately next to the existing `deleteOutput` method (found at `ui/src/lib/api.ts:150-154`), add a sibling. These are plain string builders — not `request<T>` calls — since a download needs a navigable URL for an `<a>`, not a fetch+JSON response:

```ts
  outputDownloadUrl: (name: string) => `/api/gallery/${encodePath(name)}/download`,
```

Next to `deleteWorkflow`:

```ts
  workflowDownloadUrl: (name: string) => `/api/workflows/${encodePath(name)}/download`,
```

Next to `deletePrompt`:

```ts
  promptDownloadUrl: (name: string) => `/api/prompts/${encodePath(name)}/download`,
```

- [ ] **Step 3: Add the download button to `GalleryPage.svelte`**

Import the `Download` icon alongside the existing `Trash2` import at the top of the file (match whatever import syntax `Trash2` already uses, e.g. `import { Trash2, Download } from 'lucide-svelte'`).

Immediately before the existing delete button (`Trash2` icon, `class="quiet icon danger"`, around line 177-184), add:

```svelte
<a
  class="quiet icon"
  href={api.outputDownloadUrl(selected.name)}
  download
  aria-label="Download"
  title="Download"
>
  <Download size={16} />
</a>
```

- [ ] **Step 4: Add the download button to `WorkflowPage.svelte`**

Same pattern: import `Download`, add immediately before the existing delete button (around line 105-108):

```svelte
<a
  class="quiet icon"
  href={api.workflowDownloadUrl(name)}
  download
  aria-label="Download"
  title="Download"
>
  <Download size={16} />
</a>
```

(Use whatever variable this component already has in scope for the workflow's name at that point — match the identifier the neighboring delete button's `api.deleteWorkflow(...)` call already uses.)

- [ ] **Step 5: Add the download button to `PromptEditorPage.svelte`**

Same pattern, immediately before the existing delete button (around line 505-507):

```svelte
<a
  class="quiet icon"
  href={api.promptDownloadUrl(name)}
  download
  aria-label="Download"
  title="Download"
>
  <Download size={16} />
</a>
```

(Again, match whatever identifier the neighboring `api.deletePrompt(...)` call already uses for the prompt's name.)

- [ ] **Step 6: Type-check and lint**

Run: `cd ui && npm run check && npm run lint`
Expected: PASS, no new errors

- [ ] **Step 7: Manually verify in the browser**

Run: `python -m dw.serve`, open the UI, navigate to the Gallery, Workflows, and Prompts pages, and confirm a download icon now sits next to each delete icon and clicking it saves the file (not just opens it inline).

- [ ] **Step 8: Commit**

```bash
git add ui/src/lib/api.ts ui/src/lib/pages/GalleryPage.svelte ui/src/lib/pages/WorkflowPage.svelte ui/src/lib/pages/PromptEditorPage.svelte
git commit -m "feat: add download button next to delete for gallery outputs, workflows, and prompts"
```

---

### Task 8: Documentation sweep

**Model:** haiku (pure documentation, no logic)

**Files:**
- Modify: `README.md` (Features list)
- Modify: `docs/SERVER.md`
- Modify: `docs/REPL_COMMANDS.md` (if not already fully covered by Task 5 Step 6)

**Interfaces:** none (docs only)

- [ ] **Step 1: Add two Features bullets to `README.md`**

In the `## Features` list (`README.md:16-32`), add, matching the existing bullet style:

```markdown
- **Cross-step memory management** — `residency: on_demand` components share a priority/LRU eviction pool across steps, so a memory-constrained run degrades gracefully instead of failing on OOM
- **Step-output caching** — in the REPL, a step whose resolved arguments and seed are unchanged since the last run reuses its cached result instead of re-executing
```

And update the existing "Reproducible by construction" bullet (`README.md:22`) to mention rerun collision-avoidance, since it's directly relevant to that claim:

```markdown
- **Reproducible by construction** — outputs embed their full workflow definition and seed; any image in the gallery reopens as the exact workflow that made it (see [Trust model](docs/SECURITY.md#trust-model) before reopening one someone else sent you). A rerun never overwrites a prior output — a name collision gets an incrementing suffix instead
```

- [ ] **Step 2: Document the download endpoints in `docs/SERVER.md`**

Find the section documenting the gallery/workflow/prompt HTTP API (`grep -n "DELETE /api\|## " docs/SERVER.md`) and add the three new `GET .../download` routes alongside their existing `DELETE` siblings, following that section's existing table/list format exactly.

- [ ] **Step 3: Confirm `docs/REPL_COMMANDS.md` covers step-cache clearing**

If Task 5 Step 6 already updated `memory clear`'s entry, just verify it reads clearly in context; otherwise add it now, matching that doc's existing command-entry format.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/SERVER.md docs/REPL_COMMANDS.md
git commit -m "docs: document memory manager, step cache, and download endpoints"
```

---

### Task 9: MCP tool to download an output and save it locally

**Model:** sonnet (small, but touches the MCP tool-surface contract — annotations and error handling must match the established pattern exactly)

**Design rationale:** the user asked specifically for an MCP-exposed download, so an agent (not just a browser) can pull a generated artifact onto local disk. This does **not** need Task 6's `/api/.../download` route — that route exists purely to make a *browser* force a Save-As dialog via `Content-Disposition`. `dw_mcp` already fetches raw bytes over HTTP and decides what to do with them itself (`get_output_image`/`get_output_text` in `dw_mcp/media.py` both call `client.get_bytes_if(api_path("outputs", name), ...)` against the plain `/outputs` static mount). This task adds a fourth media tool, `download_output`, that does the same fetch but writes the bytes to a local path instead of returning them inline — so it has no dependency on Task 6 and can be built in parallel with it. Scoped to gallery outputs only (not workflows/prompts) since those are already returned as inline JSON by `get_workflow`/`get_prompt` MCP tools where they exist, or can be added later the same way if needed — this task covers the artifact type the user actually asked about ("download an artifact").

**Files:**
- Modify: `dw_mcp/client.py` (add `get_bytes`, if not already present — confirm first, see Step 1)
- Modify: `dw_mcp/media.py` (new `download_output` handler)
- Modify: `dw_mcp/server.py` (register the tool)
- Modify: `docs/MCP.md` (document the new tool)
- Test: `tests/test_mcp_media.py` (extend)

**Interfaces:**
- Produces: `download_output(client, name, destination=None) -> dict` in `dw_mcp/media.py`, returning `{"name", "saved_to", "content_type", "bytes"}`.
- Produces the MCP tool `download_output(name: str, destination: str | None = None) -> dict`, registered with the `OVERWRITES` annotation (it writes to local disk and a repeat call to the same destination silently replaces the prior file — the same shape as `ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)` already defined in `dw_mcp/server.py`).

- [ ] **Step 1: Confirm `DwClient.get_bytes` already exists (it does — read `dw_mcp/client.py`'s `get_bytes` method) and does not need changes.** No code change for this step; it's a checkpoint before Step 2.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_mcp_media.py` (reusing the `serving()`/`png_bytes()` helpers already defined at the top of that file):

```python
import os

from dw_mcp.media import download_output


def test_download_output_writes_bytes_to_explicit_file_path(tmp_path):
    client = serving(png_bytes(64, 48), "image/png")
    destination = tmp_path / "saved.png"

    result = download_output(client, "run-step.0-0.0.png", destination=str(destination))

    assert destination.read_bytes() == png_bytes(64, 48)
    assert result == {
        "name": "run-step.0-0.0.png",
        "saved_to": str(destination),
        "content_type": "image/png",
        "bytes": len(png_bytes(64, 48)),
    }


def test_download_output_into_a_directory_uses_the_output_basename(tmp_path):
    client = serving(png_bytes(10, 10), "image/png")

    result = download_output(client, "sub/run-step.0-0.0.png", destination=str(tmp_path))

    saved = tmp_path / "run-step.0-0.0.png"
    assert saved.read_bytes() == png_bytes(10, 10)
    assert result["saved_to"] == str(saved)


def test_download_output_with_no_destination_saves_to_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = serving(png_bytes(10, 10), "image/png")

    result = download_output(client, "run-step.0-0.0.png")

    assert (tmp_path / "run-step.0-0.0.png").read_bytes() == png_bytes(10, 10)
    assert result["saved_to"] == str(tmp_path / "run-step.0-0.0.png")


def test_download_output_creates_missing_parent_directories(tmp_path):
    client = serving(png_bytes(10, 10), "image/png")
    destination = tmp_path / "renders" / "today" / "spoons.png"

    download_output(client, "spoons.png", destination=str(destination))

    assert destination.read_bytes() == png_bytes(10, 10)


def test_download_output_expands_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = serving(png_bytes(5, 5), "image/png")

    result = download_output(client, "spoons.png", destination="~/spoons.png")

    assert result["saved_to"] == str(tmp_path / "spoons.png")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_mcp_media.py -k download_output -v`
Expected: FAIL — `ImportError: cannot import name 'download_output' from 'dw_mcp.media'`

- [ ] **Step 4: Implement the handler**

Add to `dw_mcp/media.py` (near `delete_output`, at the end of the file), adding `import os` to the top of the file alongside the existing `base64`/`io`/`math` imports:

```python
def download_output(client, name, destination=None):
    """Fetch one output file and save it to local disk, for an agent that
    wants the artifact itself rather than a description of it.

    Unlike get_output_image/get_output_text, this accepts any content type
    and returns nothing to the conversation but a manifest of where the
    file landed - the point is a file on disk, not a payload in context.

    `destination` may be a full file path, a directory (the output's own
    basename is used inside it), or omitted (saved to the current working
    directory under its own basename). '~' expands to the user's home
    directory. Missing parent directories are created.
    """
    body, content_type = client.get_bytes(api_path("outputs", name))

    if destination is None:
        destination = os.path.basename(name)
    destination = os.path.expanduser(destination)
    if os.path.isdir(destination) or destination.endswith(os.sep):
        destination = os.path.join(destination, os.path.basename(name))

    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(destination, "wb") as file:
        file.write(body)

    return {
        "name": name,
        "saved_to": destination,
        "content_type": content_type,
        "bytes": len(body),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mcp_media.py -k download_output -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Register the MCP tool**

In `dw_mcp/server.py`, in the `# --------------------------------------------------------------- media` block, add next to the existing `get_output_image`/`get_output_text`/`delete_output` definitions:

```python
    def download_output(name: str, destination: str | None = None) -> dict:
        """Save one output file to local disk. Unlike get_output_image /
        get_output_text, this works for any file type and returns no
        content to the conversation - only where it was saved. `destination`
        may be a full path, a directory, or omitted to save into the
        current working directory under the output's own name."""
        return media.download_output(client, name, destination=destination)
```

And register it alongside the other media tools:

```python
    tool(get_output_image, READ_ONLY)
    tool(get_output_text, READ_ONLY)
    tool(download_output, OVERWRITES)
    tool(delete_output, DELETES)
```

- [ ] **Step 7: Run the full MCP test suite for regressions**

Run: `pytest tests/test_mcp_media.py tests/test_mcp_server.py -v` (adjust the second filename if `grep -rl "build_server" tests/` shows a different name for the server-registration test file)
Expected: PASS

- [ ] **Step 8: Document the new tool**

In `docs/MCP.md`, find the section listing the media tools (`grep -n "get_output_image\|delete_output" docs/MCP.md`) and add `download_output` to that list/table, following its existing format, noting it accepts any content type and writes to local disk rather than returning content.

- [ ] **Step 9: Commit**

```bash
git add dw_mcp/media.py dw_mcp/server.py tests/test_mcp_media.py docs/MCP.md
git commit -m "feat: add MCP tool to download an output artifact to local disk"
```

---

## Self-Review Notes

- **Spec coverage:** memory manager (Tasks 2-3), step cache (Tasks 4-5), download button+API everywhere a delete button exists on an artifact (Task 6-7, models excluded with stated rationale), output-overwrite research and fix (Task 1, grounded in the ComfyUI/Mellon comparison), MCP download-to-local-disk tool (Task 9) — all five requested items have a task.
- **Ordering:** Task 1 (output naming) is independent and can run first or in parallel with everything else. Tasks 2→3 and 4→5 are hard sequential pairs (core module before its call-site wiring). Task 6→7 is a hard sequential pair (backend before frontend can call it). Task 9 depends on none of the others (it uses the existing `/outputs` static mount, not Task 6's new route) and can run any time. Task 8 should run last since it references the REPL command and API routes the other tasks add.
- **Known scope boundaries, stated explicitly in-task rather than left implicit:** memory manager only covers `residency: on_demand`, not `offload`/`group_offload` (Task 2); step cache excludes sub-workflow steps (Task 4); download excludes the model manager (Task 6); MCP download is scoped to gallery outputs only, not workflows/prompts (Task 9).
