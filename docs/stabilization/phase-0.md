# Phase 0: Baseline and Correctness Fixes - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit a measured architecture baseline and fix the seven confirmed correctness bugs (B1-B5, B7, B8) without adding any feature, so later phases refactor from a correct, measured starting point.

**Architecture:** Every fix goes into the module that already owns the behaviour; no new engine module is created. One new developer script, `scripts/arch_metrics.py`, produces the numbers every later phase gate and the harness ratchet read. Each bug gets a test that fails on today's code first.

**Tech Stack:** Python 3.12, pytest (+ xdist), ruff, `filelock` (already installed via huggingface-hub; declared explicitly here), `pylint`'s `symilar` for duplicate detection (new dev dependency).

**Spec:** [ASSESSMENT.md](ASSESSMENT.md) and [ROADMAP.md](ROADMAP.md) in this directory.

## Global Constraints

- Hard freeze: no new features or functionality. Change only what a task names.
- No new module under `dw/` or `dw_mcp/`. New test files and `scripts/arch_metrics.py` are the only new files.
- Every new test fails on the code before its fix. Run it and see the failure before writing the fix.
- No count-pinning assertions (no "there are N templates"); assert behaviour. Baseline numbers live in `docs/stabilization/baseline.json`, never in a test.
- Comments explain why the code is the way it is. Do not narrate ticket history ("#415 used to...") in new comments.
- Tests: `venv/bin/python -m pytest <path> -q` for a file, `venv/bin/python -m pytest -q` (xdist is configured) for the suite. Lint: `venv/bin/python -m ruff check <files>` and `venv/bin/python -m ruff format <files>`.
- Commits: conventional prefix (`fix(...)`, `chore(...)`, `test(...)`), and end each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work on branch `stabilization/phase-0` in a worktree; never commit to `master`.

## Review Focus

1. A constraint-snapped value inside a `for_each` list entry (for example `shots[1].num_frames`) must be snapped at validate time too, not only top-level variables (Task 2).
2. `open_run` in the flat output layout must not create anything; flat has no run directories (Task 5).
3. A cache hit whose pipeline a *later cold* step borrows through `configuration.reused_components`, not only the pipeline-level key, must still load (Task 8).
4. Two threads pruning the same detail-cache entry must not raise `KeyError` (Task 4).
5. A sub-workflow step inside a `for_each` expansion must report the author's step index, not the expanded one (Task 3).

---

### Task 1: Architecture metrics script and baseline

**Files:**
- Create: `scripts/arch_metrics.py`
- Create: `tests/test_arch_metrics.py`
- Create: `docs/stabilization/baseline.json` (generated)
- Modify: `pyproject.toml` (`dev` extras: add `"pylint>=3.3"`)

**Interfaces:**
- Produces: `measure(root: pathlib.Path) -> dict[str, int | None]` with keys `modules`, `modules_over_1000_lines`, `functions_over_150_lines`, `prefix_literals`, `test_dw_patch_targets`, `claude_md_lines`, `duplicate_blocks` (None when pylint is not installed); `regressions(current: dict, baseline: dict) -> list[str]`; CLI `python scripts/arch_metrics.py [--write PATH | --check PATH]`. Every metric is lower-is-better. `--check` exits 1 and prints one line per regressed metric. Phase gates and the harness (stage B) call `--check`.

- [ ] **Step 1: Install the dev dependency**

Add `"pylint>=3.3",` to the `dev` list in `pyproject.toml`, then run `venv/bin/pip install "pylint>=3.3"`.

- [ ] **Step 2: Write the failing tests**

```python
"""scripts/arch_metrics.py - the numbers the stabilization gates and the
harness ratchet read."""

import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "arch_metrics.py"


def _load():
    spec = importlib.util.spec_from_file_location("arch_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _long_function(lines):
    body = "".join(f"    x{i} = {i}\n" for i in range(lines))
    return f"def f():\n{body}"


def test_a_long_function_and_a_long_module_are_counted(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/a.py": _long_function(151) + "\n" * 900, "dw/b.py": "x = 1\n"})
    )
    assert metrics["functions_over_150_lines"] == 1
    assert metrics["modules_over_1000_lines"] == 1
    assert metrics["modules"] == 2


def test_community_pipelines_are_not_counted(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/community_pipelines/p.py": _long_function(200)})
    )
    assert metrics["modules"] == 0
    assert metrics["functions_over_150_lines"] == 0


def test_a_reference_prefix_literal_is_counted_but_a_longer_string_is_not(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/a.py": 'P = "asset:"\nQ = "asset:cat.png is a file"\n'})
    )
    assert metrics["prefix_literals"] == 1


def test_patches_of_dw_paths_and_claude_md_lines_are_counted(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "tests/test_x.py": 'patch("dw.a.b")\npatch("os.path")\n',
                "CLAUDE.md": "one\ntwo\n",
                "ui/CLAUDE.md": "three\n",
            },
        )
    )
    assert metrics["test_dw_patch_targets"] == 1
    assert metrics["claude_md_lines"] == 3


def test_regressions_names_only_the_metrics_that_got_worse():
    module = _load()
    baseline = {"modules": 10, "prefix_literals": 5, "duplicate_blocks": None}
    current = {"modules": 11, "prefix_literals": 4, "duplicate_blocks": 3}
    assert module.regressions(current, baseline) == ["modules: 10 -> 11"]


def test_check_mode_exits_nonzero_on_a_regression(tmp_path):
    # A small tree, never the real repo: duplicate analysis of the whole
    # codebase would run on every suite run
    root = _tree(tmp_path / "repo", {"dw/a.py": "x = 1\n"})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"modules": 0}))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--check", str(baseline)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "modules: 0 ->" in result.stdout
```

- [ ] **Step 3: Run the tests and see them fail**

Run: `venv/bin/python -m pytest tests/test_arch_metrics.py -q`
Expected: every test errors with `FileNotFoundError` for `scripts/arch_metrics.py`.

- [ ] **Step 4: Write the script**

```python
#!/usr/bin/env python
"""Architecture metrics for the stabilization gates and the harness ratchet.

Every metric is lower-is-better, so a ratchet is one comparison: a metric
that rose against the committed baseline is a regression. Run with --write
to record a baseline and --check to compare against one.
"""

import argparse
import ast
import json
import pathlib
import re
import sys

REFERENCE_PREFIXES = frozenset(
    ("asset:", "output:", "prompt:", "variable:", "previous_result:",
     "constant:", "item:", "gather:")
)
# Modules allowed to spell a reference prefix. Empty until Phase 2 gives the
# prefixes one owner (dw/references.py).
PREFIX_OWNERS = frozenset()
EXCLUDED = ("community_pipelines", "node_modules", "venv", ".git")
PATCH_TARGET = re.compile(r"""patch\(\s*["']dw[._]""")


def _sources(root, *packages):
    for package in packages:
        for path in sorted((root / package).rglob("*.py")):
            if not any(part in EXCLUDED for part in path.parts):
                yield path


def _duplicate_blocks(paths):
    try:
        from pylint.checkers.symilar import Symilar
    except ImportError:
        return None
    similar = Symilar(min_lines=8, ignore_comments=True, ignore_docstrings=True,
                      ignore_imports=True, ignore_signatures=True)
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            similar.append_stream(str(path), stream)
    return len(similar._compute_sims())


def measure(root):
    root = pathlib.Path(root)
    engine = list(_sources(root, "dw", "dw_mcp"))
    metrics = {
        "modules": len(engine),
        "modules_over_1000_lines": 0,
        "functions_over_150_lines": 0,
        "prefix_literals": 0,
    }
    for path in engine:
        text = path.read_text(encoding="utf-8")
        if len(text.splitlines()) > 1000:
            metrics["modules_over_1000_lines"] += 1
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno - node.lineno > 150:
                    metrics["functions_over_150_lines"] += 1
            elif (
                isinstance(node, ast.Constant)
                and node.value in REFERENCE_PREFIXES
                and path.relative_to(root).as_posix() not in PREFIX_OWNERS
            ):
                metrics["prefix_literals"] += 1
    metrics["test_dw_patch_targets"] = sum(
        len(PATCH_TARGET.findall(path.read_text(encoding="utf-8")))
        for path in _sources(root, "tests")
    )
    metrics["claude_md_lines"] = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in root.rglob("CLAUDE.md")
        if not any(part in EXCLUDED for part in path.parts)
    )
    metrics["duplicate_blocks"] = _duplicate_blocks(engine)
    return metrics


def regressions(current, baseline):
    worse = []
    for name, before in baseline.items():
        now = current.get(name)
        if before is None or now is None:
            continue
        if now > before:
            worse.append(f"{name}: {before} -> {now}")
    return worse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=pathlib.Path(__file__).resolve().parent.parent)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", help="write the metrics to this JSON file")
    group.add_argument("--check", help="fail if any metric is worse than this JSON file")
    args = parser.parse_args(argv)
    current = measure(args.root)
    print(json.dumps(current, indent=2))
    if args.write:
        pathlib.Path(args.write).write_text(json.dumps(current, indent=2) + "\n")
    if args.check:
        worse = regressions(current, json.loads(pathlib.Path(args.check).read_text()))
        for line in worse:
            print(line)
        return 1 if worse else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

If `Symilar`'s constructor or `_compute_sims` differs in the installed pylint version, read `pylint/checkers/symilar.py` and adapt `_duplicate_blocks` only; keep the "None when unavailable" contract.

- [ ] **Step 5: Run the tests and see them pass**

Run: `venv/bin/python -m pytest tests/test_arch_metrics.py -q`
Expected: 6 passed.

- [ ] **Step 6: Record the baseline**

Run: `venv/bin/python scripts/arch_metrics.py --write docs/stabilization/baseline.json`
Expected: the JSON prints and is written. `duplicate_blocks` is an integer. Sanity-check against the assessment: roughly 133 modules, 10 over 1,000 lines, 19 functions over 150 lines. A large disagreement means the script counts something different; fix the script, not the numbers.

- [ ] **Step 7: Commit**

```bash
git add scripts/arch_metrics.py tests/test_arch_metrics.py docs/stabilization/baseline.json pyproject.toml
git commit -m "chore(stabilization): architecture metrics script and Phase 0 baseline"
```

---

### Task 2: B1 - validation and the realized record see constraint-snapped values

**Files:**
- Modify: `dw/variable_constraints.py` (add `snap_constraints`; `apply_constraints` calls it)
- Modify: `dw/workflow.py` (`expanded_definition`, ~line 519)
- Modify: `dw/realize.py` (`realize_workflow`, after `set_variables`, ~line 87)
- Test: `tests/test_variable_constraints.py`, `tests/test_realize.py`

**Interfaces:**
- Produces: `snap_constraints(definition: dict, variables: dict) -> list[tuple[str, str, object]]` in `dw/variable_constraints.py`. It rounds every `snap: "up"` value in place, both top-level variables and `for_each` list-entry fields (through the existing `entry_targets`). It returns `(message, variable_label, new_value)` per rounding. It never raises and never emits.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_variable_constraints.py`. It already imports `workflow_from_definition` and defines `H3` and `workflow_with`:

```python
class TestValidationSeesTheSnappedValue:
    """Validation must judge the value the run will use, not the one typed."""

    def _definition(self):
        definition = workflow_with({"num_frames": H3}, {"num_frames": 124})
        definition["steps"][0]["task"]["arguments"] = {"n": "variable:num_frames"}
        return definition

    def test_expanded_definition_carries_the_snapped_value(self, tmp_path):
        workflow = workflow_from_definition(self._definition(), str(tmp_path))
        expanded = workflow.expanded_definition({"num_frames": 130})
        assert expanded["steps"][0]["task"]["arguments"]["n"] == 141
        # Checks that fall back to the workflow's variables (vram_estimate)
        # must see the snapped value too
        assert expanded["variables"]["num_frames"] == 141

    def test_expanding_emits_no_warning_and_does_not_raise_on_a_refusal(self, tmp_path):
        from unittest.mock import patch

        workflow = workflow_from_definition(self._definition(), str(tmp_path))
        with patch("dw.events.emit_warning") as emitted:
            workflow.expanded_definition({"num_frames": 130})
            workflow.expanded_definition({"num_frames": 61})
        emitted.assert_not_called()

    def test_a_list_entry_value_is_snapped_at_validate_time(self, tmp_path):
        definition = workflow_with_shots(
            {"num_frames": H3}, [{"name": "a", "num_frames": 130}]
        )
        workflow = workflow_from_definition(definition, str(tmp_path))
        expanded = workflow.expanded_definition()
        assert 141 in _values_under(expanded["steps"], "num_frames")


def _values_under(node, key):
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                found.append(v)
            found.extend(_values_under(v, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_values_under(item, key))
    return found
```

Read `workflow_with_shots` (line ~242) before running the list-entry test. If the step it builds reads the field under a different argument name, adjust the assertion, not the helper.

Append to `tests/test_realize.py`:

```python
def test_the_realized_record_carries_the_snapped_value():
    from dw.realize import realize_workflow
    from tests.test_variable_constraints import H3, workflow_with

    realized, _ = realize_workflow(
        workflow_with({"num_frames": H3}, {"num_frames": 124}), {"num_frames": 130}, 1
    )
    assert realized["variables"]["num_frames"] == 141
```

Check `realize_workflow`'s signature in `dw/realize.py` first. If it needs keyword arguments (`base_dir`, `output_root`), pass `None`.

- [ ] **Step 2: Run the tests and see them fail**

Run: `venv/bin/python -m pytest tests/test_variable_constraints.py tests/test_realize.py -q -k "snapped or Snapped or snap"`
Expected: FAIL. `n` is 130 instead of 141, the realized value is 130, and the list-entry value is unsnapped.

- [ ] **Step 3: Implement `snap_constraints` and route `apply_constraints` through it**

In `dw/variable_constraints.py`, add above `apply_constraints`:

```python
def snap_constraints(definition, variables):
    """Round every value a `snap: "up"` rule rounds, in place, and say what
    changed. Pure apart from the rounding: never raises and never emits, so
    validation can see the value the run will use without the run's
    warnings or refusals (those stay in apply_constraints and
    constraint_errors)."""
    constraints = declared_constraints(definition)
    changes = []
    if not constraints or not isinstance(variables, dict):
        return changes
    for name in sorted(constraints):
        constraint = constraints[name]
        if not isinstance(constraint, dict) or name not in variables:
            continue
        notice = snap_notice(name, variables[name], constraint)
        if notice is not None:
            variables[name] = snapped(variables[name], constraint)
            changes.append((notice, name, variables[name]))
    for variable, index, name, entry in entry_targets(definition, variables):
        notice = snap_notice(name, entry[name], constraints[name])
        if notice is not None:
            entry[name] = snapped(entry[name], constraints[name])
            changes.append(
                (f"{variable}[{index}]: {notice}", f"{variable}[{index}].{name}", entry[name])
            )
    return changes
```

Rewrite `apply_constraints` so it calls `snap_constraints` first, emits one `emit_warning(message, kind="value_snapped", variable=label, value=value)` per change, and then runs the existing refusal loops (top-level and `entry_targets`) with the snapping branches removed. The refusal messages and the `ValueError` wording stay byte-for-byte the same.

- [ ] **Step 4: Call it from validation and from the realized record**

In `dw/workflow.py` `expanded_definition`, immediately after `variables = resolve_variable_values(variables)`:

```python
            # Validate what the run will use: a snap-up rule rounds before
            # substitution there, so it must here too
            snap_constraints(definition, variables)
```

Import `snap_constraints` from `.variable_constraints`, next to the existing imports from that module. `variables` was reassigned by `resolve_variable_values`, so check whether `definition["variables"]` is the same object. If the `expanded["variables"]` assertion fails, assign `definition["variables"] = variables` after snapping.

In `dw/realize.py` `realize_workflow`, immediately after `set_variables(arguments or {}, variables)`, add `snap_constraints(realized, variables)` with the same import.

- [ ] **Step 5: Run the tests and see them pass, then the neighbours**

Run: `venv/bin/python -m pytest tests/test_variable_constraints.py tests/test_realize.py tests/test_workflow.py tests/test_plan.py -q`
Expected: all pass. The existing `apply_constraints` warning and refusal tests must pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add dw/variable_constraints.py dw/workflow.py dw/realize.py tests/test_variable_constraints.py tests/test_realize.py
git commit -m "fix(validate): validation and the realized record see constraint-snapped values"
```

---

### Task 3: B3 - sub-workflow warnings are strings at the author's path, checked against the caller's arguments

**Files:**
- Modify: `dw/workflow.py` (`sub_workflow_warnings`, ~lines 609-647)
- Modify: `dw/server/app.py` (the validate route's call, ~line 1937)
- Test: `tests/test_workflow.py` (tests near lines 1295-1440 using `_tree` and `_parent`)

**Interfaces:**
- Produces: `Workflow.sub_workflow_warnings(self, arguments=None) -> list[str]`, each entry `"steps[<author index>].workflow.arguments.<name>: <message>"`, matching every other warnings source.

- [ ] **Step 1: Write the failing tests**

Read `_tree` and `_parent` in `tests/test_workflow.py` (~1295-1310) and the existing sub-workflow warning test (~1434). Replace that existing test's dict-shape assertion with the string shape, then add:

```python
def test_sub_workflow_warnings_are_strings_at_the_authors_step(tmp_path):
    parent = _parent(tmp_path, arguments={"promt": "x"})
    parent.workflow_definition["steps"].insert(
        0,
        {
            "name": "fan",
            "for_each": [{"name": "a"}, {"name": "b"}],
            "task": {"command": "no_op", "arguments": {}},
        },
    )
    warnings = parent.sub_workflow_warnings()
    assert all(isinstance(w, str) for w in warnings)
    assert warnings and warnings[0].startswith("steps[1].workflow.arguments.promt: ")
```

If `_parent` does not take `arguments=`, build the definition the way the neighbouring test does. The requirement is a sub-workflow step at author index 1 whose expanded index is 2.

- [ ] **Step 2: Run them and see them fail**

Run: `venv/bin/python -m pytest tests/test_workflow.py -q -k sub_workflow_warnings`
Expected: FAIL. The warnings are dicts, and the path reads `steps[2]`.

- [ ] **Step 3: Implement**

Change the signature to `def sub_workflow_warnings(self, arguments=None):`. Replace the expansion with:

```python
        source_indices = []
        try:
            expanded = self.expanded_definition(arguments, source_indices)
        except Exception:
            return warnings
```

Inside the loop, compute `source = source_indices[index] if index < len(source_indices) else index`, and append a string:

```python
                warnings.append(
                    f"steps[{source}].workflow.arguments.{name}: "
                    f"'{reference['path']}' declares no variable '{name}' - the "
                    "value is dropped. Declared: "
                    + (", ".join(sorted(declared)) or "<none>")
                )
```

Grep for other callers (`grep -rn "sub_workflow_warnings" dw dw_mcp tests`). In `dw/server/app.py`, change `candidate.sub_workflow_warnings()` to `candidate.sub_workflow_warnings(request.arguments)`.

- [ ] **Step 4: Run and pass, plus the server validate tests**

Run: `venv/bin/python -m pytest tests/test_workflow.py tests/test_server.py -q -k "sub_workflow or validate"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow.py dw/server/app.py tests/test_workflow.py
git commit -m "fix(validate): sub-workflow warnings are strings at the author's step and use the caller's arguments"
```

---

### Task 4: B4 - detail-cache pruning is safe under concurrent requests

**Files:**
- Modify: `dw/server/app.py` (`_prune_missing` ~238-249; delete `_prune_detail_cache` ~227-235, which has no callers)
- Test: `tests/test_server.py`

- [ ] **Step 1: Confirm `_prune_detail_cache` is dead**

Run: `grep -rn "_prune_detail_cache" dw tests`
Expected: only its definition. If there is a caller, keep the function and apply the Step 4 fix to it too.

- [ ] **Step 2: Write the failing tests**

```python
class TestDetailCachePruning:
    """Request threads share the module-level detail caches."""

    def test_an_insert_during_the_scan_does_not_raise(self):
        from unittest.mock import patch

        from dw.server import app as app_module

        cache = {"/gone/a.json": 1, "/gone/b.json": 2}

        def exists_while_another_thread_inserts(path):
            cache[f"/new/{len(cache)}.json"] = 0
            return False

        with patch.object(app_module.os.path, "exists", exists_while_another_thread_inserts):
            app_module._prune_missing(cache)
        assert "/gone/a.json" not in cache and "/gone/b.json" not in cache

    def test_an_entry_another_thread_already_pruned_is_not_an_error(self):
        from unittest.mock import patch

        from dw.server import app as app_module

        cache = {"/gone/a.json": 1, "/gone/b.json": 2}

        def exists_while_another_thread_prunes(path):
            cache.pop("/gone/b.json", None)
            return False

        with patch.object(app_module.os.path, "exists", exists_while_another_thread_prunes):
            app_module._prune_missing(cache)
        assert cache == {}
```

- [ ] **Step 3: Run them and see them fail**

Run: `venv/bin/python -m pytest tests/test_server.py -q -k DetailCachePruning`
Expected: FAIL with `RuntimeError: dictionary changed size during iteration`, and `KeyError` in the second test.

- [ ] **Step 4: Implement**

```python
def _prune_missing(cache):
    """Forget cached files that are gone from disk.

    `list(cache)` copies the keys in one step, so a request thread inserting
    meanwhile cannot break the scan, and `pop` tolerates an entry another
    thread already pruned."""
    for stale in [path for path in list(cache) if not os.path.exists(path)]:
        cache.pop(stale, None)
```

Keep the existing explanation of why existence, not the listing, is the test. Merge it into this docstring. Delete `_prune_detail_cache`.

- [ ] **Step 5: Run and pass**

Run: `venv/bin/python -m pytest tests/test_server.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "fix(server): detail-cache pruning tolerates concurrent requests"
```

---

### Task 5: B5 - opening a run claims its directory and its version atomically

**Files:**
- Modify: `dw/runs.py` (add `open_run`; delete `run_directory` and `assign_run_version`, whose only caller is `Workflow.run`, so there is one way to open a run)
- Modify: `dw/workflow.py` (`run`, ~lines 1347-1365)
- Modify: `pyproject.toml` (declare `"filelock>=3.12"` in `dependencies`; it is already installed through huggingface-hub)
- Test: `tests/test_runs.py` (`TestRunVersions`, helper `_run` at ~857)

**Interfaces:**
- Produces: `open_run(output_dir: str, file_spec: str | None, workflow_id: str, run_id: str) -> tuple[str, int]` in `dw/runs.py`, returning `(run_dir, version)`. Under a `filelock.FileLock` at `<output_dir>/<identity>/.run.lock`, it creates the identity directory, claims the run directory with `os.mkdir` (appending `-2`, `-3`, ... on `FileExistsError`), computes the version over sibling manifests excluding its own new directory, and writes a stub `manifest.json` `{"run_id": <final dir name>, "version": N, "status": "running"}`. `Workflow.run` calls it only in the run-directory layout. The flat layout calls nothing.

- [ ] **Step 1: Write the failing tests**

Add to `TestRunVersions` in `tests/test_runs.py`:

```python
    def test_two_runs_with_the_same_id_get_distinct_directories_and_versions(self, tmp_path):
        from dw.runs import open_run

        first = open_run(str(tmp_path), None, "wf", "20260928T120000Z-aaaaaaaa")
        second = open_run(str(tmp_path), None, "wf", "20260928T120000Z-aaaaaaaa")
        assert first[0] != second[0]
        assert (first[1], second[1]) == (1, 2)
        assert os.path.isdir(first[0]) and os.path.isdir(second[0])

    def test_concurrent_opens_never_share_a_version(self, tmp_path):
        import threading

        from dw.runs import open_run

        barrier = threading.Barrier(8)
        results = []

        def opener(i):
            barrier.wait()
            results.append(open_run(str(tmp_path), None, "wf", f"20260928T120000Z-{i:08x}"))

        threads = [threading.Thread(target=opener, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(version for _, version in results) == list(range(1, 9))

    def test_the_first_run_is_version_one_even_though_its_own_directory_exists(self, tmp_path):
        from dw.runs import open_run

        _, version = open_run(str(tmp_path), None, "wf", "20260928T120000Z-aaaaaaaa")
        assert version == 1
```

The last test matters: the run's own freshly claimed, manifest-less directory must not be ranked as an older sibling.

- [ ] **Step 2: Run them and see them fail**

Run: `venv/bin/python -m pytest tests/test_runs.py -q -k "open or concurrent or distinct"`
Expected: FAIL with `ImportError: cannot import name 'open_run'`.

- [ ] **Step 3: Implement `open_run`**

Read `record_run_versions`, `_run_ids` and `workflow_identity` in `dw/runs.py` first. The version must be computed over sibling runs other than the one being opened. Pass the claimed directory name as an exclusion to whichever function ranks, adding an `exclude=None` parameter to it rather than duplicating the ranking.

```python
def open_run(output_dir, file_spec, workflow_id, run_id):
    """Claim this execution's directory and version in one step.

    Two processes opening runs of one workflow at once - a CLI run beside a
    server job - used to be able to take the same directory or the same
    version, because both were computed and only later written. The
    directory is claimed with an exclusive mkdir and the version is assigned
    and written under a lock held across both."""
    identity = workflow_identity(file_spec, workflow_id)
    identity_dir = os.path.join(output_dir, identity)
    os.makedirs(identity_dir, exist_ok=True)
    with FileLock(os.path.join(identity_dir, ".run.lock")):
        candidate, counter = os.path.join(identity_dir, run_id), 1
        while True:
            try:
                os.mkdir(candidate)
                break
            except FileExistsError:
                counter += 1
                candidate = os.path.join(identity_dir, f"{run_id}-{counter}")
        name = os.path.basename(candidate)
        versions = record_run_versions(identity_dir, exclude=name)
        version = max(versions.values(), default=0) + 1
        write_manifest(candidate, {"run_id": name, "version": version, "status": "running"})
    return candidate, version
```

Adjust the names (`record_run_versions`, `write_manifest`, the manifest keys) to what `dw/runs.py` actually uses, and keep the stub manifest to keys the full manifest also writes. `_run_ids` already filters on `is_run_id(name)` and `isdir`, so `.run.lock` is not a run. Add a test that a gallery listing (`GET /api/gallery` via the `tests/test_server.py` client) of a workflow with a `.run.lock` present shows no entry for it.

Then delete `run_directory` and `assign_run_version`. Port each `assign_run_version` test in `TestRunVersions` to `open_run`. Every such test builds existing sibling runs with `_run` and asserts the next number, so the port is to call `open_run(str(tmp_path), None, <workflow id>, <a fresh run id>)` and assert `[1]` of its result. Use the workflow id that makes `workflow_identity` produce the same identity the test used, such as `"ltx2/Gyre"`. Check `workflow_identity(None, id)` for how an id maps to an identity directory. If it cannot reproduce a nested identity, pass a `file_spec` under a `workflows/ltx2/` tree instead. The ranking semantics these tests pin must not change.

- [ ] **Step 4: Use it in `Workflow.run`**

In `dw/workflow.py`, replace the `run_directory(...)` + `assign_run_version(...)` pair (~1352-1363) with:

```python
                    self._run_dir, self._run_version = open_run(
                        self.output_dir, self.file_spec, workflow_id, run_id
                    )
```

Keep the debug log line. `run_id` must now be the claimed directory's name, which can have a `-N` suffix, so set `run_id = os.path.basename(self._run_dir)` right after, because the manifest and the `run_start` event carry it.

- [ ] **Step 5: Run and pass, plus everything that reads runs**

Run: `venv/bin/python -m pytest tests/test_runs.py tests/test_workflow.py tests/test_server.py tests/test_realize.py -q`
Expected: all pass. A test that asserted the directory did not exist before the first step may need its expectation updated, because the directory now exists from the moment the run opens. Update it only if that is all it asserted.

- [ ] **Step 6: Commit**

```bash
git add dw/runs.py dw/workflow.py pyproject.toml tests/test_runs.py
git commit -m "fix(runs): claim a run's directory and version atomically"
```

---

### Task 6: B8 - validation resolves assets against the job's workspace

**Files:**
- Modify: `dw/worker.py` (`_handle_execute`, ~lines 225-320)
- Test: `tests/test_worker_execute.py` (helpers `_make_worker`, `StubWorkflow`, `_execute`)

- [ ] **Step 1: Write the failing test**

```python
class AssetRecordingWorkflow(StubWorkflow):
    def validate(self, arguments=None):
        from dw.assets import get_asset_dir

        self.asset_dir_at_validate = get_asset_dir()


def test_validation_sees_the_jobs_asset_directory(tmp_path):
    worker = _make_worker()
    workflow = AssetRecordingWorkflow()
    _execute(
        worker,
        workflow,
        {
            "workflow_path": "x.json",
            "arguments": {},
            "output_dir": str(tmp_path),
            "asset_dir": str(tmp_path / "assets"),
        },
    )
    assert workflow.asset_dir_at_validate == str(tmp_path / "assets")
```

Check what `get_asset_dir()` returns: a path or a string, and whether it takes arguments. Match the assertion to it.

- [ ] **Step 2: Run and see it fail**

Run: `venv/bin/python -m pytest tests/test_worker_execute.py -q -k asset_directory`
Expected: FAIL. The recorded directory is the default discovery result, not the job's.

- [ ] **Step 3: Implement**

Move the `asset_token = activate_asset_dir(...)` block so it runs immediately after `set_log_level(log_level)` and before `_load_workflow`. Move its `deactivate_asset_dir(asset_token)` into the outermost `finally` of `_handle_execute`, so it deactivates whatever path the execute took. Initialise `asset_token = None` beside `workflow = None`. Leave the `try/finally` around `workflow.run` with only the step-key recording and the watcher stop.

- [ ] **Step 4: Check the two server-side validation sites**

The validate route and the pre-queue check in `dw/server/app.py` (search for `validation_errors(` there) run in the server process. Check whether they activate the workspace's asset directory before validating. If they do not, write a failing test in `tests/test_server.py`: a workspace with an asset only in its own library, and a workflow whose `dissolve_videos` input is `asset:<name>`. Fix it the same way, by activating around the call. If they already do, note that in the commit message.

- [ ] **Step 5: Run and pass**

Run: `venv/bin/python -m pytest tests/test_worker_execute.py tests/test_worker.py tests/test_server.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dw/worker.py tests/test_worker_execute.py
git commit -m "fix(worker): validate a job against its own workspace's assets"
```

---

### Task 7: B7 - the step-cache key includes the pipelines a step borrows

**Files:**
- Modify: `dw/workflow.py` (`_cache_lookup`, ~line 1120-1165)
- Test: `tests/test_workflow_step_cache.py` (helpers `_pipeline_reference_workflow_def` ~153, `_shared_components_workflow_def` ~252, `FakeResult`, `_mock_pipeline_load`)

**Interfaces:**
- Consumes: `pipeline_cache_key(pipeline_definition)` from `dw/step_cache.py`.
- Produces: `component_names(pipeline_definition: dict, key: str) -> list[str]` as a module-level function in `dw/pipeline_processors/pipeline.py`. It is the pure half of `Pipeline.component_names`, reading `key` at the pipeline level and under `configuration`. The method delegates to it, so the rule stays in one place.
- Produces: `borrowed_pipeline_keys(steps: list, index: int) -> dict[str, str]` in `dw/step_cache.py` (the module that owns cache keys). It maps each earlier step this step borrows from (its `pipeline_reference.reference_name`, and for each name in its `reused_components`, at the pipeline level or under `configuration`, the latest earlier step whose `shared_components` lists that name) to that step's `pipeline_cache_key`. It is resolved statically from `steps[:index]`.

- [ ] **Step 1: Write the failing tests**

Read `_pipeline_reference_workflow_def`, `_shared_components_workflow_def` and `build_pipeline_reference_workflow_and_call_count_spy` (~181). The helper below follows the spy's pattern (patch `Step.run` and `Pipeline.load`, then stop the patchers) but records step names instead of counting:

```python
def _run_twice_recording_order(tmp_path, definition, argument_sets):
    step_cache.clear()
    workflow = Workflow(definition, str(tmp_path), "test.json")
    order = []

    def fake_step_run(self, previous_results, previous_pipelines, step_action):
        order.append(self.name)
        return FakeResult()

    patchers = [
        patch.object(Step, "run", fake_step_run),
        patch.object(Pipeline, "load", _mock_pipeline_load),
    ]
    for p in patchers:
        p.start()
    try:
        for arguments in argument_sets:
            workflow.run(arguments)
    finally:
        for p in patchers:
            p.stop()
    return order
```

Then add:

```python
def test_a_step_borrowing_a_pipeline_misses_when_the_source_model_changes(tmp_path):
    definition = _pipeline_reference_workflow_def()
    definition["variables"] = {"model_a": "m1"}
    definition["steps"][0]["pipeline"]["from_pretrained_arguments"]["model_name"] = "variable:model_a"
    order = _run_twice_recording_order(tmp_path, definition, [{"model_a": "m1"}, {"model_a": "m2"}])
    assert order == ["A", "B", "A", "B"]


def test_a_step_reusing_components_misses_when_the_sharing_model_changes(tmp_path):
    definition = _shared_components_workflow_def()
    definition["variables"] = {"model_a": "m1"}
    definition["steps"][0]["pipeline"]["from_pretrained_arguments"]["model_name"] = "variable:model_a"
    order = _run_twice_recording_order(tmp_path, definition, [{"model_a": "m1"}, {"model_a": "m2"}])
    assert order[len(order) // 2:] == order[: len(order) // 2]
```

Write `_run_twice_recording_order` from the pattern the file already uses: `_mock_pipeline_load`, a `FakeResult`, and a spy on step execution. Adapt the step names and the `model_name` path to what the helper definitions actually contain. Read them, then write the assertions against their real step names.

- [ ] **Step 2: Run and see them fail**

Run: `venv/bin/python -m pytest tests/test_workflow_step_cache.py -q -k "borrowing or reusing_components_misses"`
Expected: FAIL. The second run skips B as a stale hit.

- [ ] **Step 3: Implement**

First, in `dw/pipeline_processors/pipeline.py`, lift the body of `Pipeline.component_names` into a module-level `component_names(pipeline_definition, key)` and make the method `return component_names(self.pipeline_definition, key)`. Then add `borrowed_pipeline_keys(steps, index)` to `dw/step_cache.py`. It imports `component_names` inside the function (`from .pipeline_processors.pipeline import component_names`), which keeps `step_cache` free of a module-level dependency on the pipeline module. Confirm there is no import cycle with `venv/bin/python -c "import dw.step_cache"`. In `Workflow._cache_lookup`, after the step snapshot is taken, add:

```python
            borrowed = borrowed_pipeline_keys(steps, index)
            if borrowed:
                step_data_snapshot["__borrowed_pipelines__"] = borrowed
```

Use the variable names `_cache_lookup` already has for the steps list and the index.

- [ ] **Step 4: Run and pass, plus the whole cache suite**

Run: `venv/bin/python -m pytest tests/test_workflow_step_cache.py tests/test_step_cache.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dw/step_cache.py dw/workflow.py tests/test_workflow_step_cache.py
git commit -m "fix(step-cache): a step borrowing a pipeline misses when its source changes"
```

---

### Task 8: B2 - a cache hit does not load a pipeline nobody needs (superseded by Tasks 11 and 12)

**Files:**
- Modify: `dw/workflow.py` (the run loop ~1500-1530 and `create_step_action` ~1873-2010)
- Test: `tests/test_workflow_step_cache.py` (helper `build_test_workflow_and_call_count_spy` ~66)

**Interfaces:**
- Consumes: `borrowed_pipeline_keys` from Task 7. A later step borrows this one exactly when this step's name is a key in some later step's `borrowed_pipeline_keys(steps, j)`.
- Produces: `create_step_action(..., cache_hit: bool = False)`. On a hit, when the step's pipeline is not resident and no later step borrows it, it records the step's pipeline key and touches it but does not load, and returns `None`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_fully_cached_rerun_loads_no_pipeline(tmp_path):
    workflow, call_count = build_test_workflow_and_call_count_spy(tmp_path)
    with patch("dw.pipeline_processors.pipeline.Pipeline.load") as load:
        workflow.run({})
        loads_first_run = load.call_count
        workflow.run({})
    assert call_count() == 1
    assert load.call_count == loads_first_run
```

Read `build_test_workflow_and_call_count_spy` first. Run twice without sharing a `previous_pipelines` dict, which is the released-pipeline situation. Adapt the patch target to wherever `Pipeline.load` is looked up from.

- [ ] **Step 2: Run and see it fail**

Run: `venv/bin/python -m pytest tests/test_workflow_step_cache.py -q -k fully_cached_rerun`
Expected: FAIL. The second run loads once more.

- [ ] **Step 3: Implement**

In the run loop, compute `borrowed_later = any(step.name in borrowed_pipeline_keys(steps, j) for j in range(i + 1, len(steps)))` and pass `cache_hit=cached_result is not None and not borrowed_later` into `create_step_action`. In the pipeline branch of `create_step_action`, after `_step_pipeline_key(...)` and `touch_pipeline(...)` run, add:

```python
            if cache_hit and cache_key not in previous_pipelines:
                # A hit needs the key recorded (release_pipeline and
                # pipeline_reference address it by name), not the weights:
                # nothing this run does will call the pipeline
                return None
```

Check every use of `step_action` after the call in the run loop for `None` safety. The release path already pops by key, and `isinstance(step_action, Workflow)` is False for `None`. Update the stale comment above the call ("A hit skips the step's work, never its bookkeeping") so it says the bookkeeping no longer includes loading.

- [ ] **Step 4: Run and pass, plus the guard tests**

Run: `venv/bin/python -m pytest tests/test_workflow_step_cache.py tests/test_workflow.py tests/test_worker.py -q`
Expected: all pass. In particular, `test_pipeline_reference_still_resolves_when_referenced_step_is_cache_eligible` and `test_cache_hit_republishes_shared_components_for_a_later_cold_step` must stay green. They are the cases `borrowed_later` exists for.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow.py tests/test_workflow_step_cache.py
git commit -m "fix(step-cache): a cache hit records its pipeline without loading it"
```

---

### Task 9: B6 - record the MCP mount's shared-session limit as a decision

**Files:**
- Modify: `docs/stabilization/ASSESSMENT.md` (the B6 row)

The mount is stateless HTTP by design (`dw/server/mcp_mount.py`, "single-user"; `Context.session_id` is `None` there). A per-session workspace would need `stateless_http=False`, which gives up the property that a restart strands no sessions. This is a design decision, not a Phase 0 bug fix.

- [ ] **Step 1:** Change the B6 row's evidence to "confirmed; by design (single-user mount, stateless HTTP). Revisit in Phase 3 with the router split if multi-agent use of one server becomes a requirement."
- [ ] **Step 2:** Commit: `git commit -am "docs(stabilization): B6 is a documented single-user limit, not a Phase 0 fix"`

---

### Task 10: Phase 0 gate

- [ ] **Step 0 (before any merge):** B2 "before" timing on lem, which still runs today's `develop`. Rerun a seeded catalog template that sets `release_pipeline: true` twice with the same arguments and record the second run's wall-clock time.
- [ ] **Step 1:** Run the full suite: `venv/bin/python -m pytest -q`. Expected: no failure that also passes on `develop`. Compare against a `develop` run if anything fails.
- [ ] **Step 2:** Run `venv/bin/python -m ruff check dw dw_mcp tests scripts` and `venv/bin/python -m ruff format --check dw dw_mcp tests scripts`.
- [ ] **Step 3:** Run `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. Expected: exit 0, since Phase 0 must not make any metric worse.
- [ ] **Step 4:** Whole-branch review (`superpowers:requesting-code-review`), then merge `stabilization/phase-0` into `develop` and push.
- [ ] **Step 5:** Deploy to lem (`scripts/deploy.sh`) and run the smoke checks from `docs/RELEASING.md`.
- [ ] **Step 6:** B2 "after" timing on lem: the same template and arguments as Step 0. Record both wall-clock times in the ROADMAP status row.
- [ ] **Step 7:** Refresh `ASSESSMENT.md` bug statuses, set Phase 0 to `done` in `ROADMAP.md`, and move `hot-zone.txt` to the Phase 1 file list. Commit and push. Tag the merge commit `stabilization-gate-0` and push the tag. Gate reports are computed from the tag once Phase 1 adds `scripts/arch_report.py`.

---

### Task 11: B2 completion - defer a cached step's load until a step that runs borrows it

Added at the gate, 2026-09-28. On lem, a fully cached rerun of `templates/ltx2/two-stage` took 78.8 s before Task 8 and 84.4 s after it: `base` still loaded the whole LTX-2.5 pipeline (about 72 s) because `upscale` reuses its `vae`, and Task 8's rule ("a later step borrows it") does not ask whether that later step will run. `upscale` was itself a cache hit.

**Rule:** a cache-hit step never loads its pipeline when it is not resident. It records the pipeline as *deferred* (its key recorded and touched exactly as Task 8 does), and a later step that actually executes (a cache miss) loads every deferred pipeline it borrows immediately before it needs it. The look-ahead check Task 8 added (`hit_needs_no_pipeline` / the `borrowed_later` computation over later steps) is deleted, not refined: no prediction of later hits is needed.

**Files:**
- Modify: `dw/workflow.py` (the run loop, `create_step_action`)
- Test: `tests/test_workflow_step_cache.py`

**Interfaces:**
- Consumes: `borrowed_pipeline_keys(steps, index)` from `dw/step_cache.py` (Task 7): for step `index`, the earlier step names it borrows from.
- Produces: per-run deferred state, e.g. `self._deferred_pipelines: dict[str, <what create_step_action needs to load that step's pipeline later>]`, reset at the start of each run. `create_step_action(..., cache_hit=True)` defers instead of loading. Before a cold step's own `create_step_action`, the run loop loads each deferred pipeline named in `borrowed_pipeline_keys(steps, i)`, in step order, through the same code path a cold load uses (so `shared_components` are published and `pipeline_reference` resolves). A `release_pipeline` on a deferred step drops it from the deferred map and emits nothing.

- [ ] **Step 1: Failing tests** (tests/test_workflow_step_cache.py; use `_mock_pipeline_load` with a load counter, patch.object only):
    1. A two-step workflow where step B reuses A's shared component and both are cache hits on the second run: the second run loads nothing. This fails today: A loads.
    2. The same workflow where, on the second run, B misses (change B's arguments through a variable) but A hits: A's pipeline is loaded once, before B, and B runs with A's shared component available. This passes today and must keep passing.
    3. The same for `pipeline_reference`: A hits and B (`pipeline_reference: A`) misses. A loads once, B resolves the reference.
    4. The existing guard tests (`test_pipeline_reference_still_resolves_when_referenced_step_is_cache_eligible`, `test_cache_hit_republishes_shared_components_for_a_later_cold_step`, `test_cache_hit_still_touches_the_steps_pipeline`, `test_release_pipeline_on_a_hit_that_loaded_nothing_emits_no_release`, `test_a_fully_cached_rerun_loads_no_pipeline`) stay green, unedited.
- [ ] **Step 2:** Run the new tests and see test 1 fail with A's load counted.
- [ ] **Step 3:** Implement the deferral and the lazy load. Delete the look-ahead.
- [ ] **Step 4:** Run tests/test_workflow_step_cache.py, tests/test_pipeline_caching.py, tests/test_workflow.py, tests/test_worker.py, tests/test_events.py, then the full suite. `scripts/arch_metrics.py --check docs/stabilization/baseline.json` must exit 0.
- [ ] **Step 5:** Commit: `fix(step-cache): a cached step defers its load until a step that runs borrows it`.

The gate (Task 10, Step 6) re-times `templates/ltx2/two-stage` after this lands. Expected: the cached rerun no longer spends about 72 s loading `base`.

---

### Task 12: B7/B2 follow-up - borrowed pipeline keys come from the definition as written, not as loaded

Added at the gate, 2026-09-28. After Task 11 was deployed, a cached rerun of `templates/ltx2/two-stage` on lem took 172 s, as long as a cold run. `upscale` missed the step cache, so `refine` missed too and regenerated.

Cause: `Pipeline.load` edits its step's definition in place (for example `from_pretrained_arguments.pop("model_name")`). `borrowed_pipeline_keys` (Task 7) hashes the *earlier* step's pipeline definition when the later step is looked up:
- In the cold run, `base` has already loaded by then, so its definition is the edited one.
- In the cached run, `base` is a deferred hit (Task 11) that never loaded, so its definition is the original.

The two keys differ, and `upscale` never matches its own cache entry. The Task 7 and Task 11 tests use a mock load that edits nothing, so they could not see this.

**Rule:** each pipeline step's cache key is computed once per run, from the prepared definition before any step executes. Both the run and `cache_hits` compute it that way. `borrowed_pipeline_keys` reads those precomputed keys and never hashes a definition that a load may have changed.

**Files:**
- Modify: `dw/step_cache.py` (`borrowed_pipeline_keys` takes the precomputed keys)
- Modify: `dw/workflow.py` (`run` and `cache_hits` compute the key table once, before the step loop, and pass it through `_cache_lookup` and the deferred-borrow load)
- Test: `tests/test_workflow_step_cache.py`

**Interfaces:**
- Produces: `borrowed_pipeline_keys(steps, index, pipeline_keys)`, where `pipeline_keys: dict[str, str]` maps a step name to the `pipeline_cache_key` of its pipeline definition, computed before the run executes anything. Every caller passes it, and the function no longer imports `pipeline_cache_key`.

- [ ] **Step 1: Failing test.** Write a mock load (patch.object on `Pipeline.load`) that edits its own definition the way the real one does (`self.from_pretrained_arguments.pop("model_name", None)`, or whatever attribute path reaches the step's definition dict). Run the shared-components workflow (A shares, B reuses) twice with no argument change. The second run must be served entirely from the cache: B reused, and neither A nor B loads. Today this fails with B missing.
- [ ] **Step 2:** Run it and see B miss.
- [ ] **Step 3:** Implement the precomputed key table and thread it through.
- [ ] **Step 4:** Run the step-cache, pipeline-caching, workflow, worker and events suites, then the full suite. `arch_metrics --check` must exit 0.
- [ ] **Step 5:** Commit: `fix(step-cache): borrowed pipeline keys come from the definition as written, not as loaded`.

Deferred to Phase 2: `Pipeline.load` must stop editing the workflow definition it was given (copy on entry). The edits are a hazard beyond the cache key.
