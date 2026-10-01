# Phase 1 Implementation Plan: one path from request to run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every request reaches the GPU through one admission, one prepare pipeline and one client surface, measured by metrics v2 before and after.

**Architecture:** Metrics come first, so the phase is measured from its own baseline. Then:

1. The REPL is deleted, which leaves `JobManager` as the worker's only sender.
2. The three prepare paths collapse into one **fold** stage (variables) and one **expand** stage (definition). Validation, the run and the realized record all consume these two stages.
3. The server admits a request once in `dw/server/admission.py`. That means one `Workflow` instance, one expansion and every warning, shared by the validate, submit and rerun routes.
4. `python -m dw.run` becomes an HTTP client of `dw.serve`.

**Tech Stack:** Python 3.10+, FastAPI, httpx, pytest. Metrics use ruff C901, grimp (via import-linter), networkx and pygount.

**Spec:** [ROADMAP.md](ROADMAP.md) (the Phase 1 row, "Metrics", "Working rules") and [ASSESSMENT.md](ASSESSMENT.md) (decisions and findings). The Phase 0 plan [phase-0.md](phase-0.md) is the precedent for format and gate.

## Global Constraints

- Hard freeze: no net-new features or functionality. A task may change surface (routes, CLI flags, file layout) only where the consolidation requires it; every such change is listed in the task and in the release notes.
- Work happens on `stabilization/phase-1` branches in the worktree `/Users/don/src/dkackman/dw-stabilization`. **Tasks 1-2 merge to `develop` on their own (as `stabilization/phase-1a`) before Task 3 starts**, so the harness ratchet can run against the new baseline for the rest of the phase.
- `scripts/arch_metrics.py --check docs/stabilization/baseline.json` passes at the end of every task.
- The `modules` budget is a hard constraint. Task 3 deletes three modules and adds one. Task 5 adds one. No other task adds a module.
- Every new test fails before its fix. No count-pinning tests.
- Add no string `patch("dw...")` targets. Use `patch.object`, injected fakes, or `RunContext` event capture.
- Do not move functions between modules except where a task names the move. Phase 3 is the phase for module moves.
- Run tests locally with `venv/bin/python -m pytest` (torch is present). Nothing is shipped to lem except at the gate.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or whichever model wrote the commit).
- New dev dependencies are `import-linter>=2.15` (it brings grimp), `networkx>=3.0` and `pygount>=3.2`. `httpx>=0.28.1` moves into the base `dependencies`. No other dependency changes.

## Decisions carried into this plan (rulings, 2026-09-28)

- **Worker validation stays.** After this phase the server admits each request exactly once: one `Workflow` instance and one expansion. The worker still calls `validate()` on the definition it actually loads, because it re-reads `workflow_path` from disk and a file edited during a long queue wait would otherwise run unchecked. So the gate reads "the server admits once" and not "validated once end to end". Making the job carry an admitted snapshot is Phase 2 work, alongside the typed worker protocol. It needs run-directory identity carried separately from the definition.
- **The realized workflow stays unexpanded.** `workflow.json` keeps `for_each` (CLAUDE.md, *Type System*). The record consumes the **fold** stage's variables and never an expanded definition.
- **`dw.run` does not start a server.** If none is reachable it says how to start one and exits 2. `dw.test` (the installation check) and `dw.validate` (a pure pre-flight) stay in-process.
- **One complexity tool.** Ruff C901 supplies the ratchet, the distribution and the top ten. Gate 0's hand-computed distribution came from mccabe with nested functions folded in (1,485 functions, `create_app` 510). Task 2 regenerates it with ruff (1,719 functions, `create_app` 440) and footnotes the change.
- **A second cycle ratchet.** Beside `import_cycles` (strongly connected components, 6 today), `modules_in_import_cycles` (26 today) is ratcheted too. A 16-module knot runs through `arguments`, `result`, `pipeline`, `runs`, `step_cache` and `tasks`, and without the second count it could grow while the cycle count stayed at 6. It grew 22 → 26 during Phase 0.

## Review Focus

1. **A `constant:` variable default that fails to resolve.** Validation and the run should both name `variables.<name>`. Today only validation does. Task 4 adds `test_a_failing_constant_names_its_variable_at_run_time`.
2. **Arguments that fail `argument_errors`.** Validate should report them and still expand with the defaults; submit should answer 400 and never reach the worker. Task 5 adds `test_bad_arguments_are_refused_before_the_queue_and_validate_still_expands`.
3. **`dw.run` pointed at a local file the server cannot reach.** It should fall back to an inline definition and say that relative references now resolve on the server. A server-side failure should print the server's message and exit non-zero, with no traceback. Task 6 adds `test_a_file_the_server_cannot_reach_is_sent_inline_with_a_notice`.
4. **`dw.run` with no server, or a rejected token.** It should print one line with a remedy and exit 2, with no traceback. Task 6 adds `test_no_server_is_one_line_and_exit_2`.
5. **Rerunning a job whose `asset:` argument has since been deleted.** It should get the same 400 a fresh submit gets. Today it is queued and fails on the GPU. Task 5 adds `test_a_rerun_rechecks_its_references`.

## File map

| File | Change | Task |
| --- | --- | --- |
| `scripts/arch_metrics.py` | complexity + import-graph metrics | 1 |
| `scripts/arch_report.py` | new: the per-gate report | 2 |
| `tests/test_arch_metrics.py`, `tests/test_arch_report.py` | tests | 1, 2 |
| `docs/stabilization/baseline.json`, `ROADMAP.md` | re-baseline; Metrics text; gate 0 report regenerated | 1, 2, 8 |
| `pyproject.toml` | dev deps (1); `dw-repl` script removed (3); httpx to base (6) | 1, 3, 6 |
| `dw/repl.py`, `dw/repl_commands.py` | deleted | 3 |
| `dw/repl_worker.py` → `dw/worker_manager.py` | moved | 3 |
| `dw/workflow.py` | fold/expand stages; memoized expansion | 4 |
| `dw/realize.py`, `dw/plan.py` | consume the fold | 4 |
| `dw/server/admission.py` | new: `admit()` | 5 |
| `dw/server/app.py`, `dw/server/jobs.py` | routes and `submit` use `admit()` | 5 |
| `dw/run.py`, `tests/test_cli.py` | HTTP client of `dw.serve` | 6 |
| `dw/runs.py` | `open_run` survives a parent swept mid-claim | 7 |
| `docs/stabilization/hot-zone.txt` | Phase 1 files added | 1 |

---

### Task 1: Metrics v2 ratchets (complexity and import cycles)

**Files:**
- Modify: `scripts/arch_metrics.py` (replace whole file with the code below)
- Modify: `tests/test_arch_metrics.py` (append tests)
- Modify: `pyproject.toml` (dev extras)
- Modify: `docs/stabilization/baseline.json` (re-baseline)
- Modify: `docs/stabilization/ROADMAP.md` ("Metrics" section)
- Modify: `docs/stabilization/hot-zone.txt`

**Interfaces:**
- Produces:
  - `complexities(root) -> list[tuple[int, str, str]]`, returning `(cc, "path:line", name)` highest first.
  - `import_graph(root) -> {"cycles": list[list[str]], "edges": list[list[str]]}`.
  - `measure(root)` gains the keys `complex_functions`, `import_cycles` and `modules_in_import_cycles`.
  - `EXCLUDED` and `PACKAGES` stay module constants that Task 2 imports.

- [ ] **Step 1: Add the dev dependencies and install them**

In `pyproject.toml`, append to the `dev` extras after `"pylint>=3.3",`:

```toml
    "import-linter>=2.15",
    "networkx>=3.0",
    "pygount>=3.2",
```

Run: `venv/bin/pip install -e '.[dev]'`
Expected: grimp 3.17+, networkx and pygount installed. `venv/bin/python -c "import grimp, networkx, pygount"` exits 0.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_arch_metrics.py`; they reuse its `_load` and `_tree`)

```python
def _branches(count):
    body = "".join(f"    if x == {i}:\n        return {i}\n" for i in range(count))
    return f"def branchy(x):\n{body}    return -1\n"


def test_a_function_over_complexity_15_is_counted_and_one_at_15_is_not(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {"dw/a.py": _branches(15), "dw_mcp/b.py": _branches(14)},
        )
    )
    # n ifs score n + 1: 16 is over the limit, 15 is not
    assert metrics["complex_functions"] == 1


def test_complexities_name_the_function_and_where_it_is(tmp_path):
    found = _load().complexities(_tree(tmp_path, {"dw/a.py": _branches(3)}))
    assert found == [(4, "dw/a.py:1", "branchy")]


def test_community_pipelines_do_not_count_toward_complexity(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/community_pipelines/p.py": _branches(20)})
    )
    assert metrics["complex_functions"] == 0


def test_an_import_cycle_is_counted_with_its_modules(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/__init__.py": "",
                "dw/a.py": "from . import b\n",
                "dw/b.py": "def f():\n    from . import a\n",
                "dw/c.py": "from . import a\n",
            },
        )
    )
    # a lazy import closes a cycle as surely as a top-level one; c only
    # depends on it
    assert metrics["import_cycles"] == 1
    assert metrics["modules_in_import_cycles"] == 2


def test_a_folder_without_an_init_is_still_in_the_graph(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/__init__.py": "",
                "dw/a.py": "from .tasks import t\n",
                "dw/tasks/t.py": "from .. import a\n",
            },
        )
    )
    assert metrics["import_cycles"] == 1
    assert metrics["modules_in_import_cycles"] == 2


def test_a_type_checking_import_does_not_close_a_cycle(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/__init__.py": "",
                "dw/a.py": "from . import b\n",
                "dw/b.py": (
                    "from typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n    from . import a\n"
                ),
            },
        )
    )
    assert metrics["import_cycles"] == 0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_arch_metrics.py -q`
Expected: the six new tests fail with `KeyError: 'complex_functions'` / `AttributeError: ... 'complexities'`. The six existing tests pass.

- [ ] **Step 4: Replace `scripts/arch_metrics.py` with this** (verified against the repo: 21 / 6 / 26 at `f07eaf2d`)

```python
#!/usr/bin/env python
"""Architecture metrics for the stabilization gates and the harness ratchet.

Every metric is lower-is-better, so a ratchet is one comparison: a metric
that rose against the committed baseline is a regression. Run with --write
to record a baseline and --check to compare against one.

Counting rules, fixed so any commit measures the same way:
- Sources are dw/ and dw_mcp/, minus EXCLUDED (vendored community pipelines).
- Cyclomatic complexity is ruff's C901 (mccabe) as ruff reports it: each
  function scored on its own body, nested functions counted into it too.
- An import cycle is a strongly connected component of more than one module
  in grimp's import graph of dw + dw_mcp: lazy (function-level) imports
  count, TYPE_CHECKING imports do not. Folders without an __init__.py
  (dw/tasks, dw/pipeline_processors) are named to grimp explicitly, since it
  walks only regular packages.
"""

import argparse
import ast
import json
import os
import pathlib
import re
import subprocess
import sys

REFERENCE_PREFIXES = frozenset(
    (
        "asset:",
        "output:",
        "prompt:",
        "variable:",
        "previous_result:",
        "constant:",
        "item:",
        "gather:",
    )
)
# Modules allowed to spell a reference prefix. Empty until Phase 2 gives the
# prefixes one owner (dw/references.py).
PREFIX_OWNERS = frozenset()
EXCLUDED = ("community_pipelines", "node_modules", "venv", ".git")
PACKAGES = ("dw", "dw_mcp")
PATCH_TARGET = re.compile(r"""patch\(\s*["']dw[._]""")
COMPLEXITY_LIMIT = 15
COMPLEXITY_MESSAGE = re.compile(r"^`(?P<name>.+)` is too complex \((?P<cc>\d+) > 0\)$")

# Run in a subprocess rooted at the tree being measured: grimp finds a
# package through the import system, and this process (a pytest session,
# say) may already have another checkout's dw imported
_GRAPH = """
import json, sys, grimp, networkx
packages, namespaces, excluded = json.loads(sys.argv[1])
graph = grimp.build_graph(
    *packages, *namespaces, exclude_type_checking_imports=True, cache_dir=None
)
def measured(module):
    return module not in namespaces and not any(
        part in excluded for part in module.split(".")
    )
edges = networkx.DiGraph()
for module in filter(measured, graph.modules):
    edges.add_node(module)
    for imported in filter(measured, graph.find_modules_directly_imported_by(module)):
        edges.add_edge(module, imported)
print(json.dumps({
    "cycles": sorted(
        sorted(c) for c in networkx.strongly_connected_components(edges) if len(c) > 1
    ),
    "edges": sorted(edges.edges()),
}))
"""


def _sources(root, *packages):
    for package in packages:
        for path in sorted((root / package).rglob("*.py")):
            if not any(part in EXCLUDED for part in path.parts):
                yield path


def _present(root):
    return [name for name in PACKAGES if (root / name).is_dir()]


def _duplicate_blocks(paths):
    try:
        from pylint.checkers.symilar import Symilar
    except ImportError:
        return None
    similar = Symilar(
        min_lines=8,
        ignore_comments=True,
        ignore_docstrings=True,
        ignore_imports=True,
        ignore_signatures=True,
    )
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            similar.append_stream(str(path), stream)
    return len(similar._compute_sims())


def complexities(root):
    """Every function's cyclomatic complexity as (cc, "path:line", name),
    highest first. ruff reports a function only above its threshold, so the
    threshold is zero and every function reports."""
    root = pathlib.Path(root).resolve()
    packages = _present(root)
    if not packages:
        return []
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--exit-zero",
            "--select",
            "C901",
            "--config",
            "lint.mccabe.max-complexity=0",
            "--extend-exclude",
            ",".join(EXCLUDED),
            "--output-format",
            "json",
            *packages,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    found = []
    for item in json.loads(result.stdout):
        match = COMPLEXITY_MESSAGE.match(item["message"])
        path = pathlib.Path(item["filename"]).resolve().relative_to(root)
        where = f"{path.as_posix()}:{item['location']['row']}"
        found.append((int(match["cc"]), where, match["name"]))
    return sorted(found, key=lambda entry: (-entry[0], entry[1]))


def _namespace_packages(root, packages):
    """Folders below a measured package that hold modules but no __init__.py."""
    found = set()
    for path in _sources(root, *packages):
        folder = path.parent
        if folder.parent != root and not (folder / "__init__.py").exists():
            found.add(".".join(folder.relative_to(root).parts))
    return sorted(found)


def import_graph(root):
    """grimp's view of dw + dw_mcp as {"cycles": [[module, ...]], "edges":
    [[importer, imported]]}. Only a package with an __init__.py is walked,
    so a tree without one (a test's) has no graph."""
    root = pathlib.Path(root).resolve()
    packages = [
        name for name in _present(root) if (root / name / "__init__.py").exists()
    ]
    if not packages:
        return {"cycles": [], "edges": []}
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(root), os.environ.get("PYTHONPATH")))
        ),
    )
    spec = json.dumps([packages, _namespace_packages(root, packages), EXCLUDED])
    result = subprocess.run(
        [sys.executable, "-c", _GRAPH, spec],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def measure(root):
    root = pathlib.Path(root)
    engine = list(_sources(root, *PACKAGES))
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
    metrics["complex_functions"] = sum(
        cc > COMPLEXITY_LIMIT for cc, _, _ in complexities(root)
    )
    cycles = import_graph(root)["cycles"]
    metrics["import_cycles"] = len(cycles)
    metrics["modules_in_import_cycles"] = sum(len(cycle) for cycle in cycles)
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
    parser.add_argument(
        "--root", default=pathlib.Path(__file__).resolve().parent.parent
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", help="write the metrics to this JSON file")
    group.add_argument(
        "--check", help="fail if any metric is worse than this JSON file"
    )
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

- [ ] **Step 5: Run the tests, then measure the repository**

Run: `venv/bin/python -m pytest tests/test_arch_metrics.py -q`
Expected: 12 passed.

Run: `venv/bin/python scripts/arch_metrics.py`
Expected: every Phase 0 value unchanged (133, 10, 19, 93, 285, 964, 21), plus `"complex_functions": 21`, `"import_cycles": 6` and `"modules_in_import_cycles": 26`. If a number differs, stop and report it. Don't adjust the code to hit the number.

- [ ] **Step 6: Re-baseline**

Run: `venv/bin/python scripts/arch_metrics.py --write docs/stabilization/baseline.json`
Then run `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. Expected: exit 0.

- [ ] **Step 7: Update ROADMAP.md "Metrics"**

In the **Ratchets** bullet list, replace the "Added at the start of Phase 1" sub-bullets with:

```markdown
- Added at the start of Phase 1 ("metrics v2"), re-baselined in the same commit:
  - **Cyclomatic complexity:** functions above 15, by ruff's C901 (already a dev dependency). Baseline 21.
  - **Import cycles:** strongly connected components of more than one module in grimp's graph of `dw` + `dw_mcp`, lazy imports included, TYPE_CHECKING imports excluded. Baseline 6.
  - **Modules inside import cycles:** the sum of those components' sizes. Baseline 26, because one of the six is a 16-module knot (`arguments`, `result`, `pipeline`, `runs`, `step_cache`, `tasks`, ...) that could grow without the cycle count moving. It grew from 22 to 26 in Phase 0 (the `step_cache` → `pipeline` import). The target is 0 for both cycle ratchets, reached by Phases 2-3. Each ratchet only forbids getting worse.
```

In the paragraph under **"Every gate report carries the full metrics table"**, extend the row list with: SLOC by layer (engine, API = `dw/server` + `dw_mcp`, UI = `ui/src` without its tests; pygount code lines; for reference only), package instability, and LCOM4 on `Workflow`, `Pipeline`, `Result` and `JobManager` with their method counts. (`create_app` is a closure, not a class. It is measured when Phase 3 splits it into routers.) Add one sentence: "All complexity numbers are ruff C901; gate 0's hand table used mccabe and is regenerated by `arch_report.py`."

- [ ] **Step 8: Add Phase 1's new files to the hot zone**

Append to `docs/stabilization/hot-zone.txt` after `dw/validate.py`:

```text
dw/worker_manager.py
dw/runs.py
dw/server/admission.py
pyproject.toml
```

- [ ] **Step 9: Commit**

```bash
git add scripts/arch_metrics.py tests/test_arch_metrics.py pyproject.toml docs/stabilization/baseline.json docs/stabilization/ROADMAP.md docs/stabilization/hot-zone.txt
git commit -m "feat(metrics): complexity and import-cycle ratchets; re-baseline (metrics v2)"
```

---

### Task 2: `scripts/arch_report.py` and the regenerated gate 0 report

**Files:**
- Create: `scripts/arch_report.py`
- Create: `tests/test_arch_report.py`
- Modify: `docs/stabilization/ROADMAP.md` (the Gate 0 report)

**Interfaces:**
- Consumes: `arch_metrics.complexities`, `import_graph`, `measure`, `EXCLUDED` (Task 1).
- Produces:
  - CLI `scripts/arch_report.py [--root R] [LABEL=REF ...]`, which prints Markdown. The default columns are "Before Phase 0" (`3afd70e9`), every `stabilization-gate-N` tag, and HEAD when HEAD is not the newest tag.
  - Functions `lcom4(tree, path, class_name) -> str | None`, `instability(edges) -> {package: (ca, ce, i)}`, `coupling(root, ref) -> (Counter, pairs)` and `hotspots(revisions, functions) -> rows`.

- [ ] **Step 1: Write the failing tests** (`tests/test_arch_report.py`)

```python
"""scripts/arch_report.py - the gate report's measurements. Each is checked
on a tree small enough to count by hand, never on the real repository."""

import importlib.util
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "arch_report.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("arch_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lcom4_counts_methods_that_share_nothing_as_separate_components(tmp_path):
    (tmp_path / "m.py").write_text(
        "class C:\n"
        "    def a(self):\n        return self.x\n"
        "    def b(self):\n        self.x = 1\n"
        "    def c(self):\n        return self.a()\n"
        "    def d(self):\n        return self.y\n"
    )
    # a-b share x, c calls a; d touches only y
    assert _load().lcom4(tmp_path, "m.py", "C") == "2 (4 methods)"


def test_lcom4_of_a_missing_class_is_none(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    assert _load().lcom4(tmp_path, "m.py", "C") is None


def test_instability_counts_modules_across_package_lines():
    edges = [
        ["dw.server.app", "dw.workflow"],
        ["dw.server.jobs", "dw.workflow"],
        ["dw.workflow", "dw.tasks.task"],
        ["dw.server.app", "dw.server.jobs"],
    ]
    result = _load().instability(edges)
    # the server's two modules each reach out and nothing reaches in
    assert result["dw.server"] == (0, 2, 1.0)
    assert result["dw (core)"] == (2, 1, 0.33)
    assert result["dw.tasks"] == (1, 0, 0.0)


def _commit(repo, files, message):
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.read_text() + "x\n" if path.exists() else "x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message],
        cwd=repo,
        check=True,
    )


def test_coupling_reports_pairs_that_change_together(tmp_path):
    module = _load()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for i in range(5):
        _commit(tmp_path, ["dw/a.py", "dw/b.py"], f"both {i}")
    _commit(tmp_path, ["dw/a.py", "dw/c.py"], "once")
    revisions, pairs = module.coupling(tmp_path, "HEAD")
    assert revisions["dw/a.py"] == 6
    # five shared of a mean of (6 + 5) / 2 revisions; a-c shares only one
    assert pairs == [(5, 91, "dw/a.py", "dw/b.py")]


def test_a_changeset_too_large_to_mean_anything_is_skipped(tmp_path):
    module = _load()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    files = [f"dw/m{i}.py" for i in range(module.COUPLING_MAX_CHANGESET + 1)]
    for i in range(5):
        _commit(tmp_path, files, f"sweep {i}")
    revisions, pairs = module.coupling(tmp_path, "HEAD")
    assert pairs == [] and not revisions


def test_hotspots_multiply_churn_by_total_complexity():
    import collections

    revisions = collections.Counter({"dw/a.py": 3, "dw/b.py": 10})
    functions = [(5, "dw/a.py:1", "f"), (4, "dw/a.py:9", "g"), (1, "dw/b.py:1", "h")]
    assert _load().hotspots(revisions, functions) == [
        (27, 3, 9, "dw/a.py"),
        (10, 10, 1, "dw/b.py"),
    ]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_arch_report.py -q`
Expected: every test errors with `FileNotFoundError` on `scripts/arch_report.py`.

- [ ] **Step 3: Create `scripts/arch_report.py`** (verified: three columns in about 17 s)

```python
#!/usr/bin/env python
"""The stabilization gate report: every metric, one column per gate.

Each ref is measured by *this* script's rules over an archive of that ref,
never by the script as it stood then, so every column counts the same way.
Default columns: before Phase 0 (3afd70e9), every stabilization-gate-N
tag, and HEAD when it is not the newest tag. Markdown on stdout, for the
Gate reports section of docs/stabilization/ROADMAP.md.

Reports, never gates: nothing here fails a build.
"""

import argparse
import ast
import collections
import itertools
import pathlib
import statistics
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from arch_metrics import EXCLUDED, complexities, import_graph, measure  # noqa: E402

BEFORE_PHASE_0 = ("Before Phase 0", "3afd70e9")
ARCHIVED = ["dw", "dw_mcp", "tests", "ui/src", ":(glob)**/CLAUDE.md"]
RATCHETS = [
    ("modules", "Engine + MCP modules"),
    ("modules_over_1000_lines", "Modules over 1,000 lines"),
    ("functions_over_150_lines", "Functions over 150 lines"),
    ("complex_functions", "Functions over cyclomatic complexity 15"),
    ("import_cycles", "Import cycles"),
    ("modules_in_import_cycles", "Modules inside import cycles"),
    ("duplicate_blocks", "Duplicate-code blocks, cross-file"),
    ("prefix_literals", "Reference-prefix literals"),
    ("test_dw_patch_targets", 'Test `patch("dw...")` targets'),
    ("claude_md_lines", "CLAUDE.md lines, all files"),
]
# Layers for SLOC: reference only. Tests are counted in no layer
LAYERS = [
    ("Engine", "dw", lambda p: "server" not in p.parts),
    ("API (dw/server + dw_mcp)", "dw/server", lambda p: True),
    ("API (dw/server + dw_mcp)", "dw_mcp", lambda p: True),
    ("UI (ui/src, tests excluded)", "ui/src", lambda p: ".test." not in p.name),
]
SLOC_SUFFIXES = (".py", ".ts", ".svelte", ".js", ".css")
# Instability is measured per package, dw's own modules standing as "core"
PACKAGES = [
    ("dw_mcp", "dw_mcp"),
    ("dw.server", "dw.server"),
    ("dw.tasks", "dw.tasks"),
    ("dw.pipeline_processors", "dw.pipeline_processors"),
]
LCOM4_CLASSES = [
    ("dw/workflow.py", "Workflow"),
    ("dw/pipeline_processors/pipeline.py", "Pipeline"),
    ("dw/result.py", "Result"),
    ("dw/server/jobs.py", "JobManager"),
]
COUPLING_SINCE = "2026-08-01"
COUPLING_MIN_SHARED = 5
COUPLING_MAX_CHANGESET = 30
TOP = 10


def git(*args, root):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def default_refs(root):
    refs = [BEFORE_PHASE_0]
    tags = git("tag", "--list", "stabilization-gate-*", "--sort=creatordate", root=root)
    for tag in tags.split():
        refs.append((f"Gate {tag.rsplit('-', 1)[1]}", tag))
    head = git("rev-parse", "HEAD", root=root).strip()
    if git("rev-parse", refs[-1][1] + "^{commit}", root=root).strip() != head:
        refs.append(("HEAD", head[:8]))
    return refs


def extract(ref, root, into):
    # tarfile's filter="data" needs Python 3.10.12+ / 3.11.4+ / 3.12+
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref, "--", *ARCHIVED],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryFile() as stream:
        stream.write(archive)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(into, filter="data")


def sloc(tree):
    from pygount import SourceAnalysis

    counts = collections.Counter()
    for layer, folder, keep in LAYERS:
        for path in sorted((tree / folder).rglob("*")):
            if (
                path.is_file()
                and path.suffix in SLOC_SUFFIXES
                and keep(path.relative_to(tree))
                and not any(part in EXCLUDED for part in path.parts)
            ):
                counts[layer] += SourceAnalysis.from_file(str(path), layer).code_count
    return counts


def package_of(module):
    for name, prefix in PACKAGES:
        if module == prefix or module.startswith(prefix + "."):
            return name
    return "dw (core)"


def instability(edges):
    """Ca, Ce and I = Ce / (Ca + Ce) per package, counted in modules: Ca is
    how many modules outside the package import one inside it, Ce how many
    inside import one outside."""
    afferent = collections.defaultdict(set)
    efferent = collections.defaultdict(set)
    for importer, imported in edges:
        source, target = package_of(importer), package_of(imported)
        if source != target:
            efferent[source].add(importer)
            afferent[target].add(importer)
    result = {}
    for name in sorted(set(afferent) | set(efferent)):
        ca, ce = len(afferent[name]), len(efferent[name])
        result[name] = (ca, ce, round(ce / (ca + ce), 2) if ca + ce else 0.0)
    return result


def lcom4(tree, path, class_name):
    """LCOM4: connected components among a class's methods, two methods
    joined when they share a `self.` attribute or one calls the other. 1 is
    cohesive; more means the class is that many classes. Reported with the
    method count, since a split shows as fewer methods before it shows as
    fewer components. None when absent."""
    source = (tree / path).read_text(encoding="utf-8")
    node = next(
        (
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.ClassDef) and n.name == class_name
        ),
        None,
    )
    if node is None:
        return None
    methods = {
        m.name: m
        for m in node.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    touches = {}
    for name, method in methods.items():
        used = set()
        for sub in ast.walk(method):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                used.add(sub.attr)
        touches[name] = used
    parent = {name: name for name in methods}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for a, b in itertools.combinations(methods, 2):
        shared = (touches[a] & touches[b]) - set(methods)
        calls = b in touches[a] or a in touches[b]
        if shared or calls:
            parent[find(a)] = find(b)
    return f"{len({find(name) for name in methods})} ({len(methods)} methods)"


def coupling(root, ref):
    """File pairs that change together, code-maat's way: non-merge commits
    since COUPLING_SINCE touching at most COUPLING_MAX_CHANGESET engine
    files, pairs sharing at least COUPLING_MIN_SHARED commits, degree =
    shared / mean(revisions of each)."""
    log = git(
        "log",
        ref,
        "--no-merges",
        f"--since={COUPLING_SINCE}",
        "--name-only",
        "--format=@@%h",
        "--",
        "dw",
        "dw_mcp",
        root=root,
    )
    revisions = collections.Counter()
    shared = collections.Counter()
    for block in log.split("@@")[1:]:
        files = sorted(
            {
                f
                for f in block.split()
                if f.endswith(".py") and not any(p in EXCLUDED for p in f.split("/"))
            }
        )
        if not files or len(files) > COUPLING_MAX_CHANGESET:
            continue
        revisions.update(files)
        shared.update(itertools.combinations(files, 2))
    pairs = [
        (count, round(100 * count / ((revisions[a] + revisions[b]) / 2)), a, b)
        for (a, b), count in shared.items()
        if count >= COUPLING_MIN_SHARED
    ]
    return revisions, sorted(pairs, key=lambda p: (-p[1], -p[0], p[2]))[:TOP]


def hotspots(revisions, functions):
    """Churn (revisions in the coupling window) x the file's total
    cyclomatic complexity."""
    total = collections.Counter()
    for cc, where, _ in functions:
        total[where.rsplit(":", 1)[0]] += cc
    rows = [
        (revisions[f] * total[f], revisions[f], total[f], f)
        for f in total
        if revisions[f]
    ]
    return sorted(rows, reverse=True)[:TOP]


def column(root, ref):
    with tempfile.TemporaryDirectory() as into:
        tree = pathlib.Path(into)
        extract(ref, root, tree)
        values = measure(tree)
        functions = complexities(tree)
        edges = import_graph(tree)["edges"]
        layers = sloc(tree)
        cohesion = {f"{cls}": lcom4(tree, path, cls) for path, cls in LCOM4_CLASSES}
    scores = [cc for cc, _, _ in functions]
    values.update(
        {
            "over_30": sum(cc > 30 for cc in scores),
            "functions": len(scores),
            "median": statistics.median(scores),
            "mean": round(statistics.mean(scores), 2),
            **{
                f"over_{t}_dist": sum(cc > t for cc in scores) for t in (10, 15, 20, 30)
            },
            "sloc": layers,
            "instability": instability(edges),
            "lcom4": cohesion,
        }
    )
    return values, functions


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def report(root, refs):
    columns = [(label, ref, *column(root, ref)) for label, ref in refs]
    headers = ["Metric (lower is better)"] + [
        f"{label} (`{ref}`)" for label, ref, _, _ in columns
    ]
    rows = [[title] + [c[2].get(key) for c in columns] for key, title in RATCHETS]
    rows.insert(
        4,
        ["Functions over cyclomatic complexity 30"]
        + [c[2]["over_30"] for c in columns],
    )
    out = ["#### Metrics", "", table(headers, rows), ""]

    dist = [
        ["Functions"] + [c[2]["functions"] for c in columns],
        ["Median complexity"] + [c[2]["median"] for c in columns],
        ["Mean complexity"] + [c[2]["mean"] for c in columns],
    ] + [
        [f"Over {t}"] + [c[2][f"over_{t}_dist"] for c in columns]
        for t in (10, 15, 20, 30)
    ]
    out += [
        "#### Complexity distribution (ruff C901)",
        "",
        table(["", *headers[1:]], dist),
        "",
    ]

    layer_names = list(dict.fromkeys(name for name, _, _ in LAYERS))
    out += [
        "#### SLOC by layer (pygount code lines; reference only)",
        "",
        table(
            ["Layer", *headers[1:]],
            [[n] + [c[2]["sloc"][n] for c in columns] for n in layer_names],
        ),
        "",
    ]
    packages = sorted({p for c in columns for p in c[2]["instability"]})
    out += [
        "#### Package instability, I = Ce / (Ca + Ce)",
        "",
        table(
            ["Package", *headers[1:]],
            [
                [p]
                + [
                    "Ca {} / Ce {} / I {}".format(*c[2]["instability"][p])
                    if p in c[2]["instability"]
                    else "-"
                    for c in columns
                ]
                for p in packages
            ],
        ),
        "",
        "#### LCOM4 (components; 1 = cohesive)",
        "",
        table(
            ["Class", *headers[1:]],
            [[cls] + [c[2]["lcom4"][cls] for c in columns] for _, cls in LCOM4_CLASSES],
        ),
        "",
    ]
    label, ref, _, functions = columns[-1]
    out += [
        f"#### Ten most complex functions at {label}",
        "",
        table(
            ["Complexity", "Function"],
            [[cc, f"{where} `{name}`"] for cc, where, name in functions[:TOP]],
        ),
        "",
    ]
    revisions, pairs = coupling(root, ref)
    out += [
        f"#### Change coupling at {label} (commits since {COUPLING_SINCE})",
        "",
        table(["Shared commits", "Degree %", "File", "File"], pairs),
        "",
        f"#### Hotspots at {label} (churn x total complexity)",
        "",
        table(
            ["Score", "Commits", "Complexity", "File"], hotspots(revisions, functions)
        ),
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=pathlib.Path(__file__).resolve().parent.parent,
        type=pathlib.Path,
    )
    parser.add_argument(
        "refs",
        nargs="*",
        help="LABEL=REF columns; default: before Phase 0, every gate tag, HEAD",
    )
    args = parser.parse_args(argv)
    refs = (
        [tuple(r.split("=", 1)) for r in args.refs]
        if args.refs
        else default_refs(args.root)
    )
    print(report(args.root, refs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/test_arch_report.py tests/test_arch_metrics.py -q`
Expected: 18 passed.

- [ ] **Step 5: Regenerate the gate 0 report**

Run: `venv/bin/python scripts/arch_report.py "Before Phase 0=3afd70e9" "Gate 0=stabilization-gate-0" > /tmp/gate0.md`

In ROADMAP.md's "### Gate 0" section, replace everything from the `| Metric (lower is better) |` table through the end of the "ten most complex" table with the contents of `/tmp/gate0.md`. Keep the bullets above it: ratchets, B2 timing, and the Tasks 11/12 lesson. Delete the bullet "Change-coupling, instability and LCOM4 reports for this gate are computed from the tag once `scripts/arch_report.py` lands", because they are now present. Add one line under the tables:

"Regenerated by `scripts/arch_report.py` at Phase 1 Task 2. The complexity figures are ruff C901 throughout. The hand table this replaced used mccabe with nested functions folded in, which counted 1,485 functions and scored `create_app` at 510."

Expected values to spot-check:
- modules in cycles 22 → 26;
- functions 1,712 → 1,719, median 2;
- SLOC engine 19,980 → 20,102, API 6,498 → 6,525, UI 2,266 → 2,266.

- [ ] **Step 6: Commit**

```bash
git add scripts/arch_report.py tests/test_arch_report.py docs/stabilization/ROADMAP.md
git commit -m "feat(metrics): arch_report.py gate report; gate 0 regenerated from its tag"
```

- [ ] **Step 7: Merge point.** The controller merges `stabilization/phase-1a` (Tasks 1-2) to `develop` and pushes. It then writes the stage B harness prompt (`docs/stabilization/harness/stage-b-ratchet.md`) and tells Don the harness can be turned back on. Task 3 starts from the merged `develop`.

---

### Task 3: Remove the REPL

**Files:**
- Delete: `dw/repl.py`, `dw/repl_commands.py`, `tests/test_repl_commands.py`, `tests/test_repl_ergonomics.py`, `docs/REPL_COMMANDS.md`
- Move: `dw/repl_worker.py` → `dw/worker_manager.py` (`git mv`; the content is unchanged apart from its docstring)
- Move: `tests/test_repl_worker.py` → `tests/test_worker_manager.py`
- Move: `docs/REPL_WORKER_GUIDE.md` → `docs/WORKER_GUIDE.md` (strip the REPL sections)
- Modify: `dw/server/jobs.py:22` (import) and `:1564` (comment), `pyproject.toml` (drop `dw-repl = "dw.repl:main"`)
- Modify: `tests/test_host_memory.py` (delete `test_the_repl_prints_host_memory_either_way`, lines 93-99)
- Modify: `tests/test_security_symlinks.py` (delete `test_the_repl_listing_drops_the_link`, lines 671-684). The property it pinned is covered by `workflow_names`'s own symlink test at line 642.
- Docs:
  - `README.md` lines 11, 187-188, 234-235
  - `CLAUDE.md` lines 29-30, 40, 61-63 (the "REPL Architecture" section becomes a short "Worker" section naming `dw/worker_manager.py`; "Common Commands" loses `python -m dw.repl`)
  - `docs/SECURITY.md` lines 28, 63, 205
  - `docs/MCP.md` line 3
  - `docs/TESTING.md` lines 50-51
  - `tests/README.md` lines 24-25
  - `docs/proposals/mcp-job-notifications.md` line 46

**Interfaces:**
- Produces: `from dw.worker_manager import WorkerManager`. The class and module constants are unchanged.
- Worker protocol: unchanged. Every command is also sent by `JobManager` (`ping` is test-only and stays).

- [ ] **Step 1: Write the failing test.** In `tests/test_worker_manager.py` (after the move), change the imports to `import dw.worker_manager as worker_manager` / `from dw.worker_manager import WorkerManager`, and rename the module references in the body. Run `venv/bin/python -m pytest tests/test_worker_manager.py -q`. It fails: `ModuleNotFoundError: dw.worker_manager`.
- [ ] **Step 2: Move and delete.** `git mv dw/repl_worker.py dw/worker_manager.py`. Reword its module docstring to "Worker process management: the GPU worker's lifecycle for JobManager." Delete the files listed above, and fix `dw/server/jobs.py:22` to `from ..worker_manager import WorkerManager`.
- [ ] **Step 3: Sweep for leftovers.** Run `grep -rnw "repl\|repl_worker\|repl_commands\|DiffusersWorkflowREPL\|dw-repl" --include=*.py --include=*.md --include=*.toml --include=*.sh . | grep -v "docs/stabilization\|docs/proposals/audits\|\.remember"`. Expected: no hits once the doc edits above are made. `docs/stabilization/` and the dated audit are history and stay as written.
- [ ] **Step 4: Run the full suite.** Run `venv/bin/python -m pytest -q -x`. Expected: all pass, and the test count drops by exactly the 14 deleted tests (5 + 7 + 1 + 1).
- [ ] **Step 5: Check the ratchets.** Run `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. Expected: exit 0, with `modules` 131, `complex_functions` 20 (`_workflow_run` gone), and `claude_md_lines` lower.
- [ ] **Step 6: Commit** `refactor: remove the REPL; WorkerManager moves to dw/worker_manager.py`

---

### Task 4: One prepare pipeline (fold and expand)

**Files:**
- Modify: `dw/workflow.py`: `expanded_definition` (:500-544), `_prepare_definition` (:1041-1145), `run` (:1339-1429, the prepare and realize calls), `cache_hits` (:1230-1277)
- Modify: `dw/realize.py`: `realize_workflow` (:45-110)
- Modify: `dw/plan.py`: `build_plan` (:88-110)
- Test: `tests/test_prepare_pipeline.py` (new), plus the updates to `tests/test_realize.py` and `tests/test_variable_constraints.py:352` that the signature changes require

**Interfaces:**
- Produces, on `Workflow`:
  - `_fold(self, definition, arguments, *, fold_arguments, constrain) -> dict | None`. Stage 1, in this order:
    1. realize constants per variable, with a failure raised as `ConstantError("variables.<name>", ...)` (the code at :528-535, moved here);
    2. `set_variables(arguments, variables)` when `fold_arguments`;
    3. `variables = resolve_variable_values(variables)`;
    4. `constrain(definition, variables)`;
    5. `definition["variables"] = variables`.

    Returns `variables`, or `None` when the definition declares none.
  - `_expand(definition, variables, source_indices=None) -> dict` (staticmethod). Stage 2: `replace_variables` when `variables` is not None, then `resolve_constraint_references`, then `expand_for_each(definition, source_indices)`.
  - `expanded_definition(self, arguments=None, source_indices=None)`, with the same signature and meaning as today, now computed as:
    1. `deepcopy`;
    2. `_fold(fold_arguments=bool(arguments) and not argument_errors(definition, arguments), constrain=snap_constraints)`;
    3. `_expand(...)`.

    It is **memoized per instance**, keyed by `json.dumps(arguments, sort_keys=True, default=repr)`. Each call returns a `copy.deepcopy` of the cached definition and extends the caller's `source_indices` from the cached list. An exception is never cached.
  - `folded_variables(self, arguments=None) -> dict | None`. A deepcopy of the variables from the same memoized fold. `build_plan` uses it.
  - `_prepare_definition(self, workflow_def, arguments, base_dir) -> (workflow_def, default_seed, recorded_variables)`. It runs:
    1. `_fold(fold_arguments=True, constrain=apply_constraints)`;
    2. `recorded_variables = copy.deepcopy(variables)`, taken before anything loads;
    3. `realize_args(variables, base_dir, apply_key_conventions=False)`;
    4. `_expand(workflow_def, variables)`;
    5. then, unchanged: `apply_vram_estimate`, `elide_definition`, seed coercion.
- Produces, in `dw/realize.py`: `realize_workflow(definition, variables, seed, base_dir=None, prompt_dir=None, output_root=None, workflow_dir=None, pin_outputs=True)`. The second parameter is now the **folded variables** (or `None`) instead of raw arguments. The body's `set_variables` + `snap_constraints` (:88-91) are replaced by `realized["variables"] = copy.deepcopy(variables)` when the definition declares variables. Everything after that (the seed, `_pin`, `_record_sub_workflows`) is unchanged.
- Consumers:
  - `run()` passes `recorded_variables` to `realize_workflow`.
  - `cache_hits()` ignores the third return value.
  - `build_plan` computes `expanded = candidate.expanded_definition(arguments)` (dropping the second `Workflow(realized, ...)` construction) and `realized, annotations = realize_workflow(definition, candidate.folded_variables(arguments), seed=0, ...)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_prepare_pipeline.py`). Each builds a `Workflow` from an inline definition with `workflow_from_definition(definition, str(tmp_path))`. Define the shared fixture once at module level:

```python
import copy
import json

import pytest

from dw.workflow import ConstantError, Workflow, workflow_from_definition

# One definition exercising every stage the three paths used to disagree on:
# a snap-up constraint, a for_each whose entry references another variable,
# and a "constraint:" frame_snap. (A frame_snap in a task's arguments is a
# validation error by design - loop_frames takes no such argument - but the
# prepare stages walk the whole definition, so it is the smallest place to
# put one. Verified against f07eaf2d: the run resolves it, validation does
# not, and that is the divergence the first test catches.)
DEFINITION = {
    "id": "prep",
    "seed": 7,
    "variables": {
        "num_frames": 108,
        "tail": 5,
        "clips": [{"name": "a", "len": "variable:tail"}, {"name": "b", "len": 9}],
    },
    "variable_constraints": {
        "num_frames": {
            "modulus": 17, "remainder": 5, "min_frames": 124, "max_frames": 345,
            "snap": "up",
        }
    },
    "steps": [
        {
            "name": "clip",
            "for_each": "variable:clips",
            "task": {
                "command": "loop_frames",
                "arguments": {"video": "item:name", "num_frames": "item:len"},
            },
            "result": {"content_type": "video/mp4"},
        },
        {
            "name": "long",
            "task": {
                "command": "loop_frames",
                "arguments": {
                    "video": "gather:clip",
                    "num_frames": "variable:num_frames",
                    "frame_snap": "constraint:num_frames",
                },
            },
            "result": {"content_type": "video/mp4"},
        },
    ],
}
```

  Tests, one behavior each:

```python
def test_validation_expands_exactly_what_the_run_prepares(tmp_path):
    wf = workflow_from_definition(copy.deepcopy(DEFINITION), str(tmp_path))
    arguments = {"num_frames": 108}
    validated = wf.expanded_definition(arguments)
    prepared, seed, _ = wf._prepare_definition(
        copy.deepcopy(wf.workflow_definition), arguments, str(tmp_path)
    )
    assert validated["steps"] == prepared["steps"]
    assert seed == 7


def test_the_record_carries_the_run_s_folded_values(tmp_path):
    wf = workflow_from_definition(copy.deepcopy(DEFINITION), str(tmp_path))
    _, _, recorded = wf._prepare_definition(
        copy.deepcopy(wf.workflow_definition), {"num_frames": 108}, str(tmp_path)
    )
    from dw.realize import realize_workflow

    realized, _ = realize_workflow(wf.workflow_definition, recorded, seed=7)
    assert realized["variables"]["num_frames"] == 124  # snapped, as run
    assert realized["variables"]["clips"][0]["len"] == 5  # entry resolved
    assert realized["steps"][0]["for_each"] == "variable:clips"  # unexpanded


def test_a_failing_constant_names_its_variable_at_run_time(tmp_path):
    definition = copy.deepcopy(DEFINITION)
    definition["variables"]["sigmas"] = "constant:diffusers.no_such_module.X"
    wf = workflow_from_definition(definition, str(tmp_path))
    with pytest.raises(ConstantError) as raised:
        wf._prepare_definition(copy.deepcopy(definition), {}, str(tmp_path))
    assert raised.value.path == "variables.sigmas"


def test_expansion_is_computed_once_per_workflow_and_arguments(tmp_path):
    wf = workflow_from_definition(copy.deepcopy(DEFINITION), str(tmp_path))
    calls = []
    original = Workflow._fold

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Workflow, "_fold", counting)
        first = wf.expanded_definition({"num_frames": 108})
        first["steps"].clear()
        second = wf.expanded_definition({"num_frames": 108})
        wf.validation_errors({"num_frames": 108})
    assert len(calls) == 1
    assert second["steps"]  # the cache was not mutated through `first`
```

- [ ] **Step 2: Run them to verify they fail.**
  - Run: `venv/bin/python -m pytest tests/test_prepare_pipeline.py -q`
  - Expected failures:
    - the first test fails on step equality: validation leaves `frame_snap` as `"constraint:num_frames"` and the run resolves it;
    - the second fails because the record leaves `"variable:tail"` unresolved in the entry;
    - the third fails because a bare `ValueError`/`ImportError` is raised rather than `ConstantError`;
    - the fourth fails because `_fold` does not exist.
- [ ] **Step 3: Implement** the interfaces above in `dw/workflow.py`, `dw/realize.py` and `dw/plan.py`.
  - Keep every existing comment that explains a stage's order, moving it with its stage.
  - `validation_errors` and the six `*_warnings` helpers need no change. They already call `expanded_definition`, which is now memoized.
  - `validation_errors` still catches `ForEachError` / `ConstantError` / `VariableNotFoundError` around it. Because validation now runs `resolve_constraint_references`, an undeclared `frame_snap` name would raise a bare `ValueError` inside expansion. Do **not** add an `except ValueError`: `ConstantError` is a `ValueError` subclass, so such a clause either shadows its handler or swallows unrelated failures. Instead, move `constraint_reference_errors(self.workflow_definition)` from the tail concatenation (:846) into the pre-expansion gate beside the schema check (:684-689), and return its findings before expanding. Add `test_an_undeclared_frame_snap_name_is_reported_at_its_path_not_raised` to `tests/test_prepare_pipeline.py`.
- [ ] **Step 4: Run the new tests, then the full suite.**
  - Run `venv/bin/python -m pytest tests/test_prepare_pipeline.py -q`, then `venv/bin/python -m pytest -q`.
  - Expected: all pass.
  - Fingerprint tests in `tests/test_plan*.py` may change value, because the fingerprint is now taken over the candidate's own expansion (with `prompt:` not inlined into steps, and prompt text still covered by `annotations`). Update any **literal** fingerprint expectation only after confirming the new value is stable across two calls. Record the change for the release notes: a bound acknowledgement made before deploy gets one 409.
- [ ] **Step 5: Ratchets.** Run `scripts/arch_metrics.py --check`. Expected: exit 0. `functions_over_150_lines` may drop, since `run` shrinks.
- [ ] **Step 6: Commit** `refactor(workflow): one prepare pipeline - fold and expand shared by validation, run and record`

---

### Task 5: One admission service

**Files:**
- Create: `dw/server/admission.py`
- Modify: `dw/server/app.py`:
  - `_candidate_for` (:1141-1197) is deleted;
  - `_argument_reference_errors` (:1679) moves to `admission.py` unchanged;
  - `validate_workflow` (:1814-2080), `submit_job` (:1199-1287) and `rerun_job` (:1381-1425) call `admit()`;
  - the duplicate `constraint_errors` (:1187-1189) goes.
- Modify: `dw/server/jobs.py`: `JobManager.submit` (:789-926) stops loading and validating. `_validation_errors` (:773-787) is deleted.
- Test: `tests/test_admission.py` (new). Existing tests in `tests/test_server*.py` that called `manager.submit` with an invalid workflow and expected `ValueError` move to `tests/test_admission.py` as `admit()` tests. Their assertions on messages are kept verbatim.

**Interfaces:**
- Consumes: `Workflow.expanded_definition` (memoized, Task 4), `build_plan`, `argument_errors`, `constraint_errors`, the warning helpers, `activate_asset_dir` / `deactivate_asset_dir`.
- Produces (`dw/server/admission.py`):

```python
@dataclass
class Admission:
    workflow: Workflow            # the one instance this request builds
    arguments: dict               # as the caller sent them ({} when omitted)
    supplied: bool                # the caller sent `arguments` at all
    errors: list                  # [{path, message}]: schema, argument, constraint, reference
    warnings: list                # [str]: every warning /api/validate reports
    plan: dict | None = None      # build_plan's answer, when plan_for was given

    @property
    def ok(self):
        return not self.errors


def admit(*, workflow_path, workflow, arguments, base_dir, workspace,
          sources, ceiling_index, output_dir, workflow_dir,
          supplied=True, plan_for=None) -> Admission:
    """Load the request's workflow once and check it once, with the
    workspace's asset library active for every check (the validate route
    used to leave it off for its warnings). `plan_for`, a callable taking
    the Workflow, is called inside the same scope when the caller needs the
    plan (validate always; submit only for a bound acknowledgement)."""
```

- Produces: `JobManager.submit(...)` keeps its keyword signature, minus the validation it did. The docstring states: "Callers admit first (`dw.server.admission.admit`). submit records and queues; it does not re-check." `warnings=` carries `admission.warnings`.
- Produces: `JobManager.rerun(self, job_id, new_seed=False, acknowledged=ACK_NONE, acknowledged_cost=None, warnings=None)`. The new `warnings=` is passed through to `submit`. `rerun_job` admits the spec it gets from `manager.rerun_spec(job_id)` (with the new seed already drawn when `new_seed`), then calls `manager.rerun(..., warnings=admission.warnings)`. `rerun` must not re-check anything either.
- Surface changes (release-note them):
  - a submitted job records the full warning set `/api/validate` reports, not the smaller subset it recorded before;
  - `POST /api/jobs/{id}/rerun` now rechecks `asset:` / `prompt:` / `output:` references and answers 400 when one no longer resolves.

- [ ] **Step 1: Write the failing tests** (`tests/test_admission.py`). Use the `tests/test_server.py` fixtures (`create_app` with `ScriptedWorkerManager`) for route tests, and call `admit()` directly for unit tests. Required tests:
  - `test_a_submit_expands_once`: `POST /api/jobs` for a valid inline workflow with a `for_each`. `Workflow._fold` is counted (the counting-wrapper pattern from Task 4's test, via `monkeypatch.setattr(Workflow, "_fold", ...)`) and called exactly once.
  - `test_a_validate_expands_once`: the same for `POST /api/validate` with `arguments` supplied, where the plan is built too. Exactly one `_fold`.
  - `test_every_admission_check_sees_the_workspace_asset_library`: in a named workspace, wrap one warning helper (`monkeypatch.setattr(Workflow, "shot_span_warnings", spy)`) so it records the active asset directory. Assert that it equals the workspace's `assets/`. Today this fails, because the warnings run after `deactivate_asset_dir`.
  - `test_bad_arguments_are_refused_before_the_queue_and_validate_still_expands`: an undeclared argument. Validate returns its error **and** expansion-derived warnings computed with the defaults. `POST /api/jobs` answers 400, and `ScriptedWorkerManager.commands` stays empty.
  - `test_a_rerun_rechecks_its_references`: submit with `arguments={"image": "asset:gone.png"}` while the asset exists, delete the asset file, then `POST /api/jobs/{id}/rerun` returns 400 naming `asset:gone.png`.
  - `test_a_submitted_job_records_the_warnings_validate_reports`: a workflow that triggers a validate-only warning today (for example `unseeded_cache_warnings`: no seed, with a step the cache would have hit). The job's `warnings` contain it.
- [ ] **Step 2: Run them to verify they fail.**
  - Run `venv/bin/python -m pytest tests/test_admission.py -q`.
  - Expected: `ModuleNotFoundError: dw.server.admission`.
  - After a stub module exists, the behavioral tests fail with ≥2 `_fold` calls, the wrong asset dir, the rerun queued, and the warning missing.
- [ ] **Step 3: Implement.**
  - `admit()` takes over, in this order:
    1. the loading `_candidate_for` and the validate route do today (`workflow_from_file` / `workflow_from_definition`);
    2. one `activate_asset_dir(workspace.assets)` around everything below, with deactivation in `finally`;
    3. `validation_errors`, `argument_errors` and `_argument_reference_errors`;
    4. every warning the validate route computes today (steps 7-16 of its body), with `inherited_vram_warnings` using `ceiling_index`;
    5. `plan_for`.
  - The three routes become thin:
    - resolve `workflow_path` via `resolve_workflow_reference`;
    - call `admit()`;
    - map `errors` to the same HTTP 400 bodies they return today (tests pin those);
    - check a bound acknowledgement against `admission.plan`;
    - call `manager.submit(..., warnings=admission.warnings)`.
  - The validate response shape is unchanged.
- [ ] **Step 4: Full suite.** Run `venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Ratchets.** Run `scripts/arch_metrics.py --check`. Expected: exit 0.
  - `modules` is +1 (132).
  - `complex_functions` should drop: `_argument_reference_errors` (24) and `validate_workflow` (17) leave the closure, and `create_app`'s score falls.
  - Report the new `create_app` score.
- [ ] **Step 6: Commit** `refactor(server): one admission service for validate, submit and rerun`

---

### Task 6: `python -m dw.run` becomes a client of `dw.serve`

**Files:**
- Modify: `dw/run.py` (rewrite)
- Modify: `pyproject.toml`: move `"httpx>=0.28.1"` from the `mcp` extra into `dependencies`. The `mcp` extra keeps only `mcp`.
- Modify: `tests/test_cli.py`: `TestRunEntryPoint`'s six tests are rewritten; `TestValidateEntryPoint` is untouched.
- Modify: `CLAUDE.md` "Common Commands" and `README.md` (the `dw.run` usage and flags)

**Interfaces:**
- Consumes:
  - `dw_mcp.client.DwClient(base_url=None, timeout=30.0, transport=None, token=None, workspace=None)`, `DwApiError(message, status_code)`, `resolve_base_url` (reads `DW_MCP_URL`, defaulting to `http://127.0.0.1:8765`) and `resolve_token` (`DW_API_TOKEN`).
  - `dw_mcp/client.py` imports only httpx, so `dw.run` must not import `dw_mcp.server`.
  - The routes `GET /api/health`, `POST /api/jobs` (`JobRequest`: `workflow_path` | `workflow`, `arguments`, `workspace`), `GET /api/jobs/{id}`, `GET /api/jobs/{id}/event-log?after=` (read `dw_mcp/diagnose.py:241 wait_for_job` for the polling cadence and the page shape) and `POST /api/jobs/{id}/cancel`.
- Produces: `main(argv=None, client=None) -> int`. The CLI is:

```text
python -m dw.run WORKFLOW [name=value ...] [--server URL] [--workspace NAME] [--token TOKEN]
```

  - `WORKFLOW`:
    - When it is not an existing local file, it is sent as `workflow_path`: a catalog name, or a path the server resolves.
    - When it is an existing local file, it is first sent as `workflow_path=os.path.abspath(WORKFLOW)`. That is the local-checkout case, and it keeps run-directory identity and relative paths right.
    - If the server answers 400 because it cannot reach that path, the file's JSON is resent as `workflow`, after printing: `note: the server cannot read <path>; sent its definition inline - relative paths in it resolve against the server's workflows/`.
  - `name=value` pairs are validated as today (`validate_variable_name`, `validate_string_input`) and sent as string `arguments`. The server coerces them exactly as the in-process run did.
  - While running, one line is printed per event from the event log (`step_start`, `step_end`, warnings; the implementer picks the fields from `diagnose.py`'s summary). Finally it prints the status, `run_dir`, and each manifest file. It returns 0 on `succeeded` and 1 on `failed` / `cancelled`, printing the error.
  - Ctrl-C calls `POST /api/jobs/{id}/cancel`, prints `cancelled`, and returns 130.
  - A connection error prints `error: no dw.serve at <url> - start one with: python -m dw.serve` and returns 2. A 401/403 prints `error: the server refused the token (set DW_API_TOKEN or pass --token)` and returns 2. Neither prints a traceback.
- Removed flags: `-o/--output_dir`, `--prompt-dir`, `--asset-dir`, `--output-layout`, `--trust-workflows` and `-l/--log_level`. They are `dw.serve` flags now, and argparse's "unrecognized arguments" is the message. List them in the release notes.

- [ ] **Step 1: Write the failing tests.**
  - Rewrite `TestRunEntryPoint` in `tests/test_cli.py` against a real app. Build `create_app(...)` with `ScriptedWorkerManager` (as `tests/test_server.py` does) and bridge it to `DwClient` with an `httpx.MockTransport` whose handler forwards each request to a `fastapi.testclient.TestClient(app)`:

```python
def _bridge(app):
    local = TestClient(app)

    def handler(request):
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        response = local.request(
            request.method, request.url.path, params=request.url.params,
            content=request.content, headers=headers,
        )
        return httpx.Response(response.status_code, headers=response.headers, content=response.content)

    # TestClient's own host, so host-checking middleware sees what it expects
    return DwClient(base_url="http://testserver", transport=httpx.MockTransport(handler))
```

  - Tests, one behavior each:
    - `test_a_catalog_name_runs_and_exits_0`: the script answers success. `main([...], client=...) == 0`, and the output names `run_dir`.
    - `test_name_value_pairs_arrive_as_arguments`: the `JobRequest` the manager received carries `{"prompt": "a cat"}`.
    - `test_a_bad_pair_is_refused_before_any_request`: `oops` exits 1 with the existing message, and zero requests are made.
    - `test_a_file_the_server_cannot_reach_is_sent_inline_with_a_notice`: a JSON file in `tmp_path` outside the server's roots. Two requests are made (a path, then inline), and the notice is printed.
    - `test_a_400_for_bad_arguments_on_a_local_file_is_not_resent_inline`: a local file the server *can* reach, with an undeclared argument. Exactly one request is made, the server's error is printed, and the exit code is 1. The inline fallback keys only on the "can reach" refusal. Matching the server's message text is brittle, but the alternative is a server change this task does not make.
    - `test_a_failed_job_exits_1_with_its_error`: the script answers an error, so exit 1 and the error text is printed.
    - `test_no_server_is_one_line_and_exit_2`: a `DwClient` whose `httpx.MockTransport` handler raises `httpx.ConnectError("refused")`. No real socket is opened. Exit 2, the output starts with `error: no dw.serve at`, and there is no `Traceback`.
- [ ] **Step 2: Run them to verify they fail.** Run `venv/bin/python -m pytest tests/test_cli.py -q`. Expected: `TypeError: main() got an unexpected keyword argument 'client'` for the run tests. The validate tests pass.
- [ ] **Step 3: Implement** `dw/run.py` to the interface above, then make the `pyproject.toml` httpx move. `dw/run.py` imports nothing from `dw.workflow`, `dw.worker`, `dw.worker_manager` or `dw.server`. Check with `grep -n "^from\|^import" dw/run.py`. (`import dw` itself loads torch through `dw/__init__.py`. Restructuring that is Phase 3 work, not this task's: the point of the client is one path to the GPU, not startup time.)
- [ ] **Step 4: Full suite, then a live check.**
  - Run `venv/bin/python -m pytest -q`.
  - Then start `venv/bin/python -m dw.serve --port 8799` in the background and run `venv/bin/python -m dw.run workflows/templates/text-to-image.json --server http://127.0.0.1:8799 num_images_per_prompt=1` (on the Mac: MPS, a small model).
  - Expected: exit 0, with the printed `run_dir` holding the image and `workflow.json`.
  - Stop the server.
- [ ] **Step 5: Docs.** Update "Common Commands" in CLAUDE.md to `python -m dw.serve` first, then `python -m dw.run ...` with a one-line note that `dw.run` is a client. In README.md, update the `dw.run` flags and the output-directory text. Run `scripts/arch_metrics.py --check`. Expected: exit 0, with `claude_md_lines` not above the baseline.
- [ ] **Step 6: Commit** `refactor(run): python -m dw.run is a client of dw.serve`

---

### Task 7: `open_run` survives a parent folder swept mid-claim

**Files:**
- Modify: `dw/runs.py`: `open_run` (:595, the `makedirs` + claim loop at :643-651)
- Test: `tests/test_runs.py`

**Why:**
- The gallery sweep (`_remove_empty_identity_folders`, `dw/server/app.py:3756`) holds the run lock of the identity it deleted from while it `rmdir`s empty parents up to the output root.
- A run of a *sibling* identity under the same parent (`ltx2/a` swept while `ltx2/b` opens) holds a different lock.
- If the sweep removes `ltx2/` between `open_run`'s `os.makedirs(identity_dir)` creating it and the `os.mkdir(candidate)` claim, the claim raises `FileNotFoundError` and the run fails before its first step.
- This is Phase 0's deferred "nested-identity sweep race".

**Interfaces:** no signature change. The claim is retried: on `FileNotFoundError` from `os.mkdir(candidate)`, `os.makedirs(identity_dir, exist_ok=True)` runs again and the same candidate is retried, up to 3 attempts. After that the error is raised as before.

- [ ] **Step 1: Write the failing test** in `tests/test_runs.py` (verified to fail at `f07eaf2d` with `FileNotFoundError` on the claim):

```python
RUN_ID = "20260928T000000Z-deadbeef"


def test_a_parent_swept_between_makedirs_and_the_claim_is_recreated(
    tmp_path, monkeypatch
):
    import os

    from dw import runs

    real_mkdir = os.mkdir
    swept = []

    def sweep_then_claim(path, *args, **kwargs):
        # Only the claim itself: os.makedirs calls os.mkdir too
        if not swept and os.path.basename(path) == RUN_ID:
            identity_dir = os.path.dirname(path)
            # what a sibling identity's sweep does to the empty folders
            os.rmdir(identity_dir)
            os.rmdir(os.path.dirname(identity_dir))
            swept.append(identity_dir)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(runs.os, "mkdir", sweep_then_claim)
    run_dir, version = runs.open_run(
        str(tmp_path / "outputs"),
        str(tmp_path / "workflows" / "ltx2" / "b.json"),
        "b",
        RUN_ID,
    )
    assert swept
    assert os.path.isdir(run_dir)
    assert version == 1
```

- [ ] **Step 2: Run it to verify it fails.** Run `venv/bin/python -m pytest tests/test_runs.py -q -k swept`. Expected: `FileNotFoundError`.
- [ ] **Step 3: Implement** the bounded retry, with a comment naming the sweep it answers.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_runs.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** `fix(runs): a run claim survives a sibling's sweep removing its parent folder`

---

### Task 8: Gate 1

**Files:** `docs/stabilization/ROADMAP.md`, `docs/stabilization/baseline.json`, `docs/stabilization/hot-zone.txt`, `docs/stabilization/ASSESSMENT.md`, the Claude Doc (via the controller), and the memory file.

- [ ] **Step 1: Full suite and ratchets.** Run `venv/bin/python -m pytest -q` and `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. Both must pass.
- [ ] **Step 2: Merge and deploy.** Merge `stabilization/phase-1` to `develop` and push. Then run `ssh lem '~/diffusers-workflow/scripts/deploy.sh develop'`, and on lem run `venv/bin/pip install -e .` for httpx.
- [ ] **Step 3: Real-model timings on lem.**
  - (a) B2: submit `templates/ltx2/two-stage` twice with the same seed (the dw MCP tools, `workspace: "default"`). The second run is reused and takes about 1 s.
  - (b) The thin client for real: `ssh lem 'cd ~/diffusers-workflow && venv/bin/python -m dw.run templates/ltx2/two-stage seed=<new>'`. It should be a cold run of about 160-174 s, exit 0, and print a `run_dir`.
  - (c) `POST /api/validate` on the same workflow answers with `plan` present.
  - Record all three.
- [ ] **Step 4: Tag.** Run `git tag stabilization-gate-1 && git push origin stabilization-gate-1`.
- [ ] **Step 5: Report.** Run `venv/bin/python scripts/arch_report.py > /tmp/gate1.md`, which gives the columns Before Phase 0, Gate 0 and Gate 1. Add a "### Gate 1" section to ROADMAP.md's Gate reports with:
  - the timings;
  - the list of surface changes: `dw.run` flags (and `dw.run --workspace` now names a server workspace, not a directory; trust is `dw.serve --trust-workflows`); `dw-repl` gone; jobs record the full warning set; rerun rechecks references; plan fingerprints shift once;
  - the generated tables.

  Mark Phase 1 done in the phase table.
- [ ] **Step 6: Re-baseline at the gate.** Run `venv/bin/python scripts/arch_metrics.py --write docs/stabilization/baseline.json`, so the harness ratchet holds this phase's gains.
- [ ] **Step 7: Hot zone for Phase 2.** Rewrite `hot-zone.txt` from the Phase 2 row. The controller does this when writing `phase-2.md`. Until then, keep the Phase 1 list.
- [ ] **Step 8: Commit and push** `docs(stabilization): close gate 1`. The controller updates the Claude Doc's plan and metrics table and the memory file.
