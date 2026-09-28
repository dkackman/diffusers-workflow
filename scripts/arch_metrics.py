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
