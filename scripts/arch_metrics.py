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
- prefix_literals counts a string constant that is exactly a reference
  prefix, outside PREFIX_OWNERS.
- prefix_handling counts hand-written handling of a reference prefix, outside
  PREFIX_OWNERS, per AST node. A "reference expression" is a constant of
  dw/references.py (parsed from the measured root; a fixed list when the file
  is absent) as `<module alias>.NAME` (any import form), as a name imported
  from references (with `as`), as a module-level name assigned from either,
  or a name imported from a dw module that ends in _PREFIX or _PREFIXES.
  Forms: (a) a startswith/removeprefix/removesuffix/replace/split/partition
  call with one as an argument; (b) a slice starting at len(<one>); (c) `+`
  with one as an operand, or an f-string splicing one in; (d) a module-level
  assignment of one (or a tuple holding one); (e) a non-docstring string that
  starts with a prefix and is longer than it, or an f-string fragment that
  ends with a prefix. Prose naming a bare prefix is not counted.
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
        "builtin:",
        "constraint:",
    )
)
# Modules allowed to spell a reference prefix: the one owner (Phase 2a)
PREFIX_OWNERS = frozenset({"dw/references.py"})
EXCLUDED = ("community_pipelines", "node_modules", "venv", ".git")
PACKAGES = ("dw", "dw_mcp")
REFERENCES_MODULE = "dw/references.py"
FALLBACK_REFERENCE_NAMES = frozenset(
    "ASSET OUTPUT PROMPT VARIABLE PREVIOUS_RESULT CONSTANT ITEM GATHER BUILTIN "
    "CONSTRAINT SUBSTITUTED UNRESOLVED DEFERRED".split()
)
PREFIX_METHODS = frozenset(
    ("startswith", "removeprefix", "removesuffix", "replace", "split", "partition")
)
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


def _reference_names(root):
    """(constant names, prefix strings) defined by dw/references.py under root:
    a single prefix is an upper-case name bound to a string ending in a colon;
    a tuple is one built from those names (and other tuples)."""
    path = root / REFERENCES_MODULE
    if not path.is_file():
        return FALLBACK_REFERENCE_NAMES, REFERENCE_PREFIXES
    assigns = [
        (node.targets[0].id, node.value)
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
    ]
    singles = {
        name: value.value
        for name, value in assigns
        if isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.endswith(":")
    }
    names = set(singles)

    def built(node):
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, ast.Tuple):
            return any(built(element) for element in node.elts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return built(node.left) or built(node.right)
        return False

    grown = True
    while grown:
        grown = False
        for name, value in assigns:
            if name not in names and built(value):
                names.add(name)
                grown = True
    return frozenset(names), frozenset(singles.values())


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _Resolver:
    """What counts as a reference expression in one module: the module's own
    imports of references (any form), its bare imports from it, names imported
    from a dw module that end in _PREFIX / _PREFIXES, and module-level names
    assigned from any of those."""

    def __init__(self, tree, constants):
        self.constants = constants
        self.modules = set()
        self.names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "dw.references":
                        self.modules.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                self._from_import(node)
        grown = True
        while grown:
            grown = False
            for node in tree.body:
                target, value = _assignment(node)
                if (
                    target is not None
                    and target not in self.names
                    and self.is_reference(value)
                ):
                    self.names.add(target)
                    grown = True

    def _from_import(self, node):
        in_dw = node.level > 0 or (node.module or "").split(".")[0] == "dw"
        for alias in node.names:
            bound = alias.asname or alias.name
            if alias.name == "references" and node.module in (None, "dw"):
                self.modules.add(bound)
            elif (node.module or "").split(".")[-1] == "references" and in_dw:
                if alias.name in self.constants:
                    self.names.add(bound)
            elif in_dw and alias.name.endswith(("_PREFIX", "_PREFIXES")):
                self.names.add(bound)

    def is_reference(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.names
        if isinstance(node, ast.Attribute):
            return node.attr in self.constants and _dotted(node.value) in self.modules
        if isinstance(node, ast.Tuple):
            return any(self.is_reference(element) for element in node.elts)
        return False


def _assignment(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target, value = node.target, node.value
    else:
        return None, None
    return (target.id if isinstance(target, ast.Name) else None), value


def prefix_handling_sites(tree, constants, prefixes):
    """Every hand-written handling of a reference prefix in one parsed module,
    as (line, form) with form one of "a".."e"."""
    resolver = _Resolver(tree, constants)
    refers = resolver.is_reference
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    fragments = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }
    found = []
    for node in tree.body:
        target, value = _assignment(node)
        if target is not None and refers(value):
            found.append((node.lineno, "d"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in PREFIX_METHODS
                and any(refers(arg) for arg in node.args)
            ):
                found.append((node.lineno, "a"))
        elif isinstance(node, ast.Subscript):
            lower = node.slice.lower if isinstance(node.slice, ast.Slice) else None
            if (
                isinstance(lower, ast.Call)
                and isinstance(lower.func, ast.Name)
                and lower.func.id == "len"
                and len(lower.args) == 1
                and refers(lower.args[0])
            ):
                found.append((node.lineno, "b"))
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add) and (
                refers(node.left) or refers(node.right)
            ):
                found.append((node.lineno, "c"))
        elif isinstance(node, ast.JoinedStr):
            if any(
                isinstance(part, ast.FormattedValue) and refers(part.value)
                for part in node.values
            ):
                found.append((node.lineno, "c"))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            text = node.value
            if any(text.startswith(p) and len(text) > len(p) for p in prefixes) or (
                id(node) in fragments and text.endswith(tuple(prefixes))
            ):
                found.append((node.lineno, "e"))
    return found


def measure(root):
    root = pathlib.Path(root)
    engine = list(_sources(root, *PACKAGES))
    metrics = {
        "modules": len(engine),
        "modules_over_1000_lines": 0,
        "functions_over_150_lines": 0,
        "prefix_literals": 0,
        "prefix_handling": 0,
    }
    constants, prefixes = _reference_names(root.resolve())
    for path in engine:
        text = path.read_text(encoding="utf-8")
        if len(text.splitlines()) > 1000:
            metrics["modules_over_1000_lines"] += 1
        tree = ast.parse(text)
        if path.relative_to(root).as_posix() not in PREFIX_OWNERS:
            metrics["prefix_handling"] += len(
                prefix_handling_sites(tree, constants, prefixes)
            )
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
