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
